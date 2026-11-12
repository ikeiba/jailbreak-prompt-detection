"""Caché de ELMo: un vector por token (media de las 3 capas, 1024 dims, float16) calculado
una sola vez sobre la secuencia ya truncada de cada configuración de entrada.

Formato, por configuración y split (<raíz>/<truncado>__L<longitud>/<split>/):
- vectors.f16   matriz (total_tokens, 1024) en float16, sin cabecera, escrita por batches.
- index.npz     ids de los prompts (orden del JSONL), fila de inicio (offsets) y nº de tokens.
- progress.json batches terminados y filas escritas; complete=True cuando ha acabado.

ElmoCache lee con np.memmap: solo se cargan del disco los vectores del prompt pedido.
La escritura (ElmoTF, precompute_split) es la única parte que importa TensorFlow."""

import json
import os
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from jailbreak_detection.models.data import SPLITS, truncate

ELMO_DIM = 1024
STORE_DTYPE = np.float16
PRECOMPUTE_NOTEBOOK = "notebooks/05_elmo_precompute.ipynb"


def config_name(truncation: str, max_len: int) -> str:
    return f"{truncation}__L{max_len}"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def cache_status(root: str | Path, truncation: str, max_len: int) -> dict[str, dict | None]:
    """progress.json de cada split (None si no se ha empezado)."""
    base = Path(root) / config_name(truncation, max_len)
    out = {}
    for split in SPLITS:
        path = base / split / "progress.json"
        out[split] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return out


# --------------------------------------------------------------------------- lectura

class ElmoCache:
    """Vectores de ELMo precalculados de una configuración de entrada (todos los splits).
    Los ids de prompt son únicos entre splits, así que se indexa directamente por id."""

    def __init__(self, root: str | Path, truncation: str, max_len: int,
                 splits: Sequence[str] = SPLITS):
        self.name = config_name(truncation, max_len)
        base = Path(root) / self.name
        self._arrays, self._lookup = [], {}
        for split in splits:
            d = base / split
            progress = d / "progress.json"
            status = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() else None
            if status is None or not status.get("complete"):
                state = "no existe" if status is None else (
                    f"está incompleta ({status['batches_done']}/{status['n_batches']} batches)")
                raise FileNotFoundError(
                    f"La caché de ELMo '{self.name}' del split '{split}' {state} en {d}. "
                    f"Ejecuta {PRECOMPUTE_NOTEBOOK} (con el mismo valor de QUICK) para generarla"
                    + ("" if self.name == "head_tail__L1024" else
                       "; esta configuración es de una ablación: el 05 la genera cuando ELMo ha "
                       "ganado el paso 1 de notebooks/06_deep_encoder.ipynb (o con FORCE_ABLATIONS = True)") + ".")
            index = np.load(d / "index.npz")
            array = np.memmap(d / "vectors.f16", dtype=STORE_DTYPE, mode="r",
                              shape=(int(status["total_rows"]), ELMO_DIM))
            k = len(self._arrays)
            self._arrays.append(array)
            for pid, off, n in zip(index["ids"], index["offsets"], index["lengths"]):
                self._lookup[int(pid)] = (k, int(off), int(n))

    def __contains__(self, prompt_id: int) -> bool:
        return int(prompt_id) in self._lookup

    def length(self, prompt_id: int) -> int:
        return self._lookup[int(prompt_id)][2]

    def get(self, prompt_id: int) -> np.ndarray:
        """(n_tokens, 1024) en float32."""
        k, off, n = self._lookup[int(prompt_id)]
        return np.asarray(self._arrays[k][off:off + n], dtype=np.float32)

    def batch(self, prompt_ids: Sequence[int]) -> np.ndarray:
        """(B, max_tokens, 1024) en float32, con ceros en el padding."""
        lengths = [self.length(p) for p in prompt_ids]
        out = np.zeros((len(prompt_ids), max(lengths), ELMO_DIM), dtype=np.float32)
        for row, pid in enumerate(prompt_ids):
            out[row, :lengths[row]] = self.get(pid)
        return out


# --------------------------------------------------------------------------- escritura

class ElmoTF:
    """ELMo (simple_elmo, TensorFlow) en inferencia, con el estado del LSTM a cero en cada batch.

    El biLM de simple_elmo es stateful: guarda el estado final de cada fila del batch en 8
    variables (2 direcciones x 2 capas x (c, h)) y lo usa como estado inicial del batch
    siguiente. Aquí se reinician a cero antes de cada batch y, como cada fila del batch tiene su
    propio estado, el vector de un prompt no depende de lo procesado antes ni de con qué otros
    prompts comparta batch."""

    def __init__(self, model_dir: str | Path, batch_size: int = 32, layers: str = "average"):
        os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import logging

        import tensorflow as tf
        from simple_elmo import ElmoModel
        from simple_elmo.elmo import weight_layers

        self.tf_version = tf.__version__
        self.gpus = tf.config.list_physical_devices("GPU")
        for gpu in self.gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        self.batch_size = batch_size
        self.model = ElmoModel()
        self.model.load(str(model_dir), max_batch_size=batch_size)
        logging.getLogger("simple_elmo.elmo_helpers").setLevel(logging.WARNING)
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(config=config)
        with self.session.as_default():
            self.model.elmo_sentence_input = weight_layers(
                "input", self.model.sentence_embeddings_op, use_layers=layers)
            self.session.run(tf.compat.v1.global_variables_initializer())
        state_vars = [v for v in tf.compat.v1.global_variables()
                      if re.fullmatch(r"bilm/Variable(_\d+)?:0", v.name)]
        if len(state_vars) != 8 or any(v.shape[0] != batch_size for v in state_vars):
            raise RuntimeError("No se han encontrado las 8 variables de estado del LSTM de ELMo.")
        self._reset = tf.compat.v1.variables_initializer(state_vars)

    @property
    def device(self) -> str:
        return f"GPU ({len(self.gpus)})" if self.gpus else "CPU"

    def __call__(self, token_lists: Sequence[Sequence[str]]) -> np.ndarray:
        """(B, max_tokens, 1024) en float32. B no puede superar batch_size."""
        if len(token_lists) > self.batch_size:
            raise ValueError("El batch supera el batch_size con el que se cargó ELMo.")
        self.session.run(self._reset)
        vectors = self.model.get_elmo_vectors([list(t) for t in token_lists], warmup=False,
                                              session=self.session)
        return vectors.astype(np.float32)

    def tokens_per_second(self, token_lists: Sequence[Sequence[str]]) -> float:
        """Velocidad medida sobre un batch de ejemplo (se descarta una primera ejecución)."""
        self(token_lists[:2])
        t0 = time.time()
        self(token_lists)
        return sum(len(t) for t in token_lists) / (time.time() - t0)


def precompute_split(elmo: ElmoTF, df, truncation: str, max_len: int, root: str | Path,
                     split: str, log=print, log_every: float = 60.0,
                     max_batches: int | None = None) -> dict:
    """Calcula (o continúa) la caché de un split. Reanudable: tras cada batch se vacían los
    datos a disco y se guarda el progreso; al relanzar se recorta lo escrito después del último
    batch completo y se sigue desde ahí. Orden fijo: de más largo a más corto (con empate, el
    orden del JSONL), así un posible fallo de memoria aparece al principio.
    max_batches limita los batches de esta llamada (para pruebas)."""
    d = Path(root) / config_name(truncation, max_len) / split
    d.mkdir(parents=True, exist_ok=True)
    tokens = [truncate(t, max_len, truncation) for t in df["tokens_flat"]]
    ids = df["id"].to_numpy(dtype=np.int64)
    lengths = np.array([len(t) for t in tokens], dtype=np.int64)
    order = np.argsort(-lengths, kind="stable")
    offsets = np.empty_like(lengths)
    offsets[order] = np.concatenate([[0], np.cumsum(lengths[order])[:-1]])
    batches = [order[s:s + elmo.batch_size] for s in range(0, len(order), elmo.batch_size)]

    index_path, progress_path, vectors_path = d / "index.npz", d / "progress.json", d / "vectors.f16"
    if index_path.exists():
        old = np.load(index_path)
        if not (np.array_equal(old["ids"], ids) and np.array_equal(old["lengths"], lengths)
                and np.array_equal(old["order"], order)):
            raise ValueError(f"La caché de {d} se hizo con otros prompts; bórrala para regenerarla.")
    else:
        np.savez(index_path, ids=ids, offsets=offsets, lengths=lengths, order=order)
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {
        "split": split, "truncation": truncation, "max_len": max_len, "layers": "average",
        "dtype": "float16", "dim": ELMO_DIM, "state_reset_per_batch": True,
        "batch_size": elmo.batch_size, "n_prompts": len(ids), "n_batches": len(batches),
        "total_rows": int(lengths.sum()), "batches_done": 0, "rows_written": 0,
        "complete": False, "seconds": 0.0}
    if progress["complete"]:
        log(f"[{config_name(truncation, max_len)}/{split}] ya completa.")
        return progress
    if progress["batch_size"] != elmo.batch_size:
        raise ValueError(f"La caché de {d} se empezó con batch_size={progress['batch_size']}; "
                         "usa ese valor para continuarla o bórrala.")

    done, seconds_before = progress["batches_done"], progress["seconds"]
    expected = progress["rows_written"] * ELMO_DIM * np.dtype(STORE_DTYPE).itemsize
    if (vectors_path.stat().st_size if vectors_path.exists() else 0) < expected:
        raise RuntimeError(f"{vectors_path} tiene menos datos de los que indica progress.json; "
                           "borra la carpeta para regenerarla.")
    with open(vectors_path, "ab") as f:  # recorta lo escrito tras el último batch completo
        f.truncate(expected)
    if done:
        log(f"[{config_name(truncation, max_len)}/{split}] se reanuda: {done}/{len(batches)} "
            f"batches ya guardados.")
    start, last_log, rows_session = time.time(), time.time(), 0
    with open(vectors_path, "ab") as f:
        for b in range(done, len(batches) if max_batches is None else min(len(batches), done + max_batches)):
            batch = batches[b]
            vectors = elmo([tokens[i] for i in batch])
            for row, i in enumerate(batch):
                f.write(vectors[row, :lengths[i]].astype(STORE_DTYPE).tobytes())
            f.flush()
            os.fsync(f.fileno())
            n_rows = int(lengths[batch].sum())
            rows_session += n_rows
            progress["rows_written"] += n_rows
            progress["batches_done"] = b + 1
            progress["seconds"] = seconds_before + time.time() - start
            progress["complete"] = progress["batches_done"] == len(batches)
            _write_json(progress_path, progress)
            if time.time() - last_log > log_every or progress["complete"] or b == done:
                elapsed = time.time() - start
                speed = rows_session / elapsed
                left = (progress["total_rows"] - progress["rows_written"]) / speed
                log(f"[{config_name(truncation, max_len)}/{split}] batch {b + 1}/{len(batches)} | "
                    f"{progress['rows_written']:,}/{progress['total_rows']:,} tokens | "
                    f"{speed:,.0f} tokens/s | quedan ~{left / 60:.1f} min")
                last_log = time.time()
    if progress["complete"]:
        assert vectors_path.stat().st_size == progress["total_rows"] * ELMO_DIM * 2
    return progress
