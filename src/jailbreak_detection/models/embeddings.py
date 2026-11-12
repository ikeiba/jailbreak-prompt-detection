"""Las cuatro entradas del encoder: word2vec original (congelado o ajustable), word2vec propio
congelado y ELMo congelado, leído de la caché precalculada por el notebook 05.
Este módulo no importa TensorFlow."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from jailbreak_detection.models.data import PAD, PAD_IDX, UNK, UNK_IDX, PromptDataset, TokenIndex
from jailbreak_detection.models.elmo_cache import ELMO_DIM, ElmoCache

EMBEDDINGS = {
    # nombre: (fuente, congelado)
    "w2v_google_frozen": ("google", True),
    "w2v_google_finetuned": ("google", False),
    "w2v_domain_frozen": ("domain", True),
    "elmo_frozen": ("elmo", True),
}
GOOGLE_W2V = "word2vec-google-news-300"
SEED = 42


def build_matrix(vocab: Sequence[str], kv, seed: int = SEED) -> tuple[list[str], np.ndarray]:
    """Una fila por token de vocab que tenga vector en kv; el resto irá a <unk>.
    Fila 0 (<pad>) = ceros; fila 1 (<unk>) = vector aleatorio fijo N(0, std de los vectores)."""
    known = [t for t in vocab if t in kv.key_to_index and t not in (PAD, UNK)]
    vectors = np.stack([kv[t] for t in known]).astype(np.float32)
    unk = np.random.default_rng(seed).normal(0.0, vectors.std(), vectors.shape[1]).astype(np.float32)
    matrix = np.vstack([np.zeros((1, vectors.shape[1]), np.float32), unk[None, :], vectors])
    return known, matrix


@dataclass
class InputPipeline:
    """Cómo se convierte un batch en la entrada (B, T, input_size) de la GRU."""
    name: str
    input_size: int
    freeze: bool = True
    index: TokenIndex | None = None
    matrix: np.ndarray | None = None
    elmo_cache: ElmoCache | None = None

    def dataset(self, df: pd.DataFrame, max_len: int, truncation: str) -> PromptDataset:
        ds = PromptDataset(df["tokens_flat"], df["label"], max_len, truncation, index=self.index,
                           prompt_ids=df["id"] if self.index is None else None)
        if self.elmo_cache is not None:
            # La caché debe ser exactamente la secuencia truncada que ve el modelo
            for ref in ds.items:
                if ref.prompt_id not in self.elmo_cache:
                    raise KeyError(f"El prompt {ref.prompt_id} no está en la caché "
                                   f"'{self.elmo_cache.name}' (¿caché del modo rápido?).")
                if self.elmo_cache.length(ref.prompt_id) != ref.length:
                    raise ValueError(f"La caché '{self.elmo_cache.name}' no coincide con el "
                                     f"truncado del prompt {ref.prompt_id}.")
        return ds

    def to_device(self, inputs, device) -> torch.Tensor:
        if self.elmo_cache is None:
            return inputs.to(device)
        return torch.from_numpy(self.elmo_cache.batch(inputs)).to(device)


class EmbeddingStore:
    """Construye (y guarda en caché) cada entrada solo cuando hace falta.

    El vocabulario es el del word2vec propio (notebook 03), construido solo con train:
    se comprueba que coincide exactamente con los tipos de train."""

    def __init__(self, domain_w2v_path: str | Path, cache_dir: str | Path,
                 elmo_cache_root: str | Path, train_docs: Sequence[Sequence[str]]):
        from gensim.models import Word2Vec

        self.domain = Word2Vec.load(str(domain_w2v_path)).wv
        self.vocab = list(self.domain.index_to_key)
        train_types = {t for doc in train_docs for t in doc}
        if set(self.vocab) != train_types:
            raise ValueError("El vocabulario del word2vec propio no coincide con los tokens de train.")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.elmo_cache_root = Path(elmo_cache_root)
        self._pipelines: dict[tuple, InputPipeline] = {}

    def _matrix(self, source: str) -> tuple[list[str], np.ndarray]:
        matrix_path = self.cache_dir / f"w2v_{source}_matrix.npy"
        tokens_path = self.cache_dir / f"w2v_{source}_tokens.json"
        if matrix_path.exists() and tokens_path.exists():
            return json.loads(tokens_path.read_text(encoding="utf-8")), np.load(matrix_path)
        if source == "google":
            import gensim.downloader as api
            kv = api.load(GOOGLE_W2V)
        else:
            kv = self.domain
        known, matrix = build_matrix(self.vocab, kv)
        np.save(matrix_path, matrix)
        tokens_path.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")
        return known, matrix

    def get(self, name: str, truncation: str = "head_tail", max_len: int = 1024) -> InputPipeline:
        """Las variantes word2vec no dependen de la configuración de entrada; la de ELMo sí,
        porque sus vectores se calcularon sobre la secuencia ya truncada."""
        source, freeze = EMBEDDINGS[name]
        key = (name, truncation, max_len) if source == "elmo" else (name,)
        if key in self._pipelines:
            return self._pipelines[key]
        if source == "elmo":
            cache = ElmoCache(self.elmo_cache_root, truncation, max_len)
            pipe = InputPipeline(name, ELMO_DIM, freeze=True, elmo_cache=cache)
        else:
            known, matrix = self._matrix(source)
            index = TokenIndex(known)
            assert index.stoi[PAD] == PAD_IDX and index.stoi[UNK] == UNK_IDX
            assert len(index) == matrix.shape[0]
            pipe = InputPipeline(name, matrix.shape[1], freeze=freeze, index=index, matrix=matrix)
        self._pipelines[key] = pipe
        return pipe
