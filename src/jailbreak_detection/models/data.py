"""Carga de los splits tokenizados, truncado, vocabulario y batches agrupados por longitud."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from jailbreak_detection.features.tfidf import flatten

SPLITS = ("train", "val", "test")
PAD, UNK = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1
TRUNCATIONS = ("head_tail", "head")


def load_splits(interim_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Lee los JSONL tokenizados y añade los tokens aplanados y la etiqueta entera."""
    splits = {}
    for name in SPLITS:
        df = pd.read_json(Path(interim_dir) / f"{name}_tok.jsonl", lines=True)
        df["tokens_flat"] = df["tokens"].map(flatten)
        df["label"] = df["jailbreak"].astype(int)
        splits[name] = df
    return splits


def truncate(tokens: Sequence[str], max_len: int, mode: str = "head_tail") -> list[str]:
    """Recorta a max_len tokens.
    head_tail: primeros max_len/2 + últimos max_len/2. head: primeros max_len."""
    tokens = list(tokens)
    if len(tokens) <= max_len:
        return tokens
    if mode == "head":
        return tokens[:max_len]
    if mode == "head_tail":
        half = max_len // 2
        return tokens[:half] + tokens[len(tokens) - (max_len - half):]
    raise ValueError(f"Truncado desconocido: {mode}")


class TokenIndex:
    """Token -> índice. 0 es <pad> (vector de ceros) y 1 es <unk>, compartido por todos los OOV."""

    def __init__(self, tokens: Sequence[str]):
        self.itos = [PAD, UNK] + [t for t in tokens if t not in (PAD, UNK)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str]) -> list[int]:
        return [self.stoi.get(t, UNK_IDX) for t in tokens]

    def oov_rate(self, docs: Sequence[Sequence[str]]) -> float:
        total = sum(len(d) for d in docs)
        oov = sum(t not in self.stoi for d in docs for t in d)
        return oov / total if total else 0.0


class CachedRef(NamedTuple):
    """Referencia a los vectores precalculados (ELMo) de un prompt."""
    prompt_id: int
    length: int


class PromptDataset(Dataset):
    """Prompts ya truncados. Con index, cada item es la lista de índices (word2vec); sin index,
    una CachedRef con el id del prompt para leer sus vectores de la caché (ELMo)."""

    def __init__(self, tokens: Sequence[Sequence[str]], labels: Sequence[int],
                 max_len: int, truncation: str, index: TokenIndex | None = None,
                 prompt_ids: Sequence[int] | None = None):
        self.tokens = [truncate(t, max_len, truncation) for t in tokens]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.lengths = np.array([len(t) for t in self.tokens])
        if (self.lengths == 0).any():
            raise ValueError("Hay prompts sin tokens; pack_padded_sequence no los admite.")
        self.index = index
        if index is not None:
            self.items = [index.encode(t) for t in self.tokens]
        elif prompt_ids is not None:
            self.items = [CachedRef(int(p), int(n)) for p, n in zip(prompt_ids, self.lengths)]
        else:
            raise ValueError("Hace falta un TokenIndex o los ids de los prompts.")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return i, self.items[i], self.labels[i]


def collate(batch) -> dict:
    """Rellena con PAD_IDX si los items son índices; con CachedRef devuelve la lista de ids."""
    idx, items, labels = zip(*batch)
    if isinstance(items[0], CachedRef):
        lengths = torch.tensor([r.length for r in items], dtype=torch.long)
        inputs = [r.prompt_id for r in items]
    else:
        lengths = torch.tensor([len(x) for x in items], dtype=torch.long)
        inputs = torch.full((len(items), int(lengths.max())), PAD_IDX, dtype=torch.long)
        for row, ids in enumerate(items):
            inputs[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return {
        "inputs": inputs,
        "lengths": lengths,
        "labels": torch.tensor(np.array(labels), dtype=torch.float32),
        "index": torch.tensor(idx, dtype=torch.long),
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Batches de prompts de longitud parecida para reducir el padding.

    En entrenamiento se baraja, se parte en bloques de pool_batches batches, se ordena cada
    bloque por longitud y se barajan los batches resultantes (semilla + nº de época).
    En evaluación se ordena todo por longitud, sin aleatoriedad."""

    def __init__(self, lengths: Sequence[int], batch_size: int, shuffle: bool,
                 seed: int = 42, pool_batches: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool = batch_size * pool_batches
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        n = len(self.lengths)
        if not self.shuffle:
            order = np.argsort(self.lengths, kind="stable")
            for s in range(0, n, self.batch_size):
                yield order[s:s + self.batch_size].tolist()
            return
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        perm = rng.permutation(n)
        batches = []
        for s in range(0, n, self.pool):
            chunk = perm[s:s + self.pool]
            chunk = chunk[np.argsort(self.lengths[chunk], kind="stable")]
            batches += [chunk[b:b + self.batch_size].tolist()
                        for b in range(0, len(chunk), self.batch_size)]
        for b in rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_loader(dataset: PromptDataset, batch_size: int, shuffle: bool, seed: int = 42):
    sampler = BucketBatchSampler(dataset.lengths, batch_size, shuffle=shuffle, seed=seed)
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, collate_fn=collate)


def stratified_subset(df: pd.DataFrame, n: int, min_pos: int, seed: int = 42) -> pd.DataFrame:
    """Subconjunto pequeño para el modo rápido, con al menos min_pos jailbreaks."""
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    n_pos = min(len(pos), max(min_pos, round(n * len(pos) / len(df))))
    sample = pd.concat([pos.sample(n_pos, random_state=seed),
                        neg.sample(min(len(neg), n - n_pos), random_state=seed)])
    return sample.sort_index()


# (nº de prompts, mínimo de jailbreaks) del modo rápido; los notebooks 05 y 06 usan los mismos
QUICK_SIZES = {"train": (160, 30), "val": (96, 16), "test": (96, 16)}


def select_splits(splits: dict[str, pd.DataFrame], quick: bool) -> dict[str, pd.DataFrame]:
    """Los splits completos o, en modo rápido, los subconjuntos de QUICK_SIZES."""
    if not quick:
        return dict(splits)
    return {name: stratified_subset(splits[name], *QUICK_SIZES[name]) for name in SPLITS}


# Misma detección que el notebook 04 (y el 02): prompts con [PROMPT] o [TARGETLANGUAGE]
PROMPT_MARKER = r"\[\s*(?:prompt|targetlanguage)\s*\]"


def has_prompt_marker(df: pd.DataFrame) -> pd.Series:
    return df["prompt_clean"].str.contains(PROMPT_MARKER, case=False, regex=True, na=False)


def validation_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Máscaras de evaluación del notebook 04: todo el split y sin marcadores."""
    marker = has_prompt_marker(df).to_numpy()
    return {"Todos": np.ones(len(df), dtype=bool), "Sin marcadores": ~marker}
