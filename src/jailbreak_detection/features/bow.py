"""Persistencia de las representaciones Bag of Words."""

import json
from pathlib import Path

import joblib
from scipy.sparse import save_npz


def save_bow(vectorizer, matrices: dict, out_dir: str | Path) -> None:
    """Guarda las matrices dispersas, el vocabulario y el vectorizador BoW."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        save_npz(out_dir / f"bow_{name}.npz", matrix)
    vocab = {token: int(index) for token, index in vectorizer.vocabulary_.items()}
    (out_dir / "bow_vocabulary.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    joblib.dump(vectorizer, out_dir / "bow_vectorizer.joblib")
