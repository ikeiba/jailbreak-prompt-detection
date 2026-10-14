import json
from pathlib import Path

import joblib
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer


def identity(x):
    """Función de módulo (no lambda) para poder guardar el vectorizador con joblib."""
    return x


def flatten(tokens: list[list[str]]) -> list[str]:
    """Une los tokens de todas las oraciones de un prompt en una sola lista."""
    return [t for sentence in tokens for t in sentence]


def build_vectorizer(min_df: int = 2) -> TfidfVectorizer:
    """TF-IDF sobre tokens ya calculados: sin volver a tokenizar, sin pasar a minúsculas,
    sin eliminar stopwords. sublinear_tf amortigua la repetición en prompts muy largos."""
    return TfidfVectorizer(
        tokenizer=identity, preprocessor=identity, token_pattern=None,
        lowercase=False, min_df=min_df, sublinear_tf=True,
    )


def save_tfidf(vectorizer, matrices: dict, out_dir: str | Path) -> None:
    """Guarda las matrices dispersas (.npz), el vocabulario (.json) y el vectorizador (.joblib)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, X in matrices.items():
        save_npz(out_dir / f"tfidf_{name}.npz", X)
    vocab = {t: int(i) for t, i in vectorizer.vocabulary_.items()}
    (out_dir / "tfidf_vocabulary.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")