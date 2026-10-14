import re

import pandas as pd
from langdetect import DetectorFactory, detect_langs

DetectorFactory.seed = 0  # resultados reproducibles

_URL = re.compile(r"https?://\S+|www\.\S+")
_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z _]{2,}\]|\{\{.*?\}\}|\[[A-Z ]+\]")
MIN_LETTERS = 20


def detect_language(text: str, max_chars: int = 2000) -> tuple[str, float]:
    """Idioma probable y su probabilidad. Ignora URLs y marcadores, que confunden al detector.
    Devuelve ('short', 1.0) si hay menos de 20 letras y ('unk', 0.0) si el detector falla."""
    t = _PLACEHOLDER.sub("", _URL.sub("", text))[:max_chars]
    if len(re.findall(r"[^\W\d_]", t)) < MIN_LETTERS:
        return "short", 1.0
    try:
        best = detect_langs(t)[0]
        return best.lang, best.prob
    except Exception:
        return "unk", 0.0


def add_language_columns(df: pd.DataFrame, text_col: str = "prompt_clean") -> pd.DataFrame:
    out = df.copy()
    res = out[text_col].map(detect_language)
    out["lang"] = [lang for lang, _ in res]
    out["lang_prob"] = [prob for _, prob in res]
    return out


def remove_non_english(df: pd.DataFrame, min_prob: float = 0.9
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Elimina los prompts cuyo idioma detectado no es inglés con probabilidad >= min_prob.
    Los 'short' y 'unk' se conservan. Devuelve (df_en, eliminados)."""
    non_en = ~df["lang"].isin(["en", "short", "unk"]) & (df["lang_prob"] >= min_prob)
    return df[~non_en].copy(), df[non_en].copy()