import re

import pandas as pd
from ftfy import TextFixerConfig, fix_text

# NFKC + comillas tipográficas -> ASCII + entidades HTML + texto corrupto (mojibake)
# + caracteres de control. Todo en un paso, con la configuración explícita.
_FTFY = TextFixerConfig(
    normalization="NFKC",
    uncurl_quotes=True,
    unescape_html=True,
    fix_encoding=True,
    remove_control_chars=True,
    fix_line_breaks=True,
)

# Ancho cero que ocultan texto. No incluye U+200D (une emojis compuestos).
_ZERO_WIDTH = re.compile("[\u200b\u200c\u2060\ufeff]")
_MD_ESCAPE = re.compile(r"\\([\[\]()_*`#~>])")
# "â€" suelto: comilla doble de cierre (U+201D) cuyo último byte se perdió al corromperse.
# Se trata antes que ftfy, que lo convertiría en "†". Los "â€™", "â€œ"... completos los repara ftfy.
_TRUNCATED_QUOTE = re.compile("\u00e2\u20ac(?![\u2122\u0153\u201c\u201d\u02dc\u00a2\u00a6\x9d])")


def has_zero_width(text: str) -> bool:
    return bool(_ZERO_WIDTH.search(text))


def clean_text(text: str) -> str:
    t = _TRUNCATED_QUOTE.sub('"', text)
    t = fix_text(t, _FTFY)
    t = _ZERO_WIDTH.sub("", t)
    t = _MD_ESCAPE.sub(r"\1", t)
    t = re.sub(r"[ \t]+", " ", t)      # espacios y tabuladores repetidos -> uno
    t = re.sub(r" ?\n ?", "\n", t)     # sin espacios alrededor de los saltos de línea
    t = re.sub(r"\n+", "\n", t)        # rachas de saltos de línea -> uno
    return t.strip()


def add_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={"prompt": "prompt_raw"}).copy()
    out["has_zero_width"] = out["prompt_raw"].map(has_zero_width)   # se marca antes de quitarlos
    out["prompt_clean"] = out["prompt_raw"].map(clean_text)
    return out