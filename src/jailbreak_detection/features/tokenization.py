import re

from nltk.tokenize import casual_tokenize, sent_tokenize

# URL sin la puntuación final (", ) ] . ..."), que pertenece a la frase y no al enlace
_URL = re.compile(r"(?:https?://|www\.)\S+?(?=[.,;:!?)\]}\"']*(?:\s|$))")

# Marcadores frecuentes que se convierten en un token canónico
_NAMED = [
    (re.compile(r"\[\s*prompt\s*\]", re.I), "<prompt>"),
    (re.compile(r"\[\s*targetlanguage\s*\]", re.I), "<targetlanguage>"),
    (re.compile(r"\{\{\s*user\s*\}\}", re.I), "<user>"),
    (re.compile(r"\{\{\s*char\s*\}\}", re.I), "<char>"),
]
# Cualquier otra ranura de plantilla {{...}}
_CURLY = re.compile(r"\{\{[^{}\n]{1,60}\}\}")


def mask_special(text: str) -> str:
    """Sustituye URLs y marcadores de plantilla por tokens canónicos.
    Los [...] restantes (incluidos los que van en mayúsculas) se dejan al tokenizador."""
    t = _URL.sub("<url>", text)
    for pattern, token in _NAMED:
        t = pattern.sub(token, t)
    return _CURLY.sub("<placeholder>", t)


def split_sentences(text: str) -> list[str]:
    """Punkt, con el salto de línea como límite fuerte de oración."""
    sentences = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            sentences.extend(sent_tokenize(line))
    return sentences


def process_prompt(text: str) -> dict:
    """Enmascara, divide en oraciones y tokeniza cada oración con casual_tokenize.
    No pasa a minúsculas: eso se decide al representar."""
    sentences = split_sentences(mask_special(text))
    return {"sentences": sentences, "tokens": [casual_tokenize(s) for s in sentences]}