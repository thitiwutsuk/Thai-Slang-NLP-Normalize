"""Phase 1: tokenization using PyThaiNLP's newmm engine."""

from pythainlp.tokenize import word_tokenize


def tokenize_text(text: str) -> list[str]:
    """Tokenize Thai (and mixed-language) text, dropping whitespace-only
    tokens that `word_tokenize` keeps by default."""
    if not text:
        return []
    tokens = word_tokenize(text, engine="newmm")
    return [t for t in tokens if t.strip()]
