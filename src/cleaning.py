"""Phase 1: text cleaning (URLs, mentions, hashtag symbols, whitespace)."""

import re

URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_SYMBOL_RE = re.compile(r"#")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Strip URLs, @mentions, and hashtag symbols (keeping the hashtag's
    word), then collapse whitespace. Does not touch elongation, slang, or
    emoji -- that's Phase 2."""
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_SYMBOL_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text
