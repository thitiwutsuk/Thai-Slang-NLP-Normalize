"""Phase 2: elongation reduction (collapse repeated characters)."""

import re

import emoji as emoji_lib
from pythainlp.util import remove_trailing_repeat_consonants

_MAIYAMOK_RUN_RE = re.compile(r"ๆ{2,}")
_GENERIC_REPEAT_RE = re.compile(r"(\S)\1{2,}")


def _collapse_non_digit_non_emoji(match: re.Match) -> str:
    char = match.group(1)
    if char.isdigit() or emoji_lib.is_emoji(char):
        return match.group(0)
    return char


def reduce_elongation(text: str) -> str:
    """Collapse repeated characters ("มากกกกก" -> "มาก", "จริงๆๆๆ" -> "จริงๆ").

    Leaves digit runs ("555") and emoji runs alone -- those carry meaning
    (laughter, emphasis) rather than being a typing quirk.
    """
    text = _MAIYAMOK_RUN_RE.sub("ๆ", text)
    text = remove_trailing_repeat_consonants(text)
    text = _GENERIC_REPEAT_RE.sub(_collapse_non_digit_non_emoji, text)
    return text
