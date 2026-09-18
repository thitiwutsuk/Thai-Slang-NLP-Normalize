"""Phase 2: emoji / text-emoticon -> sentiment-token mapping."""

import re

import emoji as emoji_lib

_POSITIVE_EMOJI = {
    "😊", "😍", "🥰", "😘", "😁", "😄", "😃", "🙂", "😆", "😂", "🤣",
    "👍", "👏", "🙏", "❤️", "❤", "💕", "💖", "💗", "😻", "🥳", "✨",
}
_NEGATIVE_EMOJI = {
    "😭", "😢", "😞", "😔", "😡", "😠", "💔", "😩", "😫", "😤", "😖",
    "👎", "😨", "😰", "😱", "🙁", "☹️", "☹", "😪", "😓",
}

_TEXT_EMOTICON_RE = re.compile(
    r"(T[_.]?T|:'\(|:\(+|:\)+|\^_?\^|555+)", re.IGNORECASE
)
_NEGATIVE_TEXT_EMOTICONS = {":(", ":((", ":((("}
_POSITIVE_TEXT_EMOTICONS = {":)", ":))", ":)))", "^_^", "^^"}

POS_TAG = "[pos_emoji]"
NEG_TAG = "[neg_emoji]"
NEUTRAL_TAG = "[emoji]"


def _classify_emoji(char: str) -> str:
    if char in _POSITIVE_EMOJI:
        return POS_TAG
    if char in _NEGATIVE_EMOJI:
        return NEG_TAG
    return NEUTRAL_TAG


def _classify_text_emoticon(token: str) -> str:
    lowered = token.lower()
    if lowered.startswith("t_t") or lowered.startswith("t.t") or lowered == "5555" or lowered.startswith("555"):
        return NEG_TAG if "t" in lowered else POS_TAG  # "555..." reads as laughter
    if lowered in _POSITIVE_TEXT_EMOTICONS or lowered.startswith("^"):
        return POS_TAG
    if lowered in _NEGATIVE_TEXT_EMOTICONS:
        return NEG_TAG
    return NEUTRAL_TAG


def map_emoji_sentiment(text: str) -> str:
    """Replace emoji and common text emoticons with sentiment tokens."""
    text = text.replace("️", "").replace("︎", "")  # variation selectors
    text = _TEXT_EMOTICON_RE.sub(lambda m: _classify_text_emoticon(m.group(0)), text)
    mapped = "".join(_classify_emoji(ch) if emoji_lib.is_emoji(ch) else ch for ch in text)
    return mapped.replace("][", "] [")
