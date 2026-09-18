"""Phase 2: normalize_thai() -- combine cleaning, elongation reduction,
emoji/text-emoticon mapping, tokenization, and slang dictionary lookup."""

from src.cleaning import clean_text
from src.elongation import reduce_elongation
from src.emoji_map import map_emoji_sentiment
from src.slang_dict import slang_lookup
from src.tokenization import tokenize_text


def normalize_thai(text: str) -> dict:
    cleaned = clean_text(text)
    deelongated = reduce_elongation(cleaned)
    emoji_mapped = map_emoji_sentiment(deelongated)

    corrections = []
    if cleaned != deelongated:
        corrections.append(
            {"original": cleaned, "normalized": deelongated, "type": "elongation"}
        )
    if deelongated != emoji_mapped:
        corrections.append(
            {"original": deelongated, "normalized": emoji_mapped, "type": "emoji"}
        )

    tokens = []
    for token in tokenize_text(emoji_mapped):
        normalized_token = slang_lookup(token)
        tokens.append(normalized_token)
        if normalized_token != token:
            corrections.append(
                {"original": token, "normalized": normalized_token, "type": "slang_dict"}
            )

    return {
        "text": text,
        "normalized_text": " ".join(tokens),
        "tokens": tokens,
        "corrections": corrections,
    }
