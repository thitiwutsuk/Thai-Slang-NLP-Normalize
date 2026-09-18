from src.emoji_map import map_emoji_sentiment, NEG_TAG, POS_TAG


def test_empty_text():
    assert map_emoji_sentiment("") == ""


def test_positive_emoji():
    assert map_emoji_sentiment("😍") == POS_TAG


def test_negative_emoji():
    assert map_emoji_sentiment("😭") == NEG_TAG


def test_variation_selector_stripped():
    assert map_emoji_sentiment("❤️") == POS_TAG


def test_consecutive_emoji_kept_as_separate_tokens():
    assert map_emoji_sentiment("😭😭😭") == f"{NEG_TAG} {NEG_TAG} {NEG_TAG}"


def test_text_emoticon_crying():
    assert map_emoji_sentiment("เสียใจ T_T") == f"เสียใจ {NEG_TAG}"


def test_text_emoticon_laughter():
    assert map_emoji_sentiment("ฮามาก 5555") == f"ฮามาก {POS_TAG}"


def test_non_emoji_text_untouched():
    assert map_emoji_sentiment("สวัสดีครับ") == "สวัสดีครับ"
