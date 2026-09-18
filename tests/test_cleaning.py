from src.cleaning import clean_text


def test_empty_text():
    assert clean_text("") == ""


def test_url_only():
    assert clean_text("https://example.com/path?x=1") == ""


def test_url_inline():
    assert clean_text("ดูที่นี่ https://example.com ครับ") == "ดูที่นี่ ครับ"


def test_www_url():
    assert clean_text("เว็บ www.example.com นะ") == "เว็บ นะ"


def test_mention_only():
    assert clean_text("@some_user") == ""


def test_mention_inline():
    assert clean_text("ขอบคุณ @shopee นะคะ") == "ขอบคุณ นะคะ"


def test_hashtag_symbol_removed_word_kept():
    assert clean_text("#ช้างเอฟเอคัพ นัดชิงชนะเลิศ") == "ช้างเอฟเอคัพ นัดชิงชนะเลิศ"


def test_numbers_only():
    assert clean_text("12345") == "12345"


def test_emoji_only_untouched():
    assert clean_text("😭😭😭") == "😭😭😭"


def test_collapses_whitespace():
    assert clean_text("  หวัดดี   ครับ  \n\n ผม  ") == "หวัดดี ครับ ผม"


def test_mixed_url_mention_hashtag():
    text = "แจ้ง @shopee ดูที่ https://x.co/y ครับ #รีวิว"
    assert clean_text(text) == "แจ้ง ดูที่ ครับ รีวิว"
