from src.tokenization import tokenize_text


def test_empty_text():
    assert tokenize_text("") == []


def test_numbers_only():
    assert tokenize_text("12345") == ["12345"]


def test_emoji_only():
    tokens = tokenize_text("😭😭😭")
    assert tokens
    assert "".join(tokens) == "😭😭😭"


def test_thai_sentence():
    tokens = tokenize_text("ฉันชอบกินข้าว")
    assert tokens == ["ฉัน", "ชอบ", "กินข้าว"]


def test_no_whitespace_tokens():
    tokens = tokenize_text("สวัสดี ครับ")
    assert all(t.strip() for t in tokens)
    assert tokens == ["สวัสดี", "ครับ"]
