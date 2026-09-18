from src.elongation import reduce_elongation


def test_empty_text():
    assert reduce_elongation("") == ""


def test_trailing_elongation():
    assert reduce_elongation("อร่อยยยยย") == "อร่อย"


def test_interior_elongation():
    assert reduce_elongation("ชอบบบบมาก") == "ชอบมาก"


def test_maiyamok_run_collapsed():
    assert reduce_elongation("จริงๆๆๆๆๆ") == "จริงๆ"


def test_digits_untouched():
    assert reduce_elongation("555") == "555"
    assert reduce_elongation("อร่อย 555555") == "อร่อย 555555"


def test_emoji_run_untouched():
    assert reduce_elongation("😭😭😭") == "😭😭😭"


def test_punctuation_run_collapsed():
    assert reduce_elongation("เย้!!!!") == "เย้!"
