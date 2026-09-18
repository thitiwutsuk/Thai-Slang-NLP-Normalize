from src.normalize import normalize_thai


def test_empty_text():
    result = normalize_thai("")
    assert result == {
        "text": "",
        "normalized_text": "",
        "tokens": [],
        "corrections": [],
    }


def test_returns_expected_keys():
    result = normalize_thai("สวัสดีครับ")
    assert set(result.keys()) == {"text", "normalized_text", "tokens", "corrections"}


def test_elongation_and_slang_and_emoji_combined():
    result = normalize_thai("อาหารช้ามากกกกก มั้ยอ่ะ เค้าไม่ชอบ 😭😭😭")

    assert "มากกกกก" not in result["normalized_text"]
    assert "ไหม" in result["tokens"]
    assert "เขา" in result["tokens"]
    assert "[neg_emoji]" in result["normalized_text"]

    types = {c["type"] for c in result["corrections"]}
    assert types == {"elongation", "emoji", "slang_dict"}


def test_no_corrections_for_already_clean_text():
    result = normalize_thai("สวัสดีครับ")
    assert result["corrections"] == []
