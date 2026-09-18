from src.slang_dict import slang_lookup


def test_known_slang_mapped():
    assert slang_lookup("มั้ย") == "ไหม"
    assert slang_lookup("เค้า") == "เขา"


def test_unknown_token_passthrough():
    assert slang_lookup("อาหาร") == "อาหาร"
    assert slang_lookup("xyz123") == "xyz123"


def test_empty_token():
    assert slang_lookup("") == ""
