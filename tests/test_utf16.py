"""Tests for UTF-16 offsets."""

from piitag._utf16 import UTF16Text


def test_astral_character_uses_two_offsets() -> None:
    text = UTF16Text(
        "😀 call +34 600 100 200 or email me@x.com "
        "iban DE89370400440532013000 card 4539 1488 0343 6467"
    )

    assert len(text) == len(text.string) + 1
    assert text.find("+34 600 100 200", 0) == (8, 23)
    assert text.find("me@x.com", 23) == (33, 41)
    assert text.find("DE89370400440532013000", 41) == (47, 69)
    assert text.find("4539 1488 0343 6467", 69) == (75, 94)


def test_slice_decodes_surrogate_pairs_and_clamps_bounds() -> None:
    text = UTF16Text("A😀B")

    assert text.slice(-10, 1) == "A"
    assert text.slice(1, 3) == "😀"
    assert text.slice(3, 20) == "B"
    assert text.slice(4, 1) == ""


def test_scalar_at_is_code_unit_indexed() -> None:
    text = UTF16Text("A😀B")

    assert text.scalar_at(-1) is None
    assert text.scalar_at(0) == "A"
    assert text.scalar_at(1) == "\ud83d"
    assert text.scalar_at(2) == "\ude00"
    assert text.scalar_at(4) is None


def test_unicode_whitespace_and_word_categories() -> None:
    text = UTF16Text("A ٣Ⅻ²\u0301\u200b")

    assert text.is_whitespace_at(1)
    assert not text.is_whitespace_at(2)
    assert not text.is_whitespace_at(0)
    assert text.is_word_char_at(0)
    assert text.is_word_char_at(2)  # Arabic-Indic decimal number.
    assert text.is_word_char_at(3)  # Roman numeral letter number.
    assert text.is_word_char_at(4)  # Superscript two, other number.
    assert text.is_word_char_at(5)  # Combining acute accent.
    assert not text.is_word_char_at(6)
    assert not text.is_word_char_at(-1)


def test_find_handles_negative_start_and_missing_values() -> None:
    text = UTF16Text("abc abc")

    assert text.find("abc", -4) == (0, 3)
    assert text.find("abc", 1) == (4, 7)
    assert text.find("", 0) is None
    assert text.find("xyz", 0) is None
    assert text.find("abc", 8) is None
