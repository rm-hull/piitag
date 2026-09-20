"""Tests for the Redact unigram tokenizer."""

import struct
from pathlib import Path

import pytest

from piitag._tokenizer import Tokenizer, normalize

ROOT = Path(__file__).parent
TOKENIZER_PATH = ROOT / "fixtures" / "model" / "redact_tokenizer.bin"


def tokenizer_bytes(
    pieces: list[tuple[str, float]],
    *,
    unk_id: int = 0,
    bos_id: int = 1,
    eos_id: int = 2,
) -> bytes:
    """Build a minimal Redact tokenizer container for parser tests."""

    data = bytearray(b"RDTK\x01")
    data.extend(struct.pack("<4i", unk_id, bos_id, eos_id, len(pieces)))
    data.extend(struct.pack(f"<{len(pieces)}f", *(score for _, score in pieces)))
    encoded = [piece.encode("utf-8") for piece, _ in pieces]
    data.extend(struct.pack(f"<{len(encoded)}H", *(len(piece) for piece in encoded)))
    for piece in encoded:
        data.extend(piece)
    return bytes(data)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return Tokenizer(TOKENIZER_PATH.read_bytes())


def test_tokenizer_matches_expected_viterbi_path() -> None:
    tokenizer = Tokenizer(
        tokenizer_bytes(
            [
                ("<unk>", -10.0),
                ("<s>", 0.0),
                ("</s>", 0.0),
                ("▁", -1.0),
                ("▁a", -1.0),
                ("b", -1.0),
                ("▁ab", -0.1),
                ("▁b", -0.1),
            ]
        )
    )

    assert tokenizer.encode("ab") == [6]
    assert [token.scalars for token in tokenizer.tokenize("a b")] == [
        ("▁", "a"),
        ("▁", "b"),
    ]


def test_normalization_is_plain_nfkc() -> None:
    assert normalize("\uff21  \u00a0B") == "A   B"
    assert normalize("\uff5e") == "~"
    assert normalize("a\x01b") == "a\x01b"


def test_whitespace_is_trimmed_and_collapsed_during_tokenization() -> None:
    tokenizer = Tokenizer(
        tokenizer_bytes(
            [
                ("<unk>", -10.0),
                ("<s>", 0.0),
                ("</s>", 0.0),
                ("▁a", -0.1),
                ("▁b", -0.1),
            ]
        )
    )

    assert tokenizer.encode("  a   b  ") == [3, 4]


def test_byte_distinct_pieces_are_kept_distinct() -> None:
    tokenizer = Tokenizer(
        tokenizer_bytes(
            [
                ("<unk>", -10.0),
                ("<s>", 0.0),
                ("</s>", 0.0),
                ("▁x\u0301", -0.1),
                ("▁x\u0323", -0.2),
            ]
        )
    )

    assert tokenizer.encode("x\u0301") == [3]
    assert tokenizer.encode("x\u0323") == [4]


@pytest.mark.parametrize(
    "data",
    [
        b"RDTK",
        b"RDTK\x01" + b"\0" * 16,
        b"GSTK\x01" + b"\0" * 16,
    ],
)
def test_truncated_or_wrong_magic_tokenizer_is_rejected(data: bytes) -> None:
    with pytest.raises(ValueError):
        Tokenizer(data)


def test_downloaded_tokenizer_produces_tokens() -> None:
    tokenizer = Tokenizer(TOKENIZER_PATH.read_bytes())

    tokens = tokenizer.tokenize("😀 call +34 600 100 200")

    assert tokens
    assert all(token.id >= 0 for token in tokens)
    assert all(token.scalars for token in tokens)
