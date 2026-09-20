"""SentencePiece-style unigram tokenizer used by the Redact model."""

from __future__ import annotations

import struct
import unicodedata
from dataclasses import dataclass

_METASPACE = "\u2581"


@dataclass(frozen=True)
class Token:
    """A token ID and its normalized Unicode scalars."""

    id: int
    scalars: tuple[str, ...]


def normalize(text: str) -> str:
    """Apply Redact's NFKC normalization and whitespace handling."""

    return unicodedata.normalize("NFKC", text)


class Tokenizer:
    """Parse and run the compact Redact unigram tokenizer."""

    def __init__(self, data: bytes) -> None:
        if len(data) < 21 or data[:4] != b"RDTK":
            raise ValueError("invalid tokenizer header")

        offset = 5
        try:
            unk_id, bos_id, eos_id, count = struct.unpack_from("<4i", data, offset)
        except struct.error as error:
            raise ValueError("truncated tokenizer header") from error
        offset += 16
        if count <= 0 or count > (len(data) - offset) // 6:
            raise ValueError("invalid tokenizer vocabulary count")

        score_bytes = count * 4
        if offset + score_bytes > len(data):
            raise ValueError("truncated tokenizer scores")
        scores = list(struct.unpack_from(f"<{count}f", data, offset))
        offset += score_bytes

        length_bytes = count * 2
        if offset + length_bytes > len(data):
            raise ValueError("truncated tokenizer piece lengths")
        lengths = struct.unpack_from(f"<{count}H", data, offset)
        offset += length_bytes

        pieces: dict[bytes, int] = {}
        maximum_length = 1
        for piece_id, length in enumerate(lengths):
            end = offset + length
            if end > len(data):
                raise ValueError("truncated tokenizer piece")
            piece = data[offset:end]
            offset = end
            if piece in pieces:
                raise ValueError("duplicate tokenizer piece")
            pieces[piece] = piece_id
            maximum_length = max(
                maximum_length,
                sum(byte & 0xC0 != 0x80 for byte in piece),
            )

        if offset != len(data):
            raise ValueError("unexpected trailing tokenizer data")
        if not all(0 <= token_id < count for token_id in (unk_id, bos_id, eos_id)):
            raise ValueError("tokenizer special ID is out of range")

        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id
        self._scores = scores
        self._pieces = pieces
        self._max_length = min(maximum_length, 32)
        self._unknown_penalty = min(scores) - 10.0

    def encode(self, text: str) -> list[int]:
        """Return content-subword IDs for ``text``."""

        return [token.id for token in self.tokenize(text)]

    def tokenize(self, text: str) -> list[Token]:
        """Return Viterbi-optimal tokens for ``text``."""

        nfkc = normalize(text)
        squeezed: list[str] = []
        last_was_space = True
        for character in nfkc:
            if character == " ":
                if last_was_space:
                    continue
                last_was_space = True
            else:
                last_was_space = False
            squeezed.append(character)
        if squeezed and squeezed[-1] == " ":
            squeezed.pop()

        scalar_text = [
            _METASPACE if character == " " else character for character in squeezed
        ]
        scalar_text.insert(0, _METASPACE)
        length = len(scalar_text)

        best = [-1e18] * (length + 1)
        best[0] = 0.0
        back_positions = [-1] * (length + 1)
        back_ids = [-1] * (length + 1)

        for end in range(1, length + 1):
            start_limit = max(0, end - self._max_length)
            for start in range(start_limit, end):
                piece = "".join(scalar_text[start:end]).encode("utf-8")
                token_id = self._pieces.get(piece)
                if token_id is None:
                    continue
                score = best[start] + float(self._scores[token_id])
                if score > best[end]:
                    best[end] = score
                    back_positions[end] = start
                    back_ids[end] = token_id

            unknown_score = best[end - 1] + self._unknown_penalty
            if unknown_score > best[end]:
                best[end] = unknown_score
                back_positions[end] = end - 1
                back_ids[end] = self.unk_id

        tokens: list[Token] = []
        position = length
        while position > 0:
            start = back_positions[position]
            tokens.append(Token(back_ids[position], tuple(scalar_text[start:position])))
            position = start
        tokens.reverse()
        return tokens
