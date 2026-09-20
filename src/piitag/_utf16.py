"""UTF-16 code-unit indexed string helpers."""

from __future__ import annotations

import struct
import unicodedata

import regex  # type: ignore[import-untyped]

_ALPHABETIC = regex.compile(r"\A\p{Alphabetic}\Z")
_NUMERIC = regex.compile(r"\A\p{Numeric_Type=Numeric}\Z")
_WHITESPACE = regex.compile(r"\A\p{White_Space}\Z")


class UTF16Text:
    """UTF-16 code-unit-indexed view over a Python string."""

    def __init__(self, text: str) -> None:
        self.string = text
        encoded = text.encode("utf-16-le")
        self._units = list(struct.unpack(f"<{len(encoded) // 2}H", encoded))

    def __len__(self) -> int:
        """Return the number of UTF-16 code units."""
        return len(self._units)

    def scalar_at(self, i: int) -> str | None:
        """Return the scalar represented by one UTF-16 code unit."""
        if i < 0 or i >= len(self._units):
            return None
        return chr(self._units[i])

    def slice(self, a: int, b: int) -> str:
        """Return the clamped UTF-16 slice from ``a`` to ``b``."""
        lo = max(0, min(a, len(self._units)))
        hi = max(0, min(b, len(self._units)))
        if hi <= lo:
            return ""
        raw = struct.pack(f"<{hi - lo}H", *self._units[lo:hi])
        return raw.decode("utf-16-le", errors="replace")

    def find(self, needle: str, start: int) -> tuple[int, int] | None:
        """Find the first literal occurrence at or after a UTF-16 offset."""
        target = list(
            struct.unpack(
                f"<{len(needle.encode('utf-16-le')) // 2}H",
                needle.encode("utf-16-le"),
            )
        )
        if not target:
            return None
        first = max(0, start)
        last = len(self._units) - len(target)
        if first > last:
            return None
        for i in range(first, last + 1):
            if self._units[i : i + len(target)] == target:
                return i, i + len(target)
        return None

    def is_whitespace_at(self, i: int) -> bool:
        """Return whether the scalar at ``i`` has Unicode White_Space."""
        scalar = self.scalar_at(i)
        return scalar is not None and bool(_WHITESPACE.fullmatch(scalar))

    def is_word_char_at(self, i: int) -> bool:
        """Return whether the scalar at ``i`` is a Swift word character."""
        scalar = self.scalar_at(i)
        if scalar is None:
            return False
        if _ALPHABETIC.fullmatch(scalar) or _NUMERIC.fullmatch(scalar):
            return True
        return unicodedata.category(scalar) in {
            "Nd",
            "Nl",
            "No",
            "Mn",
            "Mc",
            "Me",
        }
