"""Regex-free span algebra for the Redact pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from ._utf16 import UTF16Text


@dataclass
class Span:
    """A mutable span with UTF-16 offsets."""

    start: int
    end: int
    label: str
    score: float = 1.0


_DET_OWNED = frozenset(
    {
        "SSN",
        "CREDIT_CARD",
        "EMAIL",
        "URL",
        "IP_ADDRESS",
        "BANK_ACCOUNT",
        "ROUTING_NUMBER",
    }
)
_NAME_FAMILIES = frozenset({"GIVEN_NAME", "SURNAME"})
_CONNECT = frozenset({"-", "'", "\u2019"})


def bioes_to_spans(tags: list[str], offsets: list[tuple[int, int]]) -> list[Span]:
    """Decode BIOES tags and UTF-16 token offsets into spans."""

    output: list[Span] = []
    label: str | None = None
    start: int | None = None
    end: int | None = None

    def close() -> None:
        nonlocal label, start, end
        if label is not None and start is not None and end is not None and end > start:
            output.append(Span(start, end, label))
        label = start = end = None

    for tag, (token_start, token_end) in zip(tags, offsets, strict=True):
        if token_end <= token_start:
            continue
        if tag == "O":
            close()
            continue
        if "-" not in tag:
            close()
            continue
        prefix, token_label = tag.split("-", 1)
        if prefix == "S":
            close()
            output.append(Span(token_start, token_end, token_label))
        elif prefix == "B":
            close()
            label, start, end = token_label, token_start, token_end
        elif prefix == "I":
            if label == token_label:
                end = token_end
            else:
                close()
                label, start, end = token_label, token_start, token_end
        elif prefix == "E":
            if label == token_label:
                end = token_end
                close()
            else:
                close()
                output.append(Span(token_start, token_end, token_label))
        else:
            close()
    close()
    return output


def _ordered(spans: list[Span]) -> list[Span]:
    return sorted(
        spans, key=lambda span: (span.start, -(span.end - span.start), span.label)
    )


def merge_same_label(spans: list[Span]) -> list[Span]:
    """Merge overlapping or adjacent spans with the same label."""

    output: list[Span] = []
    for span in _ordered(spans):
        if output and span.label == output[-1].label and span.start <= output[-1].end:
            output[-1].end = max(output[-1].end, span.end)
        elif not output or span.start >= output[-1].end:
            output.append(span)
    return output


def merge_priority(spans: list[Span]) -> list[Span]:
    """Greedily merge spans, preferring deterministic-owned labels."""

    output: list[Span] = []
    for span in _ordered(spans):
        if output and span.label == output[-1].label and span.start <= output[-1].end:
            output[-1].end = max(output[-1].end, span.end)
        elif not output or span.start >= output[-1].end:
            output.append(span)
        else:
            current = output[-1]
            current_priority = (
                _DET_OWNED.__contains__(current.label),
                current.end - current.start,
            )
            span_priority = (_DET_OWNED.__contains__(span.label), span.end - span.start)
            if span_priority > current_priority:
                output[-1] = span
    return output


def _snap_one(text: UTF16Text, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(0, min(end, len(text)))
    while start > 0:
        if text.is_word_char_at(start - 1) or (
            text.scalar_at(start - 1) in _CONNECT
            and start - 2 >= 0
            and text.is_word_char_at(start - 2)
        ):
            start -= 1
        else:
            break
    while end < len(text):
        if text.is_word_char_at(end) or (
            text.scalar_at(end) in _CONNECT
            and end + 1 < len(text)
            and text.is_word_char_at(end + 1)
        ):
            end += 1
        else:
            break
    return start, end


def snap_spans(text: UTF16Text, spans: list[Span]) -> list[Span]:
    """Expand spans across adjacent word and connector characters."""

    snapped = [
        Span(*_snap_one(text, span.start, span.end), span.label, span.score)
        for span in spans
    ]
    return merge_same_label(snapped)


# Swift-style aliases for callers porting directly from the reference.
bioesToSpans = bioes_to_spans
mergeSameLabel = merge_same_label
mergePriority = merge_priority
snapSpans = snap_spans
