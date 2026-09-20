"""Tests for Phase 5a span post-processing."""

from piitag._pipeline import (
    Span,
    bioes_to_spans,
    merge_priority,
    merge_same_label,
    snap_spans,
)
from piitag._utf16 import UTF16Text


def test_bioes_decoding_handles_all_prefixes_and_malformed_transitions() -> None:
    offsets = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
    tags = ["B-NAME", "I-NAME", "E-NAME", "S-EMAIL", "B-X", "E-Y", "O"]

    assert bioes_to_spans(tags, offsets) == [
        Span(0, 5, "NAME"),
        Span(6, 7, "EMAIL"),
        Span(8, 9, "X"),
        Span(10, 11, "Y"),
    ]


def test_bioes_invalid_tag_closes_open_span_and_zero_width_is_skipped() -> None:
    assert bioes_to_spans(
        ["B-X", "I-X", "BAD", "S-Y"],
        [(0, 2), (2, 4), (4, 4), (5, 6)],
    ) == [Span(0, 4, "X"), Span(5, 6, "Y")]


def test_merge_same_label_merges_overlaps_and_adjacent_spans() -> None:
    assert merge_same_label(
        [
            Span(5, 10, "X"),
            Span(0, 5, "X"),
            Span(3, 7, "Y"),
            Span(12, 14, "X"),
        ]
    ) == [Span(0, 10, "X"), Span(12, 14, "X")]


def test_merge_priority_prefers_deterministic_label_then_length() -> None:
    assert merge_priority(
        [
            Span(0, 10, "GIVEN_NAME"),
            Span(2, 8, "EMAIL"),
            Span(12, 15, "SURNAME"),
            Span(12, 18, "GIVEN_NAME"),
        ]
    ) == [Span(2, 8, "EMAIL"), Span(12, 18, "GIVEN_NAME")]


def test_snap_spans_expands_words_and_internal_connectors() -> None:
    text = UTF16Text("Jean-Pierre arrived")

    assert snap_spans(text, [Span(1, 4, "GIVEN_NAME")]) == [Span(0, 11, "GIVEN_NAME")]


def test_snap_spans_preserves_apostrophe_connector() -> None:
    text = UTF16Text("van der Berg")

    assert snap_spans(text, [Span(4, 7, "SURNAME")]) == [Span(4, 7, "SURNAME")]
