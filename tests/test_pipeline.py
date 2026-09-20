"""Tests for Phase 5a span post-processing."""

from piitag._pipeline import (
    Span,
    attach_building_numbers,
    attach_state_codes,
    bioes_to_spans,
    bridge_name_gaps,
    clean_spans,
    hysteresis,
    mask_text,
    merge_priority,
    merge_same_label,
    model_input,
    redact_secondary_address,
    redact_us_street,
    relabel_by_context,
    resolve,
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


def test_casing_helpers_normalize_shouting_without_changing_german_length() -> None:
    assert model_input("JOHN DOE") == "John Doe"
    assert model_input("STRAẞE") == "Straße"


def test_us_address_helpers_add_structured_spans() -> None:
    text = UTF16Text("Send to 12 Main Street, Austin, TX 78701, Apt 4.")
    spans = redact_us_street(text, [])
    spans = attach_state_codes(text, spans)
    spans = redact_secondary_address(text, spans)
    assert spans == [
        Span(8, 10, "BUILDING_NUMBER"),
        Span(11, 22, "STREET_NAME"),
        Span(32, 34, "STATE"),
        Span(35, 40, "ZIP_CODE"),
        Span(42, 47, "SECONDARY_ADDRESS"),
    ]


def test_building_number_can_be_attached_to_existing_street() -> None:
    text = UTF16Text("12 Main Street")
    assert attach_building_numbers(text, [Span(3, 14, "STREET_NAME")]) == [
        Span(0, 2, "BUILDING_NUMBER"),
        Span(3, 14, "STREET_NAME"),
    ]


def test_context_relabeling_supports_account_and_german_license_terms() -> None:
    text = "account 123456 and führerschein ABC123"
    assert relabel_by_context(
        text,
        [Span(8, 14, "PHONE"), Span(34, 40, "GOVERNMENT_ID")],
    ) == [Span(8, 14, "BANK_ACCOUNT"), Span(34, 40, "DRIVERS_LICENSE")]


def test_clean_spans_removes_titles_and_edge_punctuation() -> None:
    assert clean_spans("Dr. Jean,", [Span(0, 9, "GIVEN_NAME")]) == [
        Span(4, 8, "GIVEN_NAME")
    ]
    assert clean_spans("Dr.", [Span(0, 3, "GIVEN_NAME")]) == []


def test_name_bridging_and_hysteresis_are_fixed_point_operations() -> None:
    text = UTF16Text("Jean de la Cruz")
    spans = [Span(0, 4, "GIVEN_NAME"), Span(10, 14, "SURNAME")]
    assert bridge_name_gaps(text, spans) == [Span(0, 14, "GIVEN_NAME")]
    scored = [(Span(0, 4, "GIVEN_NAME"), 0.9), (Span(5, 7, "GIVEN_NAME"), 0.2)]
    assert hysteresis(text, scored, 0.8) == [
        Span(0, 4, "GIVEN_NAME"),
        Span(5, 7, "GIVEN_NAME"),
    ]


def test_resolve_prefers_deterministic_conflicts_and_masks_newlines() -> None:
    assert resolve(
        [Span(0, 5, "EMAIL")],
        [Span(0, 5, "EMAIL"), Span(6, 9, "GIVEN_NAME")],
    ) == [Span(0, 5, "EMAIL"), Span(6, 9, "GIVEN_NAME")]
    assert mask_text("a😀\nb", [Span(1, 3, "EMAIL")]) == "a  \nb"
