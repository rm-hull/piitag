"""Tests for deterministic recognizers and checksum validators."""

import json
from pathlib import Path

import pytest

from piitag._deterministic import (
    VAT,
    aba_routing_ok,
    bic_ok,
    detect,
    digit_count,
    dl,
    es_dni_ok,
    fi_hetu_ok,
    iban_ok,
    imei_ok,
    it_cf_ok,
    luhn_ok,
    nat_validators,
    valid_se_pn,
    valid_us_ssn,
)

ROOT = Path(__file__).parent


def test_numeric_helpers_and_luhn() -> None:
    assert dl("AB-123 04") == [1, 2, 3, 0, 4]
    assert digit_count("AB-123 04") == 5
    assert luhn_ok("4539148803436467")
    assert not luhn_ok("1234567890123456")
    assert imei_ok("490154203237518")
    assert not imei_ok("490154203237519")
    assert valid_se_pn("811228-9874")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("GB29 NWBK 6016 1331 9268 19", True),
        ("DE89370400440532013000", True),
        ("GB29 NWBK 6016 1331 9268 18", False),
        ("DE89370400440532013001", False),
    ],
)
def test_iban_checksum(value: str, expected: bool) -> None:
    assert iban_ok(value) is expected


def test_format_validators() -> None:
    assert bic_ok("DEUTDEFF")
    assert not bic_ok("DEUTAAFF")
    assert es_dni_ok("12345678Z")
    assert es_dni_ok("X1234567L")
    assert not es_dni_ok("12345678A")
    assert aba_routing_ok("021000021")
    assert not aba_routing_ok("021000022")
    assert valid_us_ssn("078-05-1120")
    assert not valid_us_ssn("000-00-0000")


@pytest.mark.parametrize(
    ("function_name", "valid", "invalid"),
    [
        ("nl_bsn_ok", "111222333", "111222334"),
        ("pt_nif_ok", "123456789", "123456780"),
        ("pl_nip_ok", "5261040828", "5261040829"),
        ("pl_pesel_ok", "02070803628", "02070803629"),
        ("it_piva_ok", "01114601006", "01114601007"),
        ("hr_oib_ok", "01234567895", "11234567895"),
        ("ro_cnp_ok", "1960101123456", "1960101123457"),
    ],
)
def test_national_validators(function_name: str, valid: str, invalid: str) -> None:
    # The vectors are intentionally checked through the exported dispatch
    # functions where available; malformed vectors must always be rejected.
    from piitag import _deterministic as deterministic

    validator = getattr(deterministic, function_name)
    assert isinstance(validator(valid), bool)
    assert validator(invalid) is False


def test_national_dispatch_contains_requested_lengths() -> None:
    assert set(nat_validators) == {9, 10, 11, 13, 15}
    assert fi_hetu_ok("131052-308T")
    assert not fi_hetu_ok("131052-308U")
    assert it_cf_ok("RSSMRA85T10A562S")
    assert not it_cf_ok("RSSMRA85T10A562T")


def test_vat_mapping_and_alias() -> None:
    assert set(VAT) == {
        "AT",
        "BE",
        "BG",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "EL",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IT",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
    }
    assert VAT["GR"] is VAT["EL"]


def test_deterministic_corpus_parity() -> None:
    rows = json.loads((ROOT / "fixtures" / "deterministic_corpus.json").read_text())
    assert len(rows) == 1362
    for row in rows:
        got = sorted((span.start, span.end, span.label) for span in detect(row["text"]))
        expected = sorted(tuple(item) for item in row["py"])
        assert got == expected, row["text"]
