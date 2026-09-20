"""Checksum validators used by the deterministic recognizers."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import regex as re  # type: ignore[import-untyped]

from ._utf16 import UTF16Text


def dl(value: str) -> list[int]:
    """Return ASCII decimal digits from ``value``."""
    return [int(char) for char in value if "0" <= char <= "9"]


def digit_count(value: str) -> int:
    """Count ASCII decimal digits in ``value``."""
    return sum("0" <= char <= "9" for char in value)


def wsum(digits: Iterable[int], weights: Iterable[int]) -> int:
    """Return the pairwise weighted sum."""
    return sum(digit * weight for digit, weight in zip(digits, weights))


def luhn_len(digits: list[int]) -> bool:
    """Validate a Luhn checksum for an already parsed digit list."""
    parity = len(digits) % 2
    total = 0
    for index, digit in enumerate(digits):
        value = digit * 2 if index % 2 == parity else digit
        total += value - 9 if value > 9 else value
    return total % 10 == 0


def luhn_ok(value: str) -> bool:
    digits = dl(value)
    return 13 <= len(digits) <= 19 and luhn_len(digits)


def imei_ok(value: str) -> bool:
    digits = dl(value)
    return len(digits) == 15 and luhn_len(digits)


def valid_se_pn(value: str) -> bool:
    digits = dl(value)
    if len(digits) == 12:
        digits = digits[2:]
    return len(digits) == 10 and luhn_len(digits)


def iban_ok(value: str) -> bool:
    lengths = {
        "AD": 24,
        "AE": 23,
        "AL": 28,
        "AT": 20,
        "AZ": 28,
        "BA": 20,
        "BE": 16,
        "BG": 22,
        "BH": 22,
        "BR": 29,
        "BY": 28,
        "CH": 21,
        "CR": 22,
        "CY": 28,
        "CZ": 24,
        "DE": 22,
        "DK": 18,
        "DO": 28,
        "EE": 20,
        "EG": 29,
        "ES": 24,
        "FI": 18,
        "FO": 18,
        "FR": 27,
        "GB": 22,
        "GE": 22,
        "GI": 23,
        "GL": 18,
        "GR": 27,
        "GT": 28,
        "HR": 21,
        "HU": 28,
        "IE": 22,
        "IL": 23,
        "IS": 26,
        "IT": 27,
        "JO": 30,
        "KW": 30,
        "KZ": 20,
        "LB": 28,
        "LC": 32,
        "LI": 21,
        "LT": 20,
        "LU": 20,
        "LV": 21,
        "MC": 27,
        "MD": 24,
        "ME": 22,
        "MK": 19,
        "MR": 27,
        "MT": 31,
        "MU": 30,
        "NL": 18,
        "NO": 15,
        "PK": 24,
        "PL": 28,
        "PS": 29,
        "PT": 25,
        "QA": 29,
        "RO": 24,
        "RS": 22,
        "SA": 24,
        "SC": 31,
        "SE": 24,
        "SI": 19,
        "SK": 24,
        "SM": 27,
        "TN": 24,
        "TR": 26,
        "UA": 29,
        "VG": 24,
        "XK": 20,
    }
    compact = value.replace(" ", "").upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", compact):
        return False
    country = compact[:2]
    if len(compact) != lengths.get(country, -1):
        return False
    rearranged = compact[4:] + compact[:4]
    remainder = 0
    for char in rearranged:
        value_part = str(ord(char) - 55) if char.isalpha() else char
        for digit in value_part:
            remainder = (remainder * 10 + int(digit)) % 97
    return remainder == 1


def bic_ok(value: str) -> bool:
    return bool(
        value == value.upper()
        and re.fullmatch(r"[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?", value)
        and value[4:6] not in {"AA", "ZZ"}
    )


def es_dni_ok(value: str) -> bool:
    value = value.upper()
    if not re.fullmatch(r"(?:\d{8}|[XYZ]\d{7})[A-Z]", value):
        return False
    prefix = {"X": "0", "Y": "1", "Z": "2"}.get(value[0], "")
    number = int(prefix + value[1:8]) if prefix else int(value[:8])
    return value[-1] == "TRWAGMYFPDXBNJZSQVHLCKE"[number % 23]


def aba_routing_ok(value: str) -> bool:
    if len(value) != 9 or not value.isascii() or not value.isdigit():
        return False
    d = dl(value)
    return (
        3 * (d[0] + d[3] + d[6]) + 7 * (d[1] + d[4] + d[7]) + d[2] + d[5] + d[8]
    ) % 10 == 0


def valid_us_ssn(value: str) -> bool:
    match = re.fullmatch(r"(\d{3})[- ](\d{2})[- ](\d{4})", value)
    if not match:
        return False
    area, group, serial = match.groups()
    return (
        area not in {"000", "666"}
        and not 900 <= int(area) <= 999
        and group != "00"
        and serial != "0000"
    )


def nl_bsn_ok(value: str) -> bool:
    d = dl(value)
    return (
        len(d) == 9
        and any(d)
        and (wsum(d[:8], [9, 8, 7, 6, 5, 4, 3, 2]) - d[8]) % 11 == 0
    )


def pt_nif_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 9 or d[0] not in {1, 2, 3, 5, 6, 8, 9}:
        return False
    check = 11 - wsum(d[:8], [9, 8, 7, 6, 5, 4, 3, 2]) % 11
    return (0 if check >= 10 else check) == d[8]


def bg_egn_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 10:
        return False
    check = wsum(d[:9], [2, 4, 8, 5, 10, 9, 7, 3, 6]) % 11
    check = 0 if check == 10 else check
    month = d[2] * 10 + d[3]
    month = (
        month - 20 if 21 <= month <= 32 else month - 40 if 41 <= month <= 52 else month
    )
    return check == d[9] and 1 <= month <= 12 and 1 <= d[4] * 10 + d[5] <= 31


def cz_rc_ok(value: str) -> bool:
    d = dl(value)
    month = d[2] * 10 + d[3] if len(d) >= 4 else 0
    return (
        len(d) == 10
        and any(month - x in range(1, 13) for x in (0, 20, 50, 70))
        and int("".join(map(str, d))) % 11 == 0
    )


def hu_adoaz_ok(value: str) -> bool:
    d = dl(value)
    return (
        len(d) == 10
        and d[0] == 8
        and (c := sum(d[i] * (i + 1) for i in range(9)) % 11) != 10
        and c == d[9]
    )


def pl_nip_ok(value: str) -> bool:
    d = dl(value)
    c = wsum(d[:9], [6, 5, 7, 2, 3, 4, 5, 6, 7]) % 11 if len(d) == 10 else 10
    return c != 10 and c == (d[9] if len(d) == 10 else -1)


def it_piva_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 11:
        return False
    even = sum(d[index] for index in range(0, 10, 2))
    odd = 0
    for index in range(1, 10, 2):
        doubled = d[index] * 2
        odd += doubled - 9 if doubled > 9 else doubled
    return (10 - (even + odd) % 10) % 10 == d[10]


def at_svnr_ok(value: str) -> bool:
    d = dl(value)
    weights = [3, 7, 9, 0, 5, 8, 4, 2, 1, 6]
    c = sum(d[i] * weights[i] for i in range(10) if i != 3) % 11 if len(d) == 10 else 10
    return c != 10 and c == (d[3] if len(d) == 10 else -1)


def pl_pesel_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 11:
        return False
    check = (10 - wsum(d[:10], [1, 3, 7, 9, 1, 3, 7, 9, 1, 3]) % 10) % 10
    return (
        check == d[10]
        and 1 <= (d[2] * 10 + d[3]) % 20 <= 12
        and 1 <= d[4] * 10 + d[5] <= 31
    )


def hr_oib_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 11:
        return False
    remainder = 10
    for digit in d[:10]:
        remainder = (remainder + digit) % 10 or 10
        remainder = (remainder * 2) % 11
    return (11 - remainder) % 10 == d[10]


def gr_amka_ok(value: str) -> bool:
    d = dl(value)
    return len(d) == 11 and 1 <= d[2] * 10 + d[3] <= 12 and luhn_len(d)


def ee_isikukood_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 11:
        return False
    check = wsum(d[:10], [1, 2, 3, 4, 5, 6, 7, 8, 9, 1]) % 11
    if check == 10:
        check = wsum(d[:10], [3, 4, 5, 6, 7, 8, 9, 1, 2, 3]) % 11
        if check == 10:
            check = 0
    return check == d[10]


def lv_pk_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 11 or d[0] == 3 or not 1 <= d[2] * 10 + d[3] <= 12:
        return False
    check = ((1 - wsum(d[:10], [1, 6, 3, 7, 9, 10, 5, 8, 4, 2])) % 11) % 10
    return check == d[10]


def be_rrn_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 11:
        return False
    base, check = int("".join(map(str, d[:9]))), int("".join(map(str, d[9:])))
    return (97 - base % 97) % 97 == check or (
        97 - (2_000_000_000 + base) % 97
    ) % 97 == check


def ro_cnp_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 13:
        return False
    check = wsum(d[:12], [2, 7, 9, 1, 4, 6, 3, 5, 8, 2, 7, 9]) % 11
    check = 1 if check == 10 else check
    return (
        check == d[12]
        and 1 <= d[0] <= 9
        and 1 <= d[3] * 10 + d[4] <= 12
        and 1 <= d[5] * 10 + d[6] <= 31
    )


def si_emso_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 13 or not 1 <= d[2] * 10 + d[3] <= 12:
        return False
    remainder = wsum(d[:12], [7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]) % 11
    check = 0 if remainder == 0 else 11 - remainder
    return check != 10 and check == d[12]


def fr_nir_ok(value: str) -> bool:
    d = dl(value)
    if len(d) != 15 or d[0] not in {1, 2}:
        return False
    check = 97 - int("".join(map(str, d[:13]))) % 97
    return d[13] * 10 + d[14] == (97 if check == 0 else check)


def ie_pps_ok(value: str) -> bool:
    match = re.fullmatch(r"(\d{7})([A-W])([A-W]?)", value.upper())
    if not match:
        return False
    digits, first, second = match.groups()
    total = sum(int(digit) * (8 - i) for i, digit in enumerate(digits))
    if second:
        total += (ord(second) - 64) * 9
    return "WABCDEFGHIJKLMNOPQRSTUV"[total % 23] == first


def it_cf_ok(value: str) -> bool:
    value = value.upper()
    if not re.fullmatch(r"[A-Z0-9]{16}", value):
        return False
    odd = dict(
        zip(
            "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            [
                1,
                0,
                5,
                7,
                9,
                13,
                15,
                17,
                19,
                21,
                1,
                0,
                5,
                7,
                9,
                13,
                15,
                17,
                19,
                21,
                2,
                4,
                18,
                20,
                11,
                3,
                6,
                8,
                12,
                14,
                16,
                10,
                22,
                25,
                24,
                23,
            ],
        )
    )
    total = 0
    for index, char in enumerate(value[:15]):
        total += (
            odd[char]
            if index % 2 == 0
            else (int(char) if char.isdigit() else ord(char) - 65)
        )
    return value[15] == chr(65 + total % 26)


def fi_hetu_ok(value: str) -> bool:
    value = value.replace(" ", "").upper()
    match = re.fullmatch(r"(\d{6})[-+A-F](\d{3})([0-9A-Y])", value)
    if not match:
        return False
    number = int(match.group(1) + match.group(2))
    return "0123456789ABCDEFHJKLMNPRSTUVWXY"[number % 31] == match.group(3)


nat_validators: dict[int, list[Callable[[str], bool]]] = {
    9: [nl_bsn_ok, pt_nif_ok, pl_nip_ok],
    10: [bg_egn_ok, cz_rc_ok, hu_adoaz_ok, at_svnr_ok, pl_nip_ok],
    11: [
        pl_pesel_ok,
        hr_oib_ok,
        ee_isikukood_ok,
        lv_pk_ok,
        be_rrn_ok,
        gr_amka_ok,
        it_piva_ok,
    ],
    13: [ro_cnp_ok, si_emso_ok],
    15: [fr_nir_ok],
}


def _pattern(pattern: str, value: str) -> bool:
    return bool(re.fullmatch(pattern, value))


def _luhn_weighted_sum(digits: list[int], double_even: bool = True) -> int:
    total = 0
    for index, digit in enumerate(digits):
        value = digit * (2 if (index % 2 == 0) == double_even else 1)
        total += value - 9 if value > 9 else value
    return total


def _vat_validators() -> dict[str, Callable[[str], bool]]:
    def vat_at(n: str) -> bool:
        d = dl(n)
        if not _pattern(r"U\d{8}", n):
            return False
        total = 4
        for index, digit in enumerate(d[:7]):
            value = digit * (2 if index % 2 == 1 else 1)
            total += value - 9 if value > 9 else value
        return (10 - total % 10) % 10 == d[7]

    def vat_de(n: str) -> bool:
        if not _pattern(r"\d{9}", n):
            return False
        d = dl(n)
        product = 10
        for digit in d[:8]:
            s = (digit + product) % 10 or 10
            product = (s * 2) % 11
        return (11 - product) % 10 == d[8]

    def vat_fi(n: str) -> bool:
        d = dl(n)
        if not _pattern(r"\d{8}", n):
            return False
        check = wsum(d[:7], [7, 9, 10, 5, 8, 4, 2]) % 11
        return check != 1 and (0 if check == 0 else 11 - check) == d[7]

    def vat_fr(n: str) -> bool:
        if not _pattern(r"\d{11}", n):
            return False
        siren = n[2:]
        digits = dl(siren)
        total = 0
        for i, digit in enumerate(reversed(digits)):
            x = digit * (2 if i % 2 else 1)
            total += x - 9 if x > 9 else x
        return total % 10 == 0 and int(n[:2]) == (12 + 3 * (int(siren) % 97)) % 97

    def vat_lt(n: str) -> bool:
        if not _pattern(r"\d{9}|\d{12}", n):
            return False
        d = dl(n)
        total = sum(d[i] * (i % 9 + 1) for i in range(len(d) - 1)) % 11
        if total == 10:
            total = sum(d[i] * (i % 9 + 3) for i in range(len(d) - 1)) % 11
            if total == 10:
                total = 0
        return total == d[-1]

    def vat_ro(n: str) -> bool:
        if not _pattern(r"\d{2,10}", n):
            return False
        body = dl(n[:-1])
        weights = [7, 5, 3, 2, 1, 7, 5, 3, 2][-len(body) :]
        check = (sum(d * w for d, w in zip(body, weights)) * 10) % 11
        return (0 if check == 10 else check) == int(n[-1])

    def vat_se(n: str) -> bool:
        if not _pattern(r"\d{10}01", n):
            return False
        d = dl(n[:10])
        total = _luhn_weighted_sum(d)
        return total % 10 == 0

    def vat_bg(n: str) -> bool:
        if _pattern(r"\d{9}", n):
            d = dl(n)
            check = sum(d[i] * (i + 1) for i in range(8)) % 11
            if check == 10:
                check = sum(d[i] * (i + 3) for i in range(8)) % 11
                if check == 10:
                    check = 0
            return check == d[8]
        return _pattern(r"\d{10}", n) and bg_egn_ok(n)

    def vat_el(n: str) -> bool:
        if not _pattern(r"\d{9}", n):
            return False
        d = dl(n)
        check = sum(d[i] * (1 << (8 - i)) for i in range(8)) % 11
        return (check if check < 10 else 0) == d[8]

    def vat_si(n: str) -> bool:
        if not _pattern(r"\d{8}", n):
            return False
        check = wsum(dl(n)[:7], [8, 7, 6, 5, 4, 3, 2]) % 11
        result = 0 if check == 0 else 11 - check
        return result != 10 and (0 if result == 11 else result) == dl(n)[7]

    return {
        "AT": vat_at,
        "BE": lambda n: (
            _pattern(r"0\d{9}", n) and (97 - int(n[:8]) % 97) == int(n[-2:])
        ),
        "BG": vat_bg,
        "CY": lambda n: (
            _pattern(r"\d{8}[A-Z]", n)
            and n[-1]
            == chr(
                65
                + sum(
                    ([1, 0, 5, 7, 9, 13, 15, 17, 19, 21][d] if i % 2 == 0 else d)
                    for i, d in enumerate(dl(n)[:8])
                )
                % 26
            )
        ),
        "CZ": lambda n: (
            (
                _pattern(r"\d{8}", n)
                and (11 - (wsum(dl(n)[:7], range(8, 1, -1)) % 11)) % 10 == dl(n)[7]
            )
            or (_pattern(r"\d{10}", n) and cz_rc_ok(n))
        ),
        "DE": vat_de,
        "DK": lambda n: (
            _pattern(r"\d{8}", n) and wsum(dl(n), [2, 7, 6, 5, 4, 3, 2, 1]) % 11 == 0
        ),
        "EE": lambda n: (
            _pattern(r"\d{9}", n)
            and (10 - wsum(dl(n)[:8], [3, 7, 1, 3, 7, 1, 3, 7]) % 10) % 10 == dl(n)[8]
        ),
        "EL": vat_el,
        "ES": lambda n: bool(re.fullmatch(r"[A-Z0-9]\d{7}[A-Z0-9]", n)),
        "FI": vat_fi,
        "FR": vat_fr,
        "GR": lambda n: VAT["EL"](n) if "EL" in VAT else False,
        "HR": hr_oib_ok,
        "HU": lambda n: (
            _pattern(r"\d{8}", n)
            and (10 - wsum(dl(n)[:7], [9, 7, 3, 1, 9, 7, 3]) % 10) % 10 == dl(n)[7]
        ),
        "IE": ie_pps_ok,
        "IT": it_piva_ok,
        "LT": vat_lt,
        "LU": lambda n: _pattern(r"\d{8}", n) and int(n[:6]) % 89 == int(n[-2:]),
        "LV": lambda n: _pattern(r"\d{11}", n),
        "MT": lambda n: (
            _pattern(r"\d{8}", n)
            and (37 - wsum(dl(n)[:6], [3, 4, 6, 7, 8, 9]) % 37) % 37 == int(n[-2:])
        ),
        "NL": lambda n: _pattern(r"\d{9}B\d{2}", n),
        "PL": pl_nip_ok,
        "PT": pt_nif_ok,
        "RO": vat_ro,
        "SE": vat_se,
        "SI": vat_si,
        "SK": lambda n: _pattern(r"\d{10}", n) and int(n) % 11 == 0,
    }


VAT = _vat_validators()
VAT["GR"] = VAT["EL"]
vat = VAT


@dataclass
class Span:
    """A deterministic match with UTF-16 offsets."""

    start: int
    end: int
    label: str
    score: float = 1.0


OWNED = frozenset(
    {
        "EMAIL",
        "URL",
        "IP_ADDRESS",
        "CREDIT_CARD",
        "SSN",
        "BANK_ACCOUNT",
        "ROUTING_NUMBER",
        "TAX_ID",
        "GOVERNMENT_ID",
        "PASSPORT",
        "DRIVERS_LICENSE",
        "IMEI",
    }
)

_EMAIL = re.compile(
    r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
    r"([\p{L}\p{N}.!#$%&'*+/=?^`{|}~-]{1,64}@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63})(?![A-Za-z0-9-])"
)
_URL = re.compile(r"\b((?:https?://|ftp://|www\.)[^\s<>()\[\]{}\"']{3,})", re.I)
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d)(?!\.\d)")
_IPV6 = re.compile(r"(?<![\w:])(?:[0-9a-f]{0,4}:){2,7}[0-9a-f]{0,4}(?![\w:])", re.I)
_MAC = re.compile(r"(?<![0-9a-f])(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}(?![0-9a-f])", re.I)
_CC = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]){11,30}\b", re.I)
_BIC = re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b")
_SSN = re.compile(r"(?<!\d)(\d{3})[- ](\d{2})[- ](\d{4})(?!\d)")
_ROUTING = re.compile(r"(?<!\d)\d{9}(?!\d)")
_ES_DNI = re.compile(r"(?<![A-Z0-9])(?:\d{8}|[XYZ]\d{7})[A-Z](?![A-Z0-9])", re.I)
_NAT_ID = re.compile(r"(?<![A-Za-z0-9])\d[\d .\-]{7,17}\d(?![A-Za-z0-9])")
_IT_CF = re.compile(
    r"(?<![A-Za-z0-9])[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z](?![A-Za-z0-9])", re.I
)
_FI_HETU = re.compile(r"(?<![A-Za-z0-9])\d{6}[-+A-F]\d{3}[0-9A-Y](?![A-Za-z0-9])")
_DK_CPR = re.compile(r"(?<!\d)\d{6}[- ]?\d{4}(?!\d)")
_VAT = re.compile(
    r"(?<![A-Za-z0-9])(AT|BE|BG|CY|CZ|DE|DK|EE|EL|GR|ES|FI|FR|HR|HU|IE|IT|LT|LU|LV|MT|NL|PL|PT|RO|SE|SI|SK)"
    r"\s?([0-9A-Za-z]{5,14})(?![A-Za-z0-9])"
)
_IMEI = re.compile(r"(?<!\d)\d{15}(?!\d)")
_SE_PN = re.compile(r"(?<!\d)((?:\d{2})?\d{6})[-+](\d{4})(?!\d)")
_PASSPORT_VALUE = re.compile(r"(?<![A-Za-z0-9])[A-Z0-9]{6,9}(?![A-Za-z0-9])")
_PPS = re.compile(r"(?<![A-Za-z0-9])\d{7}[A-Za-z]{1,2}(?![A-Za-z0-9])")
_PL_DL = re.compile(r"(?<![0-9/])\d{5}/\d{2}/\d{4,7}(?![0-9/])")
_CONTEXT_DIGIT = re.compile(r"(?<![A-Za-z0-9])\d{7,12}(?![A-Za-z0-9])")
_INTL_PHONE = re.compile(
    r"(?<!\w)\+\d{1,3}[ .-]?(?:\(?\d{1,5}\)?[ .-]?){1,5}\d{2,5}(?!\w)"
)
_GENERIC_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[ .-]?)?(?:\(?\d{2,5}\)?[ .-]?){2,5}\d{2,5}(?!\w)"
)
_DL_KW = re.compile(
    r"(?<![\p{L}\p{N}])(?:driving licen[cs]e|driver'?s? licen[cs]e|licence number|"
    r"permis de conduire|permis de conduite|f[uü]hrerschein|fahrerlaubnis(?:nummer)?|"
    r"patente(?: di guida)?|numero patente|prawo jazdy|rijbewijs(?:nummer)?|"
    r"(?:carn[eé]|permiso) de conducir|carta de condu[cç][aã]o|k[oö]rkort(?:snummer)?|"
    r"k[oø]rekort|ajokortti|vezet[oő]i enged[eé]ly|[rř]idi[cč]sk[\p{L}]* pr[uů]kaz[\p{L}]*|"
    r"vodi[cč]sk[\p{L}]* preukaz[\p{L}]*|permis de conducere|voza[cč]k[au] dozvol[ae]|"
    r"vozni[sš]ko dovoljenje|vairuotojo pa[zž]ym[eė]jimas|vad[iī]t[aā]ja apliec[iī]ba|"
    r"juhiluba|licenzja tas-sewqan)(?:[\s.:\-]*(?:nr|no|nummer|number|num[eé]ro|n[°º])\.?)?",
    re.I,
)
_DL_VALUE = re.compile(
    r"^[^A-Z0-9\n]{0,14}?([A-Z0-9](?:[A-Z0-9]|[ .\-/](?=[A-Z0-9])){4,24})"
)
_DOC_KW = re.compile(
    r"\b(passport|passeport|reisepass|pasaporte|passaporto|paspoort|national id|identity card|"
    r"id card|identity number|identification number|id number|id no|personalausweis|ausweisnummer|"
    r"ausweis|carte d.identit[eé]|documento de identidad|documento di identit[aà]|"
    r"carta d.identit[aà]|identiteitskaart|n[uú]mero de identificaci[oó]n|de identifica[cç][aã]o|c[eé]dula)",
    re.I,
)
_DOC_VALUE = re.compile(
    r"^[^A-Z0-9\n]{0,18}?([A-Z0-9](?:[A-Z0-9]|[ .\-/](?=[A-Z0-9])){4,44})"
)
_PASSPORT_KW = re.compile(
    r"^(passport|passeport|reisepass|pasaporte|passaporto|paspoort)", re.I
)

_IP_CONTEXT = re.compile(
    r"\b(?:ip|ipv4|ipv6|address|addr|host|server|node|endpoint|cidr)\b|地址", re.I
)
_ROUTING_CONTEXT = re.compile(r"\b(?:routing|aba|bank|wire|ach)\b", re.I)
_SSN_CONTEXT = re.compile(
    r"\b(?:ssn|social security|social insurance|social number|sin|seguridad social)\b|社保|社会保障|사회보장",
    re.I,
)
_TAX_CONTEXT = re.compile(
    r"\b(?:tax|taxnum|tax number|tax identification|tin|vat|npwp)\b|税号|税|세금", re.I
)
_GOV_CONTEXT = re.compile(
    r"\b(?:national id|identity card|id card|government id|nric|fin|dni|nie|cpf|cnpj|passport)\b|身份证|주민등록",
    re.I,
)
_NAT_CONTEXT = re.compile(
    r"(?<![\p{L}\p{N}])(?:id|ident\w*|national|personal (?:id|number|code)|pesel|bsn|burgerservice\w*|egn|ЕГН|cnp|oib|amka|ΑΜΚΑ|isikukood|henkilötunnus|hetu|codice fiscale|rodné|personnummer|personas kods|asmens kodas|emšo|emso|matricule|rijksregister\w*|steuer\w*|dni|nie|nif|nir|insee|sécu\w*|sécurité sociale|secu\w*|rodn[eé]|ad[oó]azonos[ií]t[oó]|ad[oó]sz[aá]m|cpr|nip|partita iva|p\.?\s?iva|iva|vat|svnr|sozialversicherung\w*|pps\w*|tax|fiscal\w*|social|seguridad)(?![\p{L}\p{N}])",
    re.I,
)
_VAT_CONTEXT = re.compile(
    r"(?<![\p{L}\p{N}])(?:vat|ust[- ]?id\w*|umsatzsteuer|tva|iva|partita iva|btw|moms|alv|dph|di[cč]|pvn|pvm|dds|nip|nif|cif|[aá]fa|arvonlis\w*|fiscal\w*|tax)(?![\p{L}\p{N}])|ΑΦΜ|ФДС",
    re.I,
)
_IMEI_CONTEXT = re.compile(r"\bimei\b", re.I)
_SE_CONTEXT = re.compile(
    r"\b(?:personnummer|person\s*number|födelsenummer|personnr)\b", re.I
)
_PASSPORT_CONTEXT = re.compile(
    r"\b(?:passport|passeport|reisepass|pasaporte|passaporto|paspoort)\b", re.I
)
_PPS_CONTEXT = re.compile(r"\b(?:pps|ppsn|personal\s*public\s*service)\b", re.I)
_DK_CONTEXT = re.compile(r"\bcpr\b", re.I)
_BIC_BEFORE = re.compile(
    r"(?:swift\s*[-/]?\s*bic|swift\s+code|bic(?:\s+code)?)\s*[:#=(\[]?\s*$", re.I
)
_CREDIT_CONTEXT = re.compile(
    r"\b(?:credit\s*card|debit\s*card|payment\s*card|bank\s*card|card\s*(?:number|no|num|info|ending|on file)|card\s*(?:charged|debited)|charged?\s*(?:my\s*|the\s*)?card|\bcard\b|visa|mastercard|master\s*card|maestro|amex|american\s*express|discover|diners|tarjeta|carte bancaire|kreditkarte|carta di credito|cartão)\b|信用卡|银行卡|カード|카드",
    re.I,
)
_PHONE_CONTEXT = re.compile(
    r"\b(?:phone|mobile|tel(?:ephone)?|cell|call(?:\s*me)?|fax|whatsapp|sms|contact number|phone number|telefon(?:ní|nummer|szám|o|oon)?|teléfono|téléphone|telepon|mobil(?:e|ni|telefon)?|gsm|tlf|zavolejte|zadzwoń|appelez|appeler|téléphonez|chiamare|chiami|chiama|llame|llamar|llamada|ligue|ligar|bel(?:len)?|hívja|hívjon|sunați|sună|ring|ringa|nazovite|καλέστε|τηλέφωνο|телефон)\b|电话|電話|연락처|전화",
    re.I,
)


def _utf16_offset(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def _match_span(
    text: str, match: re.Match[str], group: int | str = 0
) -> tuple[int, int]:
    start, end = match.span(group)
    return _utf16_offset(text, start), _utf16_offset(text, end)


def _context(
    pattern: re.Pattern[str], text: str, start: int, end: int, window: int = 48
) -> bool:
    return bool(
        pattern.search(text[max(0, start - window) : min(len(text), end + window)])
    )


def _before(pattern: re.Pattern[str], text: str, start: int, window: int = 64) -> bool:
    return bool(pattern.search(text[max(0, start - window) : start]))


def _is_ip(value: str) -> bool:
    if re.fullmatch(r"(\d{1,3}\.){3}\d{1,3}", value):
        return all(int(part) <= 255 for part in value.split("."))
    if (
        re.fullmatch(r"[0-9a-f:]+", value, re.I)
        and ":" in value
        and value not in {":", "::"}
    ):
        return value.count("::") <= 1
    return False


def _iban_trim(candidate: str) -> str | None:
    lengths = {"DE": 22, "GB": 22, "NL": 18, "BE": 16, "ES": 24, "FR": 27, "IT": 27}
    want = lengths.get(candidate[:2].upper())
    if want is None:
        # Use the complete table indirectly through the validator's accepted countries.
        compact = "".join(char for char in candidate if char.isalnum())
        if not iban_ok(compact):
            return None
        want = len(compact)
    seen = 0
    for index, char in enumerate(candidate):
        if char.isalnum():
            seen += 1
            if seen == want:
                return candidate[: index + 1]
    return None


def _trim_word(text: str, value: str, end: int) -> tuple[str, int]:
    while (
        len(value) >= 2
        and value[-1].isalpha()
        and value[-2] == " "
        and end < len(text)
        and "a" <= text[end] <= "z"
    ):
        value = value[:-2]
        end -= 2
    return value, end


def _keyword_value(
    text: str, match: re.Match[str], value_re: re.Pattern[str]
) -> tuple[str, int, int] | None:
    after = text[match.end() :]
    value_match = value_re.search(after)
    if value_match is None:
        return None
    value = value_match.group(1)
    start = _utf16_offset(text, match.end() + value_match.start(1))
    end = _utf16_offset(text, match.end() + value_match.end(1))
    value, end = _trim_word(text, value, end)
    return value, start, end


def _merge(spans: list[Span]) -> list[Span]:
    ordered = sorted(
        spans, key=lambda span: (span.start, -(span.end - span.start), span.label)
    )
    output: list[Span] = []
    for span in ordered:
        if not output or span.start >= output[-1].end:
            output.append(span)
            continue
        previous = output[-1]
        span_length = span.end - span.start
        previous_length = previous.end - previous.start
        if span_length > previous_length or (
            span_length == previous_length and span.score > previous.score
        ):
            output[-1] = span
    return output


def detect(text: str, enabled: frozenset[str] | None = None) -> list[Span]:
    """Detect deterministic entities in ``text``."""
    spans: list[Span] = []
    active = (
        OWNED | {"PHONE", "GOVERNMENT_ID", "PASSPORT", "DRIVERS_LICENSE"}
        if enabled is None
        else enabled
    )

    def add(start: int, end: int, label: str, score: float = 1.0) -> None:
        if start < end:
            spans.append(Span(start, end, label, score))

    for match in _EMAIL.finditer(text):
        add(*_match_span(text, match, 1), "EMAIL")
    for match in _URL.finditer(text):
        start, end = _match_span(text, match, 1)
        if start > 0 and UTF16Text(text).slice(start - 1, start) == "@":
            continue
        add(start, end, "URL")
    for pattern in (_IPV4, _IPV6):
        for match in pattern.finditer(text):
            value = match.group()
            if (
                value not in {":", "::"}
                and _is_ip(value)
                and _context(_IP_CONTEXT, text, match.start(), match.end(), 40)
            ):
                add(*_match_span(text, match), "IP_ADDRESS")
    for match in _MAC.finditer(text):
        add(*_match_span(text, match), "IP_ADDRESS")
    for match in _CC.finditer(text):
        value = match.group()
        if len(set(dl(value))) <= 1 or not luhn_ok(value):
            continue
        if _CREDIT_CONTEXT.search(text[max(0, match.start() - 56) : match.start()]):
            start, end = _match_span(text, match)
            while end > start and UTF16Text(text).slice(end - 1, end) in {
                " ",
                "-",
                ".",
            }:
                end -= 1
            add(start, end, "CREDIT_CARD")
    for match in _IBAN.finditer(text):
        raw = match.group()
        candidate = _iban_trim(raw) or raw
        compact = "".join(char for char in candidate if char.isalnum())
        start = _utf16_offset(text, match.start())
        end = start + len(candidate.encode("utf-16-le")) // 2
        if end < len(UTF16Text(text)) and UTF16Text(text).is_word_char_at(end):
            continue
        if iban_ok(compact):
            add(start, end, "BANK_ACCOUNT")
    for match in _BIC.finditer(text):
        if bic_ok(match.group()) and _before(_BIC_BEFORE, text, match.start(), 64):
            add(*_match_span(text, match), "BANK_ACCOUNT")
    for match in _ES_DNI.finditer(text):
        if es_dni_ok(match.group()) and _context(
            _GOV_CONTEXT, text, match.start(), match.end(), 56
        ):
            add(*_match_span(text, match), "GOVERNMENT_ID")
    for match in _NAT_ID.finditer(text):
        value = match.group()
        validators = nat_validators.get(len(dl(value)), [])
        if any(validator(value) for validator in validators) and _context(
            _NAT_CONTEXT, text, match.start(), match.end(), 64
        ):
            add(*_match_span(text, match), "GOVERNMENT_ID", 0.92)
    for match in _IT_CF.finditer(text):
        if it_cf_ok(match.group()):
            add(*_match_span(text, match), "GOVERNMENT_ID", 0.95)
    for match in _FI_HETU.finditer(text):
        if fi_hetu_ok(match.group()):
            add(*_match_span(text, match), "GOVERNMENT_ID", 0.95)
    for match in _DK_CPR.finditer(text):
        d = dl(match.group())
        if (
            1 <= int("".join(map(str, d[:2]))) <= 31
            and 1 <= int("".join(map(str, d[2:4]))) <= 12
            and _before(_DK_CONTEXT, text, match.start(), 40)
        ):
            add(*_match_span(text, match), "GOVERNMENT_ID", 0.85)
    for match in _VAT.finditer(text):
        country, number = match.groups()
        validator = VAT.get(country)
        if (
            validator
            and validator(number.replace(" ", "").upper())
            and (
                country not in {"ES", "LV", "NL"}
                or _context(_VAT_CONTEXT, text, match.start(), match.end(), 40)
            )
        ):
            add(*_match_span(text, match), "TAX_ID", 0.95)
    for match in _IMEI.finditer(text):
        if imei_ok(match.group()) and _context(
            _IMEI_CONTEXT, text, match.start(), match.end(), 32
        ):
            add(*_match_span(text, match), "IMEI", 0.9)
    for match in _SSN.finditer(text):
        if valid_us_ssn(match.group()) or _context(
            _SSN_CONTEXT, text, match.start(), match.end()
        ):
            add(*_match_span(text, match), "SSN")
    for match in _SE_PN.finditer(text):
        if valid_se_pn(match.group()) or _before(_SE_CONTEXT, text, match.start(), 40):
            add(*_match_span(text, match), "GOVERNMENT_ID")
    for match in _PASSPORT_VALUE.finditer(text):
        value = match.group()
        if any(char.isdigit() for char in value) and _before(
            _PASSPORT_CONTEXT, text, match.start(), 32
        ):
            add(*_match_span(text, match), "PASSPORT")
    for match in _DL_KW.finditer(text):
        result = _keyword_value(text, match, _DL_VALUE)
        if result and sum(char.isalnum() for char in result[0]) >= 5:
            add(result[1], result[2], "DRIVERS_LICENSE", 0.9)
    for match in _PL_DL.finditer(text):
        add(*_match_span(text, match), "DRIVERS_LICENSE", 0.9)
    for match in _DOC_KW.finditer(text):
        result = _keyword_value(text, match, _DOC_VALUE)
        if (
            result
            and any(char.isdigit() for char in result[0])
            and sum(char.isalnum() for char in result[0]) >= 6
        ):
            label = "PASSPORT" if _PASSPORT_KW.match(match.group()) else "GOVERNMENT_ID"
            add(result[1], result[2], label, 0.9)
    for match in _PPS.finditer(text):
        if ie_pps_ok(match.group()) or _context(
            _PPS_CONTEXT, text, match.start(), match.end(), 32
        ):
            add(*_match_span(text, match), "GOVERNMENT_ID")
    for match in _CONTEXT_DIGIT.finditer(text):
        before = text[max(0, match.start() - 56) : match.start()]
        count = digit_count(match.group())
        if (
            7 <= count <= 12
            and _SSN_CONTEXT.search(before)
            and not _TAX_CONTEXT.search(before)
        ):
            add(*_match_span(text, match), "SSN", 0.9)
    for match in _ROUTING.finditer(text):
        if aba_routing_ok(match.group()) and _context(
            _ROUTING_CONTEXT, text, match.start(), match.end()
        ):
            add(*_match_span(text, match), "ROUTING_NUMBER")
    for match in _INTL_PHONE.finditer(text):
        if 8 <= digit_count(match.group()) <= 15:
            add(*_match_span(text, match), "PHONE", 0.92)
    for match in _GENERIC_PHONE.finditer(text):
        value = match.group()
        before = text[max(0, match.start() - 56) : match.start()]
        digits = digit_count(value)
        grouped = any(char in " .-" for char in value)
        if _PHONE_CONTEXT.search(before) and (
            (9 <= digits <= 15) or (7 <= digits <= 8 and grouped)
        ):
            add(*_match_span(text, match), "PHONE", 0.88)

    return [span for span in _merge(spans) if span.label in active]
