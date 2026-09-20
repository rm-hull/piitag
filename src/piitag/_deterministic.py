"""Checksum validators used by the deterministic recognizers."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable


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
    10: [bg_egn_ok, cz_rc_ok, hu_adoaz_ok, at_svnr_ok],
    11: [pl_pesel_ok, hr_oib_ok, ee_isikukood_ok, lv_pk_ok, be_rrn_ok],
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
        total = 4 + _luhn_weighted_sum(d[:7])
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
            return (0 if check == 10 else check) == d[8]
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
                and (11 - wsum(dl(n)[:7], range(8, 1, -1))) % 10 == dl(n)[7]
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
