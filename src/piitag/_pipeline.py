"""Regex-free span algebra for the Redact pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import regex  # type: ignore[import-untyped]

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
_PARTICLES = frozenset(
    [
        "de",
        "del",
        "della",
        "dell",
        "di",
        "da",
        "das",
        "dos",
        "du",
        "van",
        "von",
        "der",
        "den",
        "ter",
        "la",
        "le",
        "el",
        "al",
        "bin",
        "ibn",
        "mac",
        "mc",
        "o",
        "st",
        "of",
        "y",
        "e",
    ]
)
_ALL_PARTICLES = frozenset(
    [
        "van",
        "von",
        "de",
        "del",
        "della",
        "dell",
        "di",
        "da",
        "das",
        "dos",
        "du",
        "zu",
        "af",
        "ter",
        "ten",
        "des",
        "do",
        "der",
        "den",
        "la",
        "le",
        "el",
        "y",
    ]
)
_US_STATES = frozenset(
    [
        "AL",
        "AK",
        "AZ",
        "AR",
        "CA",
        "CO",
        "CT",
        "DE",
        "DC",
        "FL",
        "GA",
        "HI",
        "ID",
        "IL",
        "IN",
        "IA",
        "KS",
        "KY",
        "LA",
        "ME",
        "MD",
        "MA",
        "MI",
        "MN",
        "MS",
        "MO",
        "MT",
        "NE",
        "NV",
        "NH",
        "NJ",
        "NM",
        "NY",
        "NC",
        "ND",
        "OH",
        "OK",
        "OR",
        "PA",
        "RI",
        "SC",
        "SD",
        "TN",
        "TX",
        "UT",
        "VT",
        "VA",
        "WA",
        "WV",
        "WI",
        "WY",
    ]
)
_US_STREET = regex.compile(
    r"\b(\d{1,6}[A-Za-z]?)\s+((?:[A-Z][A-Za-z0-9.'’-]*\s+){0,4}"
    r"(?:Street|Avenue|Boulevard|Road|Lane|Drive|Court|Place|Terrace|Circle|"
    r"Highway|Parkway|Square|Trail|Crescent|Alley|Loop|Way|St|Ave|Blvd|Rd|"
    r"Ln|Dr|Ct|Pl|Ter|Cir|Hwy|Pkwy|Sq|Trl|Aly))\b\.?(?=$|[\s,.;:)])"
)
_STATE_ZIP = regex.compile(r"(?:,\s*|\s)([A-Z]{2})\s+(\d{5}(?:-\d{4})?)\b")
_SECONDARY = regex.compile(
    r"\b(?:Apartment|Apt|Suite|Ste|Unit|Building|Bldg|Floor|Fl|Room|Rm|"
    r"Department|Dept|Trailer|Trlr|Space|Spc|Lot)(?:\.|\s|#)\s*#?\s*"
    r"(?:\d{1,4}[A-Za-z]?|[A-Za-z]\d{1,4})\b",
    regex.I,
)
_BAD_GAP = regex.compile(r"[,;:/&|()\[\]{}\"<>\n\t]")
_ACCT_LEFT = regex.compile(
    r"(?:\ba/?c\b|acct|account|konto|compte|cuenta|rekening|conta|\biban\b)\W{0,4}#?\s*$",
    regex.I,
)
_TRIM_TRAIL = set(" \t\n\r.,;:!?)]}\"»’”'")
_TRIM_LEAD = set(' \t\n\r([{"«‘“')
_STRIP_TITLES = frozenset(
    [
        "mr",
        "mrs",
        "ms",
        "miss",
        "mx",
        "master",
        "mstr",
        "dr",
        "prof",
        "professor",
        "doctor",
        "dear",
        "sir",
        "madam",
        "madame",
        "monsieur",
        "mme",
        "mlle",
        "herr",
        "frau",
        "fraulein",
        "frl",
        "mevrouw",
        "dhr",
        "signor",
        "signora",
    ]
)


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


def is_shouting(text: str) -> bool:
    letters = [char for char in text if char.isalpha()]
    return len(letters) > 1 and all(char.isupper() for char in letters)


def title_case(text: str) -> str | None:
    output: list[str] = []
    word_start = True
    for char in text:
        if char.isalpha():
            mapped = char.upper() if word_start else char.lower()
            if len(mapped) != 1:
                return None
            output.append(mapped)
            word_start = False
        else:
            output.append(char)
            word_start = True
    return "".join(output)


def model_input(text: str) -> str:
    normalized = title_case(text) if is_shouting(text) else None
    if normalized is not None and len(normalized.encode("utf-16-le")) == len(
        text.encode("utf-16-le")
    ):
        return normalized
    return text


def casing_is_informative(text: str) -> bool:
    words: list[str] = []
    current: list[str] = []
    saw_lower = False
    for char in text:
        if char.isalpha() or char in {"'", "-", "\u2019"}:
            current.append(char)
            saw_lower |= char.islower()
        else:
            if current and current[0].isalpha():
                words.append("".join(current))
            current = []
    if current and current[0].isalpha():
        words.append("".join(current))
    if len(words) < 3:
        return True
    return saw_lower and not all(word[0].isupper() for word in words)


def _utf16_index(text: str, index: int) -> int:
    return len(text[:index].encode("utf-16-le")) // 2


def _py_span(text: str, start: int, end: int) -> tuple[int, int]:
    return _utf16_index(text, start), _utf16_index(text, end)


def attach_building_numbers(text: UTF16Text, spans: list[Span]) -> list[Span]:
    output = list(spans)
    occupied = [(span.start, span.end) for span in spans]

    def free(start: int, end: int) -> bool:
        return not any(a < end and start < b for a, b in occupied)

    for span in spans:
        if span.label != "STREET_NAME":
            continue
        after = text.string[_codepoint_index(text.string, span.end) :]
        match = regex.match(
            r"^[\s,]{0,2}(\d{1,5}[a-zA-Z]?(?:[-/]\d{1,4}[a-zA-Z]?)?)\b", after
        )
        if match:
            start, end = _py_span(
                text.string,
                _codepoint_index(text.string, span.end) + match.start(1),
                _codepoint_index(text.string, span.end) + match.end(1),
            )
            if free(start, end):
                output.append(Span(start, end, "BUILDING_NUMBER"))
                occupied.append((start, end))
        base = max(0, span.start - 8)
        before = text.slice(base, span.start)
        match = regex.search(r"(\d{1,5}[a-zA-Z]?)[\s,]{0,2}$", before)
        if match:
            start, end = (
                base + _utf16_index(before, match.start(1)),
                base + _utf16_index(before, match.end(1)),
            )
            if free(start, end):
                output.append(Span(start, end, "BUILDING_NUMBER"))
                occupied.append((start, end))
    return sorted(merge_same_label(output), key=lambda span: (span.start, span.end))


def _codepoint_index(text: str, utf16_index: int) -> int:
    units = 0
    for index, char in enumerate(text):
        next_units = units + len(char.encode("utf-16-le")) // 2
        if next_units > utf16_index:
            return index
        units = next_units
    return len(text)


def redact_us_street(text: UTF16Text, spans: list[Span]) -> list[Span]:
    output = list(spans)
    for match in _US_STREET.finditer(text.string):
        building = _py_span(text.string, *match.span(1))
        street = _py_span(text.string, *match.span(2))
        output = [
            span
            for span in output
            if not (
                span.label in {"STREET_NAME", "BUILDING_NUMBER"}
                and max(span.start, building[0]) < min(span.end, street[1])
            )
        ]
        output.extend(
            [Span(*building, "BUILDING_NUMBER"), Span(*street, "STREET_NAME")]
        )
    return sorted(merge_same_label(output), key=lambda span: (span.start, span.end))


def attach_state_codes(text: UTF16Text, spans: list[Span]) -> list[Span]:
    output = list(spans)
    occupied = [(span.start, span.end) for span in spans]
    for match in _STATE_ZIP.finditer(text.string):
        state = match.group(1)
        if state not in _US_STATES:
            continue
        for group, label in ((1, "STATE"), (2, "ZIP_CODE")):
            start, end = _py_span(text.string, *match.span(group))
            if not any(a < end and start < b for a, b in occupied):
                output.append(Span(start, end, label))
                occupied.append((start, end))
    return sorted(merge_same_label(output), key=lambda span: (span.start, span.end))


def redact_secondary_address(text: UTF16Text, spans: list[Span]) -> list[Span]:
    output = list(spans)
    for match in _SECONDARY.finditer(text.string):
        start, end = _py_span(text.string, *match.span())
        output = [
            span
            for span in output
            if not (
                span.label in {"SECONDARY_ADDRESS", "BUILDING_NUMBER"}
                and span.start < end
                and start < span.end
            )
        ]
        output.append(Span(start, end, "SECONDARY_ADDRESS"))
    return sorted(merge_same_label(output), key=lambda span: (span.start, span.end))


def relabel_by_context(text: str, spans: list[Span]) -> list[Span]:
    view = UTF16Text(text)
    output: list[Span] = []
    for span in spans:
        left = view.slice(max(0, span.start - 28), span.start).lower()
        label = span.label
        if label == "PHONE" and _ACCT_LEFT.search(left):
            label = "BANK_ACCOUNT"
        elif label == "GOVERNMENT_ID" and (
            ("driv" in left and "licen" in left)
            or "führerschein" in left
            or "fuhrerschein" in left
            or "rijbewijs" in left
            or ("permis" in left and "conduire" in left)
        ):
            label = "DRIVERS_LICENSE"
        output.append(Span(span.start, span.end, label, span.score))
    return output


def clean_spans(text: str, spans: list[Span]) -> list[Span]:
    view = UTF16Text(text)
    output: list[Span] = []
    for span in spans:
        start, end = span.start, span.end
        while end > start and view.scalar_at(end - 1) in _TRIM_TRAIL:
            end -= 1
        while start < end and view.scalar_at(start) in _TRIM_LEAD:
            start += 1
        if span.label in _NAME_FAMILIES:
            while start < end:
                segment = view.slice(start, end)
                match = regex.match(r"^(\S+)\s+", segment)
                if (
                    not match
                    or match.group(1).strip(".'’").lower() not in _STRIP_TITLES
                ):
                    break
                start += len(match.group(0).encode("utf-16-le")) // 2
            core = view.slice(start, end).strip(" .'’")
            if not core or core.lower() in _STRIP_TITLES:
                continue
        if end > start:
            output.append(Span(start, end, span.label, span.score))
    return output


def _gap_is_name_like(gap: str) -> bool:
    if (
        not gap.strip()
        or len(gap.encode("utf-16-le")) // 2 > 20
        or _BAD_GAP.search(gap)
    ):
        return False
    for token in gap.split():
        word = token.strip(".-'’")
        if (
            word
            and word.lower() not in _PARTICLES
            and len(word) != 1
            and not word[0].isupper()
        ):
            return False
    return True


def bridge_name_gaps(text: UTF16Text, spans: list[Span]) -> list[Span]:
    if not casing_is_informative(text.string):
        return spans
    output: list[Span] = []
    for span in sorted(spans, key=lambda value: (value.start, value.end)):
        if (
            output
            and output[-1].label in _NAME_FAMILIES
            and span.label in _NAME_FAMILIES
            and span.start >= output[-1].end
            and _gap_is_name_like(text.slice(output[-1].end, span.start))
        ):
            output[-1].end = max(output[-1].end, span.end)
        else:
            output.append(span)
    return output


def extend_particle_names(text: UTF16Text, spans: list[Span]) -> list[Span]:
    if not casing_is_informative(text.string):
        return spans
    output: list[Span] = []
    for span in sorted(spans, key=lambda value: (value.start, value.end)):
        if span.label in _NAME_FAMILIES:
            words = text.slice(span.start, span.end).split()
            consumed = bool(words and words[-1].strip(".-'’").lower() in _ALL_PARTICLES)
            position = span.end
            while position < len(text):
                rest = text.slice(position, len(text))
                match = regex.match(r"^(\s+)([^\s,.;:!?)\]}\"']+)", rest)
                if not match:
                    break
                token = match.group(2)
                width = len(match.group(0).encode("utf-16-le")) // 2
                if token.strip(".-'’").lower() in _ALL_PARTICLES:
                    position += width
                    consumed = True
                elif consumed and token[0].isupper():
                    position += width
                else:
                    break
            if position > span.end:
                span = Span(span.start, position, span.label, span.score)
        output.append(span)
    return merge_same_label(output)


def _adjacent_name(text: UTF16Text, first: Span, second: Span) -> bool:
    left, right = sorted((first, second), key=lambda span: span.start)
    if right.start < left.end:
        return True
    gap = text.slice(left.end, right.start)
    return (
        gap.isspace() and len(gap.encode("utf-16-le")) // 2 <= 3
    ) or _gap_is_name_like(gap)


def hysteresis(
    text: UTF16Text, scored: list[tuple[Span, float]], high: float
) -> list[Span]:
    kept = [span for span, score in scored if score >= high]
    weak = [
        span for span, score in scored if score < high and span.label in _NAME_FAMILIES
    ]
    changed = True
    while changed and weak:
        changed = False
        for span in weak[:]:
            if any(
                candidate.label in _NAME_FAMILIES
                and _adjacent_name(text, span, candidate)
                for candidate in kept
            ):
                kept.append(span)
                weak.remove(span)
                changed = True
                break
    return kept


def resolve(det_spans: list[Span], ml_spans: list[Span]) -> list[Span]:
    suppressed: set[int] = set()
    kept_ml: list[Span] = []
    for ml in ml_spans:
        conflict = False
        for index, det in enumerate(det_spans):
            if ml.start < det.end and det.start < ml.end:
                if ml.label == det.label:
                    suppressed.add(index)
                else:
                    conflict = True
        if not conflict:
            kept_ml.append(ml)
    kept_det = [span for index, span in enumerate(det_spans) if index not in suppressed]
    return sorted(
        merge_same_label(kept_det + kept_ml),
        key=lambda span: (span.start, span.end, span.label),
    )


def mask_text(text: str, spans: list[Span]) -> str:
    units = list(text.encode("utf-16-le"))
    for span in spans:
        if span.label not in _DET_OWNED:
            continue
        start, end = max(0, span.start), min(len(units) // 2, span.end)
        for index in range(start, end):
            if units[index * 2 : index * 2 + 2] != b"\n\x00":
                units[index * 2 : index * 2 + 2] = b" \x00"
    return bytes(units).decode("utf-16-le")


# Swift-style aliases for callers porting directly from the reference.
bioesToSpans = bioes_to_spans
mergeSameLabel = merge_same_label
mergePriority = merge_priority
snapSpans = snap_spans
