"""Command-line interface for the piitag Redact detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import Label, Options, Redact


def _labels(value: str) -> frozenset[Label]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    try:
        return frozenset(Label(name.upper()) for name in names)
    except ValueError as error:
        valid = ", ".join(label.value for label in Label)
        raise argparse.ArgumentTypeError(
            f"unknown label {error.args[0]!r}; choose from: {valid}"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="piitag",
        description="Detect and redact personal information in text.",
    )
    parser.add_argument("text", help="text to redact")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="minimum neural confidence from 0 to 1 (default: 0.6)",
    )
    parser.add_argument(
        "--labels",
        type=_labels,
        help="comma-separated labels to redact, for example EMAIL,PHONE",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="output the redaction and items as JSON",
    )
    parser.add_argument(
        "--directory",
        type=Path,
        help="local model asset directory; omit to use the Hugging Face cache",
    )
    return parser


def _run(args: argparse.Namespace) -> str:
    result = Redact(directory=args.directory).redaction(
        args.text,
        Options(minimum_confidence=args.min_confidence, labels=args.labels),
    )
    if not args.json:
        return result.redacted_text
    return json.dumps(
        {
            "redacted_text": result.redacted_text,
            "items": [
                {
                    "label": item.label.value,
                    "original": item.original,
                    "placeholder": item.placeholder,
                    "confidence": item.confidence,
                    "start": item.start,
                    "end": item.end,
                }
                for item in result.items
            ],
        },
        ensure_ascii=False,
    )


def main(argv: list[str] | None = None) -> None:
    """Run the piitag command-line interface."""
    args = _parser().parse_args(argv)
    try:
        print(_run(args))
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        _parser().error(str(error))
