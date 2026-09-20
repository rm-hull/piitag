"""Smoke tests for the initial package scaffold."""

import piitag
from piitag.cli import _parser, _run
from piitag.model import Item, Label, Redaction


def test_package_imports() -> None:
    """The package is importable before the implementation phases land."""
    assert piitag.__doc__


def test_cli_json_output(monkeypatch) -> None:
    class FakeRedact:
        def __init__(self, *, directory=None) -> None:
            del directory

        def redaction(self, text, options):
            del text, options
            return Redaction(
                "[EMAIL_1]",
                [Item(Label.EMAIL, "a@example.com", "[EMAIL_1]", 1.0, 0, 13)],
            )

    monkeypatch.setattr("piitag.cli.Redact", FakeRedact)
    args = _parser().parse_args(["a@example.com", "--labels", "email", "--json"])

    assert '"redacted_text": "[EMAIL_1]"' in _run(args)
    assert '"label": "EMAIL"' in _run(args)
