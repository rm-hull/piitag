"""Smoke tests for the initial package scaffold."""

import piitag


def test_package_imports() -> None:
    """The package is importable before the implementation phases land."""
    assert piitag.__doc__
