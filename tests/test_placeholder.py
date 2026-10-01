"""Placeholder test confirming the test suite and package imports are wired up correctly."""

import rag


def test_package_imports() -> None:
    assert rag.__doc__ is not None
