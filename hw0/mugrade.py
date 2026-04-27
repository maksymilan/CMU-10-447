"""Local stub for course-specific mugrade package used by tests.

The public homework tests import mugrade for submission helpers, but local
unit tests do not require remote submission. Keep submit() as a no-op so test
collection succeeds without the private dependency.
"""

from typing import Any


def submit(_: Any) -> None:
    """No-op local replacement for mugrade.submit."""
    return None
