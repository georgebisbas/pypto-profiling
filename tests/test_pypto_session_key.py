"""Unit tests for the pypto session cache key (reset_persistent_windows)."""

from __future__ import annotations

import sys
from pathlib import Path

_PROFILING = Path(__file__).resolve().parents[1]
if str(_PROFILING) not in sys.path:
    sys.path.insert(0, str(_PROFILING))

from collectives.equivalence import EquivalenceCase  # noqa: E402
from collectives.runners.pypto_own import _session_key  # noqa: E402


def _case(**overrides):
    kwargs = dict(
        variant="mesh",
        p=2,
        count=65536,
        dtype="fp32",
        device_ids=[0, 1],
        platform="a2a3",
        warmup_rounds=1,
        timed_rounds=5,
    )
    kwargs.update(overrides)
    return EquivalenceCase(**kwargs)


def test_key_changes_when_reset_flag_flips():
    """Flipping ONLY reset_persistent_windows must invalidate the cached session.

    This is the landmine the --reset-persistent-windows CLI flag would have hit:
    the old key was (case, mode, dfx_dir, persistent) and silently reused the
    first run's session when only the reset flag changed between two legs.
    """
    case = _case()
    base = _session_key(case, mode="composite", dfx_dir=None, persistent=True)
    reset_true = _session_key(
        case, mode="composite", dfx_dir=None, persistent=True,
        reset_persistent_windows=True,
    )
    reset_false = _session_key(
        case, mode="composite", dfx_dir=None, persistent=True,
        reset_persistent_windows=False,
    )
    assert reset_true != base
    assert reset_false != base
    assert reset_true != reset_false


def test_key_stable_without_reset_flag():
    """Explicitly passing None must equal the legacy key (no surprise rebuilds)."""
    case = _case()
    legacy = _session_key(case, mode="composite", dfx_dir=None, persistent=True)
    explicit_none = _session_key(
        case, mode="composite", dfx_dir=None, persistent=True,
        reset_persistent_windows=None,
    )
    assert legacy == explicit_none


def test_key_sensitive_to_persistent():
    case = _case()
    non_persistent = _session_key(
        case, mode="composite", dfx_dir=None, persistent=False,
        reset_persistent_windows=None,
    )
    persistent = _session_key(
        case, mode="composite", dfx_dir=None, persistent=True,
        reset_persistent_windows=None,
    )
    assert non_persistent != persistent
