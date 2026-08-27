"""Unit tests for EquivalenceCase core_num (launch-width) handling."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROFILING = Path(__file__).resolve().parents[1]
if str(_PROFILING) not in sys.path:
    sys.path.insert(0, str(_PROFILING))

from collectives.equivalence import EquivalenceCase  # noqa: E402


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


def test_core_num_defaults_to_one():
    case = _case()
    assert case.core_num == 1
    assert case.case_id == "mesh_p2_count65536_fp32_a2a3_d0-1"


def test_core_num_appears_in_case_id():
    case = _case(core_num=8)
    assert case.case_id.endswith("_cn8")
    assert "_cn1" not in case.case_id
    # Distinct launch widths are distinct cases (distinct equivalence hashes).
    assert case.equivalence_hash() != _case(core_num=1).equivalence_hash()


def test_core_num_validation_rejects_non_positive():
    with pytest.raises(ValueError, match="core_num"):
        _case(core_num=0).validate()
    with pytest.raises(ValueError, match="core_num"):
        _case(core_num=-1).validate()


def test_core_num_round_trips_through_json(tmp_path):
    case = _case(core_num=16)
    path = tmp_path / "case.json"
    path.write_text(__import__("json").dumps(case.canonical_dict(), indent=2), encoding="utf-8")
    loaded = EquivalenceCase.from_json_file(str(path))
    assert loaded.core_num == 16
    assert loaded.case_id == case.case_id


def test_old_case_json_without_core_num_defaults_to_one(tmp_path):
    import json

    case = _case()
    data = case.canonical_dict()
    data.pop("core_num")  # pre-core_num case files have no such field
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = EquivalenceCase.from_json_file(str(path))
    assert loaded.core_num == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
