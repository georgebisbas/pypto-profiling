"""Scratch/window alignment required by the current simpler runtime."""

import pytest

from collectives.runners.simpler_own import _scratch_layout


@pytest.mark.parametrize(
    "count,nranks",
    [(256, 2), (4096, 8), (24960, 4), (65536, 2), (65536, 8)],
)
def test_scratch_layout_is_64_byte_aligned(count: int, nranks: int) -> None:
    nbytes, window_size = _scratch_layout(count, nranks)

    assert nbytes % 64 == 0
    assert window_size % 64 == 0
    assert nbytes >= count * 4 + nranks * 4
    assert window_size >= nbytes
