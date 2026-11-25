# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 MXFP8 skinny tiles: a non-converging partition candidate is skipped.

MT352x128 and MT128x352 reach an uneven multi-DU partition split whose VGPR
tile assignment repeats with a period past the unroll limit. Codegen used to
die with an AssertionError that aborted the whole job; it must now skip that
candidate and emit with a finer split.

CPU-only emit. No GPU.
"""

import os

import pytest

from config_harness import emit_kernels_from_config, solutions_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "mxfp8_skinny_vgpr_unroll_limit.yaml",
)

# problem_index -> (MacroTile0, MacroTile1, MIWaveTile)
_TILES = {
    0: (352, 128, [22, 2]),
    1: (128, 352, [2, 22]),
}


@pytest.mark.parametrize("problem_index", sorted(_TILES))
def test_skinny_solutions_are_in_scope(problem_index):
    sols = solutions_from_config(_CONFIG, arch=_ARCH, problem_index=problem_index)
    assert len(sols) == 1, f"expected one solution, got {len(sols)}"
    sol = sols[0]
    st = sol._state if hasattr(sol, "_state") else sol
    mt0, mt1, miwt = _TILES[problem_index]
    assert (st["MacroTile0"], st["MacroTile1"]) == (mt0, mt1)
    assert list(st["MIWaveTile"]) == miwt
    assert st["DepthU"] == 256
    assert st["PrefetchGlobalRead"] == 1


@pytest.mark.parametrize("problem_index", sorted(_TILES))
def test_skinny_kernels_emit(problem_index):
    results = emit_kernels_from_config(
        _CONFIG, limit=1, arch=_ARCH, problem_index=problem_index
    )
    assert results, f"no kernels from {_CONFIG} group {problem_index}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"emit failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
