# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocRoller kernels dropped for wrong results must not come back.

Regenerating the checked-in rocRoller kernels would restore these. Nothing
else would stop them: a kernel without uniformSummationOrder is still
selectable with the mode off, and both kinds below pass every predicate for
problems they get wrong. Remove a name only together with the rocRoller fix
that makes the regenerated kernel correct, and the evidence for it.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_HERE = Path(__file__).resolve().parent
_KERNELS = _HERE.parents[1] / "CustomKernels" / "rocroller"
_LOGIC = _HERE.parents[3] / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full"

_PREFIX = "RR_GEMM_TN_"
_SCALES = "_SA_BE8M0_32_SB_BE8M0_32_"

# Stream-K. The data-parallel tile loop advances by the K loop's unrolled
# counter, which leaves out the K tail: the kernel hangs when K is a single
# iteration and writes off tile boundaries at odd iteration counts of 3 or more
# (rocRoller AddStreamK.cpp, forTileDPIncrementExpr).
_STREAM_K = [
    f"{_PREFIX}{a}_{b}_{d}_{d}_Float{_SCALES}SK_WGT_128x64x128_UR_2"
    for a in ("BF8", "FP8")
    for b in ("BF8", "FP8")
    for d in ("BFloat16", "Float", "Half")
]

# Nondeterministic output with uniform summation order on or off: about 40% of
# the elements differ from the sibling kernels' at the same shape.
_RACY = [
    f"{_PREFIX}BF6_BF8_Float_Float_Float{_SCALES}WGT_256x128x128_UR_2",
    f"{_PREFIX}BF6_FP8_BFloat16_BFloat16_Float{_SCALES}WGT_256x128x128_UR_2",
    f"{_PREFIX}BF6_FP8_Float_Float_Float{_SCALES}WGT_256x128x128_UR_2",
    f"{_PREFIX}FP6_FP8_Half_Half_Float{_SCALES}WGT_256x128x128_UR_2",
    f"{_PREFIX}FP6_FP8_Float_Float_Float{_SCALES}WGT_256x128x128_UR_2",
    f"{_PREFIX}FP4_FP6_BFloat16_BFloat16_Float{_SCALES}WGT_128x128x128_UR_2",
    f"{_PREFIX}FP4_FP6_Float_Float_Float{_SCALES}WGT_128x128x128_UR_2",
]

_DROPPED = _STREAM_K + _RACY


def test_the_dropped_list_is_complete():
    assert len(_STREAM_K) == 12 and len(_RACY) == 7 and len(set(_DROPPED)) == 19


def test_no_dropped_kernel_source_is_shipped():
    shipped = {path.stem for path in _KERNELS.rglob("*.s")}
    assert shipped, f"no kernels under {_KERNELS}"
    back = sorted(shipped & set(_DROPPED))
    assert not back, f"dropped rocRoller kernels are back in {_KERNELS}: {back}"


def test_no_logic_file_names_a_dropped_kernel():
    logic = sorted(_LOGIC.rglob("GridBased/*.yaml"))
    assert logic, f"no GridBased logic under {_LOGIC}"
    # A trailing name boundary, so a WGM sibling (name + "_WGM_") does not match.
    names = re.compile(r"\b(" + "|".join(map(re.escape, _DROPPED)) + r")(?![A-Za-z0-9_])")
    back = sorted(
        f"{path.relative_to(_LOGIC)}: {name}"
        for path in logic
        for name in set(names.findall(path.read_text()))
    )
    assert not back, f"logic files name dropped rocRoller kernels: {back}"
