# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocRoller kernel configurations with nondeterministic output must not ship.

Two tile shapes give wrong, run-to-run different output with uniform summation
order on or off: {FP6,BF6} x {FP8,BF8} at 256x128x128 and FP4 x {FP6,BF6} at
128x128x128, both UR_2. Every output type and both workgroup mappings are
covered, 36 configurations in all. Capturing rocRoller's picks again would
bring them back, and every predicate would still accept them, so nothing but
this test keeps them out. The rocRoller-off build returns no algorithm where
rocRoller picks one. Remove a name only together with the rocRoller fix that
makes the regenerated kernel correct, and the evidence for it.
"""

import re
from pathlib import Path

import pytest

from logic_corpus import needs_logic_corpus
from rocroller_logic_corpus import asm_full

pytestmark = pytest.mark.unit

_HERE = Path(__file__).resolve().parent
_KERNELS = _HERE.parents[1] / "CustomKernels" / "rocroller"

_SCALES = "_SA_BE8M0_32_SB_BE8M0_32_"
_OUTPUTS = ("BFloat16", "Float", "Half")
_RACY_TILES = (
    [(a, b, "256x128x128") for a in ("FP6", "BF6") for b in ("FP8", "BF8")]
    + [("FP4", b, "128x128x128") for b in ("FP6", "BF6")]
)

_DROPPED = [
    f"RR_GEMM_TN_{a}_{b}_{d}_{d}_Float{_SCALES}WGT_{tile}_UR_2{wgm}"
    for a, b, tile in _RACY_TILES
    for d in _OUTPUTS
    for wgm in ("", "_WGM_")
]


def test_the_dropped_list_is_complete():
    assert len(_DROPPED) == len(set(_DROPPED)) == 36


def test_no_dropped_kernel_source_is_shipped():
    shipped = {path.stem for path in _KERNELS.rglob("*.s")}
    assert shipped, f"no kernels under {_KERNELS}"
    back = sorted(shipped & set(_DROPPED))
    assert not back, f"dropped rocRoller kernels are back in {_KERNELS}: {back}"


@needs_logic_corpus
def test_no_logic_file_names_a_dropped_kernel():
    root = asm_full()
    logic = sorted(root.rglob("GridBased/*.yaml"))
    assert logic, f"no GridBased logic under {root}"
    # A trailing name boundary, so a non-WGM name does not match its WGM sibling.
    names = re.compile(r"\b(" + "|".join(map(re.escape, _DROPPED)) + r")(?![A-Za-z0-9_])")
    back = sorted(
        f"{path.relative_to(root)}: {name}"
        for path in logic
        for name in set(names.findall(path.read_text()))
    )
    assert not back, f"logic files name dropped rocRoller kernels: {back}"
