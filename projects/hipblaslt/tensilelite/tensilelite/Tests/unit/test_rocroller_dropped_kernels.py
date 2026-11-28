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

In NT and TT, kernels with an FP6 or BF6 A and an FP4, FP8 or BF8 B give wrong
output at some tiles, for every output type: run-to-run different at most of
them, the same wrong answer every run at 16x16x128 and 32x32x64 with an 8-bit
B. _NONTN_FAILING lists the tile groups that fail, with the name suffixes
rocRoller picked; every A, B and output in a group fails alike, 312 kernels in
all.

The _WGM_ and _LDSSYNC suffixes name the same tile built with workgroup
mapping or with rocRoller's LDS-sync retry, and rocRoller may pick another of
these forms for another problem, so every suffix sibling of a dropped name is
kept out too.
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

_B_CLASSES = {"FP4": ("FP4",), "FP8/BF8": ("FP8", "BF8")}
_NONTN_FAILING = {
    "NT": [
        ("FP4", "128x128x128", ("", "_WGM_")),
        ("FP4", "256x256x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
        ("FP8/BF8", "16x16x128", ("", "_WGM_")),
        ("FP8/BF8", "32x32x64", ("",)),
        ("FP8/BF8", "16x64x256", ("", "_WGM_")),
        ("FP8/BF8", "64x64x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
        ("FP8/BF8", "128x64x128", ("_WGM__LDSSYNC",)),
        ("FP8/BF8", "128x128x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
        ("FP8/BF8", "256x256x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
    ],
    "TT": [
        ("FP4", "128x128x128", ("", "_WGM_")),
        ("FP4", "256x256x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
        ("FP8/BF8", "16x16x128", ("", "_WGM_")),
        ("FP8/BF8", "32x32x64", ("",)),
        ("FP8/BF8", "64x64x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
        ("FP8/BF8", "128x64x128", ("_WGM__LDSSYNC",)),
        ("FP8/BF8", "128x128x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
        ("FP8/BF8", "256x256x128", ("_LDSSYNC", "_WGM__LDSSYNC")),
    ],
}

_DROPPED += [
    f"RR_GEMM_{layout}_{a}_{b}_{d}_{d}_Float{_SCALES}WGT_{tile}_UR_2{suffix}"
    for layout, groups in _NONTN_FAILING.items()
    for b_class, tile, suffixes in groups
    for a in ("FP6", "BF6")
    for b in _B_CLASSES[b_class]
    for d in _OUTPUTS
    for suffix in suffixes
]

_SUFFIXES = r"(?:_WGM_)?(?:_LDSSYNC)?"
_DROPPED_CONFIGS = sorted({re.sub(_SUFFIXES + "$", "", name) for name in _DROPPED})
# A trailing name boundary, so a configuration does not match a longer name.
_DROPPED_NAME = re.compile(
    r"\b(?:" + "|".join(map(re.escape, _DROPPED_CONFIGS)) + ")" + _SUFFIXES + r"(?![A-Za-z0-9_])"
)


def test_the_dropped_list_is_complete():
    assert len(_DROPPED) == len(set(_DROPPED)) == 36 + 168 + 144


@pytest.mark.parametrize(
    "name, dropped",
    [
        (f"RR_GEMM_NT_FP6_FP8_Half_Half_Float{_SCALES}WGT_32x32x64_UR_2_WGM_", True),
        (f"RR_GEMM_NT_BF6_BF8_Float_Float_Float{_SCALES}WGT_64x64x128_UR_2", True),
        (f"RR_GEMM_NT_FP6_FP4_Half_Half_Float{_SCALES}WGT_128x128x128_UR_2_LDSSYNC", True),
        (f"RR_GEMM_TN_BF6_FP8_Half_Half_Float{_SCALES}WGT_256x128x128_UR_2_WGM__LDSSYNC", True),
        (f"RR_GEMM_TT_FP6_BF8_Float_Float_Float{_SCALES}WGT_32x32x64_UR_2_WGM__LDSSYNC", True),
        (f"RR_GEMM_TT_BF6_FP8_BFloat16_BFloat16_Float{_SCALES}WGT_128x128x128_UR_2", True),
        (f"RR_GEMM_NT_FP4_FP8_Half_Half_Float{_SCALES}WGT_32x32x64_UR_2_WGM_", False),
        (f"RR_GEMM_NT_FP6_FP8_Half_Half_Float{_SCALES}WGT_32x32x128_UR_2", False),
        (f"RR_GEMM_NN_FP6_FP8_Half_Half_Float{_SCALES}WGT_32x32x64_UR_2", False),
    ],
)
def test_a_suffix_sibling_of_a_dropped_name_is_dropped(name, dropped):
    assert name not in _DROPPED
    assert bool(_DROPPED_NAME.fullmatch(name)) == dropped
    assert bool(_DROPPED_NAME.findall(f"CustomKernelName: {name}\n")) == dropped


def test_no_dropped_kernel_source_is_shipped():
    shipped = {path.stem for path in _KERNELS.rglob("*.s")}
    assert shipped, f"no kernels under {_KERNELS}"
    back = sorted(name for name in shipped if _DROPPED_NAME.fullmatch(name))
    assert not back, f"dropped rocRoller kernels are back in {_KERNELS}: {back}"


@needs_logic_corpus
def test_no_logic_file_names_a_dropped_kernel():
    root = asm_full()
    logic = sorted(root.rglob("GridBased/*.yaml"))
    assert logic, f"no GridBased logic under {root}"
    back = sorted(
        f"{path.relative_to(root)}: {name}"
        for path in logic
        for name in set(_DROPPED_NAME.findall(path.read_text()))
    )
    assert not back, f"logic files name dropped rocRoller kernels: {back}"
