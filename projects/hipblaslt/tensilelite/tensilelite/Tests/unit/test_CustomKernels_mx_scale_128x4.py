# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MXScaleFormat ``HostPreSwizzle128x4`` and ``AssertAlphaValue``.

Checks the predicates that keep a custom kernel on 128x4-shuffled scales away from
every problem it cannot run: the scale format, alpha == 1, the fixed size and
strides, and the swizzled operands. The last tests parse the rocKE MX FP6 logic files.
"""

import copy
import json
import re
from pathlib import Path

import pytest
from logic_corpus import needs_logic_corpus, require_asm_full_root, require_logic_file

import tensilelite.BenchmarkProblems as bp
import tensilelite.LibraryIO as L
import tensilelite.SolutionLibrary as SolutionLibrary
from tensilelite import Contractions as C
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.CustomKernels import getCustomKernelConfig
from tensilelite.SolutionStructs.Problem import SWIZZLE_LAYOUT_IDS, ProblemType
from tensilelite.SolutionStructs.Validators.MXScaleFormat import (
    checkCustomKernelSwizzleLayout,
    validateHostPreSwizzle128x4,
    validateSwizzleLayout,
)

pytestmark = pytest.mark.unit

# The same entries for MI355X (0x75a3) and, in the base directory, MI350X (0x75a0).
_CHIP_DIRS = {"75a3": "gfx950/gfx950_id75a3/Equality", "75a0": "gfx950/gfx950/Equality"}
_TILED_B_FILES = [
    (f"gfx950_Cijk_Alik_Bljk_{types}_MXAE8B32_MXBE8B32_STA_STB_H_UserArgs.yaml", fmt)
    for types, fmt in (("F6BS", "e2m3"), ("B6BS", "e3m2"))
]
_LOGIC_FILES = [
    pytest.param(f"{_CHIP_DIRS[chip]}/{name}", fmt, chip, id=f"{fmt}-{chip}")
    for chip in _CHIP_DIRS
    for name, fmt in _TILED_B_FILES
]
# Each kernel runs 12 * L + 8 K tiles of 128: L iterations of a 12-tile main loop and a
# fixed 8-tile tail. Only these K have a kernel (L = 42 and L = 85).
_KS = (65536, 131584)
# B in HIPBLASLT_ORDER_COL: one E2M3 kernel that runs 24 * L + 8 K tiles, L = 21.
_CANONICAL_B_FILE = "gfx950_Cijk_Alik_Bljk_F6BS_MXAE8B32_MXBE8B32_STA_H_UserArgs.yaml"
_CANONICAL_B_KERNEL = (
    "rocke_gemm_mx_e2m3_e2m3_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3"
    "_4096x4096x65536_c0c1_Acanonical_k21f5fa872a"
)
# HIPBLASLT_ORDER_COL256_R128_P48_EXT on A and B, HIPBLASLT_ORDER_COL256_R128_P16_EXT on A with B
# in HIPBLASLT_ORDER_COL, and HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_EXT on A with B in
# HIPBLASLT_ORDER_COL256_R128_P48_PAGE24_SPLIT_EXT: one E2M3 and one E3M2 kernel each, at
# K = 65536 only.
_LAYOUT_FILES = [
    pytest.param(
        f"gfx950_Cijk_Alik_Bljk_{types}_MXAE8B32_MXBE8B32_{token}_H_UserArgs.yaml",
        fmt,
        layouts,
        f"rocke_gemm_mx_{fmt}_{fmt}_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3"
        f"_4096x4096x65536_{suffix}",
        id=f"{fmt}-{token}",
    )
    for types, fmt in (("F6BS", "e2m3"), ("B6BS", "e3m2"))
    for token, layouts, suffix in (
        ("STAP48_STBP48", ("P48", "P48"), "c0c1p48_k067cdb88d0"),
        ("STAP16", ("P16", "Default"), "c0c1p_Acanonical_kd2977a1791"),
        (
            "STAP48Page24_STBP48Page24Split",
            ("P48Page24", "P48Page24Split"),
            "c0c1p48_ka941ce3864",
        ),
    )
]
_FIXTURES = Path(__file__).resolve().parent / "logic_yaml" / "mx_scale_format"
# A generated MX GEMM and a non-MX GEMM, one solution each.
_GENERATED_MX_LOGIC = _FIXTURES / (
    "gfx950_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_BH_BiasSB_GG_HAS_SAB_SAV_UserArgs.yaml"
)
_NON_MX_LOGIC = _FIXTURES / "gfx950_Cijk_Alik_Bljk_F8F8S_BH_BiasS_HAS_SAV_UserArgs.yaml"


def _kernel_name(fmt, k):
    name = (
        f"rocke_gemm_mx_{fmt}_{fmt}_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3"
        f"_4096x4096x{k}_c0c1"
    )
    return name + "_k4b371438f4" if k == 65536 else name


def _flatten(predicate):
    if predicate.tag == "And":
        return [p for child in predicate.value for p in _flatten(child)]
    return [predicate]


def _predicates(solution, tag):
    return [p.value for p in _flatten(solution.problemPredicate) if p.tag == tag]


def _first_solution(path, assembler, iim):
    library = L.parseLibraryLogicFile(str(path), assembler, False, False, False, iim, False)
    return library.solutions[0]


def test_scale_format_ids_are_append_only():
    assert C.MX_SCALE_FORMAT_IDS == {
        "NoSwizzle": 0,
        "HostPreSwizzle": 1,
        "InMemorySwizzle": 2,
        "HostPreSwizzle128x4": 3,
    }
    assert validParameters["MXScaleFormat"] == ["Auto", *C.MX_SCALE_FORMAT_IDS]


@pytest.mark.parametrize(
    "state, expected",
    [
        ({}, 0),
        ({"MXScaleFormat": "Auto"}, 0),
        ({"MXScaleFormat": "NoSwizzle"}, 0),
        ({"MXScaleFormat": "HostPreSwizzle"}, 1),
        ({"MXScaleFormat": "InMemorySwizzle"}, 2),
        ({"MXScaleFormat": "HostPreSwizzle128x4"}, 3),
    ],
)
def test_scale_format_id(state, expected):
    assert C.mxScaleFormatId(state) == expected


def test_assert_alpha_value_becomes_alpha_value_predicate():
    predicate = C.ProblemPredicate.FromOriginalKeyPair(("AssertAlphaValue", 1))
    assert (predicate.tag, predicate.value) == ("AlphaValue", "1")


@pytest.mark.parametrize("value", [-1, 0, 2, "1"])
def test_assert_alpha_value_rejects_unknown_values(value):
    with pytest.raises(RuntimeError, match="AssertAlphaValue"):
        C.ProblemPredicate.FromOriginalKeyPair(("AssertAlphaValue", value))


def test_assert_alpha_value_is_a_valid_parameter():
    assert validParameters["AssertAlphaValue"] == [1]


@pytest.mark.parametrize(
    "fmt, expected", [(None, None), ("NoSwizzle", 0), ("HostPreSwizzle", 1), ("InMemorySwizzle", 2)]
)
def test_every_mx_solution_carries_its_scale_format(fmt, expected, assembler, gfx950_iim):
    # A 128x4 problem stays off an ordinary MX kernel only because that kernel carries
    # an MXScaleFormatEquals predicate below 3.
    struct = copy.deepcopy(_first_solution(_GENERATED_MX_LOGIC, assembler, gfx950_iim))
    if fmt is not None:
        struct["MXScaleFormat"] = fmt
    else:
        expected = C.mxScaleFormatId(struct)
    solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
    assert _predicates(solution, "MXScaleFormatEquals") == [expected]
    assert expected < 3


def test_non_mx_solutions_carry_no_scale_format(assembler, gfx950_iim):
    struct = _first_solution(_NON_MX_LOGIC, assembler, gfx950_iim)
    solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
    assert _predicates(solution, "MXScaleFormatEquals") == []


@pytest.mark.parametrize("fmt, valid", [(None, True), ("HostPreSwizzle128x4", False)])
def test_pre_derived_pool_solution_rejects_host_pre_swizzle_128x4(
    fmt, valid, assembler, gfx950_iim
):
    _, _, data = bp._parsePoolFile(str(_GENERATED_MX_LOGIC))
    state = copy.deepcopy(data["Solutions"][0])
    # The legacy selectors make the constructor derive again; current pool states lack them.
    state.pop("StreamK", None)
    state.pop("StreamKForceDPOnly", None)
    assert state["AssignedDerivedParameters"]
    if fmt is not None:
        state["MXScaleFormat"] = fmt
    solution = bp._create_solution_from_state(
        state,
        ProblemType(data["ProblemType"], False),
        False,
        False,
        False,
        assembler,
        gfx950_iim,
        str(_GENERATED_MX_LOGIC),
    )
    assert solution is not None
    assert solution["Valid"] is valid
    if valid:
        assert solution["AssignedDerivedParameters"]


@pytest.mark.parametrize(
    "fmt, handwritten, isa, valid",
    [
        ("HostPreSwizzle128x4", True, (9, 5, 0), True),
        ("HostPreSwizzle128x4", False, (9, 5, 0), False),
        ("HostPreSwizzle128x4", True, (9, 4, 2), False),
        ("HostPreSwizzle128x4", True, (12, 5, 0), False),
        ("HostPreSwizzle", False, (9, 5, 0), True),
        ("NoSwizzle", False, (9, 4, 2), True),
    ],
)
def test_host_pre_swizzle_128x4_is_handwritten_gfx950_only(fmt, handwritten, isa, valid):
    state = {"MXScaleFormat": fmt, "ISA": isa, "Valid": True}
    assert validateHostPreSwizzle128x4(state, handwritten, False) is valid
    assert state["Valid"] is valid


def test_swizzle_layout_ids_are_append_only():
    assert SWIZZLE_LAYOUT_IDS == {
        "Default": 0,
        "P48": 1,
        "P16": 2,
        "P48Page24": 3,
        "P48Page24Split": 4,
    }
    header = (Path(__file__).resolve().parents[3] / "include/Tensile/SwizzleLayout.hpp").read_text()
    entries = re.findall(r"^\s*(\w+)\s*=\s*(\d+),", header, re.M)
    assert {name: int(value) for name, value in entries} == SWIZZLE_LAYOUT_IDS


def _swizzled_problem_type(assembler, iim, **keys):
    library = L.parseLibraryLogicFile(
        str(_GENERATED_MX_LOGIC), assembler, False, False, False, iim, False
    )
    state = copy.deepcopy(library.problemType)
    for key, value in keys.items():
        state[key] = value
    return state


def _type_predicates(problemType, tag):
    return [p.value for p in problemType.predicates(includeType=True) if p.tag == tag]


@pytest.mark.parametrize(
    "keys, layoutA, layoutB",
    [
        ({}, None, None),
        ({"SwizzleTensorA": True}, 0, None),
        ({"SwizzleTensorA": True, "SwizzleTensorB": True}, 0, 0),
        ({"SwizzleTensorA": True, "SwizzleLayoutA": "P16"}, 2, None),
        (
            {
                "SwizzleTensorA": True,
                "SwizzleTensorB": True,
                "SwizzleLayoutA": "P48Page24",
                "SwizzleLayoutB": "P48Page24Split",
            },
            3,
            4,
        ),
    ],
)
def test_every_swizzled_problem_row_names_its_layout(keys, layoutA, layoutB, assembler, gfx950_iim):
    # An order-104 row (layout 0) must not match a problem in another layout.
    pt = C.ProblemType.FromOriginalState(_swizzled_problem_type(assembler, gfx950_iim, **keys))
    expectedA = [] if layoutA is None else [layoutA]
    expectedB = [] if layoutB is None else [layoutB]
    assert _type_predicates(pt, "SwizzleLayoutA") == expectedA
    assert _type_predicates(pt, "SwizzleLayoutB") == expectedB


@pytest.mark.parametrize(
    "keys, match",
    [
        ({"SwizzleTensorA": True, "SwizzleLayoutA": "C0C1P48"}, "SwizzleLayoutA must be one of"),
        ({"SwizzleLayoutA": "P48"}, "SwizzleLayoutA P48 requires SwizzleTensorA"),
        ({"SwizzleTensorA": True, "SwizzleLayoutB": "P48"}, "SwizzleLayoutB P48 requires SwizzleTensorB"),
    ],
)
def test_swizzle_layout_rejects_unknown_or_unswizzled_layouts(keys, match, assembler, gfx950_iim):
    state = _swizzled_problem_type(assembler, gfx950_iim, **keys)
    with pytest.raises(ValueError, match=match):
        C.ProblemType.FromOriginalState(state)


@pytest.mark.parametrize(
    "keys, token",
    [
        ({"SwizzleTensorA": True, "SwizzleTensorB": True}, "_STA_STB_"),
        ({"SwizzleTensorA": True, "SwizzleTensorB": True, "SwizzleLayoutA": "P48",
          "SwizzleLayoutB": "P48"}, "_STAP48_STBP48_"),
        ({"SwizzleTensorA": True, "SwizzleLayoutA": "P16"}, "_STAP16_"),
    ],
)
def test_problem_type_name_keeps_default_layouts_unchanged(keys, token, assembler, gfx950_iim):
    name = str(ProblemType(_swizzled_problem_type(assembler, gfx950_iim, **keys), False))
    assert token in name


@pytest.mark.parametrize(
    "layoutA, layoutB, fmt, handwritten, valid",
    [
        ("Default", "Default", None, False, True),
        ("P48", "P48", "HostPreSwizzle128x4", True, True),
        ("P48Page24", "P48Page24Split", "HostPreSwizzle128x4", True, True),
        ("P48", "P48", "HostPreSwizzle128x4", False, False),
        ("P16", "Default", "HostPreSwizzle", True, False),
        ("Default", "P48", None, True, False),
    ],
)
def test_swizzle_layout_is_a_handwritten_128x4_kernel_only(layoutA, layoutB, fmt, handwritten, valid):
    state = {
        "ProblemType": {"SwizzleLayoutA": layoutA, "SwizzleLayoutB": layoutB},
        "MXScaleFormat": fmt,
        "Valid": True,
    }
    assert validateSwizzleLayout(state, handwritten, False) is valid
    assert state["Valid"] is valid


def test_custom_kernel_config_keeps_host_pre_swizzle_128x4(tmp_path):
    (tmp_path / "mx_kernel.s").write_text(
        ".amdgpu_metadata\n"
        "---\n"
        "custom.config:\n"
        "  InternalSupportParams:\n"
        "    KernArgsVersion: 0\n"
        "  MXScaleFormat: HostPreSwizzle128x4\n"
        "  ProblemType: {MXBlockA: 32, MXBlockB: 32}\n"
        "  MatrixInstruction: [16, 16, 128, 1]\n"
        "amdhsa.kernels:\n"
        "  - .name: mx_kernel\n"
        "    .max_flat_workgroup_size: 64\n"
        "    .args:\n"
        "      - .name: D\n"
        "        .size: 8\n"
        "        .offset: 0\n"
        "        .value_kind: global_buffer\n"
        "...\n"
    )
    config = getCustomKernelConfig("mx_kernel", {}, str(tmp_path))
    assert config["MXScaleFormat"] == "HostPreSwizzle128x4"


@pytest.mark.parametrize("fmt", ["e2m3", "e3m2"])
def test_page24_kernel_takes_the_aid_placement_bit_last(fmt):
    """The kernel's die bit, the int32 at byte 40, is a CustomArgSemantic the host knows."""
    name = (
        f"rocke_gemm_mx_{fmt}_{fmt}_e8m0b32_f32acc_bf16_rcr_256x256x128_w2x2_16x16x128_s3"
        "_4096x4096x65536_c0c1p48_ka941ce3864"
    )
    args = getCustomKernelConfig(name, {})["CustomKernel"]["args"]
    assert [a["semantic"] for a in args] == [
        "AddressB", "AddressA", "AddressMXScaleB", "AddressMXScaleA", "AddressD", "AIDPlacementBit"
    ]
    assert args[-1]["type"] == "int32"
    header = (
        Path(__file__).resolve().parents[3] / "include/Tensile/ContractionSolution.hpp"
    ).read_text()
    assert "X_MACRO(AIDPlacementBit)" in header


@needs_logic_corpus
@pytest.mark.parametrize(
    "name",
    [name for name, _ in _TILED_B_FILES]
    + [_CANONICAL_B_FILE]
    + [p.values[0] for p in _LAYOUT_FILES],
)
def test_logic_files_match_across_chips(name):
    """The 0x75a0 file is the 0x75a3 file with only its device changed."""
    mi355 = require_logic_file(f"{_CHIP_DIRS['75a3']}/{name}").read_text()
    mi350 = require_logic_file(f"{_CHIP_DIRS['75a0']}/{name}").read_text()
    assert mi355.count("DeviceNames: [Device 75a3]") == 1
    assert mi350 == mi355.replace("DeviceNames: [Device 75a3]", "DeviceNames: [Device 75a0]")


@needs_logic_corpus
@pytest.mark.parametrize("rel, fmt, chip", _LOGIC_FILES)
def test_logic_file_predicates(rel, fmt, chip, assembler, gfx950_iim):
    logic = require_logic_file(rel)
    assert L.readYAML(str(logic))["DeviceNames"] == [f"Device {chip}"]
    library = L.parseLibraryLogicFile(str(logic), assembler, False, False, False, gfx950_iim, False)
    assert len(library.solutions) == len(_KS)
    rows = {tuple(key): index for key, (index, _) in library.exactLogic}
    assert set(rows) == {(4096, 4096, 1, k) for k in _KS}
    for k in _KS:
        struct = library.solutions[rows[(4096, 4096, 1, k)]]
        assert struct["CustomKernelName"] == _kernel_name(fmt, k)
        solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
        preds = {}
        for p in _flatten(solution.problemPredicate):
            preds.setdefault(p.tag, []).append((p.index, p.value))

        def values(tag):
            return set(preds[tag])

        assert values("MXScaleFormatEquals") == {(None, 3)}
        assert values("AlphaValue") == {(None, "1")}
        assert values("SwizzleTensorA") == {(None, True)}
        assert values("SwizzleTensorB") == {(None, True)}
        assert values("SizeEqual") == {(0, 4096), (1, 4096), (2, 1), (3, k)}
        assert values("StrideAEqual") == {(1, k)}
        assert values("StrideBEqual") == {(1, k)}
        assert values("StrideDEqual") == {(1, 4096)}
        assert solution.problemType.mxScaleFormat == 3
        assert not solution.problemType.useBeta


@needs_logic_corpus
@pytest.mark.parametrize("chip", _CHIP_DIRS)
def test_canonical_b_logic_file_predicates(chip, assembler, gfx950_iim):
    logic = require_logic_file(f"{_CHIP_DIRS[chip]}/{_CANONICAL_B_FILE}")
    assert L.readYAML(str(logic))["DeviceNames"] == [f"Device {chip}"]
    library = L.parseLibraryLogicFile(str(logic), assembler, False, False, False, gfx950_iim, False)
    assert len(library.solutions) == 1
    rows = {tuple(key): index for key, (index, _) in library.exactLogic}
    assert rows == {(4096, 4096, 1, 65536): 0}
    struct = library.solutions[0]
    assert struct["CustomKernelName"] == _CANONICAL_B_KERNEL
    solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
    preds = {}
    for p in _flatten(solution.problemPredicate):
        preds.setdefault(p.tag, []).append((p.index, p.value))

    def values(tag):
        return set(preds[tag])

    assert values("MXScaleFormatEquals") == {(None, 3)}
    assert values("AlphaValue") == {(None, "1")}
    assert values("SwizzleTensorA") == {(None, True)}
    assert values("SwizzleTensorB") == {(None, False)}
    assert not solution.problemType.swizzleTensorB
    assert values("SizeEqual") == {(0, 4096), (1, 4096), (2, 1), (3, 65536)}
    assert values("StrideAEqual") == {(1, 65536)}
    assert values("StrideBEqual") == {(1, 65536)}
    assert values("StrideDEqual") == {(1, 4096)}
    assert solution.problemType.mxScaleFormat == 3
    assert not solution.problemType.useBeta


@needs_logic_corpus
@pytest.mark.parametrize("chip", _CHIP_DIRS)
@pytest.mark.parametrize("name, fmt, layouts, kernel", _LAYOUT_FILES)
def test_swizzle_layout_logic_file_predicates(
    name, fmt, layouts, kernel, chip, assembler, gfx950_iim
):
    """The 48-byte, 16-byte and near-die kernels sit on their own layout's problem-type row and lazy
    library, never on the order-104 row at the same key."""
    logic = require_logic_file(f"{_CHIP_DIRS[chip]}/{name}")
    assert L.readYAML(str(logic))["DeviceNames"] == [f"Device {chip}"]
    library = L.parseLibraryLogicFile(str(logic), assembler, False, False, False, gfx950_iim, False)
    assert len(library.solutions) == 1
    rows = {tuple(key): index for key, (index, _) in library.exactLogic}
    assert rows == {(4096, 4096, 1, 65536): 0}
    struct = library.solutions[0]
    assert struct["CustomKernelName"] == kernel
    solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
    preds = {}
    for p in _flatten(solution.problemPredicate):
        preds.setdefault(p.tag, []).append((p.index, p.value))

    def values(tag):
        return set(preds[tag])

    tiledB = layouts[1] != "Default"
    assert values("MXScaleFormatEquals") == {(None, 3)}
    assert values("AlphaValue") == {(None, "1")}
    assert values("SwizzleTensorA") == {(None, True)}
    assert values("SwizzleTensorB") == {(None, tiledB)}
    assert values("SizeEqual") == {(0, 4096), (1, 4096), (2, 1), (3, 65536)}
    assert values("StrideAEqual") == {(1, 65536)}
    assert values("StrideBEqual") == {(1, 65536)}
    assert values("StrideDEqual") == {(1, 4096)}
    assert solution.problemType.mxScaleFormat == 3
    assert not solution.problemType.useBeta
    assert _type_predicates(solution.problemType, "SwizzleLayoutA") == [
        SWIZZLE_LAYOUT_IDS[layouts[0]]
    ]
    assert _type_predicates(solution.problemType, "SwizzleLayoutB") == (
        [SWIZZLE_LAYOUT_IDS[layouts[1]]] if tiledB else []
    )
    lazy = L.parseLibraryLogicFile(str(logic), assembler, False, False, False, gfx950_iim, True)
    (lazyName,) = lazy.library.lazyLibraries
    token = "_STA" + layouts[0] + ("_STB" + layouts[1] if tiledB else "") + "_"
    assert token in lazyName


def _logic_library(rel, assembler, iim, lazy, declared=None, **problemType):
    """Parses `rel` with `problemType` keys replaced. `declared` holds the SwizzleLayoutA/B
    that the kernels' custom.config states; by default the ones `problemType` sets, as a
    correctly declared kernel does."""
    data = L.read(str(require_logic_file(rel)), True)
    data["ProblemType"].update(problemType)
    if declared is None:
        declared = {k: v for k, v in problemType.items() if k.startswith("SwizzleLayout")}
    real = L.getCustomKernelConfig

    def config(*args, **kwargs):
        cfg = copy.deepcopy(real(*args, **kwargs))
        cfg.setdefault("ProblemType", {}).update(declared)
        return cfg

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(L, "getCustomKernelConfig", config)
        return L.parseLibraryLogicData(data, rel, assembler, False, False, False, iim, lazy).library


def _problem_type_rows(library):
    """(SwizzleLayoutA, SwizzleLayoutB) -> the matching tables under that problem-type row."""
    rows = {}

    def tables(node):
        if isinstance(node, SolutionLibrary.MatchingLibrary):
            return [node]
        children = [r["library"] for r in getattr(node, "rows", [])]
        children += list(getattr(node, "mapping", {}).values())
        return [t for child in children for t in tables(child)]

    def walk(node):
        for row in getattr(node, "rows", []):
            preds = {p.tag: p.value for p in _flatten(row["predicate"])}
            if "TypesEqual" in preds:
                layouts = (preds.get("SwizzleLayoutA"), preds.get("SwizzleLayoutB"))
                assert layouts not in rows
                rows[layouts] = tables(row["library"])
            else:
                walk(row["library"])
        for child in getattr(node, "mapping", {}).values():
            walk(child)

    walk(library.library)
    return rows


@needs_logic_corpus
def test_swizzle_layout_keeps_its_own_solution_table(assembler, gfx950_iim):
    """A P48 logic file with the order-104 file's keys merges into a separate table."""
    rel = f"{_CHIP_DIRS['75a3']}/{_TILED_B_FILES[0][0]}"
    p48 = {"SwizzleLayoutA": "P48", "SwizzleLayoutB": "P48"}
    merged = _logic_library(rel, assembler, gfx950_iim, False)
    merged.merge(_logic_library(rel, assembler, gfx950_iim, False, **p48))

    rows = _problem_type_rows(merged)
    assert set(rows) == {(0, 0), (1, 1)}
    for layout, tables in rows.items():
        assert len(tables) == 1
        entries = tables[0].table
        # Each table holds only its own file's two keys, so the K = 65536 row of one
        # layout never sorts against the other's.
        assert len(entries) == len(_KS)
        assert sorted(tuple(e["key"]) for e in entries) == sorted(
            {tuple(e["key"]) for e in rows[(0, 0)][0].table}
        )
        for e in entries:
            solution = e["index"].solution
            assert _predicates(solution, "SwizzleLayoutA") == [layout[0]]
            assert _predicates(solution, "SwizzleLayoutB") == [layout[1]]


@needs_logic_corpus
def test_swizzle_layout_has_its_own_lazy_library(assembler, gfx950_iim):
    rel = f"{_CHIP_DIRS['75a3']}/{_TILED_B_FILES[0][0]}"
    c0c1 = _logic_library(rel, assembler, gfx950_iim, True)
    p48 = _logic_library(
        rel, assembler, gfx950_iim, True, SwizzleLayoutA="P48", SwizzleLayoutB="P48"
    )
    (c0c1Name,) = c0c1.lazyLibraries
    (p48Name,) = p48.lazyLibraries
    assert "_STA_STB_" in c0c1Name
    assert p48Name == c0c1Name.replace("_STA_STB_", "_STAP48_STBP48_")
    c0c1.merge(p48)
    assert set(c0c1.lazyLibraries) == {c0c1Name, p48Name}


@needs_logic_corpus
@pytest.mark.parametrize(
    "logicLayout, declaredLayout",
    [("Default", "P48"), ("P48", "Default"), ("P16", "P48")],
    ids=["logic-omits", "config-omits", "disagree"],
)
def test_custom_kernel_layout_must_match_its_logic_file(
    logicLayout, declaredLayout, assembler, gfx950_iim
):
    """A kernel whose layout is stated in only one of custom.config and the logic file is
    a build error. Otherwise the logic file's Default layout would put it in the order-104
    row, at the order-104 kernel's Equality key."""
    rel = f"{_CHIP_DIRS['75a3']}/{_TILED_B_FILES[0][0]}"
    logic = {}
    if logicLayout != "Default":
        logic = {"SwizzleLayoutA": logicLayout, "SwizzleLayoutB": logicLayout}
    declared = {"SwizzleLayoutA": declaredLayout, "SwizzleLayoutB": declaredLayout}
    with pytest.raises(
        ValueError,
        match=rf"SwizzleLayoutA {declaredLayout} in custom.config but {logicLayout} in the logic file",
    ):
        _logic_library(rel, assembler, gfx950_iim, False, declared=declared, **logic)


def test_custom_kernel_layout_check_passes_matching_or_absent_layouts():
    # A logic ProblemType that omits the keys holds Default for both once parsed.
    absent = {"SwizzleLayoutA": "Default", "SwizzleLayoutB": "Default"}
    checkCustomKernelSwizzleLayout("k", {"ProblemType": {}}, absent)
    checkCustomKernelSwizzleLayout("k", {}, {})
    checkCustomKernelSwizzleLayout("k", {"ProblemType": {"SwizzleLayoutA": "Default"}}, absent)
    checkCustomKernelSwizzleLayout(
        "k",
        {"ProblemType": {"SwizzleLayoutA": "P48", "SwizzleLayoutB": "P48"}},
        {"SwizzleLayoutA": "P48", "SwizzleLayoutB": "P48"},
    )
    with pytest.raises(ValueError, match="SwizzleLayoutB P16 in custom.config but Default"):
        checkCustomKernelSwizzleLayout("k", {"ProblemType": {"SwizzleLayoutB": "P16"}}, absent)


@needs_logic_corpus
def test_no_equality_key_mixes_custom_kernel_layouts():
    """Every custom kernel in a swizzled Equality logic file declares the file's layouts in
    custom.config, so one problem-type row never holds kernels of two layouts at a key.
    Kernels predating the other layouts state no layout in either file: Default."""
    root = require_asm_full_root()
    files = sorted(
        p for p in root.rglob("*.yaml")
        if p.parent.name == "Equality" and ("_STA" in p.name or "_STB" in p.name)
    )
    assert files
    declaredAt = {}
    checked = 0
    for path in files:
        data = L.read(str(path), True)
        if isinstance(data, list):
            data = L.parseLibraryLogicList(data, str(path))
        pt = data["ProblemType"]
        logic = tuple(pt.get("SwizzleLayout" + x, "Default") for x in "AB")
        rowType = json.dumps(
            {k: v for k, v in pt.items() if not k.startswith("SwizzleLayout")},
            sort_keys=True,
            default=str,
        )
        kernels = {}
        for s in data["Solutions"]:
            name = s.get("CustomKernelName")
            if not name:
                continue
            try:
                cfg = getCustomKernelConfig(name, s.get("InternalSupportParams", {}))
            except (RuntimeError, KeyError, TypeError):
                continue
            checkCustomKernelSwizzleLayout(name, cfg, pt, str(path))
            declared = tuple(
                (cfg.get("ProblemType") or {}).get("SwizzleLayout" + x, "Default") for x in "AB"
            )
            kernels[s["SolutionIndex"]] = (name, declared)
        for sizes, (index, _) in data.get("ExactLogic") or []:
            if index not in kernels:
                continue
            name, declared = kernels[index]
            group = declaredAt.setdefault((str(path.parent), rowType, logic, tuple(sizes)), {})
            group[name] = declared
            checked += 1
    assert checked
    for (directory, _, logic, key), group in declaredAt.items():
        assert set(group.values()) == {logic}, (directory, logic, key, group)
