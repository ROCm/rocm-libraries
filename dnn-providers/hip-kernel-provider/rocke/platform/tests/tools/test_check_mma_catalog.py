# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for ``tools/check_mma_catalog.py``, the MMA catalog's compiler cross-check.

The tool's verdicts come from running clang, which these tests do not do. They
check the parts that turn clang's answers into a report: reading an intrinsic
name and a builtin prototype, writing the probe a builtin is compiled from, and
classifying measured answers against real catalog rows. The measured answers
are written here by hand, over target ids the catalog lists.

Source-tree only (the installed test tree has no ``tools/``; see CMakeLists.txt).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"  # tests/tools -> platform
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_mma_catalog as tool
from rocke.helpers.tiling.traits import load_mma_traits

CATALOG = load_mma_traits()

# The four targets mfma_f32_32x32x8f16 lists, plus one RDNA target it does not.
CDNA = ("gfx908", "gfx90a", "gfx942", "gfx950")
JUDGED = CDNA + ("gfx1100",)


def _links(*intrinsics: str) -> tool.ProbeResult:
    return tool.ProbeResult(True, intrinsics=intrinsics)


def _rejected(diagnostic: str = "error: needs target feature") -> tool.ProbeResult:
    return tool.ProbeResult(False, diagnostic=diagnostic)


# ---------------------------------------------------------------------------
# intrinsic_base_name
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "declare_line, base",
    [
        # Not overloaded: the name is kept whole.
        (
            (
                "declare <16 x float> @llvm.amdgcn.mfma.f32.32x32x8f16(<4 x half>, "
                "<4 x half>, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #1"
            ),
            "llvm.amdgcn.mfma.f32.32x32x8f16",
        ),
        # Overloaded on the result and the A/B vectors.
        (
            (
                "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16("
                "<16 x half>, <16 x half>, <8 x float>) #1"
            ),
            "llvm.amdgcn.wmma.f32.16x16x16.f16",
        ),
        # The trailing `f32` of the base name is not stripped: the signature
        # holds float vectors, not a scalar float.
        (
            (
                "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32("
                "<2 x float>, <2 x float>, i16 immarg, <8 x float>, i1 immarg, "
                "i1 immarg) #1"
            ),
            "llvm.amdgcn.wmma.f32.16x16x4.f32",
        ),
        (
            (
                "declare <4 x float> @llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4."
                "v8i32.v8i32(<8 x i32>, <8 x i32>, <4 x float>, i32 immarg, i32 immarg, "
                "i32 immarg, i32, i32 immarg, i32) #1"
            ),
            "llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4",
        ),
        (
            (
                "declare <16 x float> @llvm.amdgcn.wmma.scale16.f32.32x16x128.f4."
                "v16f32.v16i32.v8i32(<16 x i32>, <8 x i32>, i16 immarg, <16 x float>, "
                "i32 immarg, i32 immarg, i64, i32 immarg, i32 immarg, i64, i1 immarg, "
                "i1 immarg) #1"
            ),
            "llvm.amdgcn.wmma.scale16.f32.32x16x128.f4",
        ),
        # Scalar operands: the `f64` ending `16x16x4f64` is inside a name part.
        (
            (
                "declare <4 x double> @llvm.amdgcn.mfma.f64.16x16x4f64(double, double, "
                "<4 x double>, i32 immarg, i32 immarg, i32 immarg) #1"
            ),
            "llvm.amdgcn.mfma.f64.16x16x4f64",
        ),
    ],
)
def test_intrinsic_base_name_strips_only_the_type_overload_suffix(declare_line, base):
    assert tool.intrinsic_base_name(declare_line) == base


def test_intrinsic_base_name_refuses_a_line_that_is_not_a_declare():
    with pytest.raises(ValueError, match="not an LLVM declare line"):
        tool.intrinsic_base_name("define void @probe() {")


def test_amdgcn_intrinsics_reads_only_amdgcn_declares():
    module = (
        "define void @probe(ptr %out) {\n"
        "  ret void\n"
        "}\n"
        "declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16("
        "<16 x half>, <16 x half>, <8 x float>) #1\n"
        "declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)\n"
    )
    assert tool.amdgcn_intrinsics(module) == ("llvm.amdgcn.wmma.f32.16x16x16.f16",)


# ---------------------------------------------------------------------------
# Prototypes and probe sources
# ---------------------------------------------------------------------------

V4H = "__attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16"
V16F = "__attribute__((__vector_size__(16 * sizeof(float)))) float"


def test_builtin_prototypes_are_read_from_the_ast_dump():
    """clang declares each known builtin implicitly; an unknown one is absent."""
    dump = (
        "probe.c:3:3: error: use of unknown builtin '__builtin_amdgcn_mfma_unknown'\n"
        "|-FunctionDecl 0x1 <col:16> col:16 implicit used "
        f"__builtin_amdgcn_mfma_f32_32x32x8f16 '{V16F} ({V4H}, {V4H}, {V16F}, "
        "int, int, int)' extern\n"
        "|-FunctionDecl 0x2 <col:60> col:60 implicit used "
        "__builtin_amdgcn_wmma_i32_16x16x64_iu8 '__attribute__((__vector_size__"
        "(8 * sizeof(int)))) int (_Bool, int, ...)' extern\n"
    )
    prototypes = tool.parse_builtin_prototypes(dump)
    assert prototypes == {
        "__builtin_amdgcn_mfma_f32_32x32x8f16": tool.Prototype(
            V16F, (V4H, V4H, V16F, "int", "int", "int")
        ),
        # A builtin clang type-checks itself ends in `...`, which is no operand.
        "__builtin_amdgcn_wmma_i32_16x16x64_iu8": tool.Prototype(
            "__attribute__((__vector_size__(8 * sizeof(int)))) int", ("_Bool", "int")
        ),
    }


def test_probe_source_passes_integers_as_literals_and_the_rest_by_pointer():
    prototype = tool.Prototype(V16F, (V4H, V4H, V16F, "int", "int", "int"))
    source = tool.probe_source(
        "__builtin_amdgcn_mfma_f32_32x32x8f16", prototype, {4: 2}
    )
    assert f"typedef {V4H} operand0_t;" in source
    assert (
        "void probe(result_t *out, const operand0_t *p0, const operand1_t *p1, "
        "const operand2_t *p2) { *out = __builtin_amdgcn_mfma_f32_32x32x8f16("
        "*p0, *p1, *p2, 0, 2, 0); }"
    ) in source


def test_probe_source_refuses_an_immediate_past_the_last_operand():
    prototype = tool.Prototype(V16F, (V4H, V4H, V16F))
    with pytest.raises(ValueError, match=r"has 3 operands; no operand \[5\] to set"):
        tool.probe_source("__builtin_amdgcn_x", prototype, {5: 1})


def test_a_format_row_is_probed_with_its_a_and_b_format_codes():
    """An f8f6f4 builtin takes A's and B's formats as immediates (fp8 0, fp4 4)."""
    row = CATALOG.get("mfma_scale_f32_32x32x64_f8f6f4_fp8e4m3_fp4e2m1")
    assert tool.row_probe(row) == tool.Probe(
        row.op_id, row.llvm_builtin, ((3, 0), (4, 4))
    )
    plain = CATALOG.get("mfma_f32_32x32x8f16")
    assert tool.row_probe(plain).immediates == ()


def test_scale_formats_are_probed_pair_by_pair_only_where_the_builtin_takes_them():
    wmma = CATALOG.get("wmma_scale_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1")
    probes = tool.scale_format_probes(wmma)
    assert len(probes) == 9
    # The matrix formats (both fp4) stay; e4m3 (2) and e5m3 (1) take operands 7, 10.
    assert probes["e4m3", "e5m3"].immediates == ((0, 4), (2, 4), (7, 2), (10, 1))
    # The gfx950 scaled MFMA has no scale-format operand.
    mfma = CATALOG.get("mfma_scale_f32_32x32x64_f8f6f4_fp8e4m3_fp4e2m1")
    assert tool.scale_format_probes(mfma) == {}


# ---------------------------------------------------------------------------
# Classifying the answers
# ---------------------------------------------------------------------------


def test_matching_targets_and_intrinsic_report_nothing():
    row = CATALOG.get("mfma_f32_32x32x8f16")
    results = {(row.op_id, t): _links(row.llvm_intrinsic) for t in CDNA}
    results[row.op_id, "gfx1100"] = _rejected()
    assert tool.compare_targets([row], JUDGED, results) == []
    assert tool.compare_intrinsics([row], JUDGED, results) == []


def test_a_target_mismatch_names_both_directions_and_clangs_reason():
    row = CATALOG.get("mfma_f32_32x32x8f16")
    results = {(row.op_id, t): _links(row.llvm_intrinsic) for t in JUDGED}
    results[row.op_id, "gfx908"] = _rejected("error: needs target feature mai-insts")
    assert tool.compare_targets([row], JUDGED, results) == [
        tool.TargetMismatch(
            row.op_id,
            unlisted=("gfx1100",),
            not_compiling=("gfx908",),
            diagnostic="error: needs target feature mai-insts",
        )
    ]


def test_targets_this_clang_cannot_judge_are_not_compared():
    """gfx950 is listed, but this clang could not compile for it: no verdict."""
    row = CATALOG.get("mfma_f32_32x32x8f16")
    judged = ("gfx908", "gfx90a", "gfx942")
    results = {(row.op_id, t): _links(row.llvm_intrinsic) for t in judged}
    assert tool.compare_targets([row], judged, results) == []


def test_an_intrinsic_mismatch_names_the_emitted_intrinsic_and_where():
    row = CATALOG.get("mfma_f32_32x32x8f16")
    results = {
        (row.op_id, t): _links("llvm.amdgcn.mfma.f32.32x32x8bf16.1k") for t in CDNA
    }
    assert tool.compare_intrinsics([row], CDNA, results) == [
        tool.IntrinsicMismatch(
            row.op_id,
            row.llvm_intrinsic,
            ("llvm.amdgcn.mfma.f32.32x32x8bf16.1k",),
            "gfx908",
        )
    ]


def test_scale_format_sets_are_compared_operand_by_operand():
    """The accepted set of an operand is every format some compiling pair uses."""
    row = CATALOG.get("wmma_scale_f32_16x16x128_f8f6f4_fp4e2m1_fp8e4m3")
    assert row.scale.a_dtypes == ("e8m0", "e4m3", "e5m3")
    assert row.scale.b_dtypes == ("e8m0",)
    agreeing = [(a, "e8m0") for a in ("e8m0", "e5m3", "e4m3")]
    assert tool.compare_scale_formats(row, "gfx1250", agreeing) == []
    wider = agreeing + [("e8m0", "e5m3")]
    assert tool.compare_scale_formats(row, "gfx1250", wider) == [
        tool.ScaleFormatMismatch(row.op_id, "B", ("e8m0",), ("e8m0", "e5m3"), "gfx1250")
    ]


def test_a_builtin_with_no_row_is_reported_only_where_it_compiles():
    rows = [CATALOG.get("mfma_f32_32x32x8f16")]
    results = {
        ("__builtin_amdgcn_mfma_new_on_cdna", "gfx942"): _links("llvm.amdgcn.x"),
        ("__builtin_amdgcn_mfma_new_on_cdna", "gfx90a"): _rejected(),
        ("__builtin_amdgcn_wmma_nowhere", "gfx942"): _rejected(),
        ("__builtin_amdgcn_wmma_nowhere", "gfx90a"): _rejected(),
    }
    table = [
        "__builtin_amdgcn_mfma_f32_32x32x8f16",  # has a row
        "__builtin_amdgcn_mfma_new_on_cdna",
        "__builtin_amdgcn_wmma_nowhere",
    ]
    found, nowhere = tool.builtins_without_row(
        table, rows, ("gfx90a", "gfx942"), results
    )
    assert found == {"__builtin_amdgcn_mfma_new_on_cdna": ("gfx942",)}
    assert nowhere == 1


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def _report(**changes) -> tool.CrossCheckReport:
    fields = {
        "clang": "clang",
        "identity": "clang version N",
        "rows": 1,
        "judged_targets": ("gfx90a", "gfx1154"),
        "unjudged_targets": ("gfx1250",),
        "compiler_names": {"gfx90a": "gfx90a", "gfx1154": "gfx11-generic"},
        "unknown_builtins": (),
        "builtin_table": None,
        "target_mismatches": [],
        "intrinsic_mismatches": [],
        "scale_format_mismatches": [],
        "builtins_without_row": {},
        "builtins_on_no_target": 0,
    }
    fields.update(changes)
    return tool.CrossCheckReport(**fields)


def test_a_clean_report_is_green_and_names_what_it_could_not_judge():
    text = tool.render(_report())
    assert "targets  : gfx90a, gfx1154 (as gfx11-generic)" in text
    assert "builtins with no row were not checked" in text
    assert "targets it cannot compile for: gfx1250" in text
    assert text.endswith(
        "RESULT: GREEN - the catalog agrees with this clang on every target and "
        "builtin it could judge."
    )


def test_every_disagreement_makes_the_report_red():
    mismatch = tool.TargetMismatch("mfma_f32_32x32x8f16", ("gfx1100",), (), "")
    report = _report(
        target_mismatches=[mismatch],
        builtins_without_row={"__builtin_amdgcn_mfma_new_on_cdna": ("gfx942",)},
    )
    assert report.mismatches == 2
    text = tool.render(report)
    assert "mfma_f32_32x32x8f16: compiles on gfx1100, which it does not list" in text
    assert "__builtin_amdgcn_mfma_new_on_cdna: compiles on gfx942" in text
    assert "RESULT: RED - 2 disagreement(s)" in text
