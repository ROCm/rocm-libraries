# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MXScaleFormat ``HostPreSwizzle128x4`` and ``AssertAlphaValue``.

Checks the predicates that keep a custom kernel on 128x4-shuffled scales away from
every problem it cannot run: the scale format and alpha == 1.
"""

import copy

import pytest
from logic_corpus import needs_logic_corpus, require_logic_file

import tensilelite.LibraryIO as L
from tensilelite import Contractions as C
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.CustomKernels import getCustomKernelConfig
from tensilelite.SolutionStructs.Validators.MXScaleFormat import validateHostPreSwizzle128x4

pytestmark = pytest.mark.unit

# A generated MX GEMM and a non-MX GEMM from the shipped corpus.
_GENERATED_MX_LOGIC = (
    "gfx950/gfx950/GridBased/gfx950_Cijk_Alik_Bljk_F4BS_MXA32_MXB32_BH_BiasSB_GG_HAS_SAB_SAV"
    "_UserArgs.yaml"
)
_NON_MX_LOGIC = "gfx950/gfx950_id75a3/Equality/gfx950_Cijk_Alik_Bljk_F8F8S_BH_BiasS_HAS_SAV_UserArgs.yaml"


def _flatten(predicate):
    if predicate.tag == "And":
        return [p for child in predicate.value for p in _flatten(child)]
    return [predicate]


def _predicates(solution, tag):
    return [p.value for p in _flatten(solution.problemPredicate) if p.tag == tag]


def _first_solution(rel, assembler, iim):
    library = L.parseLibraryLogicFile(
        str(require_logic_file(rel)), assembler, False, False, False, iim, False
    )
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


@needs_logic_corpus
@pytest.mark.parametrize(
    "fmt, expected", [(None, None), ("NoSwizzle", 0), ("HostPreSwizzle", 1), ("InMemorySwizzle", 2)]
)
def test_every_mx_solution_carries_its_scale_format(fmt, expected, assembler, gfx950_iim):
    # A 128x4 problem stays off an ordinary MX kernel only because that kernel carries
    # an MXScaleFormat predicate below 3.
    struct = copy.deepcopy(_first_solution(_GENERATED_MX_LOGIC, assembler, gfx950_iim))
    if fmt is not None:
        struct["MXScaleFormat"] = fmt
    else:
        expected = C.mxScaleFormatId(struct)
    solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
    assert _predicates(solution, "MXScaleFormat") == [expected]
    assert expected < 3


@needs_logic_corpus
def test_non_mx_solutions_carry_no_scale_format(assembler, gfx950_iim):
    struct = _first_solution(_NON_MX_LOGIC, assembler, gfx950_iim)
    solution = C.Solution.FromSolutionStruct(struct, False, False, False, assembler, gfx950_iim)
    assert _predicates(solution, "MXScaleFormat") == []


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
