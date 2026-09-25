# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch bounds and argument ABI for the specialized decode kernel."""
from pathlib import Path

import pytest
import yaml

from Tensile.Common.Utilities import state
from Tensile.Contractions import ProblemPredicate
from Tensile.CustomKernels import getCustomKernelConfigAndAssembly, readCustomKernelConfig

pytestmark = pytest.mark.unit

NAME = "Custom_W4A16_Decode_G{group}{suffix}_UnsignedBias8_gfx1151"
GENERAL = "Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB{group}ZPU8_UserArgs_MT64x160x64_MI16x16x1_gfx1151"
DIRECTORY = Path(__file__).parents[2] / "CustomKernels"


@pytest.mark.parametrize("group,suffix", [(32, "_W4")])
def test_decode_selection_bounds(group, suffix):
    config = readCustomKernelConfig(NAME.format(group=group, suffix=suffix), DIRECTORY)
    equal = ProblemPredicate.FromOriginalKeyPair(("AssertSizeEqual", config["AssertSizeEqual"]))
    positive_k = ProblemPredicate.FromOriginalKeyPair(
        ("AssertSizeGreaterThan", config["AssertSizeGreaterThan"]))
    assert state(equal) == {
        "type": "And",
        "value": [
            {"type": "SizeEqual", "index": 1, "value": 1},
            {"type": "SizeEqual", "index": 2, "value": 1},
        ],
    }
    assert state(positive_k) == {"type": "SizeGreaterThan", "index": 3, "value": 0}
    assert config["AssertSummationElementMultiple"] == 256
    support = config["InternalSupportParams"]
    assert not support["SupportUserGSU"]
    assert not support["SupportCustomWGM"]
    assert not support["SupportCustomStaggerU"]


@pytest.mark.parametrize("group,suffix", [(32, "_W4")])
def test_decode_universal_arguments_match_matrix_kernel(group, suffix):
    def metadata(name):
        config, _ = getCustomKernelConfigAndAssembly(name, DIRECTORY)
        return yaml.safe_load(config)["amdhsa.kernels"][0]

    decode, general = metadata(NAME.format(group=group, suffix=suffix)), metadata(GENERAL.format(group=group))
    # HIP compilation must preserve the universal layout that the existing
    # host library supplies, including unused fields and trailing offsets.
    for key in (".kernarg_segment_size", ".kernarg_segment_align"):
        assert decode[key] == general[key]
    assert [(arg[".offset"], arg[".size"]) for arg in decode[".args"]] == [
        (arg[".offset"], arg[".size"]) for arg in general[".args"]
    ]
    config = readCustomKernelConfig(NAME.format(group=group, suffix=suffix), DIRECTORY)
    x, y, z = config["WorkGroup"]
    assert x * y * z == decode[".max_flat_workgroup_size"]


def test_q27b_equality_dispatch_keys():
    from types import SimpleNamespace

    from Tensile.SolutionLibrary import MatchingLibrary

    path = DIRECTORY.parents[2] / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_Q27B.yaml"
    )
    logic = yaml.safe_load(path.read_text())
    solutions = {s["SolutionIndex"]: SimpleNamespace(index=s["SolutionIndex"])
                 for s in logic["Solutions"]}
    library = MatchingLibrary.FromOriginalState(
        {"indexOrder": logic["IndexOrder"], "distance": logic["LibraryType"],
         "table": logic["ExactLogic"]}, solutions)
    serialized = state(library)
    assert serialized["distance"] == "Equality"
    assert serialized["properties"] == [
        {"type": "FreeSizeA", "index": 0}, {"type": "FreeSizeB", "index": 0},
        {"type": "BatchSize", "index": 0}, {"type": "BoundSize", "index": 0},
    ]
    expected = {(m, n, 1, k) for m, k in [
        (34816, 5120), (5120, 17408), (16384, 5120), (14336, 5120), (5120, 6144)
    ] for n in (1, 2048)}
    assert {tuple(row["key"]) for row in serialized["table"]} == expected
    assert len(serialized["table"]) == len(expected)
    for row in serialized["table"]:
        solution = logic["Solutions"][row["index"]]
        if row["key"][1] == 1:
            assert solution["CustomKernelName"] == NAME.format(group=32, suffix="_W4")
        else:
            assert solution["EnableMatrixInstruction"]
            assert solution["WorkGroupMapping"] in (1, 4)
            if row["key"] == [34816, 2048, 1, 5120]:
                assert solution["CustomKernelName"].endswith(
                    "_MT64x256x64_MI16x16x1_gfx1151")
                assert solution["WorkGroup"] == [32, 8, 1]
                assert solution["MacroTile0"] == 64
                assert solution["MacroTile1"] == 256
            else:
                assert solution["CustomKernelName"].endswith(
                    "_MT128x256x64_MI16x16x1_gfx1151")
                assert solution["WorkGroup"] == [64, 8, 1]
                assert solution["MacroTile0"] == 128
                assert solution["MacroTile1"] == 256
                assert solution["WorkGroupMapping"] == 1
    assert logic["ProblemType"]["ScaleBlockSizeA"] == 32
    assert logic["ProblemType"]["Int4EncodingA"] == "UnsignedBias8"


def test_block_scale_equality_grids_do_not_merge_duplicate_shape_keys():
    from copy import deepcopy
    from types import SimpleNamespace

    from Tensile.LibraryIO import prepareLibraryLogicDict
    from Tensile.SolutionLibrary import MasterSolutionLibrary

    class IndexOnlySolution:
        @staticmethod
        def FromSolutionStruct(solution, *_args):
            return SimpleNamespace(index=solution["SolutionIndex"])

    path = DIRECTORY.parents[2] / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8_Q27B.yaml")
    original = yaml.safe_load(path.read_text())
    merged = None
    for group, zero_point, encoding in [
        (32, True, "UnsignedBias8"),
        (32, True, "Signed"),
        (128, True, "UnsignedBias8"),
        (32, False, "UnsignedBias8"),
    ]:
        logic = deepcopy(original)
        logic["ProblemType"].update(
            ScaleBlockSizeA=group, ScaleZeroPointA=zero_point, Int4EncodingA=encoding)
        prepareLibraryLogicDict(logic)
        library, _ = MasterSolutionLibrary.FromOriginalState(
            logic, logic["Solutions"], False, False, False, None, {}, True,
            solutionClass=IndexOnlySolution)
        if merged is None:
            merged = library
        else:
            merged.merge(library)
    assert len(merged.lazyLibraries) == 4
    for library in merged.lazyLibraries.values():
        table = state(library.library)["rows"][0]["library"]["table"]
        assert len(table) == 10
        assert len({tuple(row["key"]) for row in table}) == 10
        assert all(row["index"] in library.solutions for row in table)
