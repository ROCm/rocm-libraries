# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch bounds and universal ABI for generated runtime-group decode."""
from pathlib import Path
import pytest
import yaml
from Tensile.Common.Utilities import state
from Tensile.Contractions import ProblemPredicate
from test_w4a16_block_dequant import toolchain, _emit, _emit_solution, FP16_UNSIGNED
from test_w4a16_decode_codegen import solution as decode_solution

pytestmark = pytest.mark.unit
DIRECTORY = Path(__file__).parents[2] / "CustomKernels"
SCHEDULES = {
    "_W4": [4, 4, 1, 128, 1, True, True],
    "_W4_U1_A4": [4, 1, 4, 128, 1, False, False],
    "_W4_NativePerm": [4, 4, 1, 128, 1, False, True],
}


def metadata(source):
    return yaml.safe_load(source.split(".amdgpu_metadata", 1)[1].split(".end_amdgpu_metadata", 1)[0])["amdhsa.kernels"][0]


@pytest.fixture(scope="module")
def generated_matrix(toolchain):
    from Tensile.CustomKernels import _buildCustomKernelFromMetadata

    kernel, source = _emit(toolchain, blockSize=32, depthU=64, zeroPoint=True,
                           problemType=dict(FP16_UNSIGNED, ScaleBlockSizesA=[32, 64, 128]))
    general = metadata(source)
    return general, _buildCustomKernelFromMetadata(
        "matrix", {"amdhsa.kernels": [general]}, kernel)


@pytest.mark.parametrize("suffix", SCHEDULES)
def test_decode_selection_bounds(toolchain, suffix):
    config = decode_solution(toolchain, decode=SCHEDULES[suffix], zeroPoint=True)
    assert [state(p) for p in ProblemPredicate.W4A16DecodePredicates(config)] == [
        {"type": "SizeEqual", "index": 2, "value": 1},
        {"type": "SizeGreaterThan", "index": 3, "value": 0},
        {"type": "SizeEqual", "index": 1, "value": 1},
    ]
    assert config["AssertSummationElementMultiple"] == 256
    for key in ("SupportUserGSU", "SupportCustomWGM", "SupportCustomStaggerU"):
        assert not config["InternalSupportParams"][key]


@pytest.mark.parametrize("suffix", SCHEDULES)
def test_decode_universal_arguments_match_matrix_kernel(toolchain, suffix, generated_matrix):
    kernel, source = _emit_solution(toolchain, decode_solution(toolchain, decode=SCHEDULES[suffix], zeroPoint=True))
    decode, general = metadata(source), generated_matrix[0]
    assert decode[".kernarg_segment_size"] == general[".kernarg_segment_size"] == 168
    assert decode[".kernarg_segment_align"] == general[".kernarg_segment_align"]
    assert [(a[".offset"], a[".size"]) for a in decode[".args"]] == [(a[".offset"], a[".size"]) for a in general[".args"]]
    assert decode[".args"][-1][".offset"] == 160
    assert [a[".name"] for a in decode[".args"]] == [a[".name"] for a in general[".args"]]
    assert kernel["CustomKernel"]["args"] == generated_matrix[1]["args"]
    assert kernel["CustomKernel"]["grid"] == ["TilesXYBatchGSU", "One", "One"]
    assert kernel["CustomKernel"]["macrotile"] == [4, 1, 256]


def test_q27b_equality_dispatch_keys():
    from types import SimpleNamespace

    from Tensile.SolutionLibrary import MatchingLibrary

    path = DIRECTORY.parents[2] / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
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
    ] for n in (1, 2, 3, 4, 2048)}
    assert {tuple(row["key"]) for row in serialized["table"]} == expected
    assert len(serialized["table"]) == len(expected)
    for row in serialized["table"]:
        solution = logic["Solutions"][row["index"]]
        if row["key"][1] == 1:
            suffix = {17408: "_W4_U1_A4", 6144: "_W4_NativePerm"}.get(
                row["key"][3], "_W4")
            assert solution["W4A16Decode"] == SCHEDULES[suffix]
            assert solution["CustomKernelName"] == ""
        else:
            assert solution["EnableMatrixInstruction"]
            assert solution["CustomKernelName"] == ""
            assert solution["WorkGroupMapping"] in (1, 4)
            if row["key"][1] in (2, 3, 4) or row["key"] == [34816, 2048, 1, 5120]:
                assert solution["WorkGroup"] == [32, 8, 1]
                assert solution["MacroTile0"] == 64
                assert solution["MacroTile1"] == 256
            else:
                assert solution["WorkGroup"] == [64, 8, 1]
                assert solution["MacroTile0"] == 128
                assert solution["MacroTile1"] == 256
                assert solution["WorkGroupMapping"] == 1
    assert logic["ProblemType"]["ScaleBlockSizesA"] == [32, 64, 128]
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
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml")
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
            ScaleBlockSizesA=[group], ScaleZeroPointA=zero_point, Int4EncodingA=encoding)
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
        assert len(table) == 25
        assert len({tuple(row["key"]) for row in table}) == 25
        assert all(row["index"] in library.solutions for row in table)


def test_runtime_group_predicate_accepts_only_declared_groups():
    from Tensile.Contractions import ProblemType
    from Tensile.SolutionStructs import ProblemType as OriginalProblemType

    config = dict(DataType=4, DataTypeA=24, DataTypeB=4, DestDataType=4,
                  ComputeDataType=0, HighPrecisionAccumulate=True,
                  UseScaleAB="",
                  ScaleBlockSizesA=[128, 32, 64], ScaleZeroPointA=True,
                  Int4EncodingA="UnsignedBias8")
    problem = ProblemType.FromOriginalState(OriginalProblemType(config, False))
    predicates = [state(p) for p in problem.predicates(includeType=True)]
    group = next(p for p in predicates if p["type"] == "Or")
    assert group == {"type": "Or", "value": [
        {"type": "ScaleBlockSizeA", "value": g} for g in (32, 64, 128)]}
    assert not any(p["type"] == "ScaleBlockSizeA" for p in predicates)
    assert {p["type"]: p.get("value") for p in predicates}["ScaleZeroPointA"] is True
    assert {p["type"]: p.get("value") for p in predicates}["Int4EncodingA"] == "UnsignedBias8"

    config["ScaleBlockSizesA"] = [32]
    fixed = [state(p) for p in ProblemType.FromOriginalState(
        OriginalProblemType(config, False)).predicates(includeType=True)]
    assert {"type": "ScaleBlockSizeA", "value": 32} in fixed
    assert not any(p["type"] == "Or" for p in fixed)

    # W4A16 always needs group scaling.
    del config["ScaleBlockSizesA"]
    with pytest.raises(ValueError, match="ScaleBlockSizesA must be non-empty"):
        ProblemType.FromOriginalState(OriginalProblemType(config, False))


@pytest.mark.parametrize("m,k,tile_m", [
    (4608, 3584, 128), (3584, 3584, 64),
    (37888, 3584, 128), (3584, 18944, 64),
])
@pytest.mark.parametrize("n,tile_n", [(109, 112), (128, 128)])
def test_symmetric_prefill_rounded_query_selects_narrow_tile(m, k, tile_m, n, tile_n):
    from types import SimpleNamespace
    from Tensile.SolutionLibrary import MatchingLibrary

    path = DIRECTORY.parents[2] / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_Q27B.yaml")
    logic = yaml.safe_load(path.read_text())
    solutions = {s["SolutionIndex"]: s for s in logic["Solutions"]}
    library = MatchingLibrary.FromOriginalState(
        {"indexOrder": logic["IndexOrder"], "distance": logic["LibraryType"],
         "table": logic["ExactLogic"]},
        {index: SimpleNamespace(index=index) for index in solutions})
    # Callers round 84–118-token prompts to 128 for the heuristic query.
    rows = [row for row in state(library)["table"] if row["key"] == [m, n, 1, k]]
    assert len(rows) == 1
    selected = solutions[rows[0]["index"]]
    assert (selected["MacroTile0"], selected["MacroTile1"]) == (tile_m, tile_n)
    assert not selected.get("W4A16Decode")
    assert selected.get("CustomKernelName", "") == ""


def test_runtime_group_lazy_library_shares_one_group_partition():
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
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml")
    logic = yaml.safe_load(path.read_text())
    prepareLibraryLogicDict(logic)
    library, _ = MasterSolutionLibrary.FromOriginalState(
        logic, logic["Solutions"], False, False, False, None, {}, True,
        solutionClass=IndexOnlySolution)
    assert len(library.lazyLibraries) == 1
    assert "SABBG32x64x128_ZP1_UnsignedBias8" in next(iter(library.lazyLibraries))


@pytest.mark.parametrize("groups, dataTypeA", [
    ([16, 32], 24), ([256], 24), ([32, "64"], 24), (["32"], 24), ("32", 24), (None, 24), ([], 24), ([32], 4)])
def test_invalid_runtime_groups_are_rejected(groups, dataTypeA):
    from Tensile.Contractions import ProblemType
    from Tensile.SolutionStructs import ProblemType as OriginalProblemType

    problem = OriginalProblemType(dict(
        DataType=4, DataTypeA=dataTypeA, DataTypeB=4, DestDataType=4,
        ComputeDataType=0, HighPrecisionAccumulate=True,
        UseScaleAB=""), False)
    problem["ScaleBlockSizesA"] = groups
    with pytest.raises(ValueError, match="ScaleBlockSizesA"):
        ProblemType.FromOriginalState(problem)


@pytest.mark.parametrize("groups", [[32, 64, 128], [32]])
def test_multi_group_custom_logic_requires_runtime_argument(monkeypatch, toolchain, groups):
    from Tensile import LibraryIO
    path = DIRECTORY.parents[2] / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
    logic = yaml.safe_load(path.read_text())
    logic["ProblemType"]["ScaleBlockSizesA"] = groups
    logic["Solutions"][0]["CustomKernelName"] = "missing_group_argument"
    monkeypatch.setattr(LibraryIO, "getCustomKernelConfig", lambda *_: {
        "CustomKernel": {"name": "missing_group_argument", "args": []}})
    def parse():
        return LibraryIO.parseLibraryLogicData(logic, str(path), toolchain[2], False, False,
                                              False, toolchain[1], True)

    if len(groups) > 1:
        with pytest.raises(ValueError, match="runtime group-size argument"):
            parse()
    else:
        # Fixed groups pass this check and reach the incomplete mock metadata.
        with pytest.raises(KeyError, match="threads"):
            parse()


def test_unsigned_symmetric_logic_preserves_shapes_and_selects_small_n_decode():
    logic_dir = DIRECTORY.parents[2] / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality"
    asymmetric = yaml.safe_load((logic_dir / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml").read_text())
    symmetric = yaml.safe_load((logic_dir / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_Q27B.yaml").read_text())
    original_shapes = {tuple(row[0]) for row in asymmetric["ExactLogic"]}
    assert original_shapes <= {tuple(row[0]) for row in symmetric["ExactLogic"]}
    for shape, selection in symmetric["ExactLogic"]:
        if tuple(shape) not in original_shapes:
            continue
        reference = list(shape)
        if shape[1] in (2, 3, 4):
            reference[1] = 1
        old_index = next(row[1][0] for row in asymmetric["ExactLogic"] if row[0] == reference)
        expected = list(asymmetric["Solutions"][old_index].get("W4A16Decode", []))
        if expected and shape[1] in (2, 3, 4):
            expected[4] = 4
        assert symmetric["Solutions"][selection[0]].get("W4A16Decode", []) == expected
    for shape, selection in symmetric["ExactLogic"]:
        selected = symmetric["Solutions"][selection[0]]
        schedule = selected.get("W4A16Decode", [])
        if schedule and schedule[:2] == [4, 4]:
            expected_prefetch = 4 if shape[1] == 1 and shape[3] % 1024 == 0 else 1
            assert selected.get("W4A16DecodePrefetch", 1) == expected_prefetch
    assert symmetric["ProblemType"]["ScaleZeroPointA"] is False
    assert symmetric["ProblemType"]["Int4EncodingA"] == "UnsignedBias8"
    assert symmetric["ProblemType"]["ScaleBlockSizesA"] == [32, 64, 128]


@pytest.mark.parametrize("suffix", SCHEDULES)
def test_small_n_decode_bounds_and_abi(toolchain, suffix):
    params = list(SCHEDULES[suffix])
    original, source = _emit_solution(toolchain, decode_solution(toolchain, decode=params))
    params[4] = 4
    small, small_source = _emit_solution(toolchain, decode_solution(toolchain, decode=params))
    assert [state(p) for p in ProblemPredicate.W4A16DecodePredicates(small)] == [
        {"type": "SizeEqual", "index": 2, "value": 1},
        {"type": "SizeGreaterThan", "index": 3, "value": 0},
        {"type": "SizeGreaterThan", "index": 1, "value": 0},
        {"type": "SizeLessThan", "index": 1, "value": 5},
    ]
    assert metadata(source)[".args"] == metadata(small_source)[".args"]
    assert original["CustomKernel"]["macrotile"] == small["CustomKernel"]["macrotile"]


def test_wide_decode_launch_geometry(toolchain):
    config, source = _emit_solution(toolchain, decode_solution(toolchain, decode=[2, 1, 4, 512, 1, True, True]))
    assert config["WorkGroup"] == [16, 1, 32]
    assert metadata(source)[".max_flat_workgroup_size"] == 512
    assert {"type": "SizeEqual", "index": 1, "value": 1} in [state(p) for p in ProblemPredicate.W4A16DecodePredicates(config)]


@pytest.mark.parametrize("name", ["generated_prefill", "unrelated_kernel", ""])
def test_decode_bounds_do_not_apply_to_other_kernels(name):
    assert ProblemPredicate.W4A16DecodePredicates({"CustomKernelName": name}) == []


def test_no_shipped_w4a16_custom_assembly_or_hip_source():
    assert not list(DIRECTORY.glob("RuntimeGroup*.s"))
    assert not (DIRECTORY / "Source/w4a16_decode.hip").exists()
    assert not (DIRECTORY / "Source/generate_w4a16_decode.py").exists()


def test_shipped_w4a16_kernels_do_not_support_scalar_operand_scales():
    from Tensile.Contractions import ProblemType
    from Tensile.SolutionStructs import ProblemType as OriginalProblemType

    root = DIRECTORY.parents[2] / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151"
    catalogs = list(root.glob("*/*I4*.yaml"))
    assert catalogs
    for path in catalogs:
        logic = yaml.safe_load(path.read_text())
        configs = [logic["ProblemType"]] + [s.get("ProblemType", logic["ProblemType"]) for s in logic["Solutions"]]
        for config in configs:
            assert config["UseScaleAB"] in ("", "Block"), path
            problem = ProblemType.FromOriginalState(OriginalProblemType(config, False))
            predicates = [state(p) for p in problem.predicates(includeType=True)]
            assert {"type": "UseScaleAB", "value": ""} in predicates, path
            assert any(p["type"] in ("ScaleBlockSizeA", "Or") for p in predicates), path


@pytest.mark.parametrize("groups, zero_point, encoding, token", [
    ([32, 64, 128], True, "UnsignedBias8", "_SABBGZPU8_"),
    ([32, 64, 128], False, "Signed", "_SABBG_"),
    ([32], True, "UnsignedBias8", "_SABB32ZPU8_"),
    ([32], False, "Signed", "_SABB32_"),
])
def test_problem_type_name_separates_runtime_group(groups, zero_point, encoding, token):
    from Tensile.SolutionStructs import ProblemType

    name = str(ProblemType(dict(
        DataType=4, DataTypeA=24, DataTypeB=4, DestDataType=4, ComputeDataType=0,
        HighPrecisionAccumulate=True, ScaleBlockSizesA=groups,
        ScaleZeroPointA=zero_point, Int4EncodingA=encoding), False))
    assert token in name
