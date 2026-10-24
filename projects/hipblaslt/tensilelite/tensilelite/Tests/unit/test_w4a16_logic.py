# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Shipped gfx1151 W4A16 logic and runtime-group dispatch."""
from pathlib import Path
import pytest
import yaml
from Tensile.Common.Utilities import state
from test_w4a16_block_dequant import toolchain, GFX1151_LOGIC, needs_logic

pytestmark = pytest.mark.unit
DIRECTORY = Path(__file__).parents[2] / "CustomKernels"


@needs_logic
def test_q27b_equality_dispatch_keys():
    from types import SimpleNamespace

    from Tensile.SolutionLibrary import MatchingLibrary

    path = GFX1151_LOGIC / "Equality/gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
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
        if row["key"][1] <= 4:
            # Decode: dot2 WaveSplitK with A and B fed straight from global reads.
            assert solution["WaveSplitK"]
            assert solution["DirectToVgprDot2A"] and solution["DirectToVgprDot2B"]
            assert solution.get("CustomKernelName", "") == ""
        else:
            assert solution["EnableMatrixInstruction"]
            assert solution.get("CustomKernelName", "") == ""
            assert solution["WorkGroupMapping"] in (1, 4)
            if row["key"] == [34816, 2048, 1, 5120]:
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


@needs_logic
def test_block_scale_equality_grids_do_not_merge_duplicate_shape_keys():
    from copy import deepcopy
    from types import SimpleNamespace

    from Tensile.LibraryIO import prepareLibraryLogicDict
    from Tensile.SolutionLibrary import MasterSolutionLibrary

    class IndexOnlySolution:
        @staticmethod
        def FromSolutionStruct(solution, *_args):
            return SimpleNamespace(index=solution["SolutionIndex"])

    path = GFX1151_LOGIC / "Equality/gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
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


@needs_logic
@pytest.mark.parametrize("m,k,tile_m", [
    (4608, 3584, 128), (3584, 3584, 64),
    (37888, 3584, 128), (3584, 18944, 64),
])
@pytest.mark.parametrize("n,tile_n", [(109, 112), (128, 128)])
def test_symmetric_prefill_rounded_query_selects_narrow_tile(m, k, tile_m, n, tile_n):
    from types import SimpleNamespace
    from Tensile.SolutionLibrary import MatchingLibrary

    path = GFX1151_LOGIC / "Equality/gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_Q27B.yaml"
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
    assert selected.get("CustomKernelName", "") == ""


@needs_logic
def test_runtime_group_lazy_library_shares_one_group_partition():
    from types import SimpleNamespace
    from Tensile.LibraryIO import prepareLibraryLogicDict
    from Tensile.SolutionLibrary import MasterSolutionLibrary

    class IndexOnlySolution:
        @staticmethod
        def FromSolutionStruct(solution, *_args):
            return SimpleNamespace(index=solution["SolutionIndex"])

    path = GFX1151_LOGIC / "Equality/gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
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


@needs_logic
@pytest.mark.parametrize("groups", [[32, 64, 128], [32]])
def test_multi_group_custom_logic_requires_runtime_argument(monkeypatch, toolchain, groups):
    from Tensile import LibraryIO
    path = GFX1151_LOGIC / "Equality/gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
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


def test_no_shipped_w4a16_custom_assembly_or_hip_source():
    assert not list(DIRECTORY.glob("RuntimeGroup*.s"))
    assert not (DIRECTORY / "Source/w4a16_decode.hip").exists()
    assert not (DIRECTORY / "Source/generate_w4a16_decode.py").exists()


@needs_logic
def test_shipped_w4a16_kernels_do_not_support_scalar_operand_scales():
    from Tensile.Contractions import ProblemType
    from Tensile.SolutionStructs import ProblemType as OriginalProblemType

    catalogs = list(GFX1151_LOGIC.glob("*/*I4*.yaml"))
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
