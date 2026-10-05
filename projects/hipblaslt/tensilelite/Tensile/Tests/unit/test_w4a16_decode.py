# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch bounds and argument ABI for runtime-group decode."""
from pathlib import Path

import pytest
import yaml

from Tensile.Common.Utilities import state
from Tensile.Contractions import ProblemPredicate
from Tensile.CustomKernels import _readEmbeddedYaml, readCustomKernelConfig

pytestmark = pytest.mark.unit

RUNTIME_NAME = "RuntimeGroup_Decode{suffix}_UnsignedBias8_gfx1151"
RUNTIME_GENERAL = "RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_gfx1151"
DIRECTORY = Path(__file__).parents[2] / "CustomKernels"

def runtime_name(suffix):
    if suffix == "_W4":
        return "RuntimeGroup_Decode_W4_NativePerm_LinearK_UnsignedBias8_gfx1151"
    return RUNTIME_NAME.format(suffix=suffix)


@pytest.mark.parametrize("suffix", ["_W4", "_W4_U1_A4", "_W4_NativePerm"])
def test_decode_selection_bounds(suffix):
    name = runtime_name(suffix)
    config = readCustomKernelConfig(name, DIRECTORY)
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


@pytest.mark.parametrize("suffix", ["_W4", "_W4_U1_A4", "_W4_NativePerm"])
def test_decode_universal_arguments_match_matrix_kernel(suffix):
    def metadata(name):
        return _readEmbeddedYaml(name, DIRECTORY)["amdhsa.kernels"][0]

    name = runtime_name(suffix)
    general_name = RUNTIME_GENERAL
    decode, general = metadata(name), metadata(general_name)
    # HIP compilation must preserve the universal layout that the existing
    # host library supplies, including unused fields and trailing offsets.
    for key in (".kernarg_segment_size", ".kernarg_segment_align"):
        if key == ".kernarg_segment_size":
            # HIP omits the final four padding bytes supplied by the host.
            assert decode[key] == 164
            assert (decode[key] + 7) // 8 * 8 == general[key]
        else:
            assert decode[key] == general[key]
    assert [(arg[".offset"], arg[".size"]) for arg in decode[".args"]] == [
        (arg[".offset"], arg[".size"]) for arg in general[".args"]
    ]
    assert decode[".args"][-1][".offset"] == 160
    assert decode[".args"][-1][".size"] == 4
    config = readCustomKernelConfig(name, DIRECTORY)
    x, y, z = config["WorkGroup"]
    assert x * y * z == decode[".max_flat_workgroup_size"]


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
            assert solution["CustomKernelName"] == runtime_name(suffix)
        else:
            assert solution["EnableMatrixInstruction"]
            assert solution["WorkGroupMapping"] in (1, 4)
            if row["key"][1] in (2, 3, 4) or row["key"] == [34816, 2048, 1, 5120]:
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
        logic["ProblemType"]["ScaleBlockSizesA"] = []
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
        assert len(table) == 25
        assert len({tuple(row["key"]) for row in table}) == 25
        assert all(row["index"] in library.solutions for row in table)


@pytest.mark.parametrize("suffix", ["_W4", "_W4_U1_A4", "_W4_NativePerm"])
def test_decode_custom_launch_preserves_universal_abi(suffix):
    from Tensile.CustomKernels import getCustomKernelConfig

    name = runtime_name(suffix)
    general_name = RUNTIME_GENERAL
    decode = getCustomKernelConfig(name, {}, DIRECTORY)["CustomKernel"]
    general = getCustomKernelConfig(general_name, {}, DIRECTORY)["CustomKernel"]
    general_args = [dict(a) for a in general["args"]]
    assert general_args[-1].pop("padding") == 4
    assert decode["args"] == general_args
    tail = [a["semantic"] for a in decode["args"]][24:]
    assert tail == ["AddressScaleZeroA", "BatchOffsetD", "BatchOffsetC",
                    "BatchOffsetA", "BatchOffsetB"] + ["ScaleBlockSizeA"]
    assert all(a["type"] == "int64" for a in decode["args"][25:29])
    assert decode["grid"] == ["TilesXYBatchGSU", "One", "One"]
    assert decode["macrotile"] == [4, 1, 256]


def test_runtime_group_predicate_accepts_only_declared_groups():
    from Tensile.Contractions import ProblemType
    from Tensile.SolutionStructs import ProblemType as OriginalProblemType

    config = dict(DataType=4, DataTypeA=24, DataTypeB=4, DestDataType=4,
                  ComputeDataType=0, HighPrecisionAccumulate=True,
                  UseScaleAB="Block", ScaleBlockSizeA=32,
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

    config["ScaleBlockSizesA"] = []
    fixed = ProblemType.FromOriginalState(OriginalProblemType(config, False))
    assert {"type": "ScaleBlockSizeA", "value": 32} in [
        state(p) for p in fixed.predicates(includeType=True)]


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


@pytest.mark.parametrize("groups", [[16, 32], [64, 128], ["32"], "32", None])
def test_invalid_runtime_groups_are_rejected(groups):
    from Tensile.Contractions import ProblemType
    from Tensile.SolutionStructs import ProblemType as OriginalProblemType

    problem = OriginalProblemType(dict(
        DataType=4, DataTypeA=24, DataTypeB=4, DestDataType=4,
        ComputeDataType=0, HighPrecisionAccumulate=True,
        UseScaleAB="Block", ScaleBlockSizeA=32), False)
    problem["ScaleBlockSizesA"] = groups
    with pytest.raises(ValueError, match="ScaleBlockSizesA"):
        ProblemType.FromOriginalState(problem)


def test_multi_group_logic_requires_runtime_argument(monkeypatch):
    from Tensile import LibraryIO

    path = DIRECTORY.parents[2] / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
        "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml")
    logic = yaml.safe_load(path.read_text())
    get_config = LibraryIO.getCustomKernelConfig

    def without_group_argument(*args, **kwargs):
        config = get_config(*args, **kwargs)
        config["CustomKernel"]["args"] = [
            a for a in config["CustomKernel"]["args"]
            if a.get("semantic") != "ScaleBlockSizeA"]
        return config

    monkeypatch.setattr(LibraryIO, "getCustomKernelConfig", without_group_argument)
    with pytest.raises(ValueError, match="runtime group-size argument"):
        LibraryIO.parseLibraryLogicData(logic, str(path), None, False, False,
                                       False, {}, True)


def test_unsigned_symmetric_logic_preserves_shapes_and_selects_small_n_decode():
    logic_dir = DIRECTORY.parents[2] / (
        "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality")
    asymmetric = yaml.safe_load((logic_dir / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml").read_text())
    symmetric = yaml.safe_load((logic_dir / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGU8_Q27B.yaml").read_text())
    original_shapes = {tuple(row[0]) for row in asymmetric["ExactLogic"]}
    assert original_shapes <= {tuple(row[0]) for row in symmetric["ExactLogic"]}
    old_solutions = {s["SolutionIndex"]: s["CustomKernelName"] for s in asymmetric["Solutions"]}
    new_solutions = {s["SolutionIndex"]: s["CustomKernelName"] for s in symmetric["Solutions"]}
    for shape, selection in symmetric["ExactLogic"]:
        if tuple(shape) not in original_shapes:
            continue
        reference = list(shape)
        if shape[1] in (2, 3, 4):
            reference[1] = 1
        old_index = next(row[1][0] for row in asymmetric["ExactLogic"] if row[0] == reference)
        expected = old_solutions[old_index].replace("SABBGZPU8", "SABBGU8")
        if "Decode" in expected:
            expected = expected.replace("_UnsignedBias8", "_Symmetric_UnsignedBias8")
        if shape[1] in (2, 3, 4):
            expected = expected.replace("_UnsignedBias8", "_N4_UnsignedBias8")
        if tuple(shape) in {(3584, 1, 1, 18944), (3584, 1, 1, 3584), (4608, 1, 1, 3584)}:
            expected = "RuntimeGroup_Decode_W2_U1_A4_T512_NativePerm_LinearK_Symmetric_UnsignedBias8_gfx1151"
        assert new_solutions[selection[0]] == expected
    assert symmetric["ProblemType"]["ScaleZeroPointA"] is False
    assert symmetric["ProblemType"]["Int4EncodingA"] == "UnsignedBias8"
    assert symmetric["ProblemType"]["ScaleBlockSizesA"] == [32, 64, 128]
    for solution in symmetric["Solutions"]:
        name = solution["CustomKernelName"]
        metadata = _readEmbeddedYaml(name, DIRECTORY)["amdhsa.kernels"][0]
        original_name = name.replace("SABBGU8", "SABBGZPU8").replace("_Symmetric_", "_").replace("_N4_", "_")
        if name.startswith("RuntimeGroup_Prefill_"):
            original_name = next(s["CustomKernelName"] for s in asymmetric["Solutions"]
                                 if "Decode" not in s["CustomKernelName"])
        if "_T512_" in name:
            original_name = "RuntimeGroup_Decode_W4_NativePerm_UnsignedBias8_gfx1151"
        original = _readEmbeddedYaml(original_name, DIRECTORY)["amdhsa.kernels"][0]
        assert [(arg[".offset"], arg[".size"]) for arg in metadata[".args"]] == [
            (arg[".offset"], arg[".size"]) for arg in original[".args"]]
        assert metadata[".args"][-1][".offset"] == 160
        source = (DIRECTORY / (name + ".s")).read_text()
        if "Decode" not in name:
            assert "buffer_load_d16_u8" not in source
            assert "v_pk_fma_f16" not in source
            assert "0xe408e408" in source and "0xd480d480" in source


@pytest.mark.parametrize("suffix", ["_W4", "_W4_U1_A4", "_W4_NativePerm"])
def test_small_n_decode_bounds_and_abi(suffix):
    original_name = runtime_name(suffix).replace("_UnsignedBias8", "_Symmetric_UnsignedBias8")
    name = original_name.replace("_UnsignedBias8", "_N4_UnsignedBias8")
    config = readCustomKernelConfig(name, DIRECTORY)
    assert config["AssertSizeEqual"] == {2: 1}
    assert config["AssertSizeGreaterThan"] == {1: 0, 3: 0}
    assert config["AssertSizeLessThan"] == {1: 5}
    from Tensile.CustomKernels import getCustomKernelConfig

    # The filtered custom config must retain the bound for serialization.
    filtered = getCustomKernelConfig(name, config["InternalSupportParams"], DIRECTORY)
    assert filtered["AssertSizeLessThan"] == {1: 5}
    predicate = ProblemPredicate.FromOriginalKeyPair(
        ("AssertSizeLessThan", config["AssertSizeLessThan"]))
    assert state(predicate) == {"type": "SizeLessThan", "index": 1, "value": 5}
    original = _readEmbeddedYaml(original_name, DIRECTORY)["amdhsa.kernels"][0]
    small_n = _readEmbeddedYaml(name, DIRECTORY)["amdhsa.kernels"][0]
    assert small_n[".kernarg_segment_size"] == original[".kernarg_segment_size"]
    assert [(a[".offset"], a[".size"]) for a in small_n[".args"]] == [
        (a[".offset"], a[".size"]) for a in original[".args"]]


def test_unsigned_symmetric_regeneration_preserves_measured_selections(tmp_path):
    import runpy
    from Tensile.CustomYamlLoader import load_yaml_stream

    generator = runpy.run_path(str(DIRECTORY / "Source/generate_w4a16_unsigned_symmetric.py"))
    header = "# Copyright Advanced Micro Devices, Inc., or its affiliates.\n# SPDX-License-Identifier: MIT\n# Test logic\n# Runtime groups\n"
    generated = {"Solutions": [{"SolutionIndex": 0, "CustomKernelName": "matrix"},
                               {"SolutionIndex": 1, "CustomKernelName": "decode"}],
                 "ExactLogic": [[[64, 1, 1, 256], [0, 1.0]]]}
    shared = []
    for solution in generated["Solutions"]:
        solution["MatrixInstruction"] = shared
    previous = {"Solutions": generated["Solutions"],
                "ExactLogic": [[[64, 1, 1, 256], [1, 2.0]],
                               [[128, 2, 1, 256], [1, 3.0]]]}
    text = header + yaml.safe_dump(generated)
    result = generator["preserve_exact_logic"](text, yaml.safe_dump(previous))
    assert yaml.safe_load(result)["ExactLogic"] == previous["ExactLogic"]
    path = tmp_path / "logic.yaml"
    path.write_text(result)
    assert load_yaml_stream(path, yaml.SafeLoader) == yaml.safe_load(result)
    assert generator["preserve_exact_logic"](result, result) == result
    previous["Solutions"][1]["CustomKernelName"] = "different"
    result = generator["preserve_exact_logic"](text, yaml.safe_dump(previous))
    retained = yaml.safe_load(result)
    assert retained["Solutions"][2]["CustomKernelName"] == "different"
    assert retained["Solutions"][2]["SolutionIndex"] == 2
    assert all(selection[0] == 2 for _, selection in retained["ExactLogic"])
    assert generator["preserve_exact_logic"](result, result) == result


def test_wide_decode_launch_geometry():
    name = "RuntimeGroup_Decode_W2_U1_A4_T512_NativePerm_LinearK_Symmetric_UnsignedBias8_gfx1151"
    config = readCustomKernelConfig(name, DIRECTORY)
    metadata = _readEmbeddedYaml(name, DIRECTORY)["amdhsa.kernels"][0]
    assert config["WorkGroup"] == [16, 1, 32]
    assert metadata[".max_flat_workgroup_size"] == 512
    assert config["AssertSizeEqual"] == {1: 1, 2: 1}
