# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch bounds and argument ABI for runtime-group decode."""
from pathlib import Path

import pytest
import yaml

from Tensile.Common.Utilities import state
from Tensile.Contractions import ProblemPredicate
from Tensile.CustomKernels import (_readEmbeddedYaml, getCustomKernelConfig,
                                   readCustomKernelConfig)

pytestmark = pytest.mark.unit

RUNTIME_GENERAL = "RuntimeGroup_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_UserArgs_MT64x160x64_MI16x16x1_gfx1151"
DIRECTORY = Path(__file__).parents[2] / "CustomKernels"
Q27B = DIRECTORY.parents[2] / (
    "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality/"
    "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml")

DECODE_NAMES = sorted(p.stem for p in DIRECTORY.glob("RuntimeGroup_Decode_*.s"))
ARG_BYTES = {"uint32": 4, "float32": 4, "uint64": 8, "address": 8}


def size_bounds(config):
    """Flattened predicates emitted by the kernel's AssertSize* declarations."""
    def flat(p):
        return [q for v in p["value"] for q in flat(v)] if p["type"] == "And" else [p]
    predicates = []
    for key in ("AssertSizeEqual", "AssertSizeGreaterThan", "AssertSizeLessThan"):
        if key in config:
            predicates += flat(state(ProblemPredicate.FromOriginalKeyPair((key, config[key]))))
    return predicates


def test_decode_kernels_are_discovered():
    assert len(DECODE_NAMES) == 10


def admits(bounds, n=1, batch=1, m=64, k=256):
    size = [m, n, batch, k]
    compare = {"SizeEqual": int.__eq__, "SizeGreaterThan": int.__gt__,
               "SizeLessThan": int.__lt__}
    return all(compare[p["type"]](size[p["index"]], p["value"]) for p in bounds)


@pytest.mark.parametrize("name", DECODE_NAMES)
def test_decode_selection_bounds(name):
    config = readCustomKernelConfig(name, DIRECTORY)
    # The bounds must survive the custom.config filtering applied at load time.
    bounds = size_bounds(
        getCustomKernelConfig(name, config["InternalSupportParams"], DIRECTORY))
    max_n = 4 if "_N4_" in name else 1
    assert [n for n in range(1, 8) if admits(bounds, n=n)] == list(range(1, max_n + 1))
    assert not admits(bounds, batch=2)


@pytest.mark.parametrize("name", DECODE_NAMES)
def test_decode_abi_matches_runtime_group_kernel(name):
    """Declared args reproduce the HIP kernarg layout, which is the general
    runtime-group kernel's."""
    def metadata(kernel):
        return _readEmbeddedYaml(kernel, DIRECTORY)["amdhsa.kernels"][0]

    def layout(meta):
        return [(a[".offset"], a[".size"]) for a in meta[".args"]]

    config = getCustomKernelConfig(name, {}, DIRECTORY)
    declared = config["CustomKernel"]["args"]
    decode = metadata(name)
    offset = 0
    for arg, meta in zip(declared, decode[".args"], strict=True):
        assert (offset, ARG_BYTES[arg["type"]]) == (meta[".offset"], meta[".size"]), arg
        assert (arg["type"] == "address") == (meta[".value_kind"] == "global_buffer"), arg
        offset += ARG_BYTES[arg["type"]] + arg.get("padding", 0)
    assert offset == decode[".kernarg_segment_size"]

    general = getCustomKernelConfig(RUNTIME_GENERAL, {}, DIRECTORY)["CustomKernel"]["args"]
    assert [(a["semantic"], ARG_BYTES[a["type"]]) for a in declared] == [
        (a["semantic"], ARG_BYTES[a["type"]]) for a in general]
    assert layout(decode) == layout(metadata(RUNTIME_GENERAL))

    x, y, z = config["WorkGroup"]
    assert x * y * z == decode[".max_flat_workgroup_size"]
    assert config["CustomKernel"]["threads"] == [decode[".max_flat_workgroup_size"], 1, 1]


def test_block_scale_equality_grids_do_not_merge_duplicate_shape_keys():
    from copy import deepcopy
    from types import SimpleNamespace

    from Tensile.LibraryIO import prepareLibraryLogicDict
    from Tensile.SolutionLibrary import MasterSolutionLibrary

    class IndexOnlySolution:
        @staticmethod
        def FromSolutionStruct(solution, *_args):
            return SimpleNamespace(index=solution["SolutionIndex"])

    original = yaml.safe_load(Q27B.read_text())
    rows = len(original["ExactLogic"])
    merged = None
    # None keeps the shipped runtime group.
    for override in [None, (32, True, "UnsignedBias8"), (32, True, "Signed"),
                     (128, True, "UnsignedBias8"), (32, False, "UnsignedBias8")]:
        logic = deepcopy(original)
        if override:
            group, zero_point, encoding = override
            logic["ProblemType"].update(
                ScaleBlockSizesA=[group],
                ScaleZeroPointA=zero_point, Int4EncodingA=encoding)
        prepareLibraryLogicDict(logic)
        library, _ = MasterSolutionLibrary.FromOriginalState(
            logic, logic["Solutions"], False, False, False, None, {}, True,
            solutionClass=IndexOnlySolution)
        # A runtime group shares one partition rather than one per group size.
        assert len(library.lazyLibraries) == 1
        if merged is None:
            merged = library
        else:
            merged.merge(library)
    assert len(merged.lazyLibraries) == 5
    assert any("SABBG32x64x128_ZP1_UnsignedBias8" in key for key in merged.lazyLibraries)
    for library in merged.lazyLibraries.values():
        table = state(library.library)["rows"][0]["library"]["table"]
        assert len({tuple(row["key"]) for row in table}) == len(table) == rows
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


@pytest.mark.parametrize("groups, dataTypeA", [
    ([16, 32], 24), ([256], 24), ([32, "64"], 24), ("32", 24), ([], 24), ([32], 4)])
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


@pytest.mark.parametrize("groups, raises", [([32, 64, 128], True), ([32], False)])
def test_multi_group_logic_requires_runtime_argument(monkeypatch, groups, raises):
    from Tensile import LibraryIO

    logic = yaml.safe_load(Q27B.read_text())
    logic["ProblemType"]["ScaleBlockSizesA"] = groups
    get_config = LibraryIO.getCustomKernelConfig

    def without_group_argument(*args, **kwargs):
        config = get_config(*args, **kwargs)
        config["CustomKernel"]["args"] = [
            a for a in config["CustomKernel"]["args"]
            if a.get("semantic") != "ScaleBlockSizeA"]
        return config

    monkeypatch.setattr(LibraryIO, "getCustomKernelConfig", without_group_argument)

    def parse():
        LibraryIO.parseLibraryLogicData(logic, str(Q27B), None, False, False,
                                       False, {}, True)
    if raises:
        with pytest.raises(ValueError, match="runtime group-size argument"):
            parse()
    else:
        # Fails later without an isaInfoMap; only the group-argument check matters.
        with pytest.raises(Exception) as e:
            parse()
        assert "runtime group-size argument" not in str(e.value)


def test_decode_bounds_override_the_consuming_solution():
    """custom.config wins over the logic-file solution, so the bounds stay attached."""
    from Tensile.ExecutionPolicy import normalize_execution_policy_with_defaults

    name = DECODE_NAMES[0]
    config = readCustomKernelConfig(name, DIRECTORY)
    merged = normalize_execution_policy_with_defaults(
        config, {"CustomKernelName": name, "AssertSizeEqual": {2: 4}})
    assert size_bounds(merged) == size_bounds(config)


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
            assert config["UseScaleAB"] == "", path
            problem = ProblemType.FromOriginalState(OriginalProblemType(config, False))
            predicates = [state(p) for p in problem.predicates(includeType=True)]
            assert {"type": "UseScaleAB", "value": ""} in predicates, path
            assert any(p["type"] in ("ScaleBlockSizeA", "Or") for p in predicates), path
