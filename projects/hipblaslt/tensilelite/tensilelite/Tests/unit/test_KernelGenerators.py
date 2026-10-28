# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the generator form of a logic-file CustomKernel (KernelGenerators)."""

import copy
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.unit

from Tensile import Contractions, LibraryIO
from Tensile.Common import IsaVersion
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.KernelGenerators import (
    GeneratorKernelRejected,
    SkinnyGEMM,
    generatedCodeObjectName,
    getGeneratorKernelConfig,
    getGeneratorKernelSource,
    isGeneratorKernel,
    validateGeneratorKernel,
)
from Tensile.TensileCreateLibrary import Run
from Tensile.Toolchain.Assembly import buildAssemblyCodeObjectFiles, makeAssemblyToolchain
from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

LOGIC_FILE = (
    Path(__file__).parent
    / "test_data"
    / "skinny_gemm_logic"
    / "gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml"
)

S1 = dict(bm=1, tn=1, lanesK=64, waves=4, prefetch=2, upIters=2, partialSteps=0, group=1, numXcds=1, grid3d=True, useBeta=False)
S2 = dict(bm=1, tn=1, lanesK=16, waves=4, prefetch=2, upIters=0, partialSteps=0, group=1, numXcds=1, grid3d=True, useBeta=False)
S3 = dict(bm=2, tn=2, lanesK=16, waves=4, prefetch=1, upIters=0, partialSteps=1, group=1, numXcds=1, grid3d=True, useBeta=False)
S4 = dict(bm=1, tn=1, lanesK=64, waves=4, prefetch=2, upIters=2, partialSteps=0, group=1, numXcds=1, grid3d=False, useBeta=False)
S5 = dict(bm=1, tn=2, lanesK=16, waves=2, prefetch=2, upIters=3, partialSteps=1, group=2, numXcds=8, grid3d=False, useBeta=False)


def betaVariant(params):
    return {**params, "useBeta": True}


# The fixture lists every knob set with useBeta false, then true.
FIXTURE_SOLUTIONS = [variant for p in (S1, S2, S3, S4, S5) for variant in (p, betaVariant(p))]

EXPECTED_ARGS = [
    {"type": "address", "semantic": "AddressD"},
    {"type": "address", "semantic": "AddressA"},
    {"type": "address", "semantic": "AddressB"},
    {"type": "address", "semantic": "AddressMXScaleA"},
    {"type": "address", "semantic": "AddressMXScaleB"},
    {"type": "uint64", "semantic": "ConstantZero"},
    {"type": "address", "semantic": "AddressC"},
    {"type": "int32", "semantic": "StrideD0Bytes"},
    {"type": "int32", "semantic": "StrideA0Bytes"},
    {"type": "int32", "semantic": "StrideB0Bytes"},
    {"type": "int32", "semantic": "SizeFree0"},
    {"type": "int32", "semantic": "SizeFree1"},
    {"type": "int32", "semantic": "SizeSum"},
    {"type": "int32", "semantic": "SizeSum"},
    {"type": "int32", "semantic": "StrideC0Bytes"},
    {"type": "float32", "semantic": "Alpha"},
    {"type": "float32", "semantic": "Beta"},
    {"type": "int32", "semantic": "NumWorkGroupsX"},
    {"type": "int32", "semantic": "NumWorkGroupsY"},
    {"type": "int32", "semantic": "NumWorkGroupsZ"},
]

COMMON_ASSERTS = {
    "AssertSizeEqual": {2: 1},
    "AssertStrideAEqual": {0: 1},
    "AssertStrideBEqual": {0: 1},
    "AssertStrideDEqual": {0: 1},
    "AssertStrideAMultiple": {1: 4},
    "AssertStrideBMultiple": {1: 4},
    "AssertNoBias": True,
    "AssertNoActivation": True,
    "AssertNoScaleAB": True,
    "AssertNoScaleAlphaVec": True,
}
# Without useBeta the kernel ignores C and beta; with it C is read with the vector widths of D.
NO_BETA_ASSERTS = {"AssertBetaZero": True}
BETA_ASSERTS = {"AssertStrideCEqual": {0: 1}}


def kernelNameOf(p):
    return (
        f"gemm_BM{p['bm']}_TN{p['tn']}_LANES_K{p['lanesK']}_WAVES{p['waves']}"
        f"_UP_ITERS{p['upIters']}_PARTIAL_STEPS{p['partialSteps']}_PF{p['prefetch']}"
        f"_GROUP{p['group']}_NUM_XCDS{p['numXcds']}_GRID_3D{int(p['grid3d'])}"
        f"_USE_BETA{int(p.get('useBeta', False))}"
    )


@pytest.fixture
def fakeCodegen(monkeypatch):
    """A stand-in for tensilelite_codegen so that the pure logic runs without the module."""
    module = types.ModuleType("tensilelite_codegen")

    class SkinnyGEMMConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    def skinnyGEMMKernelName(config):
        if config.lanesK not in (4, 8, 16, 64):
            raise ValueError(f"unsupported lanesK {config.lanesK}")
        return kernelNameOf(config.__dict__)

    module.SkinnyGEMMConfig = SkinnyGEMMConfig
    module.skinnyGEMMKernelName = skinnyGEMMKernelName

    def emitSkinnyGEMM(config, arch="gfx950"):
        return f"// {skinnyGEMMKernelName(config)} for {arch}\n"

    module.emitSkinnyGEMM = emitSkinnyGEMM
    monkeypatch.setitem(sys.modules, "tensilelite_codegen", module)
    return module


def block(params):
    return {"generator": "SkinnyGEMM", "generatorParams": dict(params)}


def realCodegen():
    try:
        import tensilelite_codegen
    except ImportError:
        return None
    return tensilelite_codegen


needsCodegen = pytest.mark.skipif(
    realCodegen() is None, reason="the tensilelite_codegen module is not importable"
)


# --------------------------------------------------------------------------- #
# Expansion.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "params,tile,grid,sizeGreater,sizeLess,kMultiple",
    [
        (S1, [4, 1, 4096], ["TilesLong", "TilesShort", "Batch"], 4096, 8193, 4096),
        (S2, [16, 1, 1024], ["TilesLong", "TilesShort", "Batch"], 0, 2**20 + 1, 1024),
        (S3, [32, 2, 1024], ["TilesLong", "TilesShort", "Batch"], 0, 2**20 + 1, None),
        (S4, [4, 1, 4096], ["ComputeUnits", "One", "One"], 4096, 8193, 4096),
        (S5, [16, 1, 1024], ["ComputeUnits", "One", "One"], 2048, 3073, None),
    ],
)
@pytest.mark.parametrize("useBeta", [False, True])
def test_expanded_config(fakeCodegen, params, tile, grid, sizeGreater, sizeLess, kMultiple, useBeta):
    params = {**params, "useBeta": useBeta}
    config = getGeneratorKernelConfig(block(params), {}, "gfx950")
    name = kernelNameOf(params)
    assert config["CustomKernelName"] == name
    assert config["KernelLanguage"] == "Assembly"
    assert config["GlobalSplitU"] == 1
    assert config["MatrixInstruction"] == [16, 16, 16, 1]
    assert config["EnableMatrixInstruction"] is True
    assert config["MIWaveTile"] == [1, 1]
    assert config["StaggerU"] == 0
    assert config["WavefrontSize"] == 64

    ck = config["CustomKernel"]
    assert ck["name"] == name
    assert ck["args"] == EXPECTED_ARGS
    assert ck["macrotile"] == tile
    assert ck["threads"] == [64 * params["waves"], 1, 1]
    assert ck["grid"] == grid
    assert (ck["workspaceType"], ck["workspaceSizePerElemC"], ck["workspaceSizePerElemBias"]) == ("None", 0, 0)
    assert ck["generator"] == "SkinnyGEMM"
    assert ck["generatorParams"] == params

    expected = {**COMMON_ASSERTS, **(BETA_ASSERTS if useBeta else NO_BETA_ASSERTS)}
    expected["AssertSizeGreaterThan"] = {3: sizeGreater}
    expected["AssertSizeLessThan"] = {0: 2**20 + 1, 1: 2**30 + 1, 3: sizeLess}
    if kMultiple is not None:
        expected["AssertSummationElementMultiple"] = kMultiple
    asserts = {key: value for key, value in config.items() if key.startswith("Assert")}
    assert asserts == expected


def test_expanded_args_are_copies(fakeCodegen):
    first = getGeneratorKernelConfig(block(S1), {}, "gfx950")
    first["CustomKernel"]["args"][0]["semantic"] = "Changed"
    first["CustomKernel"]["args"].pop()
    assert SkinnyGEMM.args() == EXPECTED_ARGS
    assert getGeneratorKernelConfig(block(S1), {}, "gfx950")["CustomKernel"]["args"] == EXPECTED_ARGS


def test_internal_support_params_fill_only_missing_keys(fakeCodegen):
    isp = {"KernArgsVersion": 3, "SupportUserGSU": True, "Other": 7}
    config = getGeneratorKernelConfig(block(S1), isp, "gfx950")
    assert config["InternalSupportParams"] == {
        "KernArgsVersion": 0,
        "SupportUserGSU": False,
        "PersistentLoopArgsVersion": 0,
        "Other": 7,
    }
    assert isp == {"KernArgsVersion": 3, "SupportUserGSU": True, "Other": 7}


def test_stale_keys_in_the_block_are_replaced(fakeCodegen):
    stale = {**block(S1), "name": "stale", "args": [], "macrotile": [1, 1, 1]}
    ck = getGeneratorKernelConfig(stale, {}, "gfx950")["CustomKernel"]
    assert ck["name"] == kernelNameOf(S1)
    assert ck["args"] == EXPECTED_ARGS
    assert ck["macrotile"] == [4, 1, 4096]


def test_params_are_not_aliased(fakeCodegen):
    source = block(S1)
    config = getGeneratorKernelConfig(source, {}, "gfx950")
    config["CustomKernel"]["generatorParams"]["bm"] = 99
    assert source["generatorParams"]["bm"] == 1


def test_source_comes_from_the_emitter(fakeCodegen):
    assert getGeneratorKernelSource(block(S2), "gfx950") == f"// {kernelNameOf(S2)} for gfx950\n"


def test_is_generator_kernel():
    assert isGeneratorKernel(block(S1))
    assert not isGeneratorKernel({"name": "wvSpltK_bf16_tn_m1"})
    assert not isGeneratorKernel(-1)
    assert not isGeneratorKernel(None)


# --------------------------------------------------------------------------- #
# Validation.
# --------------------------------------------------------------------------- #
def withParam(key, value):
    params = dict(S1)
    params[key] = value
    return block(params)


@pytest.mark.parametrize(
    "customKernel,message",
    [
        ({"generator": "Nope", "generatorParams": dict(S1)}, "unknown kernel generator"),
        ({"generator": "SkinnyGEMM", "generatorParams": [1, 2]}, "must be a dict"),
        ({"generator": "SkinnyGEMM"}, "must be a dict"),
        ({"generator": "SkinnyGEMM", "generatorParams": {k: v for k, v in S1.items() if k != "tn"}}, "missing tn"),
        ({"generator": "SkinnyGEMM", "generatorParams": {**S1, "extra": 1}}, "unknown keys extra"),
        (withParam("bm", True), "bm must be an int"),
        (withParam("lanesK", 16.0), "lanesK must be an int"),
        (withParam("grid3d", 1), "grid3d must be a bool"),
        (withParam("useBeta", 1), "useBeta must be a bool"),
        (withParam("useBeta", "true"), "useBeta must be a bool"),
        (withParam("useBeta", None), "useBeta must be a bool"),
        ({"generator": "SkinnyGEMM", "generatorParams": {k: v for k, v in S1.items() if k != "useBeta"}}, "missing useBeta"),
        (withParam("partialSteps", 2), "partialSteps must be 0 or 1"),
        (withParam("upIters", -1), "upIters must not be negative"),
        (withParam("lanesK", 3), "unsupported lanesK 3"),
    ],
)
def test_invalid_blocks_raise(fakeCodegen, customKernel, message):
    with pytest.raises(ValueError, match=message):
        getGeneratorKernelConfig(customKernel, {}, "gfx950")


def test_validate_generator_kernel_needs_no_codegen_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensilelite_codegen", None)
    assert validateGeneratorKernel(block(S1)) == S1
    with pytest.raises(ValueError, match="upIters must not be negative"):
        validateGeneratorKernel(withParam("upIters", -1))
    with pytest.raises(ValueError, match="unknown kernel generator"):
        validateGeneratorKernel({"generator": "Nope"})


def test_binding_errors_name_the_parameters(fakeCodegen):
    with pytest.raises(ValueError, match="invalid SkinnyGEMM generatorParams.*'lanesK': 3"):
        getGeneratorKernelConfig(withParam("lanesK", 3), {}, "gfx950")
    with pytest.raises(ValueError, match="unsupported lanesK 3"):
        getGeneratorKernelSource(withParam("lanesK", 3), "gfx950")


def test_missing_module_is_a_runtime_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "tensilelite_codegen", None)
    with pytest.raises(RuntimeError, match="tensilelite_codegen"):
        getGeneratorKernelConfig(block(S1), {}, "gfx950")
    with pytest.raises(RuntimeError, match="TENSILELITE_ENABLE_JIT_PYTHON"):
        getGeneratorKernelSource(block(S1), "gfx950")


# --------------------------------------------------------------------------- #
# Code object names and predicates.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "libraryCodeObject,expected",
    [
        ("TensileLibrary_Type_ID75a0_gfx950", "TensileLibrary_Type_ID75a0_Generated_gfx950"),
        ("TensileLibrary_gfx950", "TensileLibrary_Generated_gfx950"),
        ("TensileLibrary_lazy", "TensileLibrary_lazy_Generated_gfx950"),
        ("TensileLibrary_gfx950_extra", "TensileLibrary_gfx950_extra_Generated_gfx950"),
    ],
)
def test_generated_code_object_name(libraryCodeObject, expected):
    assert generatedCodeObjectName(libraryCodeObject, "gfx950") == expected


@pytest.mark.parametrize(
    "key,tag,value",
    [
        ("AssertNoBias", "UseBias", 0),
        ("AssertNoActivation", "Activation", "None"),
        ("AssertNoScaleAB", "UseScaleAB", ""),
        ("AssertNoScaleAlphaVec", "UseScaleAlphaVec", 0),
    ],
)
def test_no_feature_asserts_map_to_predicates(key, tag, value):
    predicate = Contractions.ProblemPredicate.FromOriginalKeyPair((key, True))
    state = predicate.state()
    assert (state["type"], state["value"]) == (tag, value)
    assert Contractions.ProblemPredicate.FromOriginalKeyPair((key, False)) is None


def test_use_beta_reaches_the_binding(fakeCodegen):
    assert SkinnyGEMM._config(betaVariant(S1)).useBeta is True
    assert SkinnyGEMM._config(S1).useBeta is False
    assert SkinnyGEMM.kernelName(betaVariant(S1)).endswith("_USE_BETA1")
    assert SkinnyGEMM.kernelName(S1).endswith("_USE_BETA0")


def test_beta_zero_assert_maps_to_a_predicate():
    assert Contractions.ProblemPredicate.FromOriginalKeyPair(("AssertBetaZero", True)).state() == {"type": "BetaZero"}


def test_c_stride_asserts_map_to_predicates():
    assert Contractions.ProblemPredicate.FromOriginalKeyPair(("AssertStrideCEqual", {0: 1})).state() == {
        "type": "StrideCEqual", "index": 0, "value": 1}
    assert Contractions.ProblemPredicate.FromOriginalKeyPair(("AssertStrideCMultiple", {1: 4})).state() == {
        "type": "StrideCMultiple", "index": 1, "value": 4}


def test_runtime_custom_kernel_has_no_generator_keys(fakeCodegen):
    ck = getGeneratorKernelConfig(block(S1), {}, "gfx950")["CustomKernel"]
    state = Contractions.CustomKernel.FromOriginalState(ck).state()
    assert "generator" not in state and "generatorParams" not in state
    assert "codeObjectFile" not in state
    ck["codeObjectFile"] = "TensileLibrary_Generated_gfx950.co"
    state = Contractions.CustomKernel.FromOriginalState(ck).state()
    assert state["codeObjectFile"] == "TensileLibrary_Generated_gfx950.co"


# --------------------------------------------------------------------------- #
# Logic parsing.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def parseEnv():
    cxxCompiler, _cCompiler, offloadBundler = validateToolchain(
        ToolchainDefaults.CXX_COMPILER,
        ToolchainDefaults.C_COMPILER,
        ToolchainDefaults.OFFLOAD_BUNDLER,
    )
    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], cxxCompiler)
    assignGlobalParameters({}, isaInfoMap)
    assembler = makeAssemblyToolchain(cxxCompiler, offloadBundler, "default").assembler
    return SimpleNamespace(assembler=assembler, isaInfoMap=isaInfoMap)


@pytest.fixture(autouse=True)
def sequentialCpuThreads(monkeypatch):
    monkeypatch.setitem(globalParameters, "CpuThreads", 1)


def parseLogic(env, raw):
    return LibraryIO.parseLibraryLogicData(
        raw, str(LOGIC_FILE), env.assembler, False, False, False, env.isaInfoMap, False
    )


def readLogic():
    return LibraryIO.read(str(LOGIC_FILE), customizedLoader=True)


def test_logic_round_trip(fakeCodegen, parseEnv):
    fields = parseLogic(parseEnv, readLogic())
    assert [s["CustomKernelName"] for s in fields.solutions] == [
        kernelNameOf(p) for p in FIXTURE_SOLUTIONS
    ]
    for solution, params in zip(fields.solutions, FIXTURE_SOLUTIONS):
        ck = solution["CustomKernel"]
        assert ck["name"] == kernelNameOf(params)
        assert ck["generatorParams"] == params
        assert ck["args"] == EXPECTED_ARGS
        assert solution["AssertNoBias"] is True
        assert solution.get("AssertBetaZero", False) is (not params["useBeta"])
        assert ("AssertStrideCEqual" in solution) is params["useBeta"]


def test_logic_predicates_reach_the_library_solution(fakeCodegen, parseEnv):
    fields = parseLogic(parseEnv, readLogic())
    solution = Contractions.Solution.FromSolutionStruct(
        fields.solutions[0], False, False, False, parseEnv.assembler, parseEnv.isaInfoMap
    )
    predicates = solution.problemPredicate.state()["value"]
    byType = {}
    for predicate in predicates:
        byType.setdefault(predicate["type"], []).append(predicate)
    assert {"type": "UseBias", "value": 0} in byType["UseBias"]
    assert {"type": "Activation", "value": "None"} in byType["Activation"]
    assert {"type": "UseScaleAB", "value": ""} in byType["UseScaleAB"]
    assert {"type": "UseScaleAlphaVec", "value": 0} in byType["UseScaleAlphaVec"]
    assert {"type": "SizeEqual", "index": 2, "value": 1} in byType["SizeEqual"]
    assert {"type": "StrideAMultiple", "index": 1, "value": 4} in byType["StrideAMultiple"]
    assert {"type": "SizeGreaterThan", "index": 3, "value": 4096} in byType["SizeGreaterThan"]
    custom = solution.customKernel.state()
    assert "generator" not in custom and "codeObjectFile" not in custom


def libraryPredicates(env, index):
    fields = parseLogic(env, readLogic())
    solution = Contractions.Solution.FromSolutionStruct(
        fields.solutions[index], False, False, False, env.assembler, env.isaInfoMap
    )
    return solution.problemPredicate.state()["value"]


def test_beta_predicates_per_variant(fakeCodegen, parseEnv):
    withoutBeta = libraryPredicates(parseEnv, FIXTURE_SOLUTIONS.index(S1))
    withBeta = libraryPredicates(parseEnv, FIXTURE_SOLUTIONS.index(betaVariant(S1)))
    assert {"type": "BetaZero"} in withoutBeta
    assert not any(p["type"] == "StrideCEqual" for p in withoutBeta)
    assert {"type": "BetaZero"} not in withBeta
    assert {"type": "StrideCEqual", "index": 0, "value": 1} in withBeta
    # ldC and ldD need no alignment, ldA and ldB do.
    for predicates in (withoutBeta, withBeta):
        assert not any(p["type"] in ("StrideCMultiple", "StrideDMultiple") for p in predicates)
        assert {"type": "StrideAMultiple", "index": 1, "value": 4} in predicates
        assert {"type": "StrideBMultiple", "index": 1, "value": 4} in predicates


def test_handwritten_custom_kernel_keeps_its_path(fakeCodegen, parseEnv, monkeypatch):
    calls = []
    real = LibraryIO.getCustomKernelConfig

    def spy(name, isp, *args):
        calls.append(name)
        return real(name, isp, *args)

    monkeypatch.setattr(LibraryIO, "getCustomKernelConfig", spy)
    raw = readLogic()
    raw["Solutions"] = [{"SolutionIndex": 0, "CustomKernel": {"name": "wvSpltK_bf16_tn_m1"}}]
    raw["ExactLogic"] = raw["ExactLogic"][:1]
    fields = parseLogic(parseEnv, raw)
    assert calls == ["wvSpltK_bf16_tn_m1"]
    assert fields.solutions[0]["CustomKernelName"] == "wvSpltK_bf16_tn_m1"
    assert "generator" not in fields.solutions[0]["CustomKernel"]


@pytest.mark.parametrize("mutate", [
    lambda params: params.pop("waves"),
    lambda params: params.update(partialSteps=2),
    lambda params: params.update(grid3d=1),
])
def test_invalid_generator_block_fails_the_parse(fakeCodegen, parseEnv, mutate):
    raw = readLogic()
    mutate(raw["Solutions"][0]["CustomKernel"]["generatorParams"])
    with pytest.raises(ValueError) as error:
        parseLogic(parseEnv, raw)
    assert not isinstance(error.value, GeneratorKernelRejected)


def matchingTables(library):
    """The rows of every matching table below a library, as (key, solution index)."""
    rows = []
    for row in getattr(library, "rows", []):
        rows += matchingTables(row["library"])
    for child in (getattr(library, "mapping", None) or {}).values():
        rows += matchingTables(child)
    for entry in getattr(library, "table", []):
        if isinstance(entry, dict) and "key" in entry:
            rows.append((entry["key"], entry["index"].solution.index))
    return rows


def assertSecondSolutionDropped(env, raw, capsys, rejected):
    """The solution at index 1 is dropped and every other solution keeps its rows."""
    original = [(key, value[0]) for key, value in readLogic()["ExactLogic"]]
    fields = parseLogic(env, raw)
    output = capsys.readouterr().out
    assert "Skipping generator kernel SkinnyGEMM" in output
    assert rejected in output and "solution index 1" in output
    keptIndices = [index for index in range(len(FIXTURE_SOLUTIONS)) if index != 1]
    assert [s["CustomKernelName"] for s in fields.solutions] == [
        kernelNameOf(FIXTURE_SOLUTIONS[index]) for index in keptIndices
    ]
    assert sorted(fields.library.solutions) == keptIndices
    rows = matchingTables(fields.library.library)
    assert sorted(rows) == sorted((key, index) for key, index in original if index != 1)


def test_generator_kernel_rejected_by_the_emitter_is_dropped(fakeCodegen, parseEnv, capsys):
    raw = readLogic()
    raw["Solutions"][1]["CustomKernel"]["generatorParams"]["lanesK"] = 3
    assertSecondSolutionDropped(parseEnv, raw, capsys, "'lanesK': 3")


@needsCodegen
def test_real_over_budget_generator_kernel_is_dropped(parseEnv, capsys):
    overBudget = dict(bm=2, tn=8, lanesK=8, waves=2, prefetch=1, upIters=10, partialSteps=0, group=1,
                      numXcds=1, grid3d=True, useBeta=True)
    raw = readLogic()
    raw["Solutions"][1]["CustomKernel"]["generatorParams"] = dict(overBudget)
    assertSecondSolutionDropped(parseEnv, raw, capsys, "register budget")


def test_rejection_is_a_value_error():
    assert issubclass(GeneratorKernelRejected, ValueError)


def test_emitter_rejection_raises_generator_kernel_rejected(fakeCodegen):
    rejected = withParam("lanesK", 3)
    for call in (
        lambda: SkinnyGEMM.emit(rejected["generatorParams"], "gfx950"),
        lambda: SkinnyGEMM.kernelName(rejected["generatorParams"]),
        lambda: getGeneratorKernelConfig(rejected, {}, "gfx950"),
    ):
        with pytest.raises(GeneratorKernelRejected, match="invalid SkinnyGEMM generatorParams.*unsupported lanesK 3"):
            call()


def test_shape_errors_are_not_rejections(fakeCodegen):
    with pytest.raises(ValueError) as error:
        getGeneratorKernelConfig(withParam("partialSteps", 2), {}, "gfx950")
    assert not isinstance(error.value, GeneratorKernelRejected)


# --------------------------------------------------------------------------- #
# Kernel source.
# --------------------------------------------------------------------------- #
def test_kernel_source_uses_the_generator_and_not_the_custom_kernel_directory(fakeCodegen, monkeypatch):
    from Tensile import KernelWriterAssembly as KwaModule

    kwa = object.__new__(KwaModule.KernelWriterAssembly)
    kwa.states = SimpleNamespace()
    kwa._getCustomKernelSource = lambda *args, **kwargs: pytest.fail("read CustomKernels")
    kernel = {
        "KernelLanguage": "Assembly",
        "ISA": [9, 5, 0],
        "CustomKernelName": kernelNameOf(S1),
        "CustomKernel": getGeneratorKernelConfig(block(S1), {}, "gfx950")["CustomKernel"],
        "CUOccupancy": -1,
    }
    errcode, code = kwa.getSourceFileString(kernel)
    assert errcode == 0
    assert code == f"// {kernelNameOf(S1)} for gfx950\n"
    assert kwa.states.kernel is kernel


def test_kernel_source_failure_is_a_generation_failure(fakeCodegen):
    from Tensile import KernelWriterAssembly as KwaModule

    kwa = object.__new__(KwaModule.KernelWriterAssembly)
    kwa.states = SimpleNamespace()
    kernel = {
        "KernelLanguage": "Assembly",
        "ISA": [9, 5, 0],
        "CustomKernelName": "broken",
        "CustomKernel": withParam("lanesK", 3),
    }
    kernel["CustomKernel"]["name"] = "broken"
    with pytest.raises(RuntimeError, match="generator kernel broken.*lanesK"):
        kwa._getGeneratedKernelSource(kernel)
    errcode, code = kwa.getSourceFileString(kernel)
    assert errcode == -2
    assert code == ""


# --------------------------------------------------------------------------- #
# Code object routing.
# --------------------------------------------------------------------------- #
def recordingLinker(links):
    def link(objects, destination):
        links[destination] = list(objects)
        Path(destination).write_bytes(b"")

    return link


def generatorAlias(fileName):
    ck = {"name": "gemm", "generator": "SkinnyGEMM", "generatorParams": dict(S1)}
    if fileName:
        ck["codeObjectFile"] = fileName
    return {"ISA": (9, 5, 0), "BaseName": "gen", "CustomKernel": ck, "codeObjectFile": "TensileLibrary_ID_gfx950"}


def test_generator_kernels_link_into_their_own_code_object(tmp_path):
    links = {}
    kernels = [
        {"ISA": (9, 5, 0), "BaseName": "plain", "codeObjectFile": "TensileLibrary_ID_gfx950"},
        generatorAlias("TensileLibrary_ID_Generated_gfx950.co"),
    ]
    coFiles = buildAssemblyCodeObjectFiles(
        recordingLinker(links), None, kernels, tmp_path, tmp_path, compress=False
    )
    assert {Path(path).name for path in links} == {
        "TensileLibrary_ID_gfx950.co.raw",
        "TensileLibrary_ID_Generated_gfx950.co.raw",
    }
    assert {path.name for path in coFiles} == {
        "TensileLibrary_ID_gfx950.co",
        "TensileLibrary_ID_Generated_gfx950.co",
    }
    for path, objects in links.items():
        expected = "gen.o" if "Generated" in path else "plain.o"
        assert [Path(o).name for o in objects] == [expected]


def test_generator_kernel_without_a_code_object_file_fails(tmp_path):
    with pytest.raises(RuntimeError, match="no code object file"):
        buildAssemblyCodeObjectFiles(
            recordingLinker({}), None, [generatorAlias(None)], tmp_path, tmp_path, compress=False
        )


def customKernelState(fileName=None):
    ck = {
        "name": "gemm",
        "args": [],
        "macrotile": [1, 1, 1],
        "threads": [64, 1, 1],
        "grid": ["One", "One", "One"],
    }
    if fileName:
        ck["codeObjectFile"] = fileName
    return ck


def fakeSolution(state):
    return SimpleNamespace(
        originalSolution=SimpleNamespace(_state=state),
        customKernel=Contractions.CustomKernel.FromOriginalState(state["CustomKernel"]),
    )


def test_codegen_definition_keeps_the_solutions_own_code_object_file():
    shared = customKernelState()
    sol = fakeSolution({"CustomKernel": customKernelState("own_Generated_gfx950.co")})
    Run._applyCustomKernelDefToSol(sol, SimpleNamespace(customKernelDef=shared))
    assert sol.originalSolution._state["CustomKernel"]["codeObjectFile"] == "own_Generated_gfx950.co"
    assert sol.customKernel.codeObjectFile == "own_Generated_gfx950.co"
    assert "codeObjectFile" not in shared


def test_codegen_definition_is_unchanged_without_an_own_file():
    shared = customKernelState()
    sol = fakeSolution({"CustomKernel": customKernelState()})
    Run._applyCustomKernelDefToSol(sol, SimpleNamespace(customKernelDef=shared))
    assert sol.originalSolution._state["CustomKernel"] is shared
    assert sol.customKernel.codeObjectFile == ""


def test_assign_generated_code_object_files(fakeCodegen):
    generated = getGeneratorKernelConfig(block(S1), {}, "gfx950")["CustomKernel"]
    shared = dict(generated)
    plain = customKernelState()
    lazyName = "TensileLibrary_Type_ID75a0_gfx950"
    nonLazy = fakeSolution({"CustomKernel": shared})
    lazy = fakeSolution({"CustomKernel": shared})
    other = fakeSolution({"CustomKernel": plain})
    master = SimpleNamespace(
        solutions={0: nonLazy, 1: other},
        lazyLibraries={lazyName: SimpleNamespace(solutions={2: lazy})},
    )
    Run._assignGeneratedCodeObjectFiles({"gfx950": master})

    nonLazyFile = "TensileLibrary_Generated_gfx950.co"
    lazyFile = "TensileLibrary_Type_ID75a0_Generated_gfx950.co"
    assert nonLazy.originalSolution._state["CustomKernel"]["codeObjectFile"] == nonLazyFile
    assert nonLazy.customKernel.codeObjectFile == nonLazyFile
    assert lazy.originalSolution._state["CustomKernel"]["codeObjectFile"] == lazyFile
    assert lazy.customKernel.codeObjectFile == lazyFile
    assert "codeObjectFile" not in shared
    assert other.originalSolution._state["CustomKernel"] is plain
    assert other.customKernel.codeObjectFile == ""


# --------------------------------------------------------------------------- #
# The real emitter.
# --------------------------------------------------------------------------- #
def emittedMetadata(source):
    match = re.search(r"\.amdgpu_metadata\n(.*?)\n\.end_amdgpu_metadata", source, re.S)
    assert match, "no metadata in the emitted source"
    return yaml.safe_load(match.group(1).split("\n...")[0])


def emittedKernelMetadata(source):
    return emittedMetadata(source)["amdhsa.kernels"][0]


def emittedExplicitArgs(source):
    return [arg for arg in emittedKernelMetadata(source)[".args"] if not arg[".value_kind"].startswith("hidden_")]


@needsCodegen
def test_real_kernel_name():
    assert SkinnyGEMM.kernelName(S1) == (
        "gemm_BM1_TN1_LANES_K64_WAVES4_UP_ITERS2_PARTIAL_STEPS0_PF2_GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA0"
    )
    for params in FIXTURE_SOLUTIONS:
        assert SkinnyGEMM.kernelName(params) == kernelNameOf(params)


@needsCodegen
def test_real_binding_rejects_invalid_knobs():
    with pytest.raises(ValueError, match="invalid SkinnyGEMM generatorParams"):
        SkinnyGEMM.kernelName({**S1, "lanesK": 3})


@needsCodegen
def test_real_source_is_code_object_v6():
    source = SkinnyGEMM.emit(S1, "gfx950")
    assert ".amdhsa_code_object_version 6" in source
    assert kernelNameOf(S1) in source


ARG_NAMES = [
    "D", "A", "B", "scalesA", "scalesB", "ws", "C", "ldD", "ldA", "ldB", "m", "n", "k", "spk",
    "ldC", "alpha", "beta", "numWorkGroupsX", "numWorkGroupsY", "numWorkGroupsZ",
]
ARG_SIZES = {"address": 8, "uint64": 8, "int32": 4, "float32": 4}


def checkEmittedArguments(source):
    """The explicit kernargs are the declared ones, packed in order, and they are the whole segment."""
    explicit = emittedExplicitArgs(source)
    declared = SkinnyGEMM.args()
    assert len(explicit) == len(declared) == len(ARG_NAMES) == 20
    assert [arg[".name"] for arg in explicit] == ARG_NAMES
    offset = 0
    for emitted, arg in zip(explicit, declared):
        assert emitted[".size"] == ARG_SIZES[arg["type"]]
        assert emitted[".offset"] == offset
        assert (emitted[".value_kind"] == "global_buffer") == (arg["type"] in ("address", "uint64"))
        offset += emitted[".size"]
    assert "hidden_" not in source
    # The segment is the arguments rounded up to 8 bytes.
    assert offset == 108
    assert emittedKernelMetadata(source)[".kernarg_segment_size"] == 112


@needsCodegen
@pytest.mark.parametrize("params", [S1, S4, betaVariant(S1), betaVariant(S4)])
def test_real_emitted_arguments_match_the_declared_ones(params):
    checkEmittedArguments(SkinnyGEMM.emit(params, "gfx950"))


# The code object version of the directive and the amdhsa.version of the metadata.
CODE_OBJECT_VERSION_METADATA = {4: [1, 1], 5: [1, 2], 6: [1, 2]}


@needsCodegen
@pytest.mark.parametrize("codeObjectVersion", [4, 5, 6])
@pytest.mark.parametrize("params", [S1, S4, betaVariant(S1), betaVariant(S4)])
def test_real_emitted_arguments_at_every_code_object_version(params, codeObjectVersion):
    source = realCodegen().emitSkinnyGEMM(
        realCodegen().SkinnyGEMMConfig(**params), "gfx950", codeObjectVersion
    )
    checkEmittedArguments(source)
    assert f".amdhsa_code_object_version {codeObjectVersion}" in source
    assert emittedMetadata(source)["amdhsa.version"] == CODE_OBJECT_VERSION_METADATA[codeObjectVersion]
    assert kernelNameOf(params) in source
