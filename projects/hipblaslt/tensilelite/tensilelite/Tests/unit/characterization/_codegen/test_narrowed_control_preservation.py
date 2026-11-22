# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Preservation tests for narrowed solution controls that are already correct.

Each control narrows one call fact of the same stock gfx950 solution (bf16 NN,
MT16x16, DepthU 64; ``data/test_data/_logic/gfx950/BBS_MT16x16_narrowed_controls.yaml``):

* positive: the narrowed kernel omits the code region the fact makes dead (the
  region is present in the reference kernel) and the kernel assembles;
* negative: the narrowed solution's library problem predicate carries the guard
  that refuses an incompatible call (the C++ evaluation of those predicates is
  pinned in ``tensilelite/tests/Predicates_test.cpp``); for DataParallel the
  derivation drops a requested GSU (the host's runtime-GSU refusal is pinned in
  ``tensilelite/tests/PersistentArgumentLayout_test.cpp``).
"""

import copy
import os
import re

import pytest
import yaml

import codegen_harness as H

pytestmark = pytest.mark.unit

_LOGIC = os.path.join(
    os.path.dirname(__file__), "data", "test_data", "_logic", "gfx950", "BBS_MT16x16_narrowed_controls.yaml"
)

# name: (ProblemType changes, Solution changes) of one logic variant
_VARIANTS = {
    "reference": ({}, {}),
    "use_beta_off": ({"UseBeta": False}, {}),
    "activation_none": ({"Activation": False, "ActivationType": "none"}, {}),
    "bias_off": ({"UseBias": 0, "BiasDataTypeList": []}, {}),
    "sav_off": ({"UseScaleAlphaVec": 0}, {}),
    # bias kept, BiasDataTypeList narrowed from [Float, BFloat16] to the call's one type
    "bias_type_bf16": ({"BiasDataTypeList": [7]}, {}),
    "asem_depthu": ({}, {"AssertSummationElementMultiple": 64}),
    "stream_k": ({}, {"TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid"}),
    # GlobalSplitU 2 and SupportUserGSU requested: DataParallel must still refuse a split.
    "data_parallel": ({}, {"TileProcessingStrategy": "DataParallel", "WorkAssignment": "StaticGrid",
                           "GlobalSplitU": 2,
                           "InternalSupportParams": {"SupportCustomStaggerU": True, "SupportCustomWGM": True,
                                                     "SupportUserGSU": True, "UseUniversalArgs": True}}),
}

# control: (narrowed variant, reference variant, region the narrowed kernel must omit)
_REGIONS = {
    "UseBeta": ("use_beta_off", "reference", r"\blabel_GW_B1_|\bsgprBeta\b"),
    "ActivationNone": ("activation_none", "reference", r"\bsgprActivationType\b|\blabel_Activation_\w+"),
    "UseBias0": ("bias_off", "reference", r"\bsgpr(AddressBias|SrdBias|BiasStride|BiasType)\b"),
    "UseScaleAlphaVec0": ("sav_off", "reference",
                          r"\bsgpr(AddressScaleAlphaVec|SrdScaleAlphaVec)\b|\bScaleAlphaVecVMul\b"),
    "BiasTypeSingle": ("bias_type_bf16", "reference", r"\blabel_Load_Biasf32_\w+"),
    "ASEMTailLoop": ("asem_depthu", "reference", r"\blabel_TailLoopBeginL\b"),
    "DataParallelStaticGrid": ("data_parallel", "stream_k",
                               r"\bsgpr(AddressWS|SrdWS|AddressFlags|StreamKLocalStart|StreamKLocalEnd)\b"),
}

# control: (narrowed variant, guard leaf of its serialized problem predicate)
_GUARDS = {
    "UseBeta": ("use_beta_off", {"type": "BetaZero"}),
    "ActivationNone": ("activation_none", {"type": "Activation", "value": "None"}),
    "UseBias0": ("bias_off", {"type": "UseBias", "value": 0}),
    "UseScaleAlphaVec0": ("sav_off", {"type": "UseScaleAlphaVec", "value": 0}),
    "BiasTypeSingle": ("bias_type_bf16", {"type": "BiasDataTypeWhiteList", "value": ["BFloat16"]}),
    "ASEMTailLoop": ("asem_depthu", {"type": "BoundSizeMultiple", "index": -1, "value": 64}),
}


@pytest.fixture(scope="module")
def logic_paths(tmp_path_factory):
    with open(_LOGIC) as f:
        base = yaml.safe_load(f)
    root = tmp_path_factory.mktemp("narrowed_controls")
    paths = {}
    for name, (problem_type, solution) in _VARIANTS.items():
        logic = copy.deepcopy(base)
        logic["ProblemType"].update(problem_type)
        logic["Solutions"][0].update(solution)
        paths[name] = root / f"{name}.yaml"
        with open(paths[name], "w") as f:
            yaml.safe_dump(logic, f, sort_keys=False)
    return paths


@pytest.fixture(scope="module")
def kernels(logic_paths):
    """variant -> (kernel name, assembly source, emitter return code)."""
    cache = {}

    def get(name):
        if name not in cache:
            results = H.emit_kernels_from_logic(logic_paths[name], canonical=False)
            assert len(results) == 1
            cache[name] = results[0]
        return cache[name]

    return get


def _library_solution(path):
    from tensilelite import Contractions

    with H._isolated_globals():
        (sol,) = H._solutions_from_logic_unguarded(path)
        return Contractions.Solution.FromSolutionStruct(
            sol, False, False, False, H.get_assembler(), H.get_isa_info_map())


def _predicate_leaves(predicate):
    from tensilelite.Properties import state

    leaves = []

    def walk(node):
        if node["type"] == "And":
            for child in node["value"]:
                walk(child)
        else:
            leaves.append(node)

    walk(state(predicate))
    return leaves


@pytest.mark.parametrize("control", sorted(_REGIONS))
def test_reference_kernel_has_region(control, kernels):
    _narrowed, reference, region = _REGIONS[control]
    _name, source, err = kernels(reference)
    assert err == 0
    assert re.search(region, source), f"{control}: the reference kernel lacks /{region}/"


@pytest.mark.parametrize("control", sorted(_REGIONS))
def test_narrowed_kernel_omits_region(control, kernels):
    narrowed, _reference, region = _REGIONS[control]
    _name, source, err = kernels(narrowed)
    assert err == 0
    found = sorted(set(m.group(0) for m in re.finditer(region, source)))
    assert not found, f"{control}: the narrowed kernel still emits {found[:5]}"


@pytest.fixture(scope="module")
def assembler():
    """The assembler TensileCreateLibrary builds (its --code-object-version default, 4)."""
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import ToolchainDefaults, validateToolchain

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "4").assembler


@pytest.mark.parametrize("control", sorted(_REGIONS))
def test_narrowed_kernel_assembles(control, kernels, assembler, tmp_path):
    narrowed, _reference, _region = _REGIONS[control]
    name, source, err = kernels(narrowed)
    assert err == 0
    src, obj = tmp_path / f"{name}.s", tmp_path / f"{name}.o"
    src.write_text(source)
    assembler("gfx950", 64, str(src), str(obj))
    assert obj.stat().st_size > 0


@pytest.mark.parametrize("control", sorted(_GUARDS))
def test_narrowed_problem_predicate_carries_guard(control, logic_paths):
    narrowed, guard = _GUARDS[control]
    reference = _predicate_leaves(_library_solution(logic_paths["reference"]).problemPredicate)
    leaves = _predicate_leaves(_library_solution(logic_paths[narrowed]).problemPredicate)
    assert guard not in reference
    assert guard in leaves, f"{control}: {guard} missing from {leaves}"


def test_bias_type_narrowing_drops_other_type_helpers(logic_paths):
    """The conversion helpers are generated per BiasDataTypeList entry: narrowing the list to
    bf16 removes the f32-bias (``_BiasS_``) helper variants and keeps the bf16 ones."""
    reference_helpers = H.emit_helpers_from_logic(logic_paths["reference"])
    narrowed_helpers = H.emit_helpers_from_logic(logic_paths["bias_type_bf16"])
    assert [h for h in reference_helpers + narrowed_helpers if h[1] != 0] == []
    reference = [name for name, _err in reference_helpers]
    narrowed = [name for name, _err in narrowed_helpers]
    assert any("_BiasS_" in n for n in reference) and any("_BiasB_" in n for n in reference)
    assert any("_BiasB_" in n for n in narrowed)
    assert not [n for n in narrowed if "_BiasS_" in n]


def test_data_parallel_derivation_drops_requested_gsu(logic_paths):
    from tensilelite.Properties import state

    sol = _library_solution(logic_paths["data_parallel"])
    mapping = state(sol.sizeMapping)
    assert mapping["tileProcessingStrategy"] == "DataParallel"
    assert mapping["workAssignment"] == "StaticGrid"
    assert mapping["globalSplitU"] == 0
    assert state(sol.internalArgsSupport)["gsu"] is False
