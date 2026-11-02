# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-solution ProblemType of the handwritten custom kernels in the shipped logic files.

A custom.config whose Features turn bias, activation or the scale-alpha vector off gives
its solution a restricted ProblemType instead of the logic file's shared one. Only the
skinny MXFP8 kernels declare that; every other shipped kernel must keep the shared
ProblemType, so a new difference cannot slip in unnoticed.
"""

import re
from pathlib import Path

import pytest
import yaml

import tensilelite.LibraryIO as L
from tensilelite.Contractions import ProblemType as ContractionsProblemType
from tensilelite.CustomKernels import getCustomKernelConfig, readCustomKernelConfig
from tensilelite.SolutionStructs.Problem import ProblemType

pytestmark = pytest.mark.unit

LOGIC_ROOT = (
    Path(__file__).resolve().parents[4] / "library/src/amd_detail/rocblaslt/src/Tensile/Logic"
)
SKINNY_SHARD = "SkinnyGEMM"
SKINNY_LOGIC = (
    "asm_full/gfx950/gfx950/Equality/"
    "gfx950_Cijk_Alik_Bljk_F8BS_MXAE8B32_MXBE8B32_BH_Bias_HA_S_SAB_SAV_UserArgs_SkinnyGEMM.yaml"
)

_LEGACY_NAME = re.compile(rb"CustomKernelName: [^'\" \n]")

def _logic_files_with_handwritten_kernels():
    """Cheap text filter: a legacy kernel name, or a CustomKernel block not marked generated."""
    for path in sorted(LOGIC_ROOT.rglob("*.yaml")):
        if SKINNY_SHARD in path.name:
            continue
        text = path.read_bytes()
        if _LEGACY_NAME.search(text) or text.count(b"CustomKernel:") > text.count(b"generated: true"):
            yield path


def _shared_problem_type(data, path):
    L.normalizeLogicProblemType(data["ProblemType"])
    return ProblemType(data["ProblemType"], False, srcFile=str(path), raiseOnTypeMismatch=False)


def _handwritten_kernel_name(solutionState):
    descriptor = solutionState.get("CustomKernel")
    if isinstance(descriptor, dict) and descriptor.get("name") and not descriptor.get("generated", False):
        return descriptor["name"]
    return solutionState.get("CustomKernelName", "")


def _predicates(problemType):
    predicates = ContractionsProblemType.FromOriginalState(problemType).predicates(True, True, True)
    return [(p.tag, getattr(p, "index", None), getattr(p, "value", None)) for p in predicates]


def _state_and_predicates(problemType):
    return dict(problemType.state), _predicates(problemType)


def _load_logic(path):
    data = yaml.load(path.read_bytes(), Loader=yaml.CSafeLoader)
    data = L.parseLibraryLogicList(data, str(path)) if isinstance(data, list) else data
    if isinstance(data, dict):
        L.prepareLibraryLogicDict(data)
    return data


def test_shipped_handwritten_kernels_keep_the_shared_problem_type():
    kernels = 0
    skipped = []
    for path in _logic_files_with_handwritten_kernels():
        data = _load_logic(path)
        shared = _shared_problem_type(data, path)
        defaults = data.get("DefaultSolution") or {}
        for solution in data["Solutions"]:
            state = L.fillSolutionDefaults(dict(solution), dict(defaults))
            name = _handwritten_kernel_name(state)
            if not name:
                continue
            try:
                getCustomKernelConfig(name, state.get("InternalSupportParams", {}))
            except (RuntimeError, KeyError, TypeError):
                skipped.append(name)
                continue
            kernels += 1
            features = readCustomKernelConfig(name).get("Features")
            result = L._problemTypeForCustomKernel(shared, features, False, str(path))
            key = (str(path.relative_to(LOGIC_ROOT)), name)
            assert _state_and_predicates(result) == _state_and_predicates(shared), key

    # The loader skips a kernel without a usable custom.config, which would hide it here.
    assert skipped == []
    # 189 solutions were scanned when written; the floor guards against a filter that matches nothing.
    assert kernels > 100


def test_shipped_skinny_kernels_are_restricted():
    path = LOGIC_ROOT / SKINNY_LOGIC
    data = _load_logic(path)
    shared = _shared_problem_type(data, path)
    sharedPredicates = _predicates(shared)
    defaults = data.get("DefaultSolution") or {}
    kernels = 0
    for solution in data["Solutions"]:
        state = L.fillSolutionDefaults(dict(solution), dict(defaults))
        name = _handwritten_kernel_name(state)
        if not name:
            continue
        features = readCustomKernelConfig(name).get("Features")
        result = L._problemTypeForCustomKernel(shared, features, False, str(path))
        assert result is not shared, name
        assert result["UseBias"] == 0, name
        assert result["UseScaleAlphaVec"] == 0, name
        assert result["ActivationType"] == "none", name
        predicates = _predicates(result)
        assert ("UseBias", None, 0) in predicates, name
        assert ("UseScaleAlphaVec", None, 0) in predicates, name
        assert ("UseBias", None, 1) not in predicates, name
        assert ("UseScaleAlphaVec", None, 1) not in predicates, name
        # Activation is emitted for the restricted solution only, and no longer as hipblaslt_all.
        activation = [p for p in predicates if p[0] == "Activation"]
        sharedActivation = [p for p in sharedPredicates if p[0] == "Activation"]
        assert activation and activation != sharedActivation, name
        assert not any(p[0] == "ActivationEnumWhiteList" for p in predicates), name
        kernels += 1
    assert kernels == 6
