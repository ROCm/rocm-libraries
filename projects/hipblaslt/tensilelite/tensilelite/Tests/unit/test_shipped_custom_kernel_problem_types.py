# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Per-solution ProblemType of the handwritten custom kernels in the shipped logic files.

A custom.config whose Features set SupportsBias, SupportsActivation or SupportsScaleAlpha
false gives its solution a restricted ProblemType only when that changes the logic file's
shared one. Many shipped kernels (vllm, rocblas, aiter, ...) declare those flags false too,
but their logic files' ProblemType already has them off, so nothing changes for them; only
the skinny MXFP8 solutions get a different ProblemType (checked in test_skinny_mxfp8_logic.py
and test_skinny_mxfp8_kernel_metadata.py).
"""

import re
from pathlib import Path

import pytest
import yaml

import tensilelite
import tensilelite.LibraryIO as L
from tensilelite.Contractions import ProblemType as ContractionsProblemType
from tensilelite.CustomKernels import getCustomKernelConfig, readCustomKernelConfig
from tensilelite.SolutionStructs.Problem import ProblemType

from logic_corpus import needs_logic_corpus, require_logic_root

pytestmark = pytest.mark.unit

SKINNY_KERNELS = {
    path.stem
    for path in (Path(tensilelite.__file__).parent / "CustomKernels" / "skinny_mxfp8").glob("*.s")
}

_LEGACY_NAME = re.compile(rb"CustomKernelName: [^'\" \n]")

def _logic_files_with_handwritten_kernels(root):
    """Cheap text filter: a legacy kernel name, or a CustomKernel block not marked generated."""
    for path in sorted(root.rglob("*.yaml")):
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


@needs_logic_corpus
def test_shipped_handwritten_kernels_keep_the_shared_problem_type():
    root = require_logic_root()
    kernels = 0
    skipped = []
    for path in _logic_files_with_handwritten_kernels(root):
        data = _load_logic(path)
        shared = _shared_problem_type(data, path)
        defaults = data.get("DefaultSolution") or {}
        for solution in data["Solutions"]:
            state = L.fillSolutionDefaults(dict(solution), dict(defaults))
            name = _handwritten_kernel_name(state)
            if not name or name in SKINNY_KERNELS:
                continue
            try:
                getCustomKernelConfig(name, state.get("InternalSupportParams", {}))
            except (RuntimeError, KeyError, TypeError):
                skipped.append(name)
                continue
            kernels += 1
            features = readCustomKernelConfig(name).get("Features")
            result = L._problemTypeForCustomKernel(shared, features, False, str(path))
            key = (str(path.relative_to(root)), name)
            assert result is shared, key
            assert _state_and_predicates(result) == _state_and_predicates(shared), key

    # The loader skips a kernel without a usable custom.config, which would hide it here.
    assert skipped == []
    # 189 solutions were scanned when written; the floor guards against a filter that matches nothing.
    assert kernels > 100

