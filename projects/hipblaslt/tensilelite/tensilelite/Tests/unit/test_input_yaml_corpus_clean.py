################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Corpus-clean CI test.

Walks every input YAML under Tensile/Tests/common/ and Tests/unit/test_data/
and asserts each one loads + parses + section-validates without raising
ConfigTypeError. This enforces the tree-cleanliness invariant going forward:
any new test YAML with a typo'd type fails this gate.

The test walks the four in-scope sections of each config (GlobalParameters,
BenchmarkProblems[*][0] ProblemType,
BenchmarkProblems[*][1].(BenchmarkCommonParameters|ForkParameters|
ForkParameters.Groups), LibraryLogic, InternalSupportParams). It does not run
kernel generation or any expensive downstream work.
"""

import os
from pathlib import Path

import pytest

from Tensile import LibraryIO
from Tensile.Common.GlobalParameters import (
    _assertGlobalParametersAreValid,
    _GLOBAL_PARAMETER_IGNORE_KEYS,
    defaultAnalysisParameters,
)
from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.Common.ValidParameters import (
    checkParametersAreValid,
    validateInternalSupportParams,
    validParameters,
)
from Tensile.SolutionStructs.Problem import (
    validateProblemTypeParameterTypes,
    _defaultProblemType,
)


_TESTS_DIR = Path(__file__).resolve().parent.parent
_YAML_ROOTS = [
    _TESTS_DIR / "common",
    _TESTS_DIR / "unit" / "test_data",
]
YAML_FILES = sorted(
    str(p)
    for root in _YAML_ROOTS
    for p in root.rglob("*.yaml")
)


def _validate_global_parameters(yaml_data, srcFile):
    if "GlobalParameters" not in yaml_data:
        return
    cfg = yaml_data["GlobalParameters"]
    if not cfg:
        return
    _assertGlobalParametersAreValid(cfg, _GLOBAL_PARAMETER_IGNORE_KEYS)


def _validate_problem_type(problem_type_dict, prefix, srcFile):
    """Run validateProblemTypeParameterTypes in raise mode."""
    # Walk the keys that ProblemType.__init__ would have populated into
    # state via assignParameterWithDefault. For corpus testing we just
    # validate what's present in the user dict (limited to recognized
    # ProblemType keys) -- the assignment step itself isn't exercised
    # so we don't need to instantiate a ProblemType.
    state = {k: v for k, v in problem_type_dict.items()
             if k in _defaultProblemType}
    validateProblemTypeParameterTypes(state, srcFile=srcFile,
                                      keyPathPrefix=prefix + ".ProblemType")


def _validate_benchmark_params(benchmark_slice, prefix, srcFile):
    """Run checkParametersAreValid on BenchmarkCommon/Fork/Groups."""
    if not isinstance(benchmark_slice, dict):
        return
    if "BenchmarkCommonParameters" in benchmark_slice:
        for d in benchmark_slice["BenchmarkCommonParameters"] or []:
            for name, values in d.items():
                if not isinstance(values, list):
                    continue
                try:
                    checkParametersAreValid(
                        (name, values), validParameters,
                        keyPathPrefix=prefix + ".BenchmarkCommonParameters",
                        srcFile=srcFile,
                    )
                except Exception as e:
                    # Only ConfigTypeError matters here; legacy invalid-
                    # value errors are out of scope for the corpus gate.
                    if isinstance(e, ConfigTypeError):
                        raise

    if "ForkParameters" in benchmark_slice:
        fp = benchmark_slice["ForkParameters"] or []
        groups = None
        for d in fp:
            for name, values in d.items():
                if name == "Groups":
                    groups = values
                    continue
                if not isinstance(values, list):
                    continue
                try:
                    checkParametersAreValid(
                        (name, values), validParameters,
                        keyPathPrefix=prefix + ".ForkParameters",
                        srcFile=srcFile,
                    )
                except Exception as e:
                    if isinstance(e, ConfigTypeError):
                        raise
        if groups:
            for gIdx, group_list in enumerate(groups):
                for eIdx, group in enumerate(group_list or []):
                    for k, v in (group or {}).items():
                        # A Groups entry may carry a list of scalars (production
                        # expands these into separate groups). Validate each
                        # scalar rather than the list object itself.
                        scalars = v if isinstance(v, list) else [v]
                        try:
                            checkParametersAreValid(
                                (k, scalars), validParameters,
                                keyPathPrefix=f"{prefix}.ForkParameters.Groups[{gIdx}][{eIdx}]",
                                srcFile=srcFile,
                            )
                        except Exception as e:
                            if isinstance(e, ConfigTypeError):
                                raise

    if "InternalSupportParams" in benchmark_slice:
        isp = benchmark_slice["InternalSupportParams"] or {}
        validateInternalSupportParams(
            isp, srcFile=srcFile,
            keyPathPrefix=prefix + ".InternalSupportParams",
        )


def _validate_library_logic(yaml_data, srcFile):
    if "LibraryLogic" not in yaml_data:
        return
    cfg = yaml_data["LibraryLogic"] or {}
    from Tensile.Common.TypeValidationErrors import formatMismatch
    from Tensile.Common.GlobalParameters import libraryLogicTypeOverrides
    errors = []
    for key, value in cfg.items():
        if key not in defaultAnalysisParameters:
            errors.append(
                f"LibraryLogic.{key}: unknown key. "
                f"Valid keys are {sorted(defaultAnalysisParameters.keys())}."
            )
            continue
        if key in libraryLogicTypeOverrides:
            expected = libraryLogicTypeOverrides[key]
        else:
            default = defaultAnalysisParameters[key]
            expected = {type(default)}
        if type(value) not in expected:
            errors.append(formatMismatch(srcFile, f"LibraryLogic.{key}", value, expected))
            continue
        if key == "SolutionImportanceMin" and not (0.0 <= value <= 1.0):
            errors.append(
                f"LibraryLogic.SolutionImportanceMin = {value!r} out of [0.0, 1.0]."
            )
    if errors:
        raise ConfigTypeError(
            f"LibraryLogic validation failed in {srcFile}:\n  "
            + "\n  ".join(errors)
        )


def _validate_one(path):
    data = LibraryIO.read(path)
    if data is None:
        return
    _validate_global_parameters(data, path)
    _validate_library_logic(data, path)
    bp = data.get("BenchmarkProblems") if isinstance(data, dict) else None
    if not bp:
        return
    for outerIdx, ptg in enumerate(bp):
        if not ptg:
            continue
        # Outer slice: BenchmarkProblems[<outerIdx>][0] is the ProblemType.
        ptDict = ptg[0] if ptg else {}
        outerPrefix = f"BenchmarkProblems[{outerIdx}][0]"
        if isinstance(ptDict, dict):
            _validate_problem_type(ptDict, outerPrefix, path)
        for innerIdx, group in enumerate(ptg[1:]):
            innerPrefix = f"BenchmarkProblems[{outerIdx}][{1 + innerIdx}]"
            _validate_benchmark_params(group, innerPrefix, path)


@pytest.mark.parametrize("yaml_path", YAML_FILES, ids=[
    os.path.relpath(p, str(_TESTS_DIR)) for p in YAML_FILES
])
def test_input_yaml_corpus_is_strict_clean(yaml_path):
    """Every in-tree input YAML loads + section-validates without ConfigTypeError."""
    try:
        _validate_one(yaml_path)
    except ConfigTypeError as e:
        pytest.fail(
            f"Strict validation failed for {yaml_path}:\n{e}\n"
            f"(Fix the YAML, or run utilities/fix_yaml_types.py over the tree.)"
        )
