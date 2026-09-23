# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Strict type validation for characterization YAML fixtures."""

from pathlib import Path

import pytest
import yaml

from Tensile.Common.TypeValidationErrors import ConfigTypeError
from Tensile.Common.ValidParameters import (
    checkParametersAreValid,
    validParameters,
    validateInternalSupportParams,
)
from Tensile.SolutionStructs.Problem import validateProblemTypeParameterTypes
from Tensile.SolutionStructs.Solution import validateParameterTypes

pytestmark = pytest.mark.unit

_YAML_ROOT = Path(__file__).resolve().parent
_YAML_FILES = sorted(
    p for suffix in ("*.yaml", "*.yml") for p in _YAML_ROOT.rglob(suffix)
)


def _path_label(path):
    return str(path.relative_to(_YAML_ROOT))


@pytest.mark.parametrize("yaml_path", _YAML_FILES, ids=_path_label)
def test_characterization_yaml_types_are_strict(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    errors = []
    _validate_node(data, yaml_path, _path_label(yaml_path), errors)

    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        pytest.fail(f"{_path_label(yaml_path)} has YAML type mismatches:\n{joined}")


def _validate_node(node, yaml_path, key_path, errors):
    if isinstance(node, dict):
        _validate_problem_type_dict(node, yaml_path, key_path, errors)
        _validate_benchmark_config(node, yaml_path, key_path, errors)
        _validate_nested_problem_type(node, yaml_path, key_path, errors)
        for key, value in node.items():
            _validate_node(value, yaml_path, f"{key_path}.{key}", errors)
    elif isinstance(node, list):
        _validate_library_logic_solutions(node, yaml_path, key_path, errors)
        for idx, item in enumerate(node):
            _validate_node(item, yaml_path, f"{key_path}[{idx}]", errors)


def _is_problem_type_dict(node):
    return isinstance(node, dict) and node.get("OperationType") in {
        "GEMM",
        "TensorContraction",
        "ConvolutionForward",
        "ConvolutionBackwardData",
        "ConvolutionBackwardWeights",
    }


def _validate_problem_type_dict(node, yaml_path, key_path, errors):
    if not _is_problem_type_dict(node):
        return
    try:
        validateProblemTypeParameterTypes(node, srcFile=str(yaml_path), keyPathPrefix=key_path)
    except ConfigTypeError as exc:
        errors.append(str(exc))


def _validate_nested_problem_type(node, yaml_path, key_path, errors):
    problem_type = node.get("ProblemType")
    if not isinstance(problem_type, dict):
        return
    try:
        validateProblemTypeParameterTypes(
            problem_type,
            srcFile=str(yaml_path),
            keyPathPrefix=f"{key_path}.ProblemType",
        )
    except ConfigTypeError as exc:
        errors.append(str(exc))


def _validate_benchmark_config(node, yaml_path, key_path, errors):
    benchmark_problems = node.get("BenchmarkProblems")
    if not isinstance(benchmark_problems, list):
        return

    for problem_idx, benchmark_problem in enumerate(benchmark_problems):
        if not isinstance(benchmark_problem, list) or len(benchmark_problem) < 2:
            continue
        problem_type = benchmark_problem[0]
        if isinstance(problem_type, dict):
            _validate_problem_type_dict(
                problem_type,
                yaml_path,
                f"{key_path}.BenchmarkProblems[{problem_idx}][0]",
                errors,
            )
        for group_idx, group in enumerate(benchmark_problem[1:], start=1):
            if isinstance(group, dict):
                _validate_parameter_group(
                    group,
                    yaml_path,
                    f"{key_path}.BenchmarkProblems[{problem_idx}][{group_idx}]",
                    errors,
                )


def _validate_parameter_group(group, yaml_path, key_path, errors):
    for section_name in ("BenchmarkCommonParameters", "ForkParameters"):
        section = group.get(section_name)
        if section is None:
            continue
        for entry_idx, entry in enumerate(section):
            if not isinstance(entry, dict):
                continue
            entry_path = f"{key_path}.{section_name}[{entry_idx}]"
            for name, values in entry.items():
                if name == "Groups":
                    _validate_groups(values, yaml_path, f"{entry_path}.Groups", errors)
                    continue
                _validate_candidate_values(name, values, yaml_path, entry_path, errors)

    internal_support = group.get("InternalSupportParams")
    if isinstance(internal_support, dict):
        try:
            validateInternalSupportParams(
                internal_support,
                srcFile=str(yaml_path),
                keyPathPrefix=f"{key_path}.InternalSupportParams",
            )
        except ConfigTypeError as exc:
            errors.append(str(exc))


def _validate_groups(groups, yaml_path, key_path, errors):
    if not isinstance(groups, list):
        return
    for group_idx, group in enumerate(groups):
        if not isinstance(group, list):
            continue
        for entry_idx, entry in enumerate(group):
            if not isinstance(entry, dict):
                continue
            entry_path = f"{key_path}[{group_idx}][{entry_idx}]"
            for name, value in entry.items():
                _validate_candidate_values(name, [value], yaml_path, entry_path, errors)


def _validate_candidate_values(name, values, yaml_path, key_path, errors):
    if not isinstance(values, list):
        values = [values]
    try:
        checkParametersAreValid(
            (name, values),
            validParameters,
            keyPathPrefix=key_path,
            srcFile=str(yaml_path),
        )
    except ConfigTypeError as exc:
        errors.append(str(exc))


def _validate_library_logic_solutions(node, yaml_path, key_path, errors):
    if len(node) < 6 or not isinstance(node[5], list):
        return
    for idx, solution in enumerate(node[5]):
        if not isinstance(solution, dict):
            continue
        for (name, actual, expected), value_repr, _ in validateParameterTypes(
            solution,
            srcFile=str(yaml_path),
        ):
            errors.append(
                f"{yaml_path}: {key_path}[5][{idx}].{name} = {value_repr} "
                f"({actual}); expected {expected}"
            )
