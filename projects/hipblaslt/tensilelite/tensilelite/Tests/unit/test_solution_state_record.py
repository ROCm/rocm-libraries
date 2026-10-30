# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ record builder has to stay on the Python builder.

solutionFromState and Solution.FromOriginalState are maintained together.
A field added on the Python side has to show up here, or this test fails.
Updating the checked-in record then fails SolutionRecord in tensilelite-tests
until the C++ builder emits the same lines. There is no Python binding: the
two tests share the record file and nothing else.

The derived states are the custom.config blocks of the three plain kernels
in the JIT sources (gfx90a, gfx942, gfx950). The record is what
FromOriginalState builds from that map: the same pieces solutionFromState
builds, and not the assembler object Solution keeps beside them.
"""

import contextlib
import io
import json
from pathlib import Path

import msgpack
import pytest
import yaml

from tensilelite.Common import state
from tensilelite.Contractions import (
    CustomKernel,
    InternalArgsSupport,
    ProblemPredicate,
    ProblemType,
    SizeMapping,
    TaskPredicate,
)
from tensilelite import Hardware

pytestmark = pytest.mark.unit

_ARCHES = ("gfx90a-plain", "gfx942-plain", "gfx950-plain")
_MX_SCALE_FORMAT = {"NoSwizzle": 0, "HostPreSwizzle": 1, "InMemorySwizzle": 2}


def records_dir():
    return Path(__file__).resolve().parent / "data" / "solution_records"


def _tags(predicate):
    found = []

    def walk(node):
        if isinstance(node, dict) and "type" in node:
            found.append(node["type"])
            value = node.get("value")
            if isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                walk(value)

    walk(state(predicate))
    return found


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def _built(derived):
    problem = ProblemType.FromOriginalState(derived["ProblemType"])
    problem.mxScaleFormat = _MX_SCALE_FORMAT.get(derived.get("MXScaleFormat", "NoSwizzle"), 0)
    with contextlib.redirect_stdout(io.StringIO()):
        hardware = Hardware.HardwarePredicate.FromHardware(
            derived["ISA"], derived.get("CUCount"), derived.get("DeviceNames"), logicFile=""
        )
    return {
        "problem": problem,
        "problemPredicate": ProblemPredicate.FromOriginalState(derived, problem),
        "taskPredicate": TaskPredicate.FromOriginalState(derived, problem),
        "hardware": hardware,
    }


def library_record(derived):
    """The library record FromOriginalState builds from one derived state."""
    built = _built(derived)
    record = {
        "solutionName": derived.get("SolutionNameMin", ""),
        "kernelName": derived.get("KernelNameMin", ""),
        "sizeMapping": state(SizeMapping.FromOriginalState(derived)),
        "internalArgsSupport": state(InternalArgsSupport.FromOriginalState(derived)),
        "problemType": state(built["problem"]),
        "problemPredicate": _tags(built["problemPredicate"]),
        "taskPredicate": _tags(built["taskPredicate"]),
        "hardwarePredicate": _tags(built["hardware"]),
    }
    if "CustomKernel" in derived:
        record["customKernel"] = state(CustomKernel.FromOriginalState(derived["CustomKernel"]))
    return record


def predicate_payload(derived):
    """Predicate arguments, not only their types. A threshold change fails this."""
    built = _built(derived)
    return {
        "problem": state(built["problemPredicate"]),
        "task": state(built["taskPredicate"]),
        "hardware": state(built["hardware"]),
    }


def record_lines(record):
    lines = []

    def emit(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                emit(f"{prefix}.{key}" if prefix else str(key), item)
        else:
            lines.append(f"{prefix}={_json(value)}")

    for key, value in record.items():
        emit(key, value)
    return lines


def _load_derived(name):
    path = records_dir() / f"{name}.yaml"
    with path.open() as handle:
        return yaml.safe_load(handle)


@pytest.mark.parametrize("name", _ARCHES)
def test_python_builder_matches_checked_in_record(name):
    derived = _load_derived(name)
    expected = (records_dir() / f"{name}.record.txt").read_text().splitlines()
    assert record_lines(library_record(derived)) == expected


@pytest.mark.parametrize("name", _ARCHES)
def test_predicate_arguments_match_checked_in_payload(name):
    derived = _load_derived(name)
    path = records_dir() / f"{name}.predicates.yaml"
    with path.open() as handle:
        expected = yaml.safe_load(handle)
    assert predicate_payload(derived) == expected


@pytest.mark.parametrize("name", _ARCHES)
def test_msgpack_state_matches_yaml(name):
    """solutionFromState reads the msgpack copy. It has to be this YAML."""
    derived = _load_derived(name)
    packed = (records_dir() / f"{name}.msgpack").read_bytes()
    assert msgpack.unpackb(packed, strict_map_key=False) == derived


def _pack(value):
    return msgpack.packb(value, use_bin_type=True)


if __name__ == "__main__":
    # Regenerates the checked-in record after a deliberate Python change.
    # The C++ test then fails until solutionFromState emits the new lines.
    for name in _ARCHES:
        derived = _load_derived(name)
        directory = records_dir()
        (directory / f"{name}.msgpack").write_bytes(_pack(derived))
        lines = record_lines(library_record(derived))
        (directory / f"{name}.record.txt").write_text("\n".join(lines) + "\n")
        (directory / f"{name}.predicates.yaml").write_text(
            yaml.safe_dump(predicate_payload(derived), sort_keys=False)
        )
        print(name, len(lines))
