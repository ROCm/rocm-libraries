# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ record builder has to stay on the Python builder.

solutionFromState and Solution.FromOriginalState are maintained together.
A field added on the Python side has to show up here, or this test fails.
Updating the checked-in record then fails SolutionRecord in tensilelite-tests
until the C++ builder emits the same lines. There is no Python binding: the
two tests share the files in test_data/solution_records and nothing else.

Each case of test_data/solution_records/fixtures.yaml is a derived state, the
map FromOriginalState and solutionFromState read: a base state of states/ with
the edits the case lists (solution_state_fixtures.py). The files built from it
are checked in:

  <name>.record.txt          one line per field of the library record
  <name>.predicates.yaml     the three predicate trees with their values, in
                             the {type, index, value} form state() gives them

The tensilelite-tests build writes each state as <name>.msgpack and the trees
as <name>.predicates.msgpack, because the C++ test has no YAML parser.
SolutionRecord builds the record from the msgpack, writes it with
MessagePackWriteSolutionData and compares the predicate trees node by node, so
a threshold, an operand or a feature the C++ builder drops fails there. Both
tests list the cases of fixtures.yaml, so adding one needs no change to either.

The gfx950-plain state is the custom.config block of a generated kernel. The
other bases come from derive_solution_records.py, which runs a real library
logic file or benchmark config through the Python derivation. The cases
together have to cover the ones listed in
test_fixtures_cover_the_predicate_families, so a swapped operand or an unread
feature cannot hide behind a symmetric kernel.
"""

import contextlib
import io
import json
import copy

import pytest
import yaml

from solution_state_fixtures import fixtures, records_dir
from solution_state_fixtures import state as derived_state
from tensilelite.Common import state
from tensilelite.Common.DataType import DataType
from tensilelite.Contractions import Solution

pytestmark = pytest.mark.unit

FIXTURES = sorted(fixtures())


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def _built(derived):
    """Solution.RecordFromOriginalState, the part of FromOriginalState the library write reads."""
    with contextlib.redirect_stdout(io.StringIO()):
        return Solution.RecordFromOriginalState(copy.deepcopy(derived))


def _linear_model(model):
    # The defaults the library loader gives a key the record leaves out.
    return {
        "slope": float(model.get("slope", 1.0)),
        "intercept": float(model.get("intercept", 0.0)),
        "max": float(model.get("max", 1000.0)),
    }


def _problem_type(problemType):
    """state() of the problem type, with the two DataType lists as names.

    An explicit list holds enum values and a default list holds DataTypes, so
    state() gives one as numbers and the other as names.
    """
    rv = state(problemType)
    for key in ("biasDataTypeWhiteList", "gateResidualDataTypeWhiteList"):
        rv[key] = [DataType(item).toName() for item in getattr(problemType, key)]
    return rv


def library_record(derived):
    """The library record FromOriginalState builds from one derived state, as the library reads it."""
    solution = _built(derived)
    # FromOriginalState leaves these unset without SolutionNameMin,
    # KernelNameMin or SolutionIndex. ContractionSolution has no unset value,
    # and applyNaming and FixSolutionIndices set them before a library write.
    name = "" if solution.name is None else solution.name
    kernelName = getattr(solution, "kernelName", "")
    index = 0 if solution.index is None else solution.index
    return {
        "solutionName": name,
        "kernelName": kernelName,
        "sizeMapping": state(solution.sizeMapping),
        "internalArgsSupport": state(solution.internalArgsSupport),
        "problemType": _problem_type(solution.problemType),
        "customKernel": state(solution.customKernel),
        "debugKernel": solution.debugKernel,
        "libraryLogicIndex": solution.libraryLogicIndex,
        "index": index,
        "ideals": {int(key): float(value) for key, value in solution.ideals.items()},
        "linearModel": _linear_model(solution.linearModel),
    }


def predicate_payload(derived):
    """Predicate arguments, not only their types. A threshold change fails this."""
    solution = _built(derived)
    return {
        "problem": state(solution.problemPredicate),
        "task": state(solution.taskPredicate),
        "hardware": state(solution.hardwarePredicate),
    }


def record_lines(record):
    lines = []

    def emit(prefix, value):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                emit(f"{prefix}.{key}" if prefix else str(key), item)
        else:
            lines.append(f"{prefix}={_json(value)}")

    for key, value in record.items():
        emit(key, value)
    return lines


@pytest.mark.parametrize("name", FIXTURES)
def test_python_builder_matches_checked_in_record(name):
    derived = derived_state(name)
    expected = (records_dir() / f"{name}.record.txt").read_text().splitlines()
    assert record_lines(library_record(derived)) == expected


@pytest.mark.parametrize("name", FIXTURES)
def test_predicate_arguments_match_checked_in_payload(name):
    derived = derived_state(name)
    path = records_dir() / f"{name}.predicates.yaml"
    with path.open() as handle:
        expected = yaml.safe_load(handle)
    assert predicate_payload(derived) == expected


def _predicate_types(node):
    found = []
    if isinstance(node, dict):
        if "type" in node:
            found.append(node["type"])
        for item in node.values():
            found.extend(_predicate_types(item))
    elif isinstance(node, list):
        for item in node:
            found.extend(_predicate_types(item))
    return found


def _predicate_values(node, kind):
    if isinstance(node, dict):
        if node.get("type") == kind:
            yield node["value"]
        for item in node.values():
            yield from _predicate_values(item, kind)
    elif isinstance(node, list):
        for item in node:
            yield from _predicate_values(item, kind)


def test_fixtures_cover_the_predicate_families():
    """A fixture set of symmetric kernels cannot tell A from B or M from N.

    Each case below is one way a C++ builder can be wrong and still match a
    square, default-feature kernel. A fixture that stops covering a case fails
    here, not silently in the C++ comparison.
    """
    states = {name: derived_state(name) for name in FIXTURES}
    payloads = {name: predicate_payload(state) for name, state in states.items()}

    def problem(state):
        return state["ProblemType"]

    def some(test):
        return [name for name, state in states.items() if test(state)]

    assert some(lambda s: s["MacroTile0"] != s["MacroTile1"]), "no non-square tile"
    assert some(lambda s: s["MacroTile0"] >= 128 or s["MacroTile1"] >= 128), "no large tile"
    assert some(
        lambda s: s["GlobalReadVectorWidthA"] != s["GlobalReadVectorWidthB"]
    ), "GlobalReadVectorWidthA is always B"
    assert len({s["DepthU"] for s in states.values()}) >= 3, "DepthU barely varies"
    assert some(lambda s: problem(s)["DataTypeA"] != problem(s)["DataTypeB"]), "A type is B type"
    assert len({tuple(s["ISA"]) for s in states.values()}) >= 4, "too few architectures"

    assert some(lambda s: problem(s).get("MXBlockA", 0) > 0), "no MX kernel"
    assert some(lambda s: s.get("MXScaleFormat", "NoSwizzle") != "NoSwizzle"), "no MX scale format"
    assert some(
        lambda s: problem(s).get("MXBlockA", 0) > 0
        and problem(s).get("MXBlockFreeA", 1) != problem(s).get("MXBlockFreeB", 1)
    ), "MXBlockFree is always 1 or the same for A and B"
    assert some(
        lambda s: len(problem(s).get("ScaleBlockSizesA", [])) > 1
        and problem(s).get("Int4EncodingA") == "UnsignedBias8"
    ), "no int4 kernel with several group sizes and UnsignedBias8"
    assert some(lambda s: problem(s).get("StochasticRounding")), "no stochastic rounding"
    assert some(lambda s: s.get("TileProcessingStrategy") == "StreamK"), "no StreamK kernel"
    assert some(lambda s: s.get("ReuseAcrossPersistent")), "no ReuseAcrossPersistent kernel"
    assert some(
        lambda s: s["InternalSupportParams"].get("PersistentLoopArgsVersion") == 2
    ), "no PersistentLoopArgsVersion 2 kernel"
    assert some(
        lambda s: s["InternalSupportParams"].get("SupportOccupancyProbe")
    ), "no occupancy probe"

    persistent = ("StreamK", "DataParallel")
    assert some(
        lambda s: s["GlobalSplitU"] > 1 and s.get("TileProcessingStrategy") not in persistent
    ), "no GlobalSplitU above 1 with GlobalSplitUCheckMinK"
    assert some(lambda s: s["GlobalSplitU"] == -1), "no GlobalSplitU -1"
    assert some(
        lambda s: s["GlobalSplitU"] > 1 and s.get("_GlobalAccumulation") != "MultipleBuffer"
    ), "no GlobalSplitU above 1 that adds DeterministicMode"
    assert some(
        lambda s: s.get("TileProcessingStrategy") == "StreamK" and s.get("StreamKAtomic") == 1
    ), "no StreamKAtomic 1"
    assert some(
        lambda s: not s["EnableMatrixInstruction"] and s.get("SynchronizerSizeCheck")
    ), "no SynchronizerSizeCheck without matrix instructions"
    assert some(lambda s: s["KernelLanguage"] == "Source"), "no source kernel"

    assert some(lambda s: not problem(s).get("UseBeta", True)), "no kernel without beta"
    assert some(
        lambda s: problem(s).get("UseBias") and "BiasDataTypeList" not in problem(s)
    ), "no bias with the default type list"
    assert some(lambda s: problem(s).get("BiasSrc", "D") != "D"), "BiasSrc is always D"
    assert some(lambda s: problem(s).get("UseE")), "no UseE"
    assert some(
        lambda s: problem(s).get("UseGateResidual") and "GateResidualDataTypeList" not in problem(s)
    ), "no gate residual with the default type list"

    assert some(
        lambda s: tuple(s["ISA"]) == (9, 5, 0)
        and isinstance(s.get("DeviceNames"), list)
        and len(s["DeviceNames"]) > 1
        and any(name != name.lower() for name in s["DeviceNames"])
    ), "no gfx950 kernel with several DeviceNames, one upper-case"
    assert some(lambda s: s.get("CUCount")), "no CUCount"
    assert some(lambda s: s["SpaceFillingAlgo"]), "no space-filling workgroup mapping"
    assert some(
        lambda s: problem(s).get("MXBlockA", 0) != problem(s).get("MXBlockB", 0)
        and problem(s).get("DataTypeMXSA") != problem(s).get("DataTypeMXSB")
    ), "MXBlock and the MX scale type are always the same for A and B"

    assert some(lambda s: problem(s).get("Sparse")), "no sparse kernel"
    assert some(lambda s: problem(s).get("MetadataLayout")), "no MetadataLayout"
    assert some(lambda s: problem(s).get("Gradient")), "no gradient"
    assert some(lambda s: problem(s).get("UseScaleCD")), "no UseScaleCD"
    assert some(lambda s: problem(s).get("OutputAmaxD")), "no OutputAmaxD"
    assert some(lambda s: problem(s).get("F32XdlMathOp", 0) != 0), "F32XdlMathOp is always Float"
    assert some(lambda s: problem(s)["ComputeDataType"] != 0), "the compute type is always Float"
    assert some(lambda s: problem(s).get("ScaleZeroPointA")), "no ScaleZeroPointA"
    assert some(
        lambda s: problem(s).get("Int4EncodingA") == "Signed" and problem(s).get("ScaleBlockSizesA")
    ), "no int4 kernel with an explicit Signed encoding"

    types = {name: set(_predicate_types(payload)) for name, payload in payloads.items()}
    assert any("Or" in kinds for kinds in types.values()), "no Or predicate"
    assert any("ScaleBlockSizeA" in kinds for kinds in types.values())
    assert any("GlobalSplitUCheckMinK" in kinds for kinds in types.values())
    for kind in (
        "DeterministicMode",
        "BetaZero",
        "CUCount",
        "PciChipId",
        "AIGreaterThanEqual",
        "AILessThanEqual",
        "Free0SizeMultiple",
        "Free1SizeMultiple",
        "BoundSizeMultiple",
        "SizeEqual",
        "SizeGreaterThan",
        "SizeLessThan",
        "StrideAEqual",
        "StrideBEqual",
        "StrideCEqual",
        "StrideDEqual",
        "BatchSizeEqual",
        "CDStridesEqual",
        "GemmA2AShape",
    ):
        assert any(kind in kinds for kinds in types.values()), f"no {kind} predicate"
    assert any(
        "Or" in _predicate_types(payload["hardware"]) for payload in payloads.values()
    ), "no Or of chip ids"
    assert any(
        {"type": "SwizzleTensorA", "value": True} in payload["problem"]["value"]
        for payload in payloads.values()
    ), "no SwizzleTensorA predicate set to true"

    gl2 = [
        value
        for payload in payloads.values()
        for value in _predicate_values(payload["problem"], "GL2PrefetchSAddrOffsetLimitCheck")
    ]
    assert gl2, "no GL2PrefetchSAddrOffsetLimitCheck predicate"
    for a, b in (
        ("TLUA", "TLUB"),
        ("DepthUA", "DepthUB"),
        ("ClusterMT0", "ClusterMT1"),
        ("MXBlockA", "MXBlockB"),
        ("MXBlockFreeA", "MXBlockFreeB"),
    ):
        assert any(value[a] != value[b] for value in gl2), f"GL2 prefetch {a} is always {b}"
    assert any(value["Metadata"] for value in gl2), "no GL2 prefetch with metadata"

    def gl2_prefetch(s):
        return s.get("PrefetchGL2", 0) > 0 and s.get("PrefetchGL2SAddr")

    assert some(
        lambda s: gl2_prefetch(s) and s["MacroTile0"] != s["MacroTile1"]
    ), "no GL2 prefetch on a non-square tile"
    assert [
        name
        for name, state in states.items()
        if state.get("PrefetchGL2", 0) > 0
        and not state.get("PrefetchGL2SAddr")
        and not any(
            _predicate_values(payloads[name]["problem"], "GL2PrefetchSAddrOffsetLimitCheck")
        )
    ], "no PrefetchGL2 without PrefetchGL2SAddr"
    assert some(
        lambda s: gl2_prefetch(s)
        and s.get("enableTDMMetadata")
        and problem(s).get("Sparse") == 2
        and s["ClusterDim"][0] != s["ClusterDim"][1]
        and s["_DepthUMetadata"] != s["DepthU"]
        and problem(s).get("TLUMetadata") != problem(s)["TLUA"]
    ), "no GL2 prefetch with metadata fields that differ from the A and B ones"
    assert some(
        lambda s: gl2_prefetch(s) and problem(s).get("MXBlockFreeB", 1) < 1
    ), "no GL2 prefetch with an MXBlockFree below 1"

    formats = {
        name: list(_predicate_values(payload["problem"], "MXScaleFormatEquals"))
        for name, payload in payloads.items()
    }
    assert {1, 3}.issubset(
        {value for values in formats.values() for value in values}
    ), "no MXScaleFormatEquals with a loose and an exact format"
    for only, other in (("MXBlockA", "MXBlockB"), ("MXBlockB", "MXBlockA")):
        assert [
            name
            for name, state in states.items()
            if problem(state).get(only, 0) and not problem(state).get(other, 0) and formats[name]
        ], f"no MXScaleFormatEquals with only {only}"
    assert [
        name
        for name, state in states.items()
        if not problem(state).get("MXBlockA", 0)
        and not problem(state).get("MXBlockB", 0)
        and not formats[name]
    ], "no kernel without MX scales"
    assert any("AlphaValue" in kinds for kinds in types.values()), "no AlphaValue predicate"

    mappings = [library_record(state)["sizeMapping"] for state in states.values()]
    for key in (
        "adaptiveGemm",
        "temporalHintD",
        "PrefetchLocalRead",
        "HalfPLR",
        "PrefetchGL2",
        "TDMInst",
        "Multicast",
    ):
        assert len({mapping[key] for mapping in mappings}) > 1, f"sizeMapping {key} never varies"


def test_record_holds_every_solution_state_key():
    """A field added to Solution.StateKeys has to reach the record and the C++ comparison."""
    record = library_record(derived_state("gfx950-plain"))
    renamed = {"name": "solutionName"}
    # The predicate trees are compared through <name>.predicates.yaml.
    trees = {"problemPredicate", "taskPredicate", "hardwarePredicate"}
    assert {renamed.get(key, key) for key in Solution.StateKeys} == set(record) | trees


@pytest.mark.parametrize(
    "name, edit",
    [
        ("gfx1250-streamk-cluster-rap-probe", lambda s: s.update(ClusterDim=[1, 3])),
        ("gfx1151-int4", lambda s: s["ProblemType"].update(Int4EncodingA="Unsigned")),
    ],
    ids=["cluster-1x3", "int4-unsigned"],
)
def test_builds_records_solution_from_state_rejects(name, edit):
    """SolutionState.hpp lists the states solutionFromState rejects and this builder accepts."""
    derived = derived_state(name)
    edit(derived)
    library_record(derived)


@pytest.mark.parametrize(
    "packed, a, b",
    [
        ("Float8BFloat8", "Float8", "BFloat8"),
        ("BFloat8Float8", "BFloat8", "Float8"),
        ("Float8BFloat8_fnuz", "Float8_fnuz", "BFloat8_fnuz"),
        ("BFloat8Float8_fnuz", "BFloat8_fnuz", "Float8_fnuz"),
    ],
)
def test_packed_fp8_type_splits_into_a_and_b(packed, a, b):
    """No fixture reaches the split; SolutionState.PackedFp8OperandsSplitIntoAAndB checks C++."""
    derived = derived_state("gfx950-plain")
    problem = derived["ProblemType"]
    for key in ("DataTypeA", "DataTypeB", "MacDataTypeA", "MacDataTypeB"):
        del problem[key]
    problem["DataType"] = DataType(packed).value
    record = library_record(derived)["problemType"]
    assert (record["aType"], record["bType"]) == (a, b)
    assert (record["computeInputTypeA"], record["computeInputTypeB"]) == (packed, packed)


@pytest.mark.parametrize("declared", (True, False))
def test_custom_kernel_uniform_summation_order_is_recorded(declared):
    derived = derived_state("gfx950-plain")
    derived["CustomKernel"]["uniformSummationOrder"] = declared
    record = library_record(derived)
    assert record["customKernel"]["uniformSummationOrder"] is declared
    assert f"customKernel.uniformSummationOrder={_json(declared)}" in record_lines(record)


def write_fixture(name):
    """The golden files of one case. Each C++ comparison reads them back."""
    derived = derived_state(name)
    directory = records_dir()
    lines = record_lines(library_record(derived))
    (directory / f"{name}.record.txt").write_text("\n".join(lines) + "\n")
    payload = predicate_payload(derived)
    (directory / f"{name}.predicates.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    return len(lines)


if __name__ == "__main__":
    # Regenerates the checked-in files after a deliberate Python change, or after
    # a base state or fixtures.yaml changed. The C++ test then fails until
    # solutionFromState builds the new record and predicate trees.
    for name in FIXTURES:
        print(name, write_fixture(name))
