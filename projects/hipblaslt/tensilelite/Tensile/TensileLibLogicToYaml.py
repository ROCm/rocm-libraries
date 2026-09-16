################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from Tensile import __version__
from Tensile import LibraryIO
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.Constants import HR
from Tensile.SolutionStructs.Problem import _defaultProblemType as defaultProblemType
from Tensile.Common.GlobalParameters import globalParameters
from Tensile.Common.Architectures import isaToGfx
from Tensile.Common import IsaVersion
from Tensile.Common.ValidParameters import validParameters
from Tensile.Common.GlobalParameters import globalParameters as globalParameterDefaults

import argparse
import ast
import inspect
import os
import sys
import re
import yaml
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, List, Optional, Tuple


#: Emitted only when no run config supplies a real device name.
FALLBACK_DEVICE_NAME = "Device 0000"

#: Entries every benchmark data file starts with: MinimumRequiredVersion and
#: ProblemSizes.
BENCHMARK_REQUIRED_HEADER_LEN = 2

#: Header entries that may follow, in this order. _writeSolutionsHeader emits
#: each one only when it is set, so the first solution state does not sit at a
#: fixed index.
BENCHMARK_OPTIONAL_HEADER_KEYS = ("BiasTypeArgs", "ActivationArgs", "GateTypeArgs")

#: Written by ClientWriter into every benchmark step directory. This is the
#: authoritative record of the settings a run actually used, and unlike the
#: tuning config it always lives inside the build tree next to the data file.
CLIENT_PARAMETERS_FILENAME = "ClientParameters.ini"

#: ini key -> GlobalParameters key for pairs the source scan cannot resolve on
#: its own. Each is a deliberate exception, not a copy of what the scan finds.
CLIENT_PARAMETER_OVERRIDES = {
    # deviceId reaches ClientWriter as a function argument; Tensile.py sources it
    # from config["GlobalParameters"]["Device"], which is a different module.
    "device-idx": "Device",
    # The scan reports these as ambiguous because the value is conditionally
    # rewritten on the way out. Reversing is still correct for benchmark steps,
    # which is what this tool reads:
    #   init-a / init-b fall back to DataInitTypeAB only when the type is -1,
    "init-a": "DataInitTypeA",
    "init-b": "DataInitTypeB",
    #   init-beta is forced to Zero when the problem has no beta,
    "init-beta": "DataInitTypeBeta",
    #   num-elements-to-validate takes max() with the Winner variant only when
    #   forBenchmark is False.
    "num-elements-to-validate": "NumElementsToValidate",
}

#: Settings whose value cannot be read back. activation-additional-args is a
#: comma-joined list; the other two repeat across lines and would be truncated
#: to their first element by the single-value ini parse.
CLIENT_PARAMETER_EXCLUSIONS = frozenset(
    ("activation-additional-args", "streamk-hybrid-mode", "rocprof-counter")
)

#: ini settings written through an int -> name helper in ClientWriter rather
#: than an Enum. The inverse is built by probing the helper, so a new mode name
#: needs no change here.
CLIENT_NAME_LOOKUPS = {"bounds-check": "boundsCheckName", "prune-mode": "pruneModeName"}

#: A tuning run writes its build tree to "<runConfigDir>/build_<stem>/", so the
#: config that produced a benchmark data file can be recovered from its path.
BUILD_DIR_PREFIX = "build_"

#: Problem type keys this script carries in dedicated fields instead of inside
#: the ProblemType block.
PROBLEM_TYPE_RELOCATED_KEYS = ("BiasDataTypeList", "GateResidualDataTypeList")

#: Always emitted, whether or not they match the registry default.
PROBLEM_TYPE_ALWAYS_EMITTED = (
    "DataType",
    "DestDataType",
    "ComputeDataType",
    "HighPrecisionAccumulate",
    "TransposeA",
    "TransposeB",
)

#: Solution keys re-emitted in richer form inside the Groups block.
SOLUTION_KEYS_IN_GROUPS = ("MatrixInstruction", "WorkGroup")


class Quoted(str):
    pass


# print ""
def quotedPresenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


class FlowList(list):
    pass


def makeFlow(value):
    if isinstance(value, list):
        return FlowList(value)
    return value


# print list in format of []
def flowSeq(dumper, value):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


# ignore null
def representNone(self, _):
    return self.represent_scalar("tag:yaml.org,2002:null", "")


yaml.add_representer(Quoted, quotedPresenter)
yaml.add_representer(FlowList, flowSeq)
yaml.add_representer(type(None), representNone)


def tPrint(verbosity: int, arg) -> None:
    """Conditionally prints input to stdout.

    If the global print level is greater than or equal to the verbosity,
    the argument is printed to stdout.

    Args:
        verbosity: Level to use for printing arg.
        arg: Item to print to stdout.
    """
    if globalParameters["ClientLogLevel"] >= verbosity:
        print(arg)
        sys.stdout.flush()


def buildFallbackGlobalParams(problemTypeState: dict) -> Dict[str, Any]:
    """GlobalParameters used only when the originating run config cannot be found.

    These are timing-harness defaults, not values recovered from the input, so
    every caller announces that they were assumed.
    """
    isInt8 = problemTypeState.get("DataType") == "I8"
    return {
        "SleepPercent": 0,
        "KernelTime": True,
        "NumElementsToValidate": 0,
        "DataInitTypeBeta": 1,
        "DataInitTypeAlpha": 1,
        "DataInitTypeA": 3 if isInt8 else 12,
        "DataInitTypeB": 3 if isInt8 else 13,
        "DataInitTypeC": 3 if isInt8 else 12,
        "DataInitTypeD": 3 if isInt8 else 12,
        "PreciseKernelTime": False,
        "Device": 0,
        "SkipSlowSolutionRatio": 0.0,
        "KeepBuildTmp": False,
    }


def setGlobalParams(
    versionString: dict,
    problemTypeState: dict,
    runSettings: Optional["RunSettings"] = None,
) -> dict:
    """Forms the GlobalParameters block.

    The originating run config is preferred so the generated config times the
    kernel under the same regime as the run it reproduces; rotating buffers,
    warmup counts and device index all live here and are unrecoverable from
    benchmark data alone.
    """
    if runSettings is not None and runSettings.globalParameters:
        res = dict(runSettings.globalParameters)
        tPrint(
            1,
            "GlobalParameters recovered from: {} -> {} keys.".format(
                runSettings.describe(), len(res)
            ),
        )
    else:
        res = buildFallbackGlobalParams(problemTypeState)
        tPrint(
            1,
            "Warning: no run config found, so GlobalParameters are assumed defaults, not "
            "the settings of the originating run. Timing-sensitive keys (RotatingBufferSize, "
            "NumWarmups, EnqueuesPerSync, SleepPercent) and Device are absent or set to 0. "
            "Pass --run-config to inherit them.",
        )

    # The input file is authoritative for the version it was produced with, and
    # it stays the first key so the emitted block keeps its conventional order.
    return {
        "MinimumRequiredVersion": versionString["MinimumRequiredVersion"],
        **{k: v for k, v in res.items() if k != "MinimumRequiredVersion"},
    }


def formProblemTypeYamlData(problemTypeState: dict) -> dict:
    if len(problemTypeState) == 0:
        raise RuntimeError(
            "Length of problem Type Parameters is empty!!, Please re-check the library logic file !"
        )

    data = {}
    dropped = []
    data["OperationType"] = problemTypeState["OperationType"]
    for problemTypeKey, problemTypeValue in problemTypeState.items():
        # Always print HighPrecisionAccumulate, TransposeA, TransposeB fields
        if problemTypeKey in [
            "DataType",
            "DestDataType",
            "ComputeDataType",
            "HighPrecisionAccumulate",
            "TransposeA",
            "TransposeB",
        ]:
            data[problemTypeKey] = problemTypeValue
            continue

        # Print default keys with no default values
        if problemTypeKey not in defaultProblemType:
            dropped.append(problemTypeKey)
        else:
            if problemTypeValue != defaultProblemType[problemTypeKey]:
                # Shipped library-logic encodes some bool fields as int 0/1;
                # ProblemType type-checks them strictly.
                if isinstance(defaultProblemType[problemTypeKey], bool):
                    problemTypeValue = bool(problemTypeValue)
                data[problemTypeKey] = makeFlow(problemTypeValue)
                continue

    warnDroppedKeys("ProblemType", dropped, set(problemTypeConfigOnlyKeys()))
    return data


def warnDroppedKeys(name: str, dropped, settable) -> None:
    """Reports keys the emitters skipped that a config could actually have set.

    The emitters keep only keys present in their registry, so anything else is
    discarded with no trace. Most such keys are values Tensile derives and
    recomputes, which is correct; this makes the exceptions visible.
    """
    lost = sorted(key for key in dropped if key in settable)
    if lost:
        tPrint(
            1,
            "Warning: {} key(s) dropped from {} because the emitter has no default for "
            "them, though a config can set them: {}. The generated config will fall back "
            "to Tensile's defaults for these.".format(len(lost), name, ", ".join(lost)),
        )


def formGroups(MIInstruction9Bits: dict) -> dict:
    data = {}
    data["Groups"] = [[]]
    group = {}
    for forkKey, forkValue in MIInstruction9Bits.items():
        group[forkKey] = forkValue
    data["Groups"][0].append(group)
    return data


def form9BitMIInst(currentSolutionState: dict) -> dict:
    MIBlock = currentSolutionState["MIBlock"]
    MIWaveTile = currentSolutionState["MIWaveTile"]
    MIWaveGroup = currentSolutionState["MIWaveGroup"]

    if len(MIBlock) == 0 or len(MIWaveTile) == 0 or len(MIWaveGroup) == 0:
        raise RuntimeError(
            "Length of MIBlock:{0}, MIWave Tile:{1},MIWaveGroup:{2} cannot be empty".format(
                len(MIBlock), len(MIWaveTile), len(MIWaveGroup)
            )
        )

    MIBlock1 = MIBlock[0:5]

    MIInstruction9Bits = MIBlock1 + MIWaveTile + MIWaveGroup

    groups = {}
    groups["MatrixInstruction"] = FlowList(MIInstruction9Bits)
    groups["WorkGroup"] = FlowList(currentSolutionState["WorkGroup"])
    groups["MIArchVgpr"] = currentSolutionState["MIArchVgpr"]

    return groups


def formForkParams(currentIndexSolution: dict, skipMI: bool) -> dict:

    data = {}
    data["InitialSolutionParameters"] = None
    kernelLang = {}
    kernelLang["KernelLanguage"] = FlowList(["Assembly"])
    data["BenchmarkCommonParameters"] = [kernelLang]

    forkData = []
    dropped = []
    for forkKey, forkValue in currentIndexSolution.items():
        temp = {}
        # # ignore MatrixInstruction
        if forkKey in SOLUTION_KEYS_IN_GROUPS:
            continue
        # Find the matching index for fork key name from list of dictionaries => defaultBenchmarkCommonParameters
        index = next(
            (i for i, d in enumerate(defaultBenchmarkCommonParameters) if forkKey in d),
            None,
        )
        if index is None:
            dropped.append(forkKey)
        else:
            forkValue = [forkValue]  # convert to list
            if forkValue != defaultBenchmarkCommonParameters[index][forkKey]:
                temp[forkKey] = FlowList(forkValue)
                forkData.append(temp)

    # ISA is carried as the architecture name; the Groups keys are re-emitted below.
    warnDroppedKeys(
        "ForkParameters",
        [key for key in dropped if key != "ISA"],
        set(validParameters) - set(SOLUTION_KEYS_IN_GROUPS),
    )

    # Skip the MI calculation if 9 bit MI is not needed or MatrixInstruction field is disabled
    isMatrixInsEnabled = False
    if "EnableMatrixInstruction" in currentIndexSolution:
        isMatrixInsEnabled = currentIndexSolution["EnableMatrixInstruction"]

        if (
            currentIndexSolution["EnableMatrixInstruction"]
            and currentIndexSolution["MatrixInstruction"]
        ):
            isMatrixInsEnabled = True
        else:
            tPrint(
                1,
                "Matrix instruction is disabled skipping the matrix instruction parameter ..",
            )

    # Iterate over MIs in Group
    if skipMI != True and isMatrixInsEnabled:
        groups = form9BitMIInst(currentIndexSolution)
    else:
        # formGroups is the only emitter for WorkGroup, since the loop above
        # skips SOLUTION_KEYS_IN_GROUPS.
        groups = {"WorkGroup": FlowList(currentIndexSolution["WorkGroup"])}

    forkData.append(formGroups(groups))

    data["ForkParameters"] = forkData

    return data


def splitBenchmarkHeader(data: list) -> Tuple[dict, int]:
    """Splits a benchmark data file into its header fields and its solutions.

    Mirrors LibraryIO.parseSolutionsData: the optional header entries are each
    written only when set, so the first solution state is found by scanning for
    the known keys rather than by assuming a fixed offset.

    Returns the header fields found and the index of the first solution state.
    """
    header = {}
    index = BENCHMARK_REQUIRED_HEADER_LEN
    for key in BENCHMARK_OPTIONAL_HEADER_KEYS:
        if index < len(data) and isinstance(data[index], dict) and key in data[index]:
            header[key] = data[index][key]
            index += 1
    return header, index


def normalizeBiasTypeArgs(biasTypeArgs: Optional[list]) -> Optional[list]:
    """Flattens the nested BiasTypeArgs shape written into benchmark data files.

    LibraryIO._writeSolutionsHeader wrapped the value in a second pair of
    brackets, so files written before that fix carry [[7]] where the benchmark
    config schema takes a flat [7]. Left nested, BiasTypeArgs in Solution.py
    hands the inner list to DataType and raises. Every already-generated file
    keeps the old shape, so accept both here rather than only fixing the writer.

    Returns None for an absent or empty value so the caller falls back to the
    problem type's BiasDataTypeList.
    """
    if biasTypeArgs is None:
        return None
    flattened = []
    for entry in biasTypeArgs:
        if isinstance(entry, (list, tuple)):
            flattened.extend(entry)
        else:
            flattened.append(entry)
    return flattened or None


def formProblemSize(
    exactLogic: Optional[list[Tuple[list, list]]],
    solutionIndex: int,
    problemTypeStat: dict,
    problemSizes: Optional[list] = None,
    biasTypeArgs: Optional[list] = None,
) -> dict:
    data = {}
    data["BenchmarkJoinParameters"] = None
    data["BenchmarkFinalParameters"] = []

    temp = {}
    # Benchmark data files carry the sizes that were actually run, so use them
    # verbatim instead of recovering a single size from the exact logic.
    if problemSizes is not None:
        temp["ProblemSizes"] = [
            {key: FlowList(value) for key, value in entry.items()}
            for entry in problemSizes
        ]
    # for origami exactLogic is not present so we need to create it
    elif exactLogic is None:
        tPrint(
            1, "Warning: For Origami liblogics, Exact logic needs to be set manually"
        )
        temp["ProblemSizes"] = [{"Exact": FlowList([1, 1, 1, 1])}]
    else:
        for size, mapping in exactLogic:
            if mapping[0] == solutionIndex:
                temp["ProblemSizes"] = [{"Exact": FlowList(size)}]

    data["BenchmarkFinalParameters"].append(temp)

    temp = {}
    biasTypeArgs = normalizeBiasTypeArgs(biasTypeArgs)
    if not biasTypeArgs:
        biasTypeArgs = problemTypeStat["BiasDataTypeList"]
    temp["BiasTypeArgs"] = FlowList(biasTypeArgs)
    gateTypeArgs = problemTypeStat.get("GateResidualDataTypeList", [])
    if gateTypeArgs:
        temp["GateTypeArgs"] = FlowList(gateTypeArgs)
    data["BenchmarkFinalParameters"].append(temp)

    return data


@dataclass(frozen=True)
class SolutionSource:
    """Everything needed to emit a config, independent of the input file format.

    Replaces the positional tuple the readers used to return: the fields are
    named at both ends, so adding one cannot silently shift the others.
    """

    versionString: dict
    scheduleName: str
    architectureName: Any
    deviceNames: List[str]
    problemType: dict
    solution: dict
    exactLogic: Optional[list] = None
    problemSizes: Optional[list] = None
    biasTypeArgs: Optional[list] = None
    #: Human readable description of where this came from.
    origin: str = ""
    #: True when scheduleName/deviceNames are stand-ins rather than recorded values.
    hasPlaceholderLibraryLogic: bool = False


class SourceReader(ABC):
    """Reads one input format into a SolutionSource.

    Adding a format means adding a reader and listing it in SOURCE_READERS;
    no existing reader or caller changes.
    """

    description: str = ""

    @staticmethod
    @abstractmethod
    def matches(data) -> bool:
        """Returns True if this reader understands the parsed yaml."""

    @staticmethod
    @abstractmethod
    def read(data, solutionIndex: int) -> SolutionSource:
        """Extracts the requested solution and its context."""


class BenchmarkDataReader(SourceReader):
    """Reads 1_BenchmarkProblems/<problem>/Data/*_Final.yaml.

    The format is a list of [MinimumRequiredVersion, ProblemSizes, then any of
    BiasTypeArgs, ActivationArgs and GateTypeArgs that were set, followed by
    *solutionStates]. Benchmark data predates the library logic
    step, so it records no schedule, device or architecture name; the
    architecture is recovered from the selected solution's ISA.
    """

    description = "benchmark data"

    @staticmethod
    def matches(data) -> bool:
        # A list-format library logic holds the schedule name (a string) at
        # index 1, and the dict format is not a list at all.
        if not (
            isinstance(data, list)
            and len(data) > BENCHMARK_REQUIRED_HEADER_LEN
            and isinstance(data[0], dict)
            and "MinimumRequiredVersion" in data[0]
            and isinstance(data[1], dict)
            and "ProblemSizes" in data[1]
        ):
            return False
        _, solutionOffset = splitBenchmarkHeader(data)
        return (
            len(data) > solutionOffset
            and isinstance(data[solutionOffset], dict)
            and "ProblemType" in data[solutionOffset]
        )

    @staticmethod
    def read(data, solutionIndex: int) -> SolutionSource:
        header, solutionOffset = splitBenchmarkHeader(data)
        solutionStates = data[solutionOffset:]
        solution = BenchmarkDataReader._selectSolution(solutionStates, solutionIndex)

        # Each benchmark solution embeds its own problem type; a library logic
        # instead carries one shared problem type for every solution.
        problemType = solution.get("ProblemType")
        if not problemType:
            raise RuntimeError(
                "Solution index {} in the benchmark data file has no ProblemType field.".format(
                    solutionIndex
                )
            )

        isa = solution.get("ISA")
        if not isa:
            raise RuntimeError(
                "Solution index {} in the benchmark data file has no ISA field, so the "
                "architecture name cannot be determined.".format(solutionIndex)
            )
        architectureName = isaToGfx(IsaVersion(*isa))

        # Every size the benchmark ran is attributed to the requested solution.
        problemSizes = data[1].get("ProblemSizes") or []
        exactLogic = [
            [entry["Exact"], [solutionIndex, 0.0]]
            for entry in problemSizes
            if "Exact" in entry
        ]

        return SolutionSource(
            versionString={"MinimumRequiredVersion": data[0].get("MinimumRequiredVersion")},
            scheduleName=architectureName,
            architectureName=architectureName,
            deviceNames=[FALLBACK_DEVICE_NAME],
            problemType=problemType,
            solution=solution,
            exactLogic=exactLogic,
            problemSizes=problemSizes,
            biasTypeArgs=normalizeBiasTypeArgs(header.get("BiasTypeArgs")),
            origin="benchmark data, solution {} ({})".format(
                solutionIndex, solution.get("SolutionNameMin", "<unnamed>")
            ),
            hasPlaceholderLibraryLogic=True,
        )

    @staticmethod
    def _selectSolution(solutionStates: list, solutionIndex: int) -> dict:
        """Matches the SolutionIndex field, falling back to position."""
        for solution in solutionStates:
            if solution.get("SolutionIndex") == solutionIndex:
                return solution

        if 0 <= solutionIndex < len(solutionStates):
            return solutionStates[solutionIndex]

        raise RuntimeError(
            "Could not find solution index {} in the benchmark data file, which holds {} "
            "solutions. Try an index in the range 0..{}.".format(
                solutionIndex, len(solutionStates), len(solutionStates) - 1
            )
        )


class LibraryLogicReader(SourceReader):
    """Reads a library logic file (3_LibraryLogic/<arch>/*.yaml), dict or list form."""

    description = "library logic"

    @staticmethod
    def matches(data) -> bool:
        # The fallback reader: rawLibraryLogic accepts both remaining shapes.
        return True

    @staticmethod
    def read(data, solutionIndex: int) -> SolutionSource:
        (
            versionString,
            scheduleName,
            architectureName,
            deviceNames,
            problemType,
            solutionStates,
            _,  # indexOrder
            exactLogic,
            _,  # rangeLogic
            _,  # otherFields
        ) = LibraryIO.rawLibraryLogic(data)

        try:
            solution = solutionStates[solutionIndex]
        except (IndexError, KeyError, TypeError):
            raise RuntimeError(
                "Could not find the matching data for the solution index:{} from the library logic file, Try different solution index".format(
                    solutionIndex
                )
            )

        if solution == "":
            raise RuntimeError(
                "Could not find the matching data for the solution index:{} from the library logic file, Try different solution index".format(
                    solutionIndex
                )
            )

        return SolutionSource(
            versionString=versionString,
            scheduleName=scheduleName,
            architectureName=architectureName,
            deviceNames=deviceNames,
            problemType=problemType,
            solution=solution,
            exactLogic=exactLogic,
            origin="library logic, solution {}".format(solutionIndex),
        )


#: Ordered most specific first; LibraryLogicReader matches anything else.
SOURCE_READERS: Tuple[type, ...] = (BenchmarkDataReader, LibraryLogicReader)


def readSource(data, solutionIndex: int) -> SolutionSource:
    """Dispatches the parsed yaml to the first reader that understands it."""
    for reader in SOURCE_READERS:
        if reader.matches(data):
            source = reader.read(data, solutionIndex)
            tPrint(1, "Input format: {}. Using {}.".format(reader.description, source.origin))
            return source

    raise RuntimeError("Unrecognized input file format.")


@dataclass(frozen=True)
class RunSettings:
    """GlobalParameters and LibraryLogic recovered from the config that produced the input.

    Benchmark data records neither, so without this the emitted config would have
    to invent both. Recovered from the build tree first (ClientParameters.ini,
    always written next to the data file) and only then from the tuning config,
    which lives outside the build tree and may be absent, moved or edited since.
    """

    globalParameters: Dict[str, Any] = field(default_factory=dict)
    libraryLogic: Dict[str, Any] = field(default_factory=dict)
    #: Top-level config sections that are neither regenerated nor merged
    #: (Backend and its settings block, for instance), carried through verbatim.
    extraSections: Dict[str, Any] = field(default_factory=dict)
    #: One line per contributing file, most authoritative first.
    sources: List[str] = field(default_factory=list)

    def describe(self) -> str:
        return "; ".join(self.sources) if self.sources else "no source"


def _globalParameterKey(node) -> Optional[str]:
    """Returns X for a globalParameters["X"] subscript, else None."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "globalParameters"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return node.slice.value
    return None


def _iniKeyValuePairs(function):
    """Yields (iniKey, valueExpression) for both ways ClientWriter emits a setting.

    writeClientConfigIni calls param("key", value); dataInitParams returns a list
    of ("key", value) tuples that its caller feeds to the same param().
    """
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "param"
            and len(node.args) == 2
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            yield node.args[0].value, node.args[1]
        elif (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[0], ast.Constant)
            and isinstance(node.elts[0].value, str)
        ):
            yield node.elts[0].value, node.elts[1]


@lru_cache(maxsize=1)
def clientParameterMap() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Derives the ClientParameters.ini reverse mapping from ClientWriter's source.

    ClientWriter is the single definition of how GlobalParameters become ini
    settings, so the reverse is read back out of it rather than restated here:
    a hand-kept copy would drift the moment a setting is added or renamed.

    A setting is only reversed when exactly one GlobalParameters key reaches it.
    Where the value is conditionally rewritten on the way out, more than one key
    (or a non-parameter value) reaches it and the setting is skipped as
    ambiguous, unless CLIENT_PARAMETER_OVERRIDES resolves it deliberately.

    Returns:
        (iniKey -> GlobalParameters key, iniKey -> Enum class name to decode with)
    """
    from Tensile import ClientWriter

    mapping: Dict[str, str] = {}
    decoders: Dict[str, str] = {}

    try:
        tree = ast.parse(inspect.getsource(ClientWriter))
    except (OSError, TypeError, SyntaxError) as e:  # pragma: no cover
        tPrint(1, "Warning: could not scan ClientWriter ({}); relying on overrides.".format(e))
        return dict(CLIENT_PARAMETER_OVERRIDES), {}

    for function in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        # Locals assigned straight from globalParameters, so a setting written as
        # DataInitName(initA).name still resolves back to DataInitTypeA.
        local: Dict[str, set] = {}
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                key = _globalParameterKey(node.value)
                if key is not None:
                    local.setdefault(name, set()).add(key)
                elif name in local:
                    # Reassigned from something else: no longer a clean alias.
                    local[name].add(None)

        for iniKey, expression in _iniKeyValuePairs(function):
            if iniKey in CLIENT_PARAMETER_EXCLUSIONS:
                continue

            keys = set()
            for node in ast.walk(expression):
                key = _globalParameterKey(node)
                if key is not None:
                    keys.add(key)
                elif isinstance(node, ast.Name) and node.id in local:
                    keys |= local[node.id]
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    candidate = getattr(ClientWriter, node.func.id, None)
                    if isinstance(candidate, type) and issubclass(candidate, Enum):
                        decoders[iniKey] = node.func.id

            if len(keys) == 1 and None not in keys:
                mapping[iniKey] = keys.pop()

    for iniKey, globalKey in CLIENT_PARAMETER_OVERRIDES.items():
        if iniKey not in CLIENT_PARAMETER_EXCLUSIONS:
            mapping[iniKey] = globalKey

    for iniKey, functionName in CLIENT_NAME_LOOKUPS.items():
        if iniKey in mapping:
            decoders[iniKey] = functionName

    return mapping, decoders


@lru_cache(maxsize=1)
def problemTypeConfigOnlyKeys() -> Tuple[str, ...]:
    """Problem type keys a config can set that formProblemTypeYamlData cannot emit.

    Derived by scanning ProblemType for keys read out of its config argument and
    subtracting the registry the emitter filters on, so the warning keeps working
    as Tensile gains problem type options.
    """
    # import_module, not "from ... import Problem": the package re-exports a
    # class of that name.
    problemModule = import_module("Tensile.SolutionStructs.Problem")

    try:
        tree = ast.parse(inspect.getsource(problemModule))
    except (OSError, TypeError, SyntaxError):  # pragma: no cover
        return ()

    read = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "config"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            read.add(node.args[0].value)
        elif (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Constant)
            and isinstance(node.left.value, str)
            and any(isinstance(op, ast.In) for op in node.ops)
            and any(isinstance(c, ast.Name) and c.id == "config" for c in node.comparators)
        ):
            read.add(node.left.value)
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "config"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            read.add(node.slice.value)

    return tuple(
        sorted(
            key
            for key in read
            if key not in defaultProblemType and key not in PROBLEM_TYPE_RELOCATED_KEYS
        )
    )


def findClientParameters(inputPath: str) -> Optional[str]:
    """Locates the ClientParameters.ini written for the step that produced the input.

    Benchmark data lives at "<problem>/Data/<step>.yaml" and the client config at
    "<problem>/<step>/source/ClientParameters.ini". Falls back to any single
    ClientParameters.ini under the problem directory.
    """
    dataDir = os.path.dirname(os.path.abspath(inputPath))
    problemDir = os.path.dirname(dataDir)
    step = os.path.splitext(os.path.basename(inputPath))[0]

    candidate = os.path.join(problemDir, step, "source", CLIENT_PARAMETERS_FILENAME)
    if os.path.isfile(candidate):
        return candidate

    found = [
        os.path.join(root, CLIENT_PARAMETERS_FILENAME)
        for root, _, files in os.walk(problemDir)
        if CLIENT_PARAMETERS_FILENAME in files
    ]
    return found[0] if len(found) == 1 else None


@lru_cache(maxsize=None)
def _invertNameLookup(functionName: str) -> Dict[str, int]:
    """Inverts an int -> name helper by probing it over its domain.

    The helpers return None once past their last mode, which bounds the probe
    without restating the mode list here.
    """
    from Tensile import ClientWriter

    function = getattr(ClientWriter, functionName)
    inverse = {}
    for mode in range(64):
        name = function(mode)
        if name is None:
            break
        inverse.setdefault(name, mode)
    return inverse


def _coerceClientValue(globalKey: str, rawValue: str, decoder: Optional[str]) -> Any:
    """Converts an ini string to the type GlobalParameters uses for that key.

    `decoder` names the Enum class ClientWriter used to write the value, so a
    setting emitted as a member name is read back as its numeric value.
    """
    if decoder is not None:
        from Tensile import ClientWriter

        decoderObject = getattr(ClientWriter, decoder)
        if isinstance(decoderObject, type) and issubclass(decoderObject, Enum):
            return decoderObject[rawValue].value
        return _invertNameLookup(decoder)[rawValue]

    default = globalParameterDefaults.get(globalKey)
    if isinstance(default, bool):
        return rawValue.strip().lower() in ("true", "1", "yes")
    if isinstance(default, int):
        return int(rawValue)
    if isinstance(default, float):
        return float(rawValue)

    # Keys with no registry entry (Device is whitelisted separately) carry no
    # type to match, so infer it from the text.
    text = rawValue.strip()
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    for parse in (int, float):
        try:
            return parse(text)
        except ValueError:
            pass
    return rawValue


def loadClientParameters(path: str) -> Dict[str, Any]:
    """Reads ClientParameters.ini back into GlobalParameters form.

    Only keys that round-trip are recovered; anything unrecognized is ignored so
    a newer client writing new keys cannot break the conversion.
    """
    raw: Dict[str, str] = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", ";", "[")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            # Repeated keys (problem-size, strides) are per-problem; keep the first.
            raw.setdefault(key.strip(), value.strip())

    mapping, decoders = clientParameterMap()
    recovered: Dict[str, Any] = {}
    # Walk the file, not the mapping, so the emitted block keeps the client's
    # own ordering regardless of how the mapping was assembled.
    for iniKey in raw:
        globalKey = mapping.get(iniKey)
        if globalKey is None:
            continue
        try:
            recovered[globalKey] = _coerceClientValue(globalKey, raw[iniKey], decoders.get(iniKey))
        except (KeyError, ValueError) as e:
            tPrint(1, "Warning: ignoring {}={!r} ({})".format(iniKey, raw[iniKey], e))
    return recovered


def dropDefaults(params: Dict[str, Any]) -> Dict[str, Any]:
    """Keeps only values that differ from Tensile's own defaults.

    Recovered settings that match the default add nothing to the config, and
    omitting them keeps the emitted block readable.
    """
    return {
        key: value
        for key, value in params.items()
        if key not in globalParameterDefaults or globalParameterDefaults[key] != value
    }


def findRunConfig(inputPath: str) -> Optional[str]:
    """Locates the tuning config that produced a benchmark data file.

    A run writes its outputs to "<dir>/build_<stem>/1_BenchmarkProblems/...", so
    the config is "<dir>/<stem>.yaml". Returns None when no such file exists.
    """
    directory = os.path.dirname(os.path.abspath(inputPath))
    while directory and directory != os.path.dirname(directory):
        name = os.path.basename(directory)
        if name.startswith(BUILD_DIR_PREFIX):
            stem = name[len(BUILD_DIR_PREFIX) :]
            for extension in (".yaml", ".yml"):
                candidate = os.path.join(os.path.dirname(directory), stem + extension)
                if os.path.isfile(candidate):
                    return candidate
            return None
        directory = os.path.dirname(directory)
    return None


#: Regenerated from the input rather than carried over from the run config.
CONFIG_SECTIONS_REGENERATED = ("GlobalParameters", "BenchmarkProblems", "LibraryLogic")

#: Selects the kernel generation backend. Search backends such as Ductile need a
#: non-empty space, which an extracted config -- every parameter pinned to one
#: value -- cannot provide, so the section is dropped and the default backend
#: generates the same kernel.
CONFIG_SECTION_BACKEND = "Backend"


def loadRunConfig(runConfigPath: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Reads GlobalParameters, LibraryLogic and any other sections from a tuning config.

    Sections this tool does not regenerate are passed through untouched: Backend
    selects the kernel generation backend, so dropping it would silently emit a
    config that builds differently from the run being reproduced.
    """
    try:
        config = LibraryIO.readYAML(runConfigPath)
    except Exception as e:  # noqa: BLE001 - a malformed config must not be fatal
        tPrint(1, "Warning: could not read run config {}: {}".format(runConfigPath, e))
        return {}, {}, {}

    if not isinstance(config, dict):
        return {}, {}, {}

    # Drop the backend selection and the settings block it names.
    backend = config.get(CONFIG_SECTION_BACKEND) or {}
    backendName = backend.get("Name", "") if isinstance(backend, dict) else ""
    dropped = {CONFIG_SECTION_BACKEND, backendName, backendName.lower()}
    if backend:
        tPrint(
            1,
            "Not carrying over the {} backend: it searches the space spanned by the "
            "ForkParameters ranges, which an extracted single-solution config does not "
            "have. The default backend generates the same kernel.".format(
                backendName or "configured"
            ),
        )

    extras = {
        key: value
        for key, value in config.items()
        if key not in CONFIG_SECTIONS_REGENERATED and key not in dropped
    }
    return config.get("GlobalParameters") or {}, config.get("LibraryLogic") or {}, extras


def resolveRunSettings(
    inputPath: str, runConfigPath: Optional[str], useRunConfig: bool
) -> Optional[RunSettings]:
    """Recovers the settings of the run that produced the input.

    Layered most authoritative first:

    1. ClientParameters.ini from the build tree - what the run actually executed,
       and always present beside the data file.
    2. The tuning config, when it can be found - supplies the build-side keys the
       client config cannot express (KeepBuildTmp, PreciseKernelTime, UseEffLike)
       and the LibraryLogic block, whose ScheduleName and DeviceNames appear
       nowhere in the build tree.

    Returns None when nothing could be recovered, leaving the caller to fall back
    to built-in defaults.
    """
    if not useRunConfig:
        return None

    globals_: Dict[str, Any] = {}
    libraryLogic: Dict[str, Any] = {}
    sources: List[str] = []

    # Every key the client config expresses is settled here, defaults included,
    # so the run config cannot override a setting the run actually used.
    clientKeys: set = set()
    clientPath = findClientParameters(inputPath)
    if clientPath is not None:
        recovered = loadClientParameters(clientPath)
        clientKeys = set(recovered)
        kept = dropDefaults(recovered)
        if recovered:
            globals_.update(kept)
            sources.append(
                "{} ({} settings, {} at default)".format(
                    clientPath, len(recovered), len(recovered) - len(kept)
                )
            )

    if runConfigPath is None:
        runConfigPath = findRunConfig(inputPath)
    elif not os.path.isfile(runConfigPath):
        raise RuntimeError("Run config not found: {}".format(runConfigPath))

    extraSections: Dict[str, Any] = {}
    if runConfigPath is not None:
        configGlobals, libraryLogic, extraSections = loadRunConfig(runConfigPath)
        # The client config recorded what ran, so it wins on every key it holds.
        extra = {k: v for k, v in configGlobals.items() if k not in clientKeys}
        globals_.update(extra)
        if extra or libraryLogic or extraSections:
            sources.append(
                "{} ({} extra settings{}{})".format(
                    runConfigPath,
                    len(extra),
                    ", LibraryLogic" if libraryLogic else "",
                    ", " + ", ".join(sorted(extraSections)) if extraSections else "",
                )
            )

    if not sources:
        return None

    return RunSettings(
        globalParameters=globals_,
        libraryLogic=libraryLogic,
        extraSections=extraSections,
        sources=sources,
    )


def formLibraryLogic(
    source: "SolutionSource", runSettings: Optional["RunSettings"] = None
) -> dict:
    """Forms the LibraryLogic block, preferring values recorded by the run config."""
    scheduleName = source.scheduleName
    deviceNames = source.deviceNames
    architectureName = source.architectureName

    recorded = runSettings.libraryLogic if runSettings is not None else {}
    if recorded:
        scheduleName = recorded.get("ScheduleName", scheduleName)
        architectureName = recorded.get("ArchitectureName", architectureName)
        deviceNames = recorded.get("DeviceNames", deviceNames)
        tPrint(1, "LibraryLogic recovered from the run config.")
    elif source.hasPlaceholderLibraryLogic:
        tPrint(
            1,
            "Warning: benchmark data records no schedule or device name, so ScheduleName={} "
            "(derived from the architecture) and DeviceNames=[{}] are placeholders. Edit them "
            "before using this config to build a library.".format(
                scheduleName, FALLBACK_DEVICE_NAME
            ),
        )

    # rawLibraryLogic may return the architecture as a dict
    # ({'Architecture': 'gfx950', 'CUCount': 128}); createLibraryLogic and
    # load_logic_gfx_arch expect the plain name.
    if isinstance(architectureName, dict):
        architectureName = architectureName.get("Architecture", architectureName)

    return {
        "ScheduleName": Quoted(scheduleName),
        "DeviceNames": FlowList([Quoted(name) for name in deviceNames]),
        "ArchitectureName": Quoted(architectureName),
    }


def writeToTensileYamlFile(tensileYamlFile: str, tensileYamlData: str) -> Optional[str]:
    ret = None
    try:
        fileDir = os.path.dirname(tensileYamlFile)
        if fileDir:
            os.makedirs(fileDir, exist_ok=True)

        with open(tensileYamlFile, "w") as f:
            yaml.dump(
                tensileYamlData,
                f,
                default_flow_style=False,
                sort_keys=False,
                Dumper=yaml.Dumper,
            )
        tPrint(1, "Config library is written to {}".format(tensileYamlFile))
        ret = tensileYamlFile

    except (OSError, IOError):
        tPrint(
            1,
            "Error: Creating file {} Please provide file name in this format <filename>.yaml.".format(
                tensileYamlFile
            ),
        )
    return ret


def TensileLibLogicToYaml(
    logicFilePath: str,
    solutionIndex: int,
    tensileYamlFile: str,
    skipMI: bool,
    runConfigPath: Optional[str] = None,
    useRunConfig: bool = True,
) -> Optional[str]:
    """Generate a config from a library logic or a benchmark data file.

    Extracts one solution and emits a Tensile config for it. GlobalParameters and
    LibraryLogic are inherited from the config that produced the input when it can
    be found, so the generated config reproduces the original run rather than a
    set of assumed defaults.

    Args:
        logicFilePath: Library logic or benchmark data yaml to extract from.
        solutionIndex: Solution index to extract.
        tensileYamlFile: Config yaml file name. Creates the dir if path is given.
        skipMI: If False ignores the MI instruction.
        runConfigPath: Tuning config to inherit run settings from. Located
            automatically from the input path when omitted.
        useRunConfig: Set False to emit built-in defaults instead of inheriting.

    Returns:
        The generated config file name, otherwise None.

    Raises:
        RuntimeError: If logicFilePath cannot be read.
        RuntimeError: If solutionIndex is not in the logicFilePath.
        RuntimeError: If tensileYamlFile string is empty or name is not valid.

    Example:
        TensileLibLogicToYaml("gfx950_Cijk_Alik_Bljk_BSS_BH_BiasS_HAS_SAV_UserArgs.yaml", 0, "config.yaml", False)
    """

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  TensileLibLogicToYaml Library v{}".format(__version__))
    tPrint(1, "#  Input: {}".format(logicFilePath))
    tPrint(1, "#  Solution Index: {}".format(solutionIndex))
    tPrint(1, "#")
    tPrint(1, HR)
    tPrint(1, "")

    libYaml = LibraryIO.readYAML(logicFilePath)
    if libYaml == "":
        raise RuntimeError(
            "Yaml file data is empty, read yaml file :{} failed".format(logicFilePath)
        )
    if solutionIndex == "":
        raise RuntimeError("At least one solution idx should be provided")

    source = readSource(libYaml, solutionIndex)
    runSettings = resolveRunSettings(logicFilePath, runConfigPath, useRunConfig)

    tensileYamlFileData = {}
    tensileYamlFileData["GlobalParameters"] = setGlobalParams(
        source.versionString, source.problemType, runSettings
    )

    benchmarkProblems = [[]]
    benchmarkProblems[0].append(formProblemTypeYamlData(source.problemType))

    benchmarkProblemsData = formForkParams(source.solution, skipMI)
    benchmarkProblemsData.update(
        formProblemSize(
            source.exactLogic,
            solutionIndex,
            source.problemType,
            source.problemSizes,
            source.biasTypeArgs,
        )
    )
    benchmarkProblems[0].append(benchmarkProblemsData)

    tensileYamlFileData["BenchmarkProblems"] = benchmarkProblems
    tensileYamlFileData["LibraryLogic"] = formLibraryLogic(source, runSettings)

    if runSettings is not None and runSettings.extraSections:
        tensileYamlFileData.update(runSettings.extraSections)

    return writeToTensileYamlFile(tensileYamlFile, tensileYamlFileData)


def parseArgs():
    argParser = argparse.ArgumentParser()
    argHelp = {
        "input": "Library logic file or benchmark data file (1_BenchmarkProblems/.../Data/*_Final.yaml) to be converted to tensile input yaml file.",
        "indices": "Comma-separated list of Solution indices from the input file to extract. Ex: 0,3,4,5",
        "output": "Base Output file name.",
        "skipMI": "Skips the MatrixInstruction field in the tensile yaml file",
        "runConfig": "Tuning config to inherit GlobalParameters and LibraryLogic from. Located automatically from the input path when omitted.",
        "noRunConfig": "Emit built-in defaults instead of inheriting run settings from the originating config.",
    }

    argParser.add_argument(
        "--input",
        "-i",
        action="store",
        type=os.path.realpath,
        required=True,
        default=None,
        help=argHelp["input"],
    )
    argParser.add_argument(
        "--indices",
        "-d",
        action="store",
        type=str,
        required=True,
        default="0",
        help=argHelp["indices"],
    )
    argParser.add_argument(
        "--output",
        "-o",
        action="store",
        type=os.path.realpath,
        required=True,
        default=None,
        help=argHelp["output"],
    )
    argParser.add_argument(
        "--run-config",
        "-c",
        action="store",
        type=os.path.realpath,
        default=None,
        required=False,
        help=argHelp["runConfig"],
    )
    argParser.add_argument(
        "--no-run-config",
        action="store_true",
        default=False,
        required=False,
        help=argHelp["noRunConfig"],
    )
    argParser.add_argument(
        "--skipMI",
        "-s",
        action="store_true",
        default=None,
        help=argHelp["skipMI"],
        required=False,
    )

    return argParser.parse_args()


def main():
    args = parseArgs()
    ids = [int(x.strip()) for x in args.indices.split(",")]
    tensileYamlFiles = []
    for id in ids:
        if len(ids) == 1:
            tensileYamlFile = args.output
        else:
            tensileYamlFile = re.sub(".yaml", f"_{int(id)}.yaml", args.output)
        TensileLibLogicToYaml(
            args.input,
            int(id),
            tensileYamlFile,
            args.skipMI,
            args.run_config,
            not args.no_run_config,
        )
        tensileYamlFiles.append(tensileYamlFile)
    tPrint(1, f"Tensile Files generated: {tensileYamlFiles}")
