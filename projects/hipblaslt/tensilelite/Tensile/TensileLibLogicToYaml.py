################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
from Tensile.Common.GlobalParameters import globalParameters, defaultBenchmarkCommonParameters
from Tensile.Common.Constants import HR
from Tensile.Common.ValidParameters import validParameters
from Tensile.SolutionStructs.Problem import _defaultProblemType as defaultProblemType

import argparse
import ast
import functools
import os
import sys
import re
import yaml
from io import StringIO
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class Quoted(str):
    """Marker type for YAML double-quoted scalars."""

    pass


def quotedPresenter(dumper: Any, data: "Quoted") -> Any:
    """YAML representer for :class:`Quoted` (double-quoted string style).

    Args:
        dumper: Active PyYAML dumper instance.
        data: String value to emit as a double-quoted scalar.

    Returns:
        A YAML scalar node for *data*.

    Raises:
        None.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


class FlowList(list):
    """Marker list type for YAML flow-style sequence output."""

    pass


def makeFlow(value: Any) -> Any:
    """Wrap lists as :class:`FlowList` for flow-style YAML emission; pass through others.

    Args:
        value: A list to wrap, or any other value returned unchanged.

    Returns:
        ``FlowList(value)`` if *value* is a ``list``, else *value*.

    Raises:
        None.
    """
    if isinstance(value, list):
        return FlowList(value)
    return value


def flowSeq(dumper: Any, value: FlowList) -> Any:
    """YAML representer for :class:`FlowList` (flow-style sequence).

    Args:
        dumper: Active PyYAML dumper instance.
        value: List values to emit in flow style ``[...]``.

    Returns:
        A YAML sequence node with ``flow_style=True``.

    Raises:
        None.
    """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


def representNone(dumper: Any, _: Any) -> Any:
    """YAML representer mapping Python ``None`` to an empty string scalar.

    Args:
        dumper: Active PyYAML dumper instance.
        _: Ignored (PyYAML passes the serialized ``None``).

    Returns:
        An empty string scalar node.

    Raises:
        None.
    """
    return dumper.represent_scalar("tag:yaml.org,2002:null", "")


yaml.add_representer(Quoted, quotedPresenter)
yaml.add_representer(FlowList, flowSeq)
yaml.add_representer(type(None), representNone)


def _formatCompactRange(
    rng: Any, start_elements: int = 2, end_elements: int = 2
) -> str:
    """Return a short string for a valid-values sequence (same idea as TuningDriver GEKO).

    Long lists are shown as first *start_elements* and last *end_elements* entries
    with an ellipsis between.

    Args:
        rng: Iterable of valid values, typically a list from ``validParameters``.
        start_elements: Number of leading entries to include before ``...``.
        end_elements: Number of trailing entries to include after ``...``.

    Returns:
        A bracketed string representation of *rng*, or ``str(rng)`` if not a list.

    Raises:
        None.
    """
    if not isinstance(rng, list):
        return str(rng)
    if len(rng) <= start_elements + end_elements:
        return str(rng)
    start = ", ".join(map(str, rng[:start_elements]))
    end = ", ".join(map(str, rng[-end_elements:]))
    return f"[{start}, ..., {end}]"


def buildForkParameterCommentMetadata() -> Dict[str, str]:
    """Build trailing comment text per fork parameter for inline documentation.

    Merges ``defaultBenchmarkCommonParameters`` the same way as TuningDriver's
    ``load_tensile_metadata``, then intersects keys with ``validParameters``.

    Returns:
        Mapping from fork parameter name to a suffix string starting with
        ``' # Default Value: ... # Range: ...'`` suitable to append to a YAML line.

    Raises:
        None.
    """
    defaultsMerged: Dict[str, Any] = {}
    for param_dict in defaultBenchmarkCommonParameters:
        defaultsMerged.update(param_dict)
    meta: Dict[str, str] = {}
    for name in set(validParameters.keys()) & set(defaultsMerged.keys()):
        default_list = defaultsMerged[name]
        range_str = _formatCompactRange(validParameters[name])
        meta[name] = f" # Default Value: {default_list} # Range: {range_str}"
    return meta


_FORK_PARAM_LINE_RE = re.compile(r"^(?P<indent>    )- (?P<key>[A-Za-z0-9_]+)(?P<rest>:.*)$")

# First row of a Groups entry (yaml.dump uses ``- - MatrixInstruction:``).
_GROUP_MATRIX_INSTRUCTION_RE = re.compile(
    r"^(?P<prefix>\s+- - MatrixInstruction)(?P<rest>:.*)$"
)
# Continuation keys in the same group block (indented mapping under ``- -``).
_GROUP_WORKGROUP_RE = re.compile(r"^(?P<prefix>\s+WorkGroup)(?P<rest>:.*)$")
_GROUP_MIARCH_VGPR_RE = re.compile(r"^(?P<prefix>\s+MIArchVgpr)(?P<rest>:.*)$")


def parseMatrixInstructionListFromColonRest(rest: str) -> Optional[List[int]]:
    """Parse the YAML list after ``MatrixInstruction:`` from the ``: [...]`` tail.

    Args:
        rest: Substring starting with ``':'`` then optional whitespace and a
            bracketed list (as emitted by ``yaml.dump`` for ``FlowList``).

    Returns:
        The parsed list of integers, or ``None`` if parsing fails or values are
        not all integers.

    Raises:
        None.
    """
    rest = rest.lstrip()
    if not rest.startswith(":"):
        return None
    tail = rest[1:].strip()
    if not tail.startswith("["):
        return None
    depth = 0
    for i, ch in enumerate(tail):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                try:
                    v = ast.literal_eval(tail[: i + 1])
                except (ValueError, SyntaxError):
                    return None
                if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                    return v
                return None
    return None


def formatMatrixInstructionCmsComment(
    mi: Sequence[int], wavefrontSize: int = 64
) -> Optional[str]:
    """Build the GEKO-style ``#CMS — MT …`` suffix for a 9-deep MatrixInstruction.

    Uses the same MFMA layout math as ``MIDesign.calculate_mfma_parameters`` in
    TuningDriver (``geko/.../mi_designer.py``).

    Args:
        mi: MatrixInstruction tuple (M, N, K, B, MIBlockM, WaveTileM, WaveTileN,
            WaveM, WaveN).
        wavefrontSize: Wavefront size in threads (default 64).

    Returns:
        A string starting with ``' #CMS — MT …'``, or ``None`` if *mi* has fewer
        than nine integer entries.

    Raises:
        None.
    """
    if len(mi) < 9:
        return None
    wave = (mi[7], mi[8])
    miBlockM = mi[4]
    waveTileM, waveTileN = mi[5], mi[6]
    matrixInstM = mi[0] * miBlockM
    mt0 = matrixInstM * waveTileM * wave[0]
    matrixInstN = mi[1] / miBlockM * mi[3]
    mt1 = int(matrixInstN * waveTileN * wave[1])
    tt0 = waveTileM
    tt1 = waveTileN * mi[1]
    wg0 = matrixInstM * wave[0]
    wg1 = int(wave[0] * wave[1] * wavefrontSize / wg0)
    return (
        f" #CMS — MT {mt0}x{mt1} - TT {tt0}x{tt1} - WG {wg0}x{wg1} - MIBlockM {miBlockM}"
    )


def injectForkParameterInlineComments(
    yamlText: str, commentByKey: Optional[Dict[str, str]] = None
) -> str:
    """Append inline comments to ForkParameters and key lines under ``Groups``.

    Fork list entries get ``# Default Value`` / ``# Range`` from metadata.
    Under ``Groups``, ``MatrixInstruction`` uses a GEKO-style ``#CMS — MT …``
    suffix when the instruction has nine components (same layout math as
    TuningDriver ``MIDesign.calculate_mfma_parameters``); shorter tuples fall
    back to default/range metadata. ``WorkGroup`` and ``MIArchVgpr`` use
    metadata only. Lines already containing ``CMS —`` or ``Default Value:``
    are left unchanged (idempotent).

    Args:
        yamlText: Full document text produced by ``yaml.dump``.
        commentByKey: Optional pre-built metadata; defaults to
            :func:`buildForkParameterCommentMetadata`.

    Returns:
        Text with fork-parameter and group MI lines annotated where applicable.

    Raises:
        None.
    """
    if commentByKey is None:
        commentByKey = buildForkParameterCommentMetadata()
    lines = yamlText.splitlines(keepends=True)
    out: List[str] = []
    inForkBlock = False
    inGroupsContent = False
    for line in lines:
        if not inForkBlock and line.startswith("    ForkParameters:"):
            inForkBlock = True
            inGroupsContent = False
            out.append(line)
            continue
        if inForkBlock and (
            line.startswith("    BenchmarkJoinParameters:")
            or line.startswith("    BenchmarkFinalParameters:")
        ):
            inForkBlock = False
            inGroupsContent = False
            out.append(line)
            continue
        if inForkBlock:
            stripped = line.rstrip("\n")
            if stripped.startswith("    - Groups:"):
                inGroupsContent = True
                out.append(line)
                continue
            if inGroupsContent:
                if "Default Value:" not in stripped and "CMS —" not in stripped:
                    gm = _GROUP_MATRIX_INSTRUCTION_RE.match(stripped)
                    if gm:
                        restClean = gm.group("rest").split("#", 1)[0].rstrip()
                        prefix = gm.group("prefix")
                        miVals = parseMatrixInstructionListFromColonRest(
                            restClean
                        )
                        cmsSuffix = None
                        if miVals is not None and len(miVals) >= 9:
                            cmsSuffix = formatMatrixInstructionCmsComment(miVals)
                        if cmsSuffix:
                            line = prefix + restClean + cmsSuffix + "\n"
                        elif "MatrixInstruction" in commentByKey:
                            line = (
                                prefix
                                + restClean
                                + commentByKey["MatrixInstruction"]
                                + "\n"
                            )
                    else:
                        gw = _GROUP_WORKGROUP_RE.match(stripped)
                        if gw and "WorkGroup" in commentByKey:
                            line = (
                                gw.group("prefix")
                                + gw.group("rest")
                                + commentByKey["WorkGroup"]
                                + "\n"
                            )
                        else:
                            gmv = _GROUP_MIARCH_VGPR_RE.match(stripped)
                            if gmv and "MIArchVgpr" in commentByKey:
                                line = (
                                    gmv.group("prefix")
                                    + gmv.group("rest")
                                    + commentByKey["MIArchVgpr"]
                                    + "\n"
                                )
                out.append(line)
                continue
            m = _FORK_PARAM_LINE_RE.match(stripped)
            if (
                m
                and m.group("key") != "Groups"
                and m.group("key") in commentByKey
                and "Default Value:" not in stripped
            ):
                suffix = commentByKey[m.group("key")]
                line = stripped + suffix + "\n"
        out.append(line)
    return "".join(out)


def tPrint(verbosity: int, arg: Any) -> None:
    """Print *arg* to stdout when the client log level is high enough.

    Args:
        verbosity: Minimum ``ClientLogLevel`` at which printing occurs.
        arg: Object to print (passed to ``print``).

    Returns:
        None.

    Raises:
        None.
    """
    if globalParameters["ClientLogLevel"] >= verbosity:
        print(arg)
        sys.stdout.flush()


def setGlobalParams(versionString: dict, problemTypeState: dict) -> dict:
    """Build ``GlobalParameters`` for a Tensile config from library logic metadata.

    Args:
        versionString: Mapping containing ``MinimumRequiredVersion``.
        problemTypeState: Problem-type dict from library logic (uses ``DataType``).

    Returns:
        Global-parameter dict suitable for the output YAML ``GlobalParameters`` key.

    Raises:
        None.
    """
    res = {}
    res["MinimumRequiredVersion"] = versionString["MinimumRequiredVersion"]
    res["SleepPercent"] = 0
    res["KernelTime"] = True
    res["NumElementsToValidate"] = 0
    res["DataInitTypeBeta"] = 1
    res["DataInitTypeAlpha"] = 1
    res["DataInitTypeA"] = 12 if problemTypeState["DataType"] != "I8" else 3
    res["DataInitTypeB"] = 13 if problemTypeState["DataType"] != "I8" else 3
    res["DataInitTypeC"] = 12 if problemTypeState["DataType"] != "I8" else 3
    res["DataInitTypeD"] = 12 if problemTypeState["DataType"] != "I8" else 3
    res["PreciseKernelTime"] = 0
    res["Device"] = 0
    res["SkipSlowSolutionRatio"] = 0
    res["KeepBuildTmp"] = False
    return res


def formProblemTypeYamlData(problemTypeState: dict) -> dict:
    """Select problem-type fields for ``BenchmarkProblems`` from full library problem type.

    Args:
        problemTypeState: Full problem-type mapping from library logic.

    Returns:
        Subset dict for the first benchmark-problem block.

    Raises:
        RuntimeError: If *problemTypeState* is empty.
    """
    if len(problemTypeState) == 0:
        raise RuntimeError(
            "Length of problem Type Parameters is empty!!, Please re-check the library logic file !"
        )

    data = {}
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
        if problemTypeKey in defaultProblemType:
            if problemTypeValue != defaultProblemType[problemTypeKey]:
                data[problemTypeKey] = makeFlow(problemTypeValue)
                continue

    return data


def formGroups(groupRow: dict) -> dict:
    """Wrap a single solution group row under ``ForkParameters: - Groups:``.

    Args:
        groupRow: Mapping (e.g. ``MatrixInstruction``, ``WorkGroup``, ``MIArchVgpr``).
            An empty dict produces a single empty mapping under ``Groups``.

    Returns:
        Dict with key ``Groups`` whose value is ``[[ groupRow ]]`` (list of lists).

    Raises:
        None.
    """
    data = {}
    data["Groups"] = [[]]
    group = {}
    for forkKey, forkValue in groupRow.items():
        group[forkKey] = forkValue
    data["Groups"][0].append(group)
    return data


def form9BitMIInst(currentSolutionState: dict) -> dict:
    """Build the ``MatrixInstruction`` / ``WorkGroup`` / ``MIArchVgpr`` group row from MI fields.

    Args:
        currentSolutionState: Solution dict containing ``MIBlock``, ``MIWaveTile``,
            ``MIWaveGroup``, ``WorkGroup``, and ``MIArchVgpr``.

    Returns:
        Mapping suitable for :func:`formGroups`.

    Raises:
        RuntimeError: If any of ``MIBlock``, ``MIWaveTile``, or ``MIWaveGroup`` is empty.
    """
    miBlock = currentSolutionState["MIBlock"]
    miWaveTile = currentSolutionState["MIWaveTile"]
    miWaveGroup = currentSolutionState["MIWaveGroup"]

    if len(miBlock) == 0 or len(miWaveTile) == 0 or len(miWaveGroup) == 0:
        raise RuntimeError(
            "Length of MIBlock:{0}, MIWave Tile:{1},MIWaveGroup:{2} cannot be empty".format(
                len(miBlock), len(miWaveTile), len(miWaveGroup)
            )
        )

    miBlockPrefix = miBlock[0:5]

    miInstruction9 = miBlockPrefix + miWaveTile + miWaveGroup

    groups = {}
    groups["MatrixInstruction"] = FlowList(miInstruction9)
    groups["WorkGroup"] = FlowList(currentSolutionState["WorkGroup"])
    groups["MIArchVgpr"] = currentSolutionState["MIArchVgpr"]

    return groups


def formForkParams(
    currentIndexSolution: dict, skipMI: Optional[bool]
) -> dict:
    """Build the second benchmark-problem block (fork parameters and ``Groups``).

    Args:
        currentIndexSolution: One entry from ``allSolutionStates`` in library logic.
        skipMI: If true, omit the MI-derived ``Groups`` row (empty group mapping).
            ``None`` (CLI default when ``--skipMI`` is absent) is treated like false.

    Returns:
        Dict with ``InitialSolutionParameters``, ``BenchmarkCommonParameters``,
        and ``ForkParameters``.

    Raises:
        None.
    """
    data = {}
    data["InitialSolutionParameters"] = None
    kernelLang = {}
    kernelLang["KernelLanguage"] = FlowList(["Assembly"])
    data["BenchmarkCommonParameters"] = [kernelLang]

    forkData = []
    for forkKey, forkValue in currentIndexSolution.items():
        entry = {}
        if forkKey in ["MatrixInstruction", "WorkGroup"]:
            continue
        index = next(
            (
                i
                for i, d in enumerate(defaultBenchmarkCommonParameters)
                if forkKey in d
            ),
            None,
        )
        if index is not None:
            forkValue = [forkValue]
            if forkValue != defaultBenchmarkCommonParameters[index][forkKey]:
                entry[forkKey] = FlowList(forkValue)
                forkData.append(entry)

    isMatrixInsEnabled = False
    if "EnableMatrixInstruction" in currentIndexSolution:
        enableMi = currentIndexSolution["EnableMatrixInstruction"]
        isMatrixInsEnabled = enableMi
        if enableMi and currentIndexSolution["MatrixInstruction"]:
            isMatrixInsEnabled = True
        else:
            tPrint(
                1,
                "Matrix instruction is disabled skipping the matrix instruction parameter ..",
            )

    if skipMI is not True and isMatrixInsEnabled:
        groupRow = form9BitMIInst(currentIndexSolution)
    else:
        groupRow = {}

    forkData.append(formGroups(groupRow))

    data["ForkParameters"] = forkData

    return data


def formProblemSize(
    exactLogic: Optional[list[Tuple[list, list]]],
    solutionIndex: int,
    problemTypeStat: dict,
) -> dict:
    """Build ``BenchmarkJoinParameters`` / ``BenchmarkFinalParameters`` for one solution.

    Args:
        exactLogic: List of ``(size, mapping)`` from library logic, or ``None`` for Origami.
        solutionIndex: Solution index to match against ``mapping[0]``.
        problemTypeStat: Problem-type dict (must include ``BiasDataTypeList``).

    Returns:
        Dict with ``BenchmarkJoinParameters`` and ``BenchmarkFinalParameters`` keys.

    Raises:
        None.
    """
    data = {}
    data["BenchmarkJoinParameters"] = None
    data["BenchmarkFinalParameters"] = []

    temp = {}
    if exactLogic is None:
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
    biasTypeArgs = problemTypeStat["BiasDataTypeList"]
    temp["BiasTypeArgs"] = FlowList(biasTypeArgs)

    data["BenchmarkFinalParameters"].append(temp)

    return data


def formLibraryLogic(
    scheduleName: str, deviceNames: list, architectureName: str
) -> dict:
    """Build the top-level ``LibraryLogic`` block for the output config.

    Args:
        scheduleName: Schedule name from library logic.
        deviceNames: Non-empty device name list (first entry is emitted).
        architectureName: Architecture string from library logic.

    Returns:
        Dict with ``ScheduleName``, ``DeviceNames``, and ``ArchitectureName``.

    Raises:
        None.
    """
    data = {}
    data["ScheduleName"] = Quoted(scheduleName)
    data["DeviceNames"] = FlowList([Quoted(deviceNames[0])])
    data["ArchitectureName"] = Quoted(architectureName)

    return data


def writeToTensileYamlFile(
    tensileYamlFile: str, tensileYamlData: dict
) -> Optional[str]:
    """Write Tensile YAML to disk, with inline fork-parameter documentation.

    Args:
        tensileYamlFile: Destination path.
        tensileYamlData: Nested dict for the tuning config (GlobalParameters,
            BenchmarkProblems, LibraryLogic).

    Returns:
        The output path on success, or ``None`` on I/O error.

    Raises:
        None.
    """
    ret = None
    try:
        fileDir = os.path.dirname(tensileYamlFile)
        if fileDir:
            os.makedirs(fileDir, exist_ok=True)

        buf = StringIO()
        yaml.dump(
            tensileYamlData,
            buf,
            default_flow_style=False,
            sort_keys=False,
            Dumper=yaml.Dumper,
        )
        body = injectForkParameterInlineComments(buf.getvalue())

        with open(tensileYamlFile, "w") as f:
            f.write(body)
        tPrint(1, "Config library is written to {}".format(tensileYamlFile))
        ret = tensileYamlFile

    except OSError:
        tPrint(
            1,
            "Error: Creating file {} Please provide file name in this format <filename>.yaml.".format(
                tensileYamlFile
            ),
        )
    return ret


def TensileLibLogicToYaml(
    logicFilePath: str,
    solutionIndex: Union[int, str],
    tensileYamlFile: str,
    skipMI: Optional[bool],
) -> Optional[str]:
    """Generate a Tensile tuning config YAML from one row of a library logic file.

    Args:
        logicFilePath: Path to the library logic YAML (or msgpack) file.
        solutionIndex: Solution index into ``allSolutionStates`` (may be ``""`` to
            trigger validation errors for empty selection).
        tensileYamlFile: Output path; parent directories are created when needed.
        skipMI: When true, omit the MI-derived ``Groups`` content. When ``None``
            (CLI default if ``--skipMI`` is omitted), matrix instructions are not skipped.

    Returns:
        Output file path on success, or ``None`` if writing the file failed.

    Raises:
        RuntimeError: If the library file yields empty data, ``solutionIndex`` is
            ``""``, or the selected solution entry is empty.
    """

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  TensileLibLogicToYaml Library v{}".format(__version__))

    tPrint(1, "#  Library Logic: {}".format(logicFilePath))
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

    fields = LibraryIO.rawLibraryLogic(libYaml)
    (
        versionString,
        scheduleName,
        architectureName,
        deviceNames,
        problemTypeState,
        allSolutionStates,
        _,  # indexOrder
        exactLogic,
        _,  # rangeLogic
        _,  # otherFields
    ) = fields

    currentIndexSolution = allSolutionStates[solutionIndex]

    if currentIndexSolution == "":
        raise RuntimeError(
            "Could not find the matching data for the solution index:{} from the library logic file, Try different solution index".format(
                solutionIndex
            )
        )

    tensileYamlFileData = {}

    tensileYamlFileData["GlobalParameters"] = setGlobalParams(
        versionString, problemTypeState
    )
    benchmarkProblems = [[]]
    benchmarkProblemsData = formProblemTypeYamlData(problemTypeState)
    benchmarkProblems[0].append(benchmarkProblemsData)

    benchmarkProblemsData = formForkParams(currentIndexSolution, skipMI)
    benchmarkProblemsData.update(
        formProblemSize(
            exactLogic,
            solutionIndex,
            problemTypeState,
        )
    )
    benchmarkProblems[0].append(benchmarkProblemsData)

    tensileYamlFileData["BenchmarkProblems"] = benchmarkProblems

    problemSizeData = formLibraryLogic(scheduleName, deviceNames, architectureName)

    tensileYamlFileData["LibraryLogic"] = problemSizeData

    return writeToTensileYamlFile(tensileYamlFile, tensileYamlFileData)


def parseArgs() -> argparse.Namespace:
    """Parse ``TensileLibLogicToYaml`` CLI arguments.

    Args:
        None (reads ``sys.argv``).

    Returns:
        Parsed ``argparse.Namespace`` (``--skipMI`` is ``None`` unless the flag is set).

    Raises:
        SystemExit: On argparse validation errors (e.g. missing required options).
    """
    argParser = argparse.ArgumentParser()
    argHelp = {
        "input": "Library logic file to be converted to tensile input yaml file.",
        "indices": "Comma-separated list of Solution indices from library logic File to extract. Ex: 0,3,4,5",
        "output": "Base Output file name.",
        "skipMI": (
            "Omit MI-derived Groups content. Omit this flag for normal behavior; "
            "present means skip (store_true with default None)."
        ),
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
        "--skipMI",
        "-s",
        action="store_true",
        default=None,
        help=argHelp["skipMI"],
        required=False,
    )

    return argParser.parse_args()


def main() -> None:
    """Run the ``TensileLibLogicToYaml`` command-line interface.

    Args:
        None (reads ``sys.argv``).

    Returns:
        None.

    Raises:
        SystemExit: Propagated from :func:`parseArgs` on invalid CLI input.
    """
    args = parseArgs()
    solutionIds = [int(x.strip()) for x in args.indices.split(",")]
    tensileYamlFiles = []
    for sid in solutionIds:
        if len(solutionIds) == 1:
            outPath = args.output
        else:
            outPath = re.sub(".yaml", f"_{int(sid)}.yaml", args.output)
        TensileLibLogicToYaml(args.input, int(sid), outPath, args.skipMI)
        tensileYamlFiles.append(outPath)
    tPrint(1, f"Tensile Files generated: {tensileYamlFiles}")
