################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import re
import collections

from pathlib import Path
from subprocess import run, PIPE
from typing import List, Optional, Set, Tuple, Union, NamedTuple, Dict

from .Types import IsaVersion
from .Utilities import print2

import rocisa

# Translate GPU targets to filter filenames in Tensile_LOGIC directory
architectureMap = {
    "all": "_",
    "gfx000": "none",
    "gfx803": "r9nano",
    "gfx900": "vega10",
    "gfx906": "vega20",
    "gfx906:xnack+": "vega20",
    "gfx906:xnack-": "vega20",
    "gfx908": "arcturus",
    "gfx908:xnack+": "arcturus",
    "gfx908:xnack-": "arcturus",
    "gfx90a": "aldebaran",
    "gfx90a:xnack+": "aldebaran",
    "gfx90a:xnack-": "aldebaran",
    "gfx942": "aquavanjaram",
    "gfx942:xnack+": "aquavanjaram",
    "gfx942:xnack-": "aquavanjaram",
    "gfx950": "gfx950",
    "gfx950:xnack+": "gfx950",
    "gfx950:xnack-": "gfx950",
    "gfx1010": "navi10",
    "gfx1011": "navi12",
    "gfx1012": "navi14",
    "gfx1030": "navi21",
    "gfx1100": "navi31",
    "gfx1101": "navi32",
    "gfx1102": "navi33",
    "gfx1103": "gfx1103",
    "gfx1150": "gfx1150",
    "gfx1151": "gfx1151",
    "gfx1200": "gfx1200",
    "gfx1201": "gfx1201",
}

gfxVariantMap = {
    "gfx906": ["gfx906:xnack+", "gfx906:xnack-"],
    "gfx908": ["gfx908:xnack+", "gfx908:xnack-"],
    "gfx90a": ["gfx90a:xnack+", "gfx90a:xnack-"],
    "gfx942": ["gfx942:xnack+", "gfx942:xnack-"],
    "gfx950": ["gfx950:xnack+", "gfx950:xnack-"],
}

SUPPORTED_ISA = [
    IsaVersion(8, 0, 3),
    IsaVersion(9, 0, 0),
    IsaVersion(9, 0, 6),
    IsaVersion(9, 0, 8),
    IsaVersion(9, 0, 10),
    IsaVersion(9, 4, 2),
    IsaVersion(9, 5, 0),
    IsaVersion(10, 1, 0),
    IsaVersion(10, 1, 1),
    IsaVersion(10, 1, 2),
    IsaVersion(10, 3, 0),
    IsaVersion(11, 0, 0),
    IsaVersion(11, 0, 1),
    IsaVersion(11, 0, 2),
    IsaVersion(11, 0, 3),
    IsaVersion(11, 5, 0),
    IsaVersion(11, 5, 1),
    IsaVersion(12, 0, 0),
    IsaVersion(12, 0, 1),
]

SUPPORTED_ARCH_DEVICE_IDS = {
    "id=74a0": "gfx942",
    "id=74a1": "gfx942",
    "id=74a2": "gfx942",
    "id=74a3": "gfx942",
    "id=74a5": "gfx942",
    "id=74a9": "gfx942",
    "id=75a0": "gfx950",
    "id=75a2": "gfx950",
    "id=75a3": "gfx950",
    # This dictionary should be extended to add device ID-based
    # filtering support for other architectures.
}

SUPPORTED_ARCH_CU_COUNTS = {
    "cu=20": "gfx942",
    "cu=38": "gfx942",
    "cu=64": "gfx942",
    "cu=80": "gfx942",
    "cu=96": "gfx942",
    "cu=228": "gfx942",
    "cu=304": "gfx942",
    # This dictionary should be extended to add CU count-based
    # filtering support for other architectures.
}

ARCH_DEVICE_ID_FALLBACKS = {
    "id=75a2": ["id=75a0"],
    "id=75a3": ["id=75a0"],
}

# Here, `None` refers to an unspecified CU count.
ARCH_CU_COUNT_FALLBACKS = {
    "cu=20": None,
    "cu=38": None,
    "cu=64": None,
    "cu=80": None,
    "cu=96": None,
    "cu=228": None,
    "cu=304": None,
}


def isaToGfx(arch: IsaVersion) -> str:
    """Converts an ISA version to a gfx architecture name.

    Args:
        arch: An object representing the major, minor, and step version of the ISA.

    Returns:
        The name of the GPU architecture (e.g., 'gfx906').
    """
    # Convert last digit to hex because reasons
    name = str(arch[0]) + str(arch[1]) + ("%x" % arch[2])
    return "gfx" + "".join(map(str, name))


SUPPORTED_GFX = [isaToGfx(isa) for isa in SUPPORTED_ISA]


def gfxToIsa(name: str) -> Optional[IsaVersion]:
    """Extracts the ISA version from a given gfx architecture name.

    Args:
        name: The gfx name of the GPU architecture (e.g., 'gfx906').

    Returns:
        An object representing the major, minor, and step version of the ISA.
            Returns None if the name does not match the expected pattern.
    """
    match = re.search(r"gfx([0-9a-fA-F]{3,})", name)
    if not match:
        return None
    ipart = match.group(1)
    step = int(ipart[-1], 16)

    ipart = ipart[:-1]
    minor = int(ipart[-1])

    ipart = ipart[:-1]
    major = int(ipart)
    return IsaVersion(major, minor, step)


def gfxToSwCodename(gfxName: str) -> Optional[str]:
    """Retrieves the common name for a given gfx architecture name.

    Args:
        gfxName: The name of the GPU architecture (e.g., gfx1100).

    Returns:
        The common name of the GPU architecture (e.g., navi31) if found in ``architectureMap``.
            Returns None if the name is not found.
    """
    if gfxName in architectureMap:
        return architectureMap[gfxName]
    else:
        for archKey in architectureMap:
            if gfxName in archKey:
                return architectureMap[archKey]
            return None


def gfxToVariants(gfx: str) -> List[str]:
    """Retrieves the list of variants for a given gfx architecture name.

    Args:
        gfx: The name of the GPU architecture (e.g., 'gfx906').

    Returns:
        List of variants for the GPU architecture.
    """
    return gfxVariantMap.get(gfx, [gfx])


def cliArchsToIsa(cliArchs: str) -> List[IsaVersion]:
    """Maps the requested gfx architectures to ISA numbers.

    Args:
        archs: str of ";" or "_" separated gfx architectures (e.g., gfx1100 or gfx90a;gfx1101).

    Returns:
        List of tuples
    """
    archs = cliArchs.split(";") if ";" in cliArchs else cliArchs.split("_")
    return SUPPORTED_ISA if "all" in archs else [gfxToIsa(''.join(map(str, arch))) for arch in archs]


def _detectGlobalCurrentISA(detectionTool, deviceId: int):
    """
    Returns returncode if detection failure
    """
    process = run([detectionTool], stdout=PIPE)
    archList = []
    for line in process.stdout.decode().split("\n"):
        arch = gfxToIsa(line.strip())
        if arch is not None:
            if arch in SUPPORTED_ISA:
                print(f"# Detected GPU {deviceId} with ISA: " + isaToGfx(arch))
                archList.append(arch)
    if process.returncode:
        print(f"{detectionTool} exited with code {process.returncode}")
    return archList[deviceId] if (len(archList) > 0 and process.returncode == 0) else process.returncode


def detectGlobalCurrentISA(deviceId: int, enumerator: str):
    """Returns the ISA version for a given device.

    Given an integer ID for a device, the ISA version tuple
    of the form (X, Y, Z) is computed using first amdgpu-arch.
    If amdgpu-arch fails, rocm_agent_enumerator is used.

    Args:
        deviceID: an integer indicating the device to inspect.

    Raises:
        Exception if both tools fail to detect ISA.
    """
    result = _detectGlobalCurrentISA(enumerator, deviceId)
    if not isinstance(result, IsaVersion):
        raise Exception("Failed to detect currect ISA")
    return result


class ArchInfo(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[str] = None


class LogicFileError(Exception):
    def __init__(self, message="Expected line is either not present or is malformed"):
        self.message = message
        super().__init__(self.message)


def _extractArchInfo(file: Union[str, Path]) -> ArchInfo:
    """
    Extracts architecture predicate information from a given logic file.

    The file is expected to have the following format:
    - Line 0: Minimum required version (e.g., "- {MinimumRequiredVersion: 4.33.0}")
    - Line 1: Code name of the architecture (e.g., "- aquavanjaram")
    - Line 2: GFX name of the architecture or a map with variant details (e.g., "- gfx950" or "- {Architecture: gfx950, CUCount: 256}")
    - Line 3: Device IDs (e.g., "- [Device 1234, Device 5678]")

    Args:
        file: Path to a logic file.
    Returns:
        ArchInfo: An object containing the extracted architecture predicates.
    Raises:
        LogicFileError: If the file does not match the expected format.
    """

    def l0(line: str):
        if not re.match(r"- \{MinimumRequiredVersion", line):
            raise LogicFileError(
                f"Expected minimum required version:\n  line: {line}  file: {file}"
            )

    def l1(line: str):
        return line[2:].strip()

    def l2(line: str):
        match1 = re.match(r"- \{Architecture: (\w+), CUCount: (\d+)\}", line)
        match2 = re.match(r"- gfx(\w+)", line)
        if match1:
            architecture, cu_count = match1.groups()
            return architecture, f"cu={cu_count}"
        elif match2:
            return line[2:].strip(), None
        else:
            raise LogicFileError(
                f"Expected architecture and CU count, or only an archiecture: line: {line}"
            )

    def l3(line: str):
        if re.match(r"- \[Device", line):
            devIds = re.findall(r"Device (\w+)", line)
            return set(f"id={id}" for id in devIds)
        else:
            raise LogicFileError(f"No device IDs found: line: {line}")

    with open(file, "r") as f:
        l0(f.readline())
        name = l1(f.readline())
        gfx, cu = l2(f.readline())
        deviceIds = l3(f.readline())

    try:
        for id in deviceIds:
            _verifyPredicate(id, gfx)
    except ValueError as e:
        raise LogicFileError(f"Invalid device ID found while parsing {file}: {e}")

    return ArchInfo(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def _verifyPredicate(predicateSpec: str, gfx: str) -> str:
    """
    Verifies that a predicate specification is valid.

    Args:
        predicateSpec: A string representing a predicate specification.
        gfx: GFX architecture to validate device ID against.

    Returns:
        The validated predicate specification.
    Raises:
        ValueError: If the predicate specification is invalid or if device ID doesn't match GFX architecture.
    """
    msgPrefix = f"Invalid predicate: {predicateSpec}"
    key, _, val = predicateSpec.partition("=")
    if key == "id":
        if predicateSpec not in SUPPORTED_ARCH_DEVICE_IDS:
            raise ValueError(f"{msgPrefix}: device ID not supported")
        if gfx and SUPPORTED_ARCH_DEVICE_IDS[predicateSpec] != gfx:
            raise ValueError(f"{msgPrefix}: device ID is not associated with {gfx}")
    elif key == "cu":
        if predicateSpec not in SUPPORTED_ARCH_CU_COUNTS:
            raise ValueError(f"{msgPrefix}: CU count not supported")
        if gfx and SUPPORTED_ARCH_CU_COUNTS[predicateSpec] != gfx:
            raise ValueError(f"{msgPrefix}: CU count is not associated with {gfx}")
    else:
        raise ValueError(f"{msgPrefix}: only device ID and CU count-based predicates are currently supported")
    return predicateSpec


def splitArchsFromPredicates(archSpecs: List[str]) -> Tuple[List[str], Optional[Dict[str, List[str]]]]:
    """
    Splits a list of architecture specifications into architectures and their predicates.

    Example inputs:
        ["gfx942"]  # No predicates
        ["gfx942[id=74a0,id=74a1]"]  # With device IDs
        ["gfx942[cu=80,cu=96]"]  # With CU counts
        ["gfx942[id=74a0,cu=80]"]  # With both

    Args:
        archSpecs: List of architecture specifications, optionally with predicates in square brackets

    Returns:
        Tuple of:
        - List of architecture names
        - Dictionary mapping architectures to their predicates (or None if no predicates)
    """
    # Match predicates in square brackets, e.g., [id=74a0,cu=80]
    pattern = re.compile(r"\[(.*?)\]")

    architectures = set()
    predicateMap = collections.defaultdict(list)

    for spec in archSpecs:
        spec = spec.strip()
        arch = spec  # Default to full spec if no predicates

        match = re.search(pattern, spec)
        if match:
            arch = spec[:match.start()].strip()
            predicates = [p.strip().lower() for p in match.group(1).split(",")]
            predicateMap[arch].extend(_verifyPredicate(p, arch) for p in predicates)

        if arch not in architectureMap:
            raise ValueError(f"Architecture {spec} not supported")

        architectures.add(arch)

    return list(architectures), predicateMap or None


def _addVariantMap(
    gfxPredicateMap: Dict[str, Set[Tuple[Path, str]]], spec: str, path: Path, fname: str
) -> bool:
    """
    Adds a logic file to a predicate map.

    Args:
        gfxPredicateMap: Nested dict mapping architectures to their predicate sets
        spec: Predicate specification
        path: Path to the logic file
        fname: Filename of the logic file
    Returns:
        True if the logic file was added to the predicate map, False otherwise
    """
    if fname not in {x for _, x in gfxPredicateMap[spec]}:
        gfxPredicateMap[spec].add((path, fname))
        return True
    return False


def _populateVariantMap(
    predicateMap: Dict[str, Dict[str, Set[Tuple[Path, str]]]],
    targetLogicFile: Path,
    fallbackKey: str,
):
    """
    Populates a predicate map with logic files, handling both exact matches and fallbacks.

    For each logic file:
    1. First tries to match against specific predicates (device IDs, CU counts)
    2. If matched to any specific predicate, removes from fallbacks
    3. If no specific matches, tries to add to fallbacks based on fallback rules

    Args:
        predicateMap: Nested dict mapping architectures to their predicate sets
        targetLogicFile: Logic file to process
        fallbackKey: Key used to store fallback matches
    """
    file = Path(targetLogicFile)
    path, fname = file.parent, file.name

    archinfo = _extractArchInfo(file)
    if archinfo.Gfx not in predicateMap:
        return

    gfxPredicateMap = predicateMap[archinfo.Gfx]
    requestedDevIds = {x for x in gfxPredicateMap if x.startswith("id=")}
    requestedCUs = {x for x in gfxPredicateMap if x.startswith("cu=")}

    fallbackDevIds = {
        fallbackId
        for v in requestedDevIds
        if v in ARCH_DEVICE_ID_FALLBACKS
        for fallbackId in ARCH_DEVICE_ID_FALLBACKS[v]
    }
    fallbackCUs = {ARCH_CU_COUNT_FALLBACKS[v] for v in requestedCUs if v in ARCH_CU_COUNT_FALLBACKS}

    isCuFallback = not requestedCUs or archinfo.CUCount in fallbackCUs
    isDevIdFallback = not requestedDevIds or (
        archinfo.DeviceIds and any(fallbackId in archinfo.DeviceIds for fallbackId in fallbackDevIds)
    )

    if isCuFallback and isDevIdFallback:
        # If the file name is not already in a requested predicate, then add it to the fallback set
        if all(
            fname not in {nm for _, nm in gfxPredicateMap[spec]}
            for spec in gfxPredicateMap
            if spec != fallbackKey
        ):
            gfxPredicateMap[fallbackKey].add((path, fname))
    else:
        removeFallbacks = []
        for spec in gfxPredicateMap:
            if spec != fallbackKey:  # Don't try to add to fallback set here
                if "id" in spec and archinfo.DeviceIds:
                    removeFallbacks.extend(
                        _addVariantMap(gfxPredicateMap, spec, path, fname)
                        for id in archinfo.DeviceIds
                        if id == spec
                    )
                if "cu" in spec and archinfo.CUCount:
                    removeFallbacks.append(
                        _addVariantMap(gfxPredicateMap, spec, path, fname)
                        if archinfo.CUCount == spec
                        else False
                    )

        if removeFallbacks and any(removeFallbacks):
            gfxPredicateMap[fallbackKey] = {
                x for x in gfxPredicateMap[fallbackKey] if x[1] != fname
            }


def filterLogicFilesByPredicates(
    logicFiles: List[str], variants: Dict[str, Dict[str, Set[Tuple[Path, str]]]]
) -> List[str]:
    """
    Filters logic files based on the requested predicates.

    Args:
        logicFiles: List of logic file paths
        variants: Dictionary mapping architectures to their predicate sets

    Returns:
        List of logic file paths that match the requested predicates
    """
    fallbackKey = "fallback"
    # A `spec` here is a variant specification passed via the command line, e.g., "cu=64"
    # This is how the code differentiates variants of the same gfx, as well as "fallback" files
    variantMap = {gfx: {spec: set() for spec in specs} for gfx, specs in variants.items()}
    for file in variantMap.values():
        file[fallbackKey] = set()

    for logicFile in logicFiles:
        _populateVariantMap(variantMap, Path(logicFile), fallbackKey)

    return [
        str(p / file)
        for gfxPredicateMap in variantMap.values()
        for files in gfxPredicateMap.values()
        for p, file in files
    ]
