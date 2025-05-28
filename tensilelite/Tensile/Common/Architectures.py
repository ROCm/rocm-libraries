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
from pathlib import Path
from subprocess import run, PIPE
from typing import List, Optional, Set, Tuple, Union, NamedTuple

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
    "id=75a0": "gfx950",
    "id=75a2": "gfx950",
    "id=75a3": "gfx950",
}

ARCH_DEVICE_ID_FALLBACKS = {
    "id=75a2": "id=75a0",
    "id=75a3": "id=75a0",
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

def isaToGfx(arch: IsaVersion) -> str:
    return rocisa.isaToGfx(arch)


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
    - Line 1: Minimum required version (e.g., "- {MinimumRequiredVersion: 4.33.0}")
    - Line 2: Code name of the architecture (e.g., "- aquavanjaram")
    - Line 3: GFX name of the architecture or a map with variant details (e.g., "- gfx950" or "- {Architecture: gfx950, CUCount: 256}")
    - Line 4: Device IDs (e.g., "- [Device 1234, Device 5678]")

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
        emulationIds = {"0049", "0050", "0051", "0052", "0054", "0062"}
        if re.match(r"- \[Device", line):
            devIds = re.findall(r"Device (\w+)", line)
            return set(f"id={id}" for id in devIds)
        if re.match(r"-\[alldevices", line.lower().replace(" ", "")):
            return None
        else:
            raise LogicFileError(f"No device IDs found: line: {line}")

    with open(file, "r") as f:
        l0(f.readline())
        name = l1(f.readline())
        gfx, cu = l2(f.readline())
        deviceIds = l3(f.readline())

    return ArchInfo(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def filterLogicFilesByArchPredicates(
    logicFiles: List[str], archs: Set[str], requestedDeviceIds: Set[str]
) -> List[str]:
    """
    Filter logic files based on architecture and requested device IDs.

    Args:
        logicFiles: List of logic file paths to filter
        archs: List of target architectures (e.g. ['gfx908', 'gfx90a'])
        requestedDeviceIds: Set of device IDs (e.g. {'id=1234', 'id=5678'})

    Returns:
        List of logic files that match architecture and requested device IDs.
        For each base filename, prefers exact matches over fallbacks.
    """
    fallbackIds = {
        ARCH_DEVICE_ID_FALLBACKS[v] for v in requestedDeviceIds if v in ARCH_DEVICE_ID_FALLBACKS
    }
    exactMatches = set()
    fallbackMatches = dict()

    for logicFile in map(Path, logicFiles):
        archInfo = _extractArchInfo(logicFile)

        if archInfo.Gfx not in archs:
            print2(
                f"Skipping {logicFile}\n  because architecture {archInfo.Gfx} not in targets {archs}"
            )
            continue

        deviceIds = archInfo.DeviceIds
        if any(devId in requestedDeviceIds for devId in deviceIds):
            exactMatches.add(logicFile)
        elif any(devId in fallbackIds for devId in deviceIds):
            fallbackMatches[logicFile.name] = logicFile
        else:
            print2(
                f"Skipping {logicFile}\n  because device IDs {deviceIds} don't match requested variants {requestedDeviceIds} or fallbacks {fallbackIds}"
            )

    validFallbacks = {
        path
        for name, path in fallbackMatches.items()
        if not any(em.name == name for em in exactMatches)
    }

    if validFallbacks:
        print2("Using fallbacks:\n  " + "\n  ".join(map(str, validFallbacks)))

    return list(exactMatches.union(validFallbacks))


def splitArchsFromPredicates(archSpecs: List[str]) -> Tuple[List[str], Optional[Set[str]]]:
    """
    Splits a list of architecture specifications into a list of architectures and a set of predicate specifications.

    Args:
        archSpecs: A list of architecture specifications.
    Returns:
        A tuple containing a list of architectures and a set of variant specifications.
    """
    pattern = r"(.*?)\[(.*?)\]"

    variants = set()
    archs = []
    for archSpec in archSpecs:
        match = re.match(pattern, archSpec)
        if match:
            archs.append(match.group(1).strip())
            variantId = verifyPredicate(match.group(2))
            variants.add(variantId)
        else:
            archs.append(archSpec)
    return (archs, variants if variants else None)


def verifyPredicate(predicateSpec: str) -> str:
    """
    Verifies that a predicate specification is valid.

    Args:
        predicateSpec: A string representing a predicate specification.
    Returns:
        The validated predicate specification.
    Raises:
        ValueError: If the predicate specification is invalid.
    """
    key, _, val = predicateSpec.partition("=")
    if key == "id":
        if predicateSpec not in SUPPORTED_ARCH_DEVICE_IDS:
            raise ValueError(f"Invalid architecture variant: device ID not supported: {predicateSpec}")
    return predicateSpec

