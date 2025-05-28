################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

import re
from pathlib import Path
from typing import NamedTuple, Optional, Union, Tuple, Set, Dict, List

from .Utilities import printWarning, print2

ARCH_VARIANT_FALLBACKS = {
    "id=75a2": "id=75a0",
    "id=75a3": "id=75a0",
}


class ArchVariant(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[str] = None


class LogicFileError(Exception):
    def __init__(self, message="Expected line is either not present or is malformed"):
        self.message = message
        super().__init__(self.message)


def _extractArchVariant(file: Union[str, Path]) -> ArchVariant:
    """
    Extracts an architecture variant from a given logic file.

    The file is expected to have the following format:
    - Line 1: Minimum required version (e.g., "- {MinimumRequiredVersion: 4.33.0}")
    - Line 2: Code name of the architecture (e.g., "- aquavanjaram")
    - Line 3: GFX name of the architecture or a map with variant details (e.g., "- gfx950" or "- {Architecture: gfx950, CUCount: 256}")
    - Line 4: Device IDs (e.g., "- [Device 1234, Device 5678]")

    Args:
        file: Path to a logic file.
    Returns:
        ArchVariant: An object containing the extracted architecture variant.
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

    return ArchVariant(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def filterVariants(
    logicFiles: List[str], archs: Set[str], requestedDeviceIds: Set[str]
) -> List[str]:
    """
    Filter logic files based on architecture and device ID variants.

    Args:
        logicFiles: List of logic file paths to filter
        archs: List of target architectures (e.g. ['gfx908', 'gfx90a'])
        requestedDeviceIds: Set of device ID variants (e.g. {'id=1234', 'id=5678'})

    Returns:
        List of logic files that match architecture and variant requirements.
        For each base filename, prefers exact variant matches over fallbacks.
    """
    fallbackIds = {
        ARCH_VARIANT_FALLBACKS[v] for v in requestedDeviceIds if v in ARCH_VARIANT_FALLBACKS
    }
    exactMatches = set()
    fallbackMatches = dict()

    for logicFile in map(Path, logicFiles):
        variant = _extractArchVariant(logicFile)

        if variant.Gfx not in archs:
            print2(
                f"Skipping {logicFile}\n  because architecture {variant.Gfx} not in targets {archs}"
            )
            continue

        deviceIds = variant.DeviceIds
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


def splitVariantsFromArchs(archSpecs: List[str]) -> Tuple[List[str], Optional[Set[str]]]:
    """
    Splits a list of architecture specifications into a list of architectures and a set of variant specifications.

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
            variantId = verifyVariant(match.group(2))
            variants.add(variantId)
        else:
            archs.append(archSpec)
    return (archs, variants if variants else None)


def verifyVariant(variantSpec: str) -> str:
    """
    Verifies that a variant specification is valid.

    Args:
        variantSpec: A string representing a variant specification.
    Returns:
        The validated variant specification.
    Raises:
        ValueError: If the variant specification is invalid.
    """
    deviceIdLength = 4
    hexChars = "1234567890abcdef"
    key, _, val = variantSpec.partition("=")
    if key != "id" and all(c in hexChars for c in val.lower()) and len(val) == deviceIdLength:
        raise ValueError(f"Invalid architecture variant: {variantSpec}")
    return variantSpec
