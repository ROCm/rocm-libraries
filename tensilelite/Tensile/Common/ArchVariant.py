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
    """Extracts an architecture variant from a given logic file.
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
            raise LogicFileError(f"Expected minimum required version:\n  line: {line}  file: {file}")

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


def filterVariants(logicFiles: List[str], archs: List[str], variants: List[str]) -> List[str]:
    """Filter logic files based on architecture and device ID variants.
    
    Args:
        logicFiles: List of logic file paths to filter
        archs: List of target architectures (e.g. ['gfx908', 'gfx90a'])
        variants: List of device ID variants (e.g. ['id=1234', 'id=5678'])
    
    Returns:
        List of logic files that match both architecture and variant requirements
    """
    keep = []
    
    for logicFile in logicFiles:
        variantInfo = _extractArchVariant(logicFile)
        
        if not variantInfo.Gfx in archs:
            print2(f"Skipping {logicFile}: Architecture {variantInfo.Gfx} not in targets {archs}")
            continue
        if not any(dev_id in variants for dev_id in variantInfo.DeviceIds):
            print2(f"Skipping {logicFile}: No matching device IDs between {variantInfo.DeviceIds} and {variants}")
            continue
        keep.append(logicFile)
    return keep 


def splitVariantsFromArchs(archSpecs: List[str]) -> Tuple[List[str], Optional[Set[str]]]:
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


def verifyVariant(variantSpec: str):
    deviceIdLength = 4
    hexChars = "1234567890abcdef"
    key, _, val = variantSpec.partition("=")
    if key != "id" and all(c in hexChars for c in val.lower()) and len(val) == deviceIdLength:
        raise ValueError(f"Invalid architecture variant: {variantSpec}")
    return variantSpec