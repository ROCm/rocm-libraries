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

from .Utilities import printWarning

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

            if any(id in emulationIds for id in devIds):
                printWarning("Emulation device ID(s) found, ignoring...")
                devIds = filter(lambda id: id not in emulationIds, devIds)
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


def _addVariantMap(variantFiles: Dict[str, Set[Tuple[Path, str]]], spec: str, path: Path, fname: str) -> bool:
    if fname not in {x for _, x in variantFiles[spec]}:
        variantFiles[spec].add((path, fname))
        return True
    return False


def _populateVariantMap(variantMap: Dict[str, Dict[str, Set[Tuple[Path, str]]]], targetLogicFile: Path, fallbackKey: str):
        file = Path(targetLogicFile)
        path, fname = file.parent, file.name

        variant = _extractArchVariant(file)
        if variant.Gfx not in variantMap:
            return

        variantFiles = variantMap[variant.Gfx]

        # If CU and ID are both None, then this is a fallback file b/c no predicates are specified
        if variant.CUCount is None and variant.DeviceIds is None:
            if all(fname not in {nm for _, nm in variantFiles[spec]} for spec in variantFiles if spec != fallbackKey):
                variantFiles[fallbackKey].add((path, fname))
        else:
            removeFallbacks= []
            for spec in variantFiles:
                if "id" in spec and variant.DeviceIds:
                    removeFallbacks.extend(_addVariantMap(variantFiles, spec, path, fname) for id in variant.DeviceIds if id == spec)
                if "cu" in spec and variant.CUCount:
                    removeFallbacks.append(_addVariantMap(variantFiles, spec, path, fname) if variant.CUCount == spec else False)

            if removeFallbacks and all(removeFallbacks):
                variantFiles["fallback"] = set(filter(lambda x: x[1] != fname, variantFiles[fallbackKey]))

def filterVariants(logicFiles: List[str], variants: Dict[str, Dict[str, Set[Tuple[Path, str]]]]) -> List[str]:
    fallback = "fallback"
    # A `spec` here is a variant specification passed via the command line, e.g., "cu=64"
    # This is how the code differentiates variants of the same gfx, as well as "fallback" files
    variantMap = {gfx: {spec: set() for spec in specs} for gfx, specs in variants.items()}
    for file in variantMap.values():
        file[fallback] = set()

    for logicFile in logicFiles:
        _populateVariantMap(variantMap, Path(logicFile), fallback)

    return [str(p / file) for variantFiles in variantMap.values() for files in variantFiles.values() for p, file in files]

def splitVariantsFromArchs(archSpecs: List[str]) -> Optional[List[Tuple[str, List[str]]]]:
    # Updated pattern to capture everything before the brackets as well
    pattern = r"(.*?)\[(.*?)\]"
    
    variants = []
    archs = []
    for archSpec in archSpecs:
        match = re.match(pattern, archSpec)
        if match:
            archs.append(match.group(1).strip())
            variantId = verifyVariant(match.group(2))
            variants.append(variantId)
    result = (archs, variants if variants else None)
    return result

def verifyVariant(variantSpec: str):
    deviceIdLength = 4
    hexChars = "1234567890abcdef"
    key, _, val = variantSpec.partition("=")
    if key != "id" and all(c in hexChars for c in val.lower()) and len(val) == deviceIdLength:
        raise ValueError(f"Invalid architecture variant: {spec}")
    return variantSpec