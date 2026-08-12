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
from .Utilities import print1

import rocisa

# Translate GPU targets to filename prefixes in tensilelite logic files
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
    "gfx1152": "gfx1152",
    "gfx1153": "gfx1153",
    "gfx1200": "gfx1200",
    "gfx1201": "gfx1201",
    "gfx1250": "gfx1250",
    # gfx1250 v0 silicon; its capability deltas are in ARCH_CAP_OVERRIDES. It
    # shares gfx1250's ISA, so `all` -- built from SUPPORTED_ISA -- cannot name
    # it and it has to be asked for explicitly.
    "gfx1250v0": "gfx1250v0",
}

gfxVariantMap = {
    "gfx906": ["gfx906:xnack+", "gfx906:xnack-"],
    "gfx908": ["gfx908:xnack+", "gfx908:xnack-"],
    "gfx90a": ["gfx90a:xnack+", "gfx90a:xnack-"],
    "gfx942": ["gfx942:xnack+", "gfx942:xnack-"],
    "gfx950": ["gfx950:xnack+", "gfx950:xnack-"],
}

# The single declaration point for gfx1250 v0's capability deltas. Both ASIC
# revisions share ISA (12,5,0) and assemble at `-mcpu=gfx1250`, so the probe
# can't tell them apart; one build is one revision, so the deltas are declared
# here and layered onto the probed table. Keys are grouped by capability nature
# (instruction-shaped vs architectural), matching the dict each consumer reads.
ARCH_CAP_OVERRIDES = {
    "gfx1250v0": {
        # Instruction-shaped: v0 lacks the fp4 32x16 WMMA opcode.
        "asmCaps": {
            "HasWMMA_f4_32x16": False,
        },
        # Architectural: v0 has no TDM-multicast. NOTE: v0 still requires the
        # XNACK-replay xcnt drain + SMEM dst/base overlap fix (RequiresXCntForVolatileVMEM),
        # so it is intentionally NOT overridden here and inherits the probed default (True).
        "archCaps": {
            "HasTDMMulticast": False,
        },
    },
}

# Compiler target for names that are not themselves valid targets. The compiler
# does not model steppings, so gfx1250v0 has to reach `-mcpu` / `--offload-arch`
# as gfx1250; otherwise clang fails with `unsupported HIP gpu architecture`.
ARCH_COMPILER_TARGET = {
    "gfx1250v0": "gfx1250",
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
    IsaVersion(11, 5, 2),
    IsaVersion(11, 5, 3),
    IsaVersion(12, 0, 0),
    IsaVersion(12, 0, 1),
    IsaVersion(12, 5, 0),
]

# Source-of-truth chip IDs used to generate build predicates.
GFX_CHIP_IDS = {
    "gfx942": ["74a0", "74a1", "74a2", "74a3", "74a5", "74a9"],
    "gfx950": ["75a0", "75b0", "75a2", "75b2", "75a3", "75b3", "75a8", "75b8"],
}

# Source-of-truth CU counts used to generate build predicates.
GFX_CU_COUNTS = {
    "gfx942": ["20", "38", "64", "80", "96", "228", "304"],
}

SUPPORTED_BUILD_CHIP_IDS = {
    f"id={chipId}": gfx for gfx, chipIds in GFX_CHIP_IDS.items() for chipId in chipIds
}

SUPPORTED_BUILD_CU_COUNTS = {
    f"cu={cuCount}": gfx for gfx, cuCounts in GFX_CU_COUNTS.items() for cuCount in cuCounts
}

SUPPORTED_CHIP_ID_FALLBACKS = {
    "id=75b0": ["id=75a0"],
    "id=75a2": ["id=75a0"],
    "id=75b2": ["id=75a0"],
    "id=75a3": ["id=75a0"],
    "id=75b3": ["id=75a0"],
    "id=75a8": ["id=75a0"],
    "id=75b8": ["id=75a0"],
}

# `None` refers to an unspecified CU count.
SUPPORTED_CU_COUNT_FALLBACKS = {
    "cu=20": None,
    "cu=38": None,
    "cu=64": None,
    "cu=80": None,
    "cu=96": None,
    "cu=228": None,
    "cu=304": None,
}

def supportsChipIdPredicate(gfx: str) -> bool:
    """
    Returns whether PCI chip ID predicates are currently enabled for a GFX architecture.

    Extend as needed to support other architectures.
    """
    return gfx == "gfx950"


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


def gfxToCompilerTarget(name: str) -> str:
    """The target to compile an architecture name with.

    The two differ only where a name carries something the compiler does not
    model, currently gfx1250's stepping. Anything else is returned unchanged,
    which keeps qualifiers like ``:xnack+`` that deriving the target from the
    ISA version would drop.

    Args:
        name: A requested gfx architecture name (e.g. 'gfx1250v0').

    Returns:
        The target for ``-mcpu`` / ``--offload-arch`` (e.g. 'gfx1250').
    """
    return ARCH_COMPILER_TARGET.get(name, name)


def expandAllArchitectures(archs: List[str]) -> List[str]:
    """Replaces the ``all`` keyword with the architectures it covers.

    ``all`` is built from SUPPORTED_ISA, so it cannot name an architecture that
    shares another's ISA; those names survive alongside it and reach the
    mixed-build guard, rather than being dropped into a silent build of the
    other stepping. Qualified specs (``gfx942:xnack+``) name architectures
    ``all`` already covers, so they stay absorbed.

    Empty entries are dropped: cmake joins ``GPU_TARGETS`` with ``;``, so a
    trailing one arrives as an empty spec the predicate splitter would reject.

    Args:
        archs: The requested architecture names, possibly including 'all'.

    Returns:
        The list with 'all' replaced by the architectures it covers.
    """
    archs = [a.strip() for a in archs if a.strip()]
    if "all" not in archs:
        return archs
    covered = set(SUPPORTED_GFX)
    return SUPPORTED_GFX + [
        a for a in archs if a != "all" and baseArchName(a) not in covered
    ]


def baseArchName(spec: str) -> str:
    """The bare architecture name in a spec, without predicates or qualifiers."""
    return spec.split("[")[0].split(":")[0].strip()


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
    # Belt-and-suspenders for the GPU-less --cpu-only switch: when CpuOnly is set,
    # return a spoofed per-arch IsaVersion (derived from gfxToIsa) instead of shelling
    # out to a device-enumeration tool. The arch comes from the CpuOnlyArch plumbing key.
    # This backstops any entry path that reaches detection without passing an arch (the
    # primary path supplies the arch via --gpu-targets and never reaches here). Returning
    # an IsaVersion makes the isinstance(...) guard in detectGlobalCurrentISA pass so the
    # "Failed to detect currect ISA" raise never fires GPU-less.
    # Imported lazily to avoid a circular import (GlobalParameters imports from this module).
    from .GlobalParameters import globalParameters
    if globalParameters.get("CpuOnly"):
        isa = gfxToIsa(globalParameters.get("CpuOnlyArch", "gfx942"))
        if isa is not None:
            print(f"# CpuOnly: spoofing GPU {deviceId} ISA as " + isaToGfx(isa))
            return isa
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


def detectHostGfxArchs() -> List[str]:
    """Enumerate the supported GPU architectures physically present on this host.

    Reuses the same device-enumeration tool selection as the rest of the
    toolchain (``ToolchainDefaults.DEVICE_ENUMERATOR`` -> ``rocm_agent_enumerator``
    or ``amdgpu-arch``) and the canonical ``gfxToIsa``/``isaToGfx`` maps. Each
    enumerated line is normalized through ``gfxToIsa`` (which strips ``:xnack±``
    and other suffixes) and filtered to ``SUPPORTED_ISA``, so CPU agents
    (``gfx000``) and unsupported devices are dropped.

    Returns:
        A de-duplicated list of canonical gfx names (e.g. ``["gfx950"]``).
        Returns an empty list when no enumerator is available or it fails --
        callers should treat "empty" as "cannot benchmark here".
    """
    # Lazy import: keep this module free of a load-time dependency on the
    # Toolchain package (which imports Common.Utilities) and avoid any import cycle.
    try:
        from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain
    except Exception:
        return []

    tool = ToolchainDefaults.DEVICE_ENUMERATOR
    try:
        toolPath = validateToolchain(tool)
    except (FileNotFoundError, ValueError):
        return []

    try:
        process = run([toolPath], stdout=PIPE, stderr=PIPE)
    except OSError:
        return []
    if process.returncode:
        return []

    archs: List[str] = []
    for line in process.stdout.decode(errors="replace").splitlines():
        isa = gfxToIsa(line.strip())
        if isa is not None and isa in SUPPORTED_ISA:
            gfx = isaToGfx(isa)
            if gfx not in archs:
                archs.append(gfx)
    return archs


def hostHasArch(arch: str) -> bool:
    """Return True iff ``arch`` matches a supported GPU present on this host.

    Comparison is done on the normalized ISA version, so ``:xnack±`` / CU
    variants on either side (requested arch or enumerated arch) compare equal.
    """
    target = gfxToIsa(arch)
    if target is None:
        return False
    return any(gfxToIsa(a) == target for a in detectHostGfxArchs())


class ArchInfo(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Optional[Set[str]]
    CUCount: Optional[str] = None


class LogicFileError(Exception):
    def __init__(self, message="Expected line is either not present or is malformed"):
        self.message = message
        super().__init__(self.message)


class _RawArchHeader(NamedTuple):
    Name: str
    Gfx: str
    DeviceIds: Set[str]
    CUCount: Optional[str] = None


_LIST_MINVER_RE = re.compile(r"- (?:\{MinimumRequiredVersion|MinimumRequiredVersion:)")
_LIST_ARCH_WITH_CU_RE = re.compile(r"- \{Architecture: (\w+), CUCount: (\d+)\}")
_LIST_ARCH_RE = re.compile(r"- gfx(\w+)")
_LIST_DEVICE_LINE_RE = re.compile(r"- \[Device")

_DICT_DEVICE_NAMES_KEY_RE = re.compile(r"^\s*DeviceNames\s*:\s*")
_DICT_DEVICE_INLINE_RE = re.compile(r"Device\s+([0-9a-fA-F]+)")
_DICT_DEVICE_ITEM_RE = re.compile(r"^\s*-\s*Device\s+[0-9a-fA-F]+\s*$")


def _extractArchInfoFromList(lines: List[str], file: Union[str, Path]) -> _RawArchHeader:
    """Parse the legacy list-format logic header into a raw architecture tuple.

    Expected header lines are:
    1) minimum required version,
    2) schedule/code name,
    3) architecture (optionally with CUCount),
    4) device ID list.
    """

    def l0(line: str):
        if not _LIST_MINVER_RE.match(line):
            raise LogicFileError(
                f"Expected minimum required version:\n  line: {line}  file: {file}"
            )

    def l1(line: str):
        return line[2:].strip()

    def l2(line: str):
        match1 = _LIST_ARCH_WITH_CU_RE.match(line)
        match2 = _LIST_ARCH_RE.match(line)
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
        if _LIST_DEVICE_LINE_RE.match(line):
            devIds = re.findall(r"Device (\w+)", line)
            # Normalize to lowercase so downstream consumers (predicate
            # tables, fallback maps, chip-ID directory matchers) all agree
            # on the canonical form.
            return set(f"id={id.lower()}" for id in devIds)
        else:
            raise LogicFileError(f"No device IDs found: line: {line}")

    if len(lines) < 4:
        raise LogicFileError(f"Expected at least 4 list-format header lines in {file}")

    l0(lines[0])
    name = l1(lines[1])
    gfx, cu = l2(lines[2])
    deviceIds = l3(lines[3])
    return _RawArchHeader(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def _extractArchInfoFromDictFast(lines: List[str], file: Union[str, Path]) -> _RawArchHeader:
    """Fast-scan dict-format logic headers without full YAML-object parsing.

    Extracts ScheduleName, ArchitectureName, optional CUCount, and DeviceNames
    (inline list or multi-line list) directly from text lines.
    """

    def find_scalar(key: str) -> Optional[str]:
        pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*(.+?)\s*$")
        for line in lines:
            match = pattern.match(line)
            if match:
                return match.group(1).strip()
        return None

    def parse_device_ids() -> Set[str]:
        for idx, line in enumerate(lines):
            if not _DICT_DEVICE_NAMES_KEY_RE.match(line):
                continue

            rhs = _DICT_DEVICE_NAMES_KEY_RE.sub("", line)

            # Inline list form: DeviceNames: [Device 75a0, Device 75a2]
            if rhs:
                ids = _DICT_DEVICE_INLINE_RE.findall(rhs)
                if not ids:
                    raise LogicFileError(f"Malformed DeviceNames entry in dict-format logic: {file}")
                return set(f"id={chip_id.lower()}" for chip_id in ids)

            # Multi-line form:
            # DeviceNames:
            #   - Device 75a0
            ids = []
            for next_line in lines[idx + 1 :]:
                if _DICT_DEVICE_ITEM_RE.match(next_line):
                    ids.extend(_DICT_DEVICE_INLINE_RE.findall(next_line))
                    continue
                if not next_line.strip():
                    continue
                if re.match(r"^\s", next_line):
                    break
                break

            if not ids:
                raise LogicFileError(f"Malformed DeviceNames entry in dict-format logic: {file}")
            return set(f"id={chip_id.lower()}" for chip_id in ids)

        raise LogicFileError(f"Expected DeviceNames list in dict-format logic: {file}")

    name = find_scalar("ScheduleName")
    if not name:
        raise LogicFileError(f"Expected ScheduleName in dict-format logic: {file}")
    name = name.strip("'\"")

    gfx = find_scalar("ArchitectureName")
    if not gfx:
        raise LogicFileError(f"Expected ArchitectureName in dict-format logic: {file}")
    gfx = gfx.strip("'\"")

    cu = None
    cu_count = find_scalar("CUCount")
    if cu_count:
        cu_count = cu_count.strip("'\"")
        if cu_count.isdigit():
            cu = f"cu={cu_count}"

    deviceIds = parse_device_ids()
    return _RawArchHeader(Name=name, Gfx=gfx, DeviceIds=deviceIds, CUCount=cu)


def _finalizeArchInfo(
    raw: _RawArchHeader,
    file: Union[str, Path],
    validateDeviceIds: bool,
) -> ArchInfo:
    if validateDeviceIds:
        try:
            for predicateSpec in raw.DeviceIds:
                _verifyPredicate(predicateSpec, raw.Gfx)
        except ValueError as e:
            raise LogicFileError(f"Invalid device ID found while parsing {file}: {e}")

    return ArchInfo(
        Name=raw.Name,
        Gfx=raw.Gfx,
        DeviceIds=raw.DeviceIds,
        CUCount=raw.CUCount,
    )


def _extractArchInfo(file: Union[str, Path], validateDeviceIds: bool = True) -> ArchInfo:
    """
    Extracts architecture predicate information from a given logic file.

    Supported logic header formats:

    1) Legacy list format:
        - Line 0: minimum required version
            (e.g., "- {MinimumRequiredVersion: 4.33.0}")
        - Line 1: schedule/code name (e.g., "- aquavanjaram")
        - Line 2: architecture, optionally with CU count
            (e.g., "- gfx950" or "- {Architecture: gfx950, CUCount: 256}")
        - Line 3: device IDs
            (e.g., "- [Device 1234, Device 5678]")

    2) Dict format (fast header scan, no full YAML-object parsing):
        - ``ScheduleName``
        - ``ArchitectureName``
        - ``CUCount`` (optional)
        - ``DeviceNames`` (inline or multi-line list)

    Args:
        file: Path to a logic file.
        validateDeviceIds: Whether to validate Device IDs against the supported
            chip-ID tables while parsing.
    Returns:
        ArchInfo: An object containing the extracted architecture predicates.
    Raises:
        LogicFileError: If the file does not match the expected format.
    """

    with open(file, "r") as f:
        lines = f.read().splitlines()

    first_nonempty = next(
        (
            line.strip()
            for line in lines
            if line.strip() and not line.lstrip().startswith("#")
        ),
        "",
    )

    if first_nonempty.startswith("-"): 
        # List format
        raw = _extractArchInfoFromList(lines, file)
    else:  
        # Dict format
        raw = _extractArchInfoFromDictFast(lines, file)

    return _finalizeArchInfo(raw, file, validateDeviceIds)


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
        if predicateSpec not in SUPPORTED_BUILD_CHIP_IDS:
            raise ValueError(f"{msgPrefix}: device ID not supported")
        if gfx and SUPPORTED_BUILD_CHIP_IDS[predicateSpec] != gfx:
            raise ValueError(f"{msgPrefix}: device ID is not associated with {gfx}")
    elif key == "cu":
        if predicateSpec not in SUPPORTED_BUILD_CU_COUNTS:
            raise ValueError(f"{msgPrefix}: CU count not supported")
        if gfx and SUPPORTED_BUILD_CU_COUNTS[predicateSpec] != gfx:
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
        if v in SUPPORTED_CHIP_ID_FALLBACKS
        for fallbackId in SUPPORTED_CHIP_ID_FALLBACKS[v]
    }
    fallbackCUs = {SUPPORTED_CU_COUNT_FALLBACKS[v] for v in requestedCUs if v in SUPPORTED_CU_COUNT_FALLBACKS}

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
