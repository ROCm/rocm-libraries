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
from ..GpuArch import detect_gpu_archs

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
    # Spelled as clang and ROCr both spell it. Shares gfx1250's ISA, so `all` --
    # built from SUPPORTED_ISA -- cannot name it; ask for it by name.
    "gfx1250-strict": "gfx1250-strict",
}

gfxVariantMap = {
    "gfx906": ["gfx906:xnack+", "gfx906:xnack-"],
    "gfx908": ["gfx908:xnack+", "gfx908:xnack-"],
    "gfx90a": ["gfx90a:xnack+", "gfx90a:xnack-"],
    "gfx942": ["gfx942:xnack+", "gfx942:xnack-"],
    "gfx950": ["gfx950:xnack+", "gfx950:xnack-"],
}

# Where a stepping's capabilities differ from what probing reports. Capabilities
# are probed by ISA version, and gfx1250-strict shares gfx1250's, so the probe
# returns gfx1250's answers for both; these deltas are layered on top.
# Sub-keys match the dict each consumer reads.
ARCH_CAP_OVERRIDES = {
    "gfx1250-strict": {
        "asmCaps": {
            "HasWMMA_f4_32x16": False,  # no fp4 32x16 WMMA opcode
        },
        "archCaps": {
            "HasTDMMulticast": False,
            # RequiresXCntForVolatileVMEM is deliberately absent: gfx1250-strict
            # still needs the XNACK-replay xcnt drain, so it inherits the probed True.
        },
    },
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


def archMacroNames(isa: IsaVersion) -> List[str]:
    """The macros clang predefines for every architecture sharing an ISA.

    A stepping shares its base architecture's ISA but is its own compiler
    target, and clang names the macro after the target, spelling hyphens as
    underscores. Guarding source on the base's macro alone would drop it from
    the stepping's compilation, so guard on all of them.

    Args:
        isa: An object representing the major, minor, and step version of the ISA.

    Returns:
        The predefined macro name of each architecture with that ISA, sorted.
        Empty if the ISA names no architecture; `all` is not one, and matching
        it would yield a macro no compilation ever defines.
    """
    if isa is None:
        return []
    names = sorted(
        {baseArchName(name) for name in architectureMap if gfxToIsa(name) == isa}
    )
    return ["__" + name.replace("-", "_") + "__" for name in names]


def steppingArchOf(spec: str) -> Optional[str]:
    """The architecture a stepping name belongs to.

    A name is a stepping exactly when it does not survive a round trip through
    its ISA: ``gfx942`` comes back as itself, ``gfx1250-strict`` comes back as
    ``gfx1250``. Deriving it means a new stepping needs no registration here.

    Args:
        spec: A requested architecture spec, qualified or not.

    Returns:
        The ISA-derived name (``gfx1250-strict`` -> ``gfx1250``), or None if the
            spec names an architecture rather than a stepping of one.
    """
    base = baseArchName(spec)
    isa = gfxToIsa(base)
    if isa is None:
        return None
    derived = isaToGfx(isa)
    return derived if derived != base else None


def archNamesByIsa(specs: List[str]) -> Dict[IsaVersion, str]:
    """The architecture each ISA is being built as, keyed by ISA version.

    Two architectures can share an ISA -- gfx1250 and gfx1250-strict both spell
    12.5.0 -- yet they are distinct compiler targets whose code objects carry
    different ELF machine codes and will not load on each other. So the ISA alone
    names neither the compiler target, nor the code object, nor the output
    subtree; the requested name has to travel from the command line to each.

    Qualifiers are dropped: ``gfx942:xnack+`` names an architecture its ISA
    already describes, and forwarding it would pin the code object to one xnack
    setting instead of leaving it xnack-agnostic.

    Args:
        specs: The requested architecture specs, qualified or not.

    Returns:
        ISA version -> architecture name.

    Raises:
        ValueError: If two requested architectures share an ISA. One key cannot
            name both, and silently keeping either would assemble one
            architecture's kernels for the other and write them under its name.
    """
    names: Dict[IsaVersion, str] = {}
    for spec in specs:
        name = baseArchName(spec)
        isa = gfxToIsa(name)
        if isa is None:
            continue
        if names.setdefault(isa, name) != name:
            raise ValueError(
                f"Architectures {names[isa]!r} and {name!r} share ISA "
                f"{tuple(isa)}, so one build cannot name both; build each "
                "one separately."
            )
    return names


def archNameForIsa(isa: IsaVersion, archNames: Optional[List[str]] = None) -> str:
    """The architecture one ISA is being built as.

    The ISA-derived name is the answer for every architecture that is the only
    one on its ISA. It is the wrong answer for a stepping, which shares an ISA
    with the architecture it steps from, so a requested name outranks it.

    Entry paths that never learn a name -- ``ISA:`` in a config, and auto-detect
    before it has a device -- pass nothing and get the derived name, which is
    what they had before steppings existed.

    Args:
        isa: The ISA version being built.
        archNames: The requested architecture specs, qualified or not.

    Returns:
        The architecture name to build this ISA as.
    """
    return archNamesByIsa(archNames or []).get(isa) or isaToGfx(isa)


def isaCollisionFreeGroups(specs: List[str]) -> List[List[str]]:
    """Partitions requested architectures into groups one build can each cover.

    A build names its target by ISA, so two architectures sharing one — a
    stepping and the architecture it steps from — cannot appear together; see
    ``archNamesByIsa``. Splitting them into groups lets a caller cover every
    requested architecture by running once per group. Nothing collides in the
    common case, which yields a single group.

    Membership is derived rather than listed, so a new stepping is partitioned
    correctly without being registered here. Specs keep their qualifiers and
    their requested order, and ``all`` is expanded first.

    Args:
        specs: The requested architecture specs, qualified or not.

    Returns:
        Groups of specs, each free of ISA collisions. Empty if specs is empty.
    """
    def collides(spec: str, group: List[str]) -> bool:
        """Whether adding spec to group is the pairing archNamesByIsa rejects.

        Sharing an ISA is not enough: qualified specs of one architecture
        (``gfx942:xnack+`` and ``gfx942:xnack-``) share both the ISA and the
        name, and one run covers them together. Only a differing name over the
        same ISA -- a stepping beside what it steps from -- has to split.
        """
        name = baseArchName(spec)
        isa = gfxToIsa(name)
        if isa is None:
            # Names no ISA, so it collides with nothing. The run that has to make
            # sense of it rejects it; the partitioning does not.
            return False
        return any(
            gfxToIsa(baseArchName(o)) == isa and baseArchName(o) != name for o in group
        )

    groups: List[List[str]] = []
    for spec in expandAllArchitectures(specs):
        for group in groups:
            if not collides(spec, group):
                group.append(spec)
                break
        else:
            groups.append([spec])
    return groups


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
    try:
        # Only the step is hexadecimal. A letter anywhere else matches the shape
        # without spelling a version, and callers are written against the None
        # this promises rather than against an exception from int().
        step = int(ipart[-1], 16)
        minor = int(ipart[-2])
        major = int(ipart[:-2])
    except ValueError:
        return None
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


# The architecture in a line of tool output. Stops before ``:xnack+`` and
# ``[cu=64]`` the way baseArchName would, but keeps a stepping's hyphenated
# suffix, which is the one part of the name its ISA cannot recover.
_REPORTED_ARCH_RE = re.compile(r"gfx[0-9a-z]+(?:-[0-9a-z]+)*")


def _supportedArchNames(reported) -> List[str]:
    """The names Tensile knows, out of what a detection tool reported.

    One entry per accepted agent, in enumeration order, repeats kept: callers
    index this positionally to answer "what is device N". Answering "which
    architectures are present" means de-duplicating it afterwards.

    A name Tensile does not know is dropped rather than accepted on the strength
    of its ISA. ``gfxToIsa``'s regex stops at the first non-hex character, so a
    name that merely looks like a stepping (gfx1250v1) still resolves to
    (12,5,0) and would otherwise be built as gfx1250 without a word.

    A line is searched for the name rather than being one: an enumerator may
    label it (``hipinfo`` prints ``gcnArchName: gfx1100``), and splitting such a
    line on its colon would yield the label.
    """
    archs: List[str] = []
    for line in reported:
        match = _REPORTED_ARCH_RE.search(str(line))
        if match is None:
            continue
        arch = match.group(0)
        isa = gfxToIsa(arch)
        if isa is not None and isa in SUPPORTED_ISA and arch in architectureMap:
            archs.append(arch)
    return archs


def _detectArchNames(detectionTool) -> List[str]:
    """Every architecture Tensile recognises on this host, best source first.

    ``detectionTool`` is asked first, as it always was: it alone reads a
    ``target.lst`` or ``HSA_OVERRIDE_GFX_VERSION`` pin, which a sandboxed image
    or a near-miss card depends on. ``detect_gpu_archs`` (amdgpu-arch, then
    rocminfo) answers from the hardware, and is the fallback for hosts where the
    enumerator cannot answer -- previously such a host simply failed.

    What steppings changed is not the order but what survives the answer: the
    name used to be rebuilt from its ISA, and gfx1250-strict shares (12,5,0)
    with gfx1250, so the rebuild silently reported the wrong stepping.
    """
    return _fromEnumerator(detectionTool) or _supportedArchNames(detect_gpu_archs())


def _fromEnumerator(detectionTool) -> List[str]:
    """Every architecture Tensile recognises, according to one named tool."""
    if detectionTool is None:
        return []

    try:
        # stderr is captured, not inherited: both tools are chatty about a
        # missing render group, and this path is reached on exactly the hosts
        # that have that problem.
        process = run([detectionTool], stdout=PIPE, stderr=PIPE)
    except OSError:
        return []
    if process.returncode:
        print(f"{detectionTool} exited with code {process.returncode}")
        return []
    return _supportedArchNames(process.stdout.decode(errors="replace").split("\n"))


def _detectGlobalCurrentArch(detectionTool, deviceId: int):
    """The architecture name detected for one device.

    The name, not the ISA: gfx1250 and gfx1250-strict both spell (12,5,0), so an
    ISA handed back here could no longer say which of them was seen, and every
    caller would resolve it to the shipping stepping.

    Returns:
        The gfx name, or 1 on failure.
    """
    # Belt-and-suspenders for the GPU-less --cpu-only switch: when CpuOnly is set,
    # return the spoofed arch instead of shelling out to a device-enumeration tool.
    # This backstops any entry path that reaches detection without passing an arch
    # (the primary path supplies the arch via --gpu-targets and never reaches here),
    # so the "Failed to detect" raises never fire GPU-less.
    # Imported lazily to avoid a circular import (GlobalParameters imports from this module).
    from .GlobalParameters import globalParameters
    if globalParameters.get("CpuOnly"):
        arch = baseArchName(globalParameters.get("CpuOnlyArch", "gfx942"))
        # "all" is a key in architectureMap but not an architecture; spoofing it
        # would put the literal string into -mcpu and into a directory name.
        if arch != "all" and arch in architectureMap:
            print(f"# CpuOnly: spoofing GPU {deviceId} as " + arch)
            return arch

    # Positional: one entry per agent, so this names the requested device rather
    # than the first architecture that happens to be present.
    archList = _detectArchNames(detectionTool)
    if deviceId >= len(archList):
        return 1
    print(f"# Detected GPU {deviceId}: " + archList[deviceId])
    return archList[deviceId]


def _detectGlobalCurrentISA(detectionTool, deviceId: int):
    """The ISA of one device, or 1 if it could not be detected.

    Detection has a fallback source now, so there is no one exit code to hand
    back; the public wrappers only ever check the type before raising.
    """
    arch = _detectGlobalCurrentArch(detectionTool, deviceId)
    return gfxToIsa(arch) if isinstance(arch, str) else arch


def detectGlobalCurrentArch(deviceId: int, enumerator: str) -> str:
    """The architecture name of a given device.

    Prefer this over ``detectGlobalCurrentISA`` wherever the answer names an
    artifact, a compiler target, or a capability set: those are per-architecture,
    and two architectures can share one ISA.

    Args:
        deviceId: an integer indicating the device to inspect.
        enumerator: the device-enumeration tool to ask.

    Raises:
        Exception if nothing could detect an architecture.
    """
    result = _detectGlobalCurrentArch(enumerator, deviceId)
    if not isinstance(result, str):
        raise Exception("Failed to detect current architecture")
    return result


def detectGlobalCurrentISA(deviceId: int, enumerator: str):
    """Returns the ISA version for a given device.

    The ISA tuple (X, Y, Z) of the architecture ``detectGlobalCurrentArch``
    names for that device. Two architectures can share one ISA, so prefer the
    name wherever the answer names an artifact, a compiler target, or a
    capability set.

    Args:
        deviceID: an integer indicating the device to inspect.

    Raises:
        Exception if no source could detect an ISA.
    """
    result = _detectGlobalCurrentISA(enumerator, deviceId)
    if not isinstance(result, IsaVersion):
        raise Exception("Failed to detect currect ISA")
    return result


def detectHostGfxArchs() -> List[str]:
    """Enumerate the supported GPU architectures physically present on this host.

    Asks the same sources in the same order as per-device detection -- the
    toolchain's enumerator, then amdgpu-arch and rocminfo -- and de-duplicates the
    answer, which per-device detection must not. Names keep the spelling the tool
    reported, stripped of target features (``:xnack±``) and checked against the
    known architectures, so CPU agents (``gfx000``) and unsupported devices are
    dropped.

    The name is not rebuilt from the ISA: gfx1250 and gfx1250-strict share
    (12,5,0), so a round trip through it would report gfx1250 for either and tell
    the caller it can benchmark on hardware whose code objects it cannot load.

    Returns:
        A de-duplicated list of gfx names (e.g. ``["gfx950"]``).
        Returns an empty list when nothing could be detected -- callers should
        treat "empty" as "cannot benchmark here".
    """
    # Lazy import: keep this module free of a load-time dependency on the
    # Toolchain package (which imports Common.Utilities) and avoid a cycle.
    # Nothing here is worth failing a benchmark-capability question over, so any
    # failure reaching this point is answered the documented way.
    try:
        from Tensile.Toolchain.Validators import ToolchainDefaults, validateToolchain

        archs = _fromEnumerator(validateToolchain(ToolchainDefaults.DEVICE_ENUMERATOR))
    except Exception:
        archs = []

    return list(dict.fromkeys(archs or _supportedArchNames(detect_gpu_archs())))


def hostHasArch(arch: str) -> bool:
    """Return True iff ``arch`` matches a supported GPU present on this host.

    Compared as bare names, so ``:xnack±`` and CU predicates on either side
    compare equal while gfx1250 and gfx1250-strict do not. Comparing their shared
    ISA instead would answer True for each on the other's silicon, and the caller
    would benchmark code objects the agent rejects.
    """
    target = baseArchName(arch)
    if gfxToIsa(target) is None:
        return False
    return target in detectHostGfxArchs()


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
# [\w-], not \w: a stepping's name carries a hyphen, and \w would stop short of
# it and then fail to find the comma that has to follow.
_LIST_ARCH_WITH_CU_RE = re.compile(r"- \{Architecture: ([\w-]+), CUCount: (\d+)\}")
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

    # Sorted, not set order: this list goes on to name output directories and
    # order compiler flags, and string hashing is salted per process, so an
    # unsorted set makes two runs of one build disagree.
    return sorted(architectures), predicateMap or None


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

    # Sorted: `files` is a set, and this order becomes the logic-file merge
    # order, which the solution indices written into the library follow.
    return sorted(
        str(p / file)
        for gfxPredicateMap in variantMap.values()
        for files in gfxPredicateMap.values()
        for p, file in files
    )
