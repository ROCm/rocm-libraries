# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Disassembly-backed checks that custom-kernel StaggerU metadata is truthful.

The host-side launch gate for uniform summation order (``checkUniformSummationOrder``
in ``ContractionSolution.cpp``) reasons entirely from a solution's declared
metadata.  It clamps StaggerU by writing zero into a bitfield of a packed kernel
argument, and refuses any solution that declares ``SupportCustomStaggerU: False``
together with a non-zero ``StaggerU``, on the grounds that such a kernel would
stagger with a compiled-in value the clamp cannot reach.  Every admitted solution
therefore rests on one of two claims about the shipped machine code:

* ``SupportCustomStaggerU: True`` -- the kernel derives StaggerU from the packed
  argument, so writing zero into it really does stop the staggering; or
* ``SupportCustomStaggerU: False`` with ``StaggerU: 0`` -- the kernel contains no
  staggering at all, so no clamp is needed.

These tests check those claims against the instructions the kernels actually
execute.  Reading the ``.s`` files cannot do it: 98 of the 119 shipped custom
kernels are pre-assembled ``.long`` machine-code blobs with no readable
mnemonics, and every kernel that declares a non-zero StaggerU -- exactly the set
the gate's safety argument turns on -- is among them.  So each kernel is
assembled for its own ``.amdgcn_target`` architecture and disassembled, and the
checks run on the decoded instruction stream.  That also sees through the
macros and ``.set`` symbol aliases the readable kernels are written in.

What counts as staggering.  ``KernelWriterAssembly.declareStaggerParms`` and
``calculateStagger`` give the mechanism two halves, and both must be present for
summation order to vary: the A/B buffer descriptors are advanced by
``StaggerUIter * GlobalReadIncs`` before the unroll loop, and inside the loop the
iteration whose counter equals ``StaggerUIter`` swaps the normal forward
increment for a large negative wrap delta.  Only the second half is decisive and
recognisable without symbols -- a compare, a pair of ``s_cselect_b32`` picking
either ``WrapU`` or ``GlobalReadIncs`` into a 64-bit temporary, and that
temporary added onto a buffer-descriptor base.  A kernel with no such site
cannot rotate its K-loop start position, whatever its metadata says.

Scope note: these checks establish that the canonical packed decode is present
in a kernel that staggers, not that the decoded value reaches every individual
wrap site; proving the latter would need a full dataflow analysis of the
disassembly.
"""

import os
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import pytest

from Tensile.Common.GlobalParameters import (
    defaultBenchmarkCommonParameters,
    defaultInternalSupportParams,
)
from Tensile.CustomKernels import (
    getAllCustomKernelNames,
    getCustomKernelContents,
    readCustomKernelConfig,
)


def _findTool(name: str) -> Optional[str]:
    """Prefer the ROCm LLVM toolchain, fall back to whatever is on PATH."""
    rocmBin = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm", "bin")
    return shutil.which(name, path=rocmBin) or shutil.which(name)


CLANG = _findTool("clang")
OBJDUMP = _findTool("llvm-objdump")

_PROBE_KERNEL = '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n.text\ns_endpgm\n'

KERNEL_NAMES = getAllCustomKernelNames()

# Assembling and disassembling all 119 kernels is a few seconds of wall time
# spread over a thread pool, so no separate slow marker: this stays in the unit
# suite where a change to a custom kernel will actually run it.
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)

_TARGET = re.compile(r'^\.amdgcn_target\s+"amdgcn-amd-amdhsa--([a-z0-9]+)', re.MULTILINE)

# One decoded instruction: a mnemonic, its operands, and the address/encoding
# comment llvm-objdump appends.  Anything without that comment is a label,
# a section header or blank, and is skipped.
_DISASM_LINE = re.compile(r"^\t([a-z][a-z0-9_]*)(?:\s+(.*?))?\s*//\s*[0-9A-F]{12}:")

# llvm-objdump falls back to raw data directives when it cannot decode a word.
# None of the shipped kernels do, and a kernel that did would be silently
# under-inspected, so it is treated as a failure rather than ignored.
_UNDECODED = re.compile(r"^\t\.(?:long|word|short|byte)\b", re.MULTILINE)

# Canonical decode of the packed StaggerU argument (declareStaggerParms):
# the stride shift, the mapping and the value itself, all masked out of the
# same register.
_PACKED_MASKS = frozenset({"0x1f00", "0xe000", "0xff"})

# How far apart the two halves of a wrap site may drift.  The generator emits
# them adjacently, but the scheduler interleaves MFMA and LDS traffic between
# them, so the search is bounded rather than adjacent.
_CSELECT_PAIR_WINDOW = 64
_CARRY_WINDOW = 32
_COMPARE_WINDOW = 48


def _defaultStaggerU() -> int:
    """StaggerU applied when a custom kernel omits the key.

    Read from defaultBenchmarkCommonParameters rather than hard-coded: the
    default is 32 (stagger on), not 0, so treating an absent key as "no
    stagger" would silently exempt a quarter of the kernels from these checks.
    """
    for entry in defaultBenchmarkCommonParameters:
        if "StaggerU" in entry:
            return entry["StaggerU"][0]
    raise AssertionError("StaggerU missing from defaultBenchmarkCommonParameters")


DEFAULT_STAGGERU = _defaultStaggerU()
DEFAULT_SUPPORT_CUSTOM_STAGGERU = defaultInternalSupportParams["SupportCustomStaggerU"]

Instruction = Tuple[str, Tuple[str, ...]]


@dataclass(frozen=True)
class WrapSite:
    """An in-loop conditional wrap of a buffer descriptor.

    ``low``/``high`` index the two ``s_cselect_b32`` instructions, ``add``
    indexes the ``s_add_u32`` that applies the chosen delta, and ``srdBase`` is
    the first register of the buffer descriptor being advanced.
    """

    low: int
    high: int
    add: int
    srdBase: int


@dataclass(frozen=True)
class KernelCode:
    name: str
    arch: str
    numInstructions: int
    wrapSites: Tuple[WrapSite, ...]
    packedDecodeRegisters: FrozenSet[int]

    @property
    def staggers(self) -> bool:
        return bool(self.wrapSites)

    @property
    def decodesPackedArgument(self) -> bool:
        return bool(self.packedDecodeRegisters)


@dataclass(frozen=True)
class KernelMetadata:
    name: str
    declaredStaggerU: Optional[int]  # None when the key is absent
    supportsCustomStaggerU: bool

    @property
    def effectiveStaggerU(self) -> int:
        if self.declaredStaggerU is None:
            return DEFAULT_STAGGERU
        return self.declaredStaggerU


def _readMetadata(name: str, directory: Optional[str] = None) -> KernelMetadata:
    config = (
        readCustomKernelConfig(name)
        if directory is None
        else readCustomKernelConfig(name, directory)
    )
    internal = config.get("InternalSupportParams", {})
    return KernelMetadata(
        name=name,
        declaredStaggerU=config.get("StaggerU"),
        supportsCustomStaggerU=internal.get(
            "SupportCustomStaggerU", DEFAULT_SUPPORT_CUSTOM_STAGGERU
        ),
    )


def _disassemble(source: str, arch: str, workDir: str, stem: str) -> str:
    """Assemble ``source`` for ``arch`` and return its disassembly."""
    asmPath = os.path.join(workDir, stem + ".s")
    objPath = os.path.join(workDir, stem + ".o")
    with open(asmPath, "w") as f:
        f.write(source)

    assemble = subprocess.run(
        [CLANG, "-x", "assembler", "-target", "amdgcn-amd-amdhsa", "-mcpu=" + arch,
         "-c", asmPath, "-o", objPath],
        capture_output=True,
        text=True,
    )
    if assemble.returncode != 0:
        raise AssertionError(f"failed to assemble {stem} for {arch}:\n{assemble.stderr}")

    disassemble = subprocess.run(
        [OBJDUMP, "-d", "--mcpu=" + arch, objPath], capture_output=True, text=True
    )
    if disassemble.returncode != 0:
        raise AssertionError(f"failed to disassemble {stem} for {arch}:\n{disassemble.stderr}")

    undecoded = _UNDECODED.search(disassemble.stdout)
    if undecoded is not None:
        raise AssertionError(
            f"{stem}: llvm-objdump could not decode part of the kernel "
            f"({undecoded.group(0).strip()}); the stagger checks would be inspecting "
            f"an incomplete instruction stream"
        )
    return disassemble.stdout


def _toolchainHandlesAmdgcn() -> bool:
    """Whether the tools that were found can actually round-trip AMDGCN assembly.

    A clang picked up off PATH is not necessarily one with the AMDGPU backend,
    and skipping is the right answer there rather than reporting every shipped
    kernel as broken.
    """
    if CLANG is None or OBJDUMP is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="staggeru-probe-") as workDir:
            return "s_endpgm" in _disassemble(_PROBE_KERNEL, "gfx942", workDir, "probe")
    except (OSError, AssertionError):
        return False


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _toolchainHandlesAmdgcn(),
        reason="needs an AMDGCN-capable LLVM assembler and llvm-objdump to disassemble "
        "the shipped custom kernels; set ROCM_PATH or put ROCm's clang and llvm-objdump "
        "on PATH",
    ),
]


def _parseInstructions(disassembly: str) -> List[Instruction]:
    instructions = []
    for line in disassembly.splitlines():
        match = _DISASM_LINE.match(line)
        if match is None:
            continue
        operands = match.group(2) or ""
        instructions.append(
            (match.group(1), tuple(op.strip() for op in operands.split(",") if op.strip()))
        )
    return instructions


def _sgpr(operand: str) -> Optional[int]:
    match = re.fullmatch(r"s(\d+)", operand)
    return int(match.group(1)) if match else None


def _bufferDescriptorBases(instructions: Sequence[Instruction]) -> FrozenSet[int]:
    """Registers that start a buffer descriptor used by a memory instruction."""
    bases = set()
    for mnemonic, operands in instructions:
        if not mnemonic.startswith(("buffer_", "tbuffer_")):
            continue
        for operand in operands:
            match = re.search(r"s\[(\d+):(\d+)\]", operand)
            if match and int(match.group(2)) - int(match.group(1)) == 3:
                bases.add(int(match.group(1)))
    return frozenset(bases)


def _findPackedDecodeRegisters(instructions: Sequence[Instruction]) -> FrozenSet[int]:
    """Registers that get the full 0x1f00 / 0xe000 / 0xff unpack applied to them."""
    masksByRegister: Dict[int, set] = {}
    for mnemonic, operands in instructions:
        if mnemonic != "s_and_b32" or len(operands) != 3:
            continue
        # Either operand order: llvm-objdump prints the literal on whichever
        # side the encoding puts it.
        for mask, source in ((operands[2], operands[1]), (operands[1], operands[2])):
            if mask.lower() not in _PACKED_MASKS:
                continue
            register = _sgpr(source)
            if register is not None:
                masksByRegister.setdefault(register, set()).add(mask.lower())
    return frozenset(r for r, masks in masksByRegister.items() if masks >= _PACKED_MASKS)


def _nearestPrecedingCompare(
    instructions: Sequence[Instruction], index: int
) -> Optional[str]:
    for k in range(index - 1, max(-1, index - _COMPARE_WINDOW), -1):
        mnemonic = instructions[k][0]
        if mnemonic.startswith("s_cmp"):
            return mnemonic
    return None


def _findStaggerWrapSites(instructions: Sequence[Instruction]) -> Tuple[WrapSite, ...]:
    """Locate every in-loop conditional wrap of a buffer descriptor.

    The shape being matched, from ``globalReadIncrement``:

        s_cmp_eq_u32  <loop counter>, <StaggerUIter>
        s_cselect_b32 <tmp>,   <WrapU+0>, <GlobalReadIncs>
        s_cselect_b32 <tmp+1>, <WrapU+1>, 0
        s_add_u32     <Srd+0>, <Srd+0>, <tmp>
        s_addc_u32    <Srd+1>, <Srd+1>, <tmp+1>

    The unconditional pre-loop stagger offset is deliberately not matched: on
    its own it shifts where a workgroup starts but not the order in which the
    same K range is summed, and it is the conditional negative increment that
    makes the rotation observable.
    """
    bases = _bufferDescriptorBases(instructions)
    sites = []
    count = len(instructions)

    for i, (mnemonic, operands) in enumerate(instructions):
        if mnemonic != "s_cselect_b32" or len(operands) != 3:
            continue
        low = _sgpr(operands[0])
        if low is None:
            continue
        if _nearestPrecedingCompare(instructions, i) != "s_cmp_eq_u32":
            continue

        # High half: same conditional, next register up, zero when not wrapping.
        highIndex = None
        for j in range(i + 1, min(count, i + _CSELECT_PAIR_WINDOW)):
            mnemonicJ, operandsJ = instructions[j]
            if mnemonicJ != "s_cselect_b32" or len(operandsJ) != 3:
                continue
            if _sgpr(operandsJ[0]) == low + 1 and operandsJ[2] == "0":
                highIndex = j
                break
        if highIndex is None:
            continue

        # The chosen delta must land on a buffer descriptor base before the
        # temporary is reused for anything else.
        addIndex = None
        srdBase = None
        for j in range(highIndex + 1, count):
            mnemonicJ, operandsJ = instructions[j]
            if mnemonicJ == "s_add_u32" and len(operandsJ) == 3:
                destination = _sgpr(operandsJ[0])
                if (
                    destination is not None
                    and destination == _sgpr(operandsJ[1])
                    and _sgpr(operandsJ[2]) == low
                    and destination in bases
                ):
                    addIndex = j
                    srdBase = destination
                    break
            if operandsJ and not mnemonicJ.startswith("s_cmp") and _sgpr(operandsJ[0]) == low:
                break  # temporary overwritten; this cselect fed something else
        if addIndex is None:
            continue

        for j in range(addIndex + 1, min(count, addIndex + _CARRY_WINDOW)):
            mnemonicJ, operandsJ = instructions[j]
            if mnemonicJ != "s_addc_u32" or len(operandsJ) != 3:
                continue
            if (
                _sgpr(operandsJ[0]) == srdBase + 1
                and _sgpr(operandsJ[1]) == srdBase + 1
                and _sgpr(operandsJ[2]) == low + 1
            ):
                sites.append(WrapSite(low=i, high=highIndex, add=addIndex, srdBase=srdBase))
                break

    return tuple(sites)


def _analyzeSource(name: str, source: str, workDir: str) -> KernelCode:
    target = _TARGET.search(source)
    if target is None:
        raise AssertionError(f"{name}: no .amdgcn_target directive, cannot pick an -mcpu")
    arch = target.group(1)
    instructions = _parseInstructions(_disassemble(source, arch, workDir, name))
    if not instructions:
        raise AssertionError(f"{name}: disassembly contained no instructions")
    return KernelCode(
        name=name,
        arch=arch,
        numInstructions=len(instructions),
        wrapSites=_findStaggerWrapSites(instructions),
        packedDecodeRegisters=_findPackedDecodeRegisters(instructions),
    )


@lru_cache(maxsize=None)
def analyzeAllKernels() -> Dict[str, KernelCode]:
    """Disassemble and analyze every shipped custom kernel, once per session.

    The kernels are assembled unmodified, straight from the shipped ``.s``
    files, into a temp directory that is discarded once the instruction streams
    have been reduced to facts.
    """
    with tempfile.TemporaryDirectory(prefix="staggeru-disasm-") as workDir:

        def analyze(name: str) -> KernelCode:
            return _analyzeSource(name, getCustomKernelContents(name), workDir)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            return {code.name: code for code in pool.map(analyze, KERNEL_NAMES)}


@lru_cache(maxsize=None)
def readAllMetadata() -> Dict[str, KernelMetadata]:
    return {name: _readMetadata(name) for name in KERNEL_NAMES}


def clampCannotReach(code: KernelCode, metadata: KernelMetadata) -> Optional[str]:
    """Why the host clamp fails to stop this kernel staggering, or None.

    This is the property the launch gate depends on, expressed over a single
    kernel so it can be run against a mutated declaration as easily as against
    a shipped one.
    """
    if not code.staggers:
        return None

    if metadata.supportsCustomStaggerU:
        if not code.decodesPackedArgument:
            return (
                f"{code.name} declares SupportCustomStaggerU: True, so the launch gate "
                f"admits it and relies on the host clamp, but its {len(code.wrapSites)} "
                f"in-loop wrap site(s) run without the canonical unpack of the packed "
                f"StaggerU argument (s_and_b32 against "
                f"{', '.join(sorted(_PACKED_MASKS))}).  Writing zero into the argument "
                f"cannot reach a StaggerU the kernel never reads, so the kernel would "
                f"keep staggering and must declare SupportCustomStaggerU: False"
            )
        return None

    if metadata.effectiveStaggerU == 0:
        return (
            f"{code.name} declares SupportCustomStaggerU: False with StaggerU: 0, which "
            f"the launch gate reads as 'this kernel does not stagger' and admits without "
            f"a clamp, but the disassembly contains {len(code.wrapSites)} in-loop "
            f"conditional wrap site(s) at buffer descriptor(s) "
            f"{sorted({site.srdBase for site in code.wrapSites})}.  Nothing can stop this "
            f"kernel rotating its K-loop start position, so uniform summation order would "
            f"be silently violated"
        )
    return None


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_every_custom_kernel_disassembles(name):
    """Coverage guard: no kernel may quietly drop out of the checks below.

    The tests this replaces skipped 98 of 119 kernels because their bodies are
    pre-assembled ``.long`` blobs.  If assembling or disassembling one ever
    starts failing, that must be a failure rather than a silent gap.
    """
    code = analyzeAllKernels()[name]
    assert code.numInstructions > 0
    assert code.arch.startswith("gfx"), f"{name}: unexpected target {code.arch}"


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_declared_stagger_is_present_in_the_machine_code(name):
    """A non-zero declared StaggerU must correspond to real staggering.

    Direction one of the invariant.  A kernel whose metadata claims it staggers
    but whose code does not is not a safety hole, but it is metadata drift: the
    gate would refuse solutions it never needed to refuse, and the declaration
    can no longer be trusted in the other direction either.
    """
    code = analyzeAllKernels()[name]
    metadata = readAllMetadata()[name]
    if metadata.effectiveStaggerU == 0:
        return

    declaration = (
        f"StaggerU: {metadata.declaredStaggerU}"
        if metadata.declaredStaggerU is not None
        else f"no StaggerU key, so the default of {DEFAULT_STAGGERU}"
    )
    assert code.staggers, (
        f"{name} declares {declaration}, but its disassembly contains no in-loop "
        f"conditional wrap: no s_cselect_b32 pair feeding a buffer-descriptor add. "
        f"The kernel cannot rotate its K-loop start position, so the declared "
        f"StaggerU no longer describes the shipped code"
    )


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_compiled_stagger_stays_reachable_by_the_host_clamp(name):
    """A kernel that staggers must be one the launch gate can stop.

    Direction two, and the one the gate's correctness rests on: either the
    kernel reads StaggerU from the packed argument the host clamps, or it
    declares a non-zero StaggerU so the gate refuses it outright.
    """
    code = analyzeAllKernels()[name]
    metadata = readAllMetadata()[name]
    unreachable = clampCannotReach(code, metadata)
    assert unreachable is None, unreachable


# Kernels that declare StaggerU: 0 and stagger anyway.  Their assembly was
# generated with staggering enabled and the declaration edited down afterwards,
# so the declared value understates the code.  They are safe today only because
# each one inherits SupportCustomStaggerU: True and does unpack the runtime
# argument, which is what test_compiled_stagger_stays_reachable_by_the_host_clamp
# verifies; the gate admits them and the host clamp really does zero their
# StaggerU.  Pinned here so a new one has to be looked at by a human rather than
# joining the exception quietly.
STAGGERS_DESPITE_DECLARING_ZERO = frozenset(
    {
        "Custom_Cijk_Ailk_Bjlk_S_MX_B_BIAS_HA_S_SAV_NTD_SK3_UserArgs_MT256x256x32_MI16x16x1_shortname0_gfx950",
        "Custom_Cijk_Ailk_Bljk_S_MX_B_BIAS_HA_S_SAV_NTD_SK3_UserArgs_MT256x256x32_MI16x16x1_shortname0_gfx950",
        "Custom_Cijk_Alik_Bljk_S_MX_B_BIAS_HA_S_SAV_NTD_SK3_UserArgs_MT256x256x32_MI16x16x1_shortname0_gfx950",
        "Custom_Cijk_Alik_Bljk_BBS_BH_MT256x256x64_MI16x16x1_UserArgs_shortname1_gfx950",
    }
)

# The shipped population, as reconciled against the disassembly.  Pinned so that
# adding or retuning a custom kernel forces the reconciliation to be redone
# rather than shifting the ground truth underneath the gate.
EXPECTED_CENSUS = {
    "kernels": 119,
    # Explicit non-zero StaggerU: 24 at 8 and 4 at 4.
    "declaredNonZero": 28,
    # Of those, the ones with no packed unpack at all: StaggerU is a literal
    # baked into the code, which is exactly why they declare
    # SupportCustomStaggerU: False and why the gate refuses them.
    "declaredNonZeroWithLiteralStagger": 24,
    "declaredNonZeroReadingPackedArgument": 4,
    "declaredZero": 60,
    # No StaggerU key at all, so they inherit the default of 32.
    "undeclared": 31,
}


def test_shipped_population_matches_the_reconciled_ground_truth():
    """The census the gate's safety argument was reviewed against."""
    codes = analyzeAllKernels()
    metadata = readAllMetadata()
    declaredNonZero = [
        name
        for name, meta in metadata.items()
        if meta.declaredStaggerU is not None and meta.declaredStaggerU != 0
    ]
    census = {
        "kernels": len(KERNEL_NAMES),
        "declaredNonZero": len(declaredNonZero),
        "declaredNonZeroWithLiteralStagger": sum(
            1 for name in declaredNonZero if not codes[name].decodesPackedArgument
        ),
        "declaredNonZeroReadingPackedArgument": sum(
            1 for name in declaredNonZero if codes[name].decodesPackedArgument
        ),
        "declaredZero": sum(1 for meta in metadata.values() if meta.declaredStaggerU == 0),
        "undeclared": sum(1 for meta in metadata.values() if meta.declaredStaggerU is None),
    }
    assert census == EXPECTED_CENSUS, (
        f"the shipped custom-kernel population no longer matches the set the uniform "
        f"summation order gate was reviewed against: {census} != {EXPECTED_CENSUS}. "
        f"Re-reconcile the new kernels against the disassembly and update EXPECTED_CENSUS"
    )
    assert all(codes[name].staggers for name in declaredNonZero), (
        "every kernel declaring a non-zero StaggerU used to contain the in-loop wrap; "
        "one no longer does"
    )


def test_kernels_that_stagger_despite_declaring_zero_are_the_known_ones():
    """The one place declared metadata understates the shipped code."""
    codes = analyzeAllKernels()
    metadata = readAllMetadata()
    observed = {
        name
        for name, meta in metadata.items()
        if meta.declaredStaggerU == 0 and codes[name].staggers
    }
    assert observed == STAGGERS_DESPITE_DECLARING_ZERO, (
        f"the set of kernels that declare StaggerU: 0 and stagger anyway changed: "
        f"{sorted(observed ^ STAGGERS_DESPITE_DECLARING_ZERO)}.  A new one is only safe "
        f"if it also declares SupportCustomStaggerU: True and unpacks the runtime "
        f"argument; confirm that and add it to STAGGERS_DESPITE_DECLARING_ZERO"
    )


def test_the_sibling_control_pair_is_told_apart():
    """Two kernels that differ in little but their StaggerU.

    ``shortname0`` declares 8 and ``shortname12`` declares 0, so a detector that
    fired on ambient buffer-descriptor arithmetic rather than on the wrap itself
    would light up both.
    """
    codes = analyzeAllKernels()
    staggering = "Custom_Cijk_Ailk_Bljk_HHS_BH_Bias_GG_AS_SAV_UserArgs_shortname0_gfx942"
    plain = "Custom_Cijk_Ailk_Bljk_HHS_BH_Bias_GG_AS_SAV_UserArgs_shortname12_gfx942"
    assert codes[staggering].staggers, f"{staggering} declares StaggerU: 8 and does wrap"
    assert not codes[plain].staggers, (
        f"{plain} declares StaggerU: 0 and has no wrap, so a detector reporting one is "
        f"matching ordinary buffer-descriptor arithmetic"
    )


def _ablate(source: str) -> Tuple[str, int]:
    """Turn a kernel's wrap selects into unconditional increments.

    Replaces ``s_cselect_b32 dst, wrap, inc`` with ``s_mov_b32 dst, inc`` at the
    wrap sites only, keeping the instruction count identical, so the kernel
    always takes the forward increment and never rotates.
    """
    pattern = re.compile(
        r"^s_cselect_b32\s+(?P<dst>[^,]+),\s*[^,]+,\s*(?P<inc>[^/\n]+?)\s*"
        r"(?P<comment>//\s*inc(?:Lower|Upper) <- \?.*)$",
        re.MULTILINE,
    )
    ablated, count = pattern.subn(
        lambda m: f"s_mov_b32 {m.group('dst')}, {m.group('inc')} {m.group('comment')}", source
    )
    return ablated, count


def test_wrap_detector_goes_quiet_when_the_wrap_is_ablated():
    """Negative control: the detector tracks the conditional wrap, not the kernel.

    A detector that fired on everything would pass every check above.  This
    takes a kernel that does wrap, rewrites only its wrap selects into
    unconditional moves in a temp copy of the source, reassembles, and requires
    the site count to fall to zero with the rest of the instruction stream
    untouched.
    """
    codes = analyzeAllKernels()
    candidates = [
        name
        for name in KERNEL_NAMES
        if codes[name].staggers and _ablate(getCustomKernelContents(name))[1] > 0
    ]
    assert candidates, (
        "no shipped kernel has both a detected wrap and readable wrap selects to "
        "ablate; the negative control can no longer run"
    )
    # One source line can be several sites: the readable kernels emit the wrap
    # from a macro instantiated per tensor, so the counts are not equal.
    name = candidates[0]
    ablated, _ = _ablate(getCustomKernelContents(name))

    with tempfile.TemporaryDirectory(prefix="staggeru-ablation-") as workDir:
        mutated = _analyzeSource(name, ablated, workDir)

    assert not mutated.wrapSites, (
        f"{name}: the detector still reports {len(mutated.wrapSites)} wrap site(s) after "
        f"every wrap select was rewritten to an unconditional move, so it is matching "
        f"something other than the conditional wrap"
    )
    assert mutated.numInstructions == codes[name].numInstructions, (
        "the ablation changed more than the wrap selects"
    )
    assert mutated.packedDecodeRegisters == codes[name].packedDecodeRegisters, (
        "the ablation disturbed the packed-argument decode"
    )


def _snippet(text: str) -> List[Instruction]:
    """Parse a hand-written instruction list the way disassembly is parsed."""
    instructions = []
    for line in text.strip().splitlines():
        mnemonic, _, operands = line.strip().partition(" ")
        instructions.append(
            (mnemonic, tuple(op.strip() for op in operands.split(",") if op.strip()))
        )
    return instructions


WRAP_SNIPPET = """
buffer_load_dword v0, v1, s[48:51], 0 offen
s_cmp_eq_u32 s11, s15
s_cselect_b32 s82, s60, s47
s_cselect_b32 s83, s61, 0
s_add_u32 s48, s48, s82
s_addc_u32 s49, s49, s83
"""


def test_wrap_detector_needs_every_part_of_the_signature():
    """Each half of the mechanism is load-bearing for a match."""
    assert len(_findStaggerWrapSites(_snippet(WRAP_SNIPPET))) == 1

    # Unconditional increment: the forward step every iteration takes.
    assert not _findStaggerWrapSites(
        _snippet(WRAP_SNIPPET.replace("s_cselect_b32 s82, s60, s47", "s_mov_b32 s82, s47"))
    )
    # Selected on something other than "is this the wrap iteration?".
    assert not _findStaggerWrapSites(
        _snippet(WRAP_SNIPPET.replace("s_cmp_eq_u32 s11, s15", "s_cmp_lt_u32 s11, s15"))
    )
    # A conditional 64-bit value that never reaches a buffer descriptor.
    assert not _findStaggerWrapSites(
        _snippet(WRAP_SNIPPET.replace("s_add_u32 s48, s48, s82", "s_add_u32 s70, s70, s82"))
    )
    # The register pair is not a buffer descriptor at all.
    assert not _findStaggerWrapSites(
        _snippet(WRAP_SNIPPET.replace("buffer_load_dword v0, v1, s[48:51], 0 offen", "s_nop 0"))
    )
    # Only the low half is conditional, so no 64-bit delta is selected.
    assert not _findStaggerWrapSites(
        _snippet(WRAP_SNIPPET.replace("s_cselect_b32 s83, s61, 0", "s_mov_b32 s83, 0"))
    )


def _writeKernelCopy(source: str, name: str, directory) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + ".s")).write_text(source)
    return str(directory)


def test_a_lying_declaration_is_caught(tmp_path):
    """Mutation control: the checks are not vacuously satisfiable.

    A kernel whose machine code really does stagger is given, in a temp copy of
    its ``.s``, each of the two declarations that would let the launch gate
    admit it without a working clamp.  Both must be reported, and the shipped
    declaration must not be.
    """
    codes = analyzeAllKernels()
    metadata = readAllMetadata()
    name = next(
        n
        for n in KERNEL_NAMES
        if codes[n].staggers
        and not codes[n].decodesPackedArgument
        and not metadata[n].supportsCustomStaggerU
        and metadata[n].effectiveStaggerU != 0
    )
    code = codes[name]
    assert clampCannotReach(code, metadata[name]) is None, "the shipped declaration is honest"

    source = getCustomKernelContents(name)

    claimsNoStagger, replaced = re.subn(
        r"^(\s*StaggerU:\s*)\d+\s*$", r"\g<1>0", source, count=1, flags=re.MULTILINE
    )
    assert replaced == 1, f"{name}: no StaggerU declaration to flip"
    mutated = _readMetadata(name, _writeKernelCopy(claimsNoStagger, name, tmp_path / "zeroed"))
    assert mutated.declaredStaggerU == 0 and not mutated.supportsCustomStaggerU
    reason = clampCannotReach(code, mutated)
    assert reason is not None and "does not stagger" in reason, (
        "a kernel that staggers with a compiled-in StaggerU was accepted after its "
        "declaration was flipped to zero"
    )

    claimsRuntimeControl, replaced = re.subn(
        r"^(\s*SupportCustomStaggerU:\s*)False\s*$",
        r"\g<1>True",
        source,
        count=1,
        flags=re.MULTILINE,
    )
    assert replaced == 1, f"{name}: no SupportCustomStaggerU declaration to flip"
    mutated = _readMetadata(
        name, _writeKernelCopy(claimsRuntimeControl, name, tmp_path / "runtime")
    )
    assert mutated.supportsCustomStaggerU
    reason = clampCannotReach(code, mutated)
    assert reason is not None and "cannot reach a StaggerU the kernel never reads" in reason, (
        "a kernel that bakes StaggerU in as a literal was accepted after claiming the "
        "host clamp could configure it"
    )
