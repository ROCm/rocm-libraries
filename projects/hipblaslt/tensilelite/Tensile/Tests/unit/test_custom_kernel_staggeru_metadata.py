# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Disassembly-backed checks that custom-kernel StaggerU metadata is truthful.

The host-side launch gate for uniform summation order (``checkUniformSummationOrder``
in ``ContractionSolution.cpp``) currently refuses handwritten custom kernels, and
the StaggerU clamp still reasons from declared metadata for every other solution.
It clamps StaggerU by writing zero into a bitfield of a packed kernel argument, and
refuses any solution declaring ``SupportCustomStaggerU: False`` with a non-zero
``StaggerU``.  These tests pin that metadata against the instructions the kernels
execute, so a later admission path cannot trust a lying declaration.

Reading the ``.s`` files cannot do it: 98 of the 119 shipped custom kernels are
pre-assembled ``.long`` blobs with no readable mnemonics, and every kernel that
declares a non-zero StaggerU -- exactly the set the gate's safety argument turns
on -- is among them.  So each kernel is assembled for its own ``.amdgcn_target``
and disassembled, which also sees through the macros and ``.set`` aliases the
readable kernels are written in.

Of the two halves the mechanism has (``KernelWriterAssembly.declareStaggerParms``
and ``calculateStagger``), only the in-loop conditional wrap of a buffer
descriptor -- shaped as in ``_findStaggerWrapSites`` below -- is decisive and
recognisable without symbols.  A kernel with no such site cannot rotate its
K-loop start position, whatever its metadata says.

Scope note: these checks establish that the canonical packed decode is present
in a kernel that staggers, not that the decoded value reaches every individual
wrap site; proving the latter would need a full dataflow analysis.
"""

import os
import re
import shutil
import subprocess
import sys
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


_ROCM_LLVM_BIN = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "llvm", "bin")


def _findTool(name: str) -> Optional[str]:
    """Prefer the ROCm LLVM toolchain, fall back to whatever is on PATH."""
    return shutil.which(name, path=_ROCM_LLVM_BIN) or shutil.which(name)


CLANG = _findTool("clang")
OBJDUMP = _findTool("llvm-objdump")

_PROBE_KERNEL = '.amdgcn_target "amdgcn-amd-amdhsa--gfx942"\n.text\ns_endpgm\n'

KERNEL_NAMES = getAllCustomKernelNames()

# Assembling and disassembling all 119 kernels is a few seconds of wall time
# spread over a thread pool, so no separate slow marker: this stays in the unit
# suite where a change to a custom kernel will actually run it.
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 2)

_TARGET = re.compile(r'^\.amdgcn_target\s+"amdgcn-amd-amdhsa--([a-z0-9]+)', re.MULTILINE)

# One decoded instruction, recognised by the address/encoding comment
# llvm-objdump appends; labels, section headers and blank lines lack it.
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


# StaggerU applied when a kernel omits the key.  Read rather than hard-coded:
# the default is 32 (stagger on), not 0, so treating an absent key as "no
# stagger" would silently exempt a quarter of the kernels from these checks.
DEFAULT_STAGGERU = next(
    (e["StaggerU"][0] for e in defaultBenchmarkCommonParameters if "StaggerU" in e), None
)
assert DEFAULT_STAGGERU is not None, "StaggerU missing from defaultBenchmarkCommonParameters"

DEFAULT_SUPPORT_CUSTOM_STAGGERU = defaultInternalSupportParams["SupportCustomStaggerU"]

Instruction = Tuple[str, Tuple[str, ...]]


@dataclass(frozen=True)
class KernelCode:
    name: str
    arch: str
    numInstructions: int
    wrapSites: Tuple[int, ...]  # buffer-descriptor base of each in-loop wrap
    packedDecodeRegisters: FrozenSet[int]

    @property
    def staggers(self) -> bool:
        return bool(self.wrapSites)

    @property
    def decodesPackedArgument(self) -> bool:
        return bool(self.packedDecodeRegisters)


@dataclass(frozen=True)
class KernelMetadata:
    declaredStaggerU: Optional[int]  # None when the key is absent
    supportsCustomStaggerU: bool

    @property
    def effectiveStaggerU(self) -> int:
        if self.declaredStaggerU is None:
            return DEFAULT_STAGGERU
        return self.declaredStaggerU


def _readMetadata(name: str, directory: Optional[str] = None) -> KernelMetadata:
    config = readCustomKernelConfig(name, directory)
    internal = config.get("InternalSupportParams", {})
    return KernelMetadata(
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
            f"{stem}: llvm-objdump could not decode {undecoded.group(0).strip()}; the "
            f"stagger checks would be inspecting an incomplete instruction stream"
        )
    return disassemble.stdout


def _toolchainHandlesAmdgcn() -> bool:
    # A clang picked up off PATH need not have the AMDGPU backend, and skipping
    # is the right answer there rather than reporting every kernel as broken.
    if CLANG is None or OBJDUMP is None:
        return False
    try:
        with tempfile.TemporaryDirectory(prefix="staggeru-probe-") as workDir:
            return "s_endpgm" in _disassemble(_PROBE_KERNEL, "gfx942", workDir, "probe")
    except (OSError, AssertionError):
        return False


def _whyDisassemblyRequired() -> Optional[str]:
    """Why an unusable toolchain is an error here rather than a reason to skip.

    These checks are the only thing holding the gate's metadata to the shipped
    code, so skipping them where they were meant to run is the worst outcome
    available: a green suite that verified nothing.  A machine carrying ROCm's
    own LLVM is taken to be such a place.  ``TENSILE_REQUIRE_AMDGCN_DISASM``
    overrides the inference in both directions, so CI can demand the checks
    whatever its layout, and a deliberately toolchain-less run can opt out.
    """
    forced = os.environ.get("TENSILE_REQUIRE_AMDGCN_DISASM")
    if forced:
        return None if forced == "0" else f"TENSILE_REQUIRE_AMDGCN_DISASM={forced} is set"
    if all(
        shutil.which(tool, path=_ROCM_LLVM_BIN) is not None
        for tool in ("clang", "llvm-objdump")
    ):
        return f"ROCm's own LLVM is installed at {_ROCM_LLVM_BIN}"
    return None


_TOOLCHAIN_USABLE = _toolchainHandlesAmdgcn()
_REQUIRED_BECAUSE = None if _TOOLCHAIN_USABLE else _whyDisassemblyRequired()

if _REQUIRED_BECAUSE is not None:
    raise RuntimeError(
        f"the custom-kernel StaggerU checks cannot run, and {_REQUIRED_BECAUSE}, so "
        f"this is an environment where they are expected to: assembling a probe "
        f"kernel for gfx942 with clang={CLANG} and objdump={OBJDUMP} did not produce "
        f"readable AMDGCN. Failing rather than skipping, because a silent skip would "
        f"leave custom-kernel StaggerU metadata unpinned against disassembly. Set "
        f"TENSILE_REQUIRE_AMDGCN_DISASM=0 to skip deliberately."
    )

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not _TOOLCHAIN_USABLE,
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


def _findStaggerWrapSites(instructions: Sequence[Instruction]) -> Tuple[int, ...]:
    """Buffer-descriptor base of every in-loop conditional wrap, one per site.

    The shape being matched, from ``globalReadIncrement``:

        s_cmp_eq_u32  <loop counter>, <StaggerUIter>
        s_cselect_b32 <tmp>,   <WrapU+0>, <GlobalReadIncs>
        s_cselect_b32 <tmp+1>, <WrapU+1>, 0
        s_add_u32     <Srd+0>, <Srd+0>, <tmp>
        s_addc_u32    <Srd+1>, <Srd+1>, <tmp+1>

    The unconditional pre-loop stagger offset is deliberately not matched: on
    its own it shifts where a workgroup starts but not the order in which the
    same K range is summed.
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
        preceding = (instructions[k][0] for k in range(i - 1, max(-1, i - _COMPARE_WINDOW), -1))
        if next((m for m in preceding if m.startswith("s_cmp")), None) != "s_cmp_eq_u32":
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
                sites.append(srdBase)
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
    """Disassemble and analyze every shipped custom kernel, once per session."""
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

    Per-kernel so it can be run against a mutated declaration as easily as
    against a shipped one.
    """
    if not code.staggers:
        return None

    if metadata.supportsCustomStaggerU:
        if not code.decodesPackedArgument:
            return (
                f"{code.name} declares SupportCustomStaggerU: True, so a clamp-based "
                f"admission would rely on the packed argument, but its {len(code.wrapSites)} wrap site(s) run "
                f"without the canonical unpack (s_and_b32 against "
                f"{', '.join(sorted(_PACKED_MASKS))}): the clamp cannot reach a StaggerU "
                f"the kernel never reads"
            )
        return None

    if metadata.effectiveStaggerU == 0:
        return (
            f"{code.name} declares SupportCustomStaggerU: False with StaggerU: 0, so "
            f"metadata claims the kernel does not stagger, but "
            f"the disassembly has {len(code.wrapSites)} wrap site(s) at buffer "
            f"descriptor(s) {sorted(set(code.wrapSites))}"
        )
    return None


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_every_custom_kernel_disassembles(name):
    """Coverage guard: no kernel may quietly drop out of the checks below."""
    code = analyzeAllKernels()[name]
    assert code.numInstructions > 0
    assert code.arch.startswith("gfx"), f"{name}: unexpected target {code.arch}"


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_declared_stagger_is_present_in_the_machine_code(name):
    """Direction one: a non-zero declared StaggerU must be real staggering.

    Metadata claiming a stagger the code does not have is not a safety hole,
    but the gate would refuse solutions it never needed to, and the
    declaration can no longer be trusted in the other direction either.
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
        f"{name} declares {declaration}, but its disassembly has no in-loop conditional "
        f"wrap: no s_cselect_b32 pair feeding a buffer-descriptor add, so the declared "
        f"StaggerU no longer describes the shipped code"
    )


@pytest.mark.parametrize("name", KERNEL_NAMES)
def test_compiled_stagger_stays_reachable_by_the_host_clamp(name):
    """Direction two, the one the gate's correctness rests on: a kernel that
    staggers either reads StaggerU from the packed argument the host clamps, or
    declares a non-zero StaggerU so the gate refuses it outright."""
    code = analyzeAllKernels()[name]
    metadata = readAllMetadata()[name]
    unreachable = clampCannotReach(code, metadata)
    assert unreachable is None, unreachable


# Kernels that declare StaggerU: 0 and stagger anyway: their assembly was
# generated with staggering enabled and the declaration edited down afterwards.
# Each is safe only because it also inherits SupportCustomStaggerU: True and does
# unpack the runtime argument.  Pinned so a new one has to be looked at by a
# human rather than joining the exception quietly.
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


def _ablate(source: str) -> Tuple[str, int]:
    """Rewrite ``s_cselect_b32 dst, wrap, inc`` into ``s_mov_b32 dst, inc`` at the
    wrap sites only, keeping the instruction count identical, so the kernel
    always takes the forward increment and never rotates."""
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
    """Ablation control: the detector tracks the conditional wrap, not the kernel.

    A detector that fired on everything would pass every check above.
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


# Hand-written near misses: (what it is, line of WRAP_SNIPPET, its replacement).
NEAR_MISSES = [
    ("unconditional forward increment", "s_cselect_b32 s82, s60, s47", "s_mov_b32 s82, s47"),
    ("not selected on the wrap iteration", "s_cmp_eq_u32 s11, s15", "s_cmp_lt_u32 s11, s15"),
    ("delta never reaches a descriptor", "s_add_u32 s48, s48, s82", "s_add_u32 s70, s70, s82"),
    (
        "target pair is not a buffer descriptor",
        "buffer_load_dword v0, v1, s[48:51], 0 offen",
        "s_nop 0",
    ),
    ("only the low half is conditional", "s_cselect_b32 s83, s61, 0", "s_mov_b32 s83, 0"),
]


def test_wrap_detector_needs_every_part_of_the_signature():
    """Negative controls: every part of the mechanism is load-bearing."""
    assert len(_findStaggerWrapSites(_snippet(WRAP_SNIPPET))) == 1
    for description, line, replacement in NEAR_MISSES:
        variant = WRAP_SNIPPET.replace(line, replacement)
        assert not _findStaggerWrapSites(_snippet(variant)), (
            f"the detector reported a wrap for a near miss with {description}"
        )


def test_a_lying_declaration_is_caught(tmp_path):
    """Mutation control: the checks are not vacuously satisfiable.

    A kernel whose machine code really does stagger is given, in a temp copy of
    its ``.s``, each of the two declarations that would claim the kernel is
    clamp-safe or non-staggering.  Both must be reported, and the shipped
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

    lies = [
        # (key flipped, pattern, replacement, what the flip must parse as, the
        #  part of the refusal it must draw)
        (
            "StaggerU",
            r"^(\s*StaggerU:\s*)\d+\s*$",
            r"\g<1>0",
            lambda m: m.declaredStaggerU == 0 and not m.supportsCustomStaggerU,
            "does not stagger",
        ),
        (
            "SupportCustomStaggerU",
            r"^(\s*SupportCustomStaggerU:\s*)False\s*$",
            r"\g<1>True",
            lambda m: m.supportsCustomStaggerU,
            "cannot reach a StaggerU the kernel never reads",
        ),
    ]

    for key, pattern, replacement, tookEffect, expected in lies:
        lie, replaced = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
        assert replaced == 1, f"{name}: no {key} declaration to flip"

        directory = tmp_path / key
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (name + ".s")).write_text(lie)
        mutated = _readMetadata(name, str(directory))
        assert tookEffect(mutated), f"{name}: the flipped {key} did not reach the metadata"

        reason = clampCannotReach(code, mutated)
        assert reason is not None and expected in reason, (
            f"{name}: flipping {key} made the declaration a lie, but the check accepted "
            f"it instead of refusing with '{expected}'"
        )


# ---------------------------------------------------------------------------
# The generated-kernel arm.
#
# Everything above is about the 119 handwritten custom kernels.  The same
# question has to be answered for generated kernels, because the uniform
# summation order gate admits a solution whose StaggerU the host clamps to 0 and
# that admission is only sound if the kernel reads StaggerU from the packed
# argument the clamp writes.  For a generated kernel that is a property of the
# code generator rather than of a checked-in file, so it is checked by running
# the generator: emit a solution that declares SupportCustomStaggerU: False with
# a non-zero StaggerU -- the shape that used to be refused outright -- and hold
# the emitted assembly to the same clampCannotReach predicate.
#
# supportsCustomStaggerU is passed as True on purpose.  Passing the solution's
# own False would take the branch that means "the gate refuses this outright",
# which is vacuous here: under the narrowed predicate the gate no longer
# refuses it, so the assertion that has to hold is the clamp-based one.
# ---------------------------------------------------------------------------

_CODEGEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "characterization", "_codegen"
)

_LOGIC_ROOT = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        *([".."] * 4),
        "library",
        "src",
        "amd_detail",
        "rocblaslt",
        "src",
        "Tensile",
        "Logic",
        "asm_full",
    )
)

# Shipped tuning logic known to contain solutions with SupportCustomStaggerU:
# False and a non-zero StaggerU.  Two files rather than one so a retune that
# drops the shape from either still leaves the check with something to assert.
COMPILED_IN_STAGGERU_LOGIC = (
    "gfx950/gfx950_id75a3/Equality/"
    "gfx950_Cijk_Ailk_Bjlk_BSS_BH_BiasS_HAS_SAV_UserArgs.yaml",
    "gfx950/gfx950_id75a3/Equality/"
    "gfx950_Cijk_Alik_Bljk_F8B8SS_BH_BiasS_HAS_SAB_SAV_UserArgs.yaml",
)


def _logicSolutions(document):
    """Solutions out of either tuning-logic schema.

    Equality logic is a mapping with a 'Solutions' key; the positional schema
    used by the StreamK and Origami files keeps them at index 5.
    """
    if isinstance(document, dict):
        return document["Solutions"]
    return document[5]


def _declaresCompiledInStaggerU(solution) -> bool:
    """The shape the CompiledInStaggerU clause used to refuse."""
    support = solution.get("InternalSupportParams", {})
    staggerU = solution.get("StaggerU", DEFAULT_STAGGERU)
    return support.get("SupportCustomStaggerU", DEFAULT_SUPPORT_CUSTOM_STAGGERU) is False and (
        staggerU != 0
    )


def _writeSingleSolutionLogic(document, solution, path):
    """A logic file holding just this solution, so emitting is cheap."""
    import copy

    import yaml

    trimmed = copy.deepcopy(document)
    only = copy.deepcopy(solution)
    only["SolutionIndex"] = 0
    if isinstance(trimmed, dict):
        trimmed["Solutions"] = [only]
        # The exact-match table indexes into the solution list this replaces, so
        # its entries have to go.  The key itself has to stay: for Equality,
        # GridBased and Range logic prepareLibraryLogicDict() reads
        # data["ExactLogic"] unconditionally, so dropping the key raises
        # KeyError.  Emptying it is what the positional branch below already
        # does to the same table, which sits at index 7.
        trimmed["ExactLogic"] = []
    else:
        trimmed[5] = [only]
        for i in range(6, len(trimmed)):
            if isinstance(trimmed[i], list):
                trimmed[i] = []
    with open(path, "w") as f:
        yaml.safe_dump(trimmed, f, default_flow_style=None, width=200)
    return path


@pytest.mark.parametrize("rel", COMPILED_IN_STAGGERU_LOGIC)
def test_generated_kernels_read_stagger_from_the_packed_argument(rel, tmp_path):
    """A generated kernel that staggers must unpack the value the host clamps.

    This is the premise the narrowed CompiledInStaggerU clause rests on:
    SupportCustomStaggerU: False means only that the host declines to write the
    packed field, leaving it 0.  It does not mean the kernel took its stagger
    from somewhere the clamp cannot reach.  If the generator ever learns to bake
    a literal StaggerU into a wrap site, this fails and the clause has to widen
    again."""
    logic = os.path.join(_LOGIC_ROOT, rel)
    if not os.path.exists(logic):
        pytest.skip(f"tuning logic tree not present: {rel}")
    if not os.path.isdir(_CODEGEN_DIR):
        pytest.skip("the code generation harness is not present")
    if _CODEGEN_DIR not in sys.path:
        sys.path.insert(0, _CODEGEN_DIR)
    codegen_harness = pytest.importorskip("codegen_harness")

    yaml = pytest.importorskip("yaml")
    with open(logic) as f:
        document = yaml.safe_load(f)
    disqualified = [s for s in _logicSolutions(document) if _declaresCompiledInStaggerU(s)]
    assert disqualified, (
        f"{rel} no longer declares SupportCustomStaggerU: False with a non-zero StaggerU, "
        f"so it cannot exercise the narrowed clause: repoint COMPILED_IN_STAGGERU_LOGIC at "
        f"logic that still does"
    )

    workDir = str(tmp_path)
    for solution in disqualified:
        path = _writeSingleSolutionLogic(
            document, solution, os.path.join(workDir, "logic-%s.yaml" % solution["SolutionIndex"])
        )
        emitted = codegen_harness.emit_kernels_from_logic(path, canonical=True)
        assert len(emitted) == 1, f"expected one kernel from {path}, got {len(emitted)}"
        name, source, err = emitted[0]
        assert err == 0 and source, f"{name}: code generation returned {err}"

        code = _analyzeSource(name, source, workDir)
        assert code.staggers, (
            f"{name} declares StaggerU: {solution.get('StaggerU')} but the generated code has no "
            f"wrap site, so this solution no longer exercises the check"
        )
        unreachable = clampCannotReach(
            code,
            KernelMetadata(
                declaredStaggerU=solution.get("StaggerU"), supportsCustomStaggerU=True
            ),
        )
        assert unreachable is None, unreachable
