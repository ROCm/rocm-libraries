################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
"""
CUOccupancy measurement utilities: ELF metadata parsing and HIP runtime query.

WHY THE ELF PASS IS NOW OPT-IN
-------------------------------
Tensile's codegen-time occupancy model originally computed CUOccupancy from the
VGPR *pool high-water mark* (checkResources), which can be higher than the actual
live registers after rocIsaPass's removeDuplicateAssignment optimization.

For large MFMA tiles on ArchAccUnifiedRegs ISAs (gfx90a/gfx942/gfx950), the
combined pool peak exceeds the physical VGPR count (512), causing CUOccupancy to
be under-reported.  Example (MT320x192x64, MIWT10_6):
  Code-gen pool:  ceil(64/8)*8 + 240 = 304  →  512//304 = 1  (WRONG)
  Actual live:    vgpr_count = 256           →  512//256 = 2  (CORRECT)
  HIP hardware:   blocks_per_CU = 2          →  CUOccupancy = 2 (CORRECT)

THE CODEGEN-TIME FIX (updateOccupancyFromScan)
----------------------------------------------
updateOccupancyFromScan() runs in kernelBody() AFTER rocIsaPass completes.  It
rescans the instruction body with regex to find the actual max VGPR/AGPR indices,
then:
  1. Calls mkb.setGprs(scanned_vgprs, scanned_agprs) — updating the kernel
     descriptor's totalVgprs = ceil(scanned/8)*8 + scanned_agprs, which becomes
     the .amdhsa_next_free_vgpr directive in the emitted .s file.
  2. Recomputes kernel["CUOccupancy"] from the same scanned values.

Because the assembler takes .amdhsa_next_free_vgpr literally (no compiler-side
VGPR reduction for -x assembler input), the assembled .o's vgpr_count equals
exactly what updateOccupancyFromScan wrote.  Reading the ELF and applying the
same occupancy formula gives the same CUOccupancy already stored in kernel[].
The ELF pass therefore cannot correct anything the scan missed: it is a faithful
readback, not an independent measurement.

RESULT: codegen-time occupancy is correct WITHOUT the ELF pass, making the ELF
pass redundant as a correction mechanism.

TWO APPROACHES (both remain available)
---------------------------------------

Approach 1: ELF metadata (GPU-free, flag --occupancy-from-elf, now OFF by default)
  Parse the AMDHSA metadata note from the assembled .o ELF file using Python's
  struct module + msgpack.  Extract vgpr_count, group_segment_fixed_size, and
  max_flat_workgroup_size.  Recompute occupancy using the same Tensile formula.
  No GPU required; overhead is O(microseconds per kernel), negligible.

  Use case: debugging only — verify that the assembled .o's vgpr_count matches
  what updateOccupancyFromScan computed.  NOT needed as a correction pass.

Approach 2: HIP runtime (needs GPU, flag --occupancy-from-hip, test-only)
  Load the linked single-arch code-object (.co.raw) via hipModuleLoad and call
  hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.  This is the definitive
  hardware query: it accounts for VGPR granularity, LDS granularity, and any
  driver-side occupancy adjustments.  Serves as an authoritative regression test
  that Tensile's formula is correct for every compiled kernel.
  Added build time: ~5-20 ms per hipModuleLoad + hipModuleGetFunction call.

  Requirement: a compatible GPU must be present (same ISA as the compiled
  kernels).  If no GPU is available or hip-python is not installed, the pass
  silently degrades to a warning and the original codegen occupancy is kept.

PIPELINE HOOK
-------------
The ELF pass hook exists in Run.py (TensileCreateLibrary) between the calls to
writeSolutionsAndKernelsTCL and passPostKernelInfoToLibrary.  It is now OPT-IN
via --occupancy-from-elf (previously it was opt-out via --no-occupancy-from-elf).

The HIP pass is NOT part of the build pipeline.  It lives in this module for use
by tests (test_occupancy_hip.py, test_occupancy_buildtime.py) that run on machines
with a compatible GPU as a CI/cross-validation mechanism.

RELATIONSHIP BETWEEN THE TWO APPROACHES (when --occupancy-from-elf is used)
----------------------------------------------------------------------------
For the currently shipped MT320x192x64 kernel (LDS=68864 B, numRegs=256):
  ELF vgpr_count  = 256  →  occ = 2   (confirms scan corrected the descriptor)
  HIP blocks/CU   = 2    →  occ = 2   (hardware confirms)

Both agree when updateOccupancyFromScan is active.  HIP adds value as an
independent validator because it measures real hardware behaviour rather than
applying the Tensile formula to values Tensile itself wrote.
"""

import collections
import os
import struct
import warnings
from math import ceil
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

# ── hip-python (optional) ─────────────────────────────────────────────────────
try:
    from hip import hip as _hip
    _HIP_AVAILABLE = True
except ImportError:
    _hip = None
    _HIP_AVAILABLE = False


# ── ELF / AMDHSA note constants ───────────────────────────────────────────────
_SHT_NOTE = 7
_NT_AMDGPU_METADATA = 32   # note type for AMDGPU AMDHSA msgpack metadata
_AMDGPU_NOTE_NAME = b"AMDGPU\x00"   # null-terminated note name


# ── HIP helpers (same API as test_occupancy_hip.py) ───────────────────────────

def _hip_check(result):
    """Raise RuntimeError on non-zero HIP error code."""
    if isinstance(result, tuple):
        err = result[0]
        if int(err) != 0:
            raise RuntimeError(f"HIP error {int(err)}")
        return result[1] if len(result) == 2 else result[1:]
    if int(result) != 0:
        raise RuntimeError(f"HIP error {int(result)}")
    return result


def query_hip_occupancy(co_path: str, kernel_func_name: str, num_threads: int) -> int:
    """Load *co_path* and return max active blocks per CU for *kernel_func_name*.

    Args:
        co_path:           Path to a single-arch code object (.co.raw or .co).
        kernel_func_name:  The kernel function name as embedded in the code object.
        num_threads:       Block size (NumThreads from the kernel parameters).

    Returns:
        Max active workgroups (blocks) per CU, equivalent to CUOccupancy for
        gfx9 architectures with 256-thread blocks (1:1 relationship).

    Raises:
        RuntimeError: If HIP returns a non-zero error code or the function is
                      not found in the module.
    """
    if not _HIP_AVAILABLE:
        raise RuntimeError("hip-python not installed; cannot query HIP occupancy")
    _hip_check(_hip.hipInit(0))
    module = _hip_check(_hip.hipModuleLoad(
        co_path.encode() if isinstance(co_path, str) else co_path
    ))
    try:
        func_name = (kernel_func_name.encode()
                     if isinstance(kernel_func_name, str)
                     else kernel_func_name)
        func = _hip_check(_hip.hipModuleGetFunction(module, func_name))
        err, blocks_per_cu = _hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            func, num_threads, 0  # dynSharedMemPerBlk=0; static LDS is in descriptor
        )
        _hip_check(err)
        return int(blocks_per_cu)
    finally:
        _hip_check(_hip.hipModuleUnload(module))


# ── ELF parsing ───────────────────────────────────────────────────────────────

class _ElfParseError(Exception):
    pass


def _parse_elf_section_headers(data: bytes):
    """Yield (sh_type, offset, size) for every ELF section.

    ELF header field offsets (see ELF spec / elf.h):

    64-bit ELF:
      Offset  Size  Field
      0       4     e_ident magic
      4       1     EI_CLASS
      5       1     EI_DATA
      6-15    10    EI_VERSION + EI_OSABI + EI_ABIVERSION + padding
      16      2     e_type
      18      2     e_machine
      20      4     e_version
      24      8     e_entry
      32      8     e_phoff
      40      8     e_shoff      ← section-header table offset
      48      4     e_flags
      52      2     e_ehsize
      54      2     e_phentsize
      56      2     e_phnum
      58      2     e_shentsize  ← section-header entry size
      60      2     e_shnum      ← number of section headers
      62      2     e_shstrndx

    32-bit ELF:
      Offset  Size  Field
      0       4     e_ident magic
      4-15    12    EI_CLASS .. padding
      16      2     e_type
      18      2     e_machine
      20      4     e_version
      24      4     e_entry
      28      4     e_phoff
      32      4     e_shoff      ← section-header table offset
      36      4     e_flags
      40      2     e_ehsize
      42      2     e_phentsize
      44      2     e_phnum
      46      2     e_shentsize  ← section-header entry size
      48      2     e_shnum      ← number of section headers
      50      2     e_shstrndx
    """
    magic = data[:4]
    if magic != b'\x7fELF':
        raise _ElfParseError("Not an ELF file")

    ei_class = data[4]   # 1=32-bit, 2=64-bit
    ei_data  = data[5]   # 1=little, 2=big
    endian   = '<' if ei_data == 1 else '>'

    if ei_class == 2:
        # 64-bit ELF: read shoff at 40 (Q), then shentsize/shnum at 58/60 (HH)
        e_shoff,    = struct.unpack_from(f'{endian}Q',  data, 40)
        e_shentsize, e_shnum = struct.unpack_from(f'{endian}HH', data, 58)
        for i in range(e_shnum):
            base = e_shoff + i * e_shentsize
            sh_type, = struct.unpack_from(f'{endian}I', data, base + 4)
            sh_off, sh_size = struct.unpack_from(f'{endian}QQ', data, base + 24)
            yield sh_type, sh_off, sh_size
    elif ei_class == 1:
        # 32-bit ELF: read shoff at 32 (I), then shentsize/shnum at 46/48 (HH)
        e_shoff,    = struct.unpack_from(f'{endian}I',  data, 32)
        e_shentsize, e_shnum = struct.unpack_from(f'{endian}HH', data, 46)
        for i in range(e_shnum):
            base = e_shoff + i * e_shentsize
            sh_type, = struct.unpack_from(f'{endian}I', data, base + 4)
            sh_off, sh_size = struct.unpack_from(f'{endian}II', data, base + 16)
            yield sh_type, sh_off, sh_size
    else:
        raise _ElfParseError(f"Unknown EI_CLASS {ei_class}")


def _iter_notes(data: bytes, section_data: bytes, endian: str):
    """Yield (name_bytes, note_type, desc_bytes) for ELF NOTE section entries."""
    pos = 0
    n = len(section_data)
    while pos + 12 <= n:
        namesz, descsz, ntype = struct.unpack_from(f'{endian}III', section_data, pos)
        pos += 12
        name = section_data[pos:pos + namesz]
        pos += (namesz + 3) & ~3   # 4-byte aligned
        desc = section_data[pos:pos + descsz]
        pos += (descsz + 3) & ~3
        yield name, ntype, desc


def read_amdhsa_metadata_from_elf(elf_path: str) -> Optional[dict]:
    """Return the decoded AMDHSA msgpack metadata dict from an AMDGPU ELF file.

    Returns None if no AMDGPU metadata note is found.

    Args:
        elf_path: Path to the assembled .o (relocatable AMDGPU ELF).

    The note section type is SHT_NOTE (7).  The AMDGPU note has name "AMDGPU"
    and type NT_AMDGPU_METADATA (32).  The note data is msgpack-encoded.
    """
    try:
        import msgpack
    except ImportError:
        raise ImportError("msgpack is required for ELF metadata reading (pip install msgpack)")

    with open(elf_path, 'rb') as f:
        data = f.read()

    if len(data) < 16:
        return None

    try:
        ei_data = data[5]
        endian = '<' if ei_data == 1 else '>'
    except IndexError:
        return None

    try:
        for sh_type, sh_off, sh_size in _parse_elf_section_headers(data):
            if sh_type != _SHT_NOTE:
                continue
            section_data = data[sh_off:sh_off + sh_size]
            for name, ntype, desc in _iter_notes(data, section_data, endian):
                # Name may be null-terminated; strip trailing nulls
                if name.rstrip(b'\x00') == b'AMDGPU' and ntype == _NT_AMDGPU_METADATA:
                    return msgpack.unpackb(desc, raw=False, strict_map_key=False)
    except (_ElfParseError, struct.error):
        return None

    return None


def read_vgpr_count_from_elf(elf_path: str, kernel_func_name: str) -> Optional[int]:
    """Return the vgpr_count for *kernel_func_name* from an AMDHSA ELF .o file.

    This is the value that was assembled from .amdhsa_next_free_vgpr in the
    source, which for pure-assembly Tensile kernels equals what the kernel
    descriptor will encode (no compiler reduction for -x assembler input).

    On ArchAccUnifiedRegs (gfx90a/gfx942/gfx950): vgpr_count =
        ceil(regular_vgprs / 8) * 8 + acc_vgprs  (the .amdhsa_accum_offset +
    remaining unified VGPRs), as computed by updateOccupancyFromScan or the
    original pool model.

    Returns None if the metadata cannot be parsed or the kernel is not found.
    """
    meta = read_amdhsa_metadata_from_elf(elf_path)
    if meta is None:
        return None

    kernels = meta.get("amdhsa.kernels", [])
    for k in kernels:
        name = k.get(".name") or k.get("name", "")
        if name == kernel_func_name:
            return k.get(".vgpr_count") or k.get("vgpr_count")

    return None


def read_kernel_descriptor_from_elf(elf_path: str, kernel_func_name: str) -> Optional[dict]:
    """Return the full per-kernel metadata dict for *kernel_func_name*.

    Keys follow the AMDHSA spec with leading dots, e.g.:
      .vgpr_count, .sgpr_count, .group_segment_fixed_size, .max_flat_workgroup_size
    """
    meta = read_amdhsa_metadata_from_elf(elf_path)
    if meta is None:
        return None
    for k in meta.get("amdhsa.kernels", []):
        name = k.get(".name") or k.get("name", "")
        if name == kernel_func_name:
            return k
    return None


# ── Occupancy computation from ELF data ───────────────────────────────────────

def compute_occupancy_from_elf_metadata(
    vgpr_count: int,
    lds_bytes: int,
    sgpr_count: int,
    num_threads: int,
    physical_vgpr: int,
    physical_sgpr: int,
    device_lds: int,
    max_waves_per_simd: int,
) -> int:
    """Compute CUOccupancy from compiled-object metadata (no GPU needed).

    This mirrors the Tensile getOccupancy formula applied to the *actual*
    vgpr_count from the compiled object, rather than the pool-estimate.

    For ArchAccUnifiedRegs: the unified physical_vgpr pool (512 on gfx950)
    is divided by vgpr_count (=.amdhsa_next_free_vgpr).

    Args:
        vgpr_count:        Unified VGPR count from the compiled .amdhsa_next_free_vgpr.
        lds_bytes:         Static LDS from group_segment_fixed_size.
        sgpr_count:        SGPR count from the compiled .sgpr_count.
        num_threads:       Block size (from .max_flat_workgroup_size).
        physical_vgpr:     Hardware physical VGPR pool per SIMD
                           (512 for ArchAccUnifiedRegs gfx9; 256 otherwise).
        physical_sgpr:     Hardware physical SGPR pool per SIMD (800 for gfx9).
        device_lds:        Device total LDS per CU (163840 B for gfx950).
        max_waves_per_simd: Hardware wave cap per SIMD (8 for ArchAccUnifiedRegs, 10 otherwise).

    Returns:
        Max active workgroups (blocks) per CU, identical to Tensile's CUOccupancy.
    """
    wave_size = 64  # gfx9 family
    num_simds = 4

    # Occupancy limits
    vgpr_occ    = physical_vgpr // max(vgpr_count, 1)
    sgpr_occ    = physical_sgpr // max(sgpr_count, 1) if sgpr_count > 0 else max_waves_per_simd

    lds_occ = device_lds // max(((lds_bytes + 255) // 256) * 256, 256) if lds_bytes > 0 else max_waves_per_simd

    # Wave-count limit from block size
    multiplier  = int(ceil(num_threads / 256.0))
    wave_occ    = max_waves_per_simd // multiplier

    return max(1, min(vgpr_occ, sgpr_occ, lds_occ, wave_occ, max_waves_per_simd))


# ── Build-time measurement passes ─────────────────────────────────────────────

def _arch_caps_for_kernel(kernel) -> Tuple[int, int, int, int]:
    """Return (physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd) for a kernel.

    Derived from the kernel's ISA tuple using the same logic as rocisa::getArchCaps.
    We avoid importing rocisa here to keep this module lightweight.
    """
    isa = tuple(kernel.get("ISA", (9, 0, 8)))

    # ArchAccUnifiedRegs: gfx90a (9,0,10), gfx942 (9,4,2), gfx950 (9,5,0)
    arch_acc_unified = isa in {(9, 0, 10), (9, 4, 2), (9, 5, 0)}

    if arch_acc_unified:
        physical_vgpr    = 512
        max_waves_per_simd = 8
    else:
        physical_vgpr    = 256
        max_waves_per_simd = 10

    physical_sgpr = 800   # gfx9 family

    # DeviceLDS: gfx950 has 160 KB; others have 64 KB
    if isa == (9, 5, 0):
        device_lds = 163840  # 160 * 1024
    elif isa in {(9, 0, 10), (9, 4, 2)}:
        device_lds = 65536   # 64 KB (gfx90a/gfx942)
    else:
        device_lds = 65536   # gfx908 and others

    return physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd


def compute_occupancy_from_asm_source(kernel, asm_source: str) -> Optional[int]:
    """Parse .amdhsa_* directives from a hand-written custom kernel .s and return CUOccupancy.

    Used by :func:`KernelWriterAssembly._getCustomKernelSource` to compute occupancy at
    build (codegen) time for custom kernels that bypass the normal ``checkResources`` path.

    For ArchAccUnifiedRegs ISAs (gfx90a/gfx942/gfx950) ``.amdhsa_next_free_vgpr`` is the
    combined unified VGPR total (regularVgprs + accVgprs) and is passed directly to
    :func:`compute_occupancy_from_elf_metadata`, which divides ``physical_vgpr`` by that
    total — matching the hardware allocation granularity.

    The regex is deliberately tolerant of inline comments after values, e.g.::

        .amdhsa_next_free_vgpr 256 // vgprs

    Args:
        kernel:     Kernel dict; must contain ``"ISA"`` (list/tuple) and ``"NumThreads"``.
        asm_source: Full text of the .s file (may contain multiple kernel blocks).

    Returns:
        Computed CUOccupancy (``>= 1``) or ``None`` if any required directive is missing
        or unparseable.  The caller must leave ``kernel["CUOccupancy"]`` unchanged on ``None``.
    """
    import re as _re

    def _parse(directive: str, text: str) -> Optional[int]:
        m = _re.search(r'\.' + directive + r'\s+(\d+)', text)
        return int(m.group(1)) if m else None

    vgpr_total = _parse('amdhsa_next_free_vgpr', asm_source)
    sgpr_total = _parse('amdhsa_next_free_sgpr', asm_source)
    lds_size   = _parse('amdhsa_group_segment_fixed_size', asm_source)

    if vgpr_total is None or sgpr_total is None or lds_size is None:
        return None

    num_threads = kernel.get('NumThreads', 256)
    phy_vgpr, phy_sgpr, device_lds, max_waves = _arch_caps_for_kernel(kernel)

    try:
        return compute_occupancy_from_elf_metadata(
            vgpr_count=vgpr_total,
            lds_bytes=lds_size,
            sgpr_count=sgpr_total,
            num_threads=num_threads,
            physical_vgpr=phy_vgpr,
            physical_sgpr=phy_sgpr,
            device_lds=device_lds,
            max_waves_per_simd=max_waves,
        )
    except Exception:
        return None


def _kernels_to_co_map(
    uniqueKernels: list,
    assemblyTmpPath: Path,
) -> Dict[str, list]:
    """Return a map: co_path → list of (kernel_idx, kernel) for that co file.

    Each kernel belongs to either the default TensileLibrary_<gfx>.co.raw or
    a per-kernel file if kernel["codeObjectFile"] is set.
    """
    co_map: Dict[str, list] = collections.defaultdict(list)
    for idx, kernel in enumerate(uniqueKernels):
        if kernel.get("KernelLanguage") != "Assembly":
            continue
        isa = tuple(kernel.get("ISA", ()))
        gfx = "gfx" + "".join(str(x) for x in isa if x is not None)
        # Match the naming in buildAssemblyCodeObjectFiles
        co_name = kernel.get("codeObjectFile", None)
        if co_name:
            co_raw = str(assemblyTmpPath / f"{co_name}.co.raw")
        else:
            co_raw = str(assemblyTmpPath / f"TensileLibrary_{gfx}.co.raw")
        co_map[co_raw].append((idx, kernel))
    return dict(co_map)


def measure_occupancy_hip(
    uniqueKernels: list,
    kernelInfo: list,
    assemblyTmpPath: Path,
    splitGSU: bool = False,
) -> list:
    """Overwrite cuoccupancy in kernelInfo using HIP hardware queries.

    For each unique kernel, loads the corresponding .co.raw code object via
    hipModuleLoad and calls hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
    This gives the authoritative hardware occupancy for every kernel.

    Args:
        uniqueKernels: List of kernel objects (from writeSolutionsAndKernelsTCL).
        kernelInfo:    List of KernelCodeGenResult NamedTuples (parallel with
                       uniqueKernels).
        assemblyTmpPath: Directory containing the assembled .o and .co.raw files.
        splitGSU:      Whether GSU kernels were split (passed through to naming).

    Returns:
        Updated list of KernelCodeGenResult with corrected cuoccupancy values.

    Degrades gracefully: if hip-python is not available, or no GPU is present,
    or a .co.raw file is missing, a warning is printed and the original
    occupancy is kept for the affected kernels.
    """
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    if not _HIP_AVAILABLE:
        warnings.warn(
            "--occupancy-from-hip: hip-python not installed; skipping HIP occupancy pass. "
            "Install with: pip install hip-python",
            stacklevel=2,
        )
        return list(kernelInfo)

    # Test that a device is available
    try:
        _hip_check(_hip.hipInit(0))
        err, count = _hip.hipGetDeviceCount()
        _hip_check(err)
        if int(count) == 0:
            warnings.warn(
                "--occupancy-from-hip: no GPU device found; skipping HIP occupancy pass.",
                stacklevel=2,
            )
            return list(kernelInfo)
    except RuntimeError as e:
        warnings.warn(
            f"--occupancy-from-hip: HIP initialisation failed ({e}); skipping pass.",
            stacklevel=2,
        )
        return list(kernelInfo)

    updated = list(kernelInfo)  # shallow copy; we replace NamedTuples individually

    # Build .co.raw → [(idx, kernel)] map
    co_map = _kernels_to_co_map(uniqueKernels, assemblyTmpPath)

    total = corrected = failed = 0

    for co_raw_path, kernel_list in co_map.items():
        if not os.path.exists(co_raw_path):
            warnings.warn(
                f"--occupancy-from-hip: {co_raw_path} not found; "
                f"skipping {len(kernel_list)} kernel(s) in this co.",
                stacklevel=2,
            )
            failed += len(kernel_list)
            continue

        try:
            _hip_check(_hip.hipInit(0))
            module = _hip_check(_hip.hipModuleLoad(co_raw_path.encode()))
        except RuntimeError as e:
            warnings.warn(
                f"--occupancy-from-hip: failed to load {co_raw_path}: {e}; "
                f"skipping {len(kernel_list)} kernel(s).",
                stacklevel=2,
            )
            failed += len(kernel_list)
            continue

        try:
            for idx, kernel in kernel_list:
                if idx >= len(updated):
                    continue
                total += 1
                kernel_func_name = getKernelNameMin(kernel, splitGSU)
                num_threads = kernel.get("NumThreads", 256)
                try:
                    func = _hip_check(_hip.hipModuleGetFunction(
                        module, kernel_func_name.encode()
                    ))
                    err, blocks_per_cu = _hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                        func, num_threads, 0
                    )
                    _hip_check(err)
                    hip_occ = int(blocks_per_cu)

                    old_occ = updated[idx].cuoccupancy
                    if hip_occ != old_occ:
                        updated[idx] = updated[idx]._replace(cuoccupancy=hip_occ)
                        corrected += 1
                        print(
                            f"  [HIP occ] {kernel_func_name[:60]}: "
                            f"{old_occ} → {hip_occ} blocks/CU"
                        )
                except RuntimeError as e:
                    warnings.warn(
                        f"--occupancy-from-hip: query failed for {kernel_func_name}: {e}",
                        stacklevel=2,
                    )
                    failed += 1
        finally:
            try:
                _hip_check(_hip.hipModuleUnload(module))
            except RuntimeError:
                pass

    print(
        f"[HIP occupancy pass] {total} kernels queried, "
        f"{corrected} corrected, {failed} failed/skipped."
    )
    return updated


def measure_occupancy_elf(
    uniqueKernels: list,
    kernelInfo: list,
    assemblyTmpPath: Path,
    splitGSU: bool = False,
) -> list:
    """Overwrite cuoccupancy in kernelInfo by reading vgpr_count from .o ELF files.

    For each unique kernel, parses the AMDHSA msgpack metadata from the assembled
    .o ELF file to extract vgpr_count, lds_bytes, and num_threads.  Recomputes
    CUOccupancy using the hardware occupancy formula without requiring a GPU.

    This is most useful as a:
    - Fast CPU-only verification pass (no GPU needed, negligible overhead)
    - Independent check that the .amdhsa_next_free_vgpr in the compiled object
      matches what updateOccupancyFromScan computed

    For pure-assembly kernels, the ELF value equals exactly what Tensile wrote
    in the .s file (the assembler does not reduce .amdhsa_next_free_vgpr).

    Args:
        uniqueKernels: List of kernel objects.
        kernelInfo:    List of KernelCodeGenResult NamedTuples.
        assemblyTmpPath: Directory with the assembled .o files.
        splitGSU:      Passed to naming functions.

    Returns:
        Updated list of KernelCodeGenResult with corrected cuoccupancy values.

    Degrades gracefully when msgpack is not installed or a .o file is missing.
    """
    try:
        import msgpack  # noqa: F401 (just verify availability)
    except ImportError:
        warnings.warn(
            "--occupancy-from-elf: msgpack not installed; skipping ELF occupancy pass. "
            "Install with: pip install msgpack",
            stacklevel=2,
        )
        return list(kernelInfo)

    from Tensile.SolutionStructs.Naming import getKernelNameMin

    updated = list(kernelInfo)
    total = corrected = failed = 0

    for idx, (kernel, result) in enumerate(zip(uniqueKernels, kernelInfo)):
        if kernel.get("KernelLanguage") != "Assembly":
            continue

        total += 1
        base_name = kernel.get("BaseName", "")
        o_path = str(assemblyTmpPath / f"{base_name}.o")

        if not os.path.exists(o_path):
            # .o may have been cleaned up already; skip gracefully
            failed += 1
            continue

        kernel_func_name = getKernelNameMin(kernel, splitGSU)
        kd = read_kernel_descriptor_from_elf(o_path, kernel_func_name)
        if kd is None:
            warnings.warn(
                f"--occupancy-from-elf: metadata not found for {kernel_func_name} in {o_path}",
                stacklevel=2,
            )
            failed += 1
            continue

        vgpr_count   = kd.get(".vgpr_count") or kd.get("vgpr_count") or 0
        lds_bytes    = kd.get(".group_segment_fixed_size") or kd.get("group_segment_fixed_size") or 0
        sgpr_count   = kd.get(".sgpr_count") or kd.get("sgpr_count") or 0
        num_threads_elf = kd.get(".max_flat_workgroup_size") or kd.get("max_flat_workgroup_size") or 0

        # Prefer the kernel parameter value (num_threads_elf from metadata is a max, not fixed)
        num_threads = kernel.get("NumThreads", num_threads_elf) or num_threads_elf

        if not vgpr_count:
            failed += 1
            continue

        physical_vgpr, physical_sgpr, device_lds, max_waves_per_simd = _arch_caps_for_kernel(kernel)

        elf_occ = compute_occupancy_from_elf_metadata(
            vgpr_count=vgpr_count,
            lds_bytes=lds_bytes,
            sgpr_count=sgpr_count,
            num_threads=num_threads,
            physical_vgpr=physical_vgpr,
            physical_sgpr=physical_sgpr,
            device_lds=device_lds,
            max_waves_per_simd=max_waves_per_simd,
        )

        old_occ = result.cuoccupancy
        if elf_occ != old_occ:
            updated[idx] = result._replace(cuoccupancy=elf_occ)
            corrected += 1
            print(
                f"  [ELF occ] {kernel_func_name[:60]}: "
                f"{old_occ} → {elf_occ}  "
                f"(vgpr={vgpr_count}, lds={lds_bytes}, threads={num_threads})"
            )

    print(
        f"[ELF occupancy pass] {total} kernels checked, "
        f"{corrected} corrected, {failed} failed/skipped."
    )
    return updated


# ── Utility: gfx string from ISA tuple ────────────────────────────────────────

def isa_to_gfx(isa: tuple) -> str:
    """Convert an ISA tuple to gfx string (e.g. (9,5,0) -> 'gfx950')."""
    return "gfx" + "".join(str(x) for x in isa)
