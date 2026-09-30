# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
HIP-based occupancy measurement helpers for occupancy unit tests.

TEST-ONLY — production code must not import from here.
Exports: HIP_AVAILABLE, _hip, _hip_check, query_hip_occupancy,
         _kernels_to_co_map, measure_occupancy_hip.
"""

import collections
import os
import warnings
from pathlib import Path
from typing import Dict

# ── hip-python (optional) ─────────────────────────────────────────────────────
try:
    from hip import hip as _hip
    HIP_AVAILABLE = True
except ImportError:
    _hip = None
    HIP_AVAILABLE = False


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
    if not HIP_AVAILABLE:
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

    if not HIP_AVAILABLE:
        warnings.warn(
            "HIP occupancy check: hip-python not installed; skipping. "
            "Install with: pip install hip-python",
            stacklevel=2,
        )
        return list(kernelInfo)

    try:
        _hip_check(_hip.hipInit(0))
        err, count = _hip.hipGetDeviceCount()
        _hip_check(err)
        if int(count) == 0:
            warnings.warn(
                "HIP occupancy check: no GPU device found; skipping.",
                stacklevel=2,
            )
            return list(kernelInfo)
    except RuntimeError as e:
        warnings.warn(
            f"HIP occupancy check: HIP initialisation failed ({e}); skipping.",
            stacklevel=2,
        )
        return list(kernelInfo)

    updated = list(kernelInfo)

    co_map = _kernels_to_co_map(uniqueKernels, assemblyTmpPath)

    total = corrected = failed = 0

    for co_raw_path, kernel_list in co_map.items():
        if not os.path.exists(co_raw_path):
            warnings.warn(
                f"HIP occupancy check: {co_raw_path} not found; "
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
                f"HIP occupancy check: failed to load {co_raw_path}: {e}; "
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
                        f"HIP occupancy check: query failed for {kernel_func_name}: {e}",
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
