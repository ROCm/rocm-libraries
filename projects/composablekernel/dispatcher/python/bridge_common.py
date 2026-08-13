#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Shared infrastructure for ctypes dispatcher bridges.

Extracted from per-op utils to avoid duplication across the five quant bridge
types (rowcolquant, tensorquant, aquant, abquant, bquant). Each per-op module
imports from here and only implements the parts that differ:
  - KernelConfig dataclass (quant-specific fields)
  - DispatcherLib._setup() (quant-specific C ABI argtypes)
  - GpuGemmRunner.run() (quant-specific validation and stride logic)
"""

import ctypes
import logging
import os
import subprocess
import sys
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple

log = logging.getLogger(__name__)

# =============================================================================
# Shared constants
# =============================================================================

DEFAULT_HIPCC    = "hipcc"
DEFAULT_GFX_ARCH = "gfx950"


# =============================================================================
# GPU architecture detection
# =============================================================================

def detect_gpu_arch(default: str = DEFAULT_GFX_ARCH) -> str:
    """Detect current GPU arch via rocm_agent_enumerator. Falls back to `default`."""
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                return line
    except Exception as e:
        log.warning("rocm_agent_enumerator failed (%s); defaulting to %s", e, default)
        return default
    log.warning("rocm_agent_enumerator returned no usable arch; defaulting to %s", default)
    return default


# =============================================================================
# CK include / static lib discovery
# =============================================================================

def get_ck_include_dir(anchor: Path) -> Optional[Path]:
    """Locate the CK include directory relative to `anchor` (typically __file__)."""
    here = anchor.resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def get_dispatcher_static_lib(ctypes_lib_src: Path) -> Optional[Path]:
    """Return libck_tile_dispatcher.a from the CMake build directory, or None."""
    dispatcher_root = ctypes_lib_src.parent.parent.parent
    static_lib = dispatcher_root / "build" / "libck_tile_dispatcher.a"
    return static_lib if static_lib.exists() else None


# =============================================================================
# Architecture-specific compiler defines
# =============================================================================

def arch_defines(gfx_arch: str) -> List[str]:
    """Return the -D flags required for the given GPU architecture."""
    defines: List[str] = []
    if "gfx12" in gfx_arch or "gfx950" in gfx_arch:
        defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]
    if "gfx950" in gfx_arch:
        defines += ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]
    return defines


# =============================================================================
# Shared compile + link
# =============================================================================

def compile_kernel(
    ctypes_lib_src: Path,
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """Compile a generated .hpp into a .so via hipcc (compile then link).

    The ctypes_lib_src (.cpp) is compiled with the .hpp force-included so that
    each .so contains exactly one kernel variant baked in at compile time.
    """
    ck_include = get_ck_include_dir(ctypes_lib_src)
    static_lib = get_dispatcher_static_lib(ctypes_lib_src)

    obj_path = so_path.with_suffix(".o")
    defines   = arch_defines(gfx_arch)

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}",
                   f"-DGFX_ARCH=\"{gfx_arch}\"",
                   *defines,
                   "-include", str(hpp_path),
                   str(ctypes_lib_src),
                   "-o", str(obj_path)]

    if ck_include:
        compile_cmd += [f"-I{ck_include}"]
    if extra_include_dirs:
        for d in extra_include_dirs:
            compile_cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(compile_cmd))

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        obj_path.unlink(missing_ok=True)
        return False

    link_cmd = [hipcc, "-shared", "-fPIC",
                f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path)]

    if static_lib:
        link_cmd += [str(static_lib)]
    link_cmd += ["-o", str(so_path)]

    log.debug("Linking %s:\n  %s", so_path.name, " ".join(link_cmd))

    try:
        result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("Link failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            obj_path.unlink(missing_ok=True)
            return False
    except subprocess.TimeoutExpired:
        log.error("Link timed out for %s", so_path.name)
        obj_path.unlink(missing_ok=True)
        return False

    obj_path.unlink(missing_ok=True)
    return True


# =============================================================================
# Shared codegen runner
# =============================================================================

def run_codegen(
    codegen_script: Path,
    config_json: str,
    output_dir: Path,
    kernel_name: str,
) -> Optional[Path]:
    """Invoke a codegen script with a JSON config; return the .hpp path or None."""
    cmd = [
        sys.executable,
        str(codegen_script),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("Codegen failed for %s:\n%s", kernel_name, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("Codegen timed out for %s", kernel_name)
        return None

    hpp = output_dir / f"{kernel_name}.hpp"
    if not hpp.exists():
        log.error("Codegen succeeded but %s not found", hpp)
        return None

    return hpp


# =============================================================================
# Shared parallel build pipeline
# =============================================================================

class _HasName(Protocol):
    @property
    def name(self) -> str: ...

    def to_codegen_config(self) -> dict: ...


def setup_multiple_dispatchers(
    configs: List[_HasName],
    codegen_script: Path,
    ctypes_lib_src: Path,
    label: str,
    output_dir: Optional[Path] = None,
    hipcc: str = DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each config: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.

    Args:
        configs:          List of kernel configs (each must expose .name and .to_codegen_config()).
        codegen_script:   Path to the per-op codegen .py script.
        ctypes_lib_src:   Path to the per-op ctypes .cpp source file.
        label:            Human-readable op name for log messages (e.g. "RowColQuant").
    """
    import json

    if not configs:
        return []

    arch     = gfx_arch or detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix=f"{label.lower()}_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info("Building %d %s kernel(s) for %s into %s", len(configs), label, arch, base_dir)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, _HasName]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: _HasName) -> Tuple[int, Optional[Path]]:
        hpp = run_codegen(
            codegen_script=codegen_script,
            config_json=json.dumps(cfg.to_codegen_config()),
            output_dir=headers_dir,
            kernel_name=cfg.name,
        )
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = compile_kernel(
            ctypes_lib_src=ctypes_lib_src,
            hpp_path=hpp,
            so_path=so,
            gfx_arch=arch,
            hipcc=hipcc,
            extra_include_dirs=extra_include_dirs,
        )
        return idx, so if ok else None

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_one, idx, cfg): (idx, cfg) for idx, cfg in deduped}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, so_path = fut.result()
                    results[idx] = so_path
                    if so_path:
                        log.info("  built %s", so_path.name)
                    else:
                        _, cfg = futures[fut]
                        log.error("  FAILED %s", cfg.name)
                except Exception as e:
                    _, cfg = futures[fut]
                    log.error("  EXCEPTION for %s: %s", cfg.name, e)
    else:
        for idx, cfg in deduped:
            _, so_path = _build_one(idx, cfg)
            results[idx] = so_path

    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]
                if results[i] is None:
                    log.debug("  dedup: %s (index %d) inherits failed build from index %d",
                              cfg.name, i, first_idx)

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d %s kernels", built, len(configs), label)
    return results


# =============================================================================
# BaseDispatcherLib — shared ctypes lifecycle
# =============================================================================

class BaseDispatcherLib:
    """
    Base class for ctypes wrappers around compiled quant GEMM .so files.

    Handles .so loading, dispatcher_initialize(), dispatcher_cleanup(), and
    the get_kernel_name / get_kernel_count accessors — which are identical
    across all five quant bridge types.

    Subclasses must implement _setup_run_fn() to register the quant-specific
    dispatcher_run_*_gemm argtypes on self._lib.
    """

    def __init__(self, so_path: Path, so_label: str):
        self.so_path = Path(so_path)
        self._cleaned_up = False
        if not self.so_path.exists():
            raise FileNotFoundError(f"{so_label} .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup_common()
        self._setup_run_fn()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup_common(self):
        lib = self._lib
        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_get_kernel_name.restype  = ctypes.c_char_p
        lib.dispatcher_get_kernel_name.argtypes = []

        lib.dispatcher_get_kernel_count.restype  = ctypes.c_int
        lib.dispatcher_get_kernel_count.argtypes = []

        lib.dispatcher_cleanup.restype  = None
        lib.dispatcher_cleanup.argtypes = []

    def _setup_run_fn(self):
        """Subclasses register the quant-specific dispatcher_run_* argtypes here."""
        raise NotImplementedError

    def get_kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else ""

    def get_kernel_count(self) -> int:
        return self._lib.dispatcher_get_kernel_count()

    def cleanup(self):
        if not self._cleaned_up:
            self._lib.dispatcher_cleanup()
            self._cleaned_up = True

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
