#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
BQuantGrouped GEMM dispatcher utilities.

Three-layer Python bridge for the dispatcher's BQuantGrouped GEMM path:

  BQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  BQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  BQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_bquant_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

Usage (end-to-end):
  configs = [BQuantKernelConfig(variant_key="fp8", layout="rcr", ...)]
  so_paths = setup_multiple_bquant_dispatchers(configs, output_dir=Path("/tmp/bq"))
  runner = BQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, BQ, BQuantGemmProblem(M=16, N=64, K=256))
"""

import ctypes
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_bquant_gemm_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "bquant_gemm_ctypes_lib.cpp"

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"

# Flags that match the tile engine / dispatcher build flags for BQuant kernels
_HIPCC_BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
    "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
    "-w",  # suppress warnings during generated-code compilation
]


# =============================================================================
# BQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class BQuantKernelConfig:
    """
    Complete description of one BQuantGrouped GEMM kernel.

    The .name property produces the exact string that unified_bquant_gemm_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    variant_key: str       # "fp8" or "bf8"
    layout: str            # "rcr" (A=RowMajor, B=ColMajor, C=RowMajor)
    pipeline: str          # "compv3"
    epilogue: str          # "cshuffle"
    scheduler: str         # "intrawave"

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128

    preshuffle_b: bool     = False
    preshuffle_bquant: bool = False

    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME."""
        parts = [
            "bquant_gemm",
            self.variant_key,
            self.layout,
            self.pipeline,
            self.epilogue,
            self.scheduler,
            f"{self.tile_m}x{self.tile_n}x{self.tile_k}",
            f"{self.warp_m}x{self.warp_n}x{self.warp_k}",
            f"{self.warp_tile_m}x{self.warp_tile_n}x{self.warp_tile_k}",
            f"qg{self.quant_group_m}x{self.quant_group_n}x{self.quant_group_k}",
        ]
        if self.preshuffle_b:
            parts.append("preshuffleb")
        if self.preshuffle_bquant:
            parts.append("preshufflebq")
        return "_".join(parts)

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_bquant_gemm_codegen.py."""
        return {
            "variant_keys": [self.variant_key],
            "layouts": [self.layout],
            "pipeline": self.pipeline,
            "epilogue": self.epilogue,
            "scheduler": self.scheduler,
            "tile_configs": [{
                "tile_m": self.tile_m,
                "tile_n": self.tile_n,
                "tile_k": self.tile_k,
                "warp_m": self.warp_m,
                "warp_n": self.warp_n,
                "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m,
                "warp_tile_n": self.warp_tile_n,
                "warp_tile_k": self.warp_tile_k,
            }],
            "quant_groups": [{
                "quant_group_m": self.quant_group_m,
                "quant_group_n": self.quant_group_n,
                "quant_group_k": self.quant_group_k,
            }],
        }


# =============================================================================
# BQuantGemmProblem
# =============================================================================


@dataclass
class BQuantGemmProblem:
    M: int
    N: int
    K: int
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    k_batch: int = 1

    @property
    def QK_B(self) -> int:
        """Number of K-groups: ceil(K / quant_group_k)."""
        return (self.K + self.quant_group_k - 1) // self.quant_group_k

    @property
    def QN_B(self) -> int:
        """Number of N-groups: ceil(N / quant_group_n)."""
        return (self.N + self.quant_group_n - 1) // self.quant_group_n


# =============================================================================
# BQuantGemmResult
# =============================================================================


@dataclass
class BQuantGemmResult:
    C: object          # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# BQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class BQuantDispatcherLib:
    """
    Loads a compiled bquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_bquant_gemm(A, B, BQ, C, M, N, K,
                                       stride_A, stride_B, stride_BQ, stride_C,
                                       QK_B, QN_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"BQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_bquant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_bquant_gemm.argtypes = [
            ctypes.c_void_p,   # A
            ctypes.c_void_p,   # B
            ctypes.c_void_p,   # BQ
            ctypes.c_void_p,   # C
            ctypes.c_int64,    # M
            ctypes.c_int64,    # N
            ctypes.c_int64,    # K
            ctypes.c_int64,    # stride_A
            ctypes.c_int64,    # stride_B
            ctypes.c_int64,    # stride_BQ
            ctypes.c_int64,    # stride_C
            ctypes.c_int64,    # QK_B
            ctypes.c_int64,    # QN_B
            ctypes.c_int,      # k_batch
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]

        lib.dispatcher_get_kernel_name.restype  = ctypes.c_char_p
        lib.dispatcher_get_kernel_name.argtypes = []

        lib.dispatcher_get_kernel_count.restype  = ctypes.c_int
        lib.dispatcher_get_kernel_count.argtypes = []

        lib.dispatcher_cleanup.restype  = None
        lib.dispatcher_cleanup.argtypes = []

    def run(
        self,
        A,
        B,
        BQ,
        C,
        M: int,
        N: int,
        K: int,
        stride_A: int,
        stride_B: int,
        stride_BQ: int,
        stride_C: int,
        QK_B: int,
        QN_B: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_bquant_gemm with ctypes-wrapped pointers.

        A, B, BQ, C should be numpy arrays (contiguous).
        Returns (status, time_ms).
        """
        import numpy as np

        A   = np.ascontiguousarray(A)
        B   = np.ascontiguousarray(B)
        BQ  = np.ascontiguousarray(BQ)
        C   = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_bquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M),
            ctypes.c_int64(N),
            ctypes.c_int64(K),
            ctypes.c_int64(stride_A),
            ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_BQ),
            ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_B),
            ctypes.c_int64(QN_B),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def get_kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else ""

    def get_kernel_count(self) -> int:
        return self._lib.dispatcher_get_kernel_count()

    def cleanup(self):
        self._lib.dispatcher_cleanup()

    def __del__(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:
            pass


# =============================================================================
# BQuantGpuGemmRunner — high-level runner
# =============================================================================


class BQuantGpuGemmRunner:
    """
    High-level runner that loads a BQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, BQ; allocates C; returns BQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = BQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, BQ, problem: BQuantGemmProblem) -> BQuantGemmResult:
        """
        Run BQuantGrouped GEMM.

        A   shape: (M, K)           dtype: fp8/bf8
        B   shape: (K, N) col-major  dtype: fp8/bf8
        BQ  shape: (QK_B, QN_B)     dtype: float/fp8
        Returns BQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_B    = problem.QK_B
        QN_B    = problem.QN_B

        # Output buffer — dtype matches CDataType (half for fp8/bf8 variants)
        C = np.zeros((M, N), dtype=np.float16)

        # Strides (in elements, row-major for A and C; col-major for B means stride = K)
        stride_A  = K   # A is row-major [M, K]
        stride_B  = K   # B is col-major [K, N] → leading dim = K
        stride_BQ = QN_B
        stride_C  = N   # C is row-major [M, N]

        rc, time_ms = self._lib.run(
            A=A, B=B, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_B=stride_B,
            stride_BQ=stride_BQ,
            stride_C=stride_C,
            QK_B=QK_B,
            QN_B=QN_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_bquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return BQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers (self-contained, do not call ctypes_utils.py)
# =============================================================================


def _detect_gpu_arch() -> str:
    """Detect current GPU arch via rocm_agent_enumerator. Falls back to gfx950."""
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                return line
    except Exception:
        pass
    return _DEFAULT_GFX_ARCH


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    # Walk up from dispatcher/python/ to find project root
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_bquant_kernel(
    config: BQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """
    Run unified_bquant_gemm_codegen.py for one config; return the .hpp path or None.
    """
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    cmd = [
        sys.executable,
        str(_CODEGEN_SCRIPT),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            log.error("Codegen failed for %s:\n%s", config.name, result.stderr)
            return None
    except subprocess.TimeoutExpired:
        log.error("Codegen timed out for %s", config.name)
        return None

    hpp = output_dir / f"{config.name}.hpp"
    if not hpp.exists():
        log.error("Codegen succeeded but %s not found", hpp)
        return None

    return hpp


def _compile_bquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """
    Compile a generated .hpp into a .so via hipcc.
    Returns True on success.
    """
    ck_include = _get_ck_include_dir()

    cmd = [hipcc] + _HIPCC_BASE_FLAGS + [
        f"--offload-arch={gfx_arch}",
        f"-DGFX_ARCH=\"{gfx_arch}\"",
        f"-include", str(hpp_path),
        str(_CTYPES_LIB_SRC),
        "-o", str(so_path),
    ]

    if ck_include:
        cmd += [f"-I{ck_include}"]

    # Dispatcher include
    dispatcher_include = _CTYPES_LIB_SRC.parent.parent.parent / "dispatcher" / "include"
    if dispatcher_include.is_dir():
        cmd += [f"-I{dispatcher_include}"]

    if extra_include_dirs:
        for d in extra_include_dirs:
            cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False


# =============================================================================
# setup_multiple_bquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_bquant_dispatchers(
    configs: List[BQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each BQuantKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="bquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info(
        "Building %d BQuant kernel(s) for %s into %s",
        len(configs), arch, base_dir,
    )

    # Deduplicate by name so we don't build the same kernel twice
    seen: Dict[str, int] = {}          # name → index of first occurrence
    deduped: List[Tuple[int, BQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    # results[i] = Path or None, aligned with input configs
    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: BQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_bquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_bquant_kernel(
            hpp_path=hpp,
            so_path=so,
            gfx_arch=arch,
            hipcc=hipcc,
            extra_include_dirs=extra_include_dirs,
        )
        return idx, so if ok else None

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
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

    # Fill in duplicates
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d BQuant kernels", built, len(configs))
    return results


# =============================================================================
# Convenience: default fp8 config (matches GemmConfigQuantDecode<fp8_t>)
# =============================================================================


def default_fp8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """Return the default fp8 BQuant config (tile = 16x64x256, warp = 1x4x1)."""
    return BQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> BQuantKernelConfig:
    """Return the default bf8 BQuant config (tile = 16x64x256, warp = 1x4x1)."""
    return BQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=16,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )
