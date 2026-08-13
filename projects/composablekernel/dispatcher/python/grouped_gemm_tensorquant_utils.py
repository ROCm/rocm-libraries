#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm TensorQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's TensorQuant Grouped GEMM path:

  TensorQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  TensorQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  TensorQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers:
  setup_multiple_tensorquant_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

TensorQuant: A and B each have a single per-tensor scalar scale.
ADataType=BDataType=fp8/bf8; AQDataType=BQDataType=float; CDataType=half.
"""

import ctypes
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from bridge_common import (
    DEFAULT_GFX_ARCH,
    DEFAULT_HIPCC,
    BaseDispatcherLib,
    detect_gpu_arch,  # noqa: F401 — re-exported for callers that imported it from here
    setup_multiple_dispatchers,
)

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_tensorquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_tensorquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from unified_grouped_gemm_tensorquant_codegen import make_tensorquant_kernel_name  # noqa: E402


# =============================================================================
# TensorQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class TensorQuantKernelConfig:
    """
    Complete description of one TensorQuant Grouped GEMM kernel.

    The .name property produces the exact string that unified_grouped_gemm_tensorquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    dtype: str       # "fp8" or "bf8"
    layout: str      # "rcr"
    pipeline: str    # "compv3"
    epilogue: str    # "cshuffle"
    scheduler: str   # "intrawave"

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    pad_m: bool = False
    pad_n: bool = False
    pad_k: bool = True
    persistent: bool = False
    block_size: int = 256
    k_block_per_cu: int = 1

    gfx_arch: str = DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME."""
        return make_tensorquant_kernel_name(
            dtype=self.dtype,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            pad_m=self.pad_m,
            pad_n=self.pad_n,
            pad_k=self.pad_k,
            persistent=self.persistent,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_grouped_gemm_tensorquant_codegen.py."""
        return {
            "dtypes": [self.dtype],
            "layouts": [self.layout],
            "pipeline": self.pipeline,
            "epilogue": self.epilogue,
            "scheduler": self.scheduler,
            "pad_m": self.pad_m,
            "pad_n": self.pad_n,
            "pad_k": self.pad_k,
            "persistent": self.persistent,
            "block_size": self.block_size,
            "k_block_per_cu": self.k_block_per_cu,
            "tile_configs": [{
                "tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "warp_m": self.warp_m, "warp_n": self.warp_n, "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m, "warp_tile_n": self.warp_tile_n, "warp_tile_k": self.warp_tile_k,
            }],
        }


# =============================================================================
# TensorQuantGemmProblem
# =============================================================================


@dataclass
class TensorQuantGemmProblem:
    M: int
    N: int
    K: int
    k_batch: int = 1


# =============================================================================
# TensorQuantGemmResult
# =============================================================================


@dataclass
class TensorQuantGemmResult:
    C: object
    time_ms: float
    kernel_name: str


# =============================================================================
# TensorQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class TensorQuantDispatcherLib(BaseDispatcherLib):
    """
    Loads a compiled tensorquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_tensorquant_gemm(A, B, AQ, BQ, C,
                                            M, N, K,
                                            stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                                            k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        super().__init__(so_path, "TensorQuant")

    def _setup_run_fn(self):
        self._lib.dispatcher_run_tensorquant_gemm.restype  = ctypes.c_int
        self._lib.dispatcher_run_tensorquant_gemm.argtypes = [
            ctypes.c_void_p,   # A
            ctypes.c_void_p,   # B
            ctypes.c_void_p,   # AQ
            ctypes.c_void_p,   # BQ
            ctypes.c_void_p,   # C
            ctypes.c_int64,    # M
            ctypes.c_int64,    # N
            ctypes.c_int64,    # K
            ctypes.c_int64,    # stride_A
            ctypes.c_int64,    # stride_B
            ctypes.c_int64,    # stride_AQ
            ctypes.c_int64,    # stride_BQ
            ctypes.c_int64,    # stride_C
            ctypes.c_int,      # k_batch
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]

    def run(
        self,
        A, B, AQ, BQ, C,
        M: int, N: int, K: int,
        stride_A: int, stride_B: int,
        stride_AQ: int, stride_BQ: int, stride_C: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """Call dispatcher_run_tensorquant_gemm with ctypes-wrapped pointers.

        B must already be F-contiguous (column-major) — the caller (GpuGemmRunner)
        converts it with asfortranarray before passing it here.  Using
        ascontiguousarray on a 2-D F-contiguous array would silently copy it back
        to C order, making the declared stride_B=K incorrect.
        """
        import numpy as np
        A  = np.ascontiguousarray(A)
        # Preserve F-contiguous layout for B (rcr: column-major B, stride_B = K).
        B  = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        AQ = np.ascontiguousarray(AQ)
        BQ = np.ascontiguousarray(BQ)
        C  = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)
        rc = self._lib.dispatcher_run_tensorquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            AQ.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M),
            ctypes.c_int64(N),
            ctypes.c_int64(K),
            ctypes.c_int64(stride_A),
            ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_AQ),
            ctypes.c_int64(stride_BQ),
            ctypes.c_int64(stride_C),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value


# =============================================================================
# TensorQuantGpuGemmRunner — high-level runner
# =============================================================================


class TensorQuantGpuGemmRunner:
    """
    High-level runner that loads a TensorQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, AQ (scalar), BQ (scalar); allocates C; returns TensorQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = TensorQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, BQ, problem: TensorQuantGemmProblem, c_dtype=None) -> TensorQuantGemmResult:
        """
        Run TensorQuant Grouped GEMM.

        A    shape: (M, K)   dtype: fp8/bf8  (row-major)
        B    shape: (K, N)   dtype: fp8/bf8  (col-major)
        AQ   shape: (1,)     dtype: float    (per-tensor A scale)
        BQ   shape: (1,)     dtype: float    (per-tensor B scale)
        c_dtype numpy dtype for the output C buffer. Defaults to np.float16.
        Returns TensorQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K

        if A.ndim != 2 or A.shape != (M, K):
            raise ValueError(f"A shape mismatch: expected ({M}, {K}), got {A.shape}")
        if B.ndim != 2 or B.shape != (K, N):
            raise ValueError(f"B shape mismatch: expected ({K}, {N}), got {B.shape}")
        if AQ.ndim != 1 or AQ.shape[0] != 1:
            raise ValueError(f"AQ shape mismatch: expected (1,), got {AQ.shape}")
        if BQ.ndim != 1 or BQ.shape[0] != 1:
            raise ValueError(f"BQ shape mismatch: expected (1,), got {BQ.shape}")
        # fp8/bf8 have no native numpy dtype; both are 1-byte elements.
        if A.itemsize != 1:
            raise ValueError(f"A dtype must be a 1-byte fp8/bf8 type, got {A.dtype} (itemsize={A.itemsize})")
        if B.itemsize != 1:
            raise ValueError(f"B dtype must be a 1-byte fp8/bf8 type, got {B.dtype} (itemsize={B.itemsize})")
        if AQ.dtype != np.float32:
            raise ValueError(f"AQ dtype must be float32, got {AQ.dtype}")
        if BQ.dtype != np.float32:
            raise ValueError(f"BQ dtype must be float32, got {BQ.dtype}")

        if c_dtype is None:
            c_dtype = np.float16
        if c_dtype != np.float16:
            raise ValueError(
                f"c_dtype must be float16 (the compiled ABI always writes CDataType=half); "
                f"got {c_dtype}"
            )

        C = np.zeros((M, N), dtype=c_dtype)

        # B is column-major (rcr layout): the kernel expects leading dim = K (stride_B = K),
        # which means elements are stored column-first in memory (Fortran order).
        # Reorder here so the raw pointer passed to C++ matches the stride we declare below.
        B = np.asfortranarray(B)
        if not B.flags["F_CONTIGUOUS"]:
            raise RuntimeError("B is not F-contiguous after asfortranarray — unexpected numpy state")

        # TensorQuant: single scalar scale per tensor → AQ/BQ strides are 1
        stride_A  = K
        stride_B  = K
        stride_AQ = 1
        stride_BQ = 1
        stride_C  = N

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_AQ=stride_AQ, stride_BQ=stride_BQ, stride_C=stride_C,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_tensorquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return TensorQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# setup_multiple_tensorquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_tensorquant_dispatchers(
    configs: List[TensorQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each TensorQuantKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.
    """
    return setup_multiple_dispatchers(
        configs=configs,
        codegen_script=_CODEGEN_SCRIPT,
        ctypes_lib_src=_CTYPES_LIB_SRC,
        label="TensorQuant",
        output_dir=output_dir,
        hipcc=hipcc,
        gfx_arch=gfx_arch,
        extra_include_dirs=extra_include_dirs,
        parallel=parallel,
        max_workers=max_workers,
    )


# =============================================================================
# Convenience: default fp8 and bf8 configs
# =============================================================================


def default_fp8_config(gfx_arch: str = DEFAULT_GFX_ARCH) -> TensorQuantKernelConfig:
    """Return the default fp8 TensorQuant config."""
    return TensorQuantKernelConfig(
        dtype="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=64,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pad_m=True,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(gfx_arch: str = DEFAULT_GFX_ARCH) -> TensorQuantKernelConfig:
    """Return the default bf8 TensorQuant config."""
    return TensorQuantKernelConfig(
        dtype="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=64,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
        pad_m=True,
        gfx_arch=gfx_arch,
    )
