#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm RowColQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's RowColQuant Grouped GEMM path:

  RowColQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  RowColQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  RowColQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers:
  setup_multiple_rowcolquant_dispatchers(configs, ...)
       codegen → hipcc → list of .so paths, all in parallel

RowColQuant: A has per-row scales [M, 1], B has per-column scales [1, N].
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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_rowcolquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_rowcolquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from unified_grouped_gemm_rowcolquant_codegen import make_rowcolquant_kernel_name  # noqa: E402


# =============================================================================
# RowColQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class RowColQuantKernelConfig:
    """
    Complete description of one RowColQuant Grouped GEMM kernel.

    The .name property produces the exact string that unified_grouped_gemm_rowcolquant_codegen.py
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
        return make_rowcolquant_kernel_name(
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
        """Produce the JSON config dict consumed by unified_grouped_gemm_rowcolquant_codegen.py."""
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
# RowColQuantGemmProblem
# =============================================================================


@dataclass
class RowColQuantGemmProblem:
    M: int
    N: int
    K: int
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        """Number of AQ elements (one per row). Used for buffer sizing only; the kernel uses broadcast strides."""
        return self.M

    @property
    def QK_B(self) -> int:
        """Number of BQ elements (one per column). Used for buffer sizing only; the kernel uses broadcast strides."""
        return self.N


# =============================================================================
# RowColQuantGemmResult
# =============================================================================


@dataclass
class RowColQuantGemmResult:
    C: object
    time_ms: float
    kernel_name: str


# =============================================================================
# RowColQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class RowColQuantDispatcherLib(BaseDispatcherLib):
    """
    Loads a compiled rowcolquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_rowcolquant_gemm(A, B, AQ, BQ, C,
                                            M, N, K,
                                            stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                                            QK_A, QK_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        super().__init__(so_path, "RowColQuant")

    def _setup_run_fn(self):
        self._lib.dispatcher_run_rowcolquant_gemm.restype  = ctypes.c_int
        self._lib.dispatcher_run_rowcolquant_gemm.argtypes = [
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
            ctypes.c_int64,    # QK_A
            ctypes.c_int64,    # QK_B
            ctypes.c_int,      # k_batch
            ctypes.POINTER(ctypes.c_float),  # time_ms
        ]

    def run(
        self,
        A, B, AQ, BQ, C,
        M: int, N: int, K: int,
        stride_A: int, stride_B: int,
        stride_AQ: int, stride_BQ: int, stride_C: int,
        QK_A: int, QK_B: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """Call dispatcher_run_rowcolquant_gemm with ctypes-wrapped pointers.

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
        rc = self._lib.dispatcher_run_rowcolquant_gemm(
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
            ctypes.c_int64(QK_A),
            ctypes.c_int64(QK_B),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value


# =============================================================================
# RowColQuantGpuGemmRunner — high-level runner
# =============================================================================


class RowColQuantGpuGemmRunner:
    """
    High-level runner that loads a RowColQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, AQ, BQ; allocates C; returns RowColQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = RowColQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, BQ, problem: RowColQuantGemmProblem, c_dtype=None) -> RowColQuantGemmResult:
        """
        Run RowColQuant Grouped GEMM.

        A       shape: (M, K)     dtype: fp8/bf8  (row-major)
        B       shape: (K, N)     dtype: fp8/bf8  (col-major)
        AQ      shape: (M,)       dtype: float    (per-row A scale)
        BQ      shape: (N,)       dtype: float    (per-col B scale)
        c_dtype numpy dtype for the output C buffer. Defaults to np.float16.
        Returns RowColQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_A = problem.QK_A  # == M
        QK_B = problem.QK_B  # == N

        if A.ndim != 2 or A.shape != (M, K):
            raise ValueError(f"A shape mismatch: expected ({M}, {K}), got {A.shape}")
        if B.ndim != 2 or B.shape != (K, N):
            raise ValueError(f"B shape mismatch: expected ({K}, {N}), got {B.shape}")
        if AQ.ndim != 1 or AQ.shape[0] != M:
            raise ValueError(f"AQ shape mismatch: expected ({M},), got {AQ.shape}")
        if BQ.ndim != 1 or BQ.shape[0] != N:
            raise ValueError(f"BQ shape mismatch: expected ({N},), got {BQ.shape}")
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

        # Strides for A, B, C (standard packed layouts).
        # stride_AQ and stride_BQ are NOT passed to the kernel; the C++ lib hardwires
        # broadcast strides (0) because the RowColQuant kernel indexes each scale vector
        # directly by the row/col index without a stride multiply.
        stride_A = K   # A row-major [M, K]
        stride_B = K   # B col-major [K, N], leading dim = K
        stride_C = N   # C row-major [M, N]

        # stride_AQ=1, stride_BQ=1 are placeholder values; the C++ lib ignores them
        # and always passes broadcast strides (0) to the kernel.
        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_AQ=1, stride_BQ=1, stride_C=stride_C,
            QK_A=QK_A, QK_B=QK_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_rowcolquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return RowColQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# setup_multiple_rowcolquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_rowcolquant_dispatchers(
    configs: List[RowColQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each RowColQuantKernelConfig: codegen → hipcc compile → .so path.

    Returns a list parallel to `configs` — each entry is the Path to the
    compiled .so, or None if that config failed.
    """
    return setup_multiple_dispatchers(
        configs=configs,
        codegen_script=_CODEGEN_SCRIPT,
        ctypes_lib_src=_CTYPES_LIB_SRC,
        label="RowColQuant",
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


def default_fp8_config(gfx_arch: str = DEFAULT_GFX_ARCH) -> RowColQuantKernelConfig:
    """Return the default fp8 RowColQuant config."""
    return RowColQuantKernelConfig(
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


def default_bf8_config(gfx_arch: str = DEFAULT_GFX_ARCH) -> RowColQuantKernelConfig:
    """Return the default bf8 RowColQuant config."""
    return RowColQuantKernelConfig(
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
