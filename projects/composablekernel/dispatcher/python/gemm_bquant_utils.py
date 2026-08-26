#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm BQuant dispatcher utilities (non-grouped, block-scale GEMM).

Three-layer Python bridge for the dispatcher's plain (non-grouped) BQuant GEMM
path from example/ck_tile/38_block_scale_gemm. Distinct from the multi-problem
grouped_gemm_bquant bridge.

  BQuantKernelConfig  -- describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  BQuantDispatcherLib -- thin ctypes wrapper around a compiled .so
  BQuantGpuGemmRunner -- high-level runner that accepts numpy arrays

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_bquant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

Usage (end-to-end):
  configs = [BQuantKernelConfig(variant_key="fp8", layout="rcr", ...)]
  so_paths = setup_multiple_bquant_dispatchers(configs, output_dir=Path("/tmp/bq"))
  runner = BQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, BQ, BQuantGemmProblem(M=16, N=64, K=256))
"""

import ctypes
import json
import functools
import logging
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Shared quant-bridge scaffolding (ctypes API install, codegen subprocess, build
# orchestration, CK include probe). Op-specific parts stay in this file.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from quant_bridge_base import (  # noqa: E402
    arch_from_so_path,
    DispatcherLibBase,
    build_dispatchers,
    find_ck_include_dir,
    generate_kernel,
    coerce_a_for_variant,
    encode_bq_for_variant,
    encode_e8m0,
    encode_fp8_bytes,
    ml_fp8_dtype,
    uses_ocp_fp8,
    variant_from_kernel_name,
    ADTYPE_ELEMSIZE_BY_VARIANT,
    BQ_QDTYPE_BY_VARIANT,
)

# =============================================================================
# Constants
# =============================================================================

# Operator family prefix -- must match NAME_PREFIX in unified_gemm_bquant_codegen.py.
NAME_PREFIX = "gemm_bquant"

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_gemm_bquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_bquant_ctypes_lib.cpp"

# Import the shared name-construction helper from codegen_common so both sides
# stay byte-exact without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import (  # noqa: E402
    make_bquant_kernel_name,
    quant_warp_tile_k,
)

# --- Tile-Engine perf flags: single source of truth (quant_bridge_flags.py) ---
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from quant_bridge_flags import (  # noqa: E402
    TE_ORDER_BQUANT as _TE_ORDER_BQUANT,
    te_perf_flags as _te_perf_flags,
)

_DEFAULT_HIPCC = "hipcc"

# Placeholder arch used ONLY for pure name-construction in the convenience
# factory functions / dataclass default (KERNEL_NAME does not depend on arch).
# It is NOT a build fallback: the build path (setup_multiple_bquant_dispatchers)
# requires a real arch, detected via _detect_gpu_arch() (which raises) or passed
# explicitly via gfx_arch=. Do not use this to silently target a build.
_NAME_ONLY_GFX_ARCH = "gfx950"

# MX variants require gfx950 (e8m0 block scale / native MX support).
_MX_VARIANTS = {"mx_bf16bf16", "mx_bf16bf8", "mx_bf16fp4"}


def _require_mx_arch(variant_key: str, gfx_arch: str) -> None:
    """Fail early (before hipcc) if an MX variant targets a non-gfx950 arch.

    MX kernels use e8m0 block scale and native MX matrix instructions that only
    exist on gfx950; the C++ ctypes lib also #errors on the mismatch, but we
    catch it here with a clear Python-level message rather than a cryptic
    compiler failure. Mirrors the get_arch+throw policy.
    """
    if variant_key in _MX_VARIANTS and gfx_arch != "gfx950":
        raise ValueError(
            f"MX variant {variant_key!r} requires gfx950 (e8m0 block scale / "
            f"native MX support); got gfx_arch={gfx_arch!r}. "
            f"Rebuild targeting gfx950 or use a non-MX variant."
        )

# Flags that match the tile engine / dispatcher build flags for BQuant kernels
_HIPCC_BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
    "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
    "-w",  # suppress warnings during generated-code compilation
]


# --- TE backend codegen flags -------------------------------------------------
# For a FAIR bridge-vs-Old-TE comparison the bridge .so must be built with the
# SAME -mllvm backend flags that CK's CMake injects into the tile_engine example
# TU (run_gemm_quant_example).  Omitting them made the bridge kernel codegen
# materially worse -- e.g. mx_bf16bf8 4096^3 measured ~+24% slower than Old-TE
# even though BOTH sides select the identical AgBgCrCompV3 CShuffle kernel; with
# these flags the two builds are backend-identical and the gap collapses.
#
# The list mirrors mx_gemm_utils._MX_CODEGEN_FLAGS byte-for-byte.  The coerce
# flag is probe-gated: CK's CMake only adds it when check_cxx_compiler_flag
# passes, and ROCm 7.2's clang REJECTS -amdgpu-coerce-illegal-types=1, so the
# probe drops it on that toolchain -- keeping BOTH sides codegen-identical.
# The exact backend flag set Old-TE's gemm_quant TU is built with.  The flag
# strings and the coerce probe live in quant_bridge_flags (single source of
# truth); only bquant's emitted ORDER -- Old-TE's order, coerce flag first --
# is specified here, because flag order can change codegen.
_BQUANT_EXTRA_FLAGS = ("--offload-compress",)


def _bquant_codegen_flags(hipcc: str = _DEFAULT_HIPCC) -> "Tuple[str, ...]":
    """Old-TE's gemm_quant codegen flags plus any probe-gated flags the compiler
    accepts -- the exact backend flag set the TE benchmark TU is built with.

    Honours ``CK_BRIDGE_NO_TE_FLAGS=1`` like every other bridge; before the
    de-fork this one op had no such escape hatch.
    """
    return tuple(_te_perf_flags(
        hipcc,
        extra=_BQUANT_EXTRA_FLAGS,
        order=_TE_ORDER_BQUANT,
        coerce_first=True,
    ))


# =============================================================================
# BQuantKernelConfig -- byte-exact naming with codegen
# =============================================================================


@dataclass
class BQuantKernelConfig:
    """
    Complete description of one non-grouped gemm_bquant GEMM kernel.

    The .name property produces the exact string that unified_gemm_bquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    variant_key: str       # "fp8", "bf8", "fp8i4", "bf8i4", "mx_bf16*"
    layout: str            # "rcr" (A=RowMajor, B=ColMajor, C=RowMajor)
    pipeline: str          # "compv3" | "preshuffleb" | "microscale"
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

    preshuffle_b: bool      = False
    preshuffle_bquant: bool  = False
    double_smem_buffer: bool = False
    k_block_per_cu: int      = 1

    gfx_arch: str = _NAME_ONLY_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME (delegates to make_bquant_kernel_name)."""
        return make_bquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_bquant=self.preshuffle_bquant,
            name_prefix=NAME_PREFIX,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_gemm_bquant_codegen.py."""
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
            "preshuffle_b": self.preshuffle_b,
            "preshuffle_bquant": self.preshuffle_bquant,
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
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
# BQuantDispatcherLib -- thin ctypes wrapper
# =============================================================================


class BQuantDispatcherLib(DispatcherLibBase):
    """
    Loads a compiled gemm_bquant .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_bquant_gemm(A, B, BQ, C, M, N, K,
                                       stride_A, stride_B, stride_BQ, stride_C,
                                       QK_B, QN_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()

    The initialize / get_kernel_name / get_kernel_count / cleanup scaffold lives
    in DispatcherLibBase; only the op-specific dispatcher_run argtypes (below) and
    the run() marshalling stay here.
    """

    _NOT_FOUND_LABEL = "BQuant"
    _RUN_SYMBOL = "dispatcher_run_bquant_gemm"
    _RUN_ARGTYPES = [
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

        A, B, BQ, C must be numpy arrays (C-contiguous, packed).
        B should be a packed (K, N) C-contiguous array -- the kernel interprets
        it as column-major via stride_B=K, not via numpy's Fortran-order flag.
        C must be the array that will receive output; a non-contiguous C would
        produce a temporary copy that is not returned to the caller.
        Returns (status, time_ms).
        """
        import numpy as np

        A   = np.ascontiguousarray(A)
        # Kernel BLayout is ColumnMajor (rcr): B[k,n] lives at offset n*K+k.
        # Supply column-major bytes for 2-D B; ascontiguousarray would force
        # row-major and silently transpose. Packed 1-D B (fp4) stays as-is.
        B   = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        # BQ is ColumnMajor [QK_B, QN_B] (leading dim QK_B) to match Old-TE's rcr
        # path and the WPQuantB pipeline; supply fortran-order bytes for 2-D BQ.
        BQ  = np.asfortranarray(BQ) if BQ.ndim == 2 else np.ascontiguousarray(BQ)
        # Inputs may be copied into a contiguous temporary because the copy is what
        # gets uploaded. C may not: the library memcpys the device result back into
        # whatever buffer this pointer names. Copying C would send the results into a
        # temporary that is discarded on return, and the caller's array would silently
        # keep its pre-call contents.
        if not C.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "C must be a C-contiguous array; it is written in place. "
                "Pass np.ascontiguousarray(C) and copy the result back yourself, "
                "or allocate C with np.empty/np.zeros."
            )

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


# =============================================================================
# QDataType-aware BQ encoding
#
# The ctypes lib reinterprets the BQ bytes as the kernel's compile-time QDataType
# (unified_gemm_bquant_codegen.BQUANT_VARIANTS[*]["ck_q"]):
#   fp8 / bf8               -> QDataType = float      (float32, 4 bytes)
#   fp8i4                   -> QDataType = fp8_t       (1 byte, OCP e4m3)
#   bf8i4                   -> QDataType = bf8_t       (1 byte, OCP e5m2)
#   mx_bf16bf16/bf8/fp4     -> QDataType = e8m0_t      (1 byte, block-scale exp)
#
# The runner must therefore hand the .so bytes in the kernel's QDataType, NOT
# always float32.  The round-5 runner passed BQ straight through as float32 for
# every variant, so for the i4 variants the kernel read 4-byte float32 patterns
# as 1-byte fp8/bf8 -> every value became NaN (all 8 fp8i4/bf8i4 configs failed).
# =============================================================================

# variant_key -> the numpy encoder that produces the kernel's QDataType bytes.
# "float32" means "no re-encode" (fp8/bf8 plain).  The value is a tag consumed
# by _encode_bq_for_variant so callers never need to know the QDataType.
_BQ_QDTYPE_BY_VARIANT: Dict[str, str] = BQ_QDTYPE_BY_VARIANT


def _variant_from_kernel_name(name: str) -> Optional[str]:
    """This op's spelling of :func:`quant_bridge_base.variant_from_kernel_name`."""
    return variant_from_kernel_name(name, NAME_PREFIX)


def _encode_e8m0(arr) -> "object":
    """Thin spelling of :func:`quant_bridge_base.encode_e8m0`."""
    return encode_e8m0(arr)


def _uses_ocp_fp8(gfx_arch: Optional[str]) -> bool:
    """Thin spelling of :func:`quant_bridge_base.uses_ocp_fp8`."""
    return uses_ocp_fp8(gfx_arch)


def _ml_fp8_dtype(dtype: str, gfx_arch: Optional[str]):
    """Thin spelling of :func:`quant_bridge_base.ml_fp8_dtype`."""
    return ml_fp8_dtype(dtype, gfx_arch)


def _encode_fp8_bytes(arr, dtype: str, gfx_arch: Optional[str] = None) -> "object":
    """Thin spelling of :func:`quant_bridge_base.encode_fp8_bytes`."""
    return encode_fp8_bytes(arr, dtype, gfx_arch=gfx_arch)


def _encode_bq_for_variant(BQ, variant_key: Optional[str],
                           gfx_arch: Optional[str] = None) -> "object":
    """Thin spelling of :func:`quant_bridge_base.encode_bq_for_variant`."""
    return encode_bq_for_variant(BQ, variant_key, gfx_arch=gfx_arch)


# variant_key -> the numpy element size (bytes) of the kernel's compile-time
# ADataType (unified_gemm_bquant_codegen.BQUANT_VARIANTS[*]["ck_a"]):
#   fp8 / bf8 / fp8i4 / bf8i4  -> ADataType = fp8_t / bf8_t   (1 byte)
#   mx_bf16bf16/bf8/fp4        -> ADataType = bf16_t          (2 bytes)
# The ctypes lib reads elements_to_bytes<ADataType>(M*K) bytes from the host A
# pointer.  If a caller hands a uint8 A (1 byte/elem) for an MX kernel whose
# ADataType is bf16 (2 bytes/elem), the .so over-reads M*K bytes past the numpy
# allocation.  At small M*K the over-read lands in slack and only corrupts the
# result; at large M*K (e.g. M=K=2048) it walks off the mapping and the HIP DMA
# fails to pin the host source ("DmaBlitManager::getBuffer failed to pin a
# resource!") -> SEGFAULT.  _ADTYPE_ELEMSIZE_BY_VARIANT lets the runner coerce A
# to the kernel's element width before the ctypes call so the copy is always
# in-bounds.
_ADTYPE_ELEMSIZE_BY_VARIANT: Dict[str, int] = ADTYPE_ELEMSIZE_BY_VARIANT


def _coerce_a_for_variant(A, variant_key: Optional[str]) -> "object":
    """Thin spelling of :func:`quant_bridge_base.coerce_a_for_variant`."""
    return coerce_a_for_variant(A, variant_key)


# =============================================================================
# BQuantGpuGemmRunner -- high-level runner
# =============================================================================


class BQuantGpuGemmRunner:
    """
    High-level runner that loads a gemm_bquant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, BQ; allocates C; returns BQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = BQuantDispatcherLib(so_path)
        # Derive the compiled-for arch from the .so filename ("lib{name}_{arch}.so")
        # so the runner can encode fp8/bf8 BQ bytes in the matching format
        # (OCP on gfx950/gfx12*, FNUZ on gfx942/gfx90a).  None if not encoded.
        self._gfx_arch = self._arch_from_so_path(Path(so_path))

    @staticmethod
    def _arch_from_so_path(so_path: Path) -> Optional[str]:
        """Thin spelling of :func:`quant_bridge_base.arch_from_so_path`.

        The local regex this replaced was anchored at the end of the stem and so
        stopped matching once the flag digest was folded into the filename
        (``..._gfx942_9bf231b3``).  It then returned None, which the fp8 codec
        reads as "assume OCP" -- right on gfx950, wrong on gfx942.
        """
        return arch_from_so_path(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, BQ, problem: BQuantGemmProblem, c_dtype=None) -> BQuantGemmResult:
        """
        Run non-grouped BQuant GEMM.

        A       shape: (M, K)           dtype: fp8/bf8/bf16
        B       shape: (K, N) col-major  dtype: fp8/bf8/pk_int4/pk_fp4/bf16
        BQ      shape: (QK_B, QN_B)     dtype: float/fp8/e8m0
        c_dtype numpy dtype for the output C buffer.  Defaults to np.float16
                (correct for fp8/bf8/fp8i4/bf8i4 variants).  Pass np.bfloat16
                for MX variants (mx_bf16bf16, mx_bf16bf8, mx_bf16fp4) whose
                CDataType is bf16.
        Returns BQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        # Split-K trap: only k_batch == 1 is validated end-to-end for this
        # bridge. k_batch > 1 is passed through to the kernel but never verified,
        # so reject it explicitly rather than risk a silently-wrong result.
        if problem.k_batch != 1:
            raise ValueError(
                f"k_batch={problem.k_batch} is not supported by the non-grouped "
                f"gemm_bquant bridge; only k_batch == 1 is validated. "
                f"Split-K (k_batch > 1) would produce an unverified result."
            )

        M, N, K = problem.M, problem.N, problem.K
        QK_B    = problem.QK_B
        QN_B    = problem.QN_B

        if c_dtype is None:
            c_dtype = np.float16

        # QDataType-aware BQ encoding: the .so reinterprets BQ bytes as the
        # kernel's compile-time QDataType.  fp8/bf8 want float32; fp8i4/bf8i4 want
        # fp8/bf8 bytes; MX wants e8m0 uint8.  Encode float32 scales to the right
        # width here (i4 previously read 4-byte float32 as 1-byte fp8 -> NaN).
        _variant = _variant_from_kernel_name(self.kernel_name)
        BQ = _encode_bq_for_variant(
            BQ, _variant,
            gfx_arch=getattr(self, "_gfx_arch", None))

        # ADataType-aware A guard: the .so copies M*K * sizeof(ADataType) bytes
        # from the host A pointer.  MX kernels have ADataType == bf16 (2 bytes);
        # a caller passing a 1-byte A (e.g. uint8) would make the .so over-read
        # M*K bytes past the numpy allocation -- harmless slack at small M*K but a
        # host-pin failure -> SEGFAULT at large M*K (e.g. M=K=2048).  Coerce A to
        # the kernel's element width so the device copy is always in bounds.
        A = _coerce_a_for_variant(A, _variant)

        # Output buffer -- dtype must match the compiled kernel's CDataType.
        C = np.zeros((M, N), dtype=c_dtype)

        # Strides (in elements, row-major for A and C; col-major for B and BQ).
        stride_A  = K     # A is row-major [M, K]
        stride_B  = K     # B is col-major [K, N] -> leading dim = K
        stride_BQ = QK_B  # BQ is col-major [QK_B, QN_B] -> leading dim = QK_B
        stride_C  = N     # C is row-major [M, N]

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

        # PermuteNEpilogue writes C with N-columns riffled WITHIN EACH N-TILE.
        # The riffle is scoped to one TileN-wide block (kNPerBlock == TileN in the
        # epilogue), NOT the full N dimension: per M-repeat the epilogue permutes
        # r = NRepeat = TileN / (WarpN * WarpTileN) column-groups of width TileN/r,
        # then advances to the next N-tile.  Undoing it therefore has to be applied
        # independently per TileN-wide slice.  The round-5 code used a GLOBAL riffle
        # with _half = N // r, which is correct only when N == TileN (single N-tile)
        # -- at N >= 2*TileN it scrambled columns (gfx942/gfx950 tester: max_rel
        # 50-74 for MX at N=256/512).  The fix here tiles the de-riffle across N.
        #
        # The de-permute action is EPILOGUE-DEPENDENT:
        #   * PreshuffleB (WPQuantB) kernels -- name token "preshuffleb".  Their
        #     column order already comes out LOGICAL (the B-weight preshuffle
        #     shuffle_b_permuteN + bq_permuteN on the host inputs already accounts
        #     for the epilogue riffle), so the correct action is IDENTITY -- no
        #     de-permute.  The gfx942 tester confirmed any C-side riffle here
        #     (forward OR inverse) SCRAMBLES columns (max_rel 57-58); identity is
        #     exact.
        #   * CompV3 / preshufflequant / MX (microscale) kernels -- the epilogue
        #     riffle is visible in device C, so apply the per-tile INVERSE riffle
        #     to recover logical column order.
        _name = self.kernel_name
        if 'permute_n' in _name:
            import re as _re
            _is_preshuffleb = bool(_re.search(r'(?:^|_)preshuffleb(?:_|$)', _name))
            if not _is_preshuffleb:
                _m = _re.search(
                    r'_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_', _name)
                if _m:
                    _tile_n = int(_m.group(2))
                    _warp_n = int(_m.group(5))
                    _wt_n = int(_m.group(8))
                    _r = _tile_n // _wt_n // _warp_n
                    # Only the last (partial) N-tile may be narrower than TileN.
                    # The riffle is only defined on a FULL TileN-wide block whose
                    # width divides evenly by r; skip any ragged tail rather than
                    # mis-riffle it.
                    if _r > 1 and (_tile_n % _r) == 0:
                        _within = _tile_n // _r
                        # Per-tile INVERSE riffle: within each TileN-wide block the
                        # index list _logical is the same one round-5 used globally,
                        # but scoped to TileN columns.  Applying it as a SCATTER
                        # (_dst[:, _logical] = _src) is the inverse of the epilogue's
                        # forward riffle -- identical direction to the validated
                        # single-N-tile (N == TileN) round-5 CompV3 path.
                        _logical = [
                            (c % _r) * _within + (c // _r) for c in range(_tile_n)
                        ]
                        _Cp = np.empty_like(C)
                        for _n0 in range(0, N, _tile_n):
                            _w = min(_tile_n, N - _n0)
                            _src = C[:, _n0:_n0 + _w]
                            if _w == _tile_n:
                                _dst = np.empty_like(_src)
                                _dst[:, _logical] = _src
                                _Cp[:, _n0:_n0 + _tile_n] = _dst
                            else:
                                # Ragged tail (N not a multiple of TileN): the
                                # epilogue still riffles a full TileN internally but
                                # only the first _w columns are stored; copy as-is.
                                _Cp[:, _n0:_n0 + _w] = _src
                        C = _Cp
        return BQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers (self-contained, do not call ctypes_utils.py)
# =============================================================================


def _detect_gpu_arch() -> str:
    """Detect current GPU arch via rocm_agent_enumerator.

    RAISES RuntimeError if the enumerator is missing, fails, or reports no
    usable gfx target. We deliberately do NOT fall back to a hardcoded arch:
    a silent default mis-targets the kernel on non-gfx950 hosts (e.g. gfx942)
    and violates the get_arch+throw policy. Callers that know their target
    should pass ``gfx_arch=`` explicitly to skip detection entirely.
    """
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "Could not detect GPU arch: 'rocm_agent_enumerator' not found. "
            "Pass gfx_arch= explicitly (e.g. gfx_arch='gfx950')."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Could not detect GPU arch via rocm_agent_enumerator: {e}. "
            f"Pass gfx_arch= explicitly."
        ) from e

    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("gfx") and line != "gfx000":
            return line

    raise RuntimeError(
        "Could not detect a usable GPU arch (rocm_agent_enumerator returned "
        f"no gfx target; stdout={result.stdout!r}). Pass gfx_arch= explicitly."
    )


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    # Walk up from dispatcher/python/ to find project root
    return find_ck_include_dir()


def _generate_bquant_kernel(
    config: BQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """
    Run unified_gemm_bquant_codegen.py for one config; return the .hpp path or None.
    """
    return generate_kernel(config, output_dir, _CODEGEN_SCRIPT)


def _get_dispatcher_static_lib() -> Optional[Path]:
    """Return libck_tile_dispatcher.a from the CMake build directory, or None."""
    dispatcher_root = _CTYPES_LIB_SRC.parent.parent.parent
    static_lib = dispatcher_root / "build" / "libck_tile_dispatcher.a"
    return static_lib if static_lib.exists() else None


def _compile_bquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    """
    Compile a generated .hpp into a .so via hipcc (compile then link).

    Two-step build:
      1. Compile to a .o object file.
      2. Link the .o into a shared .so (no dispatcher static lib needed;
         the BQuant ctypes lib does not use the registry or dispatcher).

    Returns True on success.
    """
    ck_include = _get_ck_include_dir()
    static_lib = _get_dispatcher_static_lib()

    # -- Step 1: compile to object file --------------------------------------
    obj_path = so_path.with_suffix(".o")

    # Arch-specific defines: gfx950 uses OCP fp8 (not FNUZ) and native MX support.
    # These mirror the CMakeLists.txt definitions that are normally injected by CMake
    # but are absent in the standalone hipcc build path.
    arch_defines = []
    if "gfx12" in gfx_arch or "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]
    if "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]

    # TE backend codegen flags: mirror the -mllvm set CK's CMake injects into the
    # tile_engine gemm_quant example so the bridge .so is backend-identical to
    # Old-TE.  Probe-gated (drops -amdgpu-coerce-illegal-types=1 on toolchains
    # that reject it, e.g. ROCm 7.2) so the build stays portable and fair.
    codegen_flags = list(_bquant_codegen_flags(hipcc))

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}",
                   f"-DGFX_ARCH=\"{gfx_arch}\"",
                   *arch_defines,
                   *codegen_flags,
                   "-include", str(hpp_path),
                   str(_CTYPES_LIB_SRC),
                   "-o", str(obj_path)]

    if ck_include:
        compile_cmd += [f"-I{ck_include}"]

    # NOTE: dispatcher/include is intentionally excluded here.
    # It pulls in generated_tile_backend.hpp which instantiates
    # SelectedKernel::launch(GemmHostArgs&), conflicting with the BQuant
    # kernel's launch(QuantGemmHostArgs&). The BQuant ctypes lib only needs
    # the main CK include path (ck_tile/host/tensor_shuffle_utils.hpp lives there).

    if extra_include_dirs:
        for d in extra_include_dirs:
            compile_cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(compile_cmd))

    try:
        result = subprocess.run(
            compile_cmd,
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False

    # -- Step 2: link into shared library ------------------------------------
    link_cmd = [hipcc, "-shared", "-fPIC",
                f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path)]

    if static_lib:
        link_cmd += [str(static_lib)]

    link_cmd += ["-o", str(so_path)]

    log.debug("Linking %s:\n  %s", so_path.name, " ".join(link_cmd))

    try:
        result = subprocess.run(
            link_cmd,
            capture_output=True, text=True, timeout=120,
        )
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
# setup_multiple_bquant_dispatchers -- build pipeline
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
    For each BQuantKernelConfig: codegen -> hipcc compile -> .so path.

    Returns a list parallel to `configs` -- each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()

    # Python-side MX guard: fail early (before hipcc) if any MX variant targets a
    # non-gfx950 arch, rather than relying solely on the C++ #error. Mirrors
    # get_arch+throw.
    for cfg in configs:
        _require_mx_arch(cfg.variant_key, arch)

    def _compile_fn(hpp: Path, so: Path, a: str) -> bool:
        return _compile_bquant_kernel(
            hpp_path=hpp, so_path=so, gfx_arch=a,
            hipcc=hipcc, extra_include_dirs=extra_include_dirs,
        )

    return build_dispatchers(
        configs,
        arch=arch,
        tmp_prefix="gemm_bquant_dispatcher_",
        log_label="gemm_bquant",
        generate_fn=_generate_bquant_kernel,
        compile_fn=_compile_fn,
        output_dir=output_dir,
        parallel=parallel,
        max_workers=max_workers,
    )


# =============================================================================
# Sweep expansion: JSON config -> list of BQuantKernelConfig
# =============================================================================


def expand_bquant_sweep(
    config_path: str,
    gfx_arch: str = _NAME_ONLY_GFX_ARCH,
) -> List["BQuantKernelConfig"]:
    """Expand a BQuant JSON sweep config into a list of BQuantKernelConfig objects.

    The JSON format mirrors unified_gemm_bquant_codegen.py's _build_specs
    so the same config files work for both codegen and Python utils. Every valid
    (variant, layout, tile, quant_group) combination produces one BQuantKernelConfig;
    duplicates (by .name) are collapsed.

    JSON schema:
      variant_keys:       list of dtype variants, e.g. ["fp8", "bf8"]
      layouts:            list of layout strings, e.g. ["rcr"]
      pipeline:           pipeline name, e.g. "compv3"
      epilogue:           epilogue name, e.g. "cshuffle"
      scheduler:          scheduler name, e.g. "intrawave"
      tile_configs:       list of {tile_m, tile_n, tile_k, warp_m, warp_n, warp_k,
                                   warp_tile_m, warp_tile_n, warp_tile_k}
      quant_groups:       list of {quant_group_m, quant_group_n, quant_group_k}
      pad_m/pad_n/pad_k:  bool
      block_size:         int (default 256)
      k_block_per_cu:     int (default 1)
      double_smem_buffer: bool (default false)
      preshuffle_b:       bool (default false)
      preshuffle_bquant:  bool (default false)
    """
    import itertools

    with open(config_path) as f:
        cfg = json.load(f)

    pipeline          = cfg.get("pipeline", "compv3")
    epilogue          = cfg.get("epilogue", "cshuffle")
    scheduler         = cfg.get("scheduler", "intrawave")
    block_size        = cfg.get("block_size", 256)
    k_block_per_cu    = cfg.get("k_block_per_cu", 1)
    double_smem_buffer = cfg.get("double_smem_buffer", False)
    preshuffle_b      = cfg.get("preshuffle_b", False)
    preshuffle_bquant = cfg.get("preshuffle_bquant", False)

    configs: List[BQuantKernelConfig] = []
    seen: set = set()

    for variant_key, layout, tile_dict, qg in itertools.product(
        cfg.get("variant_keys", ["fp8"]),
        cfg.get("layouts", ["rcr"]),
        cfg.get("tile_configs", []),
        cfg.get("quant_groups", [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        c = BQuantKernelConfig(
            variant_key=variant_key,
            layout=layout,
            pipeline=pipeline,
            epilogue=epilogue,
            scheduler=scheduler,
            tile_m=tile_dict["tile_m"],
            tile_n=tile_dict["tile_n"],
            tile_k=tile_dict["tile_k"],
            warp_m=tile_dict["warp_m"],
            warp_n=tile_dict["warp_n"],
            warp_k=tile_dict["warp_k"],
            warp_tile_m=tile_dict["warp_tile_m"],
            warp_tile_n=tile_dict["warp_tile_n"],
            warp_tile_k=tile_dict["warp_tile_k"],
            quant_group_m=qg.get("quant_group_m", 1),
            quant_group_n=qg.get("quant_group_n", 1),
            quant_group_k=qg.get("quant_group_k", 128),
            preshuffle_b=preshuffle_b,
            preshuffle_bquant=preshuffle_bquant,
            double_smem_buffer=double_smem_buffer,
            k_block_per_cu=k_block_per_cu,
            gfx_arch=gfx_arch,
        )
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)

    return configs


# =============================================================================
# Convenience default configs (mirror gemm_utils.hpp GemmConfig* tile choices)
#
# Non-preshuffle decode tiles (GemmConfigQuantDecode): fp8/bf8/fp8i4/bf8i4.
# warp_tile_k is ARCH-DERIVED (never hardcoded); see _warp_tile_k_for below.
# =============================================================================


def _warp_tile_k_for(gfx_arch: str, is_flatmm: bool = False) -> int:
    """Arch-derived K warp-tile, mirroring ck_tile::get_k_warp_tile<PrecType, 16, IsFlatMM>().

    (tile_gemm_shape.hpp:104-136, M_Warp_Tile=16, non-WMMA path.)  Every non-MX
    BQuant variant -- fp8, bf8, fp8i4, bf8i4 -- instantiates the GEMM config with an
    8-bit float PrecType (GemmConfig<fp8_t>/<bf8_t>; the pk_int4 B operand does NOT
    drive the K warp tile -- see gemm_bquant_quantgrouped{,_preshuffleb,_preshufflequant}
    _*.cpp, which all pass GemmConfig<ck_tile::fp8_t> / <bf8_t> even for the i4 files).
    So is_8bit_float is always True and warp_tile_k depends only on arch + pipeline:

      gfx950 (CK_GFX950_SUPPORT): 128   (both decode IsFlatMM=false and preshuffle)
      gfx942/other, decode (IsFlatMM=false)   : 32
      gfx942/other, preshuffle_b (IsFlatMM=true): 64

    This is a BLOCKING correctness constraint, not just a naming detail: a
    warp_tile_k=128 fp8/bf8 kernel *compiles* on gfx942 but silently produces
    ALL-ZEROS output (there is no valid 16x16x128 fp8/bf8 warp-gemm on gfx942).
    GPU-confirmed on the sibling tensor_quant/rowcolquant/aquant/abquant bridges.
    Note: the pre-fix code additionally hardcoded warp_tile_k=16 for the i4 decode
    variants, which get_k_warp_tile<fp8_t,16>() never returns for M_Warp_Tile=16 --
    that was wrong on BOTH arches.
    """
    return quant_warp_tile_k(gfx_arch, is_8bit_float=True, is_flat_mm=is_flatmm)


# =============================================================================
# Decode family (BQuantGemmPipelineAgBgCrCompV3, tile 16x64x256)
#   GemmConfigBQuantDecode: warp 1x4x1, warp_tile 16x16x{K_warp}
#   fp8/bf8/fp8i4/bf8i4: K_warp = 128 on gfx950, 32 on gfx942.
# =============================================================================


def default_fp8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _NAME_ONLY_GFX_ARCH,
) -> BQuantKernelConfig:
    """Default fp8 BQuant config (tile = 16x64x256, warp = 1x4x1).

    warp_tile_k is arch-derived (get_k_warp_tile<fp8_t, 16>()): 128 on gfx950,
    32 on gfx942 (128 silently outputs all-zeros on gfx942).
    """
    return BQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch),
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _NAME_ONLY_GFX_ARCH,
) -> BQuantKernelConfig:
    """Default bf8 BQuant config (tile = 16x64x256, warp = 1x4x1).

    warp_tile_k is arch-derived (get_k_warp_tile<bf8_t, 16>()): 128 on gfx950,
    32 on gfx942 (128 silently outputs all-zeros on gfx942).
    """
    return BQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch),
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_fp8i4_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _NAME_ONLY_GFX_ARCH,
) -> BQuantKernelConfig:
    """Default fp8i4 BQuant config (A=fp8, B=pk_int4, Q=fp8; tile = 16x64x256).

    warp_tile_k is arch-derived: the i4 decode kernel is instantiated as
    GemmConfigQuantDecode<ck_tile::fp8_t> (the pk_int4 B operand does NOT change
    K_Warp_Tile), so it matches fp8 -- 128 on gfx950, 32 on gfx942.  The prior
    hardcoded 16 was wrong on both arches (get_k_warp_tile<fp8_t,16>() never
    returns 16 for M_Warp_Tile=16).
    """
    return BQuantKernelConfig(
        variant_key="fp8i4",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch),
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_bf8i4_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _NAME_ONLY_GFX_ARCH,
) -> BQuantKernelConfig:
    """Default bf8i4 BQuant config (A=bf8, B=pk_int4, Q=bf8; tile = 16x64x256).

    warp_tile_k is arch-derived: the i4 decode kernel is instantiated as
    GemmConfigQuantDecode<ck_tile::bf8_t> (the pk_int4 B operand does NOT change
    K_Warp_Tile), so it matches bf8 -- 128 on gfx950, 32 on gfx942.  The prior
    hardcoded 16 was wrong on both arches.
    """
    return BQuantKernelConfig(
        variant_key="bf8i4",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch),
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


# =============================================================================
# preshuffle_b only (WPQuantBPipelineAgBgCrV2, prefill tile 128x128x128)
#   GemmConfigPreshuffleB_BQuant_Prefill: warp 1x4x1, DoubleSmemBuffer=true, kBlockPerCu=2
#   fp8/bf8 K_warp=128; pk_int4 K_warp=32.
# =============================================================================


def _preshuffleb_config(variant_key, warp_tile_k, quant_group_k, quant_group_n, gfx_arch):
    return BQuantKernelConfig(
        variant_key=variant_key, layout="rcr", pipeline="preshuffleb",
        epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1, quant_group_n=quant_group_n, quant_group_k=quant_group_k,
        preshuffle_b=True, preshuffle_bquant=False,
        double_smem_buffer=True, k_block_per_cu=2, gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleb_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """fp8 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<fp8_t>).

    IsFlatMM=true: warp_tile_k = 128 on gfx950, 64 on gfx942.
    """
    return _preshuffleb_config(
        "fp8", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


def default_bf8_preshuffleb_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """bf8 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<bf8_t>).

    IsFlatMM=true: warp_tile_k = 128 on gfx950, 64 on gfx942.
    """
    return _preshuffleb_config(
        "bf8", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


def default_fp8i4_preshuffleb_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """fp8i4 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<fp8_t>).

    Instantiated with 8-bit-float PrecType (pk_int4 B does not drive K_Warp_Tile),
    IsFlatMM=true: warp_tile_k = 128 on gfx950, 64 on gfx942 (prior hardcoded 32
    was wrong on both arches).
    """
    return _preshuffleb_config(
        "fp8i4", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


def default_bf8i4_preshuffleb_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """bf8i4 preshuffle_b prefill config (GemmConfigPreshuffleB_BQuant_Prefill<bf8_t>).

    Instantiated with 8-bit-float PrecType (pk_int4 B does not drive K_Warp_Tile),
    IsFlatMM=true: warp_tile_k = 128 on gfx950, 64 on gfx942 (prior hardcoded 32
    was wrong on both arches).
    """
    return _preshuffleb_config(
        "bf8i4", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


# =============================================================================
# preshuffle_bquant only (BQuantGemmPipelineAgBgCrCompV3, prefill tile 128x128x128)
#   GemmConfigPreshuffleBQuantPrefill: warp 1x4x1, DoubleSmemBuffer=false, kBlockPerCu=1
# =============================================================================


def _preshufflequant_config(variant_key, warp_tile_k, quant_group_k, quant_group_n, gfx_arch):
    return BQuantKernelConfig(
        variant_key=variant_key, layout="rcr", pipeline="compv3",
        epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1, quant_group_n=quant_group_n, quant_group_k=quant_group_k,
        preshuffle_b=False, preshuffle_bquant=True, gfx_arch=gfx_arch,
    )


def default_fp8_preshufflequant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """fp8 preshuffle_bquant prefill config (GemmConfigPreshuffleBQuantPrefill<fp8_t>).

    Derives from GemmConfigQuantPrefill (IsFlatMM=false): 128 on gfx950, 32 on gfx942.
    """
    return _preshufflequant_config(
        "fp8", _warp_tile_k_for(gfx_arch), quant_group_k, quant_group_n, gfx_arch)


def default_bf8_preshufflequant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """bf8 preshuffle_bquant prefill config (IsFlatMM=false: 128 gfx950, 32 gfx942)."""
    return _preshufflequant_config(
        "bf8", _warp_tile_k_for(gfx_arch), quant_group_k, quant_group_n, gfx_arch)


def default_fp8i4_preshufflequant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """fp8i4 preshuffle_bquant prefill config (8-bit PrecType; 128 gfx950, 32 gfx942)."""
    return _preshufflequant_config(
        "fp8i4", _warp_tile_k_for(gfx_arch), quant_group_k, quant_group_n, gfx_arch)


def default_bf8i4_preshufflequant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """bf8i4 preshuffle_bquant prefill config (8-bit PrecType; 128 gfx950, 32 gfx942)."""
    return _preshufflequant_config(
        "bf8i4", _warp_tile_k_for(gfx_arch), quant_group_k, quant_group_n, gfx_arch)


# =============================================================================
# preshuffle_b + preshuffle_bquant (WPQuantBPipelineAgBgCrV2, prefill tile)
#   GemmConfigPreshuffleB_PreshuffleBQuant_Prefill: same tile as preshuffleb,
#   PreshuffleB=true, BPreshuffleQuant=true, DoubleSmemBuffer=true, kBlockPerCu=2
# =============================================================================


def _preshuffleb_bquant_config(variant_key, warp_tile_k, quant_group_k, quant_group_n, gfx_arch):
    return BQuantKernelConfig(
        variant_key=variant_key, layout="rcr", pipeline="preshuffleb",
        epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1, quant_group_n=quant_group_n, quant_group_k=quant_group_k,
        preshuffle_b=True, preshuffle_bquant=True,
        double_smem_buffer=True, k_block_per_cu=2, gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleb_bquant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """fp8 preshuffle_b+preshuffle_bquant config (IsFlatMM=true: 128 gfx950, 64 gfx942)."""
    return _preshuffleb_bquant_config(
        "fp8", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


def default_bf8_preshuffleb_bquant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """bf8 preshuffle_b+preshuffle_bquant config (IsFlatMM=true: 128 gfx950, 64 gfx942)."""
    return _preshuffleb_bquant_config(
        "bf8", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


def default_fp8i4_preshuffleb_bquant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """fp8i4 preshuffle_b+preshuffle_bquant config (8-bit PrecType; 128 gfx950, 64 gfx942)."""
    return _preshuffleb_bquant_config(
        "fp8i4", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


def default_bf8i4_preshuffleb_bquant_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """bf8i4 preshuffle_b+preshuffle_bquant config (8-bit PrecType; 128 gfx950, 64 gfx942)."""
    return _preshuffleb_bquant_config(
        "bf8i4", _warp_tile_k_for(gfx_arch, is_flatmm=True), quant_group_k, quant_group_n, gfx_arch)


# =============================================================================
# MX microscale variants (A=bf16, Q=e8m0 block scale) -- gfx950 ONLY.
#   Pipeline: MicroscaleGemmPipelineAgBgCrCompV3, tile 128x128x128, warp 1x4x1.
#   mx_bf16bf16 / mx_bf16fp4: warp_tile_k=32; mx_bf16bf8: warp_tile_k=64.
#
# These warp_tile_k values are the correct gfx950 values (NOT hardcoded across
# arches): MX is gfx950-only (enforced by _require_mx_arch + the C++ #error), so
# there is no gfx942 case to derive.  Verified against Old-TE gemm_utils.hpp:
#   - mx_bf16bf16 / mx_bf16fp4: GemmConfigQuantPrefill<bf16_t> ->
#       get_k_warp_tile<bf16_t, 16>() on gfx950 = 32 (non-8bit-float branch).
#   - mx_bf16bf8: GemmConfigMixedPrecision -> K_Warp_Tile hardcoded 64 in Old-TE.
# =============================================================================


def default_mx_bf16bf16_config(quant_group_k=32, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """MX bf16+bf16 config (A=bf16, B=bf16, Q=e8m0; GemmConfigQuantPrefill<bf16_t>)."""
    return BQuantKernelConfig(
        variant_key="mx_bf16bf16", layout="rcr", pipeline="microscale",
        epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=1, quant_group_n=quant_group_n, quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_mx_bf16bf8_config(quant_group_k=128, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """MX bf16+bf8 config (A=bf16, B=bf8, Q=e8m0; GemmConfigMixedPrecision, warp_tile_k=64)."""
    return BQuantKernelConfig(
        variant_key="mx_bf16bf8", layout="rcr", pipeline="microscale",
        epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=64,
        quant_group_m=1, quant_group_n=quant_group_n, quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_mx_bf16fp4_config(quant_group_k=32, quant_group_n=1, gfx_arch=_NAME_ONLY_GFX_ARCH):
    """MX bf16+fp4 config (A=bf16, B=pk_fp4, Q=e8m0; GemmConfigQuantPrefill<bf16_t>)."""
    return BQuantKernelConfig(
        variant_key="mx_bf16fp4", layout="rcr", pipeline="microscale",
        epilogue="cshuffle", scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=32,
        quant_group_m=1, quant_group_n=quant_group_n, quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )
