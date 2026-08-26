#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
GroupedGemm ABQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's ABQuantGrouped GEMM path:

  ABQuantKernelConfig  — describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  ABQuantDispatcherLib — thin ctypes wrapper around a compiled .so
  ABQuantGpuGemmRunner — high-level runner that accepts numpy arrays

Build helpers:
  setup_multiple_abquant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

ABQuant: both A-side and B-side quantization active simultaneously.
  AQ[ceil(M/aM), ceil(K/aK)] — A-side scale (RowMajor)
  BQ[ceil(K/bK), ceil(N/bN)] — B-side scale (RowMajor)
  Constraint: aquant_group_k == bquant_group_k

Usage:
  configs = [ABQuantKernelConfig(variant_key="fp8", pipeline="compv3", ...)]
  so_paths = setup_multiple_abquant_dispatchers(configs, output_dir=Path("/tmp/abq"))
  runner = ABQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, AQ, BQ, ABQuantGemmProblem(M=128, N=128, K=128))
"""

import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import concurrent.futures

log = logging.getLogger(__name__)

# =============================================================================
# Constants
# =============================================================================

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_grouped_gemm_abquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "grouped_gemm_abquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import (  # noqa: E402
    make_abquant_kernel_name,
    quant_warp_tile_k,
    variant_is_8bit_float,
)

# --- Tile-Engine perf flags: single source of truth (quant_bridge_flags.py) ---
# Without these the .so is built with plain -O3 while the Old-TE baseline it is
# compared against carries the full TE -mllvm set, which biases every parity
# number AGAINST the bridge.
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))
from quant_bridge_flags import te_perf_flags as _te_perf_flags  # noqa: E402
from quant_bridge_base import build_dispatchers  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"

_HIPCC_BASE_FLAGS = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-shared",
    "-DCK_TILE_SINGLE_KERNEL_INCLUDE",
    "-w",
]


# =============================================================================
# ABQuantKernelConfig — byte-exact naming with codegen
# =============================================================================


@dataclass
class ABQuantKernelConfig:
    """
    Complete description of one ABQuantGrouped GEMM kernel.

    Constraint: aquant_group_k must equal bquant_group_k.

    Pipelines:
      "compv3"      — ABQuantGemmPipelineAgBgCrCompV3 (non-gfx950, GemmConfigABQuantPrefill)
      "eightwaves"  — ABQuantGemmPipelineAgBgCrEightWaves (gfx950, TransposeC=true)
      "preshuffleb" — WPABQuantBPipelineAgBgCrV2 (GemmConfigPreshuffleB_ABQuant_Prefill)
    """

    variant_key: str    # "fp8", "bf8"
    layout: str         # "rcr"
    pipeline: str       # "compv3", "eightwaves", "preshuffleb"
    epilogue: str       # "cshuffle" (effective epilogue may be permute_n)
    scheduler: str      # "intrawave"

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    # A-side quantization group
    aquant_group_m: int = 1
    aquant_group_n: int = 1
    aquant_group_k: int = 128

    # B-side quantization group
    bquant_group_m: int = 1
    bquant_group_n: int = 1
    bquant_group_k: int = 128

    preshuffle_b: bool  = False   # PreshuffleB
    preshuffle_aq: bool = False   # APreshuffleQuant
    preshuffle_bq: bool = False   # BPreshuffleQuant
    transpose_c: bool   = False   # true for eightwaves/gfx950
    double_smem_buffer: bool = False
    k_block_per_cu: int = 1

    gfx_arch: str = _DEFAULT_GFX_ARCH

    def __post_init__(self):
        if self.aquant_group_k != self.bquant_group_k:
            raise ValueError(
                f"ABQuant requires aquant_group_k == bquant_group_k, "
                f"got {self.aquant_group_k} != {self.bquant_group_k}"
            )

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME."""
        return make_abquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
            aquant_group_m=self.aquant_group_m,
            aquant_group_n=self.aquant_group_n,
            aquant_group_k=self.aquant_group_k,
            bquant_group_m=self.bquant_group_m,
            bquant_group_n=self.bquant_group_n,
            bquant_group_k=self.bquant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_aq=self.preshuffle_aq,
            preshuffle_bq=self.preshuffle_bq,
            transpose_c=self.transpose_c,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_grouped_gemm_abquant_codegen.py."""
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
            "aquant_groups": [{
                "aquant_group_m": self.aquant_group_m,
                "aquant_group_n": self.aquant_group_n,
                "aquant_group_k": self.aquant_group_k,
            }],
            "bquant_groups": [{
                "bquant_group_m": self.bquant_group_m,
                "bquant_group_n": self.bquant_group_n,
                "bquant_group_k": self.bquant_group_k,
            }],
            "preshuffle_b": self.preshuffle_b,
            "preshuffle_aq": self.preshuffle_aq,
            "preshuffle_bq": self.preshuffle_bq,
            "transpose_c": self.transpose_c,
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
        }


# =============================================================================
# ABQuantGemmProblem
# =============================================================================


@dataclass
class ABQuantGemmProblem:
    M: int
    N: int
    K: int
    aquant_group_m: int = 1
    aquant_group_n: int = 1
    aquant_group_k: int = 128
    bquant_group_m: int = 1
    bquant_group_n: int = 1
    bquant_group_k: int = 128
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        """ceil(K / aquant_group_k)"""
        return (self.K + self.aquant_group_k - 1) // self.aquant_group_k

    @property
    def QM_A(self) -> int:
        """ceil(M / aquant_group_m) — typically == M when aM=1"""
        return (self.M + self.aquant_group_m - 1) // self.aquant_group_m

    @property
    def QK_B(self) -> int:
        """ceil(K / bquant_group_k)"""
        return (self.K + self.bquant_group_k - 1) // self.bquant_group_k

    @property
    def QN_B(self) -> int:
        """ceil(N / bquant_group_n)"""
        return (self.N + self.bquant_group_n - 1) // self.bquant_group_n


# =============================================================================
# ABQuantGemmResult
# =============================================================================


@dataclass
class ABQuantGemmResult:
    C: object          # numpy array (M, N)
    time_ms: float
    kernel_name: str


# =============================================================================
# ABQuantDispatcherLib — thin ctypes wrapper
# =============================================================================


class ABQuantDispatcherLib:
    """
    Loads a compiled abquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_grouped_abquant_gemm(A, B, AQ, BQ, C, M, N, K,
                                       stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                                       QK_A, QM_A, QK_B, QN_B, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"ABQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_grouped_abquant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_grouped_abquant_gemm.argtypes = [
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
            ctypes.c_int64,    # QM_A
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
        A, B, AQ, BQ, C,
        M: int, N: int, K: int,
        stride_A: int, stride_B: int,
        stride_AQ: int, stride_BQ: int, stride_C: int,
        QK_A: int, QM_A: int, QK_B: int, QN_B: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """Call dispatcher_run_grouped_abquant_gemm. Returns (status, time_ms)."""
        import numpy as np

        A  = np.ascontiguousarray(A)
        # B is col-major [K, N]: Fortran order makes the leading dim = K (stride_B = K).
        B  = np.asfortranarray(B)
        AQ = np.ascontiguousarray(AQ)
        # BQ is col-major [QK_B, QN_B]: Fortran order makes leading dim = QK_B (stride_BQ = QK_B).
        BQ = np.asfortranarray(BQ)
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

        rc = self._lib.dispatcher_run_grouped_abquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            AQ.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K),
            ctypes.c_int64(stride_A), ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_AQ), ctypes.c_int64(stride_BQ), ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_A), ctypes.c_int64(QM_A),
            ctypes.c_int64(QK_B), ctypes.c_int64(QN_B),
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
# ABQuantGpuGemmRunner — high-level runner
# =============================================================================


def _shuffle_b_cdna(B: "np.ndarray", wt_n: int, wt_k: int) -> "np.ndarray":
    """Apply ck_tile::shuffle_b (gfx9/CDNA path) to B in-Python.

    Mirrors the gfx9 branch of tensor_shuffle_utils.hpp::shuffle_b<GemmConfig>:
      KLane            = warp_size / wt_n  (64 / wt_n)
      ItemsPerAccess   = min(16 / sizeof(fp8) = 16,  wt_k / KLane)
      view shape       = [N/wt_n, wt_n, K/items, items]   (C++ row-major)
      permute          = {0, 2, 1, 3}

    B must be a numpy array with shape (K, N) and uint8 dtype (fp8/bf8 bytes).
    The shuffle is applied before uploading to the device; the kernel reads
    the shuffled layout as [N/wt_n, K/items, wt_n, items] (after permute).
    """
    import numpy as np
    K, N = B.shape
    kLane = 64 // wt_n
    items = min(16, wt_k // kLane)
    # C++ copies B col-major (HostTensor[K,N] with stride=[1,K]) into
    # a row-major view [N/wt_n, wt_n, K/items, items], then permutes {0,2,1,3}.
    flat = B.flatten(order='F')                      # col-major byte order
    view = flat.reshape(N // wt_n, wt_n, K // items, items)
    return np.ascontiguousarray(view.transpose(0, 2, 1, 3)).reshape(B.shape)


class ABQuantGpuGemmRunner:
    """High-level runner for ABQuantGrouped GEMM."""

    def __init__(self, so_path: Path):
        self._lib = ABQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, BQ, problem: ABQuantGemmProblem, c_dtype=None) -> ABQuantGemmResult:
        """
        Run ABQuantGrouped GEMM.

        A   shape: (M, K)            dtype: fp8/bf8
        B   shape: (K, N) col-major   dtype: fp8/bf8
        AQ  shape: (QM_A, QK_A)      dtype: float32 (A-side scale, RowMajor)
        BQ  shape: (QK_B, QN_B)      dtype: float32 (B-side scale, ColumnMajor)
        c_dtype: numpy dtype for C output buffer. Defaults to np.float16.
                 Pass np.bfloat16 for MX variants whose CDataType is bf16.
        """
        import numpy as np
        import re as _re

        # Split-K gate.  Only k_batch == 1 has ever been verified on device
        # through this bridge: the round-3 default_config sweep recorded A6
        # (split-K) as NOT-COVERED for all 74 shipped configs, on both arches.
        # The underlying kernel does accept k_batch > 1 for some quant types
        # (gemm_quant_kernel.hpp:1287-1296), and the per-launch C clear the
        # split-K accumulation needs exists in quant_bridge_common.hpp -- but
        # "the kernel accepts it" is not "this bridge produces the right answer
        # with it".  Reject explicitly rather than return an unverified result;
        # lifting this needs an on-device A6 gate, not a deleted check.
        if problem.k_batch != 1:
            raise ValueError(
                f"k_batch={problem.k_batch} is not supported by the grouped_gemm_abquant "
                f"bridge; only k_batch == 1 is verified on device. Split-K "
                f"(k_batch > 1) would produce an unverified result."
            )

        M, N, K = problem.M, problem.N, problem.K
        QK_A = problem.QK_A
        QM_A = problem.QM_A
        QK_B = problem.QK_B
        QN_B = problem.QN_B

        if c_dtype is None:
            c_dtype = np.float16

        C = np.zeros((M, N), dtype=c_dtype)

        # PreshuffleB kernels require B to be host-shuffled before upload.
        # Mirrors ck_tile::shuffle_b (gfx9/CDNA path) used by the C++ test fixture.
        # After shuffling, pass B as 1-D so the ctypes layer's asfortranarray(B)
        # is a no-op (1-D arrays have only one memory order).
        _name = self.kernel_name
        if 'preshuffleb' in _name:
            _pm = _re.search(r'_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_', _name)
            if _pm:
                _wt_n = int(_pm.group(8)); _wt_k = int(_pm.group(9))
                B = _shuffle_b_cdna(B, wt_n=_wt_n, wt_k=_wt_k).ravel()

        # Stride layout:
        # AQ RowMajor [QM_A, QK_A]: stride_AQ = QK_A
        # BQ ColumnMajor [QK_B, QN_B]: stride_BQ = QK_B (leading dim = K-groups)
        stride_A   = K
        stride_B   = K    # col-major B: leading dim = K
        stride_AQ  = QK_A
        stride_BQ  = QK_B
        stride_C   = N

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_AQ=stride_AQ, stride_BQ=stride_BQ, stride_C=stride_C,
            QK_A=QK_A, QM_A=QM_A, QK_B=QK_B, QN_B=QN_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_grouped_abquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        # permute_n epilogue riffles N-columns within each tile of width tile_n.
        # Undo it per-tile so the caller gets logical (row-major) C.
        _name = self.kernel_name
        if 'permute_n' in _name:
            import re as _re
            _m = _re.search(r'_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_(\d+)x(\d+)x(\d+)_', _name)
            if _m:
                _tile_n = int(_m.group(2)); _warp_n = int(_m.group(5)); _wt_n = int(_m.group(8))
                _r = _tile_n // _wt_n // _warp_n
                if _r > 1 and (N % _tile_n) == 0:
                    _half = _tile_n // _r
                    _logical = [
                        (c // _tile_n) * _tile_n + (c % _tile_n % _r) * _half + (c % _tile_n // _r)
                        for c in range(N)
                    ]
                    _Cp = np.empty_like(C)
                    _Cp[:, _logical] = C
                    C = _Cp

        return ABQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers
# =============================================================================


def _detect_gpu_arch() -> str:
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"], capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                return line
    except Exception:
        pass
    return _DEFAULT_GFX_ARCH


def _get_ck_include_dir() -> Optional[Path]:
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_abquant_kernel(
    config: ABQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    config_json = json.dumps(config.to_codegen_config())
    cmd = [
        sys.executable, str(_CODEGEN_SCRIPT),
        "--output-dir", str(output_dir),
        "--config-json", config_json,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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


def _compile_abquant_kernel(
    hpp_path: Path,
    so_path: Path,
    gfx_arch: str,
    hipcc: str = _DEFAULT_HIPCC,
    extra_include_dirs: Optional[List[str]] = None,
) -> bool:
    ck_include = _get_ck_include_dir()

    cmd = [hipcc] + _HIPCC_BASE_FLAGS + _te_perf_flags(
        hipcc,
        extra=["-mllvm", "-enable-noalias-to-md-conversion=1",
               "-mllvm", "-greedy-reverse-local-assignment=1"],
    ) + [
        f"--offload-arch={gfx_arch}",
        f"-DGFX_ARCH=\"{gfx_arch}\"",
        "-include", str(hpp_path),
        str(_CTYPES_LIB_SRC),
        "-o", str(so_path),
    ]

    if ck_include:
        cmd += [f"-I{ck_include}"]

    # NOTE: dispatcher/include is intentionally excluded here — the abquant ctypes lib
    # calls SelectedKernel::launch() directly and does not use any dispatcher headers.

    if extra_include_dirs:
        for d in extra_include_dirs:
            cmd += [f"-I{d}"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
        return True
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False


# =============================================================================
# setup_multiple_abquant_dispatchers — build pipeline
# =============================================================================


def setup_multiple_abquant_dispatchers(
    configs: List[ABQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """For each ABQuantKernelConfig: codegen -> hipcc compile -> .so path."""
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    def _compile_fn(hpp: Path, so: Path, a: str) -> bool:
        return _compile_abquant_kernel(
            hpp_path=hpp, so_path=so, gfx_arch=a,
            hipcc=hipcc, extra_include_dirs=extra_include_dirs,
        )

    # Shared builder: dedupe-by-name, parallel fan-out, and -- the reason this
    # module no longer rolls its own loop -- a ``.so`` filename that carries a
    # digest of the compile flags.  Keyed on name+arch alone, flipping
    # CK_BRIDGE_NO_TE_FLAGS (or moving to a toolchain where the coerce probe
    # answers differently) silently reused a ``.so`` built with the other flag
    # set.  This bridge only gained switchable TE flags this cycle, so the
    # defect was newly live here.
    return build_dispatchers(
        configs,
        arch=arch,
        tmp_prefix="abquant_dispatcher_",
        log_label="ABQuant",
        generate_fn=_generate_abquant_kernel,
        compile_fn=_compile_fn,
        output_dir=output_dir,
        parallel=parallel,
        max_workers=max_workers,
        hipcc=hipcc,
    )


# =============================================================================
# Default configs (mapped from reference examples)
# =============================================================================


def default_fp8_compv3_config(
    quant_group_k: int = 128,
    bquant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """fp8 ABQuant CompV3 config (GemmConfigABQuantPrefill<fp8_t>, TransposeC=False).

    Tile: 128x128x128, warp 1x4x1. kPadK=false.

    warp_tile_k = get_k_warp_tile<fp8_t, 16, IsFlatMM=false>(): 128 on gfx950,
    32 on gfx942.  The prior text here claimed CompV3 "must stay at 32 regardless
    of arch because it does not use FlatMM"; that is not what the rule says.
    get_k_warp_tile (tile_gemm_shape.hpp:122-128) ignores IsFlatMM entirely on
    the CK_GFX950_SUPPORT branch, and GemmConfigABQuantPrefill -> GemmConfigQuantPrefill
    (gemm_utils.hpp:256-269) derives K_Warp_Tile with the same call the eightwaves
    sibling in this module already uses.  32 on gfx950 is the gfx942 value: it
    computes correctly but is not the shipped Old-TE tile.
    """
    return ABQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16,
        warp_tile_k=_warp_tile_k_for(gfx_arch, "fp8"),
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=quant_group_k,
        bquant_group_m=1, bquant_group_n=bquant_group_n, bquant_group_k=quant_group_k,
        preshuffle_b=False, preshuffle_aq=False, preshuffle_bq=False,
        transpose_c=False,
        gfx_arch=gfx_arch,
    )


def default_bf8_compv3_config(
    quant_group_k: int = 128,
    bquant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """bf8 ABQuant CompV3 config (GemmConfigABQuantPrefill<bf8_t>, TransposeC=False).

    warp_tile_k = get_k_warp_tile<bf8_t, 16, IsFlatMM=false>(): 128 gfx950, 32 gfx942.
    See :func:`default_fp8_compv3_config`.
    """
    return ABQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16,
        warp_tile_k=_warp_tile_k_for(gfx_arch, "bf8"),
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=quant_group_k,
        bquant_group_m=1, bquant_group_n=bquant_group_n, bquant_group_k=quant_group_k,
        preshuffle_b=False, preshuffle_aq=False, preshuffle_bq=False,
        transpose_c=False,
        gfx_arch=gfx_arch,
    )


def _warp_tile_k_for(gfx_arch: str, variant_key: str, is_flatmm: bool = False) -> int:
    """Arch-derived WarpTileK for one grouped-ABQuant default config.

    Thin spelling of :func:`codegen_common.quant_warp_tile_k`; mirrors
    get_k_warp_tile<PrecType, M_Warp_Tile=16, IsFlatMM>.  ``is_flatmm`` is true
    only for the preshuffle-B prefill pipeline (Old-TE's
    GemmConfigPreshuffleB_ABQuant_Prefill), false for compv3 and eightwaves.

      is_flatmm=False -> gfx950 128, gfx942 32
      is_flatmm=True  -> gfx950 128, gfx942 64
    """
    return quant_warp_tile_k(
        gfx_arch,
        is_8bit_float=variant_is_8bit_float(variant_key),
        is_flat_mm=is_flatmm,
    )


def default_fp8_eightwaves_config(
    quant_group_k: int = 128,
    bquant_group_n: int = 128,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """fp8 ABQuant gfx950 EightWaves config (GemmConfigEightWaves<fp8_t>, TransposeC=True).

    Tile: 192x256x128, warp 4x2x1 — 8-wave configuration for MI350X.
    warp_tile_k mirrors get_k_warp_tile<fp8_t, 16, IsFlatMM=false>:
      gfx950 (CK_GFX950_SUPPORT): 128
      gfx942: 32
    The value must match what the C++ kernel header computes at compile time.
    """
    return ABQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="eightwaves",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=192, tile_n=256, tile_k=128,
        warp_m=4, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch, "fp8"),
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=quant_group_k,
        bquant_group_m=1, bquant_group_n=bquant_group_n, bquant_group_k=quant_group_k,
        preshuffle_b=False, preshuffle_aq=False, preshuffle_bq=False,
        transpose_c=True,
        k_block_per_cu=1,
        gfx_arch=gfx_arch,
    )


def default_bf8_eightwaves_config(
    quant_group_k: int = 128,
    bquant_group_n: int = 128,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """bf8 ABQuant gfx950 EightWaves config (GemmConfigEightWaves<bf8_t>, TransposeC=True).

    warp_tile_k mirrors get_k_warp_tile<bf8_t, 16, IsFlatMM=false>:
      gfx950: 128, gfx942: 32. See default_fp8_eightwaves_config for rationale.
    """
    return ABQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="eightwaves",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=192, tile_n=256, tile_k=128,
        warp_m=4, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch, "bf8"),
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=quant_group_k,
        bquant_group_m=1, bquant_group_n=bquant_group_n, bquant_group_k=quant_group_k,
        preshuffle_b=False, preshuffle_aq=False, preshuffle_bq=False,
        transpose_c=True,
        k_block_per_cu=1,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleb_config(
    quant_group_k: int = 128,
    bquant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """fp8 ABQuant PreshuffleB config (GemmConfigPreshuffleB_ABQuant_Prefill<fp8_t>).

    Tile: 128x128x128, warp 2x2x1. DoubleSmemBuffer=True.
    warp_tile_k mirrors get_k_warp_tile<fp8_t, 16, IsFlatMM=true>:
      gfx950 (CK_GFX950_SUPPORT): 128
      gfx942: 64
    The value must match what the C++ kernel header computes at compile time.
    """
    return ABQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="preshuffleb",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch, "fp8", is_flatmm=True),
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=quant_group_k,
        bquant_group_m=1, bquant_group_n=bquant_group_n, bquant_group_k=quant_group_k,
        preshuffle_b=True, preshuffle_aq=False, preshuffle_bq=False,
        transpose_c=True,
        double_smem_buffer=True,
        k_block_per_cu=2,
        gfx_arch=gfx_arch,
    )


def default_bf8_preshuffleb_config(
    quant_group_k: int = 128,
    bquant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """bf8 ABQuant PreshuffleB config (GemmConfigPreshuffleB_ABQuant_Prefill<bf8_t>).

    warp_tile_k mirrors get_k_warp_tile<bf8_t, 16, IsFlatMM=true>:
      gfx950: 128, gfx942: 64. See default_fp8_preshuffleb_config for rationale.
    """
    return ABQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="preshuffleb",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for(gfx_arch, "bf8", is_flatmm=True),
        aquant_group_m=1, aquant_group_n=1, aquant_group_k=quant_group_k,
        bquant_group_m=1, bquant_group_n=bquant_group_n, bquant_group_k=quant_group_k,
        preshuffle_b=True, preshuffle_aq=False, preshuffle_bq=False,
        transpose_c=True,
        double_smem_buffer=True,
        k_block_per_cu=2,
        gfx_arch=gfx_arch,
    )
