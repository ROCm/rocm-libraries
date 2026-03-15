#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
FMHA Dispatcher Python Utilities

Provides Python wrappers for FMHA dispatcher kernels via ctypes.
Mirrors ctypes_utils.py (GEMM) and grouped_conv_utils.py (Conv).

Usage:
    from fmha_utils import FmhaDispatcherLib, FmhaRunner, FmhaProblem, cpu_attention_fwd

    runner = FmhaRunner.from_prebuilt()
    result = runner.run(Q, K, V, problem)
"""

import ctypes
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Utility helpers
# =============================================================================


try:
    from dispatcher_common import detect_gpu_arch, get_dispatcher_root
except ImportError:
    # Standalone usage without dispatcher_common on PYTHONPATH
    def get_dispatcher_root() -> Path:
        return Path(__file__).parent.parent

    def detect_gpu_arch(fallback: str = "gfx950") -> str:
        try:
            out = subprocess.check_output(
                ["rocminfo"], text=True, stderr=subprocess.DEVNULL
            )
            for line in out.splitlines():
                if "Name:" in line and "gfx" in line:
                    return line.split()[-1].strip()
        except Exception:
            pass
        return fallback


# =============================================================================
# Data types
# =============================================================================


@dataclass
class FmhaResult:
    success: bool
    output: Optional[np.ndarray] = None
    time_ms: float = 0.0
    tflops: float = 0.0
    error: str = ""


@dataclass
class FmhaProblem:
    batch: int = 2
    nhead_q: int = 8
    nhead_k: int = 8
    seqlen_q: int = 128
    seqlen_k: int = 128
    hdim_q: int = 128
    hdim_v: int = 128

    @property
    def scale(self) -> float:
        return 1.0 / (self.hdim_q**0.5)

    @property
    def num_ops(self) -> int:
        sq, sk = self.seqlen_q, self.seqlen_k
        return 2 * self.batch * self.nhead_q * sq * sk * (self.hdim_q + self.hdim_v)

    def q_shape(self):
        return (self.batch, self.nhead_q, self.seqlen_q, self.hdim_q)

    def k_shape(self):
        return (self.batch, self.nhead_k, self.seqlen_k, self.hdim_q)

    def v_shape(self):
        return (self.batch, self.nhead_k, self.seqlen_k, self.hdim_v)

    def o_shape(self):
        return (self.batch, self.nhead_q, self.seqlen_q, self.hdim_v)


@dataclass
class FmhaKernelConfig:
    """Complete kernel configuration for FMHA.

    All tile/wave/warp dimensions are explicitly named to match the
    GEMM pattern (tile_m, tile_n, tile_k) but extended for FMHA's
    two-stage computation (Q*K^T stage 0, Attn*V stage 1).
    """

    # -- Signature: what operation --
    family: str = "fwd"
    data_type: str = "fp16"
    mode: str = "batch"
    vlayout: str = "r"
    hdim_q: int = 128
    hdim_v: int = 128
    gfx_arch: str = "gfx950"

    # -- Algorithm: tile shape --
    # Stage 0 (Q * K^T): seqlen_q x seqlen_k x hdim_q
    tile_m0: int = 128  # seqlen_q tile
    tile_n0: int = 128  # seqlen_k tile
    tile_k0: int = 32  # hdim_q tile
    # Stage 1 (Attn * V): seqlen_q x hdim_v x seqlen_k
    tile_n1: int = 128  # hdim_v tile
    tile_k1: int = 32  # seqlen_k tile
    tile_k0max: int = 128  # max k0 (alignment)

    # -- Algorithm: wave config (warps per block) --
    wave_m0: int = 4
    wave_n0: int = 1
    wave_k0: int = 1
    wave_m1: int = 4
    wave_n1: int = 1
    wave_k1: int = 1
    wave_m2: int = 1
    wave_n2: int = 1
    wave_k2: int = 1

    # -- Algorithm: warp tile (elements per warp) --
    warp_m0: int = 32
    warp_n0: int = 32
    warp_k0: int = 16
    warp_m1: int = 32
    warp_n1: int = 32
    warp_k1: int = 16
    warp_m2: int = 16
    warp_n2: int = 16
    warp_k2: int = 16

    # -- Algorithm: padding --
    # Values: 0=no pad, 1=pad, 8=pad with 8-byte alignment (BWD-specific)
    pad_s: int = 1
    pad_sk: int = 1
    pad_d: int = 1
    pad_dv: int = 1

    # -- Algorithm: pipeline --
    pipeline: str = "qr_async"
    block_per_cu: int = 1
    num_wave_groups: int = 1

    # -- Signature: features --
    mask: str = "no"
    bias: str = "no"
    lse: bool = False
    dropout: bool = False
    qscale: str = "no"
    rope: str = "none"
    logits: bool = False
    paged_kv: bool = False
    sink: bool = False
    skip_min_seqlen_q: bool = False
    page_size: int = 1
    kv_memory_layout: str = "vectorized"
    kv_lookup_table: str = "sglang"
    deterministic: bool = False
    dbias: bool = False
    dropout_variant: str = ""  # BWD: "no"/"dropout_wg16"/"dropout_wg16_storerandval"
    tile_tag: str = ""  # extra tile variant discriminator (e.g. "trload", "small")

    @property
    def tile(self) -> Tuple[int, ...]:
        return (
            self.tile_m0,
            self.tile_n0,
            self.tile_k0,
            self.tile_n1,
            self.tile_k1,
            self.tile_k0max,
        )

    @property
    def wave(self) -> Tuple[int, ...]:
        return (
            self.wave_m0,
            self.wave_n0,
            self.wave_k0,
            self.wave_m1,
            self.wave_n1,
            self.wave_k1,
            self.wave_m2,
            self.wave_n2,
            self.wave_k2,
        )

    @property
    def warp(self) -> Tuple[int, ...]:
        return (
            self.warp_m0,
            self.warp_n0,
            self.warp_k0,
            self.warp_m1,
            self.warp_n1,
            self.warp_k1,
            self.warp_m2,
            self.warp_n2,
            self.warp_k2,
        )

    @property
    def padding(self) -> Tuple[bool, ...]:
        return (self.pad_s, self.pad_sk, self.pad_d, self.pad_dv)

    @property
    def name(self) -> str:
        s = self.pad_s
        k = self.pad_sk
        d = self.pad_d
        v = self.pad_dv
        parts = [
            f"fmha_{self.family}_{self.data_type}",
            self.mode,
            f"h{self.hdim_q}x{self.hdim_v}"
            if self.hdim_q != self.hdim_v
            else f"h{self.hdim_q}",
            self.pipeline,
            f"t{self.tile_m0}x{self.tile_n0}x{self.tile_k0}x{self.tile_n1}x{self.tile_k1}x{self.tile_k0max}"
            + (f".{self.tile_tag}" if self.tile_tag else ""),
            f"pad{s}{k}{d}{v}",
            f"mask={self.mask}",
            f"bias={self.bias}",
        ]
        if self.lse:
            parts.append("lse=1")
        if self.dropout:
            parts.append("drop=1")
        if self.logits:
            parts.append("logits=1")
        if self.sink:
            parts.append("sink=1")
        if self.skip_min_seqlen_q:
            parts.append("skip=1")
        if self.qscale != "no":
            parts.append(f"qs={self.qscale}")
        if self.paged_kv:
            parts.append("pkv=1")
        if self.rope != "none":
            parts.append(f"rope={self.rope}")
        if self.page_size != 1:
            parts.append(f"ps={self.page_size}")
        if self.kv_memory_layout != "vectorized":
            parts.append(f"kvl={self.kv_memory_layout}")
        if self.kv_lookup_table != "sglang":
            parts.append(f"kvt={self.kv_lookup_table}")
        if self.deterministic:
            parts.append("det=1")
        if self.dbias:
            parts.append("dbias=1")
        if self.dropout_variant and self.dropout_variant != "no":
            parts.append(f"drv={self.dropout_variant}")
        return "_".join(parts)

    def to_codegen_json(self) -> str:
        return json.dumps(
            {
                "arch": self.gfx_arch,
                "signature": {
                    "family": self.family,
                    "data_type": self.data_type,
                    "mode": self.mode,
                    "vlayout": self.vlayout,
                    "hdim_q": self.hdim_q,
                    "hdim_v": self.hdim_v,
                    "mask": self.mask,
                    "bias": self.bias,
                    "lse": self.lse,
                    "dropout": self.dropout,
                    "qscale": self.qscale,
                    "rope": self.rope,
                    "logits": self.logits,
                    "paged_kv": self.paged_kv,
                    "fp8_static_quant": False,
                    "skip_min_seqlen_q": self.skip_min_seqlen_q,
                    "sink": self.sink,
                    "dbias": False,
                    "store_randval": False,
                    "deterministic": False,
                    "kv_memory_layout": self.kv_memory_layout,
                    "kv_lookup_table": self.kv_lookup_table,
                    "page_size": self.page_size,
                },
                "algorithm": {
                    "pipeline": self.pipeline,
                    "tile": list(self.tile),
                    "wave": list(self.wave),
                    "warp": list(self.warp),
                    "padding": list(self.padding),
                    "block_per_cu": self.block_per_cu,
                    "num_wave_groups": self.num_wave_groups,
                    "max_splits_log2": 0,
                    "max_seq_len_q": 0,
                },
            }
        )


# =============================================================================
# CPU reference
# =============================================================================


def cpu_attention_fwd(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, scale: float
) -> np.ndarray:
    """CPU reference: scaled dot-product attention (supports GQA).

    Args:
        Q: [batch, nhead_q, seqlen_q, hdim_q]  float32
        K: [batch, nhead_k, seqlen_k, hdim_q]  float32
        V: [batch, nhead_k, seqlen_k, hdim_v]  float32

    Returns:
        O: [batch, nhead_q, seqlen_q, hdim_v]  float32
    """
    nhead_q = Q.shape[1]
    nhead_k = K.shape[1]
    if nhead_q != nhead_k:
        ratio = nhead_q // nhead_k
        K = np.repeat(K, ratio, axis=1)
        V = np.repeat(V, ratio, axis=1)
    S = np.matmul(Q, K.transpose(0, 1, 3, 2)) * scale
    S_max = S.max(axis=-1, keepdims=True)
    S_exp = np.exp(S - S_max)
    P = S_exp / S_exp.sum(axis=-1, keepdims=True)
    return np.matmul(P, V)


# =============================================================================
# Low-level ctypes wrapper
# =============================================================================


class FmhaDispatcherLib:
    """Wrapper for the FMHA dispatcher shared library (libdispatcher_fmha_lib.so)."""

    SEARCH_PATHS = [
        "build/examples/libdispatcher_fmha_lib.so",
        "build/libdispatcher_fmha_lib.so",
        "build/lib/libdispatcher_fmha_lib.so",
    ]

    def __init__(self, lib: ctypes.CDLL, path: Path):
        self._lib = lib
        self.path = path
        self._setup()

    def _setup(self):
        lib = self._lib
        lib.fmha_dispatcher_initialize.argtypes = [ctypes.c_char_p]
        lib.fmha_dispatcher_initialize.restype = ctypes.c_int
        lib.fmha_dispatcher_run_fwd.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_int,  # batch
            ctypes.c_int,  # nhead_q
            ctypes.c_int,  # nhead_k
            ctypes.c_int,  # seqlen_q
            ctypes.c_int,  # seqlen_k
            ctypes.c_int,  # hdim_q
            ctypes.c_int,  # hdim_v
            ctypes.c_float,  # scale
            ctypes.c_int,  # mask_type
            ctypes.c_int,  # bias_type
            ctypes.c_int,  # has_lse
            ctypes.c_int,  # has_dropout
            ctypes.c_int,  # traits_hdim_q (0=same as hdim_q)
            ctypes.c_int,  # traits_hdim_v (0=same as hdim_v)
            ctypes.c_int,  # is_v_rowmajor (1=row, 0=col)
            ctypes.c_int,  # perm (1=BHSD, 0=BSHD)
            ctypes.c_char_p,  # data_type ("fp16", "bf16")
            ctypes.c_int,  # is_group_mode
            ctypes.c_int,  # window_left (-1=no window)
            ctypes.c_int,  # window_right (-1=no window, 0=causal)
            ctypes.POINTER(ctypes.c_float),  # time_ms_out
        ]
        lib.fmha_dispatcher_run_fwd.restype = ctypes.c_int
        lib.fmha_dispatcher_run_bwd.argtypes = [
            ctypes.c_void_p,  # q
            ctypes.c_void_p,  # k
            ctypes.c_void_p,  # v
            ctypes.c_void_p,  # o
            ctypes.c_void_p,  # lse
            ctypes.c_void_p,  # do
            ctypes.c_void_p,  # dq
            ctypes.c_void_p,  # dk
            ctypes.c_void_p,  # dv
            ctypes.c_int,  # batch
            ctypes.c_int,  # nhead_q
            ctypes.c_int,  # nhead_k
            ctypes.c_int,  # seqlen_q
            ctypes.c_int,  # seqlen_k
            ctypes.c_int,  # hdim_q
            ctypes.c_int,  # hdim_v
            ctypes.c_float,  # scale
            ctypes.c_char_p,  # data_type_str
            ctypes.POINTER(ctypes.c_float),  # time_ms_out
        ]
        lib.fmha_dispatcher_run_bwd.restype = ctypes.c_int
        lib.fmha_dispatcher_kernel_count.argtypes = []
        lib.fmha_dispatcher_kernel_count.restype = ctypes.c_int
        lib.fmha_dispatcher_cleanup.argtypes = []
        lib.fmha_dispatcher_cleanup.restype = None

    @classmethod
    def find(cls) -> Optional["FmhaDispatcherLib"]:
        root = get_dispatcher_root()
        for rel in cls.SEARCH_PATHS:
            path = root / rel
            if path.exists():
                try:
                    lib = ctypes.CDLL(str(path))
                    return cls(lib, path)
                except OSError:
                    continue
        return None

    @classmethod
    def load(cls, path: str) -> "FmhaDispatcherLib":
        lib = ctypes.CDLL(path)
        return cls(lib, Path(path))

    def initialize(self, arch: str = "gfx950") -> bool:
        return self._lib.fmha_dispatcher_initialize(arch.encode()) == 0

    def run_bwd(
        self,
        q: ctypes.c_void_p,
        k: ctypes.c_void_p,
        v: ctypes.c_void_p,
        o: ctypes.c_void_p,
        lse: ctypes.c_void_p,
        do_grad: ctypes.c_void_p,
        dq: ctypes.c_void_p,
        dk: ctypes.c_void_p,
        dv: ctypes.c_void_p,
        prob: FmhaProblem,
        data_type: str = "fp16",
    ) -> Tuple[int, float]:
        time_ms = ctypes.c_float(0.0)
        rc = self._lib.fmha_dispatcher_run_bwd(
            q,
            k,
            v,
            o,
            lse,
            do_grad,
            dq,
            dk,
            dv,
            prob.batch,
            prob.nhead_q,
            prob.nhead_k,
            prob.seqlen_q,
            prob.seqlen_k,
            prob.hdim_q,
            prob.hdim_v,
            prob.scale,
            data_type.encode(),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def kernel_count(self) -> int:
        return self._lib.fmha_dispatcher_kernel_count()

    def cleanup(self):
        self._lib.fmha_dispatcher_cleanup()


# =============================================================================
# High-level GPU runner (mirrors GpuGroupedConvRunner)
# =============================================================================


class FmhaRunner:
    """High-level FMHA runner with NumPy interface and HIP memory management."""

    HIP_MEMCPY_H2D = 1
    HIP_MEMCPY_D2H = 2

    def __init__(self, dispatch_lib: FmhaDispatcherLib, arch: str = "gfx950"):
        self._lib = dispatch_lib
        self._arch = arch
        self._hip = None
        self._load_hip()
        if not dispatch_lib.initialize(arch):
            raise RuntimeError("Failed to initialize FMHA dispatcher")

    def _load_hip(self):
        for name in ["libamdhip64.so", "libamdhip64.so.6"]:
            try:
                self._hip = ctypes.CDLL(name)
                self._hip.hipMalloc.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_size_t,
                ]
                self._hip.hipMalloc.restype = ctypes.c_int
                self._hip.hipFree.argtypes = [ctypes.c_void_p]
                self._hip.hipFree.restype = ctypes.c_int
                self._hip.hipMemcpy.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_size_t,
                    ctypes.c_int,
                ]
                self._hip.hipMemcpy.restype = ctypes.c_int
                self._hip.hipMemset.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_size_t,
                ]
                self._hip.hipMemset.restype = ctypes.c_int
                return
            except OSError:
                continue
        raise RuntimeError("Could not load libamdhip64.so")

    @classmethod
    def from_prebuilt(cls, arch: Optional[str] = None) -> "FmhaRunner":
        arch = arch or detect_gpu_arch()
        lib = FmhaDispatcherLib.find()
        if lib is None:
            raise RuntimeError(
                "FMHA dispatcher library not found. Build with:\n"
                "  cd dispatcher/build && cmake .. -DBUILD_DISPATCHER_EXAMPLES=ON && make dispatcher_fmha_lib"
            )
        return cls(lib, arch)

    @classmethod
    def from_library(cls, path: str, arch: Optional[str] = None) -> "FmhaRunner":
        arch = arch or detect_gpu_arch()
        return cls(FmhaDispatcherLib.load(path), arch)

    def run(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        prob: FmhaProblem,
        mask_type: int = 0,
        bias_type: int = 0,
        has_lse: int = 0,
        has_dropout: int = 0,
    ) -> FmhaResult:
        """Run FMHA forward on GPU with automatic HIP memory management.

        Args:
            Q: [batch, nhead_q, seqlen_q, hdim_q]  float16
            K: [batch, nhead_k, seqlen_k, hdim_q]  float16
            V: [batch, nhead_k, seqlen_k, hdim_v]  float16

        Returns:
            FmhaResult with output array, timing, TFLOPS
        """
        Q_c = np.ascontiguousarray(Q.astype(np.float16))
        K_c = np.ascontiguousarray(K.astype(np.float16))
        V_c = np.ascontiguousarray(V.astype(np.float16))
        O_c = np.zeros(prob.o_shape(), dtype=np.float16)

        d_q, d_k, d_v, d_o = (ctypes.c_void_p() for _ in range(4))

        try:
            self._hip.hipMalloc(ctypes.byref(d_q), Q_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_k), K_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_v), V_c.nbytes)
            self._hip.hipMalloc(ctypes.byref(d_o), O_c.nbytes)

            self._hip.hipMemcpy(d_q, Q_c.ctypes.data, Q_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_k, K_c.ctypes.data, K_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemcpy(d_v, V_c.ctypes.data, V_c.nbytes, self.HIP_MEMCPY_H2D)
            self._hip.hipMemset(d_o, 0, O_c.nbytes)

            time_ms = ctypes.c_float(0.0)
            rc = self._lib._lib.fmha_dispatcher_run_fwd(
                d_q,
                d_k,
                d_v,
                d_o,
                prob.batch,
                prob.nhead_q,
                prob.nhead_k,
                prob.seqlen_q,
                prob.seqlen_k,
                prob.hdim_q,
                prob.hdim_v,
                prob.scale,
                mask_type,
                bias_type,
                has_lse,
                has_dropout,
                0,
                0,  # traits_hdim_q/v (0 = same as hdim)
                1,  # is_v_rowmajor
                1,  # perm (1=BHSD)
                b"fp16",
                0,  # is_group_mode
                -1,  # window_left (no window)
                -1,  # window_right (no window)
                ctypes.byref(time_ms),
            )

            if rc != 0:
                return FmhaResult(success=False, error=f"Kernel failed (rc={rc})")

            self._hip.hipMemcpy(O_c.ctypes.data, d_o, O_c.nbytes, self.HIP_MEMCPY_D2H)

            tflops = (
                prob.num_ops / (time_ms.value * 1e-3) / 1e12
                if time_ms.value > 0
                else 0.0
            )
            return FmhaResult(
                success=True, output=O_c, time_ms=time_ms.value, tflops=tflops
            )

        finally:
            for d in [d_q, d_k, d_v, d_o]:
                if d.value:
                    self._hip.hipFree(d)

    @property
    def kernel_count(self) -> int:
        return self._lib.kernel_count()

    @property
    def library_path(self) -> str:
        return str(self._lib.path)

    def cleanup(self):
        self._lib.cleanup()


# =============================================================================
# JIT Build Support (mirrors setup_multiple_gemm_dispatchers)
# =============================================================================


@dataclass
class FmhaSetupResult:
    success: bool
    config: Optional[FmhaKernelConfig] = None
    runner: Optional[FmhaRunner] = None
    library_path: str = ""
    error: str = ""
    build_time_s: float = 0.0


def _find_static_lib() -> Optional[Path]:
    root = get_dispatcher_root()
    for rel in ["build/libck_tile_dispatcher.a", "build/lib/libck_tile_dispatcher.a"]:
        p = root / rel
        if p.exists():
            return p
    return None


def _find_hipcc() -> str:
    for path in ["/opt/rocm/bin/hipcc", "/usr/bin/hipcc"]:
        if os.path.exists(path):
            return path
    return "hipcc"


def fmha_compile_flags(arch: str, hipcc: str = "") -> List[str]:
    """Base hipcc flags for compiling FMHA kernels. Shared by JIT and tile engine."""
    if not hipcc:
        hipcc = _find_hipcc()
    root = get_dispatcher_root()
    flags = [
        hipcc,
        "-c",
        "-fPIC",
        "-O3",
        f"--offload-arch={arch}",
        "-std=c++17",
        f"-I{root.parent / 'include'}",
        f"-I{root / 'include'}",
        f"-I{root.parent}",
        "-mllvm",
        "-enable-noalias-to-md-conversion=0",
        "-Wno-undefined-func-template",
        "-Wno-float-equal",
        "--offload-compress",
    ]
    if arch.startswith("gfx9"):
        flags.append("-DCK_TILE_FMHA_FWD_FAST_EXP2=1")
    return flags


def setup_fmha_dispatcher(
    config: FmhaKernelConfig,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> FmhaSetupResult:
    """JIT-compile a single FMHA kernel and return a runner.

    Cached: if the .so already exists, loads it directly (~1ms).
    Fresh build: codegen → parallel compile (kernel + ctypes) → link.
    """
    import time

    t0 = time.perf_counter()

    root = get_dispatcher_root()
    codegen_dir = root / "codegen"
    ctypes_src = root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = _find_static_lib()
    hipcc = _find_hipcc()

    if output_dir is None:
        output_dir = root / "build" / "examples" / f"fmha_jit_{config.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    lib_name = f"libdispatcher_fmha_{config.name}.so"
    lib_path = output_dir / lib_name

    # Cache hit: .so already exists, just load
    if lib_path.exists():
        try:
            runner = FmhaRunner.from_library(str(lib_path), config.gfx_arch)
            return FmhaSetupResult(
                success=True,
                config=config,
                runner=runner,
                library_path=str(lib_path),
                build_time_s=time.perf_counter() - t0,
            )
        except Exception:
            pass  # stale .so, rebuild

    if not static_lib:
        return FmhaSetupResult(
            success=False, config=config, error="libck_tile_dispatcher.a not found"
        )
    if not ctypes_src.exists():
        return FmhaSetupResult(
            success=False, config=config, error="fmha_ctypes_lib.cpp not found"
        )

    # Step 1: Codegen
    gen_cmd = [
        sys.executable,
        str(codegen_dir / "generate_fmha_fallback.py"),
        "--output-dir",
        str(output_dir),
        "--gpu-target",
        config.gfx_arch,
        "--config-json",
        config.to_codegen_json(),
    ]
    r = subprocess.run(gen_cmd, capture_output=True, text=True, cwd=str(codegen_dir))
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False, config=config, error=f"Codegen failed: {r.stderr[:500]}"
        )

    dispatch_header = output_dir / "fmha_python_dispatch.hpp"
    if not dispatch_header.exists():
        return FmhaSetupResult(
            success=False, config=config, error="Dispatch header not generated"
        )

    # Step 2: Compile kernel .cpp AND ctypes in parallel
    kernel_cpps = list(output_dir.glob("fmha_*.cpp"))
    base_flags = fmha_compile_flags(config.gfx_arch, hipcc)

    compile_jobs = []
    for cpp in kernel_cpps:
        obj = cpp.with_suffix(".o")
        compile_jobs.append((base_flags + [str(cpp), "-o", str(obj)], obj, "kernel"))

    ctypes_obj = output_dir / "fmha_ctypes_lib.o"
    ctypes_cmd = base_flags + [
        f"-I{output_dir}",
        f"-I{output_dir / 'dispatcher_wrappers'}",
        f"-include{dispatch_header}",
        f'-DGFX_ARCH="{config.gfx_arch}"',
        str(ctypes_src),
        "-o",
        str(ctypes_obj),
    ]
    compile_jobs.append((ctypes_cmd, ctypes_obj, "ctypes"))

    def _run_compile(job):
        cmd, obj, label = job
        if obj.exists():
            return (True, obj, label, "")
        r = subprocess.run(cmd, capture_output=True, text=True)
        return (r.returncode == 0, obj, label, r.stderr[:500])

    with ThreadPoolExecutor(max_workers=len(compile_jobs)) as pool:
        results = list(pool.map(_run_compile, compile_jobs))

    kernel_objs = []
    for ok, obj, label, err in results:
        if not ok:
            return FmhaSetupResult(
                success=False,
                config=config,
                error=f"{label} compile failed: {err}",
            )
        if label == "kernel":
            kernel_objs.append(str(obj))

    # Step 3: Link
    link_cmd = [
        hipcc,
        "-shared",
        "-fPIC",
        str(ctypes_obj),
        *kernel_objs,
        str(static_lib),
        "-o",
        str(lib_path),
    ]
    r = subprocess.run(link_cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return FmhaSetupResult(
            success=False, config=config, error=f"Link failed: {r.stderr[:500]}"
        )

    # Step 4: Load
    try:
        runner = FmhaRunner.from_library(str(lib_path), config.gfx_arch)
    except Exception as e:
        return FmhaSetupResult(success=False, config=config, error=f"Load failed: {e}")

    elapsed = time.perf_counter() - t0
    return FmhaSetupResult(
        success=True,
        config=config,
        runner=runner,
        library_path=str(lib_path),
        build_time_s=elapsed,
    )


def setup_multiple_fmha_dispatchers(
    configs: List[FmhaKernelConfig],
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    max_workers: Optional[int] = None,
) -> List[FmhaSetupResult]:
    """3-stage pipelined JIT: codegen(parallel) -> compile(parallel) -> link+load(parallel).

    Faster than calling setup_fmha_dispatcher() per-kernel because all hipcc
    compile jobs (kernel + ctypes from ALL kernels) share one thread pool.
    """
    if not configs:
        return []

    workers = max_workers or min(len(configs), os.cpu_count() or 4)
    root = get_dispatcher_root()
    codegen_dir = root / "codegen"
    ctypes_src = root / "bindings" / "ctypes" / "fmha_ctypes_lib.cpp"
    static_lib = _find_static_lib()
    hipcc = _find_hipcc()
    arch = configs[0].gfx_arch

    if output_dir is None:
        output_dir = root / "build" / "examples"

    results: dict[str, FmhaSetupResult] = {}

    # --- Stage 1: Parallel codegen ---
    def _codegen(cfg):
        out = output_dir / f"fmha_jit_{cfg.name}"
        lib_path = out / f"libdispatcher_fmha_{cfg.name}.so"
        if lib_path.exists():
            try:
                FmhaRunner.from_library(str(lib_path), arch)
                return (cfg.name, cfg, out, True)
            except Exception:
                pass
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            [
                sys.executable,
                str(codegen_dir / "generate_fmha_fallback.py"),
                "--output-dir",
                str(out),
                "--gpu-target",
                cfg.gfx_arch,
                "--config-json",
                cfg.to_codegen_json(),
            ],
            capture_output=True,
            text=True,
            cwd=str(codegen_dir),
        )
        ok = r.returncode == 0 and (out / "fmha_python_dispatch.hpp").exists()
        if not ok:
            results[cfg.name] = FmhaSetupResult(
                success=False, config=cfg, error=f"Codegen failed: {r.stderr[:200]}"
            )
        return (cfg.name, cfg, out, ok)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        codegen_results = list(pool.map(_codegen, configs))

    # --- Stage 2: Collect ALL compile jobs, run in one pool ---
    base_flags = fmha_compile_flags(arch, hipcc)
    compile_jobs = []  # (cmd, obj_path, kernel_name, label)

    config_dirs: dict[str, tuple[FmhaKernelConfig, Path]] = {}
    for name, cfg, out, ok in codegen_results:
        if not ok or name in results:
            continue
        config_dirs[name] = (cfg, out)
        for cpp in out.glob("fmha_*.cpp"):
            obj = cpp.with_suffix(".o")
            if not obj.exists():
                compile_jobs.append(
                    (base_flags + [str(cpp), "-o", str(obj)], obj, name, "kernel")
                )
        ctypes_obj = out / "fmha_ctypes_lib.o"
        if not ctypes_obj.exists():
            dispatch = out / "fmha_python_dispatch.hpp"
            compile_jobs.append(
                (
                    base_flags
                    + [
                        f"-I{out}",
                        f"-I{out / 'dispatcher_wrappers'}",
                        f"-include{dispatch}",
                        f'-DGFX_ARCH="{arch}"',
                        str(ctypes_src),
                        "-o",
                        str(ctypes_obj),
                    ],
                    ctypes_obj,
                    name,
                    "ctypes",
                )
            )

    failed_names: set = set()

    def _compile(job):
        cmd, obj, name, label = job
        if obj.exists():
            return (name, True, "")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            return (name, False, r.stderr[:200])
        return (name, True, "")

    if compile_jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, ok, err in pool.map(_compile, compile_jobs):
                if not ok:
                    failed_names.add(name)
                    if name not in results:
                        cfg, _ = config_dirs[name]
                        results[name] = FmhaSetupResult(
                            success=False, config=cfg, error=f"Compile: {err}"
                        )

    # --- Stage 3: Link + load ---
    def _link_load(item):
        name, (cfg, out) = item
        if name in failed_names or name in results:
            return
        objs = list(out.glob("*.o"))
        lib_path = out / f"libdispatcher_fmha_{name}.so"
        if not lib_path.exists():
            r = subprocess.run(
                [
                    hipcc,
                    "-shared",
                    "-fPIC",
                    *[str(o) for o in objs],
                    str(static_lib),
                    "-o",
                    str(lib_path),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                results[name] = FmhaSetupResult(
                    success=False, config=cfg, error=f"Link: {r.stderr[:200]}"
                )
                return
        try:
            runner = FmhaRunner.from_library(str(lib_path), arch)
            results[name] = FmhaSetupResult(
                success=True, config=cfg, runner=runner, library_path=str(lib_path)
            )
        except Exception as e:
            results[name] = FmhaSetupResult(
                success=False, config=cfg, error=f"Load: {e}"
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_link_load, config_dirs.items()))

    # Return in original order
    return [
        results.get(c.name, FmhaSetupResult(success=False, config=c, error="skipped"))
        for c in configs
    ]


# =============================================================================
# Registry (mirrors ctypes_utils.Registry)
# =============================================================================


class FmhaRegistry:
    """Kernel registry with parallel JIT build support."""

    def __init__(self, name: str = "fmha"):
        self._name = name
        self._kernels: List[FmhaKernelConfig] = []

    def register_kernel(self, config: FmhaKernelConfig):
        self._kernels.append(config)

    def __len__(self):
        return len(self._kernels)

    def build(
        self,
        verbose: bool = False,
        max_workers: Optional[int] = None,
    ) -> List[FmhaSetupResult]:
        return setup_multiple_fmha_dispatchers(
            self._kernels,
            verbose=verbose,
            max_workers=max_workers,
        )


# =============================================================================
# Validator (mirrors ctypes_utils.Validator)
# =============================================================================


class FmhaValidator:
    """Validates FMHA GPU output against a reference.

    Usage:
        validator = FmhaValidator(rtol=1e-2, atol=1e-2)
        ok, max_abs, max_rel = validator.check(gpu_output, cpu_reference)
    """

    def __init__(self, rtol: float = 1e-2, atol: float = 1e-2):
        self.rtol = rtol
        self.atol = atol

    def check(
        self, output: np.ndarray, reference: np.ndarray
    ) -> Tuple[bool, float, float]:
        """Check output against reference.

        Returns:
            (is_valid, max_abs_error, max_rel_error)
        """
        out_f32 = output.astype(np.float32)
        ref_f32 = reference.astype(np.float32)
        diff = np.abs(out_f32 - ref_f32)
        max_abs = float(diff.max())
        max_rel = float((diff / (np.abs(ref_f32) + 1e-6)).max())
        ok = bool(np.allclose(out_f32, ref_f32, atol=self.atol, rtol=self.rtol))
        return ok, max_abs, max_rel


# =============================================================================
# KernelSpec + spec_to_config (mirrors ctypes_utils.KernelSpec)
# =============================================================================


@dataclass
class FmhaKernelSpec:
    """High-level kernel specification for easy declaration.

    Mirrors GEMM's KernelSpec: specify name + key dimensions, get a
    full FmhaKernelConfig via spec_to_config().
    """

    name: str
    hdim: int = 128
    pipeline: str = "qr_async"
    # Stage 0 tile (Q*K^T)
    tile_m0: int = 128
    tile_n0: int = 128
    tile_k0: int = 32


def spec_to_config(
    spec: FmhaKernelSpec, dtype: str = "fp16", arch: str = "gfx950"
) -> FmhaKernelConfig:
    """Convert a high-level FmhaKernelSpec to a full FmhaKernelConfig."""
    hdim = spec.hdim

    # gfx12 (RDNA4) uses 16x16x16 WMMA tiles with wave32
    # gfx9 (CDNA) uses 32x32x16 warp tiles with wave64; gfx11 (RDNA3) uses wave32
    is_gfx12 = arch.startswith("gfx12")
    warp_m = 16 if is_gfx12 else 32
    warp_n = 16 if is_gfx12 else 32
    warp_k = 16

    config_kwargs = dict(
        data_type=dtype,
        hdim_q=hdim,
        hdim_v=hdim,
        pipeline=spec.pipeline,
        tile_m0=spec.tile_m0,
        tile_n0=spec.tile_n0,
        tile_k0=spec.tile_k0,
        tile_n1=hdim,
        tile_k1=spec.tile_k0,
        tile_k0max=hdim,
        gfx_arch=arch,
    )

    if is_gfx12:
        config_kwargs.update(
            warp_m0=warp_m, warp_n0=warp_n, warp_k0=warp_k,
            warp_m1=warp_m, warp_n1=warp_n, warp_k1=warp_k,
            warp_m2=warp_m, warp_n2=warp_n, warp_k2=warp_k,
        )

    return FmhaKernelConfig(**config_kwargs)


# =============================================================================
# Split-K heuristic (from fmhaarch.md Section 9.5)
# =============================================================================


def num_splits_heuristic_ck(
    batch: int,
    nheads: int,
    seqlen_q: int,
    tile_m0: int = 128,
    num_cus: int = 304,
    min_util_rate: float = 0.85,
) -> int:
    """Recommend num_splits for split-KV, matching CK's heuristic.

    Args:
        batch: batch size
        nheads: number of Q heads
        seqlen_q: query sequence length
        tile_m0: tile size in seqlen_q dimension
        num_cus: number of compute units on GPU (gfx950: 304)
        min_util_rate: minimum CU utilization threshold

    Returns:
        Recommended num_splits (1 means no split)
    """
    import math

    m_blocks = math.ceil(seqlen_q / tile_m0) if tile_m0 > 0 else 1
    batch_nheads_mblocks = batch * nheads * m_blocks

    if batch_nheads_mblocks >= num_cus * min_util_rate:
        return 1

    for splits in [2, 4, 8, 16, 32]:
        if batch_nheads_mblocks * splits >= num_cus * min_util_rate:
            return splits

    return 1
