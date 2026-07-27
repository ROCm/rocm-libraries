#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
ABQuant GEMM dispatcher utilities.

Three-layer Python bridge for A+B quantized GEMM:
  ABQuantKernelConfig    — describes one kernel; .name is byte-exact with codegen
  ABQuantDispatcherLib   — thin ctypes wrapper
  ABQuantGpuGemmRunner   — high-level runner

HostArgs mapping:
  aq_ptr    = A scale [M, QK_A], float32, row-major; stride=QK_A
  bq_ptr    = B scale [QK_B, QN_B], float32, col-major; stride=QK_B
  QK_A      = ceil(K / quant_group_k)
  QK_B      = ceil(K / quant_group_k)   (same group_k as A)
  stride_AQ = QK_A
  stride_BQ = QK_B  (col-major BQ: leading dim = QK_B)
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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_gemm_abquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_abquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_abquant_kernel_name  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"


@dataclass
class ABQuantKernelConfig:
    variant_key: str       # "fp8", "bf8"
    layout: str            # "rcr", "rrr", "crr", "ccr"
    pipeline: str          # "compv3"
    epilogue: str          # "cshuffle"
    scheduler: str         # "intrawave" or "interwave"

    tile_m: int
    tile_n: int
    tile_k: int
    warp_m: int
    warp_n: int
    warp_k: int
    warp_tile_m: int
    warp_tile_n: int
    warp_tile_k: int

    quant_group_k: int = 128
    quant_group_n: int = 1    # B N-group size; always in name as gsn{N}
    preshuffle_a: bool = False
    preshuffle_b: bool = False
    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        return make_abquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            quant_group_k=self.quant_group_k,
            quant_group_n=self.quant_group_n,
            preshuffle_b=self.preshuffle_a,      # a_preshuffle slot
            preshuffle_quant=self.preshuffle_b,  # b_preshuffle slot
        )

    def to_codegen_config(self) -> dict:
        return {
            "variant_keys": [self.variant_key],
            "layouts": [self.layout],
            "pipeline": self.pipeline,
            "epilogue": self.epilogue,
            "scheduler": self.scheduler,
            "tile_configs": [{
                "tile_m": self.tile_m, "tile_n": self.tile_n, "tile_k": self.tile_k,
                "warp_m": self.warp_m, "warp_n": self.warp_n, "warp_k": self.warp_k,
                "warp_tile_m": self.warp_tile_m, "warp_tile_n": self.warp_tile_n,
                "warp_tile_k": self.warp_tile_k,
            }],
            "quant_group_k": self.quant_group_k,
            "quant_group_n": self.quant_group_n,
            "preshuffle_a": self.preshuffle_a,
            "preshuffle_b": self.preshuffle_b,
        }


@dataclass
class ABQuantGemmProblem:
    M: int
    N: int
    K: int
    quant_group_k: int = 128
    quant_group_n: int = 1
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        return (self.K + self.quant_group_k - 1) // self.quant_group_k

    @property
    def QK_B(self) -> int:
        return self.QK_A  # same group_k for B

    @property
    def QN_B(self) -> int:
        return (self.N + self.quant_group_n - 1) // self.quant_group_n


@dataclass
class ABQuantGemmResult:
    C: object
    time_ms: float
    kernel_name: str


class ABQuantDispatcherLib:
    """
    Loads a compiled abquant_gemm .so and wraps its C API.

    int dispatcher_run_abquant_gemm(A, B, AQ, BQ, C,
                                     M, N, K,
                                     stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                                     QK_A, QK_B, QN_B, k_batch, *time_ms)
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

        lib.dispatcher_run_abquant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_abquant_gemm.argtypes = [
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

    def run(self, A, B, AQ, BQ, C, M, N, K,
            stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
            QK_A, QK_B, QN_B, k_batch=1) -> Tuple[int, float]:
        import numpy as np

        A  = np.ascontiguousarray(A)
        AQ = np.ascontiguousarray(AQ)
        B  = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        # BQ is col-major [QK_B, QN_B] — ascontiguousarray is fine since it's already
        # stored as [QK_B, QN_B] C-contiguous and the kernel reads it with stride=QK_B
        BQ = np.ascontiguousarray(BQ)
        C  = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)
        rc = self._lib.dispatcher_run_abquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            AQ.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K),
            ctypes.c_int64(stride_A), ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_AQ), ctypes.c_int64(stride_BQ), ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_A), ctypes.c_int64(QK_B), ctypes.c_int64(QN_B),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    def get_kernel_name(self) -> str:
        raw = self._lib.dispatcher_get_kernel_name()
        return raw.decode("utf-8") if raw else ""

    def cleanup(self):
        self._lib.dispatcher_cleanup()

    def __del__(self):
        try:
            self._lib.dispatcher_cleanup()
        except Exception:
            pass


class ABQuantGpuGemmRunner:
    def __init__(self, so_path: Path):
        self._lib = ABQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, AQ, B, BQ, problem: ABQuantGemmProblem, c_dtype=None) -> ABQuantGemmResult:
        """
        Run ABQuant GEMM.

        A   shape (M, K), row-major, dtype: fp8/bf8
        B   shape (K, N), col-major, dtype: fp8/bf8
        AQ  shape (M, QK_A), float32, row-major — A scale
        BQ  shape (QK_B, QN_B), float32, C-contiguous — B scale (col-major in kernel)
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_A, QK_B, QN_B = problem.QK_A, problem.QK_B, problem.QN_B

        if c_dtype is None:
            c_dtype = np.float16

        C = np.zeros((M, N), dtype=c_dtype)

        # stride_A: row-major A -> K; stride_B col-major B -> K;
        # stride_AQ: row-major [M, QK_A] -> QK_A;
        # stride_BQ: col-major [QK_B, QN_B] -> QK_B (leading dim = rows);
        stride_A  = K
        stride_B  = K
        stride_AQ = QK_A
        stride_BQ = QK_B   # BQ col-major: stride = number of rows = QK_B
        stride_C  = N

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_AQ=stride_AQ, stride_BQ=stride_BQ, stride_C=stride_C,
            QK_A=QK_A, QK_B=QK_B, QN_B=QN_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_abquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return ABQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


def _detect_gpu_arch() -> str:
    try:
        result = subprocess.run(["rocm_agent_enumerator"], capture_output=True, text=True, timeout=10)
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


def _get_dispatcher_static_lib() -> Optional[Path]:
    dispatcher_root = _CTYPES_LIB_SRC.parent.parent.parent
    static_lib = dispatcher_root / "build" / "libck_tile_dispatcher.a"
    return static_lib if static_lib.exists() else None


def _generate_abquant_kernel(config: ABQuantKernelConfig, output_dir: Path) -> Optional[Path]:
    config_json = json.dumps(config.to_codegen_config())
    cmd = [sys.executable, str(_CODEGEN_SCRIPT),
           "--output-dir", str(output_dir), "--config-json", config_json]
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


def _compile_abquant_kernel(hpp_path, so_path, gfx_arch, hipcc=_DEFAULT_HIPCC,
                            extra_include_dirs=None) -> bool:
    ck_include = _get_ck_include_dir()
    static_lib = _get_dispatcher_static_lib()
    obj_path = so_path.with_suffix(".o")

    arch_defines = []
    if "gfx12" in gfx_arch or "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]
    if "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}", f"-DGFX_ARCH=\"{gfx_arch}\"",
                   *arch_defines, "-include", str(hpp_path), str(_CTYPES_LIB_SRC),
                   "-o", str(obj_path)]
    if ck_include:
        compile_cmd += [f"-I{ck_include}"]
    if extra_include_dirs:
        for d in extra_include_dirs:
            compile_cmd += [f"-I{d}"]

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False

    link_cmd = [hipcc, "-shared", "-fPIC", f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path)]
    if static_lib:
        link_cmd += [str(static_lib)]
    link_cmd += ["-o", str(so_path)]

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


def setup_multiple_abquant_dispatchers(
    configs: List[ABQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="abquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, ABQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx, cfg):
        hpp = _generate_abquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None
        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            return idx, so
        ok = _compile_abquant_kernel(hpp, so, arch, hipcc, extra_include_dirs)
        return idx, so if ok else None

    if parallel and len(deduped) > 1:
        workers = max_workers or min(len(deduped), os.cpu_count() or 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_build_one, idx, cfg): (idx, cfg) for idx, cfg in deduped}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    idx, so_path = fut.result()
                    results[idx] = so_path
                except Exception as e:
                    _, cfg = futures[fut]
                    log.error("EXCEPTION for %s: %s", cfg.name, e)
    else:
        for idx, cfg in deduped:
            _, so_path = _build_one(idx, cfg)
            results[idx] = so_path

    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    return results


def default_fp8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """Default fp8 ABQuant config (A=fp8, B=fp8).

    WarpTileK=128 for fp8 on gfx950.
    """
    return ABQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_k=quant_group_k,
        quant_group_n=quant_group_n,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> ABQuantKernelConfig:
    """Default bf8 ABQuant config."""
    return ABQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_k=quant_group_k,
        quant_group_n=quant_group_n,
        gfx_arch=gfx_arch,
    )
