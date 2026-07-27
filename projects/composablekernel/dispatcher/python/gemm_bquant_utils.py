#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
BQuant GEMM dispatcher utilities (block_scale_gemm operator, gemm_bquant_* naming).

Three-layer Python bridge for B-matrix quantized GEMM (block_scale_gemm naming convention):

  GemmBQuantKernelConfig   -- describes one kernel; .name is byte-exact with codegen
  GemmBQuantDispatcherLib  -- thin ctypes wrapper around a compiled .so
  GemmBQuantGpuGemmRunner  -- high-level runner accepting numpy arrays

Build helper:
  setup_multiple_gemm_bquant_dispatchers(configs, ...) : codegen -> hipcc -> .so paths

NOTE: This module (gemm_bquant_utils.py) handles kernels with "gemm_bquant_*" names
matching the block_scale_gemm tile engine operator. The existing grouped_gemm_bquant_utils.py
handles the older "grouped_gemm_bquant_*" named kernels. The C ABI function name
(dispatcher_run_bquant_gemm) is the same in both.

HostArgs mapping (QuantGemmHostArgs):
  aq_ptr    = nullptr (unused for BQuant-only)
  bq_ptr    = device B scale [QK_B, QN_B], float32, row-major
  QK_A      = 0
  QK_B      = ceil(K / quant_group_k)
  stride_AQ = 0
  stride_BQ = QN_B (row-major)
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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_gemm_bquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_bquant_ctypes_lib.cpp"

_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_gemm_bquant_kernel_name  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"


@dataclass
class GemmBQuantKernelConfig:
    """
    Complete description of one gemm_bquant (block_scale_gemm) kernel.

    The .name property produces the exact string that unified_gemm_bquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    variant_key: str       # "fp8", "bf8"
    layout: str            # "rcr", "rrr", "crr", "ccr"
    pipeline: str          # "compv3"
    epilogue: str          # "default" or "cshuffle"
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

    quant_group_k: int = 128
    preshuffle_bquant: bool = False
    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        return make_gemm_bquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n,
            warp_tile_k=self.warp_tile_k,
            preshuffle_bquant=self.preshuffle_bquant,
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
            "preshuffle_bquant": self.preshuffle_bquant,
        }


@dataclass
class GemmBQuantGemmProblem:
    M: int
    N: int
    K: int
    quant_group_k: int = 128
    quant_group_n: int = 1
    k_batch: int = 1

    @property
    def QK_B(self) -> int:
        return (self.K + self.quant_group_k - 1) // self.quant_group_k

    @property
    def QN_B(self) -> int:
        return (self.N + self.quant_group_n - 1) // self.quant_group_n


@dataclass
class GemmBQuantGemmResult:
    C: object
    time_ms: float
    kernel_name: str


class GemmBQuantDispatcherLib:
    """
    Loads a compiled gemm_bquant .so and wraps its C API.

    int dispatcher_run_bquant_gemm(A, B, BQ, C,
                                    M, N, K, stride_A, stride_B, stride_BQ, stride_C,
                                    QK_B, QN_B, k_batch, *time_ms)
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"GemmBQuant .so not found: {self.so_path}")
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

    def run(self, A, B, BQ, C, M, N, K,
            stride_A, stride_B, stride_BQ, stride_C,
            QK_B, QN_B, k_batch=1) -> Tuple[int, float]:
        import numpy as np

        A  = np.ascontiguousarray(A)
        B  = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        BQ = np.ascontiguousarray(BQ)
        C  = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)
        rc = self._lib.dispatcher_run_bquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            BQ.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M), ctypes.c_int64(N), ctypes.c_int64(K),
            ctypes.c_int64(stride_A), ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_BQ), ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_B), ctypes.c_int64(QN_B),
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


class GemmBQuantGpuGemmRunner:
    """High-level runner for gemm_bquant (block_scale_gemm) kernels."""

    def __init__(self, so_path: Path):
        self._lib = GemmBQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, BQ, problem: GemmBQuantGemmProblem, c_dtype=None) -> GemmBQuantGemmResult:
        """
        Run BQuant GEMM.

        A   shape (M, K), row-major, dtype: fp8/bf8
        B   shape (K, N), col-major, dtype: fp8/bf8
        BQ  shape (QK_B, QN_B), float32, row-major — B scale tensor
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_B, QN_B = problem.QK_B, problem.QN_B

        if c_dtype is None:
            c_dtype = np.float16

        C = np.zeros((M, N), dtype=c_dtype)

        stride_A  = K
        stride_B  = K
        stride_BQ = QN_B
        stride_C  = N

        rc, time_ms = self._lib.run(
            A=A, B=B, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A, stride_B=stride_B,
            stride_BQ=stride_BQ, stride_C=stride_C,
            QK_B=QK_B, QN_B=QN_B,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_bquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return GemmBQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


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


def _generate_gemm_bquant_kernel(config: GemmBQuantKernelConfig, output_dir: Path) -> Optional[Path]:
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


def _compile_gemm_bquant_kernel(hpp_path, so_path, gfx_arch, hipcc=_DEFAULT_HIPCC,
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
            log.error("Compile failed:\n%s", result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        return False

    link_cmd = [hipcc, "-shared", "-fPIC", f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path)]
    if static_lib:
        link_cmd += [str(static_lib)]
    link_cmd += ["-o", str(so_path)]

    try:
        result = subprocess.run(link_cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            log.error("Link failed:\n%s", result.stderr[-2000:])
            obj_path.unlink(missing_ok=True)
            return False
    except subprocess.TimeoutExpired:
        obj_path.unlink(missing_ok=True)
        return False

    obj_path.unlink(missing_ok=True)
    return True


def setup_multiple_gemm_bquant_dispatchers(
    configs: List[GemmBQuantKernelConfig],
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
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="gemm_bquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, GemmBQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx, cfg):
        hpp = _generate_gemm_bquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None
        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            return idx, so
        ok = _compile_gemm_bquant_kernel(hpp, so, arch, hipcc, extra_include_dirs)
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
) -> GemmBQuantKernelConfig:
    """Default fp8 gemm_bquant config (block_scale_gemm, default epilogue).

    WarpTileK=128: get_k_warp_tile<fp8_t, 16>() = 128 on gfx950.
    Epilogue="default": block_scale_gemm uses DefaultGemm2DEpilogue by default.
    """
    return GemmBQuantKernelConfig(
        variant_key="fp8", layout="rcr",
        pipeline="compv3", epilogue="default", scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )


def default_bf8_config(
    quant_group_k: int = 128,
    quant_group_n: int = 1,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> GemmBQuantKernelConfig:
    """Default bf8 gemm_bquant config."""
    return GemmBQuantKernelConfig(
        variant_key="bf8", layout="rcr",
        pipeline="compv3", epilogue="default", scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        quant_group_k=quant_group_k,
        gfx_arch=gfx_arch,
    )
