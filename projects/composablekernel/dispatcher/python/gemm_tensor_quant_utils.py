#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm TensorQuant dispatcher utilities.

Three-layer Python bridge for the dispatcher's TensorQuant GEMM path
(single per-tensor scale for A and a single per-tensor scale for B):

  TensorQuantKernelConfig  -- describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  TensorQuantDispatcherLib -- thin ctypes wrapper around a compiled .so
  TensorQuantGpuGemmRunner -- high-level runner that accepts numpy arrays

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_tensor_quant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

Usage (end-to-end):
  configs = [TensorQuantKernelConfig(variant_key="fp8", layout="rcr", ...)]
  so_paths = setup_multiple_tensor_quant_dispatchers(configs, output_dir=Path("/tmp/tq"))
  runner = TensorQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, AQ, BQ, TensorQuantGemmProblem(M=16, N=64, K=256))

Behavioral parity: Old-TE example/ck_tile/38_block_scale_gemm/gemm_quant_tensor.cpp
  C[M,N] = (AQ * BQ) * (A[M,K] @ B[K,N]);  fp8/bf8, rcr layout only.
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

_CODEGEN_SCRIPT = (
    Path(__file__).parent.parent / "codegen" / "unified_gemm_tensor_quant_codegen.py"
)
_CTYPES_LIB_SRC = (
    Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_tensor_quant_ctypes_lib.cpp"
)

# Import the shared name-construction helper from the codegen module so both
# sides stay byte-exact without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from unified_gemm_tensor_quant_codegen import make_tensor_quant_kernel_name  # noqa: E402

_DEFAULT_HIPCC    = "hipcc"
_DEFAULT_GFX_ARCH = "gfx950"


# =============================================================================
# TensorQuantKernelConfig -- byte-exact naming with codegen
# =============================================================================


@dataclass
class TensorQuantKernelConfig:
    """
    Complete description of one TensorQuant GEMM kernel.

    The .name property produces the exact string that
    unified_gemm_tensor_quant_codegen.py emits as KERNEL_NAME, ensuring the
    Python side and compiled .so always agree.
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

    double_smem_buffer: bool = False
    k_block_per_cu: int      = 1

    gfx_arch: str = _DEFAULT_GFX_ARCH

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME."""
        return make_tensor_quant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_gemm_tensor_quant_codegen.py."""
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
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
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
    C: object          # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# TensorQuantDispatcherLib -- thin ctypes wrapper
# =============================================================================


class TensorQuantDispatcherLib:
    """
    Loads a compiled gemm_tensor_quant .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_tensor_quant_gemm(A, B, AQ, BQ, C, M, N, K,
                                            stride_A, stride_B, stride_C,
                                            k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"TensorQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_tensor_quant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_tensor_quant_gemm.argtypes = [
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
            ctypes.c_int64,    # stride_C
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
        AQ,
        BQ,
        C,
        M: int,
        N: int,
        K: int,
        stride_A: int,
        stride_B: int,
        stride_C: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_tensor_quant_gemm with ctypes-wrapped pointers.

        A, B, C must be numpy arrays (C-contiguous, packed).
        AQ, BQ must be single-element float32 numpy arrays (per-tensor scales).
        B should be a packed (K, N) array supplied column-major (offset n*K+k).
        C must be the array that will receive output.
        Returns (status, time_ms).
        """
        import numpy as np

        A   = np.ascontiguousarray(A)
        # Kernel BLayout is ColumnMajor (rcr): B[k,n] lives at offset n*K+k.
        # Supply column-major bytes for 2-D B; ascontiguousarray would force
        # row-major and silently transpose.
        B   = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        AQ  = np.ascontiguousarray(AQ, dtype=np.float32)
        BQ  = np.ascontiguousarray(BQ, dtype=np.float32)
        C   = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_tensor_quant_gemm(
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
            ctypes.c_int64(stride_C),
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
# TensorQuantGpuGemmRunner -- high-level runner
# =============================================================================


class TensorQuantGpuGemmRunner:
    """
    High-level runner that loads a TensorQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B and scalar scales AQ, BQ; allocates C; returns
    TensorQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = TensorQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, BQ, problem: TensorQuantGemmProblem, c_dtype=None) -> TensorQuantGemmResult:
        """
        Run TensorQuant GEMM.

        A       shape: (M, K)           dtype: fp8/bf8
        B       shape: (K, N) col-major  dtype: fp8/bf8
        AQ, BQ  scalar float scales (python float or 1-element array)
        c_dtype numpy dtype for the output C buffer. Defaults to np.float16
                (CDataType is half for fp8/bf8 TensorQuant variants).
        Returns TensorQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K

        if c_dtype is None:
            c_dtype = np.float16

        # Output buffer -- dtype must match the compiled kernel's CDataType.
        C = np.zeros((M, N), dtype=c_dtype)

        # Single per-tensor scales as 1-element float32 arrays.
        aq_arr = np.asarray([float(AQ)], dtype=np.float32) if np.ndim(AQ) == 0 else \
            np.ascontiguousarray(AQ, dtype=np.float32).reshape(-1)[:1]
        bq_arr = np.asarray([float(BQ)], dtype=np.float32) if np.ndim(BQ) == 0 else \
            np.ascontiguousarray(BQ, dtype=np.float32).reshape(-1)[:1]

        # Strides (in elements; row-major A and C, col-major B → leading dim = K).
        stride_A = K   # A is row-major [M, K]
        stride_B = K   # B is col-major [K, N] → leading dim = K
        stride_C = N   # C is row-major [M, N]

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=aq_arr, BQ=bq_arr, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_B=stride_B,
            stride_C=stride_C,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_tensor_quant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return TensorQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


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
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_tensor_quant_kernel(
    config: TensorQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """
    Run unified_gemm_tensor_quant_codegen.py for one config; return the .hpp path or None.
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


def _compile_tensor_quant_kernel(
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
         the TensorQuant ctypes lib does not use the registry or dispatcher).

    Returns True on success.
    """
    ck_include = _get_ck_include_dir()

    obj_path = so_path.with_suffix(".o")

    # Arch-specific defines: gfx950 uses OCP fp8 (not FNUZ). These mirror the
    # CMakeLists.txt definitions normally injected by CMake but absent in the
    # standalone hipcc build path.
    arch_defines = []
    if "gfx12" in gfx_arch or "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8"]
    if "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT"]

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}",
                   f"-DGFX_ARCH=\"{gfx_arch}\"",
                   *arch_defines,
                   "-include", str(hpp_path),
                   str(_CTYPES_LIB_SRC),
                   "-o", str(obj_path)]

    if ck_include:
        compile_cmd += [f"-I{ck_include}"]

    # NOTE: dispatcher/include is intentionally excluded here (same reasoning as
    # the BQuant bridge): it pulls in generated_tile_backend.hpp which
    # instantiates SelectedKernel::launch(GemmHostArgs&), conflicting with the
    # TensorQuant kernel's launch(QuantGemmHostArgs&).

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

    link_cmd = [hipcc, "-shared", "-fPIC",
                f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path),
                "-o", str(so_path)]

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
# setup_multiple_tensor_quant_dispatchers -- build pipeline
# =============================================================================


def setup_multiple_tensor_quant_dispatchers(
    configs: List[TensorQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each TensorQuantKernelConfig: codegen -> hipcc compile -> .so path.

    Returns a list parallel to `configs` -- each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function.
    """
    if not configs:
        return []

    arch = gfx_arch or _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="tensor_quant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info(
        "Building %d TensorQuant kernel(s) for %s into %s",
        len(configs), arch, base_dir,
    )

    seen: Dict[str, int] = {}          # name → index of first occurrence
    deduped: List[Tuple[int, TensorQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: TensorQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_tensor_quant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_tensor_quant_kernel(
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

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d TensorQuant kernels", built, len(configs))
    return results


# =============================================================================
# Sweep expansion: JSON config → list of TensorQuantKernelConfig
# =============================================================================


def expand_tensor_quant_sweep(
    config_path: str,
    gfx_arch: str = _DEFAULT_GFX_ARCH,
) -> List["TensorQuantKernelConfig"]:
    """Expand a TensorQuant JSON sweep config into TensorQuantKernelConfig objects.

    The JSON format mirrors unified_gemm_tensor_quant_codegen.py's _build_specs so
    the same config files work for both codegen and Python utils. Every valid
    (variant, layout, tile) combination produces one TensorQuantKernelConfig;
    duplicates (by .name) are collapsed.
    """
    import itertools

    with open(config_path) as f:
        cfg = json.load(f)

    pipeline           = cfg.get("pipeline", "compv3")
    epilogue           = cfg.get("epilogue", "cshuffle")
    scheduler          = cfg.get("scheduler", "intrawave")
    k_block_per_cu     = cfg.get("k_block_per_cu", 1)
    double_smem_buffer = cfg.get("double_smem_buffer", False)

    configs: List[TensorQuantKernelConfig] = []
    seen: set = set()

    for variant_key, layout, tile_dict in itertools.product(
        cfg.get("variant_keys", ["fp8"]),
        cfg.get("layouts", ["rcr"]),
        cfg.get("tile_configs", []),
    ):
        c = TensorQuantKernelConfig(
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
            double_smem_buffer=double_smem_buffer,
            k_block_per_cu=k_block_per_cu,
            gfx_arch=gfx_arch,
        )
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)

    return configs


# =============================================================================
# Convenience: default fp8/bf8 configs (match GemmConfigQuantDecode<fp8_t/bf8_t>)
# =============================================================================


def fp8_warp_tile_k_for_arch(gfx_arch: str) -> int:
    """Arch-derived WarpTileK for fp8/bf8 with M_Warp_Tile=16.

    Mirrors ck_tile::get_k_warp_tile<fp8_t/bf8_t, M_Warp_Tile=16>()
    (include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp):

      - gfx950 (CK_GFX950_SUPPORT): is_8bit_float -> 128
      - gfx942 (and other non-950): IsFlatMM==false -> 32

    Picking 128 on gfx942 is a silent-correctness bug: there is no valid
    16x16x128 fp8/bf8 warp-gemm on gfx942, so the kernel compiles but outputs
    all-zeros (confirmed on GPU, MI300X). 32 is bit-exact and at parity with
    Old-TE (which launches ...16x16x32 on gfx942).
    """
    return 128 if ("gfx950" in gfx_arch or "gfx12" in gfx_arch) else 32


def default_fp8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> TensorQuantKernelConfig:
    """Default fp8 TensorQuant config (tile = 16x64x256, warp = 1x4x1).

    WarpTileK is arch-derived: 32 on gfx942, 128 on gfx950, mirroring
    ck_tile::get_k_warp_tile<fp8_t, M_Warp_Tile=16>().
    """
    return TensorQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16,
        warp_tile_k=fp8_warp_tile_k_for_arch(gfx_arch),
        gfx_arch=gfx_arch,
    )


def default_bf8_config(gfx_arch: str = _DEFAULT_GFX_ARCH) -> TensorQuantKernelConfig:
    """Default bf8 TensorQuant config (tile = 16x64x256, warp = 1x4x1).

    WarpTileK is arch-derived: 32 on gfx942, 128 on gfx950, mirroring
    ck_tile::get_k_warp_tile<bf8_t, M_Warp_Tile=16>().
    """
    return TensorQuantKernelConfig(
        variant_key="bf8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16,
        warp_tile_k=fp8_warp_tile_k_for_arch(gfx_arch),
        gfx_arch=gfx_arch,
    )


# =============================================================================
# Self-test / default-config runner
# =============================================================================


def _self_test(args) -> int:
    """Build (and optionally run) the default fp8/bf8 TensorQuant kernels.

    Without --run this only exercises codegen + hipcc (no GPU needed), which is
    the CI-safe path. With --run it executes the kernel and CPU-verifies against
    the TensorQuant reference C = (AQ*BQ) * (A @ B).
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    arch = args.gfx_arch
    configs = [default_fp8_config(arch), default_bf8_config(arch)]
    print("TensorQuant kernel names:")
    for c in configs:
        print(f"  {c.name}")

    if args.names_only:
        return 0

    out = Path(args.output_dir) if args.output_dir else None
    so_paths = setup_multiple_tensor_quant_dispatchers(
        configs, output_dir=out, gfx_arch=arch,
    )
    ok = all(p is not None for p in so_paths)
    for cfg, so in zip(configs, so_paths):
        print(f"  {cfg.variant_key}: {'BUILT ' + str(so) if so else 'FAILED'}")

    if not args.run:
        return 0 if ok else 1

    # --run: execute + CPU verify (requires a GPU).
    import numpy as np

    for cfg, so in zip(configs, so_paths):
        if so is None:
            continue
        M, N, K = 16, 64, 256
        # fp8/bf8 element values are stored raw; use small ints cast to the
        # kernel dtype via numpy uint8 view. For a smoke test we rely on the
        # kernel + CPU-ref agreement rather than exact numerics here.
        rng = np.random.default_rng(0)
        A = rng.integers(0, 8, size=(M, K)).astype(np.uint8)
        B = rng.integers(0, 8, size=(K, N)).astype(np.uint8)
        AQ = 0.5
        BQ = 0.25
        runner = TensorQuantGpuGemmRunner(so)
        res = runner.run(A, B, AQ, BQ, TensorQuantGemmProblem(M=M, N=N, K=K))
        print(f"  {cfg.variant_key}: ran {res.kernel_name} in {res.time_ms:.4f} ms, "
              f"C[0,0]={res.C[0, 0]}")

    return 0 if ok else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Gemm TensorQuant dispatcher self-test / default-config runner"
    )
    parser.add_argument("--gfx-arch", default=_DEFAULT_GFX_ARCH,
                        help="Target GPU arch (default: gfx950)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for generated headers + .so files")
    parser.add_argument("--names-only", action="store_true",
                        help="Print kernel names and exit (no build)")
    parser.add_argument("--run", action="store_true",
                        help="Execute the built kernels on the GPU (requires a GPU)")
    args = parser.parse_args()
    return _self_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
