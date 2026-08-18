#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
AQuant (A-only quantized) GEMM dispatcher utilities.

Three-layer Python bridge for the dispatcher's AQuantGrouped GEMM path:

  AQuantKernelConfig  -- describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  AQuantDispatcherLib -- thin ctypes wrapper around a compiled .so
  AQuantGpuGemmRunner -- high-level runner that accepts numpy arrays

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_aquant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

Usage (end-to-end):
  configs = [AQuantKernelConfig(variant_key="fp8", layout="rcr", ...)]
  so_paths = setup_multiple_aquant_dispatchers(configs, output_dir=Path("/tmp/aq"))
  runner = AQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, AQ, B, AQuantGemmProblem(M=16, N=64, K=256))
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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_gemm_aquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_aquant_ctypes_lib.cpp"

# Import the shared name-construction helper from codegen_common so both sides
# stay byte-exact without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_gemm_aquant_kernel_name  # noqa: E402

# NEVER default to gfx942 -- arch must be detected or explicitly supplied.
_DEFAULT_HIPCC = "hipcc"
_SUPPORTED_ARCHS = ("gfx90a", "gfx942", "gfx950", "gfx1250")

# Layout tags AQuant supports, and whether preshufflequant is allowed for each.
# Derived from run_gemm_example_prec_type in run_gemm_quant_example.inc:
#   rcr, rrr, crr : both decode and preshufflequant
#   ccr           : decode only (rejected for APreshuffleQuant)
_LAYOUTS_DECODE = ("rcr", "rrr", "crr", "ccr")
_LAYOUTS_PRESHUFFLEQUANT = ("rcr", "rrr", "crr")

# AQ (A-scale) tensor layout per 3-char layout tag, mirroring AQUANT_AQ_LAYOUT in
# unified_gemm_aquant_codegen.py.  The scale tensor is column-major only for ccr
# (A=C B=C), row-major for rcr/rrr/crr.  A column-major AQ of shape [M, QK_A] has
# leading dimension M (not QK_A) -- see Old-TE get_default_stride in
# run_gemm_quant_example.inc (~line 528).
_LAYOUTS_AQ_COLMAJOR = frozenset({"ccr"})


def _aq_stride(layout: str, M: int, QK_A: int) -> int:
    """Leading dimension of the AQ scale tensor for a given layout tag.

    Column-major AQ (ccr) -> M; row-major AQ (rcr/rrr/crr) -> QK_A.  Consistent with
    the ColumnMajor/RowMajor AQLayout the codegen emits and the exp_stride_AQ the
    ctypes lib validates.
    """
    return M if layout in _LAYOUTS_AQ_COLMAJOR else QK_A

# fp8/bf8 A/B/Q dtype meta for the four variants (A is the quantized operand).
_VARIANT_META: Dict[str, Dict[str, str]] = {
    "fp8": {"a": "fp8", "b": "fp8", "q": "float"},
    "bf8": {"a": "bf8", "b": "bf8", "q": "float"},
    "fp8i4": {"a": "pk_int4", "b": "fp8", "q": "fp8"},
    "bf8i4": {"a": "pk_int4", "b": "bf8", "q": "bf8"},
}


def _validate_arch(arch: str) -> str:
    """Return arch if supported; raise on unknown arch (never silently default)."""
    if not any(arch.startswith(a) for a in _SUPPORTED_ARCHS):
        raise ValueError(
            f"Unsupported GPU architecture '{arch}' for AQuant "
            f"(supported: {', '.join(_SUPPORTED_ARCHS)})"
        )
    return arch


# =============================================================================
# AQuantKernelConfig -- byte-exact naming with codegen
# =============================================================================


@dataclass
class AQuantKernelConfig:
    """
    Complete description of one AQuantGrouped GEMM kernel.

    The .name property produces the exact string that unified_gemm_aquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    variant_key: str       # "fp8", "bf8", "fp8i4", "bf8i4"
    layout: str            # "rcr", "rrr", "crr", "ccr"
    scheduler: str         # "interwave" (decode) or "intrawave" (preshufflequant)

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

    preshuffle_aquant: bool  = False
    double_smem_buffer: bool = False
    k_block_per_cu: int      = 1

    # No default arch: caller must pass a valid one (or use _detect_gpu_arch()).
    gfx_arch: str = "gfx950"

    @property
    def pipeline_key(self) -> str:
        """Pipeline map key echoed in the kernel name: preshufflequant -> compv3, else mem."""
        return "compv3" if self.preshuffle_aquant else "mem"

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME (delegates to make_gemm_aquant_kernel_name)."""
        return make_gemm_aquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline_key,
            epilogue="cshuffle",
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
            quant_group_m=self.quant_group_m,
            quant_group_n=self.quant_group_n,
            quant_group_k=self.quant_group_k,
            preshuffle_aquant=self.preshuffle_aquant,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_gemm_aquant_codegen.py."""
        return {
            "variant_keys": [self.variant_key],
            "layouts": [self.layout],
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
            "preshuffle_aquant": self.preshuffle_aquant,
            "double_smem_buffer": self.double_smem_buffer,
            "k_block_per_cu": self.k_block_per_cu,
        }


# =============================================================================
# AQuantGemmProblem
# =============================================================================


@dataclass
class AQuantGemmProblem:
    M: int
    N: int
    K: int
    quant_group_m: int = 1
    quant_group_n: int = 1
    quant_group_k: int = 128
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        """Number of K-groups: ceil(K / quant_group_k)."""
        return (self.K + self.quant_group_k - 1) // self.quant_group_k


# =============================================================================
# AQuantGemmResult
# =============================================================================


@dataclass
class AQuantGemmResult:
    C: object          # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# AQuantDispatcherLib -- thin ctypes wrapper
# =============================================================================


class AQuantDispatcherLib:
    """
    Loads a compiled aquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_aquant_gemm(A, AQ, B, C, M, N, K,
                                       stride_A, stride_AQ, stride_B, stride_C,
                                       QK_A, k_batch, *time_ms)
      char* dispatcher_get_kernel_name()
      int   dispatcher_get_kernel_count()
      void  dispatcher_cleanup()
    """

    def __init__(self, so_path: Path):
        self.so_path = Path(so_path)
        if not self.so_path.exists():
            raise FileNotFoundError(f"AQuant .so not found: {self.so_path}")
        self._lib = ctypes.CDLL(str(self.so_path))
        self._setup()
        rc = self._lib.dispatcher_initialize()
        if rc != 0:
            raise RuntimeError(f"dispatcher_initialize() returned {rc}")

    def _setup(self):
        lib = self._lib

        lib.dispatcher_initialize.restype  = ctypes.c_int
        lib.dispatcher_initialize.argtypes = []

        lib.dispatcher_run_aquant_gemm.restype  = ctypes.c_int
        lib.dispatcher_run_aquant_gemm.argtypes = [
            ctypes.c_void_p,   # A
            ctypes.c_void_p,   # AQ
            ctypes.c_void_p,   # B
            ctypes.c_void_p,   # C
            ctypes.c_int64,    # M
            ctypes.c_int64,    # N
            ctypes.c_int64,    # K
            ctypes.c_int64,    # stride_A
            ctypes.c_int64,    # stride_AQ
            ctypes.c_int64,    # stride_B
            ctypes.c_int64,    # stride_C
            ctypes.c_int64,    # QK_A
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
        AQ,
        B,
        C,
        M: int,
        N: int,
        K: int,
        stride_A: int,
        stride_AQ: int,
        stride_B: int,
        stride_C: int,
        QK_A: int,
        k_batch: int = 1,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_aquant_gemm with ctypes-wrapped pointers.

        A, AQ, B, C must be numpy arrays (C-contiguous, packed).
        Returns (status, time_ms).
        """
        import numpy as np

        A   = np.ascontiguousarray(A)
        AQ  = np.ascontiguousarray(AQ)
        B   = np.ascontiguousarray(B)
        C   = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_aquant_gemm(
            A.ctypes.data_as(ctypes.c_void_p),
            AQ.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            C.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(M),
            ctypes.c_int64(N),
            ctypes.c_int64(K),
            ctypes.c_int64(stride_A),
            ctypes.c_int64(stride_AQ),
            ctypes.c_int64(stride_B),
            ctypes.c_int64(stride_C),
            ctypes.c_int64(QK_A),
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
# AQuantGpuGemmRunner -- high-level runner
# =============================================================================


class AQuantGpuGemmRunner:
    """
    High-level runner that loads an AQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, AQ, B; allocates C; returns AQuantGemmResult.
    """

    def __init__(self, so_path: Path, layout: str = "rcr"):
        self._lib = AQuantDispatcherLib(so_path)
        self._layout = layout

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, AQ, B, problem: AQuantGemmProblem, c_dtype=None) -> AQuantGemmResult:
        """
        Run AQuantGrouped GEMM.

        Operands are supplied in LOGICAL shape regardless of layout tag:
        A       logical shape: (M, K)      dtype: fp8/bf8/pk_int4
        AQ      logical shape: (M, QK_A)   dtype: float/fp8/bf8
        B       logical shape: (K, N)      dtype: fp8/bf8
        c_dtype numpy dtype for the output C buffer.  Defaults to np.float16
                (correct for all supported aquant variants, whose CDataType is half).
        Returns AQuantGemmResult with C shape (M, N).

        Stride conventions follow the compiled kernel's compile-time A/B/AQ
        layouts, encoded in the layout tag (A, B, C; C is always RowMajor). The
        runner materializes each operand into the layout the kernel expects, so
        callers always pass logical arrays and never pre-transpose.
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_A    = problem.QK_A

        if c_dtype is None:
            c_dtype = np.float16

        # Output buffer -- dtype must match the compiled kernel's CDataType (half).
        C = np.zeros((M, N), dtype=c_dtype)

        # Packed strides derived from the layout tag.
        a_char, b_char, _c_char = self._layout[0], self._layout[1], self._layout[2]
        stride_A  = K if a_char == "r" else M   # A row-major -> K, col-major -> M
        # AQ [M, QK_A]: row-major -> QK_A, column-major (ccr) -> M.
        stride_AQ = _aq_stride(self._layout, M, QK_A)
        stride_B  = N if b_char == "r" else K   # B row-major -> N, col-major -> K
        stride_C  = N                             # C is row-major [M, N]

        # Materialize each LOGICAL operand into the byte layout the kernel reads.
        # The low-level ctypes wrapper copies C-contiguously, so a column-major
        # operand is produced by passing a transposed view (ascontiguousarray of
        # X.T reproduces X's column-major bytes). Row-major operands pass through.
        A_arg  = A if a_char == "r" else np.asarray(A).T
        B_arg  = B if b_char == "r" else np.asarray(B).T
        AQ_arg = np.asarray(AQ).T if self._layout in _LAYOUTS_AQ_COLMAJOR else AQ

        rc, time_ms = self._lib.run(
            A=A_arg, AQ=AQ_arg, B=B_arg, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_AQ=stride_AQ,
            stride_B=stride_B,
            stride_C=stride_C,
            QK_A=QK_A,
            k_batch=problem.k_batch,
        )

        if rc != 0:
            raise RuntimeError(
                f"dispatcher_run_aquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        return AQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers (self-contained, do not call ctypes_utils.py)
# =============================================================================


def _detect_gpu_arch() -> str:
    """Detect current GPU arch via rocm_agent_enumerator. Raises if none found."""
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                return _validate_arch(line)
    except Exception as e:
        raise RuntimeError(f"Could not detect GPU arch via rocm_agent_enumerator: {e}")
    raise RuntimeError(
        "No supported GPU architecture detected; pass gfx_arch explicitly "
        f"(supported: {', '.join(_SUPPORTED_ARCHS)})"
    )


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
    here = Path(__file__).resolve().parent
    for parent in [here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "include"
        if (candidate / "ck_tile").is_dir():
            return candidate
    return None


def _generate_aquant_kernel(
    config: AQuantKernelConfig,
    output_dir: Path,
) -> Optional[Path]:
    """Run unified_gemm_aquant_codegen.py for one config; return the .hpp path or None."""
    config_dict = config.to_codegen_config()
    config_json = json.dumps(config_dict)

    cmd = [
        sys.executable,
        str(_CODEGEN_SCRIPT),
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


def _compile_aquant_kernel(
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
         the AQuant ctypes lib does not use the registry or dispatcher).

    Returns True on success.
    """
    ck_include = _get_ck_include_dir()

    obj_path = so_path.with_suffix(".o")

    # Arch-specific defines: gfx950 uses OCP fp8 (not FNUZ) and native MX support.
    # These mirror the CMakeLists.txt definitions normally injected by CMake but
    # absent in the standalone hipcc build path.
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

    # NOTE: dispatcher/include is intentionally excluded here.  It pulls in
    # generated_tile_backend.hpp which instantiates SelectedKernel::launch(GemmHostArgs&),
    # conflicting with the AQuant kernel's launch(QuantGemmHostArgs&).  The AQuant ctypes
    # lib only needs the main CK include path.

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
        return False

    link_cmd = [hipcc, "-shared", "-fPIC",
                f"--offload-arch={gfx_arch}", "--hip-link",
                str(obj_path), "-o", str(so_path)]

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
# setup_multiple_aquant_dispatchers -- build pipeline
# =============================================================================


def setup_multiple_aquant_dispatchers(
    configs: List[AQuantKernelConfig],
    output_dir: Optional[Path] = None,
    hipcc: str = _DEFAULT_HIPCC,
    gfx_arch: Optional[str] = None,
    extra_include_dirs: Optional[List[str]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
) -> List[Optional[Path]]:
    """
    For each AQuantKernelConfig: codegen -> hipcc compile -> .so path.

    Returns a list parallel to `configs` -- each entry is the Path to the compiled
    .so, or None if that config failed.  No GPU is required to call this function.
    """
    if not configs:
        return []

    arch = _validate_arch(gfx_arch) if gfx_arch else _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="aquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info("Building %d AQuant kernel(s) for %s into %s", len(configs), arch, base_dir)

    # Deduplicate by name so we don't build the same kernel twice.
    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, AQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: AQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_aquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_aquant_kernel(
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

    # Fill in duplicates.
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d AQuant kernels", built, len(configs))
    return results


# =============================================================================
# Sweep expansion: JSON config -> list of AQuantKernelConfig
# =============================================================================


def expand_aquant_sweep(
    config_path: str,
    gfx_arch: str = "gfx950",
) -> List["AQuantKernelConfig"]:
    """Expand an AQuant JSON sweep config into a list of AQuantKernelConfig objects.

    The JSON format mirrors unified_gemm_aquant_codegen.py's _build_specs so the same
    config files work for both codegen and Python utils.  Every valid
    (variant, layout, tile, quant_group) combination produces one AQuantKernelConfig;
    duplicates (by .name) are collapsed.
    """
    import itertools

    with open(config_path) as f:
        cfg = json.load(f)

    preshuffle_aquant  = cfg.get("preshuffle_aquant", False)
    default_scheduler  = "intrawave" if preshuffle_aquant else "interwave"
    scheduler          = cfg.get("scheduler", default_scheduler)
    double_smem_buffer = cfg.get("double_smem_buffer", False)
    k_block_per_cu     = cfg.get("k_block_per_cu", 1)

    allowed_layouts = _LAYOUTS_PRESHUFFLEQUANT if preshuffle_aquant else _LAYOUTS_DECODE

    configs: List[AQuantKernelConfig] = []
    seen: set = set()

    for variant_key, layout, tile_dict, qg in itertools.product(
        cfg.get("variant_keys", ["fp8"]),
        cfg.get("layouts", ["rcr"]),
        cfg.get("tile_configs", []),
        cfg.get("quant_groups", [{"quant_group_m": 1, "quant_group_n": 1, "quant_group_k": 128}]),
    ):
        if layout not in allowed_layouts:
            log.warning("Skipping unsupported layout %s (preshufflequant=%s)",
                        layout, preshuffle_aquant)
            continue
        c = AQuantKernelConfig(
            variant_key=variant_key,
            layout=layout,
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
            preshuffle_aquant=preshuffle_aquant,
            double_smem_buffer=double_smem_buffer,
            k_block_per_cu=k_block_per_cu,
            gfx_arch=gfx_arch,
        )
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)

    return configs


# =============================================================================
# Convenience: default configs (match GemmConfigQuantDecodeInterwave tile defaults)
#
# tile = 16x64x256, warp = 1x4x1, warp_tile = 16x16x{K_warp}
# warp_tile_k is ARCH-DERIVED (never hardcoded); see _warp_tile_k_for below.
# =============================================================================


def _warp_tile_k_for(gfx_arch: str, preshuffle_aquant: bool = False) -> int:
    """Arch-derived K warp-tile, mirroring ck_tile::get_k_warp_tile<PrecType, 16, IsFlatMM>().

    (tile_gemm_shape.hpp:104-136, M_Warp_Tile=16, non-WMMA path.)  For AQuant every
    variant -- fp8, bf8, fp8i4, bf8i4 -- instantiates the GEMM config with an 8-bit
    float PrecType (fp8_t or bf8_t; the pk_int4 A operand does not drive the K warp
    tile -- see gemm_aquant_quantgrouped{,_preshufflequant}.cpp GemmConfig<fp8/bf8_t>).
    So is_8bit_float is always True and warp_tile_k depends only on the arch and the
    pipeline (decode = IsFlatMM false, preshufflequant = IsFlatMM true):

      gfx950 (CK_GFX950_SUPPORT): 128   (both decode and preshufflequant)
      gfx942/other, decode  (IsFlatMM=false): 32
      gfx942/other, preshuf (IsFlatMM=true) : 64

    This is a BLOCKING correctness constraint, not just a naming detail: a
    warp_tile_k=128 fp8/bf8 kernel *compiles* on gfx942 but silently produces
    all-zeros output (GPU-confirmed on gfx942 MI300X for this bridge, and earlier
    on the sibling tensor_quant/rowcolquant bridges).  Old-TE uses 16x16x32 on
    gfx942 for decode and is bit-exact there with warp_tile_k=32.
    """
    if "gfx950" in gfx_arch or "gfx12" in gfx_arch:
        return 128
    # gfx942 / gfx90a / other: 8-bit-float PrecType, M_Warp_Tile=16 non-WMMA path.
    return 64 if preshuffle_aquant else 32


# =============================================================================
# Decode family (GemmConfigQuantDecodeInterwave, tile 16x64x256, IsFlatMM=false)
#   fp8/bf8/fp8i4/bf8i4: K_warp = 128 on gfx950, 32 on gfx942.
# =============================================================================


def _decode_config(
    variant_key: str,
    warp_tile_k: int,
    quant_group_k: int,
    quant_group_n: int,
    layout: str,
    gfx_arch: str,
) -> AQuantKernelConfig:
    return AQuantKernelConfig(
        variant_key=variant_key,
        layout=layout,
        scheduler="interwave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        preshuffle_aquant=False,
        gfx_arch=gfx_arch,
    )


def default_fp8_config(quant_group_k: int = 128, quant_group_n: int = 1,
                       layout: str = "rcr", gfx_arch: str = "gfx950",
                       warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """Default fp8 AQuant decode config (GemmConfigQuantDecodeInterwave<fp8_t>).

    warp_tile_k is arch-derived (get_k_warp_tile<fp8_t, 16>()): 128 on gfx950,
    32 on gfx942 (128 silently outputs all-zeros on gfx942).
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=False)
    return _decode_config("fp8", warp_tile_k, quant_group_k, quant_group_n, layout, gfx_arch)


def default_bf8_config(quant_group_k: int = 128, quant_group_n: int = 1,
                       layout: str = "rcr", gfx_arch: str = "gfx950",
                       warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """Default bf8 AQuant decode config (GemmConfigQuantDecodeInterwave<bf8_t>).

    warp_tile_k is arch-derived: 128 on gfx950, 32 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=False)
    return _decode_config("bf8", warp_tile_k, quant_group_k, quant_group_n, layout, gfx_arch)


def default_fp8i4_config(quant_group_k: int = 128, quant_group_n: int = 1,
                         layout: str = "rcr", gfx_arch: str = "gfx950",
                         warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """Default fp8i4 AQuant decode config (A=pk_int4, B=fp8, Q=fp8).

    PrecType is fp8_t (GemmConfig<fp8_t>), so warp_tile_k is arch-derived like fp8:
    128 on gfx950, 32 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=False)
    return _decode_config("fp8i4", warp_tile_k, quant_group_k, quant_group_n, layout, gfx_arch)


def default_bf8i4_config(quant_group_k: int = 128, quant_group_n: int = 1,
                         layout: str = "rcr", gfx_arch: str = "gfx950",
                         warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """Default bf8i4 AQuant decode config (A=pk_int4, B=bf8, Q=bf8).

    PrecType is bf8_t (GemmConfig<bf8_t>), so warp_tile_k is arch-derived like bf8:
    128 on gfx950, 32 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=False)
    return _decode_config("bf8i4", warp_tile_k, quant_group_k, quant_group_n, layout, gfx_arch)


# =============================================================================
# Preshufflequant family (GemmConfigPreshuffleQuantDecode, tile 16x64x256, IsFlatMM=true)
#   fp8/bf8/fp8i4/bf8i4: K_warp = 128 on gfx950, 64 on gfx942.
# =============================================================================


def _preshufflequant_config(
    variant_key: str,
    warp_tile_k: int,
    quant_group_k: int,
    quant_group_n: int,
    layout: str,
    gfx_arch: str,
) -> AQuantKernelConfig:
    return AQuantKernelConfig(
        variant_key=variant_key,
        layout=layout,
        scheduler="intrawave",
        tile_m=16, tile_n=64, tile_k=256,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        quant_group_m=1,
        quant_group_n=quant_group_n,
        quant_group_k=quant_group_k,
        preshuffle_aquant=True,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshufflequant_config(quant_group_k: int = 128, quant_group_n: int = 1,
                                       layout: str = "rcr",
                                       gfx_arch: str = "gfx950",
                                       warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """fp8 AQuant preshufflequant config (GemmConfigPreshuffleQuantDecode<fp8_t>).

    warp_tile_k is arch-derived (get_k_warp_tile<fp8_t, 16, IsFlatMM=true>()):
    128 on gfx950, 64 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=True)
    return _preshufflequant_config("fp8", warp_tile_k, quant_group_k, quant_group_n,
                                   layout, gfx_arch)


def default_bf8_preshufflequant_config(quant_group_k: int = 128, quant_group_n: int = 1,
                                       layout: str = "rcr",
                                       gfx_arch: str = "gfx950",
                                       warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """bf8 AQuant preshufflequant config (GemmConfigPreshuffleQuantDecode<bf8_t>).

    warp_tile_k is arch-derived: 128 on gfx950, 64 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=True)
    return _preshufflequant_config("bf8", warp_tile_k, quant_group_k, quant_group_n,
                                   layout, gfx_arch)


def default_fp8i4_preshufflequant_config(quant_group_k: int = 128, quant_group_n: int = 1,
                                         layout: str = "rcr",
                                         gfx_arch: str = "gfx950",
                                         warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """fp8i4 AQuant preshufflequant config (A=pk_int4, B=fp8, Q=fp8).

    PrecType is fp8_t, so warp_tile_k is arch-derived: 128 on gfx950, 64 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=True)
    return _preshufflequant_config("fp8i4", warp_tile_k, quant_group_k, quant_group_n,
                                   layout, gfx_arch)


def default_bf8i4_preshufflequant_config(quant_group_k: int = 128, quant_group_n: int = 1,
                                         layout: str = "rcr",
                                         gfx_arch: str = "gfx950",
                                         warp_tile_k: Optional[int] = None) -> AQuantKernelConfig:
    """bf8i4 AQuant preshufflequant config (A=pk_int4, B=bf8, Q=bf8).

    PrecType is bf8_t, so warp_tile_k is arch-derived: 128 on gfx950, 64 on gfx942.
    """
    if warp_tile_k is None:
        warp_tile_k = _warp_tile_k_for(gfx_arch, preshuffle_aquant=True)
    return _preshufflequant_config("bf8i4", warp_tile_k, quant_group_k, quant_group_n,
                                   layout, gfx_arch)
