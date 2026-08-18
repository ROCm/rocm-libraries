#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""
Gemm ABQuant (A+B block-scale) dispatcher utilities.

Three-layer Python bridge for the dispatcher's ABQuantGrouped GEMM path:

  ABQuantKernelConfig  -- describes one kernel; .name is byte-exact with codegen KERNEL_NAME
  ABQuantDispatcherLib -- thin ctypes wrapper around a compiled .so
  ABQuantGpuGemmRunner -- high-level runner that accepts numpy arrays

Build helpers (self-contained, do not import from gemm_utils.py):
  setup_multiple_abquant_dispatchers(configs, ...)
       codegen -> hipcc -> list of .so paths, all in parallel

ABQuant quantizes BOTH A and B:
  AQuantGroupSize is always 1x1x{aquant_group_k} (K-wise scale on A)
  BQuantGroupSize is 1x{bquant_group_n}x{bquant_group_k}

Parity target arch: gfx950 (MI350) is the default. The default-config generators
are ARCH-AWARE (a single config set cannot be byte-identical to Old-TE for both
gfx942 and gfx950): warp_tile_k is arch-derived from get_k_warp_tile<PrecType,16>()
(fp8/bf8 -> 128 on gfx950, 32 on gfx942; fp4 -> 32 everywhere), and the gfx950
eight_waves fast path (GemmConfig/GemmConfigPrefill aliases under CK_USE_GFX950)
is selected for exactly the 6 fp8/bf8 kernels that route through those aliases.

Usage (end-to-end):
  configs = [default_fp8_config()]
  so_paths = setup_multiple_abquant_dispatchers(configs, output_dir=Path("/tmp/abq"))
  runner = ABQuantGpuGemmRunner(so_paths[0])
  result = runner.run(A, B, AQ, BQ, ABQuantGemmProblem(M=128, N=128, K=256))
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

_CODEGEN_SCRIPT = Path(__file__).parent.parent / "codegen" / "unified_gemm_abquant_codegen.py"
_CTYPES_LIB_SRC = Path(__file__).parent.parent / "bindings" / "ctypes" / "gemm_abquant_ctypes_lib.cpp"

# Import the shared name-construction helper from codegen_common so both sides
# stay byte-exact without duplicating the logic.
_codegen_dir = str(Path(__file__).parent.parent / "codegen")
if _codegen_dir not in sys.path:
    sys.path.insert(0, _codegen_dir)
from codegen_common import make_gemm_abquant_kernel_name  # noqa: E402

_DEFAULT_HIPCC = "hipcc"

_COERCE_FLAG_SUPPORTED = None


def _coerce_flag_supported(hipcc: str = _DEFAULT_HIPCC) -> bool:
    """True if the local clang accepts -amdgpu-coerce-illegal-types (LLVM opt).

    clang >= 22 (ROCm 7.2) removed the flag and aborts the compile when it is
    passed. It only tightens EightWaves register allocation on older toolchains;
    the kernel is bit-accurate without it, so probe once and drop it when the
    toolchain rejects it. Cached for the process lifetime.
    """
    global _COERCE_FLAG_SUPPORTED
    if _COERCE_FLAG_SUPPORTED is None:
        try:
            clang = subprocess.run(
                [hipcc, "-print-prog-name=clang++"],
                capture_output=True, text=True, timeout=30,
            ).stdout.strip() or "clang++"
            probe = subprocess.run(
                [clang, "-x", "c++", "-c", "-o", "/dev/null",
                 "-mllvm", "-amdgpu-coerce-illegal-types=1", "-"],
                input="int main(){return 0;}",
                capture_output=True, text=True, timeout=60,
            )
            _COERCE_FLAG_SUPPORTED = "Unknown command line argument" not in (probe.stderr or "")
        except Exception:
            _COERCE_FLAG_SUPPORTED = True
    return _COERCE_FLAG_SUPPORTED

# Architectures the ABQuant bridge supports. NEVER default to gfx942 silently:
# the arch must be detected (get_arch) or explicitly supplied, and unknown archs
# raise (Python) / return an error (C++ runtime check in the ctypes lib).
_SUPPORTED_ARCHS = ("gfx942", "gfx950", "gfx1250")


def _validate_arch(arch: str) -> str:
    """Return arch if supported, else raise. Mirrors the C++ runtime arch check."""
    if not arch or not any(arch.startswith(a) for a in _SUPPORTED_ARCHS):
        raise ValueError(
            f"Unsupported GPU architecture {arch!r} for ABQuant bridge "
            f"(supported: {', '.join(_SUPPORTED_ARCHS)})"
        )
    return arch


# =============================================================================
# ABQuantKernelConfig -- byte-exact naming with codegen
# =============================================================================


@dataclass
class ABQuantKernelConfig:
    """
    Complete description of one ABQuant GEMM kernel.

    The .name property produces the exact string that unified_gemm_abquant_codegen.py
    emits as KERNEL_NAME, ensuring the Python side and compiled .so always agree.
    """

    variant_key: str       # "fp8" | "bf8" | "fp4"
    layout: str            # "rcr" (A=RowMajor, B=ColMajor, C=RowMajor)
    pipeline: str          # "compv3" | "preshuffleb" | "eightwaves"
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

    aquant_group_k: int = 128
    bquant_group_n: int = 1
    bquant_group_k: int = 128

    preshuffle_b: bool       = False
    preshuffle_bquant: bool  = False
    double_smem_buffer: bool = False
    eight_waves: bool        = False
    transpose_c: bool        = False
    pad_k: bool              = False
    k_block_per_cu: int      = 1

    gfx_arch: str = "gfx950"

    @property
    def name(self) -> str:
        """Byte-exact match to codegen KERNEL_NAME (delegates to make_gemm_abquant_kernel_name)."""
        return make_gemm_abquant_kernel_name(
            variant_key=self.variant_key,
            layout=self.layout,
            pipeline=self.pipeline,
            epilogue=self.epilogue,
            scheduler=self.scheduler,
            tile_m=self.tile_m, tile_n=self.tile_n, tile_k=self.tile_k,
            warp_m=self.warp_m, warp_n=self.warp_n, warp_k=self.warp_k,
            warp_tile_m=self.warp_tile_m, warp_tile_n=self.warp_tile_n, warp_tile_k=self.warp_tile_k,
            aquant_group_k=self.aquant_group_k,
            bquant_group_n=self.bquant_group_n,
            bquant_group_k=self.bquant_group_k,
            preshuffle_b=self.preshuffle_b,
            preshuffle_bquant=self.preshuffle_bquant,
            eight_waves=self.eight_waves,
        )

    def to_codegen_config(self) -> dict:
        """Produce the JSON config dict consumed by unified_gemm_abquant_codegen.py."""
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
            "aquant_group_k": self.aquant_group_k,
            "bquant_groups": [{
                "bquant_group_n": self.bquant_group_n,
                "bquant_group_k": self.bquant_group_k,
            }],
            "preshuffle_b": self.preshuffle_b,
            "preshuffle_bquant": self.preshuffle_bquant,
            "double_smem_buffer": self.double_smem_buffer,
            "eight_waves": self.eight_waves,
            "transpose_c": self.transpose_c,
            "pad_k": self.pad_k,
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
    aquant_group_k: int = 128
    bquant_group_n: int = 1
    bquant_group_k: int = 128
    k_batch: int = 1

    @property
    def QK_A(self) -> int:
        """Number of A K-groups: ceil(K / aquant_group_k)."""
        return (self.K + self.aquant_group_k - 1) // self.aquant_group_k

    @property
    def QK_B(self) -> int:
        """Number of B K-groups: ceil(K / bquant_group_k)."""
        return (self.K + self.bquant_group_k - 1) // self.bquant_group_k

    @property
    def QN_B(self) -> int:
        """Number of B N-groups: ceil(N / bquant_group_n)."""
        return (self.N + self.bquant_group_n - 1) // self.bquant_group_n


# =============================================================================
# ABQuantGemmResult
# =============================================================================


@dataclass
class ABQuantGemmResult:
    C: object          # numpy array
    time_ms: float
    kernel_name: str


# =============================================================================
# ABQuantDispatcherLib -- thin ctypes wrapper
# =============================================================================


class ABQuantDispatcherLib:
    """
    Loads a compiled abquant_gemm .so and wraps its C API.

    Expected .so exports:
      int  dispatcher_initialize()
      int  dispatcher_run_abquant_gemm(A, B, AQ, BQ, C, M, N, K,
                                        stride_A, stride_B, stride_AQ, stride_BQ, stride_C,
                                        QK_A, QK_B, QN_B, k_batch, *time_ms)
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
        stride_AQ: int,
        stride_BQ: int,
        stride_C: int,
        QK_A: int,
        QK_B: int,
        QN_B: int,
        k_batch: int = 1,
        aq_column_major: bool = False,
    ) -> Tuple[int, float]:
        """
        Call dispatcher_run_abquant_gemm with ctypes-wrapped pointers.

        A, B, AQ, BQ, C must be numpy arrays (C-contiguous, packed).
        B should be a packed (K, N) array supplied column-major (stride_B=K).
        C must be the array that will receive output.
        aq_column_major supplies AQ as column-major bytes (leading dim = M) for
        the n=128 EightWaves fast path; otherwise AQ is row-major (leading dim=QK_A).
        Returns (status, time_ms).
        """
        import numpy as np

        A  = np.ascontiguousarray(A)
        # Kernel BLayout is ColumnMajor (rcr): B[k,n] lives at offset n*K+k.
        # Supply column-major bytes for 2-D B; ascontiguousarray would force
        # row-major and silently transpose. Packed 1-D B (fp4) stays as-is.
        B  = np.asfortranarray(B) if B.ndim == 2 else np.ascontiguousarray(B)
        # AQLayout is ColumnMajor for the n=128 EightWaves fast path (StrideAQ=M):
        # supply Fortran-order [M, QK_A] bytes so AQ[m,qk] lives at qk*M+m.
        if aq_column_major and AQ.ndim == 2:
            AQ = np.asfortranarray(AQ)
        else:
            AQ = np.ascontiguousarray(AQ)
        # BQLayout is ColumnMajor [QK_B, QN_B]: BQ[k,n] at offset n*QK_B+k.
        BQ = np.asfortranarray(BQ) if BQ.ndim == 2 else np.ascontiguousarray(BQ)
        C  = np.ascontiguousarray(C)

        time_ms = ctypes.c_float(0.0)

        rc = self._lib.dispatcher_run_abquant_gemm(
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
            ctypes.c_int64(QN_B),
            ctypes.c_int(k_batch),
            ctypes.byref(time_ms),
        )
        return rc, time_ms.value

    @staticmethod
    def kernel_uses_column_major_aq(kernel_name: str) -> bool:
        """Whether a kernel name resolves to the ColumnMajor-AQ EightWaves path.

        Mirrors the codegen AQLayout rule (run_gemm_quant_example.inc:1013-1021):
        BQuantGroupSize::kN == 128 && M_Warp*N_Warp*K_Warp == 8. Since warps==8
        occurs only for the 4x2x1 EightWaves configs, we detect 'eightwaves' plus
        the 'bqg1x128x' N-group segment in the byte-exact kernel name.
        """
        return "eightwaves" in kernel_name and "bqg1x128x" in kernel_name

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
# ABQuantGpuGemmRunner -- high-level runner
# =============================================================================


class ABQuantGpuGemmRunner:
    """
    High-level runner that loads an ABQuant .so and executes GEMM on the GPU.

    Accepts numpy arrays for A, B, AQ, BQ; allocates C; returns ABQuantGemmResult.
    """

    def __init__(self, so_path: Path):
        self._lib = ABQuantDispatcherLib(so_path)

    @property
    def kernel_name(self) -> str:
        return self._lib.get_kernel_name()

    def run(self, A, B, AQ, BQ, problem: ABQuantGemmProblem, c_dtype=None) -> ABQuantGemmResult:
        """
        Run ABQuant GEMM.

        A       shape: (M, K)            dtype: fp8/bf8/fp4 (packed uint8)
        B       shape: (K, N) col-major  dtype: fp8/bf8/fp4 (packed uint8)
        AQ      shape: (M, QK_A)         dtype: float32 (row-major)
        BQ      shape: (QK_B, QN_B)      dtype: float32 (col-major)
        c_dtype numpy dtype for the output C buffer.  Defaults to np.float16
                (CDataType is half_t for all abquant variants).
        Returns ABQuantGemmResult with C shape (M, N).
        """
        import numpy as np

        M, N, K = problem.M, problem.N, problem.K
        QK_A    = problem.QK_A
        QK_B    = problem.QK_B
        QN_B    = problem.QN_B

        if c_dtype is None:
            c_dtype = np.float16

        # Output buffer -- dtype must match the compiled kernel's CDataType.
        C = np.zeros((M, N), dtype=c_dtype)

        # AQLayout is ColumnMajor for the n=128 EightWaves fast path (StrideAQ=M);
        # RowMajor (StrideAQ=QK_A) everywhere else.
        aq_column_major = ABQuantDispatcherLib.kernel_uses_column_major_aq(self.kernel_name)

        # Strides (in elements). A / C are row-major; B / BQ are col-major.
        stride_A  = K                    # A is row-major [M, K]
        stride_B  = K                    # B is col-major [K, N] -> leading dim = K
        stride_AQ = M if aq_column_major else QK_A  # AQ leading dim per AQLayout
        stride_BQ = QK_B                 # BQ is col-major [QK_B, QN_B] -> leading dim = QK_B
        stride_C  = N                    # C is row-major [M, N]

        rc, time_ms = self._lib.run(
            A=A, B=B, AQ=AQ, BQ=BQ, C=C,
            M=M, N=N, K=K,
            stride_A=stride_A,
            stride_B=stride_B,
            stride_AQ=stride_AQ,
            stride_BQ=stride_BQ,
            stride_C=stride_C,
            QK_A=QK_A,
            QK_B=QK_B,
            QN_B=QN_B,
            k_batch=problem.k_batch,
            aq_column_major=aq_column_major,
        )

        if rc != 0:
            # rc == -3 is the graceful "fp4 + PreshuffleB is unsupported" reject,
            # mirroring Old-TE's throw ("Preshuffling weight matrix is not supported
            # for ... bf16_fp4_gemm", run_gemm_quant_example.inc:994-1001). The ctypes
            # lib returns before any device alloc so there is no heap corruption --
            # surface it as a clear, catchable error instead of a malloc abort.
            if rc == -3:
                raise RuntimeError(
                    "Preshuffling weight matrix is not supported for bf16_fp4_gemm "
                    f"(kernel {self.kernel_name}); matches Old-TE reject"
                )
            raise RuntimeError(
                f"dispatcher_run_abquant_gemm failed with code {rc} "
                f"for kernel {self.kernel_name}"
            )

        # NOTE: no post-hoc permute_n de-permute here. Round-4's bq_permuteN fix
        # makes the kernel/ctypes epilogue write C directly in correct logical
        # column order for permute_n kernels, so the ctypes output is already
        # right. The former wrapper-side de-permute (undoing an r-group riffle)
        # was a workaround for the pre-fix scrambled output; applying it now
        # would double-correct and scramble C. Pass the ctypes output through
        # unchanged for every kernel (permute_n and non-permute_n alike).
        return ABQuantGemmResult(C=C, time_ms=time_ms, kernel_name=self.kernel_name)


# =============================================================================
# Subprocess helpers (self-contained, do not call ctypes_utils.py)
# =============================================================================


def _detect_gpu_arch() -> str:
    """Detect current GPU arch via rocm_agent_enumerator. Raises if unknown.

    Never defaults to gfx942 silently -- if detection fails or yields an
    unsupported arch, _validate_arch raises.
    """
    try:
        result = subprocess.run(
            ["rocm_agent_enumerator"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("gfx") and line != "gfx000":
                return _validate_arch(line)
    except Exception:
        pass
    raise RuntimeError(
        "Could not detect a supported GPU architecture via rocm_agent_enumerator. "
        f"Pass gfx_arch explicitly (supported: {', '.join(_SUPPORTED_ARCHS)})."
    )


def _get_ck_include_dir() -> Optional[Path]:
    """Attempt to locate the CK include directory relative to this file."""
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
    """Run unified_gemm_abquant_codegen.py for one config; return the .hpp path or None."""
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


def _compile_abquant_kernel(
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
         the ABQuant ctypes lib does not use the registry or dispatcher).

    Returns True on success.
    """
    _validate_arch(gfx_arch)
    ck_include = _get_ck_include_dir()

    # -- Step 1: compile to object file --------------------------------------
    obj_path = so_path.with_suffix(".o")

    # Arch-specific defines: gfx950 uses OCP fp8 (not FNUZ), native MX support,
    # and the CK_GFX950_SUPPORT flag that gates the eight_waves fast path.
    arch_defines = []
    if "gfx950" in gfx_arch:
        arch_defines += ["-DCK_USE_OCP_FP8", "-DCK_TILE_USE_OCP_FP8",
                         "-DCK_USE_NATIVE_MX_SUPPORT", "-DCK_GFX950_SUPPORT",
                         "-DCK_USE_GFX950"]

    # Tile-Engine performance codegen flags. The Old-TE example/test build
    # (example/ck_tile/38_block_scale_gemm, develop CMake) compiles every kernel
    # -- including the gfx950 EightWaves (192x256x128, 8-wave) block-scale fast
    # path -- with this exact -mllvm flag set. Without them, hipcc -O3 register
    # allocation for the EightWaves hot loop peaks at 256 VGPRs and spills to
    # scratch (private_segment_fixed_size > 0), which collapses occupancy and
    # makes the bridge kernel ~3x slower than the byte-identical Old-TE kernel
    # (+140..+307% on fp8/bf8 n=128). With these flags the bridge kernel matches
    # Old-TE's register profile exactly (229 VGPRs, zero scratch) and the gap
    # collapses to within +/-5%. Kept in lockstep with the develop TE flags so
    # the ctypes .so is codegen-identical to the Old-TE build (fair parity).
    perf_flags = [
        "-fno-offload-uniform-block",
        "-mllvm", "--lsr-drop-solution=1",
        "-mllvm", "-enable-post-misched=0",
        "-mllvm", "-amdgpu-early-inline-all=true",
        "-mllvm", "-amdgpu-function-calls=false",
        "-mllvm", "-enable-noalias-to-md-conversion=1",
        "-mllvm", "-greedy-reverse-local-assignment=1",
    ]
    # clang < 22 only: -amdgpu-coerce-illegal-types tightens EightWaves
    # register allocation, but clang >= 22 (ROCm 7.2) removed it and aborts
    # the compile. Include it only when the toolchain accepts it (the kernel
    # is bit-accurate without it).
    if _coerce_flag_supported(hipcc):
        perf_flags += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

    compile_cmd = [hipcc, "-c", "-fPIC", "-O3", "-std=c++17",
                   "-DCK_TILE_SINGLE_KERNEL_INCLUDE", "-w",
                   f"--offload-arch={gfx_arch}",
                   f"-DGFX_ARCH=\"{gfx_arch}\"",
                   *arch_defines,
                   *perf_flags,
                   "-include", str(hpp_path),
                   str(_CTYPES_LIB_SRC),
                   "-o", str(obj_path)]

    if ck_include:
        compile_cmd += [f"-I{ck_include}"]

    # NOTE: dispatcher/include is intentionally excluded here (same rationale as
    # the BQuant bridge): it pulls in generated_tile_backend.hpp which instantiates
    # SelectedKernel::launch(GemmHostArgs&), conflicting with the ABQuant kernel's
    # launch(QuantGemmHostArgs&). The ctypes lib only needs the main CK include path.
    if extra_include_dirs:
        for d in extra_include_dirs:
            compile_cmd += [f"-I{d}"]

    log.debug("Compiling %s:\n  %s", so_path.name, " ".join(compile_cmd))

    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=900)
        if result.returncode != 0:
            log.error("Compile failed for %s:\n%s", so_path.name, result.stderr[-2000:])
            return False
    except subprocess.TimeoutExpired:
        log.error("Compile timed out for %s", so_path.name)
        return False

    # -- Step 2: link into shared library ------------------------------------
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
# setup_multiple_abquant_dispatchers -- build pipeline
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
    """
    For each ABQuantKernelConfig: codegen -> hipcc compile -> .so path.

    Returns a list parallel to `configs` -- each entry is the Path to the
    compiled .so, or None if that config failed.

    No GPU is required to call this function. The arch must be supplied or
    detectable; _validate_arch raises on unknown archs (never silent gfx942).
    """
    if not configs:
        return []

    arch = _validate_arch(gfx_arch) if gfx_arch else _detect_gpu_arch()
    base_dir = output_dir or Path(tempfile.mkdtemp(prefix="abquant_dispatcher_"))
    base_dir.mkdir(parents=True, exist_ok=True)

    headers_dir = base_dir / "generated_kernels"
    so_dir      = base_dir / "libs"
    headers_dir.mkdir(exist_ok=True)
    so_dir.mkdir(exist_ok=True)

    log.info("Building %d ABQuant kernel(s) for %s into %s", len(configs), arch, base_dir)

    # Deduplicate by name so we don't build the same kernel twice
    seen: Dict[str, int] = {}
    deduped: List[Tuple[int, ABQuantKernelConfig]] = []
    for i, cfg in enumerate(configs):
        if cfg.name not in seen:
            seen[cfg.name] = i
            deduped.append((i, cfg))

    results: List[Optional[Path]] = [None] * len(configs)

    def _build_one(idx: int, cfg: ABQuantKernelConfig) -> Tuple[int, Optional[Path]]:
        hpp = _generate_abquant_kernel(cfg, headers_dir)
        if hpp is None:
            return idx, None

        so = so_dir / f"lib{cfg.name}_{arch}.so"
        if so.exists():
            log.info("  [cached] %s", so.name)
            return idx, so

        ok = _compile_abquant_kernel(
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

    # Fill in duplicates
    for i, cfg in enumerate(configs):
        if results[i] is None:
            first_idx = seen.get(cfg.name)
            if first_idx is not None and first_idx != i:
                results[i] = results[first_idx]

    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d ABQuant kernels", built, len(configs))
    return results


# =============================================================================
# Sweep expansion: JSON config -> list of ABQuantKernelConfig
# =============================================================================


def expand_abquant_sweep(
    config_path: str,
    gfx_arch: str = "gfx950",
) -> List["ABQuantKernelConfig"]:
    """Expand an ABQuant JSON sweep config into a list of ABQuantKernelConfig objects.

    The JSON format mirrors unified_gemm_abquant_codegen.py's _build_specs so the
    same config files work for both codegen and Python utils. Every valid
    (variant, layout, tile, bquant_group) combination produces one
    ABQuantKernelConfig; duplicates (by .name) are collapsed.
    """
    import itertools

    _validate_arch(gfx_arch)
    with open(config_path) as f:
        cfg = json.load(f)

    pipeline           = cfg.get("pipeline", "compv3")
    epilogue           = cfg.get("epilogue", "cshuffle")
    scheduler          = cfg.get("scheduler", "intrawave")
    pad_k              = cfg.get("pad_k", False)
    transpose_c        = cfg.get("transpose_c", False)
    k_block_per_cu     = cfg.get("k_block_per_cu", 1)
    double_smem_buffer = cfg.get("double_smem_buffer", False)
    preshuffle_b       = cfg.get("preshuffle_b", False)
    preshuffle_bquant  = cfg.get("preshuffle_bquant", False)
    eight_waves        = cfg.get("eight_waves", False)
    aquant_group_k     = cfg.get("aquant_group_k", 128)

    configs: List[ABQuantKernelConfig] = []
    seen: set = set()

    for variant_key, layout, tile_dict, bqg in itertools.product(
        cfg.get("variant_keys", ["fp8"]),
        cfg.get("layouts", ["rcr"]),
        cfg.get("tile_configs", []),
        cfg.get("bquant_groups", [{"bquant_group_n": 1, "bquant_group_k": 128}]),
    ):
        c = ABQuantKernelConfig(
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
            aquant_group_k=aquant_group_k,
            bquant_group_n=bqg.get("bquant_group_n", 1),
            bquant_group_k=bqg.get("bquant_group_k", 128),
            preshuffle_b=preshuffle_b,
            preshuffle_bquant=preshuffle_bquant,
            double_smem_buffer=double_smem_buffer,
            eight_waves=eight_waves,
            transpose_c=transpose_c,
            pad_k=pad_k,
            k_block_per_cu=k_block_per_cu,
            gfx_arch=gfx_arch,
        )
        if c.name not in seen:
            seen.add(c.name)
            configs.append(c)

    return configs


# =============================================================================
# Default configs -- one per Old-TE gemm_abquant_quantgrouped*.cpp entry
#
# ARCH-AWARE (see finding #1/#2/#3): one config set CANNOT be byte-identical to
# Old-TE for both gfx942 and gfx950, because the compiled kernel shape differs:
#
#   * warp_tile_k = get_k_warp_tile<PrecType, 16>()  (tile_gemm_shape.hpp:104-136)
#       gfx950: fp8/bf8 -> 128, fp4 (pk_fp4) -> 32
#       gfx942: all             -> 32
#
#   * eight_waves fast path (gemm_abquant_quantgrouped.h:8-18): under CK_USE_GFX950
#       GemmConfig<T>        aliases GemmConfigEightWaves<T>            (non-preshuffleb)
#       GemmConfigPrefill<T> aliases GemmConfigPreshuffleBEightWaves<T> (preshuffleb)
#     and run_gemm_quant_example.inc enables eight_waves iff
#       IS_FP8BLOCKSCALE && M_Warp*N_Warp*K_Warp==8 && K_Warp_Tile==128.
#     With get_k_warp_tile<fp8/bf8,16>()==128 and the EightWaves 4x2x1 warps this
#     is TRUE on gfx950 for exactly the 6 fp8/bf8 kernels that go through those
#     two aliases (non-preshuffleb non-pq 1x128x128, and preshuffleb {1,128}).
#     On gfx942 (or fp4, or the hardcoded GemmConfigABQuantPrefill entries)
#     eight_waves is FALSE.
#
#   EightWaves shape (GemmConfigEightWaves / GemmConfigPreshuffleBEightWaves):
#       M_Warp=4, N_Warp=2, K_Warp=1 ; M_Tile=192, N_Tile=256,
#       K_Tile=128/sizeof(PrecType)=128 (fp8/bf8) ; warp_tile 16x16x128 ;
#       kBlockPerCu=1 ; TransposeC defaults true.
#
# Non-eight_waves prefill variants use GemmConfigABQuantPrefill
# (= GemmConfigQuantPrefill + kPadK=false): tile 128x128x128, warp 1x4x1,
# warp_tile 16x16x{K_warp}.
# preshuffleb (non-eight_waves) uses GemmConfigPreshuffleB_ABQuant_Prefill:
# warp 2x2x1, preshuffle_b + double_smem, kBlockPerCu=2, kPadK=false.
#
# SCOPE NOTE -- fp8/bf8 preshuffleb n=1 (non-pq) is a PRE-EXISTING Old-TE bug, NOT
# a bridge defect. The fp8/bf8 preshuffleb + bquant_group_n=1 (non-preshufflequant)
# kernels "fail" ~71% of shapes on gfx950 -- but this reproduces Old-TE 1:1: Old-TE's
# own example (-v=1 verification) also fails these for group_size 1x1x128. The bridge
# faithfully mirrors Old-TE's shuffle_b / bq_permuteN / shuffle_bq host path, so
# matching Old-TE's (buggy) output IS parity. Do NOT "fix" beyond Old-TE. The
# preshuffleb + preshufflequant n=1 (permute_n) family and the {1,128} eight_waves
# families are correct/at-parity; only the non-pq n=1 group-size-1x1x128 case carries
# the upstream defect.
# =============================================================================


def _warp_tile_k_for(variant_key: str, gfx_arch: str, is_flat_mm: bool = False) -> int:
    """Arch-derived K warp-tile, mirroring ck_tile::get_k_warp_tile<PrecType, 16, IsFlatMM>().

    (tile_gemm_shape.hpp:104-136, M_Warp_Tile=16, non-WMMA path)
      gfx950 (CK_GFX950_SUPPORT): fp8/bf8 -> 128, non-8bit-float (fp4) -> 32
                                  (IsFlatMM does NOT change the gfx950 result)
      gfx942/other              : IsFlatMM==false -> 32 ; IsFlatMM==true -> 64
                                  (sizeof(PrecType)==2 i.e. 16-bit is the only 32-case
                                   under IsFlatMM, and abquant has no 16-bit variant)

    IsFlatMM is TRUE for the preshuffleb prefill configs: Old-TE's
    GemmConfigPreshuffleB_BQuant_Prefill (the base of GemmConfigPreshuffleB_ABQuant_Prefill)
    derives K_Warp_Tile via get_k_warp_tile<PrecType, 16, /*IsFlatMM=*/true>() (gemm_utils.hpp
    line 208-209), which yields 64 on gfx942 for the 8-bit (fp8/bf8/fp4-packed) variants.
    The non-preshuffle prefill (GemmConfigQuantPrefill) and the preshufflequant-only config
    (GemmConfigPreshuffleBQuantPrefill, extends GemmConfigQuantPrefill) use the default
    IsFlatMM=false -> 32 on gfx942, so they must NOT pass is_flat_mm=True.
    """
    is_8bit_float = variant_key in ("fp8", "bf8")
    if "gfx950" in gfx_arch or "gfx12" in gfx_arch:
        # CK_GFX950_SUPPORT branch: is_8bit_float ? 128 : 32.  IsFlatMM is IGNORED here
        # (the gfx950 M_Warp_Tile==16 else-branch does not depend on IsFlatMM), so fp4
        # preshuffleb stays 32 on gfx950 -- do NOT bump it to 64.
        return 128 if is_8bit_float else 32
    # gfx942/other (no CK_GFX950_SUPPORT, non-WMMA): M_Warp_Tile==16 else-branch is
    #   (sizeof(PrecType)==2 || IsFlatMM==false) ? 32 : 64.
    # abquant variants (fp8/bf8/pk_fp4) are all 1-byte, so IsFlatMM==true -> 64, else 32.
    if is_flat_mm:
        return 64
    return 32


def _uses_eight_waves(variant_key: str, gfx_arch: str) -> bool:
    """Whether this fp8/bf8-blockscale family resolves to the eight_waves fast path.

    Only gfx950 aliases GemmConfig/GemmConfigPrefill to the EightWaves configs
    (gemm_abquant_quantgrouped.h), and only fp8/bf8 have K_Warp_Tile==128, so the
    IS_FP8BLOCKSCALE && warps==8 && K_Warp_Tile==128 predicate is TRUE only here.
    """
    return "gfx950" in gfx_arch and variant_key in ("fp8", "bf8")


def _abquant_eight_waves_config(
    variant_key: str,
    pipeline: str,           # "compv3" (non-preshuffleb) | "preshuffleb"
    preshuffle_b: bool,
    bquant_group_n: int,
    gfx_arch: str,
) -> ABQuantKernelConfig:
    """gfx950 EightWaves prefill (GemmConfig[Prefill] = *EightWaves under CK_USE_GFX950).

    M_Warp=4, N_Warp=2, K_Warp=1 ; M_Tile=192, N_Tile=256, K_Tile=128 ;
    warp_tile 16x16x128 ; TransposeC=true (alias default) ; kBlockPerCu=1 ; kPadK=false.
    The eight_waves flag routes codegen to ABQuantGemmPipelineAgBgCrEightWaves and
    kernel_attr<true>; pipeline "eightwaves" overrides the preshuffleb pipeline
    selection exactly as run_gemm_quant_example.inc does.
    """
    return ABQuantKernelConfig(
        variant_key=variant_key,
        layout="rcr",
        pipeline="eightwaves",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=192, tile_n=256, tile_k=128,
        warp_m=4, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=128,
        aquant_group_k=128,
        bquant_group_n=bquant_group_n,
        bquant_group_k=128,
        preshuffle_b=preshuffle_b,
        preshuffle_bquant=False,
        double_smem_buffer=preshuffle_b,   # GemmConfigPreshuffleBEightWaves sets it
        eight_waves=True,
        transpose_c=True,
        pad_k=False,
        k_block_per_cu=1,
        gfx_arch=gfx_arch,
    )


def _abquant_prefill_config(
    variant_key: str,
    warp_tile_k: int,
    bquant_group_n: int,
    transpose_c: bool,
    gfx_arch: str,
) -> ABQuantKernelConfig:
    """Non-preshuffle ABQuant prefill (GemmConfigABQuantPrefill<PrecType, TransposeC>)."""
    return ABQuantKernelConfig(
        variant_key=variant_key,
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        aquant_group_k=128,
        bquant_group_n=bquant_group_n,
        bquant_group_k=128,
        transpose_c=transpose_c,
        pad_k=False,
        gfx_arch=gfx_arch,
    )


# =============================================================================
# Decode family (non-preshuffle, tile 128x128x128, warp 1x4x1)
#   fp8/bf8: K_warp = 128 on gfx950 (EightWaves for bquant_group_n>1), 32 on gfx942.
#   fp4:     K_warp = 32 on all arches.
# =============================================================================


def default_fp8_config(bquant_group_n: int = 1, gfx_arch: str = "gfx950") -> ABQuantKernelConfig:
    """fp8 ABQuant, non-preshuffle.

    Old-TE (gemm_abquant_quantgrouped_fp8.cpp):
      bquant_group_n=1  -> GemmConfigABQuantPrefill<fp8,false>  (hardcoded, NOT eight_waves)
      bquant_group_n=128-> GemmConfig<fp8>  = EightWaves on gfx950
    """
    if bquant_group_n > 1 and _uses_eight_waves("fp8", gfx_arch):
        return _abquant_eight_waves_config("fp8", pipeline="compv3", preshuffle_b=False,
                                           bquant_group_n=bquant_group_n, gfx_arch=gfx_arch)
    return _abquant_prefill_config("fp8", warp_tile_k=_warp_tile_k_for("fp8", gfx_arch),
                                   bquant_group_n=bquant_group_n,
                                   transpose_c=(bquant_group_n > 1), gfx_arch=gfx_arch)


def default_bf8_config(bquant_group_n: int = 1, gfx_arch: str = "gfx950") -> ABQuantKernelConfig:
    """bf8 ABQuant, non-preshuffle (same alias split as fp8)."""
    if bquant_group_n > 1 and _uses_eight_waves("bf8", gfx_arch):
        return _abquant_eight_waves_config("bf8", pipeline="compv3", preshuffle_b=False,
                                           bquant_group_n=bquant_group_n, gfx_arch=gfx_arch)
    return _abquant_prefill_config("bf8", warp_tile_k=_warp_tile_k_for("bf8", gfx_arch),
                                   bquant_group_n=bquant_group_n,
                                   transpose_c=(bquant_group_n > 1), gfx_arch=gfx_arch)


def default_fp4_config(gfx_arch: str = "gfx950") -> ABQuantKernelConfig:
    """fp4 ABQuant, non-preshuffle (only bquant_group_n=128; hardcoded
    GemmConfigABQuantPrefill<pk_fp4_raw_t>, never eight_waves, warp_tile_k=32)."""
    return _abquant_prefill_config("fp4", warp_tile_k=_warp_tile_k_for("fp4", gfx_arch),
                                   bquant_group_n=128, transpose_c=False, gfx_arch=gfx_arch)


# =============================================================================
# Preshufflequant-only family (GemmConfigPreshuffleBQuantPrefill, tile 128x128x128)
#   NOT eight_waves. IsFlatMM=false: K_warp = 128 on gfx950, 32 on gfx942.
# =============================================================================


def default_fp8_preshufflequant_config(
    bquant_group_n: int = 1, gfx_arch: str = "gfx950"
) -> ABQuantKernelConfig:
    """fp8 ABQuant + preshufflequant (GemmConfigPreshuffleBQuantPrefill<fp8>).

    BPreshuffleQuant=true, APreshuffleQuant stays false; base is GemmConfigQuantPrefill
    so TransposeC defaults false and kPadK=true. NOT eight_waves (uses the explicit
    GemmConfigPreshuffleBQuantPrefill, not the GemmConfig alias). warp_tile_k arch-derived.
    """
    return ABQuantKernelConfig(
        variant_key="fp8",
        layout="rcr",
        pipeline="compv3",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=1, warp_n=4, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=_warp_tile_k_for("fp8", gfx_arch),
        aquant_group_k=128,
        bquant_group_n=bquant_group_n,
        bquant_group_k=128,
        preshuffle_bquant=True,
        pad_k=True,
        gfx_arch=gfx_arch,
    )


# =============================================================================
# Preshuffleb family (tile 128x128x128, warp 2x2x1, DoubleSmemBuffer=true)
#   fp8/bf8: gfx950 -> EightWaves (warp 4x2x1, K_warp=128); gfx942 -> K_warp=64 (IsFlatMM=true).
#   fp4:     K_warp=32 on all arches; never eight_waves.
# =============================================================================


def _abquant_preshuffleb_config(
    variant_key: str,
    warp_tile_k: int,
    bquant_group_n: int,
    preshuffle_bquant: bool,
    transpose_c: bool,
    gfx_arch: str,
) -> ABQuantKernelConfig:
    """preshuffleb ABQuant (GemmConfigPreshuffleB_ABQuant_Prefill<PrecType,TransposeC>).

    warp 2x2x1, preshuffle_b + double_smem, kBlockPerCu=2, kPadK=false.
    """
    return ABQuantKernelConfig(
        variant_key=variant_key,
        layout="rcr",
        pipeline="preshuffleb",
        epilogue="cshuffle",
        scheduler="intrawave",
        tile_m=128, tile_n=128, tile_k=128,
        warp_m=2, warp_n=2, warp_k=1,
        warp_tile_m=16, warp_tile_n=16, warp_tile_k=warp_tile_k,
        aquant_group_k=128,
        bquant_group_n=bquant_group_n,
        bquant_group_k=128,
        preshuffle_b=True,
        preshuffle_bquant=preshuffle_bquant,
        double_smem_buffer=True,
        transpose_c=transpose_c,
        pad_k=False,
        k_block_per_cu=2,
        gfx_arch=gfx_arch,
    )


def default_fp8_preshuffleb_config(
    bquant_group_n: int = 1, gfx_arch: str = "gfx950"
) -> ABQuantKernelConfig:
    """fp8 ABQuant + preshuffleb.

    Old-TE (_preshuffleb_fp8.cpp) uses GemmConfigPrefill<fp8> for BOTH n, which on
    gfx950 aliases GemmConfigPreshuffleBEightWaves<fp8> -> eight_waves. On gfx942 it
    is GemmConfigPreshuffleB_ABQuant_Prefill<fp8,true> (warp 2x2x1, warp_tile_k=32).
    """
    if _uses_eight_waves("fp8", gfx_arch):
        return _abquant_eight_waves_config("fp8", pipeline="preshuffleb", preshuffle_b=True,
                                           bquant_group_n=bquant_group_n, gfx_arch=gfx_arch)
    return _abquant_preshuffleb_config(
        "fp8", warp_tile_k=_warp_tile_k_for("fp8", gfx_arch, is_flat_mm=True),
        bquant_group_n=bquant_group_n,
        preshuffle_bquant=False, transpose_c=True, gfx_arch=gfx_arch)


def default_bf8_preshuffleb_config(
    bquant_group_n: int = 1, gfx_arch: str = "gfx950"
) -> ABQuantKernelConfig:
    """bf8 ABQuant + preshuffleb (same GemmConfigPrefill alias split as fp8)."""
    if _uses_eight_waves("bf8", gfx_arch):
        return _abquant_eight_waves_config("bf8", pipeline="preshuffleb", preshuffle_b=True,
                                           bquant_group_n=bquant_group_n, gfx_arch=gfx_arch)
    return _abquant_preshuffleb_config(
        "bf8", warp_tile_k=_warp_tile_k_for("bf8", gfx_arch, is_flat_mm=True),
        bquant_group_n=bquant_group_n,
        preshuffle_bquant=False, transpose_c=True, gfx_arch=gfx_arch)


def default_fp4_preshuffleb_config(gfx_arch: str = "gfx950") -> ABQuantKernelConfig:
    """fp4 ABQuant + preshuffleb (only bquant_group_n=128; explicit
    GemmConfigPreshuffleB_ABQuant_Prefill<pk_fp4_raw_t>, never eight_waves)."""
    return _abquant_preshuffleb_config(
        "fp4", warp_tile_k=_warp_tile_k_for("fp4", gfx_arch, is_flat_mm=True),
        bquant_group_n=128,
        preshuffle_bquant=False, transpose_c=True, gfx_arch=gfx_arch)


# =============================================================================
# Preshuffleb + preshufflequant combined (tile 128x128x128, IsFlatMM=true)
#   NOT eight_waves on either arch. K_warp = 128 on gfx950, 64 on gfx942.
# =============================================================================


def default_fp8_preshuffleb_preshufflequant_config(
    bquant_group_n: int = 1, gfx_arch: str = "gfx950"
) -> ABQuantKernelConfig:
    """fp8 ABQuant + preshuffleb + preshufflequant
    (GemmConfigPreshuffleB_ABQuant_PreshuffleBQuant_Prefill<fp8, TransposeC>).

    Uses the EXPLICIT *_PreshuffleBQuant_Prefill config (not the GemmConfigPrefill
    alias), so it is NOT eight_waves on either arch. warp_tile_k arch-derived.
    TransposeC=false for 1x1x128, true for 1x128x128 (mirrors the Old-TE lut).
    """
    return _abquant_preshuffleb_config(
        "fp8", warp_tile_k=_warp_tile_k_for("fp8", gfx_arch, is_flat_mm=True),
        bquant_group_n=bquant_group_n,
        preshuffle_bquant=True, transpose_c=(bquant_group_n > 1), gfx_arch=gfx_arch)


# =============================================================================
# Convenience: the full Old-TE matrix as a single list of configs
# =============================================================================


def all_default_configs(gfx_arch: str = "gfx950") -> List[ABQuantKernelConfig]:
    """Return every ABQuant config that maps to a Old-TE gemm_abquant lut entry.

    dtype x layout x preshuffle matrix:
      fp8: non-preshuffle {1,128}; preshufflequant {1,128};
           preshuffleb {1,128}; preshuffleb+preshufflequant {1,128}
      bf8: non-preshuffle {1,128}; preshuffleb {1,128}
      fp4: non-preshuffle {128}; preshuffleb {128}
    """
    cfgs: List[ABQuantKernelConfig] = []
    # fp8 non-preshuffle
    cfgs.append(default_fp8_config(bquant_group_n=1, gfx_arch=gfx_arch))
    cfgs.append(default_fp8_config(bquant_group_n=128, gfx_arch=gfx_arch))
    # fp8 preshufflequant
    cfgs.append(default_fp8_preshufflequant_config(bquant_group_n=1, gfx_arch=gfx_arch))
    cfgs.append(default_fp8_preshufflequant_config(bquant_group_n=128, gfx_arch=gfx_arch))
    # fp8 preshuffleb
    cfgs.append(default_fp8_preshuffleb_config(bquant_group_n=1, gfx_arch=gfx_arch))
    cfgs.append(default_fp8_preshuffleb_config(bquant_group_n=128, gfx_arch=gfx_arch))
    # fp8 preshuffleb + preshufflequant
    cfgs.append(default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=1, gfx_arch=gfx_arch))
    cfgs.append(default_fp8_preshuffleb_preshufflequant_config(bquant_group_n=128, gfx_arch=gfx_arch))
    # bf8 non-preshuffle
    cfgs.append(default_bf8_config(bquant_group_n=1, gfx_arch=gfx_arch))
    cfgs.append(default_bf8_config(bquant_group_n=128, gfx_arch=gfx_arch))
    # bf8 preshuffleb
    cfgs.append(default_bf8_preshuffleb_config(bquant_group_n=1, gfx_arch=gfx_arch))
    cfgs.append(default_bf8_preshuffleb_config(bquant_group_n=128, gfx_arch=gfx_arch))
    # fp4 non-preshuffle + preshuffleb (only 128)
    cfgs.append(default_fp4_config(gfx_arch=gfx_arch))
    cfgs.append(default_fp4_preshuffleb_config(gfx_arch=gfx_arch))
    return cfgs


# =============================================================================
# Self-test / default-config runner (no GPU required for --list / --codegen)
# =============================================================================


def _self_test(gfx_arch: str, do_build: bool) -> int:
    """Codegen (and optionally build) every default config; report name parity."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    configs = all_default_configs(gfx_arch=gfx_arch)

    log.info("ABQuant default-config matrix (%d kernels):", len(configs))
    for c in configs:
        log.info("  %s", c.name)

    # Codegen-only parity check: generate each header and confirm the emitted
    # KERNEL_NAME string matches config.name byte-for-byte.
    tmp = Path(tempfile.mkdtemp(prefix="abquant_selftest_"))
    ok = 0
    for c in configs:
        hpp = _generate_abquant_kernel(c, tmp)
        if hpp is None:
            log.error("  CODEGEN FAILED: %s", c.name)
            continue
        text = hpp.read_text()
        if f'KERNEL_NAME = "{c.name}"' in text:
            ok += 1
        else:
            log.error("  NAME MISMATCH in generated header for %s", c.name)
    log.info("Codegen name parity: %d / %d headers match", ok, len(configs))

    if not do_build:
        return 0 if ok == len(configs) else 1

    results = setup_multiple_abquant_dispatchers(configs, gfx_arch=gfx_arch)
    built = sum(1 for r in results if r is not None)
    log.info("Built %d / %d ABQuant .so files", built, len(configs))
    return 0 if built == len(configs) else 1


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="ABQuant dispatcher self-test / default-config runner")
    parser.add_argument("--gfx", default="gfx950", help="Target GFX arch (gfx942|gfx950)")
    parser.add_argument("--build", action="store_true",
                        help="Also compile each config with hipcc (needs a build env)")
    parser.add_argument("--list", action="store_true",
                        help="Only list the default-config kernel names and exit")
    args = parser.parse_args()

    if args.list:
        for c in all_default_configs(gfx_arch=args.gfx):
            print(c.name)
        return 0

    return _self_test(args.gfx, do_build=args.build)


if __name__ == "__main__":
    raise SystemExit(main())
