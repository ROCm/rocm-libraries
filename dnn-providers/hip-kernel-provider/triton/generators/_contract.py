# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generator <-> pack contract for the Triton RMSNorm AOT objects.

Pure data; stdlib only. TritonRmsNormNative.cpp restates the symbol and field names;
keep them in sync.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --- Names ---------------------------------------------------------------
ENGINE_NAME = "hipkernel:TritonRmsNorm"  # TRITON_RMSNORM_ENGINE_NAME
FILE_STEM = "triton_rmsnorm"
UED_FILE = f"{FILE_STEM}.ued.json"
UHD_FILE = f"{FILE_STEM}.uhd.json"
UDD_FILE = f"{FILE_STEM}.udd.json"
KMD_FILE = f"{FILE_STEM}.kmd.json"
UMD_FILE = f"{FILE_STEM}_kernel_match.umd.json"
KDP_FILE = f"{FILE_STEM}.kdp.json"
MANIFEST_FILE = "manifest.json"

SYMBOL_GRAPH_MATCH = "hipkernel.triton_rmsnorm.graph_match"
SYMBOL_KERNEL_MATCH = "hipkernel.triton_rmsnorm.kernel_match"
SYMBOL_SCORE = "hipkernel.triton_rmsnorm.score"
SYMBOL_DISPATCH = "hipkernel.triton_rmsnorm.dispatch"

FAMILY_FOLDER = "Triton"

# --- UUIDv5 and layout ---------------------------------------------------
NAMESPACE = uuid.uuid5(
    uuid.NAMESPACE_URL,
    "https://github.com/ROCm/rocm-libraries/dnn-providers/hip-kernel-provider/triton",
)

OP = "rmsnorm"
ARCHES: Tuple[str, ...] = ("gfx942", "gfx950")

ID_NAME_KMD = "kmd/rmsnorm"
ID_NAME_UHD = "uhd/rmsnorm"
ID_NAME_UDD = "udd/rmsnorm"
ID_NAME_UED = "ued/rmsnorm"
ID_NAME_UMD = "umd/rmsnorm/kernel"


def id_name_kdp(arch: str) -> str:
    return f"kdp/rmsnorm/{arch}"


def id_name_ukd(arch: str, instance: str) -> str:
    return f"ukd/rmsnorm/{arch}/{instance}"


def make_id(name: str) -> uuid.UUID:
    return uuid.uuid5(NAMESPACE, name)


def ukd_file(instance: str) -> str:
    return f"{instance}.ukd.json"


def hsaco_file(instance: str) -> str:
    return f"{instance}.hsaco"


# --- Instances -----------------------------------------------------------
DTYPES: Tuple[str, ...] = ("bf16", "f16")


@dataclass(frozen=True)
class BlockConfig:
    block_n: int
    num_warps: int
    block_threads: int  # num_warps * 64
    priority: int  # UKD priority, which the pack uses as its score


BLOCK_CONFIGS: Tuple[BlockConfig, ...] = (
    BlockConfig(1024, 4, 256, 400),
    BlockConfig(2048, 4, 256, 300),
    BlockConfig(4096, 8, 512, 200),
    BlockConfig(8192, 8, 512, 100),
)


@dataclass(frozen=True)
class Instance:
    name: str
    dtype: str
    block_n: int
    num_warps: int
    block_threads: int
    priority: int


def instance_name(dtype: str, block_n: int) -> str:
    return f"rmsnorm_{dtype}_bn{block_n}"


def instances() -> Tuple[Instance, ...]:
    """The 8 instances built for every arch (dtype-major, then BLOCK_N)."""
    return tuple(
        Instance(
            instance_name(dtype, c.block_n),
            dtype,
            c.block_n,
            c.num_warps,
            c.block_threads,
            c.priority,
        )
        for dtype in DTYPES
        for c in BLOCK_CONFIGS
    )


# --- Kernel --------------------------------------------------------------
KERNEL_SYMBOL = "rmsnorm_kernel"
KERNEL_SOURCE_FILE = "kernels_src/rmsnorm.py"
TRITON_ELEMENT_TYPE: Dict[str, str] = {"bf16": "bf16", "f16": "fp16"}


def triton_signature(dtype: str) -> Dict[str, str]:
    t = TRITON_ELEMENT_TYPE[dtype]
    return {
        "x_ptr": f"*{t}",
        "scale_ptr": f"*{t}",
        "y_ptr": f"*{t}",
        "n_cols": "i32",
        "eps": "fp32",
        "BLOCK_N": "constexpr",
    }


def triton_constexprs(block_n: int) -> Dict[str, int]:
    return {"BLOCK_N": block_n}


def triton_attrs() -> Dict[Tuple[int], List[Tuple[str, int]]]:
    """tt.divisibility=16 on indices 0,1,2 only; none on n_cols (index 3).

    Triton 3.8.0 ASTSource attrs form: {(arg_index,): [(attr_name, value), ...]}.
    """
    return {
        (0,): [("tt.divisibility", 16)],
        (1,): [("tt.divisibility", 16)],
        (2,): [("tt.divisibility", 16)],
    }


# --- Kernarg ABI ---------------------------------------------------------
@dataclass(frozen=True)
class KernArg:
    name: str
    value_kind: str
    size: int
    offset: int


KERNARGS: Tuple[KernArg, ...] = (
    KernArg("x", "global_buffer", 8, 0),
    KernArg("scale", "global_buffer", 8, 8),
    KernArg("y", "global_buffer", 8, 16),
    KernArg("n_cols", "by_value", 4, 24),
    KernArg("eps", "by_value", 4, 28),
    KernArg("global_scratch", "global_buffer", 8, 32),
    KernArg("profile_scratch", "global_buffer", 8, 40),
)
MIN_KERNARG_SEGMENT_SIZE = 48

# --- KMD / descriptor fields --------------------------------------------
KMD_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("dtype", "string"),  # "bf16" / "f16"
    ("block_n", "int"),
    ("block_threads", "int"),
    ("shared_mem_bytes", "int"),
)
UED_KNOBS: Tuple = ()
UMD_SCOPE = "kernel"
UKD_SOURCE_KIND = "hsaco"

# --- Manifest ------------------------------------------------------------
MANIFEST_VERSION = 1
MANIFEST_OP = "rmsnorm"
MANIFEST_GRID_RULE = "one_program_per_row"

# --- Pins ----------------------------------------------------------------
TRITON_VERSION = "3.8.0"
TRITON_WHEEL_SHA256 = "e91ffa46d095b252248297292dd22bcbacd53a125a0c2eefbbbf74925a320bc3"
WARP_SIZE = 64  # GPUTarget("hip", arch, 64)
GPU_TARGET_BACKEND = "hip"

HIP_OPTIONS_FIXED: Dict[str, object] = {
    "num_stages": 2,
    "waves_per_eu": 0,
    "enable_fp_fusion": True,
    "allow_flush_denorm": False,
    "matrix_instr_nonkdim": 0,
    "kpack": 1,
    "num_ctas": 1,
    "launch_cooperative_grid": False,
}


def hip_options(num_warps: int) -> Dict[str, object]:
    """Full explicit HIPOptions kwargs for one instance (num_warps from the instance table)."""
    return {"num_warps": num_warps, **HIP_OPTIONS_FIXED}


# --- Error codes ---------------------------------------------------------
ERR_ENV = "TRITONGEN-ENV"
ERR_ARCH = "TRITONGEN-ARCH"
ERR_ABI = "TRITONGEN-ABI"
ERR_SCRATCH = "TRITONGEN-SCRATCH"
ERR_ERRATUM = "TRITONGEN-ERRATUM"
ERR_COMPILE = "TRITONGEN-COMPILE"
ERR_INPUT = "TRITONGEN-INPUT"
ERR_INTERNAL = "TRITONGEN-INTERNAL"
