# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Op contract for the Triton RMSNorm AOT objects.

Pure data; stdlib only. ``TritonRmsNormNative.cpp`` restates the KMD field names and the
matcher symbols derived from this op's name; keep them in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .. import _family as F

OP = "rmsnorm"
ENGINE_NAME = F.engine_name("RmsNorm")
ARCHES: Tuple[str, ...] = ("gfx942", "gfx950")
UED_SDK_VERSION = "1.2.0"  # the plugin API version of hip-kernel-provider

# --- Kernel --------------------------------------------------------------------
KERNEL_SYMBOL = "rmsnorm_kernel"
KERNEL_SOURCE_FILE = "kernels_src/rmsnorm.py"
TRITON_ELEMENT_TYPE: Dict[str, str] = {"bf16": "bf16", "f16": "fp16"}
# tt.divisibility of x, scale and y; the operand alignment every instance requires.
ALIGNMENT = 16
ALIGNED_ARG_INDICES: Tuple[int, ...] = (0, 1, 2)
# Kernel parameter -> semantic kernarg name recorded in the manifest.
ARG_NAMES: Dict[str, str] = {
    "x_ptr": "x",
    "scale_ptr": "scale",
    "y_ptr": "y",
    "n_cols": "n_cols",
    "eps": "eps",
}

# --- Instances -----------------------------------------------------------------
DTYPES: Tuple[str, ...] = ("bf16", "f16")


@dataclass(frozen=True)
class BlockConfig:
    block_n: int
    block_threads: int
    priority: int  # UKD priority, which the pack uses as its score


BLOCK_CONFIGS: Tuple[BlockConfig, ...] = (
    BlockConfig(1024, 256, 400),
    BlockConfig(2048, 256, 300),
    BlockConfig(4096, 512, 200),
    BlockConfig(8192, 512, 100),
)


@dataclass(frozen=True)
class Instance:
    name: str
    dtype: str
    block_n: int
    block_threads: int
    priority: int

    @property
    def knobs(self) -> Dict[str, int]:
        return {"block_n": self.block_n, "block_threads": self.block_threads}

    @property
    def num_warps(self) -> int:
        return self.block_threads // F.WARP_SIZE

    def row(self) -> dict:
        """The instance-table row, as hashed into the manifest's ``inputs_sha256``."""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "priority": self.priority,
            "knobs": self.knobs,
        }


def instance_name(dtype: str, block_n: int) -> str:
    return f"rmsnorm_{dtype}_bn{block_n}"


def instances() -> Tuple[Instance, ...]:
    """The 8 instances built for every arch (dtype-major, then BLOCK_N)."""
    return tuple(
        Instance(
            instance_name(dtype, c.block_n),
            dtype,
            c.block_n,
            c.block_threads,
            c.priority,
        )
        for dtype in DTYPES
        for c in BLOCK_CONFIGS
    )


def triton_signature(instance: Instance) -> Dict[str, str]:
    t = TRITON_ELEMENT_TYPE[instance.dtype]
    return {
        "x_ptr": f"*{t}",
        "scale_ptr": f"*{t}",
        "y_ptr": f"*{t}",
        "n_cols": "i32",
        "eps": "fp32",
        "BLOCK_N": "constexpr",
    }


def triton_constexprs(instance: Instance) -> Dict[str, int]:
    return {"BLOCK_N": instance.block_n}


def triton_attrs(instance: Instance) -> Dict[Tuple[int], List[Tuple[str, int]]]:
    """tt.divisibility on x, scale and y only; none on n_cols.

    Triton 3.8.0 ASTSource attrs form: {(arg_index,): [(attr_name, value), ...]}.
    """
    return {(i,): [("tt.divisibility", ALIGNMENT)] for i in ALIGNED_ARG_INDICES}


def alignment(instance: Instance) -> int:
    """The operand alignment ``instance`` was compiled for: its ``tt.divisibility``."""
    values = {
        value
        for attrs in triton_attrs(instance).values()
        for key, value in attrs
        if key == "tt.divisibility"
    }
    if len(values) != 1:
        raise F.GeneratorError(
            f"{F.ERR_INTERNAL}: {instance.name}: tt.divisibility values {sorted(values)}"
        )
    return values.pop()


# --- Descriptors -----------------------------------------------------------------
KMD_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("dtype", "string"),  # "bf16" / "f16"
    ("block_n", "int"),
    ("block_threads", "int"),
    ("shared_mem_bytes", "int"),
    ("alignment", "int"),
)
UED_KNOBS: Tuple[str, ...] = ()
UMD_SCOPE = "kernel"


def metadata(record: dict) -> dict:
    """A UKD's ``metadata`` (one value per KMD field) from its manifest record."""
    instance = {i.name: i for i in instances()}[record["name"]]
    return {
        "dtype": record["dtype"],
        "block_n": record["knobs"]["block_n"],
        "block_threads": record["knobs"]["block_threads"],
        "shared_mem_bytes": record["dynamic_lds_bytes"],
        "alignment": alignment(instance),
    }


def record_problems(record: dict) -> List[str]:
    """Op-specific faults in a manifest record (the row reduction needs LDS)."""
    if record["dynamic_lds_bytes"] <= 0:
        return [f"dynamic_lds_bytes={record['dynamic_lds_bytes']}, expected > 0"]
    return []
