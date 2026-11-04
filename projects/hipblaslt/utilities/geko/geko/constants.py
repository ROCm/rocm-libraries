# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data type definitions and field mappings for GEMM operations.

Defines constants for:
- Data type mappings between hipBLASLt and Tensile formats.
- GEMM operation field definitions and categorizations.
- Log file field specifications.
- Index type mappings for different data formats.
- Gfx-style ``ARCH`` strings for YAML / tuning config (``SUPPORTED_ARCH``), the
  retired spellings still accepted for them, and the runtime environment an
  architecture needs.

These constants ensure consistent data handling across the optimization workflow.
"""

import logging

DTYPE = {
    "bf16_r": "B",
    "f16_r": "H",
    "f32_r": "S",
    "f64_r": "D",
    "f8_r": "F8",
    "f8_fnuz_r": "F8N",
    "bf8_r": "B8",
    "xf32_r": "X",
    "f4_r": "F4",
    "f32_c": "C",
    "f64_c": "Z",
    # I8
}

# Bytes per element for hipBLASLt dtype tokens.
DTYPE_BYTES = {
    "f64_r": 8, "f32_r": 4, "f16_r": 2, "bf16_r": 2,
    "f8_r": 1, "bf8_r": 1, "i8_r": 1, "i32_r": 4,
    "xf32_r": 4, "f4_r": 1,
    "f32_c": 8, "f64_c": 16,
}

GEMM_FIELDS = (
    "transA",
    "transB",
    "batch_count",
    "m",
    "n",
    "k",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
)

GEMM_LOG_FIELDS = (
    "transA",
    "transB",
    "batch_count",
    "M",
    "N",
    "K",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "compute_type",
    "scaleA",
    "scaleB",
)
GEMM_TYPE_FIELDS = (
    "transA",
    "transB",
    "a_type",
    "b_type",
    "c_type",
    "compute_type",
)

LOG_FIELDS = (
    "function",
    "M",
    "N",
    "K",
    "lda",
    "ldb",
    "ldc",
    "ldd",
    "stride_a",
    "stride_b",
    "stride_c",
    "stride_d",
    "alpha",
    "beta",
    "transA",
    "transB",
    "batch_count",
    "scaleA",
    "scaleB",
    "scaleC",
    "scaleD",
    "swizzleA",
    "swizzleB",
    "scaleAlpha_vector",
    "gradient",
    "use_e",
    "bias_vector",
    "bias_source",
    "a_type",
    "b_type",
    "c_type",
    "d_type",
    "scale_type",
    "bias_type",
    "aux_type",
    "compute_type",
    "activation_type",
    "flush",
    "any_stride",
    "rotating",
    "cold_iters",
    "iters",
    "solution_index",
    "solution_Name",
    "kernel_name",
    "call_count",
)


INDEX_TYPE_MAP = {
    0: "f32_r",
    1: "f64_r",
    2: "f32_c",
    3: "f64_c",
    4: "f16_r",
    5: "i8_r",
    6: "i32_r",
    7: "bf16_r",
    8: "i8_r",
    9: "i64_r",
    10: "xf32",
    11: "f8_r",
    12: "bf8_r",
    13: "f8b8",
    14: "b8f8",
    15: "f8_r",
    16: "bf8_r",
    17: "f8b8",
    18: "b8f8",
    19: "f6_r",
    20: "bf6_r",
    21: "f4_r",
}

# hipblaslt_scaling_format values (hipBLASLt clients/common/include/
# hipblaslt_scaling_format.hpp) for MX block scaling, keyed by (block size,
# Tensile scale DataType). A block-scaled problem benchmarked without its value
# resolves to an unscaled kernel, so fp4 finds no solution at all and fp8
# silently measures the wrong one. 0, 1 and 2 are none, Scalar and Vector.
MX_SCALING_FORMAT = {
    (32, "E8"): 3,    # Block_32_UE8M0
    (16, "E8"): 4,    # Block_16_UE8M0
    (32, "F8"): 5,    # Block_32_UE4M3
    (16, "F8"): 6,    # Block_16_UE4M3
    (32, "E5M3"): 7,  # Block_32_UE5M3
    (16, "E5M3"): 8,  # Block_16_UE5M3
}
# Block_32_UE8M0_32_8_EXT: E8 scales on 32-element blocks, pre-swizzled into
# 32x8 tiles. hipBLASLt maps it to the same Tensile problem as 3.
MX_SCALING_FORMAT_PRESWIZZLED = 1001
# MX scale DataTypes as library logic stores them (Tensile DataTypeEnum values).
MX_SCALE_DATATYPE_ENUM = {22: "E8", 15: "F8", 23: "E5M3"}

PERF_FIELDS = (
    "hipblaslt-Gflops",
    "hipblaslt-GB/s",
    "us",
)

# --- gfx-style ``ARCH`` strings (YAML + ``geko.optim.config.get_config``) ---
# Must match keys in ``geko.config_generator.constants.HARDWARE_MAP`` / ``_ARCH_SPECS``.
SUPPORTED_ARCH: tuple[str, ...] = (
    "gfx950",
    "gfx950_128cu",
    "gfx942",
    "gfx942_80cu",
    "gfx942_38cu",
    "gfx942_20cu",
    "gfx942_228cu",
    "gfx1250",
    "gfx1250_96cu",
    "gfx1250_192cu",
    "gfx1250-strict",
    "gfx1250-strict_96cu",
    "gfx1250-strict_192cu",
)

# Retired ARCH spellings, accepted with a deprecation warning. hipBLASLt rejects
# ``gfx1250v0`` everywhere a user can name it; the A0 stepping is the
# ``gfx1250-strict`` compiler target.
LEGACY_ARCH_ALIASES: dict[str, str] = {
    "gfx1250v0": "gfx1250-strict",
    "gfx1250v0_96cu": "gfx1250-strict_96cu",
    "gfx1250v0_192cu": "gfx1250-strict_192cu",
}


def canonical_arch(arch: str) -> str:
    """Return the supported spelling of an ``ARCH`` string.

    Args:
        arch: ARCH as given on the command line or in an input config.

    Returns:
        ``arch`` itself, or its replacement when it is a retired alias, in which
        case a deprecation warning is logged.
    """
    replacement = LEGACY_ARCH_ALIASES.get(arch)
    if replacement is None:
        return arch
    logging.getLogger("GEKO").warning(
        f"ARCH '{arch}' is deprecated; using '{replacement}'. Update the input config "
        f"or command line to '{replacement}'."
    )
    return replacement


# On gfx1250 A0 the HSA runtime reports the device as gfx1250, and loads only
# gfx1250 code objects, unless its strict mode is on. Tuning gfx1250-strict
# kernels therefore needs strict mode for every process that touches the GPU.
STRICT_RUNTIME_ENV: dict[str, str] = {"HSA_DISABLE_GFX12_STRICT": "0"}


def runtime_env(architecture: str | None) -> dict[str, str]:
    """Environment variables a tuning process needs on a Tensile architecture.

    Args:
        architecture: The LibraryLogic ``ArchitectureName`` of the configs being
            tuned (a compiler target such as ``gfx1250-strict``), or None.

    Returns:
        Variables to add to the process environment; empty when the
        architecture needs none.
    """
    return dict(STRICT_RUNTIME_ENV) if architecture == "gfx1250-strict" else {}
