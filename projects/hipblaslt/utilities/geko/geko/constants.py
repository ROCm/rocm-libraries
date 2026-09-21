# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Data type definitions and field mappings for GEMM operations.

Defines constants for:
- Data type mappings between hipBLASLt and Tensile formats.
- GEMM operation field definitions and categorizations.
- Log file field specifications.
- Index type mappings for different data formats.
- Gfx-style ``ARCH`` strings for YAML / tuning config (``SUPPORTED_ARCH``).

These constants ensure consistent data handling across the optimization workflow.
"""

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
}

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
)
