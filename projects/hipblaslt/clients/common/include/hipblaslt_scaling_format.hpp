// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Lightweight scaling-format enum for client code paths that must not pull in
// hipblaslt_ostream.hpp (e.g. hipblaslt-mxdatagen compiled with -x hip).

typedef enum class _hipblaslt_scaling_format
{
    none                    = 0,
    Scalar                  = 1,
    Vector                  = 2,
    Block_32_UE8M0          = 3,
    Block_16_UE8M0          = 4,
    Block_32_UE4M3          = 5,
    Block_16_UE4M3          = 6,
    Block_32_UE5M3          = 7,
    Block_16_UE5M3          = 8,
    Block_32_UE8M0_32_8_EXT = 1001,
    // w4a16 group scales; values equal HIPBLASLT_MATMUL_MATRIX_SCALE_VEC*_{16BF,16F}[_ZP]_EXT.
    // Not MX, so isBlockScaling() is false. Scale type must equal B's.
    Block_32_BF16           = 1006,
    Block_64_BF16           = 1007,
    Block_128_BF16          = 1008,
    Block_32_BF16_ZP        = 1009,
    Block_64_BF16_ZP        = 1010,
    Block_128_BF16_ZP       = 1011,
    Block_32_F16            = 1012,
    Block_64_F16            = 1013,
    Block_128_F16           = 1014,
    Block_32_F16_ZP         = 1015,
    Block_64_F16_ZP         = 1016,
    Block_128_F16_ZP        = 1017,
} hipblaslt_scaling_format;
