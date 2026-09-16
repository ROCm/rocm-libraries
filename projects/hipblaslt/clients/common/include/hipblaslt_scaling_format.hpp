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
    // w4a16 group scales for an int4 A. The values match the public
    // HIPBLASLT_MATMUL_MATRIX_SCALE_VEC{32,64,128}[_ZP]_EXT enumerators so
    // --scaleA takes the same number the API does. Unlike the MX formats above
    // these are ordinary signed 16-bit floats consumed in the main loop, so
    // isBlockScaling() (which gates the MX data generator) stays false for them.
    // The scale element type is B's, so it is not part of the mode.
    Block_32                = 1006,
    Block_64                = 1007,
    Block_128               = 1008,
    Block_32_ZP             = 1009,
    Block_64_ZP             = 1010,
    Block_128_ZP            = 1011,
} hipblaslt_scaling_format;
