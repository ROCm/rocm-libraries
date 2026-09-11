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
} hipblaslt_scaling_format;
