// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define EXECUTION_SPECIFIER __device__

EXECUTION_SPECIFIER float bfloat16_to_float(__bf16 src_val)
{
    return static_cast<float>(src_val);
}

EXECUTION_SPECIFIER __bf16 float_to_bfloat16(float src_val)
{
    return static_cast<__bf16>(src_val);
}

#ifdef __cplusplus
}
#endif
