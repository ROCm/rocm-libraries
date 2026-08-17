// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stddef.h>
#include <stdint.h>

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
    const void* ptr;
    size_t size;
} hipdnnBackendFlatbufferData_t;

#ifdef __cplusplus
}
#endif
// NOLINTEND
