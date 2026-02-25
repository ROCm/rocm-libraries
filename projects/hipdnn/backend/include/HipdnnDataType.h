// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

typedef enum
{
    HIPDNN_DATA_FLOAT = 0,
    HIPDNN_DATA_DOUBLE,
    HIPDNN_DATA_HALF,
    HIPDNN_DATA_INT8,
    HIPDNN_DATA_INT32,
    HIPDNN_DATA_UINT8,
    HIPDNN_DATA_BFLOAT16,
    HIPDNN_DATA_FP8_E4M3,
    HIPDNN_DATA_FP8_E5M2,
} hipdnnDataType_t;
