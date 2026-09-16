// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Internal mirror of core/scaled_wmma.py: matrix and scale selectors. */
#ifndef ROCKE_SCALED_WMMA_INTERNAL_H
#define ROCKE_SCALED_WMMA_INTERNAL_H

#include <stdio.h>
#include <string.h>

static inline int rocke_wmma_matrix_format(const char* dtype)
{
    const char* names[] = {"fp8", "bf8", "fp6", "bf6", "fp4"};
    for(int i = 0; i < 5; ++i)
    {
        if(strcmp(dtype, names[i]) == 0)
        {
            return i;
        }
    }
    return -1;
}

static inline bool rocke_wmma_scaled_formats(const char* op_id, bool* scale16, int* a, int* b)
{
    const char* prefix = "wmma_scale_f32_16x16x128_";
    *scale16 = strncmp(op_id, "wmma_scale16_", 13) == 0;
    if(*scale16)
    {
        prefix = "wmma_scale16_f32_16x16x128_";
    }
    if(strncmp(op_id, prefix, strlen(prefix)) != 0)
    {
        return false;
    }
    char da[8], db[8];
    int end = 0;
    const char* suffix = op_id + strlen(prefix);
    if(sscanf(suffix, "%7[^_]_%7s%n", da, db, &end) != 2 || suffix[end] != '\0')
    {
        return false;
    }
    *a = rocke_wmma_matrix_format(da);
    *b = rocke_wmma_matrix_format(db);
    return *a >= 0 && *b >= 0;
}

static inline int rocke_wmma_scale_format(const char* dtype)
{
    if(!dtype || strcmp(dtype, "e8m0") == 0 || strcmp(dtype, "i8") == 0)
    {
        return 0;
    }
    if(strcmp(dtype, "e5m3") == 0)
    {
        return 1;
    }
    return strcmp(dtype, "e4m3") == 0 ? 2 : -1;
}

static inline const char* rocke_wmma_scale_error(int a, int b, int sa, int sb)
{
    if(sa < 0 || sb < 0)
    {
        return "scaled WMMA scale types must be e8m0, e5m3, or e4m3";
    }
    if((a != 4 && sa != 0) || (b != 4 && sb != 0))
    {
        return "scaled WMMA e5m3/e4m3 scales require an FP4 operand";
    }
    if(a == 4 && b == 4 && sa != sb)
    {
        return "scaled WMMA FP4 x FP4 requires matching scale formats";
    }
    return NULL;
}

#endif
