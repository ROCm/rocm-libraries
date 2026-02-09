/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/
#pragma once

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <stdio.h>

#include "../hiptensor_types.h"

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(expression)                   \
    {                                                 \
        hipError_t status = (expression);             \
        if(status != hipSuccess)                      \
        {                                             \
            fprintf(stderr,                           \
                    "hip error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(status),        \
                    status,                           \
                    __FILE__,                         \
                    __LINE__);                        \
            exit(EXIT_FAILURE);                       \
        }                                             \
    }
#endif

#ifndef CHECK_HIPTENSOR_ERROR
#define CHECK_HIPTENSOR_ERROR(expression)                   \
    {                                                       \
        hiptensorStatus_t status = (expression);            \
        if(status != HIPTENSOR_STATUS_SUCCESS)              \
        {                                                   \
            fprintf(stderr,                                 \
                    "hipTensor error: '%s'(%d) at %s:%d\n", \
                    hiptensorGetErrorString(status),        \
                    status,                                 \
                    __FILE__,                               \
                    __LINE__);                              \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    }
#endif
