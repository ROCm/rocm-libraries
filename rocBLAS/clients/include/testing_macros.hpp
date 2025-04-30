/* ************************************************************************
 * Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 *
 * ************************************************************************/

#pragma once

#define UNWRAP_ARGS(...) __VA_ARGS__

// ALLOC refers to constructors here

#define HOST_MEMCHECK(T_, name_, args_) \
    T_ name_(UNWRAP_ARGS args_);        \
    CHECK_HIP_ERROR(name_.memcheck());

#define DEVICE_MEMCHECK(T_, name_, args_) \
    T_ name_(UNWRAP_ARGS args_);          \
    CHECK_DEVICE_ALLOCATION(name_.memcheck());

#define RETURN_IF_MEM_ERROR(T_, name_, args_) \
    T_ name_(UNWRAP_ARGS args_);              \
    RETURN_IF_HIP_ERROR(name_.memcheck());

// DAPI refers to dual API (original and ILP64 version ending in _64)

#define DAPI_DISPATCH(name_, args_)    \
    if(arg.api & c_API_64)             \
    {                                  \
        name_##_64(UNWRAP_ARGS args_); \
    }                                  \
    else                               \
    {                                  \
        name_(UNWRAP_ARGS args_);      \
    }

#define DAPI_EXPECT(val_, name_, args_)                               \
    if(arg.api & c_API_64)                                            \
    {                                                                 \
        EXPECT_ROCBLAS_STATUS((name_##_64(UNWRAP_ARGS args_)), val_); \
    }                                                                 \
    else                                                              \
    {                                                                 \
        EXPECT_ROCBLAS_STATUS((name_(UNWRAP_ARGS args_)), val_);      \
    }

#define DAPI_CHECK(name_, args_)                              \
    if(arg.api & c_API_64)                                    \
    {                                                         \
        CHECK_ROCBLAS_ERROR((name_##_64(UNWRAP_ARGS args_))); \
    }                                                         \
    else                                                      \
    {                                                         \
        CHECK_ROCBLAS_ERROR((name_(UNWRAP_ARGS args_)));      \
    }

#define DAPI_CHECK_ALLOC_QUERY(name_, args_)                \
    if(arg.api & c_API_64)                                  \
    {                                                       \
        CHECK_ALLOC_QUERY((name_##_64(UNWRAP_ARGS args_))); \
    }                                                       \
    else                                                    \
    {                                                       \
        CHECK_ALLOC_QUERY((name_(UNWRAP_ARGS args_)));      \
    }
