/*! \file */
/* ************************************************************************
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights Reserved.
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
 * ************************************************************************ */

#pragma once

#include <hip/hip_runtime_api.h>

//
// This section is conditional to the definition
// of ROCSPARSE_WITH_MEMSTAT
//
#ifndef ROCSPARSE_WITH_MEMSTAT

#define rocsparse_hipMalloc(p_, nbytes_) hipMalloc(p_, nbytes_)
#define rocsparse_hipFree(p_) hipFree(p_)

// if hip version is atleast 5.3.0 hipMallocAsync and hipFreeAsync are defined
#if HIP_VERSION >= 50300000
#define rocsparse_hipMallocAsync(p_, nbytes_, stream_) hipMallocAsync(p_, nbytes_, stream_)
#define rocsparse_hipFreeAsync(p_, stream_) hipFreeAsync(p_, stream_)
#else
#define rocsparse_hipMallocAsync(p_, nbytes_, stream_) hipMalloc(p_, nbytes_)
#define rocsparse_hipFreeAsync(p_, stream_) hipFree(p_)
#endif

#define rocsparse_hipHostMalloc(p_, nbytes_) hipHostMalloc(p_, nbytes_)
#define rocsparse_hipHostFree(p_) hipHostFree(p_)

#define rocsparse_hipMallocManaged(p_, nbytes_) hipMallocManaged(p_, nbytes_)
#define rocsparse_hipFreeManaged(p_) hipFree(p_)

#else

#include "rocsparse-auxiliary.h"

#define ROCSPARSE_HIP_SOURCE_MSG(msg_) #msg_
#define ROCSPARSE_HIP_SOURCE_TAG(msg_) __FILE__ " " ROCSPARSE_HIP_SOURCE_MSG(msg_)

#define rocsparse_hipMalloc(p_, nbytes_) \
    rocsparse_hip_malloc((void**)(p_), (nbytes_), ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipFree(p_) rocsparse_hip_free((void**)(p_), ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipMallocAsync(p_, nbytes_, stream_) \
    rocsparse_hip_malloc_async((void**)(p_), (nbytes_), stream_, ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipFreeAsync(p_, stream_) \
    rocsparse_hip_free_async((void**)(p_), stream_, ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipHostMalloc(p_, nbytes_) \
    rocsparse_hip_host_malloc((void**)(p_), (nbytes_), ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipHostFree(p_) \
    rocsparse_hip_host_free((void**)(p_), ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipMallocManaged(p_, nbytes_) \
    rocsparse_hip_malloc_managed((void**)(p_), (nbytes_), ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#define rocsparse_hipFreeManaged(p_) \
    rocsparse_hip_free_managed((void**)(p_), ROCSPARSE_HIP_SOURCE_TAG(__LINE__))

#endif
