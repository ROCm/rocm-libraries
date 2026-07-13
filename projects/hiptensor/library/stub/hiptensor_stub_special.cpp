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

// Hand-written part of the hipTensor stub.
//
// The bulk of the stub -- every public API function returning hiptensorStatus_t,
// which simply returns HIPTENSOR_STATUS_NOT_SUPPORTED -- is generated from the
// public header by generate_api_stub.py (--mode stub). This file holds the
// few informational helpers that carry real behavior even in the stub build and
// therefore cannot be generated. If you add a public API with a non-status
// return type, implement it here.

#include <hiptensor/hiptensor.h>

const char* hiptensorGetErrorString(const hiptensorStatus_t error)
{
    switch(error)
    {
    case HIPTENSOR_STATUS_SUCCESS:
        return "HIPTENSOR_STATUS_SUCCESS";
    case HIPTENSOR_STATUS_NOT_INITIALIZED:
        return "HIPTENSOR_STATUS_NOT_INITIALIZED";
    case HIPTENSOR_STATUS_ALLOC_FAILED:
        return "HIPTENSOR_STATUS_ALLOC_FAILED";
    case HIPTENSOR_STATUS_INVALID_VALUE:
        return "HIPTENSOR_STATUS_INVALID_VALUE";
    case HIPTENSOR_STATUS_ARCH_MISMATCH:
        return "HIPTENSOR_STATUS_ARCH_MISMATCH";
    case HIPTENSOR_STATUS_EXECUTION_FAILED:
        return "HIPTENSOR_STATUS_EXECUTION_FAILED";
    case HIPTENSOR_STATUS_INTERNAL_ERROR:
        return "HIPTENSOR_STATUS_INTERNAL_ERROR";
    case HIPTENSOR_STATUS_NOT_SUPPORTED:
        return "HIPTENSOR_STATUS_NOT_SUPPORTED";
    case HIPTENSOR_STATUS_CK_ERROR:
        return "HIPTENSOR_STATUS_CK_ERROR";
    case HIPTENSOR_STATUS_HIP_ERROR:
        return "HIPTENSOR_STATUS_HIP_ERROR";
    case HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE:
        return "HIPTENSOR_STATUS_INSUFFICIENT_WORKSPACE";
    case HIPTENSOR_STATUS_INSUFFICIENT_DRIVER:
        return "HIPTENSOR_STATUS_INSUFFICIENT_DRIVER";
    case HIPTENSOR_STATUS_IO_ERROR:
        return "HIPTENSOR_STATUS_IO_ERROR";
    default:
        return "HIPTENSOR_STATUS_UNKNOWN";
    }
}

int hiptensorGetHiprtVersion()
{
    return -1;
}

size_t hiptensorGetVersion()
{
    return HIPTENSOR_MAJOR_VERSION * 1e6 + HIPTENSOR_MINOR_VERSION * 1e3 + HIPTENSOR_PATCH_VERSION;
}
