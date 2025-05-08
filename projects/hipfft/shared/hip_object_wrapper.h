/******************************************************************************
* Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef ROCFFT_HIP_OBJ_WRAPPER_H
#define ROCFFT_HIP_OBJ_WRAPPER_H

#include "rocfft_hip.h"

// RAII wrapper around HIP objects
template <typename T, auto TCreate, auto TDestroy>
struct hip_object_wrapper_t
{
    hip_object_wrapper_t()
        : obj(nullptr)
    {
    }

    void alloc()
    {
        if(obj == nullptr && TCreate(&obj) != hipSuccess)
            throw std::runtime_error("hip create failure");
    }

    void free()
    {
        if(obj)
        {
            (void)TDestroy(obj);
            obj = nullptr;
        }
    }

    operator const T&() const
    {
        return obj;
    }
    operator T&()
    {
        return obj;
    }

    operator bool() const
    {
        return obj != nullptr;
    }

    ~hip_object_wrapper_t()
    {
        free();
    }

    hip_object_wrapper_t(const hip_object_wrapper_t&) = delete;
    hip_object_wrapper_t& operator=(const hip_object_wrapper_t&) = delete;
    hip_object_wrapper_t(hip_object_wrapper_t&& other)
        : obj(other.obj)
    {
        other.obj = nullptr;
    }

private:
    T obj;
};

typedef hip_object_wrapper_t<hipStream_t, hipStreamCreate, hipStreamDestroy> hipStream_wrapper_t;
typedef hip_object_wrapper_t<hipEvent_t, hipEventCreate, hipEventDestroy>    hipEvent_wrapper_t;

#endif // ROCFFT_HIP_OBJ_WRAPPER_H
