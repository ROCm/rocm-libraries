// Copyright (C) 2016 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "../../../shared/rocfft_hip.h"

#include "callback_map.h"

struct rocfft_execution_info_t
{
    void*       workBuffer;
    size_t      workBufferSize;
    hipStream_t rocfft_stream = 0; // by default it is stream 0
    rocfft_execution_info_t()
        : workBuffer(nullptr)
        , workBufferSize(0)
    {
    }
    // User-supplied load/store callback function pointers and data.
    // If specified, there is one function+data per brick in the
    // input/output.
    void** load_cb_fns        = nullptr;
    void** load_cb_data       = nullptr;
    size_t load_cb_lds_bytes  = 0;
    void** store_cb_fns       = nullptr;
    void** store_cb_data      = nullptr;
    size_t store_cb_lds_bytes = 0;
};

void TransformPowX(const ExecPlan&                         execPlan,
                   void*                                   in_buffer[],
                   void*                                   out_buffer[],
                   rocfft_execution_info                   info,
                   size_t                                  multiPlanIdx,
                   const std::map<int, device_callback_t>& callbacks);

#endif // TRANSFORM_H
