// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_EXEC_INFO_H
#define ROCFFT_EXEC_INFO_H

#include "../../../shared/gpubuf.h"
#include <hip/hip_runtime_api.h>
#include <vector>

// User-specified execution info, to store details that the user can
// set at plan execution time.  This is really only intended to be
// used in public-facing APIs, and we should create an "internal"
// struct specified below soon after for internal use.
struct rocfft_execution_info_t
{
    rocfft_execution_info_t();

    // Vectors here have one element per visible HIP device

    // gpubufs are expected to be non-owned if they come from the
    // user, and owned if we allocate them at execute time.
    std::vector<gpubuf>      workBuffers;
    std::vector<hipStream_t> rocfft_streams;

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

// Internal execution info that we create after we know which plan we
// are executing.  Stores additional details that are only knowable
// by that time, and which the user can't directly specify.
struct rocfft_execution_info_internal : public rocfft_execution_info_t
{
    // construct from a user-specified info, which is expected to
    // live longer than this struct if specified
    rocfft_execution_info_internal(const rocfft_execution_info_t* user_info);

    // these can be copied, but we need to ensure that the original
    // continues to own any allocations
    rocfft_execution_info_internal(const rocfft_execution_info_internal& other);

private:
    // copy the work buffer pointers from another struct but make them
    // nonowned here
    void init_nonowned_work_buffers(const rocfft_execution_info_t& other);
};

#endif
