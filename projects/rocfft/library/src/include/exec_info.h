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
#include <cstddef>
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

    // gpubufs are non-owned as they come from the user
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
struct rocfft_execution_info_internal
{
    // construct from a user-specified info, which must live longer
    // than this struct if specified
    rocfft_execution_info_internal(const rocfft_execution_info_t* user_info);

    // Ensure that we have a work buffer of the specified size for
    // the specified device.  If the user specified one that's big
    // enough, use that.  If the user specified one that's not big
    // enough, throw invalid work buffer exception.  Otherwise
    // allocate one.
    void ensure_work_buffer_size(const std::vector<size_t>& sizes_bytes_per_device);
    // Get the work buffer for the specified device.
    const gpubuf& get_work_buffer(int device) const;

    // accessors for data that may be provided by user-specified exec info
    void** get_load_cb_fns() const;
    void** get_load_cb_data() const;
    size_t get_load_cb_lds_bytes() const;
    void** get_store_cb_fns() const;
    void** get_store_cb_data() const;
    size_t get_store_cb_lds_bytes() const;
    // get user-specified stream
    hipStream_t get_user_stream(int device) const;

private:
    // pointer to user-specified info - may be null if user never specified one
    const rocfft_execution_info_t* user_info = nullptr;

    // gpubufs that we own and allocate during execution
    std::vector<gpubuf> execWorkBuffers;
};

#endif
