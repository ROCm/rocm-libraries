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

#include "exec_info.h"

rocfft_execution_info_t::rocfft_execution_info_t()
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess)
        throw std::runtime_error("failed to get device count");
    workBuffers.resize(deviceCount);
    rocfft_streams.resize(deviceCount);
}

rocfft_execution_info_internal::rocfft_execution_info_internal(
    const rocfft_execution_info_t* user_info)
{
    if(!user_info)
    {
        // base class was already constructed so work/stream vectors
        // are already the right size, nothing to do
        return;
    }

    this->load_cb_fns        = user_info->load_cb_fns;
    this->load_cb_data       = user_info->load_cb_data;
    this->load_cb_lds_bytes  = user_info->load_cb_lds_bytes;
    this->store_cb_fns       = user_info->store_cb_fns;
    this->store_cb_data      = user_info->store_cb_data;
    this->store_cb_lds_bytes = user_info->store_cb_lds_bytes;
    init_nonowned_work_buffers(*user_info);

    // streams are not owned anyway, so just copy
    rocfft_streams = user_info->rocfft_streams;
}

// delegate to base class copy ctor before handling any
// internal-specific members
rocfft_execution_info_internal::rocfft_execution_info_internal(
    const rocfft_execution_info_internal& other)
    : rocfft_execution_info_internal(static_cast<const rocfft_execution_info_t*>(&other))

{
}

void rocfft_execution_info_internal::init_nonowned_work_buffers(
    const rocfft_execution_info_t& other)
{
    // Make non-owning copies of the pointers.  "other" is expected
    // to live at least as long as this.
    for(size_t i = 0; i < this->workBuffers.size(); ++i)
    {
        this->workBuffers[i]
            = gpubuf::make_nonowned(other.workBuffers[i].data(), other.workBuffers[i].size());
    }
}
