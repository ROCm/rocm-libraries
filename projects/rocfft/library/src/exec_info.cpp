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
#include "../../shared/rocfft_hip.h"
#include "logging.h"

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
    : user_info(user_info)
{
    if(user_info)
    {
        execWorkBuffers.resize(user_info->workBuffers.size());
    }
    else
    {
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess)
            throw std::runtime_error("failed to get device count");
        execWorkBuffers.resize(deviceCount);
    }
}

void rocfft_execution_info_internal::ensure_work_buffer_size(
    const std::vector<size_t>& sizes_bytes_per_device)
{
    // assert that all our vectors are the same length
    if(execWorkBuffers.size() != sizes_bytes_per_device.size()
       || (user_info && execWorkBuffers.size() != user_info->workBuffers.size()))
        throw std::logic_error("unexpected work buf lengths");

    // go over each device's memory requirements
    for(size_t device = 0; device < execWorkBuffers.size(); ++device)
    {
        const size_t size_bytes = sizes_bytes_per_device[device];
        if(size_bytes == 0)
            continue;

        // check if user gave us a buffer
        if(user_info)
        {
            auto& buf = user_info->workBuffers[device];
            if(buf)
            {
                if(buf.size() >= size_bytes)
                {
                    continue;
                }
                else
                {
                    // not big enough
                    if(LOG_TRACE_ENABLED())
                        (*LogSingleton::GetInstance().GetTraceOS())
                            << "user work buffer too small" << std::endl;
                    throw rocfft_status_invalid_work_buffer;
                }
            }
        }

        // no user work buffer was specified, allocate one
        rocfft_scoped_device dev(device);
        if(execWorkBuffers[device].alloc(size_bytes) != hipSuccess)
            throw std::runtime_error("work buffer allocation failure");
    }
}

const gpubuf& rocfft_execution_info_internal::get_work_buffer(int device) const
{
    if(user_info && user_info->workBuffers[device])
        return user_info->workBuffers[device];
    return execWorkBuffers[device];
}

void** rocfft_execution_info_internal::get_load_cb_fns() const
{
    if(user_info)
        return user_info->load_cb_fns;
    return nullptr;
}

void** rocfft_execution_info_internal::get_load_cb_data() const
{
    if(user_info)
        return user_info->load_cb_data;
    return nullptr;
}

size_t rocfft_execution_info_internal::get_load_cb_lds_bytes() const
{
    if(user_info)
        return user_info->load_cb_lds_bytes;
    return 0;
}

void** rocfft_execution_info_internal::get_store_cb_fns() const
{
    if(user_info)
        return user_info->store_cb_fns;
    return nullptr;
}

void** rocfft_execution_info_internal::get_store_cb_data() const
{
    if(user_info)
        return user_info->store_cb_data;
    return nullptr;
}

size_t rocfft_execution_info_internal::get_store_cb_lds_bytes() const
{
    if(user_info)
        return user_info->store_cb_lds_bytes;
    return 0;
}

hipStream_t rocfft_execution_info_internal::get_user_stream(int device) const
{
    if(user_info)
        return user_info->rocfft_streams[device];
    return nullptr;
}
