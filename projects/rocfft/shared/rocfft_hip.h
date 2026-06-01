// Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef __ROCFFT_HIP_H__
#define __ROCFFT_HIP_H__

#include <atomic>
#include <cstddef>
#include <hip/hip_runtime_api.h>
#include <stdexcept>
#include <string>

// exception for hip runtime error(s) specifically
struct hip_runtime_error : public std::runtime_error
{
    const hipError_t hip_error;
    hip_runtime_error(const std::string& info, hipError_t hip_status)
        : std::runtime_error::runtime_error(info)
        , hip_error(hip_status)
    {
        n_hip_failures++;
    }

    static inline size_t get_count()
    {
        return n_hip_failures;
    }

private:
    static inline std::atomic<size_t> n_hip_failures = 0;
};

class rocfft_scoped_device
{
public:
    static int device_count()
    {
        static int count = 0;
        if(count == 0)
        {
            const auto ret = hipGetDeviceCount(&count);
            if(ret != hipSuccess)
                throw hip_runtime_error("failed to get device count", ret);
        }
        return count;
    }

    rocfft_scoped_device(int device)
    {
        auto ret = hipGetDevice(&orig_device);
        if(ret != hipSuccess)
            throw hip_runtime_error("hipGetDevice failure", ret);

        ret = hipSetDevice(device);
        if(ret != hipSuccess)
            throw hip_runtime_error("hipSetDevice failure", ret);
    }
    ~rocfft_scoped_device()
    {
        (void)hipSetDevice(orig_device);
    }

    // not copyable or movable
    rocfft_scoped_device(const rocfft_scoped_device&) = delete;
    rocfft_scoped_device(rocfft_scoped_device&&)      = delete;
    rocfft_scoped_device& operator=(const rocfft_scoped_device&) = delete;

private:
    int orig_device;
};

#endif // __ROCFFT_HIP_H__
