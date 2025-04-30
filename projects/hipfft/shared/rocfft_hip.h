// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime_api.h>
#include <stdexcept>

class rocfft_scoped_device
{
public:
    rocfft_scoped_device(int device)
    {
        if(hipGetDevice(&orig_device) != hipSuccess)
            throw std::runtime_error("hipGetDevice failure");

        if(hipSetDevice(device) != hipSuccess)
            throw std::runtime_error("hipSetDevice failure");
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
