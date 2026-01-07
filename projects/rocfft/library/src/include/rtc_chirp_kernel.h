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

#ifndef ROCFFT_RTC_CHIRP_KERNEL_H
#define ROCFFT_RTC_CHIRP_KERNEL_H

#include "rtc_chirp_gen.h"
#include "rtc_kernel.h"

struct RTCKernelChirp : public RTCKernel
{
    // generate chirp kernel from precision
    static RTCKernelChirp generate(const std::string& gpu_arch, rocfft_precision precision);

    // no DeviceCallIn is available at chirp generation time -
    // these kernels are launched without it
    RTCKernelArgs get_launch_args(DeviceCallIn& data) override
    {
        return {};
    }

protected:
    RTCKernelChirp(const std::string&       kernel_name,
                   const std::vector<char>& code,
                   dim3                     gridDim,
                   dim3                     blockDim)
        : RTCKernel(kernel_name, code, gridDim, blockDim)
    {
    }
};
#endif // ROCFFT_RTC_CHIRP_KERNEL_H
