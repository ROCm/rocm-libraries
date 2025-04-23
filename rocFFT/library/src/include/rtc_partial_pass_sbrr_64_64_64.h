// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_RTC_PARTIAL_PASS_SBRR_64_64_64_KERNEL_H
#define ROCFFT_RTC_PARTIAL_PASS_SBRR_64_64_64_KERNEL_H

#include "rocfft/rocfft.h"
#include "rtc_kernel.h"
#include <hip/hip_runtime_api.h>
#include <string>

static const unsigned int PARTIAL_PASS_SBRR_64_64_64_THREADS = 128;

// generate name for partial-pass sbrr 64x64x64 compute kernel
std::string partial_pass_64_64_64_sbrr_rtc_kernel_name(rocfft_precision precision);
// generate source for partial-pass sbrr 64x64x64 compute kernel
std::string partial_pass_64_64_64_sbrr_rtc(const std::string& kernel_name,
                                           rocfft_precision   precision);

struct RTCKernelPartialPassSBRR64Cubed : public RTCKernel
{
    static RTCKernel::RTCGenerator generate_from_node(const LeafNode&    node,
                                                      const std::string& gpu_arch,
                                                      bool               enable_callbacks);

    virtual RTCKernelArgs get_launch_args(DeviceCallIn& data) override;

protected:
    RTCKernelPartialPassSBRR64Cubed(const std::string&       kernel_name,
                                    const std::vector<char>& code,
                                    dim3                     gridDim,
                                    dim3                     blockDim)
        : RTCKernel(kernel_name, code, gridDim, blockDim)
    {
    }
};

#endif // ROCFFT_RTC_PARTIAL_PASS_SBRR_64_64_64_KERNEL_H
