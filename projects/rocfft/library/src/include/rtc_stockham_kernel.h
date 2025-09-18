// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_RTC_STOCKHAM_KERNEL_H
#define ROCFFT_RTC_STOCKHAM_KERNEL_H

#include "rtc_kernel.h"

struct RTCKernelStockham : public RTCKernel
{
    RTCKernelStockham(const std::string& kernel_name, const std::vector<char>& code)
        : RTCKernel(kernel_name, code)
        , hardcoded_dim(kernel_name.find("_dim") != std::string::npos)
    {
    }

    static RTCKernel::RTCGenerator generate_from_node(const LeafNode&    node,
                                                      const std::string& gpu_arch,
                                                      bool               enable_callbacks);

    virtual RTCKernelArgs get_launch_args(DeviceCallIn& data) override;

protected:
    struct RTCStockhamGenerator : public RTCKernel::RTCGenerator
    {
        // For rtc_stockham_kernel, once it's the correct type from generate_from_node,
        // we assign the name function.
        // Changed for tuning framework: since we'd like to get the kernel name information
        // anyway, even when it's compiled.
        virtual bool valid() const override
        {
            return (generate_name) ? true : false;
        }
        // generator is the correct type, but kernel is already compiled
        virtual bool is_pre_compiled() const override
        {
            return generate_name && (!generate_src) && (!construct_rtckernel);
        }
    };

private:
    // true if the kernel is hardcoded for a number of dimensions.
    // kernels generated at runtime will be, but ahead-of-time
    // compiled kernels won't.
    bool hardcoded_dim;
};

#endif
