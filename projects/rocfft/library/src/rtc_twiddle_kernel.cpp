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

#include "rtc_twiddle_kernel.h"
#include "device/kernel-generator-embed.h"
#include "rtc_cache.h"

RTCKernelTwiddle RTCKernelTwiddle::generate(const std::string& gpu_arch,
                                            TwiddleTableType   type,
                                            rocfft_precision   precision)
{
    auto kernel_name = twiddle_rtc_kernel_name(type, precision);

    kernel_src_gen_t generator{
        [=](const std::string& kernel_name) { return twiddle_rtc(kernel_name, type, precision); }};

    auto code = RTCCache::cached_compile(kernel_name, gpu_arch, generator, generator_sum());

    return RTCKernelTwiddle{kernel_name, code, {}, {}};
}
