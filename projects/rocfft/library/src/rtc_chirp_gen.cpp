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

#include "rtc_chirp_gen.h"
#include "device/kernel-generator-embed.h"
#include "rtc_kernel.h"

std::string chirp_rtc_kernel_name(rocfft_precision precision)
{
    std::string kernel_name = "chirp_gen";
    kernel_name += rtc_precision_name(precision);
    return kernel_name;
}

const char* chirp_rtc_header = "extern \"C\" __global__ void ";

static std::string chirp_rtc_launch_bounds()
{
    std::string bounds = "__launch_bounds__(";
    bounds += std::to_string(CHIRP_THREADS);
    bounds += ") ";
    return bounds;
}

static std::string chirp_rtc_args()
{
    std::string args = "(";
    args += "size_t N";
    args += ", scalar_type* output";
    args += ")";
    return args;
}

static std::string chirp_rtc_body()
{
    std::string body = "{";
    body += R"_SRC(
        auto i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < N)
        {
            unsigned int twoN = 2 * N;
            unsigned int iSq  = i * i;

            auto f = (double)iSq / (double)twoN;

            unsigned int fRnd = floor(f);

            auto aLow = iSq;
            auto bLow = twoN * fRnd;

            auto aHi = __umulhi(i, i);
            auto bHi = __umulhi(twoN, fRnd);

            auto f1 = (aHi - bHi) * (double)(0x100000000 % twoN) / (double)twoN;
            auto f2 = (double)((aLow - bLow) % twoN) / (double)twoN;
            auto fp = (f1 - floor(f1)) + f2;

            output[i].x = cos(TWO_PI * fp);
            output[i].y = sin(TWO_PI * fp);
        }
        )_SRC";
    body += "}";
    return body;
}

std::string chirp_rtc(const std::string& kernel_name, rocfft_precision precision)
{
    std::string src;

    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += rtc_precision_type_decl(precision);
    src += "static constexpr double TWO_PI = 6.283185307179586476925286766559;\n";

    src += chirp_rtc_header;
    src += chirp_rtc_launch_bounds();
    src += kernel_name;
    src += chirp_rtc_args();
    src += chirp_rtc_body();
    return src;
}
