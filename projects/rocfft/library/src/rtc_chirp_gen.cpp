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

std::string chirp_rtc_kernel_name(rocfft_precision precision, const KIntType& itype)
{
    std::string kernel_name = "chirp_gen";
    kernel_name += rtc_kint_name(itype);
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
    args += "integer_type N";
    args += ", scalar_type* output";
    args += ")";
    return args;
}

// The body below reduces i * i modulo 2N exactly, by splitting the
// product into a high and a low half of integer_type.  Both the
// high-multiply intrinsic and the weight of the high half (2 raised to
// the width of integer_type) therefore depend on that width, so emit
// them to suit.
static std::string chirp_rtc_mulhi_decls(const KIntType& itype)
{
    std::string src;

    src += "__device__ inline integer_type chirp_mul_hi(integer_type a, integer_type b)\n{\n";
    src += itype == KIntType::U64 ? "    return __umul64hi(a, b);\n"
                                  : "    return __umulhi(a, b);\n";
    src += "}\n";

    // Weight of the high half, reduced mod m.  2^64 has no literal, so
    // reach it as (2^64 - 1) + 1 with the reduction applied twice.
    src += "__device__ inline integer_type chirp_hi_weight_mod(integer_type m)\n{\n";
    src += itype == KIntType::U64
               ? "    return static_cast<integer_type>((~0ull % m + 1ull) % m);\n"
               : "    return static_cast<integer_type>(0x100000000ull % m);\n";
    src += "}\n";

    return src;
}

static std::string chirp_rtc_body()
{
    std::string body = "{";
    body += R"_SRC(
        integer_type i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < N)
        {
            integer_type twoN = 2 * N;
            integer_type iSq  = i * i;

            auto f = (double)iSq / (double)twoN;

            integer_type fRnd = floor(f);

            auto aLow = iSq;
            auto bLow = twoN * fRnd;

            auto aHi = chirp_mul_hi(i, i);
            auto bHi = chirp_mul_hi(twoN, fRnd);

            auto f1 = (aHi - bHi) * (double)chirp_hi_weight_mod(twoN) / (double)twoN;
            auto f2 = (double)((aLow - bLow) % twoN) / (double)twoN;
            auto fp = (f1 - floor(f1)) + f2;

            output[i].x = cos(TWO_PI * fp);
            output[i].y = sin(TWO_PI * fp);
        }
        )_SRC";
    body += "}";
    return body;
}

std::string
    chirp_rtc(const std::string& kernel_name, rocfft_precision precision, const KIntType& itype)
{
    std::string src;

    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += rtc_kint_type_decl(itype);
    src += rtc_precision_type_decl(precision);
    src += "static constexpr double TWO_PI = 6.283185307179586476925286766559;\n";
    src += chirp_rtc_mulhi_decls(itype);

    src += chirp_rtc_header;
    src += chirp_rtc_launch_bounds();
    src += kernel_name;
    src += chirp_rtc_args();
    src += chirp_rtc_body();
    return src;
}
