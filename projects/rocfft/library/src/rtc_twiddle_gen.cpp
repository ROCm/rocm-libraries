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

#include "rtc_twiddle_gen.h"
#include "device/kernel-generator-embed.h"
#include "rtc_kernel.h"
#include "twiddles.h"

std::string twiddle_rtc_kernel_name(TwiddleTableType type, rocfft_precision precision)
{
    std::string kernel_name = "twiddle_gen";
    switch(type)
    {
    case TwiddleTableType::RADICES:
        kernel_name += "_radices";
        break;
    case TwiddleTableType::LENGTH_N:
        kernel_name += "_N";
        break;
    case TwiddleTableType::HALF_N:
        kernel_name += "_half_N";
        break;
    case TwiddleTableType::LARGE:
        kernel_name += "_large";
        break;
    case TwiddleTableType::PARTIAL_PASS_N:
        kernel_name += "_pp_N";
        break;
    }
    kernel_name += rtc_precision_name(precision);
    return kernel_name;
}

const char* twiddle_rtc_header = "extern \"C\" __global__ void ";

static std::string twiddle_rtc_launch_bounds(TwiddleTableType type)
{
    std::string bounds = "__launch_bounds__(";
    switch(type)
    {
    case TwiddleTableType::RADICES:
    case TwiddleTableType::LARGE:
    case TwiddleTableType::PARTIAL_PASS_N:
        bounds += std::to_string(TWIDDLES_THREADS * TWIDDLES_THREADS);
        break;
    case TwiddleTableType::LENGTH_N:
    case TwiddleTableType::HALF_N:
        bounds += std::to_string(TWIDDLES_THREADS);
        break;
    }
    bounds += ") ";
    return bounds;
}

static std::string twiddle_rtc_args(TwiddleTableType type, rocfft_precision precision)
{
    std::string args = "(";
    switch(type)
    {
    case TwiddleTableType::RADICES:
        args += "size_t length_limit";
        args += ", size_t num_radices";
        args += ", radices_t radices";
        args += ", radices_t radices_prod";
        args += ", radices_t radices_sum_prod";
        args += ", scalar_type* output";
        break;
    case TwiddleTableType::LENGTH_N:
        args += "size_t length_limit";
        args += ", size_t N";
        args += ", scalar_type* output";
        break;
    case TwiddleTableType::HALF_N:
        args += "size_t half_N";
        args += ", size_t N";
        args += ", scalar_type* output";
        break;
    case TwiddleTableType::LARGE:
        args += "double phi";
        args += ", size_t base";
        args += ", size_t X";
        args += ", size_t Y";
        args += ", scalar_type* output";
        break;
    case TwiddleTableType::PARTIAL_PASS_N:
        args += "size_t N";
        args += ", scalar_type* output";
        break;
    }
    args += ")";
    return args;
}

static std::string twiddle_rtc_body(TwiddleTableType type)
{
    std::string body = "{";
    switch(type)
    {
    case TwiddleTableType::RADICES:
        body += R"_SRC(
        auto i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < num_radices - 1)
        {
            auto L     = radices_prod.data[i];
            auto radix = radices.data[i + 1];
            auto k     = threadIdx.y + blockIdx.y * blockDim.y;

            if(k < L / radix)
            {
                double theta = TWO_PI * (k) / (L);
                auto   index = radices_sum_prod.data[i] + k * (radices.data[i + 1] - 1);

                for(size_t j = 1; j < radix && index < length_limit; ++j)
                {
                    output[index].x = cos((j)*theta);
                    output[index].y = sin((j)*theta);

                    ++index;
                }
            }
        }
        )_SRC";
        break;
    case TwiddleTableType::LENGTH_N:
        body += R"_SRC(
        auto i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < N && i < length_limit)
        {
            double c = cos(TWO_PI * i / N);
            double s = sin(TWO_PI * i / N);

            output[i].x = c;
            output[i].y = s;
        }
        )_SRC";
        break;
    case TwiddleTableType::HALF_N:
        body += R"_SRC(
        auto i = threadIdx.x + blockIdx.x * blockDim.x;

        if(i < half_N)
        {
            double c = cos(TWO_PI * i / (2 * N));
            double s = sin(TWO_PI * i / (2 * N));

            output[i].x = c;
            output[i].y = s;
        }
        )_SRC";
        break;
    case TwiddleTableType::LARGE:
        body += R"_SRC(
        auto iY = threadIdx.y + blockIdx.y * blockDim.y;

        if(iY < Y)
        {
            auto iX = threadIdx.x + blockIdx.x * blockDim.x;

            if(iX < X)
            {
                auto j = (static_cast<size_t>(1) << (iY * base)) * iX;

                double c = cos(phi * j);
                double s = sin(phi * j);

                auto index = iY * X + iX;

                output[index].x = c;
                output[index].y = s;
            }
        }
        )_SRC";
        break;
    case TwiddleTableType::PARTIAL_PASS_N:
        body += R"_SRC(
        auto i_row = threadIdx.x + blockIdx.x * blockDim.x;
        auto i_col = threadIdx.y + blockIdx.y * blockDim.y;

        if(i_row < N && i_col < N)
        {   
            auto n = i_row * i_col;
            double arg = -TWO_PI * n / N;

            output[N * i_col + i_row].x = cos(arg);
            output[N * i_col + i_row].y = -sin(arg);
        }
        )_SRC";
        break;
    }
    body += "}";
    return body;
}

std::string
    twiddle_rtc(const std::string& kernel_name, TwiddleTableType type, rocfft_precision precision)
{
    std::string src;

    src += rocfft_complex_h;
    src += common_h;
    src += device_enum_h;
    src += rtc_precision_type_decl(precision);
    src += "static constexpr double TWO_PI = -6.283185307179586476925286766559;\n";
    src += "static const unsigned int TWIDDLES_MAX_RADICES = "
           + std::to_string(TWIDDLES_MAX_RADICES) + ";\n";

    src += radices_t_str;
    src += twiddle_rtc_header;
    src += twiddle_rtc_launch_bounds(type);
    src += kernel_name;
    src += twiddle_rtc_args(type, precision);
    src += twiddle_rtc_body(type);
    return src;
}
