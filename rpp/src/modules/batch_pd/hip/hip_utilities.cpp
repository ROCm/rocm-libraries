/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "hip_declarations.hpp"

void max_size(Rpp32u *height, Rpp32u *width, unsigned int batch_size, unsigned int *max_height, unsigned int *max_width)
{
    int i;
    *max_height  = 0;
    *max_width =0;
    for (i=0; i<batch_size; i++){
        if(*max_height < height[i])
            *max_height = height[i];
        if(*max_width < width[i])
            *max_width = width[i];
    }
}

void get_kernel_name(std::string &kernel_name, const RPPTensorFunctionMetaData &tensor_info)
{
    switch (tensor_info._in_type)
    {
    case RPPTensorDataType::U8:
        switch (tensor_info._out_type)
        {
        case RPPTensorDataType::U8:
            break;
        case RPPTensorDataType::FP32:
            kernel_name = kernel_name + "_u8_fp32";
            break;
        case RPPTensorDataType::FP16:
            kernel_name = kernel_name + "_u8_fp16";
            break;
        case RPPTensorDataType::I8:
            kernel_name = kernel_name + "_u8_int8";
            break;
        default:
            break;
        }
        break;
    case RPPTensorDataType::FP32:
        kernel_name = kernel_name + "_fp32";
        break;
    case RPPTensorDataType::FP16:
        kernel_name = kernel_name + "_fp16";
        break;
    case RPPTensorDataType::I8:
        kernel_name = kernel_name + "_int8";
        break;
    default:
        break;
    }
}
