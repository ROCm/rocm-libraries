/* ************************************************************************
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "auxiliary.hpp"
#include "activation.hpp"

// clang-format off
const hipDataType string_to_hip_datatype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? HIP_R_32F  :
        value == "f16_r" || value == "h" ? HIP_R_16F  :
        value == "bf16_r"                ? HIP_R_16BF  :
        value == "i8_r"                  ? HIP_R_8I   :
#if defined(HIP_FP8_TYPE_OCP) || defined(__HIP_PLATFORM_NVIDIA__)
        value == "f8_r"                  ? HIP_R_8F_E4M3    :
        value == "bf8_r"                 ? HIP_R_8F_E5M2    :
#endif
        static_cast<hipDataType>(-1);
}

const hipsparseLtComputetype_t string_to_hipsparselt_computetype(const std::string& value)
{
    return
        value == "f32_r" || value == "s" ? HIPSPARSELT_COMPUTE_32F  :
        value == "i32_r"                 ? HIPSPARSELT_COMPUTE_32I  :
        value == "f16_r" || value == "h" ? HIPSPARSELT_COMPUTE_16F  :
        value == "tf32_r"                ? HIPSPARSELT_COMPUTE_TF32  :
        value == "tf32f_r"               ? HIPSPARSELT_COMPUTE_TF32_FAST  :
        static_cast<hipsparseLtComputetype_t>(-1);
}
// clang-format on

const hipsparselt_activation_type string_to_hipsparselt_activation_type(const std::string& value)
{
    return value == "none"          ? hipsparselt_activation_type::none
           : value == "abs"         ? hipsparselt_activation_type::abs
           : value == "clippedrelu" ? hipsparselt_activation_type::clippedrelu
           : value == "gelu"        ? hipsparselt_activation_type::gelu
           : value == "leakyrelu"   ? hipsparselt_activation_type::leakyrelu
           : value == "relu"        ? hipsparselt_activation_type::relu
           : value == "sigmoid"     ? hipsparselt_activation_type::sigmoid
           : value == "tanh"        ? hipsparselt_activation_type::tanh
           : value == "all"         ? hipsparselt_activation_type::all
           : value == "exp"         ? hipsparselt_activation_type::exp
                                    : static_cast<hipsparselt_activation_type>(-1);
}

// Convert hipsparselt_activation_type to string
const char* hipsparselt_activation_type_to_string(hipsparselt_activation_type type)
{
    switch(type)
    {
    case hipsparselt_activation_type::abs:
        return "abs";
    case hipsparselt_activation_type::clippedrelu:
        return "clippedrelu";
    case hipsparselt_activation_type::exp:
        return "exp";
    case hipsparselt_activation_type::gelu:
        return "gelu";
    case hipsparselt_activation_type::leakyrelu:
        return "leakyrelu";
    case hipsparselt_activation_type::relu:
        return "relu";
    case hipsparselt_activation_type::sigmoid:
        return "sigmoid";
    case hipsparselt_activation_type::tanh:
        return "tanh";
    case hipsparselt_activation_type::all:
        return "all";
    case hipsparselt_activation_type::none:
        return "none";
    default:
        return "invalid";
    }
}
