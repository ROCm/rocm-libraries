/*******************************************************************************
 *
 * MIT License
 *
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include "utility.hpp"
#include <sys/types.h>
#include <unistd.h>

std::string prefix(const char* layer, const char* caller)
{
    time_t now   = time(0);
    tm*    local = localtime(&now);

    std::string             format = "[%d-%02d-%02d %02d:%02d:%02d][HIPSPARSELT][%lu][%s][%s]\0";
    std::unique_ptr<char[]> buf(new char[255]);
    std::sprintf(buf.get(),
                 format.c_str(),
                 1900 + local->tm_year,
                 1 + local->tm_mon,
                 local->tm_mday,
                 local->tm_hour,
                 local->tm_min,
                 local->tm_sec,
                 getpid(),
                 layer,
                 caller);
    return std::string(buf.get());
}

const char* rocsparselt_compute_type_to_string(rocsparselt_compute_type type)
{
    switch(type)
    {
    case rocsparselt_compute_f32:
        return "f32";
    case rocsparselt_compute_i32:
        return "i32";
    }
}

const char* rocsparselt_order_to_string(rocsparselt_order order)
{
    switch(order)
    {
    case rocsparselt_order_row:
        return "row";
    case rocsparselt_order_column:
        return "col";
    }
}

const char* rocsparselt_operation_to_string(rocsparselt_operation op)
{
    switch(op)
    {
    case rocsparselt_operation_none:
        return "non_transpose";
    case rocsparselt_operation_transpose:
        return "transpose";
    case rocsparselt_operation_conjugate_transpose:
        return "conjugate_transpose";
    }
}

const char* rocsparselt_sparsity_to_string(rocsparselt_sparsity sparsity)
{
    switch(sparsity)
    {
    case rocsparselt_sparsity_50_percent:
        return "50%";
    }
}

const char* rocsparselt_matrix_type_to_string(rocsparselt_matrix_type type)
{
    switch(type)
    {
    case rocsparselt_matrix_type_unknown:
        return "unknown";
    case rocsparselt_matrix_type_dense:
        return "dense";
    case rocsparselt_matrix_type_structured:
        return "structured";
    }
}

const char* rocsparselt_layer_mode2string(rocsparselt_layer_mode layer_mode)
{
    switch(layer_mode)
    {
    case rocsparselt_layer_mode_none:
        return "None";
    case rocsparselt_layer_mode_log_error:
        return "Error";
    case rocsparselt_layer_mode_log_trace:
        return "Trace";
    case rocsparselt_layer_mode_log_hints:
        return "Hints";
    case rocsparselt_layer_mode_log_info:
        return "Info";
    case rocsparselt_layer_mode_log_api:
        return "Api";
    default:
        return "Invalid";
    }
}

const char* rocsparselt_activation_type_to_string(rocsparselt_matmul_descr_attribute type)
{
    switch(type)
    {
    case rocsparselt_matmul_activation_abs:
        return "abs";
    case rocsparselt_matmul_activation_gelu:
        return "gelu";
    case rocsparselt_matmul_activation_leakyrelu:
        return "leakyrelu";
    case rocsparselt_matmul_activation_relu:
        return "relu";
    case rocsparselt_matmul_activation_sigmoid:
        return "sigmoid";
    case rocsparselt_matmul_activation_tanh:
        return "tanh";
    default:
        return "none";
    }
}
