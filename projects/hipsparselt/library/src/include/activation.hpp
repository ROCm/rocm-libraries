/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019-2023 Advanced Micro Devices, Inc.
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

#pragma once
#include <hipsparselt/hipsparselt-export.h>
#include <iostream>
#include <string>

enum class hipsparselt_activation_type
{
    none = 0,
    abs,
    clippedrelu,
    gelu,
    leakyrelu,
    relu,
    sigmoid,
    tanh,
    all,
    exp,
};

HIPSPARSELT_EXPORT
const hipsparselt_activation_type string_to_hipsparselt_activation_type(const std::string& value);

// Convert hipsparselt_activation_type to string
HIPSPARSELT_EXPORT
const char* hipsparselt_activation_type_to_string(hipsparselt_activation_type type);
