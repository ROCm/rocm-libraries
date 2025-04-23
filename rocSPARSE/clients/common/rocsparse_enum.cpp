/*! \file */
/* ************************************************************************
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse.hpp"

#include "rocsparse_enum.hpp"

constexpr rocsparse_matrix_type_t::value_t
    rocsparse_matrix_type_t::values[rocsparse_matrix_type_t::nvalues];

constexpr rocsparse_operation_t::value_t
    rocsparse_operation_t::values[rocsparse_operation_t::nvalues];

constexpr rocsparse_storage_mode_t::value_t
    rocsparse_storage_mode_t::values[rocsparse_storage_mode_t::nvalues];

std::ostream& operator<<(std::ostream& out, const rocsparse_operation& v)
{
    out << rocsparse_operation2string(v);
    return out;
}

std::ostream& operator<<(std::ostream& out, const rocsparse_direction& v)
{
    out << rocsparse_direction2string(v);
    return out;
}
