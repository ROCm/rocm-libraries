/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_extract.hpp"
#include "rocsparse_extract_alg_default.hpp"
#include "rocsparse_utility.hpp"

extern "C" rocsparse_status rocsparse_create_extract_descr(rocsparse_extract_descr*    descr,
                                                           rocsparse_const_spmat_descr source,
                                                           rocsparse_spmat_descr       target,
                                                           rocsparse_extract_alg       alg)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_POINTER(1, source);
    ROCSPARSE_CHECKARG_POINTER(2, target);
    ROCSPARSE_CHECKARG_ENUM(3, alg);

    switch(alg)
    {
    case rocsparse_extract_alg_default:
    {
        descr[0] = new rocsparse_extract_descr_default_t(source, target);
        break;
    }
    }
    return rocsparse_status_success;
}
