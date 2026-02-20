/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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
#pragma once

#include "rocsparse_extract.hpp"

struct rocsparse_extract_descr_default_t : public _rocsparse_extract_descr
{

    rocsparse_extract_descr_default_t(rocsparse_const_spmat_descr source,
                                      rocsparse_const_spmat_descr target);

    virtual rocsparse_status buffer_size(rocsparse_handle            handle,
                                         rocsparse_const_spmat_descr source,
                                         rocsparse_spmat_descr       target,
                                         rocsparse_extract_stage     stage,
                                         size_t* __restrict__ buffer_size_in_bytes);

    virtual rocsparse_status run(rocsparse_handle            handle,
                                 rocsparse_const_spmat_descr source,
                                 rocsparse_spmat_descr       target,
                                 rocsparse_extract_stage     stage,
                                 size_t                      buffer_size,
                                 void* __restrict__ buffer);
};
