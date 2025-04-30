/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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
#ifndef ROCSPARSE_CSRITILU0_BUFFER_SIZE_HPP
#define ROCSPARSE_CSRITILU0_BUFFER_SIZE_HPP

#include "control.h"
#include "utility.h"

namespace rocsparse
{
    template <typename I, typename J>
    rocsparse_status csritilu0_buffer_size_template(rocsparse_handle     handle_,
                                                    rocsparse_itilu0_alg alg_,
                                                    J                    options_,
                                                    J                    nmaxiter_,
                                                    J                    m_,
                                                    I                    nnz_,
                                                    const I* __restrict__ ptr_,
                                                    const J* __restrict__ ind_,
                                                    rocsparse_index_base base_,
                                                    rocsparse_datatype   datatype_,
                                                    size_t* __restrict__ buffer_size_);

    template <typename I, typename J>
    rocsparse_status csritilu0_buffer_size_impl(rocsparse_handle     handle_,
                                                rocsparse_itilu0_alg alg_,
                                                J                    options_,
                                                J                    nmaxiter_,
                                                J                    m_,
                                                I                    nnz_,
                                                const I* __restrict__ ptr_,
                                                const J* __restrict__ ind_,
                                                rocsparse_index_base base_,
                                                rocsparse_datatype   datatype_,
                                                size_t* __restrict__ buffer_size_);
}

#endif // ROCSPARSE_CSRITILU0_BUFFER_SIZE_HPP
