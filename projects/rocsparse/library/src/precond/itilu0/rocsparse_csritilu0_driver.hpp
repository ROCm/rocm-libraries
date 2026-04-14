/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
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

#include "rocsparse_control.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <rocsparse_itilu0_alg ALG>
    struct csritilu0_driver_t
    {
        template <typename I, typename J>
        struct buffer_size
        {
            static rocsparse_status run(rocsparse_handle     handle_,
                                        rocsparse_itilu0_alg alg_,
                                        J                    options_,
                                        J                    nsweeps_,
                                        J                    m_,
                                        I                    nnz_,
                                        const I* __restrict__ ptr_,
                                        const J* __restrict__ ind_,
                                        rocsparse_index_base base_,
                                        rocsparse_datatype   datatype_,
                                        size_t* __restrict__ buffer_size_);
        };

        template <typename I, typename J>
        struct preprocess
        {
            static rocsparse_status run(rocsparse_handle     handle_,
                                        rocsparse_itilu0_alg alg_,
                                        J                    options_,
                                        J                    nsweeps_,
                                        J                    m_,
                                        I                    nnz_,
                                        const I* __restrict__ ptr_,
                                        const J* __restrict__ ind_,
                                        rocsparse_index_base base_,
                                        rocsparse_datatype   datatype_,
                                        size_t               buffer_size_,
                                        void* __restrict__ buffer_);
        };

        template <typename T, typename I, typename J>
        struct compute
        {
            static rocsparse_status run(rocsparse_handle     handle_,
                                        rocsparse_itilu0_alg alg_,
                                        J                    options_,
                                        J*                   nsweeps_,
                                        J                    nfreeiter_,
                                        floating_data_t<T>   tol_,
                                        J                    m_,
                                        I                    nnz_,
                                        const I* __restrict__ ptr_,
                                        const J* __restrict__ ind_,
                                        const T* __restrict__ val_,
                                        T* __restrict__ ilu0_,
                                        rocsparse_index_base base_,
                                        size_t               buffer_size_,
                                        void* __restrict__ buffer_);
        };

        template <typename T, typename J>
        struct history
        {
            static rocsparse_status run(rocsparse_handle     handle_,
                                        rocsparse_itilu0_alg alg_,
                                        J*                   niter_,
                                        T*                   data_,
                                        size_t               buffer_size_,
                                        void*                buffer_);
        };
    };
}
