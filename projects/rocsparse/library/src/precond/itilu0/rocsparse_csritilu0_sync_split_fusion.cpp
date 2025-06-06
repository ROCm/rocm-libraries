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

#include "rocsparse_common.hpp"
#include "rocsparse_csritilu0_driver.hpp"

template <>
struct rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split_fusion>
{
    //
    // History, same as algorithm 1.
    //
    template <typename T, typename J>
    struct history
    {
        static rocsparse_status run(rocsparse_handle     handle_,
                                    rocsparse_itilu0_alg alg_,
                                    J*                   niter_,
                                    T*                   data_,
                                    size_t               buffer_size_,
                                    void*                buffer_)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<rocsparse_itilu0_alg_sync_split>::history<T, J>::run(
                    handle_, alg_, niter_, data_, buffer_size_, buffer_)));
            return rocsparse_status_success;
        }
    };

    //
    // Buffer size, same as algorithm 1.
    //
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
                                    size_t* __restrict__ buffer_size_)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split>::buffer_size<I, J>::run(handle_,
                                                                             alg_,
                                                                             options_,
                                                                             nsweeps_,
                                                                             m_,
                                                                             nnz_,
                                                                             ptr_,
                                                                             ind_,
                                                                             base_,
                                                                             datatype_,
                                                                             buffer_size_)));
            return rocsparse_status_success;
        }
    };

    //
    // Preprocess, same as algorithm 1.
    //
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
                                    void* __restrict__ buffer_)

        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split>::preprocess<I, J>::run(handle_,
                                                                            alg_,
                                                                            options_,
                                                                            nsweeps_,
                                                                            m_,
                                                                            nnz_,
                                                                            ptr_,
                                                                            ind_,
                                                                            base_,
                                                                            datatype_,
                                                                            buffer_size_,
                                                                            buffer_)));
            return rocsparse_status_success;
        }
    };

    //
    // Compute, same as algorithm 1 but with algorithm 2 in the back end.
    //
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
                                    T* __restrict__ x_,
                                    rocsparse_index_base base_,
                                    size_t               buffer_size_,
                                    void* __restrict__ buffer_)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::csritilu0_driver_t<
                    rocsparse_itilu0_alg_sync_split>::compute<T, I, J>::run(handle_,
                                                                            alg_,
                                                                            options_,
                                                                            nsweeps_,
                                                                            nfreeiter_,
                                                                            tol_,
                                                                            m_,
                                                                            nnz_,
                                                                            ptr_,
                                                                            ind_,
                                                                            val_,
                                                                            x_,
                                                                            base_,
                                                                            buffer_size_,
                                                                            buffer_)));
            return rocsparse_status_success;
        }
    };
};

#define INSTANTIATE(T, I, J)                       \
    template struct rocsparse::csritilu0_driver_t< \
        rocsparse_itilu0_alg_sync_split_fusion>::compute<T, I, J>

INSTANTIATE(float, rocsparse_int, rocsparse_int);
INSTANTIATE(double, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_float_complex, rocsparse_int, rocsparse_int);
INSTANTIATE(rocsparse_double_complex, rocsparse_int, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(T, J)                          \
    template struct rocsparse::csritilu0_driver_t< \
        rocsparse_itilu0_alg_sync_split_fusion>::history<T, J>

INSTANTIATE(float, rocsparse_int);
INSTANTIATE(double, rocsparse_int);

#undef INSTANTIATE

#define INSTANTIATE(I, J)                                           \
    template struct rocsparse::csritilu0_driver_t<                  \
        rocsparse_itilu0_alg_sync_split_fusion>::buffer_size<I, J>; \
    template struct rocsparse::csritilu0_driver_t<                  \
        rocsparse_itilu0_alg_sync_split_fusion>::preprocess<I, J>;

INSTANTIATE(rocsparse_int, rocsparse_int);

#undef INSTANTIATE
