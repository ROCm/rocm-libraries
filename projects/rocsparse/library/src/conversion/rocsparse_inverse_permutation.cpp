/*! \file */
/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
#include "internal/conversion/rocsparse_inverse_permutation.h"
#include "rocsparse_inverse_permutation.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel1(I n_, const I* p_, I* q_)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(idx >= n_)
        {
            return;
        }

        q_[p_[idx] - 1] = idx + 1;
    }

    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel0(I n_, const I* p_, I* q_)
    {
        I idx = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(idx >= n_)
        {
            return;
        }
        q_[p_[idx]] = idx;
    }

    template <typename I>
    rocsparse_status inverse_permutation_core(rocsparse_handle handle_,
                                              I                n_,
                                              const I* __restrict__ p_,
                                              I* __restrict__ q_,
                                              rocsparse_index_base base_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr uint32_t BLOCKSIZE = 512;
        dim3                      blocks((n_ - 1) / BLOCKSIZE + 1);
        dim3                      threads(BLOCKSIZE);
        if(base_ == rocsparse_index_base_zero)
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::kernel0<BLOCKSIZE>), blocks, threads, 0, handle_->stream, n_, p_, q_);
        }
        else
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
                (rocsparse::kernel1<BLOCKSIZE>), blocks, threads, 0, handle_->stream, n_, p_, q_);
        }
        return rocsparse_status_success;
    }

    template <typename I>
    rocsparse_status inverse_permutation_template(rocsparse_handle handle_,
                                                  I                n_,
                                                  const I* __restrict__ p_,
                                                  I* __restrict__ q_,
                                                  rocsparse_index_base base_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(n_ == 0)
        {
            return rocsparse_status_success;
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::inverse_permutation_core(handle_, n_, p_, q_, base_));
        return rocsparse_status_success;
    }

    template <typename I>
    rocsparse_status inverse_permutation_impl(rocsparse_handle handle,
                                              I                n,
                                              const I* __restrict__ p,
                                              I* __restrict__ q,
                                              rocsparse_index_base base)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Logging
        rocsparse::log_trace(
            handle, "rocsparse_inverse_permutation", n, (const void*&)p, (const void*&)q, base);

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, n);
        ROCSPARSE_CHECKARG_ARRAY(2, n, p);
        ROCSPARSE_CHECKARG_ARRAY(3, n, q);
        ROCSPARSE_CHECKARG_ENUM(4, base);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::inverse_permutation_template(handle, n, p, q, base));
        return rocsparse_status_success;
    }
}

#define INSTANTIATE(ITYPE)                                                                       \
    template rocsparse_status rocsparse::inverse_permutation_template(rocsparse_handle,          \
                                                                      ITYPE,                     \
                                                                      const ITYPE* __restrict__, \
                                                                      ITYPE* __restrict__,       \
                                                                      rocsparse_index_base);     \
    template rocsparse_status rocsparse::inverse_permutation_impl<ITYPE>(                        \
        rocsparse_handle,                                                                        \
        ITYPE,                                                                                   \
        const ITYPE* __restrict__,                                                               \
        ITYPE* __restrict__,                                                                     \
        rocsparse_index_base)

INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */
extern "C" rocsparse_status rocsparse_inverse_permutation(rocsparse_handle     handle_,
                                                          rocsparse_int        n_,
                                                          const rocsparse_int* p_,
                                                          rocsparse_int*       q_,
                                                          rocsparse_index_base base_)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::inverse_permutation_impl(handle_, n_, p_, q_, base_));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
