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

#include "common.h"
#include "../conversion/rocsparse_csxsldu.hpp"
#include "../conversion/rocsparse_identity.hpp"
#include "common.hpp"
#include "rocsparse_csritilu0_driver.hpp"
#include "rocsparse_csritilu0x_buffer_size.hpp"
#include "rocsparse_csritilu0x_compute.hpp"
#include "rocsparse_csritilu0x_history.hpp"
#include "rocsparse_csritilu0x_preprocess.hpp"

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, typename T, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_get_permuted_array(I size_, const T* a_, T* x_, const I* perm_)
    {
        const I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(i < size_)
        {
            x_[i] = a_[perm_[i]];
        }
    }

    //
    //
    //
    template <uint32_t BLOCKSIZE, typename T, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_set_identity_array(I size_, T* x_)
    {
        const I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(i < size_)
        {
            x_[i] = static_cast<T>(1);
        }
    }

    //
    //
    //
    template <uint32_t BLOCKSIZE, typename T, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_set_permuted_array(I size_, T* a_, const T* x_, const I* perm_)
    {
        const I i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(i < size_)
        {
            a_[perm_[i]] = x_[i];
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_nrminf(size_t nitems_,
                       const T* __restrict__ x_,
                       floating_data_t<T>* __restrict__ nrm_,
                       const floating_data_t<T>* __restrict__ nrm0_)
    {
        int    tid = hipThreadIdx_x;
        size_t gid = tid + BLOCKSIZE * hipBlockIdx_x;

        __shared__ floating_data_t<T> shared[BLOCKSIZE];

        if(gid < nitems_)
        {
            shared[tid] = std::abs(x_[gid]);
        }
        else
        {
            shared[tid] = 0;
        }

        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            if(nrm0_ != nullptr)
            {
                rocsparse::atomic_max(nrm_, shared[0] / nrm0_[0]);
            }
            else
            {
                rocsparse::atomic_max(nrm_, shared[0]);
            }
        }
    }

    template <uint32_t BLOCKSIZE, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void kernel_nrminf_diff(size_t nitems_,
                            const T* __restrict__ x_,
                            const T* __restrict__ y_,
                            floating_data_t<T>* __restrict__ nrm_,
                            const floating_data_t<T>* __restrict__ nrm0_)
    {
        const uint32_t tid = hipThreadIdx_x;
        const uint32_t gid = tid + BLOCKSIZE * hipBlockIdx_x;

        __shared__ floating_data_t<T> shared[BLOCKSIZE];

        shared[tid] = (gid < nitems_) ? std::abs(x_[gid] - y_[gid]) : 0;

        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            if(nrm0_ != nullptr)
            {
                rocsparse::atomic_max(nrm_, shared[0] / nrm0_[0]);
            }
            else
            {
                rocsparse::atomic_max(nrm_, shared[0]);
            }
        }
    }
}

template <uint32_t BLOCKSIZE, typename T, typename I>
void rocsparse::get_permuted_array(
    rocsparse_handle handle_, I size_, const T* a_, T* x_, const I* perm_)
{
    dim3 blocks((size_ - 1) / BLOCKSIZE + 1);
    dim3 threads(BLOCKSIZE);
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_get_permuted_array<BLOCKSIZE, T, I>),
                                      blocks,
                                      threads,
                                      0,
                                      handle_->stream,
                                      size_,
                                      a_,
                                      x_,
                                      perm_);
}

template <uint32_t BLOCKSIZE, typename T, typename I>
void rocsparse::set_identity_array(rocsparse_handle handle_, I size_, T* x_)
{
    dim3 blocks((size_ - 1) / BLOCKSIZE + 1);
    dim3 threads(BLOCKSIZE);

    THROW_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_set_identity_array<BLOCKSIZE, T, I>),
                                      blocks,
                                      threads,
                                      0,
                                      handle_->stream,
                                      size_,
                                      x_);
}

template <uint32_t BLOCKSIZE, typename T, typename I>
void rocsparse::set_permuted_array(
    rocsparse_handle handle_, I size_, T* a_, const T* x_, const I* perm_)
{
    dim3 blocks((size_ - 1) / BLOCKSIZE + 1);
    dim3 threads(BLOCKSIZE);
    THROW_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_set_permuted_array<BLOCKSIZE, T, I>),
                                      blocks,
                                      threads,
                                      0,
                                      handle_->stream,
                                      size_,
                                      a_,
                                      x_,
                                      perm_);
}

template <uint32_t BLOCKSIZE, typename T>
rocsparse_status rocsparse::nrminf(rocsparse_handle          handle_,
                                   size_t                    nitems_,
                                   const T*                  x_,
                                   floating_data_t<T>*       nrm_,
                                   const floating_data_t<T>* nrm0_,
                                   bool                      MX)
{

    if(!MX)
    {
        RETURN_IF_HIP_ERROR(hipMemsetAsync(nrm_, 0, sizeof(floating_data_t<T>), handle_->stream));
    }
    //
    // Compute nrm max of the matrix.
    //
    size_t nitems_nblocks = (nitems_ - 1) / BLOCKSIZE + 1;
    dim3   blocks(nitems_nblocks);
    dim3   threads(BLOCKSIZE);
    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_nrminf<BLOCKSIZE, T>),
                                       blocks,
                                       threads,
                                       0,
                                       handle_->stream,
                                       nitems_,
                                       x_,
                                       nrm_,
                                       nrm0_);

    return rocsparse_status_success;
}

template <uint32_t BLOCKSIZE, typename T>
rocsparse_status rocsparse::nrminf_diff(rocsparse_handle          handle_,
                                        size_t                    nitems_,
                                        const T*                  x_,
                                        const T*                  y_,
                                        floating_data_t<T>*       nrm_,
                                        const floating_data_t<T>* nrm0_,
                                        bool                      MX)
{

    //
    // Compute nrm max of the matrix.
    //
    size_t nitems_nblocks = (nitems_ - 1) / BLOCKSIZE + 1;
    dim3   blocks(nitems_nblocks);
    dim3   threads(BLOCKSIZE);
    if(!MX)
    {
        RETURN_IF_HIP_ERROR(hipMemsetAsync(nrm_, 0, sizeof(floating_data_t<T>), handle_->stream));
    }

    RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::kernel_nrminf_diff<BLOCKSIZE, T>),
                                       blocks,
                                       threads,
                                       0,
                                       handle_->stream,
                                       nitems_,
                                       x_,
                                       y_,
                                       nrm_,
                                       nrm0_);
    return rocsparse_status_success;
}

//
#define INSTANTIATE(BLOCKSIZE, T, I)                                            \
    template void rocsparse::set_identity_array<BLOCKSIZE, T, I>(               \
        rocsparse_handle handle_, I size_, T * x_);                             \
    template void rocsparse::get_permuted_array<BLOCKSIZE, T, I>(               \
        rocsparse_handle handle_, I size_, const T* a_, T* x_, const I* perm_); \
    template void rocsparse::set_permuted_array<BLOCKSIZE, T, I>(               \
        rocsparse_handle handle_, I size_, T * a_, const T* x_, const I* perm_)

INSTANTIATE(1024, int32_t, int32_t);
INSTANTIATE(1024, float, int32_t);
INSTANTIATE(1024, double, int32_t);
INSTANTIATE(1024, rocsparse_float_complex, int32_t);
INSTANTIATE(1024, rocsparse_double_complex, int32_t);

INSTANTIATE(256, float, int32_t);
INSTANTIATE(256, double, int32_t);
INSTANTIATE(256, rocsparse_float_complex, int32_t);
INSTANTIATE(256, rocsparse_double_complex, int32_t);

INSTANTIATE(512, float, int32_t);
INSTANTIATE(512, double, int32_t);
INSTANTIATE(512, rocsparse_float_complex, int32_t);
INSTANTIATE(512, rocsparse_double_complex, int32_t);

INSTANTIATE(1024, int64_t, int64_t);
INSTANTIATE(1024, float, int64_t);
INSTANTIATE(1024, double, int64_t);
INSTANTIATE(1024, rocsparse_float_complex, int64_t);
INSTANTIATE(1024, rocsparse_double_complex, int64_t);

INSTANTIATE(256, float, int64_t);
INSTANTIATE(256, double, int64_t);
INSTANTIATE(256, rocsparse_float_complex, int64_t);
INSTANTIATE(256, rocsparse_double_complex, int64_t);

INSTANTIATE(512, float, int64_t);
INSTANTIATE(512, double, int64_t);
INSTANTIATE(512, rocsparse_float_complex, int64_t);
INSTANTIATE(512, rocsparse_double_complex, int64_t);

#undef INSTANTIATE

#define INSTANTIATE(BLOCKSIZE, T)                                                                \
    template rocsparse_status rocsparse::nrminf_diff<BLOCKSIZE, T>(                              \
        rocsparse_handle          handle_,                                                       \
        size_t                    nitems_,                                                       \
        const T*                  x_,                                                            \
        const T*                  y_,                                                            \
        floating_data_t<T>*       nrm_,                                                          \
        const floating_data_t<T>* nrm0_,                                                         \
        bool                      MX);                                                                                \
    template rocsparse_status rocsparse::nrminf<BLOCKSIZE, T>(rocsparse_handle          handle_, \
                                                              size_t                    nitems_, \
                                                              const T*                  x_,      \
                                                              floating_data_t<T>*       nrm_,    \
                                                              const floating_data_t<T>* nrm0_,   \
                                                              bool                      MX)

INSTANTIATE(1024, float);
INSTANTIATE(1024, double);
INSTANTIATE(1024, rocsparse_float_complex);
INSTANTIATE(1024, rocsparse_double_complex);

INSTANTIATE(512, float);
INSTANTIATE(512, double);
INSTANTIATE(512, rocsparse_float_complex);
INSTANTIATE(512, rocsparse_double_complex);

INSTANTIATE(256, float);
INSTANTIATE(256, double);
INSTANTIATE(256, rocsparse_float_complex);
INSTANTIATE(256, rocsparse_double_complex);

#undef INSTANTIATE
