/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_line_nnz_profile.hpp"

#include "rocsparse_common.h"
#include "rocsparse_control.hpp"
#include "rocsparse_hip.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // Reduction over line lengths len[i] = offsets[i+1] - offsets[i]
    // (base-independent, a difference of consecutive offsets), accumulating the
    // maximum into a global counter via one atomic per block.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void line_nnz_profile_kernel(int64_t nlines,
                                 const I* __restrict__ offsets,
                                 unsigned long long* __restrict__ d_max)
    {
        const int64_t tid    = hipThreadIdx_x;
        const int64_t gid    = int64_t(hipBlockIdx_x) * BLOCKSIZE + tid;
        const int64_t stride = int64_t(hipGridDim_x) * BLOCKSIZE;

        unsigned long long local_max = 0;

        for(int64_t line = gid; line < nlines; line += stride)
        {
            const unsigned long long len
                = static_cast<unsigned long long>(offsets[line + 1] - offsets[line]);
            local_max = (len > local_max) ? len : local_max;
        }

        __shared__ unsigned long long s_max[BLOCKSIZE];
        s_max[tid] = local_max;
        __syncthreads();

        for(uint32_t s = BLOCKSIZE / 2; s > 0; s >>= 1)
        {
            if(tid < s)
            {
                s_max[tid] = (s_max[tid + s] > s_max[tid]) ? s_max[tid + s] : s_max[tid];
            }
            __syncthreads();
        }

        if(tid == 0)
        {
            atomicMax(d_max, s_max[0]);
        }
    }

    template <typename I>
    static rocsparse_status compute_line_nnz_profile_dispatch(rocsparse_handle  handle,
                                                              int64_t           nlines,
                                                              const I*          offsets,
                                                              line_nnz_profile& profile)
    {
        ROCSPARSE_ROUTINE_TRACE;

        constexpr uint32_t BLOCKSIZE = 256;

        // Scratch for the reduction result. The max identity is 0, so a plain
        // device memset seeds it correctly; read back with one synchronizing copy.
        unsigned long long* d_max = nullptr;
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync((void**)&d_max, sizeof(unsigned long long), handle->stream));
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(d_max, 0, sizeof(unsigned long long), handle->stream));

        int64_t nblocks64 = (nlines - 1) / BLOCKSIZE + 1;
        if(nblocks64 > 4096)
        {
            nblocks64 = 4096;
        }
        const uint32_t nblocks = static_cast<uint32_t>(nblocks64);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::line_nnz_profile_kernel<BLOCKSIZE, I>),
                                           dim3(nblocks),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           nlines,
                                           offsets,
                                           d_max);

        unsigned long long h_max = 0;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &h_max, d_max, sizeof(unsigned long long), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(d_max, handle->stream));

        profile.max = static_cast<int64_t>(h_max);

        return rocsparse_status_success;
    }
}

rocsparse_status rocsparse::compute_line_nnz_profile(rocsparse_handle             handle,
                                                     rocsparse_indextype          offsets_indextype,
                                                     int64_t                      nlines,
                                                     int64_t                      nnz,
                                                     const void*                  offsets,
                                                     rocsparse::line_nnz_profile& profile)
{
    ROCSPARSE_ROUTINE_TRACE;

    // Already computed for this descriptor: nothing to do.
    if(profile.known)
    {
        return rocsparse_status_success;
    }

    if(nlines <= 0 || nnz <= 0 || offsets == nullptr)
    {
        return rocsparse_status_success;
    }

    // The reduction below launches a kernel and copies the result back to the
    // host (a synchronizing operation), which is illegal while the stream is
    // captured into a HIP graph. Skip it under capture and leave the profile
    // uncomputed so callers fall back to their capture-safe default.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    RETURN_IF_HIP_ERROR(hipStreamIsCapturing(handle->stream, &capture_status));
    if(capture_status != hipStreamCaptureStatusNone)
    {
        return rocsparse_status_success;
    }

    switch(offsets_indextype)
    {
    case rocsparse_indextype_i32:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::compute_line_nnz_profile_dispatch<int32_t>(
            handle, nlines, static_cast<const int32_t*>(offsets), profile));
        break;
    }
    case rocsparse_indextype_i64:
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::compute_line_nnz_profile_dispatch<int64_t>(
            handle, nlines, static_cast<const int64_t*>(offsets), profile));
        break;
    }
    case deprecated_rocsparse_indextype_u16:
    {
        // Not a valid offsets type; leave the profile uncomputed.
        return rocsparse_status_success;
    }
    }

    profile.nnz   = nnz;
    profile.known = true;

    return rocsparse_status_success;
}
