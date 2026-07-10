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
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_hip.hpp"
#include "rocsparse_utility.hpp"

namespace rocsparse
{
    // The maximum line length is computed with a classic two-pass, atomic-free
    // reduction. The line length len[i] = offsets[i+1] - offsets[i] is
    // base-independent (a difference of consecutive offsets).
    //
    // Pass 1 launches a fixed grid of BLOCKSIZE blocks of BLOCKSIZE threads.
    // Each block grid-strides over the lines, reduces its share to a single
    // block-wide maximum, and writes that one value to workspace[blockIdx.x].
    // Pass 2 launches a single block that reduces the BLOCKSIZE partials down to
    // the final maximum. No thread ever contends on a shared global location,
    // which avoids the atomic serialization of a one-atomic-per-block scheme.
    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void line_nnz_profile_part1(int64_t nlines,
                                const I* __restrict__ offsets,
                                uint64_t* __restrict__ workspace)
    {
        const int     tid    = hipThreadIdx_x;
        const int64_t gid    = int64_t(hipBlockIdx_x) * BLOCKSIZE + tid;
        const int64_t stride = int64_t(hipGridDim_x) * BLOCKSIZE;

        uint64_t local_max = 0;
        for(int64_t line = gid; line < nlines; line += stride)
        {
            const uint64_t len = static_cast<uint64_t>(offsets[line + 1] - offsets[line]);
            local_max          = rocsparse::max(local_max, len);
        }

        __shared__ uint64_t shared[BLOCKSIZE];
        shared[tid] = local_max;
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            workspace[hipBlockIdx_x] = shared[0];
        }
    }

    template <uint32_t BLOCKSIZE>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void line_nnz_profile_part2(uint64_t* __restrict__ workspace)
    {
        const int tid = hipThreadIdx_x;

        __shared__ uint64_t shared[BLOCKSIZE];
        shared[tid] = workspace[tid];
        __syncthreads();

        rocsparse::blockreduce_max<BLOCKSIZE>(tid, shared);

        if(tid == 0)
        {
            workspace[0] = shared[0];
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

        // Pass 1 writes exactly BLOCKSIZE partial maxima (one per block), so the
        // workspace holds BLOCKSIZE entries; every entry is written by pass 1
        // (idle blocks write the max identity 0), so no pre-seeding is needed.
        uint64_t* workspace = nullptr;
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(
            (void**)&workspace, sizeof(uint64_t) * BLOCKSIZE, handle->stream));

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::line_nnz_profile_part1<BLOCKSIZE, I>),
                                           dim3(BLOCKSIZE),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           nlines,
                                           offsets,
                                           workspace);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::line_nnz_profile_part2<BLOCKSIZE>),
                                           dim3(1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           workspace);

        uint64_t h_max = 0;
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(
            &h_max, workspace, sizeof(uint64_t), hipMemcpyDeviceToHost, handle->stream));
        RETURN_IF_HIP_ERROR(hipStreamSynchronize(handle->stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(workspace, handle->stream));

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

    // The reduction ends with a synchronizing device->host copy, which is illegal
    // while the stream is being captured into a HIP graph and would invalidate the
    // capture. Callers (e.g. hipSPARSE's graph-capture tests) may run the analysis
    // stage under capture, so guard here rather than relying on the caller: leave
    // the profile unknown so the selector falls back to the capture-safe default
    // (row-split). A captured graph therefore stays valid, at the cost of not
    // upgrading skewed matrices while capturing.
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
