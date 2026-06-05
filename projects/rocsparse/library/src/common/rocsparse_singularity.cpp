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

#include "rocsparse_singularity.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_logging.hpp"

namespace rocsparse
{

    template <uint32_t BLOCKSIZE, typename I>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void markers2singularity(int64_t batch_count,
                             const void* __restrict__ symbolic_,
                             const void* __restrict__ exact_,
                             const void* __restrict__ near_,
                             void* __restrict__ s_)
    {
        rocsparse_singularity* __restrict__ s
            = reinterpret_cast<rocsparse_singularity* __restrict__>(s_);
        static constexpr int64_t mx    = std::numeric_limits<I>::max();
        const I* __restrict__ symbolic = reinterpret_cast<const I* __restrict__>(symbolic_);
        const I* __restrict__ exact    = reinterpret_cast<const I* __restrict__>(exact_);
        const I* __restrict__ near     = reinterpret_cast<const I* __restrict__>(near_);
        const auto tid                 = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(tid < batch_count)
        {
            auto singularity = rocsparse_singularity_none;
            auto value       = mx;

            if((symbolic != nullptr) && (symbolic[0] != mx))
            {
                singularity = rocsparse_singularity_symbolic;
                value       = symbolic[0];
            }

            if(exact != nullptr)
            {
                if(exact[tid] < value)
                {
                    singularity = rocsparse_singularity_numeric_exact;
                    value       = exact[tid];
                }
            }

            if(near != nullptr)
            {
                if(near[tid] < value)
                {
                    singularity = rocsparse_singularity_numeric_near;
                    value       = near[tid];
                }
            }

            s[tid] = singularity;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void markers2position(int64_t batch_count,
                          const void* __restrict__ symbolic_,
                          const void* __restrict__ exact_,
                          const void* __restrict__ near_,
                          void* __restrict__ s_)
    {
        J* __restrict__ s              = reinterpret_cast<J* __restrict__>(s_);
        static constexpr I mx          = std::numeric_limits<I>::max();
        const I* __restrict__ symbolic = reinterpret_cast<const I* __restrict__>(symbolic_);
        const I* __restrict__ exact    = reinterpret_cast<const I* __restrict__>(exact_);
        const I* __restrict__ near     = reinterpret_cast<const I* __restrict__>(near_);
        const auto tid                 = hipBlockIdx_x * BLOCKSIZE + hipThreadIdx_x;
        if(tid < batch_count)
        {
            auto value = mx;
            if(symbolic != nullptr)
                value = symbolic[0];
            if(exact != nullptr)
                value = std::min(value, exact[tid]);
            if(near != nullptr)
                value = std::min(value, near[tid]);
            s[tid] = (value != mx) ? value : -1;
        }
    }

    static rocsparse_status singularity_get_async(bool                determine_singularity,
                                                  int64_t             batch_count,
                                                  rocsparse_indextype indextype,
                                                  const void* __restrict__ symbolic_,
                                                  const void* __restrict__ exact_,
                                                  const void* __restrict__ near_,
                                                  rocsparse_pointer_mode pointer_mode,
                                                  rocsparse_indextype    data_indextype,
                                                  void*                  data_,
                                                  size_t                 buffer_size_in_bytes,
                                                  void*                  buffer,
                                                  hipStream_t            stream)
    {
        ROCSPARSE_ROUTINE_TRACE;
        static constexpr uint32_t s_blocksize = 1024;
        size_t                    sizeofelm   = sizeof(int64_t);
        auto kernel = rocsparse::markers2position<s_blocksize, int64_t, int64_t>;

        if(determine_singularity)
        {
            sizeofelm = sizeof(rocsparse_singularity);
            kernel    = rocsparse::markers2singularity<s_blocksize, int64_t>;
            if(indextype == rocsparse_indextype_i32)
                kernel = rocsparse::markers2singularity<s_blocksize, int32_t>;
        }
        else
        {
            if(data_indextype == rocsparse_indextype_i32)
            {
                sizeofelm = sizeof(int32_t);
                kernel    = rocsparse::markers2position<s_blocksize, int64_t, int32_t>;
                if(indextype == rocsparse_indextype_i32)
                    kernel = rocsparse::markers2position<s_blocksize, int32_t, int32_t>;
            }
            else if(data_indextype == rocsparse_indextype_i64)
            {
                sizeofelm = sizeof(int64_t);
                kernel    = rocsparse::markers2position<s_blocksize, int64_t, int64_t>;
                if(indextype == rocsparse_indextype_i32)
                    kernel = rocsparse::markers2position<s_blocksize, int32_t, int64_t>;
            }
        }

        switch(pointer_mode)
        {
        case rocsparse_pointer_mode_host:
        {
            //
            // Let's use the internal buffer from the handle.
            //
            const char*   p_exact    = reinterpret_cast<const char*>(exact_);
            const char*   p_near     = reinterpret_cast<const char*>(near_);
            char*         data       = reinterpret_cast<char*>(data_);
            const int64_t m          = buffer_size_in_bytes / sizeofelm;
            const size_t  data_shift = m * sizeofelm;
            const size_t  p_exact_shift
                = (p_exact != nullptr) ? (m * rocsparse::indextype_sizeof(indextype)) : 0;
            const size_t p_near_shift
                = (p_near != nullptr) ? (m * rocsparse::indextype_sizeof(indextype)) : 0;
            const int64_t nfullgroups = batch_count / m;
            // Check how many rocsparse_singularity the handle can handle.
            //
            for(int64_t igroup = 0; igroup < nfullgroups; ++igroup)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((kernel),
                                                   dim3((m - 1) / s_blocksize + 1),
                                                   dim3(s_blocksize),
                                                   0,
                                                   stream,
                                                   m,
                                                   symbolic_,
                                                   p_exact + igroup * p_exact_shift,
                                                   p_near + igroup * p_near_shift,
                                                   buffer);

                RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                    data + igroup * data_shift, buffer, sizeofelm * m, hipMemcpyDefault, stream));
            }

            const int64_t rem = batch_count % m;
            if(rem > 0)
            {
                RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((kernel),
                                                   dim3((rem - 1) / s_blocksize + 1),
                                                   dim3(s_blocksize),
                                                   0,
                                                   stream,
                                                   rem,
                                                   symbolic_,
                                                   p_exact + nfullgroups * p_exact_shift,
                                                   p_near + nfullgroups * p_near_shift,
                                                   buffer);
                RETURN_IF_HIP_ERROR(hipMemcpyAsync(data + nfullgroups * data_shift,
                                                   buffer,
                                                   sizeofelm * rem,
                                                   hipMemcpyDefault,
                                                   stream));
            }

            return rocsparse_status_success;
        }

        case rocsparse_pointer_mode_device:
        {
            RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((kernel),
                                               dim3((batch_count - 1) / s_blocksize + 1),
                                               dim3(s_blocksize),
                                               0,
                                               stream,
                                               batch_count,
                                               symbolic_,
                                               exact_,
                                               near_,
                                               data_);
            return rocsparse_status_success;
        }
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    }

    rocsparse_status singularity_get_position_async(rocsparse_handle                  handle,
                                                    int64_t                           batch_count,
                                                    const rocsparse::pivot_info_t*    symbolic,
                                                    const rocsparse::singular_info_t* exact,
                                                    const rocsparse::singular_info_t* near,
                                                    rocsparse_pointer_mode position_pointer_mode,
                                                    rocsparse_indextype    position_indextype,
                                                    void*                  position)
    {
        const int64_t near_batch_count  = (near != nullptr) ? near->get_batch_count() : 0;
        const int64_t exact_batch_count = (exact != nullptr) ? exact->get_batch_count() : 0;
        if(near && near->get_position())
        {
            if(batch_count != near_batch_count)
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                       "incompatible near batch count");
            }
        }
        if(exact && exact->get_position())
        {
            if(batch_count != exact_batch_count)
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                       "incompatible exact batch_count");
            }
        }

        rocsparse_indextype indextype = rocsparse_indextype_i32;
        if(symbolic && symbolic->get_position())
        {
            indextype = symbolic->get_indextype();
        }
        else if(exact && exact->get_position())
        {
            indextype = exact->get_indextype();
        }
        else if(near && near->get_position())
        {
            indextype = near->get_indextype();
        }

        if(symbolic && symbolic->get_position())
        {
            if(indextype != symbolic->get_indextype())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }
        if(exact && exact->get_position())
        {
            if(indextype != exact->get_indextype())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }
        if(near && near->get_position())
        {
            if(indextype != near->get_indextype())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::singularity_get_async(false,
                                             batch_count,
                                             indextype,
                                             (symbolic) ? symbolic->get_position() : nullptr,
                                             (exact) ? exact->get_position() : nullptr,
                                             (near) ? near->get_position() : nullptr,
                                             position_pointer_mode,
                                             position_indextype,
                                             position,
                                             handle->buffer_size,
                                             handle->buffer,
                                             handle->stream));
        return rocsparse_status_success;
    }

    rocsparse_status singularity_get_async(rocsparse_handle                  handle,
                                           int64_t                           batch_count,
                                           const rocsparse::pivot_info_t*    symbolic,
                                           const rocsparse::singular_info_t* exact,
                                           const rocsparse::singular_info_t* near,
                                           rocsparse_pointer_mode singularity_pointer_mode,
                                           void*                  singularity)
    {

        const int64_t near_batch_count  = (near) ? near->get_batch_count() : 0;
        const int64_t exact_batch_count = (exact) ? exact->get_batch_count() : 0;
        if(near && near->get_position())
        {
            if(batch_count != near_batch_count)
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                       "incompatible near batch count");
            }
        }
        if(exact && exact->get_position())
        {
            if(batch_count != exact_batch_count)
            {
                RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                       "incompatible exact batch_count");
            }
        }

        rocsparse_indextype indextype = rocsparse_indextype_i32;
        if(symbolic && symbolic->get_position())
        {
            indextype = symbolic->get_indextype();
        }
        else if(exact && exact->get_position())
        {
            indextype = exact->get_indextype();
        }
        else if(near && near->get_position())
        {
            indextype = near->get_indextype();
        }

        if(symbolic && symbolic->get_position())
        {
            if(indextype != symbolic->get_indextype())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }
        if(exact && exact->get_position())
        {
            if(indextype != exact->get_indextype())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }
        if(near && near->get_position())
        {
            if(indextype != near->get_indextype())
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
            }
        }

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::singularity_get_async(true,
                                             batch_count,
                                             indextype,
                                             (symbolic) ? symbolic->get_position() : nullptr,
                                             (exact) ? exact->get_position() : nullptr,
                                             (near) ? near->get_position() : nullptr,
                                             singularity_pointer_mode,
                                             indextype, // not used
                                             singularity,
                                             handle->buffer_size,
                                             handle->buffer,
                                             handle->stream));
        return rocsparse_status_success;
    }

}
