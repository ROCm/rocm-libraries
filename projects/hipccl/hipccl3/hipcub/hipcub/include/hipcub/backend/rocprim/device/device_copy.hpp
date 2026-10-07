/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2024-2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_ROCPRIM_DEVICE_DEVICE_COPY_HPP_
#define HIPCUB_ROCPRIM_DEVICE_DEVICE_COPY_HPP_

#include "../../../config.hpp"
#include "./device_for.hpp"
#include "./device_transform.hpp"

#include <rocprim/device/device_copy.hpp> // IWYU pragma: export

#include <hip/hip_runtime.h>

#include _HIPCUB_LIBCXX_INCLUDE(functional)
#include _HIPCUB_STD_INCLUDE(mdspan)
#include _HIPCUB_STD_INCLUDE(type_traits)

BEGIN_HIPCUB_NAMESPACE

namespace detail
{
template<typename MappingLhsType, typename MappingRhsType>
[[nodiscard]]
HIPCUB_HOST
bool have_same_strides(const MappingLhsType& mapping_lhs, const MappingRhsType& mapping_rhs)
{
    static_assert(MappingLhsType::extents_type::rank() == MappingRhsType::extents_type::rank(),
                  "Mappings must have the same rank");
    constexpr auto ndimensions = MappingLhsType::extents_type::rank();
    for(size_t i = 0; i < ndimensions; ++i)
    {
        if(mapping_lhs.stride(i) != mapping_rhs.stride(i))
        {
            return false;
        }
    }
    return true;
}

namespace copy_mdspan
{

template<typename MdspanIn, typename MdspanOut>
struct copy_mdspan_t
{
    MdspanIn  mdspan_in;
    MdspanOut mdspan_out;

    HIPCUB_HOST_DEVICE
    copy_mdspan_t(MdspanIn mdspan_in, MdspanOut mdspan_out)
        : mdspan_in{mdspan_in}, mdspan_out{mdspan_out}
    {}

    template<typename Idx, typename... Indices>
  HIPCUB_DEVICE
    HIPCUB_FORCEINLINE void operator()(Idx, Indices... indices)
    {
        mdspan_out(indices...) = mdspan_in(indices...);
    }
};

template<typename T_In,
         typename E_In,
         typename L_In,
         typename A_In,
         typename T_Out,
         typename E_Out,
         typename L_Out,
         typename A_Out>
[[nodiscard]]
HIPCUB_RUNTIME_FUNCTION
HIPCUB_FORCEINLINE hipError_t copy(_HIPCUB_STD::mdspan<T_In, E_In, L_In, A_In>     mdspan_in,
                                   _HIPCUB_STD::mdspan<T_Out, E_Out, L_Out, A_Out> mdspan_out,
                                   hipStream_t                                     stream)
{
    if(mdspan_in.is_exhaustive() && mdspan_out.is_exhaustive() && mdspan_in.is_unique()
       && mdspan_out.is_unique()
       && detail::have_same_strides(mdspan_in.mapping(), mdspan_out.mapping()))
    {
        return DeviceTransform::Transform(
            mdspan_in.data_handle(),
            mdspan_out.data_handle(),
            mdspan_in.size(),
            _HIPCUB_LIBCXX::proclaim_copyable_arguments(_HIPCUB_STD::identity{}),
            stream);
    }
    // Note: TODO compatible layouts may use ForEachInLayout
    // return DeviceFor::ForEachInLayout(mdspan_in.mapping(),
    //                                   copy_mdspan_t{mdspan_in, mdspan_out},
    //                                   stream);
    return DeviceFor::ForEachInExtents(mdspan_in.extents(),
                                       copy_mdspan_t{mdspan_in, mdspan_out},
                                       stream);
}
} // namespace copy_mdspan
} // namespace detail

struct DeviceCopy
{
    template<typename InputBufferIt, typename OutputBufferIt, typename BufferSizeIteratorT>
    static hipError_t Batched(void*                d_temp_storage,
                              size_t&              temp_storage_bytes,
                              InputBufferIt        input_buffer_it,
                              OutputBufferIt       output_buffer_it,
                              BufferSizeIteratorT  buffer_sizes,
                              _HIPCUB_STD::int64_t num_buffers,
                              hipStream_t          stream = 0)
    {
        return rocprim::batch_copy(d_temp_storage,
                                   temp_storage_bytes,
                                   input_buffer_it,
                                   output_buffer_it,
                                   buffer_sizes,
                                   num_buffers,
                                   stream,
                                   HIPCUB_DETAIL_DEBUG_SYNC_VALUE);
    }

    template<typename T_In,
             typename Extents_In,
             typename Layout_In,
             typename Accessor_In,
             typename T_Out,
             typename Extents_Out,
             typename Layout_Out,
             typename Accessor_Out>
    [[nodiscard]]
    HIPCUB_RUNTIME_FUNCTION
    static hipError_t
        Copy(void*                                                             d_temp_storage,
             size_t&                                                           temp_storage_bytes,
             _HIPCUB_STD::mdspan<T_In, Extents_In, Layout_In, Accessor_In>     mdspan_in,
             _HIPCUB_STD::mdspan<T_Out, Extents_Out, Layout_Out, Accessor_Out> mdspan_out,
             hipStream_t                                                       stream = nullptr)
    {
        HIPCUB_ASSERT(mdspan_in.extents() == mdspan_out.extents(),
                      "input and output mdspan extents must be equal");
        if(mdspan_in.size() != 0)
        {
            HIPCUB_ASSERT(mdspan_in.data_handle() != nullptr,
                          "input mdspan data handle must not be nullptr if the size is not 0");
            HIPCUB_ASSERT(mdspan_out.data_handle() != nullptr,
                          "output mdspan data handle must not be nullptr if the size is not 0");
            auto in_start  = mdspan_in.data_handle();
            auto out_start = mdspan_out.data_handle();
            [[maybe_unused]]
            auto in_end
                = in_start + mdspan_in.mapping().required_span_size();
            [[maybe_unused]]
            auto out_end
                = out_start + mdspan_out.mapping().required_span_size();
            HIPCUB_ASSERT((in_end <= out_start) || (out_end <= in_start),
                          "input and output mdspan data ranges must not overlap");
        }
        if(d_temp_storage == nullptr)
        {
            temp_storage_bytes = 1;
            return hipSuccess;
        }
        return detail::copy_mdspan::copy(mdspan_in, mdspan_out, stream);
    }
};

END_HIPCUB_NAMESPACE

#endif // HIPCUB_ROCPRIM_DEVICE_DEVICE_COPY_HPP_
