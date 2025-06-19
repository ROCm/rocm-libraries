/*! \file */
/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_utility.hpp"

#include <rocprim/rocprim.hpp>

template <typename K, typename V, typename I>
rocsparse_status
    rocsparse::primitives::segmented_radix_sort_pairs_buffer_size(rocsparse_handle handle,
                                                                  size_t           length,
                                                                  size_t           segments,
                                                                  uint32_t         startbit,
                                                                  uint32_t         endbit,
                                                                  size_t*          buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    using config
        = rocprim::segmented_radix_sort_config<7,
                                               rocprim::kernel_config<256, 16>,
                                               rocprim::WarpSortConfig<8, 8, 256, 5, 16, 16, 256>,
                                               1>;
    rocprim::double_buffer<K> rocprim_keys(nullptr, nullptr);
    rocprim::double_buffer<V> rocprim_values(nullptr, nullptr);

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(nullptr,
                                                                    *buffer_size,
                                                                    rocprim_keys,
                                                                    rocprim_values,
                                                                    length,
                                                                    segments,
                                                                    (I*)nullptr,
                                                                    (I*)nullptr,
                                                                    startbit,
                                                                    endbit,
                                                                    handle->stream));
    return rocsparse_status_success;
}

template <typename K, typename V, typename I>
rocsparse_status rocsparse::primitives::segmented_radix_sort_pairs(rocsparse_handle  handle,
                                                                   double_buffer<K>& keys,
                                                                   double_buffer<V>& values,
                                                                   size_t            length,
                                                                   size_t            segments,
                                                                   I*                begin_offsets,
                                                                   I*                end_offsets,
                                                                   uint32_t          startbit,
                                                                   uint32_t          endbit,
                                                                   size_t            buffer_size,
                                                                   void*             buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocprim::double_buffer<K> rocprim_keys(keys.current(), keys.alternate());
    rocprim::double_buffer<V> rocprim_values(values.current(), values.alternate());

    using config
        = rocprim::segmented_radix_sort_config<7,
                                               rocprim::kernel_config<256, 16>,
                                               rocprim::WarpSortConfig<8, 8, 256, 5, 16, 16, 256>,
                                               1>;

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_pairs<config>(buffer,
                                                                    buffer_size,
                                                                    rocprim_keys,
                                                                    rocprim_values,
                                                                    length,
                                                                    segments,
                                                                    begin_offsets,
                                                                    end_offsets,
                                                                    startbit,
                                                                    endbit,
                                                                    handle->stream));
    if(keys.current() != rocprim_keys.current())
    {
        keys.swap();
    }
    if(values.current() != rocprim_values.current())
    {
        values.swap();
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(KTYPE, VTYPE, ITYPE)                                                    \
    template rocsparse_status                                                               \
        rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<KTYPE, VTYPE, ITYPE>( \
            rocsparse_handle handle,                                                        \
            size_t           length,                                                        \
            size_t           segments,                                                      \
            uint32_t         startbit,                                                      \
            uint32_t         endbit,                                                        \
            size_t * buffer_size);                                                          \
    template rocsparse_status rocsparse::primitives::segmented_radix_sort_pairs(            \
        rocsparse_handle      handle,                                                       \
        double_buffer<KTYPE>& keys,                                                         \
        double_buffer<VTYPE>& values,                                                       \
        size_t                length,                                                       \
        size_t                segments,                                                     \
        ITYPE*                begin_offsets,                                                \
        ITYPE*                end_offsets,                                                  \
        uint32_t              startbit,                                                     \
        uint32_t              endbit,                                                       \
        size_t                buffer_size,                                                  \
        void*                 buffer);

INSTANTIATE(int32_t, int32_t, const int32_t);
INSTANTIATE(int64_t, int64_t, const int64_t);
INSTANTIATE(int32_t, int32_t, int32_t);
INSTANTIATE(int64_t, int64_t, int64_t);
#undef INSTANTIATE
