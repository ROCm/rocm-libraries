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

template <typename K, typename V>
rocsparse_status rocsparse::primitives::radix_sort_pairs_buffer_size(rocsparse_handle handle,
                                                                     size_t           length,
                                                                     uint32_t         startbit,
                                                                     uint32_t         endbit,
                                                                     size_t*          buffer_size,
                                                                     bool using_double_buffers)
{
    ROCSPARSE_ROUTINE_TRACE;

    K* ptr1 = reinterpret_cast<K*>(0x4);
    V* ptr2 = reinterpret_cast<V*>(0x4);

    if(using_double_buffers)
    {
        rocprim::double_buffer<K> rocprim_keys(ptr1, ptr1);
        rocprim::double_buffer<V> rocprim_values(ptr2, ptr2);
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(nullptr,
                                                      *buffer_size,
                                                      rocprim_keys,
                                                      rocprim_values,
                                                      length,
                                                      startbit,
                                                      endbit,
                                                      handle->stream));
    }
    else
    {
        RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(nullptr,
                                                      *buffer_size,
                                                      ptr1,
                                                      ptr1,
                                                      ptr2,
                                                      ptr2,
                                                      length,
                                                      startbit,
                                                      endbit,
                                                      handle->stream));
    }

    return rocsparse_status_success;
}

template <typename K, typename V>
rocsparse_status rocsparse::primitives::radix_sort_pairs(rocsparse_handle  handle,
                                                         double_buffer<K>& keys,
                                                         double_buffer<V>& values,
                                                         size_t            length,
                                                         uint32_t          startbit,
                                                         uint32_t          endbit,
                                                         size_t            buffer_size,
                                                         void*             buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocprim::double_buffer<K> rocprim_keys(keys.current(), keys.alternate());
    rocprim::double_buffer<V> rocprim_values(values.current(), values.alternate());
    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(buffer,
                                                  buffer_size,
                                                  rocprim_keys,
                                                  rocprim_values,
                                                  length,
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

template <typename K, typename V>
rocsparse_status rocsparse::primitives::radix_sort_pairs(rocsparse_handle handle,
                                                         K*               keys_input,
                                                         K*               keys_output,
                                                         V*               values_input,
                                                         V*               values_output,
                                                         size_t           length,
                                                         uint32_t         startbit,
                                                         uint32_t         endbit,
                                                         size_t           buffer_size,
                                                         void*            buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_HIP_ERROR(rocprim::radix_sort_pairs(buffer,
                                                  buffer_size,
                                                  keys_input,
                                                  keys_output,
                                                  values_input,
                                                  values_output,
                                                  length,
                                                  startbit,
                                                  endbit,
                                                  handle->stream));

    return rocsparse_status_success;
}

#define INSTANTIATE(KTYPE, VTYPE)                                                                \
    template rocsparse_status rocsparse::primitives::radix_sort_pairs_buffer_size<KTYPE, VTYPE>( \
        rocsparse_handle handle,                                                                 \
        size_t           length,                                                                 \
        uint32_t         startbit,                                                               \
        uint32_t         endbit,                                                                 \
        size_t * buffer_size,                                                                    \
        bool using_double_buffers);                                                              \
    template rocsparse_status rocsparse::primitives::radix_sort_pairs(                           \
        rocsparse_handle      handle,                                                            \
        double_buffer<KTYPE>& keys,                                                              \
        double_buffer<VTYPE>& values,                                                            \
        size_t                length,                                                            \
        uint32_t              startbit,                                                          \
        uint32_t              endbit,                                                            \
        size_t                buffer_size,                                                       \
        void*                 buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int32_t, int64_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(KTYPE, VTYPE)                                                                    \
    template rocsparse_status rocsparse::primitives::radix_sort_pairs(rocsparse_handle handle,       \
                                                                      KTYPE*           keys_input,   \
                                                                      KTYPE*           keys_output,  \
                                                                      VTYPE*           values_input, \
                                                                      VTYPE*   values_output,        \
                                                                      size_t   length,               \
                                                                      uint32_t startbit,             \
                                                                      uint32_t endbit,               \
                                                                      size_t   buffer_size,          \
                                                                      void*    buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE
