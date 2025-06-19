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

template <typename K>
rocsparse_status rocsparse::primitives::radix_sort_keys_buffer_size(
    rocsparse_handle handle, size_t length, uint32_t startbit, uint32_t endbit, size_t* buffer_size)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocprim::double_buffer<K> rocprim_keys(nullptr, nullptr);
    RETURN_IF_HIP_ERROR(rocprim::radix_sort_keys(
        nullptr, *buffer_size, rocprim_keys, length, startbit, endbit, handle->stream));

    return rocsparse_status_success;
}

template <typename K>
rocsparse_status rocsparse::primitives::radix_sort_keys(rocsparse_handle  handle,
                                                        double_buffer<K>& keys,
                                                        size_t            length,
                                                        uint32_t          startbit,
                                                        uint32_t          endbit,
                                                        size_t            buffer_size,
                                                        void*             buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    rocprim::double_buffer<K> rocprim_keys(keys.current(), keys.alternate());

    RETURN_IF_HIP_ERROR(rocprim::radix_sort_keys(
        buffer, buffer_size, rocprim_keys, length, startbit, endbit, handle->stream));

    if(keys.current() != rocprim_keys.current())
    {
        keys.swap();
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(KTYPE)                                                                           \
    template rocsparse_status rocsparse::primitives::radix_sort_keys_buffer_size<KTYPE>(             \
        rocsparse_handle handle,                                                                     \
        size_t           length,                                                                     \
        uint32_t         startbit,                                                                   \
        uint32_t         endbit,                                                                     \
        size_t * buffer_size);                                                                       \
    template rocsparse_status rocsparse::primitives::radix_sort_keys(rocsparse_handle      handle,   \
                                                                     double_buffer<KTYPE>& keys,     \
                                                                     size_t                length,   \
                                                                     uint32_t              startbit, \
                                                                     uint32_t              endbit,   \
                                                                     size_t buffer_size,             \
                                                                     void*  buffer);

INSTANTIATE(float);
INSTANTIATE(double);
#undef INSTANTIATE
