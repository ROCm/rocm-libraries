/*! \file */
/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "control.h"
#include "rocsparse_primitives.h"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <typename K, typename I>
rocsparse_status
    rocsparse::primitives::segmented_radix_sort_keys_buffer_size(rocsparse_handle handle,
                                                                 size_t           length,
                                                                 size_t           segments,
                                                                 uint32_t         startbit,
                                                                 uint32_t         endbit,
                                                                 size_t*          buffer_size)
{
    using config
        = rocprim::segmented_radix_sort_config<7,
                                               4,
                                               rocprim::kernel_config<256, 16>,
                                               rocprim::WarpSortConfig<8, 8, 256, 5, 16, 16, 256>,
                                               1>;

    rocprim::double_buffer<K> rocprim_keys(nullptr, nullptr);

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(nullptr,
                                                                   *buffer_size,
                                                                   rocprim_keys,
                                                                   length,
                                                                   segments,
                                                                   (I*)nullptr,
                                                                   (I*)nullptr,
                                                                   startbit,
                                                                   endbit,
                                                                   handle->stream));

    return rocsparse_status_success;
}

template <typename K, typename I>
rocsparse_status rocsparse::primitives::segmented_radix_sort_keys(rocsparse_handle  handle,
                                                                  double_buffer<K>& keys,
                                                                  size_t            length,
                                                                  size_t            segments,
                                                                  I*                begin_offsets,
                                                                  I*                end_offsets,
                                                                  uint32_t          startbit,
                                                                  uint32_t          endbit,
                                                                  size_t            buffer_size,
                                                                  void*             buffer)
{
    rocprim::double_buffer<K> rocprim_keys(keys.current(), keys.alternate());

    using config
        = rocprim::segmented_radix_sort_config<7,
                                               4,
                                               rocprim::kernel_config<256, 16>,
                                               rocprim::WarpSortConfig<8, 8, 256, 5, 16, 16, 256>,
                                               1>;

    RETURN_IF_HIP_ERROR(rocprim::segmented_radix_sort_keys<config>(buffer,
                                                                   buffer_size,
                                                                   rocprim_keys,
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

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::sort_csr_column_indices_buffer_size(
    rocsparse_handle handle, J m, J n, I nnz, const I* csr_row_ptr, size_t* buffer_size)
{
    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(n);

    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
        handle, nnz, m, startbit, endbit, buffer_size)));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::sort_csr_column_indices(rocsparse_handle handle,
                                                                J                m,
                                                                J                n,
                                                                I                nnz,
                                                                const I*         csr_row_ptr,
                                                                const J*         csr_col_ind,
                                                                J*    csr_col_ind_buffer1,
                                                                J*    csr_col_ind_buffer2,
                                                                void* buffer)
{
    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(n);

    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csr_col_ind_buffer1, csr_col_ind, sizeof(J) * nnz, hipMemcpyDeviceToDevice));

    rocsparse::primitives::double_buffer<J> indices(csr_col_ind_buffer1, csr_col_ind_buffer2);

    size_t buffer_size;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
        handle, nnz, m, startbit, endbit, &buffer_size)));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(handle,
                                                                               indices,
                                                                               nnz,
                                                                               m,
                                                                               csr_row_ptr,
                                                                               csr_row_ptr + 1,
                                                                               startbit,
                                                                               endbit,
                                                                               buffer_size,
                                                                               buffer));

    if(indices.current() != csr_col_ind_buffer2)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csr_col_ind_buffer2,
                                           indices.current(),
                                           sizeof(J) * nnz,
                                           hipMemcpyDeviceToDevice,
                                           handle->stream));
    }

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::sort_csr_column_indices(rocsparse_handle handle,
                                                                J                m,
                                                                J                n,
                                                                I                nnz,
                                                                const I*         csr_row_ptr,
                                                                J*               csr_col_ind,
                                                                J*    csr_col_ind_buffer2,
                                                                void* buffer)
{
    uint32_t startbit = 0;
    uint32_t endbit   = rocsparse::clz(n);

    rocsparse::primitives::double_buffer<J> indices(csr_col_ind, csr_col_ind_buffer2);

    size_t buffer_size;
    RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, I>(
        handle, nnz, m, startbit, endbit, &buffer_size)));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::segmented_radix_sort_keys(handle,
                                                                               indices,
                                                                               nnz,
                                                                               m,
                                                                               csr_row_ptr,
                                                                               csr_row_ptr + 1,
                                                                               startbit,
                                                                               endbit,
                                                                               buffer_size,
                                                                               buffer));

    if(indices.current() != csr_col_ind)
    {
        RETURN_IF_HIP_ERROR(hipMemcpyAsync(csr_col_ind,
                                           indices.current(),
                                           sizeof(J) * nnz,
                                           hipMemcpyDeviceToDevice,
                                           handle->stream));
    }

    return rocsparse_status_success;
}

#define INSTANTIATE(KTYPE, ITYPE)                                                   \
    template rocsparse_status                                                       \
        rocsparse::primitives::segmented_radix_sort_keys_buffer_size<KTYPE, ITYPE>( \
            rocsparse_handle handle,                                                \
            size_t           length,                                                \
            size_t           segments,                                              \
            uint32_t         startbit,                                              \
            uint32_t         endbit,                                                \
            size_t * buffer_size);                                                  \
    template rocsparse_status rocsparse::primitives::segmented_radix_sort_keys(     \
        rocsparse_handle      handle,                                               \
        double_buffer<KTYPE>& keys,                                                 \
        size_t                length,                                               \
        size_t                segments,                                             \
        ITYPE*                begin_offsets,                                        \
        ITYPE*                end_offsets,                                          \
        uint32_t              startbit,                                             \
        uint32_t              endbit,                                               \
        size_t                buffer_size,                                          \
        void*                 buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE)                                                         \
    template rocsparse_status rocsparse::primitives::sort_csr_column_indices_buffer_size( \
        rocsparse_handle handle,                                                          \
        JTYPE            m,                                                               \
        JTYPE            n,                                                               \
        ITYPE            nnz,                                                             \
        const ITYPE*     csr_row_ptr,                                                     \
        size_t*          buffer_size);                                                             \
    template rocsparse_status rocsparse::primitives::sort_csr_column_indices(             \
        rocsparse_handle handle,                                                          \
        JTYPE            m,                                                               \
        JTYPE            n,                                                               \
        ITYPE            nnz,                                                             \
        const ITYPE*     csr_row_ptr,                                                     \
        const JTYPE*     csr_col_ind,                                                     \
        JTYPE*           csr_col_ind_buffer1,                                             \
        JTYPE*           csr_col_ind_buffer2,                                             \
        void*            buffer);                                                                    \
    template rocsparse_status rocsparse::primitives::sort_csr_column_indices(             \
        rocsparse_handle handle,                                                          \
        JTYPE            m,                                                               \
        JTYPE            n,                                                               \
        ITYPE            nnz,                                                             \
        const ITYPE*     csr_row_ptr,                                                     \
        JTYPE*           csr_col_ind,                                                     \
        JTYPE*           csr_col_ind_buffer2,                                             \
        void*            buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE
