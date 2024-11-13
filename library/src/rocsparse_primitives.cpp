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

#include "rocsparse_primitives.h"
#include "control.h"
#include "utility.h"

#include <rocprim/rocprim.hpp>

template <typename K, typename V>
rocsparse_status rocsparse::primitives::radix_sort_pairs_buffer_size(rocsparse_handle handle,
                                                                     size_t           length,
                                                                     uint32_t         startbit,
                                                                     uint32_t         endbit,
                                                                     size_t*          buffer_size,
                                                                     bool using_double_buffers)
{
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

template <typename K>
rocsparse_status rocsparse::primitives::radix_sort_keys_buffer_size(
    rocsparse_handle handle, size_t length, uint32_t startbit, uint32_t endbit, size_t* buffer_size)
{
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
    rocprim::double_buffer<K> rocprim_keys(keys.current(), keys.alternate());

    rocprim::radix_sort_keys(
        buffer, buffer_size, rocprim_keys, length, startbit, endbit, handle->stream);

    if(keys.current() != rocprim_keys.current())
    {
        keys.swap();
    }

    return rocsparse_status_success;
}

template <typename J>
rocsparse_status rocsparse::primitives::run_length_encode_buffer_size(rocsparse_handle handle,
                                                                      size_t           length,
                                                                      size_t*          buffer_size)
{
    RETURN_IF_HIP_ERROR(rocprim::run_length_encode(nullptr,
                                                   *buffer_size,
                                                   (J*)nullptr,
                                                   length,
                                                   (J*)nullptr,
                                                   (J*)nullptr,
                                                   (J*)nullptr,
                                                   handle->stream));

    return rocsparse_status_success;
}

template <typename J>
rocsparse_status rocsparse::primitives::run_length_encode(rocsparse_handle handle,
                                                          J*               input,
                                                          J*               unique_output,
                                                          J*               counts_output,
                                                          J*               runs_count_output,
                                                          size_t           length,
                                                          size_t           buffer_size,
                                                          void*            buffer)
{
    RETURN_IF_HIP_ERROR(rocprim::run_length_encode(buffer,
                                                   buffer_size,
                                                   input,
                                                   length,
                                                   unique_output,
                                                   counts_output,
                                                   runs_count_output,
                                                   handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::exclusive_scan_buffer_size(rocsparse_handle handle,
                                                                   J                initial_value,
                                                                   size_t           length,
                                                                   size_t*          buffer_size)
{
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(nullptr,
                                                *buffer_size,
                                                (I*)nullptr,
                                                (J*)nullptr,
                                                initial_value,
                                                length,
                                                rocprim::plus<J>(),
                                                handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::exclusive_scan(rocsparse_handle handle,
                                                       I*               input,
                                                       J*               output,
                                                       J                initial_value,
                                                       size_t           length,
                                                       size_t           buffer_size,
                                                       void*            buffer)
{
    RETURN_IF_HIP_ERROR(rocprim::exclusive_scan(buffer,
                                                buffer_size,
                                                input,
                                                output,
                                                initial_value,
                                                length,
                                                rocprim::plus<J>(),
                                                handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::inclusive_scan_buffer_size(rocsparse_handle handle,
                                                                   size_t           length,
                                                                   size_t*          buffer_size)
{
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(nullptr,
                                                *buffer_size,
                                                (I*)nullptr,
                                                (J*)nullptr,
                                                length,
                                                rocprim::plus<J>(),
                                                handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::inclusive_scan(
    rocsparse_handle handle, I* input, J* output, size_t length, size_t buffer_size, void* buffer)
{
    RETURN_IF_HIP_ERROR(rocprim::inclusive_scan(
        buffer, buffer_size, input, output, length, rocprim::plus<J>(), handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::find_max_buffer_size(rocsparse_handle handle,
                                                             size_t           length,
                                                             size_t*          buffer_size)
{
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        *buffer_size,
                                        (I*)nullptr,
                                        (J*)nullptr,
                                        length,
                                        rocprim::maximum<J>(),
                                        handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::find_max(
    rocsparse_handle handle, I* input, J* max, size_t length, size_t buffer_size, void* buffer)
{
    RETURN_IF_HIP_ERROR(rocprim::reduce(
        buffer, buffer_size, input, max, length, rocprim::maximum<J>(), handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::find_sum_buffer_size(rocsparse_handle handle,
                                                             size_t           length,
                                                             size_t*          buffer_size)
{
    RETURN_IF_HIP_ERROR(rocprim::reduce(nullptr,
                                        *buffer_size,
                                        (I*)nullptr,
                                        (J*)nullptr,
                                        length,
                                        rocprim::plus<J>(),
                                        handle->stream));

    return rocsparse_status_success;
}

template <typename I, typename J>
rocsparse_status rocsparse::primitives::find_sum(
    rocsparse_handle handle, I* input, J* sum, size_t length, size_t buffer_size, void* buffer)
{
    RETURN_IF_HIP_ERROR(rocprim::reduce(
        buffer, buffer_size, input, sum, length, rocprim::plus<J>(), handle->stream));

    return rocsparse_status_success;
}

template <typename K, typename V, typename I>
rocsparse_status
    rocsparse::primitives::segmented_radix_sort_pairs_buffer_size(rocsparse_handle handle,
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
    rocprim::double_buffer<K> rocprim_keys(keys.current(), keys.alternate());
    rocprim::double_buffer<V> rocprim_values(values.current(), values.alternate());

    using config
        = rocprim::segmented_radix_sort_config<7,
                                               4,
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

#define INSTANTIATE(JTYPE)                                                                       \
    template rocsparse_status rocsparse::primitives::run_length_encode_buffer_size<JTYPE>(       \
        rocsparse_handle handle, size_t length, size_t * buffer_size);                           \
    template rocsparse_status rocsparse::primitives::run_length_encode(rocsparse_handle handle,  \
                                                                       JTYPE*           input,   \
                                                                       JTYPE* unique_output,     \
                                                                       JTYPE* counts_output,     \
                                                                       JTYPE* runs_count_output, \
                                                                       size_t length,            \
                                                                       size_t buffer_size,       \
                                                                       void*  buffer);

INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE)                                                                   \
    template rocsparse_status rocsparse::primitives::exclusive_scan_buffer_size<ITYPE, JTYPE>(      \
        rocsparse_handle handle, JTYPE initial_value, size_t length, size_t * buffer_size);         \
    template rocsparse_status rocsparse::primitives::exclusive_scan(rocsparse_handle handle,        \
                                                                    ITYPE*           input,         \
                                                                    JTYPE*           output,        \
                                                                    JTYPE            initial_value, \
                                                                    size_t           length,        \
                                                                    size_t           buffer_size,   \
                                                                    void*            buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE)                                                                 \
    template rocsparse_status rocsparse::primitives::inclusive_scan_buffer_size<ITYPE, JTYPE>(    \
        rocsparse_handle handle, size_t length, size_t * buffer_size);                            \
    template rocsparse_status rocsparse::primitives::inclusive_scan(rocsparse_handle handle,      \
                                                                    ITYPE*           input,       \
                                                                    JTYPE*           output,      \
                                                                    size_t           length,      \
                                                                    size_t           buffer_size, \
                                                                    void*            buffer);

INSTANTIATE(uint32_t, uint32_t);
INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE)                                                           \
    template rocsparse_status rocsparse::primitives::find_max_buffer_size<ITYPE, JTYPE>(    \
        rocsparse_handle handle, size_t length, size_t * buffer_size);                      \
    template rocsparse_status rocsparse::primitives::find_max(rocsparse_handle handle,      \
                                                              ITYPE*           input,       \
                                                              JTYPE*           max,         \
                                                              size_t           length,      \
                                                              size_t           buffer_size, \
                                                              void*            buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

#define INSTANTIATE(ITYPE, JTYPE)                                                           \
    template rocsparse_status rocsparse::primitives::find_sum_buffer_size<ITYPE, JTYPE>(    \
        rocsparse_handle handle, size_t length, size_t * buffer_size);                      \
    template rocsparse_status rocsparse::primitives::find_sum(rocsparse_handle handle,      \
                                                              ITYPE*           input,       \
                                                              JTYPE*           sum,         \
                                                              size_t           length,      \
                                                              size_t           buffer_size, \
                                                              void*            buffer);

INSTANTIATE(int32_t, int32_t);
INSTANTIATE(int64_t, int64_t);
#undef INSTANTIATE

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
