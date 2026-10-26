/*! \file */
/* ************************************************************************
 * Copyright (C) 2018-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/conversion/rocsparse_coosort.h"
#include "internal/conversion/rocsparse_inverse_permutation.h"

#include "rocsparse_utility.hpp"

#include "../level1/rocsparse_gthr.hpp"
#include "coosort_device.h"
#include "rocsparse_control.hpp"
#include "rocsparse_coosort.hpp"
#include "rocsparse_gcreate_identity_permutation.hpp"
#include "rocsparse_primitives.hpp"
#include "rocsparse_spmat_descr.hpp"

namespace rocsparse
{
    // Number of bits needed to represent indices that are smaller than size.
    static uint32_t coosort_endbit(int64_t size)
    {
        // __builtin_clzll is undefined for size == 0
        return (size == 0) ? 0 : 64 - __builtin_clzll(static_cast<unsigned long long>(size));
    }

    static bool coosort_is_supported(rocsparse_indextype idx_type)
    {
        return idx_type == rocsparse_indextype_i32 || idx_type == rocsparse_indextype_i64;
    }

    // The sort orders the entries by their major index, which has major_size possible values,
    // and then by their minor index, which has minor_size possible values, within each major
    // index. Sorting by row uses the rows as the major index, and sorting by column the columns.
    template <typename J>
    static rocsparse_status coosort_rocprim_buffer_size_template(rocsparse_handle handle,
                                                                 int64_t          major_size,
                                                                 int64_t          minor_size,
                                                                 int64_t          nnz,
                                                                 size_t*          buffer_size)
    {
        const uint32_t startbit = 0;

        *buffer_size = 0;

        size_t size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
            handle, nnz, startbit, rocsparse::coosort_endbit(major_size), &size)));
        *buffer_size = rocsparse::max(size, *buffer_size);

        size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::run_length_encode_buffer_size<J>(handle, nnz, &size));
        *buffer_size = rocsparse::max(size, *buffer_size);

        size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), major_size + 1, &size)));
        *buffer_size = rocsparse::max(size, *buffer_size);

        size_t size1 = std::numeric_limits<size_t>::max();
        size_t size2 = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, J, J>(
                handle, nnz, major_size, startbit, rocsparse::coosort_endbit(minor_size), &size1)));
        RETURN_IF_ROCSPARSE_ERROR(
            (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, J>(
                handle, nnz, major_size, startbit, rocsparse::coosort_endbit(minor_size), &size2)));
        *buffer_size = rocsparse::max(rocsparse::max(size1, size2), *buffer_size);

        return rocsparse_status_success;
    }

    static rocsparse_status coosort_rocprim_buffer_size(rocsparse_handle    handle,
                                                        rocsparse_indextype idx_type,
                                                        int64_t             major_size,
                                                        int64_t             minor_size,
                                                        int64_t             nnz,
                                                        size_t*             buffer_size)
    {
        if(idx_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_rocprim_buffer_size_template<int32_t>(
                handle, major_size, minor_size, nnz, buffer_size)));
            return rocsparse_status_success;
        }
        if(idx_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_rocprim_buffer_size_template<int64_t>(
                handle, major_size, minor_size, nnz, buffer_size)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "the indices type is not supported");
    }

    // Sorts the major indices and applies the same reordering to vals. The sorted major
    // indices are copied back into major, and vals_sorted is set to whichever of vals and
    // vals_tmp holds the reordered values.
    template <typename J>
    static rocsparse_status coosort_sort_major_template(rocsparse_handle handle,
                                                        int64_t          nnz,
                                                        uint32_t         endbit,
                                                        void*            major,
                                                        void*            major_tmp,
                                                        void*            vals,
                                                        void*            vals_tmp,
                                                        void*            rocprim_buffer,
                                                        void**           vals_sorted)
    {
        const uint32_t startbit = 0;

        J* major_ = reinterpret_cast<J*>(major);

        rocsparse::primitives::double_buffer<J> keys(major_, reinterpret_cast<J*>(major_tmp));
        rocsparse::primitives::double_buffer<J> values(reinterpret_cast<J*>(vals),
                                                       reinterpret_cast<J*>(vals_tmp));

        size_t size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::radix_sort_pairs_buffer_size<J, J>(
            handle, nnz, startbit, endbit, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::radix_sort_pairs(
            handle, keys, values, nnz, startbit, endbit, size, rocprim_buffer));

        if(keys.current() != major_)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                major_, keys.current(), sizeof(J) * nnz, hipMemcpyDeviceToDevice, handle->stream));
        }

        *vals_sorted = values.current();
        return rocsparse_status_success;
    }

    static rocsparse_status coosort_sort_major(rocsparse_handle    handle,
                                               rocsparse_indextype idx_type,
                                               int64_t             nnz,
                                               uint32_t            endbit,
                                               void*               major,
                                               void*               major_tmp,
                                               void*               vals,
                                               void*               vals_tmp,
                                               void*               rocprim_buffer,
                                               void**              vals_sorted)
    {
        if(idx_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_sort_major_template<int32_t>(handle,
                                                                            nnz,
                                                                            endbit,
                                                                            major,
                                                                            major_tmp,
                                                                            vals,
                                                                            vals_tmp,
                                                                            rocprim_buffer,
                                                                            vals_sorted)));
            return rocsparse_status_success;
        }
        if(idx_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_sort_major_template<int64_t>(handle,
                                                                            nnz,
                                                                            endbit,
                                                                            major,
                                                                            major_tmp,
                                                                            vals,
                                                                            vals_tmp,
                                                                            rocprim_buffer,
                                                                            vals_sorted)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "the indices type is not supported");
    }

    // Computes the offsets of the segments of equal sorted major indices, for the segmented
    // sort of the minor indices. The number of segments is copied back to the host.
    template <typename J>
    static rocsparse_status coosort_segments_template(rocsparse_handle handle,
                                                      int64_t          nnz,
                                                      void*            major,
                                                      void*            workspace,
                                                      void*            offsets,
                                                      void*            rocprim_buffer,
                                                      int64_t*         nsegm)
    {
        J* workspace_ = reinterpret_cast<J*>(workspace);
        J* offsets_   = reinterpret_cast<J*>(offsets);

        size_t size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::run_length_encode_buffer_size<J>(handle, nnz, &size));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::primitives::run_length_encode(handle,
                                                     reinterpret_cast<J*>(major),
                                                     workspace_ + 1,
                                                     offsets_,
                                                     workspace_,
                                                     nnz,
                                                     size,
                                                     rocprim_buffer));

        J nsegm_;
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
            &nsegm_, workspace_, sizeof(J), hipMemcpyDeviceToHost, handle->stream));

        // Wait for host transfer to finish
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(handle->stream));

        size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::exclusive_scan_buffer_size<J, J>(
            handle, static_cast<J>(0), nsegm_ + 1, &size)));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::exclusive_scan(
            handle, offsets_, offsets_, static_cast<J>(0), nsegm_ + 1, size, rocprim_buffer));

        *nsegm = nsegm_;
        return rocsparse_status_success;
    }

    static rocsparse_status coosort_segments(rocsparse_handle    handle,
                                             rocsparse_indextype idx_type,
                                             int64_t             nnz,
                                             void*               major,
                                             void*               workspace,
                                             void*               offsets,
                                             void*               rocprim_buffer,
                                             int64_t*            nsegm)
    {
        if(idx_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_segments_template<int32_t>(
                handle, nnz, major, workspace, offsets, rocprim_buffer, nsegm)));
            return rocsparse_status_success;
        }
        if(idx_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_segments_template<int64_t>(
                handle, nnz, major, workspace, offsets, rocprim_buffer, nsegm)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "the indices type is not supported");
    }

    // Computes out[i] = in[mapping[i]].
    template <typename J>
    static rocsparse_status coosort_permute_template(
        rocsparse_handle handle, int64_t nnz, const void* in, const void* mapping, void* out)
    {
#define COOSORT_DIM 512
        dim3 coosort_blocks((nnz - 1) / COOSORT_DIM + 1);
        dim3 coosort_threads(COOSORT_DIM);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::coosort_permute_kernel<COOSORT_DIM>),
                                           coosort_blocks,
                                           coosort_threads,
                                           0,
                                           handle->stream,
                                           static_cast<J>(nnz),
                                           reinterpret_cast<const J*>(in),
                                           reinterpret_cast<const J*>(mapping),
                                           reinterpret_cast<J*>(out));
#undef COOSORT_DIM
        return rocsparse_status_success;
    }

    static rocsparse_status coosort_permute(rocsparse_handle    handle,
                                            rocsparse_indextype idx_type,
                                            int64_t             nnz,
                                            const void*         in,
                                            const void*         mapping,
                                            void*               out)
    {
        if(idx_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (coosort_permute_template<int32_t>(handle, nnz, in, mapping, out)));
            return rocsparse_status_success;
        }
        if(idx_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (coosort_permute_template<int64_t>(handle, nnz, in, mapping, out)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "the indices type is not supported");
    }

    // Sorts the minor indices within each segment, and applies the same reordering to vals
    // if it is not null. The results are copied into minor and perm if they end up in the
    // other buffers.
    template <typename J>
    static rocsparse_status coosort_sort_minor_template(rocsparse_handle handle,
                                                        int64_t          nnz,
                                                        int64_t          nsegm,
                                                        uint32_t         endbit,
                                                        const void*      offsets,
                                                        void*            keys,
                                                        void*            keys_tmp,
                                                        void*            vals,
                                                        void*            vals_tmp,
                                                        void*            minor,
                                                        void*            perm,
                                                        void*            rocprim_buffer)
    {
        const uint32_t startbit = 0;

        const J* offsets_ = reinterpret_cast<const J*>(offsets);

        rocsparse::primitives::double_buffer<J> keys_(reinterpret_cast<J*>(keys),
                                                      reinterpret_cast<J*>(keys_tmp));

        size_t size = std::numeric_limits<size_t>::max();
        if(vals != nullptr)
        {
            rocsparse::primitives::double_buffer<J> vals_(reinterpret_cast<J*>(vals),
                                                          reinterpret_cast<J*>(vals_tmp));

            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_pairs_buffer_size<J, J, J>(
                    handle, nnz, nsegm, startbit, endbit, &size)));
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::primitives::segmented_radix_sort_pairs(handle,
                                                                  keys_,
                                                                  vals_,
                                                                  static_cast<J>(nnz),
                                                                  static_cast<J>(nsegm),
                                                                  offsets_,
                                                                  offsets_ + 1,
                                                                  startbit,
                                                                  endbit,
                                                                  size,
                                                                  rocprim_buffer));

            if(vals_.current() != perm)
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(perm,
                                                             vals_.current(),
                                                             sizeof(J) * nnz,
                                                             hipMemcpyDeviceToDevice,
                                                             handle->stream));
            }
        }
        else
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (rocsparse::primitives::segmented_radix_sort_keys_buffer_size<J, J>(
                    handle, nnz, nsegm, startbit, endbit, &size)));
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::primitives::segmented_radix_sort_keys(handle,
                                                                 keys_,
                                                                 static_cast<J>(nnz),
                                                                 static_cast<J>(nsegm),
                                                                 offsets_,
                                                                 offsets_ + 1,
                                                                 startbit,
                                                                 endbit,
                                                                 size,
                                                                 rocprim_buffer));
        }

        if(keys_.current() != minor)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                minor, keys_.current(), sizeof(J) * nnz, hipMemcpyDeviceToDevice, handle->stream));
        }
        return rocsparse_status_success;
    }

    static rocsparse_status coosort_sort_minor(rocsparse_handle    handle,
                                               rocsparse_indextype idx_type,
                                               int64_t             nnz,
                                               int64_t             nsegm,
                                               uint32_t            endbit,
                                               const void*         offsets,
                                               void*               keys,
                                               void*               keys_tmp,
                                               void*               vals,
                                               void*               vals_tmp,
                                               void*               minor,
                                               void*               perm,
                                               void*               rocprim_buffer)
    {
        if(idx_type == rocsparse_indextype_i32)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_sort_minor_template<int32_t>(handle,
                                                                            nnz,
                                                                            nsegm,
                                                                            endbit,
                                                                            offsets,
                                                                            keys,
                                                                            keys_tmp,
                                                                            vals,
                                                                            vals_tmp,
                                                                            minor,
                                                                            perm,
                                                                            rocprim_buffer)));
            return rocsparse_status_success;
        }
        if(idx_type == rocsparse_indextype_i64)
        {
            RETURN_IF_ROCSPARSE_ERROR((coosort_sort_minor_template<int64_t>(handle,
                                                                            nnz,
                                                                            nsegm,
                                                                            endbit,
                                                                            offsets,
                                                                            keys,
                                                                            keys_tmp,
                                                                            vals,
                                                                            vals_tmp,
                                                                            minor,
                                                                            perm,
                                                                            rocprim_buffer)));
            return rocsparse_status_success;
        }
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "the indices type is not supported");
    }

    static rocsparse_status coosort_buffer_size_compute(rocsparse_handle    handle,
                                                        rocsparse_direction dir,
                                                        int64_t             m,
                                                        int64_t             n,
                                                        int64_t             nnz,
                                                        rocsparse_indextype idx_type,
                                                        size_t*             buffer_size_in_bytes)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if(!rocsparse::coosort_is_supported(idx_type))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                                   "the indices type is not supported");
        }

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            *buffer_size_in_bytes = 0;
            return rocsparse_status_success;
        }

        const bool by_row       = (dir == rocsparse_direction_row);
        size_t     rocprim_size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_rocprim_buffer_size(
            handle, idx_type, by_row ? m : n, by_row ? n : m, nnz, &rocprim_size));

        const size_t idx_size = rocsparse::indextype_sizeof(idx_type);

        *buffer_size_in_bytes = rocsparse::align_size<char>(rocprim_size);

        // rocPRIM does not support in-place sorting, so we need additional buffer
        // for all temporary arrays: two arrays of nnz indices, the workspace of the segments,
        // which holds the number of segments followed by up to nnz unique indices, and the
        // segment offsets.
        *buffer_size_in_bytes += rocsparse::align_size<char>(idx_size * nnz);
        *buffer_size_in_bytes += rocsparse::align_size<char>(idx_size * nnz);
        *buffer_size_in_bytes += rocsparse::align_size<char>(idx_size * (nnz + 1));
        *buffer_size_in_bytes += rocsparse::align_size<char>(idx_size * (rocsparse::max(m, n) + 1));

        return rocsparse_status_success;
    }

    // The legacy buffer size query does not know the direction, so the buffer must be large
    // enough to sort either by row or by column.
    static rocsparse_status coosort_legacy_buffer_size(
        rocsparse_handle handle, int64_t m, int64_t n, int64_t nnz, size_t* buffer_size_in_bytes)
    {
        size_t buffer_size_by_row = std::numeric_limits<size_t>::max();
        size_t buffer_size_by_col = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::coosort_buffer_size_compute(handle,
                                                   rocsparse_direction_row,
                                                   m,
                                                   n,
                                                   nnz,
                                                   rocsparse::get_indextype<rocsparse_int>(),
                                                   &buffer_size_by_row));
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::coosort_buffer_size_compute(handle,
                                                   rocsparse_direction_column,
                                                   m,
                                                   n,
                                                   nnz,
                                                   rocsparse::get_indextype<rocsparse_int>(),
                                                   &buffer_size_by_col));
        *buffer_size_in_bytes = rocsparse::max(buffer_size_by_row, buffer_size_by_col);
        return rocsparse_status_success;
    }

    // Sorts the entries by row (dir == rocsparse_direction_row) or by column
    // (dir == rocsparse_direction_column). If perm is not null, the sorting permutation is
    // applied to it, so it holds the sorting permutation if it starts as the identity.
    static rocsparse_status coosort_compute(rocsparse_handle    handle,
                                            rocsparse_direction dir,
                                            int64_t             m,
                                            int64_t             n,
                                            int64_t             nnz,
                                            rocsparse_indextype idx_type,
                                            void*               coo_row_ind,
                                            void*               coo_col_ind,
                                            void*               perm,
                                            size_t              buffer_size_in_bytes,
                                            void*               buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        size_t required_buffer_size = std::numeric_limits<size_t>::max();
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_buffer_size_compute(
            handle, dir, m, n, nnz, idx_type, &required_buffer_size));
        if(buffer_size_in_bytes < required_buffer_size)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
                rocsparse_status_invalid_size,
                "the buffer is smaller than the size required by the index sort");
        }

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }

        const bool     by_row       = (dir == rocsparse_direction_row);
        const int64_t  major_size   = by_row ? m : n;
        const int64_t  minor_size   = by_row ? n : m;
        void*          major        = by_row ? coo_row_ind : coo_col_ind;
        void*          minor        = by_row ? coo_col_ind : coo_row_ind;
        const uint32_t major_endbit = rocsparse::coosort_endbit(major_size);
        const uint32_t minor_endbit = rocsparse::coosort_endbit(minor_size);

        const size_t idx_size = rocsparse::indextype_sizeof(idx_type);

        // Temporary buffer entry points
        char* ptr = reinterpret_cast<char*>(buffer);

        void* work1 = ptr;
        ptr += rocsparse::align_size<char>(idx_size * nnz);

        void* work2 = ptr;
        ptr += rocsparse::align_size<char>(idx_size * (nnz + 1));

        void* work3 = ptr;
        ptr += rocsparse::align_size<char>(idx_size * (rocsparse::max(m, n) + 1));

        void* work4 = ptr;
        ptr += rocsparse::align_size<char>(idx_size * nnz);

        void* tmp_rocprim = ptr;

        int64_t nsegm = 0;

        if(perm != nullptr)
        {
            // Create the identity permutation to keep track of the reordering, so that perm
            // does not need to hold the identity.
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::gcreate_identity_permutation(handle, nnz, idx_type, work4));

            // Sorting the identity along with the major indices gives the original position
            // of each entry, which is used to reorder the minor indices and perm.
            void* mapping = nullptr;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_sort_major(handle,
                                                                    idx_type,
                                                                    nnz,
                                                                    major_endbit,
                                                                    major,
                                                                    work2,
                                                                    work4,
                                                                    work1,
                                                                    tmp_rocprim,
                                                                    &mapping));
            void* alt_map = (mapping == work4) ? work1 : work4;

            // Obtain segments for segmented sort by the minor indices
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_segments(
                handle, idx_type, nnz, major, work2, work3, tmp_rocprim, &nsegm));

            // Reorder the minor indices and perm
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::coosort_permute(handle, idx_type, nnz, minor, mapping, work2));
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::coosort_permute(handle, idx_type, nnz, perm, mapping, alt_map));

            // Sort the minor indices within each segment
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_sort_minor(handle,
                                                                    idx_type,
                                                                    nnz,
                                                                    nsegm,
                                                                    minor_endbit,
                                                                    work3,
                                                                    work2,
                                                                    minor,
                                                                    alt_map,
                                                                    mapping,
                                                                    minor,
                                                                    perm,
                                                                    tmp_rocprim));
        }
        else
        {
            // Sort by the major indices and reorder the minor indices
            void* sorted_minor = nullptr;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_sort_major(handle,
                                                                    idx_type,
                                                                    nnz,
                                                                    major_endbit,
                                                                    major,
                                                                    work2,
                                                                    minor,
                                                                    work1,
                                                                    tmp_rocprim,
                                                                    &sorted_minor));
            void* alt_minor = (sorted_minor == minor) ? work1 : minor;

            // Obtain segments for segmented sort by the minor indices
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_segments(
                handle, idx_type, nnz, major, work2, work3, tmp_rocprim, &nsegm));

            // Sort the minor indices within each segment
            RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_sort_minor(handle,
                                                                    idx_type,
                                                                    nnz,
                                                                    nsegm,
                                                                    minor_endbit,
                                                                    work3,
                                                                    sorted_minor,
                                                                    alt_minor,
                                                                    nullptr,
                                                                    nullptr,
                                                                    minor,
                                                                    nullptr,
                                                                    tmp_rocprim));
        }

        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_coosort_buffer_size(rocsparse_handle     handle,
                                                          rocsparse_int        m,
                                                          rocsparse_int        n,
                                                          rocsparse_int        nnz,
                                                          const rocsparse_int* coo_row_ind,
                                                          const rocsparse_int* coo_col_ind,
                                                          size_t*              buffer_size)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_buffer_size",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)buffer_size);

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_SIZE(1, m);
    ROCSPARSE_CHECKARG_SIZE(2, n);
    ROCSPARSE_CHECKARG_SIZE(3, nnz);
    ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
    ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
    ROCSPARSE_CHECKARG_POINTER(6, buffer_size);

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::coosort_legacy_buffer_size(handle, m, n, nnz, buffer_size));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    static rocsparse_status coosort_by_row_quickreturn(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       rocsparse_int    n,
                                                       rocsparse_int    nnz,
                                                       rocsparse_int*   coo_row_ind,
                                                       rocsparse_int*   coo_col_ind,
                                                       rocsparse_int*   perm,
                                                       void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    static rocsparse_status coosort_by_row_checkarg(rocsparse_handle handle,
                                                    rocsparse_int    m,
                                                    rocsparse_int    n,
                                                    rocsparse_int    nnz,
                                                    rocsparse_int*   coo_row_ind,
                                                    rocsparse_int*   coo_col_ind,
                                                    rocsparse_int*   perm,
                                                    void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, temp_buffer);

        const rocsparse_status status = rocsparse::coosort_by_row_quickreturn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }
}

extern "C" rocsparse_status rocsparse_coosort_by_row(rocsparse_handle handle,
                                                     rocsparse_int    m,
                                                     rocsparse_int    n,
                                                     rocsparse_int    nnz,
                                                     rocsparse_int*   coo_row_ind,
                                                     rocsparse_int*   coo_col_ind,
                                                     rocsparse_int*   perm,
                                                     void*            temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_row",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::coosort_by_row_checkarg(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    // The legacy API does not take the size of temp_buffer, which is assumed to be the size
    // returned by rocsparse_coosort_buffer_size.
    size_t buffer_size = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::coosort_legacy_buffer_size(handle, m, n, nnz, &buffer_size));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_compute(handle,
                                                         rocsparse_direction_row,
                                                         m,
                                                         n,
                                                         nnz,
                                                         rocsparse::get_indextype<rocsparse_int>(),
                                                         coo_row_ind,
                                                         coo_col_ind,
                                                         perm,
                                                         buffer_size,
                                                         temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    static rocsparse_status coosort_by_column_quickreturn(rocsparse_handle handle,
                                                          rocsparse_int    m,
                                                          rocsparse_int    n,
                                                          rocsparse_int    nnz,
                                                          rocsparse_int*   coo_row_ind,
                                                          rocsparse_int*   coo_col_ind,
                                                          rocsparse_int*   perm,
                                                          void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        // Quick return if possible
        if(m == 0 || n == 0 || nnz == 0)
        {
            return rocsparse_status_success;
        }
        return rocsparse_status_continue;
    }

    static rocsparse_status coosort_by_column_checkarg(rocsparse_handle handle,
                                                       rocsparse_int    m,
                                                       rocsparse_int    n,
                                                       rocsparse_int    nnz,
                                                       rocsparse_int*   coo_row_ind,
                                                       rocsparse_int*   coo_col_ind,
                                                       rocsparse_int*   perm,
                                                       void*            temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_SIZE(1, m);
        ROCSPARSE_CHECKARG_SIZE(2, n);
        ROCSPARSE_CHECKARG_SIZE(3, nnz);
        ROCSPARSE_CHECKARG_ARRAY(4, nnz, coo_row_ind);
        ROCSPARSE_CHECKARG_ARRAY(5, nnz, coo_col_ind);
        ROCSPARSE_CHECKARG_ARRAY(7, nnz, temp_buffer);

        const rocsparse_status status = rocsparse::coosort_by_column_quickreturn(
            handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }
}

extern "C" rocsparse_status rocsparse_coosort_by_column(rocsparse_handle handle,
                                                        rocsparse_int    m,
                                                        rocsparse_int    n,
                                                        rocsparse_int    nnz,
                                                        rocsparse_int*   coo_row_ind,
                                                        rocsparse_int*   coo_col_ind,
                                                        rocsparse_int*   perm,
                                                        void*            temp_buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    // Logging
    rocsparse::log_trace(handle,
                         "rocsparse_coosort_by_column",
                         m,
                         n,
                         nnz,
                         (const void*&)coo_row_ind,
                         (const void*&)coo_col_ind,
                         (const void*&)perm,
                         (const void*&)temp_buffer);

    const rocsparse_status status = rocsparse::coosort_by_column_checkarg(
        handle, m, n, nnz, coo_row_ind, coo_col_ind, perm, temp_buffer);

    if(status != rocsparse_status_continue)
    {
        RETURN_IF_ROCSPARSE_ERROR(status);
        return rocsparse_status_success;
    }

    // The legacy API does not take the size of temp_buffer, which is assumed to be the size
    // returned by rocsparse_coosort_buffer_size.
    size_t buffer_size = std::numeric_limits<size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::coosort_legacy_buffer_size(handle, m, n, nnz, &buffer_size));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_compute(handle,
                                                         rocsparse_direction_column,
                                                         m,
                                                         n,
                                                         nnz,
                                                         rocsparse::get_indextype<rocsparse_int>(),
                                                         coo_row_ind,
                                                         coo_col_ind,
                                                         perm,
                                                         buffer_size,
                                                         temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP

namespace rocsparse
{
    // The buffer starts with the permutation array, which tracks where each entry moves
    // while the indices are sorted, followed by scratch space shared by the index sort
    // and the value permutation.
    static size_t coosort_perm_size_in_bytes(int64_t nnz, rocsparse_indextype perm_indextype)
    {
        return rocsparse::align_size<char>(rocsparse::indextype_sizeof(perm_indextype) * nnz);
    }
}

rocsparse_status rocsparse::coosort_buffer_size(rocsparse_handle            handle,
                                                rocsparse_direction         dir,
                                                rocsparse_const_spmat_descr source,
                                                rocsparse_const_spmat_descr target,
                                                size_t*                     buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    const int64_t nnz = target->nnz;

    size_t sort_buffer_size = std::numeric_limits<std::size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_buffer_size_compute(
        handle, dir, target->rows, target->cols, nnz, target->row_type, &sort_buffer_size));

    // Values sorted in place are gathered into scratch space first, since the gather cannot
    // write over its own input.
    const size_t gather_buffer_size
        = (target->const_val_data == source->const_val_data)
              ? rocsparse::align_size<char>(rocsparse::datatype_sizeof(target->data_type) * nnz)
              : 0;

    *buffer_size_in_bytes = rocsparse::coosort_perm_size_in_bytes(nnz, target->row_type)
                            + rocsparse::max(sort_buffer_size, gather_buffer_size);

    return rocsparse_status_success;
}

rocsparse_status rocsparse::coosort(rocsparse_handle            handle,
                                    rocsparse_direction         dir,
                                    rocsparse_const_spmat_descr source,
                                    rocsparse_spmat_descr       target,
                                    size_t                      buffer_size_in_bytes,
                                    void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;

    size_t required_buffer_size = std::numeric_limits<std::size_t>::max();
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse::coosort_buffer_size(handle, dir, source, target, &required_buffer_size));
    if(buffer_size_in_bytes < required_buffer_size)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_invalid_size,
            "the buffer is smaller than the size returned by the buffer size query");
    }

    const int64_t             m         = target->rows;
    const int64_t             n         = target->cols;
    const int64_t             nnz       = target->nnz;
    const rocsparse_indextype idx_type  = target->row_type;
    const rocsparse_datatype  data_type = target->data_type;
    const size_t              idx_size  = rocsparse::indextype_sizeof(idx_type);
    const size_t              val_size  = rocsparse::datatype_sizeof(data_type);

    const size_t perm_size        = rocsparse::coosort_perm_size_in_bytes(nnz, idx_type);
    const size_t sort_buffer_size = buffer_size_in_bytes - perm_size;
    void*        perm             = buffer;
    void*        sort_buffer      = reinterpret_cast<char*>(buffer) + perm_size;

    const size_t val_stride_source = source->batch_stride * val_size;
    const size_t idx_stride_target = target->batch_stride * idx_size;
    const size_t val_stride_target = target->batch_stride * val_size;

    const char* row_ind_source = reinterpret_cast<const char*>(source->const_row_data);
    const char* col_ind_source = reinterpret_cast<const char*>(source->const_col_data);
    const char* val_source     = reinterpret_cast<const char*>(source->const_val_data);
    char*       row_ind_target = reinterpret_cast<char*>(target->row_data);
    char*       col_ind_target = reinterpret_cast<char*>(target->col_data);
    char*       val_target     = reinterpret_cast<char*>(target->val_data);

    // Only uniform batches are supported, so every batch has the sparsity pattern of the first
    // one. Its indices are sorted once, and the resulting permutation is applied to the values
    // of every batch. A single source matrix has a zero batch stride, so its values are sorted
    // into every batch of target.

    // The index sort works in place, so the indices of source are first copied into target.
    if(row_ind_target != row_ind_source)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(row_ind_target,
                                                     row_ind_source,
                                                     idx_size * nnz,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
    }
    if(col_ind_target != col_ind_source)
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(col_ind_target,
                                                     col_ind_source,
                                                     idx_size * nnz,
                                                     hipMemcpyDeviceToDevice,
                                                     handle->stream));
    }

    // The index sort applies its reordering to perm, so it must start as the identity.
    RETURN_IF_ROCSPARSE_ERROR(rocsparse::gcreate_identity_permutation(handle, nnz, idx_type, perm));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosort_compute(handle,
                                                         dir,
                                                         m,
                                                         n,
                                                         nnz,
                                                         idx_type,
                                                         row_ind_target,
                                                         col_ind_target,
                                                         perm,
                                                         sort_buffer_size,
                                                         sort_buffer));

    // Non-uniform batches are not supported: the sorted indices of the first batch are copied
    // into every batch. The batches run one after the other on the handle stream, so they share
    // the buffer.
    for(int64_t batch = 0; batch < target->batch_count; ++batch)
    {
        char* batch_row_ind_target = row_ind_target + batch * idx_stride_target;
        char* batch_col_ind_target = col_ind_target + batch * idx_stride_target;

        if(batch > 0)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_row_ind_target,
                                                         row_ind_target,
                                                         idx_size * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_col_ind_target,
                                                         col_ind_target,
                                                         idx_size * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
        }

        const char* batch_val_source = val_source + batch * val_stride_source;
        char*       batch_val_target = val_target + batch * val_stride_target;

        // The gather cannot write over its own input, so in place values go through scratch.
        const bool in_place_val = (batch_val_target == batch_val_source);
        void*      sorted_val   = in_place_val ? sort_buffer : batch_val_target;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse::gthr(handle,
                                                  nnz,
                                                  data_type,
                                                  batch_val_source,
                                                  data_type,
                                                  sorted_val,
                                                  idx_type,
                                                  perm,
                                                  rocsparse_index_base_zero));

        if(in_place_val)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(batch_val_target,
                                                         sorted_val,
                                                         val_size * nnz,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
        }
    }

    return rocsparse_status_success;
}
