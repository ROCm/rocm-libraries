/* ************************************************************************
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_sparse_to_sparse.hpp"
#include "rocsparse_utility.hpp"

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_sparse_to_sparse_alg value)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse_sparse_to_sparse_alg_default);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_sparse_to_sparse_stage value)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value)
    {
        CASE(rocsparse_sparse_to_sparse_stage_analysis);
        CASE(rocsparse_sparse_to_sparse_stage_compute);
#undef CASE
    }
    // LCOV_EXCL_START
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_sparse_to_sparse_stage value)
{
    switch(value)
    {
    case rocsparse_sparse_to_sparse_stage_analysis:
    case rocsparse_sparse_to_sparse_stage_compute:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_sparse_to_sparse_alg value)
{
    switch(value)
    {
    case rocsparse_sparse_to_sparse_alg_default:
    {
        return false;
    }
    }
    return true;
}

extern "C" rocsparse_status
    rocsparse_create_sparse_to_sparse_descr(rocsparse_sparse_to_sparse_descr* descr,
                                            rocsparse_const_spmat_descr       source,
                                            rocsparse_spmat_descr             target,
                                            rocsparse_sparse_to_sparse_alg    alg)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    ROCSPARSE_CHECKARG_POINTER(1, source);
    ROCSPARSE_CHECKARG_POINTER(2, target);
    ROCSPARSE_CHECKARG_ENUM(3, alg);

    rocsparse_format source_format;
    rocsparse_format target_format;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(target, &target_format));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(source, &source_format));
    try
    {
        descr[0]                  = new _rocsparse_sparse_to_sparse_descr;
        descr[0]->m_alg           = alg;
        descr[0]->m_target_format = target_format;
        descr[0]->m_source_format = source_format;

        const int64_t batch_count = source->batch_count;
        if(batch_count > 1)
        {
            descr[0]->batched = true;

            RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                      source->offsets_batch_stride > 0);

            RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_not_implemented,
                                      source->columns_values_batch_stride > 0);
        }
    }
    catch(const rocsparse_status& status)
    {
        return status;
    }

    return rocsparse_status_success;
}

extern "C" rocsparse_status
    rocsparse_sparse_to_sparse_permissive(rocsparse_sparse_to_sparse_descr descr)
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_POINTER(0, descr);
    descr->m_permissive = true;
    return rocsparse_status_success;
}

extern "C" rocsparse_status
    rocsparse_destroy_sparse_to_sparse_descr(rocsparse_sparse_to_sparse_descr descr)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    if(descr)
    {
        if(descr->m_intermediate != nullptr)
        {
            int64_t              m;
            int64_t              n;
            int64_t              nnz;
            void*                row;
            void*                col;
            void*                val;
            rocsparse_indextype  ptr_type;
            rocsparse_indextype  ind_type;
            rocsparse_index_base base;
            rocsparse_datatype   val_type;

            rocsparse_format format;
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(descr->m_intermediate, &format));
            RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_internal_error,
                                      format != rocsparse_format_csr);

            RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr_get(descr->m_intermediate,
                                                        &m,
                                                        &n,
                                                        &nnz,
                                                        &row,
                                                        &col,
                                                        &val,
                                                        &ptr_type,
                                                        &ind_type,
                                                        &base,
                                                        &val_type));

            // Due to the changes in the hipFree introduced in HIP 7.0
            // https://rocm.docs.amd.com/projects/HIP/en/latest/hip-7-changes.html#update-hipfree
            // we need to introduce a device synchronize here as the below hipFree calls are now asynchronous.
            // hipFree() previously had an implicit wait for synchronization purpose which is applicable for all memory allocations.
            // This wait has been disabled in the HIP 7.0 runtime for allocations made with hipMallocAsync and hipMallocFromPoolAsync.
            RETURN_IF_HIP_ERROR(hipDeviceSynchronize());

            RETURN_IF_HIP_ERROR(rocsparse_hipFree(row));
            RETURN_IF_HIP_ERROR(rocsparse_hipFree(col));
            RETURN_IF_HIP_ERROR(rocsparse_hipFree(val));
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_spmat_descr(descr->m_intermediate));
            descr->m_intermediate = nullptr;
        }
        delete descr;
    }

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
    static rocsparse_status sparse_to_sparse_core(rocsparse_handle                 handle,
                                                  rocsparse_sparse_to_sparse_descr descr,
                                                  rocsparse_const_spmat_descr      source,
                                                  rocsparse_spmat_descr            target,
                                                  rocsparse_sparse_to_sparse_stage stage,
                                                  size_t buffer_size_in_bytes,
                                                  void*  buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        static constexpr const bool compute_buffer_size_in_bytes = false;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::internal_sparse_to_sparse(handle,
                                                 descr,
                                                 source,
                                                 target,
                                                 stage,
                                                 &buffer_size_in_bytes,
                                                 buffer,
                                                 compute_buffer_size_in_bytes));
        return rocsparse_status_success;
    }

    static rocsparse_status sparse_to_sparse_quickreturn(rocsparse_handle                 handle,
                                                         rocsparse_sparse_to_sparse_descr descr,
                                                         rocsparse_const_spmat_descr      source,
                                                         rocsparse_spmat_descr            target,
                                                         rocsparse_sparse_to_sparse_stage stage,
                                                         size_t buffer_size_in_bytes,
                                                         void*  buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;
        return rocsparse_status_continue;
    }

    static rocsparse_status sparse_to_sparse_checkarg(rocsparse_handle                 handle, //0
                                                      rocsparse_sparse_to_sparse_descr descr, //1
                                                      rocsparse_const_spmat_descr      source, //2
                                                      rocsparse_spmat_descr            target, //3
                                                      rocsparse_sparse_to_sparse_stage stage, //4
                                                      size_t buffer_size_in_bytes, //5
                                                      void*  buffer) //6
    {
        ROCSPARSE_ROUTINE_TRACE;

        ROCSPARSE_CHECKARG_HANDLE(0, handle);
        ROCSPARSE_CHECKARG_POINTER(1, descr);
        ROCSPARSE_CHECKARG_POINTER(2, source);
        ROCSPARSE_CHECKARG_POINTER(3, target);
        ROCSPARSE_CHECKARG_ENUM(4, stage);
        ROCSPARSE_CHECKARG_ARRAY(6, buffer_size_in_bytes, buffer);

        const rocsparse_status status = rocsparse::sparse_to_sparse_quickreturn(
            handle, descr, source, target, stage, buffer_size_in_bytes, buffer);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        return rocsparse_status_continue;
    }

    template <typename... P>
    static rocsparse_status sparse_to_sparse_impl(P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_status status = rocsparse::sparse_to_sparse_checkarg(p...);
        if(status != rocsparse_status_continue)
        {
            RETURN_IF_ROCSPARSE_ERROR(status);
            return rocsparse_status_success;
        }

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_sparse_core(p...));
        return rocsparse_status_success;
    }
}

extern "C" rocsparse_status rocsparse_sparse_to_sparse(rocsparse_handle                 handle,
                                                       rocsparse_sparse_to_sparse_descr descr,
                                                       rocsparse_const_spmat_descr      source,
                                                       rocsparse_spmat_descr            target,
                                                       rocsparse_sparse_to_sparse_stage stage,
                                                       size_t buffer_size_in_bytes,
                                                       void*  buffer)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::sparse_to_sparse_impl(
        handle, descr, source, target, stage, buffer_size_in_bytes, buffer));
    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
