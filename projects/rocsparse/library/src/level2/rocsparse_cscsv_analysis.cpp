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

#include "rocsparse_assign_async.hpp"
#include "rocsparse_cscsv.hpp"
#include "rocsparse_cscsv_kernel.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_grid.hpp"
#include "rocsparse_utility.hpp"

#include "cscsv_device.h"

#include <map>
#include <sstream>
#include <tuple>

namespace rocsparse
{
    template <uint32_t BLOCKSIZE, uint32_t NUM_LANES, typename I, typename J>
    static rocsparse_status cscsv_analysis_launch(rocsparse_handle     handle,
                                                  int64_t              m,
                                                  const void*          csc_col_ptr,
                                                  const void*          csc_row_ind,
                                                  void*                csc_diag_ind,
                                                  void*                dependency_count,
                                                  void*                zero_pivot,
                                                  rocsparse_index_base idx_base,
                                                  rocsparse_diag_type  diag_type,
                                                  rocsparse_fill_mode  fill_mode)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::cscsv_analysis_kernel<BLOCKSIZE, NUM_LANES, I, J>),
            dim3(rocsparse::ceil_div(m * NUM_LANES, BLOCKSIZE)),
            dim3(BLOCKSIZE),
            0,
            handle->stream,
            static_cast<J>(m),
            reinterpret_cast<const I*>(csc_col_ptr),
            reinterpret_cast<const J*>(csc_row_ind),
            reinterpret_cast<I*>(csc_diag_ind),
            reinterpret_cast<J*>(dependency_count),
            reinterpret_cast<J*>(zero_pivot),
            idx_base,
            diag_type,
            fill_mode);
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename I>
    static rocsparse_status cscsv_max_nnz_launch(rocsparse_handle handle,
                                                 int64_t          m,
                                                 const void*      csc_col_ptr,
                                                 void*            workspace)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::cscsv_max_nnz_part1<BLOCKSIZE, I>),
                                           dim3(BLOCKSIZE),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           m,
                                           reinterpret_cast<const I*>(csc_col_ptr),
                                           reinterpret_cast<I*>(workspace));
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::cscsv_max_nnz_part2<BLOCKSIZE, I>),
                                           dim3(1),
                                           dim3(BLOCKSIZE),
                                           0,
                                           handle->stream,
                                           reinterpret_cast<I*>(workspace));
        return rocsparse_status_success;
    }

    static const std::map<rocsparse_indextype, rocsparse::cscsv_max_nnz_launch_t>
        s_cscsv_max_nnz_dispatch{
            {{rocsparse_indextype_i32, cscsv_max_nnz_launch<rocsparse::s_cscsv_blocksize, int32_t>},
             {rocsparse_indextype_i64,
              cscsv_max_nnz_launch<rocsparse::s_cscsv_blocksize, int64_t>}}};

    using cscsv_analysis_tpl_t = std::tuple<rocsparse_indextype, rocsparse_indextype>;

#define CONFIG(I_, J_)                                                              \
    {                                                                               \
        cscsv_analysis_tpl_t(I_, J_),                                               \
            cscsv_analysis_launch<rocsparse::s_cscsv_blocksize,                     \
                                  rocsparse::s_cscsv_analysis_num_lanes,            \
                                  typename rocsparse::indextype_traits<I_>::type_t, \
                                  typename rocsparse::indextype_traits<J_>::type_t> \
    }

    static const std::map<cscsv_analysis_tpl_t, rocsparse::cscsv_analysis_launch_t>
        s_cscsv_analysis_dispatch{{CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i32),
                                   CONFIG(rocsparse_indextype_i32, rocsparse_indextype_i64),
                                   CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i32),
                                   CONFIG(rocsparse_indextype_i64, rocsparse_indextype_i64)}};
#undef CONFIG
}

rocsparse_status rocsparse::cscsv_analysis_launch_find(rocsparse::cscsv_analysis_launch_t* launch,
                                                       rocsparse_indextype                 i_type,
                                                       rocsparse_indextype                 j_type)
{
    const auto& it = rocsparse::s_cscsv_analysis_dispatch.find(
        rocsparse::cscsv_analysis_tpl_t(i_type, j_type));
    if(it == rocsparse::s_cscsv_analysis_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid index type configuration: i_type: "
             << rocsparse::enum_utils::to_string(i_type)
             << ", j_type: " << rocsparse::enum_utils::to_string(j_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_max_nnz_launch_find(rocsparse::cscsv_max_nnz_launch_t* launch,
                                                      rocsparse_indextype                i_type)
{
    const auto& it = rocsparse::s_cscsv_max_nnz_dispatch.find(i_type);
    if(it == rocsparse::s_cscsv_max_nnz_dispatch.end())
    {
        // LCOV_EXCL_START
        std::stringstream sstr;
        sstr << "invalid index type configuration: i_type: "
             << rocsparse::enum_utils::to_string(i_type);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, sstr.str().c_str());
        // LCOV_EXCL_STOP
    }
    launch[0] = it->second;
    return rocsparse_status_success;
}

rocsparse_status rocsparse::cscsv_analysis(rocsparse_handle            handle,
                                           rocsparse_operation         trans,
                                           rocsparse_const_spmat_descr A,
                                           rocsparse_analysis_policy   analysis_policy,
                                           rocsparse_solve_policy      solve_policy,
                                           rocsparse_cscsv_info        info,
                                           size_t                      buffer_size_in_bytes,
                                           void*                       buffer)
{
    ROCSPARSE_ROUTINE_TRACE;
    if(trans == rocsparse_operation_none)
    {
        // The CSC analysis is not shared through the matrix info.
        hipStream_t               stream = handle->stream;
        const rocsparse_mat_descr descr  = A->descr;
        const int64_t             m      = A->cols;

        // Column pointers and row indices
        const rocsparse_indextype I_type   = A->col_type;
        const rocsparse_indextype J_type   = A->row_type;
        const size_t              sizeof_I = rocsparse::indextype_sizeof(I_type);
        const size_t              sizeof_J = rocsparse::indextype_sizeof(J_type);

        // Owned here until info->set takes it, so that an early return does not leak it.
        std::unique_ptr<rocsparse::trm_info_t> trm_info(new rocsparse::trm_info_t());
        trm_info->set_m(m);
        trm_info->set_nnz(A->nnz);
        trm_info->set_descr(descr);
        trm_info->set_row_ptr(A->const_col_data);
        trm_info->set_col_ind(A->const_row_data);
        trm_info->set_offset_indextype(I_type);
        trm_info->set_index_indextype(J_type);

        RETURN_IF_HIP_ERROR(
            rocsparse_hipMallocAsync(trm_info->get_ref_diag_ind(), sizeof_I * m, stream));
        void** dependency_count = info->get_ref_dependency_count(descr->fill_mode);
        RETURN_IF_HIP_ERROR(rocsparse_hipFreeAsync(*dependency_count, stream));
        *dependency_count = nullptr;
        RETURN_IF_HIP_ERROR(rocsparse_hipMallocAsync(dependency_count, sizeof_J * m, stream));
        RETURN_IF_ROCSPARSE_ERROR(info->create_zero_pivot_async(J_type, stream));

        void* zero_pivot = info->get_position();
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::assign_max_async(info->get_batch_count(), J_type, zero_pivot, stream));

        // -1 marks a missing diagonal entry
        RETURN_IF_HIP_ERROR(
            rocsparse_hipMemsetAsync(trm_info->get_diag_ind(), 0xFF, sizeof_I * m, stream));
        RETURN_IF_HIP_ERROR(rocsparse_hipMemsetAsync(*dependency_count, 0, sizeof_J * m, stream));

        if(m > 0)
        {
            rocsparse::cscsv_analysis_launch_t launch;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::cscsv_analysis_launch_find(&launch, I_type, J_type));
            RETURN_IF_ROCSPARSE_ERROR(launch(handle,
                                             m,
                                             A->const_col_data,
                                             A->const_row_data,
                                             trm_info->get_diag_ind(),
                                             *dependency_count,
                                             zero_pivot,
                                             descr->base,
                                             descr->diag_type,
                                             descr->fill_mode));
        }

        // The longest column is reduced in the handle workspace, so the analysis needs no user buffer.
        int64_t max_nnz = 0; // important to set it to zero since sizeof_I might be int32_t.
        if(m > 0)
        {
            rocsparse::cscsv_max_nnz_launch_t max_nnz_launch;
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse::cscsv_max_nnz_launch_find(&max_nnz_launch, I_type));
            RETURN_IF_ROCSPARSE_ERROR(max_nnz_launch(handle, m, A->const_col_data, handle->buffer));
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(
                &max_nnz, handle->buffer, sizeof_I, hipMemcpyDeviceToHost, stream));
        }
        RETURN_IF_HIP_ERROR(rocsparse_hipStreamSynchronize(stream));
        trm_info->set_max_nnz(max_nnz);

        info->set(rocsparse_operation_none, descr->fill_mode, trm_info.release());
        return rocsparse_status_success;
    }

    rocsparse_csrsv_info csrsv_info{};
    switch(analysis_policy)
    {
    case rocsparse_analysis_policy_reuse:
    {
        info->set_shared_csrsv_info(A->info->get_shared_csrsv_info());
        csrsv_info = info->get_csrsv_info();
        break;
    }
    case rocsparse_analysis_policy_force:
    {
        // Recomputed in place when the transposed analysis is kept in the matrix info.
        csrsv_info = info->get_csrsv_info();
        break;
    }
    }

    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsv_analysis(handle,
                                                        rocsparse::cscsv_operation_to_csr(trans),
                                                        &mat_csr,
                                                        analysis_policy,
                                                        solve_policy,
                                                        &csrsv_info,
                                                        buffer));

    switch(analysis_policy)
    {
    case rocsparse_analysis_policy_reuse:
    {
        break;
    }
    case rocsparse_analysis_policy_force:
    {
        if(info->get_csrsv_info() == nullptr)
        {
            info->set_csrsv_info(csrsv_info);
        }
        break;
    }
    }
    return rocsparse_status_success;
}

bool rocsparse::cscsv_is_analyzed(rocsparse_cscsv_info        info,
                                  rocsparse_operation         trans,
                                  rocsparse_const_spmat_descr A)
{
    if(trans == rocsparse_operation_none)
    {
        return info->get(rocsparse_operation_none, A->descr->fill_mode) != nullptr;
    }
    if(info->get_csrsv_info() == nullptr)
    {
        return false;
    }

    // The transposed analysis is stored under the CSR operation and the flipped fill mode.
    _rocsparse_mat_descr   descr_csr;
    _rocsparse_spmat_descr mat_csr;
    rocsparse::build_csr_from_csc(*A, mat_csr, descr_csr);
    return info->get_csrsv_info()->get(rocsparse::cscsv_operation_to_csr(trans),
                                       descr_csr.fill_mode)
           != nullptr;
}
