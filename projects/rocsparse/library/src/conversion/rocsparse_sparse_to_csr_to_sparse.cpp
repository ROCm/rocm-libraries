/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_sparse_to_csr_to_sparse.hpp"
#include "rocsparse_internal_spmat_print.hpp"
#include "rocsparse_utility.hpp"

rocsparse_status rocsparse::sparse_to_csr_to_sparse(rocsparse_handle                 handle_,
                                                    rocsparse_sparse_to_sparse_descr descr_,
                                                    rocsparse_const_spmat_descr      source_,
                                                    rocsparse_spmat_descr            target_,
                                                    _rocsparse_sparse_to_sparse_descr::stage stage_,
                                                    size_t* buffer_size_,
                                                    void*   buffer_)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(descr_->m_permissive == false)
    {
        std::stringstream message;
        message << "The conversion from " << source_->format << " to " << target_->format
                << " requires the calculation of an intermediate matrix with the CSR format, call "
                   "the routine rocsparse_sparse_to_sparse_permissive to allow such calculation.";
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                               message.str().c_str());
    }

    if(source_ == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    if(target_ == nullptr)
    {
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_pointer);
    }

    rocsparse_spmat_descr intermediate = nullptr;

    //
    //
    //
    switch(stage_)
    {
        //
        //
        //
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_analysis:
    {

        int64_t             csr_m{};
        int64_t             csr_n{};
        int64_t             csr_nnz{};
        rocsparse_indextype row_type = source_->row_type;
        rocsparse_indextype col_type = source_->col_type;
        //
        // Get csr csr dimensions.
        //
        switch(source_->format)
        {
        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_format_bsr:
        {
            row_type = source_->row_type;
            col_type = source_->col_type;
            csr_m    = source_->rows * source_->block_dim;
            csr_n    = source_->cols * source_->block_dim;
            csr_nnz  = source_->nnz * source_->block_dim * source_->block_dim;
            break;
        }
        case rocsparse_format_ell:
        {
            row_type = source_->row_type;
            col_type = source_->col_type;
            csr_m    = source_->rows;
            csr_n    = source_->cols;
            csr_nnz  = 0; // cannot know.

            if(target_->format == rocsparse_format_csc)
            {
                row_type = target_->col_type;
                col_type = target_->row_type;
            }
            if(target_->format == rocsparse_format_bsr)
            {
                row_type = target_->row_type;
                col_type = target_->col_type;
            }

            break;
        }
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        {
            row_type = source_->row_type;
            col_type = source_->col_type;
            csr_m    = source_->rows;
            csr_n    = source_->cols;
            csr_nnz  = source_->nnz;

            if(target_->format == rocsparse_format_csc)
            {
                row_type = target_->col_type;
                col_type = target_->row_type;
            }
            if(target_->format == rocsparse_format_bsr)
            {
                row_type = target_->row_type;
                col_type = target_->col_type;
            }

            break;
        }
        case rocsparse_format_csc:
        {
            row_type = source_->col_type;
            col_type = source_->row_type;
            csr_m    = source_->rows;
            csr_n    = source_->cols;
            csr_nnz  = source_->nnz;
            break;
        }
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }
        }
        //
        // Get intermediate descriptor.
        //
        void* TEMP_row__ = nullptr;
        void* TEMP_col__ = nullptr;
        void* TEMP_val__ = nullptr;
        if(csr_m > 0)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(
                &TEMP_row__, rocsparse::indextype_sizeof(row_type) * (csr_m + 1)));
        }
        RETURN_ROCSPARSE_ERROR_IF(rocsparse_status_type_mismatch,
                                  (csr_nnz > std::numeric_limits<int32_t>::max()
                                   && (row_type == rocsparse_indextype_i32)));

        if(csr_nnz > 0)
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(
                &TEMP_col__, rocsparse::indextype_sizeof(source_->col_type) * csr_nnz));
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(
                &TEMP_val__, rocsparse::datatype_sizeof(source_->data_type) * csr_nnz));
        }
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_csr_descr(&intermediate,
                                                             csr_m,
                                                             csr_n,
                                                             csr_nnz,
                                                             TEMP_row__,
                                                             TEMP_col__,
                                                             TEMP_val__,
                                                             row_type,
                                                             col_type,
                                                             source_->idx_base,
                                                             source_->data_type));

        //
        // Attach the intermediate.
        //
        descr_->m_intermediate = intermediate;
        rocsparse_sparse_to_sparse_descr intermediate_descr;
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_sparse_to_sparse_descr(
            &intermediate_descr, source_, intermediate, rocsparse_sparse_to_sparse_alg_default));

        //
        // Compute buffer size for intermediate conversion.
        //
        size_t buffer_size{};
        void*  buffer{};

        {
            size_t buffer_size_analysis{};
            RETURN_IF_ROCSPARSE_ERROR(
                rocsparse_sparse_to_sparse_buffer_size(handle_,
                                                       intermediate_descr,
                                                       source_,
                                                       intermediate,
                                                       rocsparse_sparse_to_sparse_stage_analysis,
                                                       &buffer_size_analysis));
            buffer_size = buffer_size_analysis;
        }

        //
        // Allocate the buffer.
        //
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

        //
        // Analysis.
        //
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse(handle_,
                                       intermediate_descr,
                                       source_,
                                       intermediate,
                                       rocsparse_sparse_to_sparse_stage_analysis,
                                       buffer_size,
                                       buffer));

        RETURN_IF_HIP_ERROR(rocsparse_hipFree(buffer));

        //
        // Here we know unknown size of intermediate if any.
        //
        switch(source_->format)
        {
        case rocsparse_format_ell:
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(
                &TEMP_col__,
                rocsparse::indextype_sizeof(intermediate->col_type) * intermediate->nnz));
            //
            // MUST BE CONDITIONAL
            //
            RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(
                &TEMP_val__,
                rocsparse::datatype_sizeof(intermediate->data_type) * intermediate->nnz));
            rocsparse_csr_set_pointers(intermediate, TEMP_row__, TEMP_col__, TEMP_val__);
            break;
        }

        case rocsparse_format_bsr:
        case rocsparse_format_coo:
        case rocsparse_format_coo_aos:
        case rocsparse_format_csc:
        {
            break;
        }
        case rocsparse_format_bell:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
        }
        case rocsparse_format_csr:
        {
            RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error);
        }
        }

        //
        // Buffer size
        //
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size(handle_,
                                                   intermediate_descr,
                                                   source_,
                                                   intermediate,
                                                   rocsparse_sparse_to_sparse_stage_compute,
                                                   &buffer_size));
        RETURN_IF_HIP_ERROR(rocsparse_hipMalloc(&buffer, buffer_size));

        //
        // Compute.
        //
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse(handle_,
                                       intermediate_descr,
                                       source_,
                                       intermediate,
                                       rocsparse_sparse_to_sparse_stage_compute,
                                       buffer_size,
                                       buffer));

        //
        // Free the buffer.
        //
        RETURN_IF_HIP_ERROR(rocsparse_hipFree(buffer));
        RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_sparse_to_sparse_descr(intermediate_descr));

        //
        // Request the buffer size of the conversion from the csr format to the target.
        //
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size(handle_,
                                                   descr_,
                                                   intermediate,
                                                   target_,
                                                   rocsparse_sparse_to_sparse_stage_analysis,
                                                   buffer_size_));
        return rocsparse_status_success;
    }

    case _rocsparse_sparse_to_sparse_descr::stage_analysis:
    {
        //
        // Request the analysis  of the conversion from the csr format to the target.
        //
        rocsparse_spmat_descr intermediate = descr_->m_intermediate;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse(handle_,
                                       descr_,
                                       intermediate,
                                       target_,
                                       rocsparse_sparse_to_sparse_stage_analysis,
                                       buffer_size_[0],
                                       buffer_));
        return rocsparse_status_success;
    }

    //
    //
    //
    case _rocsparse_sparse_to_sparse_descr::stage_buffer_size_compute:
    {
        rocsparse_spmat_descr intermediate = descr_->m_intermediate;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse_buffer_size(handle_,
                                                   descr_,
                                                   intermediate,
                                                   target_,
                                                   rocsparse_sparse_to_sparse_stage_compute,
                                                   buffer_size_));
        return rocsparse_status_success;
    }

    //
    //
    //
    case _rocsparse_sparse_to_sparse_descr::stage_compute:
    {
        rocsparse_spmat_descr intermediate = descr_->m_intermediate;
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse_sparse_to_sparse(handle_,
                                       descr_,
                                       intermediate,
                                       target_,
                                       rocsparse_sparse_to_sparse_stage_compute,
                                       buffer_size_[0],
                                       buffer_));
        return rocsparse_status_success;
    }

        //
        //
        //
    }

    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
    // LCOV_EXCL_STOP
}
