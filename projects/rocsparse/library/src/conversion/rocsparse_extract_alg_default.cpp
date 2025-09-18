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

#include "rocsparse_extract_alg_default.hpp"
#include "rocsparse_primitives.hpp"

namespace rocsparse
{

    template <typename I>
    static rocsparse_status internal_extract_buffer_size_template(rocsparse_handle    handle_,
                                                                  rocsparse_direction source_dir_,
                                                                  int64_t             source_m_,
                                                                  int64_t             source_n_,
                                                                  size_t*             buffer_size_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if((source_m_ > std::numeric_limits<I>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        if((source_n_ > std::numeric_limits<I>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        I num_seq;
        switch(source_dir_)
        {
        case rocsparse_direction_row:
        {
            num_seq = source_m_;
            break;
        }
        case rocsparse_direction_column:
        {
            num_seq = source_n_;
            break;
        }
        }

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::primitives::inclusive_scan_buffer_size<I, I>(
            handle_, num_seq + 1, buffer_size_)));

        return rocsparse_status_success;
    }

    template <typename... P>
    static rocsparse_status internal_extract_buffer_size_dispatch(rocsparse_indextype indextype_I,
                                                                  P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(indextype_I)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                                   "rocsparse_indextype_u16 is not supported");
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_buffer_size_template<int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_buffer_size_template<int64_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    template <typename I, typename J>
    static rocsparse_status internal_extract_inclusive_scan(rocsparse_handle handle_,
                                                            J                nseq_,
                                                            I* __restrict__ ptr_,
                                                            size_t buffer_size_,
                                                            void* __restrict__ buffer_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::primitives::inclusive_scan(
            handle_, ptr_, ptr_, nseq_ + 1, buffer_size_, buffer_));
        return rocsparse_status_success;
    }

    template <uint32_t BLOCKSIZE, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void extract_count_kernel(J nseq_,
                              const I* __restrict__ source_ptr_,
                              const J* __restrict__ source_ind_,
                              rocsparse_index_base base_,
                              bool                 extract_before_diagonal_,
                              rocsparse_diag_type  target_diag_,
                              I* __restrict__ target_ptr_)
    {
        const I seq = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(seq < nseq_)
        {
            I    count     = 0;
            auto predicate = [extract_before_diagonal_, target_diag_](J i, J j) {
                return (extract_before_diagonal_)
                           ? ((target_diag_ == rocsparse_diag_type_unit) ? (i > j) : (i >= j))
                           : ((target_diag_ == rocsparse_diag_type_unit) ? (i < j) : (i <= j));
            };
            for(I k = source_ptr_[seq] - base_; k < source_ptr_[seq + 1] - base_; ++k)
            {
                const J ind = source_ind_[k] - base_;
                if(predicate(seq, ind))
                {
                    ++count;
                }
            }

            //
            // We should include a force mode to include the diagonal.
            //
            target_ptr_[seq + 1] = count;
        }
    }

    template <typename I, typename J>
    ROCSPARSE_KERNEL(1)
    void extract_grab_nnz_kernel(J n_,
                                 const I* __restrict__ ptr_,
                                 rocsparse_index_base base_,
                                 int64_t* __restrict__ nnz_)
    {
        nnz_[0] = ptr_[n_] - base_;
    }

    template <typename T, typename I, typename J>
    static rocsparse_status
        internal_extract_analysis_template(rocsparse_handle    handle_,
                                           rocsparse_direction source_dir_,
                                           int64_t             source_m_,
                                           int64_t             source_n_,
                                           int64_t             source_nnz_,
                                           const void* __restrict__ source_ptr_,
                                           const void* __restrict__ source_ind_,
                                           const void* __restrict__ source_val_,
                                           rocsparse_index_base source_base_,
                                           rocsparse_fill_mode  target_fill_mode_,
                                           rocsparse_diag_type  target_diag_,
                                           void* __restrict__ target_nnz_,
                                           void* __restrict__ target_ptr_,
                                           rocsparse_index_base target_base_,
                                           size_t               buffer_size_,
                                           void* __restrict__ buffer_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if((source_m_ > std::numeric_limits<J>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        if((source_n_ > std::numeric_limits<J>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        if((source_nnz_ > std::numeric_limits<I>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        J    num_seq;
        bool extract_before_diagonal;
        switch(source_dir_)
        {
        case rocsparse_direction_row:
        {
            num_seq = source_m_;
            switch(target_fill_mode_)
            {
            case rocsparse_fill_mode_lower:
            {
                extract_before_diagonal = true;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                extract_before_diagonal = false;
                break;
            }
            }
            break;
        }
        case rocsparse_direction_column:
        {
            num_seq = source_n_;
            switch(target_fill_mode_)
            {
            case rocsparse_fill_mode_lower:
            {
                extract_before_diagonal = false;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                extract_before_diagonal = true;
                break;
            }
            }
            break;
        }
        }

        //
        // Setting count to 0.
        //
        RETURN_IF_HIP_ERROR(
            hipMemsetAsync(target_ptr_, 0, sizeof(I) * (num_seq + 1), handle_->stream));

        {
            const I base_value = static_cast<I>(target_base_);
            RETURN_IF_HIP_ERROR(hipMemcpyAsync(
                target_ptr_, &base_value, sizeof(I), hipMemcpyHostToDevice, handle_->stream));
        }

        static constexpr int nthreads_per_block = 1024;
        dim3                 threads(nthreads_per_block);
        J                    nblocks = (num_seq - 1) / nthreads_per_block + 1;
        dim3                 blocks(nblocks);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::extract_count_kernel<nthreads_per_block, I, J>),
            blocks,
            threads,
            0,
            handle_->stream,
            num_seq,
            (const I* __restrict__)source_ptr_,
            (const J* __restrict__)source_ind_,
            source_base_,
            extract_before_diagonal,
            target_diag_,
            (I* __restrict__)target_ptr_);

        RETURN_IF_ROCSPARSE_ERROR(rocsparse::internal_extract_inclusive_scan(
            handle_, num_seq, (I* __restrict__)target_ptr_, buffer_size_, buffer_));
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR((rocsparse::extract_grab_nnz_kernel<I, J>),
                                           dim3(1),
                                           dim3(1),
                                           0,
                                           handle_->stream,
                                           num_seq,
                                           (const I* __restrict__)target_ptr_,
                                           target_base_,
                                           ((int64_t* __restrict__)target_nnz_));

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename... P>
    static rocsparse_status internal_extract_analysis_dispatch_J(rocsparse_indextype indextype_J,
                                                                 P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(indextype_J)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                                   "rocsparse_indextype_u16 is not supported");
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_template<T, I, int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_template<T, I, int64_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    template <typename T, typename... P>
    static rocsparse_status internal_extract_analysis_dispatch_I(rocsparse_indextype indextype_I,
                                                                 P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(indextype_I)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                                   "rocsparse_indextype_u16 is not supported");
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_J<T, int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_J<T, int64_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    template <typename... P>
    static rocsparse_status internal_extract_analysis_dispatch(rocsparse_datatype datatype_T,
                                                               P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(datatype_T)
        {
        case rocsparse_datatype_f16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<_Float16>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_bf16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (internal_extract_analysis_dispatch_I<rocsparse_bfloat16>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<float>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<double>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (internal_extract_analysis_dispatch_I<rocsparse_float_complex>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (internal_extract_analysis_dispatch_I<rocsparse_double_complex>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_i8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<int8_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<uint8_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_i32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_analysis_dispatch_I<uint32_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    template <uint32_t BLOCKSIZE, typename T, typename I, typename J>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void internal_extract_fill_kernel(J nseq_,
                                      const I* __restrict__ source_ptr_,
                                      const J* __restrict__ source_ind_,
                                      const T* __restrict__ source_val_,
                                      rocsparse_index_base source_base_,
                                      bool                 extract_before_diagonal_,
                                      rocsparse_diag_type  target_diag_,
                                      const I* __restrict__ target_ptr_,
                                      J* __restrict__ target_ind_,
                                      T* __restrict__ target_val_,
                                      rocsparse_index_base target_base_)
    {
        const I seq = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        if(seq < nseq_)
        {
            const I end          = source_ptr_[seq + 1] - source_base_;
            I       target_start = target_ptr_[seq] - target_base_;
            auto    predicate    = [extract_before_diagonal_, target_diag_](J i, J j) {
                return (extract_before_diagonal_)
                                 ? ((target_diag_ == rocsparse_diag_type_unit) ? (i > j) : (i >= j))
                                 : ((target_diag_ == rocsparse_diag_type_unit) ? (i < j) : (i <= j));
            };

            for(I k = source_ptr_[seq] - source_base_; k < end; ++k)
            {
                const J ind = source_ind_[k] - source_base_;
                if(predicate(seq, ind))
                {
                    target_ind_[target_start] = ind + target_base_;
                    target_val_[target_start] = source_val_[k];
                    ++target_start;
                }
            }
        }
    }

    template <typename T, typename I, typename J>
    rocsparse_status internal_extract_compute_template(rocsparse_handle    handle_,
                                                       rocsparse_direction source_dir_,
                                                       int64_t             source_m_,
                                                       int64_t             source_n_,
                                                       int64_t             source_nnz_,
                                                       const void* __restrict__ source_ptr_,
                                                       const void* __restrict__ source_ind_,
                                                       const void* __restrict__ source_val_,
                                                       rocsparse_index_base source_base_,
                                                       rocsparse_fill_mode  target_fill_mode_,
                                                       rocsparse_diag_type  target_diag_,
                                                       void* __restrict__ target_ptr_,
                                                       void* __restrict__ target_ind_,
                                                       void* __restrict__ target_val_,
                                                       rocsparse_index_base target_base_,
                                                       void* __restrict__ buffer_)
    {
        ROCSPARSE_ROUTINE_TRACE;

        if((source_m_ > std::numeric_limits<J>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        if((source_n_ > std::numeric_limits<J>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        if((source_nnz_ > std::numeric_limits<I>::max()))
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error,
                                                   "out of bound dimension with index type");
        }

        J    num_seq;
        bool extract_before_diagonal;
        switch(source_dir_)
        {
        case rocsparse_direction_row:
        {
            num_seq = source_m_;
            switch(target_fill_mode_)
            {
            case rocsparse_fill_mode_lower:
            {
                extract_before_diagonal = true;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                extract_before_diagonal = false;
                break;
            }
            }
            break;
        }
        case rocsparse_direction_column:
        {
            num_seq = source_n_;
            switch(target_fill_mode_)
            {
            case rocsparse_fill_mode_lower:
            {
                extract_before_diagonal = false;
                break;
            }
            case rocsparse_fill_mode_upper:
            {
                extract_before_diagonal = true;
                break;
            }
            }
            break;
        }
        }

        static constexpr uint32_t nthreads_per_block = 1024;
        dim3                      threads(nthreads_per_block);
        J                         nblocks = (num_seq - 1) / nthreads_per_block + 1;
        dim3                      blocks(nblocks);

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::internal_extract_fill_kernel<nthreads_per_block, T, I, J>),
            blocks,
            threads,
            0,
            handle_->stream,
            num_seq,
            (const I* __restrict__)source_ptr_,
            (const J* __restrict__)source_ind_,
            (const T* __restrict__)source_val_,
            source_base_,
            extract_before_diagonal,
            target_diag_,
            (I* __restrict__)target_ptr_,
            (J* __restrict__)target_ind_,
            (T* __restrict__)target_val_,
            target_base_);

        return rocsparse_status_success;
    }

    template <typename T, typename I, typename... P>
    static rocsparse_status internal_extract_compute_dispatch_J(rocsparse_indextype indextype_J,
                                                                P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(indextype_J)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                                   "rocsparse_indextype_u16 is not supported");
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_template<T, I, int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_template<T, I, int64_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    template <typename T, typename... P>
    static rocsparse_status internal_extract_compute_dispatch_I(rocsparse_indextype indextype_I,
                                                                P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(indextype_I)
        {
        case rocsparse_indextype_u16:
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented,
                                                   "rocsparse_indextype_u16 is not supported");
        }
        case rocsparse_indextype_i32:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_J<T, int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_indextype_i64:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_J<T, int64_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

    template <typename... P>
    static rocsparse_status internal_extract_compute_dispatch(rocsparse_datatype datatype_T,
                                                              P&&... p)
    {
        ROCSPARSE_ROUTINE_TRACE;

        switch(datatype_T)
        {
        case rocsparse_datatype_f16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<_Float16>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_bf16_r:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (internal_extract_compute_dispatch_I<rocsparse_bfloat16>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<float>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<double>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f32_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (internal_extract_compute_dispatch_I<rocsparse_float_complex>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_f64_c:
        {
            RETURN_IF_ROCSPARSE_ERROR(
                (internal_extract_compute_dispatch_I<rocsparse_double_complex>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_i8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<int8_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u8_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<uint8_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_i32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<int32_t>)(p...));
            return rocsparse_status_success;
        }
        case rocsparse_datatype_u32_r:
        {
            RETURN_IF_ROCSPARSE_ERROR((internal_extract_compute_dispatch_I<uint32_t>)(p...));
            return rocsparse_status_success;
        }
        }
        // LCOV_EXCL_START
        return rocsparse_status_invalid_value;
        // LCOV_EXCL_STOP
    }

}

rocsparse_extract_descr_default_t::rocsparse_extract_descr_default_t(
    rocsparse_const_spmat_descr source, rocsparse_const_spmat_descr target)
    : _rocsparse_extract_descr(rocsparse_extract_alg_default, source, target)
{
    ROCSPARSE_ROUTINE_TRACE;

    if(source->format != target->format)
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_internal_error,
            "source and target matrices must have the same matrix format");
    }

    switch(source->format)
    {
    case rocsparse_format_csr:
    {
        this->m_direction = rocsparse_direction_row;
        break;
    }
    case rocsparse_format_csc:
    {
        this->m_direction = rocsparse_direction_column;
        break;
    }
    case rocsparse_format_ell:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_bsr:
    case rocsparse_format_bell:
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_not_implemented,
            "supported matrix format are rocsparse_format_csr and rocsparse_format_csc");
    }
    }

    if(source->descr->storage_mode != target->descr->storage_mode)
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_internal_error,
            "source and target matrices must have the same storage mode");
    }
}

rocsparse_status
    rocsparse_extract_descr_default_t::buffer_size(rocsparse_handle            handle,
                                                   rocsparse_const_spmat_descr source,
                                                   rocsparse_spmat_descr       target,
                                                   rocsparse_extract_stage     stage,
                                                   size_t* __restrict__ buffer_size_in_bytes)
{
    ROCSPARSE_ROUTINE_TRACE;

    switch(stage)
    {
    case rocsparse_extract_stage_analysis:
    {

        RETURN_IF_ROCSPARSE_ERROR((rocsparse::internal_extract_buffer_size_dispatch(
            (this->m_direction == rocsparse_direction_row) ? target->row_type : target->col_type,
            handle,
            this->m_direction,
            source->rows,
            source->cols,
            buffer_size_in_bytes)));

        this->m_stage_analysis_buffer_size = buffer_size_in_bytes[0];
        break;
    }

    case rocsparse_extract_stage_compute:
    {
        this->m_stage_compute_buffer_size = 0;
        break;
    }
    }

    return rocsparse_status_success;
}

rocsparse_status rocsparse_extract_descr_default_t::run(rocsparse_handle            handle,
                                                        rocsparse_const_spmat_descr source,
                                                        rocsparse_spmat_descr       target,
                                                        rocsparse_extract_stage     stage,
                                                        size_t                      buffer_size,
                                                        void* __restrict__ buffer)

{
    ROCSPARSE_ROUTINE_TRACE;

    const rocsparse_fill_mode target_fill_mode = target->descr->fill_mode;
    const rocsparse_diag_type target_diag_type = target->descr->diag_type;

    const void* __restrict__ const_source_ptr_data = nullptr;
    const void* __restrict__ const_source_ind_data = nullptr;
    void* __restrict__ target_ptr_data             = nullptr;
    void* __restrict__ target_ind_data             = nullptr;

    switch(source->format)
    {
    case rocsparse_format_csr:
    {
        const_source_ptr_data = source->row_data;
        const_source_ind_data = source->col_data;
        target_ptr_data       = target->row_data;
        target_ind_data       = target->col_data;
        break;
    }
    case rocsparse_format_csc:
    {
        const_source_ptr_data = source->col_data;
        const_source_ind_data = source->row_data;
        target_ptr_data       = target->col_data;
        target_ind_data       = target->row_data;
        break;
    }
    case rocsparse_format_ell:
    case rocsparse_format_coo:
    case rocsparse_format_coo_aos:
    case rocsparse_format_bsr:
    case rocsparse_format_bell:
    {
        THROW_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            rocsparse_status_not_implemented,
            "supported matrix format are rocsparse_format_csr and rocsparse_format_csc");
    }
    }

    switch(stage)
    {
    case rocsparse_extract_stage_analysis:
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::internal_extract_analysis_dispatch(
            source->data_type,
            (this->m_direction == rocsparse_direction_row) ? target->row_type : target->col_type,
            (this->m_direction == rocsparse_direction_row) ? target->col_type : target->row_type,
            handle,
            this->m_direction,
            source->rows,
            source->cols,
            source->nnz,
            const_source_ptr_data,
            const_source_ind_data,
            source->const_val_data,
            source->idx_base,
            target_fill_mode,
            target_diag_type,
            this->m_device_nnz,
            target_ptr_data,
            target->idx_base,
            buffer_size,
            buffer)));

        break;
    }

    case rocsparse_extract_stage_compute:
    {
        RETURN_IF_ROCSPARSE_ERROR((rocsparse::internal_extract_compute_dispatch(
            source->data_type,
            (this->m_direction == rocsparse_direction_row) ? target->row_type : target->col_type,
            (this->m_direction == rocsparse_direction_row) ? target->col_type : target->row_type,
            handle,
            this->m_direction,
            source->rows,
            source->cols,
            source->nnz,
            const_source_ptr_data,
            const_source_ind_data,
            source->const_val_data,
            source->idx_base,
            //
            target_fill_mode,
            target_diag_type,
            target_ptr_data,
            target_ind_data,
            target->val_data,
            target->idx_base,
            //
            buffer)));
        break;
    }
    }
    return rocsparse_status_success;
}
