/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <sstream>

#include "rocsparse.h"
#include "rocsparse_common.h"
#include "rocsparse_common.hpp"
#include "rocsparse_control.hpp"
#include "rocsparse_determine_indextype.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "rocsparse_coosm.hpp"
#include "rocsparse_cscsm.hpp"
#include "rocsparse_csrsm.hpp"

// LCOV_EXCL_START
template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spsm_alg value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spsm_alg_default);
#undef CASE
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}

template <>
const char* rocsparse::enum_utils::to_string(rocsparse_spsm_stage value_)
{
#define CASE(C) \
    case C:     \
        return #C
    switch(value_)
    {
        CASE(rocsparse_spsm_stage_buffer_size);
        CASE(rocsparse_spsm_stage_preprocess);
        CASE(rocsparse_spsm_stage_compute);
#undef CASE
    }
    THROW_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
// LCOV_EXCL_STOP

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsm_alg value_)
{
    switch(value_)
    {
    case rocsparse_spsm_alg_default:
    {
        return false;
    }
    }
    return true;
}

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spsm_stage value_)
{
    switch(value_)
    {
    case rocsparse_spsm_stage_buffer_size:
    case rocsparse_spsm_stage_preprocess:
    case rocsparse_spsm_stage_compute:
    {
        return false;
    }
    }
    return true;
}

namespace rocsparse
{

    rocsparse_status spsm(rocsparse_handle            handle,
                          rocsparse_operation         trans_A,
                          rocsparse_operation         trans_B,
                          const void*                 alpha_pointer,
                          rocsparse_const_spmat_descr matA,
                          rocsparse_const_dnmat_descr matB,
                          const rocsparse_dnmat_descr matC,
                          rocsparse_spsm_alg          alg,
                          rocsparse_spsm_stage        stage,
                          size_t*                     buffer_size,
                          void*                       temp_buffer)
    {
        ROCSPARSE_ROUTINE_TRACE;

        const rocsparse_datatype alpha_datatype = matA->data_type;

        _rocsparse_dnvec_descr alpha_st(
            1, 1, alpha_datatype, alpha_pointer, nullptr, 1, 0, handle->pointer_mode);
        rocsparse_dnvec_descr alpha = &alpha_st;

        const bool is_C_column_order = (matC->order == rocsparse_order_column);
        switch(stage)
        {
        case rocsparse_spsm_stage_buffer_size:
        {
            _rocsparse_dnmat_descr matC_row_order{true,
                                                  matC->rows,
                                                  matC->cols,
                                                  matC->cols,
                                                  (void*)0x4,
                                                  (const void*)0x4,
                                                  matC->data_type,
                                                  rocsparse_order_row,
                                                  1,
                                                  0};

            switch(matA->format)
            {
#ifndef ROCSPARSE_WITH_CSC_TRSM
            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

#else

            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_buffer_size(handle,
                                                                       matC->cols,
                                                                       trans_A,
                                                                       trans_B,
                                                                       alpha,
                                                                       matA,
                                                                       &matC_row_order,
                                                                       buffer_size,
                                                                       nullptr));
                break;
            }
#endif
            case rocsparse_format_csr:
            {

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_buffer_size(handle,
                                                                       matC->cols,
                                                                       trans_A,
                                                                       trans_B,
                                                                       alpha,
                                                                       matA,
                                                                       &matC_row_order,
                                                                       buffer_size,
                                                                       nullptr));
                break;
            }

            case rocsparse_format_coo:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_buffer_size(handle,
                                                                       matC->cols,
                                                                       trans_A,
                                                                       trans_B,
                                                                       alpha,
                                                                       matA,
                                                                       &matC_row_order,
                                                                       buffer_size,
                                                                       nullptr));

                break;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            if(is_C_column_order)
            {
                *buffer_size += rocsparse::align_size(rocsparse::datatype_sizeof(matB->data_type)
                                                      * matB->rows * matB->cols);
            }
            return rocsparse_status_success;
        }

        case rocsparse_spsm_stage_preprocess:
        {
            void* csrsm_buffer = temp_buffer;
            if(is_C_column_order)
            {
                csrsm_buffer = reinterpret_cast<char*>(temp_buffer)
                               + rocsparse::align_size(rocsparse::datatype_sizeof(matB->data_type)
                                                       * matB->rows * matB->cols);
            }

            const auto nrhs = matC->cols;
            //
            // Preprocess.
            //
            _rocsparse_dnmat_descr local_Z{true,
                                           matA->rows,
                                           nrhs,
                                           nrhs,
                                           (void*)0x4,
                                           (const void*)0x4,
                                           matC->data_type,
                                           rocsparse_order_row,
                                           1,
                                           0};

            switch(matA->format)
            {
#ifndef ROCSPARSE_WITH_CSC_TRSM
            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

#else
            case rocsparse_format_csc:
            {
                auto csrsm_info = matA->info->get_csrsm_info();
                if(csrsm_info->get(trans_A, matA->descr->fill_mode) == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::cscsm_analysis(handle,
                                                  nrhs,
                                                  trans_A,
                                                  trans_B,
                                                  alpha,
                                                  matA,
                                                  &local_Z,
                                                  rocsparse_analysis_policy_force,
                                                  &csrsm_info,
                                                  std::numeric_limits<size_t>::max(),
                                                  csrsm_buffer,
                                                  nullptr));
                }
                return rocsparse_status_success;
            }
#endif

            case rocsparse_format_csr:
            {
                rocsparse_csrsm_info csrsm_info = matA->info->get_csrsm_info();
                if(csrsm_info->get(trans_A, matA->descr->fill_mode) == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::csrsm_analysis(handle,
                                                  nrhs,
                                                  trans_A,
                                                  trans_B,
                                                  alpha,
                                                  matA,
                                                  &local_Z,
                                                  rocsparse_analysis_policy_force,
                                                  &csrsm_info,
                                                  std::numeric_limits<size_t>::max(),
                                                  csrsm_buffer,
                                                  nullptr));
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo:
            {
                rocsparse_csrsm_info csrsm_info = matA->info->get_csrsm_info();
                if(csrsm_info->get(trans_A, matA->descr->fill_mode) == nullptr)
                {
                    RETURN_IF_ROCSPARSE_ERROR(
                        rocsparse::coosm_analysis(handle,
                                                  nrhs,
                                                  trans_A,
                                                  trans_B,
                                                  alpha,
                                                  matA,
                                                  &local_Z,
                                                  rocsparse_analysis_policy_force,
                                                  &csrsm_info,
                                                  std::numeric_limits<size_t>::max(),
                                                  csrsm_buffer,
                                                  nullptr));
                }
                return rocsparse_status_success;
            }

            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }
        }

        case rocsparse_spsm_stage_compute:
        {

            const bool is_B_row_order    = (matB->order == rocsparse_order_row);
            const bool is_B_op_transpose = (trans_B != rocsparse_operation_none);

            rocsparse_error* p_error{};
            void*            matsm_buffer               = temp_buffer;
            size_t           matsm_buffer_size_in_bytes = std::numeric_limits<size_t>::max();
            if(is_C_column_order)
            {
                const size_t nbytes = rocsparse::align_size(
                    rocsparse::datatype_sizeof(matB->data_type) * matB->rows * matB->cols);
                matsm_buffer = reinterpret_cast<char*>(temp_buffer) + nbytes;
                matsm_buffer_size_in_bytes -= nbytes;
            }

            const auto nrhs = matC->cols;

            _rocsparse_dnmat_descr Z_st{true,
                                        matA->rows,
                                        nrhs,
                                        nrhs,
                                        temp_buffer,
                                        temp_buffer,
                                        matB->data_type,
                                        rocsparse_order_row,
                                        matB->batch_count,

                                        (matB->batch_stride == 0) ? 0 : (matA->rows * nrhs)};

            rocsparse_dnmat_descr Z = (is_C_column_order) ? &Z_st : matC;

            if(is_B_op_transpose)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::dnmat_transpose(handle, nullptr, matB, Z, p_error));
            }
            else if(is_B_row_order)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::dnmat_copy_data(handle, nullptr, matB, Z, p_error));
            }
            else
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::dnmat_switch_order(handle, nullptr, matB, Z, p_error));
            }

            //
            // Compute
            //
            switch(matA->format)
            {
            case rocsparse_format_csr:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse::csrsm_compute(handle,
                                                                   nrhs,
                                                                   trans_A,
                                                                   trans_B,
                                                                   alpha,
                                                                   matA,
                                                                   Z,
                                                                   matA->info->get_csrsm_info(),
                                                                   matsm_buffer_size_in_bytes,
                                                                   matsm_buffer,
                                                                   p_error));

                break;
            }

            case rocsparse_format_coo:
            {

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::coosm_compute(handle,
                                                                   nrhs,
                                                                   trans_A,
                                                                   trans_B,
                                                                   alpha,
                                                                   matA,
                                                                   Z,
                                                                   matA->info->get_csrsm_info(),
                                                                   matsm_buffer_size_in_bytes,
                                                                   matsm_buffer,
                                                                   p_error));

                break;
            }

#ifndef ROCSPARSE_WITH_CSC_TRSM

            case rocsparse_format_csc:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }

#else
            case rocsparse_format_csc:
            {

                RETURN_IF_ROCSPARSE_ERROR(rocsparse::cscsm_compute(handle,
                                                                   nrhs,
                                                                   trans_A,
                                                                   trans_B,
                                                                   alpha,
                                                                   matA,
                                                                   Z,
                                                                   matA->info->get_csrsm_info(),
                                                                   matsm_buffer_size_in_bytes,
                                                                   matsm_buffer,
                                                                   p_error));

                break;
            }
#endif

            case rocsparse_format_coo_aos:
            case rocsparse_format_bsr:
            case rocsparse_format_ell:
            case rocsparse_format_bell:
            case rocsparse_format_sell:
            {
                RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_not_implemented);
            }
            }

            if(is_C_column_order)
            {
                RETURN_IF_ROCSPARSE_ERROR(
                    rocsparse::dnmat_switch_order(handle, nullptr, Z, matC, p_error));
            }
            return rocsparse_status_success;
        }
        }
    }

}

/*
 * ===========================================================================
 *    C wrapper
 * ===========================================================================
 */

extern "C" rocsparse_status rocsparse_spsm(rocsparse_handle            handle, //0
                                           rocsparse_operation         trans_A, //1
                                           rocsparse_operation         trans_B, //2
                                           const void*                 alpha, //3
                                           rocsparse_const_spmat_descr matA, //4
                                           rocsparse_const_dnmat_descr matB, //5
                                           const rocsparse_dnmat_descr matC, //6
                                           rocsparse_datatype          compute_type, //7
                                           rocsparse_spsm_alg          alg, //8
                                           rocsparse_spsm_stage        stage, //9
                                           size_t*                     buffer_size, //10
                                           void*                       temp_buffer) //11
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_ENUM(1, trans_A);
    ROCSPARSE_CHECKARG_ENUM(2, trans_B);
    ROCSPARSE_CHECKARG_POINTER(3, alpha);
    ROCSPARSE_CHECKARG_POINTER(4, matA);
    ROCSPARSE_CHECKARG(4, matA, matA->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(4, matA, matA->batch_count != 1, rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG_POINTER(5, matB);
    ROCSPARSE_CHECKARG(5, matB, matB->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(5, matB, matB->batch_count != 1, rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG_POINTER(6, matC);
    ROCSPARSE_CHECKARG(6, matC, matC->init == false, rocsparse_status_not_initialized);
    ROCSPARSE_CHECKARG(6, matC, matC->batch_count != 1, rocsparse_status_not_implemented);
    ROCSPARSE_CHECKARG_ENUM(7, compute_type);
    ROCSPARSE_CHECKARG(7,
                       compute_type,
                       (compute_type != matA->data_type || compute_type != matB->data_type
                        || compute_type != matC->data_type),
                       rocsparse_status_not_implemented);

    ROCSPARSE_CHECKARG_ENUM(8, alg);
    ROCSPARSE_CHECKARG_ENUM(9, stage);

    switch(stage)
    {
    case rocsparse_spsm_stage_buffer_size:
    {
        ROCSPARSE_CHECKARG_POINTER(10, buffer_size);
        break;
    }
    case rocsparse_spsm_stage_preprocess:
    {
        break;
    }
    case rocsparse_spsm_stage_compute:
    {
        break;
    }
    }

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::spsm(
        handle, trans_A, trans_B, alpha, matA, matB, matC, alg, stage, buffer_size, temp_buffer));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
