/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "internal/generic/rocsparse_sptrsm.h"
#include "rocsparse_control.hpp"
#include "rocsparse_enum_utils.hpp"
#include "rocsparse_handle.hpp"
#include "rocsparse_utility.hpp"

#include "../../conversion/rocsparse_convert_array.hpp"
#include "../../conversion/rocsparse_convert_scalar.hpp"
#include "../rocsparse_sptrsm_descr.hpp"
#include "internal/level3/rocsparse_csrsm.h"
#include "rocsparse_common.h"
#include "rocsparse_coosm.hpp"
#include "rocsparse_csrsm.hpp"

template <>
inline bool rocsparse::enum_utils::is_invalid(rocsparse_sptrsm_input value)
{
    switch(value)
    {
    case rocsparse_sptrsm_input_alg:
    case rocsparse_sptrsm_input_operation_A:
    case rocsparse_sptrsm_input_operation_X:
    case rocsparse_sptrsm_input_compute_datatype:
    case rocsparse_sptrsm_input_scalar_datatype:
    case rocsparse_sptrsm_input_scalar_alpha:
    case rocsparse_sptrsm_input_analysis_policy:
    {
        return false;
    }
    }
    return true;
};

extern "C" rocsparse_status rocsparse_sptrsm_set_input(rocsparse_handle       handle,
                                                       rocsparse_sptrsm_descr sptrsm_descr,
                                                       rocsparse_sptrsm_input input,
                                                       const void*            data,
                                                       size_t                 data_size_in_bytes,
                                                       rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, sptrsm_descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, data);

    switch(input)
    {
    case rocsparse_sptrsm_input_alg:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsm_descr->get_stage() != ((rocsparse_sptrsm_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsm_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsm_stage was executed");
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_sptrsm_alg),
                           rocsparse_status_invalid_size);
        const rocsparse_sptrsm_alg alg = *reinterpret_cast<const rocsparse_sptrsm_alg*>(data);
        sptrsm_descr->set_alg(alg);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_scalar_alpha:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(const void*),
                           rocsparse_status_invalid_size);
        sptrsm_descr->set_scalar_alpha(data);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_scalar_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        sptrsm_descr->set_scalar_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_compute_datatype:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsm_descr->get_stage() != ((rocsparse_sptrsm_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsm_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsm_stage was executed");
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_size);
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        sptrsm_descr->set_compute_datatype(datatype);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_analysis_policy:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsm_descr->get_stage() != ((rocsparse_sptrsm_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsm_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsm_stage was executed");
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_analysis_policy),
                           rocsparse_status_invalid_size);
        const rocsparse_analysis_policy policy
            = *reinterpret_cast<const rocsparse_analysis_policy*>(data);
        sptrsm_descr->set_analysis_policy(policy);
        return rocsparse_status_success;
    }
    case rocsparse_sptrsm_input_operation_A:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsm_descr->get_stage() != ((rocsparse_sptrsm_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsm_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsm_stage was executed");
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(data);
        sptrsm_descr->set_operation_A(op);
        return rocsparse_status_success;
    }

    case rocsparse_sptrsm_input_operation_X:
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            sptrsm_descr->get_stage() != ((rocsparse_sptrsm_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_sptrsm_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_sptrsm_stage was executed");
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_operation),
                           rocsparse_status_invalid_size);
        const rocsparse_operation op = *reinterpret_cast<const rocsparse_operation*>(data);
        sptrsm_descr->set_operation_X(op);
        return rocsparse_status_success;
    }
    }
    // LCOV_EXCL_START
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value);
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
