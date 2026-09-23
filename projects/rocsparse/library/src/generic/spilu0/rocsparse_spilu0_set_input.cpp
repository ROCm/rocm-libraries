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

#include "rocsparse_spilu0_descr.hpp"
#include "rocsparse_utility.hpp"

#include "internal/generic/rocsparse_spilu0.h"

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spilu0_alg value)
{
    switch(value)
    {
    case rocsparse_spilu0_alg_default:
    {
        return false;
    }
    }
    return true;
};

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spilu0_input value)
{
    switch(value)
    {
    case rocsparse_spilu0_input_alg:
    case rocsparse_spilu0_input_compute_datatype:
    case rocsparse_spilu0_input_analysis_policy:
    case rocsparse_spilu0_input_boost_enable:
    case rocsparse_spilu0_input_boost_tolerance:
    case rocsparse_spilu0_input_boost_value:
    case rocsparse_spilu0_input_singularity_tolerance:
    {
        return false;
    }
    }
    return true;
};

template <>
bool rocsparse::enum_utils::is_invalid(rocsparse_spilu0_stage value)
{
    switch(value)
    {
    case rocsparse_spilu0_stage_analysis:
    case rocsparse_spilu0_stage_compute:
    {
        return false;
    }
    }
    return true;
};

extern "C" rocsparse_status rocsparse_spilu0_set_input(rocsparse_handle       handle,
                                                       rocsparse_spilu0_descr spilu0_descr,
                                                       rocsparse_spilu0_input input,
                                                       const void*            data,
                                                       size_t                 data_size_in_bytes,
                                                       rocsparse_error*       p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;

    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(1, spilu0_descr);
    ROCSPARSE_CHECKARG_ENUM(2, input);
    ROCSPARSE_CHECKARG_POINTER(3, data);

    switch(input)
    {

    case rocsparse_spilu0_input_alg:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_spilu0_alg),
                           rocsparse_status_invalid_value);

        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            spilu0_descr->get_stage() != ((rocsparse_spilu0_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_spilu0_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_spilu0_stage was executed");

        const rocsparse_spilu0_alg alg = *reinterpret_cast<const rocsparse_spilu0_alg*>(data);
        spilu0_descr->set_alg(alg);
        return rocsparse_status_success;
    }

    case rocsparse_spilu0_input_boost_enable:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(int32_t),
                           rocsparse_status_invalid_value);
        spilu0_descr->get_boost()->set_enable(*reinterpret_cast<const int32_t*>(data));
        return rocsparse_status_success;
    }

    case rocsparse_spilu0_input_boost_value:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(void*),
                           rocsparse_status_invalid_value);
        spilu0_descr->get_boost()->set_val_pointer_mode(handle->pointer_mode);
        spilu0_descr->get_boost()->set_val(data);
        return rocsparse_status_success;
    }

    case rocsparse_spilu0_input_boost_tolerance:
    {
        ROCSPARSE_CHECKARG(
            4,
            data_size_in_bytes,
            (data_size_in_bytes != sizeof(float) && data_size_in_bytes != sizeof(double)),
            rocsparse_status_invalid_value);
        spilu0_descr->get_boost()->set_tol(data);
        spilu0_descr->get_boost()->set_tol_pointer_mode(handle->pointer_mode);
        spilu0_descr->get_boost()->set_tol_datatype(rocsparse_datatype_f64_r);
        return rocsparse_status_success;
    }

    case rocsparse_spilu0_input_singularity_tolerance:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(const void*),
                           rocsparse_status_invalid_value);

        if(nullptr != spilu0_descr->get_csrilu0_info())
        {
            spilu0_descr->get_csrilu0_info()->get_singularity_numeric_near()->set_tolerance_pointer(
                data, handle->pointer_mode, rocsparse_datatype_f64_r);
        }
        else
        {
            spilu0_descr->set_tolerance_pointer(data);
            spilu0_descr->set_tolerance_pointer_mode(handle->pointer_mode);
        }

        return rocsparse_status_success;
    }

    case rocsparse_spilu0_input_analysis_policy:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_analysis_policy),
                           rocsparse_status_invalid_value);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            spilu0_descr->get_stage() != ((rocsparse_spilu0_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_spilu0_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_spilu0_stage was executed");

        const auto analysis_policy = *reinterpret_cast<const rocsparse_analysis_policy*>(data);
        spilu0_descr->set_analysis_policy(analysis_policy);
        return rocsparse_status_success;
    }

    case rocsparse_spilu0_input_compute_datatype:
    {
        ROCSPARSE_CHECKARG(4,
                           data_size_in_bytes,
                           data_size_in_bytes != sizeof(rocsparse_datatype),
                           rocsparse_status_invalid_value);
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
            spilu0_descr->get_stage() != ((rocsparse_spilu0_stage)-1)
                ? rocsparse_status_invalid_value
                : rocsparse_status_success,
            "rocsparse_spilu0_set_input cannot modify the descriptor after any of the stages "
            "rocsparse_spilu0_stage was executed");
        const rocsparse_datatype datatype = *reinterpret_cast<const rocsparse_datatype*>(data);
        spilu0_descr->set_compute_datatype(datatype);
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
