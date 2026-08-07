/* **************************************************************************
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include "rocsolver_handle.hpp"
#include "exceptions.hpp"
#include "rocblas.hpp"

#include <memory>

ROCSOLVER_BEGIN_NAMESPACE

//------------------------------------------------------------------------------
rocblas_status rocsolver_set_alg_mode_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           const rocsolver_alg_mode mode)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;
    if(mode == rocsolver_alg_mode_mixed)
        return rocblas_status_invalid_value;

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data == nullptr)
    {
        handle_ptr = std::make_shared<rocsolver_handle_data_>();
        handle_data = (rocsolver_handle_data)handle_ptr.get();
        handle_data->checksum = sizeof(rocsolver_handle_data_);

        ROCBLAS_CHECK(rocblas_internal_set_data_ptr(handle, handle_ptr));
    }
    else
    {
        if(handle_data->checksum != sizeof(rocsolver_handle_data_))
            return rocblas_status_internal_error;
    }

    switch(func)
    {
    case rocsolver_function_gesvd:
    case rocsolver_function_bdsqr:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->bdsqr_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_sterf:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->sterf_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_steqr:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->steqr_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_syev_heev:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->sterf_mode = mode;
            handle_data->steqr_mode = mode;
            return rocblas_status_success;
        }
        break;
    case rocsolver_function_hetrd:
        if(mode == rocsolver_alg_mode_1stage || mode == rocsolver_alg_mode_2stage
           || mode == rocsolver_alg_mode_auto)
        {
            handle_data->hetrd_mode = mode;
            return rocblas_status_success;
        }
        break;
    }

    return rocblas_status_invalid_value;
}
catch(...)
{
    return exception2rocblas_status();
}

//------------------------------------------------------------------------------
rocblas_status rocsolver_get_alg_mode_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           rocsolver_alg_mode* mode)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data && handle_data->checksum != sizeof(rocsolver_handle_data_))
        return rocblas_status_internal_error;

    // Get mode from handle_data, or default if handle_data not yet initialized.
    switch(func)
    {
    case rocsolver_function_gesvd:
    case rocsolver_function_bdsqr:
        *mode = handle_data ? handle_data->bdsqr_mode : rocsolver_alg_mode_gpu;
        break;
    case rocsolver_function_sterf:
        *mode = handle_data ? handle_data->sterf_mode : rocsolver_alg_mode_gpu;
        break;
    case rocsolver_function_steqr:
        *mode = handle_data ? handle_data->steqr_mode : rocsolver_alg_mode_gpu;
        break;
    case rocsolver_function_syev_heev:
        if(!handle_data)
            *mode = rocsolver_alg_mode_gpu;
        else if(handle_data->sterf_mode == handle_data->steqr_mode)
            *mode = handle_data->sterf_mode;
        else
            *mode = rocsolver_alg_mode_mixed;
        break;
    case rocsolver_function_hetrd:
        *mode = handle_data ? handle_data->hetrd_mode : rocsolver_alg_mode_1stage;
        break;
    default: return rocblas_status_invalid_value;
    }

    return rocblas_status_success;
}
catch(...)
{
    return exception2rocblas_status();
}

//------------------------------------------------------------------------------
static int64_t default_opt_i64(rocsolver_function /*func*/, rocsolver_option opt)
{
    switch(opt)
    {
    case rocsolver_option_nb:                return 64;
    case rocsolver_option_kd:               return 32;
    case rocsolver_option_switch_size:      return 128;
    case rocsolver_option_2stage_switch_size: return 8000;
    case rocsolver_option_dc_leaf_size:     return 16;
    default: return 0;
    }
}

static double default_opt_fp64(rocsolver_function /*func*/, rocsolver_option opt)
{
    switch(opt)
    {
    case rocsolver_option_svd_qr_ratio:        return 1.6;
    case rocsolver_option_svd_values_qr_ratio: return 1.2;
    default: return 0.0;
    }
}

//------------------------------------------------------------------------------
rocblas_status rocsolver_set_opt_impl(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_option opt,
                                      const int64_t value)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    switch(opt)
    {
    case rocsolver_option_nb:
    case rocsolver_option_kd:
    case rocsolver_option_switch_size:
    case rocsolver_option_2stage_switch_size:
    case rocsolver_option_dc_leaf_size: break;
    default: return rocblas_status_invalid_value;
    }

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data == nullptr)
    {
        handle_ptr  = std::make_shared<rocsolver_handle_data_>();
        handle_data = (rocsolver_handle_data)handle_ptr.get();
        handle_data->checksum = sizeof(rocsolver_handle_data_);
        ROCBLAS_CHECK(rocblas_internal_set_data_ptr(handle, handle_ptr));
    }
    else if(handle_data->checksum != sizeof(rocsolver_handle_data_))
        return rocblas_status_internal_error;

    handle_data->opts_i64[{func, opt}] = value;
    return rocblas_status_success;
}
catch(...)
{
    return exception2rocblas_status();
}

//------------------------------------------------------------------------------
rocblas_status rocsolver_get_opt_impl(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_option opt,
                                      int64_t* value)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    switch(opt)
    {
    case rocsolver_option_nb:
    case rocsolver_option_kd:
    case rocsolver_option_switch_size:
    case rocsolver_option_2stage_switch_size:
    case rocsolver_option_dc_leaf_size: break;
    default: return rocblas_status_invalid_value;
    }

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data && handle_data->checksum != sizeof(rocsolver_handle_data_))
        return rocblas_status_internal_error;

    if(handle_data)
    {
        auto it = handle_data->opts_i64.find({func, opt});
        if(it != handle_data->opts_i64.end())
        {
            *value = it->second;
            return rocblas_status_success;
        }
    }
    *value = default_opt_i64(func, opt);
    return rocblas_status_success;
}
catch(...)
{
    return exception2rocblas_status();
}

//------------------------------------------------------------------------------
rocblas_status rocsolver_set_opt_fp64_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           const rocsolver_option opt,
                                           const double value)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    switch(opt)
    {
    case rocsolver_option_svd_qr_ratio:
    case rocsolver_option_svd_values_qr_ratio: break;
    default: return rocblas_status_invalid_value;
    }

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data == nullptr)
    {
        handle_ptr  = std::make_shared<rocsolver_handle_data_>();
        handle_data = (rocsolver_handle_data)handle_ptr.get();
        handle_data->checksum = sizeof(rocsolver_handle_data_);
        ROCBLAS_CHECK(rocblas_internal_set_data_ptr(handle, handle_ptr));
    }
    else if(handle_data->checksum != sizeof(rocsolver_handle_data_))
        return rocblas_status_internal_error;

    handle_data->opts_fp64[{func, opt}] = value;
    return rocblas_status_success;
}
catch(...)
{
    return exception2rocblas_status();
}

//------------------------------------------------------------------------------
rocblas_status rocsolver_get_opt_fp64_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           const rocsolver_option opt,
                                           double* value)
try
{
    if(!handle)
        return rocblas_status_invalid_handle;

    switch(opt)
    {
    case rocsolver_option_svd_qr_ratio:
    case rocsolver_option_svd_values_qr_ratio: break;
    default: return rocblas_status_invalid_value;
    }

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data && handle_data->checksum != sizeof(rocsolver_handle_data_))
        return rocblas_status_internal_error;

    if(handle_data)
    {
        auto it = handle_data->opts_fp64.find({func, opt});
        if(it != handle_data->opts_fp64.end())
        {
            *value = it->second;
            return rocblas_status_success;
        }
    }
    *value = default_opt_fp64(func, opt);
    return rocblas_status_success;
}
catch(...)
{
    return exception2rocblas_status();
}

ROCSOLVER_END_NAMESPACE

extern "C" {

rocblas_status rocsolver_set_alg_mode(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_alg_mode mode)
{
    return rocsolver::rocsolver_set_alg_mode_impl(handle, func, mode);
}

rocblas_status rocsolver_get_alg_mode(rocblas_handle handle,
                                      const rocsolver_function func,
                                      rocsolver_alg_mode* mode)
{
    return rocsolver::rocsolver_get_alg_mode_impl(handle, func, mode);
}

rocblas_status rocsolver_set_opt(rocblas_handle handle,
                                 const rocsolver_function func,
                                 const rocsolver_option opt,
                                 const int64_t value)
{
    return rocsolver::rocsolver_set_opt_impl(handle, func, opt, value);
}

rocblas_status rocsolver_get_opt(rocblas_handle handle,
                                 const rocsolver_function func,
                                 const rocsolver_option opt,
                                 int64_t* value)
{
    return rocsolver::rocsolver_get_opt_impl(handle, func, opt, value);
}

rocblas_status rocsolver_set_opt_fp64(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_option opt,
                                      const double value)
{
    return rocsolver::rocsolver_set_opt_fp64_impl(handle, func, opt, value);
}

rocblas_status rocsolver_get_opt_fp64(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_option opt,
                                      double* value)
{
    return rocsolver::rocsolver_get_opt_fp64_impl(handle, func, opt, value);
}

} // extern C
