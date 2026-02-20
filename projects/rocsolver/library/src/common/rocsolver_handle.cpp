/* **************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "rocblas.hpp"

#include <memory>

ROCSOLVER_BEGIN_NAMESPACE

rocblas_status rocsolver_set_alg_mode_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           const rocsolver_alg_mode mode)
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
    case rocsolver_function_sterf:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->sterf_mode = mode;
            return rocblas_status_success;
        }
    case rocsolver_function_steqr:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->steqr_mode = mode;
            return rocblas_status_success;
        }
    case rocsolver_function_syev_heev:
        if(mode == rocsolver_alg_mode_gpu || mode == rocsolver_alg_mode_hybrid)
        {
            handle_data->sterf_mode = mode;
            handle_data->steqr_mode = mode;
            return rocblas_status_success;
        }
    }

    return rocblas_status_invalid_value;
}

rocblas_status rocsolver_get_alg_mode_impl(rocblas_handle handle,
                                           const rocsolver_function func,
                                           rocsolver_alg_mode* mode)
{
    if(!handle)
        return rocblas_status_invalid_handle;

    std::shared_ptr<void> handle_ptr;
    ROCBLAS_CHECK(rocblas_internal_get_data_ptr(handle, handle_ptr));
    rocsolver_handle_data handle_data = (rocsolver_handle_data)handle_ptr.get();

    if(handle_data == nullptr)
    {
        *mode = rocsolver_alg_mode_gpu;
    }
    else
    {
        if(handle_data->checksum != sizeof(rocsolver_handle_data_))
            return rocblas_status_internal_error;

        switch(func)
        {
        case rocsolver_function_gesvd:
        case rocsolver_function_bdsqr: *mode = handle_data->bdsqr_mode; break;
        case rocsolver_function_sterf: *mode = handle_data->sterf_mode; break;
        case rocsolver_function_steqr: *mode = handle_data->steqr_mode; break;
        case rocsolver_function_syev_heev:
            if(handle_data->sterf_mode == handle_data->steqr_mode)
                *mode = handle_data->sterf_mode;
            else
                *mode = rocsolver_alg_mode_mixed;
            break;
        default: return rocblas_status_invalid_value;
        }
    }

    return rocblas_status_success;
}

ROCSOLVER_END_NAMESPACE

extern "C" {

rocblas_status rocsolver_set_alg_mode(rocblas_handle handle,
                                      const rocsolver_function func,
                                      const rocsolver_alg_mode mode)
try
{
    return rocsolver::rocsolver_set_alg_mode_impl(handle, func, mode);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}

rocblas_status rocsolver_get_alg_mode(rocblas_handle handle,
                                      const rocsolver_function func,
                                      rocsolver_alg_mode* mode)
try
{
    return rocsolver::rocsolver_get_alg_mode_impl(handle, func, mode);
}
catch(...)
{
    return rocsolver::exception_to_rocblas_status();
}
}
