/*! \file */
/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "rocsparse_control.hpp"
#include "rocsparse_csrmv.hpp"
#include "rocsparse_logging.hpp"

namespace rocsparse
{
    template <typename T, typename Y>
    rocsparse_status csrmv_extract_gamma_and_z_arrays(rocsparse_handle             handle,
                                                      rocsparse_int                num_extra,
                                                      rocsparse_const_dnvec_descr  gamma_vec,
                                                      rocsparse_const_dnvec_descr* z_vecs,
                                                      T*        gamma_device_array,
                                                      const Y** z_array)
    {
        ROCSPARSE_ROUTINE_TRACE;

        using Z = Y;

        if(num_extra <= 0)
        {
            return rocsparse_status_success;
        }

        // Extract z pointers directly to device array using hipMemsetAsync and hipMemcpyAsync
        for(rocsparse_int i = 0; i < num_extra; ++i)
        {
            if(z_vecs != nullptr && z_vecs[i] != nullptr)
            {
                // Copy the pointer value directly to device array element
                RETURN_IF_HIP_ERROR(hipMemcpyAsync((void*)(&z_array[i]),
                                                   &(z_vecs[i]->const_values),
                                                   sizeof(const Z*),
                                                   hipMemcpyHostToDevice,
                                                   handle->stream));
            }
            else
            {
                // Set z_array[i] to nullptr using hipMemsetAsync
                RETURN_IF_HIP_ERROR(
                    hipMemsetAsync((void*)(&z_array[i]), 0, sizeof(const Z*), handle->stream));
            }
        }

        // Extract gamma values from the dnvec descriptor
        if(gamma_vec != nullptr)
        {
            // The gamma values are in the dnvec, copy them to the device array
            const T* gamma_data = reinterpret_cast<const T*>(gamma_vec->const_values);

            RETURN_IF_HIP_ERROR(hipMemcpyAsync((void*)gamma_device_array,
                                               gamma_data,
                                               num_extra * sizeof(T),
                                               handle->pointer_mode == rocsparse_pointer_mode_host
                                                   ? hipMemcpyHostToDevice
                                                   : hipMemcpyDeviceToDevice,
                                               handle->stream));
        }
        else
        {
            // If gamma_vec is null, set all gamma values to zero
            RETURN_IF_HIP_ERROR(hipMemsetAsync(
                (void*)gamma_device_array, 0, num_extra * sizeof(T), handle->stream));
        }

        return rocsparse_status_success;
    }

// Explicit instantiations for all combinations used in csrmv
#define INSTANTIATE_EXTRACT(TTYPE, YTYPE)                                  \
    template rocsparse_status rocsparse::csrmv_extract_gamma_and_z_arrays( \
        rocsparse_handle             handle,                               \
        rocsparse_int                num_extra,                            \
        rocsparse_const_dnvec_descr  gamma_vec,                            \
        rocsparse_const_dnvec_descr* z_vecs,                               \
        TTYPE*                       gamma_device_array,                   \
        const YTYPE**                z_array);

    // Uniform precision
    INSTANTIATE_EXTRACT(float, float);
    INSTANTIATE_EXTRACT(double, double);
    INSTANTIATE_EXTRACT(rocsparse_float_complex, rocsparse_float_complex);
    INSTANTIATE_EXTRACT(rocsparse_double_complex, rocsparse_double_complex);

    // Mixed precision
    INSTANTIATE_EXTRACT(int32_t, int32_t);
    INSTANTIATE_EXTRACT(_Float16, float);
    INSTANTIATE_EXTRACT(float, _Float16);
    INSTANTIATE_EXTRACT(rocsparse_bfloat16, float);
    INSTANTIATE_EXTRACT(float, rocsparse_bfloat16);
    INSTANTIATE_EXTRACT(rocsparse_bfloat16, rocsparse_bfloat16);

#undef INSTANTIATE_EXTRACT
}
