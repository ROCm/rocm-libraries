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

#include "rocsparse-auxiliary.h"
#include "rocsparse_utility.hpp"

#include <hip/hip_runtime.h>
#include <type_traits>

namespace rocsparse
{

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dnvec_copy_data_device(I size,
                                                     const T* __restrict__ source,
                                                     int64_t source_inc,
                                                     T* __restrict__ target,
                                                     int64_t target_inc)
    {
        const auto tid = hipThreadIdx_x;
        const auto gid = static_cast<int64_t>(BLOCKSIZE) * hipBlockIdx_x + tid;
        if(gid < size)
        {
            target[gid * target_inc] = source[gid * source_inc];
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dnvec_copy_data_kernel(I        batch_count,
                                I        size,
                                const T* source,
                                int64_t  source_inc,
                                int64_t  source_stride,
                                T*       target,
                                int64_t  target_inc,
                                int64_t  target_stride)
    {
        for(I batch_index = hipBlockIdx_y; batch_index < batch_count; batch_index += hipGridDim_y)
        {
            if(batch_index < batch_count)
            {
                rocsparse::dnvec_copy_data_device<BLOCKSIZE, I, T>(
                    size,
                    source + batch_index * source_stride,
                    source_inc,
                    target + batch_index * target_stride,
                    target_inc);
            }
        }
    }

    template <typename I, typename T>
    static rocsparse_status dnvec_copy_data_kernel_launch(rocsparse_handle            handle,
                                                          rocsparse_const_dnvec_descr source,
                                                          rocsparse_dnvec_descr       target)
    {
        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::dnvec_copy_data_kernel<1024, I, T>),
            dim3(((source->size - 1) / 1024 + 1),
                 std::min(target->batch_count, static_cast<int64_t>(65535))),
            dim3(1024),
            0,
            handle->stream,

            static_cast<I>(target->batch_count),
            source->size,

            reinterpret_cast<const T*>(source->const_values),
            source->inc,
            source->batch_stride,

            reinterpret_cast<T*>(target->values),
            target->inc,
            target->batch_stride);

        return rocsparse_status_success;
    }

    typedef rocsparse_status (*dnvec_copy_data_kernel_launch_t)(rocsparse_handle,
                                                                rocsparse_const_dnvec_descr,
                                                                rocsparse_dnvec_descr);

    template <typename I>
    static dnvec_copy_data_kernel_launch_t find_T(rocsparse_datatype T_datatype)
    {
        switch(T_datatype)
        {
            // LCOV_EXCL_START
        case rocsparse_datatype_f16_r:
            return dnvec_copy_data_kernel_launch<I, _Float16>;
        case rocsparse_datatype_i32_r:
            return dnvec_copy_data_kernel_launch<I, int32_t>;
        case rocsparse_datatype_u32_r:
            return dnvec_copy_data_kernel_launch<I, uint32_t>;
        case rocsparse_datatype_i8_r:
            return dnvec_copy_data_kernel_launch<I, int8_t>;
        case rocsparse_datatype_u8_r:
            return dnvec_copy_data_kernel_launch<I, uint8_t>;
        case rocsparse_datatype_bf16_r:
            return dnvec_copy_data_kernel_launch<I, rocsparse_bfloat16>;
            // LCOV_EXCL_STOP
        case rocsparse_datatype_f32_r:
            return dnvec_copy_data_kernel_launch<I, float>;
        case rocsparse_datatype_f64_r:
            return dnvec_copy_data_kernel_launch<I, double>;
        case rocsparse_datatype_f32_c:
            return dnvec_copy_data_kernel_launch<I, rocsparse_float_complex>;
        case rocsparse_datatype_f64_c:
            return dnvec_copy_data_kernel_launch<I, rocsparse_double_complex>;
        }
        return nullptr;
    }

    static dnvec_copy_data_kernel_launch_t
        dnvec_copy_find_kernel_launch(rocsparse_indextype I_indextype,
                                      rocsparse_datatype  T_datatype)
    {
        switch(I_indextype)
        {
        case rocsparse_indextype_i32:
            return find_T<int32_t>(T_datatype);
        case rocsparse_indextype_i64:
            return find_T<int64_t>(T_datatype);
        case deprecated_rocsparse_indextype_u16:
            return nullptr;
        }
        return nullptr;
    }

}

rocsparse_status rocsparse::dnvec_copy_data(rocsparse_handle            handle,
                                            rocsparse_const_dnvec_descr source,
                                            rocsparse_dnvec_descr       target,
                                            rocsparse_error*            p_error)
{
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
        (target->size == source->size) ? rocsparse_status_success : rocsparse_status_invalid_value,
        "size mismatch");

    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((target->data_type == source->data_type)
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_value,
                                           "datatype mismatch");

    const rocsparse_indextype I_indextype
        = ((source->size <= std::numeric_limits<int32_t>::max())
           && (target->batch_count <= std::numeric_limits<int32_t>::max()))
              ? rocsparse_indextype_i32
              : rocsparse_indextype_i64;

    const auto sizelm = rocsparse::datatype_sizeof(target->data_type);

    if(target->batch_count == 1)
    {
        if((target->inc == 1) && (source->inc == 1))
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(target->values,
                                                         source->const_values,
                                                         sizelm * target->size,
                                                         hipMemcpyDeviceToDevice,
                                                         handle->stream));
            return rocsparse_status_success;
        }
        else
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpy2DAsync(target->values,
                                                           sizelm * target->inc,
                                                           source->const_values,
                                                           sizelm * source->inc,
                                                           sizelm,
                                                           target->size,
                                                           hipMemcpyDeviceToDevice,
                                                           handle->stream));
            return rocsparse_status_success;
        }
    }
    else if((target->batch_stride >= target->size) && (source->batch_stride >= source->size))
    {
        if((target->inc == 1) && (source->inc == 1))
        {
            RETURN_IF_HIP_ERROR(rocsparse_hipMemcpy2DAsync(target->values,
                                                           sizelm * target->batch_stride,
                                                           source->const_values,
                                                           sizelm * source->batch_stride,
                                                           sizelm * source->size,
                                                           target->batch_count,
                                                           hipMemcpyDeviceToDevice,
                                                           handle->stream));
            return rocsparse_status_success;
        }
    }
    else if(((target->batch_stride == 1) && (target->inc >= target->batch_count))
            && ((source->batch_stride == 1) && (source->inc >= source->batch_count)))
    {
        RETURN_IF_HIP_ERROR(rocsparse_hipMemcpy2DAsync(target->values,
                                                       sizelm * target->inc,
                                                       source->const_values,
                                                       sizelm * source->inc,
                                                       sizelm * target->batch_count,
                                                       target->size,
                                                       hipMemcpyDeviceToDevice,
                                                       handle->stream));
        return rocsparse_status_success;
    }

    auto f = rocsparse::dnvec_copy_find_kernel_launch(I_indextype, source->data_type);
    if(f == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_invalid_value,
                                               "dense_copy find failed");
    }

    RETURN_IF_ROCSPARSE_ERROR(f(handle, source, target));
    return rocsparse_status_success;
}
