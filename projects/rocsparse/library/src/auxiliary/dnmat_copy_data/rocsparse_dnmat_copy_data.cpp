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

#include "rocsparse_utility.hpp"

#include "rocsparse-types.h"
#include "rocsparse-version.h"
#include "rocsparse/rocsparse-export.h"

namespace rocsparse
{

    typedef rocsparse_status (*dnmat_copy_data_kernel_launch_t)(rocsparse_handle            handle,
                                                                rocsparse_const_dnvec_descr alpha,
                                                                rocsparse_const_dnmat_descr source,
                                                                rocsparse_dnmat_descr       target);

    static dnmat_copy_data_kernel_launch_t
        dnmat_copy_data_find_kernel_launch(rocsparse_indextype I_indextype,
                                           rocsparse_datatype  T_datatype);

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dnmat_copy_data_device(I               nrows,
                                                     I               ncols,
                                                     const T*        alpha,
                                                     rocsparse_order source_order,
                                                     const T* __restrict__ source,
                                                     int64_t         source_ld,
                                                     rocsparse_order target_order,
                                                     T* __restrict__ target,
                                                     int64_t target_ld)
    {

        const auto tid = hipThreadIdx_x;
        const auto gid = BLOCKSIZE * hipBlockIdx_x + tid;
        if(gid < static_cast<int64_t>(nrows) * ncols)
        {
            const auto i  = gid % nrows;
            const auto j  = gid / nrows;
            const T    s  = source[(source_order == rocsparse_order_row) ? (i * source_ld + j)
                                                                         : (i + source_ld * j)];
            const T    ss = (alpha) ? alpha[0] * s : s;
            if(source_order == target_order)
                target[(source_order == rocsparse_order_row) ? (i * target_ld + j)
                                                             : (i + target_ld * j)]
                    = ss;
            else
                target[(source_order == rocsparse_order_row) ? (j * target_ld + i)
                                                             : (j + target_ld * i)]
                    = ss;
        }
    }

    template <uint32_t BLOCKSIZE, typename I, typename T>
    ROCSPARSE_KERNEL(BLOCKSIZE)
    void dnmat_copy_data_kernel(I batch_count,
                                I nrows,
                                I ncols,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                int64_t         alpha_stride,
                                rocsparse_order source_order,
                                const T*        source,
                                int64_t         source_ld,
                                int64_t         source_stride,
                                rocsparse_order target_order,
                                T*              target,
                                int64_t         target_ld,
                                int64_t         target_stride,
                                bool            is_host)
    {
        const T* alpha = {};
        I        batch_index;
        for(batch_index = hipBlockIdx_y; batch_index < batch_count; batch_index += hipGridDim_y)
        {
            if(batch_index < batch_count)
            {
                alpha = (is_host) ? &alpha_union.value
                                  : (alpha_union.pointer
                                         ? (alpha_union.pointer + batch_index * alpha_stride)
                                         : nullptr);

                rocsparse::dnmat_copy_data_device<BLOCKSIZE, I, T>(
                    nrows,
                    ncols,
                    alpha,
                    source_order,
                    source + batch_index * source_stride,
                    source_ld,
                    target_order,
                    target + batch_index * target_stride,
                    target_ld);
            }
        }
    }

    template <typename I, typename T>
    static rocsparse_status dnmat_copy_data_kernel_launch(rocsparse_handle            handle,
                                                          rocsparse_const_dnvec_descr alpha,
                                                          rocsparse_const_dnmat_descr source,
                                                          rocsparse_dnmat_descr       target)
    {
        const T* alpha_const_values
            = (alpha) ? reinterpret_cast<const T*>(alpha->const_values) : nullptr;
        const int64_t alpha_stride = (alpha) ? alpha->batch_stride : 0;
        const auto    alpha_mode   = (alpha) ? alpha->pointer_mode : rocsparse_pointer_mode_device;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::dnmat_copy_data_kernel<512, I, T>),
            dim3(((source->rows * source->cols - 1) / 512 + 1),
                 std::min(target->batch_count, static_cast<int64_t>(65535))),
            dim3(512),
            0,
            handle->stream,

            static_cast<I>(target->batch_count),

            source->rows,
            source->cols,

            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(alpha_mode, alpha_const_values),
            alpha_stride,

            source->order,
            reinterpret_cast<const T*>(source->const_values),
            source->ld,
            source->batch_stride,

            target->order,
            reinterpret_cast<T*>(target->values),
            target->ld,
            target->batch_stride,
            (alpha_mode == rocsparse_pointer_mode_host));
        return rocsparse_status_success;
    }

    template <typename I>
    static dnmat_copy_data_kernel_launch_t find_T(rocsparse_datatype T_datatype)
    {
        switch(T_datatype)
        {
            // LCOV_EXCL_START
        case rocsparse_datatype_f16_r:
            return dnmat_copy_data_kernel_launch<I, _Float16>;
        case rocsparse_datatype_i32_r:
            return dnmat_copy_data_kernel_launch<I, int32_t>;
        case rocsparse_datatype_u32_r:
            return dnmat_copy_data_kernel_launch<I, uint32_t>;
        case rocsparse_datatype_i8_r:
            return dnmat_copy_data_kernel_launch<I, int8_t>;
        case rocsparse_datatype_u8_r:
            return dnmat_copy_data_kernel_launch<I, uint8_t>;
        case rocsparse_datatype_bf16_r:
            return dnmat_copy_data_kernel_launch<I, rocsparse_bfloat16>;
            // LCOV_EXCL_STOP
        case rocsparse_datatype_f32_r:
            return dnmat_copy_data_kernel_launch<I, float>;
        case rocsparse_datatype_f64_r:
            return dnmat_copy_data_kernel_launch<I, double>;
        case rocsparse_datatype_f32_c:
            return dnmat_copy_data_kernel_launch<I, rocsparse_float_complex>;
        case rocsparse_datatype_f64_c:
            return dnmat_copy_data_kernel_launch<I, rocsparse_double_complex>;
        }
        return nullptr;
    }

    static dnmat_copy_data_kernel_launch_t
        dnmat_copy_data_find_kernel_launch(rocsparse_indextype I_indextype,
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

rocsparse_status rocsparse::dnmat_copy_data(rocsparse_handle            handle,
                                            rocsparse_const_dnvec_descr alpha,
                                            rocsparse_const_dnmat_descr source,
                                            rocsparse_dnmat_descr       target,
                                            rocsparse_error*            p_error)
{

    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((target->data_type == source->data_type)
                                               ? rocsparse_status_success
                                               : rocsparse_status_not_implemented,
                                           "mixed precision not implemented");

    RETURN_IF_ROCSPARSE_ERROR(((source->rows == target->rows)) ? rocsparse_status_success
                                                               : rocsparse_status_invalid_value);

    RETURN_IF_ROCSPARSE_ERROR(((source->cols == target->cols)) ? rocsparse_status_success
                                                               : rocsparse_status_invalid_value);

    RETURN_IF_ROCSPARSE_ERROR(((source->batch_count == target->batch_count)
                               || ((source->batch_count == 1) && (source->batch_stride == 0)))
                                  ? rocsparse_status_success
                                  : rocsparse_status_invalid_value);

    if(source->order != target->order)
    {
        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::dnmat_switch_order(handle, alpha, source, target, p_error));
        return rocsparse_status_success;
    }

    //
    // Must be same data.
    //

    if(alpha)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((source->data_type == alpha->data_type)
                                                   ? rocsparse_status_success
                                                   : rocsparse_status_not_implemented,
                                               "mixed precision not implemented");
        RETURN_IF_ROCSPARSE_ERROR(((alpha->batch_count == target->batch_count)
                                   || ((alpha->batch_count == 1) && (alpha->batch_stride == 0)))
                                      ? rocsparse_status_success
                                      : rocsparse_status_invalid_value);
    }

    if(alpha == nullptr)
    {
        const int64_t M = (target->order == rocsparse_order_column) ? target->rows : target->cols;

        const int64_t N = (target->order == rocsparse_order_column) ? target->cols : target->rows;

        const int64_t MxN    = M * N;
        const auto    sizelm = rocsparse::datatype_sizeof(target->data_type);
        if(target->batch_count == 1)
        {
            if((source->ld == M) && (target->ld == M))
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpyAsync(target->values,
                                                             source->const_values,
                                                             sizelm * MxN,
                                                             hipMemcpyDeviceToDevice,
                                                             handle->stream));
            }
            else
            {
                RETURN_IF_HIP_ERROR(rocsparse_hipMemcpy2DAsync(target->values,
                                                               sizelm * target->ld,
                                                               source->const_values,
                                                               sizelm * source->ld,
                                                               sizelm * M,
                                                               N,
                                                               hipMemcpyDeviceToDevice,
                                                               handle->stream));
            }
            return rocsparse_status_success;
        }
        else if((target->batch_count == source->batch_count) && (source->batch_stride != 0))
        {
            const auto    batch_count    = target->batch_count;
            const int64_t Mx_batch_count = M * batch_count;

            const rocsparse_order target_layout
                = ((target->ld == M) && (target->batch_stride >= MxN)) ? rocsparse_order_row
                  : ((target->ld >= Mx_batch_count) && (target->batch_stride == M))
                      ? rocsparse_order_column
                      : ((rocsparse_order)-1);

            const rocsparse_order source_layout
                = ((source->ld == M) && (source->batch_stride >= MxN)) ? rocsparse_order_row
                  : ((source->ld >= Mx_batch_count) && (source->batch_stride == M))
                      ? rocsparse_order_column
                  : ((source->batch_stride == 0) && (source->ld == M)) ? target_layout
                                                                       : ((rocsparse_order)-1);

            if((source_layout == target_layout) && (target_layout != ((rocsparse_order)-1)))
            {
                const auto   layout        = target_layout;
                const bool   is_row_order  = (rocsparse_order_row == layout);
                const size_t sequence_size = (is_row_order) ? MxN : Mx_batch_count;

                const size_t nsequences = (is_row_order) ? batch_count : N;

                const int64_t source_ld = (is_row_order) ? source->batch_stride : source->ld;

                const int64_t target_ld = (is_row_order) ? target->batch_stride : target->ld;

                if((target_ld == sequence_size) && (source_ld == sequence_size))
                {
                    RETURN_IF_HIP_ERROR(
                        rocsparse_hipMemcpyAsync(target->values,
                                                 source->const_values,
                                                 sizelm * sequence_size * nsequences,
                                                 hipMemcpyDeviceToDevice,
                                                 handle->stream));
                }
                else
                {
                    RETURN_IF_HIP_ERROR(rocsparse_hipMemcpy2DAsync(target->values,
                                                                   sizelm * target_ld,
                                                                   source->const_values,
                                                                   sizelm * source_ld,
                                                                   sizelm * sequence_size,
                                                                   nsequences,
                                                                   hipMemcpyDeviceToDevice,
                                                                   handle->stream));
                }
                return rocsparse_status_success;
            }
        }
    }

    const rocsparse_indextype I_indextype
        = (((source->rows * source->cols) <= std::numeric_limits<int32_t>::max())
           && (target->batch_count <= std::numeric_limits<int32_t>::max()))
              ? rocsparse_indextype_i32
              : rocsparse_indextype_i64;

    auto f = rocsparse::dnmat_copy_data_find_kernel_launch(I_indextype, source->data_type);
    if(f == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, "find failed");
    }

    RETURN_IF_ROCSPARSE_ERROR(f(handle, alpha, source, target));

    return rocsparse_status_success;
}

rocsparse_status rocsparse_dnmat_copy_data(rocsparse_handle            handle,
                                           rocsparse_const_dnvec_descr alpha,
                                           rocsparse_const_dnmat_descr source,
                                           rocsparse_dnmat_descr       target,
                                           rocsparse_error*            p_error)
try
{
    ROCSPARSE_ROUTINE_TRACE;
    ROCSPARSE_CHECKARG_HANDLE(0, handle);
    ROCSPARSE_CHECKARG_POINTER(2, source);
    ROCSPARSE_CHECKARG_POINTER(3, target);

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnmat_copy_data(handle, alpha, source, target, p_error));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
