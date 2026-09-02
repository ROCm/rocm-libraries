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

rocsparse_status rocsparse_dnmat_transpose(rocsparse_handle            handle,
                                           rocsparse_const_dnvec_descr alpha,
                                           rocsparse_const_dnmat_descr X,
                                           rocsparse_dnmat_descr       Y,
                                           rocsparse_error*            p_error);

namespace rocsparse
{

    template <uint32_t BN, uint32_t BM, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dnmat_transpose_order_row_device(I        nrows,
                                                               I        ncols,
                                                               const T* alpha,
                                                               const T* __restrict__ source,
                                                               int64_t source_ld,
                                                               T* __restrict__ target,
                                                               int64_t target_ld)
    {
        const auto lid = hipThreadIdx_x & (BN - 1);
        const auto wid = hipThreadIdx_x / BN;

        const I a_s = hipBlockIdx_x * BN + wid;
        const I b_i = hipBlockIdx_x * BN + lid;

        __shared__ T sdata[BN][BN + 1];

        for(I j = 0; j < ncols; j += BN)
        {
            __syncthreads();

            const I a_j = j + lid;
            for(uint32_t k = 0; k < BN; k += BM)
            {
                const I a_i = a_s + k;
                if((a_j < ncols) && (a_i < nrows))
                {
                    sdata[wid + k][lid] = source[a_i * source_ld + a_j]; // A row_order
                }
            }

            __syncthreads();

            const I b_s = j + wid;
            for(uint32_t k = 0; k < BN; k += BM)
            {
                const I b_j = b_s + k;
                if((b_i < nrows) && (b_j < ncols))
                {
                    const auto s                  = sdata[lid][wid + k];
                    const auto value              = (alpha) ? alpha[0] * s : s;
                    target[b_j * target_ld + b_i] = value;
                }
            }
        }
    }

    // Perform dense matrix transposition
    template <uint32_t BN, uint32_t BM, typename I, typename T>
    ROCSPARSE_DEVICE_ILF void dnmat_transpose_order_column_device(I        nrows,
                                                                  I        ncols,
                                                                  const T* alpha,
                                                                  const T* __restrict__ source,
                                                                  int64_t source_ld,
                                                                  T* __restrict__ target,
                                                                  int64_t target_ld)
    {
        const uint32_t lid = threadIdx.x & (BN - 1);
        const uint32_t wid = threadIdx.x / BN;

        const I a_i = blockIdx.x * BN + lid;
        const I b_s = blockIdx.x * BN + wid;

        __shared__ T sdata[BN][BN + 1];

        for(I j = 0; j < ncols; j += BN)
        {
            __syncthreads();

            const auto a_s = j + wid;
            for(uint32_t k = 0; k < BN; k += BM)
            {
                const auto a_j = a_s + k;
                if(a_i < nrows && a_j < ncols)
                {
                    sdata[wid + k][lid] = source[a_j * source_ld + a_i]; // A col order
                }
            }

            __syncthreads();

            const auto b_j = j + lid;
            for(uint32_t k = 0; k < BN; k += BM)
            {
                const auto b_i = b_s + k;
                if(b_j < ncols && b_i < nrows)
                {
                    const auto s                  = sdata[lid][wid + k];
                    const auto value              = (alpha) ? alpha[0] * s : s;
                    target[b_i * target_ld + b_j] = value;
                }
            }
        }
    }

    template <uint32_t DIMC, uint32_t DIMR, typename I, typename T>
    ROCSPARSE_KERNEL(DIMC* DIMR)
    void dense_transpose_kernel(I batch_count,
                                I nrows,
                                I ncols,
                                ROCSPARSE_DEVICE_HOST_SCALAR_PARAMS(T, alpha),
                                int64_t         alpha_stride,
                                rocsparse_order A_order,
                                const T*        A,
                                int64_t         lda,
                                int64_t         A_stride,
                                T*              B,
                                int64_t         ldb,
                                int64_t         B_stride,
                                bool            is_host)
    {
        switch(A_order)
        {
        case rocsparse_order_column:
        {

            for(I batch_index = hipBlockIdx_y; batch_index < batch_count;
                batch_index += hipGridDim_y)
            {

                if(batch_index < batch_count)
                {
                    const T* alpha = (is_host)
                                         ? &alpha_union.value
                                         : ((alpha_union.pointer)
                                                ? (alpha_union.pointer + batch_index * alpha_stride)
                                                : nullptr);

                    alpha = (alpha) ? ((*alpha != static_cast<T>(1)) ? alpha : nullptr) : nullptr;

                    rocsparse::dnmat_transpose_order_column_device<DIMC, DIMR, I, T>(
                        nrows,
                        ncols,
                        alpha,
                        A + batch_index * A_stride,
                        lda,
                        B + batch_index * B_stride,
                        ldb);
                }
            }
            break;
        }

        case rocsparse_order_row:
        {
            for(I batch_index = hipBlockIdx_y; batch_index < batch_count;
                batch_index += hipGridDim_y)
            {
                if(batch_index < batch_count)
                {
                    const T* alpha = (is_host)
                                         ? &alpha_union.value
                                         : ((alpha_union.pointer)
                                                ? (alpha_union.pointer + batch_index * alpha_stride)
                                                : nullptr);
                    alpha = (alpha) ? ((*alpha != static_cast<T>(1)) ? alpha : nullptr) : nullptr;
                    rocsparse::dnmat_transpose_order_row_device<DIMC, DIMR, I, T>(
                        nrows,
                        ncols,
                        alpha,
                        A + batch_index * A_stride,
                        lda,
                        B + batch_index * B_stride,
                        ldb);
                }
            }
            break;
        }
        }
    }

    template <typename I, typename T>
    static rocsparse_status launch(rocsparse_handle            handle,
                                   rocsparse_const_dnvec_descr alpha,
                                   rocsparse_const_dnmat_descr source,
                                   rocsparse_dnmat_descr       target)
    {

        static constexpr uint32_t BN = 32;
        static constexpr uint32_t BM = 8;

        dim3 gdim((source->rows - 1) / BN + 1,
                  std::min(target->batch_count, static_cast<int64_t>(65535)));
        dim3 tdim(BN * BM);

        const T* alpha_const_values
            = (alpha) ? reinterpret_cast<const T*>(alpha->const_values) : nullptr;
        const int64_t alpha_stride = (alpha) ? alpha->batch_stride : 0;
        const auto    alpha_mode   = (alpha) ? alpha->pointer_mode : rocsparse_pointer_mode_device;

        RETURN_IF_HIPLAUNCHKERNELGGL_ERROR(
            (rocsparse::dense_transpose_kernel<BN, BM, I, T>),
            gdim,
            tdim,
            0,
            handle->stream,

            static_cast<I>(target->batch_count),
            static_cast<I>(source->rows),
            static_cast<I>(source->cols),
            ROCSPARSE_SCALAR_HOST_DEVICE_ARGUMENT(alpha_mode, alpha_const_values),
            alpha_stride,

            source->order,
            reinterpret_cast<const T*>(source->const_values),
            source->ld,
            source->batch_stride,

            // target->order,
            reinterpret_cast<T*>(target->values),
            target->ld,
            target->batch_stride,

            (alpha_mode == rocsparse_pointer_mode_host));
        return rocsparse_status_success;
    }

    typedef rocsparse_status (*launch_t)(rocsparse_handle            handle,
                                         rocsparse_const_dnvec_descr alpha,
                                         rocsparse_const_dnmat_descr source,
                                         rocsparse_dnmat_descr       target);

    template <typename I>
    static launch_t find_T(rocsparse_datatype T_datatype)
    {
        switch(T_datatype)
        {
            // LCOV_EXCL_START
        case rocsparse_datatype_f16_r:
            return launch<I, _Float16>;
        case rocsparse_datatype_i32_r:
            return launch<I, int32_t>;
        case rocsparse_datatype_u32_r:
            return launch<I, uint32_t>;
        case rocsparse_datatype_i8_r:
            return launch<I, int8_t>;
        case rocsparse_datatype_u8_r:
            return launch<I, uint8_t>;
        case rocsparse_datatype_bf16_r:
            return launch<I, rocsparse_bfloat16>;
            // LCOV_EXCL_STOP

        case rocsparse_datatype_f32_r:
            return launch<I, float>;
        case rocsparse_datatype_f64_r:
            return launch<I, double>;
        case rocsparse_datatype_f32_c:
            return launch<I, rocsparse_float_complex>;
        case rocsparse_datatype_f64_c:
            return launch<I, rocsparse_double_complex>;
        }
        return nullptr;
    }

    static launch_t find(rocsparse_indextype I_indextype, rocsparse_datatype T_datatype)
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

rocsparse_status rocsparse::dnmat_transpose(rocsparse_handle            handle,
                                            rocsparse_const_dnvec_descr alpha,
                                            rocsparse_const_dnmat_descr source,
                                            rocsparse_dnmat_descr       target,
                                            rocsparse_error*            p_error)
{

    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(
        ((target->rows == source->cols) && (target->cols == source->rows))
            ? rocsparse_status_success
            : rocsparse_status_invalid_value,
        "check inputs, dimension mismatch");
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR((target->data_type == source->data_type)
                                               ? rocsparse_status_success
                                               : rocsparse_status_not_implemented,
                                           "mixed precision not implemented");

    if(target->order != source->order)
    {
        _rocsparse_dnmat_descr shadow_target{true,
                                             target->cols,
                                             target->rows,
                                             target->ld,
                                             target->values,
                                             target->const_values,
                                             target->data_type,
                                             source->order,
                                             target->batch_count,
                                             target->batch_stride};

        RETURN_IF_ROCSPARSE_ERROR(
            rocsparse::dnmat_copy_data(handle, alpha, source, &shadow_target, p_error));
    }
    else
    {

        RETURN_IF_ROCSPARSE_ERROR(((source->batch_count == target->batch_count)
                                   || ((source->batch_count == 1) && (source->batch_stride == 0)))
                                      ? rocsparse_status_success
                                      : rocsparse_status_invalid_value);

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

        const rocsparse_indextype I_indextype
            = ((source->rows * source->cols) <= std::numeric_limits<int32_t>::max())
                  ? rocsparse_indextype_i32
                  : rocsparse_indextype_i64;
        auto f = find(I_indextype, source->data_type);
        if(f == nullptr)
        {
            RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, "find failed");
        }

        RETURN_IF_ROCSPARSE_ERROR(f(handle, alpha, source, target));
    }
    return rocsparse_status_success;
}

rocsparse_status rocsparse::dnmat_switch_order(rocsparse_handle            handle,
                                               rocsparse_const_dnvec_descr alpha,
                                               rocsparse_const_dnmat_descr source,
                                               rocsparse_dnmat_descr       target,
                                               rocsparse_error*            p_error)
{
    RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(((target->order != source->order))
                                               ? rocsparse_status_success
                                               : rocsparse_status_invalid_value,
                                           "check inputs, order mismatch");

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

    const rocsparse_indextype I_indextype = (source->rows <= std::numeric_limits<int32_t>::max()
                                             && source->cols <= std::numeric_limits<int32_t>::max())
                                                ? rocsparse_indextype_i32
                                                : rocsparse_indextype_i64;
    auto                      f           = find(I_indextype, source->data_type);
    if(f == nullptr)
    {
        RETURN_WITH_MESSAGE_IF_ROCSPARSE_ERROR(rocsparse_status_internal_error, "find failed");
    }

    RETURN_IF_ROCSPARSE_ERROR(f(handle, alpha, source, target));
    return rocsparse_status_success;
}

rocsparse_status rocsparse_dnmat_transpose(rocsparse_handle            handle,
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

    RETURN_IF_ROCSPARSE_ERROR(rocsparse::dnmat_transpose(handle, alpha, source, target, p_error));

    return rocsparse_status_success;
    // LCOV_EXCL_START
}
catch(...)
{
    RETURN_ROCSPARSE_EXCEPTION();
}
// LCOV_EXCL_STOP
