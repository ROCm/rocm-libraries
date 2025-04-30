/* ************************************************************************
 * Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#include "rocblas_rotg_kernels.hpp"
#include "check_numerics_vector.hpp"
#include "handle.hpp"
#include "logging.hpp"

/*
template <typename T, typename U>
ROCBLAS_KERNEL_NO_BOUNDS
    rocblas_rotg_check_numerics_vector_kernel(T                         a_in,
                                              rocblas_stride            offset_a,
                                              rocblas_stride            stride_a,
                                              T                         b_in,
                                              rocblas_stride            offset_b,
                                              rocblas_stride            stride_b,
                                              U                         c_in,
                                              rocblas_stride            offset_c,
                                              rocblas_stride            stride_c,
                                              T                         s_in,
                                              rocblas_stride            offset_s,
                                              rocblas_stride            stride_s,
                                              rocblas_check_numerics_t* abnormal)
{
    auto a = load_ptr_batch(a_in, blockIdx.x, offset_a, stride_a);
    auto b = load_ptr_batch(b_in, blockIdx.x, offset_b, stride_b);
    auto c = load_ptr_batch(c_in, blockIdx.x, offset_c, stride_c);
    auto s = load_ptr_batch(s_in, blockIdx.x, offset_s, stride_s);

    //Check every element of the vectors a, b, c, s for a zero/NaN/Inf/denormal value
    if(rocblas_iszero(*a) || rocblas_iszero(*b) || rocblas_iszero(*c) || rocblas_iszero(*s))
        abnormal->has_zero = true;
    if(rocblas_isnan(*a) || rocblas_isnan(*b) || rocblas_isnan(*c) || rocblas_isnan(*s))
        abnormal->has_NaN = true;
    if(rocblas_isinf(*a) || rocblas_isinf(*b) || rocblas_isinf(*c) || rocblas_isinf(*s))
        abnormal->has_Inf = true;
    if(rocblas_isdenorm(*a) || rocblas_isdenorm(*b) || rocblas_isdenorm(*c) || rocblas_isdenorm(*s))
        abnormal->has_denorm = true;
}*/

template <typename T, typename U>
rocblas_status rocblas_rotg_check_numerics_template(const char*    function_name,
                                                    rocblas_handle handle,
                                                    T              a_in,
                                                    rocblas_stride offset_a,
                                                    rocblas_stride stride_a,
                                                    T              b_in,
                                                    rocblas_stride offset_b,
                                                    rocblas_stride stride_b,
                                                    U              c_in,
                                                    rocblas_stride offset_c,
                                                    rocblas_stride stride_c,
                                                    T              s_in,
                                                    rocblas_stride offset_s,
                                                    rocblas_stride stride_s,
                                                    int64_t        batch_count,
                                                    const int      check_numerics,
                                                    bool           is_input)
{
    if(!batch_count)
        return rocblas_status_success;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        rocblas_status check_numerics_status
            = rocblas_internal_check_numerics_vector_template(function_name,
                                                              handle,
                                                              1,
                                                              a_in,
                                                              offset_a,
                                                              1,
                                                              stride_a,
                                                              batch_count,
                                                              check_numerics,
                                                              is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                1,
                                                                                b_in,
                                                                                offset_b,
                                                                                1,
                                                                                stride_b,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                1,
                                                                                c_in,
                                                                                offset_c,
                                                                                1,
                                                                                stride_c,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);
        if(check_numerics_status != rocblas_status_success)
            return check_numerics_status;

        check_numerics_status = rocblas_internal_check_numerics_vector_template(function_name,
                                                                                handle,
                                                                                1,
                                                                                s_in,
                                                                                offset_s,
                                                                                1,
                                                                                stride_s,
                                                                                batch_count,
                                                                                check_numerics,
                                                                                is_input);

        return check_numerics_status;
    }
    else
    {
        //Creating structure host object
        rocblas_check_numerics_t h_abnormal;

        for(int64_t i = 0; i < batch_count; i++)
        {
            auto a = load_ptr_batch(a_in, i, offset_a, stride_a);
            auto b = load_ptr_batch(b_in, i, offset_b, stride_b);
            auto c = load_ptr_batch(c_in, i, offset_c, stride_c);
            auto s = load_ptr_batch(s_in, i, offset_s, stride_s);

            //Check every element of the x vector for a NaN/zero/Inf/denormal value
            if(rocblas_iszero(*a) || rocblas_iszero(*b) || rocblas_iszero(*c) || rocblas_iszero(*s))
                h_abnormal.has_zero = true;
            if(rocblas_isnan(*a) || rocblas_isnan(*b) || rocblas_isnan(*c) || rocblas_isnan(*s))
                h_abnormal.has_NaN = true;
            if(rocblas_isinf(*a) || rocblas_isinf(*b) || rocblas_isinf(*c) || rocblas_isinf(*s))
                h_abnormal.has_Inf = true;
            if(rocblas_isdenorm(*a) || rocblas_isdenorm(*b) || rocblas_isdenorm(*c)
               || rocblas_isdenorm(*s))
                h_abnormal.has_denorm = true;
        }

        return rocblas_check_numerics_abnormal_struct(
            function_name, check_numerics, is_input, &h_abnormal);
    }
}

#undef INST_ROTG_LAUNCHER

#ifdef INSTANTIATE_ROTG_CHECK_NUMERICS
#error INSTANTIATE_ROTG_CHECK_NUMERICS already defined
#endif

#define INSTANTIATE_ROTG_CHECK_NUMERICS(T_, U_)                           \
    template rocblas_status rocblas_rotg_check_numerics_template<T_, U_>( \
        const char*    function_name,                                     \
        rocblas_handle handle,                                            \
        T_             a_in,                                              \
        rocblas_stride offset_a,                                          \
        rocblas_stride stride_a,                                          \
        T_             b_in,                                              \
        rocblas_stride offset_b,                                          \
        rocblas_stride stride_b,                                          \
        U_             c_in,                                              \
        rocblas_stride offset_c,                                          \
        rocblas_stride stride_c,                                          \
        T_             s_in,                                              \
        rocblas_stride offset_s,                                          \
        rocblas_stride stride_s,                                          \
        int64_t        batch_count,                                       \
        const int      check_numerics,                                    \
        bool           is_input);

// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INSTANTIATE_ROTG_CHECK_NUMERICS(float*, float*)
INSTANTIATE_ROTG_CHECK_NUMERICS(double*, double*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_float_complex*, float*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_double_complex*, double*)

// instantiate for rocblas_Xrotg_batched
INSTANTIATE_ROTG_CHECK_NUMERICS(float* const*, float* const*)
INSTANTIATE_ROTG_CHECK_NUMERICS(double* const*, double* const*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_float_complex* const*, float* const*)
INSTANTIATE_ROTG_CHECK_NUMERICS(rocblas_double_complex* const*, double* const*)

#undef INSTANTIATE_ROTG_CHECK_NUMERICS

//
// kernels and launcher

template <typename T, typename U, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
__forceinline__ __device__ __host__ void rocblas_rotg_calc(T& a, T& b, U& c, T& s)
{
    T scale = rocblas_abs(a) + rocblas_abs(b);
    if(scale == 0.0)
    {
        c = 1.0;
        s = 0.0;
        a = 0.0;
        b = 0.0;
    }
    else
    {
        T sa  = a / scale;
        T sb  = b / scale;
        T r   = scale * sqrt(sa * sa + sb * sb);
        T roe = rocblas_abs(a) > rocblas_abs(b) ? a : b;
        r     = copysign(r, roe);
        c     = a / r;
        s     = b / r;
        T z   = 1.0;
        if(rocblas_abs(a) > rocblas_abs(b))
            z = s;
        if(rocblas_abs(b) >= rocblas_abs(a) && c != 0.0)
            z = 1.0 / c;
        a = r;
        b = z;
    }
}

template <typename T, typename U, std::enable_if_t<rocblas_is_complex<T>, int> = 0>
__forceinline__ __device__ __host__ void rocblas_rotg_calc(T& a, T& b, U& c, T& s)
{
    if(rocblas_abs(a) != 0.)
    {
        auto scale = rocblas_abs(a) + rocblas_abs(b);
        auto sa    = rocblas_abs(a / scale);
        auto sb    = rocblas_abs(b / scale);
        auto norm  = scale * sqrt(sa * sa + sb * sb);
        auto alpha = a / rocblas_abs(a);
        c          = rocblas_abs(a) / norm;
        s          = alpha * conj(b) / norm;
        a          = alpha * norm;
    }
    else
    {
        c = 0;
        s = {1, 0};
        a = b;
    }
}

template <rocblas_int NB, typename T, typename U>
ROCBLAS_KERNEL(NB)
rocblas_rotg_kernel(T              a_in,
                    rocblas_stride offset_a,
                    rocblas_stride stride_a,
                    T              b_in,
                    rocblas_stride offset_b,
                    rocblas_stride stride_b,
                    U              c_in,
                    rocblas_stride offset_c,
                    rocblas_stride stride_c,
                    T              s_in,
                    rocblas_stride offset_s,
                    rocblas_stride stride_s,
                    int32_t        batch_count)
{
    int idx = blockIdx.x * NB + threadIdx.x;
    if(idx >= batch_count)
        return;

    auto a = load_ptr_batch(a_in, idx, offset_a, stride_a);
    auto b = load_ptr_batch(b_in, idx, offset_b, stride_b);
    auto c = load_ptr_batch(c_in, idx, offset_c, stride_c);
    auto s = load_ptr_batch(s_in, idx, offset_s, stride_s);
    rocblas_rotg_calc(*a, *b, *c, *s);
}

template <typename API_INT, typename T, typename U>
rocblas_status rocblas_internal_rotg_launcher(rocblas_handle handle,
                                              T              a_in,
                                              rocblas_stride offset_a,
                                              rocblas_stride stride_a,
                                              T              b_in,
                                              rocblas_stride offset_b,
                                              rocblas_stride stride_b,
                                              U              c_in,
                                              rocblas_stride offset_c,
                                              rocblas_stride stride_c,
                                              T              s_in,
                                              rocblas_stride offset_s,
                                              rocblas_stride stride_s,
                                              API_INT        batch_count)
{
    if(batch_count <= 0)
        return rocblas_status_success;

    if(rocblas_pointer_mode_device == handle->pointer_mode)
    {
        static constexpr int NB = 32; // non batched use so keep small
        dim3                 blocks((batch_count - 1) / NB + 1);
        dim3                 threads(NB);

        ROCBLAS_LAUNCH_KERNEL(rocblas_rotg_kernel<NB>,
                              blocks,
                              threads,
                              0,
                              handle->get_stream(),
                              a_in,
                              offset_a,
                              stride_a,
                              b_in,
                              offset_b,
                              stride_b,
                              c_in,
                              offset_c,
                              stride_c,
                              s_in,
                              offset_s,
                              stride_s,
                              (int32_t)batch_count);
    }
    else
    {
        // TODO: make this faster for a large number of batches.
        for(int i = 0; i < batch_count; i++)
        {
            auto a = load_ptr_batch(a_in, i, offset_a, stride_a);
            auto b = load_ptr_batch(b_in, i, offset_b, stride_b);
            auto c = load_ptr_batch(c_in, i, offset_c, stride_c);
            auto s = load_ptr_batch(s_in, i, offset_s, stride_s);

            rocblas_rotg_calc(*a, *b, *c, *s);
        }
    }

    return rocblas_status_success;
}

// If there are any changes in template parameters in the files *rotg*.cpp
// instantiations below will need to be manually updated to match the changes.

#ifdef INST_ROTG_LAUNCHER
#error INST_ROTG_LAUNCHER already defined
#endif

#define INST_ROTG_LAUNCHER(API_INT_, T_, U_)                                  \
    template rocblas_status rocblas_internal_rotg_launcher<API_INT_, T_, U_>( \
        rocblas_handle handle,                                                \
        T_             a_in,                                                  \
        rocblas_stride offset_a,                                              \
        rocblas_stride stride_a,                                              \
        T_             b_in,                                                  \
        rocblas_stride offset_b,                                              \
        rocblas_stride stride_b,                                              \
        U_             c_in,                                                  \
        rocblas_stride offset_c,                                              \
        rocblas_stride stride_c,                                              \
        T_             s_in,                                                  \
        rocblas_stride offset_s,                                              \
        rocblas_stride stride_s,                                              \
        API_INT_       batch_count);

// instantiate for rocblas_Xrotg and rocblas_Xrotg_strided_batched
INST_ROTG_LAUNCHER(rocblas_int, float*, float*)
INST_ROTG_LAUNCHER(rocblas_int, double*, double*)
INST_ROTG_LAUNCHER(rocblas_int, rocblas_float_complex*, float*)
INST_ROTG_LAUNCHER(rocblas_int, rocblas_double_complex*, double*)

// instantiate for rocblas_Xrotg_batched
INST_ROTG_LAUNCHER(rocblas_int, float* const*, float* const*)
INST_ROTG_LAUNCHER(rocblas_int, double* const*, double* const*)
INST_ROTG_LAUNCHER(rocblas_int, rocblas_float_complex* const*, float* const*)
INST_ROTG_LAUNCHER(rocblas_int, rocblas_double_complex* const*, double* const*)
