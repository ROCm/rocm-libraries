/* ************************************************************************
 * Copyright (C) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rocblas_matrix.hpp"
#include "int64_helpers.hpp"
#include "rocblas_math.hpp"

__device__ inline uint64_t
    pseudo_random(size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, size_t offset = 0)
{
    uint64_t s = (i + j * M + b * M * N) * 6364136223846793005ULL + 1442695040888963407ULL;
    // Run a few extra iterations to make the generators diverge
    // in case the seeds are still poor (consecutive ints)
    for(int i = 0; i < 2 + offset; i++)
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
    }
    return s;
}

template <typename T>
struct trig_functor
{
    __device__ T
        operator()(size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool seedReset)
    {
        size_t idx = i + j * M + b * M * N;
        return T(seedReset ? cos(idx) : sin(idx));
    }
};

template <typename T>
struct identity_functor
{
    __device__ T operator()(size_t i, size_t j, size_t, rocblas_int, rocblas_int, bool)
    {
        return i == j ? T(1) : T(0);
    }
};

template <typename T>
struct zero_functor
{
    __device__ T operator()(size_t, size_t, size_t, rocblas_int, rocblas_int, bool)
    {
        return T(0);
    }
};

template <typename T>
struct hpl_functor
{
    __device__ T operator()(size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
    {
        auto r = pseudo_random(i, j, b, M, N);
        return T(double(r) / double(std::numeric_limits<decltype(r)>::max()) - 0.5);
    }
};

template <typename T>
struct rand_int_functor
{
    __device__ T operator()(size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
    {
        return T(double(pseudo_random(i, j, b, M, N) % 10 + 1));
    }
};

template <typename T>
struct rand_int_zero_one_functor
{
    __device__ T operator()(size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
    {
        return T(double(pseudo_random(i, j, b, M, N) % 2));
    }
};

template <typename T>
struct rand_nan_functor
{
    template <typename UINT_T, int SIG, int EXP, typename REAL_T = T>
    __device__ __host__ REAL_T random_nan_data(uint64_t r)
    {
        static_assert(sizeof(UINT_T) == sizeof(REAL_T), "Type sizes do not match");
        union
        {
            UINT_T u;
            REAL_T fp;
        } x;
        do
            x.u = (UINT_T)r;
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }
    __device__ T operator()(size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
    {
        return T(0);
    }
};

template <>
__device__ double rand_nan_functor<double>::operator()(
    size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
{
    return random_nan_data<uint64_t, 52, 11>(pseudo_random(i, j, b, M, N));
}
template <>
__device__ float rand_nan_functor<float>::operator()(
    size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
{
    return random_nan_data<uint32_t, 23, 8>(pseudo_random(i, j, b, M, N));
}
template <>
__device__ rocblas_half rand_nan_functor<rocblas_half>::operator()(
    size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
{
    return random_nan_data<uint16_t, 10, 5>(pseudo_random(i, j, b, M, N));
}
template <>
__device__ rocblas_bfloat16 rand_nan_functor<rocblas_bfloat16>::operator()(
    size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
{
    return random_nan_data<uint16_t, 7, 8>(pseudo_random(i, j, b, M, N));
}
template <>
__device__ rocblas_double_complex rand_nan_functor<rocblas_double_complex>::operator()(
    size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
{
    return rocblas_double_complex(
        random_nan_data<uint64_t, 52, 11, double>(pseudo_random(i, j, b, M, N, 0)),
        random_nan_data<uint64_t, 52, 11, double>(pseudo_random(i, j, b, M, N, 1)));
}
template <>
__device__ rocblas_float_complex rand_nan_functor<rocblas_float_complex>::operator()(
    size_t i, size_t j, size_t b, rocblas_int M, rocblas_int N, bool)
{
    return rocblas_float_complex(
        random_nan_data<uint32_t, 23, 8, float>(pseudo_random(i, j, b, M, N, 0)),
        random_nan_data<uint32_t, 23, 8, float>(pseudo_random(i, j, b, M, N, 1)));
}

template <typename T>
struct non_rep_bf16_vals_functor
{
    const rocblas_half ieee_half_vals[4] = {2028, 2034, 2036, 2038};
    __device__ T       operator()(size_t i, size_t j, size_t, rocblas_int M, rocblas_int, bool)
    {
        return T(ieee_half_vals[(i + j * M) % 4]);
    }
};

template <typename T>
struct alt_impl_small_functor
{
    __device__ T operator()(size_t i, size_t j, size_t, rocblas_int M, rocblas_int, bool)
    {
        const rocblas_half ieee_half_small(0.0000607967376708984375);
        return T(ieee_half_small);
    }
};

template <typename T>
struct alt_impl_big_functor
{
    __device__ T operator()(size_t i, size_t j, size_t, rocblas_int M, rocblas_int, bool)
    {
        const rocblas_half ieee_half_large(65280.0);
        return T(ieee_half_large);
    }
};

template <int DIM_X, int DIM_Y, typename T, typename F>
ROCBLAS_KERNEL(DIM_X* DIM_Y)
rocblas_init_matrix_kernel(F                         f,
                           rocblas_int               M,
                           rocblas_int               N,
                           T*                        Aa,
                           rocblas_stride            offset,
                           int64_t                   lda,
                           rocblas_stride            stride,
                           rocblas_check_matrix_type matrix_type,
                           char                      uplo,
                           bool                      seedReset,
                           bool                      alternating_sign)
{
    rocblas_int i = blockIdx.x * blockDim.x + threadIdx.x;
    rocblas_int j = blockIdx.y * blockDim.y + threadIdx.y;
    rocblas_int b = blockIdx.z;
    if(i >= M || j >= N)
    {
        return;
    }
    auto* A     = load_ptr_batch(Aa, b, offset, stride);
    auto  value = f(i, j, b, M, N, seedReset);
    if(alternating_sign)
    {
        value = (i ^ j) & 1 ? value : negate(value);
    }
    if(matrix_type == rocblas_client_general_matrix)
    {
        A[i + j * lda] = value;
    }
    else
    {
        if(i == j)
        {
            A[i + j * lda]
                = (matrix_type == rocblas_client_hermitian_matrix) ? std::real(value) : value;
        }
        else if(j < i)
        {
            if(matrix_type == rocblas_client_hermitian_matrix)
            {
                A[i + j * lda] = value;
                A[j + i * lda] = conjugate(value);
            }
            else if(matrix_type == rocblas_client_symmetric_matrix)
            {
                A[i + j * lda] = A[j + i * lda] = value;
            }
            else if(matrix_type == rocblas_client_triangular_matrix)
            {
                A[uplo == 'U' ? j + i * lda : i + j * lda] = value;
                A[uplo == 'U' ? i + j * lda : j + i * lda] = T(0);
            }
            // TODO diagonally dominant?
        }
    }
}

template <typename T, typename F>
rocblas_status rocblas_init_matrix_template(F&&                             f,
                                            rocblas_handle                  handle,
                                            char                            uplo,
                                            rocblas_check_matrix_type       matrix_type,
                                            device_strided_batch_matrix<T>& dA,
                                            bool                            seedReset = false,
                                            bool alternating_sign                     = false)
{
    //Graph capture do not support any use of sync APIs.
    //Quick return: check numerics not supported
    if(handle->is_stream_in_capture_mode())
    {
        return rocblas_status_internal_error;
    }

    T*             A              = dA[0];
    int64_t        m_64           = dA.m();
    int64_t        n_64           = dA.n();
    int64_t        lda            = dA.lda();
    int64_t        batch_count_64 = dA.batch_count();
    rocblas_stride offset         = 0; // is this ever not zero?
    rocblas_stride stride         = dA.stride();

    //Quick return if possible. Not Argument error
    if(!m_64 || !n_64 || !batch_count_64 || !A)
        return rocblas_status_success;

    hipStream_t rocblas_stream = handle->get_stream();

    for(int64_t b_base = 0; b_base < batch_count_64; b_base += c_i64_grid_YZ_chunk)
    {
        auto    a_ptr       = adjust_ptr_batch(A, b_base, stride);
        int32_t batch_count = int32_t(std::min(batch_count_64 - b_base, c_i64_grid_YZ_chunk));

        for(int64_t n_base = 0; n_base < n_64; n_base += c_i64_grid_YZ_chunk)
        {
            int32_t n = int32_t(std::min(n_64 - n_base, c_i64_grid_YZ_chunk));

            rocblas_stride col_offset = offset + n_base * lda;

            for(int64_t m_base = 0; m_base < m_64; m_base += c_i64_grid_X_chunk)
            {
                int32_t m = int32_t(std::min(m_64 - m_base, c_i64_grid_X_chunk));

                rocblas_stride shift = col_offset + m_base;

                static constexpr int DIM_X    = 16;
                static constexpr int DIM_Y    = 16;
                rocblas_int          blocks_X = (m - 1) / DIM_X + 1;
                rocblas_int          blocks_Y = (n - 1) / DIM_Y + 1;

                dim3 blocks(blocks_X, blocks_Y, batch_count);
                dim3 threads(DIM_X, DIM_Y);

                ROCBLAS_LAUNCH_KERNEL((rocblas_init_matrix_kernel<DIM_X, DIM_Y>),
                                      blocks,
                                      threads,
                                      0,
                                      rocblas_stream,
                                      f,
                                      m,
                                      n,
                                      a_ptr,
                                      shift,
                                      lda,
                                      stride,
                                      matrix_type,
                                      uplo,
                                      seedReset,
                                      alternating_sign);
            }
        }
    }
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(rocblas_stream));
    return rocblas_status_success;
}

template <typename T, bool altInit>
void rocblas_init_matrix(rocblas_handle                  handle,
                         device_strided_batch_matrix<T>& dA,
                         const Arguments&                arg,
                         rocblas_check_nan_init          nan_init,
                         rocblas_check_matrix_type       matrix_type,
                         bool                            seedReset,
                         bool                            alternating_sign)
{
    // TODO seedReset
    // seed is the same for every call to this routine

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        CHECK_HIP_ERROR(
            rocblas_init_matrix_template(rand_nan_functor<T>(), handle, arg.uplo, matrix_type, dA));
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        CHECK_HIP_ERROR(
            rocblas_init_matrix_template(rand_nan_functor<T>(), handle, arg.uplo, matrix_type, dA));
    }
    else if(arg.initialization == rocblas_initialization::hpl)
    {
        CHECK_HIP_ERROR(rocblas_init_matrix_template(
            hpl_functor<T>(), handle, arg.uplo, matrix_type, dA, false, alternating_sign));
    }
    else if(arg.initialization == rocblas_initialization::rand_int)
    {
        CHECK_HIP_ERROR(rocblas_init_matrix_template(
            rand_int_functor<T>(), handle, arg.uplo, matrix_type, dA, false, alternating_sign));
    }
    else if(arg.initialization == rocblas_initialization::rand_int_zero_one)
    {
        CHECK_HIP_ERROR(rocblas_init_matrix_template(rand_int_zero_one_functor<T>(),
                                                     handle,
                                                     arg.uplo,
                                                     matrix_type,
                                                     dA,
                                                     false,
                                                     alternating_sign));
    }
    else if(arg.initialization == rocblas_initialization::trig_float)
    {
        CHECK_HIP_ERROR(rocblas_init_matrix_template(
            trig_functor<T>(), handle, arg.uplo, matrix_type, dA, seedReset));
    }
    else if(arg.initialization == rocblas_initialization::denorm)
    {
        if(altInit)
            CHECK_HIP_ERROR(rocblas_init_matrix_template(
                alt_impl_small_functor<T>(), handle, arg.uplo, matrix_type, dA));
        else
            CHECK_HIP_ERROR(rocblas_init_matrix_template(
                alt_impl_big_functor<T>(), handle, arg.uplo, matrix_type, dA));
    }
    else if(arg.initialization == rocblas_initialization::denorm2)
    {
        if(altInit)
            CHECK_HIP_ERROR(rocblas_init_matrix_template(
                non_rep_bf16_vals_functor<T>(), handle, arg.uplo, matrix_type, dA));
        else
            CHECK_HIP_ERROR(rocblas_init_matrix_template(
                identity_functor<T>(), handle, arg.uplo, matrix_type, dA));
    }
    else if(arg.initialization == rocblas_initialization::zero)
    {
        CHECK_HIP_ERROR(hipMemset(dA[0], 0, sizeof(T) * dA.nmemb()));
    }
    else
    {
#ifdef GOOGLE_TEST
        FAIL() << "unknown initialization type";
        return;
#else
        rocblas_cerr << "unknown initialization type" << std::endl;
        rocblas_abort();
#endif
    }
}

#ifdef INST
#error INST IS ALREADY DEFINED
#endif
#define INST(T, altInit)                                                                 \
    template void rocblas_init_matrix<T, altInit>(rocblas_handle handle,                 \
                                                  device_strided_batch_matrix<T> & dA,   \
                                                  const Arguments&          arg,         \
                                                  rocblas_check_nan_init    nan_init,    \
                                                  rocblas_check_matrix_type matrix_type, \
                                                  bool                      seedReset,   \
                                                  bool                      alternate_sign)

INST(int8_t, true);
INST(int, true);
INST(float, true);
INST(double, true);
INST(rocblas_float_complex, true);
INST(rocblas_double_complex, true);
INST(rocblas_half, true);
INST(rocblas_bfloat16, true);

INST(int8_t, false);
INST(int, false);
INST(float, false);
INST(double, false);
INST(rocblas_float_complex, false);
INST(rocblas_double_complex, false);
INST(rocblas_half, false);
INST(rocblas_bfloat16, false);

#undef INST
