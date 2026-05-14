/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "hipblaslt_datatype2string.hpp"
#include "hipblaslt_init.hpp"
#include "hipblaslt_ostream.hpp"
#include "hipblaslt_random.hpp"
#include "hipblaslt_test.hpp"
#include <hipblaslt/hipblaslt.h>
#include <type_traits>

template <typename T, typename F>
__global__ void fill_kernel(T* A, size_t size, size_t offset, F f)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
        A[idx + offset] = f(idx + offset);
}

template <typename T, typename F>
void fill_batch(T* A, size_t M, size_t N, size_t lda, size_t stride, size_t batch_count, const F& f)
{
    size_t           size_64   = stride >= lda ? lda * N + size_t(batch_count - 1) * stride : lda * N;
    if constexpr(
        false
#if defined(HIPBLASLT_USE_FP4)
        || std::is_same_v<T, hipblaslt_f4x2>
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
        || std::is_same_v<T, hipblaslt_f6x16> || std::is_same_v<T, hipblaslt_bf6x16>
#endif
    )
    {
        using type = T;
        size_64    = size_64 / type::packed_size;
    }
    constexpr size_t c_i32_max = size_t(std::numeric_limits<int32_t>::max());
    for(size_t offset = 0; offset < size_64; offset += c_i32_max)
    {
        size_t size       = std::min(size_64 - offset, c_i32_max);
        size_t block_size = 256;
        size_t grid_size  = (size + block_size - 1) / block_size;
        fill_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(A, size, offset, f);
    }
    CHECK_HIP_ERROR(hipGetLastError());
}

__device__ uint32_t pseudo_random_device(size_t idx)
{
    // Numerical Recipes ranqd1, Chapter 7.1, ?An Even Quicker Generator, Eq. 7.1.6. parameters from Knuth and H. W. Lewis
    auto s = idx * 1664525 + 1013904223;
    // Run a few extra iterations to make the generators diverge
    // in case the seeds are still poor (consecutive ints)
    // Marsaglia, G. (2003). "Xorshift RNGs". Journal of Statistical Software. 8 (14). doi:10.18637/jss.v008.i14
    s ^= (s << 13) ^ ( s >> 17) ^ (s << 5);
    s ^= (s << 13) ^ ( s >> 17) ^ (s << 5);
    s ^= (s << 13) ^ ( s >> 17) ^ (s << 5);
    return s;
}

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
__device__ T random_int(size_t idx)
{
    return T(pseudo_random_device(idx) % 10 + 1.f);
}

/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
__device__ hipblasLtHalf random_int<hipblasLtHalf>(size_t idx)
{
    return hipblasLtHalf(pseudo_random_device(idx) % 5 - 2.f);
}

/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
__device__ hip_bfloat16 random_int<hip_bfloat16>(size_t idx)
{
    return hip_bfloat16(pseudo_random_device(idx) % 5 - 2.f);
}

/*! \brief  generate a random number in range [1,2,3] */
template <>
__device__ int8_t random_int<int8_t>(size_t idx)
{
    return pseudo_random_device(idx) % 3 + 1;
}

/*! \brief  generate a random number in range [0, 1, 2] for integer_exact init */
template <typename T>
__device__ T small_int_positive(size_t idx)
{
    return T(pseudo_random_device(idx) % 3);
}

template <>
__device__ hipblasLtHalf small_int_positive<hipblasLtHalf>(size_t idx)
{
    return hipblasLtHalf(pseudo_random_device(idx) % 3);
}

template <>
__device__ hip_bfloat16 small_int_positive<hip_bfloat16>(size_t idx)
{
    return hip_bfloat16(pseudo_random_device(idx) % 3);
}

template <>
__device__ int8_t small_int_positive<int8_t>(size_t idx)
{
    return static_cast<int8_t>(pseudo_random_device(idx) % 3);
}

#if defined(HIPBLASLT_USE_FP4)
/*! \brief  generate a random number in range [-4,-3,-2,-1,0,1,2,3,4] */
template <>
__device__ hipblaslt_f4x2 random_int<hipblaslt_f4x2>(size_t idx)
{
    auto r0 = static_cast<int>(pseudo_random_device(2 * idx) % 9) - 4;
    auto r1 = static_cast<int>(pseudo_random_device(2 * idx + 1) % 9) - 4;
    return hipblaslt_f4x2(float(r0), float(r1));
}
#endif

#if defined(HIPBLASLT_USE_FP6)
/*! \brief  generate a random number in range [-7, -6, ..., 7] */
template <>
__device__ hipblaslt_f6x16 random_int<hipblaslt_f6x16>(size_t idx)
{
    using type               = hipblaslt_f6x16;
    int r[type::packed_size] = {0};
    for(size_t i = 0; i < type::packed_size; i++)
    {
        r[i] = static_cast<int>(pseudo_random_device(type::packed_size * idx + i) % 15) - 7;
    }
    return hipblaslt_f6x16(float(r[0]),
                           float(r[1]),
                           float(r[2]),
                           float(r[3]),
                           float(r[4]),
                           float(r[5]),
                           float(r[6]),
                           float(r[7]),
                           float(r[8]),
                           float(r[9]),
                           float(r[10]),
                           float(r[11]),
                           float(r[12]),
                           float(r[13]),
                           float(r[14]),
                           float(r[15]));
}
#endif

#if defined(HIPBLASLT_USE_BF6)
/*! \brief  generate a random number in range [-28, -27, ..., 28] */
template <>
__device__ hipblaslt_bf6x16 random_int<hipblaslt_bf6x16>(size_t idx)
{
    using type               = hipblaslt_bf6x16;
    int r[type::packed_size] = {0};
    for(size_t i = 0; i < type::packed_size; i++)
    {
        r[i] = static_cast<int>(pseudo_random_device(type::packed_size * idx + i) % 57) - 28;
    }
    return hipblaslt_bf6x16(float(r[0]),
                            float(r[1]),
                            float(r[2]),
                            float(r[3]),
                            float(r[4]),
                            float(r[5]),
                            float(r[6]),
                            float(r[7]),
                            float(r[8]),
                            float(r[9]),
                            float(r[10]),
                            float(r[11]),
                            float(r[12]),
                            float(r[13]),
                            float(r[14]),
                            float(r[15]));
}
#endif

/*! \brief  generate a random number in range [2^-3,2^-2,2^-1,2^0,]2^1,2^2,2^3]] */
template <>
__device__ hipblaslt_e8 random_int<hipblaslt_e8>(size_t idx)
{
    hipblaslt_e8 val;
    val.data = ((pseudo_random_device(idx) % 7 - 3) + 127);
    return val;
}

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
__device__ T random_hpl(size_t idx)
{
    auto r = pseudo_random_device(idx);
    return T(double(r) / double(std::numeric_limits<decltype(r)>::max()) - 0.5);
}

/*! \brief  generate a random number in [-1.0,1.0] doubles  */
template <>
__device__ int8_t random_hpl(size_t idx)
{
    auto r = pseudo_random_device(idx);
    auto v = nearbyint(double(r) / double(std::numeric_limits<decltype(r)>::max()) * 2. - 1.);
    return int8_t(v > 127.f ? 127.f : v < -128.f ? -128.f : v);
}

template <typename T>
struct is_std_complex : std::false_type
{
};

template <typename U>
struct is_std_complex<std::complex<U>> : std::true_type
{
};

template <typename T>
struct get_real_type
{
    using type = T; // Default for non-complex types (e.g., float, double)
};

template <typename U>
struct get_real_type<std::complex<U>>
{
    using type = U; // Correctly extracts float or double
};

template <typename T>
using get_real_type_t = typename get_real_type<T>::type;

/*! \brief  generate a random number in [0.,1.0]  */
template <typename T>
__device__ T uniform_01(size_t idx)
{
    auto r = pseudo_random_device(idx);
    return T(double(r) / double(std::numeric_limits<decltype(r)>::max()));
}

/*! \brief  generate a random number in [0.,1.0]  */
template <>
__device__ int8_t uniform_01(size_t idx)
{
    auto r = pseudo_random_device(idx);
    auto v = nearbyint(double(r) / double(std::numeric_limits<decltype(r)>::max()));
    return int8_t(v > 127.f ? 127.f : v < -128.f ? -128.f : v);
}

#if defined(HIPBLASLT_USE_FP4)
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
__device__ hipblaslt_f4x2 random_hpl(size_t idx)
{
    constexpr auto cvt_max_ui32_to_double
        = static_cast<double>(std::numeric_limits<uint32_t>::max());
    auto r0 = static_cast<double>(pseudo_random_device(2 * idx)) / cvt_max_ui32_to_double - 0.5;
    auto r1 = static_cast<double>(pseudo_random_device(2 * idx + 1)) / cvt_max_ui32_to_double - 0.5;
    return hipblaslt_f4x2(r0, r1);
}
#endif

#if defined(HIPBLASLT_USE_FP6)
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
__device__ hipblaslt_f6x16 random_hpl(size_t idx)
{
    using type                          = hipblaslt_f6x16;
    double         r[type::packed_size] = {0.0};
    constexpr auto cvt_max_ui32_to_double
        = static_cast<double>(std::numeric_limits<uint32_t>::max());
    for(size_t i = 0; i < type::packed_size; i++)
    {
        r[i] = static_cast<double>(pseudo_random_device(type::packed_size * idx + i))
                   / cvt_max_ui32_to_double
               - 0.5;
    }
    return hipblaslt_f6x16(r[0],
                           r[1],
                           r[2],
                           r[3],
                           r[4],
                           r[5],
                           r[6],
                           r[7],
                           r[8],
                           r[9],
                           r[10],
                           r[11],
                           r[12],
                           r[13],
                           r[14],
                           r[15]);
}
#endif

#if defined(HIPBLASLT_USE_BF6)
/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <>
__device__ hipblaslt_bf6x16 random_hpl(size_t idx)
{
    using type                          = hipblaslt_bf6x16;
    double         r[type::packed_size] = {0.0};
    constexpr auto cvt_max_ui32_to_double
        = static_cast<double>(std::numeric_limits<uint32_t>::max());
    for(size_t i = 0; i < type::packed_size; i++)
    {
        r[i] = static_cast<double>(pseudo_random_device(type::packed_size * idx + i))
                   / cvt_max_ui32_to_double
               - 0.5;
    }
    return hipblaslt_bf6x16(r[0],
                            r[1],
                            r[2],
                            r[3],
                            r[4],
                            r[5],
                            r[6],
                            r[7],
                            r[8],
                            r[9],
                            r[10],
                            r[11],
                            r[12],
                            r[13],
                            r[14],
                            r[15]);
}
#endif

/*! \brief  generate a random number in range [2^-3,2^-2,2^-1,2^0,]2^1,2^2,2^3]] */
template <>
__device__ hipblaslt_e8 random_hpl<hipblaslt_e8>(size_t idx)
{
    hipblaslt_e8 val;
    val.data = ((pseudo_random_device(idx) % 7 - 3) + 127);
    return val;
}

/*! \brief  generate a float value using trig function (e.g., sin or cos) based on logical 3D index. */
template <typename T, typename Func>
__device__ T
    trig_float(size_t idx, size_t M, size_t N, size_t lda, size_t stride, Func func)
{
    // M_PI is not part of ISO C/C++ and is not defined by <cmath> on Windows (MSVC CRT / clang-cl)
    // unless _USE_MATH_DEFINES is set before the first <math.h>/<cmath> include. Use a local
    // constexpr instead so this translation unit builds on all platforms without macro gymnastics.
    constexpr double two_pi = 6.28318530717958647692528676655900576;
    auto calc = [&](size_t k) {
        auto b = k / stride;
        auto j = (k - b * stride) / lda;
        auto i = (k - b * stride) - j * lda;
        return fmod(double(i + j * M + b * M * N), two_pi);
    };

#if defined(HIPBLASLT_USE_FP4)
    if constexpr(std::is_same_v<T, hipblaslt_f4x2>)
        return hipblaslt_f4x2(func(calc(2 * idx)), func(calc(2 * idx + 1)));
    else
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
    if constexpr(std::is_same_v<T, hipblaslt_f6x16> || std::is_same_v<T, hipblaslt_bf6x16>)
    {
        using type = T;
        return T(func(calc(type::packed_size * idx)),
                 func(calc(type::packed_size * idx + 1)),
                 func(calc(type::packed_size * idx + 2)),
                 func(calc(type::packed_size * idx + 3)),
                 func(calc(type::packed_size * idx + 4)),
                 func(calc(type::packed_size * idx + 5)),
                 func(calc(type::packed_size * idx + 6)),
                 func(calc(type::packed_size * idx + 7)),
                 func(calc(type::packed_size * idx + 8)),
                 func(calc(type::packed_size * idx + 9)),
                 func(calc(type::packed_size * idx + 10)),
                 func(calc(type::packed_size * idx + 11)),
                 func(calc(type::packed_size * idx + 12)),
                 func(calc(type::packed_size * idx + 13)),
                 func(calc(type::packed_size * idx + 14)),
                 func(calc(type::packed_size * idx + 15)));
    }
    else
#endif
        return T(func(calc(idx)));
}

template <typename T>
__device__ T norm_dist(uint32_t base_seed, size_t idx)
{
    hipblaslt_norm_dist::XorwowState state;
    hipblaslt_norm_dist::init_xorwow(&state, base_seed + idx); // Unique seed per thread
    return T(hipblaslt_norm_dist::box_muller_normal(&state));
}

#if defined(HIPBLASLT_USE_FP4)
template <>
__device__ hipblaslt_f4x2 norm_dist(uint32_t base_seed, size_t idx)
{
    float r0 = norm_dist<float>(base_seed, 2 * idx);
    float r1 = norm_dist<float>(base_seed, 2 * idx + 1);
    return hipblaslt_f4x2(r0, r1);
}
#endif

#if defined(HIPBLASLT_USE_FP6)
template <>
__device__ hipblaslt_f6x16 norm_dist(uint32_t base_seed, size_t idx)
{
    using type                 = hipblaslt_f6x16;
    float r[type::packed_size] = {0.f};
    for(size_t i = 0; i < type::packed_size; i++)
    {
        r[i] = norm_dist<float>(base_seed, type::packed_size * idx + i);
    }
    return hipblaslt_f6x16(r[0],
                           r[1],
                           r[2],
                           r[3],
                           r[4],
                           r[5],
                           r[6],
                           r[7],
                           r[8],
                           r[9],
                           r[10],
                           r[11],
                           r[12],
                           r[13],
                           r[14],
                           r[15]);
}
#endif

#if defined(HIPBLASLT_USE_BF6)
template <>
__device__ hipblaslt_bf6x16 norm_dist(uint32_t base_seed, size_t idx)
{
    using type                 = hipblaslt_bf6x16;
    float r[type::packed_size] = {0.f};
    for(size_t i = 0; i < type::packed_size; i++)
    {
        r[i] = norm_dist<float>(base_seed, type::packed_size * idx + i);
    }
    return hipblaslt_bf6x16(r[0],
                            r[1],
                            r[2],
                            r[3],
                            r[4],
                            r[5],
                            r[6],
                            r[7],
                            r[8],
                            r[9],
                            r[10],
                            r[11],
                            r[12],
                            r[13],
                            r[14],
                            r[15]);
}
#endif

template <typename T>
void hipblaslt_init_device(ABC_dims                 abc,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           T*                       A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           size_t                   stride,
                           size_t                   batch_count)
{
    using T_real = get_real_type_t<T>;

    // Helper to construct std::complex types from two real components
    auto make_std_complex = [] __device__ __host__ (T_real r, T_real i) -> T {
        if constexpr(std::is_same_v<T, std::complex<float>>)
            return std::complex<float>(r, i);
        else if constexpr(std::is_same_v<T, std::complex<double>>)
            return std::complex<double>(r, i);
        else
            return T(r);
    };

    if(is_nan)
    {
        if constexpr(
            false
#if defined(HIPBLASLT_USE_FP4)
            || std::is_same_v<T, hipblaslt_f4x2>
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
            || std::is_same_v<T, hipblaslt_f6x16>
            || std::is_same_v<T, hipblaslt_bf6x16>
#endif
            || std::is_same_v<T, hipblaslt_e8>
            || std::is_same_v<T, hipblaslt_e5m3>)
        {
            hipblaslt_cerr << "No support nan for HIP_R_4F_E2M1 and HIP_R_6F_E2M3 and "
                              "HIP_R_6F_E3M2 in hipblaslt_init_device"
                           << std::endl;
        }
        else
        {
            std::array<T, 100> rand_nans;
            for(auto& r : rand_nans)
            {
                if constexpr(is_std_complex<T>::value)
                  // Use T_real for casting and the helper for construction
                  r = make_std_complex(static_cast<T_real>(hipblaslt_nan_rng()),
                                     static_cast<T_real>(hipblaslt_nan_rng()));
                else
                  r = T(hipblaslt_nan_rng());
            }
            fill_batch(A, M, N, lda, stride, batch_count, [rand_nans](size_t idx) -> T {
                return rand_nans[pseudo_random_device(idx) % rand_nans.size()];
            });
        }
    }
    else
    {
        switch(init)
        {
        case hipblaslt_initialization::rand_int:
            if(abc == ABC_dims::A || abc == ABC_dims::C)
            {
                if constexpr(is_std_complex<T>::value)
                    fill_batch(A, M, N, lda, stride, batch_count, [make_std_complex](size_t idx) -> T {                    
                        return make_std_complex(random_int<T_real>(idx), random_int<T_real>(idx + 1000000) );
                    });
                else
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {                    
                        return random_int<T>(idx);               
                });
            }
            else if(abc == ABC_dims::B)
            {
                if constexpr(is_std_complex<T>::value)
                {
                if(stride >= lda)
                {
                    stride = std::max(lda * N, stride);
                    fill_batch(A, M, N, lda, stride, batch_count, [stride, lda, make_std_complex](size_t idx) -> T {
                        auto b     = idx / stride;
                        auto j     = (idx - b * stride) / lda;
                        auto i     = (idx - b * stride) - j * lda;
                        auto real_val = random_int<T_real>(idx);
                        auto imag_val = random_int<T_real>(idx + 1000000); // Offset for different seed
                        auto complex_val = make_std_complex(real_val, imag_val);
                        return (i ^ j) & 1 ? complex_val : negate(complex_val);
                    });
                }
                else
                {
                    fill_batch(A, M, N, lda, stride, batch_count, [stride, lda, make_std_complex](size_t idx) -> T {
                        auto j     = idx / lda;
                        auto b     = (idx - j * lda) / stride;
                        auto i     = (idx - j * lda) - b * stride;
                        auto real_val = random_int<T_real>(idx);
                        auto imag_val = random_int<T_real>(idx + 1000000); // Offset for different seed
                        auto complex_val = make_std_complex(real_val, imag_val);
                        return (i ^ j) & 1 ? complex_val : negate(complex_val);
                    });
                }
                }
                else{
                if(stride >= lda)
                {
                    stride = std::max(lda * N, stride);
                    fill_batch(A, M, N, lda, stride, batch_count, [stride, lda] __host__ __device__ (size_t idx) -> T {
                        auto b     = idx / stride;
                        auto j     = (idx - b * stride) / lda;
                        auto i     = (idx - b * stride) - j * lda;
                        auto value = random_int<T>(idx);
                        return (i ^ j) & 1 ? value : negate(value);
                    });
                }
                else
                {
                    fill_batch(A, M, N, lda, stride, batch_count, [stride, lda] __host__ __device__ (size_t idx) -> T {
                        auto j     = idx / lda;
                        auto b     = (idx - j * lda) / stride;
                        auto i     = (idx - j * lda) - b * stride;
                        auto value = random_int<T>(idx);
                        return (i ^ j) & 1 ? value : negate(value);
                    });
                }
                } 
            }
            break;
        case hipblaslt_initialization::trig_float:
            if constexpr(is_std_complex<T>::value)
            {
                if(stride >= lda)
            {
                stride = std::max(lda * N, stride);
                    fill_batch(
                        A, M, N, lda, stride, batch_count, [M, N, stride, lda, make_std_complex](size_t idx) -> T {
                            auto b = idx / stride;
                            auto j = (idx - b * stride) / lda;
                            auto i = (idx - b * stride) - j * lda;
                            auto arg = double(i + j * M + b * M * N);
                            auto real_val = sin(random_int<T_real>(arg));
                            auto imag_val = cos(random_int<T_real>(arg + 1000000)); // Offset for different seed
                            auto complex_val = make_std_complex(real_val, imag_val);
                            return complex_val;
                        });
            }
            else
            {
                    fill_batch(
                        A, M, N, lda, stride, batch_count, [M, N, stride, lda, make_std_complex](size_t idx) -> T {
                            auto j = idx / lda;
                            auto b = (idx - j * lda) / stride;
                            auto i = (idx - j * lda) - b * stride;
                            auto arg = double(i + j * M + b * M * N);
                            auto real_val = sin(random_int<T_real>(arg));
                            auto imag_val = cos(random_int<T_real>(arg + 1000000)); // Offset for different seed
                            auto complex_val = make_std_complex(real_val, imag_val);
                            return complex_val;
                        });
            }
            }
            else
            {
                if(stride >= lda)
            {
                stride = std::max(lda * N, stride);
                    fill_batch(
                        A, M, N, lda, stride, batch_count, [M, N, stride, abc, lda](size_t idx) -> T {
                            auto b = idx / stride;
                            auto j = (idx - b * stride) / lda;
                            auto i = (idx - b * stride) - j * lda;
                            auto arg = double(i + j * M + b * M * N);
                            return T((abc == ABC_dims::B) ? cos(arg) : sin(arg));
                        });
            }
            else
            {
                    fill_batch(
                        A, M, N, lda, stride, batch_count, [M, N, stride, abc, lda](size_t idx) -> T {
                            auto j = idx / lda;
                            auto b = (idx - j * lda) / stride;
                            auto i = (idx - j * lda) - b * stride;
                            auto arg = double(i + j * M + b * M * N);
                            return T((abc == ABC_dims::B) ? cos(arg) : sin(arg));
                        });
            }
            }
            break;
        case hipblaslt_initialization::hpl:
            if constexpr(is_std_complex<T>::value)
                fill_batch(A, M, N, lda, stride, batch_count, [make_std_complex] __host__ __device__ (size_t idx) -> T {
                    return make_std_complex(random_hpl<T_real>(idx), random_hpl<T_real>(idx + 1000000));
                });
            else
            fill_batch(A, M, N, lda, stride, batch_count, [] __host__ __device__ (size_t idx) -> T {
                return random_hpl<T>(idx);
            });
            break;
        case hipblaslt_initialization::special:
            if(abc == ABC_dims::A)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(hipblasLtHalf(65280.0));
                });
            else if(abc == ABC_dims::B)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(hipblasLtHalf(0.0000607967376708984375));
                });
            else if(abc == ABC_dims::C)
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return T(pseudo_random_device(idx) % 10 + 1.f);
                });
            break;
        case hipblaslt_initialization::zero:
            if constexpr(is_std_complex<T>::value)
                fill_batch(A, M, N, lda, stride, batch_count, [make_std_complex] __host__ __device__ (size_t idx) -> T { 
                    return make_std_complex(T_real(0), T_real(0));
                });
            else
            fill_batch(A, M, N, lda, stride, batch_count, [] __host__ __device__ (size_t idx) -> T {
                if constexpr(
                    false
#if defined(HIPBLASLT_USE_FP4)
                    || std::is_same_v<T, hipblaslt_f4x2>
#endif
#if defined(HIPBLASLT_USE_FP6) && defined(HIPBLASLT_USE_BF6)
                    || std::is_same_v<T, hipblaslt_f6x16> || std::is_same_v<T, hipblaslt_bf6x16>
#endif
                )
                    return T(0.f);
                else
                    return T(0);
            });
            break;
        case hipblaslt_initialization::norm_dist:
            {
                std::random_device rd;
                auto base_seed = rd(); // Get a random seed for each run
                fill_batch(A, M, N, lda, stride, batch_count, [base_seed] __device__ (size_t idx) -> T {
                    hipblaslt_norm_dist::XorwowState state;
                    hipblaslt_norm_dist::init_xorwow(&state, base_seed + idx); // Unique seed per thread
                    return T(hipblaslt_norm_dist::box_muller_normal(&state));
                });
                break;
            }
        case hipblaslt_initialization::uniform_01:
            fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                return uniform_01<T>(idx);
            });
            break;
        case hipblaslt_initialization::integer_exact:
            // A and C: [0,1,2] (C with beta); B: checkerboard ±[0,1,2]
            if(abc == ABC_dims::A || abc == ABC_dims::C)
            {
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t idx) -> T {
                    return small_int_positive<T>(idx);
                });
            }
            else if(abc == ABC_dims::B)
            {
                // Checkerboard ±: (i^j)&1 so first element of each row and column alternates
                // Use an effective stride that is never zero and at least large enough
                // to contain one full matrix, to avoid division by a potentially zero stride.
                // Offset PRNG index for B so {0,1,2} magnitudes differ from A (same idx would
                // correlate via pseudo_random_device).
                constexpr size_t kBSeedOffset = 1000003; // large prime
                size_t effective_stride = stride ? std::max(stride, lda * N) : lda * N;
                fill_batch(A, M, N, lda, effective_stride, batch_count, [effective_stride, lda](size_t idx) -> T {
                    auto b        = idx / effective_stride;
                    auto in_batch = idx - b * effective_stride;
                    auto j        = in_batch / lda;
                    auto i        = in_batch - j * lda;
                    auto value    = small_int_positive<T>(idx + kBSeedOffset);
                    return (i ^ j) & 1 ? value : negate(value);
                });
            }
            break;
        case hipblaslt_initialization::fp16_accumulator_probe:
            if constexpr(std::is_same_v<T, hipblasLtHalf>)
            {
                if(abc == ABC_dims::A)
                {
                    const float fmax = 65504.f - 4.f;
                    fill_batch(A, M, N, lda, stride, batch_count, [fmax](size_t) -> T {
                        return T(hipblasLtHalf(fmax));
                    });
                }
                else if(abc == ABC_dims::B)
                {
                    // Match integer_exact B: use effective_stride in fill_batch so batch_count>1 with
                    // stride==0 still covers every batch slab (stride_b defaults to 0 in Arguments).
                    size_t effective_stride = stride ? std::max(stride, lda * N) : lda * N;
                    fill_batch(A, M, N, lda, effective_stride, batch_count, [effective_stride, lda](size_t idx) -> T {
                        auto b        = idx / effective_stride;
                        auto in_batch = idx - b * effective_stride;
                        auto n        = in_batch / lda;
                        auto k        = in_batch - n * lda;
                        (void)n;
                        const float f2 = 2.f;
                        if((k % 2) == 0)
                            return T(hipblasLtHalf(f2));
                        return T(hipblasLtHalf(-f2));
                    });
                }
                else
                {
                    fill_batch(A, M, N, lda, stride, batch_count, [](size_t) -> T { return T(0); });
                }
            }
            else
            {
                fill_batch(A, M, N, lda, stride, batch_count, [](size_t) -> T { return T(0); });
            }
            break;
        default:
            hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
            break;
        }
    }
}

void hipblaslt_init_device(ABC_dims                 abc,
                           hipblaslt_initialization init,
                           bool                     is_nan,
                           void*                    A,
                           size_t                   M,
                           size_t                   N,
                           size_t                   lda,
                           hipDataType              type,
                           size_t                   stride,
                           size_t                   batch_count)
{
    switch(type)
    {
    case HIP_R_32F:
        hipblaslt_init_device<float>(
            abc, init, is_nan, static_cast<float*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_64F:
        hipblaslt_init_device<double>(
            abc, init, is_nan, static_cast<double*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16F:
        hipblaslt_init_device<hipblasLtHalf>(
            abc, init, is_nan, static_cast<hipblasLtHalf*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_16BF:
        hipblaslt_init_device<hip_bfloat16>(
            abc, init, is_nan, static_cast<hip_bfloat16*>(A), M, N, lda, stride, batch_count);
        break;
#if HIP_FP8_TYPE_FNUZ
    case HIP_R_8F_E4M3_FNUZ:
        hipblaslt_init_device<hipblaslt_f8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_f8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2_FNUZ:
        hipblaslt_init_device<hipblaslt_bf8_fnuz>(
            abc, init, is_nan, static_cast<hipblaslt_bf8_fnuz*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if HIP_FP8_TYPE_OCP
    case HIP_R_8F_E4M3:
        hipblaslt_init_device<hipblaslt_f8>(
            abc, init, is_nan, static_cast<hipblaslt_f8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_E5M2:
        hipblaslt_init_device<hipblaslt_bf8>(
            abc, init, is_nan, static_cast<hipblaslt_bf8*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_R_32I:
        hipblaslt_init_device<int32_t>(
            abc, init, is_nan, static_cast<int32_t*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8I:
        hipblaslt_init_device<hipblasLtInt8>(
            abc, init, is_nan, static_cast<hipblasLtInt8*>(A), M, N, lda, stride, batch_count);
        break;
    case HIP_R_8F_UE8M0:
        hipblaslt_init_device<hipblaslt_e8>(
            abc, init, is_nan, static_cast<hipblaslt_e8*>(A), M, N, lda, stride, batch_count);
        break;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch"
    case HIP_R_8F_E5M3_EXT:
#pragma GCC diagnostic pop
        hipblaslt_init_device<hipblaslt_e5m3>(
            abc, init, is_nan, static_cast<hipblaslt_e5m3*>(A), M, N, lda, stride, batch_count);
        break;
#if defined(HIPBLASLT_USE_FP6)
    case static_cast<hipDataType>(HIP_R_6F_E2M3):
        hipblaslt_init_device<hipblaslt_f6x16>(
            abc, init, is_nan, static_cast<hipblaslt_f6x16*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if defined(HIPBLASLT_USE_BF6)
    case static_cast<hipDataType>(HIP_R_6F_E3M2):
        hipblaslt_init_device<hipblaslt_bf6x16>(
            abc, init, is_nan, static_cast<hipblaslt_bf6x16*>(A), M, N, lda, stride, batch_count);
        break;
#endif
#if defined(HIPBLASLT_USE_FP4)
    case static_cast<hipDataType>(HIP_R_4F_E2M1):
        hipblaslt_init_device<hipblaslt_f4x2>(
            abc, init, is_nan, static_cast<hipblaslt_f4x2*>(A), M, N, lda, stride, batch_count);
        break;
#endif
    case HIP_C_32F:
        hipblaslt_init_device<std::complex<float>>(abc,
                                                   init,
                                                   is_nan,
                                                   static_cast<std::complex<float>*>(A),
                                                   M,
                                                   N,
                                                   lda,
                                                   stride,
                                                   batch_count);
        break;
    case HIP_C_64F:
        hipblaslt_init_device<std::complex<double>>(abc,
                                                    init,
                                                    is_nan,
                                                    static_cast<std::complex<double>*>(A),
                                                    M,
                                                    N,
                                                    lda,
                                                    stride,
                                                    batch_count);
        break;
    default:
        hipblaslt_cerr << "Error type in hipblaslt_init_device" << std::endl;
        break;
    }
}
