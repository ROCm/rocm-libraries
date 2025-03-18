// Copyright (C) 2016 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef COMMON_H
#define COMMON_H
#include <hip/hip_runtime.h>
#include <hip/hip_vector_types.h>

#if defined(__HIPCC_RTC__) || defined(__CUDACC_RTC__)
typedef signed int   int32_t;
typedef unsigned int uint32_t;
#endif

#ifdef WIN32
#define ROCFFT_DEVICE_EXPORT __declspec(dllexport)
#else
#define ROCFFT_DEVICE_EXPORT
#endif

// NB:
//   All kernels were compiled based on the assumption that the default max
//   work group size is 256. This default value in compiler might change in
//   future. Each kernel has to explicitly set proper sizes through
//   __launch_bounds__ or __attribute__.
//   Further performance tuning might be done later.
static const unsigned int LAUNCH_BOUNDS_R2C_C2R_KERNEL = 256;

#ifdef __HIP_PLATFORM_NVIDIA__
#include "vector_types.h"
#include <cuComplex.h>

__device__ inline rocfft_complex<float> operator-(const rocfft_complex<float>& a,
                                                  const rocfft_complex<float>& b)
{
    return rocfft_complex<float>(a.x - b.x, a.y - b.y);
}
__device__ inline rocfft_complex<float> operator+(const rocfft_complex<float>& a,
                                                  const rocfft_complex<float>& b)
{
    return rocfft_complex<float>(a.x + b.x, a.y + b.y);
}
__device__ inline rocfft_complex<float> operator*(const float& a, const rocfft_complex<float>& b)
{
    return rocfft_complex<float>(a * b.x, a * b.y);
}
__device__ inline rocfft_complex<float> operator*=(rocfft_complex<float>&       a,
                                                   const rocfft_complex<float>& b)
{
    a = cuCmulf(a, b);
    return a;
}
__device__ inline rocfft_complex<float> operator*=(rocfft_complex<float>& a, const float& b)
{
    a = cuCmulf(a, rocfft_complex<float>(b, b));
    return a;
}
__device__ inline rocfft_complex<float> operator-(const rocfft_complex<float>& a)
{
    return cuCmulf(a, rocfft_complex<float>(-1.0, -1.0));
}

__device__ inline rocfft_complex<double> operator-(const rocfft_complex<double>& a,
                                                   const rocfft_complex<double>& b)
{
    return rocfft_complex<double>(a.x - b.x, a.y - b.y);
}
__device__ inline rocfft_complex<double> operator+(const rocfft_complex<double>& a,
                                                   const rocfft_complex<double>& b)
{
    return rocfft_complex<double>(a.x + b.x, a.y + b.y);
}
__device__ inline rocfft_complex<double> operator*(const double& a, const rocfft_complex<double>& b)
{
    return rocfft_complex<double>(a * b.x, a * b.y);
}
__device__ inline rocfft_complex<double> operator*=(rocfft_complex<double>&       a,
                                                    const rocfft_complex<double>& b)
{
    a = cuCmul(a, b);
    return a;
}
__device__ inline rocfft_complex<double> operator*=(rocfft_complex<double>& a, const double& b)
{
    a = cuCmul(a, rocfft_complex<double>(b, b));
    return a;
}
__device__ inline rocfft_complex<double> operator-(const rocfft_complex<double>& a)
{
    return cuCmul(a, rocfft_complex<double>(-1.0, -1.0));
}

#endif

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

enum class EmbeddedType : int
{
    NONE        = 0, // Works as the regular complex to complex FFT kernel
    Real2C_POST = 1, // Works with even-length real2complex post-processing
    C2Real_PRE  = 2, // Works with even-length complex2real pre-processing
};

// TODO: rework this
//
//
// NB:
// SBRC kernels can be used in various scenarios. Instead of tmeplate all
// combinations, we define/enable the cases in using only. In this way,
// the logic in POWX_LARGE_SBRC_GENERATOR() would be simple. People could
// add more later or find a way to simply POWX_LARGE_SBRC_GENERATOR().
enum SBRC_TYPE
{
    SBRC_2D = 2, // for one step in 1D middle size decomposition

    SBRC_3D_FFT_TRANS_XY_Z = 3, // for 3D C2C middle size fused kernel
    SBRC_3D_FFT_TRANS_Z_XY = 4, // for 3D R2C middle size fused kernel
    SBRC_3D_TRANS_XY_Z_FFT = 5, // for 3D C2R middle size fused kernel

    // for 3D R2C middle size, to fuse FFT, Even-length real2complex, and Transpose_Z_XY
    SBRC_3D_FFT_ERC_TRANS_Z_XY = 6,

    // for 3D C2R middle size, to fuse Transpose_XY_Z, Even-length complex2real, and FFT
    SBRC_3D_TRANS_XY_Z_ECR_FFT = 7,
};

enum SBRC_TRANSPOSE_TYPE
{
    NONE, // indicating this is a non-sbrc type, an SBRC kernel shouldn't have this
    DIAGONAL, // best, but requires cube sizes
    TILE_ALIGNED, // OK, doesn't require handling unaligned corner case
    TILE_UNALIGNED,
};

enum DirectRegType
{
    // the direct-to-from-reg codes are not even generated from generator
    // or is generated but we don't want to use it in some arch
    FORCE_OFF_OR_NOT_SUPPORT,
    TRY_ENABLE_IF_SUPPORT, // Use the direct-to-from-reg function
};

enum IntrinsicAccessType
{
    DISABLE_BOTH, // turn-off intrinsic buffer load/store
    ENABLE_LOAD_ONLY, // turn-on intrinsic buffer load only
    ENABLE_BOTH, // turn-on both intrinsic buffer load/store
};

enum BluesteinType
{
    BT_NONE,
    BT_SINGLE_KERNEL, // implementation for small lengths (that fit in LDS)
    BT_MULTI_KERNEL, // large lengths
    BT_MULTI_KERNEL_FUSED, // large lengths with fused intermediate Bluestein operations
};

enum BluesteinFuseType
{ // Fused operation types for multi-kernel Bluestein
    BFT_NONE,
    BFT_FWD_CHIRP, // fused chirp + padding + forward fft
    BFT_FWD_CHIRP_MUL, // fused chirp / input Hadamard product + padding + forward fft
    BFT_INV_CHIRP_MUL, // fused convolution Hadamard product + inverse fft + chirp Hadamard product
};

template <class T>
struct real_type;

template <>
struct real_type<rocfft_complex<float>>
{
    typedef float type;
};

template <>
struct real_type<rocfft_complex<double>>
{
    typedef double type;
};

template <>
struct real_type<rocfft_complex<rocfft_fp16>>
{
    typedef rocfft_fp16 type;
};

template <class T>
using real_type_t = typename real_type<T>::type;

template <class T>
struct complex_type;

template <>
struct complex_type<float>
{
    typedef rocfft_complex<float> type;
};

template <>
struct complex_type<double>
{
    typedef rocfft_complex<double> type;
};

template <class T>
using complex_type_t = typename complex_type<T>::type;

/// example of using complex_type_t:
// complex_type_t<float> float_complex_val;
// complex_type_t<double> double_complex_val;

template <typename T>
__device__ T TWLstep1(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    return result;
}

template <typename T>
__device__ T TWLstep2(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
               (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    return result;
}

template <typename T>
__device__ T TWLstep3(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
               (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
               (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    return result;
}

template <typename T>
__device__ T TWLstep4(const T* twiddles, size_t u)
{
    size_t j      = u & 255;
    T      result = twiddles[j];
    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[256 + j].x - result.y * twiddles[256 + j].y),
               (result.y * twiddles[256 + j].x + result.x * twiddles[256 + j].y));
    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[512 + j].x - result.y * twiddles[512 + j].y),
               (result.y * twiddles[512 + j].x + result.x * twiddles[512 + j].y));
    u >>= 8;
    j      = u & 255;
    result = T((result.x * twiddles[768 + j].x - result.y * twiddles[768 + j].y),
               (result.y * twiddles[768 + j].x + result.x * twiddles[768 + j].y));
    return result;
}

#define TWIDDLE_STEP_MUL_FWD(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T              W = TWFUNC(TWIDDLES, INDEX);        \
        real_type_t<T> TR, TI;                             \
        TR    = (W.x * REG.x) - (W.y * REG.y);             \
        TI    = (W.y * REG.x) + (W.x * REG.y);             \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#define TWIDDLE_STEP_MUL_INV(TWFUNC, TWIDDLES, INDEX, REG) \
    {                                                      \
        T              W = TWFUNC(TWIDDLES, INDEX);        \
        real_type_t<T> TR, TI;                             \
        TR    = (W.x * REG.x) + (W.y * REG.y);             \
        TI    = -(W.y * REG.x) + (W.x * REG.y);            \
        REG.x = TR;                                        \
        REG.y = TI;                                        \
    }

#endif // COMMON_H
