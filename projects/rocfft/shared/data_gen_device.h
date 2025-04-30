// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DATA_GEN_DEVICE_H
#define DATA_GEN_DEVICE_H

// rocRAND can generate warnings if inline asm is not available for
// some architectures.  data generation isn't performance-critical,
// so just disable inline asm to prevent the warnings.
#define ROCRAND_DISABLE_INLINE_ASM

#include "../shared/arithmetic.h"
#include "../shared/device_properties.h"
#include "../shared/gpubuf.h"
#include "../shared/increment.h"
#include "../shared/rocfft_complex.h"
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <limits>
#include <vector>

static const unsigned int DATA_GEN_THREADS    = 8;
static const unsigned int DATA_GEN_GRID_Y_MAX = 64;

template <typename Tcomplex>
__device__ static void conjugate(const size_t pos, const size_t cpos, Tcomplex* x)
{
    x[pos].x = x[cpos].x;
    x[pos].y = -x[cpos].y;
}

template <typename Tfloat>
__device__ static void conjugate(const size_t pos, const size_t cpos, Tfloat* xreal, Tfloat* ximag)
{
    xreal[pos] = xreal[cpos];
    ximag[pos] = -ximag[cpos];
}

template <typename Tcomplex>
__device__ static void set_imag_zero(const size_t pos, Tcomplex* x)
{
    x[pos].y = 0.0;
}

template <typename Tfloat>
__device__ static void set_imag_zero(const size_t pos, Tfloat* xreal, Tfloat* ximag)
{
    ximag[pos] = 0.0;
}

// For complex-to-real transforms, the input data must be Hermitiam-symmetric.
// That is, u_k is the complex conjugate of u_{-k}, where k is the wavevector in Fourier
// space.  For multi-dimensional data, this means that we only need to store a bit more
// than half of the complex values; the rest are redundant.  However, there are still
// some restrictions:
// * the origin and Nyquist value(s) must be real-valued
// * some of the remaining values are still redundant, and you might get different results
//   than you expect if the values don't agree.

template <typename Tcomplex>
__global__ static void impose_hermitian_symmetry_interleaved_1D_kernel(Tcomplex*    x,
                                                                       const size_t Nx,
                                                                       const size_t xstride,
                                                                       const size_t dist,
                                                                       const size_t batch_total,
                                                                       const bool   Nxeven)
{
    auto id_batch = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    static_assert(sizeof(id_batch) == sizeof(size_t));

    if(id_batch < batch_total)
    {
        id_batch *= dist;

        set_imag_zero(id_batch, x);

        if(Nxeven)
            set_imag_zero(id_batch + (Nx / 2) * xstride, x);
    }
}

template <typename Tfloat>
__global__ static void impose_hermitian_symmetry_planar_1D_kernel(Tfloat*      xreal,
                                                                  Tfloat*      ximag,
                                                                  const size_t Nx,
                                                                  const size_t xstride,
                                                                  const size_t dist,
                                                                  const size_t batch_total,
                                                                  const bool   Nxeven)
{
    auto id_batch = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    static_assert(sizeof(id_batch) == sizeof(size_t));

    if(id_batch < batch_total)
    {
        id_batch *= dist;

        set_imag_zero(id_batch, xreal, ximag);

        if(Nxeven)
            set_imag_zero(id_batch + (Nx / 2) * xstride, xreal, ximag);
    }
}

template <typename Tcomplex>
__global__ static void impose_hermitian_symmetry_interleaved_2D_kernel(Tcomplex*    x,
                                                                       const size_t Nx,
                                                                       const size_t Ny,
                                                                       const size_t xstride,
                                                                       const size_t ystride,
                                                                       const size_t dist,
                                                                       const size_t batch_total,
                                                                       const size_t x_total,
                                                                       const bool   Nxeven,
                                                                       const bool   Nyeven)
{
    auto       id_batch = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const auto id_x     = static_cast<size_t>(threadIdx.y) + blockIdx.y * blockDim.y;
    static_assert(sizeof(id_batch) == sizeof(size_t));
    static_assert(sizeof(id_x) == sizeof(size_t));

    if(id_batch < batch_total)
    {
        id_batch *= dist;

        if(id_x == 0)
            set_imag_zero(id_batch, x);

        if(id_x == 0 && Nxeven)
            set_imag_zero(id_batch + (Nx / 2) * xstride, x);

        if(id_x == 0 && Nyeven)
            set_imag_zero(id_batch + ystride * (Ny / 2), x);

        if(id_x == 0 && Nxeven && Nyeven)
            set_imag_zero(id_batch + xstride * (Nx / 2) + ystride * (Ny / 2), x);

        if(id_x < x_total)
        {
            conjugate(id_batch + xstride * (Nx - (id_x + 1)), id_batch + xstride * (id_x + 1), x);

            if(Nyeven)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny / 2),
                          id_batch + xstride * (id_x + 1) + ystride * (Ny / 2),
                          x);
        }
    }
}

template <typename Tfloat>
__global__ static void impose_hermitian_symmetry_planar_2D_kernel(Tfloat*      xreal,
                                                                  Tfloat*      ximag,
                                                                  const size_t Nx,
                                                                  const size_t Ny,
                                                                  const size_t xstride,
                                                                  const size_t ystride,
                                                                  const size_t dist,
                                                                  const size_t batch_total,
                                                                  const size_t x_total,
                                                                  const bool   Nxeven,
                                                                  const bool   Nyeven)
{
    auto       id_batch = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const auto id_x     = static_cast<size_t>(threadIdx.y) + blockIdx.y * blockDim.y;
    static_assert(sizeof(id_batch) == sizeof(size_t));
    static_assert(sizeof(id_x) == sizeof(size_t));

    if(id_batch < batch_total)
    {
        id_batch *= dist;

        if(id_x == 0)
            set_imag_zero(id_batch, xreal, ximag);

        if(id_x == 0 && Nxeven)
            set_imag_zero(id_batch + (Nx / 2) * xstride, xreal, ximag);

        if(id_x == 0 && Nyeven)
            set_imag_zero(id_batch + ystride * (Ny / 2), xreal, ximag);

        if(id_x == 0 && Nxeven && Nyeven)
            set_imag_zero(id_batch + xstride * (Nx / 2) + ystride * (Ny / 2), xreal, ximag);

        if(id_x < x_total)
        {
            conjugate(id_batch + xstride * (Nx - (id_x + 1)),
                      id_batch + xstride * (id_x + 1),
                      xreal,
                      ximag);

            if(Nyeven)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny / 2),
                          id_batch + xstride * (id_x + 1) + ystride * (Ny / 2),
                          xreal,
                          ximag);
        }
    }
}

template <typename Tcomplex>
__global__ static void impose_hermitian_symmetry_interleaved_3D_kernel(Tcomplex*    x,
                                                                       const size_t Nx,
                                                                       const size_t Ny,
                                                                       const size_t Nz,
                                                                       const size_t xstride,
                                                                       const size_t ystride,
                                                                       const size_t zstride,
                                                                       const size_t dist,
                                                                       const size_t batch_total,
                                                                       const size_t x_total,
                                                                       const size_t y_total,
                                                                       const size_t y_total_half,
                                                                       const bool   Nxeven,
                                                                       const bool   Nyeven,
                                                                       const bool   Nzeven)
{
    auto       id_batch = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const auto id_x     = static_cast<size_t>(threadIdx.y) + blockIdx.y * blockDim.y;
    const auto id_y     = static_cast<size_t>(threadIdx.z) + blockIdx.z * blockDim.z;
    static_assert(sizeof(id_batch) == sizeof(size_t));
    static_assert(sizeof(id_x) == sizeof(size_t));
    static_assert(sizeof(id_y) == sizeof(size_t));

    if(id_batch < batch_total)
    {
        auto id_x_y_zero = (id_x == 0 && id_y == 0);

        id_batch *= dist;

        if(id_x_y_zero)
            set_imag_zero(id_batch, x);

        if(Nxeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2), x);

        if(Nyeven && id_x_y_zero)
            set_imag_zero(id_batch + ystride * (Ny / 2), x);

        if(Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + zstride * (Nz / 2), x);

        if(Nxeven && Nyeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2) + ystride * (Ny / 2), x);

        if(Nxeven && Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2) + zstride * (Nz / 2), x);

        if(Nyeven && Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + ystride * (Ny / 2) + zstride * (Nz / 2), x);

        if(Nxeven && Nyeven && Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2) + ystride * (Ny / 2) + zstride * (Nz / 2),
                          x);

        if(id_x == 0 && id_y < y_total_half)
            conjugate(id_batch + ystride * (Ny - (id_y + 1)), id_batch + ystride * (id_y + 1), x);

        if(Nxeven && id_x == 0 && id_y < y_total_half)
            conjugate(id_batch + xstride * (Nx / 2) + ystride * (Ny - (id_y + 1)),
                      id_batch + xstride * (Nx / 2) + ystride * (id_y + 1),
                      x);

        if(id_x < x_total && id_y == 0)
            conjugate(id_batch + xstride * (Nx - (id_x + 1)), id_batch + xstride * (id_x + 1), x);

        if(Nyeven && id_x < x_total && id_y == 0)
            conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny / 2),
                      id_batch + xstride * (id_x + 1) + ystride * (Ny / 2),
                      x);

        if(id_x < x_total && id_y < y_total)
            conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny - (id_y + 1)),
                      id_batch + xstride * (id_x + 1) + ystride * (id_y + 1),
                      x);

        if(Nzeven)
        {
            if(id_x < x_total && id_y == 0)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + zstride * (Nz / 2),
                          id_batch + xstride * (id_x + 1) + zstride * (Nz / 2),
                          x);

            if(Nyeven && id_x < x_total && id_y == 0)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + zstride * (Nz / 2),
                          id_batch + xstride * (id_x + 1) + zstride * (Nz / 2),
                          x);

            if(id_x == 0 && id_y < y_total_half)
                conjugate(id_batch + ystride * (Ny - (id_y + 1)) + zstride * (Nz / 2),
                          id_batch + ystride * (id_y + 1) + zstride * (Nz / 2),
                          x);

            if(Nxeven && id_x == 0 && id_y < y_total_half)
                conjugate(id_batch + xstride * (Nx / 2) + ystride * (Ny - (id_y + 1))
                              + zstride * (Nz / 2),
                          id_batch + xstride * (Nx / 2) + ystride * (id_y + 1) + zstride * (Nz / 2),
                          x);

            if(id_x < x_total && id_y < y_total)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny - (id_y + 1))
                              + zstride * (Nz / 2),
                          id_batch + xstride * (id_x + 1) + ystride * (id_y + 1)
                              + zstride * (Nz / 2),
                          x);
        }
    }
}

template <typename Tfloat>
__global__ static void impose_hermitian_symmetry_planar_3D_kernel(Tfloat*      xreal,
                                                                  Tfloat*      ximag,
                                                                  const size_t Nx,
                                                                  const size_t Ny,
                                                                  const size_t Nz,
                                                                  const size_t xstride,
                                                                  const size_t ystride,
                                                                  const size_t zstride,
                                                                  const size_t dist,
                                                                  const size_t batch_total,
                                                                  const size_t x_total,
                                                                  const size_t y_total,
                                                                  const size_t y_total_half,
                                                                  const bool   Nxeven,
                                                                  const bool   Nyeven,
                                                                  const bool   Nzeven)
{
    auto       id_batch = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x;
    const auto id_x     = static_cast<size_t>(threadIdx.y) + blockIdx.y * blockDim.y;
    const auto id_y     = static_cast<size_t>(threadIdx.z) + blockIdx.z * blockDim.z;
    static_assert(sizeof(id_batch) == sizeof(size_t));
    static_assert(sizeof(id_x) == sizeof(size_t));
    static_assert(sizeof(id_y) == sizeof(size_t));

    if(id_batch < batch_total)
    {
        auto id_x_y_zero = (id_x == 0 && id_y == 0);

        id_batch *= dist;

        if(id_x_y_zero)
            set_imag_zero(id_batch, xreal, ximag);

        if(Nxeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2), xreal, ximag);

        if(Nyeven && id_x_y_zero)
            set_imag_zero(id_batch + ystride * (Ny / 2), xreal, ximag);

        if(Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + zstride * (Nz / 2), xreal, ximag);

        if(Nxeven && Nyeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2) + ystride * (Ny / 2), xreal, ximag);

        if(Nxeven && Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2) + zstride * (Nz / 2), xreal, ximag);

        if(Nyeven && Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + ystride * (Ny / 2) + zstride * (Nz / 2), xreal, ximag);

        if(Nxeven && Nyeven && Nzeven && id_x_y_zero)
            set_imag_zero(id_batch + xstride * (Nx / 2) + ystride * (Ny / 2) + zstride * (Nz / 2),
                          xreal,
                          ximag);

        if(id_x == 0 && id_y < y_total_half)
            conjugate(id_batch + ystride * (Ny - (id_y + 1)),
                      id_batch + ystride * (id_y + 1),
                      xreal,
                      ximag);

        if(Nxeven && id_x == 0 && id_y < y_total_half)
            conjugate(id_batch + xstride * (Nx / 2) + ystride * (Ny - (id_y + 1)),
                      id_batch + xstride * (Nx / 2) + ystride * (id_y + 1),
                      xreal,
                      ximag);

        if(id_x < x_total && id_y == 0)
            conjugate(id_batch + xstride * (Nx - (id_x + 1)),
                      id_batch + xstride * (id_x + 1),
                      xreal,
                      ximag);

        if(Nyeven && id_x < x_total && id_y == 0)
            conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny / 2),
                      id_batch + xstride * (id_x + 1) + ystride * (Ny / 2),
                      xreal,
                      ximag);

        if(id_x < x_total && id_y < y_total)
            conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny - (id_y + 1)),
                      id_batch + xstride * (id_x + 1) + ystride * (id_y + 1),
                      xreal,
                      ximag);

        if(Nzeven)
        {
            if(id_x < x_total && id_y == 0)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + zstride * (Nz / 2),
                          id_batch + xstride * (id_x + 1) + zstride * (Nz / 2),
                          xreal,
                          ximag);

            if(Nyeven && id_x < x_total && id_y == 0)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + zstride * (Nz / 2),
                          id_batch + xstride * (id_x + 1) + zstride * (Nz / 2),
                          xreal,
                          ximag);

            if(id_x == 0 && id_y < y_total_half)
                conjugate(id_batch + ystride * (Ny - (id_y + 1)) + zstride * (Nz / 2),
                          id_batch + ystride * (id_y + 1) + zstride * (Nz / 2),
                          xreal,
                          ximag);

            if(Nxeven && id_x == 0 && id_y < y_total_half)
                conjugate(id_batch + xstride * (Nx / 2) + ystride * (Ny - (id_y + 1))
                              + zstride * (Nz / 2),
                          id_batch + xstride * (Nx / 2) + ystride * (id_y + 1) + zstride * (Nz / 2),
                          xreal,
                          ximag);

            if(id_x < x_total && id_y < y_total)
                conjugate(id_batch + xstride * (Nx - (id_x + 1)) + ystride * (Ny - (id_y + 1))
                              + zstride * (Nz / 2),
                          id_batch + xstride * (id_x + 1) + ystride * (id_y + 1)
                              + zstride * (Nz / 2),
                          xreal,
                          ximag);
        }
    }
}

#ifdef USE_HIPRAND

#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

template <typename T>
struct input_val_1D
{
    T val1 = 0;

    __host__ __device__ input_val_1D operator+(const input_val_1D other)
    {
        return {val1 + other.val1};
    }
};

template <typename T>
struct input_val_2D
{
    T val1 = 0;
    T val2 = 0;

    __host__ __device__ input_val_2D operator+(const input_val_2D other)
    {
        return {val1 + other.val1, val2 + other.val2};
    }
};

template <typename T>
struct input_val_3D
{
    T val1 = 0;
    T val2 = 0;
    T val3 = 0;

    __host__ __device__ input_val_3D operator+(const input_val_3D other)
    {
        return {val1 + other.val1, val2 + other.val2, val3 + other.val3};
    }
};

template <typename T>
static input_val_1D<T> get_input_val(const T& val)
{
    return input_val_1D<T>{val};
}

template <typename T>
static input_val_2D<T> get_input_val(const std::tuple<T, T>& val)
{
    return input_val_2D<T>{std::get<0>(val), std::get<1>(val)};
}

template <typename T>
static input_val_3D<T> get_input_val(const std::tuple<T, T, T>& val)
{
    return input_val_3D<T>{std::get<0>(val), std::get<1>(val), std::get<2>(val)};
}

template <typename T>
__device__ static size_t
    compute_index(const input_val_1D<T>& length, const input_val_1D<T>& stride, size_t base)
{
    return (length.val1 * stride.val1) + base;
}

template <typename T>
__device__ static size_t
    compute_index(const input_val_2D<T>& length, const input_val_2D<T>& stride, size_t base)
{
    return (length.val1 * stride.val1) + (length.val2 * stride.val2) + base;
}

template <typename T>
__device__ static size_t
    compute_index(const input_val_3D<T>& length, const input_val_3D<T>& stride, size_t base)
{
    return (length.val1 * stride.val1) + (length.val2 * stride.val2) + (length.val3 * stride.val3)
           + base;
}

template <typename T>
static inline input_val_1D<T> make_unit_stride(const input_val_1D<T>& whole_length)
{
    return input_val_1D<T>{1};
}

template <typename T>
static inline input_val_2D<T> make_unit_stride(const input_val_2D<T>& whole_length)
{
    return input_val_2D<T>{1, whole_length.val1};
}

template <typename T>
static inline input_val_3D<T> make_unit_stride(const input_val_3D<T>& whole_length)
{
    return input_val_3D<T>{1, whole_length.val1, whole_length.val1 * whole_length.val2};
}

template <typename T>
__device__ static input_val_1D<T> get_length(const size_t i, const input_val_1D<T>& whole_length)
{
    auto xlen = whole_length.val1;

    auto xidx = i % xlen;

    return input_val_1D<T>{xidx};
}

template <typename T>
__device__ static input_val_2D<T> get_length(const size_t i, const input_val_2D<T>& whole_length)
{
    auto xlen = whole_length.val1;
    auto ylen = whole_length.val2;

    auto xidx = i % xlen;
    auto yidx = i / xlen % ylen;

    return input_val_2D<T>{xidx, yidx};
}

template <typename T>
__device__ static input_val_3D<T> get_length(const size_t i, const input_val_3D<T>& whole_length)
{
    auto xlen = whole_length.val1;
    auto ylen = whole_length.val2;
    auto zlen = whole_length.val3;

    auto xidx = i % xlen;
    auto yidx = i / xlen % ylen;
    auto zidx = i / xlen / ylen % zlen;

    return input_val_3D<T>{xidx, yidx, zidx};
}

template <typename T>
__device__ static size_t get_batch(const size_t i, const input_val_1D<T>& whole_length)
{
    auto xlen = whole_length.val1;

    auto yidx = i / xlen;

    return yidx;
}

template <typename T>
__device__ static size_t get_batch(const size_t i, const input_val_2D<T>& whole_length)
{
    auto xlen = whole_length.val1;
    auto ylen = whole_length.val2;

    auto zidx = i / xlen / ylen;

    return zidx;
}

template <typename T>
__device__ static size_t get_batch(const size_t i, const input_val_3D<T>& length)
{
    auto xlen = length.val1;
    auto ylen = length.val2;
    auto zlen = length.val3;

    auto widx = i / xlen / ylen / zlen;

    return widx;
}

__device__ static double make_random_val(hiprandStatePhilox4_32_10* gen_state, double offset)
{
    return hiprand_uniform_double(gen_state) + offset;
}

__device__ static float make_random_val(hiprandStatePhilox4_32_10* gen_state, float offset)
{
    return hiprand_uniform(gen_state) + offset;
}

__device__ static rocfft_fp16 make_random_val(hiprandStatePhilox4_32_10* gen_state,
                                              rocfft_fp16                offset)
{
    return static_cast<rocfft_fp16>(hiprand_uniform(gen_state)) + offset;
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_random_interleaved_data_kernel(const Tint             whole_length,
                                            const Tint             zero_length,
                                            const size_t           idist,
                                            const size_t           isize,
                                            const Tint             istride,
                                            rocfft_complex<Treal>* data,
                                            const Tint             field_lower,
                                            const size_t           field_lower_batch,
                                            const Tint             field_contig_stride,
                                            const size_t           field_contig_dist)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        auto i_length = get_length(i, whole_length);
        auto i_batch  = get_batch(i, whole_length);

        // brick index to write to
        auto write_idx = compute_index(i_length, istride, i_batch * idist);
        // logical index in the field
        auto logical_idx = compute_index(i_length + field_lower,
                                         field_contig_stride,
                                         (i_batch + field_lower_batch) * field_contig_dist);
        auto seed        = compute_index(zero_length, istride, i_batch * field_contig_dist);

        hiprandStatePhilox4_32_10 gen_state;
        hiprand_init(seed, logical_idx, 0, &gen_state);

        data[write_idx].x = make_random_val(&gen_state, static_cast<Treal>(-0.5));
        data[write_idx].y = make_random_val(&gen_state, static_cast<Treal>(-0.5));
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_interleaved_data_kernel(const Tint             whole_length,
                                     const size_t           idist,
                                     const size_t           isize,
                                     const Tint             istride,
                                     const Tint             ustride,
                                     const Treal            inv_scale,
                                     rocfft_complex<Treal>* data)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        const auto i_length = get_length(i, whole_length);
        const auto i_batch  = get_batch(i, whole_length);
        const auto i_base   = i_batch * idist;

        const auto val = static_cast<Treal>(-0.5)
                         + static_cast<Treal>(
                               static_cast<unsigned long long>(compute_index(i_length, ustride, 0)))
                               * inv_scale;

        const auto idx = compute_index(i_length, istride, i_base);

        data[idx].x = val;
        data[idx].y = val;
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_random_planar_data_kernel(const Tint   whole_length,
                                       const Tint   zero_length,
                                       const size_t idist,
                                       const size_t isize,
                                       const Tint   istride,
                                       Treal*       real_data,
                                       Treal*       imag_data,
                                       const Tint   field_lower,
                                       const size_t field_lower_batch,
                                       const Tint   field_contig_stride,
                                       const size_t field_contig_dist)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        auto i_length = get_length(i, whole_length);
        auto i_batch  = get_batch(i, whole_length);

        // brick index to write to
        auto write_idx = compute_index(i_length, istride, i_batch * idist);
        // logical index in the field
        auto logical_idx = compute_index(i_length + field_lower,
                                         field_contig_stride,
                                         (i_batch + field_lower_batch) * field_contig_dist);
        auto seed        = compute_index(zero_length, istride, i_batch * field_contig_dist);

        hiprandStatePhilox4_32_10 gen_state;
        hiprand_init(seed, logical_idx, 0, &gen_state);

        real_data[write_idx] = make_random_val(&gen_state, static_cast<Treal>(-0.5));
        imag_data[write_idx] = make_random_val(&gen_state, static_cast<Treal>(-0.5));
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_planar_data_kernel(const Tint   whole_length,
                                const size_t idist,
                                const size_t isize,
                                const Tint   istride,
                                const Tint   ustride,
                                const Treal  inv_scale,
                                Treal*       real_data,
                                Treal*       imag_data)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        const auto i_length = get_length(i, whole_length);
        const auto i_batch  = get_batch(i, whole_length);
        const auto i_base   = i_batch * idist;

        const auto val = static_cast<Treal>(-0.5)
                         + static_cast<Treal>(
                               static_cast<unsigned long long>(compute_index(i_length, ustride, 0)))
                               * inv_scale;

        const auto idx = compute_index(i_length, istride, i_base);

        real_data[idx] = val;
        imag_data[idx] = val;
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_random_real_data_kernel(const Tint   whole_length,
                                     const Tint   zero_length,
                                     const size_t idist,
                                     const size_t isize,
                                     const Tint   istride,
                                     Treal*       data,
                                     const Tint   field_lower,
                                     const size_t field_lower_batch,
                                     const Tint   field_contig_stride,
                                     const size_t field_contig_dist)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        auto i_length = get_length(i, whole_length);
        auto i_batch  = get_batch(i, whole_length);

        // brick index to write to
        auto write_idx = compute_index(i_length, istride, i_batch * idist);
        // logical index in the field
        auto logical_idx = compute_index(i_length + field_lower,
                                         field_contig_stride,
                                         (i_batch + field_lower_batch) * field_contig_dist);
        auto seed        = compute_index(zero_length, istride, i_batch * field_contig_dist);

        hiprandStatePhilox4_32_10 gen_state;
        hiprand_init(seed, logical_idx, 0, &gen_state);

        data[write_idx] = make_random_val(&gen_state, static_cast<Treal>(-0.5));
    }
}

template <typename Tint, typename Treal>
__global__ static void __launch_bounds__(DATA_GEN_THREADS)
    generate_real_data_kernel(const Tint   whole_length,
                              const size_t idist,
                              const size_t isize,
                              const Tint   istride,
                              const Tint   ustride,
                              const Treal  inv_scale,
                              Treal*       data,
                              const Tint   field_lower         = {},
                              const size_t field_lower_batch   = 0,
                              const Tint   field_contig_stride = {},
                              const size_t field_contig_dist   = 0)
{
    auto const i = static_cast<size_t>(threadIdx.x) + blockIdx.x * blockDim.x
                   + blockIdx.y * gridDim.x * DATA_GEN_THREADS;
    static_assert(sizeof(i) >= sizeof(isize));
    if(i < isize)
    {
        const auto i_length = get_length(i, whole_length);
        const auto i_batch  = get_batch(i, whole_length);
        const auto i_base   = i_batch * idist;

        const auto val = static_cast<Treal>(-0.5)
                         + static_cast<Treal>(
                               static_cast<unsigned long long>(compute_index(i_length, ustride, 0)))
                               * inv_scale;

        const auto idx = compute_index(i_length, istride, i_base);

        data[idx] = val;
    }
}

// get grid dimensions for data gen kernel
static dim3 generate_data_gridDim(const size_t isize)
{
    auto blockSize = DATA_GEN_THREADS;
    // total number of blocks needed in the grid
    auto numBlocks_setup = DivRoundingUp<size_t>(isize, blockSize);

    // Total work items per dimension in the grid is counted in
    // uint32_t.  Since each thread initializes one element, very
    // large amounts of data will overflow this total size if we do
    // all this work in one grid dimension, causing launch failure.
    //
    // CUDA also generally allows for effectively unlimited grid X
    // dim, but Y and Z are more limited.
    auto gridDim_y = std::min<unsigned int>(DATA_GEN_GRID_Y_MAX, numBlocks_setup);
    auto gridDim_x = DivRoundingUp<unsigned int>(numBlocks_setup, DATA_GEN_GRID_Y_MAX);
    return {gridDim_x, gridDim_y};
}

// get grid dimensions for hermitian symmetrizer kernel
static dim3 generate_hermitian_gridDim(const std::vector<size_t>& length,
                                       const size_t               batch,
                                       const size_t               blockSize)
{
    dim3 gridDim;

    switch(length.size())
    {
    case 1:
        gridDim = dim3(DivRoundingUp<size_t>(batch, blockSize));
        break;
    case 2:
        gridDim = dim3(DivRoundingUp<size_t>(batch, blockSize),
                       DivRoundingUp<size_t>((length[0] + 1) / 2 - 1, blockSize));
        break;
    case 3:
        gridDim = dim3(DivRoundingUp<size_t>(batch, blockSize),
                       DivRoundingUp<size_t>((length[0] + 1) / 2 - 1, blockSize),
                       DivRoundingUp<size_t>(length[1] - 1, blockSize));
        break;
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }

    return gridDim;
}

static dim3 generate_blockDim(const std::vector<size_t>& length, const size_t blockSize)
{
    dim3 blockDim;

    switch(length.size())
    {
    case 1:
        blockDim = dim3(blockSize);
        break;
    case 2:
        blockDim = dim3(blockSize, blockSize);
        break;
    case 3:
        blockDim = dim3(blockSize, blockSize, blockSize);
        break;
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }

    return blockDim;
}

template <typename Tint, typename Treal>
static void generate_random_interleaved_data(const Tint&            whole_length,
                                             const size_t           idist,
                                             const size_t           isize,
                                             const Tint&            whole_stride,
                                             rocfft_complex<Treal>* input_data,
                                             const hipDeviceProp_t& deviceProp,
                                             const Tint&            field_lower,
                                             const size_t           field_lower_batch,
                                             const Tint&            field_contig_stride,
                                             const size_t           field_contig_dist)
{
    auto                         input_length = get_input_val(whole_length);
    const decltype(input_length) zero_length;
    auto                         input_stride = get_input_val(whole_stride);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check("generate_random_interleaved_data_kernel", gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generate_random_interleaved_data_kernel<decltype(input_length), Treal>),
        gridDim,
        blockDim,
        0, // sharedMemBytes
        0, // stream
        input_length,
        zero_length,
        idist,
        isize,
        input_stride,
        input_data,
        get_input_val(field_lower),
        field_lower_batch,
        get_input_val(field_contig_stride),
        field_contig_dist);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_random_interleaved_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
static void generate_interleaved_data(const Tint&            whole_length,
                                      const size_t           idist,
                                      const size_t           isize,
                                      const Tint&            whole_stride,
                                      const size_t           nbatch,
                                      rocfft_complex<Treal>* input_data,
                                      const hipDeviceProp_t& deviceProp)
{
    const auto input_length = get_input_val(whole_length);
    const auto input_stride = get_input_val(whole_stride);
    const auto unit_stride  = make_unit_stride(input_length);

    const auto inv_scale
        = static_cast<Treal>(1.0)
          / static_cast<Treal>(static_cast<unsigned long long>(isize) / nbatch - 1);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check("generate_interleaved_data_kernel", gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generate_interleaved_data_kernel<decltype(input_length), Treal>),
        gridDim,
        blockDim,
        0, // sharedMemBytes
        0, // stream
        input_length,
        idist,
        isize,
        input_stride,
        unit_stride,
        inv_scale,
        input_data);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_interleaved_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
static void generate_random_planar_data(const Tint&            whole_length,
                                        const size_t           idist,
                                        const size_t           isize,
                                        const Tint&            whole_stride,
                                        Treal*                 real_data,
                                        Treal*                 imag_data,
                                        const hipDeviceProp_t& deviceProp,
                                        const Tint&            field_lower,
                                        const size_t           field_lower_batch,
                                        const Tint&            field_contig_stride,
                                        const size_t           field_contig_dist)
{
    const auto                   input_length = get_input_val(whole_length);
    const decltype(input_length) zero_length;
    const auto                   input_stride = get_input_val(whole_stride);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check("generate_random_planar_data_kernel", gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generate_random_planar_data_kernel<decltype(input_length), Treal>),
        gridDim,
        blockDim,
        0, // sharedMemBytes
        0, // stream
        input_length,
        zero_length,
        idist,
        isize,
        input_stride,
        real_data,
        imag_data,
        get_input_val(field_lower),
        field_lower_batch,
        get_input_val(field_contig_stride),
        field_contig_dist);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_random_planar_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
static void generate_planar_data(const Tint&            whole_length,
                                 const size_t           idist,
                                 const size_t           isize,
                                 const Tint&            whole_stride,
                                 const size_t           nbatch,
                                 Treal*                 real_data,
                                 Treal*                 imag_data,
                                 const hipDeviceProp_t& deviceProp)
{
    const auto input_length = get_input_val(whole_length);
    const auto input_stride = get_input_val(whole_stride);
    const auto unit_stride  = make_unit_stride(input_length);

    const auto inv_scale
        = static_cast<Treal>(1.0)
          / static_cast<Treal>(static_cast<unsigned long long>(isize) / nbatch - 1);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check("generate_planar_data_kernel", gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(generate_planar_data_kernel<decltype(input_length), Treal>),
                       gridDim,
                       blockDim,
                       0, // sharedMemBytes
                       0, // stream
                       input_length,
                       idist,
                       isize,
                       input_stride,
                       unit_stride,
                       inv_scale,
                       real_data,
                       imag_data);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_planar_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
static void generate_random_real_data(const Tint&            whole_length,
                                      const size_t           idist,
                                      const size_t           isize,
                                      const Tint&            whole_stride,
                                      Treal*                 input_data,
                                      const hipDeviceProp_t& deviceProp,
                                      const Tint             field_lower,
                                      const size_t           field_lower_batch,
                                      const Tint             field_contig_stride,
                                      const size_t           field_contig_dist)
{
    const auto                   input_length = get_input_val(whole_length);
    const decltype(input_length) zero_length;
    const auto                   input_stride = get_input_val(whole_stride);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check("generate_random_real_data_kernel", gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(generate_random_real_data_kernel<decltype(input_length), Treal>),
        gridDim,
        blockDim,
        0, // sharedMemBytes
        0, // stream
        input_length,
        zero_length,
        idist,
        isize,
        input_stride,
        input_data,
        get_input_val(field_lower),
        field_lower_batch,
        get_input_val(field_contig_stride),
        field_contig_dist);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_random_real_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tint, typename Treal>
static void generate_real_data(const Tint&            whole_length,
                               const size_t           idist,
                               const size_t           isize,
                               const Tint&            whole_stride,
                               const size_t           nbatch,
                               Treal*                 input_data,
                               const hipDeviceProp_t& deviceProp)
{
    const auto input_length = get_input_val(whole_length);
    const auto input_stride = get_input_val(whole_stride);
    const auto unit_stride  = make_unit_stride(input_length);

    const auto inv_scale
        = static_cast<Treal>(1.0)
          / static_cast<Treal>(static_cast<unsigned long long>(isize) / nbatch - 1);

    dim3 gridDim = generate_data_gridDim(isize);
    dim3 blockDim{DATA_GEN_THREADS};

    launch_limits_check("generate_real_data_kernel", gridDim, blockDim, deviceProp);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(generate_real_data_kernel<decltype(input_length), Treal>),
                       gridDim,
                       blockDim,
                       0, // sharedMemBytes
                       0, // stream
                       input_length,
                       idist,
                       isize,
                       input_stride,
                       unit_stride,
                       inv_scale,
                       input_data);
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("generate_real_data_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tcomplex>
static void impose_hermitian_symmetry_interleaved(const std::vector<size_t>& length,
                                                  const std::vector<size_t>& ilength,
                                                  const std::vector<size_t>& stride,
                                                  const size_t               dist,
                                                  const size_t               batch,
                                                  Tcomplex*                  input_data,
                                                  const hipDeviceProp_t&     deviceProp)
{
    auto blockSize = DATA_GEN_THREADS;
    auto blockDim  = generate_blockDim(length, blockSize);
    auto gridDim   = generate_hermitian_gridDim(length, batch, blockSize);

    switch(length.size())
    {
    case 1:
    {
        launch_limits_check(
            "impose_hermitian_symmetry_interleaved_1D_kernel", gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_1D_kernel<Tcomplex>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data,
                           length[0],
                           stride[0],
                           dist,
                           batch,
                           length[0] % 2 == 0);

        break;
    }
    case 2:
    {
        launch_limits_check(
            "impose_hermitian_symmetry_interleaved_2D_kernel", gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_2D_kernel<Tcomplex>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data,
                           length[0],
                           length[1],
                           stride[0],
                           stride[1],
                           dist,
                           batch,
                           (ilength[0] + 1) / 2 - 1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);

        break;
    }
    case 3:
    {
        launch_limits_check(
            "impose_hermitian_symmetry_interleaved_3D_kernel", gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_interleaved_3D_kernel<Tcomplex>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data,
                           length[0],
                           length[1],
                           length[2],
                           stride[0],
                           stride[1],
                           stride[2],
                           dist,
                           batch,
                           (ilength[0] + 1) / 2 - 1,
                           ilength[1] - 1,
                           (ilength[1] + 1) / 2 - 1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0,
                           length[2] % 2 == 0);
        break;
    }
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("impose_hermitian_symmetry_interleaved_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}

template <typename Tfloat>
static void impose_hermitian_symmetry_planar(const std::vector<size_t>& length,
                                             const std::vector<size_t>& ilength,
                                             const std::vector<size_t>& stride,
                                             const size_t               dist,
                                             const size_t               batch,
                                             Tfloat*                    input_data_real,
                                             Tfloat*                    input_data_imag,
                                             const hipDeviceProp_t&     deviceProp)
{
    auto blockSize = DATA_GEN_THREADS;
    auto blockDim  = generate_blockDim(length, blockSize);
    auto gridDim   = generate_hermitian_gridDim(length, batch, blockSize);

    switch(length.size())
    {
    case 1:
    {
        launch_limits_check(
            "impose_hermitian_symmetry_planar_1D_kernel", gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_planar_1D_kernel<Tfloat>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data_real,
                           input_data_imag,
                           length[0],
                           stride[0],
                           dist,
                           batch,
                           length[0] % 2 == 0);

        break;
    }
    case 2:
    {
        launch_limits_check(
            "impose_hermitian_symmetry_planar_2D_kernel", gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_planar_2D_kernel<Tfloat>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data_real,
                           input_data_imag,
                           length[0],
                           length[1],
                           stride[0],
                           stride[1],
                           dist,
                           batch,
                           (ilength[0] + 1) / 2 - 1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0);

        break;
    }
    case 3:
    {
        launch_limits_check(
            "impose_hermitian_symmetry_planar_3D_kernel", gridDim, blockDim, deviceProp);

        hipLaunchKernelGGL(impose_hermitian_symmetry_planar_3D_kernel<Tfloat>,
                           gridDim,
                           blockDim,
                           0,
                           0,
                           input_data_real,
                           input_data_imag,
                           length[0],
                           length[1],
                           length[2],
                           stride[0],
                           stride[1],
                           stride[2],
                           dist,
                           batch,
                           (ilength[0] + 1) / 2 - 1,
                           ilength[1] - 1,
                           (ilength[1] + 1) / 2 - 1,
                           length[0] % 2 == 0,
                           length[1] % 2 == 0,
                           length[2] % 2 == 0);
        break;
    }
    default:
        throw std::runtime_error("Invalid dimension for impose_hermitian_symmetry");
    }
    auto err = hipGetLastError();
    if(err != hipSuccess)
        throw std::runtime_error("impose_hermitian_symmetry_planar_kernel launch failure: "
                                 + std::string(hipGetErrorName(err)));
}
#endif // USE_HIPRAND
#endif // DATA_GEN_DEVICE_H
