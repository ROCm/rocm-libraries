// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/*! @file fft.hpp
 *  @brief Main entry point for the rocFFT device-side, header-only FFT library.
 *
 *  @details This header provides a cuFFTDx-style, device-callable FFT API. A
 *  transform is selected entirely at compile time through the
 *  ::rocfft::device::FFT class template, which maps a (length, precision,
 *  direction) triple onto a pre-generated Stockham implementation. No host-side
 *  plan creation or runtime dispatch is involved.
 *
 *  Example:
 *  @code
 *  #include <rocfft/device/fft.hpp>
 *
 *  using MyFFT = rocfft::device::FFT<64, rocfft::device::Precision::Single,
 *                                     rocfft::device::Direction::Forward>;
 *
 *  constexpr size_t smem = MyFFT::shared_memory_size;
 *  constexpr size_t ept  = MyFFT::elements_per_thread;
 *
 *  __device__ void my_fft(rocfft_complex<float>* data, char* smem) {
 *      MyFFT::execute(data, smem, threadIdx.x);
 *  }
 *  @endcode
 */

#pragma once

#ifndef ROCFFT_DEVICE_FFT_HPP
#define ROCFFT_DEVICE_FFT_HPP

#include "precision.hpp"
#include "direction.hpp"
#include "detail/complex.hpp"
#include "detail/radix.hpp"
#include "detail/dispatch.hpp"
#include "stockham/index.hpp"

namespace rocfft
{
namespace device
{

/*! @brief Internal implementation detail of ::rocfft::device::FFT.
 *  @details Holds the per-configuration `execute` specializations that drive the
 *  pre-generated Stockham device functions. Not part of the public API.
 */
namespace detail_fft
{

//! @brief Primary template — intentionally left undefined; specializations below.
template <unsigned Length, unsigned WGS, Precision P, Direction D>
struct FFTExecute;

// Macro to define a forward FFTExecute specialization.
#define ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(LENGTH, WGS, PREC_ENUM, PREC_TAG, SCALAR, MAX_RAD)    \
    template <>                                                                                     \
    struct FFTExecute<LENGTH, WGS, PREC_ENUM, Direction::Forward>                                   \
    {                                                                                               \
        using scalar_type = rocfft_complex<SCALAR>;                                                 \
                                                                                                    \
        static __device__ void execute(scalar_type* data, char* shared_scratch,                     \
                                       unsigned int thread)                                         \
        {                                                                                           \
            constexpr unsigned ept = LENGTH / WGS;                                                  \
            using namespace stockham::detail_l##LENGTH##_wgs##WGS##_##PREC_TAG##p;                  \
                                                                                                    \
            scalar_type  R[MAX_RAD];                                                                \
            auto*        lds_real    = reinterpret_cast<real_type_t<scalar_type>*>(shared_scratch);  \
            auto*        lds_complex = reinterpret_cast<scalar_type*>(shared_scratch);               \
            const unsigned stride_lds = LENGTH;                                                     \
            const unsigned offset_lds = 0;                                                          \
                                                                                                    \
            for(unsigned i = 0; i < ept; ++i)                                                       \
                lds_complex[offset_lds + thread + i * WGS] = data[i];                               \
            __syncthreads();                                                                        \
                                                                                                    \
            lds_to_reg_input_length##LENGTH##_device<scalar_type, SB_UNIT, true>(                   \
                R, lds_complex, stride_lds, offset_lds, thread, true);                              \
            forward_full_pass_length##LENGTH##_SBRR_device<scalar_type, false, SB_UNIT, true,       \
                                                           false>(                                  \
                R, lds_real, lds_complex, twiddle_storage, stride_lds, offset_lds, thread,     \
                true);                                                                              \
            lds_from_reg_output_length##LENGTH##_device<scalar_type, SB_UNIT, true>(                \
                R, lds_complex, stride_lds, offset_lds, thread, true);                              \
            __syncthreads();                                                                        \
                                                                                                    \
            for(unsigned i = 0; i < ept; ++i)                                                       \
                data[i] = lds_complex[offset_lds + thread + i * WGS];                               \
        }                                                                                           \
    }

// Macro to define an inverse FFTExecute specialization.
#define ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(LENGTH, WGS, PREC_ENUM, PREC_TAG, SCALAR, MAX_RAD)    \
    template <>                                                                                     \
    struct FFTExecute<LENGTH, WGS, PREC_ENUM, Direction::Inverse>                                   \
    {                                                                                               \
        using scalar_type = rocfft_complex<SCALAR>;                                                 \
                                                                                                    \
        static __device__ void execute(scalar_type* data, char* shared_scratch,                     \
                                       unsigned int thread)                                         \
        {                                                                                           \
            constexpr unsigned ept = LENGTH / WGS;                                                  \
            using namespace stockham::detail_l##LENGTH##_wgs##WGS##_##PREC_TAG##p;                  \
                                                                                                    \
            scalar_type  R[MAX_RAD];                                                                \
            auto*        lds_real    = reinterpret_cast<real_type_t<scalar_type>*>(shared_scratch);  \
            auto*        lds_complex = reinterpret_cast<scalar_type*>(shared_scratch);               \
            const unsigned stride_lds = LENGTH;                                                     \
            const unsigned offset_lds = 0;                                                          \
                                                                                                    \
            for(unsigned i = 0; i < ept; ++i)                                                       \
                lds_complex[offset_lds + thread + i * WGS] = data[i];                               \
            __syncthreads();                                                                        \
                                                                                                    \
            lds_to_reg_input_length##LENGTH##_device<scalar_type, SB_UNIT, true>(                   \
                R, lds_complex, stride_lds, offset_lds, thread, true);                              \
            inverse_full_pass_length##LENGTH##_SBRR_device<scalar_type, false, SB_UNIT, true,       \
                                                           false>(                                  \
                R, lds_real, lds_complex, twiddle_storage, stride_lds, offset_lds, thread,     \
                true);                                                                              \
            lds_from_reg_output_length##LENGTH##_device<scalar_type, SB_UNIT, true>(                \
                R, lds_complex, stride_lds, offset_lds, thread, true);                              \
            __syncthreads();                                                                        \
                                                                                                    \
            for(unsigned i = 0; i < ept; ++i)                                                       \
                data[i] = lds_complex[offset_lds + thread + i * WGS];                               \
        }                                                                                           \
    }

// Instantiate for all supported configurations.
// length 32: factors (4,8), max_radix=8
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(32, 8, Precision::Single, s, float, 8);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(32, 8, Precision::Single, s, float, 8);
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(32, 8, Precision::Double, d, double, 8);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(32, 8, Precision::Double, d, double, 8);

// length 64: factors (2,4,8), max_radix=8
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(64, 32, Precision::Single, s, float, 8);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(64, 32, Precision::Single, s, float, 8);
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(64, 32, Precision::Double, d, double, 8);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(64, 32, Precision::Double, d, double, 8);

// length 128: factors (16,8), max_radix=16
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(128, 64, Precision::Single, s, float, 16);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(128, 64, Precision::Single, s, float, 16);
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(128, 64, Precision::Double, d, double, 16);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(128, 64, Precision::Double, d, double, 16);

// length 256: factors (16,16), max_radix=16
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(256, 128, Precision::Single, s, float, 16);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(256, 128, Precision::Single, s, float, 16);
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(256, 128, Precision::Double, d, double, 16);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(256, 128, Precision::Double, d, double, 16);

// length 512: factors (8,8,8), max_radix=8
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(512, 256, Precision::Single, s, float, 8);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(512, 256, Precision::Single, s, float, 8);
ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE(512, 128, Precision::Double, d, double, 8);
ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE(512, 128, Precision::Double, d, double, 8);

#undef ROCFFT_DEVICE_DEFINE_FORWARD_EXECUTE
#undef ROCFFT_DEVICE_DEFINE_INVERSE_EXECUTE

} // namespace detail_fft

/*! @brief Device-callable, single-block FFT for a fixed compile-time configuration.
 *
 *  @details Selects a pre-generated Stockham transform from the supported matrix
 *  and exposes it through a static ::execute entry point plus a set of
 *  compile-time traits (element count, workgroup size, shared-memory footprint).
 *  All template parameters are resolved at compile time; an unsupported
 *  combination fails to instantiate.
 *
 *  Supported lengths are 32, 64, 128, 256, and 512 in both single and double
 *  precision.
 *
 *  @tparam Length Transform length (number of complex elements). Must be one of
 *          the supported lengths.
 *  @tparam P      Numeric precision; see ::rocfft::device::Precision.
 *  @tparam D      Transform direction; see ::rocfft::device::Direction.
 */
template <unsigned Length, Precision P, Direction D>
struct FFT
{
private:
    using dispatch = detail::FFTDispatch<Length, P>;
    using exec     = detail_fft::FFTExecute<Length, dispatch::workgroup_size, P, D>;

public:
    using complex_type = typename precision_traits<P>::complex_type; //!< Complex element type (rocfft_complex<float|double>).
    using real_type    = typename precision_traits<P>::real_type;    //!< Underlying real scalar type (float or double).

    static constexpr unsigned length             = Length;                 //!< Transform length in complex elements.
    static constexpr unsigned workgroup_size      = dispatch::workgroup_size; //!< Threads expected to cooperate on one transform.
    static constexpr unsigned elements_per_thread = dispatch::ept;          //!< Complex elements each thread holds in registers (length / workgroup_size).
    // Stockham lds_to_reg reads up to (max_radix - 1) positions beyond
    // the transform length for some thread indices.  Pad the LDS allocation
    // by one register block so those reads stay within bounds.
    static constexpr unsigned shared_memory_size  = (Length + dispatch::max_radix) * sizeof(complex_type); //!< Required shared-memory scratch size in bytes (padded for over-read).

    /*! @brief Execute the transform in place on a single thread block.
     *  @param data          Per-thread register array of ::elements_per_thread
     *                       complex values; overwritten with the transform result.
     *  @param shared_scratch Shared-memory buffer of at least ::shared_memory_size
     *                       bytes, shared by all ::workgroup_size threads.
     *  @param thread_id     Calling thread's index within the workgroup
     *                       (typically `threadIdx.x`), in [0, ::workgroup_size).
     *  @note All participating threads must call this function; it issues block
     *        synchronization internally.
     */
    static __device__ void execute(complex_type* data, char* shared_scratch,
                                   unsigned int thread_id)
    {
        exec::execute(data, shared_scratch, thread_id);
    }
};

} // namespace device
} // namespace rocfft

#endif // ROCFFT_DEVICE_FFT_HPP
