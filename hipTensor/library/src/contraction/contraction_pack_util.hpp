/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#ifndef HIPTENSOR_CONTRACTION_PACK_UTIL_HPP
#define HIPTENSOR_CONTRACTION_PACK_UTIL_HPP

#include "data_types.hpp"
#include "util.hpp"
#include <hiptensor/hiptensor.hpp>

namespace hiptensor
{
    /**
     * \brief This function performs multiply-accumulate of the form E = accum * alpha + D * beta
     *
     */
    template <typename DataType>
    __global__ void mfma(DataType*                     mE_real,
                         DataType*                     mE_imag,
                         DataType*                     mD_real,
                         DataType*                     mD_imag,
                         HIP_vector_type<DataType, 2>* mE_grid,
                         HIP_vector_type<double, 2>    alpha,
                         HIP_vector_type<double, 2>    beta,
                         int                           length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx < length)
        {
            if constexpr(std::is_same_v<DataType, float>)
            {
                mE_grid[idx] = hipCaddf(hipCmulf(make_hipFloatComplex(mE_real[idx], mE_imag[idx]),
                                                 hipComplexDoubleToFloat(alpha)),
                                        hipCmulf(make_hipFloatComplex(mD_real[idx], mD_imag[idx]),
                                                 hipComplexDoubleToFloat(beta)));
            }
            else if constexpr(std::is_same_v<DataType, double>)
            {
                mE_grid[idx]
                    = hipCadd(hipCmul(make_hipDoubleComplex(mE_real[idx], mE_imag[idx]), alpha),
                              hipCmul(make_hipDoubleComplex(mD_real[idx], mD_imag[idx]), beta));
            }
        }
    }

    /**
     * \brief This function performs multiply of the form C = accum * alpha
     *
     */
    template <typename DataType>
    __global__ void multiply(DataType*                     mE_real,
                             DataType*                     mE_imag,
                             HIP_vector_type<DataType, 2>* mE_grid,
                             HIP_vector_type<double, 2>    alpha,
                             int                           length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx < length)
        {
            if constexpr(std::is_same_v<DataType, float>)
            {
                mE_grid[idx] = hipCmulf(make_hipFloatComplex(mE_real[idx], mE_imag[idx]),
                                        hipComplexDoubleToFloat(alpha));
            }
            else if constexpr(std::is_same_v<DataType, double>)
            {
                mE_grid[idx] = hipCmul(make_hipDoubleComplex(mE_real[idx], mE_imag[idx]), alpha);
            }
        }
    }

    /**
     * \brief This function unpacks structured data (hipFloatComplex / hipDoubleComplex)
     *        into non-structured data (float / double).
     */
    template <typename InputType, typename OutputType>
    __global__ void
        unpack(const InputType* in, OutputType* out_real, OutputType* out_img, int length)
    {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if(idx < length)
        {
            if constexpr(std::is_same_v<InputType, hipFloatComplex>)
            {
                out_real[idx] = hipCrealf(in[idx]);
                out_img[idx]  = hipCimagf(in[idx]);
            }
            else if constexpr(std::is_same_v<InputType, hipDoubleComplex>)
            {
                out_real[idx] = hipCreal(in[idx]);
                out_img[idx]  = hipCimag(in[idx]);
            }
        }
    }

    struct DeviceDeleter
    {
        void operator()(void* ptr)
        {
            CHECK_HIP_ERROR(hipFree(ptr));
        }
    };

    template <typename T>
    auto allocDevice(int64_t numElements)
    {
        T* data;
        CHECK_HIP_ERROR(hipMalloc(&data, numElements * sizeof(T)));
        return std::unique_ptr<T, DeviceDeleter>(data, DeviceDeleter());
    }

} // namespace hiptensor

#endif // HIPTENSOR_CONTRACTION_PACK_UTIL_HPP
