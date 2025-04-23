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

#ifndef HIPTENSOR_TEST_CONTRACTION_DEVICE_COMMON_HPP
#define HIPTENSOR_TEST_CONTRACTION_DEVICE_COMMON_HPP

template <typename T>
__device__ inline double toDouble(T const& val)
{
    // TODO: deprecate. Not needed anymore because bfloat16_t has float operator
    return static_cast<double>(val);
}

__device__ inline double maxDouble(double a, double b)
{
    if(std::isinf(a) || std::isinf(b))
    {
        return std::numeric_limits<double>::infinity();
    }
    // Check for NaN
    else if(std::isnan(a) || std::isnan(b))
    {
        return std::numeric_limits<double>::signaling_NaN();
    }
    return a > b ? a : b;
}

__global__ static void
    maxReduceKernel(double* relativeError, uint32_t elements, uint32_t offset, uint32_t maxElements)
{
    double* localRelativeError = relativeError + (offset * elements * blockIdx.x);

    for(int i = elements >> 1; i > 0; i = i >> 1)
    {
        if(threadIdx.x < i && offset * (elements * blockIdx.x + threadIdx.x + i) < maxElements)
        {
            localRelativeError[offset * threadIdx.x]
                = maxDouble(localRelativeError[offset * threadIdx.x],
                            localRelativeError[offset * (threadIdx.x + i)]);
        }
        __syncthreads();
    }
}

__device__ inline unsigned pcg_hash(unsigned input)
{
    unsigned state = input * 747796405u + 2891336453u;
    unsigned word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// gen random float in range [-range, range)
template <unsigned range = 1>
__device__ inline float gen_random_float(unsigned input)
{
    return (static_cast<float>(pcg_hash(input)) / static_cast<float>(UINT_MAX) - 0.5f)
           * static_cast<float>(range) * 2;
}

// fill kernel for 'elementSize' elements
template <typename DataType>
__global__ void fillKernel(DataType* data, uint32_t elementSize, uint32_t seed)
{
    uint32_t index       = (blockIdx.x * blockDim.x + threadIdx.x);
    uint32_t seededIndex = static_cast<uint32_t>(uint64_t(index + seed) % UINT_MAX);

    if(index < elementSize)
    {
        if constexpr(std::is_same_v<DataType, hipFloatComplex>)
        {
            auto value  = gen_random_float(seededIndex);
            data[index] = make_hipFloatComplex(value, value);
        }
        else if constexpr(std::is_same_v<DataType, hipDoubleComplex>)
        {
            auto value  = static_cast<double>(gen_random_float(seededIndex));
            data[index] = make_hipDoubleComplex(value, value);
        }
        else
        {
            auto value  = gen_random_float(seededIndex);
            data[index] = static_cast<DataType>(value);
        }
    }
}

// fill kernel wrapper for 'elementSize' elements with a specific value
template <typename DataType>
__global__ void fillValKernel(DataType* data, uint32_t elementSize, DataType value)
{
    uint32_t index = (blockIdx.x * blockDim.x + threadIdx.x);

    if(index < elementSize)
    {
        data[index] = value;
    }
}

template <typename DDataType>
__global__ void compareEqualKernel(DDataType* deviceD,
                                   DDataType* hostD,
                                   double*    relativeError,
                                   uint32_t   elementsD)
{
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < elementsD)
    {
        DDataType valDevice = deviceD[index];
        DDataType valHost   = hostD[index];

        auto numerator = fabs(toDouble(valDevice) - toDouble(valHost));
        auto divisor   = fabs(toDouble(valDevice)) + fabs(toDouble(valHost)) + 1.0;

        if(std::isinf(numerator) || std::isinf(divisor))
        {
            relativeError[index] = std::numeric_limits<DDataType>::infinity();
        }
        else if(std::isnan(numerator) || std::isnan(divisor))
        {
            relativeError[index] = std::numeric_limits<DDataType>::signaling_NaN();
        }
        else
        {
            relativeError[index] = numerator / divisor;
        }
    }
}

#endif // HIPTENSOR_TEST_CONTRACTION_DEVICE_COMMON_HPP
