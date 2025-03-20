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

#ifndef HIPTENSOR_LIBRARY_DATA_TYPES_HPP
#define HIPTENSOR_LIBRARY_DATA_TYPES_HPP

// clang-format off
// Include order needs to be preserved
#include <hip/library_types.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <iostream>

#include <hiptensor/hiptensor_types.hpp>

// clang-format on

namespace hiptensor
{
    using index_t = int32_t;
    // Used to map to empty tensors
    struct NoneType;

    struct ScalarData
    {
        hiptensorComputeType_t mType;
        union
        {
            double           mReal;
            hipDoubleComplex mComplex;
        };

        ScalarData() = default;
        ScalarData(hiptensorComputeType_t type, double real, double imag = 0)
        {
            mType = type;
            if(type == HIPTENSOR_COMPUTE_C32F || type == HIPTENSOR_COMPUTE_C64F)
            {
                mComplex = make_hipDoubleComplex(real, imag);
            }
            else
            {
                mReal = real;
            }
        }
        operator float() const
        {
            return static_cast<float>(mReal);
        }
        operator double() const
        {
            return mReal;
        }
        operator hipFloatComplex() const
        {
            return hipComplexDoubleToFloat(mComplex);
        }
        operator hipDoubleComplex() const
        {
            return mComplex;
        }
    };

    static constexpr hipDataType NONE_TYPE = (hipDataType)31;

    // Map type to runtime HipDataType
    template <typename T>
    struct HipDataType;

    template <typename T>
    static constexpr auto HipDataType_v = HipDataType<T>::value;

    // Get data size in bytes from id
    uint32_t hipDataTypeSize(hipDataType id);

    // Convert hipDataType to hiptensorComputeType_t
    hiptensorComputeType_t convertToComputeType(hipDataType hipType);

    // Read a single value from void pointer, casted to T
    template <typename T>
    T readVal(void const* value, hipDataType id);

    template <typename T>
    T readVal(void const* value, hiptensorComputeType_t id);

    void writeVal(void const* addr, hiptensorComputeType_t id, ScalarData value);

    std::string computeTypeToString(hiptensorComputeType_t computeType);
    std::string hipTypeToString(hipDataType hipType);
    std::string opTypeToString(hiptensorOperator_t opType);
    std::string algoTypeToString(hiptensorAlgo_t algoType);
    std::string logLevelToString(hiptensorLogLevel_t);
    std::string workSizePrefToString(hiptensorWorksizePreference_t workSize);
} // namespace hiptensor

bool operator==(hipDataType hipType, hiptensorComputeType_t computeType);
bool operator==(hiptensorComputeType_t computeType, hipDataType hipType);

bool operator!=(hipDataType hipType, hiptensorComputeType_t computeType);
bool operator!=(hiptensorComputeType_t computeType, hipDataType hipType);

namespace std
{
    std::string to_string(const hiptensor::ScalarData& value);
}

#include "data_types_impl.hpp"

#endif // HIPTENSOR_LIBRARY_DATA_TYPES_HPP
