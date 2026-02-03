/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <iostream>

#include "data_types.hpp"

namespace hiptensor
{
    // Specialize overrides for runtime HipTensorDataType
    template <>
    struct HipTensorDataType<hip_bfloat16>
    {
        static constexpr auto value = HIPTENSOR_R_16BF;
    };

    template <>
    struct HipTensorDataType<_Float16>
    {
        static constexpr auto value = HIPTENSOR_R_16F;
    };

    template <>

    struct HipTensorDataType<float>
    {
        static constexpr auto value = HIPTENSOR_R_32F;
    };

    template <>
    struct HipTensorDataType<double>
    {
        static constexpr auto value = HIPTENSOR_R_64F;
    };

    template <>
    struct HipTensorDataType<int8_t>
    {
        static constexpr auto value = HIPTENSOR_R_8I;
    };

    template <>
    struct HipTensorDataType<uint8_t>
    {
        static constexpr auto value = HIPTENSOR_R_8U;
    };

    template <>
    struct HipTensorDataType<int16_t>
    {
        static constexpr auto value = HIPTENSOR_R_16I;
    };

    template <>
    struct HipTensorDataType<uint16_t>
    {
        static constexpr auto value = HIPTENSOR_R_16U;
    };

    template <>
    struct HipTensorDataType<int32_t>
    {
        static constexpr auto value = HIPTENSOR_R_32I;
    };

    template <>
    struct HipTensorDataType<uint32_t>
    {
        static constexpr auto value = HIPTENSOR_R_32U;
    };

    template <>
    struct HipTensorDataType<int64_t>
    {
        static constexpr auto value = HIPTENSOR_R_64I;
    };

    template <>
    struct HipTensorDataType<uint64_t>
    {
        static constexpr auto value = HIPTENSOR_R_64U;
    };

    template <>
    struct HipTensorDataType<hipFloatComplex>
    {
        static constexpr auto value = HIPTENSOR_C_32F;
    };

    template <>
    struct HipTensorDataType<hipDoubleComplex>
    {
        static constexpr auto value = HIPTENSOR_C_64F;
    };

    template <>
    struct HipTensorDataType<NoneType>
    {
        static constexpr auto value = NONE_TYPE;
    };

    template <typename T>
    T readVal(void const* value, hiptensorDataType_t id)
    {
        if(id == HIPTENSOR_R_16BF)
        {
            return static_cast<T>(*(hip_bfloat16*)value);
        }
        else if(id == HIPTENSOR_R_16F)
        {
            return static_cast<T>(*(_Float16*)value);
        }
        else if(id == HIPTENSOR_R_32F)
        {
            return static_cast<T>(*(float*)value);
        }
        else if(id == HIPTENSOR_R_64F)
        {
            return static_cast<T>(*(double*)value);
        }
        else if(id == HIPTENSOR_R_8I)
        {
            return static_cast<T>(*(int8_t*)value);
        }
        else if(id == HIPTENSOR_R_8U)
        {
            return static_cast<T>(*(uint8_t*)value);
        }
        else if(id == HIPTENSOR_R_16I)
        {
            return static_cast<T>(*(int16_t*)value);
        }
        else if(id == HIPTENSOR_R_16U)
        {
            return static_cast<T>(*(uint16_t*)value);
        }
        else if(id == HIPTENSOR_R_32I)
        {
            return static_cast<T>(*(int32_t*)value);
        }
        else if(id == HIPTENSOR_R_32U)
        {
            return static_cast<T>(*(uint32_t*)value);
        }
        else if(id == HIPTENSOR_R_64I)
        {
            return static_cast<T>(*(int64_t*)value);
        }
        else if(id == HIPTENSOR_R_64U)
        {
            return static_cast<T>(*(uint64_t*)value);
        }
        else if constexpr(std::is_same_v<T, hipFloatComplex> && id == HIPTENSOR_C_32F)
        {
            return static_cast<T>(*(hipFloatComplex*)value);
        }
        else if constexpr(std::is_same_v<T, hipDoubleComplex> && id == HIPTENSOR_C_64F)
        {
            return static_cast<T>(*(hipDoubleComplex*)value);
        }
        else
        {
#if !NDEBUG
            std::cout << "Unhandled hip datatype: " << id << std::endl;
#endif // !NDEBUG
            return 0;
        }
    }

    template <typename T>
    T readVal(void const* value, hiptensorComputeDescriptor_t id)
    {
        if(id == HIPTENSOR_COMPUTE_DESC_16F)
        {
            return static_cast<T>(*(_Float16*)value);
        }
        else if(id == HIPTENSOR_COMPUTE_DESC_32F)
        {
            return static_cast<T>(*(float*)value);
        }
        else if(id == HIPTENSOR_COMPUTE_DESC_64F)
        {
            return static_cast<T>(*(double*)value);
        }
        else
        {
#if !NDEBUG
            std::cout << "Unhandled hiptensorComputeDescriptor_t: " << id << std::endl;
#endif // !NDEBUG
            return 0;
        }
    }

    // @cond
    template <>
    ScalarData readVal(void const* value, hiptensorComputeDescriptor_t id);
    // @endcond

} // namespace hiptensor
