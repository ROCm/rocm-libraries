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

#include <cstring>
#include <iostream>

#include "data_types.hpp"
#include "logger.hpp"

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

    // Host scalars come from user pointers that carry no alignment guarantee (a complex double
    // passed as double[2] is only 8-byte aligned), so they are copied byte-wise, never
    // dereferenced as the wider type.
    template <typename T>
    T loadUnaligned(void const* p)
    {
        T v;
        std::memcpy(&v, p, sizeof(T));
        return v;
    }

    template <typename T>
    void storeUnaligned(void* p, T const& v)
    {
        std::memcpy(p, &v, sizeof(T));
    }

    template <typename T>
    T readVal(void const* value, hiptensorDataType_t id)
    {
        if(id == HIPTENSOR_R_16BF)
        {
            return static_cast<T>(loadUnaligned<hip_bfloat16>(value));
        }
        else if(id == HIPTENSOR_R_16F)
        {
            return static_cast<T>(loadUnaligned<_Float16>(value));
        }
        else if(id == HIPTENSOR_R_32F)
        {
            return static_cast<T>(loadUnaligned<float>(value));
        }
        else if(id == HIPTENSOR_R_64F)
        {
            return static_cast<T>(loadUnaligned<double>(value));
        }
        else if(id == HIPTENSOR_R_8I)
        {
            return static_cast<T>(loadUnaligned<int8_t>(value));
        }
        else if(id == HIPTENSOR_R_8U)
        {
            return static_cast<T>(loadUnaligned<uint8_t>(value));
        }
        else if(id == HIPTENSOR_R_16I)
        {
            return static_cast<T>(loadUnaligned<int16_t>(value));
        }
        else if(id == HIPTENSOR_R_16U)
        {
            return static_cast<T>(loadUnaligned<uint16_t>(value));
        }
        else if(id == HIPTENSOR_R_32I)
        {
            return static_cast<T>(loadUnaligned<int32_t>(value));
        }
        else if(id == HIPTENSOR_R_32U)
        {
            return static_cast<T>(loadUnaligned<uint32_t>(value));
        }
        else if(id == HIPTENSOR_R_64I)
        {
            return static_cast<T>(loadUnaligned<int64_t>(value));
        }
        else if(id == HIPTENSOR_R_64U)
        {
            return static_cast<T>(loadUnaligned<uint64_t>(value));
        }
        else if constexpr(std::is_same_v<T, hipFloatComplex> && id == HIPTENSOR_C_32F)
        {
            return static_cast<T>(loadUnaligned<hipFloatComplex>(value));
        }
        else if constexpr(std::is_same_v<T, hipDoubleComplex> && id == HIPTENSOR_C_64F)
        {
            return static_cast<T>(loadUnaligned<hipDoubleComplex>(value));
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
        switch(id)
        {
        case HIPTENSOR_COMPUTE_DESC_16F:
            return static_cast<T>(loadUnaligned<_Float16>(value));
        case HIPTENSOR_COMPUTE_DESC_16BF:
            return static_cast<T>(static_cast<float>(loadUnaligned<hip_bfloat16>(value)));
        case HIPTENSOR_COMPUTE_DESC_32F:
            return static_cast<T>(loadUnaligned<float>(value));
        case HIPTENSOR_COMPUTE_DESC_64F:
            return static_cast<T>(loadUnaligned<double>(value));
        // A scalar read is not defined for these. Listed explicitly so -Wswitch
        // flags any enumerator added to hiptensorComputeDescriptor_t later.
        case HIPTENSOR_COMPUTE_DESC_C32F: // served by the ScalarData specialization
        case HIPTENSOR_COMPUTE_DESC_C64F:
        case HIPTENSOR_COMPUTE_DESC_8U:
        case HIPTENSOR_COMPUTE_DESC_8I:
        case HIPTENSOR_COMPUTE_DESC_32U:
        case HIPTENSOR_COMPUTE_DESC_32I:
        case HIPTENSOR_COMPUTE_DESC_NONE:
            break;
        }
        char msg[96];
        snprintf(msg, sizeof(msg), "Unhandled compute descriptor: %d", static_cast<int>(id));
        Logger::instance()->logError("readVal", msg);
        return 0;
    }

    // @cond
    template <>
    ScalarData readVal(void const* value, hiptensorComputeDescriptor_t id);
    // @endcond

} // namespace hiptensor
