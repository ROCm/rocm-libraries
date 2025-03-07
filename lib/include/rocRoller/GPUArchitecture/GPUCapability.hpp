/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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

#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    class GPUCapability
    {
    public:
        enum Value : uint8_t
        {
            SupportedISA = 0,
            HasExplicitCO,
            HasExplicitNC,

            HasDirectToLds,
            HasWiderDirectToLds,
            HasAddLshl,
            HasLshlOr,
            HasSMulHi,
            HasCodeObjectV3,
            HasCodeObjectV4,
            HasCodeObjectV5,

            HasMFMA,
            HasMFMA_fp8,
            HasMFMA_f8f6f4,
            HasMFMA_f64,
            HasMFMA_bf16_32x32x4,
            HasMFMA_bf16_32x32x4_1k,
            HasMFMA_bf16_32x32x8_1k,
            HasMFMA_bf16_16x16x8,
            HasMFMA_bf16_16x16x16_1k,

            HasMFMA_16x16x32_f16,
            HasMFMA_32x32x16_f16,
            HasMFMA_16x16x32_bf16,
            HasMFMA_32x32x16_bf16,

            HasAccumOffset,
            HasGlobalOffset,

            v_mac_f16,

            v_fma_f16,
            v_fmac_f16,

            v_pk_fma_f16,
            v_pk_fmac_f16,

            v_mad_mix_f32,
            v_fma_mix_f32,

            v_dot2_f32_f16,
            v_dot2c_f32_f16,

            v_dot4c_i32_i8,
            v_dot4_i32_i8,

            v_mac_f32,
            v_fma_f32,
            v_fmac_f32,

            v_mov_b64,

            HasAtomicAdd,

            MaxVmcnt,
            MaxLgkmcnt,
            MaxExpcnt,
            SupportedSource,

            Waitcnt0Disabled,
            SeparateVscnt,
            CMPXWritesSGPR,
            HasWave32,
            HasWave64,
            DefaultWavefrontSize,
            HasAccCD,
            ArchAccUnifiedRegs,
            PackedWorkitemIDs,

            HasXnack,

            UnalignedVGPRs,
            UnalignedSGPRs,

            MaxLdsSize,

            HasNaNoo,

            HasDSReadTransposeB16,
            HasDSReadTransposeB8,
            HasDSReadTransposeB6,
            HasDSReadTransposeB4,

            HasPRNG,

            HasPermLanes16,
            HasPermLanes32,

            Count,
        };

        GPUCapability() = default;
        // cppcheck-suppress noExplicitConstructor
        constexpr GPUCapability(Value input)
            : m_value(input)
        {
        }
        explicit GPUCapability(std::string const& input)
            : m_value(GPUCapability::m_stringMap.at(input))
        {
        }
        explicit GPUCapability(int input)
            : m_value(static_cast<Value>(input))
        {
        }

        constexpr bool operator==(GPUCapability a) const
        {
            return m_value == a.m_value;
        }
        constexpr bool operator==(Value a) const
        {
            return m_value == a;
        }
        constexpr bool operator!=(GPUCapability a) const
        {
            return m_value != a.m_value;
        }
        constexpr bool operator<(GPUCapability a) const
        {
            return m_value < a.m_value;
        }

        operator uint8_t() const
        {
            return static_cast<uint8_t>(m_value);
        }

        std::string toString() const;

        static std::string toString(Value);

        struct Hash
        {
            std::size_t operator()(const GPUCapability& input) const
            {
                return std::hash<uint8_t>()((uint8_t)input.m_value);
            };
        };

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

    private:
        Value                                               m_value = Value::Count;
        static const std::unordered_map<std::string, Value> m_stringMap;
    };

    std::ostream& operator<<(std::ostream&, GPUCapability::Value);
}

#include <rocRoller/GPUArchitecture/GPUCapability_impl.hpp>
