
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
            HasAddLshl,
            HasLshlOr,
            HasSMulHi,
            HasCodeObjectV3,
            HasCodeObjectV4,
            HasCodeObjectV5,

            HasMFMA,
            HasMFMA_fp8,
            HasMFMA_f64,
            HasMFMA_bf16,
            HasMFMA_bf16_1k,

            HasAccumOffset,
            HasFlatOffset,

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

            HasEccHalf,
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
}

#include "GPUCapability_impl.hpp"
