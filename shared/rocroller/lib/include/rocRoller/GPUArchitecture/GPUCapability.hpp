// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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
            HasExplicitScalarCO,
            HasExplicitScalarCOCI,
            HasExplicitVectorCO,
            HasExplicitVectorCOCI,
            HasExplicitVectorRevCO,
            HasExplicitVectorRevCOCI,
            HasExplicitVectorRevNC,
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
            HasMFMA_scale_f8f6f4,
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

            HasWMMA,
            HasWMMA_F16_ACC,
            HasWMMA_f32_16x16x16_f16,
            HasWMMA_f16_16x16x16_f16,
            HasWMMA_bf16_16x16x16_bf16,
            HasWMMA_f16_16x16x32_f16,
            HasWMMA_bf16_16x16x32_bf16,
            HasWMMA_f32_16x16x32_f16,
            HasWMMA_f32_16x16x16_f8,
            HasWMMA_f32_16x16x64_f8,
            HasWMMA_f16_16x16x64_f8,
            HasWMMA_f32_16x16x128_f8,
            HasWMMA_f16_16x16x128_f8,
            HasWMMA_f32_16x16x4_f32,

            HasWMMA_f8f6f4,
            HasWMMA_scale_f8f6f4,
            HasWMMA_scale16_f8f6f4,
            HasWMMA_32x16x128_f4,
            HasWMMA_scale_32x16x128_f4,
            HasWMMA_scale16_32x16x128_f4,

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

            v_add3_u32,

            s_barrier,
            s_barrier_signal,

            HasAtomicAdd,

            MaxVmcnt,
            MaxLgkmcnt,
            MaxExpcnt,
            HasExpcnt,
            MaxTensorcnt,
            HasTensorcnt,
            SupportedSource,

            Waitcnt0Disabled,
            SeparateVscnt,
            HasSplitWaitCounters,
            CMPXWritesSGPR,
            HasWave32,
            HasWave64,
            DefaultWavefrontSize,
            HasAccCD,
            ArchAccUnifiedRegs,
            PackedWorkitemIDs,

            HasBlockScaling32,
            HasBlockScaling16,
            DefaultScaleBlockSize,
            HasE8M0Scale,
            HasE5M3Scale,
            HasE4M3Scale,

            HasXnack,

            UnalignedVGPRs,
            UnalignedSGPRs,

            MaxLdsSize,

            HasNaNoo,

            HasDSReadTransposeB16,
            HasDSReadTransposeB8,
            HasDSReadTransposeB6,
            HasDSReadTransposeB4,

            ds_read_b64_tr_b16,
            ds_read_b64_tr_b8,
            ds_read_b96_tr_b6,
            ds_read_b64_tr_b4,

            ds_load_tr16_b128,
            ds_load_tr8_b64,
            ds_load_tr6_b96,
            ds_load_tr4_b64,

            DSReadTransposeB6PaddingBytes,

            HasPRNG,

            HasPermLanes16,
            HasPermLanes32,

            WorkgroupIdxViaTTMP,
            HasBufferOutOfBoundsCheckOption,
            HasBufferFormatSpecInSOffsetField,

            HasXCC,
            DefaultRemapXCCValue,

            /**
             * The maximum number of SGPRs that can be preloaded with kernel
             * arguments at the beginning of the kernel.
             *
             * Note that this is the absolute maximum for the architecture, and that the actual
             * maximum for a given kernel will be reduced if other user SPGRs (such as the
             * kernel argument pointer) are also needed.
             */
            MaxPreloadedKernargs,

            PartiallyActiveWaveSize,

            HasVGPRIndexing,

            HasWorkgroupClusters,

            HasTDM,

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
