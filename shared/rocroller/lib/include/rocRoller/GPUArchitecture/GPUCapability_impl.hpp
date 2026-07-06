// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
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

#include <rocRoller/GPUArchitecture/GPUCapability.hpp>

namespace rocRoller
{
    inline std::string GPUCapability::toString() const
    {
        return GPUCapability::toString(m_value);
    }

    inline std::ostream& operator<<(std::ostream& stream, GPUCapability::Value v)
    {
        return stream << GPUCapability::toString(v);
    }

    inline std::string GPUCapability::toString(GPUCapability::Value value)
    {
        auto it = std::find_if(m_stringMap.begin(),
                               m_stringMap.end(),
                               [&value](auto const& mapping) { return value == mapping.second; });

        if(it == m_stringMap.end())
            return "";
        return it->first;
    }

    inline const std::unordered_map<std::string, GPUCapability::Value> GPUCapability::m_stringMap
        = {
            {"SupportedISA", Value::SupportedISA},
            {"HasExplicitScalarCO", Value::HasExplicitScalarCO},
            {"HasExplicitScalarCOCI", Value::HasExplicitScalarCOCI},
            {"HasExplicitVectorCO", Value::HasExplicitVectorCO},
            {"HasExplicitVectorCOCI", Value::HasExplicitVectorCOCI},
            {"HasExplicitVectorRevCO", Value::HasExplicitVectorRevCO},
            {"HasExplicitVectorRevCOCI", Value::HasExplicitVectorRevCOCI},
            {"HasExplicitVectorRevNC", Value::HasExplicitVectorRevNC},
            {"HasExplicitNC", Value::HasExplicitNC},

            {"HasDirectToLds", Value::HasDirectToLds},
            {"HasWiderDirectToLds", Value::HasWiderDirectToLds},
            {"HasAddLshl", Value::HasAddLshl},
            {"HasLshlOr", Value::HasLshlOr},
            {"HasSMulHi", Value::HasSMulHi},
            {"HasCodeObjectV3", Value::HasCodeObjectV3},
            {"HasCodeObjectV4", Value::HasCodeObjectV4},
            {"HasCodeObjectV5", Value::HasCodeObjectV5},

            {"HasMFMA", Value::HasMFMA},
            {"HasMFMA_fp8", Value::HasMFMA_fp8},
            {"HasMFMA_f8f6f4", Value::HasMFMA_f8f6f4},
            {"HasMFMA_scale_f8f6f4", Value::HasMFMA_scale_f8f6f4},
            {"HasMFMA_f64", Value::HasMFMA_f64},
            {"HasMFMA_bf16_32x32x4", Value::HasMFMA_bf16_32x32x4},
            {"HasMFMA_bf16_32x32x4_1k", Value::HasMFMA_bf16_32x32x4_1k},
            {"HasMFMA_bf16_16x16x8", Value::HasMFMA_bf16_16x16x8},
            {"HasMFMA_bf16_16x16x16_1k", Value::HasMFMA_bf16_16x16x16_1k},

            {"HasMFMA_16x16x32_f16", Value::HasMFMA_16x16x32_f16},
            {"HasMFMA_32x32x16_f16", Value::HasMFMA_32x32x16_f16},
            {"HasMFMA_16x16x32_bf16", Value::HasMFMA_16x16x32_bf16},
            {"HasMFMA_32x32x16_bf16", Value::HasMFMA_32x32x16_bf16},

            {"HasWMMA", Value::HasWMMA},
            {"HasWMMA_F16_ACC", Value::HasWMMA_F16_ACC},
            {"HasWMMA_f32_16x16x16_f16", Value::HasWMMA_f32_16x16x16_f16},
            {"HasWMMA_f16_16x16x16_f16", Value::HasWMMA_f16_16x16x16_f16},
            {"HasWMMA_bf16_16x16x16_bf16", Value::HasWMMA_bf16_16x16x16_bf16},
            {"HasWMMA_f16_16x16x32_f16", Value::HasWMMA_f16_16x16x32_f16},
            {"HasWMMA_bf16_16x16x32_bf16", Value::HasWMMA_bf16_16x16x32_bf16},
            {"HasWMMA_f32_16x16x32_f16", Value::HasWMMA_f32_16x16x32_f16},
            {"HasWMMA_f32_16x16x16_f8", Value::HasWMMA_f32_16x16x16_f8},
            {"HasWMMA_f32_16x16x64_f8", Value::HasWMMA_f32_16x16x64_f8},
            {"HasWMMA_f16_16x16x64_f8", Value::HasWMMA_f16_16x16x64_f8},
            {"HasWMMA_f32_16x16x128_f8", Value::HasWMMA_f32_16x16x128_f8},
            {"HasWMMA_f16_16x16x128_f8", Value::HasWMMA_f16_16x16x128_f8},
            {"HasWMMA_f32_16x16x4_f32", Value::HasWMMA_f32_16x16x4_f32},

            {"HasWMMA_f8f6f4", Value::HasWMMA_f8f6f4},
            {"HasWMMA_scale_f8f6f4", Value::HasWMMA_scale_f8f6f4},
            {"HasWMMA_scale16_f8f6f4", Value::HasWMMA_scale16_f8f6f4},
            {"HasWMMA_32x16x128_f4", Value::HasWMMA_32x16x128_f4},
            {"HasWMMA_scale_32x16x128_f4", Value::HasWMMA_scale_32x16x128_f4},
            {"HasWMMA_scale16_32x16x128_f4", Value::HasWMMA_scale16_32x16x128_f4},

            {"HasAccumOffset", Value::HasAccumOffset},
            {"HasGlobalOffset", Value::HasGlobalOffset},

            {"v_mac_f16", Value::v_mac_f16},

            {"v_fma_f16", Value::v_fma_f16},
            {"v_fmac_f16", Value::v_fmac_f16},

            {"v_pk_fma_f16", Value::v_pk_fma_f16},
            {"v_pk_fmac_f16", Value::v_pk_fmac_f16},

            {"v_mad_mix_f32", Value::v_mad_mix_f32},
            {"v_fma_mix_f32", Value::v_fma_mix_f32},

            {"v_dot2_f32_f16", Value::v_dot2_f32_f16},
            {"v_dot2c_f32_f16", Value::v_dot2c_f32_f16},

            {"v_dot4c_i32_i8", Value::v_dot4c_i32_i8},
            {"v_dot4_i32_i8", Value::v_dot4_i32_i8},

            {"v_mac_f32", Value::v_mac_f32},
            {"v_fma_f32", Value::v_fma_f32},
            {"v_fmac_f32", Value::v_fmac_f32},

            {"v_mov_b64", Value::v_mov_b64},

            {"v_add3_u32", Value::v_add3_u32},

            {"s_barrier", Value::s_barrier},
            {"s_barrier_signal", Value::s_barrier_signal},

            {"HasAtomicAdd", Value::HasAtomicAdd},

            {"MaxVmcnt", Value::MaxVmcnt},
            {"MaxLgkmcnt", Value::MaxLgkmcnt},
            {"MaxExpcnt", Value::MaxExpcnt},
            {"HasExpcnt", Value::HasExpcnt},
            {"MaxTensorcnt", Value::MaxTensorcnt},
            {"HasTensorcnt", Value::HasTensorcnt},
            {"SupportedSource", Value::SupportedSource},

            {"Waitcnt0Disabled", Value::Waitcnt0Disabled},
            {"SeparateVscnt", Value::SeparateVscnt},
            {"HasSplitWaitCounters", Value::HasSplitWaitCounters},
            {"CMPXWritesSGPR", Value::CMPXWritesSGPR},
            {"HasWave32", Value::HasWave32},
            {"HasAccCD", Value::HasAccCD},
            {"ArchAccUnifiedRegs", Value::ArchAccUnifiedRegs},
            {"PackedWorkitemIDs", Value::PackedWorkitemIDs},

            {"HasXnack", Value::HasXnack},
            {"HasWave64", Value::HasWave64},
            {"DefaultWavefrontSize", Value::DefaultWavefrontSize},

            {"HasBlockScaling32", Value::HasBlockScaling32},
            {"HasBlockScaling16", Value::HasBlockScaling16},
            {"DefaultScaleBlockSize", Value::DefaultScaleBlockSize},
            {"HasE8M0Scale", Value::HasE8M0Scale},
            {"HasE5M3Scale", Value::HasE5M3Scale},
            {"HasE4M3Scale", Value::HasE4M3Scale},

            {"UnalignedVGPRs", Value::UnalignedVGPRs},
            {"UnalignedSGPRs", Value::UnalignedSGPRs},

            {"MaxLdsSize", Value::MaxLdsSize},

            {"HasNaNoo", Value::HasNaNoo},

            {"HasDSReadTransposeB16", Value::HasDSReadTransposeB16},
            {"HasDSReadTransposeB8", Value::HasDSReadTransposeB8},
            {"HasDSReadTransposeB6", Value::HasDSReadTransposeB6},
            {"HasDSReadTransposeB4", Value::HasDSReadTransposeB4},

            {"ds_read_b64_tr_b16", Value::ds_read_b64_tr_b16},
            {"ds_read_b64_tr_b8", Value::ds_read_b64_tr_b8},
            {"ds_read_b96_tr_b6", Value::ds_read_b96_tr_b6},
            {"ds_read_b64_tr_b4", Value::ds_read_b64_tr_b4},

            {"ds_load_tr16_b128", Value::ds_load_tr16_b128},
            {"ds_load_tr8_b64", Value::ds_load_tr8_b64},
            {"ds_load_tr6_b96", Value::ds_load_tr6_b96},
            {"ds_load_tr4_b64", Value::ds_load_tr4_b64},

            {"DSReadTransposeB6PaddingBytes", Value::DSReadTransposeB6PaddingBytes},

            {"HasPRNG", Value::HasPRNG},

            {"HasPermLanes16", Value::HasPermLanes16},
            {"HasPermLanes32", Value::HasPermLanes32},

            {"WorkgroupIdxViaTTMP", Value::WorkgroupIdxViaTTMP},
            {"HasBufferOutOfBoundsCheckOption", Value::HasBufferOutOfBoundsCheckOption},
            {"HasBufferFormatSpecInSOffsetField", Value::HasBufferFormatSpecInSOffsetField},

            {"HasXCC", Value::HasXCC},
            {"DefaultRemapXCCValue", Value::DefaultRemapXCCValue},

            {"MaxPreloadedKernargs", Value::MaxPreloadedKernargs},

            {"PartiallyActiveWaveSize", Value::PartiallyActiveWaveSize},

            {"HasVGPRIndexing", Value::HasVGPRIndexing},

            {"HasWorkgroupClusters", Value::HasWorkgroupClusters},

            {"HasTDM", Value::HasTDM},
    };
}
