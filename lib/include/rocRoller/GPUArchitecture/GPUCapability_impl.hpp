
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

namespace rocRoller
{
    inline std::string GPUCapability::toString() const
    {
        return GPUCapability::toString(m_value);
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
            {"HasExplicitCO", Value::HasExplicitCO},
            {"HasExplicitNC", Value::HasExplicitNC},

            {"HasDirectToLds", Value::HasDirectToLds},
            {"HasAddLshl", Value::HasAddLshl},
            {"HasLshlOr", Value::HasLshlOr},
            {"HasSMulHi", Value::HasSMulHi},
            {"HasCodeObjectV3", Value::HasCodeObjectV3},
            {"HasCodeObjectV4", Value::HasCodeObjectV4},
            {"HasCodeObjectV5", Value::HasCodeObjectV5},

            {"HasMFMA", Value::HasMFMA},
            {"HasMFMA_fp8", Value::HasMFMA_fp8},
            {"HasMFMA_f64", Value::HasMFMA_f64},
            {"HasMFMA_bf16_32x32x4", Value::HasMFMA_bf16_32x32x4},
            {"HasMFMA_bf16_32x32x4_1k", Value::HasMFMA_bf16_32x32x4_1k},
            {"HasMFMA_bf16_16x16x8", Value::HasMFMA_bf16_16x16x8},
            {"HasMFMA_bf16_16x16x16_1k", Value::HasMFMA_bf16_16x16x16_1k},

            {"HasAccumOffset", Value::HasAccumOffset},
            {"HasFlatOffset", Value::HasFlatOffset},

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

            {"HasAtomicAdd", Value::HasAtomicAdd},

            {"MaxVmcnt", Value::MaxVmcnt},
            {"MaxLgkmcnt", Value::MaxLgkmcnt},
            {"MaxExpcnt", Value::MaxExpcnt},
            {"SupportedSource", Value::SupportedSource},

            {"Waitcnt0Disabled", Value::Waitcnt0Disabled},
            {"SeparateVscnt", Value::SeparateVscnt},
            {"CMPXWritesSGPR", Value::CMPXWritesSGPR},
            {"HasWave32", Value::HasWave32},
            {"HasAccCD", Value::HasAccCD},
            {"ArchAccUnifiedRegs", Value::ArchAccUnifiedRegs},
            {"PackedWorkitemIDs", Value::PackedWorkitemIDs},

            {"HasXnack", Value::HasXnack},
            {"HasWave64", Value::HasWave64},
            {"DefaultWavefrontSize", Value::DefaultWavefrontSize},

            {"UnalignedVGPRs", Value::UnalignedVGPRs},
            {"UnalignedSGPRs", Value::UnalignedSGPRs},

            {"MaxLdsSize", Value::MaxLdsSize},
    };
}
