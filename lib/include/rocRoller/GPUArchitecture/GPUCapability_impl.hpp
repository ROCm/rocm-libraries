
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

namespace rocRoller
{
    inline std::string GPUCapability::ToString() const
    {
        return GPUCapability::ToString(m_value);
    }

    inline std::string GPUCapability::ToString(GPUCapability::Value m_value)
    {
        for(const auto& mapping : m_stringMap)
        {
            if(mapping.second == m_value)
            {
                return mapping.first;
            }
        }
        return "";
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

            {"HasMFMA", Value::HasMFMA},
            {"HasMFMA_f64", Value::HasMFMA_f64},
            {"HasMFMA_bf16_1k", Value::HasMFMA_bf16_1k},

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

            {"HasAtomicAdd", Value::HasAtomicAdd},

            {"MaxVmcnt", Value::MaxVmcnt},
            {"MaxLgkmcnt", Value::MaxLgkmcnt},
            {"MaxExpcnt", Value::MaxExpcnt},
            {"SupportedSource", Value::SupportedSource},

            {"HasEccHalf", Value::HasEccHalf},
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
