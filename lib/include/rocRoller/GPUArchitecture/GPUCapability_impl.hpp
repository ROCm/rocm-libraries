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

    inline std::ostream& operator<<(std::ostream& stream, GPUCapability::Value v)
    {
        return stream << toString(v);
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
            {"HasWMMA_f8", Value::HasWMMA_f8},

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

            {"UnalignedVGPRs", Value::UnalignedVGPRs},
            {"UnalignedSGPRs", Value::UnalignedSGPRs},

            {"MaxLdsSize", Value::MaxLdsSize},

            {"HasNaNoo", Value::HasNaNoo},

            {"HasDSReadTransposeB16", Value::HasDSReadTransposeB16},
            {"HasDSReadTransposeB8", Value::HasDSReadTransposeB8},
            {"HasDSReadTransposeB6", Value::HasDSReadTransposeB6},
            {"HasDSReadTransposeB4", Value::HasDSReadTransposeB4},

            {"HasPRNG", Value::HasPRNG},

            {"HasPermLanes16", Value::HasPermLanes16},
            {"HasPermLanes32", Value::HasPermLanes32},

            {"WorkgroupIdxViaTTMP", Value::WorkgroupIdxViaTTMP},
            {"HasBufferOutOfBoundsCheckOption", Value::HasBufferOutOfBoundsCheckOption},
    };
}
