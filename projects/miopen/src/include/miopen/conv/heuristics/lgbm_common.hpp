// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_COMMON_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_COMMON_HPP

#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_predict.hpp> // LgbmEntry
#include <miopen/conv/problem_description.hpp>     // conv::Direction
#include <miopen/miopen.h>                         // miopenDataType_t

#include <cmath>
#include <string>

namespace miopen {
namespace ai {
namespace lgbm {

// Feature-encoding helpers shared by the layer-1 solver picker (lgbm_pick.cpp)
// and the layer-2 perf-config picker (lgbm_pcfg_pick.cpp). Kept in one place so
// the on-the-wire encoding cannot drift between the two callers.

// Write a numeric feature into an LgbmEntry: missing == -1 marks a NaN/absent
// feature, otherwise the value lives in fvalue.
inline void SetNumeric(LgbmEntry& e, double v)
{
    if(std::isnan(v))
        e.missing = -1;
    else
    {
        e.missing = 0;
        e.fvalue  = v;
    }
}

// Map MIOpen's conv::Direction enum to the perf-DB convention the models were
// trained on: 1=Forward, 2=BackwardData, 4=BackwardWeights.
inline int DirectionPerfDbCode(conv::Direction d)
{
    switch(d)
    {
    case conv::Direction::Forward: return 1;
    case conv::Direction::BackwardData: return 2;
    case conv::Direction::BackwardWeights: return 4;
    }
    return 1;
}

// Map MIOpen's data type to the perf-DB string used in the model vocab / bucket
// key. Only the four dtypes in the vocab are named; anything else returns ""
// (the missing category). An if-chain avoids -Wswitch-enum, which would require
// listing every miopenDataType_t value.
inline std::string DataTypeName(miopenDataType_t t)
{
    if(t == miopenHalf)
        return "fp16";
    if(t == miopenFloat)
        return "fp32";
    if(t == miopenBFloat16)
        return "bf16";
    if(t == miopenInt8)
        return "int8";
    return "";
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_COMMON_HPP
