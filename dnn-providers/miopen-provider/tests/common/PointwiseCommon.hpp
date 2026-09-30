// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>

namespace pointwise_common
{

using hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;

struct ModeCase
{
    PointwiseMode mode;
    const char* name;
};

inline const std::vector<ModeCase>& getBinaryModeCases()
{
    static const std::vector<ModeCase> s_cases = {{PointwiseMode::ADD, "Add"},
                                                  {PointwiseMode::SUB, "Sub"},
                                                  {PointwiseMode::MUL, "Mul"},
                                                  {PointwiseMode::MAX_OP, "Max"},
                                                  {PointwiseMode::MIN_OP, "Min"}};
    return s_cases;
}

} // namespace pointwise_common
