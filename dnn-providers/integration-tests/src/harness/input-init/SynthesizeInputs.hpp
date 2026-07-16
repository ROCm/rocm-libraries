// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "harness/input-init/SynthesisConfig.hpp"

namespace hipdnn_integration_tests
{

using InputTensorMap
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

struct SynthesisResult
{
    bool filled = false;
    std::string reason;

    static SynthesisResult ok()
    {
        return {true, {}};
    }
    static SynthesisResult unsupported(std::string why)
    {
        return {false, std::move(why)};
    }
};

SynthesisResult synthesizeInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                                 InputTensorMap& inputs,
                                 const std::vector<int64_t>& ownedUids,
                                 SynthesisConfig& config);

} // namespace hipdnn_integration_tests
