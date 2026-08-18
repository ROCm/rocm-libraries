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

#include "harness/input-init/InputFillRecipes.hpp"

namespace hipdnn_integration_tests
{

using InputTensorMap
    = std::unordered_map<int64_t, std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>;

struct FillResult
{
    bool filled = false;
    std::string reason;

    static FillResult ok()
    {
        return {true, {}};
    }
    static FillResult unsupported(std::string why)
    {
        return {false, std::move(why)};
    }
};

FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes);

} // namespace hipdnn_integration_tests
