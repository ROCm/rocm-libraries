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
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

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

/// Where the filled inputs will be consumed.
///
/// DEVICE lets a large tensor be generated straight into device memory with rocRAND
/// instead of by the host's serial RNG, which costs seconds per tensor at the largest
/// shapes. Such a tensor is device-resident afterwards, so the first non-const host
/// access migrates it; a const access cannot. HOST never touches the device, which is
/// what the unit tests rely on, since they have none.
enum class FillPlacement
{
    HOST,
    DEVICE,
};

FillResult fillInputs(const hipdnn_flatbuffers_sdk::data_objects::Graph& graph,
                      InputTensorMap& inputs,
                      const std::vector<int64_t>& ownedUids,
                      InputFillRecipes& recipes,
                      FillPlacement placement);

} // namespace hipdnn_integration_tests
