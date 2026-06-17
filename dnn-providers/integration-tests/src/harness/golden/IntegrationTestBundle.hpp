// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/BundleMetadata.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>

namespace hipdnn_integration_tests::golden
{

struct IntegrationTestBundle
{
    hipdnn_test_sdk::utilities::GraphAndTensorMap graphAndTensors;
    std::optional<hipdnn_test_sdk::utilities::BundleMetadata> metadata;
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>> goldenOutputs;
};

inline IntegrationTestBundle loadIntegrationTestBundle(const std::filesystem::path& jsonPath)
{
    IntegrationTestBundle bundle;
    bundle.metadata = hipdnn_test_sdk::utilities::loadBundleMetadata(jsonPath);
    bundle.graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(jsonPath);
    bundle.goldenOutputs = bundle.graphAndTensors.extractAndClearOutputTensorData();
    return bundle;
}

} // namespace hipdnn_integration_tests::golden
