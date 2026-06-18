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

// Pre-computed expected output tensors, keyed by tensor UID. These are the
// reference values a runner compares its computed output against.
using GoldenOutputs
    = std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>;

// One test's worth of bundle data loaded from disk.
//
// Both companions are optional, for different reasons:
//   metadata     — backwards compatibility: older bundles ship without a
//                  .meta.json, and loadBundleMetadata() returns nullopt for
//                  them rather than failing.
//   goldenOutputs— not every bundle carries reference output data. A bundle may
//                  exist only to be executed (e.g. an engine smoke test) with no
//                  golden values to compare against; such a bundle loads fine
//                  but has nullopt here, and the runner must skip the compare.
struct IntegrationTestBundle
{
    hipdnn_test_sdk::utilities::GraphAndTensorMap graphAndTensors;
    std::optional<hipdnn_test_sdk::utilities::BundleMetadata> metadata;
    std::optional<GoldenOutputs> goldenOutputs;
};

// Load a bundle fully into memory. This is the "happy path" loader: it assumes
// the bundle's .bin tensor blobs are present and throws (via loadGraphAndTensors)
// if any are missing. Callers that must tolerate not-yet-fetched data (DVC)
// should classify first with checkBundlePreload() and skip before calling this —
// see the harness SetUp() for that pattern.
inline IntegrationTestBundle loadIntegrationTestBundle(const std::filesystem::path& jsonPath)
{
    IntegrationTestBundle bundle;
    bundle.metadata = hipdnn_test_sdk::utilities::loadBundleMetadata(jsonPath);
    bundle.graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(jsonPath);

    // A bundle with no declared output tensors has no golden data to extract —
    // leave goldenOutputs as nullopt so the runner knows there is nothing to
    // compare. Otherwise swap the loaded output tensors out as the reference.
    if(!bundle.graphAndTensors.outputTensorUids.empty())
    {
        bundle.goldenOutputs = bundle.graphAndTensors.extractAndClearOutputTensorData();
    }

    return bundle;
}

} // namespace hipdnn_integration_tests::golden
