// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "harness/VariantPack.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle::fixtures
{

/// One batchnorm-inference bundle on disk, as four harness suites all need it.
///
/// This graph used to be pasted into every suite that drives the harness. The
/// shape is arbitrary — one node, five leaf inputs, one output — and no test
/// asserts anything about batchnorm itself; it exists only so the harness has a
/// loadable bundle with a comparable output.
constexpr int64_t K_OUTPUT_UID = 5;
constexpr size_t K_OUTPUT_ELEMS = 120;
constexpr float K_OUTPUT_VALUE = 3.5f;

inline void writeBundleFiles(const std::filesystem::path& dir,
                             const std::string& name,
                             bool includeGoldenOutput)
{
    std::filesystem::create_directories(dir);
    std::ofstream(dir / (name + ".json"))
        << R"({"nodes": [{"inputs": {"x_tensor_uid": 0, "mean_tensor_uid": 1, )"
           R"("inv_variance_tensor_uid": 2, "scale_tensor_uid": 3, "bias_tensor_uid": 4}, )"
           R"("outputs": {"y_tensor_uid": 5}, "type": "BatchnormInferenceAttributes", )"
           R"("compute_data_type": "float", "name": ""}], "tensors": [)"
           R"({"name": "", "uid": 0, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 1, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 2, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 3, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 4, "strides": [3, 1, 1, 1], "dims": [1, 3, 1, 1], )"
           R"("data_type": "float", "virtual": false}, )"
           R"({"name": "", "uid": 5, "strides": [60, 20, 5, 1], "dims": [2, 3, 4, 5], )"
           R"("data_type": "float", "virtual": false}], "io_data_type": "float", )"
           R"("compute_data_type": "float", "intermediate_data_type": "float", "name": ""})";

    std::ofstream(dir / (name + ".meta.json"))
        << R"({"format_version": 1, "operation": "BatchnormInference"})";

    const auto basePath = (dir / name).string();
    const auto writeFloatBin = [&](int64_t uid, size_t elems, float value) {
        const std::vector<float> data(elems, value);
        std::ofstream out(basePath + ".tensor" + std::to_string(uid) + ".bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    };

    writeFloatBin(0, 120, 0.0f); // x
    writeFloatBin(1, 3, 0.0f); // mean
    writeFloatBin(2, 3, 0.0f); // inv_variance
    writeFloatBin(3, 3, 0.0f); // scale
    writeFloatBin(4, 3, 0.0f); // bias

    if(includeGoldenOutput)
    {
        writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, K_OUTPUT_VALUE); // y (golden)
    }
}

/// Writes the bundle under `parentDir/name` and loads it.
inline std::shared_ptr<IntegrationTestBundle> loadBundle(const std::filesystem::path& parentDir,
                                                         const std::string& name,
                                                         bool includeGoldenOutput)
{
    const auto dir = parentDir / name;
    writeBundleFiles(dir, name, includeGoldenOutput);

    auto result = loadIntegrationTestBundle(dir / (name + ".json"));
    EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
    return std::make_shared<IntegrationTestBundle>(
        std::move(std::get<IntegrationTestBundle>(result)));
}

/// Fills the bundle's single output uid in `variantPack` with `value`.
inline void writeOutput(VariantPack& variantPack, float value)
{
    auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
    std::fill(ptr, ptr + K_OUTPUT_ELEMS, value);
}

} // namespace hipdnn_integration_tests::bundle::fixtures
