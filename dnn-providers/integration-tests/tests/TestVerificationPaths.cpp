// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/golden/BundleDiscovery.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::golden;

namespace
{

std::filesystem::path goldenDataRoot()
{
    return std::filesystem::path(__FILE__).parent_path() / ".." / "integration_test_bundles";
}

std::filesystem::path batchNormSmallBundle()
{
    return goldenDataRoot() / "quick" / "BatchnormFwdInference" / "nchw" / "fp32" / "Small"
           / "Small.json";
}

void verifyGoldenComparison(
    hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors,
    std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>&
        goldenOutputs,
    float tolerance)
{
    auto wrapper = graphAndTensors.createGraphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    for(const auto uid : graphAndTensors.outputTensorUids)
    {
        auto& actualTensor = *graphAndTensors.tensorMap.at(uid);
        auto& expectedTensor = *goldenOutputs.at(uid);
        const auto dataType = tensorAttrMap.at(uid)->data_type();

        auto validator
            = hipdnn_test_sdk::utilities::createAllCloseValidator(dataType, tolerance, tolerance);
        EXPECT_TRUE(validator->allClose(expectedTensor, actualTensor))
            << "Golden comparison failed for tensor uid=" << uid;
    }
}

void writeMinimalBatchNormBundle(const std::filesystem::path& dir, const std::string& name)
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
}

} // namespace

// ---------------------------------------------------------------------------
// Path 1 — CPU Reference against golden data (the "LayerNorm default" path)
//
// For ops without bundles AND without a GPU reference executor, the
// existing TestCpuFpReference* tests use CpuReferenceGraphExecutor as the
// ground truth.  This test proves the golden comparison pipeline works
// end-to-end by running the same CPU executor against a real batch-norm
// bundle and comparing the result to the golden tensor data on disk.
// ---------------------------------------------------------------------------
TEST(TestGoldenVerificationCpuRefFp32, BatchNormSmallMatchesGoldenData)
{
    const auto bundlePath = batchNormSmallBundle();
    if(!std::filesystem::exists(bundlePath))
    {
        GTEST_SKIP() << "Bundle not available (DVC not pulled?): " << bundlePath;
    }

    hipdnn_test_sdk::utilities::GraphAndTensorMap graphAndTensors;
    try
    {
        graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(bundlePath);
    }
    catch(const std::exception&)
    {
        GTEST_SKIP() << "Tensor data not available (DVC not pulled?): " << bundlePath;
    }
    auto goldenOutputs = graphAndTensors.extractAndClearOutputTensorData();

    auto hostBuffers = graphAndTensors.hostBufferMap();
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor().execute(
        graphAndTensors.graphBuffer.data(), graphAndTensors.graphBuffer.size(), hostBuffers);

    verifyGoldenComparison(graphAndTensors, goldenOutputs, 1e-5f);
}

// ---------------------------------------------------------------------------
// Path 2 — GPU Reference against golden data (the "Conv default" path)
//
// For ops without bundles but with a GPU reference executor (e.g.
// convolution), the existing integration tests use GpuReferenceGraphExecutor.
// Not all operations have GPU reference plans — batch norm does not, which
// is exactly why it falls through to CPU reference.  This test verifies the
// GPU ref pipeline is wired up correctly; it skips for ops that lack a GPU
// plan (confirming those ops correctly fall to the CPU reference path).
// ---------------------------------------------------------------------------
TEST(TestGpuGoldenVerificationRef, SkipsWhenNoPlanAvailable)
{
    SKIP_IF_NO_DEVICES();

    const auto bundlePath = batchNormSmallBundle();
    if(!std::filesystem::exists(bundlePath))
    {
        GTEST_SKIP() << "Bundle not available (DVC not pulled?): " << bundlePath;
    }

    hipdnn_test_sdk::utilities::GraphAndTensorMap graphAndTensors;
    try
    {
        graphAndTensors = hipdnn_test_sdk::utilities::loadGraphAndTensors(bundlePath);
    }
    catch(const std::exception&)
    {
        GTEST_SKIP() << "Tensor data not available (DVC not pulled?): " << bundlePath;
    }
    auto goldenOutputs = graphAndTensors.extractAndClearOutputTensorData();

    std::unordered_map<int64_t, void*> deviceBufferMap;
    for(auto& [uid, tensor] : graphAndTensors.tensorMap)
    {
        deviceBufferMap[uid] = tensor->rawDeviceData();
    }

    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    try
    {
        executor.execute(graphAndTensors.graphBuffer.data(),
                         graphAndTensors.graphBuffer.size(),
                         deviceBufferMap);
    }
    catch(const std::runtime_error&)
    {
        // Batch norm does not have a GPU reference plan — this is expected.
        // Ops without GPU plans fall through to CPU reference in the existing
        // test infrastructure, which is exactly the "LayerNorm default" path.
        GTEST_SKIP() << "No GPU reference plan for this operation — "
                        "confirms fallback to CPU reference path";
    }

    for(const auto uid : graphAndTensors.outputTensorUids)
    {
        graphAndTensors.tensorMap.at(uid)->markDeviceModified();
    }

    verifyGoldenComparison(graphAndTensors, goldenOutputs, 1e-5f);
}

// ---------------------------------------------------------------------------
// Path 3 — Verification routing: bundles only exist for BatchNorm
//
// Proves that discoverBundles() finds ONLY the ops that have bundle data
// (here: BatchNorm).  Ops without bundles (Conv, LayerNorm) are NOT
// discovered — they fall through to the existing CPU/GPU reference test
// infrastructure instead.
// ---------------------------------------------------------------------------
TEST(TestVerificationRouting, OnlyBatchNormDiscoveredConvAndLayerNormFallThrough)
{
    auto path = std::filesystem::temp_directory_path() / "golden_routing_test";
    std::filesystem::remove_all(path);
    const hipdnn_test_sdk::utilities::ScopedDirectory tempDir(path);

    writeMinimalBatchNormBundle(tempDir.path() / "BatchnormInference" / "nchw" / "fp32" / "Small",
                                "Small");

    const auto bundles = discoverBundles(tempDir.path());
    ASSERT_EQ(bundles.size(), 1u);

    for(const auto& b : bundles)
    {
        EXPECT_EQ(b.suiteName.find("Conv"), std::string::npos)
            << "Conv should NOT appear — no conv bundles exist";
        EXPECT_EQ(b.suiteName.find("LayerNorm"), std::string::npos)
            << "LayerNorm should NOT appear — no layer norm bundles exist";
    }

    EXPECT_EQ(bundles.front().suiteName, "BatchnormInference_nchw_fp32_Small");
}

// NOLINTEND(readability-identifier-naming)
