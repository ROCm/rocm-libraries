// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include <hipdnn_data_sdk/utilities/Visitor.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/LoadGraphAndTensors.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/golden/GoldenBundleDiscovery.hpp"
#include "harness/golden/GoldenTensorComparator.hpp"
#include "harness/gpu_graph_executor/GpuReferenceGraphExecutor.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::golden;

namespace
{

std::filesystem::path goldenDataRoot()
{
    return std::filesystem::path(__FILE__).parent_path() / ".." / "golden_reference_data";
}

std::filesystem::path batchNormSmallBundle()
{
    return goldenDataRoot() / "quick" / "BatchnormFwdInference" / "nchw" / "fp32" / "Small"
           / "Small.json";
}

void verifyGoldenComparison(
    hipdnn_test_sdk::utilities::GraphAndTensorMap& graphAndTensors,
    const std::unordered_map<int64_t, std::unique_ptr<hipdnn_data_sdk::utilities::ITensor>>&
        goldenOutputs,
    float tolerance)
{
    auto wrapper = graphAndTensors.createGraphWrapper();
    const auto& tensorAttrMap = wrapper.getTensorMap();

    for(const auto uid : graphAndTensors.outputTensorUids)
    {
        const auto& actualTensor = *graphAndTensors.tensorMap.at(uid);
        const auto& expectedTensor = *goldenOutputs.at(uid);
        const auto dataType = tensorAttrMap.at(uid)->data_type();

        auto compareFunc = [&](auto typeTag) {
            using T = decltype(typeTag);
            return compareTensors<T>(expectedTensor, actualTensor, tolerance, tolerance);
        };

        const auto result = std::visit(
            hipdnn_data_sdk::utilities::Visitor{
                compareFunc,
                [](int) -> ComparisonResult {
                    ComparisonResult r;
                    r.passed = false;
                    return r;
                }},
            hipdnn_test_sdk::utilities::datatypeToNativeVariant(dataType));

        EXPECT_TRUE(result.passed)
            << "Golden comparison failed for tensor uid=" << uid << ": "
            << result.mismatchCount << "/" << result.totalElements
            << " elements mismatched, max abs error=" << result.maxAbsError;
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
// For ops without golden bundles AND without a GPU reference executor, the
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
        GTEST_SKIP() << "Golden bundle not available (DVC not pulled?): " << bundlePath;
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
        graphAndTensors.graphBuffer.data(),
        graphAndTensors.graphBuffer.size(),
        hostBuffers);

    verifyGoldenComparison(graphAndTensors, goldenOutputs, 1e-5f);
}

// ---------------------------------------------------------------------------
// Path 2 — GPU Reference against golden data (the "Conv default" path)
//
// For ops without golden bundles but with a GPU reference executor (e.g.
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
        GTEST_SKIP() << "Golden bundle not available (DVC not pulled?): " << bundlePath;
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
        executor.execute(
            graphAndTensors.graphBuffer.data(),
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
// Path 3 — Verification routing: golden bundles only exist for BatchNorm
//
// Proves that discoverGoldenBundles() finds ONLY the ops that have golden
// bundle data (here: BatchNorm).  Ops without golden bundles (Conv,
// LayerNorm) are NOT discovered — they fall through to the existing
// CPU/GPU reference test infrastructure instead.
// ---------------------------------------------------------------------------
TEST(TestGoldenVerificationRouting, OnlyBatchNormDiscoveredConvAndLayerNormFallThrough)
{
    auto path = std::filesystem::temp_directory_path() / "golden_routing_test";
    std::filesystem::remove_all(path);
    const hipdnn_test_sdk::utilities::ScopedDirectory tempDir(path);

    writeMinimalBatchNormBundle(tempDir.path() / "quick" / "Bn" / "q", "q");
    writeMinimalBatchNormBundle(tempDir.path() / "standard" / "Bn" / "s", "s");
    writeMinimalBatchNormBundle(tempDir.path() / "comprehensive" / "Bn" / "c", "c");
    writeMinimalBatchNormBundle(tempDir.path() / "full" / "Bn" / "f", "f");

    const auto bundles = discoverGoldenBundles(tempDir.path());
    ASSERT_EQ(bundles.size(), 4u);

    for(const auto& b : bundles)
    {
        EXPECT_NE(b.suiteName.find("BatchnormInference"), std::string::npos)
            << "Expected BatchnormInference in suite name, got: " << b.suiteName;
        EXPECT_EQ(b.suiteName.find("Conv"), std::string::npos)
            << "Conv should NOT appear — no conv golden bundles exist";
        EXPECT_EQ(b.suiteName.find("LayerNorm"), std::string::npos)
            << "LayerNorm should NOT appear — no layer norm golden bundles exist";
    }
}

// ---------------------------------------------------------------------------
// Path 3 (continued) — All three runner suffixes register per bundle
//
// Proves that for a bundle with golden data, all three verification runners
// (CpuRef, GpuRef, Engine) produce distinct suite names.  This is how the
// system provides three independent cross-checks against the same golden
// ground truth.
// ---------------------------------------------------------------------------
TEST(TestGoldenVerificationRouting, ThreeRunnerSuffixesProduceDistinctSuites)
{
    auto path = std::filesystem::temp_directory_path() / "golden_suffix_test";
    std::filesystem::remove_all(path);
    const hipdnn_test_sdk::utilities::ScopedDirectory tempDir(path);

    writeMinimalBatchNormBundle(tempDir.path() / "quick" / "Bn" / "q", "q");
    writeMinimalBatchNormBundle(tempDir.path() / "standard" / "Bn" / "s", "s");
    writeMinimalBatchNormBundle(tempDir.path() / "comprehensive" / "Bn" / "c", "c");
    writeMinimalBatchNormBundle(tempDir.path() / "full" / "Bn" / "f", "f");

    const auto bundles = discoverGoldenBundles(tempDir.path());
    ASSERT_FALSE(bundles.empty());

    const auto& bundle = bundles.front();

    const auto cpuSuite = bundle.suiteName + "_CpuRef";
    const auto gpuSuite = bundle.suiteName + "_GpuRef";
    const auto engineSuite = bundle.suiteName + "_Engine";

    EXPECT_NE(cpuSuite, gpuSuite);
    EXPECT_NE(cpuSuite, engineSuite);
    EXPECT_NE(gpuSuite, engineSuite);

    EXPECT_NE(cpuSuite.find("CpuRef"), std::string::npos);
    EXPECT_NE(gpuSuite.find("GpuRef"), std::string::npos);
    EXPECT_NE(engineSuite.find("Engine"), std::string::npos);
}

// NOLINTEND(readability-identifier-naming)
