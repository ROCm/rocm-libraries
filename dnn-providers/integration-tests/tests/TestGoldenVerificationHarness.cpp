// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Unit tests for IntegrationGraphGoldenReferenceVerificationHarness's core
// contract: how it translates an executor's behaviour into a GTest outcome.
//
//   executor throws (unsupported graph) -> SKIP
//   executor writes matching output     -> PASS (no failure recorded)
//   executor writes mismatching output  -> FAIL
//
// The harness reports these via GTest itself (GTEST_SKIP / EXPECT_TRUE), so we
// drive it under a ScopedFakeTestPartResultReporter and inspect the captured
// TestPartResults rather than letting them affect *this* test.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests::golden;

namespace
{

// Exposes the harness's protected SetUp/TestBody so a test can drive the full
// lifecycle directly. requiresDevice is forced false: these tests run on CPU-only
// CI and the stub executors never touch the GPU.
class TestableHarness : public IntegrationGraphGoldenReferenceVerificationHarness
{
public:
    explicit TestableHarness(ExecuteFunc executor)
        : IntegrationGraphGoldenReferenceVerificationHarness(std::move(executor),
                                                             /*requiresDevice=*/false)
    {
    }

    using IntegrationGraphGoldenReferenceVerificationHarness::SetUp;
    using IntegrationGraphGoldenReferenceVerificationHarness::TestBody;
};

class TestGoldenHarnessFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        auto path
            = std::filesystem::temp_directory_path()
              / ("golden_harness_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    // The float value every output element is seeded with on disk, so "matching"
    // vs "mismatching" executor output is unambiguous (not the all-zero buffer
    // that the harness already zeroes outputs to before running).
    static constexpr float K_OUTPUT_VALUE = 3.5f;

    // uid 5 (the batchnorm y output): dims [2,3,4,5] -> 120 floats.
    static constexpr int64_t K_OUTPUT_UID = 5;
    static constexpr size_t K_OUTPUT_ELEMS = 120;

    // Writes a schema-valid nchw/fp32 batchnorm-inference graph, its metadata,
    // and all tensor .bin blobs. Inputs are zero-filled; the output blob (uid 5)
    // is filled with K_OUTPUT_VALUE so it is a distinct, non-zero golden.
    static void writeBundleFiles(const std::filesystem::path& dir, const std::string& name)
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
            std::ofstream out(basePath + ".tensor" + std::to_string(uid) + ".bin",
                              std::ios::binary);
            out.write(reinterpret_cast<const char*>(data.data()),
                      static_cast<std::streamsize>(data.size() * sizeof(float)));
        };

        writeFloatBin(0, 120, 0.0f); // x
        writeFloatBin(1, 3, 0.0f); // mean
        writeFloatBin(2, 3, 0.0f); // inv_variance
        writeFloatBin(3, 3, 0.0f); // scale
        writeFloatBin(4, 3, 0.0f); // bias
        writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, K_OUTPUT_VALUE); // y (golden)
    }

    // Loads a fully-populated bundle (graph + tensors + metadata) ready to run.
    std::shared_ptr<IntegrationTestBundle> loadRunnableBundle(const std::string& name) const
    {
        const auto dir = _tempDir / name;
        writeBundleFiles(dir, name);
        auto result = loadIntegrationTestBundle(dir / (name + ".json"));
        EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
        return std::make_shared<IntegrationTestBundle>(
            std::move(std::get<IntegrationTestBundle>(result)));
    }

    // Fills the output tensor (uid 5) in the live tensor map with `value`. A
    // stand-in for what a real executor would compute into the output buffer.
    static void writeOutput(IntegrationTestBundle& bundle, float value)
    {
        auto& outTensor = *bundle.tensors->at(K_OUTPUT_UID);
        const std::vector<float> data(K_OUTPUT_ELEMS, value);
        outTensor.fillWithData(data.data(), data.size() * sizeof(float));
    }

    // Drives the harness's full lifecycle (SetUp then TestBody) for the given
    // bundle + executor, capturing every TestPartResult the harness records.
    static void
        runCapturing(std::shared_ptr<IntegrationTestBundle> bundle,
                     IntegrationGraphGoldenReferenceVerificationHarness::ExecuteFunc executor,
                     ::testing::TestPartResultArray* results)
    {
        TestableHarness harness(std::move(executor));
        harness.setBundle(std::move(bundle), "unit-test-bundle");

        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        harness.TestBody();
    }

    static bool anySkipped(const ::testing::TestPartResultArray& results)
    {
        for(int i = 0; i < results.size(); ++i)
        {
            if(results.GetTestPartResult(i).skipped())
            {
                return true;
            }
        }
        return false;
    }

    static bool anyFailed(const ::testing::TestPartResultArray& results)
    {
        for(int i = 0; i < results.size(); ++i)
        {
            if(results.GetTestPartResult(i).failed())
            {
                return true;
            }
        }
        return false;
    }
};

} // namespace

// An executor that throws ("unsupported graph") must yield a SKIP, not a FAIL —
// the harness translates the throw into GTEST_SKIP.
TEST_F(TestGoldenHarnessFixture, ExecutorThrowsYieldsSkip)
{
    ::testing::TestPartResultArray results;
    runCapturing(
        loadRunnableBundle("throws"),
        [](IntegrationTestBundle&) {
            throw std::runtime_error("engine does not support this graph");
        },
        &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_FALSE(anyFailed(results));
}

// An executor that writes output matching the golden reference must yield a
// PASS: no failure and no skip recorded.
TEST_F(TestGoldenHarnessFixture, MatchingOutputYieldsPass)
{
    ::testing::TestPartResultArray results;
    runCapturing(
        loadRunnableBundle("match"),
        [](IntegrationTestBundle& bundle) { writeOutput(bundle, K_OUTPUT_VALUE); },
        &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

// An executor that writes output differing from the golden reference must yield
// a FAIL.
TEST_F(TestGoldenHarnessFixture, MismatchingOutputYieldsFail)
{
    ::testing::TestPartResultArray results;
    runCapturing(
        loadRunnableBundle("mismatch"),
        [](IntegrationTestBundle& bundle) { writeOutput(bundle, K_OUTPUT_VALUE + 100.0f); },
        &results);

    EXPECT_TRUE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

// NOLINTEND(readability-identifier-naming)
