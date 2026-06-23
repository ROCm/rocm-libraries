// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Tests the TOML tolerance override and test_skips wiring in the golden harness:
//
//   1. lookupToleranceOverride replaces atol/rtol when matched
//   2. Per-op default (priority 2) is used when override returns nullopt
//   3. lookupSkip skips the test when matched
//   4. Test runs normally when lookupSkip returns nullopt

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/TestConfig.hpp"
#include "harness/golden/IntegrationGraphGoldenReferenceVerificationHarness.hpp"
#include "harness/golden/IntegrationTestBundle.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::golden;

namespace
{

using EngineStub = std::function<void(std::unordered_map<int64_t, void*>&)>;

class ToleranceTestableHarness : public IntegrationGraphGoldenReferenceVerificationHarness
{
public:
    ToleranceTestableHarness(EngineStub engineStub,
                             std::optional<ToleranceOverride> tolOverride,
                             std::optional<std::string> skipReason)
        : IntegrationGraphGoldenReferenceVerificationHarness(/*requiresDevice=*/false)
        , _engineStub(std::move(engineStub))
        , _tolOverride(std::move(tolOverride))
        , _skipReason(std::move(skipReason))
    {
    }

    using IntegrationGraphGoldenReferenceVerificationHarness::SetUp;
    using IntegrationGraphGoldenReferenceVerificationHarness::TestBody;

protected:
    VerificationMode getVerificationMode() const override
    {
        return VerificationMode::GOLDEN;
    }

    void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& variantPack) override
    {
        _engineStub(variantPack);
    }

    void runReferenceExecutor(ReferenceExecutorType /*type*/,
                              std::unordered_map<int64_t, void*>& /*variantPack*/) override
    {
    }

    std::unique_ptr<IReferenceGraphExecutor>
        makeReferenceExecutor(ReferenceExecutorType /*type*/) override
    {
        return nullptr;
    }

    void applyMetadataGuards() const override {}

    std::optional<std::string> lookupSkip(const std::string& /*testName*/) const override
    {
        return _skipReason;
    }

    std::optional<ToleranceOverride>
        lookupToleranceOverride(const std::string& /*testName*/) const override
    {
        return _tolOverride;
    }

private:
    EngineStub _engineStub;
    std::optional<ToleranceOverride> _tolOverride;
    std::optional<std::string> _skipReason;
};

class TestGoldenToleranceAndSkips : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    static constexpr float K_OUTPUT_VALUE = 3.5f;
    static constexpr int64_t K_OUTPUT_UID = 5;
    static constexpr size_t K_OUTPUT_ELEMS = 120;

    void SetUp() override
    {
        auto path
            = std::filesystem::temp_directory_path()
              / ("tol_skip_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    static void writeBundleFiles(const std::filesystem::path& dir,
                                 const std::string& name,
                                 float goldenValue)
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

        writeFloatBin(0, 120, 0.0f);
        writeFloatBin(1, 3, 0.0f);
        writeFloatBin(2, 3, 0.0f);
        writeFloatBin(3, 3, 0.0f);
        writeFloatBin(4, 3, 0.0f);
        writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, goldenValue);
    }

    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      float goldenValue) const
    {
        const auto dir = _tempDir / name;
        writeBundleFiles(dir, name, goldenValue);
        auto result = loadIntegrationTestBundle(dir / (name + ".json"));
        EXPECT_TRUE(std::holds_alternative<IntegrationTestBundle>(result));
        return std::make_shared<IntegrationTestBundle>(
            std::move(std::get<IntegrationTestBundle>(result)));
    }

    static void writeOutput(std::unordered_map<int64_t, void*>& variantPack, float value)
    {
        auto* ptr = static_cast<float*>(variantPack.at(K_OUTPUT_UID));
        std::fill(ptr, ptr + K_OUTPUT_ELEMS, value);
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

    static void runCapturing(std::shared_ptr<IntegrationTestBundle> bundle,
                             EngineStub engineStub,
                             std::optional<ToleranceOverride> tolOverride,
                             std::optional<std::string> skipReason,
                             ::testing::TestPartResultArray* results)
    {
        ToleranceTestableHarness harness(
            std::move(engineStub), std::move(tolOverride), std::move(skipReason));
        harness.setBundle(std::move(bundle), "tol-skip-test-bundle");

        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        if(!anySkipped(*results))
        {
            harness.TestBody();
        }
    }
};

// Engine writes output that differs from golden by 0.05.
// BN inference fp32 default tolerance is 2e-4 — so this FAILS without override.
// A TOML override with atol=0.1 makes it pass.
TEST_F(TestGoldenToleranceAndSkips, ToleranceOverrideApplied)
{
    constexpr float goldenValue = 1.0f;
    constexpr float engineValue = 1.05f;
    auto bundle = loadBundle("tol_override", goldenValue);

    ::testing::TestPartResultArray results;
    runCapturing(
        bundle,
        [](auto& vp) { writeOutput(vp, engineValue); },
        ToleranceOverride{0.1f, 0.1f},
        std::nullopt,
        &results);

    EXPECT_FALSE(anyFailed(results)) << "Should pass with the relaxed TOML override tolerance";
}

// Same scenario but lookupToleranceOverride returns nullopt.
// The per-op default (BN inference fp32 = 2e-4) is used, so the 0.05 diff FAILS.
TEST_F(TestGoldenToleranceAndSkips, DefaultToleranceUsedWhenNoOverride)
{
    constexpr float goldenValue = 1.0f;
    constexpr float engineValue = 1.05f;
    auto bundle = loadBundle("tol_default", goldenValue);

    ::testing::TestPartResultArray results;
    runCapturing(
        bundle,
        [](auto& vp) { writeOutput(vp, engineValue); },
        std::nullopt,
        std::nullopt,
        &results);

    EXPECT_TRUE(anyFailed(results)) << "Should fail with the tight per-op default tolerance";
}

// lookupSkip returns a reason string — SetUp() should GTEST_SKIP.
TEST_F(TestGoldenToleranceAndSkips, SkipApplied)
{
    auto bundle = loadBundle("skip_test", 1.0f);

    ::testing::TestPartResultArray results;
    runCapturing(
        bundle,
        [](auto& /*vp*/) {},
        std::nullopt,
        std::string("known failure on gfx1100"),
        &results);

    EXPECT_TRUE(anySkipped(results)) << "Test should be skipped when lookupSkip returns a reason";
}

// lookupSkip returns nullopt — test runs normally (and passes because engine matches golden).
TEST_F(TestGoldenToleranceAndSkips, NoSkipRunsNormally)
{
    constexpr float value = 1.0f;
    auto bundle = loadBundle("no_skip", value);

    ::testing::TestPartResultArray results;
    runCapturing(
        bundle, [](auto& vp) { writeOutput(vp, value); }, std::nullopt, std::nullopt, &results);

    EXPECT_FALSE(anyFailed(results)) << "Test should run and pass when no skip is set";
    EXPECT_FALSE(anySkipped(results)) << "Test should not be skipped when lookupSkip is nullopt";
}

} // namespace

// NOLINTEND(readability-identifier-naming)
