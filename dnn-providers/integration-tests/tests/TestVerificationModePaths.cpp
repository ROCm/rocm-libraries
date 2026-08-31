// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Verification mode dispatch: AUTO/GOLDEN/GPU/CPU paths.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "harness/ReferenceCapabilityError.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

namespace
{

using EngineStub = std::function<void(std::unordered_map<int64_t, void*>&)>;
using RefStub = std::function<void(ReferenceExecutorType, std::unordered_map<int64_t, void*>&)>;

class ModeTestableHarness : public IntegrationBundleVerificationHarness
{
public:
    ModeTestableHarness(VerificationMode mode, EngineStub engineStub, RefStub refStub)
        : IntegrationBundleVerificationHarness(/*requiresDevice=*/false)
        , _mode(mode)
        , _engineStub(std::move(engineStub))
        , _refStub(std::move(refStub))
    {
    }

    using IntegrationBundleVerificationHarness::SetUp;
    using IntegrationBundleVerificationHarness::TestBody;

protected:
    VerificationMode getVerificationMode() const override
    {
        return _mode;
    }

    void executeGraphThroughEngine(std::unordered_map<int64_t, void*>& variantPack) override
    {
        _engineStub(variantPack);
    }

    void runReferenceExecutor(ReferenceExecutorType type,
                              std::unordered_map<int64_t, void*>& variantPack) override
    {
        _refStub(type, variantPack);
    }

    std::unique_ptr<IReferenceGraphExecutor>
        makeReferenceExecutor(ReferenceExecutorType /*type*/) override
    {
        return nullptr;
    }

    bool isEnforcingSupportClaims() const override
    {
        return false;
    }

    void applyMetadataGuards() const override {}

    // Record the routing decision instead of running it. The real implementation
    // calls getSharedHandle(), which this deviceless harness must not reach.
    // skipUnverifiable() is how the real enforceAtLevel() exits when it cannot
    // verify; using it here keeps TestBody()'s "verified nothing" guard satisfied,
    // which a bare GTEST_SKIP() would not once the reporter intercepts the skip.
    // Assertions read enforcedLevel(), so they prove routing specifically.
    void enforceAtLevel(EnforcementLevel level) override
    {
        _enforcedLevel = level;
        skipUnverifiable("enforceAtLevel stubbed (deviceless)");
    }

public:
    std::optional<EnforcementLevel> enforcedLevel() const
    {
        return _enforcedLevel;
    }

private:
    VerificationMode _mode;
    EngineStub _engineStub;
    RefStub _refStub;
    std::optional<EnforcementLevel> _enforcedLevel;
};

class TestVerificationModePathsFixture : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        auto path
            = std::filesystem::temp_directory_path()
              / ("vmode_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    static constexpr float K_OUTPUT_VALUE = 3.5f;
    static constexpr int64_t K_OUTPUT_UID = 5;
    static constexpr size_t K_OUTPUT_ELEMS = 120;

    static void writeBundleFiles(const std::filesystem::path& dir,
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

        if(includeGoldenOutput)
        {
            writeFloatBin(K_OUTPUT_UID, K_OUTPUT_ELEMS, K_OUTPUT_VALUE);
        }
    }

    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      bool includeGoldenOutput) const
    {
        const auto dir = _tempDir / name;
        writeBundleFiles(dir, name, includeGoldenOutput);
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

    static void runCapturing(std::shared_ptr<IntegrationTestBundle> bundle,
                             VerificationMode mode,
                             EngineStub engineStub,
                             RefStub refStub,
                             ::testing::TestPartResultArray* results)
    {
        ModeTestableHarness harness(mode, std::move(engineStub), std::move(refStub));
        harness.setBundle(std::move(bundle), "vmode-test-bundle");

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

    static EngineStub matchingEngine()
    {
        return [](std::unordered_map<int64_t, void*>& vp) { writeOutput(vp, K_OUTPUT_VALUE); };
    }

    static EngineStub mismatchingEngine()
    {
        return [](std::unordered_map<int64_t, void*>& vp) {
            writeOutput(vp, K_OUTPUT_VALUE + 100.0f);
        };
    }

    static RefStub matchingRef()
    {
        return [](ReferenceExecutorType, std::unordered_map<int64_t, void*>& vp) {
            writeOutput(vp, K_OUTPUT_VALUE);
        };
    }

    static RefStub capabilityMissRef()
    {
        return [](ReferenceExecutorType, std::unordered_map<int64_t, void*>&) {
            throw ReferenceCapabilityError("stub: unsupported op");
        };
    }

    static RefStub gpuMissCpuMatchRef()
    {
        return [](ReferenceExecutorType type, std::unordered_map<int64_t, void*>& vp) {
            if(type == ReferenceExecutorType::GPU)
            {
                throw ReferenceCapabilityError("stub: no GPU ref plan");
            }
            writeOutput(vp, K_OUTPUT_VALUE);
        };
    }
};

} // namespace

// ── AUTO mode ───────────────────────────────────────────────────────────────

TEST_F(TestVerificationModePathsFixture, AutoWithGoldenUsesGoldenAndPasses)
{
    ::testing::TestPartResultArray results;
    bool refCalled = false;
    runCapturing(
        loadBundle("auto_golden", /*includeGoldenOutput=*/true),
        VerificationMode::AUTO,
        matchingEngine(),
        [&](ReferenceExecutorType, std::unordered_map<int64_t, void*>&) { refCalled = true; },
        &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
    EXPECT_FALSE(refCalled) << "Reference executor should NOT run when golden data is present";
}

TEST_F(TestVerificationModePathsFixture, AutoWithGoldenMismatchFails)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_golden_mm", /*includeGoldenOutput=*/true),
                 VerificationMode::AUTO,
                 mismatchingEngine(),
                 matchingRef(),
                 &results);

    EXPECT_TRUE(anyFailed(results));
}

TEST_F(TestVerificationModePathsFixture, AutoNoGoldenRefSucceedsPasses)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_gpu", /*includeGoldenOutput=*/false),
                 VerificationMode::AUTO,
                 matchingEngine(),
                 matchingRef(),
                 &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, AutoNoGoldenRefMissFallsThroughToCpu)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_fallthrough", /*includeGoldenOutput=*/false),
                 VerificationMode::AUTO,
                 matchingEngine(),
                 gpuMissCpuMatchRef(),
                 &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, AutoNoGoldenBothRefsMissSkips)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("auto_both_miss", /*includeGoldenOutput=*/false),
                 VerificationMode::AUTO,
                 matchingEngine(),
                 capabilityMissRef(),
                 &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_FALSE(anyFailed(results));
}

// ── GOLDEN mode ─────────────────────────────────────────────────────────────

TEST_F(TestVerificationModePathsFixture, GoldenModeWithDataPasses)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("golden_ok", /*includeGoldenOutput=*/true),
                 VerificationMode::GOLDEN,
                 matchingEngine(),
                 capabilityMissRef(),
                 &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, GoldenModeWithoutDataSkips)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("golden_absent", /*includeGoldenOutput=*/false),
                 VerificationMode::GOLDEN,
                 matchingEngine(),
                 matchingRef(),
                 &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_FALSE(anyFailed(results));
}

// ── Explicit GPU mode ───────────────────────────────────────────────────────
// "Device" in these case names denotes VerificationMode::GPU (the device-side
// reference executor). The literal "Gpu" keyword is reserved by the test-name
// linter for the suite name and so cannot appear in the case name.

TEST_F(TestVerificationModePathsFixture, DeviceModeRefSucceedsPasses)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("gpu_ok", /*includeGoldenOutput=*/true),
                 VerificationMode::GPU,
                 matchingEngine(),
                 matchingRef(),
                 &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, DeviceModeCapabilityMissSkips)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("gpu_miss", /*includeGoldenOutput=*/true),
                 VerificationMode::GPU,
                 matchingEngine(),
                 capabilityMissRef(),
                 &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_FALSE(anyFailed(results));
}

// ── Explicit CPU mode ───────────────────────────────────────────────────────

TEST_F(TestVerificationModePathsFixture, CpuModeRefSucceedsPasses)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("cpu_ok", /*includeGoldenOutput=*/true),
                 VerificationMode::CPU,
                 matchingEngine(),
                 matchingRef(),
                 &results);

    EXPECT_FALSE(anyFailed(results));
    EXPECT_FALSE(anySkipped(results));
}

TEST_F(TestVerificationModePathsFixture, CpuModeCapabilityMissSkips)
{
    ::testing::TestPartResultArray results;
    runCapturing(loadBundle("cpu_miss", /*includeGoldenOutput=*/true),
                 VerificationMode::CPU,
                 matchingEngine(),
                 capabilityMissRef(),
                 &results);

    EXPECT_TRUE(anySkipped(results));
    EXPECT_FALSE(anyFailed(results));
}

// ── Enforcement-level gate ──────────────────────────────────────────────────
// runComparison() routes on enforcement level alone. The --enforce-support-claims
// flag controls what happens *inside* enforceAtLevel(), not whether it runs, so a
// non-FULL bundle must reach it even with the flag off (isEnforcingSupportClaims()
// is false in this harness). These cover the routing branch only; enforceAtLevel()
// itself is stubbed because the real one needs a device.

TEST_F(TestVerificationModePathsFixture, NonFullBundleRoutesToEnforcementPath)
{
    auto bundle = loadBundle("enforce_gate", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::APPLICABILITY;

    bool engineCalled = false;
    ModeTestableHarness harness(
        VerificationMode::AUTO,
        [&](std::unordered_map<int64_t, void*>&) { engineCalled = true; },
        matchingRef());
    harness.setBundle(std::move(bundle), "vmode-test-bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        harness.TestBody();
    }

    EXPECT_EQ(harness.enforcedLevel(), EnforcementLevel::APPLICABILITY)
        << "non-FULL bundle must route to enforceAtLevel()";
    EXPECT_FALSE(engineCalled) << "comparison path must not run for a non-FULL bundle";
    EXPECT_FALSE(anyFailed(results));
}

TEST_F(TestVerificationModePathsFixture, FullBundleRoutesToComparisonPath)
{
    auto bundle = loadBundle("enforce_gate_full", /*includeGoldenOutput=*/true);
    bundle->metadata.enforcementLevel = EnforcementLevel::FULL;

    bool engineCalled = false;
    ModeTestableHarness harness(
        VerificationMode::AUTO,
        [&](std::unordered_map<int64_t, void*>& variantPack) {
            engineCalled = true;
            matchingEngine()(variantPack);
        },
        matchingRef());
    harness.setBundle(std::move(bundle), "vmode-test-bundle");

    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.SetUp();
        harness.TestBody();
    }

    EXPECT_FALSE(harness.enforcedLevel().has_value())
        << "FULL bundle must not route to enforceAtLevel()";
    EXPECT_TRUE(engineCalled) << "FULL bundle must run the comparison path";
    EXPECT_FALSE(anyFailed(results));
}

// NOLINTEND(readability-identifier-naming)
