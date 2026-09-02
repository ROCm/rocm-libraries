// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exception-type routing: EngineNotApplicableError→SKIP,
// ReferenceCapabilityError→SKIP, generic exception→FAIL.

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
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceNotApplicableError.hpp>

#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/EngineNotApplicableError.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

namespace
{

using EngineStub = std::function<void(std::unordered_map<int64_t, void*>&)>;
using RefStub = std::function<void(ReferenceExecutorType, std::unordered_map<int64_t, void*>&)>;

class ErrorPathHarness : public IntegrationBundleVerificationHarness
{
public:
    ErrorPathHarness(VerificationMode mode, EngineStub engineStub, RefStub refStub)
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

    void applyMetadataGuards() const override {}

private:
    VerificationMode _mode;
    EngineStub _engineStub;
    RefStub _refStub;
};

class TestErrorPaths : public ::testing::Test
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
              / ("err_path_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

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
                             VerificationMode mode,
                             EngineStub engineStub,
                             RefStub refStub,
                             ::testing::TestPartResultArray* results)
    {
        ErrorPathHarness harness(mode, std::move(engineStub), std::move(refStub));
        harness.setBundle(std::move(bundle), "err-path-test-bundle");

        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        if(!anySkipped(*results))
        {
            harness.TestBody();
        }
    }

    static EngineStub matchingEngine()
    {
        return [](std::unordered_map<int64_t, void*>& vp) { writeOutput(vp, K_OUTPUT_VALUE); };
    }

    static RefStub unusedRef()
    {
        return [](ReferenceExecutorType, std::unordered_map<int64_t, void*>&) {
            FAIL() << "Reference executor should not be called";
        };
    }
};

TEST_F(TestErrorPaths, EngineNotApplicableSkips)
{
    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("eng_not_applicable", /*includeGoldenOutput=*/true),
        VerificationMode::GOLDEN,
        [](auto&) { throw EngineNotApplicableError("stub: engine does not support graph"); },
        unusedRef(),
        &results);

    EXPECT_TRUE(anySkipped(results)) << "EngineNotApplicableError should produce a SKIP";
    EXPECT_FALSE(anyFailed(results));
}

TEST_F(TestErrorPaths, EngineCrashFails)
{
    auto bundle = loadBundle("eng_crash", /*includeGoldenOutput=*/true);

    EXPECT_THROW(
        {
            ::testing::TestPartResultArray results;
            runCapturing(
                std::move(bundle),
                VerificationMode::GOLDEN,
                [](auto&) { throw std::runtime_error("stub: unexpected engine crash"); },
                unusedRef(),
                &results);
        },
        std::runtime_error)
        << "A generic engine exception must propagate, not be silently swallowed";
}

TEST_F(TestErrorPaths, RefCapabilityMissSkips)
{
    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("ref_cap_miss", /*includeGoldenOutput=*/false),
        VerificationMode::CPU,
        matchingEngine(),
        [](ReferenceExecutorType, auto&) {
            throw ReferenceCapabilityError("stub: no plan for this op");
        },
        &results);

    EXPECT_TRUE(anySkipped(results)) << "ReferenceCapabilityError should produce a SKIP";
    EXPECT_FALSE(anyFailed(results));
}

TEST_F(TestErrorPaths, RefCrashFails)
{
    ::testing::TestPartResultArray results;
    runCapturing(
        loadBundle("ref_crash", /*includeGoldenOutput=*/false),
        VerificationMode::CPU,
        matchingEngine(),
        [](ReferenceExecutorType, auto&) {
            throw std::runtime_error("stub: ref crashed on supported op");
        },
        &results);

    EXPECT_TRUE(anyFailed(results))
        << "A generic ref exception must route to RUNTIME_ERROR and FAIL the test";
}

TEST_F(TestErrorPaths, AdapterTranslatesNotApplicableToCapabilityError)
{
    const CpuReferenceGraphExecutorAdapter adapter;

    hipdnn_test_sdk::utilities::CpuReferenceNotApplicableError notApplicable("stub");
    EXPECT_TRUE(dynamic_cast<const std::runtime_error*>(&notApplicable) != nullptr)
        << "CpuReferenceNotApplicableError must derive from std::runtime_error";

    ReferenceCapabilityError capError("stub");
    EXPECT_TRUE(dynamic_cast<const std::runtime_error*>(&capError) != nullptr)
        << "ReferenceCapabilityError must derive from std::runtime_error";

    try
    {
        throw hipdnn_test_sdk::utilities::CpuReferenceNotApplicableError("test");
    }
    catch(const ReferenceCapabilityError&)
    {
        FAIL() << "CpuReferenceNotApplicableError must NOT be caught as ReferenceCapabilityError";
    }
    catch(const hipdnn_test_sdk::utilities::CpuReferenceNotApplicableError&)
    {
        SUCCEED();
    }
}

TEST_F(TestErrorPaths, GenericRuntimeErrorNotCaughtAsNotApplicable)
{
    try
    {
        throw std::runtime_error("generic crash");
    }
    catch(const hipdnn_test_sdk::utilities::CpuReferenceNotApplicableError&)
    {
        FAIL() << "std::runtime_error must NOT be caught as CpuReferenceNotApplicableError";
    }
    catch(const std::runtime_error&)
    {
        SUCCEED();
    }
}

} // namespace

// NOLINTEND(readability-identifier-naming)
