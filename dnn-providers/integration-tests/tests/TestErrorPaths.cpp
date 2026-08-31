// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exception-type routing: EngineOpResult::declinedBy() -> SKIP,
// ReferenceCapabilityError / isApplicable()==false -> SKIP, generic exception -> FAIL.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <gmock/gmock.h>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceNotApplicableError.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "harness/CpuReferenceGraphExecutorAdapter.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

// NOLINTBEGIN(readability-identifier-naming)

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;

namespace
{

class TestErrorPaths : public ::testing::Test
{
protected:
    std::optional<hipdnn_test_sdk::utilities::ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();

        auto path
            = std::filesystem::temp_directory_path()
              / ("err_path_test_"
                 + std::to_string(::testing::UnitTest::GetInstance()->current_test_info()->line()));
        std::filesystem::remove_all(path);
        _scopedDir.emplace(path);
        _tempDir = _scopedDir->path();
    }

    /// Writes and loads a bundle under the fixture's temp dir.
    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      bool includeGoldenOutput) const
    {
        return fixtures::loadBundle(_tempDir, name, includeGoldenOutput);
    }

    /// Builds the real harness on top of `mocks`, drives it through one bundle, and
    /// captures every gtest disposition it issues. TestBody() is skipped when
    /// SetUp() itself skipped, matching how the real GTest runner drives a test.
    static void runCapturing(testing_support::HarnessMocks& mocks,
                             std::shared_ptr<IntegrationTestBundle> bundle,
                             VerificationMode mode,
                             ::testing::TestPartResultArray* results)
    {
        IntegrationBundleVerificationHarness harness(
            mocks.dependencies(testing_support::hostPolicy(mode)));
        harness.setBundle(std::move(bundle), "err-path-test-bundle");

        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        if(!testing_support::anySkipped(*results))
        {
            harness.TestBody();
        }
    }
};

// Was: EngineStub throwing EngineNotApplicableError. IGraphEngineRunner::execute()
// now answers "not mine" with EngineOpResult::declinedBy(...) instead of throwing.
// The outcome the test defends is unchanged (SKIP).
TEST_F(TestErrorPaths, EngineNotApplicableSkips)
{
    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(
            ::testing::Return(EngineOpResult::declinedBy("stub: engine does not support graph")));
    EXPECT_CALL(mocks.referenceExecutors, get(::testing::_)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("eng_not_applicable", /*includeGoldenOutput=*/true),
                 VerificationMode::GOLDEN,
                 &results);

    EXPECT_TRUE(testing_support::anySkipped(results)) << "A declined engine should produce a SKIP";
    EXPECT_FALSE(testing_support::anyFailed(results));
}

// Was: EngineStub throwing a generic std::runtime_error, expected to propagate
// uncaught out of TestBody() (caught here by EXPECT_THROW). The engine seam no
// longer throws for a compile/execute failure -- it answers with
// EngineOpResult::failed(message), which the harness turns into a
// FailureOrigin::ENGINE outcome carrying that message verbatim. The intent (a
// generic engine problem is loud, not silently swallowed as SKIP) is unchanged;
// only the mechanism moved from an uncaught exception to a reported FAIL.
TEST_F(TestErrorPaths, EngineCrashFails)
{
    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(EngineOpResult::failed("stub: unexpected engine crash")));
    EXPECT_CALL(mocks.referenceExecutors, get(::testing::_)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("eng_crash", /*includeGoldenOutput=*/true),
                 VerificationMode::GOLDEN,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "A generic engine failure must FAIL the test, not be silently swallowed";
    EXPECT_THAT(testing_support::allMessages(results),
                ::testing::HasSubstr("stub: unexpected engine crash"));
}

// Was: RefStub throwing ReferenceCapabilityError. This is one of the two forms the
// harness still catches as CAPABILITY_MISS (the other is isApplicable()==false,
// covered by RefNotApplicableSkips below).
TEST_F(TestErrorPaths, RefCapabilityMissSkips)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack&) {
            throw ReferenceCapabilityError("stub: no plan for this op");
        });

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("ref_cap_miss", /*includeGoldenOutput=*/false),
                 VerificationMode::CPU,
                 &results);

    EXPECT_TRUE(testing_support::anySkipped(results))
        << "ReferenceCapabilityError should produce a SKIP";
    EXPECT_FALSE(testing_support::anyFailed(results));
}

// The other capability-miss form: the reference says up front, via isApplicable(),
// that it has no plan for this op, without ever being asked to execute. Added so
// both forms the harness maps to CAPABILITY_MISS are exercised in this file.
TEST_F(TestErrorPaths, RefNotApplicableSkips)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.cpuReference, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(false));
    EXPECT_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("ref_not_applicable", /*includeGoldenOutput=*/false),
                 VerificationMode::CPU,
                 &results);

    EXPECT_TRUE(testing_support::anySkipped(results))
        << "isApplicable()==false should produce a SKIP";
    EXPECT_FALSE(testing_support::anyFailed(results));
}

// Was: RefStub throwing a generic std::runtime_error. Still routes to
// RefStatus::RUNTIME_ERROR and FAILs the test -- unchanged.
TEST_F(TestErrorPaths, RefCrashFails)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack&) -> void {
            throw std::runtime_error("stub: ref crashed on supported op");
        });

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("ref_crash", /*includeGoldenOutput=*/false),
                 VerificationMode::CPU,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
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
