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
#include <vector>

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
        _scopedDir.emplace(scratch::makeDir("err_path_test_"));
        _tempDir = _scopedDir->path();
    }

    /// Writes and loads a bundle under the fixture's temp dir.
    std::shared_ptr<IntegrationTestBundle> loadBundle(const std::string& name,
                                                      bool includeGoldenOutput) const
    {
        return fixtures::loadBundle(_tempDir, name, includeGoldenOutput);
    }

    /// Builds the real harness on top of `mocks` and drives it through one bundle.
    static void runCapturing(testing_support::HarnessMocks& mocks,
                             std::shared_ptr<IntegrationTestBundle> bundle,
                             VerificationMode mode,
                             ::testing::TestPartResultArray* results)
    {
        IntegrationBundleVerificationHarness harness(
            mocks.dependencies(testing_support::hostPolicy(mode)));
        harness.setBundle(std::move(bundle), "err-path-test-bundle");
        testing_support::driveHarness(harness, results);
    }
};

// The engine seam answers "not mine" with EngineOpResult::declinedBy(...) rather
// than throwing: a decline is an answer, and the harness turns it into a SKIP.
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

// A generic engine problem is loud, never silently swallowed as a SKIP. The runner
// answers with EngineOpResult::failed(message) and the harness turns it into a
// FailureOrigin::ENGINE outcome carrying that message.
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

// An empty message on a FAILED outcome means "the failure is already on the gtest
// record with more detail than this could add", which only the comparison can
// promise. The engine seam cannot, so the harness has to name the failure itself —
// an engine error that produces a green test is the exact shape this harness
// exists to rule out.
TEST_F(TestErrorPaths, EngineFailureWithNoMessageStillFails)
{
    testing_support::HarnessMocks mocks;
    ON_CALL(mocks.engineRunner, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(EngineOpResult::failed("")));

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("eng_silent_crash", /*includeGoldenOutput=*/true),
                 VerificationMode::GOLDEN,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "an engine failure with no message must still fail the test";
    EXPECT_THAT(testing_support::allMessages(results),
                ::testing::HasSubstr("err-path-test-bundle"));
}

// One of the two forms the harness maps to CAPABILITY_MISS; the other is
// isApplicable()==false, covered by RefNotApplicableSkips below.
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
// that it has no plan for this op, without ever being asked to execute.
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

// A reference that breaks on an op it accepted is a real defect in the reference,
// so it routes to RefStatus::RUNTIME_ERROR and fails the test.
TEST_F(TestErrorPaths, RefCrashFails)
{
    testing_support::HarnessMocks mocks;
    testing_support::engineWrites(
        mocks.engineRunner, &fixtures::writeOutput, fixtures::K_OUTPUT_VALUE);
    ON_CALL(mocks.cpuReference, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack&) -> void {
            throw std::runtime_error("stub: ref crashed on supported op");
        });

    std::vector<std::string> refErrors;
    testing_support::captureReferenceErrors(mocks.reporter, refErrors);

    ::testing::TestPartResultArray results;
    runCapturing(mocks,
                 loadBundle("ref_crash", /*includeGoldenOutput=*/false),
                 VerificationMode::CPU,
                 &results);

    EXPECT_TRUE(testing_support::anyFailed(results))
        << "A generic ref exception must route to RUNTIME_ERROR and FAIL the test";
    ASSERT_EQ(refErrors.size(), 1U)
        << "A crashing reference must publish exactly one reference error";
    EXPECT_THAT(refErrors.front(), ::testing::HasSubstr("stub: ref crashed on supported op"));
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
