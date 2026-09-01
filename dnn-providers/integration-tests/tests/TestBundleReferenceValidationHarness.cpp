// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// BundleReferenceValidationHarness is the sole gate on our checked-in golden data
// and, by design, has no verification skip path — so every way it can decline to
// check something has to be a loud failure. This file pins that: the SetUp()
// guards, useDevice() asking the injected executor rather than trusting the
// registration enum, and each branch of TestBody().

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "ScratchDirectory.hpp"
#include "harness/ReferenceCapabilityError.hpp"
#include "harness/bundle/BundleReferenceValidationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "mocks/MockReferenceExecutors.hpp"

using namespace hipdnn_integration_tests;
using namespace hipdnn_integration_tests::bundle;
using hipdnn_test_sdk::utilities::ScopedDirectory;

// NOLINTBEGIN(readability-identifier-naming)

namespace
{

class TestBundleReferenceValidationHarness : public ::testing::Test
{
protected:
    std::optional<ScopedDirectory> _scopedDir;
    std::filesystem::path _tempDir;
    ::testing::NiceMock<MockReferenceExecutors> _referenceExecutors;
    ::testing::NiceMock<MockReferenceGraphExecutor> _gpuExecutor;

    void SetUp() override
    {
        testing_support::ensureTestConfigInitialized();
        _scopedDir.emplace(scratch::makeDir("bundle_reference_validation_"));
        _tempDir = _scopedDir->path();

        using ::testing::Return;
        using ::testing::ReturnRef;

        ON_CALL(_referenceExecutors, get(ReferenceExecutorType::GPU))
            .WillByDefault(ReturnRef(_gpuExecutor));
        ON_CALL(_gpuExecutor, isApplicable(::testing::_, ::testing::_)).WillByDefault(Return(true));
    }

    std::shared_ptr<IReferenceExecutors> executors()
    {
        return testing_support::nonOwning<IReferenceExecutors>(_referenceExecutors);
    }

    // Runs SetUp() the way GTest would, capturing every disposition it issues.
    static void driveSetUp(BundleReferenceValidationHarness& harness,
                           ::testing::TestPartResultArray* results)
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
    }

    // SetUp() then TestBody(), the way GTest's own runner drives a test: TestBody()
    // is skipped when SetUp() already issued a skip.
    static void drive(BundleReferenceValidationHarness& harness,
                      ::testing::TestPartResultArray* results)
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
        if(!testing_support::anySkipped(*results))
        {
            harness.TestBody();
        }
    }

    // Gives a harness a bundle that has golden data. The harness itself is built by
    // each case: ::testing::Test is non-copyable, so it cannot be handed back.
    void setGoldenBundle(BundleReferenceValidationHarness& harness)
    {
        harness.setBundle(fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/true),
                          _tempDir / "Bundle");
    }
};

} // namespace

// Registration only ever creates this harness for a bundle that already has
// golden data (ReferenceOpCoverage.hpp), so reaching SetUp() without it is a
// registration bug, not a property worth skipping over.
TEST_F(TestBundleReferenceValidationHarness, SetUpFailsForABundleRegisteredWithNoGoldenData)
{
    auto bundle = fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/false);
    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::CPU, /*requiresDevice=*/false, executors());
    harness.setBundle(bundle, _tempDir / "Bundle");

    ::testing::TestPartResultArray results;
    driveSetUp(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("no golden data"), std::string::npos);
}

// The second guard covers a state the file-based fixtures cannot produce (the
// loader only ever sets hasGoldenOutputs once tensors is already populated -- see
// IntegrationTestBundle.hpp's loadTensorDataIfPresent), so this bundle is built by
// hand rather than through fixtures::loadBundle.
TEST_F(TestBundleReferenceValidationHarness, SetUpFailsForABundleRegisteredWithNoTensorData)
{
    auto bundle = std::make_shared<IntegrationTestBundle>();
    bundle->hasGoldenOutputs = true; // clears the first guard so the second one fires

    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::CPU, /*requiresDevice=*/false, executors());
    harness.setBundle(bundle, "no-tensor-data-bundle");

    ::testing::TestPartResultArray results;
    driveSetUp(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("no tensor data"), std::string::npos);
}

// The regression this rework fixes: useDevice() used to read _referenceType ==
// GPU, which is a property of how the harness was registered, not of what the
// executor it was actually handed can take. A registered-GPU harness driving an
// executor that reports it does not need device memory must stay on host
// pointers, or handing it device pointers anyway is exactly the silent crash the
// old comment warned about.
TEST_F(TestBundleReferenceValidationHarness,
       UseDeviceStaysOffTheDeviceWhenTheExecutorReportsNoDeviceNeed)
{
    using ::testing::_;
    using ::testing::AtLeast;
    using ::testing::Return;

    EXPECT_CALL(_gpuExecutor, requiresDeviceMemory())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(false));
    ON_CALL(_gpuExecutor, execute(_, _, _))
        .WillByDefault([](void*, size_t, const VariantPack& variantPack) {
            auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
            std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
        });

    auto bundle = fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/true);
    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/true, executors());
    harness.setBundle(bundle, _tempDir / "Bundle");

    // Driven through TestBody() directly rather than SetUp()+TestBody(): SetUp()'s
    // SKIP_IF_NO_DEVICES() gate is keyed off the registration flag alone and would
    // skip this case on a deviceless machine before useDevice() ever ran the
    // executor's own answer -- exactly the seam this test isolates. The mock never
    // touches device memory when it reports false, so no device is needed here.
    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.TestBody();
    }

    EXPECT_FALSE(testing_support::anyFailed(results));
}

// The other half: registered as device-requiring and handed an executor that
// actually wants device memory, useDevice() must still say yes. Reading real
// values back off the device needs a real device (ITensor::rawDeviceData()
// hipMallocs lazily), so this is gated the same way every GPU-path case in this
// binary is; on a deviceless machine it skips rather than requiring a GPU.
TEST_F(TestBundleReferenceValidationHarness,
       UseDeviceReachesTheDeviceWhenBothTheRegistrationAndTheExecutorWantIt)
{
    SKIP_IF_NO_DEVICES();

    using ::testing::_;
    using ::testing::AtLeast;
    using ::testing::Return;

    EXPECT_CALL(_gpuExecutor, requiresDeviceMemory())
        .Times(AtLeast(1))
        .WillRepeatedly(Return(true));

    auto bundle = fixtures::loadBundle(_tempDir, "Bundle", /*includeGoldenOutput=*/true);
    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/true, executors());
    harness.setBundle(bundle, _tempDir / "Bundle");

    // Not driven through SetUp(): same seam as above. The guard at the top of this
    // test already confirmed a device is present for this process.
    ::testing::TestPartResultArray results;
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, &results);
        harness.TestBody();
    }

    // No assertion on `results`: the mock executor never wrote real values into
    // the device buffer it was handed, so a comparison mismatch is expected here.
    // What this pins is requiresDeviceMemory() being asked at all (the
    // EXPECT_CALL above) and the device allocation/read-back path completing
    // without throwing.
}

// Registration promised this reference covers every node type in the graph, so a
// reference that then says it cannot run it is a gap in the reference. It must be
// loud: a skip here is a bundle nobody checked.
TEST_F(TestBundleReferenceValidationHarness, InapplicableReferenceFailsRatherThanSkips)
{
    ON_CALL(_gpuExecutor, isApplicable(::testing::_, ::testing::_))
        .WillByDefault(::testing::Return(false));
    EXPECT_CALL(_gpuExecutor, execute(::testing::_, ::testing::_, ::testing::_)).Times(0);

    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/false, executors());
    setGoldenBundle(harness);

    ::testing::TestPartResultArray results;
    drive(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_NE(testing_support::allMessages(results).find("is required to support this graph"),
              std::string::npos)
        << testing_support::allMessages(results);
}

// Same contract by the other route: the reference accepts the graph up front and
// then throws a capability error once it looks properly. Still a gap, still loud.
TEST_F(TestBundleReferenceValidationHarness, CapabilityErrorFromTheReferenceFails)
{
    ON_CALL(_gpuExecutor, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack&) {
            throw ReferenceCapabilityError("stub: no plan for this shape");
        });

    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/false, executors());
    setGoldenBundle(harness);

    ::testing::TestPartResultArray results;
    drive(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_FALSE(testing_support::anySkipped(results));
    EXPECT_NE(testing_support::allMessages(results).find("stub: no plan for this shape"),
              std::string::npos);
}

// A reference that breaks outright is reported with its own message, not folded
// into the capability case — the two mean different things to whoever fixes it.
TEST_F(TestBundleReferenceValidationHarness, ReferenceThatThrowsIsReportedWithItsMessage)
{
    ON_CALL(_gpuExecutor, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack&) {
            throw std::runtime_error("stub: reference exploded");
        });

    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/false, executors());
    setGoldenBundle(harness);

    ::testing::TestPartResultArray results;
    drive(harness, &results);

    EXPECT_TRUE(testing_support::anyFailed(results));
    EXPECT_NE(testing_support::allMessages(results).find("stub: reference exploded"),
              std::string::npos);
}

// The green path, and the mismatch path beside it: a reference whose output equals
// the golden data passes, and one that drifts fails naming the tensor. Together
// they pin that the comparison is actually consulted rather than assumed.
TEST_F(TestBundleReferenceValidationHarness, MatchingReferenceOutputPasses)
{
    ON_CALL(_gpuExecutor, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack& variantPack) {
            auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
            std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE);
        });

    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/false, executors());
    setGoldenBundle(harness);

    ::testing::TestPartResultArray results;
    drive(harness, &results);

    EXPECT_FALSE(testing_support::anyFailed(results)) << testing_support::allMessages(results);
    EXPECT_FALSE(testing_support::anySkipped(results));
}

TEST_F(TestBundleReferenceValidationHarness, DriftedReferenceOutputFailsNamingTheTensor)
{
    ON_CALL(_gpuExecutor, execute(::testing::_, ::testing::_, ::testing::_))
        .WillByDefault([](void*, size_t, const VariantPack& variantPack) {
            auto* ptr = static_cast<float*>(variantPack.at(fixtures::K_OUTPUT_UID));
            std::fill(ptr, ptr + fixtures::K_OUTPUT_ELEMS, fixtures::K_OUTPUT_VALUE + 100.0f);
        });

    BundleReferenceValidationHarness harness(
        ReferenceExecutorType::GPU, /*requiresDevice=*/false, executors());
    setGoldenBundle(harness);

    ::testing::TestPartResultArray results;
    drive(harness, &results);

    ASSERT_TRUE(testing_support::anyFailed(results));

    // One failure per drifted tensor, and this bundle has exactly one output.
    int failures = 0;
    for(int i = 0; i < results.size(); ++i)
    {
        if(results.GetTestPartResult(i).failed())
        {
            ++failures;
        }
    }
    EXPECT_EQ(failures, 1) << testing_support::allMessages(results);

    const std::string messages = testing_support::allMessages(results);
    EXPECT_NE(messages.find("Golden data validation"), std::string::npos) << messages;
    EXPECT_NE(messages.find("UID " + std::to_string(fixtures::K_OUTPUT_UID)), std::string::npos)
        << messages;
}

// NOLINTEND(readability-identifier-naming)
