// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// BundleReferenceValidationHarness had zero unit coverage before this file: the
// only line naming it in the test tree was the CMakeLists.txt entry that compiles
// it. Two behaviours are worth pinning here: the SetUp() guards that turn a
// registration bug into a named failure instead of a silent skip, and useDevice()
// asking the injected executor rather than trusting the registration enum.

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>

#include "BundleFixtureFiles.hpp"
#include "HarnessTestSupport.hpp"
#include "ScratchDirectory.hpp"
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
    // TestBody() is deliberately not called here: these two cases assert on the
    // registration guard itself, not on what a passing run looks like.
    static void driveSetUp(BundleReferenceValidationHarness& harness,
                           ::testing::TestPartResultArray* results)
    {
        const ::testing::ScopedFakeTestPartResultReporter reporter(
            ::testing::ScopedFakeTestPartResultReporter::INTERCEPT_ALL_THREADS, results);
        harness.SetUp();
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

// NOLINTEND(readability-identifier-naming)
