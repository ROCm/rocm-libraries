// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <gtest/gtest.h>

#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "harness/IReferenceExecutors.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/TestConfig.hpp"
#include "harness/bundle/IntegrationBundleVerificationHarness.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Validates checked-in golden data against a reference executor. No engine is
/// involved, so no support claims exist to enforce and none are queried.
///
/// This is deliberately a separate harness rather than a mode of the engine
/// harness. Verifying an engine and verifying our own golden data are different
/// jobs with different failure meanings, and folding the second into the first is
/// what produced a "verification mode" that structurally never reached an engine —
/// and therefore never checked the claims the engine harness exists to enforce.
///
/// **There is no verification skip path.** Registration only creates a test when
/// the bundle has golden data *and* every node type is in this reference's
/// required-op set (see ReferenceOpCoverage.hpp). Given that, a reference that
/// cannot run the graph is a gap in the reference, so it fails rather than skips.
/// The one skip is device availability, checked before that gate: the GPU
/// reference needs a device, so its suite skips on a runner that has none.
class BundleReferenceValidationHarness : public ::testing::Test
{
public:
    BundleReferenceValidationHarness(ReferenceExecutorType referenceType,
                                     bool requiresDevice,
                                     std::shared_ptr<IReferenceExecutors> referenceExecutors)
        : _referenceType(referenceType)
        , _requiresDevice(requiresDevice)
        , _referenceExecutors(std::move(referenceExecutors))
    {
    }

    void setBundle(std::shared_ptr<IntegrationTestBundle> bundle, std::filesystem::path path)
    {
        _bundle = std::move(bundle);
        _bundlePath = std::move(path);
    }

    static const char* referenceLabel(ReferenceExecutorType type)
    {
        return type == ReferenceExecutorType::GPU ? "GpuRef" : "CpuRef";
    }

    // Public rather than protected: the unit tests drive a real harness directly
    // instead of subclassing it, so they need to call these. GTest calls them
    // through the base class either way.
    //
    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override;

private:
    // Borrowed from the run's pool: the executors are reusable, so this harness
    // does not build one per bundle.
    IReferenceGraphExecutor& referenceExecutor() const;

    bool useDevice() const;
    OutputTensors allocateOutputs() const;
    std::unordered_map<int64_t, void*> buildVariantPack(OutputTensors& outputs) const;

    ReferenceExecutorType _referenceType;
    bool _requiresDevice;
    std::shared_ptr<IReferenceExecutors> _referenceExecutors;
    std::filesystem::path _bundlePath;
    std::shared_ptr<IntegrationTestBundle> _bundle;
};

} // namespace hipdnn_integration_tests::bundle
