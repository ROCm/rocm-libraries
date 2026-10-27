// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

#include "harness/IReferenceExecutors.hpp"
#include "harness/IReferenceGraphExecutor.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"
#include "harness/reference-validation/ReferenceOpCoverage.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Runs one bundle through a reference executor with no expected output to compare
/// against (`hipdnn_golden_data_tests --smoke`). It answers two questions about the
/// reference, not about an engine: does it accept the graph, and does it produce
/// finite output for it.
///
/// Inputs come from the bundle's metadata fill recipes and seed, as the engine
/// harness generates them for a bundle without tensor blobs. A test fails when the
/// reference reports the graph inapplicable, throws, or leaves a NaN or Inf in any
/// output element. Outputs are prefilled with a NaN sentinel, so an element the
/// reference never wrote also fails. Integer and other non-float outputs are not
/// scanned; the test records which ones it skipped.
class BundleReferenceSmokeHarness : public ::testing::Test
{
public:
    BundleReferenceSmokeHarness(ReferenceExecutorType referenceType,
                                std::shared_ptr<IReferenceExecutors> referenceExecutors)
        : _referenceType(referenceType)
        , _referenceExecutors(std::move(referenceExecutors))
    {
    }

    /// `knownGap` is the knownReferenceGaps() entry for this bundle, if any; the test
    /// then skips with the entry's reason instead of running.
    void setBundle(std::shared_ptr<IntegrationTestBundle> bundle,
                   std::filesystem::path path,
                   std::optional<KnownReferenceGap> knownGap)
    {
        _bundle = std::move(bundle);
        _bundlePath = std::move(path);
        _knownGap = std::move(knownGap);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override;

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TestBody() override;

private:
    IReferenceGraphExecutor& referenceExecutor() const;
    bool useDevice() const;

    ReferenceExecutorType _referenceType;
    std::shared_ptr<IReferenceExecutors> _referenceExecutors;
    std::filesystem::path _bundlePath;
    std::optional<KnownReferenceGap> _knownGap;
    std::shared_ptr<IntegrationTestBundle> _bundle;
};

/// Non-finite elements found in one output tensor, over its logical elements only:
/// padding between strided elements is never written by a reference and is skipped.
struct NonFiniteScan
{
    bool scanned = false; ///< false when the element type is not a float type
    std::size_t elements = 0;
    std::size_t nans = 0;
    std::size_t infs = 0;
    std::optional<std::vector<int64_t>> firstBadIndex;
};

NonFiniteScan scanNonFinite(hipdnn_data_sdk::utilities::ITensor& tensor);

} // namespace hipdnn_integration_tests::bundle
