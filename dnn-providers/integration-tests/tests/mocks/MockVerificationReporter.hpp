// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <gmock/gmock.h>

#include "harness/bundle/IVerificationReporter.hpp"

namespace hipdnn_integration_tests::bundle
{

/// Catches what a test body published, so assertions read the harness's own output
/// rather than clearing process-wide singletons in SetUp and hoping.
class MockVerificationReporter : public IVerificationReporter
{
public:
    MOCK_METHOD(void, recordCoverage, (const CoverageUpdate& update), (override));
    MOCK_METHOD(void, recordVerdict, (const SupportResult& record), (override));
    MOCK_METHOD(void,
                recordUnverifiable,
                (const std::string& bundlePath, const std::string& reason),
                (override));
    MOCK_METHOD(void,
                recordReferenceError,
                (const std::string& bundlePath, const std::string& reason),
                (override));
};

} // namespace hipdnn_integration_tests::bundle
