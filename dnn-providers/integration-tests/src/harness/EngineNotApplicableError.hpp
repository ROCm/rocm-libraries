// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

namespace hipdnn_integration_tests
{

// Signals "the engine under test does not support this graph" — the named
// engine was not in the ranked list, or no engine was ranked at all.
// Distinct from ReferenceCapabilityError (reference executor can't run an op)
// and from a generic std::exception (unexpected crash).
//
// runEngineCapturingOutputs catches this specifically so that an unsupported
// graph becomes a GTEST_SKIP, while genuine engine crashes propagate as
// uncaught exceptions (reported as test failures by GTest).
class EngineNotApplicableError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace hipdnn_integration_tests
