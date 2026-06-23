// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <stdexcept>
#include <string>

namespace hipdnn_integration_tests
{

// Signals "this reference executor has no plan for this op" — a CAPABILITY MISS,
// not a runtime failure. The golden-verification harness distinguishes three
// reference outcomes:
//
//   A  capability miss  — ref cannot run this op   -> ReferenceCapabilityError
//   B  disagreement     — ref ran, output != engine-> mismatch at compare time
//   C  runtime error    — ref CAN run it but threw -> any other std::exception
//
// In `auto` mode a case-A miss falls through to the next reference; in explicit
// gpu/cpu mode it SKIPs. A case-C error is loud (auto: fall through + loud
// report; explicit / end-of-auto: FAIL). Throwing the right type at the source
// is what lets the harness tell A from C.
//
// Deriving from std::runtime_error keeps existing `catch(const std::exception&)`
// / `catch(const std::runtime_error&)` call sites working unchanged.
//
// NOTE: the GPU reference executor (our code) throws this directly at its
// capability-miss sites. The CPU reference executor (test_sdk) throws
// CpuReferenceNotApplicableError for capability misses; the
// CpuReferenceGraphExecutorAdapter translates that into a ReferenceCapabilityError.
// Other test_sdk exceptions propagate as genuine runtime failures (case C).
class ReferenceCapabilityError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace hipdnn_integration_tests
