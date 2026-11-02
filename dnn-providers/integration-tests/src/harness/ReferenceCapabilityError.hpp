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
// A case-A miss falls through to the next reference; when none is left the bundle
// has no oracle and FAILs. Most misses are caught by isApplicable() before the
// reference executes; this exception covers the ones only execute() can see.
// A case-C error also falls through, but is listed in the reference-error report,
// and FAILs if nothing after it verifies the bundle. Throwing the right type at
// the source is what lets the harness tell A from C.
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
