// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

namespace hipdnn_test_sdk::utilities
{

// Signals that the CPU reference executor has no plan for a graph node —
// either the node type is not in the signature-key switch or the plan
// builder's isApplicable() returned false. Distinct from a generic
// std::runtime_error, which indicates a genuine runtime failure in a
// plan that *should* have worked.
class CpuReferenceNotApplicableError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

} // namespace hipdnn_test_sdk::utilities
