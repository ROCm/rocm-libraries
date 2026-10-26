// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace roc::hostnumerics {
enum class VerificationStatus { Passed, Mismatch, Unsupported, Incomplete };

// Unsupported denotes numerical semantics outside a verifier's contract.
// Incomplete denotes an exhausted work/storage budget or incomplete observation.
// Neither outcome is a successful numerical check. Invalid API arguments throw.
struct VerificationFailure {
    VerificationStatus status;
    std::string reason;
};
}  // namespace roc::hostnumerics
