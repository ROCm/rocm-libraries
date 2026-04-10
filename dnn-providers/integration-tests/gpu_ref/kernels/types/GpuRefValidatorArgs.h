// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Shared argument struct for GPU reference validator kernels.
// Included by both device code (HipRTC) and host launch code.
// Only POD types allowed — no host or device includes.

#pragma once

// NOLINTBEGIN(misc-non-private-member-variables-in-classes,
//             readability-identifier-naming)
struct ValidatorArgs
{
    const void* reference;
    const void* implementation;
    int* failureFlag; // single int: 0 = all passed, 1 = any element failed
    long long totalElements;
    double absoluteTolerance;
    double relativeTolerance;
};
// NOLINTEND(misc-non-private-member-variables-in-classes,
//           readability-identifier-naming)
