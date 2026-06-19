// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>

namespace TensileLite::testing::detail
{
    // Fail the next optional alt allocation after this many successful calls.
    void setOptionalAltAllocationFailureCountdown(size_t callsBeforeFailure);
    void clearOptionalAltAllocationFailure();
    bool shouldFailOptionalAltAllocation();
} // namespace TensileLite::testing::detail
