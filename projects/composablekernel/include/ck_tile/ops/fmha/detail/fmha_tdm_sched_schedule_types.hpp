// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

enum class FmhaTdmSchedLoadKind : index_t
{
    KRead,
    VRead,
    IgnoredLegacy,
};

// These are compiler scheduling points, not hardware synchronization, except Drain.
enum class FmhaTdmSchedSchedulePoint : index_t
{
    AfterWmma,
    BetweenTokenHalves,
    AfterTokens,
    // FmhaTdmSchedGroupedDrainPlacement only: a full wait for the WMMAs in flight, and the operand
    // keep-alive after the RAW fence of the stage's last row.
    Drain,
    KeepPending,
};

} // namespace ck_tile
