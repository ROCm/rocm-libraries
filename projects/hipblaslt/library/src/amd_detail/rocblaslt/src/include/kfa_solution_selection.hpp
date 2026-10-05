// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

#include <vector>

// One survivor of chooseSolutionIndexParameters, after the rocRoller filters
// and before the M/N/K divisibility test. Stream-K and non-temporal bits are
// carried so a checked-in kernel that lacks them is not chosen in their place.
struct KfaTile
{
    int  m = 0;
    int  n = 0;
    int  k = 0;
    bool workgroupMapping = false;
    bool streamK          = false;
    bool nonTemporalA     = false;
    bool nonTemporalB     = false;
};

// Ranked tile list for a problem rocRoller would accept. Returns
// rocblaslt_status_invalid_value for the same bias, batch, scale-type, and
// accumulation rejects as getRocRollerBestSolutions. On success, tiles is the
// filtered rank order (divisibility is applied by the caller).
rocblaslt_status rankKfaTiles(const RocblasltContractionProblem& prob,
                              std::vector<KfaTile>&              tiles);
