// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

// RocRollerSelectedTile and the kernel-name match, in a header that depends on
// nothing but <string> so they can be unit-tested GPU-free.
#include "rocroller_kernel_match.hpp"

#include <vector>

// Same rejects and the same rank order as getRocRollerBestSolutions.
// Implemented by calling chooseSolutionIndexParameters.
rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles);
