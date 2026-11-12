// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

// RocRollerSelectedTile, the kernel match and the walk, in headers that depend
// on nothing GPU-side so they can be unit-tested GPU-free.
#include "rocroller_kernel_select.hpp"

#include <vector>

// Same rejects and the same rank order as getRocRollerBestSolutions.
// Implemented by calling chooseSolutionIndexParameters. A problem with an
// operand of 4 GiB or more gets no tiles: rocRoller refuses it at every tile.
rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles);
