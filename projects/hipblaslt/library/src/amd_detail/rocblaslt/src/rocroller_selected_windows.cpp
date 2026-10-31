// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Windows build of rankRocRollerSelectedTiles. This file does not include
// rocRoller's headers. No Windows GPU target has checked-in rocRoller kernels,
// so a tile returned here would have nothing to run.

#include "include/rocroller_selected.hpp"

rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles)
{
    static_cast<void>(prob);
    tiles.clear();
    return rocblaslt_status_success;
}
