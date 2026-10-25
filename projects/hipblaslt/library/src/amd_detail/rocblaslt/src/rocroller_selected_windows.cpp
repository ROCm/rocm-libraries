// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Windows build of rankRocRollerSelectedTiles. rocRoller's data-type headers
// do not compile under the MSVC record layout. No Windows GPU target has
// checked-in rocRoller kernels, so no ranked tile could match one; returning
// no tiles gives callers the same empty result.

#include "include/rocroller_selected.hpp"

rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles)
{
    static_cast<void>(prob);
    tiles.clear();
    return rocblaslt_status_success;
}
