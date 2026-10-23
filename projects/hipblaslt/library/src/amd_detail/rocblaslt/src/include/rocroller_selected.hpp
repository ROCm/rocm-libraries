// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocblaslt.h"

// RocRollerSelectedTile and the kernel-name match, in a header that depends on
// nothing but <string> so they can be unit-tested GPU-free.
#include "rocroller_kernel_match.hpp"

#include <vector>

// The problem properties rocRoller declines outright: a bias, a batch, a scale
// type other than f32, and accumulation other than f32. Cheap and derived from
// the problem alone, so every rocRoller entry point can agree on the answer.
// Callers route a declined problem to TensileLite instead of failing it.
bool rocRollerSupportsProblem(const RocblasltContractionProblem& prob);

// Same rejects and the same rank order as getRocRollerBestSolutions.
// Implemented by calling chooseSolutionIndexParameters.
rocblaslt_status rankRocRollerSelectedTiles(const RocblasltContractionProblem& prob,
                                            std::vector<RocRollerSelectedTile>& tiles);
