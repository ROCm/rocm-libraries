// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// The tile ranker, with the hardware model it scores against taken as an
// argument rather than queried.
//
// Kept apart from solution_selection.hpp so that the rest of the rocRoller
// headers, which describe a kernel only in the types hipBLASLt already uses
// for one, do not acquire an origami include.

#include "solution_selection.hpp"

#include "origami/hardware.hpp"

/**
 * @brief The hardware model the ranker scores against, for the current HIP device.
 *
 * Cached per device. The ranker used to build this from device 0 unconditionally;
 * asking for the device the caller is actually on is what makes the answer right
 * on a multi-GPU or partitioned system.
 */
const origami::hardware_t& rankerHardwareForCurrentDevice();

/**
 * @brief Choose the SolutionIndexParameters to use for a given problem.
 *
 * The whole of the ranker. The overload taking a RocblasltContractionProblem is
 * a front end onto this: it translates the descriptor into a KernelType and a
 * RankerProblem and calls here.
 *
 * Taking the hardware as an argument lets a host-only test rank against a fixed
 * synthetic device, with no GPU present.
 */
std::vector<SolutionIndexParameters>
    chooseSolutionIndexParameters(const KernelType&          kernelType,
                                  const RankerProblem&       prob,
                                  int                        requestedAlgoCount,
                                  const origami::hardware_t& hardware);
