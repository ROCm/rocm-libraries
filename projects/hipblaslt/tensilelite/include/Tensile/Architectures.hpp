// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string_view>

namespace TensileLite
{
    // PCI device ids of the GPUs TensileLite builds libraries for, one list for
    // every C++ caller. SUPPORTED_BUILD_CHIP_IDS in
    // tensilelite/Common/Architectures.py is the source of the values; the
    // catalog tests (test_solution_state_catalog.py and
    // SolutionRecord.CatalogMatchesSolutionState) fail when the two lists
    // differ.
    inline constexpr std::array<std::string_view, 14> SupportedBuildChipIds = {"74a0",
                                                                               "74a1",
                                                                               "74a2",
                                                                               "74a3",
                                                                               "74a5",
                                                                               "74a9",
                                                                               "75a0",
                                                                               "75b0",
                                                                               "75a2",
                                                                               "75b2",
                                                                               "75a3",
                                                                               "75b3",
                                                                               "75a8",
                                                                               "75b8"};
} // namespace TensileLite
