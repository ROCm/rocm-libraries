// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace hip_kernel_provider::kernel_ingestor_engine::megakernel
{

/// Why a persistent grid cannot run on a device. @c field is the bracketed key of the
/// decline line ("cus" or "occupancy"); @c detail is for people.
struct CoResidencyProblem
{
    std::string_view field;
    std::string detail;
};

/// The megakernel spins on a grid barrier, so all @p gridCus workgroups must be resident
/// at once: one per CU of the device. Returns the problem when the device has another CU
/// count (@p deviceCus) or fits less than one workgroup per CU (@p blocksPerCu), else
/// nullopt. Pure, so the rule is testable without a device.
inline std::optional<CoResidencyProblem>
    coResidencyProblem(int64_t gridCus, int deviceCus, int blocksPerCu)
{
    if(gridCus != deviceCus)
    {
        return CoResidencyProblem{"cus",
                                  "the grid has " + std::to_string(gridCus)
                                      + " workgroups, the device has " + std::to_string(deviceCus)
                                      + " CUs"};
    }
    if(blocksPerCu < 1)
    {
        return CoResidencyProblem{"occupancy",
                                  std::to_string(blocksPerCu)
                                      + " workgroup(s) fit per CU; the persistent grid needs 1"};
    }
    return std::nullopt;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::megakernel
