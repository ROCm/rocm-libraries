// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Reads a checked-in rocRoller kernel's tile fields from its Tensile solution.
// This header includes TensileLite, so only hipblaslt and the host tests
// include it. hipblaslt-rocroller-selection is built without those headers.

#include "rocroller_kernel_select.hpp"

#include <Tensile/ContractionSolution.hpp>

#include <algorithm>

inline RocRollerKernelFields rocRollerKernelFields(const TensileLite::ContractionSolution& solution)
{
    const auto&           kernel = solution.customKernel;
    RocRollerKernelFields fields;
    fields.tileM   = static_cast<int>(kernel.macrotile.x);
    fields.tileN   = static_cast<int>(kernel.macrotile.y);
    fields.tileK   = static_cast<int>(kernel.macrotile.z);
    fields.unroll  = kernel.unroll;
    fields.streamK = kernel.workspaceType == TensileLite::CustomWorkspaceType::RocRollerStreamK;
    fields.workgroupMapping
        = std::any_of(kernel.args.begin(), kernel.args.end(), [](auto const& arg) {
              return arg.semantic == TensileLite::CustomArgSemantic::RR_WorkgroupMapping;
          });
    return fields;
}
