// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// A client case that runs one solution of a built library by explicit index,
// constructed from that solution's own TensileLite record.

#include "hipblaslt_arguments.hpp"

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionSolution.hpp>

#include <cstdint>
#include <string>
#include <variant>

// Choices the solution does not determine.
struct SolutionCaseOptions
{
    // Complete DepthU blocks of K that each split reduces.
    int64_t reductionBlocks = 8;
    double  alpha           = 1;
    double  beta            = 0;
    // gfx1250 lays out MX scales for tensor data movement; other targets do not.
    bool gfx1250 = false;
};

// A solution whose case the client cannot express, and why.
struct UnsupportedSolution
{
    std::string reason;
};

// The case for `solution`, stored at `index` in its library: one macro tile of
// output, K covering options.reductionBlocks complete DepthU blocks per split
// (two splits for automatic split-K, none for persistent schedules), adjusted
// to sizes the solution's own size predicates accept, with the gradient, E and
// amaxD outputs it always writes. Grouped solutions get two equal members.
// Data is packed; the norm check is the verdict.
std::variant<Arguments, UnsupportedSolution>
    solutionIndexCase(int                                     index,
                      TensileLite::ContractionSolution const& solution,
                      SolutionCaseOptions const&              options);

// Whether `solution` runs on `hardware` only because of its compute-unit count:
// the same device with another count is rejected.
bool requiresCUCount(TensileLite::ContractionSolution const& solution,
                     TensileLite::Hardware const&            hardware);
