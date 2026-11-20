// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// A client case that runs one solution of a built library by explicit index,
// constructed from that solution's own TensileLite record.

#include "hipblaslt_arguments.hpp"

#include <Tensile/ContractionSolution.hpp>

#include <cstdint>
#include <string>
#include <variant>

// Choices the solution does not determine. reductionBlocks must be positive;
// alpha and beta must be finite.
struct SolutionCaseOptions
{
    // Complete DepthU blocks of K that each split reduces.
    int64_t reductionBlocks = 8;
    double  alpha           = 1;
    double  beta            = 0;
};

// A solution whose case the client cannot express, and why.
struct UnsupportedSolution
{
    std::string reason;
};

// A candidate case for `solution`: one macro tile of
// output, K covering options.reductionBlocks complete DepthU blocks per split
// (two splits for automatic split-K, none for persistent schedules), adjusted
// to sizes the solution's own size predicates accept, with the gradient, E and
// amaxD outputs it always writes. Grouped solutions get two equal members.
// Data is packed; norm and allclose checks validate the outputs. This does not
// prove the solution is eligible: the caller must run the normal client checks
// against the matching library and device. Unsupported cases carry a reason.
std::variant<Arguments, UnsupportedSolution>
    solutionIndexCase(TensileLite::ContractionSolution const& solution,
                      SolutionCaseOptions const&              options = {});
