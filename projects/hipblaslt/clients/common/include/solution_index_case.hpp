// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "hipblaslt_arguments.hpp"

#include <Tensile/ContractionSolution.hpp>

#include <cstdint>
#include <string>
#include <variant>

namespace hipblaslt_client
{
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

    // Build a small workload for an already-selected solution: translate its
    // types, layouts and required outputs, then choose dimensions accepted by
    // its size predicates. This constructs client Arguments; it does not run
    // the workload or choose a kernel.
    //
    // Prefer one macro tile of output and options.reductionBlocks complete
    // DepthU blocks of K per split (two splits for automatic split-K, none for
    // persistent schedules). Predicate bounds and multiples supply alternative
    // dimensions. Grouped solutions get two equal members. Data is packed;
    // norm and allclose checks validate the outputs.
    //
    // This is one candidate, not a general constraint solver or proof of device
    // eligibility. The caller must run the normal client checks against the
    // matching library and device. Unsupported cases carry a reason.
    std::variant<Arguments, UnsupportedSolution>
        solutionIndexCase(TensileLite::ContractionSolution const& solution,
                          SolutionCaseOptions const&              options = {});
}
