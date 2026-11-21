// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionProblem.hpp>

#include <cstddef>

namespace rocblaslt
{
    // Grouped selection writes the group size and the cu-fallback status into each
    // gemm before evaluating its problem predicates (SynchronizerSizeCheck reads the
    // one, WorkgroupMappingXCCCheck the other). A launch-time check must see the same
    // state, on a copy: on the caller's gemm it would depend on whether selection
    // had already run on this object.
    inline TensileLite::ContractionProblemGemm
        groupedPredicateProblem(TensileLite::ContractionProblemGroupedGemm const& grouped,
                                std::size_t                                       index,
                                bool                                              isCUFallback)
    {
        auto gemm = grouped.gemms[index];
        gemm.setGroupedGemmCount(static_cast<int>(grouped.gemms.size()));
        gemm.setParams().setFallbackStatus(isCUFallback);
        return gemm;
    }
}
