// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-tensilelite-internal.hpp"
#include <hipblaslt/hipblaslt-jit-tensilelite.hpp>

namespace hipblaslt_ext::experimental::jit::tensilelite::detail
{
    // Provider-private plan consumed immediately by its builder. There is no
    // public prediction schema or cache-before-compilation protocol yet.
    struct PredictionPlan
    {
        std::string requestPath;
        std::string summary;
    };
    PredictionPlan planGemm(const jit::detail::GemmRequest& request,
                            const jit::detail::Target&      target,
                            const Options&                  options);
    PredictionPlan predictGemmPlan(const TensileLite::ContractionProblemGemm& problem,
                                   const TensileLite::Hardware&               hardware,
                                   const Options&                             options,
                                   const std::string&                         scaleModeA,
                                   const std::string&                         scaleModeB);
}
