// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "../../include/rocblaslt-types.h"

namespace rocblaslt
{
    template <typename Problem>
    bool logVectorAlpha(const Problem& problem)
    {
        return !problem.getParams().deviceScalarAlpha() && problem.useScaleAlphaVec();
    }

    template <typename Problem>
    const char* benchAlphaFlag(const Problem& problem)
    {
        if(problem.getParams().deviceScalarAlpha())
            return "--device_scalar_alpha";
        return logVectorAlpha(problem) ? "--scaleAlpha_vector" : "";
    }

    inline const char* traceAlphaLabel(rocblaslt_pointer_mode mode)
    {
        if(mode == rocblaslt_pointer_mode_alpha_device_scalar_beta_host_ext)
            return "alphaScalarDevice";
        return mode == rocblaslt_pointer_mode_host ? "alpha" : "alphaVector";
    }
}
