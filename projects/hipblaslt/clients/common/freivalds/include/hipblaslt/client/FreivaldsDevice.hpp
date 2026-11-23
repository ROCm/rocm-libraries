// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime_api.h>
#include <roc/hostnumerics/freivalds.hpp>
#include <variant>

namespace hipblaslt::client
{
    struct FreivaldsDeviceObservation
    {
        // A full logical host copy is used only when the device cannot produce
        // a complete compact observation. Both alternatives go to plan.check().
        std::variant<roc::hostnumerics::FreivaldsObservation, roc::hostnumerics::Tensor> output;
        size_t readbackBytes      = 0;
        size_t deviceScratchBytes = 0;
        double projectionSeconds  = 0;
        double readbackSeconds    = 0;
        double reductionSeconds   = 0;
    };

    // Observe one logical matrix. data addresses the allocation described by
    // layout, including its offset; layout.shape() must equal request.shape.
    // The caller keeps device storage alive until this synchronous call returns.
    // Throws on invalid descriptors or HIP failures. It never returns a partial
    // projection as a complete observation. No GEMM or numerical verdict occurs here.
    FreivaldsDeviceObservation observeFreivalds(const roc::hostnumerics::FreivaldsRequest& request,
                                                const void*                                data,
                                                const roc::hostnumerics::Layout&           layout,
                                                hipStream_t                                stream);
}
