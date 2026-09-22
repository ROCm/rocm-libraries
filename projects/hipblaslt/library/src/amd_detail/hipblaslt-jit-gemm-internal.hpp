// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-gemm-tag.hpp"
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>

namespace hipblaslt_ext::experimental::detail
{
    struct JitContext
    {
        int                                    device = -1;
        std::shared_ptr<hipDeviceProp_t>       properties;
        std::shared_ptr<TensileLite::Hardware> hardware;
        std::shared_ptr<TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>>
                                                           library;
        std::unique_ptr<TensileLite::hip::SolutionAdapter> adapter;
        std::string                                        manifest, kernel;
    };

    // Returns null only for untagged algorithms. Tagged failures throw, never
    // falling through to a prebuilt index or lazy library initialization.
    std::shared_ptr<JitContext> resolveJitAlgo(const rocblaslt_matmul_algo& algo, int device);
}
