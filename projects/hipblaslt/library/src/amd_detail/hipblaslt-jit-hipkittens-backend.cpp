// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-hipkittens.hpp"

#include <iostream>
#include <memory>

namespace hipblaslt_jit
{
    // The HipKittens process backend, or null after a warning when its headers
    // are missing or do not match this build.
    std::shared_ptr<const Jit> makeHipKittensJit()
    {
        hipblaslt_ext::experimental::jit::Backend     backend;
        hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
        const auto status = hipblaslt_ext::experimental::jit::hipkittens::createBackend(
            {}, backend, diagnostics);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            std::cerr << "hipblaslt warning: " << diagnostics.message << std::endl;
            return nullptr;
        }
        return hipblaslt_ext::experimental::jit::detail::BackendAccess::get(backend);
    }
}
