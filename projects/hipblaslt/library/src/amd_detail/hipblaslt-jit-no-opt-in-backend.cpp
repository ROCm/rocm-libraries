// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"

// The opt-in provider of builds without opt-in JIT backends.
namespace hipblaslt_jit
{
    std::vector<OptInProcessBackend> optInProcessBackends()
    {
        return {};
    }
}
