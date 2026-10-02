// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-hipkittens.hpp"

// The opt-in provider of builds with the HipKittens backend.
namespace hipblaslt_jit
{
    std::vector<OptInProcessBackend> optInProcessBackends()
    {
        return {{{"hipkittens", "HipKittens"},
                 hipblaslt_ext::experimental::jit::hipkittens::detail::makeProcessBackend}};
    }
}
