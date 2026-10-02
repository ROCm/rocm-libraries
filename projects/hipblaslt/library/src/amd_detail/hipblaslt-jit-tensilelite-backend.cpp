// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-tensilelite.hpp"

// The provider of builds that generate with TensileLite.
namespace hipblaslt_jit
{
    Status makeDefaultProcessBackend(ProcessBackend& made)
    {
        return hipblaslt_ext::experimental::jit::tensilelite::detail::makeProcessBackend(made);
    }

    ProcessBackendName defaultProcessBackendName()
    {
        return {"tensilelite", "TensileLite"};
    }
}
