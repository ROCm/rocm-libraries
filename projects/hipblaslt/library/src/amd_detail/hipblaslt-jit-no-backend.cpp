// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-heuristic.hpp"

// The provider of builds without a JIT generator.
namespace hipblaslt_jit
{
    Status makeDefaultProcessBackend(ProcessBackend&)
    {
        return {Status::Code::NotSupported,
                Stage::Configure,
                "hipBLASLt was built without a JIT generator backend"};
    }
}
