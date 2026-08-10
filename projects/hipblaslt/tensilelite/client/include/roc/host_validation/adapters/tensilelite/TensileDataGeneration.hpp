// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private TensileLite adapter.

#include <cstdint>
#include <roc/host_validation/generation.hpp>

namespace roc::host_validation::tensilelite_adapter
{
    inline int indexedUniformInteger(uint64_t stream, uint64_t index, int lower, int upper)
    {
        return roc::host_validation::indexedUniformInteger(
            0x54454e53494c454cULL, stream, index, lower, upper);
    }
} // namespace roc::host_validation::tensilelite_adapter
