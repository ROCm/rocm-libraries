// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private TensileLite adapter.

#include <cstdint>
#include <roc/host_validation/generation.hpp>

namespace roc::host_validation::tensilelite_adapter
{
    // Preserve the existing semantic-name hash. The offset basis is FNV-like,
    // not the standard 64-bit FNV-1a offset basis.
    inline constexpr uint64_t dataInitializationFnvLikeOffsetBasis = 1469598103934665603ULL;
    inline constexpr uint64_t dataInitializationFnvLikePrime       = 1099511628211ULL;

    inline constexpr uint64_t dataInitializationSeedSalt = 0x54454e53494c454cULL;
    inline constexpr uint64_t sparsePruningSeed          = 0x5350415253453234ULL;

    inline GenerationRecipeSettings dataInitializationSettings(uint64_t seed, uint64_t sequence)
    {
        return {.seed = counterRandom(seed, dataInitializationSeedSalt, sequence)};
    }
} // namespace roc::host_validation::tensilelite_adapter
