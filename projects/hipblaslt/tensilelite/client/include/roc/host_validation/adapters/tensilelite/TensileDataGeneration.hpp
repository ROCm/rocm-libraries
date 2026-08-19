// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Product-private TensileLite adapter.

#include <cstdint>
#include <roc/host_validation/generation_compatibility.hpp>

namespace roc::host_validation::tensilelite_adapter
{
    // Preserve the existing semantic-name hash. The offset basis is FNV-like,
    // not the standard 64-bit FNV-1a offset basis.
    inline constexpr uint64_t dataInitializationFnvLikeOffsetBasis = 1469598103934665603ULL;
    inline constexpr uint64_t dataInitializationFnvLikePrime       = 1099511628211ULL;

    // Preserve TensileLite's historical random 2:4 pruning sequence.
    inline constexpr uint64_t sparsePruningCompatibilitySeed   = 0x54454e53494c454cULL;
    inline constexpr uint64_t sparsePruningCompatibilityStream = 1;

    // Move a legacy random stream into the seed while keeping the typed recipe
    // domain fixed. This preserves counterRandom values exactly.
    inline uint64_t
        seedForLegacyGenerationStream(uint64_t seed,
                                      uint64_t stream,
                                      uint64_t recipeDomain
                                      = generation_random_domain_version_1::realComponent)
    {
        return seed ^ (stream + counter_random_version_1::domainOffset)
               ^ (recipeDomain + counter_random_version_1::domainOffset);
    }

    inline GenerationRecipeSettings settingsForLegacyGenerationStream(uint64_t seed,
                                                                      uint64_t stream)
    {
        return {.seed = seedForLegacyGenerationStream(seed, stream)};
    }

    struct LegacyCartesianGenerationStreams
    {
        uint64_t seed;
        uint64_t realStream;
        uint64_t imaginaryStream;
    };

    // A typed Cartesian recipe has one seed and fixed component domains, so it
    // cannot represent arbitrary legacy real/imaginary stream pairs. Keep the
    // exact compatibility conversion product-private and use it only for that
    // case.
    inline void generateWithLegacyCartesianStreams(Tensor                  destination,
                                                   const GenerationRecipe& recipe,
                                                   const LegacyCartesianGenerationStreams& streams)
    {
        GenerationOptions options = legacyOptionsFromGenerationRecipe(recipe);
        options.seed              = streams.seed;
        options.real.stream       = streams.realStream;
        options.imaginary.stream  = streams.imaginaryStream;
        roc::host_validation::generate(destination, options);
    }

    inline int indexedUniformInteger(uint64_t stream, uint64_t index, int lower, int upper)
    {
        return roc::host_validation::indexedUniformInteger(
            sparsePruningCompatibilitySeed, stream, index, lower, upper);
    }
} // namespace roc::host_validation::tensilelite_adapter
