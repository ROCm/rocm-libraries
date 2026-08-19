// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_validation/generation_primitives.hpp>

namespace hipblaslt::host_validation
{
    inline constexpr uint64_t defaultInitializationSeed    = 69'069;
    inline constexpr uint64_t oneSpecialInitializationSeed = 12'345;

    namespace initialization
    {
        inline constexpr uint64_t sequenceSeedSalt = 0x484950424c41534cULL;
        inline constexpr uint64_t integerExactMatrixBSequence = 1;

        inline uint64_t seedForSequence(uint64_t seed, uint64_t sequence)
        {
            return roc::host_validation::counterRandom(seed, sequenceSeedSalt, sequence);
        }

        // Device initialization uses index-based generators. These offsets
        // select independent values without changing logical tensor indices.
        inline constexpr size_t complexImaginaryIndexOffset    = 1'000'000;
        inline constexpr size_t integerExactMatrixBIndexOffset = 1'000'003;

        // Host and device one-special initialization use the same recurrence
        // to select the overwritten element and sentinel kind.
        inline constexpr uint32_t oneSpecialLcgMultiplier = 1'103'515'245u;
        inline constexpr uint32_t oneSpecialLcgIncrement  = 12'345u;
        inline constexpr unsigned oneSpecialLcgValueShift = 16;
        inline constexpr int      oneSpecialValueCount    = 3;
    } // namespace initialization
} // namespace hipblaslt::host_validation
