// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

namespace hipblaslt::host_numerics
{
    inline constexpr uint64_t defaultInitializationSeed    = 69'069;
    inline constexpr uint64_t oneSpecialInitializationSeed = 12'345;

    namespace initialization
    {
        enum class OperandSequence : uint64_t
        {
            MatrixA = 0,
            MatrixB = 1,
            MatrixC = 2,
            Bias    = 3,
        };

        inline uint64_t seedForSequence(uint64_t seed, uint64_t sequence)
        {
            return seed + sequence;
        }

        inline uint64_t seedForSequence(uint64_t seed, OperandSequence sequence)
        {
            return seedForSequence(seed, static_cast<uint64_t>(sequence));
        }

        // Device initialization uses index-based generators. These offsets
        // select independent values without changing logical tensor indices.
        inline constexpr size_t complexImaginaryIndexOffset    = 1'000'000;
        inline constexpr size_t integerExactMatrixBIndexOffset = 1'000'003;
    } // namespace initialization
} // namespace hipblaslt::host_numerics
