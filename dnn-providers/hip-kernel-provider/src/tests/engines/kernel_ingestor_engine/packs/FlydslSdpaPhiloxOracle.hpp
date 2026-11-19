// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// TEST-LOCAL ORACLE: replace with hipDNN's reference once it serves dropout. Kept afterwards
// only as the known-answer pin of the mask's bits (see FlydslSdpaDropout.hpp), and kept whole
// if that reference adopts another RNG mapping, since then nothing else checks the bits.

#include <array>
#include <cstdint>

/**
 * @file FlydslSdpaPhiloxOracle.hpp
 * @brief The dropout mask every FlyDSL SDPA kernel draws, computed on the host: Philox4x32-10
 *        (Salmon et al., 2011) wired and laid out as AOTriton's forward kernel uses it.
 *
 * The scheme (AOTriton `modules/flash/kernel/fwd_kernel.py` and `dropout.py`, at b5e8cfb, the
 * aiw kernel's `philox.py`):
 * - one counter per 4 keys of a query row: counter = offset + ((b * Hq + h) * Sq + i) *
 *   cdiv(Skv, 4) + j / 4, where offset is offset1 + offset2, b the batch, h the query head, and
 *   Sq, Skv the graph's (padded) sequence lengths;
 * - the counter's low word in c0 and high word in c1, c2 = c3 = 0, the key (seed low, seed
 *   high); key j's random is output word j % 4;
 * - key j of row i is kept iff that word, read as int32, is greater than the threshold
 *   (FlydslSdpaDropout.hpp dropoutThreshold), a signed compare.
 *
 * Word order: AOTriton's main branch (0.15, b5e8cfb) puts word r in column 4g + r, and so do the
 * FlyDSL kernels. The AOTriton 0.11.1 that PyTorch 2.10 bundles puts words 1 and 2 the other way
 * round (columns 4g + {0, 2, 1, 3}); measured against both, see WORD_FOR_COLUMN_AOTRITON_0_11.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::philox_oracle
{

using Words = std::array<uint32_t, 4>;

/// Philox4x32 with @p rounds rounds over counter @p counter and key (@p key0, @p key1).
inline Words philox4x32(Words counter, uint32_t key0, uint32_t key1, int rounds = 10)
{
    constexpr uint64_t MULTIPLIER_0 = 0xD2511F53U;
    constexpr uint64_t MULTIPLIER_1 = 0xCD9E8D57U;
    constexpr uint32_t WEYL_0 = 0x9E3779B9U;
    constexpr uint32_t WEYL_1 = 0xBB67AE85U;
    for(int round = 0; round < rounds; ++round)
    {
        const uint64_t product0 = MULTIPLIER_0 * counter[0];
        const uint64_t product1 = MULTIPLIER_1 * counter[2];
        counter = {static_cast<uint32_t>(product1 >> 32) ^ counter[1] ^ key0,
                   static_cast<uint32_t>(product1),
                   static_cast<uint32_t>(product0 >> 32) ^ counter[3] ^ key1,
                   static_cast<uint32_t>(product0)};
        key0 += WEYL_0;
        key1 += WEYL_1;
    }
    return counter;
}

/// The four randoms of one counter, wired as Triton's `randint4x` wires a 64-bit seed and offset.
inline Words randomsAt(uint64_t seed, uint64_t counter)
{
    return philox4x32(
        {static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), 0U, 0U},
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32));
}

/// The word each key column takes, by column % 4: AOTriton main's (and the FlyDSL kernels').
inline constexpr std::array<int, 4> WORD_FOR_COLUMN = {0, 1, 2, 3};
/// The same, as the AOTriton 0.11.1 bundled with PyTorch 2.10 lays them out.
inline constexpr std::array<int, 4> WORD_FOR_COLUMN_AOTRITON_0_11 = {0, 2, 1, 3};

/// Where one mask element's random lives in the stream.
struct MaskCoordinates
{
    int64_t batch = 0;
    int64_t head = 0;
    int64_t row = 0;
    int64_t key = 0;
    int64_t numHeads = 1;
    int64_t seqLenQ = 1;
    int64_t seqLenKv = 1;
};

/// The counter holding element @p at, for a Philox offset of @p offset.
inline uint64_t counterOf(uint64_t offset, const MaskCoordinates& at)
{
    const auto rowStride = static_cast<uint64_t>((at.seqLenKv + 3) / 4);
    const auto plane = static_cast<uint64_t>(at.batch * at.numHeads + at.head);
    return offset
           + (plane * static_cast<uint64_t>(at.seqLenQ) + static_cast<uint64_t>(at.row)) * rowStride
           + static_cast<uint64_t>(at.key / 4);
}

/// Element @p at's random, as the kernels compare it: an int32.
inline int32_t randomOf(uint64_t seed,
                        uint64_t offset,
                        const MaskCoordinates& at,
                        const std::array<int, 4>& wordForColumn = WORD_FOR_COLUMN)
{
    const auto words = randomsAt(seed, counterOf(offset, at));
    return static_cast<int32_t>(
        words[static_cast<size_t>(wordForColumn[static_cast<size_t>(at.key % 4)])]);
}

/// Whether element @p at is kept under threshold @p threshold.
inline bool keeps(uint64_t seed,
                  uint64_t offset,
                  const MaskCoordinates& at,
                  int32_t threshold,
                  const std::array<int, 4>& wordForColumn = WORD_FOR_COLUMN)
{
    return randomOf(seed, offset, at, wordForColumn) > threshold;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::philox_oracle
