// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

/// @file PhiloxDropout.hpp
/// @brief The keep mask of hipDNN's Philox dropout, computed on the host (SdpaSemanticsRegister.md
///        SEM-015, "Philox mask bits").
///
/// The mask is a hipDNN extension that every engine serving Philox dropout draws: Philox4x32-10
/// keyed by the seed, one counter per 4 keys of a (batch, query head, query row), laid out as
/// AOTriton's forward kernel lays it out:
///
///   counter(b, h, i, j) = offset + ((b * Hq + h) * Sq + i) * ceil(Skv / 4) + j / 4
///
/// with the counter's low word in c0, its high word in c1 and c2 = c3 = 0, the seed's low and
/// high words as the key, and output word j % 4 as the random of key j. Element (b, h, i, j) is
/// kept iff that random, read as int32, exceeds `dropoutThreshold(p)`. Sq and Skv are the
/// graph's declared extents; the plane product is computed in 64 bits. Kept elements scale by
/// `dropoutScale(p)` = 1 / (1 - p), and the softmax's statistics are the undropped ones.
///
/// This is the reference the gauntlet grades a backward graph's Philox dropout against: an engine
/// that regenerates the mask from seed and offset is correct when its gradient equals the
/// explicit-mask gradient of the mask drawn here. The bits are pinned by a known-answer test
/// (Random123's Philox4x32-10 vectors) and by the FlyDSL engines' own host oracle.
namespace hipdnn_integration_tests::philox
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

/// The int32 a random must exceed to be kept: (p - 0.5) * 2^32 in f32, clamped to int32.
inline int32_t dropoutThreshold(float p)
{
    const float scaled = (p - 0.5F) * 4294967296.0F;
    if(!(scaled < 2147483648.0F))
    {
        return std::numeric_limits<int32_t>::max();
    }
    if(!(scaled > -2147483648.0F))
    {
        return std::numeric_limits<int32_t>::min();
    }
    return static_cast<int32_t>(scaled);
}

/// The factor a kept probability takes: 1 / (1 - p), in f32.
inline float dropoutScale(float p)
{
    return 1.0F / (1.0F - p);
}

/// The declared extents the counter layout is defined over.
struct MaskExtent
{
    int64_t batch = 1;
    int64_t heads = 1;
    int64_t seqQ = 1;
    int64_t seqKv = 1;

    int64_t elements() const
    {
        return batch * heads * seqQ * seqKv;
    }
};

/// The four randoms of one counter, wired as Triton's `randint4x` wires a 64-bit seed and offset.
inline Words randomsAt(uint64_t seed, uint64_t counter)
{
    return philox4x32(
        {static_cast<uint32_t>(counter), static_cast<uint32_t>(counter >> 32), 0U, 0U},
        static_cast<uint32_t>(seed),
        static_cast<uint32_t>(seed >> 32));
}

/// Whether element (@p b, @p h, @p i, @p j) is kept.
inline bool keeps(uint64_t seed,
                  uint64_t offset,
                  const MaskExtent& extent,
                  int64_t b,
                  int64_t h,
                  int64_t i,
                  int64_t j,
                  int32_t threshold)
{
    const auto rowStride = static_cast<uint64_t>((extent.seqKv + 3) / 4);
    const auto plane = static_cast<uint64_t>(b * extent.heads + h);
    const uint64_t counter
        = offset
          + (plane * static_cast<uint64_t>(extent.seqQ) + static_cast<uint64_t>(i)) * rowStride
          + static_cast<uint64_t>(j / 4);
    const auto words = randomsAt(seed, counter);
    return static_cast<int32_t>(words[static_cast<size_t>(j % 4)]) > threshold;
}

/// Fills the dense row-major FLOAT mask [B, Hq, Sq, Skv] at @p mask with 1 for a kept element
/// and 0 for a dropped one. Each counter is evaluated once for its four keys.
inline void fillKeepMask(
    float* mask, uint64_t seed, uint64_t offset, const MaskExtent& extent, float probability)
{
    const int32_t threshold = dropoutThreshold(probability);
    const auto rowStride = static_cast<uint64_t>((extent.seqKv + 3) / 4);
    size_t out = 0;
    for(int64_t b = 0; b < extent.batch; ++b)
    {
        for(int64_t h = 0; h < extent.heads; ++h)
        {
            const auto plane = static_cast<uint64_t>(b * extent.heads + h);
            for(int64_t i = 0; i < extent.seqQ; ++i)
            {
                const uint64_t rowBase
                    = offset
                      + (plane * static_cast<uint64_t>(extent.seqQ) + static_cast<uint64_t>(i))
                            * rowStride;
                for(int64_t group = 0; group * 4 < extent.seqKv; ++group)
                {
                    const auto words = randomsAt(seed, rowBase + static_cast<uint64_t>(group));
                    for(int64_t lane = 0; lane < 4 && group * 4 + lane < extent.seqKv; ++lane)
                    {
                        mask[out++]
                            = static_cast<int32_t>(words[static_cast<size_t>(lane)]) > threshold
                                  ? 1.0F
                                  : 0.0F;
                    }
                }
            }
        }
    }
}

} // namespace hipdnn_integration_tests::philox
