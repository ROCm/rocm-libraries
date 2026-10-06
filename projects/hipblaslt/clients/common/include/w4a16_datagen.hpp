// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Generate packed W4A16 inputs and dequantized floats for the CPU reference.

#include "datatype_interface.hpp"
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <random>
#include <vector>

namespace w4a16
{
    // Unsigned encodings: a nibble q in [0,15] represents q - 8.
    constexpr int implicitZeroPoint = 8;

    inline void storeAs(void* base, size_t idx, float v, hipDataType t)
    {
        if(t == HIP_R_16F)
            static_cast<hipblasLtHalf*>(base)[idx] = static_cast<hipblasLtHalf>(v);
        else
            static_cast<hip_bfloat16*>(base)[idx] = static_cast<hip_bfloat16>(v);
    }

    inline float loadAs(const void* base, size_t idx, hipDataType t)
    {
        return t == HIP_R_16F ? static_cast<float>(static_cast<const hipblasLtHalf*>(base)[idx])
                              : static_cast<float>(static_cast<const hip_bfloat16*>(base)[idx]);
    }

    /// Linear element index -> (byte, nibble), shared by both encodings.
    inline void nibbleAddr(size_t idx, size_t& byteIdx, size_t& nibble)
    {
        byteIdx = idx / 2;
        nibble  = idx % 2;
    }

    inline void writeNibble(uint8_t* base, size_t byteIdx, size_t nibble, uint8_t raw)
    {
        if(nibble)
            base[byteIdx] = static_cast<uint8_t>((base[byteIdx] & 0x0F) | (raw << 4));
        else
            base[byteIdx] = static_cast<uint8_t>((base[byteIdx] & 0xF0) | raw);
    }

    /// Must match c_blockScaleAZeroPointAlignment in tensile_host.cpp.
    constexpr size_t zeroPointAlignment = 256;

    /// Zero-points follow scales at the next alignment boundary.
    inline size_t zeroPointOffset(int64_t m, int64_t kGroups)
    {
        const size_t scaleBytes = static_cast<size_t>(m) * static_cast<size_t>(kGroups) * 2;
        return (scaleBytes + zeroPointAlignment - 1) / zeroPointAlignment * zeroPointAlignment;
    }

    /// Total bytes of the scaleA allocation, scales plus any zero-point region.
    inline size_t scaleBytes(int64_t m, int64_t kGroups, bool zeroPoint)
    {
        if(!zeroPoint)
            return static_cast<size_t>(m) * static_cast<size_t>(kGroups) * 2;
        return zeroPointOffset(m, kGroups)
               + static_cast<size_t>((m + 7) / 8) * static_cast<size_t>(kGroups) * 4;
    }
}

/// Generate packed weights, group scales, and optional zero-points. Return dequantized
/// weights at row stride lda, rounded through scaleType to match the kernel.
inline std::vector<float> generateW4A16Input(void*       packedA,
                                             void*       scale,
                                             hipDataType scaleType,
                                             int64_t     m,
                                             int64_t     k,
                                             int64_t     lda,
                                             int         groupSize,
                                             bool        zeroPoint,
                                             int32_t     encoding,
                                             uint32_t    seed = 24681u)
{
    using namespace w4a16;

    const int64_t kGroups     = (k + groupSize - 1) / groupSize;
    const bool    unsignedEnc = encoding != HIPBLASLT_INT4_ENCODING_SIGNED_EXT;
    auto*         weights     = static_cast<uint8_t*>(packedA);
    auto*         zeros       = static_cast<uint8_t*>(scale) + zeroPointOffset(m, kGroups);

    std::mt19937                          rng(seed);
    std::uniform_int_distribution<int>    nibbleDist(0, 15);
    // Positive and O(1), so the dequantized weights stay in bf16's good range.
    std::uniform_real_distribution<float> scaleDist(0.02f, 0.08f);

    std::vector<float> zByGroup(static_cast<size_t>(m) * kGroups, 0.0f);
    for(int64_t row = 0; row < m; ++row)
    {
        for(int64_t grp = 0; grp < kGroups; ++grp)
        {
            const size_t si = static_cast<size_t>(row) * kGroups + grp;
            storeAs(scale, si, scaleDist(rng), scaleType);

            int z = unsignedEnc ? implicitZeroPoint : 0;
            if(zeroPoint)
            {
                const int raw = nibbleDist(rng);
                // Same domain as the weights: raw [0,15] unsigned, else two's complement.
                z = unsignedEnc ? raw : (raw ^ 0x8) - 8;
                // Eight rows per word; K-groups are contiguous words.
                const size_t e = 8 * (static_cast<size_t>(row / 8) * kGroups + grp) + (row & 7);
                writeNibble(zeros, e / 2, e & 1, static_cast<uint8_t>(raw));
            }
            zByGroup[si] = static_cast<float>(z);
        }
    }

    std::vector<float> deq(static_cast<size_t>(lda) * static_cast<size_t>(m), 0.0f);
    for(int64_t row = 0; row < m; ++row)
    {
        for(int64_t col = 0; col < k; ++col)
        {
            const int raw = nibbleDist(rng);
            size_t    byteIdx, nibble;
            nibbleAddr(static_cast<size_t>(row) * lda + col, byteIdx, nibble);
            writeNibble(weights, byteIdx, nibble, static_cast<uint8_t>(raw));

            const float  q  = unsignedEnc ? static_cast<float>(raw)
                                          : static_cast<float>((raw ^ 0x8) - 8);
            const size_t si = static_cast<size_t>(row) * kGroups + col / groupSize;
            const float  s  = loadAs(scale, si, scaleType);
            const float  v  = (q - zByGroup[si]) * s;
            deq[static_cast<size_t>(row) * lda + col]
                = scaleType == HIP_R_16F ? static_cast<float>(static_cast<hipblasLtHalf>(v))
                                         : static_cast<float>(static_cast<hip_bfloat16>(v));
        }
    }
    return deq;
}
