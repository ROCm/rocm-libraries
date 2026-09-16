// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// ---------------------------------------------------------------------------
// Input generation for the w4a16 path: int4 weights in A, one 16-bit scale per
// K group of a row, and optionally a packed int4 zero-point per group.
//
// The contract mirrors generateMXInput: the packed narrow data and its scales
// are written to the caller's buffers, and the dequantized values are returned
// as floats laid out exactly like A, so the CPU reference runs as an ordinary
// float GEMM with no knowledge of the encoding.
// ---------------------------------------------------------------------------

#include "datatype_interface.hpp"
#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <random>
#include <vector>

namespace w4a16
{
    // Both unsigned weight encodings, and an absent zero-point tensor, share
    // this bias: an unsigned nibble q in [0,15] represents q - 8.
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

    /// Linear element index -> (byte, nibble) for the three weight encodings.
    /// Signed and UnsignedBias8 differ only in how the nibble is read, so they
    /// share an address; UnsignedBias8ExLlama additionally interleaves the eight
    /// nibbles of each dword as [0,2,4,6,1,3,5,7], which puts elements 2k and
    /// 2k+1 in the low and high halves of the dword.
    inline void nibbleAddr(int32_t encoding, size_t idx, size_t& byteIdx, size_t& nibble)
    {
        if(encoding != HIPBLASLT_INT4_ENCODING_UNSIGNED_BIAS8_EXLLAMA_EXT)
        {
            byteIdx = idx / 2;
            nibble  = idx % 2;
            return;
        }
        // Element j of a dword sits at bit shift (j/2)*4 + (j%2)*16, i.e. byte
        // (j%2)*2 + j/4 of the dword, nibble (j/2)%2.
        const size_t dword = idx / 8;
        const size_t j     = idx % 8;
        byteIdx            = dword * 4 + (j % 2) * 2 + (j / 4);
        nibble             = (j / 2) % 2;
    }

    inline void writeNibble(uint8_t* base, size_t byteIdx, size_t nibble, uint8_t raw)
    {
        if(nibble)
            base[byteIdx] = static_cast<uint8_t>((base[byteIdx] & 0x0F) | (raw << 4));
        else
            base[byteIdx] = static_cast<uint8_t>((base[byteIdx] & 0xF0) | raw);
    }

    /// Byte offset of the zero-point region inside the single allocation the
    /// user passes as scaleA: the scales come first, then the zero-points at the
    /// next 128-byte boundary.
    inline size_t zeroPointOffset(int64_t m, int64_t kGroups)
    {
        const size_t scaleBytes = static_cast<size_t>(m) * static_cast<size_t>(kGroups) * 2;
        return (scaleBytes + 127) / 128 * 128;
    }

    /// Total bytes of the scaleA allocation, scales plus any zero-point region.
    inline size_t scaleBytes(int64_t m, int64_t kGroups, bool zeroPoint)
    {
        if(!zeroPoint)
            return static_cast<size_t>(m) * static_cast<size_t>(kGroups) * 2;
        return zeroPointOffset(m, kGroups)
               + static_cast<size_t>((m + 1) / 2) * static_cast<size_t>(kGroups);
    }
}

/// Fill `packedA` with int4 weights and `scale` with their group scales (and,
/// when `zeroPoint`, the packed int4 zero-points that follow), and return the
/// dequantized weights as floats.
///
/// A is stored K-contiguous with row stride `lda` (the TN layout the kernels
/// require), so element (m, k) sits at m*lda + k -- in the returned vector as
/// well, which is what lets the reference GEMM take it in place of A.
///
/// The dequantized value is rounded through `scaleType` because that is what the
/// kernel does: it converts (q - z) * s to bf16/fp16 on the way into LDS, so
/// leaving the reference in f32 would make the kernel's own rounding look like
/// an error.
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
    // Scales stay positive and O(1) so the dequantized weights land in a range
    // where bf16 has its usual relative precision.
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
                // z is read back in the same domain as the weights: raw [0,15]
                // for the unsigned encodings, two's complement for the signed one.
                z = unsignedEnc ? raw : (raw ^ 0x8) - 8;
                // Zero-points share the scales' [M][kGroups] order but pack two
                // rows per byte: byte = (row/2)*kGroups + g, nibble = row & 1.
                // That is what keeps a thread's nibble fixed by its row for the
                // whole K walk.
                const size_t e = 2 * (static_cast<size_t>(row / 2) * kGroups + grp) + (row & 1);
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
            nibbleAddr(encoding, static_cast<size_t>(row) * lda + col, byteIdx, nibble);
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
