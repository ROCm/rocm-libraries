// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

namespace asm_sdpa_engine
{

/**
 * @brief Parameters for SDPA forward kernel execution.
 *
 * Holds tensor UIDs, dimensions, strides, and attention scale
 * extracted from the operation graph.
 */
struct SdpaFwdParams
{
    // Tensor UIDs
    int64_t qUid;
    int64_t kUid;
    int64_t vUid;
    int64_t oUid;

    // Tensor dimensions
    unsigned int batchSize; // B
    unsigned int numHeadsQ; // H_q
    unsigned int numHeadsKv; // H_kv
    unsigned int seqLenQ; // S_q
    unsigned int seqLenKv; // S_kv
    unsigned int headDimQk; // D_qk (128 for POC)
    unsigned int headDimV; // D_v

    // Q tensor strides (in elements)
    unsigned int qStrideSeq;
    unsigned int qStrideRow;
    unsigned int qStrideHead;
    unsigned int qStrideBatch;

    // K tensor strides (in elements)
    unsigned int kStrideSeq;
    unsigned int kStrideHead;
    unsigned int kStrideBatch;

    // V tensor strides (in elements)
    unsigned int vStrideSeq;
    unsigned int vStrideHead;
    unsigned int vStrideBatch;

    // O tensor strides (in elements)
    unsigned int oStrideSeq;
    unsigned int oStrideHead;
    unsigned int oStrideBatch;

    // Tile size
    unsigned int tileSizeQo;

    // Architecture
    std::string archString;

    // Mask type
    bool noMask;

    // Attention scale
    float attnScale;
};

} // namespace asm_sdpa_engine
