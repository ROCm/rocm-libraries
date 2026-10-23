// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_test_sdk::detail
{

// Host argument checks shared by the paged SDPA forward references
// (CpuFpReferenceSdpaRagged::forwardPaged and GpuFpReferenceSdpaRagged::fpropPaged), so both
// accept exactly the inputs that docs/SdpaPagedAttention.md and SdpaFwdNode::pre_validate_node
// allow. Every check throws std::invalid_argument prefixed with `who`.

// The extents of a paged problem, read from shapes that passed checkPagedSdpaShapes.
struct PagedSdpaShape
{
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numHeadsK = 0;
    int64_t numHeadsV = 0;
    int64_t blockSize = 0;
    int64_t numBlocksK = 0;
    int64_t numBlocksV = 0;
    int64_t maxBlocks = 0; // page-table width M == ceil(max_seq_len_kv / block_size)
};

// q [B, H, Sq, D], K container [num_blocks_k, H_k, block_size, D], V container
// [num_blocks_v, H_v, block_size, Dv], o [B, H, Sq, Dv], every dim positive (frontend tensors have
// only positive dims, so Sq >= 1; an empty sequence is seq_len_q[b] == 0, not Sq == 0).
// max_seq_len_kv >= 1, both page tables [B, 1, ceil(max_seq_len_kv / block_size), 1], and
// seq_len_kv / seq_len_q (when given) [B, 1, 1, 1].
inline PagedSdpaShape checkPagedSdpaShapes(const std::vector<int64_t>& qDims,
                                           const std::vector<int64_t>& kDims,
                                           const std::vector<int64_t>& vDims,
                                           const std::vector<int64_t>& oDims,
                                           const std::vector<int64_t>& pageTableKDims,
                                           const std::vector<int64_t>& pageTableVDims,
                                           const std::vector<int64_t>& seqLenKvDims,
                                           const std::vector<int64_t>* seqLenQDims,
                                           int64_t maxSeqLenKv,
                                           const std::string& who)
{
    const auto fail
        = [&who](const std::string& msg) { throw std::invalid_argument(who + ": " + msg); };

    if(qDims.size() != 4 || kDims.size() != 4 || vDims.size() != 4 || oDims.size() != 4)
    {
        fail("q/o must be rank-4 [B, H, Sq, D] and K/V containers rank-4 [num_blocks, H, "
             "block_size, D]");
    }

    PagedSdpaShape shape;
    shape.batch = qDims[0];
    shape.numHeads = qDims[1];
    shape.numBlocksK = kDims[0];
    shape.numHeadsK = kDims[1];
    shape.blockSize = kDims[2];
    shape.numBlocksV = vDims[0];
    shape.numHeadsV = vDims[1];

    const auto allPositive = [](const std::vector<int64_t>& dims) {
        for(const auto dim : dims)
        {
            if(dim <= 0)
            {
                return false;
            }
        }
        return true;
    };
    if(!allPositive(qDims) || !allPositive(kDims) || !allPositive(vDims))
    {
        fail("all dimensions must be positive");
    }
    if(kDims[3] != qDims[3])
    {
        fail("Q head_dim != K head_dim");
    }
    if(vDims[2] != shape.blockSize)
    {
        fail("K and V block sizes must match");
    }
    if(shape.numHeads % shape.numHeadsK != 0 || shape.numHeads % shape.numHeadsV != 0)
    {
        fail("numHeads must be divisible by numHeadsK and numHeadsV");
    }
    if(oDims[0] != shape.batch || oDims[1] != shape.numHeads || oDims[2] != qDims[2]
       || oDims[3] != vDims[3])
    {
        fail("output shape must be [B, H, Sq, Dv]");
    }

    if(maxSeqLenKv < 1)
    {
        fail("max_seq_len_kv must be >= 1 (got " + std::to_string(maxSeqLenKv) + ")");
    }
    shape.maxBlocks = (maxSeqLenKv + shape.blockSize - 1) / shape.blockSize;
    const std::vector<int64_t> pageTableDims{shape.batch, 1, shape.maxBlocks, 1};
    if(pageTableKDims != pageTableDims || pageTableVDims != pageTableDims)
    {
        fail("page tables must be [B, 1, ceil(max_seq_len_kv / block_size), 1] = [B, 1, "
             + std::to_string(shape.maxBlocks) + ", 1]");
    }

    const std::vector<int64_t> seqLenDims{shape.batch, 1, 1, 1};
    if(seqLenKvDims != seqLenDims || (seqLenQDims != nullptr && *seqLenQDims != seqLenDims))
    {
        fail("seq_len_kv and seq_len_q must be [B, 1, 1, 1]");
    }
    return shape;
}

// Q and O are both ragged or both dense, and a ragged Q comes with seq_len_q.
inline void
    checkPagedSdpaQLayout(bool raggedQ, bool raggedO, bool hasSeqLenQ, const std::string& who)
{
    if(raggedQ != raggedO)
    {
        throw std::invalid_argument(who + ": Q and O must both be ragged or both dense");
    }
    if(raggedQ && !hasSeqLenQ)
    {
        throw std::invalid_argument(who + ": ragged Q needs seq_len_q");
    }
}

// Batch b: 0 <= seq_len_kv[b] <= max_seq_len_kv and, when seq_len_q is given,
// 0 <= seq_len_q[b] <= qExtent, the rows Q holds for that batch.
inline void checkPagedSdpaSeqLens(int64_t b,
                                  int64_t seqKv,
                                  int64_t maxSeqLenKv,
                                  std::optional<int64_t> seqQ,
                                  int64_t qExtent,
                                  const std::string& who)
{
    if(seqKv < 0 || seqKv > maxSeqLenKv)
    {
        throw std::invalid_argument(
            who + ": seq_len_kv[" + std::to_string(b) + "] = " + std::to_string(seqKv)
            + " is outside [0, max_seq_len_kv = " + std::to_string(maxSeqLenKv) + "]");
    }
    if(seqQ.has_value() && (*seqQ < 0 || *seqQ > qExtent))
    {
        throw std::invalid_argument(
            who + ": seq_len_q[" + std::to_string(b) + "] = " + std::to_string(*seqQ)
            + " is outside [0, " + std::to_string(qExtent) + "], the rows q holds for that batch");
    }
}

// A used page-table entry [b, j] of `table` names a block in [0, numBlocks).
inline void checkPagedSdpaBlockId(int64_t b,
                                  int64_t j,
                                  int64_t blockId,
                                  int64_t numBlocks,
                                  const char* table,
                                  const std::string& who)
{
    if(blockId < 0 || blockId >= numBlocks)
    {
        throw std::invalid_argument(who + ": " + table + "[" + std::to_string(b) + ", "
                                    + std::to_string(j) + "] = " + std::to_string(blockId)
                                    + " is outside [0, " + std::to_string(numBlocks) + ")");
    }
}

} // namespace hipdnn_test_sdk::detail
