// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/// \file
/// Gather a paged KV cache into the dense layout the GPU reference already
/// understands.
///
/// The reference executors are dense and stride-based: their SDPA argument
/// struct is q/k/v pointers plus strides, so a graph whose K/V are addressed
/// through a block table has nothing to compare against and is declined up
/// front. That decline is correct in itself -- computing something wrong would
/// be worse -- but it leaves every paged engine with no correctness story from
/// the shared harness.
///
/// The integration-tests README states the remedy and the reason to prefer it:
///
///   "Closing a gap is an adapter, not a new reference. All three are pure
///    address remapping onto the same mathematics: gather paged K/V to dense
///    through the block table ... then call the existing
///    GpuFpReferenceSdpa::fprop. Teaching the plan to do that unblocks every
///    engine of that shape at once."
///
///   "Avoid hand-rolling a private reference for one engine. A reference
///    derived from the implementation under test proves the two agree, not that
///    either is correct, and a shared misunderstanding cancels out silently."
///
/// So this header does exactly one thing: it re-addresses bytes. It performs no
/// attention arithmetic, applies no mask and makes no dtype decision, which is
/// what keeps `fprop` the single source of truth for what the answer should be.
///
/// **The layouts, taken from the committed paged bundles rather than assumed:**
///   * paged K/V: `[num_blocks, num_kv_heads, page_size, head_size]`
///   * page table: `[num_seqs, max_blocks_per_seq]`, int32 block ids
///   * `seq_len_kv`: `[num_seqs]` LENGTHS, not offsets -- paged K/V carry no
///     ragged offsets because the page table *is* the per-sequence indirection
///   * Q: `[1, num_query_heads, total_q, head_size]`, sequences packed along
///     `total_q`
///
/// A sequence's live KV is `ceil(len / page_size)` blocks gathered in page
/// order and then trimmed to the exact length, so the unused tail of a
/// partly-filled last page never contributes. Dropping that trim is the most
/// likely way to get a plausible-but-wrong answer here: the slots exist, they
/// hold whatever the allocator left, and they would silently join the softmax.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace hipdnn_test_sdk::detail
{

/// Geometry of one paged KV cache, derived from the graph's own tensors.
struct PagedKvGeometry
{
    int64_t numBlocks    = 0;
    int64_t numKvHeads   = 0;
    int64_t pageSize     = 0;
    int64_t headSize     = 0;
    int64_t numSeqs      = 0;
    int64_t maxBlocksPerSeq = 0;

    /// Dense per-sequence extent the gather produces: the largest KV length any
    /// sequence declares, rounded up to whole pages. Every sequence writes into
    /// the same dense buffer shape so the reference sees one rectangular batch.
    int64_t denseSeqLen = 0;
};

/// Derive the geometry, refusing anything whose shape does not match the layout
/// this adapter documents. Throwing here is deliberate: a silent reinterpretation
/// of a differently-shaped cache is precisely the "plausible wrong number" this
/// whole file exists to avoid.
inline PagedKvGeometry derivePagedKvGeometry(const std::vector<int64_t>& kDims,
                                             const std::vector<int64_t>& pageTableDims,
                                             const std::vector<int64_t>& kvLengths)
{
    if(kDims.size() != 4)
    {
        throw std::invalid_argument(
            "PagedKvGather: paged K/V must be rank-4 [num_blocks, num_kv_heads, page_size, "
            "head_size], got rank " + std::to_string(kDims.size()));
    }
    if(pageTableDims.size() != 2)
    {
        throw std::invalid_argument(
            "PagedKvGather: page table must be rank-2 [num_seqs, max_blocks_per_seq], got rank "
            + std::to_string(pageTableDims.size()));
    }

    PagedKvGeometry geometry;
    geometry.numBlocks       = kDims[0];
    geometry.numKvHeads      = kDims[1];
    geometry.pageSize        = kDims[2];
    geometry.headSize        = kDims[3];
    geometry.numSeqs         = pageTableDims[0];
    geometry.maxBlocksPerSeq = pageTableDims[1];

    if(static_cast<int64_t>(kvLengths.size()) != geometry.numSeqs)
    {
        throw std::invalid_argument(
            "PagedKvGather: seq_len_kv has " + std::to_string(kvLengths.size())
            + " entries but the page table describes " + std::to_string(geometry.numSeqs)
            + " sequences");
    }

    int64_t maxLength = 0;
    for(size_t seq = 0; seq < kvLengths.size(); ++seq)
    {
        const int64_t length = kvLengths[seq];
        if(length <= 0)
        {
            throw std::invalid_argument("PagedKvGather: sequence " + std::to_string(seq)
                                        + " has non-positive KV length "
                                        + std::to_string(length));
        }
        const int64_t blocksNeeded = (length + geometry.pageSize - 1) / geometry.pageSize;
        if(blocksNeeded > geometry.maxBlocksPerSeq)
        {
            throw std::invalid_argument(
                "PagedKvGather: sequence " + std::to_string(seq) + " needs "
                + std::to_string(blocksNeeded) + " blocks but the page table holds "
                + std::to_string(geometry.maxBlocksPerSeq));
        }
        maxLength = std::max(maxLength, length);
    }
    geometry.denseSeqLen = maxLength;
    return geometry;
}

/// Dense destination shape for a gathered cache: `[num_seqs, num_kv_heads,
/// denseSeqLen, head_size]`, which is the BHSD form the reference already reads.
inline std::vector<int64_t> denseKvDims(const PagedKvGeometry& geometry)
{
    return {geometry.numSeqs, geometry.numKvHeads, geometry.denseSeqLen, geometry.headSize};
}

/// Packed strides for \ref denseKvDims.
inline std::vector<int64_t> denseKvStrides(const PagedKvGeometry& geometry)
{
    const int64_t d = geometry.headSize;
    const int64_t s = geometry.denseSeqLen * d;
    const int64_t h = geometry.numKvHeads * s;
    return {h, s, d, 1};
}

/// Copy one paged cache into a dense buffer, on device.
///
/// \param pagedBase     device pointer to `[num_blocks, num_kv_heads, page_size, head_size]`
/// \param denseBase     device pointer to \ref denseKvDims, pre-allocated and zeroed
/// \param blockIds      host copy of the page table, `[num_seqs, max_blocks_per_seq]`
/// \param kvLengths     host copy of per-sequence KV lengths
/// \param elementSize   bytes per element (dtype-agnostic: this moves bytes)
///
/// Issued as one `hipMemcpyDeviceToDevice` per (sequence, head, page) run. That
/// is more copies than a bespoke kernel would need, and it is the right trade
/// for a reference path: no new kernel to be wrong, and the cost is paid once
/// per graph rather than per iteration.
inline void gatherPagedKvToDense(const void* pagedBase,
                                 void* denseBase,
                                 const PagedKvGeometry& geometry,
                                 const std::vector<int32_t>& blockIds,
                                 const std::vector<int64_t>& kvLengths,
                                 size_t elementSize)
{
    const auto* src = static_cast<const std::uint8_t*>(pagedBase);
    auto* dst       = static_cast<std::uint8_t*>(denseBase);

    // Source strides, in elements, for [num_blocks, num_kv_heads, page_size, head_size].
    const int64_t srcHeadStride  = geometry.pageSize * geometry.headSize;
    const int64_t srcBlockStride = geometry.numKvHeads * srcHeadStride;

    // Destination strides, in elements, for [num_seqs, num_kv_heads, denseSeqLen, head_size].
    const int64_t dstHeadStride = geometry.denseSeqLen * geometry.headSize;
    const int64_t dstSeqStride  = geometry.numKvHeads * dstHeadStride;

    for(int64_t seq = 0; seq < geometry.numSeqs; ++seq)
    {
        const int64_t length       = kvLengths[static_cast<size_t>(seq)];
        const int64_t blocksNeeded = (length + geometry.pageSize - 1) / geometry.pageSize;

        for(int64_t block = 0; block < blocksNeeded; ++block)
        {
            const int64_t blockId = blockIds[static_cast<size_t>(seq * geometry.maxBlocksPerSeq
                                                                 + block)];
            if(blockId < 0 || blockId >= geometry.numBlocks)
            {
                throw std::out_of_range(
                    "PagedKvGather: page table entry [" + std::to_string(seq) + ", "
                    + std::to_string(block) + "] = " + std::to_string(blockId)
                    + " is outside the cache (" + std::to_string(geometry.numBlocks)
                    + " blocks)");
            }

            // TRIM THE LAST PAGE. Only the live tokens of the final block belong
            // to this sequence; the rest of that page is unowned. Copying it
            // whole would feed uninitialised values into the softmax and produce
            // a wrong answer that still looks numerically reasonable.
            const int64_t tokenBase  = block * geometry.pageSize;
            const int64_t tokenCount = std::min(geometry.pageSize, length - tokenBase);

            for(int64_t head = 0; head < geometry.numKvHeads; ++head)
            {
                const int64_t srcOffset = blockId * srcBlockStride + head * srcHeadStride;
                const int64_t dstOffset
                    = seq * dstSeqStride + head * dstHeadStride + tokenBase * geometry.headSize;

                const size_t bytes
                    = static_cast<size_t>(tokenCount * geometry.headSize) * elementSize;
                const auto status
                    = hipMemcpy(dst + static_cast<size_t>(dstOffset) * elementSize,
                                src + static_cast<size_t>(srcOffset) * elementSize,
                                bytes,
                                hipMemcpyDeviceToDevice);
                if(status != hipSuccess)
                {
                    throw std::runtime_error(std::string("PagedKvGather: hipMemcpy failed: ")
                                             + hipGetErrorString(status));
                }
            }
        }
    }
}

} // namespace hipdnn_test_sdk::detail
