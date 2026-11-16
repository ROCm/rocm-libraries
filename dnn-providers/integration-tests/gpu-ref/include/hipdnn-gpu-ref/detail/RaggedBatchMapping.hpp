// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Per-batch row mapping of a ragged tensor, shared by the host and the HipRTC SDPA kernel
// (this file is embedded as a kernel header). Plain C++ only: no standard headers, no host-only
// types.
//
// Batch b of a ragged tensor starts at element `offsets[b] * multiplier` (its origin) and holds
// `length` token rows, row s at `origin + s * seqStride`. The length is `seqLen[b * seqLenStride]` when a
// length tensor is bound, otherwise the width of the offset block, `(offsets[b + 1] - offsets[b])
// * multiplier / seqStride`, capped at `sMax`. This is the Data SDK's `raggedBatches` rule
// (RaggedTensor.hpp) evaluated for one batch.

#pragma once

#if defined(__HIPCC__) || defined(__HIPCC_RTC__)
#define GPU_REF_RAGGED_HD __host__ __device__
#else
#define GPU_REF_RAGGED_HD
#endif

namespace hipdnn_gpu_ref
{

// Origin (elements) and live token rows of one batch.
struct RaggedBatchRows
{
    long long origin;
    long long length;
};

// `offsets` holds batch + 1 entries, entry i at `offsets[i * offsetStride]`. `seqLen` is nullptr
// when no length tensor is bound; with `seqLen` only entry b is read, and the length of batch b
// is `seqLen[b * seqLenStride]`. Both strides are the tensors' batch-axis strides in elements.
GPU_REF_RAGGED_HD inline RaggedBatchRows raggedBatchRows(const int* offsets,
                                                         long long offsetStride,
                                                         const int* seqLen,
                                                         long long seqLenStride,
                                                         long long b,
                                                         long long multiplier,
                                                         long long seqStride,
                                                         long long sMax)
{
    const long long origin = static_cast<long long>(offsets[b * offsetStride]) * multiplier;
    if(seqLen != nullptr)
    {
        return {origin, static_cast<long long>(seqLen[b * seqLenStride])};
    }
    const long long end = static_cast<long long>(offsets[(b + 1) * offsetStride]) * multiplier;
    long long length = (end - origin) / seqStride;
    // No standard headers in HipRTC kernel sources, so std::min is unavailable here.
    if(length > sMax) // NOLINT(readability-use-std-min-max)
    {
        length = sMax;
    }
    return {origin, length};
}

// Element offset of token row s of a batch, or -1 when s is not a live row (s >= length).
GPU_REF_RAGGED_HD inline long long
    raggedTokenElement(const RaggedBatchRows& rows, long long s, long long seqStride)
{
    return (s >= 0 && s < rows.length) ? rows.origin + s * seqStride : -1;
}

} // namespace hipdnn_gpu_ref
