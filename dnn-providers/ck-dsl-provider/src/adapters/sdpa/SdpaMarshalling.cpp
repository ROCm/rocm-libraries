// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "SdpaMarshalling.hpp"

#include <stdexcept>
#include <string>

namespace ck_dsl_provider {

namespace {

void requirePositiveBlockSize(std::int32_t block_size) {
    if (block_size <= 0) {
        throw std::invalid_argument("SdpaMarshalling: block_size must be positive (got " +
                                    std::to_string(block_size) + ")");
    }
}

void requireLengthVector(const std::vector<std::int32_t>& v, std::int32_t expected,
                         const char* role) {
    if (static_cast<std::int32_t>(v.size()) != expected) {
        throw std::invalid_argument(std::string("SdpaMarshalling: ") + role + " must have length " +
                                    std::to_string(expected) + " (got " + std::to_string(v.size()) +
                                    ")");
    }
}

/// Build the degenerate consecutive-block ``block_tables`` table:
/// ``num_seqs`` rows of ``stride`` entries, filled ``0, 1, 2, ...`` so
/// sequence ``s`` owns physical blocks ``[s*stride, (s+1)*stride)``. This
/// is the one-block-per-position layout the dense and varlen paths run on
/// the always-paged unified kernel.
std::vector<std::int32_t> consecutiveBlockTable(std::int32_t num_seqs, std::int32_t stride) {
    std::vector<std::int32_t> table;
    table.reserve(static_cast<std::size_t>(num_seqs) * static_cast<std::size_t>(stride));
    std::int32_t physical = 0;
    for (std::int32_t s = 0; s < num_seqs; ++s) {
        for (std::int32_t b = 0; b < stride; ++b) {
            table.push_back(physical);
            ++physical;
        }
    }
    return table;
}

/// Prefix sum of ``q_lens`` into a length ``num_seqs + 1`` cu_seqlens_q
/// (``[0, q_lens[0], q_lens[0]+q_lens[1], ...]``).
std::vector<std::int32_t> prefixSum(const std::vector<std::int32_t>& q_lens) {
    std::vector<std::int32_t> cu;
    cu.reserve(q_lens.size() + 1);
    std::int32_t running = 0;
    cu.push_back(running);
    for (const std::int32_t len : q_lens) {
        running += len;
        cu.push_back(running);
    }
    return cu;
}

}  // namespace

std::int32_t blockTableStride(std::int32_t max_seqlen_k, std::int32_t block_size) {
    requirePositiveBlockSize(block_size);
    if (max_seqlen_k <= 0) {
        return 0;
    }
    return (max_seqlen_k + block_size - 1) / block_size;
}

SdpaMarshalledArrays marshalDenseDegenerate(const SdpaMarshalInputs& in) {
    requirePositiveBlockSize(in.block_size);
    if (in.num_seqs <= 0) {
        throw std::invalid_argument("SdpaMarshalling: num_seqs must be positive (got " +
                                    std::to_string(in.num_seqs) + ")");
    }

    const std::int32_t stride = blockTableStride(in.max_seqlen_k, in.block_size);

    SdpaMarshalledArrays out;
    out.block_table_stride = stride;
    out.block_tables = consecutiveBlockTable(in.num_seqs, stride);

    // Uniform query prefix sum [0, Sq, 2Sq, ...] (length num_seqs + 1).
    out.cu_seqlens_q.reserve(static_cast<std::size_t>(in.num_seqs) + 1);
    for (std::int32_t s = 0; s <= in.num_seqs; ++s) {
        out.cu_seqlens_q.push_back(s * in.seqlen_q);
    }

    // Uniform per-sequence KV length: Skv repeated num_seqs times.
    out.seqused_k.assign(static_cast<std::size_t>(in.num_seqs), in.seqlen_k);

    return out;
}

SdpaMarshalledArrays marshalVarlen(const SdpaMarshalInputs& in,
                                   const std::vector<std::int32_t>& q_lens,
                                   const std::vector<std::int32_t>& k_lens) {
    requirePositiveBlockSize(in.block_size);
    requireLengthVector(q_lens, in.num_seqs, "q_lens");
    requireLengthVector(k_lens, in.num_seqs, "k_lens");

    const std::int32_t stride = blockTableStride(in.max_seqlen_k, in.block_size);

    SdpaMarshalledArrays out;
    out.block_table_stride = stride;
    out.block_tables = consecutiveBlockTable(in.num_seqs, stride);
    out.cu_seqlens_q = prefixSum(q_lens);
    out.seqused_k = k_lens;

    return out;
}

SdpaMarshalledArrays marshalRealPaged(const SdpaMarshalInputs& in,
                                      const std::vector<std::int32_t>& block_tables,
                                      const std::vector<std::int32_t>& q_lens,
                                      const std::vector<std::int32_t>& k_lens) {
    requirePositiveBlockSize(in.block_size);
    requireLengthVector(q_lens, in.num_seqs, "q_lens");
    requireLengthVector(k_lens, in.num_seqs, "k_lens");

    const std::int32_t stride = blockTableStride(in.max_seqlen_k, in.block_size);
    const std::int32_t expectedTableSize = in.num_seqs * stride;
    if (static_cast<std::int32_t>(block_tables.size()) != expectedTableSize) {
        throw std::invalid_argument(
            "SdpaMarshalling: real-paged block_tables size must equal num_seqs * "
            "ceil(max_seqlen_k/block_size) = " +
            std::to_string(expectedTableSize) + " (got " + std::to_string(block_tables.size()) +
            ")");
    }

    SdpaMarshalledArrays out;
    out.block_table_stride = stride;
    out.block_tables = block_tables;
    out.cu_seqlens_q = prefixSum(q_lens);
    out.seqused_k = k_lens;

    return out;
}

}  // namespace ck_dsl_provider
