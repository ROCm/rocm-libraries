// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <vector>

namespace ck_dsl_provider {

/// Host-side marshalling for the unified paged/varlen tiled-2D attention
/// kernel. The kernel's 18-arg ABI takes three host-prepared integer
/// arrays -- the paged ``block_tables`` table, the ``cu_seqlens_q``
/// prefix-sum (the kernel's ``query_start_len_ptr``), and the
/// ``seqused_k`` per-sequence KV lengths (the kernel's ``seq_lens_ptr``)
/// -- plus the ``block_table_stride`` scalar.
///
/// These functions are PURE: they compute the arrays from the problem
/// shape with no HIP, no device allocation, and no I/O, so they are
/// directly unit-testable on a CPU-only host. The Phase-4 GPU path will
/// upload the returned ``std::vector<std::int32_t>`` buffers to device
/// memory and bind them to the corresponding kernel arg slots; that
/// device binding is intentionally out of scope here (Phase 2 verifies
/// everything up to the binary, no launch).
///
/// **Block-table addressing contract** (mirrors the kernel at
/// ``instances/common/attention_unified.py`` -- the paged-KV descriptor
/// computes ``seq_idx * ceil(max_seqlen_k / block_size) + block_idx``):
/// the per-sequence stride into ``block_tables`` is
/// ``ceil(max_seqlen_k / block_size)`` regardless of an individual
/// sequence's actual KV length, so every sequence's block row is the
/// same width.

/// Inputs describing one unified-attention problem for marshalling.
/// Pure value type; populated by the plan/execute path from the spec.
struct SdpaMarshalInputs {
    /// Number of sequences (``num_seqs``). For the dense path this is the
    /// batch ``B``; for varlen it is the number of packed sequences.
    std::int32_t num_seqs{0};

    /// Paged KV block size in tokens (one of {16, 32, 64}). Must be > 0.
    std::int32_t block_size{0};

    /// Maximum KV sequence length across all sequences. Sizes the
    /// per-sequence ``block_tables`` row width (the stride).
    std::int32_t max_seqlen_k{0};

    /// Fixed query length per sequence for the dense path (``Sq``). Unused
    /// by the varlen builders (which take explicit per-sequence lengths).
    std::int32_t seqlen_q{0};

    /// Fixed KV length per sequence for the dense path (``Skv``). Unused
    /// by the varlen builders.
    std::int32_t seqlen_k{0};
};

/// ``ceil(max_seqlen_k / block_size)`` -- the per-sequence row width into
/// the ``block_tables`` table and the ``block_table_stride`` kernel arg.
/// Throws ``std::invalid_argument`` when ``block_size <= 0``.
[[nodiscard]] std::int32_t blockTableStride(std::int32_t max_seqlen_k, std::int32_t block_size);

/// All host arrays + the stride for one marshalled problem.
struct SdpaMarshalledArrays {
    /// ``[num_seqs, block_table_stride]`` row-major paged block table.
    /// Each entry is a physical KV-cache block index.
    std::vector<std::int32_t> block_tables;

    /// ``cu_seqlens_q`` -- the query-token prefix sum, length
    /// ``num_seqs + 1``. ``cu_seqlens_q[i]`` is the first query token of
    /// sequence ``i``; ``cu_seqlens_q[num_seqs]`` is the total query
    /// count. Bound to the kernel's ``query_start_len_ptr``.
    std::vector<std::int32_t> cu_seqlens_q;

    /// ``seqused_k`` -- per-sequence KV length, length ``num_seqs``. Bound
    /// to the kernel's ``seq_lens_ptr``.
    std::vector<std::int32_t> seqused_k;

    /// ``ceil(max_seqlen_k / block_size)``; bound to the kernel's
    /// ``block_table_stride`` i32 arg.
    std::int32_t block_table_stride{0};
};

/// Marshal the DENSE problem onto the degenerate one-block-per-position
/// paged layout the unified kernel always runs. The ``block_tables``
/// table is filled with consecutive physical block indices
/// ``0, 1, 2, ...`` row-major: sequence ``s`` occupies physical blocks
/// ``[s * stride, (s + 1) * stride)``, one block per ``block_size``-token
/// span. ``cu_seqlens_q`` is the uniform prefix sum
/// ``[0, Sq, 2*Sq, ..., num_seqs*Sq]`` and ``seqused_k`` is ``Skv``
/// repeated ``num_seqs`` times. Throws ``std::invalid_argument`` on a
/// non-positive ``block_size`` or ``num_seqs``.
[[nodiscard]] SdpaMarshalledArrays marshalDenseDegenerate(const SdpaMarshalInputs& in);

/// Marshal a VARLEN problem from explicit per-sequence lengths. ``q_lens``
/// and ``k_lens`` must each have length ``num_seqs``. ``cu_seqlens_q`` is
/// the prefix sum of ``q_lens`` (length ``num_seqs + 1``); ``seqused_k``
/// is a copy of ``k_lens``. The ``block_tables`` table is the degenerate
/// consecutive-block layout sized to ``max_seqlen_k`` (the kernel reads
/// only the first ``ceil(k_lens[s]/block_size)`` blocks of each row).
/// Throws ``std::invalid_argument`` when the length vectors are the wrong
/// size or ``block_size <= 0``.
[[nodiscard]] SdpaMarshalledArrays marshalVarlen(const SdpaMarshalInputs& in,
                                                 const std::vector<std::int32_t>& q_lens,
                                                 const std::vector<std::int32_t>& k_lens);

/// Marshal a REAL-PAGED problem: the caller already owns a physical
/// ``block_tables`` table (from the graph's Page_table_K/V tensor) and the
/// per-sequence KV lengths. This validates the table size against the
/// computed stride, copies it through, builds ``seqused_k`` from
/// ``k_lens``, and synthesizes ``cu_seqlens_q`` from ``q_lens``. Throws
/// ``std::invalid_argument`` when the table size is not
/// ``num_seqs * stride`` or the length vectors are the wrong size.
[[nodiscard]] SdpaMarshalledArrays marshalRealPaged(const SdpaMarshalInputs& in,
                                                    const std::vector<std::int32_t>& block_tables,
                                                    const std::vector<std::int32_t>& q_lens,
                                                    const std::vector<std::int32_t>& k_lens);

}  // namespace ck_dsl_provider
