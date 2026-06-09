// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

namespace ck_tile {
namespace dispatcher {

struct HstuProblem
{
    int batch            = 1;
    int num_head         = 4;
    int hdim_qk          = 128;
    int hdim_v           = 128;
    int max_seqlen_q     = 0;
    int total_tokens     = 0;
    std::string data_type = "bf16";
    bool use_causal      = true;
    bool use_softmax     = false;
    int window_size      = 0;
    int contextual_seqlen = 0;
    int min_full_attn_seqlen = 0;
    bool has_targets     = false;

    [[nodiscard]] bool is_valid() const
    {
        return batch > 0 && num_head > 0 && hdim_qk > 0 && hdim_v > 0 && total_tokens > 0;
    }

    [[nodiscard]] std::uint64_t num_ops() const
    {
        // Dense upper bound; benchmark may apply causal sparsity factor in Python.
        return 2ULL * static_cast<std::uint64_t>(batch) * num_head * max_seqlen_q *
               max_seqlen_q * (hdim_qk + hdim_v);
    }
};

} // namespace dispatcher
} // namespace ck_tile
