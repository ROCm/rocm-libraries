// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// C entry points of FmhaDecodePaged5dKernel for serving frameworks (see
// fmha_decode_paged5d.py for the PyTorch wrapper). The launch configuration depends only on the
// batch, the kv heads, the window and an upper bound of the context lengths, so one call
// sequence can be captured in a HIP graph and replayed while the context lengths (read on the
// device) change below that bound.

#include "fmha_decode_paged5d_runner.hpp"

extern "C" {

// Bytes of scratch workspace for these sizes.
size_t ck_fmha_decode_paged5d_workspace_bytes(int batch, int nhead_k, int max_seqlen_k, int window)
{
    const auto c = ck_tile::fmha_decode_paged5d_config(batch, nhead_k, max_seqlen_k, window);
    return ck_tile::fmha_decode_paged5d_workspace_bytes(batch, nhead_k, c.num_splits, c.num_groups);
}

// int32 counters for this batch; zero them once, calls of any shape share them (not concurrently).
size_t ck_fmha_decode_paged5d_num_counters(int batch, int nhead_k)
{
    return ck_tile::fmha_decode_paged5d_num_counters(batch, nhead_k);
}

// q, o: [batch][nhead_q][128] bf16; k, v: SHUFFLE 5D fp8 e4m3 caches; block_table: int32 rows
// of block_table_stride entries; seqlen_k: int32 [batch], each <= max_seqlen_k; k/v descale:
// one float each. Returns a hipError_t.
int ck_fmha_decode_paged5d(const void* q,
                           const void* k,
                           const void* v,
                           const int32_t* block_table,
                           int block_table_stride,
                           const int32_t* seqlen_k,
                           const float* k_descale,
                           const float* v_descale,
                           void* o,
                           void* workspace,
                           int32_t* counters,
                           int batch,
                           int nhead_q,
                           int nhead_k,
                           int max_seqlen_k,
                           int window,
                           float scale_s,
                           hipStream_t stream)
{
    if(nhead_k <= 0 || nhead_q % nhead_k != 0 || nhead_q / nhead_k > 16)
        return hipErrorInvalidValue;
    const auto c = ck_tile::fmha_decode_paged5d_config(batch, nhead_k, max_seqlen_k, window);
    const ck_tile::FmhaDecodePaged5dHostArgs args{q,
                                                  k,
                                                  v,
                                                  block_table,
                                                  seqlen_k,
                                                  k_descale,
                                                  v_descale,
                                                  o,
                                                  workspace,
                                                  counters,
                                                  batch,
                                                  nhead_q,
                                                  nhead_k,
                                                  block_table_stride,
                                                  window,
                                                  c.pages_per_split,
                                                  c.num_groups,
                                                  scale_s};
    return fmha_decode_paged5d::launch(args, c, stream);
}
}
