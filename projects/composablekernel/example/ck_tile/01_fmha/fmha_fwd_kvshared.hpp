// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "fmha_fwd.hpp"

#include <cmath>
#include <string>
#include <utility>

// Attention with K and V one tensor, for a single head:
//   q   [batch, seqlen_q, hdim]  (fp16/bf16, contiguous)
//   kv  [batch, seqlen_k, hdim]  K and V (contiguous)
//   o   [batch, seqlen_q, hdim]  dtype of q
//   lse [batch, seqlen_q]        fp32, as fmha_fwd() stores it
// These helpers fill the arguments of fmha_fwd() and fmha_fwd_splitkv(). K and V get the
// same pointer and strides, which the dispatch recognizes: on gfx1250, up to 16 query rows
// with hdim 128 or 256 run qr_tdm_kvshared, which stages each K/V tile once for both
// gemms. Other shapes and archs run the regular pipelines on the same arguments.

struct fmha_fwd_kvshared_problem
{
    std::string data_type; // "fp16" or "bf16"
    const void* q_ptr;
    const void* kv_ptr;
    void* o_ptr;
    void* lse_ptr; // nullptr: no LSE
    ck_tile::index_t batch;
    ck_tile::index_t seqlen_q;
    ck_tile::index_t seqlen_k;
    ck_tile::index_t hdim;
    float scale_s; // softmax scale, e.g. 1 / sqrt(hdim)
};

namespace detail {
// The fields fmha_fwd_args and fmha_fwd_splitkv_args share, for one head of contiguous
// [batch, seqlen, hdim] tensors with K and V one tensor.
template <typename Args>
void fill_fmha_fwd_kvshared_args(Args& a, const fmha_fwd_kvshared_problem& p)
{
    a.q_ptr    = p.q_ptr;
    a.k_ptr    = p.kv_ptr;
    a.v_ptr    = p.kv_ptr;
    a.bias_ptr = nullptr;
    a.lse_ptr  = p.lse_ptr;
    a.o_ptr    = p.o_ptr;

    a.seqstart_q_ptr = nullptr;
    a.seqstart_k_ptr = nullptr;
    a.sink_ptr       = nullptr;

    a.seqlen_q     = p.seqlen_q;
    a.seqlen_k     = p.seqlen_k;
    a.batch        = p.batch;
    a.max_seqlen_q = p.seqlen_q;
    a.hdim_q       = p.hdim;
    a.hdim_v       = p.hdim;
    a.nhead_q      = 1;
    a.nhead_k      = 1;

    a.scale_s         = p.scale_s;
    a.logits_soft_cap = 0.f;

    a.stride_q    = p.hdim;
    a.stride_k    = p.hdim;
    a.stride_v    = p.hdim;
    a.stride_bias = 0;
    a.stride_o    = p.hdim;

    a.nhead_stride_q    = p.seqlen_q * p.hdim;
    a.nhead_stride_k    = p.seqlen_k * p.hdim;
    a.nhead_stride_v    = p.seqlen_k * p.hdim;
    a.nhead_stride_bias = 0;
    a.nhead_stride_lse  = p.seqlen_q;
    a.nhead_stride_o    = p.seqlen_q * p.hdim;

    a.batch_stride_q    = p.seqlen_q * p.hdim;
    a.batch_stride_k    = p.seqlen_k * p.hdim;
    a.batch_stride_v    = p.seqlen_k * p.hdim;
    a.batch_stride_bias = 0;
    a.batch_stride_lse  = p.seqlen_q;
    a.batch_stride_o    = p.seqlen_q * p.hdim;

    a.window_size_left  = -1;
    a.window_size_right = -1;
    a.sink_size         = 0;
    a.mask_type         = static_cast<ck_tile::index_t>(mask_enum::no_mask);
}
} // namespace detail

// fmha_fwd(): one block per (batch, 16 query rows), each running over all of K/V.
inline std::pair<fmha_fwd_traits, fmha_fwd_args>
make_fmha_fwd_kvshared_args(const fmha_fwd_kvshared_problem& p)
{
    fmha_fwd_traits t{};
    t.hdim_q              = p.hdim;
    t.hdim_v              = p.hdim;
    t.data_type           = p.data_type;
    t.is_group_mode       = false;
    t.is_v_rowmajor       = true;
    t.has_logits_soft_cap = false;
    t.mask_type           = mask_enum::no_mask;
    t.bias_type           = bias_enum::no_bias;
    t.has_lse             = p.lse_ptr != nullptr;
    t.has_dropout         = false;
    t.qscale_type         = quant_scale_enum::no_scale;

    fmha_fwd_args a{};
    detail::fill_fmha_fwd_kvshared_args(a, p);
    a.q_descale_ptr              = nullptr;
    a.k_descale_ptr              = nullptr;
    a.v_descale_ptr              = nullptr;
    a.rand_val_ptr               = nullptr;
    a.block_scale_seqstart_q_ptr = nullptr;
    a.block_scale_seqstart_k_ptr = nullptr;
    a.seqstart_v_scale_ptr       = nullptr;
    a.min_seqlen_q               = 0;
    a.p_drop                     = 0.f;
    a.s_randval                  = false;
    a.drop_seed_offset           = std::make_pair(uint64_t{0}, uint64_t{0});
    return {t, a};
}

// Bytes of the fmha_fwd_splitkv() workspace: per split, an fp32 LSE and an fp32 O.
inline std::pair<std::size_t, std::size_t>
fmha_fwd_kvshared_splitkv_workspace_bytes(const fmha_fwd_kvshared_problem& p,
                                          ck_tile::index_t num_splits)
{
    const std::size_t rows = static_cast<std::size_t>(p.batch) * num_splits * p.seqlen_q;
    return {rows * sizeof(float), rows * p.hdim * sizeof(float)};
}

// fmha_fwd_splitkv(): K/V cut into num_splits ranges, one block per (batch, split), then a
// combine kernel; for small batches, which leave most CUs idle with fmha_fwd().
inline std::pair<fmha_fwd_splitkv_traits, fmha_fwd_splitkv_args>
make_fmha_fwd_splitkv_kvshared_args(const fmha_fwd_kvshared_problem& p,
                                    ck_tile::index_t num_splits,
                                    void* lse_acc_ptr,
                                    void* o_acc_ptr)
{
    fmha_fwd_splitkv_traits t{};
    t.hdim_q              = p.hdim;
    t.hdim_v              = p.hdim;
    t.data_type           = p.data_type;
    t.is_group_mode       = false;
    t.is_v_rowmajor       = true;
    t.has_logits_soft_cap = false;
    t.mask_type           = mask_enum::no_mask;
    t.bias_type           = bias_enum::no_bias;
    t.has_lse             = p.lse_ptr != nullptr;

    fmha_fwd_splitkv_args a{};
    detail::fill_fmha_fwd_kvshared_args(a, p);
    a.lse_acc_ptr              = lse_acc_ptr;
    a.o_acc_ptr                = o_acc_ptr;
    a.block_table_ptr          = nullptr;
    a.batch_stride_block_table = 0;
    a.page_block_size          = 0;
    a.is_gappy                 = false;
    a.cache_batch_idx          = nullptr;
    a.seqlen_k_ptr             = nullptr;
    a.num_splits               = num_splits;
    a.scale_p                  = 1.f;
    a.scale_o                  = 1.f;

    // workspace [batch, num_splits, seqlen_q(, hdim)] (one head)
    a.stride_o_acc         = p.hdim;
    a.nhead_stride_lse_acc = num_splits * p.seqlen_q;
    a.nhead_stride_o_acc   = num_splits * p.seqlen_q * p.hdim;
    a.batch_stride_lse_acc = num_splits * p.seqlen_q;
    a.batch_stride_o_acc   = num_splits * p.seqlen_q * p.hdim;
    a.split_stride_lse_acc = p.seqlen_q;
    a.split_stride_o_acc   = p.seqlen_q * p.hdim;
    return {t, a};
}
