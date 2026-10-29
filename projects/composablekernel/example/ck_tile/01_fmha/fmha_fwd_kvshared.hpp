// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "fmha_fwd.hpp"

#include <string>
#include <utility>

// Attention with K and V one tensor, for a single head:
//   q   [batch, seqlen_q, hdim]  (fp16/bf16, contiguous)
//   kv  [batch, seqlen_k, hdim]  K and V (contiguous)
//   o   [batch, seqlen_q, hdim]  dtype of q
//   lse [batch, seqlen_q]        fp32, as fmha_fwd() stores it
// make_fmha_fwd_kvshared_args() fills the arguments of fmha_fwd(). K and V get the same
// pointer and strides, which the dispatch recognizes: on gfx1250, up to 16 query rows with
// hdim 128 or 256 run qr_tdm_kvshared, which stages each K/V tile once for both gemms.
// Other shapes and archs run the regular pipelines on the same arguments.

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
