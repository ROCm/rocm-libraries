// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "sparse_attention.h"
#include "sparse_attn_fwd.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/hip_check_error.hpp"
#include "ck_tile/host/stream_config.hpp"
#include <hip/hip_runtime.h>
#include <type_traits>
#include <cassert>
#include <iostream>

namespace {

ck_tile::index_t get_causal_type(mask_enum type)
{
    switch(type)
    {
    case mask_enum::mask_top_left: return 1;
    case mask_enum::mask_bottom_right: return 2;
    case mask_enum::no_mask: // fall through
    case mask_enum::window_generic: return 0;
    }
    return 0;
}

struct StrideInfo
{
    ck_tile::index_t stride_q, stride_k, stride_v, stride_o;
    ck_tile::index_t nhead_stride_q, nhead_stride_k, nhead_stride_v, nhead_stride_o;
    ck_tile::index_t batch_stride_q, batch_stride_k, batch_stride_v, batch_stride_o;
};

StrideInfo compute_strides(int nhead,
                                  int nhead_k,
                                  int seqlen_q,
                                  int seqlen_k,
                                  int hdim_q,
                                  int hdim_v,
                                  bool i_perm,
                                  bool o_perm,
                                  bool is_v_rowmajor)
{
    StrideInfo s{};
    const auto sq = static_cast<ck_tile::index_t>(seqlen_q);
    const auto sk = static_cast<ck_tile::index_t>(seqlen_k);

    s.stride_q = (i_perm ? hdim_q : nhead * hdim_q);
    s.stride_k = (i_perm ? hdim_q : nhead_k * hdim_q);
    s.stride_v = is_v_rowmajor ? (i_perm ? hdim_v : nhead_k * hdim_v)
                               : (i_perm ? sk : nhead_k * sk);
    s.stride_o = (o_perm ? hdim_v : nhead * hdim_v);

    s.nhead_stride_q = (i_perm ? sq * hdim_q : hdim_q);
    s.nhead_stride_k = (i_perm ? sk * hdim_q : hdim_q);
    s.nhead_stride_v = is_v_rowmajor ? (i_perm ? sk * hdim_v : hdim_v)
                                     : (i_perm ? hdim_v * sk : sk);
    s.nhead_stride_o = (o_perm ? sq * hdim_v : hdim_v);

    s.batch_stride_q = (nhead * sq * hdim_q);
    s.batch_stride_k = (nhead_k * sk * hdim_q);
    s.batch_stride_v = (nhead_k * hdim_v * sk);
    s.batch_stride_o = (nhead * sq * hdim_v);
    return s;
}

template <typename TraitsT>
void fill_common_traits(TraitsT& traits,
                        int hdim_q, int hdim_v,
                        const std::string& data_type,
                        bool is_v_rowmajor,
                        const mask_info& mask,
                        const sparse_attn_bias_args& bias,
                        float logits_soft_cap)
{
    traits.hdim_q        = hdim_q;
    traits.hdim_v        = hdim_v;
    traits.data_type     = data_type;
    traits.is_v_rowmajor = is_v_rowmajor;
    traits.mask_type     = mask.type;
    traits.bias_type     = bias.type;
    traits.has_logits_soft_cap = (logits_soft_cap > 0.0f);
}

template <typename ArgsT>
void fill_common_args(ArgsT& args,
                      const ck_tile::DeviceMem& q_buf,
                      const ck_tile::DeviceMem& k_buf,
                      const ck_tile::DeviceMem& v_buf,
                      const ck_tile::DeviceMem& o_buf,
                      int batch, int nhead, int nhead_k,
                      int seqlen_q, int seqlen_k, int max_seqlen_q,
                      int hdim_q, int hdim_v,
                      const StrideInfo& st,
                      const mask_info& mask,
                      float scale_s,
                      const sparse_attn_bias_args& bias,
                      float logits_soft_cap)
{
    args.q_ptr          = q_buf.GetDeviceBuffer();
    args.k_ptr          = k_buf.GetDeviceBuffer();
    args.v_ptr          = v_buf.GetDeviceBuffer();
    args.o_ptr          = o_buf.GetDeviceBuffer();
    args.batch          = batch;
    args.seqlen_q       = seqlen_q;
    args.seqlen_k       = seqlen_k;
    args.max_seqlen_q   = max_seqlen_q;
    args.hdim_q         = hdim_q;
    args.hdim_v         = hdim_v;
    args.nhead_q        = nhead;
    args.nhead_k        = nhead_k;
    args.scale_s        = scale_s;
    args.stride_q       = st.stride_q;
    args.stride_k       = st.stride_k;
    args.stride_v       = st.stride_v;
    args.stride_o       = st.stride_o;
    args.nhead_stride_q = st.nhead_stride_q;
    args.nhead_stride_k = st.nhead_stride_k;
    args.nhead_stride_v = st.nhead_stride_v;
    args.nhead_stride_o = st.nhead_stride_o;
    args.batch_stride_q = st.batch_stride_q;
    args.batch_stride_k = st.batch_stride_k;
    args.batch_stride_v = st.batch_stride_v;
    args.batch_stride_o = st.batch_stride_o;
    args.window_size_left  = mask.left;
    args.window_size_right = mask.right;
    args.mask_type         = static_cast<ck_tile::index_t>(mask.type);
    args.bias_ptr          = bias.ptr;
    args.stride_bias       = bias.stride_bias;
    args.nhead_stride_bias = bias.nhead_stride_bias;
    args.batch_stride_bias = bias.batch_stride_bias;
    args.logits_soft_cap   = logits_soft_cap;
}

} // anonymous namespace

template <typename DataType_>
float jenga_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,
                             const ck_tile::HostTensor<DataType_>& TK,
                             const ck_tile::HostTensor<DataType_>& TV,
                             const ck_tile::HostTensor<uint8_t>& Tmask,
                             ck_tile::HostTensor<DataType_>& Y,
                             int batch,
                             int nhead,
                             int nhead_k,
                             int seqlen_q,
                             int seqlen_k,
                             int hdim_q,
                             int hdim_v,
                             bool i_perm,
                             bool o_perm,
                             bool is_v_rowmajor,
                             int max_seqlen_q,
                             int max_seqlen_k,
                             const ck_tile::stream_config& stream_config,
                             const std::string& mask_str,
                             const sparse_attn_bias_args& bias,
                             float scale_s_user,
                             float logits_soft_cap,
                             const std::vector<int32_t>& seqstart_q_host,
                             const std::vector<int32_t>& seqstart_k_host,
                             const std::vector<int32_t>& seqstart_q_block_host,
                             const std::vector<int32_t>& mask_batch_offsets)
{
    static_assert(std::is_same_v<DataType_, ck_tile::half_t> ||
                      std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "Jenga sparse attention supports fp16/bf16 only.");

    const bool is_group_mode = !seqstart_q_host.empty();
    const char* tag = is_group_mode ? "[jenga group]" : "[jenga]";

    const std::string data_type =
        std::is_same_v<DataType_, ck_tile::bf16_t> ? "bf16" : "fp16";

    const int32_t total_q = is_group_mode ? seqstart_q_host.back() : seqlen_q;
    const int32_t total_k = is_group_mode ? seqstart_k_host.back() : seqlen_k;
    if(max_seqlen_q == 0) max_seqlen_q = total_q;
    if(max_seqlen_k == 0) max_seqlen_k = total_k;
    (void)max_seqlen_k;

    if(bias.type < 0 || bias.type > 2)
    {
        std::cerr << tag << " invalid bias.type=" << bias.type << " (expected 0/1/2)\n";
        return -1.0f;
    }
    // jenga + ALIBI shows numerical drift amplified by alibi slope*k (edge-tile
    // mask elision). Reject in both modes to avoid silently wrong outputs.
    if(bias.type == 2)
    {
        std::cerr << tag << " alibi bias not supported — known numerical drift "
                     "(use sparge or vsa for alibi workloads).\n";
        return -1.0f;
    }
    // Batch mode also rejects mask=t/b/swa for the same edge-tile mask elision
    // reason; group mode is unaffected.
    if(!is_group_mode)
    {
        const mask_info probe = mask_info::decode(mask_str, seqlen_q, seqlen_k);
        if(probe.type != mask_enum::no_mask)
        {
            std::cerr << tag << " -mask=" << mask_str << " not supported in batch mode "
                         "— known numerical drift from kernel/CPU-ref mask divergence "
                         "(use mask=0 or sparge / vsa for masked workloads).\n";
            return -1.0f;
        }
    }

    const float scale_s = (scale_s_user != 0.0f)
                              ? scale_s_user
                              : 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q));
    mask_info mask = mask_info::decode(mask_str, total_q, total_k);
    auto st = compute_strides(nhead, nhead_k, total_q, total_k,
                              hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor);

    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(TV.get_element_space_size_in_bytes());
    ck_tile::DeviceMem mask_buf(Tmask.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(Y.get_element_space_size_in_bytes());
    o_buf.SetZero();

    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    v_buf.ToDevice(TV.data());
    mask_buf.ToDevice(Tmask.data());

    ck_tile::DeviceMem seqstart_q_buf(is_group_mode ? seqstart_q_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_buf(is_group_mode ? seqstart_k_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_q_block_buf(is_group_mode ? seqstart_q_block_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem mask_batch_offsets_buf(is_group_mode ? mask_batch_offsets.size() * sizeof(int32_t) : 0);
    if(is_group_mode)
    {
        seqstart_q_buf.ToDevice(seqstart_q_host.data());
        seqstart_k_buf.ToDevice(seqstart_k_host.data());
        seqstart_q_block_buf.ToDevice(seqstart_q_block_host.data());
        mask_batch_offsets_buf.ToDevice(mask_batch_offsets.data());
    }

    fmha_jenga_fwd_traits fmha_traits;
    fill_common_traits(fmha_traits, hdim_q, hdim_v, data_type, is_v_rowmajor, mask, bias, logits_soft_cap);
    fmha_traits.is_group_mode = is_group_mode;

    fmha_jenga_fwd_args args;
    assert(nhead % nhead_k == 0);
    fill_common_args(args, q_buf, k_buf, v_buf, o_buf,
                     batch, nhead, nhead_k, total_q, total_k, max_seqlen_q,
                     hdim_q, hdim_v, st, mask, scale_s, bias, logits_soft_cap);
    args.block_relation_onehot_ptr = mask_buf.GetDeviceBuffer();
    if(is_group_mode)
    {
        args.seqstart_q_ptr        = seqstart_q_buf.GetDeviceBuffer();
        args.seqstart_k_ptr        = seqstart_k_buf.GetDeviceBuffer();
        args.seqstart_q_block_ptr  = seqstart_q_block_buf.GetDeviceBuffer();
        args.mask_batch_offset_ptr = mask_batch_offsets_buf.GetDeviceBuffer();
    }

    float ave_time = fmha_jenga_fwd(fmha_traits, args, stream_config);
    if(ave_time < 0)
    {
        std::cerr << tag << " ERROR: dispatch failed (returned " << ave_time
                  << "). mask_type=" << static_cast<int>(fmha_traits.mask_type)
                  << ", data_type=" << fmha_traits.data_type << std::endl;
    }

    HIP_CHECK_ERROR(hipStreamSynchronize(stream_config.stream_id_));
    o_buf.FromDevice(Y.data(), Y.get_element_space_size_in_bytes());
    return ave_time;
}

template <typename DataType_>
float vsa_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,
                           const ck_tile::HostTensor<DataType_>& TK,
                           const ck_tile::HostTensor<DataType_>& TV,
                           const ck_tile::HostTensor<int32_t>& TLUT,
                           const ck_tile::HostTensor<int32_t>& TVBN,
                           ck_tile::HostTensor<DataType_>& Y,
                           int batch,
                           int nhead,
                           int nhead_k,
                           int seqlen_q,
                           int seqlen_k,
                           int hdim_q,
                           int hdim_v,
                           bool i_perm,
                           bool o_perm,
                           bool is_v_rowmajor,
                           int max_seqlen_q,
                           int max_seqlen_k,
                           const ck_tile::stream_config& stream_config,
                           const std::string& mask_str,
                           const sparse_attn_bias_args& bias,
                           float scale_s_user,
                           float logits_soft_cap,
                           const std::vector<int32_t>& seqstart_q_host,
                           const std::vector<int32_t>& seqstart_k_host,
                           const std::vector<int32_t>& seqstart_q_block_host,
                           const std::vector<int32_t>& lut_batch_offsets)
{
    static_assert(std::is_same_v<DataType_, ck_tile::half_t> ||
                      std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "VSA sparse attention supports fp16/bf16 only.");

    const bool is_group_mode = !seqstart_q_host.empty();
    const char* tag = is_group_mode ? "[vsa group]" : "[vsa]";

    const std::string data_type =
        std::is_same_v<DataType_, ck_tile::bf16_t> ? "bf16" : "fp16";

    const int32_t total_q = is_group_mode ? seqstart_q_host.back() : seqlen_q;
    const int32_t total_k = is_group_mode ? seqstart_k_host.back() : seqlen_k;
    if(max_seqlen_q == 0) max_seqlen_q = total_q;
    if(max_seqlen_k == 0) max_seqlen_k = total_k;
    (void)max_seqlen_k;

    if(bias.type < 0 || bias.type > 2)
    {
        std::cerr << tag << " invalid bias.type=" << bias.type << " (expected 0/1/2)\n";
        return -1.0f;
    }

    const float scale_s = (scale_s_user != 0.0f)
                              ? scale_s_user
                              : 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q));
    mask_info mask = mask_info::decode(mask_str, total_q, total_k);
    auto st = compute_strides(nhead, nhead_k, total_q, total_k,
                              hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor);

    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(TV.get_element_space_size_in_bytes());
    ck_tile::DeviceMem lut_buf(TLUT.get_element_space_size_in_bytes());
    ck_tile::DeviceMem vbn_buf(TVBN.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(Y.get_element_space_size_in_bytes());
    o_buf.SetZero();

    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    v_buf.ToDevice(TV.data());
    lut_buf.ToDevice(TLUT.data());
    vbn_buf.ToDevice(TVBN.data());

    ck_tile::DeviceMem seqstart_q_buf(is_group_mode ? seqstart_q_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_buf(is_group_mode ? seqstart_k_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_q_block_buf(is_group_mode ? seqstart_q_block_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem lut_batch_offsets_buf(is_group_mode ? lut_batch_offsets.size() * sizeof(int32_t) : 0);
    if(is_group_mode)
    {
        seqstart_q_buf.ToDevice(seqstart_q_host.data());
        seqstart_k_buf.ToDevice(seqstart_k_host.data());
        seqstart_q_block_buf.ToDevice(seqstart_q_block_host.data());
        lut_batch_offsets_buf.ToDevice(lut_batch_offsets.data());
    }

    fmha_vsa_fwd_traits fmha_traits;
    fill_common_traits(fmha_traits, hdim_q, hdim_v, data_type, is_v_rowmajor, mask, bias, logits_soft_cap);
    fmha_traits.is_group_mode = is_group_mode;

    fmha_vsa_fwd_args args;
    assert(nhead % nhead_k == 0);
    fill_common_args(args, q_buf, k_buf, v_buf, o_buf,
                     batch, nhead, nhead_k, total_q, total_k, max_seqlen_q,
                     hdim_q, hdim_v, st, mask, scale_s, bias, logits_soft_cap);
    args.lut_ptr             = lut_buf.GetDeviceBuffer();
    args.valid_block_num_ptr = vbn_buf.GetDeviceBuffer();
    if(is_group_mode)
    {
        args.seqstart_q_ptr       = seqstart_q_buf.GetDeviceBuffer();
        args.seqstart_k_ptr       = seqstart_k_buf.GetDeviceBuffer();
        args.seqstart_q_block_ptr = seqstart_q_block_buf.GetDeviceBuffer();
        args.lut_batch_offset_ptr = lut_batch_offsets_buf.GetDeviceBuffer();
    }

    float ave_time = fmha_vsa_fwd(fmha_traits, args, stream_config);
    if(ave_time < 0)
    {
        std::cerr << tag << " ERROR: dispatch failed (returned " << ave_time
                  << "). mask_type=" << static_cast<int>(fmha_traits.mask_type)
                  << ", data_type=" << fmha_traits.data_type << std::endl;
    }

    HIP_CHECK_ERROR(hipStreamSynchronize(stream_config.stream_id_));
    o_buf.FromDevice(Y.data(), Y.get_element_space_size_in_bytes());
    return ave_time;
}

template <typename DataType_>
float sparge_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,
                                    const ck_tile::HostTensor<DataType_>& TK,
                                    const ck_tile::HostTensor<DataType_>& TV,
                                    ck_tile::HostTensor<DataType_>& Y,
                                    int batch,
                                    int nhead,
                                    int nhead_k,
                                    int seqlen_q,
                                    int seqlen_k,
                                    int hdim_q,
                                    int hdim_v,
                                    bool i_perm,
                                    bool o_perm,
                                    bool is_v_rowmajor,
                                    const ck_tile::sparge_hyperparam_args& hp,
                                    const std::string& mask_str,
                                    bool attention_sink,
                                    int block_size,
                                    int max_seqlen_q,
                                    int max_seqlen_k,
                                    float* sparsity_out,
                                    const ck_tile::stream_config& stream_config,
                                    const sparse_attn_bias_args& bias,
                                    float scale_s_user,
                                    float logits_soft_cap,
                                    const std::vector<int32_t>& seqlen_qs,
                                    const std::vector<int32_t>& seqlen_ks)
{
    static_assert(std::is_same_v<DataType_, ck_tile::half_t> ||
                      std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "SpargeAttention supports fp16/bf16 only.");

    const bool is_group_mode = !seqlen_qs.empty();
    const char* tag = is_group_mode ? "[sparge group]" : "[sparge]";

    assert(hdim_q == 128 && hdim_v == 128 && block_size == 128 &&
           "SpargeAttention currently supports only hdim=128, block_size=128");

    if(bias.type < 0 || bias.type > 2)
    {
        std::cerr << tag << " invalid bias.type=" << bias.type
                  << " (expected 0=no, 1=elementwise, 2=alibi).\n";
        return -1.0f;
    }
    if(!is_group_mode && bias.type != 0 && bias.ptr == nullptr)
    {
        std::cerr << tag << " error: bias.type=" << bias.type
                  << " but bias.ptr==nullptr.\n";
        return -1.0f;
    }
    if((hp.cdfthreshd > 0.0f) == (hp.topk > 0.0f))
    {
        std::cerr << tag << " error: exactly one of hp.cdfthreshd / hp.topk must be > 0\n";
        return -1.0f;
    }
    if(!is_group_mode && hp.simthreshold_per_head_ptr != nullptr && hp.simthreshold <= 0.0f)
    {
        std::cerr << tag << " warning: simthreshold_per_head_ptr set but scalar "
                     "hp.simthreshold <= 0 → ptr ignored\n";
    }
    // Mask-prediction sort-buffer cap (kMaxKBlocksPow2=256 in
    // block_sparge_mask_pipelines.hpp): per-sequence seqlen_k must fit in 256
    // K-blocks; above this the kernel-side assert fires only after LDS corruption.
    {
        constexpr int kMaxKBlocks = 256;
        if(is_group_mode)
        {
            for(size_t b = 0; b < seqlen_ks.size(); ++b)
            {
                const int num_k_blocks_b = (seqlen_ks[b] + block_size - 1) / block_size;
                if(num_k_blocks_b > kMaxKBlocks)
                {
                    std::cerr << tag << " error: seqlen_k[" << b << "]=" << seqlen_ks[b]
                              << " (=" << num_k_blocks_b << " K-blocks @ block_size="
                              << block_size << ") exceeds kMaxKBlocksPow2=" << kMaxKBlocks
                              << " (max " << kMaxKBlocks * block_size << " tokens per seq).\n";
                    return -1.0f;
                }
            }
        }
        else
        {
            const int num_k_blocks = (seqlen_k + block_size - 1) / block_size;
            if(num_k_blocks > kMaxKBlocks)
            {
                std::cerr << tag << " error: seqlen_k=" << seqlen_k
                          << " (=" << num_k_blocks << " K-blocks @ block_size="
                          << block_size << ") exceeds kMaxKBlocksPow2=" << kMaxKBlocks
                          << " (max " << kMaxKBlocks * block_size << " tokens).\n";
                return -1.0f;
            }
        }
    }

    // Group mode: locally cumsum seqlen_qs/ks → seqstart and per-batch block tables.
    auto cumsum = [](const std::vector<int32_t>& v) {
        std::vector<int32_t> out(v.size() + 1, 0);
        for(size_t i = 0; i < v.size(); ++i) out[i + 1] = out[i] + v[i];
        return out;
    };
    std::vector<int32_t> seqstart_q, seqstart_k, seqstart_q_block, seqstart_k_block, lut_batch_offsets;
    int32_t total_q_tokens = seqlen_q;
    int32_t total_k_tokens = seqlen_k;
    if(is_group_mode)
    {
        seqstart_q     = cumsum(seqlen_qs);
        seqstart_k     = cumsum(seqlen_ks);
        total_q_tokens = seqstart_q.back();
        total_k_tokens = seqstart_k.back();

        std::vector<int32_t> q_blocks(batch), k_blocks(batch);
        for(int b = 0; b < batch; ++b)
        {
            q_blocks[b] = (seqlen_qs[b] + block_size - 1) / block_size;
            k_blocks[b] = (seqlen_ks[b] + block_size - 1) / block_size;
        }
        seqstart_q_block = cumsum(q_blocks);
        seqstart_k_block = cumsum(k_blocks);
        lut_batch_offsets.assign(batch + 1, 0);
        for(int b = 0; b < batch; ++b)
            lut_batch_offsets[b + 1] = lut_batch_offsets[b] + q_blocks[b] * k_blocks[b];
    }

    if(max_seqlen_q == 0) max_seqlen_q = total_q_tokens;
    if(max_seqlen_k == 0) max_seqlen_k = total_k_tokens;
    (void)max_seqlen_k;

    mask_info mask    = mask_info::decode(mask_str, total_q_tokens, total_k_tokens);
    ck_tile::index_t causal_type = get_causal_type(mask.type);
    const float scale_s = (scale_s_user != 0.0f)
                              ? scale_s_user
                              : 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q));
    const std::string dtype_str =
        std::is_same_v<DataType_, ck_tile::half_t> ? "fp16" : "bf16";

    (void)hipGetLastError();
    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(TV.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(Y.get_element_space_size_in_bytes());
    o_buf.SetZero();
    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    v_buf.ToDevice(TV.data());

    ck_tile::DeviceMem seqstart_q_buf(is_group_mode ? seqstart_q.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_buf(is_group_mode ? seqstart_k.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_q_block_buf(is_group_mode ? seqstart_q_block.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_block_buf(is_group_mode ? seqstart_k_block.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem lut_batch_offsets_buf(is_group_mode ? lut_batch_offsets.size() * sizeof(int32_t) : 0);
    if(is_group_mode)
    {
        seqstart_q_buf.ToDevice(seqstart_q.data());
        seqstart_k_buf.ToDevice(seqstart_k.data());
        seqstart_q_block_buf.ToDevice(seqstart_q_block.data());
        seqstart_k_block_buf.ToDevice(seqstart_k_block.data());
        lut_batch_offsets_buf.ToDevice(lut_batch_offsets.data());
    }

    auto st = compute_strides(nhead, nhead_k, total_q_tokens, total_k_tokens,
                              hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor);

    fmha_sparge_fwd_traits fmha_traits;
    fill_common_traits(fmha_traits, hdim_q, hdim_v, dtype_str, is_v_rowmajor, mask, bias, logits_soft_cap);
    fmha_traits.is_group_mode = is_group_mode;

    fmha_sparge_fwd_args args;
    fill_common_args(args, q_buf, k_buf, v_buf, o_buf,
                     batch, nhead, nhead_k, total_q_tokens, total_k_tokens, max_seqlen_q,
                     hdim_q, hdim_v, st, mask, scale_s, bias, logits_soft_cap);
    args.pp_block_size  = block_size;
    args.causal_type    = causal_type;
    args.attention_sink = attention_sink;
    args.hp             = hp;
    args.sparsity_out   = sparsity_out;
    if(is_group_mode)
    {
        args.seqstart_q_ptr        = seqstart_q_buf.GetDeviceBuffer();
        args.seqstart_k_ptr        = seqstart_k_buf.GetDeviceBuffer();
        args.seqstart_q_block_ptr  = seqstart_q_block_buf.GetDeviceBuffer();
        args.seqstart_k_block_ptr  = seqstart_k_block_buf.GetDeviceBuffer();
        args.lut_batch_offset_ptr  = lut_batch_offsets_buf.GetDeviceBuffer();
        args.total_q_blocks        = seqstart_q_block.back();
        args.total_k_blocks        = seqstart_k_block.back();
        args.total_qk_blocks       = lut_batch_offsets.back();
    }

    {
        const size_t workspace_bytes = is_group_mode
            ? compute_sparge_workspace_layout_group(
                  args, args.total_q_blocks, args.total_k_blocks, args.total_qk_blocks).total_bytes
            : compute_sparge_workspace_layout(args).total_bytes;
        ck_tile::DeviceMem workspace(workspace_bytes);
        args.workspace_ptr = workspace.GetDeviceBuffer();

        float ave_time = fmha_sparge_fwd(fmha_traits, args, stream_config);
        if(ave_time < 0)
        {
            std::cerr << tag << " ERROR: dispatch failed (returned " << ave_time
                      << "). mask_type=" << static_cast<int>(fmha_traits.mask_type)
                      << ", data_type=" << fmha_traits.data_type << std::endl;
        }

        HIP_CHECK_ERROR(hipStreamSynchronize(stream_config.stream_id_));
        o_buf.FromDevice(Y.data(), Y.get_element_space_size_in_bytes());
        return ave_time;
    }
}

#define INSTANTIATE_JENGA(T)                                                           \
    template float jenga_sparse_attention<T>(const ck_tile::HostTensor<T>&,            \
                                             const ck_tile::HostTensor<T>&,            \
                                             const ck_tile::HostTensor<T>&,            \
                                             const ck_tile::HostTensor<uint8_t>&,      \
                                             ck_tile::HostTensor<T>&,                  \
                                             int, int, int, int, int, int, int,       \
                                             bool, bool, bool, int, int,              \
                                             const ck_tile::stream_config&,           \
                                             const std::string&,                      \
                                             const sparse_attn_bias_args&,            \
                                             float, float,                            \
                                             const std::vector<int32_t>&,             \
                                             const std::vector<int32_t>&,             \
                                             const std::vector<int32_t>&,             \
                                             const std::vector<int32_t>&)

#define INSTANTIATE_VSA(T)                                                             \
    template float vsa_sparse_attention<T>(const ck_tile::HostTensor<T>&,              \
                                           const ck_tile::HostTensor<T>&,              \
                                           const ck_tile::HostTensor<T>&,              \
                                           const ck_tile::HostTensor<int32_t>&,        \
                                           const ck_tile::HostTensor<int32_t>&,        \
                                           ck_tile::HostTensor<T>&,                    \
                                           int, int, int, int, int, int, int,         \
                                           bool, bool, bool, int, int,                \
                                           const ck_tile::stream_config&,             \
                                           const std::string&,                        \
                                           const sparse_attn_bias_args&,              \
                                           float, float,                              \
                                           const std::vector<int32_t>&,               \
                                           const std::vector<int32_t>&,               \
                                           const std::vector<int32_t>&,               \
                                           const std::vector<int32_t>&)

#define INSTANTIATE_SPARGE(T)                                                          \
    template float sparge_sparse_attention<T>(const ck_tile::HostTensor<T>&,           \
                                              const ck_tile::HostTensor<T>&,           \
                                              const ck_tile::HostTensor<T>&,           \
                                              ck_tile::HostTensor<T>&,                 \
                                              int, int, int, int, int, int, int,      \
                                              bool, bool, bool,                       \
                                              const ck_tile::sparge_hyperparam_args&, \
                                              const std::string&, bool,               \
                                              int, int, int,                          \
                                              float*,                                  \
                                              const ck_tile::stream_config&,          \
                                              const sparse_attn_bias_args&,           \
                                              float, float,                           \
                                              const std::vector<int32_t>&,            \
                                              const std::vector<int32_t>&)

INSTANTIATE_JENGA(ck_tile::half_t);
INSTANTIATE_JENGA(ck_tile::bf16_t);
INSTANTIATE_VSA(ck_tile::half_t);
INSTANTIATE_VSA(ck_tile::bf16_t);
INSTANTIATE_SPARGE(ck_tile::half_t);
INSTANTIATE_SPARGE(ck_tile::bf16_t);

#undef INSTANTIATE_JENGA
#undef INSTANTIATE_VSA
#undef INSTANTIATE_SPARGE
