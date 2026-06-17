// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "sparse_attention.h"
#include "sparse_attn_fwd.hpp"
#include "ck_tile/core.hpp"
#include "ck_tile/ops/sageattention/pipeline/tile_sageattn_traits.hpp"
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
    case mask_enum::no_mask:
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
    const float scale_s = (scale_s_user != 0.0f)
                              ? scale_s_user
                              : 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q));
    mask_info mask = mask_info::decode(mask_str, total_q, total_k);
    auto st = compute_strides(nhead, nhead_k, total_q, total_k,
                              hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor);

    (void)hipGetLastError();
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
                           const std::vector<int32_t>& mask_batch_offsets)
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

    (void)hipGetLastError();
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
    ck_tile::DeviceMem mask_batch_offsets_buf(is_group_mode ? mask_batch_offsets.size() * sizeof(int32_t) : 0);
    if(is_group_mode)
    {
        seqstart_q_buf.ToDevice(seqstart_q_host.data());
        seqstart_k_buf.ToDevice(seqstart_k_host.data());
        seqstart_q_block_buf.ToDevice(seqstart_q_block_host.data());
        mask_batch_offsets_buf.ToDevice(mask_batch_offsets.data());
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
        args.mask_batch_offset_ptr = mask_batch_offsets_buf.GetDeviceBuffer();
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
                                    const sparge_hyperparam_args& hp,
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
                                    const std::vector<int32_t>& seqstart_q_host,
                                    const std::vector<int32_t>& seqstart_k_host,
                                    const std::vector<int32_t>& seqstart_q_block_host,
                                    const std::vector<int32_t>& seqstart_k_block_host,
                                    const std::vector<int32_t>& mask_batch_offsets,
                                    std::vector<int32_t>* out_lut,
                                    std::vector<int32_t>* out_vbn)
{
    static_assert(std::is_same_v<DataType_, ck_tile::half_t> ||
                      std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "SpargeAttention supports fp16/bf16 only.");

    const bool is_group_mode = !seqstart_q_host.empty();
    const char* tag = is_group_mode ? "[sparge group]" : "[sparge]";

    if(hdim_q != 128 || hdim_v != 128 || block_size != 128)
    {
        std::cerr << tag << " only hdim=block_size=128 supported.\n";
        return -1.0f;
    }

    assert(nhead % nhead_k == 0);

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
    // Mask-prediction sort-buffer cap: codegen emits kMaxKBlocksPow2 variants up to 1024 blocks;
    // beyond that no variant exists, so reject here.
    {
        constexpr int kMaxKBlocks = 1024;
        if(is_group_mode)
        {
            for(int b = 0; b < batch; ++b)
            {
                const int seqlen_k_b = seqstart_k_host[b + 1] - seqstart_k_host[b];
                const int num_k_blocks_b = (seqlen_k_b + block_size - 1) / block_size;
                if(num_k_blocks_b > kMaxKBlocks)
                {
                    std::cerr << tag << " error: seqlen_k[" << b << "]=" << seqlen_k_b
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

    int32_t total_q_tokens = is_group_mode ? seqstart_q_host.back() : seqlen_q;
    int32_t total_k_tokens = is_group_mode ? seqstart_k_host.back() : seqlen_k;

    if(max_seqlen_q == 0) max_seqlen_q = total_q_tokens;
    if(max_seqlen_k == 0) max_seqlen_k = total_k_tokens;
    (void)max_seqlen_k;

    mask_info mask    = mask_info::decode(mask_str, total_q_tokens, total_k_tokens);
    ck_tile::index_t causal_type = get_causal_type(mask.type);
    const float scale_s = (scale_s_user != 0.0f)
                              ? scale_s_user
                              : 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q));
    const std::string data_type =
        std::is_same_v<DataType_, ck_tile::bf16_t> ? "bf16" : "fp16";

    (void)hipGetLastError();
    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(TV.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(Y.get_element_space_size_in_bytes());
    o_buf.SetZero();
    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    v_buf.ToDevice(TV.data());

    ck_tile::DeviceMem seqstart_q_buf(is_group_mode ? seqstart_q_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_buf(is_group_mode ? seqstart_k_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_q_block_buf(is_group_mode ? seqstart_q_block_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_block_buf(is_group_mode ? seqstart_k_block_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem mask_batch_offsets_buf(is_group_mode ? mask_batch_offsets.size() * sizeof(int32_t) : 0);
    if(is_group_mode)
    {
        seqstart_q_buf.ToDevice(seqstart_q_host.data());
        seqstart_k_buf.ToDevice(seqstart_k_host.data());
        seqstart_q_block_buf.ToDevice(seqstart_q_block_host.data());
        seqstart_k_block_buf.ToDevice(seqstart_k_block_host.data());
        mask_batch_offsets_buf.ToDevice(mask_batch_offsets.data());
    }

    auto st = compute_strides(nhead, nhead_k, total_q_tokens, total_k_tokens,
                              hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor);

    fmha_sparge_fwd_traits fmha_traits;
    fill_common_traits(fmha_traits, hdim_q, hdim_v, data_type, is_v_rowmajor, mask, bias, logits_soft_cap);
    fmha_traits.is_group_mode = is_group_mode;

    fmha_sparge_fwd_args args;
    // Group-mode seqlen convention (sparge): args.seqlen_q/k carry the PACKED TOTALS
    // (total_q_tokens/total_k_tokens), since the sparge codegen indexes the variable-length
    // group layout directly off seqlen_q/k and has no separate total_*_tokens field. This
    // DIFFERS from sparge_sage below, which puts MAX seqlen in seqlen_q/k and the totals in
    // args.total_q_tokens/total_k_tokens. Each entry is internally consistent with its own
    // codegen; the same-named field means different things, so do NOT unify one without the
    // other -- they must stay matched with their respective codegen.
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
        args.mask_batch_offset_ptr  = mask_batch_offsets_buf.GetDeviceBuffer();
        args.total_q_blocks        = seqstart_q_block_host.back();
        args.total_k_blocks        = seqstart_k_block_host.back();
        args.total_qk_blocks       = mask_batch_offsets.back();
    }

    {
        const auto ws = is_group_mode
            ? compute_sparge_workspace_layout_group(
                  args, args.total_q_blocks, args.total_k_blocks, args.total_qk_blocks)
            : compute_sparge_workspace_layout(args);
        ck_tile::DeviceMem workspace(ws.total_bytes);
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

        // Optionally hand back the device's LUT + valid-block counts (lut / vbn regions)
        // so validation can reference the kernel's actual selection.
        if(out_lut != nullptr)
        {
            out_lut->resize(ws.lut_bytes / sizeof(int32_t));
            HIP_CHECK_ERROR(hipMemcpy(out_lut->data(),
                                      static_cast<const char*>(workspace.GetDeviceBuffer()) +
                                          ws.lut_off,
                                      ws.lut_bytes, hipMemcpyDeviceToHost));
        }
        if(out_vbn != nullptr)
        {
            out_vbn->resize(ws.vbn_bytes / sizeof(int32_t));
            HIP_CHECK_ERROR(hipMemcpy(out_vbn->data(),
                                      static_cast<const char*>(workspace.GetDeviceBuffer()) +
                                          ws.vbn_off,
                                      ws.vbn_bytes, hipMemcpyDeviceToHost));
        }
        return ave_time;
    }
}

template <typename DataType_>
float sparge_sage_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,
                                   const ck_tile::HostTensor<DataType_>& TK,
                                   const ck_tile::HostTensor<ck_tile::fp8_t>& TVfp8,
                                   const ck_tile::HostTensor<float>& TVdescale,
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
                                   const sparge_hyperparam_args& hp,
                                   int block_size,
                                   bool attention_sink,
                                   const std::string& qscale,
                                   const std::string& data_type,
                                   const std::vector<int32_t>& seqstart_q_host,
                                   const std::vector<int32_t>& seqstart_k_host,
                                   const std::vector<int32_t>& seqstart_q_block_host,
                                   const std::vector<int32_t>& seqstart_k_block_host,
                                   const std::vector<int32_t>& mask_batch_offsets,
                                   std::vector<int32_t>* out_lut,
                                   std::vector<int32_t>* out_vbn)
{
    static_assert(std::is_same_v<DataType_, ck_tile::bf16_t>,
                  "sparge_sage stage 3 supports bf16 input only");
    const bool is_group_mode = !seqstart_q_host.empty();
    const char* tag = is_group_mode ? "[sparge_sage group]" : "[sparge_sage]";

    if(hdim_q != 128 || hdim_v != 128 || block_size != 128)
    {
        std::cerr << tag << " only hdim=block_size=128 supported.\n";
        return -1.0f;
    }

    assert(nhead % nhead_k == 0);

    if(bias.type < 0 || bias.type > 2)
    {
        std::cerr << tag << " invalid bias.type=" << bias.type
                  << " (expected 0=no, 1=elementwise, 2=alibi).\n";
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

    int32_t total_q_tokens = is_group_mode ? seqstart_q_host.back() : seqlen_q;
    int32_t total_k_tokens = is_group_mode ? seqstart_k_host.back() : seqlen_k;

    if(max_seqlen_q == 0) max_seqlen_q = total_q_tokens;
    if(max_seqlen_k == 0) max_seqlen_k = total_k_tokens;
    (void)max_seqlen_k;

    mask_info mask = mask_info::decode(mask_str, total_q_tokens, total_k_tokens);
    ck_tile::index_t causal_type = get_causal_type(mask.type);
    const float scale_s = (scale_s_user != 0.0f)
                              ? scale_s_user
                              : 1.0f / ck_tile::sqrt(static_cast<float>(hdim_q));

    (void)hipGetLastError();
    ck_tile::DeviceMem q_buf(TQ.get_element_space_size_in_bytes());
    ck_tile::DeviceMem k_buf(TK.get_element_space_size_in_bytes());
    ck_tile::DeviceMem v_buf(TVfp8.get_element_space_size_in_bytes());
    ck_tile::DeviceMem o_buf(Y.get_element_space_size_in_bytes());
    ck_tile::DeviceMem vdescale_buf(TVdescale.get_element_space_size_in_bytes());
    o_buf.SetZero();
    q_buf.ToDevice(TQ.data());
    k_buf.ToDevice(TK.data());
    v_buf.ToDevice(TVfp8.data());
    vdescale_buf.ToDevice(TVdescale.data());

    ck_tile::DeviceMem seqstart_q_buf(is_group_mode ? seqstart_q_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_buf(is_group_mode ? seqstart_k_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_q_block_buf(
        is_group_mode ? seqstart_q_block_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem seqstart_k_block_buf(
        is_group_mode ? seqstart_k_block_host.size() * sizeof(int32_t) : 0);
    ck_tile::DeviceMem mask_batch_offsets_buf(
        is_group_mode ? mask_batch_offsets.size() * sizeof(int32_t) : 0);
    if(is_group_mode)
    {
        seqstart_q_buf.ToDevice(seqstart_q_host.data());
        seqstart_k_buf.ToDevice(seqstart_k_host.data());
        seqstart_q_block_buf.ToDevice(seqstart_q_block_host.data());
        seqstart_k_block_buf.ToDevice(seqstart_k_block_host.data());
        mask_batch_offsets_buf.ToDevice(mask_batch_offsets.data());
    }

    // Strides over the packed token totals (group) or per-seq dims (batch).
    auto st = compute_strides(nhead, nhead_k, total_q_tokens, total_k_tokens,
                              hdim_q, hdim_v, i_perm, o_perm, is_v_rowmajor);

    fmha_sparge_sage_fwd_args args;
    // Group-mode seqlen convention (sparge_sage): args.seqlen_q/k carry the MAX seqlen
    // (sizes mask smem + grid m-tiles); per-batch lengths live in seqstart, and the PACKED
    // TOTALS are stored separately in args.total_q_tokens/total_k_tokens (set below). This
    // DIFFERS from plain sparge above, which packs the totals directly into seqlen_q/k and
    // has no total_*_tokens field. The same-named seqlen_q/k field thus means different
    // things between the two entries; keep each matched with its respective codegen and do
    // NOT unify one side in isolation.
    fill_common_args(args, q_buf, k_buf, v_buf, o_buf,
                     batch, nhead, nhead_k, seqlen_q, seqlen_k, seqlen_q,
                     hdim_q, hdim_v, st, mask, scale_s, bias, logits_soft_cap);
    args.v_descale_ptr  = vdescale_buf.GetDeviceBuffer();
    args.pp_block_size  = block_size;
    args.causal_type       = causal_type;
    // qscale -> per-token-group quant granularity. Pin the host literals below to
    // TileSageAttnTraits so a trait change can't silently desync the descale stride.
    {
        using QSE = ck_tile::BlockSageAttentionQuantScaleEnum;
        static_assert(
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERWARP>::kBlockScaleSizeQ == 32 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERWARP>::kBlockScaleSizeK == 64 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTHREAD>::kBlockScaleSizeQ == 4 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTHREAD>::kBlockScaleSizeK == 16 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::BLOCKSCALE>::kBlockScaleSizeQ == 128 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::BLOCKSCALE>::kBlockScaleSizeK == 128 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTENSOR>::kBlockScaleSizeQ == 128 &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTENSOR>::kBlockScaleSizeK == 128,
            "host block_scale_size literals must match TileSageAttnTraits::kBlockScaleSizeQ/K");
        // Guard the group workspace's scales-per-block (spb = pp_block_size / block_scale_size,
        // see compute_sparge_sage_workspace_layout_group): block_scale_size must not exceed the
        // (only supported) pp_block_size=128, else integer division collapses spb to 0 and the
        // q/k_scale region is sized to zero. All literals above are <=128; assert it holds.
        constexpr int kPpBlockSize = 128; // enforced by the hdim/block_size==128 check above
        static_assert(
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERWARP>::kBlockScaleSizeQ <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERWARP>::kBlockScaleSizeK <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTHREAD>::kBlockScaleSizeQ <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTHREAD>::kBlockScaleSizeK <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::BLOCKSCALE>::kBlockScaleSizeQ <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::BLOCKSCALE>::kBlockScaleSizeK <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTENSOR>::kBlockScaleSizeQ <= kPpBlockSize &&
            ck_tile::TileSageAttnTraits<false, false, false, false, QSE::PERTENSOR>::kBlockScaleSizeK <= kPpBlockSize,
            "block_scale_size must be <= pp_block_size (=128) or workspace scales-per-block (spb) collapses to 0");
    }
    if(qscale == "perblock" || qscale == "pertensor")
    {
        // PERTENSOR uses one global per-(b,h) scale; 128 keeps the workspace sizing
        // (ceil(S/128) scales) >= the single scale the global-absmax quant writes.
        args.block_scale_size_q = 128;
        args.block_scale_size_k = 128;
    }
    else if(qscale == "perthread")
    {
        args.block_scale_size_q = 4;
        args.block_scale_size_k = 16;
    }
    else // perwarp (default)
    {
        args.block_scale_size_q = 32;
        args.block_scale_size_k = 64;
    }
    args.nhead_stride_v_descale = hdim_v;
    args.batch_stride_v_descale = nhead_k * hdim_v;
    args.hp = hp;
    args.attention_sink = attention_sink;

    if(is_group_mode)
    {
        args.seqstart_q_ptr       = seqstart_q_buf.GetDeviceBuffer();
        args.seqstart_k_ptr       = seqstart_k_buf.GetDeviceBuffer();
        args.seqstart_q_block_ptr = seqstart_q_block_buf.GetDeviceBuffer();
        args.seqstart_k_block_ptr = seqstart_k_block_buf.GetDeviceBuffer();
        args.mask_batch_offset_ptr = mask_batch_offsets_buf.GetDeviceBuffer();
        args.total_q_blocks       = seqstart_q_block_host.back();
        args.total_k_blocks       = seqstart_k_block_host.back();
        args.total_qk_blocks      = mask_batch_offsets.back();
        // Packed token totals kept separate from seqlen_q/k (which hold MAX, see above): the
        // sparge_sage codegen reads variable-length group data off total_*_tokens, unlike plain
        // sparge which overloads seqlen_q/k for the totals. Must stay matched with the codegen.
        args.total_q_tokens       = total_q_tokens;
        args.total_k_tokens       = total_k_tokens;
    }

    // smooth_k: host-compute per-channel global K-mean km = k.mean(dim=-2) per (b, head_k, channel),
    // fed to BOTH device and reference K quant. Q is never centered; fp32 accumulation in the same
    // element order as the host reference's compute_global_k_mean. Disabled -> km_ptr nullptr.
    ck_tile::DeviceMem km_buf;
    if(hp.smooth_k)
    {
        std::vector<float> km_host(static_cast<size_t>(batch) * nhead_k * hdim_q, 0.0f);
        const auto* k_data = TK.data();
        for(int b = 0; b < batch; ++b)
        {
            const int sk = is_group_mode ? (seqstart_k_host[b + 1] - seqstart_k_host[b]) : seqlen_k;
            const int64_t b_tok_off =
                is_group_mode ? static_cast<int64_t>(seqstart_k_host[b]) : 0;
            const int64_t b_base =
                is_group_mode ? (b_tok_off * st.stride_k)
                              : (static_cast<int64_t>(b) * st.batch_stride_k);
            for(int h = 0; h < nhead_k; ++h)
                for(int d = 0; d < hdim_q; ++d)
                {
                    float sum = 0.0f;
                    for(int s = 0; s < sk; ++s)
                    {
                        const int64_t off = b_base +
                            static_cast<int64_t>(h) * st.nhead_stride_k +
                            static_cast<int64_t>(s) * st.stride_k + d;
                        sum += ck_tile::type_convert<float>(k_data[off]);
                    }
                    km_host[(static_cast<size_t>(b) * nhead_k + h) * hdim_q + d] =
                        sum / static_cast<float>(sk);
                }
        }
        km_buf.Realloc(km_host.size() * sizeof(float));
        km_buf.ToDevice(km_host.data());
        args.km_ptr = km_buf.GetDeviceBuffer();
    }
    else
    {
        args.km_ptr = nullptr;
    }

    const auto ws = is_group_mode ? compute_sparge_sage_workspace_layout_group(args)
                                  : compute_sparge_sage_workspace_layout(args);
    ck_tile::DeviceMem workspace(ws.total_bytes);
    workspace.SetZero();
    args.workspace_ptr = workspace.GetDeviceBuffer();

    // sparge_sage's traits struct differs from the shared fmha_jenga_fwd_traits used by
    // fill_common_traits (it carries qscale/has_mask, not is_v_rowmajor/mask_type), so it
    // is populated explicitly here.
    fmha_sparge_sage_fwd_traits traits;
    traits.hdim_q        = hdim_q;
    traits.hdim_v        = hdim_v;
    traits.data_type     = data_type; // "i8fp8bf16" (INT8 Q/K) or "fp8bf16" (FP8 Q/K)
    traits.qscale        = qscale;
    traits.is_group_mode = is_group_mode;
    traits.has_mask      = (causal_type != 0);
    traits.bias_type     = bias.type;

    float ave_time = fmha_sparge_sage_fwd(traits, args, stream_config);
    if(ave_time < 0)
    {
        std::cerr << tag << " ERROR: dispatch failed (returned " << ave_time
                  << "). mask_type=" << static_cast<int>(traits.has_mask)
                  << ", data_type=" << traits.data_type << std::endl;
    }

    HIP_CHECK_ERROR(hipStreamSynchronize(stream_config.stream_id_));
    o_buf.FromDevice(Y.data(), Y.get_element_space_size_in_bytes());

    // Optionally hand back the device's LUT + valid-block counts (base.lut / base.vbn regions)
    // so validation can reference the kernel's actual selection.
    if(out_lut != nullptr)
    {
        out_lut->resize(ws.base.lut_bytes / sizeof(int32_t));
        HIP_CHECK_ERROR(hipMemcpy(out_lut->data(),
                                  static_cast<const char*>(workspace.GetDeviceBuffer()) +
                                      ws.base.lut_off,
                                  ws.base.lut_bytes, hipMemcpyDeviceToHost));
    }
    if(out_vbn != nullptr)
    {
        out_vbn->resize(ws.base.vbn_bytes / sizeof(int32_t));
        HIP_CHECK_ERROR(hipMemcpy(out_vbn->data(),
                                  static_cast<const char*>(workspace.GetDeviceBuffer()) +
                                      ws.base.vbn_off,
                                  ws.base.vbn_bytes, hipMemcpyDeviceToHost));
    }
    return ave_time;
}

#define INSTANTIATE_SPARGE_SAGE(T)                                                     \
    template float sparge_sage_sparse_attention<T>(                                    \
        const ck_tile::HostTensor<T>&, const ck_tile::HostTensor<T>&,                  \
        const ck_tile::HostTensor<ck_tile::fp8_t>&, const ck_tile::HostTensor<float>&, \
        ck_tile::HostTensor<T>&, int, int, int, int, int, int, int, bool, bool, bool,  \
        int, int, const ck_tile::stream_config&, const std::string&,                   \
        const sparse_attn_bias_args&, float, float,                                    \
        const sparge_hyperparam_args&, int, bool,                                      \
        const std::string&, const std::string&,                                        \
        const std::vector<int32_t>&, const std::vector<int32_t>&,                      \
        const std::vector<int32_t>&, const std::vector<int32_t>&,                      \
        const std::vector<int32_t>&,                                                   \
        std::vector<int32_t>*, std::vector<int32_t>*)

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
                                              const sparge_hyperparam_args&, \
                                              const std::string&, bool,               \
                                              int, int, int,                          \
                                              float*,                                  \
                                              const ck_tile::stream_config&,          \
                                              const sparse_attn_bias_args&,           \
                                              float, float,                           \
                                              const std::vector<int32_t>&,            \
                                              const std::vector<int32_t>&,            \
                                              const std::vector<int32_t>&,            \
                                              const std::vector<int32_t>&,            \
                                              const std::vector<int32_t>&,            \
                                              std::vector<int32_t>*, std::vector<int32_t>*)

INSTANTIATE_JENGA(ck_tile::half_t);
INSTANTIATE_JENGA(ck_tile::bf16_t);
INSTANTIATE_VSA(ck_tile::half_t);
INSTANTIATE_VSA(ck_tile::bf16_t);
INSTANTIATE_SPARGE(ck_tile::half_t);
INSTANTIATE_SPARGE(ck_tile::bf16_t);
INSTANTIATE_SPARGE_SAGE(ck_tile::bf16_t);

#undef INSTANTIATE_JENGA
#undef INSTANTIATE_VSA
#undef INSTANTIATE_SPARGE
#undef INSTANTIATE_SPARGE_SAGE
