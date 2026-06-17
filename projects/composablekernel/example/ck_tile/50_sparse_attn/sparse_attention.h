// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/stream_config.hpp"

struct sparge_hyperparam_args
{
    float cdfthreshd   = 1.0f; 
    float topk         = 0.0f;
    float simthreshold = 0.0f;
    float pvthreshd    = 0.0f;

    const void* cdfthreshd_per_head_ptr   = nullptr;
    const void* topk_per_head_ptr         = nullptr;
    const void* simthreshold_per_head_ptr = nullptr;
    const void* pvthreshd_per_head_ptr    = nullptr;

    bool smooth_k = true;
};

struct sparse_attn_bias_args
{
    int type      = 0;   // 0=no_bias, 1=elementwise, 2=alibi
    int rank      = 0;
    const void*  ptr               = nullptr;
    std::int32_t stride_bias       = 0;
    std::int32_t nhead_stride_bias = 0;
    std::int32_t batch_stride_bias = 0;
};

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
                             const std::string& mask_str = "0",
                             const sparse_attn_bias_args& bias = {},
                             float scale_s = 0.0f,
                             float logits_soft_cap = 0.0f,
                             const std::vector<int32_t>& seqstart_q_host       = {},
                             const std::vector<int32_t>& seqstart_k_host       = {},
                             const std::vector<int32_t>& seqstart_q_block_host = {},
                             const std::vector<int32_t>& mask_batch_offsets    = {});

template <typename DataType_>
float vsa_sparse_attention(
    const ck_tile::HostTensor<DataType_>& TQ,
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
    const std::string& mask_str = "0",
    const sparse_attn_bias_args& bias = {},
    float scale_s = 0.0f,
    float logits_soft_cap = 0.0f,
    const std::vector<int32_t>& seqstart_q_host       = {},
    const std::vector<int32_t>& seqstart_k_host       = {},
    const std::vector<int32_t>& seqstart_q_block_host = {},
    const std::vector<int32_t>& mask_batch_offsets     = {});

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
                                    bool is_v_rowmajor      = true,
                                    const sparge_hyperparam_args& hp = {},
                                    const std::string& mask_str = "0",
                                    bool attention_sink     = false,
                                    int block_size          = 128,
                                    int max_seqlen_q        = 0,
                                    int max_seqlen_k        = 0,
                                    float* sparsity_out     = nullptr,
                                    const ck_tile::stream_config& stream_config = {},
                                    const sparse_attn_bias_args& bias = {},
                                    float scale_s           = 0.0f,
                                    float logits_soft_cap   = 0.0f,
                                    // group / varlen: caller-precomputed prefix-sum + block tables
                                    // (non-empty seqstart_q_host -> group mode)
                                    const std::vector<int32_t>& seqstart_q_host       = {},
                                    const std::vector<int32_t>& seqstart_k_host       = {},
                                    const std::vector<int32_t>& seqstart_q_block_host = {},
                                    const std::vector<int32_t>& seqstart_k_block_host = {},
                                    const std::vector<int32_t>& mask_batch_offsets    = {},
                                    // Optional: download the device's selected-block LUT + valid-block
                                    // counts so validation can reference the kernel's actual selection.
                                    std::vector<int32_t>* out_lut = nullptr,
                                    std::vector<int32_t>* out_vbn = nullptr);

// sparge_sage (quantized sparge): bf16 Q/K (device-quantized in the fused preprocess),
// caller-provided FP8 V + per-channel v_descale. Returns GPU time in ms (negative on error).
template <typename DataType_>
float sparge_sage_sparse_attention(const ck_tile::HostTensor<DataType_>& TQ,    // bf16 [B,H,S,D]
                                   const ck_tile::HostTensor<DataType_>& TK,    // bf16 [B,Hk,Sk,D]
                                   const ck_tile::HostTensor<ck_tile::fp8_t>& TVfp8,  // [B,Hk,Sk,Dv]
                                   const ck_tile::HostTensor<float>& TVdescale, // [B,Hk,Dv]
                                   ck_tile::HostTensor<DataType_>& Y,           // bf16 [B,H,S,Dv]
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
                                   const std::string& mask_str = "0",
                                   // Bias: 0=no_bias, 1=elementwise, 2=alibi (slopes or dense
                                   // [.., Sq, Sk] in bias.ptr; strides as for sage attn).
                                   const sparse_attn_bias_args& bias = {},
                                   float scale_s = 0.0f,
                                   float logits_soft_cap = 0.0f,
                                   const sparge_hyperparam_args& hp = {},
                                   int block_size = 128,
                                   bool attention_sink = false,
                                   const std::string& qscale = "perwarp",
                                   const std::string& data_type = "i8fp8bf16",
                                   // Group / varlen mode: caller-precomputed prefix-sum + block
                                   // tables (non-empty seqstart_q_host -> packed Q/K/V/O).
                                   const std::vector<int32_t>& seqstart_q_host       = {},
                                   const std::vector<int32_t>& seqstart_k_host       = {},
                                   const std::vector<int32_t>& seqstart_q_block_host = {},
                                   const std::vector<int32_t>& seqstart_k_block_host = {},
                                   const std::vector<int32_t>& mask_batch_offsets    = {},
                                   // Optional: download the device's selected-block LUT + valid-block
                                   // counts so validation can reference the kernel's actual selection.
                                   std::vector<int32_t>* out_lut = nullptr,
                                   std::vector<int32_t>* out_vbn = nullptr);
