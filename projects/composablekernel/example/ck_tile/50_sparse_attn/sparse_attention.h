// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
// Returns GPU kernel time in ms (negative on error).
// mask_str: "0" / "1"|"t" / "2"|"b" / "t:l,r" / "b:l,r".
// Group mode (varlen) is signalled by passing non-empty seqstart vectors,
// mirroring 01_fmha's fmha_fwd() dispatch convention.

#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "ck_tile/core.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/sparse_attn/sparge_hyperparam.hpp"

struct sparse_attn_bias_args
{
    int type      = 0;   // 0=no_bias, 1=elementwise, 2=alibi (sync with bias_enum)
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
                             // Group-mode (empty → batch mode):
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
    // Group-mode (empty → batch mode):
    const std::vector<int32_t>& seqstart_q_host       = {},
    const std::vector<int32_t>& seqstart_k_host       = {},
    const std::vector<int32_t>& seqstart_q_block_host = {},
    const std::vector<int32_t>& lut_batch_offsets     = {});

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
                                    const ck_tile::sparge_hyperparam_args& hp = {},
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
                                    // Group-mode (empty → batch mode):
                                    const std::vector<int32_t>& seqlen_qs = {},
                                    const std::vector<int32_t>& seqlen_ks = {});
