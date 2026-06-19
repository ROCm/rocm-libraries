// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace hipdnn_data_sdk::detail::autotune_config
{

namespace op
{
inline constexpr const char* CONV_FPROP = "conv_fprop";
inline constexpr const char* CONV_DGRAD = "conv_dgrad";
inline constexpr const char* CONV_WGRAD = "conv_wgrad";
inline constexpr const char* SDPA_FWD = "sdpa_fwd";
inline constexpr const char* SDPA_BWD = "sdpa_bwd";
inline constexpr const char* MATMUL = "matmul";
inline constexpr const char* BATCHNORM_TRAINING = "batchnorm_training";
inline constexpr const char* BATCHNORM_INFERENCE = "batchnorm_inference";
inline constexpr const char* BATCHNORM_INFERENCE_VARIANCE_EXT = "batchnorm_inference_variance_ext";
inline constexpr const char* BATCHNORM_BACKWARD = "batchnorm_backward";
inline constexpr const char* LAYERNORM = "layernorm";
inline constexpr const char* RMSNORM = "rmsnorm";
inline constexpr const char* RMSNORM_BACKWARD = "rmsnorm_backward";
inline constexpr const char* REDUCTION = "reduction";
inline constexpr const char* RESAMPLE_FWD = "resample_fwd";
inline constexpr const char* POINTWISE = "pointwise";
} // namespace op

namespace json
{
inline constexpr const char* VERSION = "version";
inline constexpr const char* ENGINE_OVERRIDES = "engine_overrides";
inline constexpr const char* OP = "op";
inline constexpr const char* CRITERIA = "criteria";
inline constexpr const char* ENGINE_NAME = "engine_name";
inline constexpr const char* TENSORS = "tensors";
inline constexpr const char* TENSOR_ID = "tensor_id";
inline constexpr const char* DIM = "dim";
inline constexpr const char* STRIDE = "stride";
} // namespace json

namespace version
{
inline constexpr int64_t DEFAULT = 1;
inline constexpr int64_t NAMED_TENSOR_IDS = 2;
inline constexpr int64_t CURRENT = NAMED_TENSOR_IDS;
} // namespace version

namespace criterion
{
inline constexpr const char* NORM_FWD_PHASE = "norm_fwd_phase";
inline constexpr const char* REDUCTION_MODE = "reduction_mode";
inline constexpr const char* RESAMPLE_MODE = "resample_mode";
inline constexpr const char* PADDING_MODE = "padding_mode";
inline constexpr const char* POINTWISE_MODE = "pointwise_mode";
} // namespace criterion

namespace tensor
{
inline constexpr const char* X = "x_tensor_uid";
inline constexpr const char* W = "w_tensor_uid";
inline constexpr const char* DY = "dy_tensor_uid";
inline constexpr const char* Q = "q_tensor_uid";
inline constexpr const char* K = "k_tensor_uid";
inline constexpr const char* V = "v_tensor_uid";
inline constexpr const char* SCALE = "scale_tensor_uid";
inline constexpr const char* ATTN_MASK = "attn_mask_tensor_uid";
inline constexpr const char* SEQ_LEN_Q = "seq_len_q_tensor_uid";
inline constexpr const char* SEQ_LEN_KV = "seq_len_kv_tensor_uid";
inline constexpr const char* SEED = "seed_tensor_uid";
inline constexpr const char* OFFSET = "offset_tensor_uid";
inline constexpr const char* DROPOUT_MASK = "dropout_mask_tensor_uid";
inline constexpr const char* DROPOUT_SCALE = "dropout_scale_tensor_uid";
inline constexpr const char* PAGE_TABLE_K = "page_table_k_tensor_uid";
inline constexpr const char* PAGE_TABLE_V = "page_table_v_tensor_uid";
inline constexpr const char* BLOCK_MASK = "block_mask_tensor_uid";
inline constexpr const char* SINK_TOKEN = "sink_token_tensor_uid";
inline constexpr const char* DESCALE_Q = "descale_q_tensor_uid";
inline constexpr const char* DESCALE_K = "descale_k_tensor_uid";
inline constexpr const char* DESCALE_V = "descale_v_tensor_uid";
inline constexpr const char* DESCALE_S = "descale_s_tensor_uid";
inline constexpr const char* SCALE_S = "scale_s_tensor_uid";
inline constexpr const char* SCALE_O = "scale_o_tensor_uid";
inline constexpr const char* O = "o_tensor_uid";
inline constexpr const char* DO = "do_tensor_uid";
inline constexpr const char* STATS = "stats_tensor_uid";
inline constexpr const char* DROPOUT_SCALE_INV = "dropout_scale_inv_tensor_uid";
inline constexpr const char* A = "a_tensor_uid";
inline constexpr const char* B = "b_tensor_uid";
inline constexpr const char* BIAS = "bias_tensor_uid";
inline constexpr const char* EPSILON = "epsilon_tensor_uid";
inline constexpr const char* PREV_RUNNING_MEAN = "prev_running_mean_tensor_uid";
inline constexpr const char* PREV_RUNNING_VARIANCE = "prev_running_variance_tensor_uid";
inline constexpr const char* MOMENTUM = "momentum_tensor_uid";
inline constexpr const char* MEAN = "mean_tensor_uid";
inline constexpr const char* INV_VARIANCE = "inv_variance_tensor_uid";
inline constexpr const char* VARIANCE = "variance_tensor_uid";
inline constexpr const char* INV_RMS = "inv_rms_tensor_uid";
inline constexpr const char* INPUT = "in_tensor_uid";
inline constexpr const char* IN_0 = "in_0_tensor_uid";
inline constexpr const char* IN_1 = "in_1_tensor_uid";
inline constexpr const char* IN_2 = "in_2_tensor_uid";
} // namespace tensor

} // namespace hipdnn_data_sdk::detail::autotune_config
