// Copyright 2026 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// Custom op definitions for hipDNN operations that map to fusilli CustomOp
// nodes. Each namespace provides:
//
//   validateTemplate() — Check if the hipDNN attributes use only features
//                        supported by the MLIR template. Returns NotImplemented
//                        for unsupported configurations.
//
//   validateInputs()   — Check tensor dimension constraints (rank, matching
//                        dims).
//
//   buildMLIR()        — Construct the MLIR template string for the CustomOp.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_CUSTOM_OPS_H
#define FUSILLI_PLUGIN_CUSTOM_OPS_H

#include <fusilli.h>
#include <hipdnn_data_sdk/data_objects/sdpa_attributes_generated.h>

#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

// SDPA MLIR templates for torch.aten.scaled_dot_product_attention.
//
// Templates are stored as R-string literals so the MLIR structure is
// directly readable in source. Standard CustomOp placeholders
// ({FUNC_NAME}, {IN<i>_DTYPE}, {OUT0_DTYPE}) are resolved by
// CustomOpNode::resolveMlirPlaceholders(). Scalar placeholders
// ({DROPOUT_P}, {IS_CAUSAL}, {SCALE_CONST}, {SCALE_TYPE}, {ENABLE_GQA})
// are resolved by SdpaImport::buildMLIR().
//
// Tensor rank is hardcoded to 4 ([batch, heads, seq_len, head_dim]).

// SDPA template: 3 tensor inputs (Q, K, V), attention mask is none.
// Positional args: {0}=DROPOUT_P, {1}=IS_CAUSAL, {2}=SCALE_CONST,
//                  {3}=SCALE_TYPE, {4}=ENABLE_GQA
// clang-format off
static constexpr std::string_view kSdpaNoMask = R"mlir(
  func.func private @{{FUNC_NAME}}(
      %arg0: !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>,
      %arg1: !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
      %arg2: !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>)
      -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}> {{
    %none_mask = torch.constant.none
    %dropout = torch.constant.float {0}
    %is_causal = torch.constant.bool {1}
    %scale = {2}
    %enable_gqa = torch.constant.bool {4}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %none_mask, %dropout, %is_causal, %scale, %enable_gqa :
        !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>, !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
        !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>, !torch.none, !torch.float, !torch.bool,
        {3}, !torch.bool -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
    return %0 : !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
  }}
)mlir";

// SDPA template: 4 tensor inputs (Q, K, V, attn_mask).
// Positional args: same as kSdpaNoMask.
static constexpr std::string_view kSdpaWithMask = R"mlir(
  func.func private @{{FUNC_NAME}}(
      %arg0: !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>,
      %arg1: !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
      %arg2: !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>,
      %arg3: !torch.vtensor<[?,?,?,?],{{IN3_DTYPE}}>)
      -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}> {{
    %dropout = torch.constant.float {0}
    %is_causal = torch.constant.bool {1}
    %scale = {2}
    %enable_gqa = torch.constant.bool {4}
    %0 = torch.aten.scaled_dot_product_attention %arg0, %arg1, %arg2,
        %arg3, %dropout, %is_causal, %scale, %enable_gqa :
        !torch.vtensor<[?,?,?,?],{{IN0_DTYPE}}>, !torch.vtensor<[?,?,?,?],{{IN1_DTYPE}}>,
        !torch.vtensor<[?,?,?,?],{{IN2_DTYPE}}>, !torch.vtensor<[?,?,?,?],{{IN3_DTYPE}}>,
        !torch.float, !torch.bool,
        {3}, !torch.bool -> !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
    return %0 : !torch.vtensor<[?,?,?,?],{{OUT0_DTYPE}}>
  }}
)mlir";
// clang-format on

namespace SdpaImport {
// Check if the SDPA attributes use only features supported by the MLIR
// template. Returns NotImplemented for unsupported configurations.
inline fusilli::ErrorObject
validateTemplate(const hipdnn_data_sdk::data_objects::SdpaAttributes *attrs) {
  if (attrs->dropout_probability().has_value() &&
      *attrs->dropout_probability() > 0.0f)
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with dropout not supported.");
  if (attrs->alibi_mask())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with alibi mask not supported.");
  if (attrs->padding_mask())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with padding mask not supported.");
  if (attrs->stats_tensor_uid().has_value())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with stats output not supported.");
  if (attrs->seed_tensor_uid().has_value() ||
      attrs->offset_tensor_uid().has_value() ||
      attrs->dropout_mask_tensor_uid().has_value() ||
      attrs->dropout_scale_tensor_uid().has_value())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with dropout tensors not supported.");
  if (attrs->page_table_k_tensor_uid().has_value() ||
      attrs->page_table_v_tensor_uid().has_value())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with paged attention not supported.");
  if (attrs->block_mask_tensor_uid().has_value() ||
      attrs->sink_token_tensor_uid().has_value())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with block mask not supported.");
  if (attrs->left_bound().has_value() || attrs->right_bound().has_value())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with sliding window not supported.");
  if (attrs->descale_q_tensor_uid().has_value() ||
      attrs->descale_k_tensor_uid().has_value() ||
      attrs->descale_v_tensor_uid().has_value() ||
      attrs->descale_s_tensor_uid().has_value() ||
      attrs->scale_s_tensor_uid().has_value() ||
      attrs->scale_o_tensor_uid().has_value() ||
      attrs->amax_s_tensor_uid().has_value() ||
      attrs->amax_o_tensor_uid().has_value())
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA with FP8 quantization not supported.");
  if (attrs->diagonal_alignment() !=
      hipdnn_data_sdk::data_objects::DiagonalAlignment::TOP_LEFT)
    return fusilli::error(
        fusilli::ErrorCode::NotImplemented,
        "SDPA with non-TOP_LEFT diagonal alignment not supported.");
  // This is out of an over-abundance of caution, IREE doesn't need a backend
  // implementation hit.
  if (attrs->implementation() !=
      hipdnn_data_sdk::data_objects::AttentionImplementation::AUTO)
    return fusilli::error(
        fusilli::ErrorCode::NotImplemented,
        "SDPA with explicit implementation strategy not supported.");

  // Causal attention has an explicit attn_mask, torch will reject causal with
  // additional attn_mask.
  if (attrs->causal_mask() && attrs->attn_mask_tensor_uid().has_value())
    return fusilli::error(
        fusilli::ErrorCode::NotImplemented,
        "SDPA with both causal mask and attention mask not supported.");

  return fusilli::ok();
}

// Validate tensor dimension constraints for SDPA.
// Expected layout: Q[B,H,S,D], K[B,H_kv,S,D], V[B,H_kv,S,D] where H is a
// multiple of H_kv (equal for standard MHA and > 1 for GQA).
inline fusilli::ErrorObject
validateInputs(const std::shared_ptr<fusilli::TensorAttr> &q,
               const std::shared_ptr<fusilli::TensorAttr> &k,
               const std::shared_ptr<fusilli::TensorAttr> &v) {
  const auto &qDim = q->getDim();
  const auto &kDim = k->getDim();
  const auto &vDim = v->getDim();

  constexpr size_t kRequiredRank = 4;
  if (qDim.size() != kRequiredRank || kDim.size() != kRequiredRank ||
      vDim.size() != kRequiredRank)
    return fusilli::error(fusilli::ErrorCode::InvalidAttribute,
                          "SDPA tensors must be rank 4 [B, H, S, D].");
  if (qDim[0] != kDim[0] || qDim[0] != vDim[0])
    return fusilli::error(fusilli::ErrorCode::InvalidAttribute,
                          "SDPA batch dimensions must match.");
  if (kDim[1] != vDim[1])
    return fusilli::error(fusilli::ErrorCode::InvalidAttribute,
                          "SDPA K and V num_heads must match.");
  // GQA: query heads must be a multiple of KV heads.
  if (qDim[1] % kDim[1] != 0)
    return fusilli::error(fusilli::ErrorCode::InvalidAttribute,
                          "SDPA Q num_heads must be a multiple of "
                          "K/V num_heads for GQA.");
  if (kDim[2] != vDim[2])
    return fusilli::error(fusilli::ErrorCode::InvalidAttribute,
                          "SDPA K and V seq_len must match.");
  if (qDim[3] != kDim[3])
    return fusilli::error(fusilli::ErrorCode::InvalidAttribute,
                          "SDPA Q and K head_dim must match.");

  // #TODO(iree/issues/21858) GQA with f32 triggers an IREE distribution
  // failure.
  bool isGqa = qDim[1] != kDim[1];
  if (isGqa && q->getDataType() == fusilli::DataType::Float)
    return fusilli::error(fusilli::ErrorCode::NotImplemented,
                          "SDPA GQA with f32 not supported.");

  return fusilli::ok();
}

// Build the MLIR template string for the CustomOp by selecting the
// appropriate R-string template and resolving scalar placeholders.
inline std::string buildMLIR(bool hasAttnMask, float dropoutP, bool isCausal,
                             std::optional<float> scale = std::nullopt,
                             bool enableGqa = false) {
  std::string dropoutStr = std::format("{:e}", dropoutP);
  std::string isCausalStr = isCausal ? "true" : "false";
  std::string scaleConstStr =
      scale.has_value() ? std::format("torch.constant.float {:e}", *scale)
                        : "torch.constant.none";
  std::string scaleTypeStr = scale.has_value() ? "!torch.float" : "!torch.none";
  std::string enableGqaStr = enableGqa ? "true" : "false";

  return std::vformat(hasAttnMask ? kSdpaWithMask : kSdpaNoMask,
                      std::make_format_args(dropoutStr,    // {0} DROPOUT_P
                                            isCausalStr,   // {1} IS_CAUSAL
                                            scaleConstStr, // {2} SCALE_CONST
                                            scaleTypeStr,  // {3} SCALE_TYPE
                                            enableGqaStr   // {4} ENABLE_GQA
                                            ));
}
} // namespace SdpaImport

#endif // FUSILLI_PLUGIN_CUSTOM_OPS_H
