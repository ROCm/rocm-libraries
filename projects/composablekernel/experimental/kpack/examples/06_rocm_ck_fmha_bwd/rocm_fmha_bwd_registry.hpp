// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Variant registry for programmatic kernel selection.
// Host-only header — no CK Tile dependency.

#pragma once

#include "rocm_fmha_bwd_api.hpp"

namespace rocm_ck {

/// Descriptor for a compiled FMHA BWD OGradDotO variant in the kpack archive.
struct FmhaBwdVariantDescriptor
{
    const char* name;
    FmhaBwdOGradDotOKernel kernel;
};

/// Complete table of all compiled variants, matching the kpack archive contents.
/// Each entry corresponds to a .hip file and its make_kernel configuration.
// clang-format off
static constexpr FmhaBwdVariantDescriptor ALL_FMHA_BWD_VARIANTS[] = {
    {"fmha_bwd_ograd_dot_o_fp16_d128_batch", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_bf16_d128_batch", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::BF16, .hdim_v = 128, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d64_batch", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 64, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d128_group", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::GROUP},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d128_batch_npad", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128, .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}})},
};
// clang-format on

static constexpr int ALL_FMHA_BWD_VARIANTS_COUNT =
    sizeof(ALL_FMHA_BWD_VARIANTS) / sizeof(ALL_FMHA_BWD_VARIANTS[0]);

/// Find the best variant matching the given config.
///
/// Matching logic: exact match on signature (dtype + hdim_v + mode). Among
/// matches, prefer the variant whose padding flags match the algorithm's
/// pad_seqlen_q/pad_hdim_v. Falls back to padded if no exact match.
///
/// Returns nullptr if no variant matches the signature.
constexpr const FmhaBwdVariantDescriptor* findVariant(FmhaBwdOGradDotOConfig cfg)
{
    const auto& sig  = cfg.signature;
    const auto& algo = cfg.algorithm;

    const FmhaBwdVariantDescriptor* best_nopad  = nullptr;
    const FmhaBwdVariantDescriptor* best_padded = nullptr;

    for(int i = 0; i < ALL_FMHA_BWD_VARIANTS_COUNT; ++i)
    {
        const auto& v = ALL_FMHA_BWD_VARIANTS[i];
        if(v.kernel.dtype != sig.dtype || v.kernel.hdim_v != sig.hdim_v ||
           v.kernel.mode != sig.mode)
            continue;

        if(!v.kernel.pad_seqlen_q && !v.kernel.pad_hdim_v)
            best_nopad = &ALL_FMHA_BWD_VARIANTS[i];
        else
            best_padded = &ALL_FMHA_BWD_VARIANTS[i];
    }

    // If caller doesn't need padding and a no-pad variant exists, use it
    if(!algo.pad_seqlen_q && !algo.pad_hdim_v && best_nopad)
        return best_nopad;

    // Otherwise use padded (works for all alignments)
    return best_padded ? best_padded : best_nopad;
}

} // namespace rocm_ck
