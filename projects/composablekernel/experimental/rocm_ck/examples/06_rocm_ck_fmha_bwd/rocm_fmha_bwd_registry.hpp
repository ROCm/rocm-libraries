// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Variant registry for programmatic kernel selection.
// Host-only header — no CK Tile dependency.
//
// Three kernel families, three variant arrays, three findVariant overloads.

#pragma once

#include "rocm_fmha_bwd_api.hpp"

#include <iterator> // std::size

namespace rocm_ck {

// =========================================================================
// OGradDotO variants
// =========================================================================

struct FmhaBwdOGradDotOVariant
{
    const char* name;
    FmhaBwdOGradDotOKernel kernel;
};

// clang-format off
static constexpr FmhaBwdOGradDotOVariant ALL_OGRAD_DOT_O_VARIANTS[] = {
    {"fmha_bwd_ograd_dot_o_fp16_d128_batch", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_bf16_d128_batch", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::BF16, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d64_batch", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 64,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d128_group", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}})},
    {"fmha_bwd_ograd_dot_o_fp16_d128_batch_npad", make_kernel(FmhaBwdOGradDotOConfig{
         .signature = {.dtype = DataType::FP16, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}})},
};
// clang-format on

static constexpr int ALL_OGRAD_DOT_O_VARIANTS_COUNT = std::size(ALL_OGRAD_DOT_O_VARIANTS);

/// Find the best OGradDotO variant matching the given config.
constexpr const FmhaBwdOGradDotOVariant* findVariant(FmhaBwdOGradDotOConfig cfg)
{
    const auto& sig  = cfg.signature;
    const auto& algo = cfg.algorithm;

    const FmhaBwdOGradDotOVariant* best_nopad  = nullptr;
    const FmhaBwdOGradDotOVariant* best_padded = nullptr;

    for(int i = 0; i < ALL_OGRAD_DOT_O_VARIANTS_COUNT; ++i)
    {
        const auto& v = ALL_OGRAD_DOT_O_VARIANTS[i];
        if(v.kernel.dtype != sig.dtype || v.kernel.hdim_v != sig.hdim_v ||
           v.kernel.mode != sig.mode)
            continue;

        if(!v.kernel.pad_seqlen_q && !v.kernel.pad_hdim_v)
            best_nopad = &ALL_OGRAD_DOT_O_VARIANTS[i];
        else
            best_padded = &ALL_OGRAD_DOT_O_VARIANTS[i];
    }

    if(!algo.pad_seqlen_q && !algo.pad_hdim_v && best_nopad)
        return best_nopad;

    return best_padded ? best_padded : best_nopad;
}

// =========================================================================
// DqDkDv variants
// =========================================================================

struct FmhaBwdDQDKDVVariant
{
    const char* name;
    FmhaBwdDQDKDVKernel kernel;
};

// clang-format off
static constexpr FmhaBwdDQDKDVVariant ALL_DQDKDV_VARIANTS[] = {
    {"fmha_bwd_dqdkdv_fp16_d128_batch", make_kernel(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_bf16_d128_batch", make_kernel(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::BF16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_cmask", make_kernel(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.has_mask = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_batch_det", make_kernel(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {.is_deterministic = true,
                       .pad_hdim_q = 8, .pad_hdim_v = 8}})},
    {"fmha_bwd_dqdkdv_fp16_d128_group", make_kernel(FmhaBwdDQDKDVConfig{
         .signature = {.dtype = DataType::FP16,
                       .hdim_q = 128, .hdim_v = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {.pad_hdim_q = 8, .pad_hdim_v = 8}})},
};
// clang-format on

static constexpr int ALL_DQDKDV_VARIANTS_COUNT = std::size(ALL_DQDKDV_VARIANTS);

/// Find the best DqDkDv variant matching the given config.
/// Matches on signature (dtype, hdim_q, hdim_v, mode) and feature flags.
constexpr const FmhaBwdDQDKDVVariant* findVariant(FmhaBwdDQDKDVConfig cfg)
{
    const auto& sig  = cfg.signature;
    const auto& algo = cfg.algorithm;

    for(int i = 0; i < ALL_DQDKDV_VARIANTS_COUNT; ++i)
    {
        const auto& v = ALL_DQDKDV_VARIANTS[i];
        if(v.kernel.dtype != sig.dtype || v.kernel.hdim_q != sig.hdim_q ||
           v.kernel.hdim_v != sig.hdim_v || v.kernel.mode != sig.mode)
            continue;

        // Feature flags must match exactly
        if(v.kernel.has_mask != algo.has_mask || v.kernel.has_dropout != algo.has_dropout ||
           v.kernel.is_deterministic != algo.is_deterministic ||
           v.kernel.bias_type != algo.bias_type)
            continue;

        return &ALL_DQDKDV_VARIANTS[i];
    }
    return nullptr;
}

// =========================================================================
// ConvertDQ variants
// =========================================================================

struct FmhaBwdConvertDQVariant
{
    const char* name;
    FmhaBwdConvertDQKernel kernel;
};

// clang-format off
static constexpr FmhaBwdConvertDQVariant ALL_CONVERT_DQ_VARIANTS[] = {
    {"fmha_bwd_convert_dq_fp16_d128_batch_det", make_kernel(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                       .mode = FmhaMode::BATCH},
         .algorithm = {}})},
    {"fmha_bwd_convert_dq_fp16_d128_group_det", make_kernel(FmhaBwdConvertDQConfig{
         .signature = {.dtype = DataType::FP16, .hdim_q = 128,
                       .mode = FmhaMode::GROUP},
         .algorithm = {}})},
};
// clang-format on

static constexpr int ALL_CONVERT_DQ_VARIANTS_COUNT = std::size(ALL_CONVERT_DQ_VARIANTS);

/// Find the best ConvertDQ variant matching the given config.
constexpr const FmhaBwdConvertDQVariant* findVariant(FmhaBwdConvertDQConfig cfg)
{
    const auto& sig = cfg.signature;

    for(int i = 0; i < ALL_CONVERT_DQ_VARIANTS_COUNT; ++i)
    {
        const auto& v = ALL_CONVERT_DQ_VARIANTS[i];
        if(v.kernel.dtype != sig.dtype || v.kernel.hdim_q != sig.hdim_q ||
           v.kernel.mode != sig.mode)
            continue;

        return &ALL_CONVERT_DQ_VARIANTS[i];
    }
    return nullptr;
}

} // namespace rocm_ck
