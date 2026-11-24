// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockAttentionQuantScaleEnum
{
    NO_SCALE      = 0,
    PERTENSOR     = 1,
    BLOCKSCALE    = 2,
    KV_BLOCKSCALE = 3, // Q per-tensor, K/V per-page block scale
    MX            = 4, // Microscaling
    PERHEAD       = 5, // one descale per (batch, head)
    DYNAMIC_Q     = 6, // bf16 Q quantized per row in the kernel, K/V per-tensor
};

template <BlockAttentionQuantScaleEnum>
struct BlockAttentionQuantScaleEnumToStr;

template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::NO_SCALE>
{
    static constexpr const char* name = "";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::PERTENSOR>
{
    static constexpr const char* name = "pertensor";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::BLOCKSCALE>
{
    static constexpr const char* name = "blockscale";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::KV_BLOCKSCALE>
{
    static constexpr const char* name = "kv_blockscale";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::MX>
{
    static constexpr const char* name = "mx";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::PERHEAD>
{
    static constexpr const char* name = "perhead";
};
template <>
struct BlockAttentionQuantScaleEnumToStr<BlockAttentionQuantScaleEnum::DYNAMIC_Q>
{
    static constexpr const char* name = "dynq";
};

} // namespace ck_tile
