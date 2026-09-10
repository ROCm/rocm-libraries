// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck_tile {

// This class is used for codegen pattern matching
enum class BlockFmhaPipelineEnum
{
    QRKSVS = 0,
    QRKSVS_ASYNC,
    QSKSVS,
    QRKSVS_ASYNC_TRLOAD,
    QRKSVS_ASYNC_TRLOAD_V3,
    QRKSVS_HPAD,
    QRKSVS_TDM,
    QRKSVS_TDM_D192_V128,
    QRKSVS_TDM_V128,
};

template <BlockFmhaPipelineEnum>
struct BlockFmhaPipelineEnumToStr;

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS>
{
    static constexpr const char* name = "qr";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC>
{
    static constexpr const char* name = "qr_async";
};
template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QSKSVS>
{
    static constexpr const char* name = "qs";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_ASYNC_TRLOAD>
{
    static constexpr const char* name = "qr_async_trload";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_HPAD>
{
    static constexpr const char* name = "qr_hpad";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_TDM>
{
    static constexpr const char* name = "qr_tdm";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_TDM_D192_V128>
{
    static constexpr const char* name = "qr_tdm_d192_v128";
};

template <>
struct BlockFmhaPipelineEnumToStr<BlockFmhaPipelineEnum::QRKSVS_TDM_V128>
{
    static constexpr const char* name = "qr_tdm_v128";
};

} // namespace ck_tile
