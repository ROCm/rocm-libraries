// SPDX-License-Identifier: MIT
// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.

#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_enum.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_load.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_d192_v128_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/tile_fmha_shape.hpp"
#include "ck_tile/ops/tdm.hpp"

namespace {

using DataType = ck_tile::bf16_t;

using D192BlockShape = ck_tile::TileFmhaShape<ck_tile::sequence<128, 128, 32, 128, 32, 192>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              ck_tile::sequence<4, 1, 1>,
                                              ck_tile::sequence<16, 16, 32>,
                                              true>;

struct D192PolicyProblem
{
    struct AttentionVariant
    {
    };
    struct FmhaMask
    {
    };

    using QDataType             = DataType;
    using KDataType             = DataType;
    using VDataType             = DataType;
    using SaccDataType          = float;
    using SMPLComputeDataType   = float;
    using BiasDataType          = DataType;
    using RandValOutputDataType = std::uint8_t;
    using LSEDataType           = float;
    using PDataType             = DataType;
    using OaccDataType          = float;
    using ODataType             = DataType;
    using BlockFmhaShape        = D192BlockShape;

    static constexpr ck_tile::index_t kBlockSize                   = 128;
    [[maybe_unused]] static constexpr ck_tile::index_t kBlockPerCu = 1;
    [[maybe_unused]] static constexpr bool kIsGroupMode            = false;
    [[maybe_unused]] static constexpr bool kPadSeqLenQ             = false;
    [[maybe_unused]] static constexpr bool kPadSeqLenK             = false;
    [[maybe_unused]] static constexpr bool kPadHeadDimQ            = false;
    [[maybe_unused]] static constexpr bool kPadHeadDimV            = false;
    static constexpr bool kHasLogitsSoftCap                        = false;
    static constexpr bool kHasDropout                              = false;
    [[maybe_unused]] static constexpr bool kStoreLSE               = false;
    [[maybe_unused]] static constexpr bool kHasSink                = false;
    static constexpr auto BiasEnum = ck_tile::BlockAttentionBiasEnum::NO_BIAS;
};

using D192Policy   = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128Policy;
using D192Pipeline = ck_tile::BlockFmhaPipelineQRKSVSTdmD192V128<D192PolicyProblem>;

struct DummyWindow
{
};

struct DummyPositionEncoding
{
};

static_assert(!std::is_invocable_v<D192Pipeline,
                                   const DummyWindow&,
                                   const DummyWindow&,
                                   const DummyWindow&,
                                   const DummyWindow&,
                                   DummyWindow&,
                                   D192PolicyProblem::FmhaMask,
                                   DummyPositionEncoding,
                                   float,
                                   void*,
                                   float>);

static_assert(D192Policy::IsSupportedProblem<D192PolicyProblem>());
static_assert(std::string_view{D192Pipeline::name} == "qr_tdm_d192_v128");
static_assert(std::string_view{ck_tile::BlockFmhaPipelineEnumToStr<
                  ck_tile::BlockFmhaPipelineEnum::QRKSVS_TDM_D192_V128>::name} ==
              "qr_tdm_d192_v128");
static_assert(D192Pipeline::kUsesUntransposedVKernelPath);
static_assert(D192Pipeline::kUsesTdmAffineDramPath);
static_assert(D192Pipeline::kUsesFixedSegmentedLdsArena);
static_assert(D192Pipeline::kBlockPerCu == 1);
static_assert(D192Policy::GetQKReductionSteps<D192PolicyProblem>() == 6);
static_assert(D192Policy::kKPrefetchTensorCount == CK_TILE_FMHA_GFX125_D192_K_PREFETCH_TENSORCNT);
static_assert(D192Policy::kVPrefetchTensorCount == CK_TILE_FMHA_GFX125_D192_V_PREFETCH_TENSORCNT);
static_assert(D192Policy::kPrefetchTailDrain ==
              (CK_TILE_FMHA_GFX125_D192_PREFETCH_TAIL_DRAIN != 0));
static_assert(D192Policy::kQkStage0TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE0_TAIL_DSCNT);
static_assert(D192Policy::kQkStage1TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE1_TAIL_DSCNT);
static_assert(D192Policy::kQkStage2TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE2_TAIL_DSCNT);
static_assert(D192Policy::kQkStage3TailDsCount == CK_TILE_FMHA_GFX125_D192_QK_STAGE3_TAIL_DSCNT);
static_assert(D192Policy::kPvStage0TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE0_TAIL_DSCNT);
static_assert(D192Policy::kPvStage1TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE1_TAIL_DSCNT);
static_assert(D192Policy::kPvStage2TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE2_TAIL_DSCNT);
static_assert(D192Policy::kPvStage3TailDsCount == CK_TILE_FMHA_GFX125_D192_PV_STAGE3_TAIL_DSCNT);
// Two arena layouts: the 64 KiB-aligned original and the packed one selected by
// CK_TILE_FMHA_GFX125_D192_LDS_PACK. Both must keep the buffers non-overlapping
// and 16-byte aligned; only the offsets differ.
#if CK_TILE_FMHA_GFX125_D192_LDS_PACK
static_assert(D192Policy::GetLdsOffsetK0() == 0x00000);
static_assert(D192Policy::GetLdsOffsetK1() == 0x0c800);
static_assert(D192Policy::GetLdsOffsetV0() == 0x19000);
static_assert(D192Policy::GetLdsOffsetV1() == 0x22000);
static_assert(D192Policy::GetLdsArenaSize() == 0x2b000);
static_assert(D192Pipeline::GetSmemSize() == 0x2b000);
#else
static_assert(D192Policy::GetLdsOffsetK0() == 0x00000);
static_assert(D192Policy::GetLdsOffsetK1() == 0x10000);
static_assert(D192Policy::GetLdsOffsetV0() == 0x20000);
static_assert(D192Policy::GetLdsOffsetV1() == 0x30000);
static_assert(D192Policy::GetLdsArenaSize() == 0x39000);
static_assert(D192Pipeline::GetSmemSize() == 0x39000);
#endif

constexpr auto kPolicyKLdsWriteDesc = D192Policy::MakeKLdsWriteBlockDescriptor<D192PolicyProblem>();
constexpr auto kPolicyKLdsReadDesc  = D192Policy::MakeKLdsReadBlockDescriptor<D192PolicyProblem>();
constexpr auto kPolicyVLdsWriteDesc = D192Policy::MakeVLdsWriteBlockDescriptor<D192PolicyProblem>();
constexpr auto kPolicyVLdsReadDesc  = D192Policy::MakeVLdsReadBlockDescriptor<D192PolicyProblem>();

static_assert(kPolicyKLdsWriteDesc.get_lengths()[ck_tile::number<0>{}] == 128);
static_assert(kPolicyKLdsWriteDesc.get_lengths()[ck_tile::number<1>{}] == 200);
static_assert(kPolicyKLdsReadDesc.get_lengths()[ck_tile::number<0>{}] == 128);
static_assert(kPolicyKLdsReadDesc.get_lengths()[ck_tile::number<1>{}] == 192);
static_assert(kPolicyVLdsWriteDesc.get_lengths()[ck_tile::number<0>{}] == 128);
static_assert(kPolicyVLdsWriteDesc.get_lengths()[ck_tile::number<1>{}] == 128);
static_assert(kPolicyVLdsReadDesc.get_lengths()[ck_tile::number<0>{}] == 128);
static_assert(kPolicyVLdsReadDesc.get_lengths()[ck_tile::number<1>{}] == 128);
static_assert(kPolicyKLdsWriteDesc.calculate_offset(ck_tile::make_tuple(ck_tile::number<1>{},
                                                                        ck_tile::number<0>{})) ==
              200);
static_assert(kPolicyKLdsReadDesc.calculate_offset(ck_tile::make_tuple(ck_tile::number<1>{},
                                                                       ck_tile::number<0>{})) ==
              200);
static_assert(kPolicyVLdsWriteDesc.calculate_offset(ck_tile::make_tuple(ck_tile::number<1>{},
                                                                        ck_tile::number<0>{})) ==
              144);
static_assert(kPolicyVLdsReadDesc.calculate_offset(ck_tile::make_tuple(ck_tile::number<1>{},
                                                                       ck_tile::number<0>{})) ==
              144);
static_assert(kPolicyKLdsWriteDesc.get_element_space_size() == 128 * 200);
static_assert(kPolicyVLdsWriteDesc.get_element_space_size() == 127 * 144 + 128);
static_assert(D192Policy::GetSmemSizeK<D192PolicyProblem>() == 0xc800);
static_assert(D192Policy::GetSmemSizeV<D192PolicyProblem>() == 0x9000);
static_assert(D192Policy::GetSmemSize<D192PolicyProblem>() == 0x39000);

constexpr auto kPolicyKPadding = D192Policy::GetLdsPaddingConfigK<D192PolicyProblem>();
constexpr auto kPolicyVPadding = D192Policy::GetLdsPaddingConfigV<D192PolicyProblem>();
static_assert(kPolicyKPadding[ck_tile::number<0>{}] == false);
static_assert(kPolicyKPadding[ck_tile::number<1>{}] == 0);
static_assert(kPolicyKPadding[ck_tile::number<2>{}] == 0);
static_assert(kPolicyVPadding[ck_tile::number<0>{}] == true);
static_assert(kPolicyVPadding[ck_tile::number<1>{}] == 7);
static_assert(kPolicyVPadding[ck_tile::number<2>{}] == 5);

constexpr ck_tile::index_t kRows                  = D192Policy::kLdsRows;
constexpr ck_tile::index_t kKValidCols            = D192Policy::kKValidWidth;
constexpr ck_tile::index_t kKTileCols             = D192Policy::kKPhysicalStride;
constexpr ck_tile::index_t kKStride               = D192Policy::kKPhysicalStride;
constexpr ck_tile::index_t kVCols                 = D192Policy::kVLogicalWidth;
constexpr ck_tile::index_t kVStride               = D192Policy::kVPhysicalStride;
constexpr ck_tile::index_t kVReadRows             = 32;
constexpr ck_tile::index_t kBlockSize             = 128;
constexpr std::size_t kArenaBytes                 = D192Policy::kLdsArenaSize;
constexpr std::size_t kSegmentBytes               = 0x10000;
constexpr std::size_t kKFootprintBytes            = D192Policy::kKFootprintBytes;
constexpr std::size_t kVFootprintBytes            = D192Policy::kVFootprintBytes;
constexpr std::array<std::size_t, 4> kRegionBases = {D192Policy::kLdsOffsetK0,
                                                     D192Policy::kLdsOffsetK1,
                                                     D192Policy::kLdsOffsetV0,
                                                     D192Policy::kLdsOffsetV1};
constexpr std::array<std::size_t, 4> kRegionSizes = {
    kKFootprintBytes, kKFootprintBytes, kVFootprintBytes, kVFootprintBytes};
constexpr std::array<std::uint8_t, 4> kRegionCanaries = {0x11, 0x22, 0x33, 0x44};
constexpr std::uint8_t kUntouchedCanary               = 0xa5;
constexpr std::uint32_t kSeed                         = 0xd1920128;

struct ProbeArgs
{
    const void* input;
    void* dump;
    void* transpose;
    ck_tile::index_t rows;
    ck_tile::index_t valid_cols;
    ck_tile::index_t input_stride;
    ck_tile::index_t lds_offset;
};

struct DescriptorArgs
{
    const void* input;
    void* output;
    ck_tile::index_t input_stride;
};

constexpr std::size_t kDescriptorWordCount = 28;

template <ck_tile::index_t Rows, ck_tile::index_t Cols, ck_tile::index_t Warps>
CK_TILE_DEVICE constexpr auto make_tdm_distribution()
{
    static_assert(Rows % Warps == 0);

    return ck_tile::make_static_tile_distribution(
        ck_tile::tile_distribution_encoding<
            ck_tile::sequence<>,
            ck_tile::tuple<ck_tile::sequence<Warps, Rows / Warps>, ck_tile::sequence<Cols>>,
            ck_tile::tuple<ck_tile::sequence<1>>,
            ck_tile::tuple<ck_tile::sequence<0>>,
            ck_tile::sequence<1, 2>,
            ck_tile::sequence<1, 0>>{},
        ck_tile::bool_constant<true>{});
}

template <ck_tile::index_t Rows,
          ck_tile::index_t ValidCols,
          ck_tile::index_t TileCols,
          ck_tile::index_t LdsStride,
          ck_tile::index_t Warps,
          bool Pad,
          bool FullArenaInit = true,
          bool CopyDump      = true>
struct TdmDumpKernel
{
    static constexpr ck_tile::index_t kBlockSize = Warps * 32;

    CK_TILE_DEVICE void operator()(ProbeArgs args) const
    {
        __shared__ char arena[kArenaBytes];

        constexpr std::size_t copy_bytes = Rows * LdsStride * sizeof(DataType);
        constexpr std::size_t init_bytes = FullArenaInit ? kArenaBytes : copy_bytes;
        for(std::size_t i = threadIdx.x; i < init_bytes; i += blockDim.x)
        {
            const std::size_t offset = FullArenaInit ? i : args.lds_offset + i;
            arena[offset]            = static_cast<char>(kUntouchedCanary);
        }
        __syncthreads();

        const auto* input = static_cast<const DataType*>(args.input);
        auto input_view   = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            input,
            ck_tile::make_tuple(args.rows, args.valid_cols),
            ck_tile::make_tuple(args.input_stride, 1));
        constexpr auto input_distribution = []() {
            if constexpr(Rows == kRows && TileCols == kKTileCols && LdsStride == kKStride && !Pad)
            {
                return D192Policy::MakeKDramTileDistribution<D192PolicyProblem>();
            }
            else if constexpr(Rows == kRows && TileCols == kVCols && LdsStride == kVStride && Pad)
            {
                return D192Policy::MakeVDramTileDistribution<D192PolicyProblem>();
            }
            else
            {
                return make_tdm_distribution<Rows, TileCols, Warps>();
            }
        }();
        auto input_window = ck_tile::make_tile_window(
            input_view,
            ck_tile::make_tuple(ck_tile::number<Rows>{}, ck_tile::number<TileCols>{}),
            {0, 0},
            input_distribution);

        auto* lds                     = reinterpret_cast<DataType*>(arena + args.lds_offset);
        constexpr auto lds_descriptor = []() {
            if constexpr(Rows == kRows && TileCols == kKTileCols && LdsStride == kKStride && !Pad)
            {
                return D192Policy::MakeKLdsWriteBlockDescriptor<D192PolicyProblem>();
            }
            else if constexpr(Rows == kRows && TileCols == kVCols && LdsStride == kVStride && Pad)
            {
                return D192Policy::MakeVLdsWriteBlockDescriptor<D192PolicyProblem>();
            }
            else
            {
                return ck_tile::make_naive_tensor_descriptor(
                    ck_tile::make_tuple(ck_tile::number<Rows>{}, ck_tile::number<TileCols>{}),
                    ck_tile::make_tuple(ck_tile::number<LdsStride>{}, ck_tile::number<1>{}),
                    ck_tile::number<1>{},
                    ck_tile::number<1>{});
            }
        }();
        auto lds_view =
            ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(lds, lds_descriptor);
        auto lds_window = ck_tile::make_tile_window(
            lds_view,
            ck_tile::make_tuple(ck_tile::number<Rows>{}, ck_tile::number<TileCols>{}),
            {0, 0});

        ck_tile::TDMConfig config{};
        if constexpr(Pad)
        {
            config.pad_enable              = true;
            config.pad_config.pad_interval = 5;
            config.pad_config.pad_amount   = 7;
        }

        ck_tile::load_tile_tdm(config, lds_window, input_window);
        ck_tile::s_wait_tensorcnt_barrier<0>();

        if constexpr(CopyDump)
        {
            auto* dump = static_cast<std::uint8_t*>(args.dump);
            for(std::size_t i = threadIdx.x; i < copy_bytes; i += blockDim.x)
            {
                dump[i] = static_cast<std::uint8_t>(arena[args.lds_offset + i]);
            }
        }
    }
};

template <ck_tile::index_t Rows,
          ck_tile::index_t ValidCols,
          ck_tile::index_t TileCols,
          ck_tile::index_t SecondDimStride>
struct RawDescriptorCaptureKernel
{
    static constexpr ck_tile::index_t kBlockSize = 32;

    CK_TILE_DEVICE void operator()(DescriptorArgs args) const
    {
        __shared__ char arena[kArenaBytes];

        const uint32_t global_dims[2]    = {ValidCols, Rows};
        const uint64_t global_strides[2] = {static_cast<uint64_t>(args.input_stride),
                                            SecondDimStride};
        const uint16_t box_dims[2]       = {TileCols, Rows};
        ck_tile::TDMConfig config{};
        const auto descriptor = ck_tile::createTDMDescriptor<DataType, 2>(
            args.input, arena, global_dims, global_strides, box_dims, config);
        const auto groups = descriptor.getResourceDescriptorGroup();

        if(threadIdx.x == 0)
        {
            auto* output      = static_cast<std::uint32_t*>(args.output);
            const auto group0 = ck_tile::bit_cast<ck_tile::array<std::uint32_t, 4>>(
                groups.get(ck_tile::number<0>{}));
            const auto group1 = ck_tile::bit_cast<ck_tile::array<std::uint32_t, 8>>(
                groups.get(ck_tile::number<1>{}));
            const auto group2 = ck_tile::bit_cast<ck_tile::array<std::uint32_t, 4>>(
                groups.get(ck_tile::number<2>{}));
            const auto group3 = ck_tile::bit_cast<ck_tile::array<std::uint32_t, 4>>(
                groups.get(ck_tile::number<3>{}));
            const auto group4 = ck_tile::bit_cast<ck_tile::array<std::uint32_t, 8>>(
                groups.get(ck_tile::number<4>{}));

            ck_tile::static_for<0, 4, 1>{}(
                [&](auto i) { output[i] = static_cast<std::uint32_t>(group0[i]); });
            ck_tile::static_for<0, 8, 1>{}(
                [&](auto i) { output[4 + i] = static_cast<std::uint32_t>(group1[i]); });
            ck_tile::static_for<0, 4, 1>{}(
                [&](auto i) { output[12 + i] = static_cast<std::uint32_t>(group2[i]); });
            ck_tile::static_for<0, 4, 1>{}(
                [&](auto i) { output[16 + i] = static_cast<std::uint32_t>(group3[i]); });
            ck_tile::static_for<0, 8, 1>{}(
                [&](auto i) { output[20 + i] = static_cast<std::uint32_t>(group4[i]); });
        }
    }
};

template <ck_tile::index_t Rows,
          ck_tile::index_t ValidCols,
          ck_tile::index_t TileCols,
          ck_tile::index_t SecondDimStride,
          bool FullArenaInit = false>
struct RawTdmDumpKernel
{
    static constexpr ck_tile::index_t kBlockSize = 32;

    CK_TILE_DEVICE void operator()(ProbeArgs args) const
    {
        __shared__ char arena[kArenaBytes];

        constexpr std::size_t copy_bytes = Rows * TileCols * sizeof(DataType);
        constexpr std::size_t init_bytes = FullArenaInit ? kArenaBytes : copy_bytes;
        for(std::size_t i = threadIdx.x; i < init_bytes; i += blockDim.x)
        {
            const std::size_t offset = FullArenaInit ? i : args.lds_offset + i;
            arena[offset]            = static_cast<char>(kUntouchedCanary);
        }
        __syncthreads();

        const uint32_t global_dims[2]    = {ValidCols, Rows};
        const uint64_t global_strides[2] = {static_cast<uint64_t>(args.input_stride),
                                            SecondDimStride};
        const uint16_t box_dims[2]       = {TileCols, Rows};
        ck_tile::TDMConfig config{};
        auto* lds             = reinterpret_cast<DataType*>(arena + args.lds_offset);
        const auto descriptor = ck_tile::createTDMDescriptor<DataType, 2>(
            args.input, lds, global_dims, global_strides, box_dims, config);

        ck_tile::amd_tdm_load(descriptor);
        ck_tile::s_wait_tensorcnt_barrier<0>();

        auto* dump = static_cast<std::uint8_t*>(args.dump);
        for(std::size_t i = threadIdx.x; i < copy_bytes; i += blockDim.x)
        {
            dump[i] = static_cast<std::uint8_t>(arena[args.lds_offset + i]);
        }
    }
};

template <bool Incremental>
struct VPadTransposeKernel
{
    static constexpr ck_tile::index_t kBlockSize = ::kBlockSize;

    [[maybe_unused]] CK_TILE_DEVICE void operator()(ProbeArgs args) const
    {
        __shared__ char arena[kArenaBytes];

        for(std::size_t i = threadIdx.x; i < kArenaBytes; i += blockDim.x)
        {
            arena[i] = static_cast<char>(kUntouchedCanary);
        }
        __syncthreads();

        const auto* input = static_cast<const DataType*>(args.input);
        auto input_view   = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            input,
            ck_tile::make_tuple(args.rows, args.valid_cols),
            ck_tile::make_tuple(args.input_stride, 1));
        auto input_window = ck_tile::make_tile_window(
            input_view,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kVCols>{}),
            {0, 0},
            make_tdm_distribution<kRows, kVCols, 4>());

        auto* lds               = reinterpret_cast<DataType*>(arena + args.lds_offset);
        constexpr auto lds_desc = D192Policy::MakeVLdsWriteBlockDescriptor<D192PolicyProblem>();
        auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(lds, lds_desc);
        auto lds_write_window = ck_tile::make_tile_window(
            lds_view,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kVCols>{}),
            {0, 0});

        ck_tile::TDMConfig config{};
        config.pad_enable              = true;
        config.pad_config.pad_interval = 5;
        config.pad_config.pad_amount   = 7;
        ck_tile::load_tile_tdm(config, lds_write_window, input_window);
        ck_tile::s_wait_tensorcnt_barrier<0>();

        auto* transpose  = static_cast<DataType*>(args.transpose);
        auto output_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            transpose,
            ck_tile::make_tuple(args.valid_cols, args.rows),
            ck_tile::make_tuple(args.rows, 1));
        ck_tile::static_for<0, kRows / kVReadRows, 1>{}([&](auto row_block) {
            constexpr ck_tile::index_t row_offset = row_block * kVReadRows;
            auto lds_read_window                  = ck_tile::make_tile_window(
                lds_view,
                ck_tile::make_tuple(ck_tile::number<kVReadRows>{}, ck_tile::number<kVCols>{}),
                {row_offset, 0},
                D192Policy::MakeVRegTileDistribution<D192PolicyProblem>());
            auto transposed = [&]() {
                if constexpr(Incremental)
                {
                    using Window = ck_tile::remove_cvref_t<decltype(lds_read_window)>;
                    using Output = decltype(ck_tile::load_tile_transpose(lds_read_window));
                    Output result;
                    ck_tile::static_for<0, Window::NumAccessPerCoord, 1>{}([&](auto access) {
                        ck_tile::FmhaD192TransposeLoad::LoadAccess<decltype(access)::value>(
                            result, lds_read_window);
                    });
                    return result;
                }
                else
                {
                    return ck_tile::load_tile_transpose(lds_read_window);
                }
            }();
            auto output_window = ck_tile::make_tile_window(
                output_view,
                ck_tile::make_tuple(ck_tile::number<kVCols>{}, ck_tile::number<kVReadRows>{}),
                {0, row_offset},
                transposed.get_tile_distribution());
            ck_tile::store_tile(output_window, transposed);
        });

        auto* dump = static_cast<std::uint8_t*>(args.dump);
        for(std::size_t i = threadIdx.x; i < kVFootprintBytes; i += blockDim.x)
        {
            dump[i] = static_cast<std::uint8_t>(arena[args.lds_offset + i]);
        }
    }
};

struct KIncrementalLoadKernel
{
    static constexpr ck_tile::index_t kBlockSize = ::kBlockSize;

    CK_TILE_DEVICE void operator()(ProbeArgs args) const
    {
        __shared__ char arena[kArenaBytes];

        const auto* input = static_cast<const DataType*>(args.input);
        auto input_view   = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            input,
            ck_tile::make_tuple(args.rows, args.valid_cols),
            ck_tile::make_tuple(args.input_stride, 1));
        auto input_window = ck_tile::make_tile_window(
            input_view,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kKTileCols>{}),
            {0, 0},
            D192Policy::MakeKDramTileDistribution<D192PolicyProblem>());

        auto* lds               = reinterpret_cast<DataType*>(arena);
        constexpr auto lds_desc = D192Policy::MakeKLdsWriteBlockDescriptor<D192PolicyProblem>();
        auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(lds, lds_desc);
        auto lds_write_window = ck_tile::make_tile_window(
            lds_view,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kKTileCols>{}),
            {0, 0});

        ck_tile::TDMConfig config{};
        ck_tile::load_tile_tdm(config, lds_write_window, input_window);
        ck_tile::s_wait_tensorcnt_barrier<0>();

        auto lds_read_window = ck_tile::make_tile_window(
            lds_view,
            ck_tile::make_tuple(ck_tile::number<kVReadRows>{}, ck_tile::number<kKValidCols>{}),
            {0, 0},
            D192Policy::MakeKSuRegTileDistribution<D192PolicyProblem>());
        using Output = decltype(ck_tile::load_tile(lds_read_window));
        Output loaded;
        ck_tile::static_for<0, 24, 1>{}([&](auto instruction) {
            ck_tile::FmhaD192Load::LoadInstruction<decltype(instruction)::value>(loaded,
                                                                                 lds_read_window);
        });

        auto* output     = static_cast<DataType*>(args.transpose);
        auto output_view = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            output,
            ck_tile::make_tuple(ck_tile::number<kVReadRows>{}, ck_tile::number<kKValidCols>{}),
            ck_tile::make_tuple(ck_tile::number<kKValidCols>{}, ck_tile::number<1>{}));
        auto output_window = ck_tile::make_tile_window(
            output_view,
            ck_tile::make_tuple(ck_tile::number<kVReadRows>{}, ck_tile::number<kKValidCols>{}),
            {0, 0},
            loaded.get_tile_distribution());
        ck_tile::store_tile(output_window, loaded);
    }
};

template <bool IsK>
struct SuTdmDumpKernel
{
    static constexpr ck_tile::index_t kBlockSize = ::kBlockSize;

    CK_TILE_DEVICE void operator()(ProbeArgs args) const
    {
        __shared__ char arena[kArenaBytes];
        constexpr ck_tile::index_t valid_cols = IsK ? kKValidCols : kVCols;
        constexpr ck_tile::index_t tile_cols  = IsK ? kKTileCols : kVCols;
        constexpr ck_tile::index_t lds_stride = IsK ? kKStride : kVStride;
        constexpr std::size_t footprint_bytes = kRows * lds_stride * sizeof(DataType);

        for(std::size_t i = threadIdx.x; i < footprint_bytes; i += blockDim.x)
        {
            arena[i] = static_cast<char>(kUntouchedCanary);
        }
        __syncthreads();

        const auto* input = static_cast<const DataType*>(args.input);
        auto input_view   = ck_tile::make_naive_tensor_view<ck_tile::address_space_enum::global>(
            input,
            ck_tile::make_tuple(args.rows, args.valid_cols),
            ck_tile::make_tuple(args.input_stride, 1));
        auto input_window = [&]() {
            if constexpr(IsK)
            {
                return ck_tile::make_tile_window(
                    input_view,
                    ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kKTileCols>{}),
                    {0, 0},
                    D192Policy::MakeKDramTileDistribution<D192PolicyProblem>());
            }
            else
            {
                return ck_tile::make_tile_window(
                    input_view,
                    ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<kVCols>{}),
                    {0, 0},
                    D192Policy::MakeVDramTileDistribution<D192PolicyProblem>());
            }
        }();

        auto* lds               = reinterpret_cast<DataType*>(arena);
        constexpr auto lds_desc = []() {
            if constexpr(IsK)
            {
                return D192Policy::MakeKLdsWriteBlockDescriptor<D192PolicyProblem>();
            }
            else
            {
                return D192Policy::MakeVLdsWriteBlockDescriptor<D192PolicyProblem>();
            }
        }();
        auto lds_view = ck_tile::make_tensor_view<ck_tile::address_space_enum::lds>(lds, lds_desc);

        ck_tile::TDMConfig config{};
        if constexpr(!IsK)
        {
            config.pad_enable              = true;
            config.pad_config.pad_interval = 5;
            config.pad_config.pad_amount   = 7;
        }

        auto lds_window = ck_tile::make_tile_window(
            lds_view,
            ck_tile::make_tuple(ck_tile::number<kRows>{}, ck_tile::number<tile_cols>{}),
            {0, 0});
        ck_tile::static_for<0, 4, 1>{}([&](auto su) {
            if constexpr(IsK)
            {
                D192Policy::LoadKSuTdm<decltype(su)::value, D192PolicyProblem>(
                    config, lds_window, input_window);
            }
            else
            {
                D192Policy::LoadVSuTdm<decltype(su)::value, D192PolicyProblem>(
                    config, lds_window, input_window);
            }
        });
        ck_tile::s_wait_tensorcnt_barrier<0>();

        auto* dump = static_cast<std::uint8_t*>(args.dump);
        for(std::size_t i = threadIdx.x; i < footprint_bytes; i += blockDim.x)
        {
            dump[i] = static_cast<std::uint8_t>(arena[i]);
        }
    }
};

struct ArenaKernel
{
    static constexpr ck_tile::index_t kBlockSize = ::kBlockSize;

    [[maybe_unused]] CK_TILE_DEVICE void operator()(void* output) const
    {
        __shared__ char arena[kArenaBytes];
        auto* bytes = reinterpret_cast<std::uint8_t*>(arena);

        for(std::size_t i = threadIdx.x; i < kArenaBytes; i += blockDim.x)
        {
            bytes[i] = kUntouchedCanary;
        }
        __syncthreads();

        for(ck_tile::index_t region = 0; region < 4; ++region)
        {
            for(std::size_t i = threadIdx.x; i < kRegionSizes[region]; i += blockDim.x)
            {
                bytes[kRegionBases[region] + i] = kRegionCanaries[region];
            }
        }
        __syncthreads();

        auto* out = static_cast<std::uint8_t*>(output);
        for(std::size_t i = threadIdx.x; i < kArenaBytes; i += blockDim.x)
        {
            out[i] = bytes[i];
        }
    }
};

void check_hip(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
    }
}

std::uint16_t source_pattern(std::size_t row, std::size_t col)
{
    const auto value = static_cast<std::uint16_t>((0x1234u + row * 257u + col * 13u) & 0x7fffu);
    return value == 0 ? 1 : value;
}

std::vector<std::uint16_t> make_source(std::size_t rows, std::size_t cols)
{
    std::vector<std::uint16_t> source(rows * cols);
    for(std::size_t row = 0; row < rows; ++row)
    {
        for(std::size_t col = 0; col < cols; ++col)
        {
            source[row * cols + col] = source_pattern(row, col);
        }
    }
    return source;
}

void write_dump(const std::string& dump_dir,
                const std::string& name,
                const std::vector<std::uint8_t>& bytes)
{
    if(dump_dir.empty())
    {
        return;
    }

    const std::string path = dump_dir + "/" + name + ".bin";
    std::ofstream file(path, std::ios::binary);
    if(!file)
    {
        throw std::runtime_error("cannot open dump file: " + path);
    }
    file.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

template <typename Kernel>
void launch_probe(const ProbeArgs& args)
{
    ck_tile::stream_config stream_config{nullptr, false, 0, 0, 1};
    ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<1>(Kernel{}, dim3(1), dim3(Kernel::kBlockSize), 0, args));
    check_hip(hipGetLastError(), "probe launch");
    check_hip(hipDeviceSynchronize(), "probe synchronize");
}

template <typename Kernel>
std::array<std::uint32_t, kDescriptorWordCount> capture_descriptor(const void* input,
                                                                   ck_tile::index_t input_stride)
{
    ck_tile::DeviceMem output(kDescriptorWordCount * sizeof(std::uint32_t));
    const DescriptorArgs args{input, output.GetDeviceBuffer(), input_stride};
    ck_tile::stream_config stream_config{nullptr, false, 0, 0, 1};
    ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<1>(Kernel{}, dim3(1), dim3(Kernel::kBlockSize), 0, args));
    check_hip(hipGetLastError(), "descriptor capture launch");
    check_hip(hipDeviceSynchronize(), "descriptor capture synchronize");

    std::array<std::uint32_t, kDescriptorWordCount> words{};
    output.FromDevice(words.data());
    return words;
}

void emit_descriptor_json(const std::string& name,
                          const char* descriptor_source,
                          const std::array<std::uint32_t, kDescriptorWordCount>& words)
{
    const auto emit_group = [&](std::size_t begin, std::size_t end) {
        std::cout << '[';
        for(std::size_t i = begin; i < end; ++i)
        {
            std::cout << (i == begin ? "" : ",") << words[i];
        }
        std::cout << ']';
    };

    const auto tensor_dim0        = (words[5] >> 16) | ((words[6] & 0xffffu) << 16);
    const auto tensor_dim1        = (words[6] >> 16) | ((words[7] & 0xffffu) << 16);
    const auto tile_dim0          = words[7] >> 16;
    const auto tile_dim1          = words[8] & 0xffffu;
    const auto tensor_dim0_stride = static_cast<std::uint64_t>(words[9]) |
                                    (static_cast<std::uint64_t>(words[10] & 0xffffu) << 32);
    const auto tensor_dim1_stride =
        static_cast<std::uint64_t>(words[10] >> 16) | (static_cast<std::uint64_t>(words[11]) << 16);

    std::cout << "{\"case\":\"" << name << "\",\"descriptor_source\":\"" << descriptor_source
              << "\",\"descriptor_words\":{\"group0\":";
    emit_group(0, 4);
    std::cout << ",\"group1\":";
    emit_group(4, 12);
    std::cout << ",\"group2\":";
    emit_group(12, 16);
    std::cout << ",\"group3\":";
    emit_group(16, 20);
    std::cout << ",\"group4\":";
    emit_group(20, 28);
    std::cout << "},\"decoded\":{\"tensor_dim0\":" << tensor_dim0
              << ",\"tensor_dim1\":" << tensor_dim1 << ",\"tile_dim0\":" << tile_dim0
              << ",\"tile_dim1\":" << tile_dim1 << ",\"tensor_dim0_stride\":" << tensor_dim0_stride
              << ",\"tensor_dim1_stride\":" << tensor_dim1_stride << "}}\n"
              << std::flush;
}

struct ProbeResult
{
    std::string name;
    std::size_t source_base         = 0;
    std::size_t destination_base    = 0;
    std::size_t global_valid_extent = 0;
    std::size_t tdm_tile_extent     = 0;
    std::size_t lds_row_stride      = 0;
    std::size_t pad_interval        = 0;
    std::size_t pad_amount          = 0;
    std::size_t mismatch_count      = 0;
    std::size_t transfers           = 1;
    bool byte_validation            = true;
};

void emit_json(const ProbeResult& result)
{
    std::cout << "{\"case\":\"" << result.name << "\",\"seed\":" << kSeed
              << ",\"source_base\":" << result.source_base
              << ",\"destination_base\":" << result.destination_base
              << ",\"global_valid_extent\":" << result.global_valid_extent
              << ",\"tdm_tile_extent\":" << result.tdm_tile_extent
              << ",\"lds_row_stride\":" << result.lds_row_stride
              << ",\"pad_interval\":" << result.pad_interval
              << ",\"pad_amount\":" << result.pad_amount
              << ",\"byte_mismatch_count\":" << result.mismatch_count
              << ",\"transfers\":" << result.transfers
              << ",\"byte_validation\":" << (result.byte_validation ? "true" : "false")
              << ",\"result\":\""
              << (!result.byte_validation      ? "execution_pass"
                  : result.mismatch_count == 0 ? "pass"
                                               : "fail")
              << "\"}\n";
}

ProbeResult run_k_probe(std::size_t lds_offset,
                        ck_tile::index_t valid_cols,
                        const std::string& name,
                        const std::string& dump_dir,
                        bool split_su = false)
{
    const auto source = make_source(kRows, valid_cols);
    ck_tile::DeviceMem input(source.size() * sizeof(std::uint16_t));
    ck_tile::DeviceMem dump(kKFootprintBytes);
    input.ToDevice(source.data());

    const ProbeArgs args{input.GetDeviceBuffer(),
                         dump.GetDeviceBuffer(),
                         nullptr,
                         kRows,
                         valid_cols,
                         valid_cols,
                         static_cast<ck_tile::index_t>(lds_offset)};
    if(split_su)
    {
        launch_probe<SuTdmDumpKernel<true>>(args);
    }
    else
    {
        using Kernel = TdmDumpKernel<kRows, kKValidCols, kKTileCols, kKStride, 4, false>;
        launch_probe<Kernel>(args);
    }

    std::vector<std::uint8_t> bytes(kKFootprintBytes);
    dump.FromDevice(bytes.data());
    std::size_t mismatches = 0;
    for(std::size_t row = 0; row < kRows; ++row)
    {
        for(std::size_t col = 0; col < kKTileCols; ++col)
        {
            const std::uint16_t actual = static_cast<std::uint16_t>(
                bytes[(row * kKStride + col) * 2] |
                (static_cast<std::uint16_t>(bytes[(row * kKStride + col) * 2 + 1]) << 8));
            const std::uint16_t expected =
                col < static_cast<std::size_t>(valid_cols) ? source[row * valid_cols + col] : 0;
            mismatches += actual != expected;
        }
    }
    write_dump(dump_dir, name, bytes);
    return {name,
            0,
            lds_offset,
            static_cast<std::size_t>(valid_cols),
            kKTileCols,
            kKStride,
            0,
            0,
            mismatches};
}

ProbeResult run_k_incremental_load_probe(const std::string& name)
{
    const auto source = make_source(kRows, kKValidCols);
    ck_tile::DeviceMem input(source.size() * sizeof(std::uint16_t));
    ck_tile::DeviceMem output(kVReadRows * kKValidCols * sizeof(std::uint16_t));
    input.ToDevice(source.data());

    const ProbeArgs args{input.GetDeviceBuffer(),
                         nullptr,
                         output.GetDeviceBuffer(),
                         kRows,
                         kKValidCols,
                         kKValidCols,
                         0};
    launch_probe<KIncrementalLoadKernel>(args);

    std::vector<std::uint16_t> loaded(kVReadRows * kKValidCols);
    output.FromDevice(loaded.data());
    std::size_t mismatches = 0;
    for(std::size_t row = 0; row < kVReadRows; ++row)
    {
        for(std::size_t col = 0; col < kKValidCols; ++col)
        {
            mismatches += loaded[row * kKValidCols + col] != source[row * kKValidCols + col];
        }
    }

    return {name, 0, 0, kKValidCols, kKValidCols, kKStride, 0, 0, mismatches};
}

template <ck_tile::index_t Rows,
          ck_tile::index_t SecondDimStride,
          bool UsePublicPath,
          bool FullArenaInit,
          bool CopyDump>
ProbeResult run_descriptor_probe(const std::string& name, const std::string& dump_dir)
{
    constexpr ck_tile::index_t valid_cols = kKValidCols;
    constexpr ck_tile::index_t tile_cols  = kKTileCols;
    constexpr std::size_t footprint_bytes = Rows * tile_cols * sizeof(DataType);

    const auto source = make_source(Rows, valid_cols);
    ck_tile::DeviceMem input(source.size() * sizeof(std::uint16_t));
    ck_tile::DeviceMem dump(footprint_bytes);
    input.ToDevice(source.data());

    using CaptureKernel = RawDescriptorCaptureKernel<Rows, valid_cols, tile_cols, SecondDimStride>;
    const auto words    = capture_descriptor<CaptureKernel>(input.GetDeviceBuffer(), valid_cols);
    emit_descriptor_json(name, UsePublicPath ? "raw_equivalent" : "raw_executed", words);

    const ProbeArgs args{
        input.GetDeviceBuffer(), dump.GetDeviceBuffer(), nullptr, Rows, valid_cols, valid_cols, 0};
    if constexpr(UsePublicPath)
    {
        using Kernel = TdmDumpKernel<Rows,
                                     valid_cols,
                                     tile_cols,
                                     tile_cols,
                                     1,
                                     false,
                                     FullArenaInit,
                                     CopyDump>;
        launch_probe<Kernel>(args);
    }
    else
    {
        static_assert(CopyDump, "raw descriptor probes require byte validation");
        using Kernel =
            RawTdmDumpKernel<Rows, valid_cols, tile_cols, SecondDimStride, FullArenaInit>;
        launch_probe<Kernel>(args);
    }

    if constexpr(CopyDump)
    {
        std::vector<std::uint8_t> bytes(footprint_bytes);
        dump.FromDevice(bytes.data());
        std::size_t mismatches = 0;
        for(std::size_t row = 0; row < Rows; ++row)
        {
            for(std::size_t col = 0; col < tile_cols; ++col)
            {
                const std::uint16_t actual = static_cast<std::uint16_t>(
                    bytes[(row * tile_cols + col) * 2] |
                    (static_cast<std::uint16_t>(bytes[(row * tile_cols + col) * 2 + 1]) << 8));
                const std::uint16_t expected =
                    col < valid_cols ? source[row * valid_cols + col] : 0;
                mismatches += actual != expected;
            }
        }
        write_dump(dump_dir, name, bytes);
        return {name, 0, 0, valid_cols, tile_cols, tile_cols, 0, 0, mismatches};
    }
    else
    {
        ProbeResult result{name, 0, 0, valid_cols, tile_cols, tile_cols, 0, 0, 0};
        result.byte_validation = false;
        return result;
    }
}

ProbeResult run_guard_probe(const std::string& dump_dir)
{
    int device = 0;
    check_hip(hipGetDevice(&device), "hipGetDevice");

    hipMemAllocationProp prop{};
    prop.type          = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id   = device;

    std::size_t granularity = 0;
    check_hip(
        hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum),
        "hipMemGetAllocationGranularity");

    void* reservation = nullptr;
    check_hip(hipMemAddressReserve(&reservation, 2 * granularity, granularity, nullptr, 0),
              "hipMemAddressReserve");

    hipMemGenericAllocationHandle_t handle{};
    check_hip(hipMemCreate(&handle, granularity, &prop, 0), "hipMemCreate");
    check_hip(hipMemMap(reservation, granularity, 0, handle, 0), "hipMemMap");

    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags    = hipMemAccessFlagsProtReadWrite;
    check_hip(hipMemSetAccess(reservation, granularity, &access, 1), "hipMemSetAccess");

    constexpr std::size_t source_bytes = kKValidCols * sizeof(DataType);
    auto* guarded_input = static_cast<std::uint8_t*>(reservation) + granularity - source_bytes;
    const auto source   = make_source(1, kKValidCols);
    check_hip(hipMemcpy(guarded_input, source.data(), source_bytes, hipMemcpyHostToDevice),
              "guard source copy");

    ck_tile::DeviceMem dump(kKTileCols * sizeof(DataType));
    const ProbeArgs args{
        guarded_input, dump.GetDeviceBuffer(), nullptr, 1, kKValidCols, kKValidCols, 0};
    using Kernel = TdmDumpKernel<1, kKValidCols, kKTileCols, kKStride, 1, false>;
    launch_probe<Kernel>(args);

    std::vector<std::uint8_t> bytes(kKTileCols * sizeof(DataType));
    dump.FromDevice(bytes.data());
    std::size_t mismatches = 0;
    for(std::size_t col = 0; col < kKTileCols; ++col)
    {
        const std::uint16_t actual = static_cast<std::uint16_t>(
            bytes[col * 2] | (static_cast<std::uint16_t>(bytes[col * 2 + 1]) << 8));
        const std::uint16_t expected = col < kKValidCols ? source[col] : 0;
        mismatches += actual != expected;
    }
    write_dump(dump_dir, "k_guard_page", bytes);

    check_hip(hipMemUnmap(reservation, granularity), "hipMemUnmap");
    check_hip(hipMemRelease(handle), "hipMemRelease");
    check_hip(hipMemAddressFree(reservation, 2 * granularity), "hipMemAddressFree");

    return {"k_guard_page",
            granularity - source_bytes,
            0,
            kKValidCols,
            kKTileCols,
            kKStride,
            0,
            0,
            mismatches};
}

ProbeResult run_v_probe(std::size_t lds_offset,
                        const std::string& name,
                        const std::string& dump_dir,
                        bool check_transpose,
                        bool incremental = false,
                        bool split_su    = false)
{
    const auto source = make_source(kRows, kVCols);
    ck_tile::DeviceMem input(source.size() * sizeof(std::uint16_t));
    ck_tile::DeviceMem dump(kVFootprintBytes);
    ck_tile::DeviceMem transpose(check_transpose ? source.size() * sizeof(std::uint16_t) : 0);
    input.ToDevice(source.data());

    const ProbeArgs args{input.GetDeviceBuffer(),
                         dump.GetDeviceBuffer(),
                         check_transpose ? transpose.GetDeviceBuffer() : nullptr,
                         kRows,
                         kVCols,
                         kVCols,
                         static_cast<ck_tile::index_t>(lds_offset)};
    if(check_transpose)
    {
        if(incremental)
        {
            launch_probe<VPadTransposeKernel<true>>(args);
        }
        else
        {
            launch_probe<VPadTransposeKernel<false>>(args);
        }
    }
    else
    {
        if(split_su)
        {
            launch_probe<SuTdmDumpKernel<false>>(args);
        }
        else
        {
            using Kernel = TdmDumpKernel<kRows, kVCols, kVCols, kVStride, 4, true>;
            launch_probe<Kernel>(args);
        }
    }

    std::vector<std::uint8_t> bytes(kVFootprintBytes);
    dump.FromDevice(bytes.data());
    std::size_t mismatches   = 0;
    const auto* source_bytes = reinterpret_cast<const std::uint8_t*>(source.data());
    for(std::size_t row = 0; row < kRows; ++row)
    {
        const std::size_t output_row = row * kVStride * sizeof(DataType);
        const std::size_t input_row  = row * kVCols * sizeof(DataType);
        for(std::size_t byte = 0; byte < kVCols * sizeof(DataType); ++byte)
        {
            mismatches += bytes[output_row + byte] != source_bytes[input_row + byte];
        }
        for(std::size_t byte = kVCols * sizeof(DataType); byte < kVStride * sizeof(DataType);
            ++byte)
        {
            mismatches += bytes[output_row + byte] != kUntouchedCanary;
        }
    }

    if(check_transpose)
    {
        std::vector<std::uint16_t> transposed(source.size());
        transpose.FromDevice(transposed.data());
        for(std::size_t row = 0; row < kRows; ++row)
        {
            for(std::size_t col = 0; col < kVCols; ++col)
            {
                mismatches += transposed[col * kRows + row] != source[row * kVCols + col];
            }
        }
    }

    write_dump(dump_dir, name, bytes);
    return {name, 0, lds_offset, kVCols, kVCols, kVStride, 5, 7, mismatches};
}

ProbeResult run_arena_probe(const std::string& dump_dir)
{
    ck_tile::DeviceMem output(kArenaBytes);
    ck_tile::stream_config stream_config{nullptr, false, 0, 0, 1};
    ck_tile::launch_kernel(
        stream_config,
        ck_tile::make_kernel<1>(
            ArenaKernel{}, dim3(1), dim3(ArenaKernel::kBlockSize), 0, output.GetDeviceBuffer()));
    check_hip(hipGetLastError(), "arena launch");
    check_hip(hipDeviceSynchronize(), "arena synchronize");

    std::vector<std::uint8_t> bytes(kArenaBytes);
    output.FromDevice(bytes.data());
    std::size_t mismatches = 0;
    for(std::size_t offset = 0; offset < kArenaBytes; ++offset)
    {
        std::uint8_t expected = kUntouchedCanary;
        for(std::size_t region = 0; region < kRegionBases.size(); ++region)
        {
            if(offset >= kRegionBases[region] &&
               offset < kRegionBases[region] + kRegionSizes[region])
            {
                expected = kRegionCanaries[region];
                break;
            }
        }
        mismatches += bytes[offset] != expected;
    }
    write_dump(dump_dir, "arena_0x39000", bytes);
    return {"arena_0x39000", 0, 0, 0, 0, 0, 0, 0, mismatches, 4};
}

ProbeResult run_boundary_case(const std::string& name, const std::string& dump_dir)
{
    const std::int64_t delta  = name == "segment_end_before" ? -16
                                : name == "segment_end_at"   ? 0
                                                             : 16;
    const auto checked_offset = [delta](std::size_t boundary, std::size_t footprint) {
        constexpr auto max_signed = std::numeric_limits<std::int64_t>::max();
        if(boundary > static_cast<std::size_t>(max_signed) ||
           footprint > static_cast<std::size_t>(max_signed))
        {
            throw std::overflow_error("boundary offset input exceeds int64_t");
        }

        const auto signed_boundary  = static_cast<std::int64_t>(boundary);
        const auto signed_footprint = static_cast<std::int64_t>(footprint);
        if(signed_boundary < signed_footprint)
        {
            throw std::invalid_argument("boundary is smaller than the probe footprint");
        }

        const auto base = signed_boundary - signed_footprint;
        if((delta < 0 && base < -delta) || (delta > 0 && base > max_signed - delta))
        {
            throw std::overflow_error("boundary offset adjustment is out of range");
        }

        const auto offset = base + delta;
        if(offset < 0)
        {
            throw std::invalid_argument("boundary offset must be nonnegative");
        }
        return static_cast<std::size_t>(offset);
    };

    std::size_t mismatches = 0;
    for(const std::size_t boundary : {kSegmentBytes, 2 * kSegmentBytes})
    {
        const std::size_t k_offset = checked_offset(boundary, kKFootprintBytes);
        const std::size_t v_offset = checked_offset(boundary, kVFootprintBytes);
        const auto k_result        = run_k_probe(k_offset, kKTileCols, name + "_k", dump_dir);
        const auto v_result        = run_v_probe(v_offset, name + "_v", dump_dir, false);
        mismatches += k_result.mismatch_count + v_result.mismatch_count;
    }

    return {name, 0, 0, 0, 0, 0, 0, 0, mismatches, 4};
}

std::string get_option(int argc, char** argv, const std::string& option)
{
    for(int i = 1; i + 1 < argc; ++i)
    {
        if(argv[i] == option)
        {
            return argv[i + 1];
        }
    }
    return {};
}

bool run_case(const std::string& name, const std::string& dump_dir)
{
    ProbeResult result;
    if(name == "raw_a_stride0")
    {
        result = run_descriptor_probe<1, 0, false, false, true>(name, dump_dir);
    }
    else if(name == "raw_b_stride1")
    {
        result = run_descriptor_probe<1, 1, false, false, true>(name, dump_dir);
    }
    else if(name == "public_c_stride1")
    {
        result = run_descriptor_probe<1, 1, true, true, true>(name, dump_dir);
    }
    else if(name == "raw_d_rows8_stride0")
    {
        result = run_descriptor_probe<8, 0, false, false, true>(name, dump_dir);
    }
    else if(name == "raw_e_full_arena_stride1")
    {
        result = run_descriptor_probe<1, 1, false, true, true>(name, dump_dir);
    }
    else if(name == "public_f_issue_wait_only")
    {
        result = run_descriptor_probe<1, 1, true, false, false>(name, dump_dir);
    }
    else if(name == "public_g_min_init_copy")
    {
        result = run_descriptor_probe<1, 1, true, false, true>(name, dump_dir);
    }
    else if(name == "k_valid192_tile200")
    {
        result = run_k_probe(0, kKValidCols, name, dump_dir);
    }
    else if(name == "k_guard_page")
    {
        result = run_guard_probe(dump_dir);
    }
    else if(name == "k_incremental_load")
    {
        result = run_k_incremental_load_probe(name);
    }
    else if(name == "k_su_tdm")
    {
        result = run_k_probe(0, kKValidCols, name, dump_dir, true);
    }
    else if(name == "v_pad_and_transpose")
    {
        result = run_v_probe(0, name, dump_dir, true);
    }
    else if(name == "v_pad_and_transpose_incremental")
    {
        result = run_v_probe(0, name, dump_dir, true, true);
    }
    else if(name == "v_su_tdm")
    {
        result = run_v_probe(0, name, dump_dir, false, false, true);
    }
    else if(name == "arena_0x39000")
    {
        result = run_arena_probe(dump_dir);
    }
    else if(name == "segment_end_before" || name == "segment_end_at" ||
            name == "segment_cross_after")
    {
        result = run_boundary_case(name, dump_dir);
    }
    else
    {
        throw std::runtime_error("unknown case: " + name);
    }

    emit_json(result);
    return result.mismatch_count == 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        int device = 0;
        check_hip(hipGetDevice(&device), "hipGetDevice");
        hipDeviceProp_t props{};
        check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");
        if(std::string(props.gcnArchName).find("gfx125") == std::string::npos)
        {
            std::cerr << "test_tdm_fmha_d192_layout requires gfx125, got " << props.gcnArchName
                      << '\n';
            return 77;
        }

        const std::string requested_case        = get_option(argc, argv, "--case");
        const std::string dump_dir              = get_option(argc, argv, "--dump-dir");
        const std::array<std::string, 18> cases = {"raw_a_stride0",
                                                   "raw_b_stride1",
                                                   "public_c_stride1",
                                                   "raw_d_rows8_stride0",
                                                   "raw_e_full_arena_stride1",
                                                   "public_f_issue_wait_only",
                                                   "public_g_min_init_copy",
                                                   "k_valid192_tile200",
                                                   "k_guard_page",
                                                   "k_incremental_load",
                                                   "k_su_tdm",
                                                   "v_pad_and_transpose",
                                                   "v_pad_and_transpose_incremental",
                                                   "v_su_tdm",
                                                   "arena_0x39000",
                                                   "segment_end_before",
                                                   "segment_end_at",
                                                   "segment_cross_after"};

        if(!requested_case.empty())
        {
            return run_case(requested_case, dump_dir) ? 0 : 1;
        }

        bool passed = true;
        for(const auto& name : cases)
        {
            passed &= run_case(name, dump_dir);
        }
        return passed ? 0 : 1;
    }
    catch(const std::exception& error)
    {
        std::cerr << "{\"result\":\"error\",\"message\":\"" << error.what() << "\"}\n";
        return 2;
    }
}
