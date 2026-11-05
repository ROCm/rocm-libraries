// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "test_gemm_streamk_types.hpp"
#include "test_gemm_streamk_util.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v3.hpp"
#include "gtest/gtest.h"

namespace {

using Row  = ck_tile::tensor_layout::gemm::RowMajor;
using Col  = ck_tile::tensor_layout::gemm::ColumnMajor;
using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;
using F8   = ck_tile::fp8_t;
using BF8  = ck_tile::bf8_t;

using ck_tile::StreamKReductionStrategy::Atomic;
using ck_tile::StreamKReductionStrategy::Linear;
using ck_tile::StreamKReductionStrategy::Tree;

constexpr bool NonPersistent    = false;
constexpr bool Persistent       = true;
constexpr bool CShuffleEpilogue = false;
constexpr bool TdmEpilogue      = true;
constexpr bool Unpadded         = false;
constexpr bool Padded           = true;

// GemmPipelineAgBgCrCompTDMV3 supports 4-wave (2x2x1) and 16-wave (4x4x1) block
// configurations, with 16x16x32 WMMA warp tiles for 16-bit inputs and 32x32x128
// WMMA warp tiles for 8-bit inputs on gfx1250.
template <ck_tile::index_t MPerBlock,
          ck_tile::index_t NPerBlock,
          ck_tile::index_t KPerBlock,
          ck_tile::index_t WarpTileM   = 16,
          ck_tile::index_t WarpTileN   = 16,
          ck_tile::index_t WarpTileK   = 32,
          ck_tile::index_t BlockWarpsM = 2,
          ck_tile::index_t BlockWarpsN = 2>
struct GemmConfigTdmV3
{
    using BlockTile  = ck_tile::sequence<MPerBlock, NPerBlock, KPerBlock>;
    using BlockWarps = ck_tile::sequence<BlockWarpsM, BlockWarpsN, 1>;
    using WarpTile   = ck_tile::sequence<WarpTileM, WarpTileN, WarpTileK>;
};

// 4-wave (2x2x1) 16-bit tile configurations (16x16x32 warp tile)
using GemmConfigTdmV3_4W_64x64x64    = GemmConfigTdmV3<64, 64, 64>;
using GemmConfigTdmV3_4W_128x128x64  = GemmConfigTdmV3<128, 128, 64>;
using GemmConfigTdmV3_4W_128x128x128 = GemmConfigTdmV3<128, 128, 128>;

// 16-wave (4x4x1) 16-bit tile configuration (16x16x32 warp tile)
using GemmConfigTdmV3_16W_256x256x64 = GemmConfigTdmV3<256, 256, 64, 16, 16, 32, 4, 4>;

// 4-wave (2x2x1) 8-bit tile configurations (32x32x128 warp tile)
using GemmConfigTdmV3_4W_128x128x128_F8 = GemmConfigTdmV3<128, 128, 128, 32, 32, 128>;
using GemmConfigTdmV3_4W_128x128x256_F8 = GemmConfigTdmV3<128, 128, 256, 32, 32, 128>;

// 16-wave (4x4x1) 8-bit tile configurations (32x32x128 warp tile)
using GemmConfigTdmV3_16W_256x256x128_F8 = GemmConfigTdmV3<256, 256, 128, 32, 32, 128, 4, 4>;
using GemmConfigTdmV3_16W_256x256x256_F8 = GemmConfigTdmV3<256, 256, 256, 32, 32, 128, 4, 4>;

template <typename ALayoutType,
          typename BLayoutType,
          typename CLayoutType,
          typename ADataTypeT,
          typename BDataTypeT,
          typename AccDataTypeT,
          typename CDataTypeT,
          typename GemmConfigType,
          ck_tile::StreamKReductionStrategy ReductionStrategyV,
          bool PersistentV,
          bool UseTdmChainedEpilogueV = false,
          bool PadMNKV                = false>
struct StreamKTdmV3Config
{
    using ALayout     = ALayoutType;
    using BLayout     = BLayoutType;
    using CLayout     = CLayoutType;
    using ADataType   = ADataTypeT;
    using BDataType   = BDataTypeT;
    using AccDataType = AccDataTypeT;
    using CDataType   = CDataTypeT;
    using GemmConfig  = GemmConfigType;

    template <typename Problem>
    using PipelineType = ck_tile::GemmPipelineAgBgCrCompTDMV3<Problem>;

    static constexpr auto ReductionStrategy     = ReductionStrategyV;
    static constexpr bool Persistent            = PersistentV;
    static constexpr bool UseTdmChainedEpilogue = UseTdmChainedEpilogueV;
    static constexpr bool PadM                  = PadMNKV;
    static constexpr bool PadN                  = PadMNKV;
    static constexpr bool PadK                  = PadMNKV;
};

template <typename Config, bool UseTdmChainedEpilogue = Config::UseTdmChainedEpilogue>
struct BuildStreamKTdmV3Kernel
{
    using GemmShape           = ck_tile::TileGemmShape<typename Config::GemmConfig::BlockTile,
                                                       typename Config::GemmConfig::BlockWarps,
                                                       typename Config::GemmConfig::WarpTile>;
    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<Config::PadM,
                                                                 Config::PadN,
                                                                 Config::PadK,
                                                                 true,
                                                                 typename Config::ALayout,
                                                                 typename Config::BLayout,
                                                                 typename Config::CLayout,
                                                                 false,
                                                                 false,
                                                                 Config::Persistent>;
    using TilePartitioner =
        ck_tile::StreamKTilePartitioner<GemmShape, Config::ReductionStrategy, Config::Persistent>;
    using UniversalGemmProblem =
        ck_tile::UniversalGemmPipelineProblem<typename Config::ADataType,
                                              typename Config::BDataType,
                                              typename Config::AccDataType,
                                              GemmShape,
                                              GemmUniversalTraits,
                                              ck_tile::GemmPipelineScheduler::Intrawave>;
    using GemmPipeline = typename Config::template PipelineType<UniversalGemmProblem>;

    static constexpr ck_tile::index_t WarpTileM =
        Config::GemmConfig::WarpTile::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t WarpTileN =
        Config::GemmConfig::WarpTile::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t WarpTileK =
        Config::GemmConfig::WarpTile::at(ck_tile::number<2>{});

    using EpilogueProblem =
        ck_tile::CShuffleEpilogueProblem<typename Config::ADataType,
                                         typename Config::BDataType,
                                         ck_tile::tuple<>,
                                         typename Config::AccDataType,
                                         typename Config::CDataType,
                                         ck_tile::tuple<>,
                                         typename Config::CLayout,
                                         ck_tile::element_wise::PassThrough,
                                         TilePartitioner::MPerBlock,
                                         TilePartitioner::NPerBlock,
                                         Config::GemmConfig::BlockWarps::at(ck_tile::number<0>{}),
                                         Config::GemmConfig::BlockWarps::at(ck_tile::number<1>{}),
                                         WarpTileM,
                                         WarpTileN,
                                         WarpTileK,
                                         UniversalGemmProblem::TransposeC,
                                         1,
                                         false,
                                         1,
                                         1,
                                         /*DoubleSmemBuffer*/ true>;

    using EpiloguePipeline = std::conditional_t<UseTdmChainedEpilogue,
                                                ck_tile::TdmEpilogue<EpilogueProblem>,
                                                ck_tile::CShuffleEpilogue<EpilogueProblem>>;

    using Kernel = ck_tile::StreamKKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
};

template <typename Layout>
ck_tile::index_t
resolve_stride(ck_tile::index_t rows, ck_tile::index_t cols, ck_tile::index_t stride = 0)
{
    return stride > 0 ? stride : (std::is_same_v<Layout, Row> ? cols : rows);
}

template <typename Config, bool UseTdmChainedEpilogue = Config::UseTdmChainedEpilogue>
void run_streamk_tdm_v3_case(ck_tile::index_t M,
                             ck_tile::index_t N,
                             ck_tile::index_t K,
                             ck_tile::index_t stride_A = 0,
                             ck_tile::index_t stride_B = 0,
                             ck_tile::index_t stride_C = 0,
                             int max_active_wgs        = 0)
{
    using ADataType   = typename Config::ADataType;
    using BDataType   = typename Config::BDataType;
    using AccDataType = typename Config::AccDataType;
    using CDataType   = typename Config::CDataType;
    using ALayout     = typename Config::ALayout;
    using BLayout     = typename Config::BLayout;
    using CLayout     = typename Config::CLayout;
    using Kernel      = typename BuildStreamKTdmV3Kernel<Config, UseTdmChainedEpilogue>::Kernel;

    stride_A = resolve_stride<ALayout>(M, K, stride_A);
    stride_B = resolve_stride<BLayout>(K, N, stride_B);
    stride_C = resolve_stride<CLayout>(M, N, stride_C);

    const auto make_desc = [](std::size_t rows, std::size_t cols, std::size_t stride, auto layout) {
        if constexpr(std::is_same_v<decltype(layout), Row>)
        {
            return ck_tile::HostTensorDescriptor({rows, cols}, {stride, std::size_t{1}});
        }
        else
        {
            return ck_tile::HostTensorDescriptor({rows, cols}, {std::size_t{1}, stride});
        }
    };

    ck_tile::HostTensor<ADataType> a_m_k(make_desc(M, K, stride_A, ALayout{}));
    ck_tile::HostTensor<BDataType> b_k_n(make_desc(K, N, stride_B, BLayout{}));
    ck_tile::HostTensor<CDataType> c_m_n_dev_result(make_desc(M, N, stride_C, CLayout{}));

    ck_tile::FillUniformDistributionIntegerValue<ADataType>{-5, 5, /*seed*/ 11939}(a_m_k);
    ck_tile::FillUniformDistributionIntegerValue<BDataType>{-5, 5, /*seed*/ 11940}(b_k_n);

    ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
    ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
    ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

    a_m_k_dev_buf.ToDevice(a_m_k.data());
    b_k_n_dev_buf.ToDevice(b_k_n.data());
    c_m_n_dev_buf.SetZero();
    c_m_n_dev_result.SetZero();

    ck_tile::StreamKHostArgs args{a_m_k_dev_buf.GetDeviceBuffer(),
                                  b_k_n_dev_buf.GetDeviceBuffer(),
                                  c_m_n_dev_buf.GetDeviceBuffer(),
                                  M,
                                  N,
                                  K,
                                  stride_A,
                                  stride_B,
                                  stride_C};

    auto kargs = max_active_wgs > 0 ? Kernel::MakeKernelArgs(args, max_active_wgs, 1)
                                    : Kernel::MakeKernelArgs(args);
    if(!Kernel::IsSupportedArgument(kargs))
    {
        throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
    }

    ck_tile::DeviceMem workspace_data(Kernel::GetWorkSpaceSize(kargs));
    workspace_data.SetZero();
    kargs.workspace_ptr = workspace_data.GetDeviceBuffer();

    dim3 grids  = Kernel::GridSize(kargs.tile_partitioner);
    dim3 blocks = Kernel::BlockSize();

    ck_tile::launch_kernel(ck_tile::stream_config{nullptr, false, 0, 0, 1},
                           ck_tile::make_kernel<1>(Kernel{}, grids, blocks, 0, kargs));

    c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());

    ck_tile::HostTensor<CDataType> c_m_n_dev_ref(make_desc(M, N, stride_C, CLayout{}));
    ck_tile::DeviceMem ref_c_m_n_dev_buf(c_m_n_dev_ref.get_element_space_size_in_bytes());
    ref_c_m_n_dev_buf.SetZero();

    ck_tile::
        reference_gemm_gpu<ADataType, BDataType, AccDataType, CDataType, ALayout, BLayout, CLayout>(
            static_cast<ADataType*>(a_m_k_dev_buf.GetDeviceBuffer()),
            static_cast<BDataType*>(b_k_n_dev_buf.GetDeviceBuffer()),
            static_cast<CDataType*>(ref_c_m_n_dev_buf.GetDeviceBuffer()),
            M,
            N,
            K,
            stride_A,
            stride_B,
            stride_C);
    ref_c_m_n_dev_buf.FromDevice(c_m_n_dev_ref.data());

    const float max_accumulated_value =
        *std::max_element(c_m_n_dev_ref.mData.begin(), c_m_n_dev_ref.mData.end());
    const auto rtol_atol = calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(
        K, kargs.tile_partitioner.estimate_num_wgs_per_tile(), max_accumulated_value);
    const bool pass = ck_tile::check_err(c_m_n_dev_result,
                                         c_m_n_dev_ref,
                                         "Error: Incorrect results!",
                                         rtol_atol.at(ck_tile::number<0>{}),
                                         rtol_atol.at(ck_tile::number<1>{}));
    EXPECT_TRUE(pass);
}

template <typename Config>
class TestCkTileStreamKTdmV3Gfx1250 : public ::testing::Test
{
    protected:
    static constexpr ck_tile::index_t TileM =
        Config::GemmConfig::BlockTile::at(ck_tile::number<0>{});
    static constexpr ck_tile::index_t TileN =
        Config::GemmConfig::BlockTile::at(ck_tile::number<1>{});
    static constexpr ck_tile::index_t TileK =
        Config::GemmConfig::BlockTile::at(ck_tile::number<2>{});

    void RunSplitCases()
    {
        // Single output tile (TileM x TileN) with 4 K iterations and max_active_wgs = 2:
        // forces a 2-CTA Stream-K split (2 K-tiles per CTA) and exercises cross-workgroup
        // accumulation for every tile shape (64x64, 128x128, 256x256).
        run_streamk_tdm_v3_case<Config>(TileM, TileN, 4 * TileK, 0, 0, 0, 2u);
#if !GFX1250_STREAMK_SMOKE
        // 6 output tiles (2 * TileM x 3 * TileN) with 4 K iterations per tile and
        // max_active_wgs = 4: 4 DP tiles (running hot loop + Even tail) plus 2 SK tiles
        // split across 4 CTAs.
        run_streamk_tdm_v3_case<Config>(2 * TileM, 3 * TileN, 4 * TileK, 0, 0, 0, 4u);
#endif
    }
};

// clang-format off
using StreamKTdmV3Configs = ::testing::Types<
#if GFX1250_STREAMK_SMOKE
    // Emulator-safe Atomic smoke coverage across 4-wave and 16-wave, 16-bit and 8-bit,
    // all 4 A/B layouts, Persistent/NonPersistent, and Padded/Unpadded.
    // Note: TdmEpilogue overwrites C and pairs with Linear/Tree only.
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64,      Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64,      Atomic, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_64x64x64,        Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64,      Atomic, Persistent,    CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64,     Atomic, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8,  Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8,  Atomic, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Atomic, Persistent,    CShuffleEpilogue, Unpadded>
#else
    // 4-wave (2x2x1) 64x64x64 16-bit across all 4 A/B layouts
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_64x64x64,   Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_64x64x64,   Linear, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_64x64x64,   Tree,   NonPersistent, TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_64x64x64,   Linear, Persistent,    TdmEpilogue,      Padded>,

    // 4-wave (2x2x1) 128x128x64 16-bit across all 4 A/B layouts, reductions, persistence, and epilogues
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Linear, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Tree,   Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Linear, Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Atomic, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Linear, NonPersistent, TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Tree,   Persistent,    CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Linear, Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Tree,   NonPersistent, CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x64, Atomic, Persistent,    CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Col, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Linear, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x64, Tree,   Persistent,    TdmEpilogue,      Unpadded>,

    // 4-wave (2x2x1) 128x128x128 16-bit across all 4 A/B layouts
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x128, Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x128, Linear, Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x128, Tree,   NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Col, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_4W_128x128x128, Linear, Persistent,    CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Col, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_4W_128x128x128, Tree,   Persistent,    TdmEpilogue,      Unpadded>,

    // 16-wave (4x4x1) 256x256x64 16-bit across all 4 A/B layouts
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64, Atomic, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_16W_256x256x64, Linear, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64, Tree,   Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Row, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_16W_256x256x64, Linear, Persistent,    CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Col, Row, Row, BF16, BF16, F32, BF16, GemmConfigTdmV3_16W_256x256x64, Linear, NonPersistent, TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Col, Col, Row, F16,  F16,  F32, F16,  GemmConfigTdmV3_16W_256x256x64, Tree,   Persistent,    CShuffleEpilogue, Unpadded>,

    // 4-wave (2x2x1) 8-bit (fp8/bf8 -> bf16, 32x32x128 warp tile, RCR)
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Linear, Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Linear, Persistent,    CShuffleEpilogue, Padded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x128_F8, Tree,   NonPersistent, TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8, Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8, Linear, Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_4W_128x128x256_F8, Tree,   Persistent,    CShuffleEpilogue, Unpadded>,

    // 16-wave (4x4x1) 8-bit (fp8/bf8 -> bf16, 32x32x128 warp tile, RCR)
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x128_F8, Atomic, NonPersistent, CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_16W_256x256x128_F8, Linear, Persistent,    TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Atomic, Persistent,    CShuffleEpilogue, Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, F8,   F8,   F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Linear, NonPersistent, TdmEpilogue,      Unpadded>,
    StreamKTdmV3Config<Row, Col, Row, BF8,  BF8,  F32, BF16, GemmConfigTdmV3_16W_256x256x256_F8, Tree,   Persistent,    CShuffleEpilogue, Unpadded>
#endif
>;
// clang-format on

TYPED_TEST_SUITE(TestCkTileStreamKTdmV3Gfx1250, StreamKTdmV3Configs);

TYPED_TEST(TestCkTileStreamKTdmV3Gfx1250, SplitAndMixedTileCases) { this->RunSplitCases(); }

// A single active workgroup on a 1-tile problem sets sk_ctas == 1 and dp_tiles == 0,
// covering the 1-CTA full-tile path in StreamKKernel::StreamKGemm (store_tile directly to C).
// Emulator-safe (1 workgroup, no cross-CTA wait), so it also runs in GFX1250_STREAMK_SMOKE.
TEST(StreamKTdmV3EdgeCases, OneCtaFullTile)
{
    using Config4W  = StreamKTdmV3Config<Row,
                                         Col,
                                         Row,
                                         BF16,
                                         BF16,
                                         F32,
                                         BF16,
                                         GemmConfigTdmV3_4W_128x128x64,
                                         Linear,
                                         NonPersistent,
                                         CShuffleEpilogue,
                                         Unpadded>;
    using Config16W = StreamKTdmV3Config<Row,
                                         Col,
                                         Row,
                                         BF16,
                                         BF16,
                                         F32,
                                         BF16,
                                         GemmConfigTdmV3_16W_256x256x64,
                                         Linear,
                                         Persistent,
                                         TdmEpilogue,
                                         Unpadded>;

    run_streamk_tdm_v3_case<Config4W>(128, 128, 256, 0, 0, 0, 1);
    run_streamk_tdm_v3_case<Config16W>(256, 256, 256, 0, 0, 0, 1);
}

#if !GFX1250_STREAMK_SMOKE
// Exercises all four CompTDMV3 K-loop dispatch paths per CTA in a 2-CTA split:
//   - num_loop == 1 (has_hot_loop = false, TailNumber::One):  K = 2 * 64 = 128
//   - num_loop == 2 (has_hot_loop = false, TailNumber::Two):  K = 4 * 64 = 256
//   - num_loop == 3 (has_hot_loop = true,  TailNumber::Odd):  K = 6 * 64 = 384
//   - num_loop == 4 (has_hot_loop = true,  TailNumber::Even): K = 8 * 64 = 512
TEST(StreamKTdmV3EdgeCases, TailNumLoopPaths)
{
    using Config = StreamKTdmV3Config<Row,
                                      Col,
                                      Row,
                                      BF16,
                                      BF16,
                                      F32,
                                      BF16,
                                      GemmConfigTdmV3_4W_128x128x64,
                                      Linear,
                                      Persistent,
                                      CShuffleEpilogue,
                                      Unpadded>;

    run_streamk_tdm_v3_case<Config>(128, 128, 128, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<Config>(128, 128, 256, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<Config>(128, 128, 384, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<Config>(128, 128, 512, 0, 0, 0, 2u);
}

// Unaligned M, N, and K with PadM=true, PadN=true, PadK=true (as in PR #15005)
// under 1-CTA, 2-CTA, and multi-CTA Stream-K splits across all four A/B layouts,
// 4-wave and 16-wave configs, 16-bit and 8-bit inputs, and CShuffle/TDM epilogues.
TEST(StreamKTdmV3EdgeCases, PaddedUnalignedMNKAllLayouts)
{
    using RcrBf16_4W  = StreamKTdmV3Config<Row,
                                           Col,
                                           Row,
                                           BF16,
                                           BF16,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_4W_128x128x64,
                                           Linear,
                                           Persistent,
                                           CShuffleEpilogue,
                                           Padded>;
    using RrrFp16_4W  = StreamKTdmV3Config<Row,
                                           Row,
                                           Row,
                                           F16,
                                           F16,
                                           F32,
                                           F16,
                                           GemmConfigTdmV3_4W_128x128x64,
                                           Atomic,
                                           NonPersistent,
                                           CShuffleEpilogue,
                                           Padded>;
    using CrrBf16_4W  = StreamKTdmV3Config<Col,
                                           Row,
                                           Row,
                                           BF16,
                                           BF16,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_4W_64x64x64,
                                           Tree,
                                           Persistent,
                                           TdmEpilogue,
                                           Padded>;
    using CcrFp16_4W  = StreamKTdmV3Config<Col,
                                           Col,
                                           Row,
                                           F16,
                                           F16,
                                           F32,
                                           F16,
                                           GemmConfigTdmV3_4W_128x128x64,
                                           Linear,
                                           NonPersistent,
                                           TdmEpilogue,
                                           Padded>;
    using RcrBf16_16W = StreamKTdmV3Config<Row,
                                           Col,
                                           Row,
                                           BF16,
                                           BF16,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_16W_256x256x64,
                                           Linear,
                                           Persistent,
                                           CShuffleEpilogue,
                                           Padded>;
    using RcrF8_4W    = StreamKTdmV3Config<Row,
                                           Col,
                                           Row,
                                           F8,
                                           F8,
                                           F32,
                                           BF16,
                                           GemmConfigTdmV3_4W_128x128x128_F8,
                                           Linear,
                                           Persistent,
                                           CShuffleEpilogue,
                                           Padded>;

    // 1-CTA, 2-CTA, and multi-CTA unaligned M/N/K across all four A/B layouts
    run_streamk_tdm_v3_case<RcrBf16_4W>(100, 120, 150, 0, 0, 0, 1u);
    run_streamk_tdm_v3_case<RcrBf16_4W>(100, 120, 200, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<RcrBf16_4W>(65, 72, 160, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<RcrBf16_4W>(193, 264, 225, 0, 0, 0, 4u);

    run_streamk_tdm_v3_case<RrrFp16_4W>(100, 120, 200, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<RrrFp16_4W>(200, 304, 200, 0, 0, 0, 4u);

    run_streamk_tdm_v3_case<CrrBf16_4W>(65, 72, 160, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<CrrBf16_4W>(150, 200, 180, 0, 0, 0, 4u);

    run_streamk_tdm_v3_case<CcrFp16_4W>(100, 120, 200, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<CcrFp16_4W>(200, 304, 200, 0, 0, 0, 4u);

    // 16-wave 256x256x64 padded unaligned M/N/K (2-CTA and multi-CTA)
    run_streamk_tdm_v3_case<RcrBf16_16W>(200, 240, 200, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<RcrBf16_16W>(400, 520, 200, 0, 0, 0, 4u);

    // 8-bit 128x128x128 padded unaligned M/N/K (2-CTA and multi-CTA)
    run_streamk_tdm_v3_case<RcrF8_4W>(100, 120, 300, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<RcrF8_4W>(200, 304, 300, 0, 0, 0, 4u);
}

// TDM clips A and B in hardware, so unaligned M and K work even when unpadded.
// For row-major C:
//   - Unpadded CShuffleEpilogue requires N % TileN == 0.
//   - Padded CShuffleEpilogue requires N % GetVectorSizeC() == 0 (8 elements for bf16/fp16).
//   - TdmEpilogue clips C in hardware and accepts any N, including N % 8 != 0.
TEST(StreamKTdmV3EdgeCases, UnpaddedAndOddNAlignment)
{
    using UnpaddedConfig = StreamKTdmV3Config<Row,
                                              Col,
                                              Row,
                                              BF16,
                                              BF16,
                                              F32,
                                              BF16,
                                              GemmConfigTdmV3_4W_128x128x64,
                                              Linear,
                                              NonPersistent,
                                              CShuffleEpilogue,
                                              Unpadded>;
    using PaddedConfig   = StreamKTdmV3Config<Row,
                                              Col,
                                              Row,
                                              BF16,
                                              BF16,
                                              F32,
                                              BF16,
                                              GemmConfigTdmV3_4W_128x128x64,
                                              Linear,
                                              NonPersistent,
                                              CShuffleEpilogue,
                                              Padded>;

    // Unpadded CShuffleEpilogue: unaligned M and K with tile-aligned N succeeds.
    run_streamk_tdm_v3_case<UnpaddedConfig, CShuffleEpilogue>(100, 128, 200, 0, 0, 0, 2u);
    // Unpadded CShuffleEpilogue: N not a multiple of TileN is rejected.
    EXPECT_THROW(
        (run_streamk_tdm_v3_case<UnpaddedConfig, CShuffleEpilogue>(128, 120, 256, 0, 0, 0, 2u)),
        std::runtime_error);

    // Padded CShuffleEpilogue: N = 120 (multiple of 8) succeeds; N = 102 (not multiple of 8)
    // is rejected.
    run_streamk_tdm_v3_case<PaddedConfig, CShuffleEpilogue>(100, 120, 200, 0, 0, 0, 2u);
    EXPECT_THROW(
        (run_streamk_tdm_v3_case<PaddedConfig, CShuffleEpilogue>(100, 102, 200, 0, 0, 0, 2u)),
        std::runtime_error);

    // TdmEpilogue clips C in hardware, so N = 102 (not divisible by 8) succeeds in both
    // 2-CTA and multi-CTA splits.
    run_streamk_tdm_v3_case<UnpaddedConfig, TdmEpilogue>(100, 102, 200, 0, 0, 0, 2u);
    run_streamk_tdm_v3_case<UnpaddedConfig, TdmEpilogue>(200, 302, 200, 0, 0, 0, 4u);
}
#endif

} // namespace
