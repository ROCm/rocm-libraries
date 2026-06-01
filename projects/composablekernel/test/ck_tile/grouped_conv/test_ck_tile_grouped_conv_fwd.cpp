// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp"

using namespace ck_tile;

// =====================================================================
// Test Conv Configs
// =====================================================================
struct TestConvConfig
{
    static constexpr index_t VectorSizeA = 4;
    static constexpr index_t VectorSizeB = 4;
    static constexpr index_t VectorSizeC = 4;

    static constexpr index_t M_Tile = 128;
    static constexpr index_t N_Tile = 128;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::COMPUTE_V3;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
    static constexpr index_t kBlockPerCu      = 1;
};

// Small-tile config for unit-vector tile sizes (1x1 coverage per tile)
struct UnitTileConfig
{
    static constexpr index_t VectorSizeA = 1;
    static constexpr index_t VectorSizeB = 1;
    static constexpr index_t VectorSizeC = 1;

    static constexpr index_t M_Tile = 64;
    static constexpr index_t N_Tile = 64;
    static constexpr index_t K_Tile = 32;

    static constexpr index_t M_Warp = 2;
    static constexpr index_t N_Warp = 2;
    static constexpr index_t K_Warp = 1;

    static constexpr index_t M_Warp_Tile = 16;
    static constexpr index_t N_Warp_Tile = 16;
    static constexpr index_t K_Warp_Tile = 16;

    static constexpr bool DoubleSmemBuffer    = false;
    static constexpr GemmPipeline Pipeline    = GemmPipeline::COMPUTE_V3;
    static constexpr index_t NumWaveGroups    = 1;
    static constexpr index_t NumGroupsToMerge = 1;
    static constexpr auto Scheduler           = GemmPipelineScheduler::Intrawave;
    static constexpr index_t kBlockPerCu      = 1;
};

// =====================================================================
// BuildFwdKernel: Assembles the full forward kernel type for testing
// =====================================================================
template <typename PrecType,
          typename ConvConfig,
          typename InLayout,
          typename WeiLayout,
          typename OutLayout,
          index_t NDimSpatial     = 2,
          bool EnableSplitImage_  = false>
struct BuildFwdKernel
{
    using GemmShape = TileGemmShape<
        sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
        sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
        sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile>>;

    using ConvTraits = GroupedConvTraits<NDimSpatial,
                                         ConvolutionSpecialization::Default,
                                         InLayout,
                                         WeiLayout,
                                         tuple<>,
                                         OutLayout,
                                         ConvConfig::VectorSizeA,
                                         ConvConfig::VectorSizeB,
                                         ConvConfig::VectorSizeC,
                                         ConvConfig::NumGroupsToMerge,
                                         EnableSplitImage_>;

    using TilePartitioner = GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        ConvTraits::FixedGemmParams::TilePartitionerGroupNum,
        ConvTraits::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits =
        TileGemmUniversalTraits<ConvTraits::FixedGemmParams::kPadM,
                                ConvTraits::FixedGemmParams::kPadN,
                                ConvTraits::FixedGemmParams::kPadK,
                                ConvConfig::DoubleSmemBuffer,
                                typename ConvTraits::AsLayoutFwd,
                                typename ConvTraits::BsLayoutFwd,
                                typename ConvTraits::CLayoutFwd,
                                ConvTraits::FixedGemmParams::TransposeC,
                                ConvTraits::FixedGemmParams::UseStructuredSparsity,
                                ConvTraits::FixedGemmParams::Persistent,
                                ConvConfig::NumWaveGroups>;

    using UniversalGemmProblem =
        UniversalGemmPipelineProblem<PrecType,
                                     PrecType,
                                     float,
                                     GemmShape,
                                     GemmUniversalTraits,
                                     ConvConfig::Scheduler,
                                     element_wise::PassThrough,
                                     element_wise::PassThrough,
                                     PrecType,
                                     PrecType,
                                     ConvTraits::FixedGemmParams::FixedVectorSize,
                                     ConvTraits::VectorSizeA,
                                     ConvTraits::VectorSizeB>;

    using GemmPipeline = GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;

    using EpilogueProblem = CShuffleEpilogueProblem<PrecType,
                                                    PrecType,
                                                    tuple<>,
                                                    float,
                                                    PrecType,
                                                    typename ConvTraits::ImplicitGemmDsLayout,
                                                    typename ConvTraits::FixedGemmParams::ELayout,
                                                    element_wise::PassThrough,
                                                    TilePartitioner::MPerBlock,
                                                    TilePartitioner::NPerBlock,
                                                    ConvConfig::M_Warp,
                                                    ConvConfig::N_Warp,
                                                    ConvConfig::M_Warp_Tile,
                                                    ConvConfig::N_Warp_Tile,
                                                    ConvConfig::K_Warp_Tile,
                                                    ConvTraits::FixedGemmParams::TransposeC,
                                                    ConvConfig::NumWaveGroups,
                                                    ConvTraits::FixedGemmParams::FixedVectorSize,
                                                    ConvTraits::VectorSizeC>;

    using Epilogue = CShuffleEpilogue<EpilogueProblem>;

    using type = GroupedConvolutionForwardKernel<ConvTraits, TilePartitioner, GemmPipeline, Epilogue>;
};

// =====================================================================
// Helper: Create 2D forward host args with null device pointers
// =====================================================================
static GroupedConvFwdHostArgs<> create_2d_fwd_host_args(index_t G,
                                                         index_t N,
                                                         index_t K,
                                                         index_t C,
                                                         index_t Hi,
                                                         index_t Wi,
                                                         index_t Y  = 3,
                                                         index_t X  = 3,
                                                         index_t sy = 1,
                                                         index_t sx = 1,
                                                         index_t dy = 1,
                                                         index_t dx = 1,
                                                         index_t lp_y = 1,
                                                         index_t lp_x = 1,
                                                         index_t rp_y = 1,
                                                         index_t rp_x = 1)
{
    auto conv_param = conv::ConvParam{2, G, N, K, C,
                                      {Y, X}, {Hi, Wi},
                                      {sy, sx}, {dy, dx},
                                      {lp_y, lp_x}, {rp_y, rp_x}};
    return GroupedConvFwdHostArgs<>{conv_param, nullptr, nullptr, {}, nullptr, 1};
}

// Helper: Create 1D forward host args
static GroupedConvFwdHostArgs<> create_1d_fwd_host_args(index_t G,
                                                         index_t N,
                                                         index_t K,
                                                         index_t C,
                                                         index_t Wi,
                                                         index_t X  = 3,
                                                         index_t sx = 1,
                                                         index_t dx = 1,
                                                         index_t lp_x = 1,
                                                         index_t rp_x = 1)
{
    auto conv_param = conv::ConvParam{1, G, N, K, C,
                                      {X}, {Wi},
                                      {sx}, {dx},
                                      {lp_x}, {rp_x}};
    return GroupedConvFwdHostArgs<>{conv_param, nullptr, nullptr, {}, nullptr, 1};
}

// Helper: Create 3D forward host args
static GroupedConvFwdHostArgs<> create_3d_fwd_host_args(index_t G,
                                                         index_t N,
                                                         index_t K,
                                                         index_t C,
                                                         index_t Di,
                                                         index_t Hi,
                                                         index_t Wi,
                                                         index_t Z  = 3,
                                                         index_t Y  = 3,
                                                         index_t X  = 3)
{
    auto conv_param = conv::ConvParam{3, G, N, K, C,
                                      {Z, Y, X}, {Di, Hi, Wi},
                                      {1, 1, 1}, {1, 1, 1},
                                      {1, 1, 1}, {1, 1, 1}};
    return GroupedConvFwdHostArgs<>{conv_param, nullptr, nullptr, {}, nullptr, 1};
}

// =====================================================================
// Test class
// =====================================================================
class GroupedConvFwdSplitImageTest : public ::testing::Test
{
};

// =====================================================================
// Test: GetSplitImageInfo with low threshold forces split on small tensor
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, GetSplitImageInfoLowThreshold)
{
    // NDimSpatial=2, default specialization, vec=4
    using TransformType = TransformConvFwdToGemm<2,
                                                  ConvolutionSpecialization::Default,
                                                  4, 4, 4,
                                                  1,     // NumGroupsToMerge
                                                  false, // SplitN
                                                  half_t,
                                                  half_t>;

    // G=1, N=2, C=64, K=64, H_out=128, W_out=128
    // Output elements = 2 * 128 * 128 * 64 = 2,097,152
    // Output bytes = 2,097,152 * 2 = 4,194,304

    // With threshold=1024, should definitely split
    auto info = TransformType::GetSplitImageInfo(1, 2, 64, 64, 1, 128, 128, 1024);
    EXPECT_TRUE(info.should_split);
    EXPECT_GE(info.num_h_pieces * info.num_w_pieces, 2);
}

TEST_F(GroupedConvFwdSplitImageTest, GetSplitImageInfoHighThreshold)
{
    using TransformType = TransformConvFwdToGemm<2,
                                                  ConvolutionSpecialization::Default,
                                                  4, 4, 4,
                                                  1, false,
                                                  half_t, half_t>;

    // With default TwoGB threshold on small tensor, should NOT split
    auto info = TransformType::GetSplitImageInfo(1, 2, 64, 64, 1, 128, 128);
    EXPECT_FALSE(info.should_split);
    EXPECT_EQ(info.num_d_pieces, 1);
    EXPECT_EQ(info.num_h_pieces, 1);
    EXPECT_EQ(info.num_w_pieces, 1);
}

// =====================================================================
// Test: PopulateSplitImageKargs 2-piece H-split
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, PopulateSplitImageKargs2PieceHSplit)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestConvConfig,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,    // NDimSpatial
                                           true  // EnableSplitImage
                                           >::type;

    // G=1, N=2, K=64, C=64, Hi=130, Wi=130 → Ho=128, Wo=128 (filter=3x3, pad=1)
    auto host_args = create_2d_fwd_host_args(1, 2, 64, 64, 130, 130);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Use a threshold that forces exactly 2 H-pieces
    // Output per piece memory: N * H_piece * W * K * sizeof(half_t)
    // Full: 2 * 128 * 128 * 64 * 2 = 4,194,304
    // With threshold = 2,097,152 (half), should split H into 2
    const long_index_t threshold = 2 * 128 * 128 * 64 * static_cast<long_index_t>(sizeof(half_t)) / 2;

    dim3 grids = PopulateSplitImageKargs<Kernel>(kargs, host_args, threshold);

    EXPECT_EQ(kargs.num_spatial_pieces, 2);
    EXPECT_EQ(kargs.split_image.num_h_pieces, 2);
    EXPECT_EQ(kargs.split_image.num_w_pieces, 1);
    EXPECT_EQ(kargs.split_image.total_h, 128);
    EXPECT_EQ(kargs.split_image.total_w, 128);

    // Verify pieces cover the full H range
    EXPECT_EQ(kargs.split_image.pieces[0].h_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[1].h_start, 64);
    EXPECT_EQ(kargs.split_image.pieces[0].h_size + kargs.split_image.pieces[1].h_size, 128);

    // Block ranges should be contiguous
    EXPECT_EQ(kargs.split_image.pieces[0].block_start, 0);
    EXPECT_EQ(kargs.split_image.pieces[0].block_end, kargs.split_image.pieces[1].block_start);

    // Grid x should equal total blocks across all pieces
    EXPECT_EQ(static_cast<index_t>(grids.x), kargs.split_image.pieces[1].block_end);
    EXPECT_GT(grids.x, 0u);
}

// =====================================================================
// Test: PopulateSplitImageKargs 4-piece H+W split
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, PopulateSplitImageKargs4PieceHWSplit)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestConvConfig,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true>::type;

    auto host_args = create_2d_fwd_host_args(1, 2, 64, 64, 130, 130);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Use very low threshold to force both H and W splitting
    // Need threshold small enough that 2 H-pieces still exceeds it
    // 2 * 64 * 128 * 64 * 2 = 2,097,152 per H-piece → threshold < this
    const long_index_t threshold = 512 * 1024; // 512 KB

    dim3 grids = PopulateSplitImageKargs<Kernel>(kargs, host_args, threshold);

    EXPECT_GE(kargs.num_spatial_pieces, 4);
    // With such a low threshold, expect both H and W to be split
    EXPECT_GE(kargs.split_image.num_h_pieces * kargs.split_image.num_w_pieces, 4);

    // All pieces should have non-zero spatial sizes
    for(index_t i = 0; i < kargs.num_spatial_pieces; i++)
    {
        EXPECT_GT(kargs.split_image.pieces[i].h_size, 0);
        EXPECT_GT(kargs.split_image.pieces[i].w_size, 0);
    }

    // Block ranges should be contiguous (each piece starts where previous ended)
    for(index_t i = 1; i < kargs.num_spatial_pieces; i++)
    {
        EXPECT_EQ(kargs.split_image.pieces[i].block_start,
                  kargs.split_image.pieces[i - 1].block_end);
    }

    // Grid x should equal total blocks
    EXPECT_EQ(static_cast<index_t>(grids.x),
              kargs.split_image.pieces[kargs.num_spatial_pieces - 1].block_end);
}

// =====================================================================
// Test: 3D D-split (D-dimension splits first per hierarchical priority)
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, PopulateSplitImageKargs3DDSplit)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestConvConfig,
                                           tensor_layout::convolution::NDHWGC,
                                           tensor_layout::convolution::GKZYXC,
                                           tensor_layout::convolution::NDHWGK,
                                           3,    // NDimSpatial
                                           true  // EnableSplitImage
                                           >::type;

    // 3D conv: Di=66, Hi=66, Wi=66 → Do=64, Ho=64, Wo=64 (filter=3x3x3, pad=1)
    auto host_args = create_3d_fwd_host_args(1, 2, 64, 64, 66, 66, 66);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Output: 2 * 64 * 64 * 64 * 64 * 2 = 67,108,864 bytes
    // Threshold: force D-split first
    const long_index_t threshold = 2 * 32 * 64 * 64 * 64 * static_cast<long_index_t>(sizeof(half_t));

    dim3 grids = PopulateSplitImageKargs<Kernel>(kargs, host_args, threshold);

    // D should split first (hierarchical: D -> H -> W)
    EXPECT_GE(kargs.split_image.num_d_pieces, 2);
    EXPECT_GT(grids.x, 0u);

    // First piece should start at D=0
    EXPECT_EQ(kargs.split_image.pieces[0].d_start, 0);
}

// =====================================================================
// Test: Remainder handling (non-divisible spatial dims)
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, RemainderHandling)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           UnitTileConfig,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true>::type;

    // Non-divisible: Hi=103, Wi=103, filter=1x1, stride=1, pad=0 → Ho=103, Wo=103
    auto host_args = create_2d_fwd_host_args(1, 1, 32, 32, 103, 103,
                                              1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Force split with low threshold
    const long_index_t threshold = 1024;
    dim3 grids = PopulateSplitImageKargs<Kernel>(kargs, host_args, threshold);

    EXPECT_GT(kargs.num_spatial_pieces, 1);

    // Verify spatial coverage: all pieces together should cover the full output
    index_t total_h_covered = 0;
    for(index_t i = 0; i < kargs.num_spatial_pieces; i++)
    {
        EXPECT_GT(kargs.split_image.pieces[i].h_size, 0);
        // For the W dimension: check first piece row
        if(kargs.split_image.pieces[i].h_start == 0)
        {
            // Pieces with same h_start should have contiguous w ranges
        }
    }

    // Last piece should end at the total output boundary
    // The pieces should cover the full H dimension
    // Check by summing unique H ranges
    const index_t num_h = kargs.split_image.num_h_pieces;
    const index_t num_w = kargs.split_image.num_w_pieces;
    for(index_t h = 0; h < num_h; h++)
    {
        index_t h_start = kargs.split_image.pieces[h * num_w].h_start;
        index_t h_size  = kargs.split_image.pieces[h * num_w].h_size;
        total_h_covered += h_size;
        // Verify piece starts right after previous piece
        if(h > 0)
        {
            index_t prev_h_start = kargs.split_image.pieces[(h - 1) * num_w].h_start;
            index_t prev_h_size  = kargs.split_image.pieces[(h - 1) * num_w].h_size;
            EXPECT_EQ(h_start, prev_h_start + prev_h_size);
        }
    }
    EXPECT_EQ(total_h_covered, 103);

    EXPECT_GT(grids.x, 0u);
}

// =====================================================================
// Test: No-split baseline (threshold=TwoGB on small tensor)
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, NoSplitBaseline)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestConvConfig,
                                           tensor_layout::convolution::NHWGC,
                                           tensor_layout::convolution::GKYXC,
                                           tensor_layout::convolution::NHWGK,
                                           2,
                                           true>::type;

    // Small tensor: G=1, N=1, K=8, C=8, Hi=9, Wi=9
    auto host_args = create_2d_fwd_host_args(1, 1, 8, 8, 9, 9);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Default 2GB threshold - should NOT split
    dim3 grids = PopulateSplitImageKargs<Kernel>(kargs, host_args);

    // When no split, num_spatial_pieces stays at default 1
    EXPECT_EQ(kargs.num_spatial_pieces, 1);

    // Grid should match normal GridSize
    dim3 normal_grids = Kernel::GridSize(kargs);
    EXPECT_EQ(grids.x, normal_grids.x);
    EXPECT_EQ(grids.y, normal_grids.y);
    EXPECT_EQ(grids.z, normal_grids.z);
}

// =====================================================================
// Test: has_split_image_v trait
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, HasSplitImageTrait)
{
    using KernelWithSplit = typename BuildFwdKernel<half_t,
                                                    TestConvConfig,
                                                    tensor_layout::convolution::NHWGC,
                                                    tensor_layout::convolution::GKYXC,
                                                    tensor_layout::convolution::NHWGK,
                                                    2,
                                                    true>::type;

    using KernelWithoutSplit = typename BuildFwdKernel<half_t,
                                                       TestConvConfig,
                                                       tensor_layout::convolution::NHWGC,
                                                       tensor_layout::convolution::GKYXC,
                                                       tensor_layout::convolution::NHWGK,
                                                       2,
                                                       false>::type;

    EXPECT_TRUE(has_split_image_v<KernelWithSplit>);
    EXPECT_FALSE(has_split_image_v<KernelWithoutSplit>);
}

// =====================================================================
// Test: 1D W-split
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, PopulateSplitImageKargs1DWSplit)
{
    using Kernel = typename BuildFwdKernel<half_t,
                                           TestConvConfig,
                                           tensor_layout::convolution::NWGC,
                                           tensor_layout::convolution::GKXC,
                                           tensor_layout::convolution::NWGK,
                                           1,    // NDimSpatial
                                           true  // EnableSplitImage
                                           >::type;

    // 1D: Wi=258, filter=3, stride=1, pad=1 → Wo=256
    auto host_args = create_1d_fwd_host_args(1, 2, 64, 64, 258);
    auto kargs     = Kernel::MakeKernelArgs(host_args);

    // Force W-split
    const long_index_t threshold = 2 * 128 * 64 * static_cast<long_index_t>(sizeof(half_t));

    dim3 grids = PopulateSplitImageKargs<Kernel>(kargs, host_args, threshold);

    EXPECT_GE(kargs.split_image.num_w_pieces, 2);
    EXPECT_EQ(kargs.split_image.num_h_pieces, 1);
    EXPECT_EQ(kargs.split_image.num_d_pieces, 1);

    // W pieces should be contiguous
    EXPECT_EQ(kargs.split_image.pieces[0].w_start, 0);
    index_t total_w_covered = 0;
    for(index_t i = 0; i < kargs.num_spatial_pieces; i++)
    {
        total_w_covered += kargs.split_image.pieces[i].w_size;
    }
    EXPECT_EQ(total_w_covered, 256);

    EXPECT_GT(grids.x, 0u);
}

// =====================================================================
// Test: calculate_spatial_piece directly
// =====================================================================
TEST_F(GroupedConvFwdSplitImageTest, CalculateSpatialPieceDirect)
{
    // Simple tile partitioner mock - use actual type
    using GemmShape = TileGemmShape<
        sequence<128, 128, 32>,
        sequence<2, 2, 1>,
        sequence<16, 16, 16>>;
    using TP = GemmSpatiallyLocalTilePartitioner<GemmShape, 8, 4>;

    // 2 H-pieces on 128x128 output, N=2, K=64
    auto piece0 = calculate_spatial_piece<TP>(0, 1, 2, 1, 1, 64, 128, 1, 128, 128, 2, 64, 0);
    auto piece1 = calculate_spatial_piece<TP>(1, 1, 2, 1, 1, 64, 128, 1, 128, 128, 2, 64,
                                              piece0.block_end);

    // Piece 0: H=[0, 64), W=[0, 128)
    EXPECT_EQ(piece0.h_start, 0);
    EXPECT_EQ(piece0.h_size, 64);
    EXPECT_EQ(piece0.w_start, 0);
    EXPECT_EQ(piece0.w_size, 128);

    // Piece 1: H=[64, 128), W=[0, 128)
    EXPECT_EQ(piece1.h_start, 64);
    EXPECT_EQ(piece1.h_size, 64);
    EXPECT_EQ(piece1.w_start, 0);
    EXPECT_EQ(piece1.w_size, 128);

    // Contiguous blocks
    EXPECT_EQ(piece0.block_start, 0);
    EXPECT_EQ(piece1.block_start, piece0.block_end);
    EXPECT_GT(piece1.block_end, piece1.block_start);
}
