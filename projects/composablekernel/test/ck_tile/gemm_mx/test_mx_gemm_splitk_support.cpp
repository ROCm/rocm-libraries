// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"

namespace {

template <ck_tile::index_t VectorSize>
bool supports_split_k(ck_tile::index_t split_k)
{
    using Row         = ck_tile::tensor_layout::gemm::RowMajor;
    using Col         = ck_tile::tensor_layout::gemm::ColumnMajor;
    using Shape       = ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 128>,
                                               ck_tile::sequence<2, 2, 1>,
                                               ck_tile::sequence<16, 16, 128>>;
    using Partitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<Shape, 8, 4>;
    using Problem     = ck_tile::MxGemmPipelineProblem<
            ck_tile::fp8_t,
            ck_tile::fp8_t,
            float,
            Shape,
            ck_tile::TileGemmUniversalTraits<true, true, false, true, Row, Col, Row, true>,
            ck_tile::GemmPipelineScheduler::Intrawave>;
    using Pipeline        = ck_tile::GemmPipelineAgBgCrCompAsync<Problem>;
    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<ck_tile::fp8_t,
                                                             ck_tile::fp8_t,
                                                             ck_tile::tuple<>,
                                                             float,
                                                             ck_tile::fp16_t,
                                                             ck_tile::tuple<>,
                                                             Row,
                                                             ck_tile::element_wise::PassThrough,
                                                             128,
                                                             128,
                                                             2,
                                                             2,
                                                             16,
                                                             16,
                                                             128,
                                                             true,
                                                             1,
                                                             true,
                                                             VectorSize,
                                                             1,
                                                             false,
                                                             ck_tile::fp8_t,
                                                             ck_tile::fp8_t,
                                                             true>;
    using Kernel =
        ck_tile::MxGemmKernel<Partitioner, Pipeline, ck_tile::CShuffleEpilogue<EpilogueProblem>>;
    // No launch is needed: these dimensions satisfy all constraints except the
    // scalar FP16 atomic store when split_k is greater than one.
    const ck_tile::MxGemmHostArgs<> args({nullptr},
                                         {nullptr},
                                         {nullptr},
                                         {nullptr},
                                         {},
                                         nullptr,
                                         split_k,
                                         128,
                                         128,
                                         256,
                                         {256},
                                         {256},
                                         {},
                                         128);
    return Kernel::IsSupportedArgument(Kernel::MakeKernelArgs(args));
}

TEST(MxGemmSplitKSupport, ScalarCShuffleSupportsOrdinaryGemm)
{
    EXPECT_TRUE(supports_split_k<1>(1));
    EXPECT_FALSE(supports_split_k<1>(2));
}

TEST(MxGemmSplitKSupport, PairedCShuffleRetainsSplitK)
{
    EXPECT_TRUE(supports_split_k<2>(1));
    EXPECT_TRUE(supports_split_k<2>(2));
}

} // namespace
