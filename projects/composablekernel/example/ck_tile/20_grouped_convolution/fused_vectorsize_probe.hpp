// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// AICK-1303: shared kernel-type machinery for the fused-VectorSize dynamic-VGPR
// experiment. Builds the real GroupedConvolutionForwardKernel for VectorSize
// 1/2/4/8 (fp16, 2D NHWGC, ConvConfigComputeV3_WMMA). Included by:
//   - fused_vectorsize_probe.cpp  (defines the __global__ kernels; .hsaco source)
//   - fused_vectorsize_harness.cpp (builds kargs + launches via hipModule)
#pragma once

#include "ck_tile/host.hpp"
#include "grouped_convolution_utils.hpp"
#include "conv_configs.hpp"

// Build the real conv Kernel type for a given (VSA,VSB,VSC). Mirrors
// GroupedConvolutionForwardInvoker::grouped_conv_fwd type construction, with the
// vector sizes supplied as template parameters instead of read from ConvConfig.
template <int VSA, int VSB, int VSC>
struct FusedConvKernelBuilder
{
    static constexpr ck_tile::index_t NDimSpatial = 2;
    using PrecType                                = ck_tile::half_t;
    using InDataType                              = ck_tile::half_t;
    using WeiDataType                             = ck_tile::half_t;
    using AccDataType                             = float;
    using OutDataType                             = ck_tile::half_t;
    using InLayout  = ck_tile::tensor_layout::convolution::NHWGC;
    using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;
    using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;
    using ConvConfig = ConvConfigComputeV3_WMMA<PrecType>;

    using GemmShape = ck_tile::TileGemmShape<
        ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,
        ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,
        ck_tile::sequence<ConvConfig::M_Warp_Tile,
                          ConvConfig::N_Warp_Tile,
                          ConvConfig::K_Warp_Tile>>;

    static constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;
    using GroupedConvTraitsType    = ck_tile::GroupedConvTraits<NDimSpatial,
                                                             ConvSpec,
                                                             InLayout,
                                                             WeiLayout,
                                                             ck_tile::tuple<>,
                                                             OutLayout,
                                                             VSA,
                                                             VSB,
                                                             VSC,
                                                             ConvConfig::NumGroupsToMerge>;

    using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<
        GemmShape,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,
        GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;

    using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<
        GroupedConvTraitsType::FixedGemmParams::kPadM,
        GroupedConvTraitsType::FixedGemmParams::kPadN,
        GroupedConvTraitsType::FixedGemmParams::kPadK,
        ConvConfig::DoubleSmemBuffer,
        typename GroupedConvTraitsType::AsLayoutFwd,
        typename GroupedConvTraitsType::BsLayoutFwd,
        typename GroupedConvTraitsType::CLayoutFwd,
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,
        GroupedConvTraitsType::FixedGemmParams::Persistent,
        ConvConfig::NumWaveGroups,
        GroupedConvTraitsType::FixedGemmParams::Preshuffle,
        GroupedConvTraitsType::FixedGemmParams::LDSVectorSize,
        ck_tile::DataCachePrefetchKind::None,
        ck_tile::DataCachePrefetchKind::None,
        false,
        false /*LargeTensors*/>;

    using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<
        InDataType,
        WeiDataType,
        AccDataType,
        GemmShape,
        GemmUniversalTraits,
        ConvConfig::Scheduler,
        ck_tile::element_wise::PassThrough,
        ck_tile::element_wise::PassThrough,
        OutDataType,
        OutDataType,
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeA,
        GroupedConvTraitsType::VectorSizeB>;

    using GemmPipeline = typename PipelineTypeTraits<
        ConvConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

    using EpilogueProblem = ck_tile::CShuffleEpilogueProblem<
        InDataType,
        WeiDataType,
        ck_tile::tuple<>,
        AccDataType,
        OutDataType,
        typename GroupedConvTraitsType::ImplicitGemmDsLayout,
        typename GroupedConvTraitsType::FixedGemmParams::ELayout,
        ck_tile::element_wise::PassThrough,
        TilePartitioner::MPerBlock,
        TilePartitioner::NPerBlock,
        ConvConfig::M_Warp,
        ConvConfig::N_Warp,
        ConvConfig::M_Warp_Tile,
        ConvConfig::N_Warp_Tile,
        ConvConfig::K_Warp_Tile,
        GroupedConvTraitsType::FixedGemmParams::TransposeC,
        ConvConfig::NumWaveGroups,
        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,
        GroupedConvTraitsType::VectorSizeC>;

    using ConvEpilogue = ck_tile::CShuffleEpilogue<EpilogueProblem>;

    using Kernel = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                            TilePartitioner,
                                                            GemmPipeline,
                                                            ConvEpilogue>;
};

using K1 = FusedConvKernelBuilder<1, 1, 1>::Kernel;
using K2 = FusedConvKernelBuilder<2, 2, 2>::Kernel;
using K4 = FusedConvKernelBuilder<4, 4, 4>::Kernel;
using K8 = FusedConvKernelBuilder<8, 8, 8>::Kernel;

// The four kernels' kargs differ only nominally (VectorSize does not change the
// argument layout); the fused kernel relies on this to share one kargs blob.
static_assert(sizeof(K1::GroupedConvFwdKernelArgsSpecialized) ==
                  sizeof(K8::GroupedConvFwdKernelArgsSpecialized),
              "VectorSize must not change the kernel-args layout");
