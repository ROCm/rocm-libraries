// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Example conversion of DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle template specialization
// to use make_xdl_instance_from_old_params constexpr function

#include "builder_conv_xdl.hpp"

namespace miopen {
namespace ck_builder {

// Original template specialization (first BF16 instance from
// device_grouped_conv_fwd_xdl_bilinear_instance.hpp): NOTE: The actual layouts used depend on the
// template alias instantiation context. This example uses NGCHW/GKCYX/NGKHW which matches the
// layout combination used in MIOpen tests.
//
// DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
//     NDimSpatial,    // 2 (for 2D convolution)
//     ALayout,        // NGCHW (used in MIOpen)
//     BLayout,        // GKCYX (used in MIOpen)
//     DsLayout,       // Tuple<NGKHW>
//     ELayout,        // NGKHW (used in MIOpen)
//     BF16,           // ADataType
//     BF16,           // BDataType
//     F32,            // AccDataType
//     BF16,           // CShuffleDataType
//     Tuple<BF16>,    // DsDataType
//     BF16,           // EDataType
//     PassThrough,    // AElementwiseOperation
//     PassThrough,    // BElementwiseOperation
//     Bilinear,       // CDEElementwiseOperation
//     ConvSpec,       // ConvolutionForwardSpecialization (e.g., Default)
//     GemmMNKPadding, // GemmSpecialization::MNKPadding
//     1,              // NumGemmKPrefetchStage
//     64,             // BlockSize
//     64,             // MPerBlock
//     64,             // NPerBlock
//     32,             // KPerBlock
//     8,              // AK1
//     8,              // BK1
//     32,             // MPerXDL
//     32,             // NPerXDL
//     2,              // MXdlPerWave
//     2,              // NXdlPerWave
//     S<4, 16, 1>,    // ABlockTransferThreadClusterLengths_AK0_M_AK1
//     S<1, 0, 2>,     // ABlockTransferThreadClusterArrangeOrder
//     S<1, 0, 2>,     // ABlockTransferSrcAccessOrder
//     2,              // ABlockTransferSrcVectorDim
//     1,              // ABlockTransferSrcScalarPerVector
//     8,              // ABlockTransferDstScalarPerVector_AK1
//     1,              // ABlockLdsExtraM
//     S<4, 16, 1>,    // BBlockTransferThreadClusterLengths_BK0_N_BK1
//     S<1, 0, 2>,     // BBlockTransferThreadClusterArrangeOrder
//     S<1, 0, 2>,     // BBlockTransferSrcAccessOrder
//     2,              // BBlockTransferSrcVectorDim
//     1,              // BBlockTransferSrcScalarPerVector
//     8,              // BBlockTransferDstScalarPerVector_BK1
//     1,              // BBlockLdsExtraN
//     1,              // CShuffleMXdlPerWavePerShuffle
//     1,              // CShuffleNXdlPerWavePerShuffle
//     S<1, 16, 1, 4>, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
//     1               // CDEBlockTransferScalarPerVector_NPerBlock
//     // Optional parameters (using defaults):
//     // BF16,        // AComputeDataType (defaults to ADataType)
//     // BF16,        // BComputeDataType (defaults to AComputeDataType)
//     // Default,     // LoopSched (defaults to make_default_loop_scheduler())
//     // 1            // NumGroupsToMerge (defaults to 1)
// >

// Converted to constexpr function call:
constexpr auto example_instance = make_xdl_instance_from_old_params(
    // 1. NDimSpatial
    2,

    // 2-4. Layouts (converted from CK types to builder enums)
    ckb::TensorLayout::NGCHW, // ALayout (input) - matches MIOpen test usage
    ckb::TensorLayout::GKCYX, // BLayout (weight) - matches MIOpen test usage
    ckb::TensorLayout::NGKHW, // ELayout (output) - matches MIOpen test usage

    // 5-9. Data types (converted from CK types to builder enums)
    ckb::DataType::FP32, // ADataType (input)
    ckb::DataType::FP32, // BDataType (weight)
    ckb::DataType::FP32, // AccDataType
    ckb::DataType::FP32, // CShuffleDataType
    ckb::DataType::FP32, // EDataType (output)

    // 10-11. Specializations
    ckb::ConvSpecialization::DEFAULT,    // ConvForwardSpecialization
    ckb::GemmSpecialization::MNKPadding, // GemmSpecialization

    // 12. NumGemmKPrefetchStage
    1,

    // 13-16. Block dimensions
    64, // BlockSize
    64, // MPerBlock
    64, // NPerBlock
    16, // KPerBlock

    // 17-22. XDL parameters
    4,  // AK1
    4,  // BK1
    32, // MPerXDL
    32, // NPerXDL
    2,  // MXdlPerWave
    2,  // NXdlPerWave

    // 23-29. A block transfer parameters
    {4, 16, 1}, // ABlockTransferThreadClusterLengths_AK0_M_AK1
    {1, 0, 2},  // ABlockTransferThreadClusterArrangeOrder
    {1, 0, 2},  // ABlockTransferSrcAccessOrder
    2,          // ABlockTransferSrcVectorDim
    1,          // ABlockTransferSrcScalarPerVector
    4,          // ABlockTransferDstScalarPerVector_AK1
    true,       // ABlockLdsExtraM (1 -> true)

    // 30-36. B block transfer parameters
    {4, 16, 1}, // BBlockTransferThreadClusterLengths_BK0_N_BK1
    {1, 0, 2},  // BBlockTransferThreadClusterArrangeOrder
    {1, 0, 2},  // BBlockTransferSrcAccessOrder
    2,          // BBlockTransferSrcVectorDim
    1,          // BBlockTransferSrcScalarPerVector
    4,          // BBlockTransferDstScalarPerVector_BK1
    true,       // BBlockLdsExtraN (1 -> true)

    // 37-40. C shuffle parameters
    1, // CShuffleMXdlPerWavePerShuffle
    1, // CShuffleNXdlPerWavePerShuffle
    std::array<std::size_t, 4>{
        1, 8, 1, 8}, // CDEBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
    1,               // CDEBlockTransferScalarPerVector_NPerBlock

    // 41-42. Compute data types (using defaults from template)
    ckb::DataType::FP32, // AComputeDataType (defaults to ADataType)
    ckb::DataType::FP32, // BComputeDataType (defaults to AComputeDataType)

    // 43. Loop scheduler (using default)
    ckb::PipelineScheduler::DEFAULT, // LoopSched

    // 44. Groups to merge (using default)
    1 // NumGroupsToMerge
);

constexpr auto NGCHW = ckb::TensorLayout::NGCHW;
constexpr auto GKCYX = ckb::TensorLayout::GKCYX;
constexpr auto NGKHW = ckb::TensorLayout::NGKHW;
constexpr auto FP32  = ckb::DataType::FP32;

constexpr auto example_instances()
{
    std::array result = {make_xdl_instance_from_old_params(2,
                                                           NGCHW,
                                                           GKCYX,
                                                           NGKHW,
                                                           FP32,
                                                           FP32,
                                                           FP32,
                                                           FP32,
                                                           FP32,
                                                           ckb::ConvSpecialization::DEFAULT,
                                                           ckb::GemmSpecialization::MNKPadding,
                                                           1,
                                                           64,
                                                           64,
                                                           64,
                                                           16,
                                                           4,
                                                           4,
                                                           32,
                                                           32,
                                                           2,
                                                           2,
                                                           {4, 16, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           1,
                                                           4,
                                                           true,
                                                           {4, 16, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           1,
                                                           4,
                                                           true,
                                                           1,
                                                           1,
                                                           {1, 8, 1, 8},
                                                           1,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::PipelineScheduler::DEFAULT,
                                                           1),
                         make_xdl_instance_from_old_params(2,
                                                           ckb::TensorLayout::NGCHW,
                                                           ckb::TensorLayout::GKCYX,
                                                           ckb::TensorLayout::NGKHW,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::ConvSpecialization::DEFAULT,
                                                           ckb::GemmSpecialization::MNKPadding,
                                                           1,
                                                           64,
                                                           64,
                                                           32,
                                                           16,
                                                           4,
                                                           4,
                                                           32,
                                                           32,
                                                           2,
                                                           1,
                                                           {4, 16, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           4,
                                                           4,
                                                           true,
                                                           {4, 16, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           4,
                                                           4,
                                                           true,
                                                           1,
                                                           1,
                                                           {1, 8, 1, 8},
                                                           1,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::PipelineScheduler::DEFAULT,
                                                           1),
                         make_xdl_instance_from_old_params(2,
                                                           ckb::TensorLayout::NGCHW,
                                                           ckb::TensorLayout::GKCYX,
                                                           ckb::TensorLayout::NGKHW,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::ConvSpecialization::DEFAULT,
                                                           ckb::GemmSpecialization::MNKPadding,
                                                           1,
                                                           256,
                                                           64,
                                                           64,
                                                           32,
                                                           8,
                                                           8,
                                                           16,
                                                           16,
                                                           2,
                                                           2,
                                                           {4, 64, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           1,
                                                           8,
                                                           true,
                                                           {4, 64, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           2,
                                                           8,
                                                           true,
                                                           1,
                                                           1,
                                                           {1, 32, 1, 4},
                                                           1,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::PipelineScheduler::DEFAULT,
                                                           1),
                         make_xdl_instance_from_old_params(2,
                                                           ckb::TensorLayout::NGCHW,
                                                           ckb::TensorLayout::GKCYX,
                                                           ckb::TensorLayout::NGKHW,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::ConvSpecialization::DEFAULT,
                                                           ckb::GemmSpecialization::MNKPadding,
                                                           1,
                                                           128,
                                                           64,
                                                           128,
                                                           16,
                                                           4,
                                                           4,
                                                           32,
                                                           32,
                                                           2,
                                                           2,
                                                           {4, 32, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           4,
                                                           4,
                                                           true,
                                                           {4, 32, 1},
                                                           {1, 0, 2},
                                                           {1, 0, 2},
                                                           2,
                                                           4,
                                                           4,
                                                           true,
                                                           1,
                                                           1,
                                                           {1, 8, 1, 16},
                                                           4,
                                                           ckb::DataType::FP32,
                                                           ckb::DataType::FP32,
                                                           ckb::PipelineScheduler::DEFAULT,
                                                           1)};

    return result;
}

// The result is an XdlInstance containing:
// - example_instance.signature: XdlSignature with spatial_dim, direction, layouts, data_types
// - example_instance.algorithm: XdlAlgorithm with all tuning parameters

// Type mappings used:
// CK Type                          -> Builder Enum
// ====================================================
// ck::bhalf_t (BF16)              -> ckb::DataType::BF16
// ck::half_t (F16)                -> ckb::DataType::FP16
// float (F32)                     -> ckb::DataType::FP32
// int8_t                          -> ckb::DataType::I8
// ck::tf32_t (TF32)               -> ckb::DataType::FP32 (with compute type)
//
// tensor_layout::convolution::NGCHW -> ckb::TensorLayout::NGCHW
// tensor_layout::convolution::GKCYX -> ckb::TensorLayout::GKCYX
// tensor_layout::convolution::NGKHW -> ckb::TensorLayout::NGKHW
// tensor_layout::convolution::GNHWC -> ckb::TensorLayout::GNHWC
// tensor_layout::convolution::GKYXC -> ckb::TensorLayout::GKYXC
// tensor_layout::convolution::GNHWK -> ckb::TensorLayout::GNHWK
// tensor_layout::convolution::NHWGC -> ckb::TensorLayout::NHWGC
// tensor_layout::convolution::NHWGK -> ckb::TensorLayout::NHWGK
//
// ConvolutionForwardSpecialization::Default           -> ckb::ConvSpecialization::DEFAULT
// ConvolutionForwardSpecialization::Filter1x1Pad0     ->
// ckb::ConvSpecialization::FILTER_1X1_PAD0
// ConvolutionForwardSpecialization::Filter1x1Stride1Pad0 ->
// ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0
//
// GemmSpecialization::MNKPadding  -> ckb::GemmSpecialization::MNKPadding
//
// LoopScheduler::Default          -> ckb::PipelineScheduler::DEFAULT
// LoopScheduler::Interwave        -> ckb::PipelineScheduler::INTERWAVE
//
// Sequence<...> (S<...>)          -> std::array<std::size_t, N>{...}
// Integer 0 or 1 for bool params  -> false or true

constexpr XdlInstance make_instance() { return example_instance; }
} // namespace ck_builder
} // namespace miopen
