// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/instance_data/xdl.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

// Test instance using the DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle helper
// This matches the F32 instance from CK's device_grouped_conv_fwd_xdl_instance.hpp:252
//
// Original CK instance:
// DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
//     NDimSpatial, ALayout, BLayout, DsLayout, ELayout, F32, F32, F32, F32, DsDataTypes, F32,
//     PassThrough, PassThrough, OutElementOp, ConvSpec, GemmMNKPadding,
//     1,                // NumGemmKPrefetch
//     64,               // BlockSize
//     64, 64, 16,       // MPerBlock, NPerBlock, KPerBlock
//     4, 4,             // AK1, BK1
//     32, 32,           // MPerXDL, NPerXDL
//     2, 2,             // MXdlPerWave, NXdlPerWave
//     S<4, 16, 1>,      // ABlockTransferThreadClusterLengths_K0_M_K1
//     S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
//     S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
//     2, 1, 4,          // ABlockTransferSrcVectorDim, SrcScalarPerVector, DstScalarPerVector_K1
//     1,                // ABlockLdsExtraM
//     S<4, 16, 1>,      // BBlockTransferThreadClusterLengths_K0_N_K1
//     S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
//     S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
//     2, 1, 4,          // BBlockTransferSrcVectorDim, SrcScalarPerVector, DstScalarPerVector_K1
//     1,                // BBlockLdsExtraN
//     1, 1,             // CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle
//     S<1, 8, 1, 8>,    // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
//     1>                // CBlockTransferScalarPerVector_NPerBlock

[[maybe_unused]] constexpr auto grouped_conv_2d_fwd_xdl_f32_instance_1 =
    DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<0>(
        // Spatial dimension and layouts
        2,                        // spatial_dim (2D convolution)
        ckb::TensorLayout::GNHWC, // input_layout
        ckb::TensorLayout::GKYXC, // weight_layout
        {},                       // ds_layouts (no D tensors for ABD variant)
        ckb::TensorLayout::GNHWK, // output_layout

        // Data types
        ckb::DataType::FP32, // input_data_type
        ckb::DataType::FP32, // weight_data_type
        ckb::DataType::FP32, // acc_data_type
        ckb::DataType::FP32, // cshuffle_data_type
        {},                  // ds_data_types (no D tensors)
        ckb::DataType::FP32, // output_data_type

        // Elementwise operations
        ckb::ElementwiseOperation::PASS_THROUGH, // input_elementwise_op
        ckb::ElementwiseOperation::PASS_THROUGH, // weight_elementwise_op
        ckb::ElementwiseOperation::PASS_THROUGH, // output_elementwise_op

        // Specializations
        ckb::ConvSpecialization::DEFAULT,    // conv_fwd_specialization
        ckb::GemmSpecialization::MNKPadding, // gemm_specialization

        // Prefetch and block dimensions
        1,  // num_gemm_k_prefetch_stage
        64, // block_size
        64, // m_per_block
        64, // n_per_block
        16, // k_per_block

        // XDL parameters
        4,  // ak1
        4,  // bk1
        32, // m_per_xdl
        32, // n_per_xdl
        2,  // m_xdl_per_wave
        2,  // n_xdl_per_wave

        // A block transfer parameters
        {4, 16, 1}, // a_thread_cluster_lengths
        {1, 0, 2},  // a_thread_cluster_arrange_order
        {1, 0, 2},  // a_block_transfer_src_access_order
        2,          // a_block_transfer_src_vector_dim
        1,          // a_block_transfer_src_scalar_per_vector
        4,          // a_block_transfer_dst_scalar_per_vector_k1
        true,       // a_block_lds_extra_m

        // B block transfer parameters
        {4, 16, 1}, // b_thread_cluster_lengths
        {1, 0, 2},  // b_thread_cluster_arrange_order
        {1, 0, 2},  // b_block_transfer_src_access_order
        2,          // b_block_transfer_src_vector_dim
        1,          // b_block_transfer_src_scalar_per_vector
        4,          // b_block_transfer_dst_scalar_per_vector_k1
        true,       // b_block_lds_extra_n

        // C shuffle parameters
        1,            // c_shuffle_m_xdl_per_wave_per_shuffle
        1,            // c_shuffle_n_xdl_per_wave_per_shuffle
        {1, 8, 1, 8}, // c_thread_cluster_lengths
        1,            // c_block_transfer_scalar_per_vector

        // Compute data types
        ckb::DataType::FP32, // input_compute_type
        ckb::DataType::FP32, // weight_compute_type

        // Optional parameters
        ckb::PipelineScheduler::DEFAULT, // loop_scheduler
        1                                // num_conv_groups_to_merge
    );

static_assert(is_valid_kernel_instance<grouped_conv_2d_fwd_xdl_f32_instance_1>);

// Verify the instance satisfies the concept
static_assert(
    ckb::factory::FwdXdlAlgorithm<decltype(grouped_conv_2d_fwd_xdl_f32_instance_1.algorithm)>);

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
