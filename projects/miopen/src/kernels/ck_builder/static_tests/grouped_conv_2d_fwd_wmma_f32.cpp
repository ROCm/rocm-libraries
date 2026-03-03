// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/instance_data/wmma.hpp>
#include <array>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

// Test instance using the DeviceGroupedConvFwdMultipleD_Wmma_CShuffle helper
// This is based on the F16 instance from CK's device_grouped_conv_fwd_wmma_instance.hpp:62
// Note: WMMA instances in CK library are primarily F16. This is an F16 instance for testing.
//
// Original CK instance (F16):
// DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<
//     NDimSpatial, ALayout, BLayout, DsLayout, ELayout,
//     F16, F16, F32, F16, DsDatatype, F16,
//     PassThrough, PassThrough, CDEElementOp,
//     ConvSpec, GemmMNKPadding,
//     1,                // NumGemmKPrefetch
//     128,              // BlockSize
//     64, 64, 32,       // MPerBlock, NPerBlock, KPerBlock
//     8,                // K1
//     16, 16,           // MPerWmma, NPerWmma
//     2, 2,             // MRepeat (m_wmma_per_wave), NRepeat (n_wmma_per_wave)
//     S<4, 32, 1>,      // ABlockTransferThreadClusterLengths_K0_M_K1
//     S<1, 0, 2>,       // ABlockTransferThreadClusterArrangeOrder
//     S<1, 0, 2>,       // ABlockTransferSrcAccessOrder
//     2, 1, 8,          // ABlockTransferSrcVectorDim, SrcScalarPerVector, DstScalarPerVector_K1
//     1,                // ABlockLdsExtraM
//     S<4, 32, 1>,      // BBlockTransferThreadClusterLengths_K0_N_K1
//     S<1, 0, 2>,       // BBlockTransferThreadClusterArrangeOrder
//     S<1, 0, 2>,       // BBlockTransferSrcAccessOrder
//     2, 1, 8,          // BBlockTransferSrcVectorDim, SrcScalarPerVector, DstScalarPerVector_K1
//     1,                // BBlockLdsExtraN
//     1, 1,             // CShuffleMXdlPerWavePerShuffle, CShuffleNXdlPerWavePerShuffle
//     S<1, 32, 1, 4>,   // CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
//     1>                // CBlockTransferScalarPerVector_NPerBlock

constexpr auto grouped_conv_2d_fwd_wmma_f16_instance_1 =
    DeviceGroupedConvFwdMultipleD_Wmma_CShuffle<0>(
        // Spatial dimension and layouts
        2,                        // spatial_dim (2D convolution)
        ckb::TensorLayout::GNHWC, // input_layout
        ckb::TensorLayout::GKYXC, // weight_layout
        {},                       // ds_layouts (no D tensors)
        ckb::TensorLayout::GNHWK, // output_layout

        // Data types (F16 input/output with F32 accumulation)
        ckb::DataType::FP16, // input_data_type
        ckb::DataType::FP16, // weight_data_type
        ckb::DataType::FP32, // acc_data_type
        ckb::DataType::FP16, // cshuffle_data_type
        {},                  // ds_data_types (no D tensors)
        ckb::DataType::FP16, // output_data_type

        // Elementwise operations
        ckb::ElementwiseOperation::PASS_THROUGH, // input_elementwise_op
        ckb::ElementwiseOperation::PASS_THROUGH, // weight_elementwise_op
        ckb::ElementwiseOperation::PASS_THROUGH, // output_elementwise_op

        // Specializations
        ckb::ConvSpecialization::DEFAULT,    // conv_fwd_specialization
        ckb::GemmSpecialization::MNKPadding, // gemm_specialization

        // Prefetch and block dimensions
        1,   // num_gemm_k_prefetch_stage
        128, // block_size
        64,  // m_per_block
        64,  // n_per_block
        32,  // k_per_block

        // WMMA parameters
        8,  // k1
        16, // m_per_wmma
        16, // n_per_wmma
        2,  // m_wmma_per_wave (MRepeat)
        2,  // n_wmma_per_wave (NRepeat)

        // A block transfer parameters
        {4, 32, 1}, // a_thread_cluster_lengths
        {1, 0, 2},  // a_thread_cluster_arrange_order
        {1, 0, 2},  // a_block_transfer_src_access_order
        2,          // a_block_transfer_src_vector_dim
        1,          // a_block_transfer_src_scalar_per_vector
        8,          // a_block_transfer_dst_scalar_per_vector_k1
        true,       // a_block_lds_extra_m

        // B block transfer parameters
        {4, 32, 1}, // b_thread_cluster_lengths
        {1, 0, 2},  // b_thread_cluster_arrange_order
        {1, 0, 2},  // b_block_transfer_src_access_order
        2,          // b_block_transfer_src_vector_dim
        1,          // b_block_transfer_src_scalar_per_vector
        8,          // b_block_transfer_dst_scalar_per_vector_k1
        true,       // b_block_lds_extra_n

        // C shuffle parameters
        1,             // c_shuffle_m_xdl_per_wave_per_shuffle
        1,             // c_shuffle_n_xdl_per_wave_per_shuffle
        {1, 32, 1, 4}, // c_thread_cluster_lengths
        1,             // c_block_transfer_scalar_per_vector

        // Optional parameters
        ckb::PipelineScheduler::DEFAULT, // loop_scheduler
        ckb::PipelineVersion::V1         // pipeline_version
    );

static_assert(is_valid_kernel_instance<grouped_conv_2d_fwd_wmma_f16_instance_1>);

// Verify the instance satisfies the concept
static_assert(
    ckb::factory::FwdWmmaAlgorithm<decltype(grouped_conv_2d_fwd_wmma_f16_instance_1.algorithm)>);

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
