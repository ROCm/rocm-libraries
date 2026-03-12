// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/ck_builder/instance_data/dl.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

// First simple F32 instance from device_grouped_conv_fwd_dl_instance.hpp
// DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<2, F32, F32, DsDatatype, F32, F32, InLayout,
// WeiLayout, DsLayout, OutLayout, PassThrough, PassThrough, CDEElementOp, ConvSpec, GemmMNKPadding,
//     8,    16,      4,    2,  1,          1,          2,      1,       S<4, 2>,       S<1, 1>,
//     S<2, 1, 2, 1>,      S<1, 1, 8, 1>,    S<1, 2, 0, 3>,  S<1, 2, 0, 3>,       S<1, 1, 1, 1>,
//     S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 1, 1, 1>,        S<2, 1, 4, 1>,  S<1, 2, 0, 3>,
//     S<1, 2, 0, 3>,       S<1, 1, 1, 1>,      S<1, 2, 0, 3>,       S<1, 1, 1, 1>, S<0, 1, 2, 3, 4,
//     5>,               5,                 1>

[[maybe_unused]] constexpr auto instance_grouped_conv_2d_fwd_dl_f32_1 =
    DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK<0>(
        // Spatial dimension
        2,
        // Data types
        ckb::DataType::FP32, // input_data_type
        ckb::DataType::FP32, // weight_data_type
        {},                  // ds_data_types (empty for NumDTensor=0)
        ckb::DataType::FP32, // output_data_type
        ckb::DataType::FP32, // acc_data_type
        // Layouts
        ckb::TensorLayout::GNHWC, // input_layout
        ckb::TensorLayout::GKYXC, // weight_layout
        {},                       // ds_layouts (empty for NumDTensor=0)
        ckb::TensorLayout::GNHWK, // output_layout
        // Elementwise operations
        ckb::ElementwiseOperation::PASS_THROUGH, // input_elementwise_op
        ckb::ElementwiseOperation::PASS_THROUGH, // weight_elementwise_op
        ckb::ElementwiseOperation::PASS_THROUGH, // output_elementwise_op
        // Specializations
        ckb::ConvSpecialization::DEFAULT,    // conv_fwd_specialization
        ckb::GemmSpecialization::MNKPadding, // gemm_specialization
        // Block size and dimensions
        8,  // block_size
        16, // m_per_block
        4,  // n_per_block
        2,  // k0_per_block
        1,  // k1
        // Thread parameters
        1, // m1_per_thread
        2, // n1_per_thread
        1, // k_per_thread
        // Thread cluster
        {4, 2}, // m1n1_thread_cluster_m1_xs
        {1, 1}, // m1n1_thread_cluster_n1_xs
        // A Block Transfer (K0_M0_M1_K1)
        {2, 1, 2, 1}, // a_block_transfer_thread_slice_lengths
        {1, 1, 8, 1}, // a_block_transfer_thread_cluster_lengths
        {1, 2, 0, 3}, // a_block_transfer_thread_cluster_arrange_order
        {1, 2, 0, 3}, // a_block_transfer_src_access_order
        {1, 1, 1, 1}, // a_block_transfer_src_vector_tensor_lengths
        {1, 2, 0, 3}, // a_block_transfer_src_vector_tensor_contiguous_dim_order
        {1, 1, 1, 1}, // a_block_transfer_dst_vector_tensor_lengths
        // B Block Transfer (K0_N0_N1_K1)
        {1, 1, 1, 1}, // b_block_transfer_thread_slice_lengths
        {2, 1, 4, 1}, // b_block_transfer_thread_cluster_lengths
        {1, 2, 0, 3}, // b_block_transfer_thread_cluster_arrange_order
        {1, 2, 0, 3}, // b_block_transfer_src_access_order
        {1, 1, 1, 1}, // b_block_transfer_src_vector_tensor_lengths
        {1, 2, 0, 3}, // b_block_transfer_src_vector_tensor_contiguous_dim_order
        {1, 1, 1, 1}, // b_block_transfer_dst_vector_tensor_lengths
        // C Thread Transfer
        {0, 1, 2, 3, 4, 5}, // c_thread_transfer_src_dst_access_order
        5,                  // c_thread_transfer_src_dst_vector_dim
        1                   // c_thread_transfer_dst_scalar_per_vector
    );

static_assert(is_valid_kernel_instance<instance_grouped_conv_2d_fwd_dl_f32_1>);

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
