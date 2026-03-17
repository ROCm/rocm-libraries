// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/shared.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

// Stub for unsupported DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3
// This template class is not supported by CK Builder.
template <std::size_t NumDTensor>
constexpr int DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3(
    // 1. NDimSpatial
    std::size_t spatial_dim,
    // 2-5. Layouts
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    // 6-11. Data types
    ckb::DataType input_data_type,
    ckb::DataType weight_data_type,
    ckb::DataType acc_data_type,
    ckb::DataType cshuffle_data_type,
    const std::array<ckb::DataType, NumDTensor>& ds_data_types,
    ckb::DataType output_data_type,
    // 12-14. Elementwise operations
    ckb::ElementwiseOperation input_elementwise_op,
    ckb::ElementwiseOperation weight_elementwise_op,
    ckb::ElementwiseOperation output_elementwise_op,
    // 15-16. Specializations
    ckb::ConvSpecialization conv_fwd_specialization,
    ckb::GemmSpecialization gemm_specialization,
    // 17-20. Block dimensions
    std::size_t block_size,
    std::size_t m_per_block,
    std::size_t n_per_block,
    std::size_t k_per_block,
    // 21-22. AK1/BK1
    std::size_t ak1,
    std::size_t bk1,
    // 23-26. WMMA parameters
    std::size_t m_per_wmma,
    std::size_t n_per_wmma,
    std::size_t m_wmma_per_wave,
    std::size_t n_wmma_per_wave,
    // 27-33. A block transfer parameters
    std::array<std::size_t, 3> a_thread_cluster_lengths,
    std::array<std::size_t, 3> a_thread_cluster_arrange_order,
    std::array<std::size_t, 3> a_block_transfer_src_access_order,
    std::size_t a_block_transfer_src_vector_dim,
    std::size_t a_block_transfer_src_scalar_per_vector,
    std::size_t a_block_transfer_dst_scalar_per_vector_k1,
    bool a_block_lds_extra_m,
    // 34-40. B block transfer parameters
    std::array<std::size_t, 3> b_thread_cluster_lengths,
    std::array<std::size_t, 3> b_thread_cluster_arrange_order,
    std::array<std::size_t, 3> b_block_transfer_src_access_order,
    std::size_t b_block_transfer_src_vector_dim,
    std::size_t b_block_transfer_src_scalar_per_vector,
    std::size_t b_block_transfer_dst_scalar_per_vector_k1,
    bool b_block_lds_extra_n,
    // 41-44. C shuffle parameters
    std::size_t c_shuffle_m_xdl_per_wave_per_shuffle,
    std::size_t c_shuffle_n_xdl_per_wave_per_shuffle,
    std::array<std::size_t, 4> c_thread_cluster_lengths,
    std::size_t c_block_transfer_scalar_per_vector,
    // 45-46. Pipeline parameters
    ckb::PipelineScheduler loop_scheduler,
    ckb::PipelineVersion pipeline_version,
    // 47. Use thread tile transfer
    bool use_thread_tile_transfer = true,
    // 48-49. Compute data types
    ckb::DataType input_compute_type  = ckb::DataType::UNDEFINED_DATA_TYPE,
    ckb::DataType weight_compute_type = ckb::DataType::UNDEFINED_DATA_TYPE,
    // 50. Groups to merge
    std::size_t num_conv_groups_to_merge = 1)
{
    // Use dependent false to ensure error only triggers on instantiation
    static_assert(
        NumDTensor != NumDTensor,
        "DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3 is not supported by CK Builder");
    return 0;
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
