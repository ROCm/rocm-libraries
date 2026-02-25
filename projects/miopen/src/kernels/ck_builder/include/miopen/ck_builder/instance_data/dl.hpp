// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/instance_data/common.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

struct DlAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;

    ThreadBlock thread_block;

    // DL-specific thread configuration
    struct ThreadConfig
    {
        std::size_t k0_per_block;
        std::size_t k1;
        std::size_t m1_per_thread;
        std::size_t n1_per_thread;
        std::size_t k_per_thread;
    } thread_config;

    static_assert(ckb::DlThreadConfigDescriptor<ThreadConfig>);

    // DL-specific thread cluster
    struct ThreadCluster
    {
        std::array<size_t, 2> m1_xs;
        std::array<size_t, 2> n1_xs;
    } thread_cluster;

    static_assert(ckb::DlThreadClusterDescriptor<ThreadCluster>);

    // DL-specific transfer descriptors
    struct Transfer
    {
        // DL block transfer for A (4D: K0_M0_M1_K1)
        struct DlBlockTransfer4D
        {
            std::array<size_t, 4> thread_slice_lengths;
            std::array<size_t, 4> thread_cluster_lengths;
            std::array<size_t, 4> thread_cluster_arrange_order;
            std::array<size_t, 4> src_access_order;
            std::array<size_t, 4> src_vector_tensor_lengths;
            std::array<size_t, 4> src_vector_tensor_contiguous_dim_order;
            std::array<size_t, 4> dst_vector_tensor_lengths;
        };

        static_assert(ckb::DlBlockTransferDescriptor4D<DlBlockTransfer4D>);

        DlBlockTransfer4D a;
        DlBlockTransfer4D b;

        // DL epilogue descriptor for C (6D)
        struct DlEpilogue
        {
            std::array<size_t, 6> src_dst_access_order;
            std::size_t src_dst_vector_dim;
            std::size_t dst_scalar_per_vector;
        } c;

        static_assert(ckb::DlEpilogueDescriptor<DlEpilogue>);
    } transfer;

    // Specializations
    ConvSpecial fwd_specialization;
    GemmSpecial gemm_specialization;

    ElementwiseOps elementwise_ops;
};

static_assert(ckb::factory::FwdDlAlgorithm<DlAlgorithm>);

using DlSignature = ConvSignature;

// Struct to hold both signature and algorithm
struct DlInstance
{
    DlSignature signature;
    DlAlgorithm algorithm;
};

// Constexpr function to create DlInstance from old DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK
// template parameters Parameters are in the same order as the template parameters
template <std::size_t NumDTensor>
constexpr DlInstance DeviceGroupedConvFwdDlMultipleD_NHWC_KYXC_NHWK(
    // 1. NDimSpatial
    std::size_t spatial_dim,
    // 2-5. Data types
    ckb::DataType input_data_type,
    ckb::DataType weight_data_type,
    const std::array<ckb::DataType, NumDTensor>& ds_data_types,
    ckb::DataType output_data_type,
    ckb::DataType acc_data_type,
    // 6-9. Layouts
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    const std::array<ckb::TensorLayout, NumDTensor>& ds_layouts,
    ckb::TensorLayout output_layout,
    // 10-12. Elementwise operations
    ckb::ElementwiseOperation input_elementwise_op,
    ckb::ElementwiseOperation weight_elementwise_op,
    ckb::ElementwiseOperation output_elementwise_op,
    // 13-14. Specializations
    ckb::ConvSpecialization conv_fwd_specialization,
    ckb::GemmSpecialization gemm_specialization,
    // 15. Block size
    std::size_t block_size,
    // 16-18. Block dimensions
    std::size_t m_per_block,
    std::size_t n_per_block,
    std::size_t k0_per_block,
    // 19. K1
    std::size_t k1,
    // 20-22. Thread parameters
    std::size_t m1_per_thread,
    std::size_t n1_per_thread,
    std::size_t k_per_thread,
    // 23-24. Thread cluster
    std::array<std::size_t, 2> m1n1_thread_cluster_m1_xs,
    std::array<std::size_t, 2> m1n1_thread_cluster_n1_xs,
    // 25-31. A Block Transfer (K0_M0_M1_K1)
    std::array<std::size_t, 4> a_block_transfer_thread_slice_lengths,
    std::array<std::size_t, 4> a_block_transfer_thread_cluster_lengths,
    std::array<std::size_t, 4> a_block_transfer_thread_cluster_arrange_order,
    std::array<std::size_t, 4> a_block_transfer_src_access_order,
    std::array<std::size_t, 4> a_block_transfer_src_vector_tensor_lengths,
    std::array<std::size_t, 4> a_block_transfer_src_vector_tensor_contiguous_dim_order,
    std::array<std::size_t, 4> a_block_transfer_dst_vector_tensor_lengths,
    // 32-38. B Block Transfer (K0_N0_N1_K1)
    std::array<std::size_t, 4> b_block_transfer_thread_slice_lengths,
    std::array<std::size_t, 4> b_block_transfer_thread_cluster_lengths,
    std::array<std::size_t, 4> b_block_transfer_thread_cluster_arrange_order,
    std::array<std::size_t, 4> b_block_transfer_src_access_order,
    std::array<std::size_t, 4> b_block_transfer_src_vector_tensor_lengths,
    std::array<std::size_t, 4> b_block_transfer_src_vector_tensor_contiguous_dim_order,
    std::array<std::size_t, 4> b_block_transfer_dst_vector_tensor_lengths,
    // 39-41. C Thread Transfer
    std::array<std::size_t, 6> c_thread_transfer_src_dst_access_order,
    std::size_t c_thread_transfer_src_dst_vector_dim,
    std::size_t c_thread_transfer_dst_scalar_per_vector)
{
    // TODO: ds_layouts and ds_data_types are not yet stored in the instance data but will be used
    // in future work. They are present now so that the parameter list aligns with the original CK
    // template this function is based on.
    static_assert(NumDTensor == 0,
                  "ds_layouts and ds_data_types are not yet stored in instance data");
    (void)ds_data_types;
    (void)ds_layouts;

    // Our project auto-formatting makes this initializer hard to read
    // clang-format off
    return DlInstance{
        .signature = {
            .spatial_dim            = spatial_dim,
            .direction              = ckb::ConvDirection::FORWARD,
            .input                  = {
                .config = {
                    .layout       = input_layout,
                    .data_type    = input_data_type,
                    .compute_type = input_data_type
                }
            },
            .weight = {
                .config = {
                    .layout       = weight_layout,
                    .data_type    = weight_data_type,
                    .compute_type = weight_data_type
                }
            },
            .output = {
                .config = {
                    .layout       = output_layout,
                    .data_type    = output_data_type,
                    .compute_type = output_data_type
                }
            },
            .data_type              = input_data_type,
            .accumulation_data_type = acc_data_type
        },
        .algorithm = {
            .thread_block = {
                .block_size = block_size,
                .tile_size  = {
                    .m = m_per_block,
                    .n = n_per_block,
                    .k = k0_per_block
                }
            },
            .thread_config = {
                .k0_per_block  = k0_per_block,
                .k1            = k1,
                .m1_per_thread = m1_per_thread,
                .n1_per_thread = n1_per_thread,
                .k_per_thread  = k_per_thread
            },
            .thread_cluster = {
                .m1_xs = m1n1_thread_cluster_m1_xs,
                .n1_xs = m1n1_thread_cluster_n1_xs
            },
            .transfer = {
                .a = {
                    .thread_slice_lengths                    = a_block_transfer_thread_slice_lengths,
                    .thread_cluster_lengths                  = a_block_transfer_thread_cluster_lengths,
                    .thread_cluster_arrange_order            = a_block_transfer_thread_cluster_arrange_order,
                    .src_access_order                        = a_block_transfer_src_access_order,
                    .src_vector_tensor_lengths               = a_block_transfer_src_vector_tensor_lengths,
                    .src_vector_tensor_contiguous_dim_order  = a_block_transfer_src_vector_tensor_contiguous_dim_order,
                    .dst_vector_tensor_lengths               = a_block_transfer_dst_vector_tensor_lengths
                },
                .b = {
                    .thread_slice_lengths                    = b_block_transfer_thread_slice_lengths,
                    .thread_cluster_lengths                  = b_block_transfer_thread_cluster_lengths,
                    .thread_cluster_arrange_order            = b_block_transfer_thread_cluster_arrange_order,
                    .src_access_order                        = b_block_transfer_src_access_order,
                    .src_vector_tensor_lengths               = b_block_transfer_src_vector_tensor_lengths,
                    .src_vector_tensor_contiguous_dim_order  = b_block_transfer_src_vector_tensor_contiguous_dim_order,
                    .dst_vector_tensor_lengths               = b_block_transfer_dst_vector_tensor_lengths
                },
                .c = {
                    .src_dst_access_order     = c_thread_transfer_src_dst_access_order,
                    .src_dst_vector_dim       = c_thread_transfer_src_dst_vector_dim,
                    .dst_scalar_per_vector    = c_thread_transfer_dst_scalar_per_vector
                }
            },
            .fwd_specialization  = conv_fwd_specialization,
            .gemm_specialization = gemm_specialization,
            .elementwise_ops = {
                .input_op  = input_elementwise_op,
                .weight_op = weight_elementwise_op,
                .output_op = output_elementwise_op
            }
        }
    };
    // clang-format on
}

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
