// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/instance_data/xdl.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

struct WmmaAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;
    using PipeSched   = ckb::PipelineScheduler;
    using PipeVer     = ckb::PipelineVersion;

    ThreadBlock thread_block;

    struct GridwiseGemm
    {
        std::size_t k1;
        std::size_t m_per_wmma;
        std::size_t n_per_wmma;
        std::size_t m_wmma_per_wave;
        std::size_t n_wmma_per_wave;
    } gridwise_gemm;

    TransferABC transfer;

    ConvSpecial fwd_specialization;
    GemmSpecial gemm_specialization;

    std::size_t num_gemm_k_prefetch_stages;
    std::size_t num_conv_groups_to_merge;
    PipeSched loop_scheduler;
    PipeVer pipeline_version;

    ElementwiseOps elementwise_ops;
};

static_assert(ckb::factory::FwdWmmaAlgorithm<WmmaAlgorithm>);

// Reuse XdlSignature from xdl.hpp
using WmmaSignature = XdlSignature;

// Struct to hold both signature and algorithm
struct WmmaInstance
{
    WmmaSignature signature;
    WmmaAlgorithm algorithm;
};

// Constexpr function to create WmmaInstance from DeviceGroupedConvFwdMultipleD_Wmma_CShuffle
// template parameters
template <std::size_t NumDTensor>
constexpr WmmaInstance DeviceGroupedConvFwdMultipleD_Wmma_CShuffle(
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
    // 17. NumGemmKPrefetchStage
    std::size_t num_gemm_k_prefetch_stage,
    // 18-21. Block dimensions
    std::size_t block_size,
    std::size_t m_per_block,
    std::size_t n_per_block,
    std::size_t k_per_block,
    // 22. K1 (single value for WMMA, not ak1/bk1)
    std::size_t k1,
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
    // 45. Loop scheduler
    ckb::PipelineScheduler loop_scheduler = ckb::PipelineScheduler::DEFAULT,
    // 46. Pipeline version
    ckb::PipelineVersion pipeline_version = ckb::PipelineVersion::V1)
{
    // TODO: ds_layouts and ds_data_types are not yet stored in the instance data but will be used
    // in future work. They are present now so that the parameter list aligns with the original CK
    // template this function is based on.
    static_assert(NumDTensor == 0,
                  "ds_layouts and ds_data_types are not yet stored in instance data");
    (void)ds_layouts;
    (void)ds_data_types;

    // cshuffle_data_type is not stored because CK Builder derives it internally from the primary
    // data type (see TileConvTensorTypes in conv_tile_tensor_type.hpp).
    (void)cshuffle_data_type;

    // Our project auto-formatting makes this initializer hard to read
    // clang-format off
    return WmmaInstance{
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
                    .k = k_per_block
                }
            },
            .gridwise_gemm = {
                .k1              = k1,
                .m_per_wmma      = m_per_wmma,
                .n_per_wmma      = n_per_wmma,
                .m_wmma_per_wave = m_wmma_per_wave,
                .n_wmma_per_wave = n_wmma_per_wave
            },
            .transfer = {
                .a = {
                    .block_transfer = {
                        .k0  = a_thread_cluster_lengths[0],
                        .m_n = a_thread_cluster_lengths[1],
                        .k1  = a_thread_cluster_lengths[2]
                    },
                    .lds_transfer = {
                        .src_vector_dim            = a_block_transfer_src_vector_dim,
                        .src_scalar_per_vector     = a_block_transfer_src_scalar_per_vector,
                        .lds_dst_scalar_per_vector = a_block_transfer_dst_scalar_per_vector_k1,
                        .is_direct_load            = false,
                        .lds_padding               = a_block_lds_extra_m
                    },
                    .thread_cluster_arrange_order = {
                        .order = a_thread_cluster_arrange_order
                    },
                    .src_access_order = {
                        .order = a_block_transfer_src_access_order
                    }
                },
                .b = {
                    .block_transfer = {
                        .k0  = b_thread_cluster_lengths[0],
                        .m_n = b_thread_cluster_lengths[1],
                        .k1  = b_thread_cluster_lengths[2]
                    },
                    .lds_transfer = {
                        .src_vector_dim            = b_block_transfer_src_vector_dim,
                        .src_scalar_per_vector     = b_block_transfer_src_scalar_per_vector,
                        .lds_dst_scalar_per_vector = b_block_transfer_dst_scalar_per_vector_k1,
                        .is_direct_load            = false,
                        .lds_padding               = b_block_lds_extra_n
                    },
                    .thread_cluster_arrange_order = {
                        .order = b_thread_cluster_arrange_order
                    },
                    .src_access_order = {
                        .order = b_block_transfer_src_access_order
                    }
                },
                .c = {
                    .thread_cluster_dims = {
                        .m_block        = c_thread_cluster_lengths[0],
                        .m_wave_per_xdl = c_thread_cluster_lengths[1],
                        .n_block        = c_thread_cluster_lengths[2],
                        .n_wave_per_xdl = c_thread_cluster_lengths[3]
                    },
                    .epilogue = {
                        .m_xdl_per_wave_per_shuffle = c_shuffle_m_xdl_per_wave_per_shuffle,
                        .n_per_wave_per_shuffle     = c_shuffle_n_xdl_per_wave_per_shuffle,
                        .scalar_per_vector          = c_block_transfer_scalar_per_vector
                    }
                }
            },
            .fwd_specialization         = conv_fwd_specialization,
            .gemm_specialization        = gemm_specialization,
            .num_gemm_k_prefetch_stages = num_gemm_k_prefetch_stage,
            .num_conv_groups_to_merge   = 1,
            .loop_scheduler             = loop_scheduler,
            .pipeline_version           = pipeline_version,
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
