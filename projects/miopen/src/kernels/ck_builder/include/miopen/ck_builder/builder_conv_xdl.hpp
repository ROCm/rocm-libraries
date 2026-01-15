// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <ck_tile/builder/types.hpp>
#include <ck_tile/builder/conv_builder.hpp>

namespace ckb = ck_tile::builder;

struct XdlAlgorithm
{
    using ConvSpecial = ckb::ConvSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;
    using PipeSched   = ckb::PipelineScheduler;

    struct ThreadBlock
    {
        unsigned int block_size;
        struct TileSize
        {
            unsigned int m;
            unsigned int n;
            unsigned int k;
        } tile_size;
    } thread_block;

    static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

    struct GridwiseGemm
    {
        unsigned int ak1;
        unsigned int bk1;
        struct XdlParams
        {
            unsigned int m_per_xdl      = 16;
            unsigned int n_per_xdl      = 16;
            unsigned int m_xdl_per_wave = 4;
            unsigned int n_xdl_per_wave = 1;
        } xdl_params;
        static_assert(ckb::GridwiseXdlGemmDescriptor<XdlParams>);
    } gridwise_gemm;

    static_assert(ckb::GridwiseFwdXdlGemmDescriptor<GridwiseGemm>);
    struct TransferABC
    {
        struct TransferAB
        {
            struct BlockTransfer
            {
                unsigned int k0;
                unsigned int m_n;
                unsigned int k1;
            } block_transfer;
            struct LdsTransfer
            {
                unsigned int src_vector_dim;
                unsigned int src_scalar_per_vector;
                unsigned int lds_dst_scalar_per_vector;
                bool is_direct_load;
                bool lds_padding;
            } lds_transfer;
            struct BlockTransferAccessOrder
            {
                std::array<size_t, 3> order{0, 2, 1};
            } block_transfer_access_order;
            struct SrcAccessOrder
            {
                std::array<size_t, 3> order{0, 2, 1};
            } src_access_order;
        };
        TransferAB a;
        TransferAB b;
        struct TransferC
        {
            struct ThreadClusterDims
            {
                unsigned int m_block;
                unsigned int m_wave_per_xdl;
                unsigned int n_block;
                unsigned int n_wave_per_xdl;
            } thread_cluster_dims;
            struct Epilogue
            {
                unsigned int m_xdl_per_wave_per_shuffle;
                unsigned int n_per_wave_per_shuffle;
                unsigned int scalar_per_vector;
            } epilogue;
        } c;
    } transfer;

    // TODO: Fix CK Builder schema to not require these defaults.
    ConvSpecial fwd_specialization;
    GemmSpecial gemm_specialization;

    std::size_t num_gemm_k_prefetch_stages;
    std::size_t num_conv_groups_to_merge;
    PipeSched loop_scheduler;
};

struct XdlSignature
{
    int spatial_dim;
    ckb::ConvDirection direction;
    struct InputTensorDescriptor
    {
        struct Config
        {
            ckb::TensorLayout layout;
            ckb::DataType data_type;
            ckb::DataType compute_type;
        } config;
    } input;

    struct WeightTensorDescriptor
    {
        struct Config
        {
            ckb::TensorLayout layout;
            ckb::DataType data_type;
            ckb::DataType compute_type;
        } config;
    } weight;

    struct OutputTensorDescriptor
    {
        struct Config
        {
            ckb::TensorLayout layout;
            ckb::DataType data_type;
            ckb::DataType compute_type;
        } config;
    } output;
    ckb::DataType data_type;
    ckb::DataType accumulation_data_type;
};

// Struct to hold both signature and algorithm
struct XdlInstance
{
    XdlSignature signature;
    XdlAlgorithm algorithm;
};

// Constexpr function to create XdlInstance from old DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
// template parameters Parameters are in the same order as the template parameters
constexpr XdlInstance make_xdl_instance_from_old_params(
    // 1. NDimSpatial
    unsigned int spatial_dim,
    // 2-5. Layouts
    ckb::TensorLayout input_layout,
    ckb::TensorLayout weight_layout,
    ckb::TensorLayout output_layout,
    // 6-11. Data types
    ckb::DataType input_data_type,
    ckb::DataType weight_data_type,
    ckb::DataType acc_data_type,
    ckb::DataType cshuffle_data_type,
    ckb::DataType output_data_type,
    // 12-14. Elementwise operations (not stored in XdlSignature/XdlAlgorithm currently)
    // 15-16. Specializations
    ckb::ConvSpecialization conv_fwd_specialization,
    ckb::GemmSpecialization gemm_specialization,
    // 17. NumGemmKPrefetchStage
    unsigned int num_gemm_k_prefetch_stage,
    // 18-21. Block dimensions
    unsigned int block_size,
    unsigned int m_per_block,
    unsigned int n_per_block,
    unsigned int k_per_block,
    // 22-27. XDL parameters
    unsigned int ak1,
    unsigned int bk1,
    unsigned int m_per_xdl,
    unsigned int n_per_xdl,
    unsigned int m_xdl_per_wave,
    unsigned int n_xdl_per_wave,
    // 28-34. A block transfer parameters
    std::array<std::size_t, 3> a_thread_cluster_lengths,
    std::array<std::size_t, 3> a_thread_cluster_arrange_order,
    std::array<std::size_t, 3> a_block_transfer_src_access_order,
    unsigned int a_block_transfer_src_vector_dim,
    unsigned int a_block_transfer_src_scalar_per_vector,
    unsigned int a_block_transfer_dst_scalar_per_vector_k1,
    bool a_block_lds_extra_m,
    // 35-41. B block transfer parameters
    std::array<std::size_t, 3> b_thread_cluster_lengths,
    std::array<std::size_t, 3> b_thread_cluster_arrange_order,
    std::array<std::size_t, 3> b_block_transfer_src_access_order,
    unsigned int b_block_transfer_src_vector_dim,
    unsigned int b_block_transfer_src_scalar_per_vector,
    unsigned int b_block_transfer_dst_scalar_per_vector_k1,
    bool b_block_lds_extra_n,
    // 42-45. C shuffle parameters
    unsigned int c_shuffle_m_xdl_per_wave_per_shuffle,
    unsigned int c_shuffle_n_xdl_per_wave_per_shuffle,
    std::array<std::size_t, 4> c_thread_cluster_lengths,
    unsigned int c_block_transfer_scalar_per_vector,
    // 46-47. Compute data types
    ckb::DataType input_compute_type,
    ckb::DataType weight_compute_type,
    // 48. Loop scheduler
    ckb::PipelineScheduler loop_scheduler,
    // 49. Groups to merge
    unsigned int num_conv_groups_to_merge)
{
    return XdlInstance{
        .signature = {.spatial_dim            = spatial_dim,
                      .direction              = ckb::ConvDirection::FORWARD,
                      .input                  = {.config = {.layout       = input_layout,
                                           .data_type    = input_data_type,
                                           .compute_type = input_compute_type}},
                      .weight                 = {.config = {.layout       = weight_layout,
                                            .data_type    = weight_data_type,
                                            .compute_type = weight_compute_type}},
                      .output                 = {.config =
                                     {
                                         .layout       = output_layout,
                                         .data_type    = output_data_type,
                                         .compute_type = output_data_type // Output compute type
                                                                          // same as data type
                                     }},
                      .data_type              = input_data_type,
                      .accumulation_data_type = acc_data_type},
        .algorithm =
            {.thread_block  = {.block_size = block_size,
                              .tile_size  = {.m = m_per_block, .n = n_per_block, .k = k_per_block}},
             .gridwise_gemm = {.ak1 = ak1,
                               .bk1 = bk1,
                               .xdl_params{.m_per_xdl      = m_per_xdl,
                                           .n_per_xdl      = n_per_xdl,
                                           .m_xdl_per_wave = m_xdl_per_wave,
                                           .n_xdl_per_wave = n_xdl_per_wave}},
             .transfer =
                 {.a = {.block_transfer = {.k0  = static_cast<int>(a_thread_cluster_lengths[0]),
                                           .m_n = static_cast<int>(a_thread_cluster_lengths[1]),
                                           .k1  = static_cast<int>(a_thread_cluster_lengths[2])},
                        .lds_transfer   = {.src_vector_dim = a_block_transfer_src_vector_dim,
                                         .src_scalar_per_vector =
                                             a_block_transfer_src_scalar_per_vector,
                                         .lds_dst_scalar_per_vector =
                                             a_block_transfer_dst_scalar_per_vector_k1,
                                         .is_direct_load = false,
                                         .lds_padding    = a_block_lds_extra_m},
                        .block_transfer_access_order = {.order = a_thread_cluster_arrange_order},
                        .src_access_order = {.order = a_block_transfer_src_access_order}},
                  .b = {.block_transfer = {.k0  = static_cast<int>(b_thread_cluster_lengths[0]),
                                           .m_n = static_cast<int>(b_thread_cluster_lengths[1]),
                                           .k1  = static_cast<int>(b_thread_cluster_lengths[2])},
                        .lds_transfer   = {.src_vector_dim = b_block_transfer_src_vector_dim,
                                         .src_scalar_per_vector =
                                             b_block_transfer_src_scalar_per_vector,
                                         .lds_dst_scalar_per_vector =
                                             b_block_transfer_dst_scalar_per_vector_k1,
                                         .is_direct_load = false,
                                         .lds_padding    = b_block_lds_extra_n},
                        .block_transfer_access_order = {.order = b_thread_cluster_arrange_order},
                        .src_access_order = {.order = b_block_transfer_src_access_order}},
                  .c = {.thread_cluster_dims =
                            {.m_block        = static_cast<int>(c_thread_cluster_lengths[0]),
                             .m_wave_per_xdl = static_cast<int>(c_thread_cluster_lengths[1]),
                             .n_block        = static_cast<int>(c_thread_cluster_lengths[2]),
                             .n_wave_per_xdl = static_cast<int>(c_thread_cluster_lengths[3])},
                        .epilogue = {.m_xdl_per_wave_per_shuffle =
                                         c_shuffle_m_xdl_per_wave_per_shuffle,
                                     .n_per_wave_per_shuffle = c_shuffle_n_xdl_per_wave_per_shuffle,
                                     .scalar_per_vector = c_block_transfer_scalar_per_vector}}},
             .fwd_specialization         = conv_fwd_specialization,
             .gemm_specialization        = gemm_specialization,
             .num_gemm_k_prefetch_stages = static_cast<std::size_t>(num_gemm_k_prefetch_stage),
             .num_conv_groups_to_merge   = static_cast<std::size_t>(num_conv_groups_to_merge),
             .loop_scheduler             = loop_scheduler}};
}
