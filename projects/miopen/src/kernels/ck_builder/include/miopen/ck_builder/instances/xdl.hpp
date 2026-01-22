// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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
        std::size_t block_size;
        struct TileSize
        {
            std::size_t m;
            std::size_t n;
            std::size_t k;
        } tile_size;
    } thread_block;

    static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

    struct GridwiseGemm
    {
        std::size_t ak1;
        std::size_t bk1;
        struct XdlParams
        {
            std::size_t m_per_xdl      = 16;
            std::size_t n_per_xdl      = 16;
            std::size_t m_xdl_per_wave = 4;
            std::size_t n_xdl_per_wave = 1;
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
                std::size_t k0;
                std::size_t m_n;
                std::size_t k1;
            } block_transfer;
            struct LdsTransfer
            {
                std::size_t src_vector_dim;
                std::size_t src_scalar_per_vector;
                std::size_t lds_dst_scalar_per_vector;
                bool is_direct_load;
                bool lds_padding;
            } lds_transfer;
            struct BlockTransferAccessOrder
            {
                std::array<size_t, 3> order{0, 2, 1};
            } thread_cluster_arrange_order;
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
                std::size_t m_block;
                std::size_t m_wave_per_xdl;
                std::size_t n_block;
                std::size_t n_wave_per_xdl;
            } thread_cluster_dims;
            struct Epilogue
            {
                std::size_t m_xdl_per_wave_per_shuffle;
                std::size_t n_per_wave_per_shuffle;
                std::size_t scalar_per_vector;
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

static_assert(ckb::factory::FwdXdlAlgorithm<XdlAlgorithm>);

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