// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <miopen/ck_builder/shared.hpp>

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {
namespace ckb = ck_tile::builder;

struct ThreadBlock
{
    std::size_t block_size;
    struct TileSize
    {
        std::size_t m;
        std::size_t n;
        std::size_t k;
    } tile_size;
};

static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

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
};

// Elementwise operations applied during convolution
// - input_op: Applied to input tensor (A) before GEMM
// - weight_op: Applied to weight tensor (B) before GEMM
// - output_op: Applied to output tensor (CDE) after GEMM (epilogue)
struct ElementwiseOps
{
    ckb::ElementwiseOperation input_op;
    ckb::ElementwiseOperation weight_op;
    ckb::ElementwiseOperation output_op;
};

struct TensorDescriptor
{
    struct Config
    {
        ckb::TensorLayout layout;
        ckb::DataType data_type;
        ckb::DataType compute_type;
    } config;
};

struct ConvSignature
{
    std::size_t spatial_dim;
    ckb::ConvDirection direction;
    TensorDescriptor input;
    TensorDescriptor weight;
    TensorDescriptor output;
    ckb::DataType data_type;
    ckb::DataType accumulation_data_type;
};

} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
