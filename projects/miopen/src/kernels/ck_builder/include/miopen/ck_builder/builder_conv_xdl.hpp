// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <ck_tile/builder/types.hpp>
#include <ck_tile/builder/conv_builder.hpp>
#include <miopen/ck_builder/builder_factory.hpp>

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

template <auto KernelDescriptor>
constexpr void instantiate_kernel(std::vector<BaseOperatorPtr>& kernels)
{
    using Builder = ckb::ConvBuilder<KernelDescriptor.signature, KernelDescriptor.algorithm>;
    do_builder_checks<Builder>();

    kernels.push_back(std::make_unique<typename Builder::Instance>());
}

template <typename T, T... values>
constexpr void build_kernels_helper(std::vector<BaseOperatorPtr>& kernels)
{
    std::array<BaseOperatorPtr, sizeof...(values)> result{};
    ((instantiate_kernel<values>(kernels)), ...);
}

template <typename T, std::size_t N, std::array<T, N> arr, std::size_t... I>
constexpr void build_kernels_impl(std::vector<BaseOperatorPtr>& kernels, std::index_sequence<I...>)
{
    build_kernels_helper<T, arr[I]...>(kernels);
}

template <typename ArrayType>
struct array_traits;

template <typename T, std::size_t N>
struct array_traits<std::array<T, N>>
{
    using value_type                  = T;
    static constexpr std::size_t size = N;
};

template <auto arr>
constexpr void build_kernels(std::vector<BaseOperatorPtr>& kernels)
{
    using T                 = typename array_traits<decltype(arr)>::value_type;
    constexpr std::size_t N = array_traits<decltype(arr)>::size;
    build_kernels_impl<T, N, arr>(kernels, std::make_index_sequence<N>{});
}

template <typename T, std::size_t N1, std::size_t N2, std::size_t... I1, std::size_t... I2>
constexpr std::array<T, N1 + N2> concat2_impl(const std::array<T, N1>& a,
                                              const std::array<T, N2>& b,
                                              std::index_sequence<I1...>,
                                              std::index_sequence<I2...>)
{
    return {a[I1]..., b[I2]...};
}

template <typename T, std::size_t N1, std::size_t N2>
constexpr std::array<T, N1 + N2> concat2(const std::array<T, N1>& a, const std::array<T, N2>& b)
{
    return concat2_impl(a, b, std::make_index_sequence<N1>{}, std::make_index_sequence<N2>{});
}

// Variadic: concatenate many arrays recursively
template <typename T, std::size_t N>
constexpr std::array<T, N> concat(const std::array<T, N>& a)
{
    return a;
}

template <typename T, std::size_t N1, std::size_t N2, std::size_t... Ns>
constexpr auto
concat(const std::array<T, N1>& a, const std::array<T, N2>& b, const std::array<T, Ns>&... rest)
{
    return concat(concat2(a, b), rest...);
}

// Constexpr function to create XdlInstance from old DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle
// template parameters Parameters are in the same order as the template parameters
constexpr XdlInstance make_xdl_instance_from_old_params(
    // 1. NDimSpatial
    std::size_t spatial_dim,
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
    std::size_t num_gemm_k_prefetch_stage,
    // 18-21. Block dimensions
    std::size_t block_size,
    std::size_t m_per_block,
    std::size_t n_per_block,
    std::size_t k_per_block,
    // 22-27. XDL parameters
    std::size_t ak1,
    std::size_t bk1,
    std::size_t m_per_xdl,
    std::size_t n_per_xdl,
    std::size_t m_xdl_per_wave,
    std::size_t n_xdl_per_wave,
    // 28-34. A block transfer parameters
    std::array<std::size_t, 3> a_thread_cluster_lengths,
    std::array<std::size_t, 3> a_thread_cluster_arrange_order,
    std::array<std::size_t, 3> a_block_transfer_src_access_order,
    std::size_t a_block_transfer_src_vector_dim,
    std::size_t a_block_transfer_src_scalar_per_vector,
    std::size_t a_block_transfer_dst_scalar_per_vector_k1,
    bool a_block_lds_extra_m,
    // 35-41. B block transfer parameters
    std::array<std::size_t, 3> b_thread_cluster_lengths,
    std::array<std::size_t, 3> b_thread_cluster_arrange_order,
    std::array<std::size_t, 3> b_block_transfer_src_access_order,
    std::size_t b_block_transfer_src_vector_dim,
    std::size_t b_block_transfer_src_scalar_per_vector,
    std::size_t b_block_transfer_dst_scalar_per_vector_k1,
    bool b_block_lds_extra_n,
    // 42-45. C shuffle parameters
    std::size_t c_shuffle_m_xdl_per_wave_per_shuffle,
    std::size_t c_shuffle_n_xdl_per_wave_per_shuffle,
    std::array<std::size_t, 4> c_thread_cluster_lengths,
    std::size_t c_block_transfer_scalar_per_vector,
    // 46-47. Compute data types
    ckb::DataType input_compute_type,
    ckb::DataType weight_compute_type,
    // 48. Loop scheduler
    ckb::PipelineScheduler loop_scheduler = ckb::PipelineScheduler::DEFAULT,
    // 49. Groups to merge
    std::size_t num_conv_groups_to_merge = 1)
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
                 {.a = {.block_transfer = {.k0  = a_thread_cluster_lengths[0],
                                           .m_n = a_thread_cluster_lengths[1],
                                           .k1  = a_thread_cluster_lengths[2]},
                        .lds_transfer   = {.src_vector_dim = a_block_transfer_src_vector_dim,
                                         .src_scalar_per_vector =
                                             a_block_transfer_src_scalar_per_vector,
                                         .lds_dst_scalar_per_vector =
                                             a_block_transfer_dst_scalar_per_vector_k1,
                                         .is_direct_load = false,
                                         .lds_padding    = a_block_lds_extra_m},
                        .block_transfer_access_order = {.order = a_thread_cluster_arrange_order},
                        .src_access_order = {.order = a_block_transfer_src_access_order}},
                  .b = {.block_transfer = {.k0  = b_thread_cluster_lengths[0],
                                           .m_n = b_thread_cluster_lengths[1],
                                           .k1  = b_thread_cluster_lengths[2]},
                        .lds_transfer   = {.src_vector_dim = b_block_transfer_src_vector_dim,
                                         .src_scalar_per_vector =
                                             b_block_transfer_src_scalar_per_vector,
                                         .lds_dst_scalar_per_vector =
                                             b_block_transfer_dst_scalar_per_vector_k1,
                                         .is_direct_load = false,
                                         .lds_padding    = b_block_lds_extra_n},
                        .block_transfer_access_order = {.order = b_thread_cluster_arrange_order},
                        .src_access_order = {.order = b_block_transfer_src_access_order}},
                  .c = {.thread_cluster_dims = {.m_block        = c_thread_cluster_lengths[0],
                                                .m_wave_per_xdl = c_thread_cluster_lengths[1],
                                                .n_block        = c_thread_cluster_lengths[2],
                                                .n_wave_per_xdl = c_thread_cluster_lengths[3]},
                        .epilogue            = {.m_xdl_per_wave_per_shuffle =
                                         c_shuffle_m_xdl_per_wave_per_shuffle,
                                     .n_per_wave_per_shuffle = c_shuffle_n_xdl_per_wave_per_shuffle,
                                     .scalar_per_vector = c_block_transfer_scalar_per_vector}}},
             .fwd_specialization         = conv_fwd_specialization,
             .gemm_specialization        = gemm_specialization,
             .num_gemm_k_prefetch_stages = num_gemm_k_prefetch_stage,
             .num_conv_groups_to_merge   = num_conv_groups_to_merge,
             .loop_scheduler             = loop_scheduler}};
}

namespace miopen {
namespace conv {
namespace ck_builder {
namespace instance {

using InLayout                             = ck::tensor_layout::convolution::NGCHW;
using WeiLayout                            = ck::tensor_layout::convolution::GKCYX;
using OutLayout                            = ck::tensor_layout::convolution::NGKHW;
using PassThrough                          = ck::tensor_operation::element_wise::PassThrough;
using EmptyTuple                           = ck::Tuple<>;
static constexpr ck::index_t NumDimSpatial = 2;
template <typename DataType>
using DeviceOpGFwdDefault =
    ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD<NumDimSpatial,
                                                                  InLayout,
                                                                  WeiLayout,
                                                                  ck::Tuple<>,
                                                                  OutLayout,
                                                                  DataType,
                                                                  DataType,
                                                                  ck::Tuple<>,
                                                                  DataType,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  PassThrough,
                                                                  DataType,
                                                                  DataType>;

using DeviceOpGFWdDefaultFloat = DeviceOpGFwdDefault<float>;

constexpr auto NGCHW = ckb::TensorLayout::NGCHW;
constexpr auto GKCYX = ckb::TensorLayout::GKCYX;
constexpr auto NGKHW = ckb::TensorLayout::NGKHW;
constexpr auto FP32  = ckb::DataType::FP32;

constexpr auto
create_device_grouped_conv_fwd_xdl_f32_instance_data(std::size_t spatialDim,
                                                     ckb::TensorLayout inLayout,
                                                     ckb::TensorLayout weiLayout,
                                                     ckb::TensorLayout outLayout,
                                                     ckb::ConvSpecialization convSpecialization)
{
    // Adapted from the composable_kernel project, file:
    // library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp

    // clang-format off
    std::array result = {
        // Instance 1: Generic instance
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 8, 1, 8}, 1,
            FP32, FP32),
        
        // Instance 2: Small conv.K and conv.C
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 1,
            FP32, FP32),
        
        // Instance 3
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 1, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 4
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 256, 128, 16, 4, 4, 32, 32, 4, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 5
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 256, 16, 4, 4, 32, 32, 2, 4,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 6
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 128, 16, 4, 4, 32, 32, 4, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 7
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 8
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 9
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 64, 128, 16, 4, 4, 32, 32, 2, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 10
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 64, 16, 4, 4, 32, 32, 2, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 11
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 64, 16, 4, 4, 32, 32, 2, 1,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 12
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 64, 128, 16, 4, 4, 32, 32, 1, 2,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 13
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 128, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 14
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 128, 32, 128, 16, 4, 4, 32, 32, 1, 2,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 32, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 16}, 4,
            FP32, FP32),
        
        // Instance 15
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 64, 32, 16, 4, 4, 32, 32, 2, 1,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 16
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 64, 32, 64, 16, 4, 4, 32, 32, 1, 2,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 16, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 8, 1, 8}, 4,
            FP32, FP32),
        
        // Instance 17
        make_xdl_instance_from_old_params(
            spatialDim, inLayout, weiLayout, outLayout,
            FP32, FP32, FP32, FP32, FP32,
            convSpecialization, ckb::GemmSpecialization::MNKPadding,
            1, 256, 128, 192, 16, 4, 4, 32, 32, 2, 3,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            {4, 64, 1}, {1, 0, 2}, {1, 0, 2}, 2, 4, 4, true,
            1, 1, {1, 16, 1, 16}, 4,
            FP32, FP32)

        // clang-format on
    };

    return result;
}

using BaseOperator    = ck::tensor_operation::device::BaseOperator;
using BaseOperatorPtr = std::unique_ptr<BaseOperator>;

template <auto arr>
void build_k()
{
    auto s = arr.size();
    std::cout << s << std::endl;
}

template <>
struct DeviceOperationInstanceFactory<DeviceOpGFWdDefaultFloat>
{
    static std::vector<BaseOperatorPtr> GetInstances()
    {
        std::vector<BaseOperatorPtr> instances{};

        constexpr std::array<XdlInstance, 17> defaultInstanceData =
            create_device_grouped_conv_fwd_xdl_f32_instance_data(2,
                                                                 ckb::TensorLayout::NGCHW,
                                                                 ckb::TensorLayout::GKCYX,
                                                                 ckb::TensorLayout::NGKHW,
                                                                 ckb::ConvSpecialization::DEFAULT);
        constexpr auto filter1x1Pad0InstanceData =
            create_device_grouped_conv_fwd_xdl_f32_instance_data(
                2,
                ckb::TensorLayout::NGCHW,
                ckb::TensorLayout::GKCYX,
                ckb::TensorLayout::NGKHW,
                ckb::ConvSpecialization::FILTER_1X1_PAD0);
        constexpr auto filter1x1Stride1Pad0InstanceData =
            create_device_grouped_conv_fwd_xdl_f32_instance_data(
                2,
                ckb::TensorLayout::NGCHW,
                ckb::TensorLayout::GKCYX,
                ckb::TensorLayout::NGKHW,
                ckb::ConvSpecialization::FILTER_1X1_STRIDE1_PAD0);
        constexpr auto oddCInstanceData =
            create_device_grouped_conv_fwd_xdl_f32_instance_data(2,
                                                                 ckb::TensorLayout::NGCHW,
                                                                 ckb::TensorLayout::GKCYX,
                                                                 ckb::TensorLayout::NGKHW,
                                                                 ckb::ConvSpecialization::ODD_C);

        constexpr auto instanceData = concat(defaultInstanceData,
                                             filter1x1Pad0InstanceData,
                                             filter1x1Stride1Pad0InstanceData,
                                             oddCInstanceData);

        build_kernels<instanceData>(instances);

        return instances;
    }
};
} // namespace instance
} // namespace ck_builder
} // namespace conv
} // namespace miopen
