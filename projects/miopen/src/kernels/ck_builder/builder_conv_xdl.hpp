// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

struct XdlAlgorithm
{
    using ConvSpecial = ckb::ConvFwdSpecialization;
    using GemmSpecial = ckb::GemmSpecialization;
    using PipeVers    = ckb::PipelineVersion;
    using PipeSched   = ckb::PipelineScheduler;

    struct ThreadBlock
    {
        int block_size;
        struct TileSize
        {
            int m;
            int n;
            int k;
        } tile_size;
    } thread_block;

    static_assert(ckb::ThreadBlockDescriptor<ThreadBlock>);

    struct GridwiseGemm
    {
        int ak1;
        int bk1;
        int m_per_xdl;
        int n_per_xdl;
        int m_xdl_per_wave;
        int n_xdl_per_wave;
    } gridwise_gemm;

    static_assert(ckb::GridwiseXdlGemmDescriptor<GridwiseGemm>);

    struct TransferABC
    {
        struct TransferAB
        {
            struct BlockTransfer
            {
                int k0;
                int m_n;
                int k1;
            } block_transfer;
            struct LdsTransfer
            {
                int src_vector_dim;
                int src_scalar_per_vector;
                int lds_dst_scalar_per_vector;
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
                int m_block;
                int m_wave_per_xdl;
                int n_block;
                int n_wave_per_xdl;
            } thread_cluster_dims;
            struct Epilogue
            {
                int m_xdl_per_wave_per_shuffle;
                int n_per_wave_per_shuffle;
                int scalar_per_vector;
            } epilogue;
        } c;
    } transfer;

    // TODO: Fix CK Builder schema to not require these defaults.
    ConvSpecial fwd_specialization;
    GemmSpecial gemm_specialization;

    std::size_t num_gemm_k_prefetch_stages;
    std::size_t num_groups_to_merge;
    PipeSched loop_scheduler;
};

static_assert(ckb::factory::IsXdlAlgorithm<DefaultAlgorithm> &&
              !ckb::factory::IsXdlV3Algorithm<DefaultAlgorithm>);

struct Signature
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

// Constexpr function to create XdlAlgorithm from old DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle template parameters
constexpr XdlAlgorithm make_xdl_algorithm_from_old_params(
    // Specializations
    ckb::ConvFwdSpecialization conv_fwd_specialization,
    ckb::GemmSpecialization gemm_specialization,
    
    // Numeric parameters
    int num_gemm_k_prefetch_stage,
    int block_size,
    int m_per_block,
    int n_per_block,
    int k_per_block,
    int ak1,
    int bk1,
    int m_per_xdl,
    int n_per_xdl,
    int m_xdl_per_wave,
    int n_xdl_per_wave,
    
    // A block transfer parameters
    std::array<std::size_t, 3> a_thread_cluster_lengths,
    std::array<std::size_t, 3> a_thread_cluster_arrange_order,
    std::array<std::size_t, 3> a_block_transfer_src_access_order,
    int a_block_transfer_src_vector_dim,
    int a_block_transfer_src_scalar_per_vector,
    int a_block_transfer_dst_scalar_per_vector_k1,
    bool a_block_lds_extra_m,
    
    // B block transfer parameters
    std::array<std::size_t, 3> b_thread_cluster_lengths,
    std::array<std::size_t, 3> b_thread_cluster_arrange_order,
    std::array<std::size_t, 3> b_block_transfer_src_access_order,
    int b_block_transfer_src_vector_dim,
    int b_block_transfer_src_scalar_per_vector,
    int b_block_transfer_dst_scalar_per_vector_k1,
    bool b_block_lds_extra_n,
    
    // C shuffle parameters
    int c_shuffle_m_xdl_per_wave_per_shuffle,
    int c_shuffle_n_xdl_per_wave_per_shuffle,
    std::array<std::size_t, 4> c_thread_cluster_lengths,
    int c_block_transfer_scalar_per_vector,
    
    // Loop scheduler
    ckb::PipelineScheduler loop_scheduler,
    
    // Groups to merge
    int num_groups_to_merge
)
{
    return XdlAlgorithm{
        .thread_block = {
            .block_size = block_size,
            .tile_size = {
                .m = m_per_block,
                .n = n_per_block,
                .k = k_per_block
            }
        },
        .gridwise_gemm = {
            .ak1 = ak1,
            .bk1 = bk1,
            .m_per_xdl = m_per_xdl,
            .n_per_xdl = n_per_xdl,
            .m_xdl_per_wave = m_xdl_per_wave,
            .n_xdl_per_wave = n_xdl_per_wave
        },
        .transfer = {
            .a = {
                .block_transfer = {
                    .k0 = static_cast<int>(a_thread_cluster_lengths[0]),
                    .m_n = static_cast<int>(a_thread_cluster_lengths[1]),
                    .k1 = static_cast<int>(a_thread_cluster_lengths[2])
                },
                .lds_transfer = {
                    .src_vector_dim = a_block_transfer_src_vector_dim,
                    .src_scalar_per_vector = a_block_transfer_src_scalar_per_vector,
                    .lds_dst_scalar_per_vector = a_block_transfer_dst_scalar_per_vector_k1,
                    .is_direct_load = false,
                    .lds_padding = a_block_lds_extra_m
                },
                .block_transfer_access_order = {
                    .order = a_thread_cluster_arrange_order
                },
                .src_access_order = {
                    .order = a_block_transfer_src_access_order
                }
            },
            .b = {
                .block_transfer = {
                    .k0 = static_cast<int>(b_thread_cluster_lengths[0]),
                    .m_n = static_cast<int>(b_thread_cluster_lengths[1]),
                    .k1 = static_cast<int>(b_thread_cluster_lengths[2])
                },
                .lds_transfer = {
                    .src_vector_dim = b_block_transfer_src_vector_dim,
                    .src_scalar_per_vector = b_block_transfer_src_scalar_per_vector,
                    .lds_dst_scalar_per_vector = b_block_transfer_dst_scalar_per_vector_k1,
                    .is_direct_load = false,
                    .lds_padding = b_block_lds_extra_n
                },
                .block_transfer_access_order = {
                    .order = b_thread_cluster_arrange_order
                },
                .src_access_order = {
                    .order = b_block_transfer_src_access_order
                }
            },
            .c = {
                .thread_cluster_dims = {
                    .m_block = static_cast<int>(c_thread_cluster_lengths[0]),
                    .m_wave_per_xdl = static_cast<int>(c_thread_cluster_lengths[1]),
                    .n_block = static_cast<int>(c_thread_cluster_lengths[2]),
                    .n_wave_per_xdl = static_cast<int>(c_thread_cluster_lengths[3])
                },
                .epilogue = {
                    .m_xdl_per_wave_per_shuffle = c_shuffle_m_xdl_per_wave_per_shuffle,
                    .n_per_wave_per_shuffle = c_shuffle_n_xdl_per_wave_per_shuffle,
                    .scalar_per_vector = c_block_transfer_scalar_per_vector
                }
            }
        },
        .fwd_specialization = conv_fwd_specialization,
        .gemm_specialization = gemm_specialization,
        .num_gemm_k_prefetch_stages = static_cast<std::size_t>(num_gemm_k_prefetch_stage),
        .num_groups_to_merge = static_cast<std::size_t>(num_groups_to_merge),
        .loop_scheduler = loop_scheduler
    };
}
