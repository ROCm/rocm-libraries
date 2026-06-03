// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/common.hpp"
#include "ck_tile/host/concat.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/core/utility/env.hpp"
#include "ck_tile/core/utility/type_traits.hpp"
#include "ck_tile/core/utility/persistent_async_input_scheduler.hpp"
#include "ck_tile/core/arch/workgroup_barrier.hpp"

namespace ck_tile {

/// @brief The Universal GEMM kernel host arguments.
///
/// @par Overview
///      This structure is passed to @ref UniversalGemmKernel "UniversalGemmKernel" when creating
///      kernel arguments object. It contain all necessary information required to build proper
///      kernel argument and launch kernel on GPU. This structure defines the GEMM problem
///      configuration by stating all required information like M,N,K sizes and respective strides.
///      NumATensor describes the number of A tensors. The minimum number of tensors is 1(required).
///      NumBTensor describes the number of B tensors. The minimum number of tensors is 1(required).
///      NumDTensor describes the number of D tensors. The minimum number of tensors is 0(not
///      required).
template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct UniversalGemmHostArgs
{
    CK_TILE_HOST UniversalGemmHostArgs(
        const std::array<const void*, NumATensor>& as_ptr_,
        const std::array<const void*, NumBTensor>& bs_ptr_,
        const std::array<const void*, NumDTensor>& ds_ptr_,
        [[clang::lifetimebound]] void* e_ptr_,
        index_t k_batch_,
        index_t M_,
        index_t N_,
        index_t K_,
        const std::array<index_t, NumATensor>& stride_As_,
        const std::array<index_t, NumBTensor>& stride_Bs_,
        const std::array<index_t, NumDTensor>& stride_Ds_,
        index_t stride_E_,
        PersistentAsyncInputScheduler async_input_scheduler_ = PersistentAsyncInputScheduler{})
        : as_ptr(as_ptr_),
          bs_ptr(bs_ptr_),
          ds_ptr(ds_ptr_),
          e_ptr(e_ptr_),
          M(M_),
          N(N_),
          K(K_),
          stride_As(stride_As_),
          stride_Bs(stride_Bs_),
          stride_Ds(stride_Ds_),
          stride_E(stride_E_),
          k_batch(k_batch_),
          async_input_scheduler(async_input_scheduler_)
    {
    }

    const std::array<const void*, NumATensor> as_ptr;
    const std::array<const void*, NumBTensor> bs_ptr;
    const std::array<const void*, NumDTensor> ds_ptr;
    union
    {
        void* e_ptr;
        void* c_ptr;
    };
    index_t M;
    index_t N;
    index_t K;
    const std::array<index_t, NumATensor> stride_As;
    const std::array<index_t, NumBTensor> stride_Bs;
    const std::array<index_t, NumDTensor> stride_Ds;
    union
    {
        index_t stride_E;
        index_t stride_C;
    };

    index_t k_batch;
    PersistentAsyncInputScheduler async_input_scheduler;
};

/// @brief The GEMM kernel device arguments.
template <index_t NumATensor = 1, index_t NumBTensor = 1, index_t NumDTensor = 0>
struct UniversalGemmKernelArgs
{
    /// @brief The As input tensor's pointer to device memory.
    const std::array<const void*, NumATensor> as_ptr;
    /// @brief The Bs input tensor's pointer to device memory.
    const std::array<const void*, NumBTensor> bs_ptr;
    /// @brief The Ds input tensor's pointer to device memory.
    const std::array<const void*, NumDTensor> ds_ptr;
    /// @brief The E output tensor's pointer to device memory.
    void* e_ptr;
    /// @brief GEMM's M dimension size.
    index_t M;
    /// @brief GEMM's N dimension size.
    index_t N;
    /// @brief GEMM's K dimension size.
    index_t K;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of As tensor.
    std::array<index_t, NumATensor> stride_As;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Bs tensor.
    std::array<index_t, NumBTensor> stride_Bs;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of Ds tensor.
    std::array<index_t, NumDTensor> stride_Ds;
    /// @brief The distance between consecutive elements of non-contiguous dimension
    ///        (in memory) of E tensor.
    index_t stride_E;
    index_t k_batch;
    /// @brief Persistent async input scheduler for chunk-based tile scheduling.
    PersistentAsyncInputScheduler async_input_scheduler = {};
};

template <typename TilePartitioner, index_t NumATensor, index_t NumBTensor, index_t NumDTensor>
struct StreamKKernelArgs : ck_tile::UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>
{
    StreamKKernelArgs(const UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>& host_args,
                      index_t max_active_wgs,
                      int num_xccs_ = 1)
        : UniversalGemmKernelArgs<NumATensor, NumBTensor, NumDTensor>{host_args.as_ptr,
                                                                      host_args.bs_ptr,
                                                                      host_args.ds_ptr,
                                                                      host_args.e_ptr,
                                                                      host_args.M,
                                                                      host_args.N,
                                                                      host_args.K,
                                                                      host_args.stride_As,
                                                                      host_args.stride_Bs,
                                                                      host_args.stride_Ds,
                                                                      host_args.stride_E,
                                                                      host_args.k_batch},
          // The workspace pointer is set to nullptr because we must first
          // instantiate the TilePartitioner to get the necessary size
          workspace_ptr{nullptr},
          tile_partitioner{host_args.M, host_args.N, host_args.K, max_active_wgs},
          num_xccs{num_xccs_}

    {
    }
    /**
     * @brief  A pointer to a buffer in device memory for accumulating partial via reduction
     * strategy.
     */
    void* workspace_ptr;
    /**
     * @brief  An instance of the TilePartioner class for assisting with mapping workgroups to
     * the C tensor.
     */
    TilePartitioner tile_partitioner;
    index_t num_xccs;
};

/// @brief The Universal GEMM kernel template.
///
/// @paragraph Overview Overview
///            This class provides the generic matrix multiplication kernel template. By semantic
///            division of GEMM algorithm into following parts we achieve flexible, versatile
///            and robust kernel implementation.
///
///            @li @b Prolog - The start of GEMM kernel implementation in @ref operator()
///                function call operator" which determines the work scope of each workgroup.
///            @li @b GemmPipeline - The core part @a "heart" of matrix multiplication algorithm.
///                This is the place where each workgroup is loading data from global memory and
///                carrying out dot products.
///            @li @b Epilogue - The @a "final" part of matrix multiplication implementation
///                 responsible for storing results to global memory. This is also the place where
///                 any additional operator fusion may take place.
///
///            Additionally both @ref GemmPipeline_ "GemmPipeline" and @ref EpiloguePipeline_
///            "EpiloguePipeline" are parameterized with so called @a Policy which determines all
///            internal details of those functional parts. You can think of it like both gemm and
///            epilogue pipelines provides the control-flow logic controlled by policies. Moreover
///            the policy is responsible for definition of all necessary data layouts and thread's
///            work distribution.
///
/// @tparam TilePartitioner_    The type of class providing mapping of workgroup index into the
///                             output data tile to be calculated. It determines the workgroup to
///                             data relationship (or in other words - which data would be
///                             processed and calculated by which workgroup).
/// @tparam GemmPipeline_       The type of class which provides the core part of matrix
///                             multiplication. This class should provide implementation of data
///                             loading from global memory and performing block-wise matrix
///                             multiplication. You can think of it as a work done by single
///                             workgroup point of view.
/// @tparam EpiloguePipeline_   The type of class providing the final part of matrix
///                             multiplication implementation. It is responsible for storing
///                             results calculated by @ref GemmPipeline_ "GemmPipeline" to
///                             the output E tensor in global memory.
template <typename TilePartitioner_,
          typename GemmPipeline_,
          typename EpiloguePipeline_,
          typename Derived_ = void>
struct UniversalGemmKernel
{
    using TilePartitioner  = remove_cvref_t<TilePartitioner_>;
    using GemmPipeline     = remove_cvref_t<GemmPipeline_>;
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;

    using SelfType = std::conditional_t<std::is_void_v<Derived_>, UniversalGemmKernel, Derived_>;

    static constexpr bool ADataTypeIsTuple =
        is_detected<is_tuple, typename GemmPipeline::AsDataType>::value;
    static constexpr bool BDataTypeIsTuple =
        is_detected<is_tuple, typename GemmPipeline::BsDataType>::value;
    static constexpr bool DDataTypeIsTuple =
        is_detected<is_tuple, typename EpiloguePipeline::DsDataType>::value;
    static constexpr bool ALayoutIsTuple =
        is_detected<is_tuple, typename GemmPipeline::AsLayout>::value;
    static constexpr bool BLayoutIsTuple =
        is_detected<is_tuple, typename GemmPipeline::BsLayout>::value;
    static constexpr bool DLayoutIsTuple =
        is_detected<is_tuple, typename EpiloguePipeline::DsLayout>::value;

    using AsLayout = std::conditional_t<ALayoutIsTuple,
                                        remove_cvref_t<typename GemmPipeline::AsLayout>,
                                        remove_cvref_t<tuple<typename GemmPipeline::ALayout>>>;
    using BsLayout = std::conditional_t<BLayoutIsTuple,
                                        remove_cvref_t<typename GemmPipeline::BsLayout>,
                                        remove_cvref_t<tuple<typename GemmPipeline::BLayout>>>;

    using DsLayout = std::conditional_t<DLayoutIsTuple,
                                        remove_cvref_t<typename EpiloguePipeline::DsLayout>,
                                        remove_cvref_t<tuple<typename EpiloguePipeline::DsLayout>>>;

    using AsDataType = std::conditional_t<ADataTypeIsTuple,
                                          remove_cvref_t<typename GemmPipeline::AsDataType>,
                                          remove_cvref_t<tuple<typename GemmPipeline::ADataType>>>;

    using BsDataType = std::conditional_t<BDataTypeIsTuple,
                                          remove_cvref_t<typename GemmPipeline::BsDataType>,
                                          remove_cvref_t<tuple<typename GemmPipeline::BDataType>>>;

    using DsDataType =
        std::conditional_t<DDataTypeIsTuple,
                           remove_cvref_t<typename EpiloguePipeline::DsDataType>,
                           remove_cvref_t<tuple<typename EpiloguePipeline::DsDataType>>>;

    using CLayout   = remove_cvref_t<typename GemmPipeline::CLayout>;
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using AElementWise = remove_cvref_t<typename GemmPipeline::AElementWise>;
    using BElementWise = remove_cvref_t<typename GemmPipeline::BElementWise>;

    static constexpr index_t kBlockSize = GemmPipeline::BlockSize;

    // Detect persistent kernel support to select appropriate entry point
    struct has_persistent_kernel
    {
        template <typename T>
        using has_persistent_type = decltype(T::UsePersistentKernel);

        static constexpr bool value = []() {
            if constexpr(is_detected<has_persistent_type, GemmPipeline>{})
                return GemmPipeline::UsePersistentKernel;
            else
                return false;
        }();
    };
    static constexpr bool PersistentKernel = has_persistent_kernel::value;
    template <typename T, typename = void>
    struct UseStreamK
    {
        static constexpr bool value       = false;
        static constexpr bool use_atomics = false;
    };

    template <typename T>
    struct UseStreamK<T, std::void_t<decltype(T::ReductionStrategy)>>
    {
        static constexpr bool value = true;
        static constexpr bool use_atomics =
            T::ReductionStrategy == StreamKReductionStrategy::Atomic;
    };

    static constexpr bool IsStreamK = UseStreamK<TilePartitioner>::value;

    // Detect custom output offset support for advanced partitioning schemes
    static constexpr bool ClusterLaunch =
        !IsStreamK &&
        (TilePartitioner::BlockGemmShape::kclusterM * TilePartitioner::BlockGemmShape::kclusterN *
             TilePartitioner::BlockGemmShape::kclusterK >
         1);

    // Check if TilePartitioner has GetOutputOffset method with kargs and k_id
    struct has_tile_partitioner_output_offset_impl
    {
        template <typename T, typename KernelArgs>
        using has_get_output_offset_t =
            decltype(T::GetOutputOffset(std::declval<KernelArgs>(), std::declval<index_t>()));

        static constexpr bool value = []() {
            if constexpr(is_detected<has_get_output_offset_t, TilePartitioner>{})
                return true;
            else
                return false;
        }();
    };
    static constexpr bool has_tile_partitioner_output_offset =
        has_tile_partitioner_output_offset_impl::value;

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>{};

    static constexpr index_t NumATensor = AsDataType::size();
    static constexpr index_t NumBTensor = BsDataType::size();
    static constexpr index_t NumDTensor = DsDataType::size();

    using ADataType = remove_cvref_t<std::tuple_element_t<I0, AsDataType>>;
    using BDataType = remove_cvref_t<std::tuple_element_t<I0, BsDataType>>;

    static constexpr index_t APackedSize = numeric_traits<ADataType>::PackedSize;
    static constexpr index_t BPackedSize = numeric_traits<BDataType>::PackedSize;

    static_assert(AsLayout::size() == AsDataType::size(),
                  "The size of AsLayout and AsDataType should be the same");

    static_assert(BsLayout::size() == BsDataType::size(),
                  "The size of BsLayout and BsDataType should be the same");

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    static_assert(!GemmPipeline::BlockGemmShape::PermuteA, "Not implemented!");

    using KernelArgs = std::conditional_t<
        IsStreamK,
        StreamKKernelArgs<TilePartitioner, NumATensor, NumBTensor, NumDTensor>,
        UniversalGemmKernelArgs<AsLayout::size(), BsLayout::size(), DsLayout::size()>>;
    using Kernel = UniversalGemmKernel<TilePartitioner, GemmPipeline, EpiloguePipeline>;
    using StreamKOps =
        StreamKReductionOps<TilePartitioner,
                            GemmPipeline,
                            StreamKKernelArgs<TilePartitioner, NumATensor, NumBTensor, NumDTensor>>;

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        // clang-format off
        return concat('_', "gemm", gemm_prec_str<ADataType, BDataType>(), GemmPipeline::GetName());
        // clang-format on
    }

    CK_TILE_HOST static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {

        auto grid = TilePartitioner::GridSize(M, N);
        if constexpr(std::is_same_v<decltype(grid), dim3>)
        {
            // GridSize returns dim3: preserve x, y dimensions and add z for batch; used in cluster
            // launch
            return dim3(grid.x, grid.y, KBatch);
        }
        else
        {
            // GridSize returns index_t: use as 1D grid
            return dim3(grid, 1, KBatch);
        }
    }

    CK_TILE_HOST static constexpr auto ClusterSize()
    {
        return dim3(TilePartitioner::BlockGemmShape::kclusterM,
                    TilePartitioner::BlockGemmShape::kclusterN,
                    TilePartitioner::BlockGemmShape::kclusterK);
    }

    /**
     * @brief Compute the grid size for the Stream K kernel using the tile_partitioner.
     * @return The grid size.
     */
    template <bool V = IsStreamK>
    CK_TILE_HOST static auto GridSize(const TilePartitioner& tile_partitioner)
    {
        if constexpr(IsStreamK)
        {
            return tile_partitioner.grid_size();
        }
        else
        {
            static_assert(!V,
                          "GridSize(tile_partitioner) only supported for Stream-K tile "
                          "partitioners.");
        }
    }

    /**
     * @brief Computes the buffer size needed to store accumulation results for Stream K.
     * @return The buffer size needed.
     */
    template <bool V = IsStreamK>
    CK_TILE_HOST static std::enable_if_t<V, uint32_t> GetWorkSpaceSize(const KernelArgs& kargs)
    {
        return kargs.tile_partitioner.get_workspace_size(
            sizeof(typename EpiloguePipeline::AccDataType));
    }

    /**
     * @brief Fallback workspace size for non-Stream-K kernels.
     * @return zero
     */
    template <bool V = IsStreamK>
    CK_TILE_HOST static std::enable_if_t<!V, uint32_t>
    GetWorkSpaceSize([[maybe_unused]] const KernelArgs& kargs)
    {
        return 0;
    }

    /**
     * @brief Calculate grid size that maximizes hardware utilization for persistent kernels.
     * @return Grid size that fills all compute units at maximum occupancy.
     * @note Persistent kernels loop over tiles, so grid size should match hardware capacity
     *       rather than problem size.
     */
    CK_TILE_HOST static auto MaxOccupancyGridSize(const stream_config& s) -> dim3
    {
        const auto kernel = kentry<1, SelfType, typename SelfType::KernelArgs>;
        int occupancy;
        ck_tile::hip_check_error(
            hipOccupancyMaxActiveBlocksPerMultiprocessor(&occupancy, kernel, BlockSize().x, 0));

        const int grid_size = get_available_compute_units(s) * max(occupancy, 1);
        return dim3(grid_size, 1, 1);
    }

    CK_TILE_HOST static auto BlockSize()
    {
        if(ck_tile::is_wave32())
        {
            return dim3(kBlockSize / 2);
        }
        else
        {
            return dim3(kBlockSize);
        }
    }

    CK_TILE_HOST static KernelArgs
    MakeKernelArgs(const UniversalGemmHostArgs<NumATensor, NumBTensor, NumDTensor>& hostArgs,
                   int num_cu    = NumCU(),
                   int occupancy = Occupancy<Kernel, KernelArgs, kBlockSize>())
    {
        if constexpr(!IsStreamK)
        {
            return KernelArgs{hostArgs.as_ptr,
                              hostArgs.bs_ptr,
                              hostArgs.ds_ptr,
                              hostArgs.e_ptr,
                              hostArgs.M,
                              hostArgs.N,
                              hostArgs.K,
                              hostArgs.stride_As,
                              hostArgs.stride_Bs,
                              hostArgs.stride_Ds,
                              hostArgs.stride_E,
                              hostArgs.k_batch,
                              hostArgs.async_input_scheduler};
        }
        else
        {
            const index_t max_active_wgs = num_cu * occupancy;
            const int num_xccs           = get_num_xccs();
            return KernelArgs{hostArgs, max_active_wgs, num_xccs};
        }
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return max(GemmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }

    struct SplitKBatchOffset
    {
        // Balances K-dimension work across batches to maximize parallelism while minimizing
        // load imbalance. Uses ceil division to distribute remainder work evenly.
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const index_t k_id = blockIdx.z)
        {
            constexpr auto K1     = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t num_all = amd_wave_read_first_lane(
                kargs.K / K1); // num of all loops not including potential tail
            index_t num_full = amd_wave_read_first_lane(num_all % kargs.k_batch);
            num_full         = num_full == 0 ? kargs.k_batch : num_full;

            const index_t num_full_iters =
                amd_wave_read_first_lane(std::max(integer_divide_ceil(num_all, kargs.k_batch), 1));
            const index_t full_k_read    = num_full_iters * K1;
            const index_t partial_k_read = (num_full_iters - 1) * K1;

            static_for<0, NumATensor, 1>{}([&](auto index) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, AiLayout>)
                {
                    as_k_split_offset[index] =
                        amd_wave_read_first_lane(std::min(k_id, num_full) * full_k_read +
                                                 std::max(k_id - num_full, 0) * partial_k_read);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, AiLayout>)
                {
                    as_k_split_offset[index] =
                        amd_wave_read_first_lane((std::min(k_id, num_full) * full_k_read +
                                                  std::max(k_id - num_full, 0) * partial_k_read) *
                                                 kargs.stride_As[index]);
                }
            });

            static_for<0, NumBTensor, 1>{}([&](auto index) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
                if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BiLayout>)
                {
                    bs_k_split_offset[index] =
                        amd_wave_read_first_lane((std::min(k_id, num_full) * full_k_read +
                                                  std::max(k_id - num_full, 0) * partial_k_read) *
                                                 kargs.stride_Bs[index]);
                }
                else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BiLayout>)
                {
                    bs_k_split_offset[index] =
                        amd_wave_read_first_lane(std::min(k_id, num_full) * full_k_read +
                                                 std::max(k_id - num_full, 0) * partial_k_read);
                }
            });

            if(k_id == kargs.k_batch - 1)
            {
                splitted_k = kargs.K - std::min(k_id, num_full) * full_k_read -
                             std::max(k_id - num_full, 0) * partial_k_read;
            }
            else if(k_id < num_full)
            {
                splitted_k = full_k_read;
            }
            else
            {
                splitted_k = partial_k_read;
            }
        }

        std::array<index_t, NumATensor> as_k_split_offset;
        std::array<index_t, NumBTensor> bs_k_split_offset;
        index_t splitted_k;
    };

    // for skipping validation of launch parameters especially for TDM where padding is unused
    struct has_skip_check_valid_launch_params
    {
        template <typename T>
        using has_skip_check_type = decltype(T::skipCheckValidLaunchParams);

        static constexpr bool value = []() {
            if constexpr(is_detected<has_skip_check_type, GemmPipeline>{})
                return GemmPipeline::skipCheckValidLaunchParams;
            else
                return false;
        }();
    };

    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(has_skip_check_valid_launch_params::value)
        {
            return true;
        }
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Conditions not met for Kbatch >1 !");
                }
                return false;
            }
        }

        if(integer_divide_ceil(kargs.K, GemmPipeline::BlockGemmShape::WarpTile::at(number<2>{})) <
           kargs.k_batch)
        {
            if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
            {
                CK_TILE_ERROR("KBatch is too large, part of GPU wouldn't be utilized!");
            }
            return false;
        }

        const auto vectorSizeA = is_wave32() ? GemmPipeline::template GetVectorSizeA<true>()
                                             : GemmPipeline::template GetVectorSizeA<false>();
        bool AsTensorIsValid   = {true};
        static_for<0, NumATensor, 1>{}([&](auto index) {
            using AiLayout = remove_cvref_t<std::tuple_element_t<index.value, AsLayout>>;
            if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
                   GemmPipeline::kPadK == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support K that is not a multiple of k_batch * KPerBlock "
                            "without padding!");
                    }
                    AsTensorIsValid = false;
                }
                if(kargs.K % vectorSizeA != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("K is not a multiple of vector load size for A tensor!");
                    }
                    AsTensorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support M that is not a multiple of MPerBlock without padding!");
                    }
                    AsTensorIsValid = false;
                }
                if(kargs.M % vectorSizeA != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("M is not a multiple of vector load size for A tensor!");
                    }
                    AsTensorIsValid = false;
                }
            }
        });

        bool BsTensorIsValid   = {true};
        const auto vectorSizeB = is_wave32() ? GemmPipeline::template GetVectorSizeB<true>()
                                             : GemmPipeline::template GetVectorSizeB<false>();
        static_for<0, NumBTensor, 1>{}([&](auto index) {
            using BiLayout = remove_cvref_t<std::tuple_element_t<index.value, BsLayout>>;
            if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support N that is not a multiple of NPerBlock without padding!");
                    }
                    BsTensorIsValid = false;
                }
                if(kargs.N % vectorSizeB != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("N is not a multiple of vector load size for B tensor!");
                    }
                    BsTensorIsValid = false;
                }
            }
            else
            {
                if(kargs.K % (TilePartitioner::KPerBlock * kargs.k_batch) != 0 &&
                   GemmPipeline::kPadK == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR(
                            "Can't support K that is not a multiple of k_batch * KPerBlock "
                            "without padding!");
                    }
                    BsTensorIsValid = false;
                }
                if(kargs.K % vectorSizeB != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("K is not a multiple of vector load size for B tensor!");
                    }
                    BsTensorIsValid = false;
                }
            }
        });

        bool DTensorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, CLayout> == false)
            {
                DTensorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                      "NPerBlock without padding!");
                    }
                    DTensorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    }
                    DTensorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                      "MPerBlock without padding!");
                    }
                    DTensorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                    {
                        CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    }
                    DTensorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && GemmPipeline::kPadN == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support N that is not a multiple of NPerBlock without padding!");
                }
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && GemmPipeline::kPadM == false)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Can't support M that is not a multiple of MPerBlock without padding!");
                }
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for C tensor!");
                }
                return false;
            }
        }

        // Verify async scheduler parameters to prevent division-by-zero and invalid memory access
        if(kargs.async_input_scheduler.chunk_signals != nullptr)
        {
            if(kargs.async_input_scheduler.tiles_per_chunk_m == 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("tiles_per_chunk_m must be positive when chunk_signals is set!");
                }
                return false;
            }
            if(kargs.async_input_scheduler.num_chunks == 0)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("num_chunks must be positive when chunk_signals is set!");
                }
                return false;
            }
        }

        // Verify Stream-K conditions
        if constexpr(IsStreamK)
        {
            // Stream-K can only be used with k_batch of 1
            if(kargs.k_batch > 1)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR("Stream-K can only be used when k_batch is 1!");
                }

                return false;
            }

            // Stream-K does not support multiple D tensors with the Atomic reduction strategy
            // because multiple workgroups will run the Epilogue, applying the D tensors more than
            // once.
            if(NumDTensor > 0 && UseStreamK<TilePartitioner>::use_atomics)
            {
                if(ck_tile::EnvIsEnabled(CK_TILE_ENV(CK_TILE_LOGGING)))
                {
                    CK_TILE_ERROR(
                        "Stream-K does not support D tensors with the Atomic reduction strategy!");
                }

                return false;
            }
        }

        return AsTensorIsValid && BsTensorIsValid && DTensorIsValid;
    }

    template <typename ALayout>
    CK_TILE_DEVICE static auto
    MakeDefaultATensorDescriptor(const index_t M, const index_t stride, const index_t k_size)
    {
        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(make_tuple(M, k_size),
                                                make_tuple(stride, 1),
                                                number<GemmPipeline::GetVectorSizeA()>{},
                                                number<1>{});
        }
        else
        {
            return make_naive_tensor_descriptor(make_tuple(k_size, M),
                                                make_tuple(stride, 1),
                                                number<GemmPipeline::GetVectorSizeA()>{},
                                                number<1>{});
        }
    }

    template <typename BLayout>
    CK_TILE_DEVICE static auto MakeDefaultBTensorDescriptor(const index_t N,
                                                            const index_t K,
                                                            const index_t stride,
                                                            const index_t k_size)
    {
        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
            {
                constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                const index_t K0              = k_size / K1;
                constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                const auto b_k0_n_k1_desc     = make_naive_tensor_descriptor(make_tuple(K0, N, K1),
                                                                         make_tuple(N * K1, K1, I1),
                                                                         number<VectorSizeB>{},
                                                                         number<1>{});
                return transform_tensor_descriptor(
                    b_k0_n_k1_desc,
                    make_tuple(make_merge_transform(make_tuple(K0, K1)),
                               make_pass_through_transform(N)),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<0>{}, sequence<1>{}));
            }
            else
            {
                return make_naive_tensor_descriptor(make_tuple(k_size, N),
                                                    make_tuple(stride, 1),
                                                    number<GemmPipeline::GetVectorSizeB()>{},
                                                    number<1>{});
            }
        }
        else
        {
            if constexpr(GemmPipeline::BlockGemmShape::PermuteB)
            {
                constexpr index_t K1          = GemmPipeline::GetSmemPackB();
                const index_t K0              = k_size / K1;
                constexpr index_t VectorSizeB = std::min(K1, GemmPipeline::GetVectorSizeB());
                const auto b_k0_n_k1_desc     = make_naive_tensor_descriptor(make_tuple(K0, N, K1),
                                                                         make_tuple(N * K1, K1, I1),
                                                                         number<VectorSizeB>{},
                                                                         number<1>{});
                return transform_tensor_descriptor(
                    b_k0_n_k1_desc,
                    make_tuple(make_merge_transform(make_tuple(K0, K1)),
                               make_pass_through_transform(N)),
                    make_tuple(sequence<0, 2>{}, sequence<1>{}),
                    make_tuple(sequence<1>{}, sequence<0>{}));
            }
            else
            {
                if constexpr(GemmPipeline::Preshuffle)
                {
                    index_t kFlatK =
                        GemmPipeline::BlockGemmShape::flatKPerWarp *
                        (k_size / GemmPipeline::BlockGemmShape::WarpTile::at(number<2>{}));
                    index_t kFlatN = N * K / kFlatK;

                    return make_naive_tensor_descriptor(make_tuple(kFlatN, kFlatK),
                                                        make_tuple(kFlatK, 1),
                                                        number<GemmPipeline::GetVectorSizeB()>{},
                                                        number<1>{});
                }
                else
                {
                    return make_naive_tensor_descriptor(make_tuple(N, k_size),
                                                        make_tuple(stride, 1),
                                                        number<GemmPipeline::GetVectorSizeB()>{},
                                                        number<1>{});
                }
            }
        }
    }

    template <typename DLayout, index_t VectorSizeD>
    CK_TILE_DEVICE static auto
    MakeDefaultDTensorDescriptor(const index_t M, const index_t N, const index_t stride)
    {
        if constexpr(std::is_same_v<DLayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(
                make_tuple(M, N), make_tuple(stride, 1), number<VectorSizeD>{}, number<1>{});
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(N, M), make_tuple(stride, 1), number<VectorSizeD>{}, number<1>{});
        }
    }

    CK_TILE_DEVICE static auto
    MakeDefaultETensorDescriptor(const index_t M, const index_t N, const index_t stride)
    {
        if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
        {
            return make_naive_tensor_descriptor(make_tuple(M, N),
                                                make_tuple(stride, 1),
                                                number<EpiloguePipeline::GetVectorSizeC()>{},
                                                number<1>{});
        }
        else
        {
            return make_naive_tensor_descriptor(
                make_tuple(M, N), make_tuple(1, stride), number<1>{}, number<1>{});
        }
    }

    template <typename AsTensorDesc>
    CK_TILE_DEVICE static auto
    MakeABlockWindows(const std::array<const ADataType*, NumATensor>& as_ptr,
                      const AsTensorDesc& as_desc,
                      const index_t i_m)
    {
        // Step 1: Create tensor views
        const auto& as_tensor_view = generate_tuple(
            [&](auto i) {
                using AiDataType = remove_cvref_t<std::tuple_element_t<i.value, AsDataType>>;
                return make_tensor_view<address_space_enum::global>(
                    static_cast<const AiDataType*>(as_ptr[i]), as_desc[i]);
            },
            number<NumATensor>{});

        // Step 2: Create padded views
        const auto& as_pad_view = generate_tuple(
            [&](auto i) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(as_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                }
                else
                {
                    return pad_tensor_view(as_tensor_view[i],
                                           make_tuple(number<TilePartitioner::KPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumATensor>{});

        // Step 3: Create tile windows
        const auto& as_block_window = generate_tuple(
            [&](auto i) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                if constexpr(std::is_same_v<AiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(as_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::KPerBlock>{}),
                                            {i_m, 0});
                }
                else
                {
                    return make_tile_window(as_pad_view[i],
                                            make_tuple(number<TilePartitioner::KPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {0, i_m});
                }
            },
            number<NumATensor>{});

        return as_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeABlockWindows(const std::array<const ADataType*, NumATensor>& as_ptr,
                      const KernelArgs& kargs,
                      const index_t k_size,
                      const index_t i_m)
    {
        // Step 1: Create tensor descriptors for A tensors
        const auto& as_tensor_desc = generate_tuple(
            [&](auto i) {
                using AiLayout = remove_cvref_t<std::tuple_element_t<i.value, AsLayout>>;
                return MakeDefaultATensorDescriptor<AiLayout>(kargs.M, kargs.stride_As[i], k_size);
            },
            number<NumATensor>{});

        return MakeABlockWindows(as_ptr, as_tensor_desc, i_m);
    }

    template <typename BsTensorDesc>
    CK_TILE_DEVICE static auto
    MakeBBlockWindows(const std::array<const BDataType*, NumBTensor>& bs_ptr,
                      const BsTensorDesc& bs_desc,
                      const index_t i_n)
    {
        // Step 1: Create tensor views
        const auto& bs_tensor_view = generate_tuple(
            [&](auto i) {
                using BiDataType = remove_cvref_t<std::tuple_element_t<i.value, BsDataType>>;
                return make_tensor_view<address_space_enum::global>(
                    static_cast<const BiDataType*>(bs_ptr[i]), bs_desc[i]);
            },
            number<NumBTensor>{});

        // Step 2: Create padded views
        const auto& bs_pad_view = generate_tuple(
            [&](auto i) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
                {
                    return pad_tensor_view(bs_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::KPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadK>{});
                }
                else
                {
                    return pad_tensor_view(bs_tensor_view[i],
                                           make_tuple(number<TilePartitioner::KPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
            },
            number<NumBTensor>{});

        // Step 3: Create tile windows
        const auto& bs_block_window = generate_tuple(
            [&](auto i) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                if constexpr(GemmPipeline::Preshuffle)
                {
                    return make_tile_window(
                        bs_pad_view[i],
                        make_tuple(number<GemmPipeline::BlockGemmShape::flatNPerWarp>{},
                                   number<GemmPipeline::BlockGemmShape::flatKPerWarp>{}),
                        {static_cast<int>(i_n / GemmPipeline::BlockGemmShape::WarpTile::at(I1)),
                         0});
                }
                else
                {
                    if constexpr(std::is_same_v<BiLayout, tensor_layout::gemm::ColumnMajor>)
                    {
                        return make_tile_window(bs_pad_view[i],
                                                make_tuple(number<TilePartitioner::NPerBlock>{},
                                                           number<TilePartitioner::KPerBlock>{}),
                                                {i_n, 0});
                    }
                    else
                    {
                        return make_tile_window(bs_pad_view[i],
                                                make_tuple(number<TilePartitioner::KPerBlock>{},
                                                           number<TilePartitioner::NPerBlock>{}),
                                                {0, i_n});
                    }
                }
            },
            number<NumBTensor>{});

        return bs_block_window;
    }

    CK_TILE_DEVICE static auto
    MakeBBlockWindows(const std::array<const BDataType*, NumBTensor>& bs_ptr,
                      const KernelArgs& kargs,
                      const index_t k_size,
                      const index_t i_n)
    {
        const auto& bs_tensor_desc = generate_tuple(
            [&](auto i) {
                using BiLayout = remove_cvref_t<std::tuple_element_t<i.value, BsLayout>>;
                return MakeDefaultBTensorDescriptor<BiLayout>(
                    kargs.N, kargs.K, kargs.stride_Bs[i], k_size);
            },
            number<NumBTensor>{});

        return MakeBBlockWindows(bs_ptr, bs_tensor_desc, i_n);
    }

    template <typename DsTensorDesc>
    CK_TILE_DEVICE static auto MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                                                 const DsTensorDesc& ds_desc,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        // Step 1: Create tensor views
        const auto& ds_tensor_view = generate_tuple(
            [&](auto i) {
                using DDataType_ = remove_cvref_t<std::tuple_element_t<i.value, DsDataType>>;
                return make_tensor_view<address_space_enum::global>(
                    static_cast<const DDataType_*>(ds_ptr[i]), ds_desc[i]);
            },
            number<NumDTensor>{});

        // Step 2: Create padded views
        const auto& ds_pad_view = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::MPerBlock>{},
                                                      number<TilePartitioner::NPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadN>{});
                }
                else
                {
                    return pad_tensor_view(ds_tensor_view[i],
                                           make_tuple(number<TilePartitioner::NPerBlock>{},
                                                      number<TilePartitioner::MPerBlock>{}),
                                           sequence<false, GemmPipeline::kPadM>{});
                }
            },
            number<NumDTensor>{});

        // Step 3: Create tile windows
        const auto& ds_block_window = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::MPerBlock>{},
                                                       number<TilePartitioner::NPerBlock>{}),
                                            {i_m, i_n});
                }
                else
                {
                    return make_tile_window(ds_pad_view[i],
                                            make_tuple(number<TilePartitioner::NPerBlock>{},
                                                       number<TilePartitioner::MPerBlock>{}),
                                            {i_n, i_m});
                }
            },
            number<NumDTensor>{});

        return ds_block_window;
    }

    CK_TILE_DEVICE static auto MakeDBlockWindows(const std::array<const void*, NumDTensor>& ds_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {
        const auto& ds_tensor_desc = generate_tuple(
            [&](auto i) {
                using DiLayout = remove_cvref_t<std::tuple_element_t<i.value, DsLayout>>;
                return MakeDefaultDTensorDescriptor<DiLayout, EpiloguePipeline::GetVectorSizeD(i)>(
                    kargs.M, kargs.N, kargs.stride_Ds[i]);
            },
            number<NumDTensor>{});

        return MakeDBlockWindows(ds_ptr, ds_tensor_desc, i_m, i_n);
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set, typename ETensorDesc>
    CK_TILE_DEVICE static auto MakeCBlockWindows(
        EDataType* e_ptr,
        const index_t i_m,
        const index_t i_n,
        const ETensorDesc& e_desc) // Argument order differs from A,B,D to disambiguate overloads
    {
        // Step 1: Create tensor view for E/C tensor
        const auto& e_tensor_view =
            make_tensor_view<address_space_enum::global, DstInMemOp>(e_ptr, e_desc);

        // For bf16_t and atomic_add global_atomic_add is used instead of buffer_atomic_add
        // Add padding for not contiguous dim due to the lack of OOB check
        constexpr bool pad_not_contiguous_dim =
            std::is_same_v<EDataType, bf16_t> && DstInMemOp == memory_operation_enum::atomic_add;

        // Step 2: Create padded view
        const auto& e_pad_view = [&]() {
            if constexpr(std::is_same_v<CLayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<pad_not_contiguous_dim, GemmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(e_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::NPerBlock>{}),
                                       sequence<GemmPipeline::kPadM, pad_not_contiguous_dim>{});
            }
        }();

        // Step 3: Create tile window
        auto e_block_window = make_tile_window(
            e_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<TilePartitioner::NPerBlock>{}),
            {i_m, i_n});

        return e_block_window;
    }

    template <memory_operation_enum DstInMemOp = memory_operation_enum::set>
    CK_TILE_DEVICE static auto MakeCBlockWindows(EDataType* e_ptr,
                                                 const KernelArgs& kargs,
                                                 const index_t i_m,
                                                 const index_t i_n)
    {

        const auto& e_tensor_desc = MakeDefaultETensorDescriptor(kargs.M, kargs.N, kargs.stride_E);
        return MakeCBlockWindows<DstInMemOp>(e_ptr, i_m, i_n, e_tensor_desc);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param as_ptr input As pointer
     * @param bs_ptr input Bs pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param splitk_batch_offset splitk_batch_offset Utility structure used to calculate k batch.
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     *
     */
    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       const KernelArgs& kargs,
                                       const SplitKBatchOffset& splitk_batch_offset,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n)
    {
        const index_t k_size   = amd_wave_read_first_lane(splitk_batch_offset.splitted_k);
        const index_t num_loop = TilePartitioner::GetLoopNum(k_size);
        RunGemm(as_ptr,
                bs_ptr,
                ds_ptr,
                e_ptr,
                smem_ptr,
                kargs,
                num_loop,
                block_idx_m,
                block_idx_n,
                k_size);
    }

    /**
     * @brief Runs single GEMM problem cooperatively by whole workgroup.
     *
     * @param as_ptr input As pointer
     * @param bs_ptr input Bs pointer
     * @param ds_ptr input Ds pointer
     * @param e_ptr output E pointer
     * @param smem_ptr The start memory pointer of the shared memory block.
     * @param kargs GEMM kernel arguments
     * @param num_loop The number of iterations (at the macro tile level) in the K dimension
     * this workgroup will perform in the C tile
     * @param block_idx_m The GEMM's output M dimension tile index processed by this workgroup.
     * @param block_idx_n The GEMM's output N dimension tile index processed by this workgroup.
     * @param k_size The size of K this workgroup is currently processing
     *
     */
    CK_TILE_DEVICE static void RunGemm(const std::array<const ADataType*, NumATensor>& as_ptr,
                                       const std::array<const BDataType*, NumBTensor>& bs_ptr,
                                       const std::array<const void*, NumDTensor>& ds_ptr,
                                       EDataType* e_ptr,
                                       void* smem_ptr,
                                       const KernelArgs& kargs,
                                       const index_t num_loop,
                                       const index_t block_idx_m,
                                       const index_t block_idx_n,
                                       const index_t k_size)
    {
        // cluster launch GridDim is aligned to clusterDim, need to skip out-of-bound blocks
        if constexpr(ClusterLaunch)
        {
            if(block_idx_m >= kargs.M || block_idx_n >= kargs.N)
                return;
        }

        // Create block windows using specialized methods
        const auto& as_block_window = MakeABlockWindows(as_ptr, kargs, k_size, block_idx_m);
        const auto& bs_block_window = MakeBBlockWindows(bs_ptr, kargs, k_size, block_idx_n);
        const auto& ds_block_window = MakeDBlockWindows(ds_ptr, kargs, block_idx_m, block_idx_n);

        // Run GEMM cooperatively by whole workgroup.
        const auto& c_block_tile = GemmPipeline{}.template operator()(
            as_block_window, AElementWise{}, bs_block_window, BElementWise{}, num_loop, smem_ptr);

        const index_t k_batch = amd_wave_read_first_lane(kargs.k_batch);
        // Run Epilogue Pipeline
        if(k_batch == 1) // k_batch is always 1 for Stream-K but we only use set for non-atomic
                         // reduction strategy
        {
            if constexpr(!UseStreamK<TilePartitioner>::use_atomics)
            {
                auto c_block_window = MakeCBlockWindows<memory_operation_enum::set>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
            }
            else
            {
                auto c_block_window = MakeCBlockWindows<memory_operation_enum::atomic_add>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
            }
        }
#if !defined(CK_TILE_FORCE_SINGLE_TAIL_HANDLER)
        else
        {
            if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 == 0 ||
                         !is_any_of<EDataType, fp16_t, bf16_t>::value)
            {
                auto c_block_window = MakeCBlockWindows<memory_operation_enum::atomic_add>(
                    e_ptr, kargs, block_idx_m, block_idx_n);
                EpiloguePipeline{}(c_block_window, c_block_tile, ds_block_window, smem_ptr);
            }
        }
#endif
    }

    CK_TILE_DEVICE static auto
    GetTileCoordinates(const KernelArgs& kargs) -> tuple<index_t, index_t>
    {
        index_t iM, iN;

        if constexpr(ClusterLaunch)
        {
            // Cluster launch: use 2D block indexing
            const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
            const auto blockIdY = amd_wave_read_first_lane(blockIdx.y);
            const auto [tile_m, tile_n] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdX, blockIdY);
            iM = tile_m;
            iN = tile_n;
        }
        else
        {
            // Regular launch: use 1D block indexing
            const auto blockId = amd_wave_read_first_lane(blockIdx.x);
            const auto [tile_m, tile_n] =
                TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockId);
            iM = tile_m;
            iN = tile_n;
        }

        const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
        const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

        return make_tuple(i_m, i_n);
    }

    // Helper functions for persistent kernel with cluster support
    CK_TILE_DEVICE static auto GetBlockId() -> index_t
    {
        if constexpr(ClusterLaunch)
        {
            // For 2D cluster launch: convert 2D block index to 1D
            const auto blockIdX = amd_wave_read_first_lane(blockIdx.x);
            const auto blockIdY = amd_wave_read_first_lane(blockIdx.y);
            const auto gridDimX = amd_wave_read_first_lane(gridDim.x);
            return blockIdY * gridDimX + blockIdX;
        }
        else
        {
            // For 1D regular launch
            return amd_wave_read_first_lane(get_block_id());
        }
    }

    CK_TILE_DEVICE static auto GetGridSize() -> index_t
    {
        if constexpr(ClusterLaunch)
        {
            // For 2D cluster launch: total blocks = gridDim.x * gridDim.y
            return amd_wave_read_first_lane(gridDim.x * gridDim.y);
        }
        else
        {
            // For 1D regular launch
            return amd_wave_read_first_lane(get_grid_size());
        }
    }

    // Helper to get total number of tiles, handling both dim3 and index_t return types
    template <typename... Args>
    CK_TILE_HOST_DEVICE static auto GetNumTiles(Args&&... args) -> index_t
    {
        auto grid_size = TilePartitioner::GridSize(std::forward<Args>(args)...);

        using GridSizeType = decltype(grid_size);

        if constexpr(std::is_same_v<GridSizeType, dim3>)
        {
            // GridSize returns dim3: compute total tiles as x * y * z
            return amd_wave_read_first_lane(grid_size.x * grid_size.y * grid_size.z);
        }
        else
        {
            // GridSize returns scalar (index_t): use directly
            return amd_wave_read_first_lane(grid_size);
        }
    }

    /**
     * @brief Runs the main Stream - K algorithm.
     * @param kargs Stream - K kernel arguments.
     * @param cta_idx The current Stream - K workgroup's index.
     * @param smem_ptr_0 Pointer to LDS.
     * @note It is assumed that the first Stream - K workgroup has a `cta_idx` of zero. If a
     * non-persistent data-parallel (DP) section is used, then a Stream-K workgroup's `cta_idx`
     * *should be something like `blockIdx.x` minus number of DP workgroups.
     */
    CK_TILE_DEVICE
    void StreamKGemm(KernelArgs& kargs, index_t cta_idx, void* smem_ptr_0) const
    {
        const StreamKOps sk_ops{};
        index_t iter_start, iter_end;
        kargs.tile_partitioner.get_iter_boundaries(iter_start, iter_end, cta_idx);

        while(iter_start < iter_end)
        {
            // Get the 1D tile index in the C tensor that this workgroup will work in for this
            // iteration of the loop.
            index_t tile_idx =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_tile_index(iter_start));

            // Get the start and end boundaries for the current tile.
            index_t tile_iter_start, tile_iter_end;
            kargs.tile_partitioner.get_tile_boundaries(tile_iter_start, tile_iter_end, tile_idx);

            // Get the start and end iteration within the current tile for the workgroup.
            index_t local_iter_start = amd_wave_read_first_lane(
                kargs.tile_partitioner.get_local_iter(iter_start, tile_iter_start));
            index_t local_iter_end =
                amd_wave_read_first_lane(kargs.tile_partitioner.get_local_iter_end(
                    tile_iter_start, iter_end, tile_iter_end));

            // Get the iteration length.
            index_t num_loop_sk = local_iter_end - local_iter_start;

            // Determine the total size along the K dimension the workgroup is using in this
            // iteration (used to construct tensor views).
            index_t k_size = amd_wave_read_first_lane(
                kargs.tile_partitioner.get_k_size(num_loop_sk, local_iter_end));

            // Get the K offsets for the A and B tensors
            auto [i_k_as, i_k_bs] =
                GetKOffsets<AsLayout, BsLayout, TilePartitioner::KPerBlock, NumATensor, NumBTensor>(
                    local_iter_start, kargs.stride_As, kargs.stride_Bs);

            const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
            const index_t i_m           = c_macro_tile_idx[I0] * TilePartitioner::MPerBlock;
            const index_t i_n           = c_macro_tile_idx[I1] * TilePartitioner::NPerBlock;

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) + i_k_as[i];
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) + i_k_bs[i];
            });

            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Atomic)
            {

                RunGemm(as_ptr,
                        bs_ptr,
                        kargs.ds_ptr,
                        e_ptr,
                        smem_ptr_0,
                        kargs,
                        num_loop_sk,
                        i_m,
                        i_n,
                        k_size);
            }
            else if(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear ||
                    TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Tree)
            {

                // Create block windows using specialized methods
                const auto& as_block_window = MakeABlockWindows(as_ptr, kargs, k_size, i_m);
                const auto& bs_block_window = MakeBBlockWindows(bs_ptr, kargs, k_size, i_n);
                const auto& ds_block_window = MakeDBlockWindows(kargs.ds_ptr, kargs, i_m, i_n);

                // Run GEMM cooperatively by whole workgroup.
                const auto& c_block_tile = GemmPipeline{}.template operator()(as_block_window,
                                                                              AElementWise{},
                                                                              bs_block_window,
                                                                              BElementWise{},
                                                                              num_loop_sk,
                                                                              smem_ptr_0);

                auto tile_started = iter_start == tile_iter_start;
                auto tile_ended   = iter_end >= tile_iter_end;

                if constexpr(TilePartitioner::ReductionStrategy == StreamKReductionStrategy::Linear)
                {
                    if(!tile_started)
                    {
                        sk_ops.StorePartial(kargs, cta_idx, c_block_tile);
                        sk_ops.SignalStorePartialDone(kargs, cta_idx);
                    }
                    else
                    {
                        auto accum_block_tile = c_block_tile;
                        if(!tile_ended)
                        {
                            const index_t iter_per_tile =
                                kargs.tile_partitioner.get_iters_per_tile();
                            const index_t iter_per_cta =
                                kargs.tile_partitioner.get_iters_per_sk_cta();
                            const index_t extra_iters = kargs.tile_partitioner.get_extra_iters();
                            int accum_iters           = local_iter_end - local_iter_start;
                            int next_cta              = cta_idx + 1;

                            while(accum_iters < iter_per_tile)
                            {
                                sk_ops.WaitStorePartialDone(kargs, next_cta);

                                using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                                sk_ops.AddBlockTile(
                                    accum_block_tile,
                                    sk_ops.template LoadPartial<typename BlockType::DataType>(
                                        kargs, next_cta, c_block_tile.get_tile_distribution()));

                                accum_iters += iter_per_cta + (next_cta < extra_iters);
                                ++next_cta;
                            }
                        }

                        auto c_block_window = MakeCBlockWindows<TilePartitioner::MemoryOperation>(
                            e_ptr, kargs, i_m, i_n);
                        EpiloguePipeline{}(
                            c_block_window, accum_block_tile, ds_block_window, smem_ptr_0);
                    }
                }
                else // Tree Reduction
                {
                    auto accum_block_tile      = c_block_tile;
                    index_t tile_local_cta_idx = amd_wave_read_first_lane(
                        kargs.tile_partitioner.get_tile_local_cta_index(tile_iter_start, cta_idx));

                    index_t stride = amd_wave_read_first_lane(1);

                    for(;; stride <<= 1)
                    {
                        const index_t partner_cta_idx = amd_wave_read_first_lane(cta_idx + stride);
                        const index_t partner_start_iter = amd_wave_read_first_lane(
                            kargs.tile_partitioner.get_start_iter(partner_cta_idx));
                        bool partner_in_tile =
                            amd_wave_read_first_lane(partner_start_iter < tile_iter_end);

                        // If the partner of the workgroup who started the tile is not in this
                        // tile, then the work for this tile is done and results can be stored
                        // in the C tensor.
                        if(tile_started && !partner_in_tile)
                        {
                            auto c_block_window =
                                MakeCBlockWindows<TilePartitioner::MemoryOperation>(
                                    e_ptr, kargs, i_m, i_n);
                            EpiloguePipeline{}(
                                c_block_window, accum_block_tile, ds_block_window, smem_ptr_0);
                            break;
                        }

                        // It's this workgroup's turn to read from partials.
                        if(tile_local_cta_idx % (stride << 1) == 0)
                        {
                            // If this workgroup's partner is in the tile then it can read from
                            // partials and accumulate results.
                            if(partner_in_tile)
                            {
                                sk_ops.WaitStorePartialDone(kargs, partner_cta_idx);
                                using BlockType = remove_cvref_t<decltype(c_block_tile)>;
                                sk_ops.AddBlockTile(
                                    accum_block_tile,
                                    sk_ops.template LoadPartial<typename BlockType::DataType>(
                                        kargs,
                                        partner_cta_idx,
                                        c_block_tile.get_tile_distribution()));
                            }
                        }
                        // Otherwise, it's this workgroup's turn to write to partials. All
                        // workgroups, except the workgroup who starts the tile, will write to
                        // partials.
                        else
                        {
                            sk_ops.StorePartial(kargs, cta_idx, accum_block_tile);
                            sk_ops.SignalStorePartialDone(kargs, cta_idx);
                            // Once the workgroup writes to partials, it has no more work to do
                            // for this tile.
                            break;
                        }
                    }
                }
            }
            else
            {
                static_assert(
                    "An implementation does not exist for the chosen reduction strategy.");
            }

            // Prepare for next Stream-K loop iteration.
            iter_start = tile_iter_end;
            block_sync_lds();
        }
    }

    // Non-persistent kernel entry point
    template <bool U = PersistentKernel, bool V = IsStreamK, typename KArgs>
    CK_TILE_DEVICE std::enable_if_t<!U && !V> operator()(KArgs kargs) const
    {
        const auto [i_m, i_n] = GetTileCoordinates(kargs);

        const SplitKBatchOffset splitk_batch_offset(kargs);

        // options
        std::array<const ADataType*, NumATensor> as_ptr;
        static_for<0, NumATensor, 1>{}([&](auto i) {
            as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                        splitk_batch_offset.as_k_split_offset[i] / APackedSize;
        });

        std::array<const BDataType*, NumBTensor> bs_ptr;
        static_for<0, NumBTensor, 1>{}([&](auto i) {
            bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                        splitk_batch_offset.bs_k_split_offset[i] / BPackedSize;
        });

        // Calculate output offset from tile partitioner and apply to output pointer
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
        if constexpr(has_tile_partitioner_output_offset)
        {
            const index_t output_offset = TilePartitioner::GetOutputOffset(kargs, blockIdx.z);
            e_ptr += output_offset;
        }

        // allocate LDS
        __shared__ char smem_ptr[GetSmemSize()];

        SelfType::RunGemm(
            as_ptr, bs_ptr, kargs.ds_ptr, e_ptr, smem_ptr, kargs, splitk_batch_offset, i_m, i_n);
    }

    /**
     * @brief Entry point for the Stream-K Kernel with non-persistent DP.
     *
     * @par Overview
     *     For the Non-Persistent kernel, each data parallel workgroup will
     *     compute the results for their assigned macro-tile by calling `BaseGemm()`.
     *     The Stream-K workgroups will do their assigned work by calling
     *     `StreamKGemm()`, which calls `BaseGemm()` in the Stream-K loop.
     */
    template <bool U = PersistentKernel, bool V = IsStreamK, typename KArgs>
    CK_TILE_DEVICE std::enable_if_t<!U && V> operator()(KArgs kargs) const
    {
        // Allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        index_t block_idx   = ck_tile::get_block_1d_id();
        index_t grid_size   = kargs.tile_partitioner.grid_size().x;
        index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();
        index_t dp_ctas     = kargs.tile_partitioner.get_dp_ctas();
        bool is_dp_ctas     = block_idx < kargs.tile_partitioner.get_dp_ctas();

        block_idx = kargs.tile_partitioner.remap_xcd(block_idx, grid_size, kargs.num_xccs);
        // Check if has the data parallel section
        if(is_dp_ctas)
        {
            const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(block_idx);
            const index_t i_m           = c_macro_tile_idx[I0] * TilePartitioner::MPerBlock;
            const index_t i_n           = c_macro_tile_idx[I1] * TilePartitioner::NPerBlock;

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}(
                [&](auto i) { as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]); });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}(
                [&](auto i) { bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]); });

            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            RunGemm(as_ptr,
                    bs_ptr,
                    kargs.ds_ptr,
                    e_ptr,
                    smem_ptr_0,
                    kargs,
                    dp_num_loop,
                    i_m,
                    i_n,
                    kargs.K);
        }
        else
        {
            // Stream-K
            StreamKGemm(kargs, block_idx - dp_ctas, smem_ptr_0);
        }
    }

    // Persistent kernel entry point
    template <bool U = PersistentKernel, bool V = IsStreamK, typename KArgs>
    CK_TILE_DEVICE std::enable_if_t<U && !V> operator()(KArgs kargs) const
    {
        const auto grid_size = GetGridSize();
        const auto num_tiles = GetNumTiles(kargs.M, kargs.N);
        const auto num_work  = amd_wave_read_first_lane(num_tiles * kargs.k_batch);
        auto block_id        = GetBlockId();

        while(block_id < num_work)
        {
            s_waitcnt_barrier();
            const auto tile_idx = amd_wave_read_first_lane(block_id % num_tiles);
            const auto [iM, iN] = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(tile_idx);
            // Apply pivot to M tile index first, then use the same pivoted index
            // for both data-tile selection and chunk-signal wait.
            auto iM_eff = amd_wave_read_first_lane(iM);

            if(kargs.async_input_scheduler.chunk_signals != nullptr)
            {
                const auto tile_idx_pivot =
                    amd_wave_read_first_lane(kargs.async_input_scheduler.tile_idx_pivot_m);
                const auto tiles_m = amd_wave_read_first_lane(
                    integer_divide_ceil(kargs.M, TilePartitioner::MPerBlock));
                if(tiles_m > 0)
                {
                    iM_eff = amd_wave_read_first_lane((iM_eff + tile_idx_pivot) % tiles_m);
                }
            }

            const index_t i_m = amd_wave_read_first_lane(iM_eff * TilePartitioner::MPerBlock);
            const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);

            // Synchronize with producer to ensure input data is ready before processing tile
            if(kargs.async_input_scheduler.chunk_signals != nullptr)
            {
                const auto tiles_per_chunk =
                    amd_wave_read_first_lane(kargs.async_input_scheduler.tiles_per_chunk_m);
                const auto num_chunks =
                    amd_wave_read_first_lane(kargs.async_input_scheduler.num_chunks);
                if(tiles_per_chunk > 0 && num_chunks > 0)
                {
                    // Pivot allows rotating chunk assignments for load balancing
                    const auto chunk_idx =
                        amd_wave_read_first_lane((iM_eff / tiles_per_chunk) % num_chunks);
                    workgroup_barrier chunk_barrier(kargs.async_input_scheduler.chunk_signals);
                    chunk_barrier.wait_eq_wave(/*value=*/1, /*offset=*/chunk_idx);
                }
            }

            // Get the SplitK offset for this block
            const auto k_batch = amd_wave_read_first_lane(block_id / num_tiles);
            const SplitKBatchOffset splitk_batch_offset(kargs, k_batch);

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}([&](auto i) {
                as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]) +
                            splitk_batch_offset.as_k_split_offset[i] / APackedSize;
            });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}([&](auto i) {
                bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]) +
                            splitk_batch_offset.bs_k_split_offset[i] / BPackedSize;
            });

            // Calculate output offset from tile partitioner and apply to output pointer
            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);
            if constexpr(has_tile_partitioner_output_offset)
            {
                const index_t output_offset = TilePartitioner::GetOutputOffset(kargs, k_batch);
                e_ptr += output_offset;
            }

            // allocate LDS
            __shared__ char smem_ptr[GetSmemSize()];
            // Run the GEMM
            SelfType::RunGemm(as_ptr,
                              bs_ptr,
                              kargs.ds_ptr,
                              e_ptr,
                              smem_ptr,
                              kargs,
                              splitk_batch_offset,
                              i_m,
                              i_n);

            // Advance to the next work item
            block_id += grid_size;
            if(block_id >= num_work)
            {
                break;
            }
        }
    }

    /**
     * @brief Entry point for the Stream-K Kernel with persistent DP.
     *
     * @par Overview
     *     For the Persistent kernel, each workgroup will first compute their
     *     assigned data-parallel tiles. Each data parallel tile will be computed
     *     by calling `BaseGemm()`. Then the workgroups will proceed with the
     *     Stream-K portion by calling `StreamKGemm()`, which calls `BaseGemm()`
     *     in the Stream-K loop.
     */
    template <bool U = PersistentKernel, bool V = IsStreamK, typename KArgs>
    CK_TILE_DEVICE std::enable_if_t<U && V> operator()(KArgs kargs) const
    {
        // Allocate LDS
        __shared__ char smem_ptr_0[GetSmemSize()];

        index_t block_idx   = ck_tile::get_block_1d_id();
        index_t grid_size   = kargs.tile_partitioner.grid_size().x;
        index_t dp_num_loop = kargs.tile_partitioner.get_iters_per_tile();
        block_idx = kargs.tile_partitioner.remap_xcd(block_idx, grid_size, kargs.num_xccs);

        // Data-parallel section
        for(index_t tile_idx = block_idx; tile_idx < kargs.tile_partitioner.get_dp_tiles();
            tile_idx += kargs.tile_partitioner.get_max_active_wgs())
        {
            const auto c_macro_tile_idx = kargs.tile_partitioner.get_output_tile_index(tile_idx);
            const index_t i_m           = c_macro_tile_idx[I0] * TilePartitioner::MPerBlock;
            const index_t i_n           = c_macro_tile_idx[I1] * TilePartitioner::NPerBlock;

            std::array<const ADataType*, NumATensor> as_ptr;
            static_for<0, NumATensor, 1>{}(
                [&](auto i) { as_ptr[i] = static_cast<const ADataType*>(kargs.as_ptr[i]); });

            std::array<const BDataType*, NumBTensor> bs_ptr;
            static_for<0, NumBTensor, 1>{}(
                [&](auto i) { bs_ptr[i] = static_cast<const BDataType*>(kargs.bs_ptr[i]); });

            EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

            RunGemm(as_ptr,
                    bs_ptr,
                    kargs.ds_ptr,
                    e_ptr,
                    smem_ptr_0,
                    kargs,
                    dp_num_loop,
                    i_m,
                    i_n,
                    kargs.K);
            block_sync_lds();
        }

        // Stream-K section
        StreamKGemm(kargs, block_idx, smem_ptr_0);
    }
};
} // namespace ck_tile
