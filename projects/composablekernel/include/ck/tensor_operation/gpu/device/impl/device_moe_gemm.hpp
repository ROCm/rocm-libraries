// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <iostream>
#include <sstream>

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp"
#include "ck/host_utility/device_prop.hpp"
#include "ck/host_utility/kernel_launch.hpp"
#include "ck/host_utility/flush_cache.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename DsDataType,
          typename CDataType,
          typename GemmAccDataType,
          typename CShuffleDataType,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          GemmSpecialization GemmSpec,
          index_t BlockSize,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t KPerBlock,
          index_t AK1,
          index_t BK1,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t MXdlPerWave,
          index_t NXdlPerWave,
          typename ABlockTransferThreadClusterLengths_AK0_M_AK1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_AK1,
          bool ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_BK1,
          bool BBlockLdsExtraN,
          index_t CShuffleMXdlPerWavePerShuffle,
          index_t CShuffleNXdlPerWavePerShuffle,
          typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          typename CDEShuffleBlockTransferScalarPerVectors,
          BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
          BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1,
          index_t ActivationOP                        = 0,
          bool NSwizzle                               = false,
          bool IsInputGemm                            = true,
          bool MulRoutedWeight                        = true,
          bool PerTokenQuant                          = true,
          typename IndexType                          = index_t,
          typename ComputeTypeA                       = CDataType,
          typename ComputeTypeB                       = ComputeTypeA,
          typename LDSTypeA                           = ComputeTypeA,
          typename LDSTypeB                           = ComputeTypeB>
struct DeviceMoeGemm : public DeviceGemmMultipleDSplitKBPreShuffle<ALayout,
                                                                   BLayout,
                                                                   DsLayout,
                                                                   CLayout,
                                                                   ADataType,
                                                                   BDataType,
                                                                   DsDataType,
                                                                   CDataType,
                                                                   AElementwiseOperation,
                                                                   BElementwiseOperation,
                                                                   CElementwiseOperation>
{
    static constexpr index_t NumDTensor = DsDataType::Size();
    using GridwiseGemm =
        GridwiseMoeGemm<ALayout,
                        BLayout,
                        DsLayout,
                        CLayout,
                        ADataType,
                        BDataType,
                        GemmAccDataType,
                        CShuffleDataType,
                        DsDataType,
                        CDataType,
                        AElementwiseOperation,
                        BElementwiseOperation,
                        CElementwiseOperation,
                        GemmSpec,
                        BlockSize,
                        MPerBlock,
                        NPerBlock,
                        KPerBlock,
                        AK1,
                        BK1,
                        MPerXDL,
                        NPerXDL,
                        MXdlPerWave,
                        NXdlPerWave,
                        ABlockTransferThreadClusterLengths_AK0_M_AK1,
                        ABlockTransferThreadClusterArrangeOrder,
                        ABlockTransferSrcAccessOrder,
                        ABlockTransferSrcVectorDim,
                        ABlockTransferSrcScalarPerVector,
                        ABlockTransferDstScalarPerVector_AK1,
                        false,
                        ABlockLdsExtraM,
                        BBlockTransferThreadClusterLengths_BK0_N_BK1,
                        BBlockTransferThreadClusterArrangeOrder,
                        BBlockTransferSrcAccessOrder,
                        BBlockTransferSrcVectorDim,
                        BBlockTransferSrcScalarPerVector,
                        BBlockTransferDstScalarPerVector_BK1,
                        false,
                        BBlockLdsExtraN,
                        CShuffleMXdlPerWavePerShuffle,
                        CShuffleNXdlPerWavePerShuffle,
                        CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
                        CDEShuffleBlockTransferScalarPerVectors,
                        BlkGemmPipeSched,
                        BlkGemmPipelineVer,
                        ActivationOP,
                        NSwizzle,
                        IsInputGemm,
                        MulRoutedWeight,
                        PerTokenQuant,
                        IndexType,
                        ComputeTypeA,
                        ComputeTypeB,
                        LDSTypeA,
                        LDSTypeB>;

    using Argument = typename GridwiseGemm::Argument;

    static constexpr index_t APackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<ADataType>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    static constexpr index_t BPackedSize = []() {
        if constexpr(is_same_v<remove_cvref_t<BDataType>, pk_i4_t>)
            return 2;
        else
            return 1;
    }();

    int GetPreShuffleParameters() override { return NPerXDL; }

    // Invoker
    struct Invoker : public BaseInvoker
    {
        float Run(const Argument& arg, const StreamConfig& stream_config = StreamConfig{})
        {
            if(stream_config.log_level_ > 0)
            {
                arg.Print();
            }

            if(!GridwiseGemm::CheckValidity(arg))
            {
                throw std::runtime_error("wrong! GridwiseGemm has invalid setting");
            }

            index_t gdx, gdy, gdz;
            std::tie(gdx, gdy, gdz) = GridwiseGemm::CalculateGridSize(arg.M, arg.N);

            float ave_time = 0;

            index_t k_grain = arg.KBatch * KPerBlock;
            index_t K_split = (arg.K + k_grain - 1) / k_grain * KPerBlock;

            const bool has_main_k_block_loop = GridwiseGemm::CalculateHasMainKBlockLoop(K_split);

            const auto RunKernel = [&](const auto& kernel) {
                if(stream_config.flush_cache)
                {

                    std::array<std::size_t, NumDTensor> DsSize;

                    Argument arg_ = arg;

                    const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
                        arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
                    const auto b_grid_desc_bk0_n_bk1 = GridwiseGemm::MakeBGridDescriptor_BK0_N_BK1(
                        arg_.K, arg_.KPadded, arg_.N, arg_.NPadded, arg_.StrideB, arg_.BK0);

                    auto size_a_buffer = a_grid_desc_ak0_m_ak1.GetElementSpaceSize() *
                                         sizeof(ADataType) / APackedSize;
                    auto size_b_buffer = b_grid_desc_bk0_n_bk1.GetElementSpaceSize() *
                                         sizeof(BDataType) / BPackedSize;

                    const auto ds_grid_desc_m_n = GridwiseGemm::MakeDsGridDescriptor_M_N(
                        arg_.M, arg_.MPadded, arg_.N, arg_.NPadded, arg_.StrideDs);

                    static_for<0, NumDTensor, 1>{}([&](auto i) {
                        using DDataType = remove_cvref_t<tuple_element_t<i.value, DsDataType>>;
                        DsSize[i] = ds_grid_desc_m_n[i].GetElementSpaceSize() * sizeof(DDataType);
                    });
                    ck::utility::RotatingMemWrapperMultiD<Argument, DsDataType> rotating_mem(
                        arg_, stream_config.rotating_count, size_a_buffer, size_b_buffer, DsSize);
                    rotating_mem.Print();

                    auto run_flush_cache = [&]() {
                        // flush icache
                        ck::utility::flush_icache();
                        // rotating mem
                        rotating_mem.Next();
                        // clear c mem
                        if(arg_.KBatch > 1)
                            hipGetErrorString(hipMemsetAsync(arg_.p_c_grid,
                                                             0,
                                                             arg_.M * arg_.N * sizeof(CDataType),
                                                             stream_config.stream_id_));
                    };

                    ave_time = ck::utility::launch_and_time_kernel_with_preprocess<false>(
                        stream_config,
                        run_flush_cache,
                        kernel,
                        dim3(gdx, gdy, gdz),
                        dim3(BlockSize),
                        0,
                        arg_);
                }
                else
                {
                    if(arg.KBatch > 1)
                        hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
                                                         0,
                                                         arg.M * arg.N * sizeof(CDataType),
                                                         stream_config.stream_id_));

                    ave_time = launch_and_time_kernel(
                        stream_config, kernel, dim3(gdx, gdy, gdz), dim3(BlockSize), 0, arg);
                }
            };

            constexpr auto estimated_reg_a = MPerBlock * KPerBlock * sizeof(ADataType) / BlockSize /
                                             4 * (1 + GridwiseGemm::NWave);
            constexpr auto estimated_reg_b = NPerBlock * KPerBlock * sizeof(BDataType) / BlockSize /
                                             4 * (2) * (IsInputGemm ? 2 : 1);
            constexpr auto estimated_reg_c = MPerBlock * NPerBlock * sizeof(GemmAccDataType) /
                                             BlockSize / 4 * (IsInputGemm ? 2 : 1);
            constexpr auto estimated_reg_total =
                estimated_reg_a + estimated_reg_b + estimated_reg_c;

            constexpr index_t minimum_occupancy = (estimated_reg_total >= 256) ? 1 : 2;

            constexpr auto MemoryDataOp =
                IsInputGemm ? InMemoryDataOperationEnum::Set : InMemoryDataOperationEnum::AtomicAdd;
            if(has_main_k_block_loop)
            {
                // Tail number always full
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    {
                        if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                        {
                            const auto kernel = kernel_moe_gemm<GridwiseGemm,
                                                                true,
                                                                MemoryDataOp,
                                                                minimum_occupancy,
                                                                TailNumber::Odd>;
                            RunKernel(kernel);
                        }
                        else
                        {
                            const auto kernel = kernel_moe_gemm<GridwiseGemm,
                                                                true,
                                                                MemoryDataOp,
                                                                minimum_occupancy,
                                                                TailNumber::Even>;
                            RunKernel(kernel);
                        }
                    }
                }
                else if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v2 ||
                                  BlkGemmPipelineVer == BlockGemmPipelineVersion::v3)
                {
                    if(GridwiseGemm::CalculateKBlockLoopTailNum(K_split) == TailNumber::Odd)
                    {
                        const auto kernel = kernel_moe_gemm_2lds<GridwiseGemm,
                                                                 true,
                                                                 MemoryDataOp,
                                                                 minimum_occupancy,
                                                                 TailNumber::Odd>;
                        RunKernel(kernel);
                    }
                    else
                    {
                        const auto kernel = kernel_moe_gemm_2lds<GridwiseGemm,
                                                                 true,
                                                                 MemoryDataOp,
                                                                 minimum_occupancy,
                                                                 TailNumber::Even>;
                        RunKernel(kernel);
                    }
                }
                else
                {
                    throw std::runtime_error("todo: only v1 & v2 support now");
                }
            }
#if 1
            else
            {
                // Tail number always 1
                if constexpr(BlkGemmPipelineVer == BlockGemmPipelineVersion::v1)
                {
                    const auto kernel = kernel_moe_gemm<GridwiseGemm,
                                                        true,
                                                        InMemoryDataOperationEnum::Set,
                                                        minimum_occupancy,
                                                        TailNumber::Odd>;
                    RunKernel(kernel);
                }
            }
#endif

            return ave_time;
        }

        // polymorphic
        float Run(const BaseArgument* p_arg,
                  const StreamConfig& stream_config = StreamConfig{}) override
        {
            return Run(*dynamic_cast<const Argument*>(p_arg), stream_config);
        }
    };

    static constexpr bool IsValidCompilationParameter()
    {
        // TODO: properly implement this check
        return true;
    }

    static bool IsSupportedArgument(const Argument& arg)
    {
        // only impl kbatch 1 now
        if(arg.KBatch > 1)
        {
            return false;
        }
        if(!ck::is_xdl_supported())
        {
            return false;
        }

        if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> && arg.KBatch > 1)
        {
            return false;
        }

        if((arg.K % AK1 != 0 || arg.K % BK1 != 0) && !(GemmSpec == GemmSpecialization::MKPadding ||
                                                       GemmSpec == GemmSpecialization::NKPadding ||
                                                       GemmSpec == GemmSpecialization::MNKPadding ||
                                                       GemmSpec == GemmSpecialization::KPadding))
        {
            return false;
        }
        if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
        {
            return false;
        }

        return GridwiseGemm::CheckValidity(arg);
    }

    // polymorphic
    bool IsSupportedArgument(const BaseArgument* p_arg) override
    {
        return IsSupportedArgument(*dynamic_cast<const Argument*>(p_arg));
    }

    static auto MakeArgument(const void* p_sorted_token_ids,
                             const void* p_sorted_expert_ids,
                             const void* p_max_token_id,
                             const void* p_a,
                             const void* p_b,
                             std::array<const void*, NumDTensor> p_ds,
                             void* p_c,
                             index_t NumTokens,
                             index_t TopK,
                             index_t M,
                             index_t N,
                             index_t K,
                             index_t StrideA,
                             index_t StrideB,
                             std::array<index_t, NumDTensor> StrideDs,
                             index_t StrideC,
                             index_t KBatch,
                             AElementwiseOperation a_element_op,
                             BElementwiseOperation b_element_op,
                             CElementwiseOperation c_element_op)
    {
        return Argument{static_cast<const index_t*>(p_sorted_token_ids),
                        static_cast<const index_t*>(p_sorted_expert_ids),
                        static_cast<const index_t*>(p_max_token_id),
                        static_cast<const ADataType*>(p_a),
                        static_cast<const BDataType*>(p_b),
                        p_ds,
                        static_cast<CDataType*>(p_c),
                        NumTokens,
                        TopK,
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        StrideDs,
                        StrideC,
                        KBatch,
                        a_element_op,
                        b_element_op,
                        c_element_op};
    }

    static auto MakeInvoker() { return Invoker{}; }

    // polymorphic
    std::unique_ptr<BaseArgument> MakeArgumentPointer(const void* p_a,
                                                      const void* p_b,
                                                      std::array<const void*, NumDTensor> p_ds,
                                                      void* p_c,
                                                      index_t M,
                                                      index_t N,
                                                      index_t K,
                                                      index_t StrideA,
                                                      index_t StrideB,
                                                      std::array<ck::index_t, NumDTensor> StrideDs,
                                                      index_t StrideC,
                                                      index_t KBatch,
                                                      AElementwiseOperation a_element_op,
                                                      BElementwiseOperation b_element_op,
                                                      CElementwiseOperation c_element_op) override
    {
        return std::make_unique<Argument>(nullptr,
                                          nullptr,
                                          nullptr,
                                          static_cast<const ADataType*>(p_a),
                                          static_cast<const BDataType*>(p_b),
                                          p_ds,
                                          static_cast<CDataType*>(p_c),
                                          M, // randoms set, no use
                                          0,
                                          M,
                                          N,
                                          K,
                                          StrideA,
                                          StrideB,
                                          StrideDs,
                                          StrideC,
                                          KBatch,
                                          a_element_op,
                                          b_element_op,
                                          c_element_op);
    }

    // polymorphic
    std::unique_ptr<BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>(Invoker{});
    }

    // polymorphic
    std::string GetTypeString() const override
    {
        auto str = std::stringstream();

        std::map<BlockGemmPipelineScheduler, std::string> BlkGemmPipelineSchedulerToString{
            {BlockGemmPipelineScheduler::Intrawave, "Intrawave"},
            {BlockGemmPipelineScheduler::Interwave, "Interwave"}};

        std::map<BlockGemmPipelineVersion, std::string> BlkGemmPipelineVersionToString{
            {BlockGemmPipelineVersion::v1, "v1"}, {BlockGemmPipelineVersion::v2, "v2"}};

        // clang-format off
        str << "DeviceMoeGEmm"
            << "<"
            << getGemmSpecializationString(GemmSpec) << ", "
            << std::string(ALayout::name)[0]
            << std::string(BLayout::name)[0]
            << std::string(CLayout::name)[0]
            << ">"
            << " BlkSize: "
            << BlockSize << ", "
            << "BlkTile: "
            << MPerBlock<<"x"<<NPerBlock<<"x"<<KPerBlock << ", "
            << "WaveTile: "
            << MPerXDL<<"x"<<NPerXDL << ", "
            << "WaveMap: "
            << MXdlPerWave<<"x" << NXdlPerWave<<", "
            << "VmemReadVec: "
            << ABlockTransferSrcScalarPerVector<<"x"<<BBlockTransferSrcScalarPerVector<<", "
            << "BlkGemmPipelineScheduler: "
            << BlkGemmPipelineSchedulerToString[BlkGemmPipeSched] << ", "
            << "BlkGemmPipelineVersion: "
            << BlkGemmPipelineVersionToString[BlkGemmPipelineVer] << ", "
            << "BlkGemmPipelinePrefetchStages: "
            << GridwiseGemm::BlockwiseGemmPipe::PrefetchStages;
        // clang-format on

        return str.str();
    }
};

} // namespace device
} // namespace tensor_operation
} // namespace ck
