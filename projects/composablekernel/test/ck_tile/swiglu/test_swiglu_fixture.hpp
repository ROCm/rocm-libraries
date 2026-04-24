// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <gtest/gtest.h>

#include "ck_tile/core/container/sequence.hpp"
#include "ck_tile/core/container/tuple.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/host/check_err.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/host/host_tensor.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/host/reference/reference_swiglu.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/swiglu/swiglu_kernel.hpp"

#include "swiglu_test_utils.hpp"
#include "test_utils.hpp"

#include <cstddef>
#include <iostream>
#include <memory>

namespace ck_tile {
template <typename ADataType,
          typename BDataType,
          typename CDataType,
          typename AccDataType,
          typename ALayout_,
          typename BLayout_,
          typename CLayout_,
          index_t TileM,
          index_t TileN,
          index_t TileK,
          index_t WarpM,
          index_t WarpN,
          index_t WarpK,
          index_t WarpTileM,
          index_t WarpTileN,
          index_t WarpTileK,
          bool PadM,
          bool PadN,
          bool PadK,
          ck_tile::GemmPipeline Pipeline_,
          ck_tile::GemmPipelineScheduler Scheduler_,
          index_t BlockPerCu_,
          typename ActMul_>
struct TestParams
{
    using DTypes = tuple<ADataType, BDataType, CDataType, AccDataType>;
    DTypes dtypes;

    using ALayout = ALayout_;
    using BLayout = BLayout_;
    using CLayout = CLayout_;
    using Layouts = tuple<ALayout, BLayout, CLayout>;
    Layouts layouts;

    using Padding = utils::boolseq<PadM, PadN, PadK>;
    Padding padding;

    using ActMul = ActMul_;
    ActMul act_mul;

    using Pipeline = constant<Pipeline_>;
    Pipeline pipeline;

    using Scheduler = constant<Scheduler_>;
    Scheduler scheduler;

    using BlockPerCu = constant<BlockPerCu_>;
    BlockPerCu block_per_cu;

    using Tile = sequence<TileM, TileN, TileK>;
    Tile tile;

    using Warp = sequence<WarpM, WarpN, WarpK>;
    Warp warp;

    using WarpTile = sequence<WarpTileM, WarpTileN, WarpTileK>;
    WarpTile warp_tile;

    constexpr static bool DoubleSmemBuffer =
        Pipeline_ == GemmPipeline::COMPUTE_V4 || Pipeline_ == GemmPipeline::COMPUTE_ASYNC;
    constexpr static bool TransposeC                = false;
    constexpr static bool StructuredSparsity        = false;
    constexpr static bool Persistent                = false;
    constexpr static bool Preshuffle                = false;
    constexpr static index_t NumWaveGroup           = 1;
    constexpr static index_t TileParitionerGroupNum = 8;
    constexpr static index_t TileParitionerM01      = 4;

    using GemmShape = ck_tile::TileGemmShape<Tile, Warp, WarpTile>;

    using TilePartitioner = ck_tile::
        GemmSpatiallyLocalTilePartitioner<GemmShape, TileParitionerGroupNum, TileParitionerM01>;

    using BaseGemmUniversalTraits = ck_tile::TileGemmUniversalTraits<PadM,
                                                                     PadN,
                                                                     PadK,
                                                                     DoubleSmemBuffer,
                                                                     ALayout,
                                                                     BLayout,
                                                                     CLayout,
                                                                     TransposeC,
                                                                     StructuredSparsity,
                                                                     Persistent,
                                                                     NumWaveGroup,
                                                                     Preshuffle>;

    using BaseGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                  BDataType,
                                                                  AccDataType,
                                                                  GemmShape,
                                                                  BaseGemmUniversalTraits,
                                                                  Scheduler_,
                                                                  element_wise::PassThrough,
                                                                  element_wise::PassThrough,
                                                                  ADataType>;
    using BaseGemmPipeline =
        typename GemmPipelineTypeSelector<Pipeline_, BaseGemmProblem>::pipeline;

    using EpilogueProblem  = ck_tile::CShuffleEpilogueProblem<ADataType,
                                                              BDataType,
                                                              tuple<>,
                                                              AccDataType,
                                                              CDataType,
                                                              tuple<>,
                                                              CLayout,
                                                              element_wise::PassThrough,
                                                              TilePartitioner::MPerBlock,
                                                              TilePartitioner::NPerBlock,
                                                              WarpM,
                                                              WarpN,
                                                              WarpTileM,
                                                              WarpTileN,
                                                              WarpTileK,
                                                              BaseGemmProblem::TransposeC,
                                                              1,     /*kNumWaveGroups_*/
                                                              false, /*FixedVectorSize_*/
                                                              1,     /*VectorSizeC_*/
                                                              1,     /*BlockedXDLN_PerWarp_*/
                                                              DoubleSmemBuffer>;
    using EpiloguePipeline = ck_tile::CShuffleEpilogue<EpilogueProblem>;

    using Kernel = SwiGLUKernel<TilePartitioner, BaseGemmPipeline, EpiloguePipeline, ActMul>;

    using HostArgs = typename Kernel::HostArgs;

    struct TestCase
    {
        index_t a_stride{};
        index_t b_stride{};
        index_t c_stride{};

        std::unique_ptr<HostTensor<ADataType>> a_host;
        std::unique_ptr<HostTensor<BDataType>> b0_host;
        std::unique_ptr<HostTensor<BDataType>> b1_host;
        std::unique_ptr<HostTensor<CDataType>> c_host;
        std::unique_ptr<HostTensor<CDataType>> c_ref_host;

        std::unique_ptr<DeviceMem> a_dev;
        std::unique_ptr<DeviceMem> b0_dev;
        std::unique_ptr<DeviceMem> b1_dev;
        std::unique_ptr<DeviceMem> c_dev;

        auto setup(const utils::SwiGLUShape& shape) -> void
        {
            using namespace ck_tile::utils;

            a_stride = default_stride(0, shape.m, shape.k, shape.a_is_row_major);
            b_stride = default_stride(0, shape.k, shape.n, shape.b_is_row_major);
            c_stride = default_stride(0, shape.m, shape.n, shape.c_is_row_major);

            a_host  = make_host_tensor<ADataType>(shape.m, shape.k, shape.a_is_row_major, a_stride);
            b0_host = make_host_tensor<BDataType>(shape.k, shape.n, shape.b_is_row_major, b_stride);
            b1_host = make_host_tensor<BDataType>(shape.k, shape.n, shape.b_is_row_major, b_stride);
            c_host  = make_host_tensor<CDataType>(shape.m, shape.n, shape.c_is_row_major, c_stride);
            c_ref_host =
                make_host_tensor<CDataType>(shape.m, shape.n, shape.c_is_row_major, c_stride);

            init_normal(*a_host, 1);
            init_normal(*b0_host, 2);
            init_normal(*b1_host, 3);
            init_constant(*c_host, 0);
            init_constant(*c_ref_host, 0);

            a_dev  = make_device_buffer(*a_host);
            b0_dev = make_device_buffer(*b0_host);
            b1_dev = make_device_buffer(*b1_host);
            c_dev  = make_device_buffer(*c_host);
        }

        [[nodiscard]] auto make_host_args(const utils::SwiGLUShape& shape) const -> HostArgs
        {
            return {
                // Buffers
                .a_ptr  = a_dev->GetDeviceBuffer(),
                .b0_ptr = b0_dev->GetDeviceBuffer(),
                .b1_ptr = b1_dev->GetDeviceBuffer(),
                .c_ptr  = c_dev->GetDeviceBuffer(),
                // Shape
                .m = index_t(shape.m),
                .n = index_t(shape.n),
                .k = index_t(shape.k),
                // Strides
                .a_stride  = a_stride,
                .b0_stride = b_stride,
                .b1_stride = b_stride,
                .c_stride  = c_stride,
            };
        }

        auto invoke_kernel(const utils::SwiGLUShape& shape) -> void
        {
            auto stream_config = ck_tile::stream_config(nullptr, false, 0);

            auto hargs = make_host_args(shape);
            auto kargs = Kernel::MakeKernelArgs(hargs);
            if(!Kernel::IsSupportedArgument(kargs))
                throw std::runtime_error(
                    "Arguments not supported! (run with CK_TILE_LOGGING=1 for more info)");

            dim3 block_size = Kernel::BlockSize();
            dim3 grid_size;
            if constexpr(Persistent)
            {
                grid_size = Kernel::MaxOccupancyGridSize(stream_config);
            }
            else
            {
                grid_size = Kernel::GridSize(hargs.m, hargs.n, 1);
            }

            if(stream_config.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: \n"
                          << "  grid:  " << grid_size.x << ", " << grid_size.y << ", "
                          << grid_size.z << " \n " << "  block: " << block_size.x << ", "
                          << block_size.y << ", " << block_size.z << " \n"
                          << "  shape: " << shape << " \n " << "  block_per_cu: " << BlockPerCu_
                          << " \n ";
            }

            auto kernel_callable =
                ck_tile::make_kernel<BlockPerCu_>(Kernel{}, grid_size, block_size, 0, kargs);
            ck_tile::launch_kernel(stream_config, std::move(kernel_callable));
            utils::device_to_host(*c_dev, *c_host);
        }

        auto validate_output() -> void
        {
            double max_val{};
            swiglu_reference<AccDataType>(
                *a_host, *b0_host, *b1_host, *c_ref_host, {ActMul{}}, &max_val);

            auto [rtol, atol] =
                calculate_rtol_atol<ADataType, BDataType, AccDataType, CDataType>(1, max_val);
            atol = atol * 2;

            bool no_errors =
                ck_tile::check_err(*c_host, *c_ref_host, "Error: Incorrect results!", rtol, atol);
            EXPECT_TRUE(no_errors);
        }

        auto performance(const utils::SwiGLUShape& shape) -> void
        {
            ck_tile::stream_config stream_config{
                .stream_id_    = nullptr,
                .time_kernel_  = true,
                .log_level_    = 0,
                .cold_niters_  = 100,
                .nrepeat_      = 200,
                .is_gpu_timer_ = true,
                .flush_cache_  = true,
            };

            auto hargs = make_host_args(shape);
            auto kargs = Kernel::MakeKernelArgs(hargs);
            if(!Kernel::IsSupportedArgument(kargs))
                throw std::runtime_error(
                    "Arguments not supported! (run with CK_TILE_LOGGING=1 for more info)");

            dim3 block_size = Kernel::BlockSize();
            dim3 grid_size;
            if constexpr(Persistent)
            {
                grid_size = Kernel::MaxOccupancyGridSize(stream_config);
            }
            else
            {
                grid_size = Kernel::GridSize(hargs.m, hargs.n, 1);
            }

            // auto rotating_mem_ptr0    =
            // std::make_unique<ck_tile::RotatingMemWrapper<DTypes::ADataType, DTypes::BDataType>>(
            //         as0,
            //         bs0,
            //         stream_config_perf.rotating_count_,
            //         data.xs.host->get_element_space_size_in_bytes(),
            //         data.vs.host->get_element_space_size_in_bytes());
            // auto rotating_mem_ptr1 =
            // std::make_unique<ck_tile::RotatingMemWrapper<DTypes::ADataType, DTypes::BDataType>>(
            //         as1,
            //         bs1,
            //         stream_config_perf.rotating_count_,
            //         data.xs.host->get_element_space_size_in_bytes(),
            //         data.vs.host->get_element_space_size_in_bytes());
            // rotating_mem_ptr0->Print();
            // rotating_mem_ptr1->Print();

            // auto preprocess = [&](auto) {
            //     ck_tile::flush_icache();
            //     // rotating_mem_ptr0->Next();
            //     // rotating_mem_ptr1->Next();
            //     // clear_gemm_output();
            // };
            // auto preprocess = []() {};

            if(stream_config.log_level_ > 0)
            {
                std::cout << "Launching kernel with args: \n"
                          << "  grid:  " << grid_size.x << ", " << grid_size.y << ", "
                          << grid_size.z << " \n " << "  block: " << block_size.x << ", "
                          << block_size.y << ", " << block_size.z << " \n"
                          << "  shape: " << shape << " \n " << "  block_per_cu: " << BlockPerCu_
                          << " \n ";
            }

            auto kernel_callable =
                ck_tile::make_kernel<BlockPerCu_>(Kernel{}, grid_size, block_size, 0, kargs);
            auto ave_time = ck_tile::launch_kernel(stream_config, kernel_callable);

            auto ms    = static_cast<double>(ave_time);
            auto flops = static_cast<double>(utils::get_flops(shape));
            auto bytes =
                static_cast<double>(utils::get_num_bytes<ADataType, BDataType, CDataType>(shape));
            auto tflops = flops / ms / 1e9;
            auto gbps   = bytes / ms / 1e6;
            std::cout << " > " << ave_time << " ms, " << tflops << " TFlops, " << gbps << " GB/s\n";
        }
    };

    auto run_test(const utils::SwiGLUShape& shape) const -> void
    {
        TestCase test{};
        test.setup(shape);
        test.invoke_kernel(shape);
        test.validate_output();
        test.performance(shape);
    }
};

template <typename TestParams_, bool PadM, bool PadN, bool PadK>
auto with_padding()
{
    return TestParams{
        .dtypes       = typename TestParams_::DTypes{},
        .layouts      = typename TestParams_::Layouts{},
        .padding      = utils::boolseq<PadM, PadN, PadK>{},
        .act_mul      = typename TestParams_::ActMul{},
        .pipeline     = typename TestParams_::Pipeline{},
        .scheduler    = typename TestParams_::Scheduler{},
        .block_per_cu = typename TestParams_::BlockPerCu{},
        .tile         = typename TestParams_::Tile{},
        .warp         = typename TestParams_::Warp{},
        .warp_tile    = typename TestParams_::WarpTile{},
    };
}

template <typename TestParams_, typename ActMul>
auto with_act_mul()
{
    return TestParams{
        .dtypes       = typename TestParams_::DTypes{},
        .layouts      = typename TestParams_::Layouts{},
        .padding      = typename TestParams_::Padding{},
        .act_mul      = ActMul{},
        .pipeline     = typename TestParams_::Pipeline{},
        .scheduler    = typename TestParams_::Scheduler{},
        .block_per_cu = typename TestParams_::BlockPerCu{},
        .tile         = typename TestParams_::Tile{},
        .warp         = typename TestParams_::Warp{},
        .warp_tile    = typename TestParams_::WarpTile{},
    };
}
} // namespace ck_tile
