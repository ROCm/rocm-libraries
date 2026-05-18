#pragma once

#include "ck_tile/host/stream_config.hpp"
#include "ck_tile/ops/epilogue/cshuffle_epilogue.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "ck_tile/ops/swiglu/swiglu_kernel.hpp"
#include "ck_tile/ops/swiglu/act_and_mul.hpp"

#include "config.hpp"

namespace ck_tile::swiglu_example {

struct UniversalInvoker
{
    template <typename GemmConfig,
              typename ADataType,
              typename BDataType,
              typename DsDataType,
              typename AccDataType,
              typename CDataType,
              typename ALayout,
              typename BLayout,
              typename DsLayout,
              typename ELayout,
              bool Persistent,
              typename CDEElementWise>
    static float swiglu(const ck_tile::SwiGLUHostArgs& args, const ck_tile::stream_config& s)
    {
        using GemmShape = ck_tile::TileGemmShape<
            sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
            sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
            sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile>,
            GemmConfig::PermuteA,
            GemmConfig::PermuteB>;

        using TilePartitioner =
            ck_tile::GemmSpatiallyLocalTilePartitioner<GemmShape,
                                                       GemmConfig::TileParitionerGroupNum,
                                                       GemmConfig::TileParitionerM01>;

        using GemmUniversalTraits =
            ck_tile::TileGemmUniversalTraits<GemmConfig::kPadM,
                                             GemmConfig::kPadN,
                                             GemmConfig::kPadK,
                                             GemmConfig::DoubleSmemBuffer,
                                             ALayout,
                                             BLayout,
                                             ELayout,
                                             GemmConfig::TransposeC,
                                             GemmConfig::UseStructuredSparsity,
                                             Persistent,
                                             GemmConfig::NumWaveGroups,
                                             GemmConfig::Preshuffle>;

        constexpr auto scheduler = GemmConfig::Scheduler;

        using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<ADataType,
                                                                           BDataType,
                                                                           AccDataType,
                                                                           GemmShape,
                                                                           GemmUniversalTraits,
                                                                           scheduler>;

        using GemmPipeline = typename PipelineTypeTraits<
            GemmConfig::Pipeline>::template GemmPipeline<UniversalGemmProblem>;

        using GemmEpilogue = ck_tile::CShuffleEpilogue<
            ck_tile::CShuffleEpilogueProblem<ADataType,
                                             BDataType,
                                             DsDataType,
                                             AccDataType,
                                             CDataType,
                                             DsLayout,
                                             ELayout,
                                             CDEElementWise,
                                             TilePartitioner::MPerBlock,
                                             TilePartitioner::NPerBlock,
                                             GemmConfig::M_Warp,
                                             GemmConfig::N_Warp,
                                             GemmConfig::M_Warp_Tile,
                                             GemmConfig::N_Warp_Tile,
                                             GemmConfig::K_Warp_Tile,
                                             UniversalGemmProblem::TransposeC,
                                             GemmConfig::NumWaveGroups,
                                             false, /*FixedVectorSize_*/
                                             1,     /*VectorSizeC_*/
                                             1,     /*BlockedXDLN_PerWarp_*/
                                             GemmConfig::DoubleSmemBuffer /*DoubleSmemBuffer*/>>;

        using Kernel = SwiGLUKernel<TilePartitioner, GemmPipeline, GemmEpilogue, ck_tile::SwishMul>;
        // using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

        auto kargs = Kernel::MakeKernelArgs(args);

        const dim3 grids  = Persistent ? Kernel::MaxOccupancyGridSize(s)
                                       : Kernel::GridSize(args.m, args.n, args.k_batch); // k_hatch
        const dim3 blocks = Kernel::BlockSize();

        if(!Kernel::IsSupportedArgument(kargs))
        {
            throw std::runtime_error("Wrong! Arguments not supported! Skipping gemm!\n");
        }

        if(s.log_level_ > 0)
        {
            std::cout << "Launching kernel with args: " << Kernel::GetName() << '\n'
                      << "shape: " << GemmShape::GetName() << '\n'
                      << "problem: " << UniversalGemmProblem::GetName() << '\n'
                      << "pipeline: " << GemmPipeline::GetName() << '\n'
                      << "grid: {" << grids.x << ", " << grids.y << ", " << grids.z << "}"
                      << ", blocks: {" << blocks.x << ", " << blocks.y << ", " << blocks.z << "}"
                      << std::endl;
        }

        // Declare rotating_mem_ptr here so it stays in scope until it is needed
        std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr0;
        std::unique_ptr<ck_tile::RotatingMemWrapper<ADataType, BDataType>> rotating_mem_ptr1;
        std::function<void()> preprocess;

        auto clear_gemm_output = [&]() {
            if(args.k_batch > 1)
                hipGetErrorString(hipMemsetAsync(
                    args.c_ptr, 0, args.m * args.n * sizeof(CDataType), s.stream_id_));
        };

        if(s.flush_cache_)
        {
            std::cout << "Flushing cache..." << std::endl;

            ck_tile::HostTensor<ADataType> a_m(ck_tile::host_tensor_descriptor(
                args.m, args.k, args.a_stride, is_row_major(ALayout{})));
            ck_tile::HostTensor<BDataType> b0_n(ck_tile::host_tensor_descriptor(
                args.k, args.n, args.b0_stride, is_row_major(BLayout{})));
            ck_tile::HostTensor<BDataType> b1_n(ck_tile::host_tensor_descriptor(
                args.k, args.n, args.b1_stride, is_row_major(BLayout{})));

            auto size_a_buffer  = a_m.get_element_space_size_in_bytes();
            auto size_b0_buffer = b0_n.get_element_space_size_in_bytes();
            auto size_b1_buffer = b1_n.get_element_space_size_in_bytes();

            rotating_mem_ptr0 = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b0_buffer);
            rotating_mem_ptr1 = std::make_unique<ck_tile::RotatingMemWrapper<ADataType, BDataType>>(
                kargs.as_ptr[0], kargs.bs_ptr[0], s.rotating_count_, size_a_buffer, size_b1_buffer);
            rotating_mem_ptr0->Print();
            rotating_mem_ptr1->Print();

            preprocess = [&]() {
                ck_tile::flush_icache();
                rotating_mem_ptr0->Next();
                rotating_mem_ptr1->Next();
                clear_gemm_output();
            };
        }
        else
        {
            preprocess = clear_gemm_output;
        }

        return ck_tile::launch_kernel_time_mask(
            s,
            preprocess,
            ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
    }
};

} // namespace ck_tile::swiglu_example
