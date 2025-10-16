/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <miopen/config.h>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <iostream>

// Include Composable Kernel headers for 3D convolution with channel last layout
#if MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL
// Include CK tile utility header for 3D convolution
#include <miopen/solver/implicitgemm_ck_tile_util.hpp>
// Include specific CK tile headers if needed beyond what's in the utility
#include <miopen/solver/grouped_convolution_ck_tiles_utils.hpp>
#endif

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL

// Use type aliases from the new CK Tile utility header
using namespace miopen::solver::conv_ck_tile;

// Number of spatial dimensions for 3D convolution
static constexpr ck_tile::index_t NumDimSpatial = 3;

template <typename DataType>
using DeviceOp3DChannelLastFwd =
    ck_tile::GroupedConvFwdKernelArgs<
        ck_tile::GroupedConvTraits<NumDimSpatial,
                                   ck_tile::ConvolutionSpecialization::Default,
                                   ck_tile::tensor_layout::convolution::NDHWGC, // Channel Last Input with Group
                                   ck_tile::tensor_layout::convolution::GKZYXC, // Weight
                                   ck_tile::tuple<>,
                                   ck_tile::tensor_layout::convolution::NDHWGK>>; // Channel Last Output with Group

// Type alias for Host Arguments
using GroupedConvFwdHostArgs = ck_tile::GroupedConvFwdHostArgs;

namespace {

// Structure to hold arguments for the Composable Kernel
template <typename DataType>
struct CKArgs3DChannelLastFwd
{
    CKArgs3DChannelLastFwd(const ProblemDescription& problem)
    {
        // Extract dimensions from the problem description
        G  = ProblemInterpreter::GetGroupCountG(problem);
        N  = ProblemInterpreter::GetBatchN(problem);
        K1 = ProblemInterpreter::GetOutputChannelK(problem);
        C1 = ProblemInterpreter::GetInputChannelC(problem);
        C  = C1 / G; // Number of input Channels per group
        K  = K1 / G; // Number of output Channels per group
        Di = ProblemInterpreter::GetInputDepthDi(problem);
        Hi = ProblemInterpreter::GetInputHeightHi(problem);
        Wi = ProblemInterpreter::GetInputWidthWi(problem);
        Do = ProblemInterpreter::GetOutputDepthDo(problem);
        Ho = ProblemInterpreter::GetOutputHeightHo(problem);
        Wo = ProblemInterpreter::GetOutputWidthWo(problem);
        Z  = ProblemInterpreter::GetFilterDepthZ(problem);
        Y  = ProblemInterpreter::GetFilterHeightY(problem);
        X  = ProblemInterpreter::GetFilterWidthX(problem);

        // Set strides, dilations, and padding
        filter_strides   = {ProblemInterpreter::GetAdjustedConvolutionStrideD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionStrideH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionStrideW(problem)};
        filter_dilations = {ProblemInterpreter::GetAdjustedConvolutionDilationD(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationH(problem),
                            ProblemInterpreter::GetAdjustedConvolutionDilationW(problem)};
        lPadding         = {ProblemInterpreter::GetInputLeftPadD(problem),
                            ProblemInterpreter::GetInputLeftPadH(problem),
                            ProblemInterpreter::GetInputLeftPadW(problem)};
        rPadding         = {ProblemInterpreter::GetAdjustedInputRightPadD(problem),
                            ProblemInterpreter::GetAdjustedInputRightPadH(problem),
                            ProblemInterpreter::GetAdjustedInputRightPadW(problem)};
    }

    // Helper function to create ConvParam
    ck_tile::conv::ConvParam MakeConvParam() const
    {
        return ck_tile::conv::ConvParam(
            NumDimSpatial,  // num_dim_spatial
            G,              // group_count
            N,              // n_batch
            K,              // n_out_channels
            C,              // n_in_channels
            {Z, Y, X},      // filters_len
            {Di, Hi, Wi},   // input_len
            filter_strides, // strides
            filter_dilations, // dilations
            lPadding,       // left_pads
            rPadding        // right_pads
        );
    }

    // Function to create Host Arguments for CK tile
    GroupedConvFwdHostArgs MakeHostArgs(const miopen::conv::DataInvokeParams& data_ctx) const
    {
        auto conv_param = MakeConvParam();

        // Get tensor descriptors
        const auto& tensors = data_ctx.tensors;

        // Create Host Arguments
        GroupedConvFwdHostArgs host_args(
            conv_param,
            static_cast<const void*>(tensors.in),  // in_ptr
            static_cast<const void*>(tensors.w),   // wei_ptr
            {},                                    // ds_ptr (empty for now)
            static_cast<void*>(tensors.out),       // out_ptr
            1                                      // k_batch (default to 1)
        );

        return host_args;
    }

    // Function to create Kernel Arguments for CK tile
    // This is the main function that will be called by the invoker
    std::unique_ptr<DeviceOp3DChannelLastFwd<DataType>>
    MakeArgument(const miopen::conv::DataInvokeParams& data_ctx) const
    {
        // Create Host Arguments
        auto host_args = MakeHostArgs(data_ctx);

        // Create Kernel Arguments directly from Host Arguments
        // In CK Tile, GroupedConvFwdKernelArgs is the argument type itself
        auto argument = std::make_unique<DeviceOp3DChannelLastFwd<DataType>>(host_args);

        return argument;
    }

    // Tensor dimensions
    int G;  // Groups
    int N;  // Batch size
    int K1; // Output channels
    int C1; // Input channels
    int C;  // Input channels per group
    int K;  // Output channels per group
    int Di; // Input depth
    int Hi; // Input height
    int Wi; // Input width
    int Do; // Output depth
    int Ho; // Output height
    int Wo; // Output width
    int Z;  // Filter depth
    int Y;  // Filter height
    int X;  // Filter width

    // Convolution parameters
    std::vector<ck_tile::index_t> filter_strides;
    std::vector<ck_tile::index_t> filter_dilations;
    std::vector<ck_tile::index_t> lPadding;
    std::vector<ck_tile::index_t> rPadding;
};

} // namespace

// Performance configuration methods implementation
void PerformanceConfigConv3DChannelLastFwdWmmaops::HeuristicInit(
    const miopen::conv::ProblemDescription&)
{
    instance_id = 0;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::SetNextValue(
    const miopen::conv::ProblemDescription&)
{
    // For simplicity
    return false;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::IsValidValue() const
{
    // For simplicity
    return true;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::IsValid(
    const miopen::conv::ProblemDescription&) const
{
    // For simplicity,  assume any configuration is valid
    return true;
}

bool PerformanceConfigConv3DChannelLastFwdWmmaops::operator==(
    const PerformanceConfigConv3DChannelLastFwdWmmaops& other) const
{
    return instance_id == other.instance_id;
}


// Check if this solver is applicable for the given problem
bool ConvHipImplicitGemm3DChannelLastFwdWmmaops::IsApplicable(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    // Check if the solver is enabled by environment variable
    if(env::disabled(MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS))
    {
        return false;
    }

    // Check if HIP backend is used
    if(!ctx.use_hip_kernels)
    {
        return false;
    }

    // Check if the hardware is supported
    if(!ck_tile_utility::is_ck_tile_supported_hardware(ctx.GetStream()))
    {
        return false;
    }

    // Check if it's a 3D convolution
    if(!problem.Is3d())
    {
        return false;
    }

    // Check if it's a forward convolution
    if(!problem.IsDirectionForward())
    {
        return false;
    }

    // Check if it's channel last layout (NHWC for 2D, NDHWC for 3D)
    if(!problem.IsLayoutNHWC())
    {
        return false;
    }

    // Check data type support (specifically for FP16 with FP32 accumulation)
    // This solver is designed for FP16 input data type with FP32 accumulation
    if(!problem.IsFp16())
    {
        return false;
    }

    // Check if tensors fit into int
    if(!problem.AllTensorsDimsFitIntoInt())
    {
        return false;
    }

    // Check if it's a grouped convolution (including group count of 1)
    // This solver is designed for grouped convolutions
    if(problem.GetGroupCount() < 1)
    {
        return false;
    }

    // Check if tensors are not casted
    if(problem.IsTensorsCasted())
    {
        return false;
    }

    // Additional checks can be added here based on specific requirements

    return true;
}

// Get the default performance configuration
PerformanceConfigConv3DChannelLastFwdWmmaops
ConvHipImplicitGemm3DChannelLastFwdWmmaops::GetDefaultPerformanceConfig(
    const ExecutionContext&, const ProblemDescription& problem) const
{
    // For now, return a default configuration
    PerformanceConfigConv3DChannelLastFwdWmmaops config;
    config.HeuristicInit(problem);
    return config;
}

// Check if a performance configuration is valid
bool ConvHipImplicitGemm3DChannelLastFwdWmmaops::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConv3DChannelLastFwdWmmaops& config) const
{
    // For now, we assume any configuration is valid
    // In a real implementation, you would validate the configuration
    return config.IsValid(ctx, problem);
}

// Search for the best performance configuration
PerformanceConfigConv3DChannelLastFwdWmmaops
ConvHipImplicitGemm3DChannelLastFwdWmmaops::Search(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

// Get the solution for the given problem and configuration
ConvSolution ConvHipImplicitGemm3DChannelLastFwdWmmaops::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConv3DChannelLastFwdWmmaops& config) const
{
    ConvSolution sol;

    // Create CK arguments - use half_t for FP16 input data type
    CKArgs3DChannelLastFwd<ck_tile::half_t> ck_args(problem);

    // Set up the invoker factory
    sol.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {

            const auto& data_ctx = primitive_params.CastTo<miopen::conv::DataInvokeParams>();

            // Create the host arguments
            auto host_args = ck_args.MakeHostArgs(data_ctx);
            
            // Define types for the kernel
            using DataType = ck_tile::half_t;  // FP16 input/output
            using AccDataType = float;         // FP32 accumulation

            // Create a stream_config object for CK Tile
            ck_tile::stream_config ck_stream_config{handle.GetStream(), handle.IsProfilingEnabled()};
            
            constexpr int kBlockPerCu = 1;

            constexpr ck_tile::index_t M_Tile =16;
            constexpr ck_tile::index_t N_Tile = 16;
            constexpr ck_tile::index_t K_Tile = 16;

            constexpr ck_tile::index_t M_Warp = 1;
            constexpr ck_tile::index_t N_Warp = 1;
            constexpr ck_tile::index_t K_Warp = 1;

            constexpr ck_tile::index_t M_Warp_Tile = 16;
            constexpr ck_tile::index_t N_Warp_Tile = 16;
            constexpr ck_tile::index_t K_Warp_Tile = 16;

            constexpr ck_tile::index_t VectorSizeA = 1;
            constexpr ck_tile::index_t VectorSizeB = 1;
            constexpr ck_tile::index_t VectorSizeC = 1;

            // Define types matching the example
            using InDataType = DataType;   // FP16
            using WeiDataType = DataType;  // FP16
            using AccDataType = float;     // FP32
            using OutDataType = DataType;  // FP16
            using DsDataType = ck_tile::tuple<>;
            
            // example use template with this NDimSpatial
            constexpr ck_tile::index_t NDimSpatial = 3;
            constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;
            using InLayout = ck_tile::tensor_layout::convolution::NDHWGC;
            using WeiLayout = ck_tile::tensor_layout::convolution::GKZYXC;
            using OutLayout = ck_tile::tensor_layout::convolution::NDHWGK;
            using DsLayout = ck_tile::tuple<>;

            // Implicit GEMM Traits - exactly as in the example
            using CodegenShape =
                ck_tile::TileGemmShape<ck_tile::sequence<M_Tile, N_Tile, K_Tile>,
                                       ck_tile::sequence<M_Warp, N_Warp, K_Warp>,
                                       ck_tile::sequence<M_Warp_Tile, N_Warp_Tile, K_Warp_Tile>>;

            using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenShape>;
            using GroupedConvTraitsType =
                ck_tile::GroupedConvTraits<NDimSpatial, ConvSpec, InLayout, WeiLayout, DsLayout, OutLayout>;
            
            using CodegenPipelineProblem =
                ck_tile::GemmPipelineProblem<InDataType,
                                                WeiDataType,
                                                AccDataType,
                                                CodegenShape,
                                                typename GroupedConvTraitsType::GroupedConvImplicitGemmTraits,
                                                InDataType,
                                                true,
                                                VectorSizeA,
                                                VectorSizeB>;
            
            using CodegenPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;

            const auto Run = [&](const auto memory_operation_) {
                constexpr auto memory_operation = memory_operation_.value;

                using ConvEpilogue = ck_tile::CShuffleEpilogue<
                    ck_tile::CShuffleEpilogueProblem<InDataType,
                                                     WeiDataType,
                                                     DsDataType,
                                                     AccDataType,
                                                     OutDataType,
                                                     typename GroupedConvTraitsType::ImplicitGemmDsLayout,
                                                     ck_tile::tensor_layout::gemm::RowMajor,
                                                     ck_tile::element_wise::PassThrough,
                                                     TilePartitioner::MPerBlock,
                                                     TilePartitioner::NPerBlock,
                                                     M_Warp,
                                                     N_Warp,
                                                     M_Warp_Tile,
                                                     N_Warp_Tile,
                                                     K_Warp_Tile,
                                                     CodegenPipelineProblem::TransposeC,
                                                     memory_operation,
                                                     1,
                                                     true,
                                                     VectorSizeC>>;

                using KernelType = ck_tile::GroupedConvolutionForwardKernel<GroupedConvTraitsType,
                                                                            TilePartitioner,
                                                                            CodegenPipeline,
                                                                            ConvEpilogue>;
                auto kargs = KernelType::MakeKernelArgs(host_args);

                const dim3 grids = KernelType::GridSize(kargs);
                const dim3 blocks = KernelType::BlockSize();

                if(!KernelType::IsSupportedArgument(kargs))
                {
                    throw std::runtime_error("Wrong! Arguments not supported! Skipping conv!\n");
                }

                float ave_time = ck_tile::launch_kernel(
                    ck_stream_config, ck_tile::make_kernel<kBlockPerCu>(KernelType{}, grids, blocks, 0, kargs));

                return ave_time;
            };

            float elapsed_time = Run(ck_tile::integral_constant<ck_tile::memory_operation_enum,
                                                               ck_tile::memory_operation_enum::set>{});

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(elapsed_time);
            }
        };
    };

    return sol;
}

// Get the workspace size required for this solver
size_t ConvHipImplicitGemm3DChannelLastFwdWmmaops::GetWorkspaceSize(
    const ExecutionContext& ctx, const ProblemDescription& problem) const
{
    // For now, return 0 as we're not using workspace 
    return 0;
}

} // namespace conv
} // namespace solver
} // namespace miopen

#endif // MIOPEN_BACKEND_HIP && MIOPEN_USE_CKTILE_COMPOSABLEKERNEL