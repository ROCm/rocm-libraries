// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/invokers/gen_x_w_y_pad.hpp>
#include <miopen/env.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/mlo_internal.hpp>
#include <miopen/visit_float.hpp>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DIRECT_HIP_FWD11X11)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool ConvHipDirectFwd11x11::IsApplicable(const ExecutionContext& ctx,
                                         const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DIRECT_HIP_FWD11X11))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    const auto& name = ctx.GetStream().GetDeviceName();
    // gfx8 is intentionally excluded: it is end-of-life and not supported for new HIP solvers.
    if(!(StartsWith(name, "gfx90") || StartsWith(name, "gfx94") || StartsWith(name, "gfx95") ||
         StartsWith(name, "gfx103") || StartsWith(name, "gfx110") || StartsWith(name, "gfx115") ||
         StartsWith(name, "gfx120")))
        return false;
    if(!problem.Is2d())
        return false;
    if(problem.HasNonPackedTensors())
        return false;
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;
    if(problem.IsAsymmetricPadH() || problem.IsAsymmetricPadW())
        return false;
    if(!(problem.IsFp32() || problem.IsFp16() || problem.IsBfp16()))
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.IsLayoutDefault())
        return false;

    return problem.IsDirectionForward() && problem.GetGroupCount() == 1 &&
           problem.GetDilationH() == 1 && problem.GetDilationW() == 1 &&
           problem.GetWeightsHeight() == 11 && problem.GetWeightsWidth() == 11 &&
           problem.GetKernelStrideH() == 4 && problem.GetKernelStrideW() == 4;
}

ConvSolution ConvHipDirectFwd11x11::GetSolution(const ExecutionContext& /*ctx*/,
                                                const ProblemDescription& problem) const
{
    ConvSolution result;

    const auto hw_wave_sz = 64;
    const int LG2_WAVE_SZ = mloLg2(hw_wave_sz);

    const int wei_cstride = problem.GetWeightsWidth() * problem.GetWeightsHeight();
    const int wei_bstride = static_cast<int>(problem.GetInChannels()) * wei_cstride;

    // Number of batch iterations per workgroup
    result.n_stacks                 = std::min(problem.GetBatchSize(), static_cast<std::size_t>(1));
    const std::size_t N_BATCH_LOOPS = 1;
    const int n_batch_blks =
        static_cast<int>((problem.GetBatchSize() + N_BATCH_LOOPS * result.n_stacks - 1) /
                         (N_BATCH_LOOPS * result.n_stacks));

    const int N_FILTER_SPLITS0 = static_cast<int>(
        (problem.GetWeightsWidth() + problem.GetKernelStrideW() - 1) / problem.GetKernelStrideW());
    const int N_FILTER_SPLITS1 = static_cast<int>(
        (problem.GetWeightsHeight() + problem.GetKernelStrideH() - 1) / problem.GetKernelStrideH());

    result.out_pix_tile0 = N_FILTER_SPLITS0;
    result.out_pix_tile1 = 1;

    result.in_tile0 = 1;
    result.in_tile1 = 1;

    const int n_waves      = 4;
    const int GRP_SZ       = hw_wave_sz * n_waves;
    const int lg2_n_waves  = mloLg2(n_waves);
    const int N_WAVES_MASK = (1 << lg2_n_waves) - 1;

    const int PROCESING_WIDTH =
        static_cast<int>((problem.GetOutWidth() + result.out_pix_tile0 - 1) / result.out_pix_tile0);

    const int OUT_EXTENT1 =
        std::min(static_cast<int>(problem.GetOutHeight()), GRP_SZ / PROCESING_WIDTH);

    const int read_unit = 10;
    const std::string READ_TYPE =
        (read_unit == 1) ? "_FLOAT" : "_FLOAT" + std::to_string(read_unit);

    const int n_out_stacks = 1;
    int n_in_stacks        = 1;
    n_in_stacks            = std::min(static_cast<int>(problem.GetInChannels()), n_in_stacks);

    result.n_out_pix_tiles = static_cast<std::size_t>(std::min(
        static_cast<std::size_t>(6), (problem.GetOutChannels() + n_out_stacks - 1) / n_out_stacks));
    result.n_in_data_tiles = 1;

    const int total_out_maps = static_cast<int>(result.n_out_pix_tiles) * n_out_stacks;

    result.grp_tile0    = GRP_SZ;
    result.grp_tile1    = 1;
    const int grp_tile2 = 1;

    // Determine whether a second pass is needed for remainder rows
    const int n_extents =
        static_cast<int>((problem.GetOutHeight() + OUT_EXTENT1 - 1) / OUT_EXTENT1);
    const int n_output_map_blocks =
        static_cast<int>((problem.GetOutChannels() + total_out_maps - 1) / total_out_maps);

    int last_out_extent1 = static_cast<int>(problem.GetOutHeight()) -
                           static_cast<int>(std::max(static_cast<std::size_t>(1),
                                                     problem.GetOutHeight() / OUT_EXTENT1)) *
                               OUT_EXTENT1;
    last_out_extent1 = (last_out_extent1 < 0) ? 0 : last_out_extent1;

    int n_extents_pass1 = n_extents;
    int n_batches_pass2 = 1;
    bool second_pass    = false;
    if(0 < last_out_extent1 && last_out_extent1 <= OUT_EXTENT1 / 2)
    {
        n_extents_pass1 = static_cast<int>(
            std::max(static_cast<std::size_t>(1), problem.GetOutHeight() / OUT_EXTENT1));
        n_batches_pass2 = std::max(1, GRP_SZ / (PROCESING_WIDTH * last_out_extent1));
        second_pass     = true;
    }

    // Build compile-time parameters for the HIP kernel
    const auto build_params = KernelBuildParameters{
        {"MLO_GRP_SZ", GRP_SZ},
        {"MLO_GRP_SZ0", result.grp_tile0},
        {"MLO_GRP_SZ1", result.grp_tile1},
        {"MLO_GRP_SZ2", grp_tile2},
        {"MLO_FILTER_SIZE0", problem.GetWeightsWidth()},
        {"MLO_FILTER_SIZE1", problem.GetWeightsHeight()},
        {"MLO_FILTER_PAD0", problem.GetPadW()},
        {"MLO_FILTER_PAD1", problem.GetPadH()},
        {"MLO_FILTER_STRIDE0", problem.GetKernelStrideW()},
        {"MLO_FILTER_STRIDE1", problem.GetKernelStrideH()},
        {"STRIDE_W", problem.GetKernelStrideW()},
        {"STRIDE_H", problem.GetKernelStrideH()},
        {"MLO_N_OUTPUTS", problem.GetOutChannels()},
        {"MLO_N_INPUTS", problem.GetInChannels()},
        {"MLO_BATCH_SZ", problem.GetBatchSize()},
        {"MLO_N_BATCH_LOOPS", N_BATCH_LOOPS},
        {"MLO_OUT_BATCH_STRIDE", problem.GetOutBatchStride()},
        {"MLO_OUT_CHANNEL_STRIDE", problem.GetOutChannelStride()},
        {"MLO_OUT_STRIDE", problem.GetOutStrideH()},
        {"MLO_IN_BATCH_STRIDE", problem.GetInBatchStride()},
        {"MLO_IN_CHANNEL_STRIDE", problem.GetInChannelStride()},
        {"MLO_IN_STRIDE", problem.GetInStrideH()},
        {"MLO_WEI_BATCH_STRIDE", wei_bstride},
        {"MLO_WEI_CHANNEL_STRIDE", wei_cstride},
        {"MLO_IN_WIDTH", problem.GetInWidth()},
        {"MLO_IN_HEIGHT", problem.GetInHeight()},
        {"MLO_OUT_WIDTH", problem.GetOutWidth()},
        {"MLO_OUT_HEIGHT", problem.GetOutHeight()},
        {"MLO_IN_TILE1", result.in_tile1},
        {"MLO_IN_TILE0", result.in_tile0},
        {"MLO_N_LCL_BATCHS", result.n_stacks},
        {"MLO_N_LCL_OUT_MAPS", result.n_out_pix_tiles},
        {"MLO_N_LCL_IN_MAPS", result.n_in_data_tiles},
        {"MLO_IN_PIX_TILE1", 1},
        {"MLO_IN_PIX_TILE0", 1},
        {"MLO_OUT_PIX_TILE1", result.out_pix_tile1},
        {"MLO_OUT_PIX_TILE0", result.out_pix_tile0},
        {"MLO_OUT_STACKS", n_out_stacks},
        {"MLO_IN_STACKS", n_in_stacks},
        {"MLO_N_WAVES", n_waves},
        {"MLO_N_FILTER_SPLITS0", N_FILTER_SPLITS0},
        {"MLO_N_FILTER_SPLITS1", N_FILTER_SPLITS1},
        {"MLO_PROCESSING_WIDTH", PROCESING_WIDTH},
        {"MLO_OUT_EXTENT1", OUT_EXTENT1},
        {"MLO_LAST_OUT_EXTENT1", last_out_extent1},
        {"MLO_N_LCL_BATCHS_PASS2", n_batches_pass2},
        {"MLO_TILE_REPLICATE0", 2},
        {"MLO_TILE_REPLICATE1", 1},
        {"MLO_READ_TYPE", READ_TYPE},
        {"MLO_READ_UNIT", read_unit},
        {"MLO_HW_WAVE_SZ", hw_wave_sz},
        {"MLO_LG2_WAVE_SZ", LG2_WAVE_SZ},
        {"MLO_N_WAVES_MASK", N_WAVES_MASK},
        {"MLO_CONV_BIAS", problem.GetBias()},
        {"MIOPEN_USE_FP32", static_cast<int>(problem.IsFp32())},
        {"MIOPEN_USE_FP16", static_cast<int>(problem.IsFp16())},
        {"MIOPEN_USE_BFP16", static_cast<int>(problem.IsBfp16())},
    };

    const std::string comp_options = build_params.GenerateFor(kbp::HIP{});

    // 1st pass kernel
    {
        KernelInfo k;
        k.kernel_file  = "MIOpenConvFwd_LxL_11.cpp";
        k.kernel_name  = "MIOpenCvFwd11x11";
        k.comp_options = comp_options;

        k.l_wk = {static_cast<size_t>(result.grp_tile0),
                  static_cast<size_t>(result.grp_tile1),
                  static_cast<size_t>(grp_tile2)};

        k.g_wk = {static_cast<size_t>(GRP_SZ) * static_cast<size_t>(n_extents_pass1),
                  static_cast<size_t>(n_output_map_blocks),
                  static_cast<size_t>(n_batch_blks)};

        result.construction_params.push_back(k);
    }

    // 2nd pass kernel (optional)
    if(second_pass)
    {
        KernelInfo k;
        k.kernel_file  = "MIOpenConvFwd_LxL_11.cpp";
        k.kernel_name  = "MIOpenCvFwd11x11_2";
        k.comp_options = comp_options;

        k.l_wk = {static_cast<size_t>(result.grp_tile0),
                  static_cast<size_t>(result.grp_tile1),
                  static_cast<size_t>(grp_tile2)};

        const int n_batch_blks_pass2 =
            static_cast<int>((problem.GetBatchSize() + n_batches_pass2 - 1) / n_batches_pass2);
        k.g_wk = {static_cast<size_t>(GRP_SZ),
                  static_cast<size_t>(n_output_map_blocks),
                  static_cast<size_t>(n_batch_blks_pass2)};

        result.construction_params.push_back(k);

        result.invoker_factory = [](const std::vector<Kernel>& kernels) {
            if(kernels.size() != 2)
                MIOPEN_THROW("Two kernels were expected by solver");

            return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
                const auto& invoke_params =
                    primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
                const auto& tensors = invoke_params.tensors;

                const auto first_pass_kernel  = handle.Run(kernels[0]);
                const auto second_pass_kernel = handle.Run(kernels[1]);

                float padding_val = 0;
                float elapsed     = 0;

                visit_float(tensors.inDesc.GetType(), [&](auto as_float) {
                    first_pass_kernel(tensors.in, tensors.w, tensors.out, as_float(padding_val));
                });

                if(handle.IsProfilingEnabled())
                    elapsed += handle.GetKernelTime();

                visit_float(tensors.inDesc.GetType(), [&](auto as_float) {
                    second_pass_kernel(tensors.in, tensors.w, tensors.out, as_float(padding_val));
                });

                if(handle.IsProfilingEnabled())
                {
                    elapsed += handle.GetKernelTime();
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(elapsed);
                }
            };
        };
    }
    else
    {
        result.invoker_factory = &miopen::conv::MakeGenericXWYPadInvoker;
    }

    return result;
}

} // namespace conv
} // namespace solver
} // namespace miopen
