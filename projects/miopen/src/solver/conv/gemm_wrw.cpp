// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solvers.hpp>

#include <miopen/conv/wrw_invoke_params.hpp>
#include <miopen/errors.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/solver/gemm_common.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/util.hpp>

#include <ranges>
#include <set>

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool GemmWrwBase::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    const auto& dyDesc             = problem.GetIn();
    const auto& dwDesc             = problem.GetWeights();
    const auto& xDesc              = problem.GetOut();
    const auto rblas_fp8_supported = IsFP8Supported(ctx.GetStream().GetDeviceName());
    if(problem.IsTensorsCasted())
    {
        if(!rblas_fp8_supported)
        {
            MIOPEN_LOG_I2("GEMM not supported with casted tensors on this GPU architecture");
            return false;
        }
        if(xDesc.GetCastType() && dyDesc.GetCastType())
        {
            const auto a_cast_type = xDesc.GetCastType();
            const auto b_cast_type = dyDesc.GetCastType();
            if(a_cast_type != miopenFloat8_fnuz && b_cast_type != miopenBFloat8_fnuz)
            {
                MIOPEN_LOG_W("Casting is only supported for the miopenFloat8_fnuz and "
                             "miopenBFloat8_fnuz data types");
                return false;
            }
            if(a_cast_type != miopenFloat8_fnuz && b_cast_type != miopenBFloat8_fnuz)
            {
                MIOPEN_LOG_W("Casting is only supported for the miopenFloat8_fnuz and "
                             "miopenBFloat8_fnuz data types");
                return false;
            }
        }
        else
        {
            MIOPEN_LOG_I("Both the input and output tensors need to be casted");
            return false;
        }
    }
    if(problem.IsFp8() && !rblas_fp8_supported)
    {
        MIOPEN_LOG_I2("GEMM not applicable for F8 on this GPU architecture");
        return false;
    }

    if(problem.HasNonPackedTensors())
        return false;

    return problem.IsDirectionBackwardWrW() && problem.IsLayoutDefault() &&
           !(gemm::IsAnyBufferBf16(xDesc, dyDesc, dwDesc) && !gemm::IsBf16Supported) &&
           !(gemm::IsAnyBufferFp16(xDesc, dyDesc, dwDesc) && !gemm::IsFp16Supported);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

float GemmWrwBase::GetWti(const ExecutionContext&, const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dwDesc = problem.GetWeights();
    const auto& xDesc  = problem.GetOut();
    const auto& conv   = problem.GetConv();

    int n_gemm_strided_batched           = 1; // not strided-batched by default
    int n_gemm_strided_batched_sequental = 1; // not strided-batched-sequental by default
    int n_gemm_runs                      = 1;
    int n_Im2ColGPU                      = 0;

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());
    const auto wei_spatial =
        dwDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());

    // if not 1x1
    if((miopen::any_of(wei_spatial, [](auto v) { return v != 1; }) ||
        miopen::any_of(conv.GetConvPads(), [](auto v) { return v != 0; }) ||
        miopen::any_of(conv.GetConvStrides(), [](auto v) { return v != 1; })))
    {
        n_Im2ColGPU            = in_n;
        n_gemm_strided_batched = conv.group_count;
        n_gemm_runs            = in_n;
    }
    // 1x1 does not require im2col or workspace
    else if(miopen::any_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::any_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::any_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
    {
        n_gemm_strided_batched_sequental = conv.group_count;
        n_gemm_runs                      = in_n;
    }

    auto wti = 0.7; // Memory overhead for WrW is bigger then for Fwd/Bwd.
    wti *= gemm::SlowdownFactor(n_gemm_runs, 0.9, 0.9);
    wti *= gemm::SlowdownFactor(n_gemm_strided_batched, 1.0, 0.95);
    wti *= gemm::SlowdownFactor(n_gemm_strided_batched_sequental, 1.0, 0.9);
    wti *= gemm::SlowdownFactor(n_Im2ColGPU, 0.4, 0.8);
    return wti;
#else
    std::ignore = problem;
    return 0;
#endif
}

bool GemmWrw1x1_stride1::IsSlow(const ExecutionContext& context,
                                const ProblemDescription& problem) const
{
    const std::string& arch        = context.GetStream().GetDeviceName();
    const std::set<std::string> mi = {"gfx942", "gfx955"};
    const bool is_mi               = mi.find(arch) != mi.end();
    const bool is_gfx11            = StartsWith(arch, "gfx11");
    const bool is_gfx12            = StartsWith(arch, "gfx12");

    auto b                  = problem.GetBatchSize();
    auto s                  = problem.GetOutHeight() * problem.GetOutWidth();
    auto c                  = problem.GetInChannels() + problem.GetOutChannels();
    auto g                  = problem.GetGroupCount();
    auto spatial_per_batch  = s / b;
    auto channels_per_group = c / g;

    if(is_gfx11 || is_gfx12)
    {
        // GemmWrw1x1_stride1 - Batch-based filtering
        // Analysis: 8.4% terrible cases - moderate filtering benefit
        //
        // INVERTED PATTERN discovered: Terrible cases have HIGH batch but LOW channels
        // - Batch separation: 16-32x (terrible > decent)
        // - CPG separation: 0.12-0.60x (terrible < decent)
        // - SWPG separation: 0.12-0.41x (terrible < decent)
        //
        // Physical interpretation: High batch + low channels = poor wave occupancy
        //
        // Threshold: batch > 16 AND cpg < 1400
        // Performance: FPR=3-15%, TPR=73-87%, Score=1.65-1.79
        if(b > 16 && channels_per_group < 1400)
            return true;
    }
    else if(is_mi)
    {
        // SPB-ONLY: Batch fragmentation detection
        // SPB < 48.0: Each batch item has < 48 pixels of spatial work
        if(spatial_per_batch < 48.0)
            return true;
    }

    return false;
}

bool GemmWrw1x1_stride1::IsApplicable(const ExecutionContext& context,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmWrwBase::IsApplicable(context, problem))
        return false;

    const auto& dwDesc = problem.GetWeights();
    const auto& conv   = problem.GetConv();

    const auto wei_spatial =
        dwDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());

    return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; });
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmWrw1x1_stride1::GetSolution(const ExecutionContext&,
                                             const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc     = problem.GetIn();
    const auto& dwDesc     = problem.GetWeights();
    const auto& xDesc      = problem.GetOut();
    const auto& conv       = problem.GetConv();
    const auto group_count = conv.group_count;

    if(group_count > 1)
    {
        MIOPEN_LOG_FUNCTION("groupconv, 1x1");
    }
    else
    {
        MIOPEN_LOG_FUNCTION("convolution, 1x1");
    }

    // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
    const auto tmp_gemm_desc = [&]() {
        auto tmp          = group_count > 1
                                ? CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count)
                                : CreateGemmStridedBatchedDescriptorConv1x1BwdWeight(dyDesc, xDesc, dwDesc);
        tmp.deterministic = problem.GetConv().attribute.deterministic;
        if(problem.IsTensorsCasted())
        {
            // IsApplicable ensures that both are casted
            if(dyDesc.GetCastType())
                tmp.a_cast_type = *dyDesc.GetCastType();
            if(xDesc.GetCastType())
                tmp.b_cast_type = *xDesc.GetCastType();
        }
        tmp.conv_attributes = problem.GetConv().attribute;
        return tmp;
    }();

    const auto in_spatial =
        xDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());
    const auto out_spatial =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());

    const auto out_spatial_size = std::accumulate(
        out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto in_spatial_size = std::accumulate(
        in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(xDesc.GetLengths());

    const auto wei_k = dwDesc.GetLengths()[0];

    auto solution = ConvSolution{miopenStatusSuccess};

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params = primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& dy          = conv_params.tensors.dy;
            const auto& dw          = conv_params.tensors.dw;
            const auto& dwDesc_     = conv_params.tensors.dwDesc;
            const auto& x           = conv_params.tensors.x;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("conv, 1x1");
            }

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            // Zeroing out the output buffer
            float zero = 0.0f;
            SetTensor(handle, dwDesc_, dw, &zero);

            if(group_count > 1)
            {
                float time = 0.0f;

                for(std::size_t i = 0; i < in_n; i++)
                {
                    const auto out_offset = i * wei_k * out_spatial_size;
                    const auto in_offset  = i * in_c * in_spatial_size;

                    const auto status = CallGemmStridedBatched(handle,
                                                               gemm_desc,
                                                               dy,
                                                               out_offset,
                                                               x,
                                                               in_offset,
                                                               dw,
                                                               0,
                                                               GemmBackend_t::rocblas);

                    if(status != miopenStatusSuccess)
                        MIOPEN_THROW("GemmWrw1x1_stride1 execution failure.");

                    if(handle.IsProfilingEnabled())
                        time += handle.GetKernelTime();
                }

                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time);
                }
            }
            else
            {
                // dw = sum_over_batch(dy[i] * transpose(x[i])), i is batch id
                const auto status = CallGemmStridedBatchedSequential(
                    handle, gemm_desc, dy, 0, x, 0, dw, 0, GemmBackend_t::rocblas);

                if(status != miopenStatusSuccess)
                    MIOPEN_THROW("GemmWrw1x1_stride1 execution failure.");
            }
        };
    };

    return solution;
#else
    std::ignore = problem;
    return {};
#endif
}

size_t GemmWrwUniversal::GetWorkspaceSize(const ExecutionContext& context,
                                          const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    auto& handle       = context.GetStream();
    const auto& dyDesc = problem.GetIn();
    const auto& dwDesc = problem.GetWeights();
    const auto& conv   = problem.GetConv();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto out_spatial =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto wei_spatial =
        dwDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto wei_c = dwDesc.GetLengths()[1];

    auto ws_size = GetTypeSize(dyDesc.GetType()) * wei_c *
                   std::accumulate(out_spatial.begin(),
                                   out_spatial.end(),
                                   std::size_t(1),
                                   std::multiplies<std::size_t>()) *
                   std::accumulate(wei_spatial.begin(),
                                   wei_spatial.end(),
                                   std::size_t(1),
                                   std::multiplies<std::size_t>()) *
                   conv.group_count;

    // For bf16: extra workspace for fp32 accumulation buffer (same shape as dw)
    const auto xDesc           = problem.GetOut();
    const auto in_n            = xDesc.GetLengths()[0];
    const auto need_fp32_accum = (dyDesc.GetType() == miopenBFloat16) && (in_n > 1);
    if(need_fp32_accum)
    {
        const auto fp32_accum_size = GetTypeSize(miopenFloat) * dwDesc.GetElementSize();
        // Use padded layout: im2col buffer at offset 0, fp32 accum buffer at offset ws_size
        // (aligned to 256 bytes)
        ws_size = ((ws_size + 255) & ~std::size_t{255}) + fp32_accum_size;
    }

    if(ws_size > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_LOG_I2("GemmWrwUniversal: " << ws_size << " > " << handle.GetMaxMemoryAllocSize());
        return 0;
    }
    return ws_size;
#else
    std::ignore = context;
    std::ignore = problem;
    return 0;
#endif
}

bool GemmWrwUniversal::IsSlow(const ExecutionContext& context,
                              const ProblemDescription& problem) const
{
    const std::string& arch        = context.GetStream().GetDeviceName();
    const std::set<std::string> mi = {"gfx942", "gfx955"};
    const bool is_mi               = mi.find(arch) != mi.end();
    const bool is_gfx11            = StartsWith(arch, "gfx11");
    const bool is_gfx12            = StartsWith(arch, "gfx12");

    auto b                 = problem.GetBatchSize();
    auto s                 = problem.GetOutHeight() * problem.GetOutWidth();
    auto spatial_per_batch = s / b;

    if(is_gfx11 || is_gfx12)
    {
        // GemmWrwUniversal - SPB-only filtering
        // Analysis: 18.4% terrible cases - significant filtering benefit
        //
        // Terrible cases have high batch (16x) but very low SPB (0.00x)
        // This indicates extreme batch fragmentation
        //
        // SPB < 100: Low spatial-per-batch = batch fragmentation
        // Performance: FPR=19-27%, TPR=72-92%, Score=1.49-1.66
        if(spatial_per_batch < 100)
            return true;
    }
    else if(is_mi)
    {
        // SPB-ONLY: Batch fragmentation detection
        // SPB < 48.0: Each batch item has < 48 pixels of spatial work
        if(spatial_per_batch < 48.0)
            return true;
    }

    return false;
}

bool GemmWrwUniversal::IsApplicable(const ExecutionContext& context,
                                    const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmWrwBase::IsApplicable(context, problem))
        return false;

    return !GemmWrw1x1_stride1{}.IsApplicable(context, problem) &&
           GetWorkspaceSize(context, problem) != 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmWrwUniversal::GetSolution(const ExecutionContext& context,
                                           const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc     = problem.GetIn();
    const auto& dwDesc     = problem.GetWeights();
    const auto& xDesc      = problem.GetOut();
    const auto& conv       = problem.GetConv();
    const auto group_count = conv.group_count;

    // dw = dy * transpose(Im2Col(x))
    const auto tmp_gemm_desc = [&]() {
        auto tmp          = group_count > 1
                                ? CreateGemmDescriptorGroupConvBwdWeight(dyDesc, xDesc, dwDesc, group_count)
                                : CreateGemmDescriptorConvBwdWeight(dyDesc, xDesc, dwDesc);
        tmp.deterministic = problem.GetConv().attribute.deterministic;
        if(problem.IsTensorsCasted())
        {
            // IsApplicable ensures that both are casted
            if(dyDesc.GetCastType())
                tmp.a_cast_type = *dyDesc.GetCastType();
            if(xDesc.GetCastType())
                tmp.b_cast_type = *xDesc.GetCastType();
        }
        tmp.conv_attributes = problem.GetConv().attribute;
        return tmp;
    }();

    const auto spatial_dims   = conv.GetSpatialDimension();
    const auto conv_pads      = conv.GetConvPads();
    const auto conv_strides   = conv.GetConvStrides();
    const auto conv_dilations = conv.GetConvDilations();
    const auto workspace_req  = GetWorkspaceSize(context, problem);
    const auto in_n           = xDesc.GetLengths()[0];
    const auto in_c           = xDesc.GetLengths()[1];
    const auto wei_k          = dwDesc.GetLengths()[0];

    // bf16: accumulate in fp32 workspace when batch_size > 1
    const auto data_type      = dyDesc.GetType();
    const auto use_fp32_accum = (data_type == miopenBFloat16) && (in_n > 1);
    const auto lowp_quant     = conv.lowp_quant;
    const auto dw_lengths     = dwDesc.GetLengths();
    const auto dw_strides     = dwDesc.GetStrides();
    // im2col workspace size (before padding/alignment)
    const auto im2col_ws_size = [&]() {
        const auto wei_c = dwDesc.GetLengths()[1];
        const auto out_sp =
            dyDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dims);
        const auto wei_sp =
            dwDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dims);
        return GetTypeSize(data_type) * wei_c *
               std::accumulate(
                   out_sp.begin(), out_sp.end(), std::size_t(1), std::multiplies<std::size_t>()) *
               std::accumulate(
                   wei_sp.begin(), wei_sp.end(), std::size_t(1), std::multiplies<std::size_t>()) *
               conv.group_count;
    }();
    // Offset to fp32 accumulation buffer within workspace (256-byte aligned)
    const auto fp32_accum_offset =
        use_fp32_accum ? ((im2col_ws_size + 255) & ~std::size_t{255}) : std::size_t{0};

    const auto in_spatial_ =
        xDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());
    const auto wei_spatial_ =
        dwDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());
    const auto out_spatial_ =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());

    const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
    const auto wei_spatial = std::vector<std::size_t>(wei_spatial_.begin(), wei_spatial_.end());
    const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

    const auto out_spatial_size = std::accumulate(
        out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto in_spatial_size = std::accumulate(
        in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<miopen::conv::WrWInvokeParams>();
            const auto& dy             = conv_params.tensors.dy;
            const auto& dyDesc_        = conv_params.tensors.dyDesc;
            const auto& dwDesc_        = conv_params.tensors.dwDesc;
            const auto& dw             = conv_params.tensors.dw;
            const auto& x              = conv_params.tensors.x;
            const auto& workspace      = conv_params.workSpace;
            const auto& workspace_size = conv_params.workSpaceSize;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");
            }

            if(workspace_req > 0 && (workspace == nullptr || workspace_size < workspace_req))
            {
                MIOPEN_THROW("Not enough workspace for GemmWrwUniversal. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");
            }

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            // Zeroing out the output buffer
            float zero = 0.0f;
            float time = 0;

            // For bf16 with batch > 1: accumulate into fp32 workspace, then cast
            Data_t accum_buf = dw;
            if(use_fp32_accum)
            {
                accum_buf = static_cast<Data_t>(static_cast<char*>(workspace) + fp32_accum_offset);
                TensorDescriptor fp32Desc(miopenFloat, dw_lengths, dw_strides);
                SetTensor(handle, fp32Desc, accum_buf, &zero);
            }
            else
            {
                SetTensor(handle, dwDesc_, dw, &zero);
            }

            if(handle.IsProfilingEnabled())
                time += handle.GetKernelTime();

            for(std::size_t i = 0; i < in_n; i++)
            {
                const auto out_offset = i * wei_k * out_spatial_size;
                const auto in_offset  = i * in_c * in_spatial_size;

                time += Im2ColGPU(handle,
                                  spatial_dims,
                                  x,
                                  in_offset,
                                  in_c,
                                  in_spatial,
                                  wei_spatial,
                                  out_spatial,
                                  conv_pads,
                                  conv_strides,
                                  conv_dilations,
                                  workspace,
                                  dyDesc_.GetType());

                miopenStatus_t status;

                if(group_count > 1)
                {
                    if(use_fp32_accum)
                    {
                        status = CallGemmStridedBatched(handle,
                                                        gemm_desc,
                                                        dy,
                                                        out_offset,
                                                        workspace,
                                                        0,
                                                        accum_buf,
                                                        0,
                                                        miopenFloat);
                    }
                    else
                    {
                        status = CallGemmStridedBatched(handle,
                                                        gemm_desc,
                                                        dy,
                                                        out_offset,
                                                        workspace,
                                                        0,
                                                        dw,
                                                        0,
                                                        GemmBackend_t::rocblas);
                    }
                }
                else
                {
                    if(use_fp32_accum)
                    {
                        // dw = dy * transpose(Im2Col(x))  -- accumulated in fp32
                        status = CallGemm(handle,
                                          gemm_desc,
                                          dy,
                                          out_offset,
                                          workspace,
                                          0,
                                          accum_buf,
                                          0,
                                          miopenFloat);
                    }
                    else
                    {
                        // dw = dy * transpose(Im2Col(x))
                        status = CallGemm(handle,
                                          gemm_desc,
                                          dy,
                                          out_offset,
                                          workspace,
                                          0,
                                          dw,
                                          0,
                                          GemmBackend_t::rocblas);
                    }
                }

                if(status != miopenStatusSuccess)
                    MIOPEN_THROW("GemmWrw1x1_stride1 execution failure.");

                // Update times for both the kernels
                if(handle.IsProfilingEnabled())
                    time += handle.GetKernelTime();
            }

            // Cast fp32 accumulation buffer back to bf16
            if(use_fp32_accum)
            {
                TensorDescriptor fp32Desc(miopenFloat, dw_lengths, dw_strides);
                CastTensor(handle, &lowp_quant, false, fp32Desc, accum_buf, dwDesc_, dw, 0, 0);
                if(handle.IsProfilingEnabled())
                    time += handle.GetKernelTime();
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(time);
            }
        };
    };

    return solution;
#else
    std::ignore = context;
    std::ignore = problem;
    return {};
#endif
}

} // namespace conv
} // namespace solver
} // namespace miopen
