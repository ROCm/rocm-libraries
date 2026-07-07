// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/solvers.hpp>

#include <miopen/algorithm.hpp>
#include <miopen/gemm_v2.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_ops.hpp>
#include <miopen/util.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/solver/gemm_common.hpp>

#include <ranges>
#include <set>

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

bool GemmBwdBase::IsApplicable(const ExecutionContext& ctx, const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!problem.AllTensorsDimsFitIntoInt())
        return false;

    const auto& dyDesc             = problem.GetIn();
    const auto& wDesc              = problem.GetWeights();
    const auto& dxDesc             = problem.GetOut();
    const auto rblas_fp8_supported = IsFP8Supported(ctx.GetStream().GetDeviceName());
    if(problem.IsTensorsCasted())
    {
        if(!rblas_fp8_supported)
        {
            MIOPEN_LOG_I2("GEMM not supported with casted tensors on this GPU architecture");
            return false;
        }
        if(dyDesc.GetCastType() && wDesc.GetCastType())
        {
            const auto a_cast_type = dyDesc.GetCastType();
            const auto b_cast_type = wDesc.GetCastType();
            if(a_cast_type != miopenFloat8_fnuz && a_cast_type != miopenBFloat8_fnuz)
            {
                MIOPEN_LOG_W("Casting is only supported for the miopenFloat8_fnuz and "
                             "miopenBFloat8_fnuz data types");
                return false;
            }
            if(b_cast_type != miopenFloat8_fnuz && b_cast_type != miopenBFloat8_fnuz)
            {
                MIOPEN_LOG_W("Casting is only supported for the miopenFloat8_fnuz and "
                             "miopenBFloat8_fnuz data types");
                return false;
            }
        }
        else
        {
            MIOPEN_LOG_I("Both the output and weights tensors need to be casted");
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

    return problem.IsDirectionBackwardData() && problem.IsLayoutDefault() &&
           !(gemm::IsAnyBufferBf16(dxDesc, dyDesc, wDesc) && !gemm::IsBf16Supported) &&
           !(gemm::IsAnyBufferFp16(dxDesc, dyDesc, wDesc) && !gemm::IsFp16Supported);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

float GemmBwdBase::GetWti(const ExecutionContext&, const ProblemDescription& problem) const
{
    const auto& conv   = problem.GetConv();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();

    const auto prefer_point_output_shape =
        miopen::conv::IsBwdDataPointOutput3dStrideEqFilter(problem);

    int n_SetTensor            = 0;
    int n_transpose_NCHW2CNHW  = 0;
    int n_transpose_CNHW2NCHW  = 0;
    int n_gemm_strided_batched = 1; // not strided-batched by default
    int n_gemm_runs            = 1;
    int n_Col2ImGPU            = 0;

    std::size_t in_n, in_c;
    std::tie(in_n, in_c)    = tie_pick<0, 1>()(dxDesc.GetLengths());
    std::size_t spatial_dim = conv.GetSpatialDimension();
    auto wei_spatial = wDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);

    // 1x1 does not require col2im
    if(conv.GetSpatialDimension() == 2 &&
       miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
       miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
       miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }))
    {
        n_SetTensor            = 1;
        n_transpose_NCHW2CNHW  = 1;
        n_gemm_strided_batched = conv.group_count;
        n_transpose_CNHW2NCHW  = 1;
    }
    // 1x1_stride=1 convolutions use GEMM and zero workspace
    else if(miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
            miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
            miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; }))
    {
        n_gemm_strided_batched = in_n;
    }
    // if not 1x1
    else
    {
        if(prefer_point_output_shape)
        {
            // Keep the WTI model aligned with the 3D execution path:
            // one strided-batched GEMM + one batched Col2Im launch.
            n_gemm_strided_batched = in_n;
            n_gemm_runs            = 1;
            n_Col2ImGPU            = 1;
        }
        else
        {
            // Preserve the previous non-3D behavior in WTI estimation.
            n_gemm_strided_batched = conv.group_count;
            n_gemm_runs            = in_n;
            n_Col2ImGPU            = in_n;
        }
    }

    auto wti = 1.0;
    wti *= gemm::SlowdownFactor(n_SetTensor, 0.95, 0.99);
    wti *= gemm::SlowdownFactor(n_transpose_NCHW2CNHW, 0.7, 0.9);
    wti *= gemm::SlowdownFactor(n_transpose_CNHW2NCHW, 0.7, 0.9);
    wti *= gemm::SlowdownFactor(n_gemm_runs, 0.9, 0.9);
    wti *= gemm::SlowdownFactor(n_gemm_strided_batched, 1.0, 0.95);
    wti *= gemm::SlowdownFactor(n_Col2ImGPU, 0.4, 0.8);
    if(prefer_point_output_shape && wti < 10.0)
        wti = 10.0;
    return wti;
}

size_t GemmBwd1x1_stride2::GetWorkspaceSize(const ExecutionContext& context,
                                            const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    auto& handle       = context.GetStream();
    const auto& conv   = problem.GetConv();
    const auto& dyDesc = problem.GetIn();
    const auto& dxDesc = problem.GetOut();

    const auto in_n = dxDesc.GetLengths()[0];
    const auto in_c = dxDesc.GetLengths()[1];

    const auto out_spatial =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(conv.GetSpatialDimension());

    const auto dx_t_size = in_n * in_c *
                           std::accumulate(out_spatial.begin(),
                                           out_spatial.end(),
                                           std::size_t(1),
                                           std::multiplies<std::size_t>()) *
                           GetTypeSize(dxDesc.GetType());

    const auto dy_t_size  = dyDesc.GetElementSize() * GetTypeSize(dyDesc.GetType());
    const auto gemm_trans = dx_t_size + dy_t_size;

    if(gemm_trans > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_LOG_I2("GemmBwd1x1_stride2: " << gemm_trans << " > "
                                             << handle.GetMaxMemoryAllocSize());
        return 0;
    }
    return gemm_trans;
#else
    std::ignore = context;
    std::ignore = problem;

    return 0;
#endif
}

bool GemmBwd1x1_stride2::IsSlow(const ExecutionContext& context,
                                const ProblemDescription& problem) const
{
    const std::string& arch        = context.GetStream().GetDeviceName();
    const std::set<std::string> mi = {"gfx942", "gfx955"};
    const bool is_mi               = mi.find(arch) != mi.end();
    const bool is_gfx11            = StartsWith(arch, "gfx11");
    const bool is_gfx12            = StartsWith(arch, "gfx12");

    auto s                      = problem.GetOutHeight() * problem.GetOutWidth();
    auto c                      = problem.GetInChannels() + problem.GetOutChannels();
    auto g                      = problem.GetGroupCount();
    auto channels_per_group     = c / g;
    auto spatial_work_per_group = s * channels_per_group;

    if(is_gfx11 || is_gfx12)
    {
        return false;
    }
    else if(is_mi)
    {
        // PRIMARY: Extreme low CPG detection
        // SWPG < 400k: Moderate spatial-channel work
        // CPG < 192: Low channels per group (critical discriminator)
        if(spatial_work_per_group < 400000 && channels_per_group < 192)
            return true;
    }

    return false;
}

bool GemmBwd1x1_stride2::IsApplicable(const ExecutionContext& context,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmBwdBase::IsApplicable(context, problem))
        return false;

    const auto& conv  = problem.GetConv();
    const auto& wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial =
        wDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);

    return conv.GetSpatialDimension() == 2 &&
           miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 2; }) &&
           GetWorkspaceSize(context, problem) > 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmBwd1x1_stride2::GetSolution(const ExecutionContext& context,
                                             const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    const auto& conv   = problem.GetConv();

    const auto group_count = conv.group_count;

    GemmDescriptor tmp_gemm_desc = [&]() {
        auto tmp =
            group_count > 1
                ? CreateGemmDescriptorGroupConvCNHWBwdData(wDesc, dyDesc, dxDesc, group_count)
                : CreateGemmDescriptorConvCNHWBwdData(wDesc, dyDesc, dxDesc);
        tmp.deterministic = problem.GetConv().attribute.deterministic;
        if(problem.IsTensorsCasted())
        {
            // IsApplicable ensures that both are casted
            if(dyDesc.GetCastType())
                tmp.a_cast_type = *wDesc.GetCastType();
            if(wDesc.GetCastType())
                tmp.b_cast_type = *dyDesc.GetCastType();
        }
        tmp.conv_attributes = problem.GetConv().attribute;
        return tmp;
    }();
    std::size_t in_n, in_c;
    std::tie(in_n, in_c) = tie_pick<0, 1>()(dxDesc.GetLengths());

    const auto wei_k       = wDesc.GetLengths()[0];
    const auto spatial_dim = conv.GetSpatialDimension();
    const auto in_spatial_ =
        dxDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto out_spatial_ =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
    const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());
    const auto strides     = conv.GetConvStrides();

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
            const auto& workspace      = conv_params.workSpace;
            const auto& workspace_size = conv_params.workSpaceSize;
            const auto& dy             = conv_params.tensors.in;
            const auto& dyDesc_        = conv_params.tensors.inDesc;
            const auto& w              = conv_params.tensors.w;
            const auto& dx             = conv_params.tensors.out;
            const auto& dxDesc_        = conv_params.tensors.outDesc;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1 u2xv2");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1 u2xv2");
            }

            if((workspace_req > 0 && workspace == nullptr) || workspace_size < workspace_req)
            {
                MIOPEN_THROW("Not enough workspace for GemmBwd1x1_stride2. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");
            }

            // Initialization required for upsampling in bwd direction
            float zero = 0.f;
            SetTensor(handle, dxDesc_, dx, &zero);

            float time_gemm = 0;
            if(handle.IsProfilingEnabled())
                time_gemm = handle.GetKernelTime();

            // dx = CNHW2NCHW(transpose(w) * NCHW2CNHW(dy))
            transpose_NCHW2CNHW(handle,
                                in_n,
                                wei_k,
                                out_spatial[0],
                                out_spatial[1],
                                out_spatial[0],
                                out_spatial[1],
                                dy,
                                workspace,
                                0,
                                0,
                                1,
                                1,
                                dyDesc_.GetType());

            if(handle.IsProfilingEnabled())
                time_gemm += handle.GetKernelTime();

            miopenStatus_t gemm_status;

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            if(group_count > 1)
            {
                gemm_status = CallGemmStridedBatched(handle,
                                                     gemm_desc,
                                                     w,
                                                     0,
                                                     workspace,
                                                     0,
                                                     workspace,
                                                     dyDesc_.GetElementSize(),
                                                     GemmBackend_t::rocblas);
            }
            else
            {
                // tensors.dx = CNHW2NCHW(transpose(tensors.w) * NCHW2CNHW(tensors.dy))
                gemm_status = CallGemm(handle,
                                       gemm_desc,
                                       w,
                                       0,
                                       workspace,
                                       0,
                                       workspace,
                                       dyDesc_.GetElementSize(),
                                       GemmBackend_t::rocblas);
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GemmBwd1x1_stride2 execution failure.");

            if(handle.IsProfilingEnabled())
                time_gemm += handle.GetKernelTime();

            transpose_CNHW2NCHW(handle,
                                in_n,
                                in_c,
                                out_spatial[0],
                                out_spatial[1],
                                in_spatial[0],
                                in_spatial[1],
                                workspace,
                                dx,
                                dyDesc_.GetElementSize(),
                                0,
                                strides[0],
                                strides[1],
                                dyDesc_.GetType());

            if(handle.IsProfilingEnabled())
            {
                time_gemm += handle.GetKernelTime();
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_gemm);
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

size_t GemmBwd1x1_stride1::GetWorkspaceSize(const ExecutionContext&,
                                            const ProblemDescription&) const
{
    return 0;
}

bool GemmBwd1x1_stride1::IsSlow(const ExecutionContext& context,
                                const ProblemDescription& problem) const
{
    const std::string& arch        = context.GetStream().GetDeviceName();
    const std::set<std::string> mi = {"gfx942", "gfx955"};
    const bool is_mi               = mi.find(arch) != mi.end();
    const bool is_gfx11            = StartsWith(arch, "gfx11");
    const bool is_gfx12            = StartsWith(arch, "gfx12");

    auto s                      = problem.GetOutHeight() * problem.GetOutWidth();
    auto c                      = problem.GetInChannels() + problem.GetOutChannels();
    auto g                      = problem.GetGroupCount();
    auto channels_per_group     = c / g;
    auto spatial_work_per_group = s * channels_per_group;

    if(is_gfx11 || is_gfx12)
    {
        return false;
    }
    else if(is_mi)
    {
        // PRIMARY: Memory-bound small problem detection
        // SWPG < 200k: Low spatial-channel work (memory-bound)
        // CPG < 640: Moderate channels (poor reuse)
        if(spatial_work_per_group < 200000 && channels_per_group < 640)
            return true;
    }

    return false;
}

bool GemmBwd1x1_stride1::IsApplicable(const ExecutionContext& context,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmBwdBase::IsApplicable(context, problem))
        return false;

    const auto& conv  = problem.GetConv();
    const auto& wDesc = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto wei_spatial =
        wDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);

    return miopen::all_of(wei_spatial, [](auto v) { return v == 1; }) &&
           miopen::all_of(conv.GetConvPads(), [](auto v) { return v == 0; }) &&
           miopen::all_of(conv.GetConvStrides(), [](auto v) { return v == 1; });
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmBwd1x1_stride1::GetSolution(const ExecutionContext&,
                                             const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto group_count = problem.GetConv().group_count;
    const auto spatial_dim = problem.GetConv().GetSpatialDimension();

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = 0;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
            const auto& dy          = conv_params.tensors.in;
            const auto& w           = conv_params.tensors.w;
            const auto& dx          = conv_params.tensors.out;
            const auto& dyDesc      = conv_params.tensors.inDesc;
            const auto& wDesc       = conv_params.tensors.wDesc;
            const auto& dxDesc      = conv_params.tensors.outDesc;

            const auto in_n = dxDesc.GetLengths()[0];

            // dx = transpose(w) * dy
            const auto tmp_gemm_desc = [&]() {
                auto tmp =
                    group_count > 1
                        ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                        : CreateGemmStridedBatchedDescriptorConv1x1BwdData(wDesc, dyDesc, dxDesc);
                tmp.deterministic = problem.GetConv().attribute.deterministic;
                if(problem.IsTensorsCasted())
                {
                    // IsApplicable ensures that both are casted
                    if(dyDesc.GetCastType())
                        tmp.a_cast_type = *wDesc.GetCastType();
                    if(wDesc.GetCastType())
                        tmp.b_cast_type = *dyDesc.GetCastType();
                }
                tmp.conv_attributes = problem.GetConv().attribute;
                return tmp;
            }();

            const auto in_c  = dxDesc.GetLengths()[1];
            const auto wei_k = wDesc.GetLengths()[0];

            const auto in_spatial =
                dxDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
            const auto out_spatial =
                dyDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);

            std::size_t out_spatial_size = std::accumulate(out_spatial.begin(),
                                                           out_spatial.end(),
                                                           std::size_t(1),
                                                           std::multiplies<std::size_t>());

            std::size_t in_spatial_size = std::accumulate(in_spatial.begin(),
                                                          in_spatial.end(),
                                                          std::size_t(1),
                                                          std::multiplies<std::size_t>());

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, 1x1");
            }

            miopenStatus_t gemm_status = miopenStatusUnknownError;

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            if(group_count > 1)
            {
                float time = 0.f;
                for(std::size_t i = 0; i < in_n; i++)
                {
                    std::size_t out_offset = i * wei_k * out_spatial_size;

                    std::size_t in_offset = i * in_c * in_spatial_size;

                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         dy,
                                                         out_offset,
                                                         dx,
                                                         in_offset,
                                                         GemmBackend_t::rocblas);

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
                gemm_status = CallGemmStridedBatched(
                    handle, gemm_desc, w, 0, dy, 0, dx, 0, GemmBackend_t::rocblas);
            }

            if(gemm_status != miopenStatusSuccess)
                MIOPEN_THROW("GemmBwd1x1_stride1 execution failure.");
        };
    };

    return solution;
#else
    std::ignore = problem;
    return {};
#endif
}

size_t GemmBwdRest::GetWorkspaceSize(const ExecutionContext& context,
                                     const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    auto& handle       = context.GetStream();
    const auto& conv   = problem.GetConv();
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();

    const auto spatial_dim = conv.GetSpatialDimension();

    const auto wei_spatial =
        wDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto out_spatial =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);

    const auto wei_c = wDesc.GetLengths()[1];
    const auto in_n  = dyDesc.GetLengths()[0];

    auto gemm_size = wei_c *
                     std::accumulate(wei_spatial.begin(),
                                     wei_spatial.end(),
                                     std::size_t(1),
                                     std::multiplies<std::size_t>()) *
                     std::accumulate(out_spatial.begin(),
                                     out_spatial.end(),
                                     std::size_t(1),
                                     std::multiplies<std::size_t>()) *
                     GetTypeSize(dyDesc.GetType()) * conv.group_count;

    // 3D regular convolution uses one strided-batched GEMM over N, so workspace must hold
    // all N GEMM outputs before batched Col2Im.
    if(miopen::conv::IsBwdDataPointOutput3dStrideEqFilter(problem))
        gemm_size *= in_n;

    if(gemm_size > handle.GetMaxMemoryAllocSize())
    {
        MIOPEN_LOG_I2("GemmBwdRest: " << gemm_size << " > " << handle.GetMaxMemoryAllocSize());
        return 0;
    }
    return gemm_size;
#else
    std::ignore = context;
    std::ignore = problem;

    return 0;
#endif
}

bool GemmBwdRest::IsSlow(const ExecutionContext& context, const ProblemDescription& problem) const
{
    const std::string& arch        = context.GetStream().GetDeviceName();
    const std::set<std::string> mi = {"gfx942", "gfx955"};
    const bool is_mi               = mi.find(arch) != mi.end();
    const bool is_gfx11            = StartsWith(arch, "gfx11");
    const bool is_gfx12            = StartsWith(arch, "gfx12");

    auto b                      = problem.GetBatchSize();
    auto s                      = problem.GetOutHeight() * problem.GetOutWidth();
    auto c                      = problem.GetInChannels() + problem.GetOutChannels();
    auto g                      = problem.GetGroupCount();
    auto spatial_per_batch      = s / b;
    auto channels_per_group     = c / g;
    auto spatial_work_per_group = s * channels_per_group;

    if(is_gfx11 || is_gfx12)
    {
        // GemmBwdRest - Multi-metric filtering
        // Analysis: 51.6% terrible cases - significant filtering benefit
        //
        // PRIMARY: Memory-bound small problem detection
        // SWPG < 1.6M: Low spatial-channel work
        // CPG < 360: Low channels
        if(spatial_work_per_group < 1600000 && channels_per_group < 360)
            return true;

        // SECONDARY: Batch fragmentation detection
        // SPB < 0.8: Extreme batch fragmentation
        if(spatial_per_batch < 0.8)
            return true;
    }
    else if(is_mi)
    {
        // PRIMARY: Memory-bound small problem detection
        // SWPG < 3M: Low spatial-channel work
        // CPG < 112: Very low channels
        if(spatial_work_per_group < 3000000 && channels_per_group < 112)
            return true;

        // SECONDARY: Batch fragmentation detection
        // SPB < 40.0: Each batch item has < 40 pixels of spatial work
        if(spatial_per_batch < 40.0)
            return true;
    }

    return false;
}

bool GemmBwdRest::IsApplicable(const ExecutionContext& context,
                               const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    if(!GemmBwdBase::IsApplicable(context, problem))
        return false;

    return !GemmBwd1x1_stride2{}.IsApplicable(context, problem) &&
           !GemmBwd1x1_stride1{}.IsApplicable(context, problem) &&
           GetWorkspaceSize(context, problem) > 0;
#else
    std::ignore = context;
    std::ignore = problem;
    return false;
#endif
}

ConvSolution GemmBwdRest::GetSolution(const ExecutionContext& context,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_USE_GEMM
    const auto& dyDesc = problem.GetIn();
    const auto& wDesc  = problem.GetWeights();
    const auto& dxDesc = problem.GetOut();
    const auto& conv   = problem.GetConv();

    const auto group_count = conv.group_count;

    const auto spatial_dim = conv.GetSpatialDimension();
    const auto in_spatial_ =
        dxDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto wei_spatial_ =
        wDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto out_spatial_ =
        dyDesc.GetLengths() | std::views::drop(2) | std::views::take(spatial_dim);
    const auto in_spatial  = std::vector<std::size_t>(in_spatial_.begin(), in_spatial_.end());
    const auto wei_spatial = std::vector<std::size_t>(wei_spatial_.begin(), wei_spatial_.end());
    const auto out_spatial = std::vector<std::size_t>(out_spatial_.begin(), out_spatial_.end());

    // dx = transpose(w) * dy
    const auto tmp_gemm_desc = [&]() {
        auto tmp          = group_count > 1
                                ? CreateGemmDescriptorGroupConvBwdData(wDesc, dyDesc, dxDesc, group_count)
                                : CreateGemmDescriptorConvBwdData(wDesc, dyDesc, dxDesc);
        tmp.deterministic = problem.GetConv().attribute.deterministic;
        if(problem.IsTensorsCasted())
        {
            // IsApplicable ensures that both are casted
            if(dyDesc.GetCastType())
                tmp.a_cast_type = *wDesc.GetCastType();
            if(wDesc.GetCastType())
                tmp.b_cast_type = *dyDesc.GetCastType();
        }
        tmp.conv_attributes = problem.GetConv().attribute;
        return tmp;
    }();
    const auto spatial_dims = conv.GetSpatialDimension();
    const auto pads         = conv.GetConvPads();
    const auto strides      = conv.GetConvStrides();
    const auto dilations    = conv.GetConvDilations();

    const auto in_n  = dxDesc.GetLengths()[0];
    const auto in_c  = dxDesc.GetLengths()[1];
    const auto wei_k = wDesc.GetLengths()[0];

    const auto out_spatial_size = std::accumulate(
        out_spatial.begin(), out_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto in_spatial_size = std::accumulate(
        in_spatial.begin(), in_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());
    const auto wei_spatial_size = std::accumulate(
        wei_spatial.begin(), wei_spatial.end(), std::size_t(1), std::multiplies<std::size_t>());

    const auto workspace_req = GetWorkspaceSize(context, problem);

    auto solution         = ConvSolution{miopenStatusSuccess};
    solution.workspace_sz = workspace_req;

    solution.invoker_factory = [=](const std::vector<Kernel>&) {
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& conv_params    = primitive_params.CastTo<miopen::conv::DataInvokeParams>();
            const auto& workspace      = conv_params.workSpace;
            const auto& workspace_size = conv_params.workSpaceSize;
            const auto& dy             = conv_params.tensors.in;
            const auto& dyDesc_        = conv_params.tensors.inDesc;
            const auto& w              = conv_params.tensors.w;
            const auto& dx             = conv_params.tensors.out;

            if(group_count > 1)
            {
                MIOPEN_LOG_FUNCTION("groupconv, non 1x1");
            }
            else
            {
                MIOPEN_LOG_FUNCTION("convolution, non 1x1");
            }

            if((workspace_req > 0 && workspace == nullptr) || workspace_size < workspace_req)
            {
                MIOPEN_THROW("Not enough workspace for GemmBwdRest. (" +
                             std::to_string(workspace_size) + " < " +
                             std::to_string(workspace_req) + ")");
            }

            const auto gemm_desc = [&]() {
                auto tmp            = tmp_gemm_desc;
                tmp.gfx90a_alt_impl = conv_params.gfx90aFp16alt;
                return tmp;
            }();

            float time_gemm = 0;

            // Specialized 3D path for non-grouped convolutions:
            // launch one regular GEMM over N, then one batched Col2Im.
            if(miopen::conv::IsBwdDataPointOutput3dStrideEqFilter(problem))
            {
                auto single_gemm_desc        = gemm_desc;
                single_gemm_desc.batch_count = 1;
                single_gemm_desc.strideA     = 0;
                single_gemm_desc.strideB     = 0;
                single_gemm_desc.strideC     = 0;
                // C[N, C*Z*Y*X] = DY[N, K] * W[K, C*Z*Y*X]
                single_gemm_desc.m = static_cast<int>(in_n);
                single_gemm_desc.n = static_cast<int>(in_c * wei_spatial_size * out_spatial_size);
                single_gemm_desc.transA = false;
                single_gemm_desc.transB = false;
                single_gemm_desc.lda    = static_cast<int>(wei_k);
                single_gemm_desc.ldb    = single_gemm_desc.n;
                single_gemm_desc.ldc    = single_gemm_desc.n;

                constexpr auto batched_backend =
#if MIOPEN_USE_HIPBLASLT
                    GemmBackend_t::hipblaslt;
#else
                    GemmBackend_t::rocblas;
#endif

                const auto gemm_status =
                    CallGemm(handle, single_gemm_desc, dy, 0, w, 0, workspace, 0, batched_backend);
                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GemmBwdRest single GEMM execution failure.");

                if(handle.IsProfilingEnabled())
                    time_gemm += handle.GetKernelTime();

                time_gemm += Col2Im3dGPUBatched(handle,
                                                workspace,
                                                out_spatial[0],
                                                out_spatial[1],
                                                out_spatial[2],
                                                wei_spatial[0],
                                                wei_spatial[1],
                                                wei_spatial[2],
                                                static_cast<uint32_t>(pads[0]),
                                                static_cast<uint32_t>(pads[1]),
                                                static_cast<uint32_t>(pads[2]),
                                                static_cast<uint32_t>(strides[0]),
                                                static_cast<uint32_t>(strides[1]),
                                                static_cast<uint32_t>(strides[2]),
                                                static_cast<uint32_t>(dilations[0]),
                                                static_cast<uint32_t>(dilations[1]),
                                                static_cast<uint32_t>(dilations[2]),
                                                static_cast<uint32_t>(in_c),
                                                static_cast<uint32_t>(in_spatial[0]),
                                                static_cast<uint32_t>(in_spatial[1]),
                                                static_cast<uint32_t>(in_spatial[2]),
                                                static_cast<uint32_t>(in_n),
                                                static_cast<uint64_t>(in_c) * wei_spatial_size *
                                                    out_spatial_size,
                                                dx,
                                                static_cast<uint64_t>(in_c) * in_spatial_size,
                                                0,
                                                dyDesc_.GetType());
                if(handle.IsProfilingEnabled())
                {
                    handle.ResetKernelTime();
                    handle.AccumKernelTime(time_gemm);
                }
                return;
            }

            for(std::size_t i = 0; i < in_n; i++)
            {
                std::size_t out_offset = i * wei_k * out_spatial_size;
                std::size_t in_offset  = i * in_c * in_spatial_size;

                miopenStatus_t gemm_status;

                // tensors.dx = transpose(tensors.w) * tensors.dy
                if(group_count > 1)
                {
                    gemm_status = CallGemmStridedBatched(handle,
                                                         gemm_desc,
                                                         w,
                                                         0,
                                                         dy,
                                                         out_offset,
                                                         workspace,
                                                         0,
                                                         GemmBackend_t::rocblas);
                }
                else
                {
                    gemm_status = CallGemm(handle,
                                           gemm_desc,
                                           w,
                                           0,
                                           dy,
                                           out_offset,
                                           workspace,
                                           0,
                                           GemmBackend_t::rocblas);
                }

                if(gemm_status != miopenStatusSuccess)
                    MIOPEN_THROW("GemmBwdRest execution failure.");

                if(handle.IsProfilingEnabled())
                    time_gemm += handle.GetKernelTime();

                time_gemm += Col2ImGPU(handle,
                                       spatial_dims,
                                       workspace,
                                       out_spatial,
                                       wei_spatial,
                                       pads,
                                       strides,
                                       dilations,
                                       in_c,
                                       in_spatial,
                                       dx,
                                       in_offset,
                                       dyDesc_.GetType());
            }

            if(handle.IsProfilingEnabled())
            {
                handle.ResetKernelTime();
                handle.AccumKernelTime(time_gemm);
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
