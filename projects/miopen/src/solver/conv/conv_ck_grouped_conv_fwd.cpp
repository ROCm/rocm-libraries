// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/handle.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/check_numerics.hpp>
#include <miopen/env.hpp>
#include <miopen/fusion/solvers.hpp>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_BWD_DATA_2D)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP
namespace {

// General-shape depthwise forward kernel (miopen_conv2d_depthwise_fwd_generic.cpp). The CK
// instances in ck_depthwise_fwd_impl.cpp are compiled for fixed filter/stride/pad/spatial/batch
// tuples; this kernel covers the remaining packed NCHW shapes. Its two variants are tuning
// values of the same performance config, next to the CK instance names.
constexpr const char* generic_lds_id    = "DepthwiseFwd2DGenericLds";
constexpr const char* generic_direct_id = "DepthwiseFwd2DGenericDirect";

constexpr int generic_block_size = 256;
// 32 KiB of LDS per workgroup leaves room for several workgroups per CU.
constexpr int generic_lds_floats = 8192;
// Register footprint bounds for the unrolled filter loops.
constexpr int generic_max_filter   = 15;
constexpr int generic_max_stride   = 8;
constexpr int generic_max_dilation = 8;

bool IsGenericId(const std::string& id) { return id == generic_lds_id || id == generic_direct_id; }

bool IsGenericArch(const std::string& arch)
{
    // Measured on gfx950 only.
    return arch == "gfx950" || arch.starts_with("gfx950:");
}

struct GenericDepthwiseConfig
{
    bool use_lds        = false;
    int vo              = 4;
    int tile_h          = 0;
    int tile_w          = 0;
    int tile_planes     = 0;
    int lds_pitch       = 0;
    int tiles_w         = 0;
    int tiles_h         = 0;
    int strips          = 0;
    std::size_t threads = 0;
};

int RoundUp(int v, int m) { return (v + m - 1) / m * m; }

int Span4(int vo, int sw, int fx, int dw) { return RoundUp((vo - 1) * sw + (fx - 1) * dw + 1, 4); }

// Returns false when the problem cannot be launched (index or grid size limits).
bool MakeGenericConfig(const ProblemDescription& problem, bool use_lds, GenericDepthwiseConfig& cfg)
{
    const auto planes = static_cast<std::int64_t>(problem.GetBatchSize()) *
                        static_cast<std::int64_t>(problem.GetInChannels());
    const auto ho = static_cast<int>(problem.GetOutHeight());
    const auto wo = static_cast<int>(problem.GetOutWidth());
    const auto fy = static_cast<int>(problem.GetWeightsHeight());
    const auto fx = static_cast<int>(problem.GetWeightsWidth());
    const auto sh = problem.GetKernelStrideH();
    const auto sw = problem.GetKernelStrideW();
    const auto dh = problem.GetDilationH();
    const auto dw = problem.GetDilationW();
    // Grids above 2^31 work-items are not supported by this launcher.
    constexpr std::int64_t max_threads = std::numeric_limits<std::int32_t>::max();

    cfg         = {};
    cfg.use_lds = use_lds;
    if(!use_lds)
    {
        // Measured on MI355X: two outputs per thread for strided filters, eight for narrow rows.
        cfg.vo             = sw > 1 ? 2 : (wo <= 16 ? 8 : 4);
        cfg.strips         = (wo + cfg.vo - 1) / cfg.vo;
        const auto threads = planes * ho * cfg.strips + generic_block_size - 1;
        if(threads > max_threads)
            return false;
        cfg.threads = static_cast<std::size_t>(threads / generic_block_size * generic_block_size);
        return true;
    }

    cfg.vo             = (sw == 1 && wo >= 32) ? 8 : 4;
    const auto in_rows = [&](int th) { return (th - 1) * sh + (fy - 1) * dh + 1; };
    const auto pitch   = [&](int tw) {
        const int cols = (tw - 1) * sw + (fx - 1) * dw + 1;
        const int end  = (tw / cfg.vo - 1) * cfg.vo * sw + Span4(cfg.vo, sw, fx, dw);
        return RoundUp(std::max(cols, end), 4);
    };
    const auto lds_need = [&](int th, int tw, int pb) {
        return static_cast<std::int64_t>(pb) * (in_rows(th) * pitch(tw) + fy * fx);
    };

    // One strip of vo outputs per thread: aim for 256 strips per workgroup.
    int tile_w = std::min(RoundUp(wo, cfg.vo), 64 * cfg.vo);
    while(tile_w > cfg.vo && lds_need(1, tile_w, 1) > generic_lds_floats)
        tile_w -= cfg.vo;
    if(lds_need(1, tile_w, 1) > generic_lds_floats)
        return false;
    const int strips  = tile_w / cfg.vo;
    int tile_h        = std::min(ho, (generic_block_size + strips - 1) / strips);
    const int tiles_h = (ho + tile_h - 1) / tile_h;
    tile_h            = (ho + tiles_h - 1) / tiles_h; // balance the row tiles
    while(tile_h > 1 && lds_need(tile_h, tile_w, 1) > generic_lds_floats)
        --tile_h;

    const int tiles_w = (wo + tile_w - 1) / tile_w;
    int tile_planes   = 1;
    if(tile_h == ho && tiles_w == 1)
    {
        // Small planes: several planes per workgroup, but keep enough workgroups to fill the GPU.
        tile_planes = std::max(1, generic_block_size / (ho * strips));
        while(tile_planes > 1 && lds_need(tile_h, tile_w, tile_planes) > generic_lds_floats)
            --tile_planes;
        while(tile_planes > 1 && (planes + tile_planes - 1) / tile_planes < 1024)
            tile_planes = (tile_planes + 1) / 2;
    }
    cfg.tile_h        = tile_h;
    cfg.tile_w        = tile_w;
    cfg.tile_planes   = tile_planes;
    cfg.lds_pitch     = pitch(tile_w);
    cfg.tiles_w       = tiles_w;
    cfg.tiles_h       = (ho + tile_h - 1) / tile_h;
    const auto blocks = static_cast<std::int64_t>(cfg.tiles_w) * cfg.tiles_h *
                        ((planes + tile_planes - 1) / tile_planes);
    if(blocks * generic_block_size > max_threads)
        return false;
    cfg.threads = static_cast<std::size_t>(blocks * generic_block_size);
    return true;
}

bool IsGenericDepthwiseFwdApplicable(const std::string& arch, const ProblemDescription& problem)
{
    if(!IsGenericArch(arch))
        return false;
    if(!problem.Is2d() || !problem.IsDirectionForward() || !problem.IsLayoutDefault() ||
       (!problem.IsFp16() && !problem.IsBfp16()) || problem.IsTensorsCasted() ||
       problem.HasMixedDataTypes() || problem.HasNonPackedTensors() ||
       !problem.AllTensorsLengthsFitIntoInt())
        return false;
    if(problem.GetConv().mode != miopenConvolution ||
       problem.GetConv().paddingMode != miopenPaddingDefault)
        return false;

    const auto groups = static_cast<std::size_t>(problem.GetGroupCount());
    if(groups == 0 || problem.GetInChannels() != groups || problem.GetOutChannels() != groups)
        return false;

    const auto fy = static_cast<int>(problem.GetWeightsHeight());
    const auto fx = static_cast<int>(problem.GetWeightsWidth());
    if(fy < 1 || fx < 1 || fy > generic_max_filter || fx > generic_max_filter)
        return false;
    const auto sh = problem.GetKernelStrideH();
    const auto sw = problem.GetKernelStrideW();
    const auto dh = problem.GetDilationH();
    const auto dw = problem.GetDilationW();
    if(sh < 1 || sw < 1 || sh > generic_max_stride || sw > generic_max_stride || dh < 1 || dw < 1 ||
       dh > generic_max_dilation || dw > generic_max_dilation)
        return false;
    if(problem.GetPadH() < 0 || problem.GetPadW() < 0)
        return false;
    if(problem.GetBatchSize() == 0 || problem.GetOutHeight() == 0 || problem.GetOutWidth() == 0)
        return false;

    // Offsets inside one plane and plane counts are 32-bit in the kernel.
    constexpr auto int_max = static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max());
    if(problem.GetInHeight() * problem.GetInWidth() > int_max ||
       problem.GetOutHeight() * problem.GetOutWidth() > int_max ||
       problem.GetBatchSize() * problem.GetInChannels() > int_max)
        return false;

    GenericDepthwiseConfig cfg;
    return MakeGenericConfig(problem, false, cfg) && MakeGenericConfig(problem, true, cfg);
}

// Heuristic variant choice, from MI355X measurements: reading through the cache wins for 3x3
// filters with stride, for narrow rows and for small problems; LDS staging wins otherwise.
bool GenericPrefersLds(const ProblemDescription& problem)
{
    const auto taps    = problem.GetWeightsHeight() * problem.GetWeightsWidth();
    const auto wo      = problem.GetOutWidth();
    const auto outputs = problem.GetOut().GetElementSize();
    if(wo <= 8)
        return false;
    if(taps <= 9 &&
       (problem.GetKernelStrideW() > 1 || wo <= 32 || outputs < (std::size_t{1} << 20)))
        return false;
    return true;
}

ConvSolution GetGenericDepthwiseFwdSolution(const ProblemDescription& problem, bool use_lds)
{
    GenericDepthwiseConfig cfg;
    if(!MakeGenericConfig(problem, use_lds, cfg))
        return ConvSolution{miopenStatusInternalError};

    KernelInfo kernel;
    kernel.kernel_file = "miopen_conv2d_depthwise_fwd_generic.cpp";
    kernel.kernel_name = "miopen_conv2d_depthwise_fwd_generic";
    kernel.l_wk        = {static_cast<std::size_t>(generic_block_size), 1, 1};
    kernel.g_wk        = {cfg.threads, 1, 1};

    std::ostringstream options;
    options << (problem.IsFp16() ? " -DIO_DTYPE=__half" : " -DIO_DTYPE=hip_bfloat16")
            << " -DDW_FY=" << problem.GetWeightsHeight() << " -DDW_FX=" << problem.GetWeightsWidth()
            << " -DDW_SH=" << problem.GetKernelStrideH()
            << " -DDW_SW=" << problem.GetKernelStrideW() << " -DDW_DH=" << problem.GetDilationH()
            << " -DDW_DW=" << problem.GetDilationW() << " -DDW_VO=" << cfg.vo
            << " -DDW_USE_LDS=" << (use_lds ? 1 : 0);
    if(use_lds)
        options << " -DDW_LDS_FLOATS=" << generic_lds_floats;
    kernel.comp_options = options.str();

    const auto channels = static_cast<int>(problem.GetInChannels());
    const auto planes   = static_cast<int>(problem.GetBatchSize() * problem.GetInChannels());
    const auto hi       = static_cast<int>(problem.GetInHeight());
    const auto wi       = static_cast<int>(problem.GetInWidth());
    const auto ho       = static_cast<int>(problem.GetOutHeight());
    const auto wo       = static_cast<int>(problem.GetOutWidth());
    const auto ph       = problem.GetPadH();
    const auto pw       = problem.GetPadW();

    ConvSolution result;
    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_params) {
            const auto& tensors = primitive_params.CastTo<miopen::conv::DataInvokeParams>().tensors;
            if(cfg.use_lds)
                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 channels,
                                 planes,
                                 hi,
                                 wi,
                                 ho,
                                 wo,
                                 ph,
                                 pw,
                                 cfg.tile_h,
                                 cfg.tile_w,
                                 cfg.tile_planes,
                                 cfg.lds_pitch,
                                 cfg.tiles_w,
                                 cfg.tiles_h);
            else
                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 channels,
                                 planes,
                                 hi,
                                 wi,
                                 ho,
                                 wo,
                                 ph,
                                 pw,
                                 cfg.strips);
        };
    };
    result.construction_params.push_back(std::move(kernel));
    return result;
}

} // namespace
#endif

void PerformanceConfigConvDepthwiseFwd2D::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";
    valid_kernels.clear();

#if MIOPEN_BACKEND_HIP
    const auto data_type = problem.GetInDataType();
    if(data_type != miopenHalf && data_type != miopenBFloat16)
        return;

    const auto arch    = GetCurrentDeviceName();
    const auto& loader = CkImplLibLoader::Get(arch);
    if(loader.IsLoaded())
        valid_kernels =
            loader.FillValidKernels(CKSolverType::DepthwiseFwd, problem, data_type, false);

    // CK instances keep the default slot when they exist; the general kernel is appended so
    // tuning can compare it against them.
    if(IsGenericDepthwiseFwdApplicable(arch, problem))
    {
        if(GenericPrefersLds(problem))
        {
            valid_kernels.emplace_back(generic_lds_id);
            valid_kernels.emplace_back(generic_direct_id);
        }
        else
        {
            valid_kernels.emplace_back(generic_direct_id);
            valid_kernels.emplace_back(generic_lds_id);
        }
    }

    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[0];
    }
#endif
}

bool PerformanceConfigConvDepthwiseFwd2D::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit({}, problem);
        return true;
    }
    if(index + 1 < valid_kernels.size())
    {
        index++;
        kernel_id = valid_kernels[index];
    }
    else
    {
        return false;
    }
    return true;
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseFwd2D::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(kernel_id.empty())
        return false;

    if(IsGenericId(kernel_id))
        return IsGenericDepthwiseFwdApplicable(GetCurrentDeviceName(), problem);

    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;

    const auto data_type = problem.GetInDataType();
    return loader.IsArgsSupported(CKSolverType::DepthwiseFwd, problem, kernel_id, data_type, false);
#else
    return IsValidValue();
#endif
}

bool PerformanceConfigConvDepthwiseFwd2D::operator==(
    const PerformanceConfigConvDepthwiseFwd2D& other) const
{
    return kernel_id == other.kernel_id;
}

bool ConvDepthwiseFwd2D::IsApplicable(const ExecutionContext& ctx,
                                      const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    if(!problem.IsLayoutDefault())
        return false;

    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;

    if(!problem.IsDirectionForward())
        return false;

    // Only depthwise convolution is supported
    if((problem.GetGroupCount() != problem.GetOutChannels()) ||
       (problem.GetGroupCount() != problem.GetInChannels()))
        return false;

    const std::string arch = ctx.GetStream().GetDeviceName();
    if(IsGenericDepthwiseFwdApplicable(arch, problem))
        return true;

    const auto& loader = CkImplLibLoader::Get(arch);
    if(!loader.IsLoaded())
        return false;

    const auto data_type = problem.GetInDataType();
    return loader.IsApplicable(CKSolverType::DepthwiseFwd, problem, data_type, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

PerformanceConfigConvDepthwiseFwd2D ConvDepthwiseFwd2D::GetDefaultPerformanceConfig(
    [[maybe_unused]] const ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseFwd2D pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseFwd2D::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseFwd2D& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseFwd2D
ConvDepthwiseFwd2D::Search(const ExecutionContext& ctx,
                           const miopen::conv::ProblemDescription& problem,
                           const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvDepthwiseFwd2D::GetSolution(const ExecutionContext& ctx,
                                const miopen::conv::ProblemDescription& problem,
                                const PerformanceConfigConvDepthwiseFwd2D& config) const
{
#if MIOPEN_BACKEND_HIP
    if(IsGenericId(config.kernel_id))
    {
        if(!IsGenericDepthwiseFwdApplicable(ctx.GetStream().GetDeviceName(), problem))
            return ConvSolution{miopenStatusInternalError};
        auto solution = GetGenericDepthwiseFwdSolution(problem, config.kernel_id == generic_lds_id);
        solution.solver_id = SolverDbId();
        return solution;
    }

    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    // Guard: only request a solution for a kernel_id that this build's CK instance factory
    // actually contains. A SystemDB entry can carry a depthwise kernel_id tuned against a
    // different CK kernel set (e.g. another commit); without this check the unrecognized id
    // reaches ck_impl_depthwise_fwd_get_solution, which throws
    // "No matching kernel found for kernel_id" (CK_IMPL_STATUS_INVALID_VALUE) and aborts the
    // process instead of failing gracefully. IsArgsSupported performs the same factory lookup
    // gracefully, matching IsValidPerformanceConfig() / PerformanceConfig::IsValid().
    const auto data_type = problem.GetInDataType();
    if(config.kernel_id.empty() ||
       !loader.IsArgsSupported(
           CKSolverType::DepthwiseFwd, problem, config.kernel_id, data_type, false))
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(CKSolverType::DepthwiseFwd, ctx, problem, config.kernel_id, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    std::ignore = config;
    return ConvSolution{miopenStatusInternalError};
#endif
}

// ===========================================================================
// ConvDepthwiseBwdData2D -- input-gradient (dgrad) via the forward CK depthwise
// kernel with 180-degree-rotated filter (valid for stride-1 depthwise).
// ===========================================================================

void PerformanceConfigConvDepthwiseBwdData2D::HeuristicInit(
    [[maybe_unused]] const ExecutionContext& ctx,
    [[maybe_unused]] const ProblemDescription& problem)
{
    index     = 0;
    kernel_id = "";
#if MIOPEN_BACKEND_HIP
    const auto data_type = problem.GetInDataType();
    if(data_type != miopenHalf && data_type != miopenBFloat16)
        return;
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return;
    valid_kernels =
        loader.FillValidKernels(CKSolverType::DepthwiseBwdData, problem, data_type, false);
    if(!valid_kernels.empty())
    {
        index     = 0;
        kernel_id = valid_kernels[0];
    }
#endif
}

bool PerformanceConfigConvDepthwiseBwdData2D::SetNextValue(const ProblemDescription& problem)
{
    if(valid_kernels.empty())
    {
        HeuristicInit({}, problem);
        return true;
    }
    if(index + 1 < valid_kernels.size())
    {
        index++;
        kernel_id = valid_kernels[index];
    }
    else
    {
        return false;
    }
    return true;
}

bool PerformanceConfigConvDepthwiseBwdData2D::IsValidValue() const
{
    return index < valid_kernels.size();
}

bool PerformanceConfigConvDepthwiseBwdData2D::IsValid(
    [[maybe_unused]] const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(kernel_id.empty())
        return false;
    const auto& loader = CkImplLibLoader::Get(GetCurrentDeviceName());
    if(!loader.IsLoaded())
        return false;
    const auto data_type = problem.GetInDataType();
    return loader.IsArgsSupported(
        CKSolverType::DepthwiseBwdData, problem, kernel_id, data_type, false);
#else
    return IsValidValue();
#endif
}

bool PerformanceConfigConvDepthwiseBwdData2D::operator==(
    const PerformanceConfigConvDepthwiseBwdData2D& other) const
{
    return kernel_id == other.kernel_id;
}

bool ConvDepthwiseBwdData2D::IsApplicable(const ExecutionContext& ctx,
                                          const ProblemDescription& problem) const
{
#if MIOPEN_BACKEND_HIP
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_BWD_DATA_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    if(!problem.IsLayoutDefault())
        return false;
    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;
    if(!problem.IsDirectionBackwardData())
        return false;
    // Only depthwise convolution is supported
    if((problem.GetGroupCount() != problem.GetOutChannels()) ||
       (problem.GetGroupCount() != problem.GetInChannels()))
        return false;

    const std::string arch = ctx.GetStream().GetDeviceName();
    const auto& loader     = CkImplLibLoader::Get(arch);
    if(!loader.IsLoaded())
        return false;
    const auto data_type = problem.GetInDataType();
    return loader.IsApplicable(CKSolverType::DepthwiseBwdData, problem, data_type, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    return false;
#endif
}

PerformanceConfigConvDepthwiseBwdData2D ConvDepthwiseBwdData2D::GetDefaultPerformanceConfig(
    [[maybe_unused]] const ExecutionContext& ctx,
    const miopen::conv::ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseBwdData2D pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseBwdData2D::IsValidPerformanceConfig(
    const ExecutionContext&,
    const miopen::conv::ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseBwdData2D& config) const
{
    return config.IsValid((problem));
}

PerformanceConfigConvDepthwiseBwdData2D
ConvDepthwiseBwdData2D::Search(const ExecutionContext& ctx,
                               const miopen::conv::ProblemDescription& problem,
                               const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution
ConvDepthwiseBwdData2D::GetSolution(const ExecutionContext& ctx,
                                    const miopen::conv::ProblemDescription& problem,
                                    const PerformanceConfigConvDepthwiseBwdData2D& config) const
{
#if MIOPEN_BACKEND_HIP
    const auto& loader = CkImplLibLoader::Get(ctx.GetStream().GetDeviceName());
    if(!loader.IsLoaded())
        return ConvSolution{miopenStatusInternalError};

    // Guard (same rationale as ConvDepthwiseFwd2D::GetSolution): only request a solution for a
    // kernel_id this build's CK dgrad factory actually contains. A stale SystemDB entry (tuned
    // against a different CK kernel set) would otherwise reach
    // ck_impl_depthwise_bwd_data_get_solution, which throws "No matching kernel found for
    // kernel_id" and aborts the process instead of failing gracefully. IsArgsSupported performs the
    // same factory lookup gracefully.
    const auto data_type = problem.GetInDataType();
    if(config.kernel_id.empty() ||
       !loader.IsArgsSupported(
           CKSolverType::DepthwiseBwdData, problem, config.kernel_id, data_type, false))
        return ConvSolution{miopenStatusInternalError};

    return loader.GetSolution(
        CKSolverType::DepthwiseBwdData, ctx, problem, config.kernel_id, false);
#else
    std::ignore = ctx;
    std::ignore = problem;
    std::ignore = config;
    return ConvSolution{miopenStatusInternalError};
#endif
}
} // namespace conv
} // namespace solver
} // namespace miopen
