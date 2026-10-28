// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel_info.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

// Not MIOPEN_DEBUG_CONV_DEPTHWISE_FWD_2D: that flag gates the CK-based ConvDepthwiseFwd2D in
// conv_ck_grouped_conv_fwd.cpp, and the declaration macro yields one shared instance across
// translation units, so reusing it would make a single flag gate two unrelated solvers.
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_GENERIC_DEPTHWISE_FWD_2D)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

namespace {

// General-shape depthwise forward kernel (miopen_conv2d_depthwise_fwd_generic.cpp). The CK
// instances behind ConvDepthwiseFwd2D are compiled for fixed filter/stride/pad/spatial/batch
// tuples; this kernel covers the remaining packed NCHW shapes. Its two variants (cache reads
// and LDS staging) are the tuning values of this solver's performance config.
constexpr int generic_block_size = 256;
// 32 KiB of LDS per workgroup leaves room for several workgroups per CU.
constexpr int generic_lds_floats = 8192;
// Register footprint bounds for the unrolled filter loops.
constexpr int generic_max_filter   = 15;
constexpr int generic_max_stride   = 8;
constexpr int generic_max_dilation = 8;

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

// The admitted shapes reach INT32_MAX in a single dimension, so every ceiling, every tile
// count and every row-balancing sum below is formed in int64 and narrowed only once the
// value is known to fit.
std::int64_t CeilDiv(std::int64_t v, std::int64_t m) { return v / m + (v % m != 0 ? 1 : 0); }

std::int64_t RoundUp(std::int64_t v, std::int64_t m) { return CeilDiv(v, m) * m; }

int Span4(int vo, int sw, int fx, int dw)
{
    return static_cast<int>(RoundUp(
        static_cast<std::int64_t>(vo - 1) * sw + static_cast<std::int64_t>(fx - 1) * dw + 1, 4));
}

// Returns false when the problem cannot be launched (index or grid size limits).
bool MakeGenericConfig(const ProblemDescription& problem, bool use_lds, GenericDepthwiseConfig& cfg)
{
    const auto planes = static_cast<std::int64_t>(problem.GetBatchSize()) *
                        static_cast<std::int64_t>(problem.GetInChannels());
    const auto ho = static_cast<std::int64_t>(problem.GetOutHeight());
    const auto wo = static_cast<std::int64_t>(problem.GetOutWidth());
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
        cfg.vo            = sw > 1 ? 2 : (wo <= 16 ? 8 : 4);
        const auto strips = CeilDiv(wo, cfg.vo);
        // planes and ho each fit int32; divide first so their product cannot wrap int64.
        if(planes > max_threads / ho)
            return false;
        const auto threads = planes * ho * strips + generic_block_size - 1;
        if(threads > max_threads)
            return false;
        cfg.strips  = static_cast<int>(strips);
        cfg.threads = static_cast<std::size_t>(threads / generic_block_size * generic_block_size);
        return true;
    }

    cfg.vo             = (sw == 1 && wo >= 32) ? 8 : 4;
    const auto in_rows = [&](std::int64_t th) {
        return (th - 1) * sh + static_cast<std::int64_t>(fy - 1) * dh + 1;
    };
    const auto pitch = [&](std::int64_t tw) {
        const auto cols = (tw - 1) * sw + static_cast<std::int64_t>(fx - 1) * dw + 1;
        const auto end  = (tw / cfg.vo - 1) * cfg.vo * sw + Span4(cfg.vo, sw, fx, dw);
        return RoundUp(std::max(cols, end), 4);
    };
    const auto lds_need = [&](std::int64_t th, std::int64_t tw, std::int64_t pb) {
        return pb * (in_rows(th) * pitch(tw) + static_cast<std::int64_t>(fy) * fx);
    };

    // One strip of vo outputs per thread: aim for 256 strips per workgroup.
    auto tile_w = std::min(RoundUp(wo, cfg.vo), static_cast<std::int64_t>(64 * cfg.vo));
    while(tile_w > cfg.vo && lds_need(1, tile_w, 1) > generic_lds_floats)
        tile_w -= cfg.vo;
    if(lds_need(1, tile_w, 1) > generic_lds_floats)
        return false;
    const auto strips = tile_w / cfg.vo;
    auto tile_h       = std::min(ho, CeilDiv(generic_block_size, strips));
    tile_h            = CeilDiv(ho, CeilDiv(ho, tile_h)); // balance the row tiles
    while(tile_h > 1 && lds_need(tile_h, tile_w, 1) > generic_lds_floats)
        --tile_h;

    const auto tiles_w       = CeilDiv(wo, tile_w);
    const auto tiles_h       = CeilDiv(ho, tile_h);
    std::int64_t tile_planes = 1;
    if(tile_h == ho && tiles_w == 1)
    {
        // Small planes: several planes per workgroup, but keep enough workgroups to fill the GPU.
        tile_planes = std::max<std::int64_t>(1, generic_block_size / (ho * strips));
        while(tile_planes > 1 && lds_need(tile_h, tile_w, tile_planes) > generic_lds_floats)
            --tile_planes;
        while(tile_planes > 1 && CeilDiv(planes, tile_planes) < 1024)
            tile_planes = (tile_planes + 1) / 2;
    }
    cfg.tile_h      = static_cast<int>(tile_h);
    cfg.tile_w      = static_cast<int>(tile_w);
    cfg.tile_planes = static_cast<int>(tile_planes);
    cfg.lds_pitch   = static_cast<int>(pitch(tile_w));
    cfg.tiles_w     = static_cast<int>(tiles_w);
    cfg.tiles_h     = static_cast<int>(tiles_h);
    // tiles_w * tiles_h is bounded by the output plane, which fits int32, so multiplying by
    // the plane-tile count stays well inside int64.
    const auto blocks = tiles_w * tiles_h * CeilDiv(planes, tile_planes);
    if(blocks > max_threads / generic_block_size)
        return false;
    cfg.threads = static_cast<std::size_t>(blocks * generic_block_size);
    return true;
}

// Shape, layout and geometry gating. It is shared by the two variants, which then differ
// only in whether they can launch the problem.
bool IsGenericDepthwiseFwdShapeSupported(const std::string& arch, const ProblemDescription& problem)
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
    return true;
}

// The variants have different launch limits: the LDS tile has to fit in 32 KiB, the direct
// variant only needs its grid to fit. They are independent, so a shape one of them can run is
// not rejected because the other cannot.
bool IsGenericVariantApplicable(const std::string& arch,
                                const ProblemDescription& problem,
                                bool use_lds)
{
    if(!IsGenericDepthwiseFwdShapeSupported(arch, problem))
        return false;
    GenericDepthwiseConfig cfg;
    return MakeGenericConfig(problem, use_lds, cfg);
}

bool IsGenericDepthwiseFwdApplicable(const std::string& arch, const ProblemDescription& problem)
{
    if(!IsGenericDepthwiseFwdShapeSupported(arch, problem))
        return false;
    GenericDepthwiseConfig cfg;
    return MakeGenericConfig(problem, false, cfg) || MakeGenericConfig(problem, true, cfg);
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

void PerformanceConfigConvGenericDepthwiseFwd2D::HeuristicInit(const ExecutionContext& ctx,
                                                               const ProblemDescription& problem)
{
    const auto arch      = ctx.GetStream().GetDeviceName();
    const bool direct_ok = IsGenericVariantApplicable(arch, problem, false);
    const bool lds_ok    = IsGenericVariantApplicable(arch, problem, true);
    if(GenericPrefersLds(problem))
        use_lds = lds_ok ? 1 : (direct_ok ? 0 : -1);
    else
        use_lds = direct_ok ? 0 : (lds_ok ? 1 : -1);
}

bool PerformanceConfigConvGenericDepthwiseFwd2D::SetNextValue(const ProblemDescription&)
{
    if(use_lds >= 1)
        return false;
    ++use_lds;
    return true;
}

bool PerformanceConfigConvGenericDepthwiseFwd2D::IsValidValue() const
{
    return use_lds == 0 || use_lds == 1;
}

bool PerformanceConfigConvGenericDepthwiseFwd2D::IsValid(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem) const
{
    return IsValidValue() &&
           IsGenericVariantApplicable(ctx.GetStream().GetDeviceName(), problem, use_lds == 1);
}

bool PerformanceConfigConvGenericDepthwiseFwd2D::operator==(
    const PerformanceConfigConvGenericDepthwiseFwd2D& other) const
{
    return use_lds == other.use_lds;
}

bool ConvGenericDepthwiseFwd2D::IsApplicable(const ExecutionContext& ctx,
                                             const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_GENERIC_DEPTHWISE_FWD_2D))
        return false;
    if(!ctx.use_hip_kernels)
        return false;
    return IsGenericDepthwiseFwdApplicable(ctx.GetStream().GetDeviceName(), problem);
}

PerformanceConfigConvGenericDepthwiseFwd2D
ConvGenericDepthwiseFwd2D::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                       const ProblemDescription& problem) const
{
    PerformanceConfigConvGenericDepthwiseFwd2D config;
    config.HeuristicInit(ctx, problem);
    return config;
}

bool ConvGenericDepthwiseFwd2D::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvGenericDepthwiseFwd2D& config) const
{
    return config.IsValid(ctx, problem);
}

PerformanceConfigConvGenericDepthwiseFwd2D
ConvGenericDepthwiseFwd2D::Search(const ExecutionContext& ctx,
                                  const ProblemDescription& problem,
                                  const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

ConvSolution ConvGenericDepthwiseFwd2D::GetSolution(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvGenericDepthwiseFwd2D& config) const
{
    if(!config.IsValid(ctx, problem))
        return ConvSolution{miopenStatusInternalError};

    auto solution      = GetGenericDepthwiseFwdSolution(problem, config.use_lds == 1);
    solution.solver_id = SolverDbId();
    return solution;
}

} // namespace conv
} // namespace solver
} // namespace miopen
