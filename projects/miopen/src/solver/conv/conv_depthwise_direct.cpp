// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ConvDepthwiseDirect — standalone MIOpen-native depthwise convolution solver.
//
// Harvests the hand-written hipconv VALU depthwise kernels (RDNA wave32) into
// MIOpen's JIT/tuning structure. The four device cores live in
// src/kernels/miopen_depthwise_valu_kernels.hpp; the extern "C" wrapper is
// src/kernels/miopen_depthwise_valu.cpp; this file owns the curated candidate
// table, its selection/validation logic, the grid math, and the tunable
// PerformanceConfig that drives one HIPRTC compile per config.
//
// Tuning model (mirrors ConvHipConv): a PerformanceConfig is an index into the
// candidate table below (best-first per kernel size), plus a checksum of the
// selected row so a perf-db entry written against an older table is rejected on
// restore rather than silently mapping to a different kernel. GenericSearch
// benchmarks the 1-3 valid candidates; the v2 W-strip floor is always valid so
// the valid subset is never empty on RDNA and selection can never regress.
//
// The gfx942/CDNA3 track is scaffolded (Arch::Cdna3 exists in the table's design
// and IsApplicable branches on it), but the CDNA3 cores are net-new authoring
// and are deferred to a gfx942 machine; IsApplicable reports not-applicable on
// gfx942 today so nothing unvalidated runs.

#include <miopen/config.h>
#include <miopen/conv/data_invoke_params.hpp>
#include <miopen/conv/solvers.hpp>
#include <miopen/env.hpp>
#include <miopen/generic_search.hpp>
#include <miopen/handle.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/kernel_info.hpp>
#include <miopen/solver/problem_description_interpreter.hpp>
#include <miopen/stringutils.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <optional>
#include <string>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_CONV_DEPTHWISE_DIRECT)

namespace miopen {
namespace solver {
namespace conv {

using ProblemDescription = miopen::conv::ProblemDescription;

#if MIOPEN_BACKEND_HIP

namespace {

using PI = ProblemInterpreter;

// Which architecture family a candidate row targets.
enum class Arch
{
    Rdna, // gfx11xx / gfx120x, wave32 VALU cores (miopen_depthwise_valu.cpp)
    Cdna3 // gfx942, MFMA cores — scaffold only (cores authored later)
};

// Which device core a candidate maps to. For the RDNA VALU family the wrapper's
// VARIANT macro is the integer value of this enum (0=WStrip .. 3=Lds). The Mfma
// variant selects the gfx942 wrapper (miopen_depthwise_mfma.cpp), which owns its
// own VARIANT numbering; see GetSolution's kernel-file selection.
enum class Variant
{
    WStrip    = 0, // v2_wstrip     — generic floor (any stride/dilation/kernel size)
    Microtile = 1, // v3a_microtile — register halo reuse (s=d=1, compile-time KH/KW)
    Fused     = 2, // v4_fused      — LDS block stage + register micro-tile
    Lds       = 3, // v3b_lds       — LDS block stage + tap loop from LDS
    Mfma      = 4, // gfx942 MFMA   — SCAFFOLD (cores authored in M3; rows are wip)
};

// A tuning point. Designated-initialiser order below matches this declaration.
struct Config
{
    Arch arch;
    Variant variant;
    int kh = 0, kw = 0; // required kernel size for the halo variants (0 = any, v2)
    int wstrip = 4;     // v2 W-strip: output columns per thread
    int th = 0, tw = 0; // v3a microtile output tile
    int bh = 0, bw = 0, bk = 0, rh = 0, rw = 0; // v4/v3b block + register micro-tile
    int kd = 0, bd = 0, rd = 0; // 3D LDS (v3b_lds_core_ndhwc): depth kernel/block/register.
                                // bd > 0 marks a Lds row as the 3D variant.
    int mtile = 0;              // Mfma tile (gfx942 scaffold)
    // Routing guards (0 = unused). Applied on top of the kernel-size match.
    int n_eq      = 0; // require N == n_eq
    int n_min     = 0; // require N >= n_min
    int plane_min = 0; // require output height Ho >= plane_min
    int plane_max = 0; // require output height Ho <= plane_max
    // Work-in-progress: a scaffold row whose device core is not authored yet.
    // wip rows are excluded from every valid subset (is_valid_config), so the
    // solver is cleanly not-applicable for problems only they could serve, and
    // GetSolution never emits their (guarded stub) kernel. Flip to false when the
    // core lands — that is the only change needed to bring the arch live.
    bool wip = false;
};

// The gfx1151-measured routing table (from hipconv depthwise_valu.cpp configs[]),
// best-first within each kernel size. The v2 W-strip floor is last and always
// valid, so the valid subset for any depthwise shape on RDNA is never empty.
//
//   3x3 -> microtile (4x4 default; 2x4 for the N==1 latency case; 2x8 for a
//          small plane with a batch to fill it)
//   5x5 -> fused (16x16x32 r4x4 when the plane is big enough; else 8x8x32 r2x2)
//   7x7 -> fused (same split as 5x5)
//   9x9 -> lds 8x8x16 r2x2 (register-light; v4's patch would spill)
//   everything else (11x11+, stride!=1, non-square) -> v2 floor.
constexpr Config configs[] = {
    // 3x3 microtile
    {.arch    = Arch::Rdna,
     .variant = Variant::Microtile,
     .kh      = 3,
     .kw      = 3,
     .th      = 2,
     .tw      = 4,
     .n_eq    = 1},
    {.arch      = Arch::Rdna,
     .variant   = Variant::Microtile,
     .kh        = 3,
     .kw        = 3,
     .th        = 2,
     .tw        = 8,
     .n_min     = 2,
     .plane_max = 7},
    {.arch = Arch::Rdna, .variant = Variant::Microtile, .kh = 3, .kw = 3, .th = 4, .tw = 4},
    // 5x5 fused
    {.arch      = Arch::Rdna,
     .variant   = Variant::Fused,
     .kh        = 5,
     .kw        = 5,
     .bh        = 16,
     .bw        = 16,
     .bk        = 32,
     .rh        = 4,
     .rw        = 4,
     .plane_min = 8},
    {.arch    = Arch::Rdna,
     .variant = Variant::Fused,
     .kh      = 5,
     .kw      = 5,
     .bh      = 8,
     .bw      = 8,
     .bk      = 32,
     .rh      = 2,
     .rw      = 2},
    // 7x7 fused
    {.arch      = Arch::Rdna,
     .variant   = Variant::Fused,
     .kh        = 7,
     .kw        = 7,
     .bh        = 16,
     .bw        = 16,
     .bk        = 32,
     .rh        = 4,
     .rw        = 4,
     .plane_min = 8},
    {.arch    = Arch::Rdna,
     .variant = Variant::Fused,
     .kh      = 7,
     .kw      = 7,
     .bh      = 8,
     .bw      = 8,
     .bk      = 32,
     .rh      = 2,
     .rw      = 2},
    // 9x9 lds
    {.arch    = Arch::Rdna,
     .variant = Variant::Lds,
     .kh      = 9,
     .kw      = 9,
     .bh      = 8,
     .bw      = 8,
     .bk      = 16,
     .rh      = 2,
     .rw      = 2},
    // 3x3x3 lds (3D NDHWC): stages a halo'd depth block into LDS so the 27-tap
    // stencil reads from LDS, not DRAM. bd > 0 marks this Lds row as the 3D
    // variant. block = (BD/RD)(BH/RH)(BW/RW)BK = 512; LDS = 6*10*10*16*2 = 18.75 KB.
    // Preferred over the v2 floor for 3x3x3 s=d=1; the floor still catches the rest.
    {.arch    = Arch::Rdna,
     .variant = Variant::Lds,
     .kh      = 3,
     .kw      = 3,
     .bh      = 8,
     .bw      = 8,
     .bk      = 8,
     .rh      = 2,
     .rw      = 2,
     .kd      = 3,
     .bd      = 4,
     .rd      = 2},
    // universal floor: any kernel size, any stride/dilation.
    {.arch = Arch::Rdna, .variant = Variant::WStrip, .wstrip = 4},
    // ---- gfx942 / CDNA3 scaffold (M2) ----------------------------------------
    // Full plumbing is present and reviewable, but the MFMA core is net-new
    // authoring (M3, on a CDNA3 machine). Marked wip so is_valid_config always
    // rejects it: ConvDepthwiseDirect is cleanly not-applicable on gfx942 today,
    // and GetSolution never emits the guarded stub. Flipping wip=false once the
    // core lands is the only change needed to bring gfx942 live.
    {.arch = Arch::Cdna3, .variant = Variant::Mfma, .kh = 3, .kw = 3, .mtile = 16, .wip = true},
};
constexpr int N_CONFIGS = sizeof(configs) / sizeof(configs[0]);

constexpr int divup(int a, int b) { return (a + b - 1) / b; }
constexpr long divup(long a, long b) { return (a + b - 1) / b; }

// FP16/BF16 are 2 bytes; the LDS budget math (and the kernel's static_assert)
// assume 16-bit elements.
constexpr int kIoBytes = 2;

std::optional<Arch> GetArch(const std::string& name)
{
    // RDNA is gfx11xx (RDNA3/3.5) and gfx120x (RDNA4). NOT gfx125x (gfx1250 is
    // multi-XCD CDNA5, wave32 despite the gfx12 prefix) — matching MIOpen's
    // convention elsewhere (batchnorm MIO_BN_GFX120X/GFX125X, conv_wino_rage).
    if(StartsWith(name, "gfx11") || StartsWith(name, "gfx120"))
        return Arch::Rdna;
    if(name == "gfx942")
        return Arch::Cdna3;
    return std::nullopt;
}

// Comma-free (the perf-db serializer is CSV) checksum of a row's *kernel
// identity* — variant + kernel size + tile. Rows that compile the same kernel
// but differ only in routing guards intentionally share an id.
std::string describe(const Config& c)
{
    switch(c.variant)
    {
    case Variant::WStrip: return "v2_w" + std::to_string(c.wstrip);
    case Variant::Microtile:
        return "v3a_k" + std::to_string(c.kh) + "x" + std::to_string(c.kw) + "_t" +
               std::to_string(c.th) + "x" + std::to_string(c.tw);
    case Variant::Fused:
        return "v4_k" + std::to_string(c.kh) + "x" + std::to_string(c.kw) + "_b" +
               std::to_string(c.bh) + "x" + std::to_string(c.bw) + "x" + std::to_string(c.bk) +
               "_r" + std::to_string(c.rh) + "x" + std::to_string(c.rw);
    case Variant::Lds:
        if(c.bd > 0) // 3D NDHWC variant (v3b_lds_core_ndhwc) — distinct kernel identity
            return "v3b3d_k" + std::to_string(c.kd) + "x" + std::to_string(c.kh) + "x" +
                   std::to_string(c.kw) + "_b" + std::to_string(c.bd) + "x" + std::to_string(c.bh) +
                   "x" + std::to_string(c.bw) + "x" + std::to_string(c.bk) + "_r" +
                   std::to_string(c.rd) + "x" + std::to_string(c.rh) + "x" + std::to_string(c.rw);
        return "v3b_k" + std::to_string(c.kh) + "x" + std::to_string(c.kw) + "_b" +
               std::to_string(c.bh) + "x" + std::to_string(c.bw) + "x" + std::to_string(c.bk) +
               "_r" + std::to_string(c.rh) + "x" + std::to_string(c.rw);
    case Variant::Mfma: // gfx942 scaffold (wip; never selected)
        return "mfma_k" + std::to_string(c.kh) + "x" + std::to_string(c.kw) + "_m" +
               std::to_string(c.mtile);
    }
    return {};
}

// Is this candidate a legal choice for (device arch, problem)? Mirrors hipconv
// depthwise_valu.cpp is_valid_config, plus a host-side LDS-budget pre-check so
// GenericSearch never dispatches a config that would fail the kernel's JIT
// static_assert.
bool is_valid_config(Arch dev_arch, const ProblemDescription& problem, const Config& cfg)
{
    if(cfg.arch != dev_arch)
        return false;

    // Scaffold rows whose device core is not authored yet are never valid, so the
    // solver reports not-applicable for problems only they could serve and
    // GetSolution never emits their guarded stub kernel.
    if(cfg.wip)
        return false;

    // v2 W-strip serves any depthwise shape, in either layout — the floor is
    // always valid. (GetSolution emits the channel-last or channel-first core via
    // the LAYOUT macro from problem.IsLayoutNHWC().)
    if(cfg.variant == Variant::WStrip)
        return true;

    // Channel-first (NCHW / NCDHW) is served by the WStrip floor only; the
    // halo/LDS variants are channel-last (NHWC / NDHWC) native. Reject them here.
    if(!problem.IsLayoutNHWC())
        return false;

    // 3D LDS variant (v3b_lds_core_ndhwc): NDHWC, s=d=1 on all three axes,
    // compile-time KD/KH/KW match. Checked before the 2D-only guard below.
    if(cfg.variant == Variant::Lds && cfg.bd > 0)
    {
        if(!problem.Is3d())
            return false;
        if(PI::GetAdjustedConvolutionStrideD(problem) != 1 ||
           PI::GetAdjustedConvolutionStrideH(problem) != 1 ||
           PI::GetAdjustedConvolutionStrideW(problem) != 1)
            return false;
        if(PI::GetAdjustedConvolutionDilationD(problem) != 1 ||
           PI::GetAdjustedConvolutionDilationH(problem) != 1 ||
           PI::GetAdjustedConvolutionDilationW(problem) != 1)
            return false;
        if(PI::GetFilterDepthZ(problem) != cfg.kd || PI::GetFilterHeightY(problem) != cfg.kh ||
           PI::GetFilterWidthX(problem) != cfg.kw)
            return false;
        // LDS budget (must match the kernel's static_assert:
        // (BD+KD-1)*(BH+KH-1)*(BW+KW-1)*BK*sizeof(T) <= 64 KB).
        const long lds = static_cast<long>(cfg.bd + cfg.kd - 1) * (cfg.bh + cfg.kh - 1) *
                         (cfg.bw + cfg.kw - 1) * cfg.bk * kIoBytes;
        return lds <= 65536;
    }

    // The halo variants bake a compile-time kernel size and s=d=1, and are 2D
    // only (3D routes to the v2 floor).
    if(!problem.Is2d())
        return false;
    if(PI::GetAdjustedConvolutionStrideH(problem) != 1 ||
       PI::GetAdjustedConvolutionStrideW(problem) != 1)
        return false;
    if(PI::GetAdjustedConvolutionDilationH(problem) != 1 ||
       PI::GetAdjustedConvolutionDilationW(problem) != 1)
        return false;
    if(PI::GetFilterHeightY(problem) != cfg.kh || PI::GetFilterWidthX(problem) != cfg.kw)
        return false;

    // Shape-class routing guards.
    const int n  = PI::GetBatchN(problem);
    const int ho = PI::GetOutputHeightHo(problem);
    if(cfg.n_eq != 0 && n != cfg.n_eq)
        return false;
    if(cfg.n_min != 0 && n < cfg.n_min)
        return false;
    if(cfg.plane_min != 0 && ho < cfg.plane_min)
        return false;
    if(cfg.plane_max != 0 && ho > cfg.plane_max)
        return false;

    // LDS budget for the two-level-blocked variants (must match the kernel's
    // static_assert: (BH+KH-1)*(BW+KW-1)*BK*sizeof(T) <= 64 KB).
    if(cfg.variant == Variant::Fused || cfg.variant == Variant::Lds)
    {
        const long lds =
            static_cast<long>(cfg.bh + cfg.kh - 1) * (cfg.bw + cfg.kw - 1) * cfg.bk * kIoBytes;
        if(lds > 65536)
            return false;
    }
    return true;
}

struct LaunchDims
{
    int block;
    std::size_t grid; // number of workgroups
};

LaunchDims get_launch_params(const Config& cfg, const ProblemDescription& problem)
{
    const int N  = PI::GetBatchN(problem);
    const int C  = PI::GetInputChannelC(problem);
    const int Ho = PI::GetOutputHeightHo(problem);
    const int Wo = PI::GetOutputWidthWo(problem);

    LaunchDims d{256, 1};
    switch(cfg.variant)
    {
    case Variant::WStrip: {
        const int Do = problem.Is3d() ? PI::GetOutputDepthDo(problem) : 1;
        d.block      = 256;
        if(!problem.IsLayoutNHWC())
        {
            // Channel-first (NCHW / NCDHW) floor: one thread per output element,
            // wo innermost (coalesced on the contiguous width axis). No W-strip.
            const long total = static_cast<long>(N) * C * Do * Ho * Wo;
            d.grid           = static_cast<std::size_t>(divup(total, static_cast<long>(d.block)));
            break;
        }
        // Channel-last (NHWC / NDHWC): one thread per (n, do_, ho, w-strip, c).
        // Do folds in for 3D; it is 1 in 2D so the 2D grid is unchanged.
        const long total = static_cast<long>(N) * Do * Ho * divup(Wo, cfg.wstrip) * C;
        d.grid           = static_cast<std::size_t>(divup(total, static_cast<long>(d.block)));
        break;
    }
    case Variant::Microtile: {
        const long total = static_cast<long>(N) * divup(Ho, cfg.th) * divup(Wo, cfg.tw) * C;
        d.block          = 256;
        d.grid           = static_cast<std::size_t>(divup(total, static_cast<long>(d.block)));
        break;
    }
    case Variant::Fused:
    case Variant::Lds: {
        if(cfg.bd > 0) // 3D LDS variant: fold the depth block into grid and block.
        {
            const int Do    = PI::GetOutputDepthDo(problem);
            const long grid = static_cast<long>(N) * divup(Do, cfg.bd) * divup(Ho, cfg.bh) *
                              divup(Wo, cfg.bw) * divup(C, cfg.bk);
            d.block = (cfg.bd / cfg.rd) * (cfg.bh / cfg.rh) * (cfg.bw / cfg.rw) * cfg.bk;
            d.grid  = static_cast<std::size_t>(grid);
            break;
        }
        const long grid =
            static_cast<long>(N) * divup(Ho, cfg.bh) * divup(Wo, cfg.bw) * divup(C, cfg.bk);
        d.block = (cfg.bh / cfg.rh) * (cfg.bw / cfg.rw) * cfg.bk;
        d.grid  = static_cast<std::size_t>(grid);
        break;
    }
    case Variant::Mfma:
        // gfx942 scaffold: wip rows never reach here (is_valid_config rejects
        // them). Grid/block are finalised with the MFMA tiling in M3.
        assert(false && "Mfma launch params not authored (ConvDepthwiseDirect M3)");
        break;
    }
    return d;
}

} // namespace

// -------------------------- PerformanceConfig --------------------------

void PerformanceConfigConvDepthwiseDirect::HeuristicInit(const ExecutionContext& ctx,
                                                         const ProblemDescription& problem)
{
    index = -1;
    config_id.clear();
    const auto arch = GetArch(ctx.GetStream().GetDeviceName());
    if(!arch)
        return;
    // First valid row in table order is the gfx1151-measured routing default.
    for(int i = 0; i < N_CONFIGS; ++i)
    {
        if(is_valid_config(*arch, problem, configs[i]))
        {
            index     = i;
            config_id = describe(configs[i]);
            break;
        }
    }
}

bool PerformanceConfigConvDepthwiseDirect::SetNextValue(const ProblemDescription&)
{
    if(index + 1 >= N_CONFIGS)
        return false;
    ++index;
    config_id = describe(configs[index]);
    return true;
}

bool PerformanceConfigConvDepthwiseDirect::IsValid(const ExecutionContext& ctx,
                                                   const ProblemDescription& problem) const
{
    if(index < 0 || index >= N_CONFIGS)
        return false;
    const auto arch = GetArch(ctx.GetStream().GetDeviceName());
    if(!arch)
        return false;
    // Drift guard on restore: reject a perf-db entry whose stored id no longer
    // matches the row at that index (table reordered/edited between versions).
    if(!config_id.empty() && config_id != describe(configs[index]))
        return false;
    return is_valid_config(*arch, problem, configs[index]);
}

bool PerformanceConfigConvDepthwiseDirect::operator==(
    const PerformanceConfigConvDepthwiseDirect& other) const
{
    return index == other.index;
}

// -------------------------- Solver --------------------------

PerformanceConfigConvDepthwiseDirect
ConvDepthwiseDirect::GetDefaultPerformanceConfig(const ExecutionContext& ctx,
                                                 const ProblemDescription& problem) const
{
    PerformanceConfigConvDepthwiseDirect pp;
    pp.HeuristicInit(ctx, problem);
    return pp;
}

bool ConvDepthwiseDirect::IsValidPerformanceConfig(
    const ExecutionContext& ctx,
    const ProblemDescription& problem,
    const PerformanceConfigConvDepthwiseDirect& config) const
{
    return config.IsValidValue() && config.IsValid(ctx, problem);
}

PerformanceConfigConvDepthwiseDirect
ConvDepthwiseDirect::Search(const ExecutionContext& ctx,
                            const ProblemDescription& problem,
                            const AnyInvokeParams& invoke_ctx) const
{
    return GenericSearch(*this, ctx, problem, invoke_ctx);
}

bool ConvDepthwiseDirect::IsApplicable(const ExecutionContext& ctx,
                                       const ProblemDescription& problem) const
{
    if(env::disabled(MIOPEN_DEBUG_CONV_DEPTHWISE_DIRECT))
        return false;
    if(!ctx.use_hip_kernels)
        return false;

    const auto arch = GetArch(ctx.GetStream().GetDeviceName());
    if(!arch)
        return false;
    // Note: gfx942 (Cdna3) is admitted here, but every Cdna3 config row is wip,
    // so the valid-subset check at the end fails and the solver reports
    // not-applicable on gfx942 until the MFMA core lands (M3).

    if(!problem.Is2d() && !problem.Is3d()) // 3D routes to the v2 WStrip floor only
        return false;
    if(!problem.IsDirectionForward())
        return false;
    if(!problem.IsFp16() && !problem.IsBfp16())
        return false;
    // Channel-last (NHWC / NDHWC — all variants) or channel-first (NCHW / NCDHW —
    // the WStrip floor only; is_valid_config rejects the halo/LDS variants there).
    if(!problem.IsLayoutNHWC() && !problem.IsLayoutDefault())
        return false;
    if(problem.IsTensorsCasted())
        return false;
    if(!problem.AllTensorsLengthsFitIntoInt()) // kernels use int index math
        return false;

    const auto g = static_cast<int>(problem.GetGroupCount());
    const int c  = PI::GetInputChannelC(problem);
    const int k  = PI::GetOutputChannelK(problem);
    if(g == 0 || c != g || k != g) // depthwise: k == c == groups, channels/group == 1
        return false;

    // Matches upstream: v2 implements dilation correctly but it is not routed yet.
    if(PI::GetAdjustedConvolutionDilationH(problem) != 1 ||
       PI::GetAdjustedConvolutionDilationW(problem) != 1)
        return false;
    if(problem.Is3d() && PI::GetAdjustedConvolutionDilationD(problem) != 1)
        return false;

    // At least one candidate must be valid (the WStrip floor guarantees this on RDNA).
    return std::ranges::any_of(
        configs, [&](const Config& config) { return is_valid_config(*arch, problem, config); });
}

ConvSolution
ConvDepthwiseDirect::GetSolution(const ExecutionContext& /*ctx*/,
                                 const ProblemDescription& problem,
                                 const PerformanceConfigConvDepthwiseDirect& config) const
{
    assert(config.index >= 0 && config.index < N_CONFIGS);
    const Config& cfg = configs[config.index];
    // wip rows are excluded from every valid subset, so a valid PerformanceConfig
    // can never resolve to one; guarding here keeps a stub kernel from ever being
    // emitted even if the table/selection logic drifts.
    assert(!cfg.wip && "ConvDepthwiseDirect::GetSolution reached a wip (unauthored) config");
    const LaunchDims dims = get_launch_params(cfg, problem);

    ConvSolution result;

    // The RDNA VALU family and the (scaffolded) gfx942 MFMA family live in
    // separate wrapper translation units, each with its own VARIANT numbering.
    const bool is_mfma = cfg.arch == Arch::Cdna3;

    KernelInfo kernel;
    kernel.kernel_file = is_mfma ? "miopen_depthwise_mfma.cpp" : "miopen_depthwise_valu.cpp";
    kernel.kernel_name = is_mfma ? "miopen_depthwise_mfma" : "miopen_depthwise_valu";
    kernel.l_wk        = {static_cast<std::size_t>(dims.block), 1, 1};
    kernel.g_wk        = {dims.grid * static_cast<std::size_t>(dims.block), 1, 1};

    const bool is3d = problem.Is3d();

    KernelBuildParameters build{
        {"IO_DTYPE", problem.IsFp16() ? std::string("__half") : std::string("__hip_bfloat16")},
        {"VARIANT", static_cast<int>(cfg.variant)},
        {"NDIMS", is3d ? 3 : 2}, // 3D selects the NDHWC WStrip core + depth args
        // LAYOUT: 0 = channel-last (NHWC/NDHWC), 1 = channel-first (NCHW/NCDHW).
        // Only the WStrip floor has a channel-first core; is_valid_config
        // guarantees any channel-first problem resolves to that row.
        {"LAYOUT", problem.IsLayoutNHWC() ? 0 : 1},
    };
    switch(cfg.variant)
    {
    // Macro names carry a MIO_DW_ prefix: the device cores are templated on
    // parameters named WSTRIP/KH/KW/TH/TW/BH/BW/BK/RH/RW, so an unprefixed -D
    // would textually rewrite the header's template-parameter list. See
    // miopen_depthwise_valu.cpp.
    case Variant::WStrip: build.Define("MIO_DW_WSTRIP", cfg.wstrip); break;
    case Variant::Microtile:
        build.Define("MIO_DW_KH", cfg.kh);
        build.Define("MIO_DW_KW", cfg.kw);
        build.Define("MIO_DW_TH", cfg.th);
        build.Define("MIO_DW_TW", cfg.tw);
        break;
    case Variant::Fused:
    case Variant::Lds:
        build.Define("MIO_DW_KH", cfg.kh);
        build.Define("MIO_DW_KW", cfg.kw);
        build.Define("MIO_DW_BH", cfg.bh);
        build.Define("MIO_DW_BW", cfg.bw);
        build.Define("MIO_DW_BK", cfg.bk);
        build.Define("MIO_DW_RH", cfg.rh);
        build.Define("MIO_DW_RW", cfg.rw);
        if(cfg.bd > 0) // 3D LDS variant: add the depth kernel/block/register dims.
        {
            build.Define("MIO_DW_KD", cfg.kd);
            build.Define("MIO_DW_BD", cfg.bd);
            build.Define("MIO_DW_RD", cfg.rd);
        }
        break;
    case Variant::Mfma: // gfx942 scaffold (wip; asserted unreachable above)
        build.Define("MIO_DW_KH", cfg.kh);
        build.Define("MIO_DW_KW", cfg.kw);
        build.Define("MIO_DW_MTILE", cfg.mtile);
        break;
    }
    kernel.comp_options = std::string(" ") + build.GenerateFor(kbp::HIP{});

    result.construction_params.push_back(kernel);

    // Conv geometry that is not carried on the tensor descriptors is captured
    // here from the problem; N/C/H/W are read from the descriptors at invoke.
    const int kh = PI::GetFilterHeightY(problem);
    const int kw = PI::GetFilterWidthX(problem);
    const int ph = PI::GetInputLeftPadH(problem);
    const int pw = PI::GetInputLeftPadW(problem);
    const int sh = PI::GetAdjustedConvolutionStrideH(problem);
    const int sw = PI::GetAdjustedConvolutionStrideW(problem);
    const int dh = PI::GetAdjustedConvolutionDilationH(problem);
    const int dw = PI::GetAdjustedConvolutionDilationW(problem);

    // Depth geometry — only meaningful (and only appended to the launch) for 3D.
    const int kd = is3d ? PI::GetFilterDepthZ(problem) : 1;
    const int pd = is3d ? PI::GetInputLeftPadD(problem) : 0;
    const int sd = is3d ? PI::GetAdjustedConvolutionStrideD(problem) : 1;
    const int dd = is3d ? PI::GetAdjustedConvolutionDilationD(problem) : 1;

    result.invoker_factory = [=](const std::vector<Kernel>& kernels) {
        const auto kern = kernels[0];
        return [=](const Handle& handle, const AnyInvokeParams& primitive_parameters) {
            const auto& data_ctx = primitive_parameters.CastTo<miopen::conv::DataInvokeParams>();
            const auto& tensors  = data_ctx.tensors;
            const auto& in_len   = tensors.inDesc.GetLengths();
            const auto& out_len  = tensors.outDesc.GetLengths();
            if(is3d)
            {
                // NDHWC logical lengths: in {N, C, Di, Hi, Wi}, out {N, C, Do, Ho, Wo}.
                const int N  = static_cast<int>(in_len[0]);
                const int C  = static_cast<int>(in_len[1]);
                const int Di = static_cast<int>(in_len[2]);
                const int Hi = static_cast<int>(in_len[3]);
                const int Wi = static_cast<int>(in_len[4]);
                const int Do = static_cast<int>(out_len[2]);
                const int Ho = static_cast<int>(out_len[3]);
                const int Wo = static_cast<int>(out_len[4]);
                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 N,
                                 C,
                                 Hi,
                                 Wi,
                                 Ho,
                                 Wo,
                                 kh,
                                 kw,
                                 ph,
                                 pw,
                                 sh,
                                 sw,
                                 dh,
                                 dw,
                                 Di,
                                 Do,
                                 kd,
                                 pd,
                                 sd,
                                 dd);
            }
            else
            {
                // NHWC logical lengths: in {N, C, Hi, Wi}, out {N, C, Ho, Wo}.
                const int N  = static_cast<int>(in_len[0]);
                const int C  = static_cast<int>(in_len[1]);
                const int Hi = static_cast<int>(in_len[2]);
                const int Wi = static_cast<int>(in_len[3]);
                const int Ho = static_cast<int>(out_len[2]);
                const int Wo = static_cast<int>(out_len[3]);
                handle.Run(kern)(tensors.in,
                                 tensors.w,
                                 tensors.out,
                                 N,
                                 C,
                                 Hi,
                                 Wi,
                                 Ho,
                                 Wo,
                                 kh,
                                 kw,
                                 ph,
                                 pw,
                                 sh,
                                 sw,
                                 dh,
                                 dw);
            }
        };
    };

    return result;
}

#else

void PerformanceConfigConvDepthwiseDirect::HeuristicInit(const ExecutionContext&,
                                                         const ProblemDescription&)
{
}

bool PerformanceConfigConvDepthwiseDirect::SetNextValue(const ProblemDescription&) { return false; }

bool PerformanceConfigConvDepthwiseDirect::IsValid(const ExecutionContext&,
                                                   const ProblemDescription&) const
{
    return false;
}

bool PerformanceConfigConvDepthwiseDirect::operator==(
    const PerformanceConfigConvDepthwiseDirect& other) const
{
    return index == other.index;
}

PerformanceConfigConvDepthwiseDirect
ConvDepthwiseDirect::GetDefaultPerformanceConfig(const ExecutionContext&,
                                                 const ProblemDescription&) const
{
    return {};
}

bool ConvDepthwiseDirect::IsValidPerformanceConfig(
    const ExecutionContext&,
    const ProblemDescription&,
    const PerformanceConfigConvDepthwiseDirect&) const
{
    return false;
}

PerformanceConfigConvDepthwiseDirect ConvDepthwiseDirect::Search(const ExecutionContext&,
                                                                 const ProblemDescription&,
                                                                 const AnyInvokeParams&) const
{
    return {};
}

bool ConvDepthwiseDirect::IsApplicable(const ExecutionContext&, const ProblemDescription&) const
{
    return false;
}

ConvSolution ConvDepthwiseDirect::GetSolution(const ExecutionContext&,
                                              const ProblemDescription&,
                                              const PerformanceConfigConvDepthwiseDirect&) const
{
    return ConvSolution{miopenStatusNotImplemented};
}

#endif

} // namespace conv
} // namespace solver
} // namespace miopen
