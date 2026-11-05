// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvDgradGeometry.hpp"

/// @file Gfx950ConvWgradGeometry.hpp
/// The launch contract of hipkernel:Gfx950ConvWgradAot, the shape-generic backward-weight
/// (wgrad) AOT-instance engine: the instances its catalog describes, the problem it
/// matches, and the launch plan of one instance on one problem -- the served predicate, the
/// split-K degree, the kernels it launches with their grids and arguments, the workspace
/// and the dW zero-fill. It ports gfx950_conv_wgrad_aot_plan of the rocKE adapter
/// rocke/library/builders/common/convolution_backward_filter.py, which is rocKE's own offer
/// (identity_supports_problem, direct_spec_for_problem, resolve_conv_split_k) plus the
/// launch bounds; serializePlan / planDigest are that adapter's Plan.serialize / digest.
///
/// Unlike dgrad there is no tilde sub-GEMM table and no width fold: a wgrad implicit-GEMM
/// binary bakes no geometry and runs the graph's problem as stated. Its split-K degree is
/// a runtime kernarg (ks_count), resolved per problem by the gfx950 split-K cost model
/// (rocke/platform/python/rocke/helpers/split_k_planner.py) and carried on the plan.
namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_wgrad
{

using gfx950_conv_dgrad::boundedProduct;
using gfx950_conv_dgrad::bufferRangesDoNotOverlap;
using gfx950_conv_dgrad::ceilDiv;
using gfx950_conv_dgrad::dgradOutputExtent;
using gfx950_conv_dgrad::fitsI32;
using gfx950_conv_dgrad::I32_MAX;
using gfx950_conv_dgrad::magicDivision;
using gfx950_conv_dgrad::MAX_BLOCK_THREADS;
using gfx950_conv_dgrad::MAX_GRID_DIM;
using gfx950_conv_dgrad::WAVE_SIZE;
using gfx950_conv_dgrad::wideProduct;

/// The 160 KiB LDS budget every direct wgrad validator states.
constexpr int64_t DIRECT_LDS_BUDGET_BYTES = 160 * 1024;

/// a * b for nonnegative operands, saturating at INT64_MAX: knob products only ever meet
/// small upper bounds, which a saturated value still exceeds.
inline int64_t satMul(int64_t a, int64_t b)
{
    constexpr int64_t MAX = std::numeric_limits<int64_t>::max();
    if(a == 0 || b == 0)
    {
        return 0;
    }
    return a > MAX / b ? MAX : a * b;
}

inline int64_t satMul(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const auto factor : factors)
    {
        product = satMul(product, factor);
    }
    return product;
}

inline int64_t satAdd(int64_t a, int64_t b)
{
    constexpr int64_t MAX = std::numeric_limits<int64_t>::max();
    return a > MAX - b ? MAX : a + b;
}

/// Python's a % b for b > 0 (never negative).
inline int64_t floorMod(int64_t a, int64_t b)
{
    const auto r = a % b;
    return r < 0 ? r + b : r;
}

// ---------------------------------------------------------------------------------------
// The problem
// ---------------------------------------------------------------------------------------

/// The activation dtype of the problem and of an instance (dY, X and an implicit-GEMM dW).
enum class ElementType : int64_t
{
    FP16 = 0,
    BF16 = 1,
};

inline std::optional<ElementType> parseElementType(std::string_view value)
{
    if(value == "fp16")
    {
        return ElementType::FP16;
    }
    if(value == "bf16")
    {
        return ElementType::BF16;
    }
    return std::nullopt;
}

/// One backward-weight problem as the engine matches it (Gfx950ConvWgradProblem). C and K
/// are totals across all groups. dY is [N, Ho, Wo, K] NHWK, X is [N, Hi, Wi, C] NHWC and
/// dW is [K, Y, X, C / groups] KYXC.
struct Problem
{
    int64_t n;
    int64_t c;
    int64_t k;
    int64_t hi;
    int64_t wi;
    int64_t y;
    int64_t x;
    int64_t strideH;
    int64_t strideW;
    int64_t padH;
    int64_t padW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t groups = 1;
    ElementType dtype = ElementType::FP16;
};

/// What the problem implies: the dY extents, the per-group channel counts and the three
/// tensors' sizes. tensorBytes is (dY, X, dW) with a 16-bit dW, as an implicit-GEMM
/// instance writes it; dwElements sizes the fp32 dW of a direct instance.
struct Geometry
{
    int64_t ho;
    int64_t wo;
    int64_t cpg;
    int64_t kpg;
    std::array<int64_t, 3> tensorBytes;
    int64_t dwElements;
};

/// _problem_error: positive extents, nonnegative padding, groups in [1, 65535] dividing C
/// and K, a positive dY, and dY, X and an fp32 dW each within the signed 32-bit kernel
/// ABI. wgrad's reduction spans the whole batch, so there is no per-image escape valve.
inline std::optional<Geometry> deriveGeometry(const Problem& problem)
{
    if(problem.n <= 0 || problem.n > I32_MAX || problem.c <= 0 || problem.c > I32_MAX
       || problem.k <= 0 || problem.k > I32_MAX || problem.groups < 1
       || problem.groups > MAX_GRID_DIM || problem.c % problem.groups != 0
       || problem.k % problem.groups != 0)
    {
        return std::nullopt;
    }
    const auto ho = dgradOutputExtent(
        problem.hi, problem.y, problem.strideH, problem.padH, problem.dilationH);
    const auto wo = dgradOutputExtent(
        problem.wi, problem.x, problem.strideW, problem.padW, problem.dilationW);
    if(!ho || !wo)
    {
        return std::nullopt;
    }
    const auto cpg = problem.c / problem.groups;
    const auto kpg = problem.k / problem.groups;
    const auto dyBytes = boundedProduct({problem.n, *ho, *wo, problem.k, 2});
    const auto xBytes = boundedProduct({problem.n, problem.hi, problem.wi, problem.c, 2});
    const auto dwF32Bytes = boundedProduct({problem.k, problem.y, problem.x, cpg, 4});
    if(!dyBytes || !xBytes || !dwF32Bytes)
    {
        return std::nullopt;
    }
    return Geometry{*ho, *wo, cpg, kpg, {*dyBytes, *xBytes, *dwF32Bytes / 2}, *dwF32Bytes / 4};
}

// ---------------------------------------------------------------------------------------
// The split-K cost model (rocke/platform/python/rocke/helpers/split_k_planner.py)
// ---------------------------------------------------------------------------------------

/// gfx950's fitted SplitKCostModel (_COST_MODELS["gfx950"]), in units of one K iteration's
/// load latency.
struct SplitKCostModel
{
    int64_t numCus = 256;
    int64_t maxWavesPerCu = 32;
    double flopsPerLatency = 843448.3;
    double loadBytesPerLatency = 19716.8;
    double ctaCost = 0.012296;
    double atomicBf16BytesPerLatency = 328579.7;
    double atomicFp16BytesPerLatency = 328579.7;
    double atomicFp32BytesPerLatency = 1344520.3;
    double fillBytesPerLatency = 43105.9;
    int64_t maxSplitK = 1024;
    double ladderMargin = 0.05;
    int64_t numXcds = 8;
};

constexpr int64_t MAX_GRID_DIM_Z = 65535;
/// conv_split_k.ATOMIC_16BIT_MAX_SPLIT_K: the packed 16-bit atomic's accuracy cap.
constexpr int64_t ATOMIC_16BIT_MAX_SPLIT_K = 128;
/// conv_split_k.MAX_WGRAD_WS_BYTES: the two-stage scratch is addressed through an i32
/// ws_bytes.
constexpr int64_t MAX_WGRAD_WS_BYTES = (int64_t{1} << 31) - 4;

/// SplitKProblem: the GEMM a degree is chosen for. zBatch is always batch here (wgrad
/// rides the group and the slice on grid z together).
struct SplitKProblem
{
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t batch;
    int64_t operandBytes;
    int64_t outputBytes;
};

/// SplitKKernel: block tile and waves per block.
struct SplitKKernel
{
    int64_t tileM;
    int64_t tileN;
    int64_t tileK;
    int64_t wavesPerBlock;
};

inline int64_t wavesBoundBlocksPerCu(const SplitKKernel& kernel, const SplitKCostModel& cost)
{
    return std::max<int64_t>(1, cost.maxWavesPerCu / kernel.wavesPerBlock);
}

/// SplitKReduction: how the slices fold. A one-stage instance has the packed 16-bit atomic
/// into dW (capped at ATOMIC_16BIT_MAX_SPLIT_K); a two-stage one f32 atomics into a
/// workspace of workspaceCopies dW-sized slabs, cleared and folded into dW afterwards.
struct SplitKReduction
{
    double atomicBytesPerLatency;
    std::optional<int64_t> maxSplitK;
    bool workspace = false;
    int64_t workspaceCopies = 1;
};

/// predict_split_k_cost. The arithmetic follows the Python expression for expression --
/// integer products stay exact integers until they meet a float, as Python's do -- so a
/// tie or a ladder-margin comparison resolves the same way in both.
inline double predictSplitKCost(const SplitKProblem& problem,
                                const SplitKKernel& kernel,
                                int64_t splitK,
                                const SplitKReduction& reduction,
                                const SplitKCostModel& cost)
{
    const auto tm = kernel.tileM;
    const auto tn = kernel.tileN;
    const auto tk = kernel.tileK;
    const auto tiles = ceilDiv(problem.m, tm) * ceilDiv(problem.n, tn) * problem.batch;
    const auto kIters = ceilDiv(problem.k, tk);
    const auto ctas = tiles * splitK;
    const auto iters = ceilDiv(kIters, splitK);
    const auto slots = cost.numCus * wavesBoundBlocksPerCu(kernel, cost);

    const double flopIter
        = 2.0 * static_cast<double>(tm) * static_cast<double>(tn) * static_cast<double>(tk);
    const auto mEff = std::min(tm, problem.m);
    const auto nEff = std::min(tn, problem.n);
    const auto byteIter = (mEff + nEff) * tk * problem.operandBytes;

    const auto tauFloat = [&](double blocksPerCu) {
        return 1.0
               + std::max(blocksPerCu * flopIter / cost.flopsPerLatency,
                          blocksPerCu * static_cast<double>(byteIter) / cost.loadBytesPerLatency);
    };
    const auto tauInt = [&](int64_t blocksPerCu) {
        return 1.0
               + std::max(static_cast<double>(blocksPerCu) * flopIter / cost.flopsPerLatency,
                          static_cast<double>(blocksPerCu * byteIter) / cost.loadBytesPerLatency);
    };

    const auto fullRounds = ctas / slots;
    const auto rest = ctas % slots;
    double t = static_cast<double>(fullRounds * iters)
               * tauFloat(static_cast<double>(slots) / static_cast<double>(cost.numCus));
    if(rest != 0)
    {
        t += static_cast<double>(iters) * tauInt(ceilDiv(rest, cost.numCus));
    }
    t += static_cast<double>(ctas) * cost.ctaCost;
    if(splitK > 1)
    {
        t += static_cast<double>(ctas * mEff * nEff) * 4.0 / reduction.atomicBytesPerLatency;
        const auto outBytes = problem.batch * problem.m * problem.n * problem.outputBytes;
        if(reduction.workspace)
        {
            // Clear, then a fold that reads the workspace and writes the output.
            auto wsBytes = static_cast<double>(problem.batch * problem.m * problem.n) * 4.0;
            wsBytes *= static_cast<double>(reduction.workspaceCopies);
            t += (2.0 * wsBytes + static_cast<double>(outBytes)) / cost.fillBytesPerLatency;
        }
        else
        {
            t += static_cast<double>(outBytes) / cost.fillBytesPerLatency;
        }
    }
    return t;
}

inline bool isPowerOfTwo(int64_t value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

/// _degrees: the split degrees in [2, cap] worth scoring, ascending.
inline std::vector<int64_t> splitKDegrees(const SplitKProblem& problem,
                                          const SplitKKernel& kernel,
                                          const SplitKCostModel& cost,
                                          int64_t cap)
{
    const auto kIters = ceilDiv(problem.k, kernel.tileK);
    const auto groupTiles = ceilDiv(problem.m, kernel.tileM) * ceilDiv(problem.n, kernel.tileN);
    const auto tiles = groupTiles * problem.batch;
    const auto cus = cost.numCus;
    const std::array<int64_t, 2> rounds{cus, cus * wavesBoundBlocksPerCu(kernel, cost)};
    int64_t step = 1;
    if(problem.batch > 1)
    {
        step = cost.numXcds / std::gcd(groupTiles, cost.numXcds);
    }
    std::vector<int64_t> degrees;
    for(int64_t s = 2; s <= cap; ++s)
    {
        if(isPowerOfTwo(s))
        {
            degrees.push_back(s);
        }
        else if(step > 1)
        {
            if(s % step == 0)
            {
                degrees.push_back(s);
            }
        }
        else if(cus % s == 0 || s % cus == 0 || ceilDiv(kIters, s) < ceilDiv(kIters, s - 1)
                || std::any_of(rounds.begin(), rounds.end(), [&](int64_t u) {
                       return ceilDiv(tiles * (s + 1), u) > ceilDiv(tiles * s, u);
                   }))
        {
            degrees.push_back(s);
        }
    }
    return degrees;
}

/// select_split_k_plan with one reduction: the cheapest degree (1 only when
/// @p allowUnsplit), ties to the smaller one, and a degree off the power-of-two ladder only
/// when it beats the best ladder degree by ladderMargin. std::nullopt when no degree is
/// admissible (Python raises).
inline std::optional<int64_t> selectSplitK(const SplitKProblem& problem,
                                           const SplitKKernel& kernel,
                                           const SplitKReduction& reduction,
                                           bool allowUnsplit,
                                           const SplitKCostModel& cost)
{
    const auto kIters = ceilDiv(problem.k, kernel.tileK);
    const auto zCap = MAX_GRID_DIM_Z / problem.batch;
    const auto cap = std::min({kIters, zCap, cost.maxSplitK});
    std::vector<std::pair<double, int64_t>> candidates;
    if(allowUnsplit)
    {
        candidates.emplace_back(predictSplitKCost(problem, kernel, 1, reduction, cost), 1);
    }
    const auto reductionCap = reduction.maxSplitK ? std::min(cap, *reduction.maxSplitK) : cap;
    for(const auto s : splitKDegrees(problem, kernel, cost, reductionCap))
    {
        candidates.emplace_back(predictSplitKCost(problem, kernel, s, reduction, cost), s);
    }
    if(candidates.empty())
    {
        return std::nullopt;
    }
    const auto best = *std::min_element(candidates.begin(), candidates.end());
    if(isPowerOfTwo(best.second))
    {
        return best.second;
    }
    std::optional<std::pair<double, int64_t>> ladder;
    for(const auto& candidate : candidates)
    {
        if(isPowerOfTwo(candidate.second) && (!ladder || candidate < *ladder))
        {
            ladder = candidate;
        }
    }
    if(ladder && ladder->first <= best.first * (1.0 + cost.ladderMargin))
    {
        return ladder->second;
    }
    return best.second;
}

// ---------------------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------------------

/// The kernel_family metadata value (KERNEL_FAMILY_* in convolution_backward_filter.py).
enum class KernelFamily : int64_t
{
    IMPLICIT_GEMM = 0,
    DIRECT = 1,
    /// direct_wgrad_fold and direct_wgrad_ws_reduce: packaged so the folded pipelines
    /// and the two-stage implicit GEMM can load them, never admitted by the kernel matcher.
    HELPER = 2,
};

inline std::optional<KernelFamily> parseKernelFamily(int64_t value)
{
    switch(value)
    {
    case static_cast<int64_t>(KernelFamily::IMPLICIT_GEMM):
        return KernelFamily::IMPLICIT_GEMM;
    case static_cast<int64_t>(KernelFamily::DIRECT):
        return KernelFamily::DIRECT;
    case static_cast<int64_t>(KernelFamily::HELPER):
        return KernelFamily::HELPER;
    default:
        return std::nullopt;
    }
}

/// The direct_variant metadata value; a wgrad direct identity's algorithm is its name.
enum class DirectVariant
{
    NONE, ///< "none": implicit GEMM.
    WGRAD, ///< "direct_wgrad": DirectConvWgradSpec (runtime channel counts).
    TILED, ///< "direct_wgrad_tiled": DirectWgradTiledSpec.
    DW, ///< "direct_wgrad_dw": DwWgradLdsSpec (folded).
    DW_COLS, ///< "direct_wgrad_dw_cols": DwWgradColsSpec (folded).
    DW_MFMA, ///< "direct_wgrad_dw_mfma": DwWgradMfmaSpec (folded).
    GATHER, ///< "direct_wgrad_gather": WgradGatherSpec (folded, per-channel fold).
    GROUPED_MFMA, ///< "direct_wgrad_grouped_mfma": GroupedWgradMfmaSpec (folded, per-channel).
    FOLD, ///< "direct_wgrad_fold": DwWgradFoldSpec (helper).
    WS_REDUCE, ///< "direct_wgrad_ws_reduce": WgradReduceSpec, the two-stage Stage 2 (helper).
};

inline std::optional<DirectVariant> parseDirectVariant(std::string_view value)
{
    static constexpr std::array<std::pair<std::string_view, DirectVariant>, 10> NAMES{{
        {"none", DirectVariant::NONE},
        {"direct_wgrad", DirectVariant::WGRAD},
        {"direct_wgrad_tiled", DirectVariant::TILED},
        {"direct_wgrad_dw", DirectVariant::DW},
        {"direct_wgrad_dw_cols", DirectVariant::DW_COLS},
        {"direct_wgrad_dw_mfma", DirectVariant::DW_MFMA},
        {"direct_wgrad_gather", DirectVariant::GATHER},
        {"direct_wgrad_grouped_mfma", DirectVariant::GROUPED_MFMA},
        {"direct_wgrad_fold", DirectVariant::FOLD},
        {"direct_wgrad_ws_reduce", DirectVariant::WS_REDUCE},
    }};
    for(const auto& [name, variant] : NAMES)
    {
        if(name == value)
        {
            return variant;
        }
    }
    return std::nullopt;
}

/// _HELPER_VARIANTS: the kernels kernel_family 2 packages.
inline bool isHelperVariant(DirectVariant variant)
{
    return variant == DirectVariant::FOLD || variant == DirectVariant::WS_REDUCE;
}

/// _FOLDED_VARIANTS: the compute pass stores partial slots the fold helper sums into dW.
inline bool isFoldedVariant(DirectVariant variant)
{
    return variant == DirectVariant::DW || variant == DirectVariant::DW_COLS
           || variant == DirectVariant::DW_MFMA || variant == DirectVariant::GATHER
           || variant == DirectVariant::GROUPED_MFMA;
}

/// _PER_CHANNEL_FOLD_VARIANTS: the fold runs over KH * KW * cpg taps per output channel.
inline bool isPerChannelFoldVariant(DirectVariant variant)
{
    return variant == DirectVariant::GATHER || variant == DirectVariant::GROUPED_MFMA;
}

/// The epilogue an implicit-GEMM instance stores dW with; only cshuffle emits the packed
/// 16-bit split-K atomic.
enum class Epilogue
{
    DEFAULT,
    CSHUFFLE,
};

/// One implicit-GEMM instance: the build-time fields of a rocKE wgrad KernelIdentity that
/// bound which problems its binary computes and how it launches (Gfx950ConvWgradAotSpec).
/// A wgrad implicit-GEMM binary bakes no filter, stride, padding or channel count.
struct ImplicitInstance
{
    int64_t tileM;
    int64_t tileN;
    int64_t tileK;
    int64_t warpM;
    int64_t warpN;
    int64_t vectorA = 0; ///< dY load width along kpg; 0 is the builder's choice.
    int64_t vectorB = 0; ///< X load width along cpg.
    int64_t vectorC = 0; ///< dW store width along cpg.
    Epilogue epilogue = Epilogue::CSHUFFLE;
    int64_t groupMerge = 1;
    /// The catalog's split_k placeholder (-1, 1 or > 1). Never the launched degree, but a
    /// one-stage binary built with one > 1 carries the packed-atomic epilogue, whose
    /// problem rule identity_supports_problem applies.
    int64_t splitK = 1;
    bool twoStage = false;
    bool grouped = true;
    bool asyncDma = false;
    /// Scratch slabs per group of a two-stage binary; 0 is the builder default.
    int64_t wsReplicas = 0;
};

/// WgradConvSpec's default ws_replicas (_DEFAULT_WS_REPLICAS).
constexpr int64_t DEFAULT_WS_REPLICAS = 8;

inline int64_t effectiveWsReplicas(const ImplicitInstance& instance)
{
    return instance.wsReplicas > 0 ? instance.wsReplicas : DEFAULT_WS_REPLICAS;
}

/// What a direct binary compiles in. 0 cpg / kpg: a runtime channel count.
struct DirectCapabilities
{
    int64_t filterH;
    int64_t filterW;
    int64_t strideH;
    int64_t strideW;
    int64_t padH;
    int64_t padW;
    int64_t dilationH;
    int64_t dilationW;
    int64_t cpg;
    int64_t kpg;
};

/// The knobs of a direct or helper instance as the descriptor states them: 0 where the
/// identity omitted one, which rocKE then gives its spec default (knobOr below).
struct DirectKnobs
{
    int64_t hoPerBlock = 0;
    int64_t mfmaK = 0;
    int64_t wavesC = 0;
    int64_t wavesK = 0;
    int64_t wavesQ = 0;
    int64_t prefetch = 0;
    int64_t rowUnroll = 0;
    int64_t rowsPerSeg = 0;
    int64_t targetBlocks = 0;
    int64_t unfoldN = 0;
    int64_t waveC16 = 0;
    int64_t waveK16 = 0;
    int64_t lanesC = 0;
    int64_t lanesW = 0;
    int64_t channels = 0;
    int64_t strip = 0;
    int64_t threads = 0;
    int64_t qTiles = 0;
    int64_t rowsPerChunk = 0;
    int64_t waveQ4 = 0;
    int64_t taps = 0;
};

/// The knob fields in KMD order (_KNOB_DEFAULTS in convolution_backward_filter.py).
constexpr std::array<std::pair<std::string_view, int64_t DirectKnobs::*>, 21> DIRECT_KNOB_FIELDS{{
    {"ho_per_block", &DirectKnobs::hoPerBlock},
    {"mfma_k", &DirectKnobs::mfmaK},
    {"waves_c", &DirectKnobs::wavesC},
    {"waves_k", &DirectKnobs::wavesK},
    {"waves_q", &DirectKnobs::wavesQ},
    {"prefetch", &DirectKnobs::prefetch},
    {"row_unroll", &DirectKnobs::rowUnroll},
    {"rows_per_seg", &DirectKnobs::rowsPerSeg},
    {"target_blocks", &DirectKnobs::targetBlocks},
    {"unfold_n", &DirectKnobs::unfoldN},
    {"wave_c16", &DirectKnobs::waveC16},
    {"wave_k16", &DirectKnobs::waveK16},
    {"lanes_c", &DirectKnobs::lanesC},
    {"lanes_w", &DirectKnobs::lanesW},
    {"channels", &DirectKnobs::channels},
    {"strip", &DirectKnobs::strip},
    {"threads", &DirectKnobs::threads},
    {"q_tiles", &DirectKnobs::qTiles},
    {"rows_per_chunk", &DirectKnobs::rowsPerChunk},
    {"wave_q4", &DirectKnobs::waveQ4},
    {"taps", &DirectKnobs::taps},
}};

/// Whether @p variant states @p knob (AOT_WGRAD_DIRECT_KNOBS); every other knob holds 0.
inline bool directVariantUses(DirectVariant variant, std::string_view knob)
{
    const auto among = [&](std::initializer_list<std::string_view> knobs) {
        return std::find(knobs.begin(), knobs.end(), knob) != knobs.end();
    };
    switch(variant)
    {
    case DirectVariant::WGRAD:
        return among({"ho_per_block", "mfma_k", "waves_c", "waves_k", "waves_q"});
    case DirectVariant::TILED:
        return among({"prefetch",
                      "row_unroll",
                      "rows_per_seg",
                      "target_blocks",
                      "unfold_n",
                      "wave_c16",
                      "wave_k16",
                      "waves_c",
                      "waves_k"});
    case DirectVariant::DW:
        return among({"lanes_c", "lanes_w", "prefetch", "rows_per_seg", "target_blocks"});
    case DirectVariant::DW_COLS:
        return among({"channels", "strip", "target_blocks", "threads"});
    case DirectVariant::DW_MFMA:
        return among({"prefetch", "q_tiles", "rows_per_chunk", "target_blocks", "unfold_n"});
    case DirectVariant::GATHER:
        return among({"prefetch", "rows_per_seg", "target_blocks", "wave_k16", "waves_k"});
    case DirectVariant::GROUPED_MFMA:
        return among({"prefetch", "rows_per_chunk", "target_blocks", "wave_q4", "waves_q"});
    case DirectVariant::FOLD:
        return knob == "taps";
    case DirectVariant::WS_REDUCE:
    case DirectVariant::NONE:
    default:
        return false;
    }
}

/// One direct or helper instance (the direct fields of Gfx950ConvWgradAotSpec).
/// wsReplicas is the Stage 2 fold's slab count, which it records in the implicit-GEMM
/// ws_replicas field; 0 for every other variant.
struct DirectInstance
{
    DirectVariant variant;
    DirectCapabilities compiled{};
    DirectKnobs knobs{};
    int64_t wsReplicas = 0;
};

/// A knob's value, or rocKE's spec default where the instance left it at 0 (every wgrad
/// direct knob's 0 is "not stated", so direct_knobs() omits it and the spec default rules).
inline int64_t knobOr(int64_t value, int64_t fallback)
{
    return value != 0 ? value : fallback;
}

/// One instance of either family, with the dtype it was built for.
struct AotInstance
{
    KernelFamily family;
    ElementType dtype;
    std::optional<ImplicitInstance> implicit;
    std::optional<DirectInstance> direct;
};

// ---------------------------------------------------------------------------------------
// Reading an instance from its metadata (supports_gfx950_conv_wgrad_aot's field rules)
// ---------------------------------------------------------------------------------------

/// The implicit-GEMM tuning fields; a direct or helper instance holds 0 in each.
constexpr std::array<std::string_view, 8> GEMM_TUNING_FIELDS{
    "tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_tile_m", "warp_tile_n", "warp_tile_k"};
/// The implicit-GEMM 0/1 flags (lds_k_outer .. two_stage).
constexpr std::array<std::string_view, 6> GEMM_FLAG_FIELDS{
    "lds_k_outer", "grouped", "prefetch_addr", "async_dma", "unroll_k", "two_stage"};
/// The capability fields a direct instance states and an implicit-GEMM one leaves at 0.
constexpr std::array<std::string_view, 10> DIRECT_CAPABILITY_FIELDS{"filter_h",
                                                                    "filter_w",
                                                                    "stride_h",
                                                                    "stride_w",
                                                                    "pad_h",
                                                                    "pad_w",
                                                                    "dilation_h",
                                                                    "dilation_w",
                                                                    "cpg",
                                                                    "kpg"};

/// gfx950_conv_wgrad_aot_plan's refusal of a helper, verbatim.
constexpr std::string_view HELPER_REFUSAL = "a helper is launched only by its pipeline";

/// The instance a kernel's metadata describes (supports_gfx950_conv_wgrad_aot's field
/// rules), when it is a complete instance of one family. @p metadata answers
/// intField(name) -> std::optional<int64_t> and stringField(name) -> const std::string*.
/// kernel_family is read first, so no family's checks run on another's placeholders; a
/// missing or malformed field refuses the kernel. rocKE's own spec validators are not
/// re-run here: the plan runs them on the real problem.
template <typename Metadata>
std::optional<AotInstance> parseAotInstance(const Metadata& metadata)
{
    const auto equalsString = [&](std::string_view field, std::string_view expected) {
        const auto* value = metadata.stringField(field);
        return value != nullptr && *value == expected;
    };
    const auto equalsInt = [&](std::string_view field, int64_t expected) {
        return metadata.intField(field) == std::optional<int64_t>(expected);
    };
    const auto isZero = [&](std::string_view field) { return equalsInt(field, 0); };
    const auto intIn = [&](std::string_view field, int64_t low, int64_t high) {
        const auto value = metadata.intField(field);
        return value && *value >= low && *value <= high ? value : std::nullopt;
    };
    const auto* dtypeName = metadata.stringField("dtype");
    const auto dtype = dtypeName == nullptr ? std::nullopt : parseElementType(*dtypeName);
    if(!dtype || !equalsString("layout", "NHWC") || !equalsInt("wave_size", WAVE_SIZE))
    {
        return std::nullopt;
    }
    const auto familyValue = metadata.intField("kernel_family");
    const auto family = familyValue ? parseKernelFamily(*familyValue) : std::nullopt;
    const auto* variantName = metadata.stringField("direct_variant");
    const auto variant = variantName == nullptr ? std::nullopt : parseDirectVariant(*variantName);
    if(!family || !variant)
    {
        return std::nullopt;
    }

    if(*family == KernelFamily::IMPLICIT_GEMM)
    {
        const auto* pipeline = metadata.stringField("pipeline");
        const auto* epilogue = metadata.stringField("epilogue");
        if(*variant != DirectVariant::NONE || pipeline == nullptr || epilogue == nullptr
           || metadata.stringField("lds_layout") == nullptr
           || (*pipeline != "mem" && *pipeline != "compv3" && *pipeline != "compv4")
           || (*epilogue != "default" && *epilogue != "cshuffle")
           || !std::all_of(DIRECT_CAPABILITY_FIELDS.begin(), DIRECT_CAPABILITY_FIELDS.end(), isZero)
           || !std::all_of(DIRECT_KNOB_FIELDS.begin(),
                           DIRECT_KNOB_FIELDS.end(),
                           [&](const auto& f) { return isZero(f.first); }))
        {
            return std::nullopt;
        }
        std::array<int64_t, GEMM_TUNING_FIELDS.size()> tuning{};
        for(size_t i = 0; i < GEMM_TUNING_FIELDS.size(); ++i)
        {
            const auto value = intIn(GEMM_TUNING_FIELDS[i], 1, I32_MAX);
            if(!value)
            {
                return std::nullopt;
            }
            tuning[i] = *value;
        }
        std::array<int64_t, 3> vectors{};
        constexpr std::array<std::string_view, 3> VECTOR_FIELDS{
            "vector_size_a", "vector_size_b", "vector_size_c"};
        for(size_t i = 0; i < vectors.size(); ++i)
        {
            const auto value = metadata.intField(VECTOR_FIELDS[i]);
            // 0: the builder picks the width (the identity records None as 0).
            if(!value || (*value != 0 && *value != 1 && *value != 2 && *value != 4 && *value != 8))
            {
                return std::nullopt;
            }
            vectors[i] = *value;
        }
        std::array<bool, GEMM_FLAG_FIELDS.size()> flags{};
        for(size_t i = 0; i < GEMM_FLAG_FIELDS.size(); ++i)
        {
            const auto value = intIn(GEMM_FLAG_FIELDS[i], 0, 1);
            if(!value)
            {
                return std::nullopt;
            }
            flags[i] = *value == 1;
        }
        const auto splitK = intIn("split_k", -1, I32_MAX);
        const auto groupMerge = intIn("group_merge", 1, I32_MAX);
        const auto wsReplicas = intIn("ws_replicas", 0, I32_MAX);
        const auto ldsKPad = intIn("lds_k_pad", 0, I32_MAX);
        const bool twoStage = flags[5];
        if(!splitK || *splitK == 0 || !groupMerge || !wsReplicas || !ldsKPad
           || (twoStage && *splitK == 1))
        {
            return std::nullopt;
        }
        ImplicitInstance instance{tuning[0], tuning[1], tuning[2], tuning[3], tuning[4]};
        instance.vectorA = vectors[0];
        instance.vectorB = vectors[1];
        instance.vectorC = vectors[2];
        instance.epilogue = *epilogue == "cshuffle" ? Epilogue::CSHUFFLE : Epilogue::DEFAULT;
        instance.groupMerge = *groupMerge;
        instance.splitK = *splitK;
        instance.twoStage = twoStage;
        instance.wsReplicas = *wsReplicas;
        instance.grouped = flags[1];
        instance.asyncDma = flags[3];
        if(instance.warpM > MAX_BLOCK_THREADS / WAVE_SIZE
           || instance.warpN > MAX_BLOCK_THREADS / (WAVE_SIZE * instance.warpM))
        {
            return std::nullopt;
        }
        return AotInstance{*family, *dtype, instance, std::nullopt};
    }

    const bool helper = *family == KernelFamily::HELPER;
    // The Stage 2 fold records its slab count in ws_replicas; every other direct or
    // helper instance holds the placeholder 0 there.
    const auto wsReplicas
        = *variant == DirectVariant::WS_REDUCE
              ? intIn("ws_replicas", 1, I32_MAX)
              : (isZero("ws_replicas") ? std::optional<int64_t>(0) : std::nullopt);
    if(*variant == DirectVariant::NONE || helper != isHelperVariant(*variant)
       || !equalsString("pipeline", "direct") || !equalsString("epilogue", "direct")
       || !equalsString("lds_layout", "default")
       || !std::all_of(GEMM_TUNING_FIELDS.begin(), GEMM_TUNING_FIELDS.end(), isZero)
       || !std::all_of(GEMM_FLAG_FIELDS.begin(), GEMM_FLAG_FIELDS.end(), isZero)
       || !isZero("vector_size_a") || !isZero("vector_size_b") || !isZero("vector_size_c")
       || !equalsInt("group_merge", 1) || !wsReplicas || !isZero("lds_k_pad")
       || !equalsInt("split_k", 1))
    {
        return std::nullopt;
    }
    std::array<int64_t, DIRECT_CAPABILITY_FIELDS.size()> compiled{};
    for(size_t i = 0; i < DIRECT_CAPABILITY_FIELDS.size(); ++i)
    {
        const auto value = intIn(DIRECT_CAPABILITY_FIELDS[i], 0, I32_MAX);
        if(!value)
        {
            return std::nullopt;
        }
        compiled[i] = *value;
    }
    const DirectCapabilities caps{compiled[0],
                                  compiled[1],
                                  compiled[2],
                                  compiled[3],
                                  compiled[4],
                                  compiled[5],
                                  compiled[6],
                                  compiled[7],
                                  compiled[8],
                                  compiled[9]};
    if(helper)
    {
        if(!std::all_of(compiled.begin(), compiled.end(), [](int64_t v) { return v == 0; }))
        {
            return std::nullopt;
        }
    }
    else
    {
        if(caps.filterH < 1 || caps.filterH != caps.filterW || caps.dilationH != 1
           || caps.dilationW != 1 || caps.strideH < 1 || caps.strideH != caps.strideW
           || caps.padH != caps.padW)
        {
            return std::nullopt;
        }
        // cpg == 0 (kpg == 0) is a runtime channel count: direct_wgrad and _tiled take
        // both as kernargs, _gather bakes cpg only, every other family bakes both.
        switch(*variant)
        {
        case DirectVariant::WGRAD:
        case DirectVariant::TILED:
            if(caps.cpg != 0 || caps.kpg != 0)
            {
                return std::nullopt;
            }
            break;
        case DirectVariant::GATHER:
            if(caps.cpg == 0 || caps.kpg != 0)
            {
                return std::nullopt;
            }
            break;
        default:
            if(caps.cpg == 0 || caps.kpg == 0)
            {
                return std::nullopt;
            }
            break;
        }
    }
    DirectKnobs knobs{};
    for(const auto& [field, member] : DIRECT_KNOB_FIELDS)
    {
        const auto value = metadata.intField(field);
        if(!value
           || (directVariantUses(*variant, field) ? (*value < 0 || *value > I32_MAX) : *value != 0))
        {
            return std::nullopt;
        }
        knobs.*member = *value;
    }
    return AotInstance{
        *family, *dtype, std::nullopt, DirectInstance{*variant, caps, knobs, *wsReplicas}};
}

// ---------------------------------------------------------------------------------------
// The launch plan (gfx950_conv_wgrad_aot_plan)
// ---------------------------------------------------------------------------------------

/// The kernel a step launches (PLAN_KERNEL_CODES).
enum class PlanKernel : int64_t
{
    SELF = 0,
    FOLD = 1,
    WS_REDUCE = 2,
};

/// The ABI a step launches with (PLAN_ABI_CODES).
enum class Abi : int64_t
{
    /// (dY*, X*, dW*, dY_bytes, X_bytes, dW_bytes) then 42 i32 (IMPLICIT_KERNARG_NAMES).
    IMPLICIT = 0,
    /// (A*, B*, D*, A_bytes, B_bytes, D_bytes) then 14 i32: dY, X and an fp32 D.
    DIRECT_WGRAD = 1,
    /// (A*, D*, A_bytes, D_bytes, parts, channels): the fp32 partial slots, then dW.
    FOLD = 2,
    /// IMPLICIT with the scratch pair before ks / ks_count: the values are the 43 i32
    /// (IMPLICIT_TWO_STAGE_KERNARG_NAMES), the pointer ws_ptr the WS workspace.
    IMPLICIT_TWO_STAGE = 3,
    /// (ws_ptr*, dw_ptr*, wg_M, wg_N, ws_bytes, dw_bytes, groups): the Stage 2 fold.
    WS_REDUCE = 4,
};

/// The buffers a step binds (PLAN_BUFFER_CODES).
enum class Buffer : int64_t
{
    NONE = -1,
    DY = 0,
    X = 1,
    DW = 2,
    WS = 3,
};

/// One launch of a plan (Gfx950ConvWgradStep): values are the by-value i32 arguments after
/// the operands and their byte sizes, in ABI order; the workgroup is (block, 1, 1).
struct Step
{
    PlanKernel kernel;
    Abi abi;
    std::array<int64_t, 3> grid;
    int64_t block;
    std::array<Buffer, 3> buffers;
    std::vector<int32_t> values;
};

struct Workspace
{
    Buffer name;
    int64_t offset;
    int64_t bytes;
};

/// Everything the engine does to run one instance on one problem (Gfx950ConvWgradPlan):
/// zero dW when zeroDw and the WS workspace when zeroWs, then launch the steps. splitK is
/// the resolved split-K degree, a launch value of the implicit-GEMM step (ks_count) and
/// never part of a binary's identity; 1 wherever a launch does not split.
struct Plan
{
    Problem launched;
    Geometry geometry;
    std::vector<Step> steps;
    std::vector<Workspace> workspaces;
    bool zeroDw = false;
    int64_t splitK = 1;
    bool zeroWs = false;

    int64_t workspaceBytes() const
    {
        int64_t end = 0;
        for(const auto& region : workspaces)
        {
            end = std::max(end, region.offset + region.bytes);
        }
        return end;
    }

    std::optional<Workspace> workspace(Buffer name) const
    {
        for(const auto& region : workspaces)
        {
            if(region.name == name)
            {
                return region;
            }
        }
        return std::nullopt;
    }
};

/// Gfx950ConvWgradPlan.serialize: [zero_dw, zero_ws, split_k, workspace_bytes, #workspaces,
/// (name, offset, bytes)..., #steps, then per step kernel, abi, grid x/y/z, block, the
/// three buffer codes, #values, values...].
inline std::vector<int64_t> serializePlan(const Plan& plan)
{
    std::vector<int64_t> out{plan.zeroDw ? 1 : 0,
                             plan.zeroWs ? 1 : 0,
                             plan.splitK,
                             plan.workspaceBytes(),
                             static_cast<int64_t>(plan.workspaces.size())};
    for(const auto& region : plan.workspaces)
    {
        out.insert(out.end(), {static_cast<int64_t>(region.name), region.offset, region.bytes});
    }
    out.push_back(static_cast<int64_t>(plan.steps.size()));
    for(const auto& step : plan.steps)
    {
        out.insert(out.end(),
                   {static_cast<int64_t>(step.kernel),
                    static_cast<int64_t>(step.abi),
                    step.grid[0],
                    step.grid[1],
                    step.grid[2],
                    step.block});
        for(const auto buffer : step.buffers)
        {
            out.push_back(static_cast<int64_t>(buffer));
        }
        out.push_back(static_cast<int64_t>(step.values.size()));
        out.insert(out.end(), step.values.begin(), step.values.end());
    }
    return out;
}

/// Gfx950ConvWgradPlan.digest: FNV-1a 64 over serializePlan, each value as 8
/// little-endian bytes.
inline uint64_t planDigest(const Plan& plan)
{
    uint64_t hash = 0xCBF29CE484222325ULL;
    for(const auto value : serializePlan(plan))
    {
        const auto bits = static_cast<uint64_t>(value);
        for(int byte = 0; byte < 8; ++byte)
        {
            hash ^= (bits >> (8 * byte)) & 0xFFU;
            hash *= 0x100000001B3ULL;
        }
    }
    return hash;
}

/// _grid_reason: a grid of @p block-thread workgroups launches when every axis is
/// positive, x is at most I32_MAX and y and z at most 65535. (The wgrad adapter bounds x
/// by I32_MAX alone, without dgrad's 2^32 work-item cap; it is followed exactly.)
inline bool gridFits(const std::array<int64_t, 3>& grid, int64_t block)
{
    return block >= 1 && block <= MAX_BLOCK_THREADS && grid[0] >= 1 && grid[1] >= 1 && grid[2] >= 1
           && grid[0] <= I32_MAX && grid[1] <= MAX_GRID_DIM && grid[2] <= MAX_GRID_DIM;
}

/// The 42 by-value i32 of the wgrad implicit-GEMM ABI after (dY*, X*, dW*, dY_bytes,
/// X_bytes, dW_bytes): _wgrad_arg_names in rocke/library/kernels/common/conv_abi.py
/// (one-stage, 2-D). The kernel reads them positionally; the names are compared against
/// the packaged symbol's recorded arguments.
constexpr std::array<std::string_view, 42> IMPLICIT_KERNARG_NAMES{"p_N",
                                                                  "p_Hi",
                                                                  "p_Wi",
                                                                  "p_C",
                                                                  "p_K",
                                                                  "p_Y",
                                                                  "p_X",
                                                                  "p_sH",
                                                                  "p_sW",
                                                                  "p_pH",
                                                                  "p_pW",
                                                                  "p_dH",
                                                                  "p_dW",
                                                                  "p_groups",
                                                                  "p_Ho",
                                                                  "p_Wo",
                                                                  "p_cpg",
                                                                  "p_kpg",
                                                                  "p_wg_M",
                                                                  "p_wg_N",
                                                                  "p_wg_K",
                                                                  "p_dY_stride_n",
                                                                  "p_dY_stride_ho",
                                                                  "p_dY_stride_wo",
                                                                  "p_X_stride_n",
                                                                  "p_X_stride_hi",
                                                                  "p_X_stride_wi",
                                                                  "p_dW_stride_k",
                                                                  "p_dW_stride_y",
                                                                  "p_dW_stride_x",
                                                                  "p_magic_k_Ho_mult",
                                                                  "p_magic_k_Ho_shift",
                                                                  "p_magic_k_Wo_mult",
                                                                  "p_magic_k_Wo_shift",
                                                                  "p_magic_n_X_mult",
                                                                  "p_magic_n_X_shift",
                                                                  "p_magic_n_cpg_mult",
                                                                  "p_magic_n_cpg_shift",
                                                                  "p_num_pid_m",
                                                                  "p_num_pid_n",
                                                                  "ks",
                                                                  "ks_count"};

/// The 14 values after (A*, B*, D*, A_bytes, B_bytes, D_bytes) of the direct wgrad ABI
/// (conv_direct_arg_names(direction="wgrad")).
constexpr std::array<std::string_view, 14> DIRECT_WGRAD_KERNARG_NAMES{"p_N",
                                                                      "p_Hi",
                                                                      "p_Wi",
                                                                      "p_Ho",
                                                                      "p_Wo",
                                                                      "p_groups",
                                                                      "p_total_c",
                                                                      "p_total_k",
                                                                      "p_dY_stride_n",
                                                                      "p_dY_stride_ho",
                                                                      "p_dY_stride_wo",
                                                                      "p_X_stride_n",
                                                                      "p_X_stride_hi",
                                                                      "p_X_stride_wi"};

/// The fold helper's scalars after (A*, D*, A_bytes, D_bytes) (DW_WGRAD_FOLD_ARGS).
constexpr std::array<std::string_view, 2> FOLD_KERNARG_NAMES{"parts", "channels"};

/// Where the two-stage ABI puts its scratch pair: after the first 40 implicit-GEMM values
/// (through p_num_pid_n), before ks / ks_count (conv_abi._wgrad_arg_names).
constexpr size_t TWO_STAGE_WS_POSITION = 40;

/// The Stage 2 fold's scalars after (ws_ptr*, dw_ptr*) (wgrad_reduce_signature).
constexpr std::array<std::string_view, 5> WS_REDUCE_KERNARG_NAMES{
    "wg_M", "wg_N", "ws_bytes", "dw_bytes", "groups"};

/// WgradReduceSpec's default tile: one thread per element of a 4 x 64 dW patch.
constexpr int64_t WS_REDUCE_TILE_M = 4;
constexpr int64_t WS_REDUCE_TILE_N = 64;

/// The values for IMPLICIT_KERNARG_NAMES: ConvArgs.to_launch_values for a wgrad
/// ConvProblem at @p splitK. GEMM M = kpg, N = Y * X * cpg, K = N * Ho * Wo; strides are
/// packed NHWK dY, NHWC X and KYXC dW in elements; the magic pairs divide by Ho, Wo
/// (k -> n, ho, wo) and X, cpg (n -> y, x, c); num_pid_m / num_pid_n are the unmerged tile
/// counts; ks is wg_K split splitK ways rounded up to whole tile_k tiles. std::nullopt when
/// a value leaves i32.
inline std::optional<std::vector<int32_t>> implicitKernargs(const Problem& p,
                                                            const Geometry& g,
                                                            const ImplicitInstance& instance,
                                                            int64_t splitK)
{
    const auto wgN = p.y * p.x * g.cpg;
    const auto wgK = p.n * g.ho * g.wo;
    const auto magicHo = magicDivision(g.ho);
    const auto magicWo = magicDivision(g.wo);
    const auto magicX = magicDivision(p.x);
    const auto magicCpg = magicDivision(g.cpg);
    if(!magicHo || !magicWo || !magicX || !magicCpg)
    {
        return std::nullopt;
    }
    const std::array<int64_t, 42> values{p.n,
                                         p.hi,
                                         p.wi,
                                         p.c,
                                         p.k,
                                         p.y,
                                         p.x,
                                         p.strideH,
                                         p.strideW,
                                         p.padH,
                                         p.padW,
                                         p.dilationH,
                                         p.dilationW,
                                         p.groups,
                                         g.ho,
                                         g.wo,
                                         g.cpg,
                                         g.kpg,
                                         g.kpg,
                                         wgN,
                                         wgK,
                                         g.ho * g.wo * p.k,
                                         g.wo * p.k,
                                         p.k,
                                         p.hi * p.wi * p.c,
                                         p.wi * p.c,
                                         p.c,
                                         wgN,
                                         p.x * g.cpg,
                                         g.cpg,
                                         magicHo->multiplier,
                                         magicHo->shift,
                                         magicWo->multiplier,
                                         magicWo->shift,
                                         magicX->multiplier,
                                         magicX->shift,
                                         magicCpg->multiplier,
                                         magicCpg->shift,
                                         ceilDiv(g.kpg, instance.tileM),
                                         ceilDiv(wgN, instance.tileN),
                                         ceilDiv(wgK, instance.tileK * splitK) * instance.tileK,
                                         splitK};
    std::vector<int32_t> out;
    out.reserve(values.size());
    for(const auto value : values)
    {
        if(!fitsI32(value))
        {
            return std::nullopt;
        }
        out.push_back(static_cast<int32_t>(value));
    }
    return out;
}

/// The 14 values of the direct wgrad block (DIRECT_WGRAD_KERNARG_NAMES): the extents, then
/// dY's NHWK and X's NHWC strides. Bounded by the tensors' element counts, which
/// deriveGeometry keeps within i32.
inline std::vector<int32_t> directWgradValues(const Problem& p, const Geometry& g)
{
    const std::array<int64_t, 14> values{p.n,
                                         p.hi,
                                         p.wi,
                                         g.ho,
                                         g.wo,
                                         p.groups,
                                         p.c,
                                         p.k,
                                         g.ho * g.wo * p.k,
                                         g.wo * p.k,
                                         p.k,
                                         p.hi * p.wi * p.c,
                                         p.wi * p.c,
                                         p.c};
    return std::vector<int32_t>(values.begin(), values.end());
}

// ---- implicit GEMM -------------------------------------------------------------------

/// _vec: the widest of 8, 4, 2, 1 dividing @p n (WgradConvSpec.default_vector_sizes).
inline int64_t defaultVectorWidth(int64_t n)
{
    for(const auto width : {8, 4, 2})
    {
        if(n % width == 0)
        {
            return width;
        }
    }
    return 1;
}

/// wgrad_atomic_epilogue_available on a wave64 target with a 16-bit dW: the packed
/// <2 x dtype> atomic needs an even dW row (Y * X * cpg) and an even store width --
/// @p vectorC, or the builder's widest one dividing cpg when 0.
inline bool packedAtomicEpilogueAvailable(const Problem& p, const Geometry& g, int64_t vectorC)
{
    const auto wgN = p.y * p.x * g.cpg;
    const auto storeVec = vectorC != 0 ? vectorC : defaultVectorWidth(g.cpg);
    return wgN % 2 == 0 && storeVec % 2 == 0;
}

/// identity_supports_problem for a wgrad implicit-GEMM binary (llvm flavor aside: the
/// pack ships one): group_merge's depthwise rules, the operands' vector alignment, the
/// grouped code path, a one-stage split-K binary's packed atomic, async DMA's recorded
/// chunk width (never recorded by a wgrad identity, so an async binary serves nothing) and
/// the per-axis grid cap of _grid_fits.
inline bool
    implicitSupportsProblem(const Problem& p, const Geometry& g, const ImplicitInstance& instance)
{
    const auto gm = std::max<int64_t>(1, instance.groupMerge);
    std::array<int64_t, 3> extents{g.kpg, g.cpg, g.cpg};
    if(gm > 1)
    {
        // _group_merge_fits: depthwise only, gm divides groups, the merged GEMM in one tile.
        if(g.cpg != 1 || g.kpg != 1 || p.groups % gm != 0 || g.kpg * gm > instance.tileM
           || p.y * p.x * g.cpg * gm > instance.tileN)
        {
            return false;
        }
        extents = {g.kpg * gm, g.cpg * gm, g.cpg};
    }
    const std::array<int64_t, 3> widths{instance.vectorA, instance.vectorB, instance.vectorC};
    for(size_t i = 0; i < extents.size(); ++i)
    {
        if(widths[i] != 0 && extents[i] != 0 && extents[i] % widths[i] != 0)
        {
            return false;
        }
    }
    if(p.groups > 1 && !instance.grouped)
    {
        return false;
    }
    if(instance.splitK > 1 && !instance.twoStage
       && !packedAtomicEpilogueAvailable(p, g, instance.vectorC))
    {
        return false;
    }
    if(instance.asyncDma)
    {
        return false;
    }
    const auto gy = ceilDiv(g.kpg * gm, instance.tileM);
    const auto gx = ceilDiv(p.y * p.x * g.cpg * gm, instance.tileN);
    return gy <= MAX_GRID_DIM && gx <= MAX_GRID_DIM && p.groups / gm <= MAX_GRID_DIM;
}

/// wgrad_workspace_nbytes: groups * ws_replicas * wg_M * wg_N f32 -- the unmerged dW,
/// replicated; std::nullopt past MAX_WGRAD_WS_BYTES.
inline std::optional<int64_t>
    twoStageWorkspaceBytes(const Problem& p, const Geometry& g, const ImplicitInstance& instance)
{
    const auto bytes
        = boundedProduct({p.groups, effectiveWsReplicas(instance), g.kpg, p.y * p.x * g.cpg, 4});
    if(!bytes || *bytes > MAX_WGRAD_WS_BYTES)
    {
        return std::nullopt;
    }
    return bytes;
}

/// gfx950_conv_wgrad_aot_split_k for a two-stage binary: resolve_conv_split_k("wgrad",
/// reduction="workspace", allow_unsplit=False). It has no unsplit form, so a problem
/// whose scratch leaves i32, or with one K tile or no gridDim.z budget, has no degree
/// (std::nullopt; Python raises). Otherwise select_split_k_plan scores the f32 workspace
/// reduction of ws_replicas copies, uncapped, for the merged GEMM.
inline std::optional<int64_t>
    resolveTwoStageSplitK(const Problem& p, const Geometry& g, const ImplicitInstance& instance)
{
    const auto gm = std::max<int64_t>(1, instance.groupMerge);
    const auto gridGroups = std::max<int64_t>(1, p.groups / gm);
    const auto zCap = MAX_GRID_DIM_Z / gridGroups;
    if(!twoStageWorkspaceBytes(p, g, instance))
    {
        return std::nullopt;
    }
    const SplitKProblem gemm{
        g.kpg * gm, p.y * p.x * g.cpg * gm, p.n * g.ho * g.wo, gridGroups, 2, 2};
    if(ceilDiv(gemm.k, instance.tileK) < 2 || zCap < 2)
    {
        return std::nullopt;
    }
    const SplitKCostModel cost{};
    const auto waves = std::max<int64_t>(1, instance.warpM * instance.warpN);
    const SplitKReduction workspace{
        cost.atomicFp32BytesPerLatency, std::nullopt, true, effectiveWsReplicas(instance)};
    return selectSplitK(gemm,
                        SplitKKernel{instance.tileM, instance.tileN, instance.tileK, waves},
                        workspace,
                        false,
                        cost);
}

/// gfx950_conv_wgrad_aot_split_k: resolve_conv_split_k("wgrad", reduction="atomic",
/// epilogue=the instance's, allow_unsplit=True) on gfx950, where the cost model exists.
/// The packed 16-bit atomic is the only reduction a one-stage binary has; without it (a
/// merged tile, the default epilogue, or an odd cpg) or without room to split (one K tile,
/// or no gridDim.z budget) the launch runs unsplit. Otherwise select_split_k_plan picks
/// the degree for the GEMM kpg x (Y * X * cpg) x (N * Ho * Wo) over groups.
inline int64_t resolveSplitK(const Problem& p, const Geometry& g, const ImplicitInstance& instance)
{
    if(instance.splitK == 1)
    {
        // A one-stage binary built at split_k == 1 carries no packed-atomic epilogue at
        // all (see the ImplicitInstance::splitK comment) -- epilogue alone (cshuffle vs
        // default) does not distinguish it from an atomic-capable sk2 binary, so a
        // dynamically resolved degree > 1 here would have every grid-z block overwrite
        // dW instead of accumulating into it. Matches the Python-side fix in
        // dispatch.grouped_convolution._wgrad_split_k /
        // convolution_backward_filter.gfx950_conv_wgrad_aot_split_k.
        return 1;
    }
    const auto gm = std::max<int64_t>(1, instance.groupMerge);
    const auto gridGroups = std::max<int64_t>(1, p.groups / gm);
    const auto zCap = MAX_GRID_DIM_Z / gridGroups;
    const bool atomicOk = packedAtomicEpilogueAvailable(p, g, 0) && gm == 1
                          && instance.epilogue != Epilogue::DEFAULT;
    const SplitKProblem gemm{
        g.kpg * gm, p.y * p.x * g.cpg * gm, p.n * g.ho * g.wo, gridGroups, 2, 2};
    if(!atomicOk || ceilDiv(gemm.k, instance.tileK) < 2 || zCap < 2)
    {
        return 1;
    }
    const SplitKCostModel cost{};
    const auto waves = std::max<int64_t>(1, instance.warpM * instance.warpN);
    const SplitKReduction atomic{p.dtype == ElementType::BF16 ? cost.atomicBf16BytesPerLatency
                                                              : cost.atomicFp16BytesPerLatency,
                                 ATOMIC_16BIT_MAX_SPLIT_K};
    return selectSplitK(gemm,
                        SplitKKernel{instance.tileM, instance.tileN, instance.tileK, waves},
                        atomic,
                        true,
                        cost)
        .value_or(1);
}

/// _implicit_plan: rocKE's offer, the split-K degree, the grid (N tiles, M tiles,
/// groups / group_merge * split_k) of warp_m * warp_n * 64 threads and the kernargs.
/// One-stage: dW zeroed whenever the slices atomic-add into it. Two-stage: the WS scratch
/// (offset 0) zeroed, Stage 1 atomic-adding its slices into it, then the Stage 2 fold
/// writing dW whole over (ceil(wg_N / 64), ceil(kpg / 4), groups) of 256 threads.
inline std::optional<Plan> implicitPlan(const Problem& problem, const ImplicitInstance& instance)
{
    const auto geometry = deriveGeometry(problem);
    if(!geometry || !implicitSupportsProblem(problem, *geometry, instance))
    {
        return std::nullopt;
    }
    const auto& g = *geometry;
    const auto threads = instance.warpM * instance.warpN * WAVE_SIZE;
    std::optional<int64_t> resolved
        = instance.twoStage ? resolveTwoStageSplitK(problem, g, instance)
                            : std::optional<int64_t>(resolveSplitK(problem, g, instance));
    if(!resolved)
    {
        return std::nullopt;
    }
    const auto splitK = *resolved;
    const auto gridGroups
        = std::max<int64_t>(1, problem.groups) / std::max<int64_t>(1, instance.groupMerge);
    const std::array<int64_t, 3> grid{ceilDiv(problem.y * problem.x * g.cpg, instance.tileN),
                                      ceilDiv(g.kpg, instance.tileM),
                                      gridGroups * splitK};
    if(!gridFits(grid, threads))
    {
        return std::nullopt;
    }
    auto values = implicitKernargs(problem, g, instance, splitK);
    if(!values)
    {
        return std::nullopt;
    }
    if(!instance.twoStage)
    {
        Plan plan{problem, g, {}, {}, splitK > 1, splitK};
        plan.steps.push_back(Step{PlanKernel::SELF,
                                  Abi::IMPLICIT,
                                  grid,
                                  threads,
                                  {Buffer::DY, Buffer::X, Buffer::DW},
                                  std::move(*values)});
        return plan;
    }
    const auto wsBytes = *twoStageWorkspaceBytes(problem, g, instance);
    values->insert(values->begin() + TWO_STAGE_WS_POSITION, static_cast<int32_t>(wsBytes));
    const auto wgN = problem.y * problem.x * g.cpg;
    const std::array<int64_t, 3> reduceGrid{
        ceilDiv(wgN, WS_REDUCE_TILE_N), ceilDiv(g.kpg, WS_REDUCE_TILE_M), problem.groups};
    constexpr int64_t REDUCE_THREADS = WS_REDUCE_TILE_M * WS_REDUCE_TILE_N;
    if(!gridFits(reduceGrid, REDUCE_THREADS))
    {
        return std::nullopt;
    }
    Plan plan{problem, g, {}, {{Buffer::WS, 0, wsBytes}}, false, splitK, true};
    plan.steps.push_back(Step{PlanKernel::SELF,
                              Abi::IMPLICIT_TWO_STAGE,
                              grid,
                              threads,
                              {Buffer::DY, Buffer::X, Buffer::DW},
                              std::move(*values)});
    plan.steps.push_back(Step{PlanKernel::WS_REDUCE,
                              Abi::WS_REDUCE,
                              reduceGrid,
                              REDUCE_THREADS,
                              {Buffer::WS, Buffer::NONE, Buffer::DW},
                              {static_cast<int32_t>(g.kpg),
                               static_cast<int32_t>(wgN),
                               static_cast<int32_t>(wsBytes),
                               static_cast<int32_t>(g.tensorBytes[2]),
                               static_cast<int32_t>(problem.groups)}});
    return plan;
}

// ---- direct pipelines ----------------------------------------------------------------

/// The direct kernels' view of the problem (DirectConvProblem): one stride and one padding.
struct DirectProblem
{
    int64_t n;
    int64_t h;
    int64_t w;
    int64_t groups;
    int64_t cpg;
    int64_t kpg;
    int64_t kh;
    int64_t kw;
    int64_t pad;
    int64_t stride;
    int64_t ho;
    int64_t wo;

    int64_t totalC() const
    {
        return groups * cpg;
    }
    int64_t totalK() const
    {
        return groups * kpg;
    }
};

/// One direct compute pass on the real problem (direct_spec_for_problem's spec): its grid
/// and block, and for a folded pipeline the partial-slot workspace, the partition count
/// and the fold's channel count.
struct DirectLaunch
{
    std::array<int64_t, 3> grid;
    int64_t block;
    int64_t workspaceBytes = 0;
    int64_t partitions = 0;
    int64_t foldChannels = 0;
};

/// DirectConvWgradSpec (runtime channels): validate() and is_valid_wgrad_spec (gfx950
/// has every MFMA atom and ds_read_tr they ask for), direct_spec_for_problem's "no C tile
/// wider than cpg" rule, then direct_launch_geometry.
inline std::optional<DirectLaunch> directWgradLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    const auto wavesK = knobOr(k.wavesK, 1);
    const auto wavesC = knobOr(k.wavesC, 1);
    const auto wavesQ = knobOr(k.wavesQ, 1);
    const auto hoPerBlock = knobOr(k.hoPerBlock, 4);
    const auto mfmaK = knobOr(k.mfmaK, 32);
    const auto threads = satMul({wavesK, wavesC, wavesQ, WAVE_SIZE});
    if(p.kpg < 16 || p.cpg < 16 || p.kh < 1 || p.kh > 8 || p.kw < 1 || p.kw > 8
       || satMul(wavesK, wavesC) > 16 || threads > MAX_BLOCK_THREADS || (mfmaK != 16 && mfmaK != 32)
       || p.stride != 1 || wavesC > ceilDiv(p.cpg, 16))
    {
        return std::nullopt;
    }
    const auto blockK = wavesK * 16;
    const auto blockC = wavesC * 16;
    const auto woTiles = ceilDiv((p.w + 2 * p.pad - p.kw) / p.stride + 1, mfmaK);
    return DirectLaunch{{p.groups * ceilDiv(p.kpg, blockK) * ceilDiv(p.cpg, blockC),
                         ceilDiv(p.h, hoPerBlock),
                         p.n * ceilDiv(woTiles, wavesQ)},
                        threads};
}

/// DirectWgradTiledSpec (q_groups 1, no ping-pong): is_valid_wgrad_tiled_spec, then
/// launch_geometry -- the channel tiles times ceil(target_blocks / tiles) segment splits.
inline std::optional<DirectLaunch> tiledLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    const auto waveK16 = knobOr(k.waveK16, 2);
    const auto waveC16 = knobOr(k.waveC16, 2);
    const auto wavesK = knobOr(k.wavesK, 4);
    const auto wavesC = knobOr(k.wavesC, 2);
    const auto rowsPerSeg = knobOr(k.rowsPerSeg, 30);
    const auto prefetch = knobOr(k.prefetch, 2);
    const auto targetBlocks = knobOr(k.targetBlocks, 256);
    const auto rowUnroll = k.rowUnroll;
    const auto unfoldN = knobOr(k.unfoldN, 1);
    const auto oneTwoFour = [](int64_t v) { return v == 1 || v == 2 || v == 4; };
    const auto nWaves = satMul(wavesK, wavesC);
    const auto threads = satMul(nWaves, WAVE_SIZE);
    if(p.stride != 1 || !oneTwoFour(waveK16) || !oneTwoFour(waveC16) || !isPowerOfTwo(wavesK)
       || !isPowerOfTwo(wavesC) || !oneTwoFour(unfoldN) || threads > MAX_BLOCK_THREADS
       || p.cpg % 8 != 0 || p.kpg % 8 != 0 || p.kpg < 16 * waveK16 || p.cpg < 16 * waveC16
       || satMul({p.kh, p.kw, waveK16, waveC16, 4}) > 192 || targetBlocks < 1)
    {
        return std::nullopt;
    }
    const auto buffers = prefetch + 1;
    if(rowUnroll % buffers != 0)
    {
        return std::nullopt;
    }
    const auto unroll = rowUnroll != 0 ? rowUnroll : std::lcm(buffers, p.kh);
    if(rowsPerSeg % unroll != 0 || p.kh > 8 || p.kw > 8)
    {
        return std::nullopt;
    }
    const auto blockK = 16 * waveK16 * wavesK;
    const auto blockC = 16 * waveC16 * wavesC;
    const auto stripW = 32 / unfoldN;
    const auto seg = stripW + p.kw - 1;
    const auto ik = ceilDiv(ceilDiv(32 * blockK / 8, WAVE_SIZE), nWaves);
    const auto ic = ceilDiv(ceilDiv(unfoldN * seg * blockC / 8, WAVE_SIZE), nWaves);
    const auto slot = nWaves * (ik + ic) * WAVE_SIZE * 8 * 2;
    if(buffers * slot > DIRECT_LDS_BUDGET_BYTES)
    {
        return std::nullopt;
    }
    const auto channelTiles = p.groups * ceilDiv(p.kpg, blockK) * ceilDiv(p.cpg, blockC);
    const auto segments = ceilDiv(p.n, unfoldN) * ceilDiv(p.wo, stripW) * ceilDiv(p.h, rowsPerSeg);
    const auto splits = std::min(segments, ceilDiv(targetBlocks, channelTiles));
    return DirectLaunch{{channelTiles * splits, 1, 1}, threads};
}

/// DwWgradLdsSpec (cols_per_lane 1): is_valid_dw_wgrad_lds_spec, then (channel tiles,
/// partitions) of lanes_c * lanes_w threads storing ceil-many partial slots.
inline std::optional<DirectLaunch> dwLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    const auto lanesC = knobOr(k.lanesC, 8);
    const auto lanesW = knobOr(k.lanesW, 32);
    const auto prefetch = knobOr(k.prefetch, 2);
    const auto rowsPerSeg = knobOr(k.rowsPerSeg, 32);
    const auto targetBlocks = knobOr(k.targetBlocks, 1024);
    const auto threads = satMul(lanesC, lanesW);
    if(p.cpg != 1 || p.kpg != 1 || p.groups % 8 != 0 || threads < WAVE_SIZE
       || threads % WAVE_SIZE != 0 || threads > MAX_BLOCK_THREADS || p.kh * p.kw * 8 > 128)
    {
        return std::nullopt;
    }
    const auto blockC = 8 * lanesC;
    const auto blockWo = lanesW;
    const auto unroll = prefetch + ceilDiv(p.kh, p.stride);
    const auto nWaves = threads / WAVE_SIZE;
    const auto dmaInstrs
        = [&](int64_t cols) { return ceilDiv(ceilDiv(cols * (blockC / 8), WAVE_SIZE), nWaves); };
    const auto xCols = (blockWo - 1) * p.stride + p.kw;
    const auto xRow = satMul({threads, dmaInstrs(xCols), 8});
    const auto dyRow = threads * dmaInstrs(blockWo) * 8;
    const auto lds = satAdd(satAdd(satMul({p.stride, unroll, xRow, 2}), satMul({unroll, dyRow, 2})),
                            lanesW * blockC * 4);
    if(lds > DIRECT_LDS_BUDGET_BYTES)
    {
        return std::nullopt;
    }
    const auto channelTiles = ceilDiv(p.groups, blockC);
    const auto rps = ceilDiv(rowsPerSeg, unroll) * unroll;
    const auto units = p.n * ceilDiv(p.wo, blockWo) * ceilDiv(p.ho, rps);
    const auto partitions = std::min(units, ceilDiv(targetBlocks, channelTiles));
    return DirectLaunch{{channelTiles, partitions, 1},
                        threads,
                        partitions * p.kh * p.kw * p.groups * 4,
                        partitions,
                        p.groups};
}

/// DwWgradColsSpec: direct_spec_for_problem's baked channel count (the "channels" knob,
/// which a 0 leaves unstated -- rocKE then raises KeyError, a refusal), is_valid_dw_wgrad_
/// cols_spec, then (KH, partitions) of `threads` threads.
inline std::optional<DirectLaunch> dwColsLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    const auto strip = knobOr(k.strip, 8);
    const auto threads = knobOr(k.threads, 64);
    const auto targetBlocks = knobOr(k.targetBlocks, 2048);
    const auto c = p.groups;
    if(k.channels == 0 || k.channels != c || p.cpg != 1 || p.kpg != 1 || c > 8 || p.kw * c > 96
       || threads < WAVE_SIZE || threads % WAVE_SIZE != 0 || threads > MAX_BLOCK_THREADS
       || (strip % 8) * c % 8 != 0 || (strip % 8) * (p.stride % 8) * c % 8 != 0
       || (p.w * c) % 8 != 0 || (p.wo * c) % 8 != 0 || p.wo % strip != 0
       || threads * p.kw * c * 4 > DIRECT_LDS_BUDGET_BYTES)
    {
        return std::nullopt;
    }
    const auto units = ceilDiv(p.n * p.ho * (p.wo / strip), threads);
    const auto partitions = std::min(units, ceilDiv(targetBlocks, p.kh));
    return DirectLaunch{{p.kh, partitions, 1},
                        threads,
                        partitions * p.kh * p.kw * p.groups * 4,
                        partitions,
                        p.groups};
}

/// DwWgradMfmaSpec: is_valid_dw_wgrad_mfma_spec (4 waves, 64 channels per block), then
/// (channel tiles, column tiles, row-chunk walkers).
inline std::optional<DirectLaunch> dwMfmaLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    constexpr int64_t WAVES = 4;
    constexpr int64_t BLOCK_C = 64;
    constexpr int64_t CHUNKS = BLOCK_C / 8;
    const auto qTiles = knobOr(k.qTiles, 8);
    const auto rowsPerChunk = knobOr(k.rowsPerChunk, 16);
    const auto prefetch = knobOr(k.prefetch, 2);
    const auto targetBlocks = knobOr(k.targetBlocks, 2048);
    const auto unfoldN = knobOr(k.unfoldN, 1);
    if(p.cpg != 1 || p.kpg != 1 || p.stride != 1 || p.pad > p.kh - 1 || p.pad > p.kw - 1
       || p.groups % 8 != 0 || qTiles < 1 || qTiles > 8 || prefetch < 1 || prefetch > 4
       || unfoldN < 1 || unfoldN > 4)
    {
        return std::nullopt;
    }
    const auto blockQ = 4 * qTiles;
    const auto imagePitch = std::max({p.wo, p.wo + p.kw - 1 - p.pad, p.pad + p.w});
    if(unfoldN > 1 && unfoldN * imagePitch > blockQ)
    {
        return std::nullopt;
    }
    const auto sg = ceilDiv(p.kw, 4);
    const int64_t preshift = std::min<int64_t>(4, p.kw) - 1 > 2 ? 1 : 0;
    const auto winWords = 2 * (qTiles - 1) + 2 * (sg - 1) + 3;
    const auto inPitch = 8 * ceilDiv(winWords + preshift, 4);
    const auto inGroups = ceilDiv(2 * (winWords + preshift), 4);
    const auto ix = ceilDiv(ceilDiv(4 * inGroups * CHUNKS, WAVE_SIZE), WAVES);
    const auto iy = ceilDiv(ceilDiv(blockQ * CHUNKS, WAVE_SIZE), WAVES);
    const auto stage = prefetch * WAVES * (ix + iy) * WAVE_SIZE * 16;
    const auto lds = stage + (BLOCK_C * inPitch + BLOCK_C * (blockQ + 4)) * 2;
    if(lds > DIRECT_LDS_BUDGET_BYTES)
    {
        return std::nullopt;
    }
    const auto channelTiles = ceilDiv(p.groups, BLOCK_C);
    const auto nQ = ceilDiv(p.wo, blockQ);
    const auto units = ceilDiv(p.n, unfoldN) * ceilDiv(p.h, rowsPerChunk);
    const auto nZ = std::min(units, std::max<int64_t>(1, ceilDiv(targetBlocks, channelTiles * nQ)));
    const auto partitions = nQ * nZ;
    return DirectLaunch{{channelTiles, nQ, nZ},
                        WAVES * WAVE_SIZE,
                        partitions * p.kh * p.kw * p.groups * 4,
                        partitions,
                        p.groups};
}

/// WgradGatherSpec: is_valid_wgrad_gather_spec (dense, cpg baked), then (K tiles,
/// partitions); its fold runs per output channel over KH * KW * cpg taps.
inline std::optional<DirectLaunch> gatherLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    const auto waveK16 = knobOr(k.waveK16, 2);
    const auto wavesK = knobOr(k.wavesK, 1);
    const auto prefetch = knobOr(k.prefetch, 2);
    const auto rowsPerSeg = knobOr(k.rowsPerSeg, 16);
    const auto targetBlocks = knobOr(k.targetBlocks, 1024);
    const auto threads = satMul(wavesK, WAVE_SIZE);
    const auto atoms = ceilDiv(p.kw * p.cpg, 16);
    if(p.groups != 1 || p.kpg % 8 != 0 || p.kw * p.cpg > 96 || (p.w * p.cpg) % 8 != 0
       || (32 * p.stride * p.cpg) % 8 != 0 || satMul({p.kh, waveK16, atoms, 4}) > 192
       || threads > MAX_BLOCK_THREADS)
    {
        return std::nullopt;
    }
    const auto blockK = 16 * waveK16 * wavesK;
    const auto unroll = prefetch + ceilDiv(p.kh, p.stride);
    const auto xElems = floorMod(-p.pad * p.cpg, 8) + ((32 - 1) * p.stride + p.kw) * p.cpg;
    const auto ix = ceilDiv(ceilDiv(ceilDiv(xElems, 8), WAVE_SIZE), wavesK);
    const auto iy = ceilDiv(ceilDiv(32 * blockK / 8, WAVE_SIZE), wavesK);
    const auto xSlot = threads * ix * 8 + 16;
    const auto dySlot = threads * iy * 8;
    if(satMul(satAdd(satMul({p.stride, unroll, xSlot}), satMul(unroll, dySlot)), 2)
       > DIRECT_LDS_BUDGET_BYTES)
    {
        return std::nullopt;
    }
    const auto kTiles = ceilDiv(p.kpg, blockK);
    const auto rps = ceilDiv(rowsPerSeg, unroll) * unroll;
    const auto units = p.n * ceilDiv(p.wo, 32) * ceilDiv(p.ho, rps);
    const auto partitions = std::min(units, ceilDiv(targetBlocks, kTiles));
    const auto taps = p.kh * p.kw * p.cpg;
    return DirectLaunch{
        {kTiles, partitions, 1}, threads, partitions * taps * p.kpg * 4, partitions, p.kpg};
}

/// GroupedWgradMfmaSpec: is_valid_grouped_wgrad_mfma_spec (cpg = kpg in {4, 8}), then
/// (channel tiles, column tiles, row-chunk walkers) of waves_q waves; its fold runs per
/// output channel (all groups' K) over KH * KW * cpg taps.
inline std::optional<DirectLaunch> groupedMfmaLaunch(const DirectProblem& p, const DirectKnobs& k)
{
    constexpr int64_t BLOCK_C = 64;
    constexpr int64_t CHUNKS = BLOCK_C / 8;
    const auto wavesQ = knobOr(k.wavesQ, 4);
    const auto waveQ4 = knobOr(k.waveQ4, 1);
    const auto rowsPerChunk = knobOr(k.rowsPerChunk, 56);
    const auto prefetch = knobOr(k.prefetch, 2);
    const auto targetBlocks = knobOr(k.targetBlocks, 1024);
    if(p.cpg != p.kpg || (p.cpg != 4 && p.cpg != 8) || p.stride != 1 || p.pad > p.kh - 1
       || p.pad > p.kw - 1 || p.totalC() % 8 != 0
       || (wavesQ != 1 && wavesQ != 2 && wavesQ != 4 && wavesQ != 8)
       || (waveQ4 != 1 && waveQ4 != 2 && waveQ4 != 4) || prefetch < 2 || prefetch > 4
       || p.kh * p.kw * (p.cpg / 4) * 4 > 128)
    {
        return std::nullopt;
    }
    const auto blockQ = 4 * waveQ4 * wavesQ;
    const auto ix = ceilDiv(ceilDiv((blockQ + p.kw - 1) * CHUNKS, WAVE_SIZE), wavesQ);
    const auto iy = ceilDiv(ceilDiv(blockQ * CHUNKS, WAVE_SIZE), wavesQ);
    const auto lds = prefetch * wavesQ * (ix + iy) * WAVE_SIZE * 16 + wavesQ * WAVE_SIZE * 4 * 4;
    if(lds > DIRECT_LDS_BUDGET_BYTES)
    {
        return std::nullopt;
    }
    const auto channelTiles = ceilDiv(p.totalC(), BLOCK_C);
    const auto nQ = ceilDiv(p.wo, blockQ);
    const auto units = p.n * ceilDiv(p.h, rowsPerChunk);
    const auto nZ = std::min(units, std::max<int64_t>(1, ceilDiv(targetBlocks, channelTiles * nQ)));
    const auto partitions = nQ * nZ;
    const auto taps = p.kh * p.kw * p.cpg;
    return DirectLaunch{{channelTiles, nQ, nZ},
                        wavesQ * WAVE_SIZE,
                        partitions * taps * p.totalK() * 4,
                        partitions,
                        p.totalK()};
}

/// identity_supports_problem for a direct binary: its filter, stride, padding and
/// dilation are the problem's, and so are its channel counts where it bakes them.
inline bool directCapabilitiesMatch(const DirectCapabilities& compiled,
                                    const Problem& problem,
                                    const Geometry& geometry)
{
    return compiled.filterH == problem.y && compiled.filterW == problem.x
           && compiled.strideH == problem.strideH && compiled.strideW == problem.strideH
           && compiled.dilationH == problem.dilationH && compiled.dilationW == problem.dilationW
           && compiled.padH == problem.padH && compiled.padW == problem.padH
           && (compiled.cpg == 0 || compiled.cpg == geometry.cpg)
           && (compiled.kpg == 0 || compiled.kpg == geometry.kpg);
}

/// direct_wgrad_fold_knobs: the fold's taps per output channel -- KH * KW, times cpg for
/// the per-channel (gather, grouped MFMA) folds.
inline int64_t foldTaps(DirectVariant variant, int64_t kh, int64_t kw, int64_t cpg)
{
    return kh * kw * (isPerChannelFoldVariant(variant) ? cpg : 1);
}

/// DwWgradFoldSpec.launch_geometry: ceil(channels * taps / 64) blocks of 16 waves.
constexpr int64_t FOLD_COLUMNS = 64;
constexpr int64_t FOLD_THREADS = 16 * 64;

/// _direct_plan: one stride and one padding, rocKE's capability match, the variant's
/// validator and launch geometry on the real problem, then
///   * direct_wgrad / _tiled: one launch atomically adding into a zeroed fp32 dW;
///   * the five folded pipelines: the compute pass storing its partial slots into the
///     workspace (WS at offset 0), then the fold summing them into dW, written whole.
inline std::optional<Plan> directPlan(const Problem& problem, const DirectInstance& instance)
{
    if(instance.variant == DirectVariant::NONE || isHelperVariant(instance.variant)
       || problem.strideH != problem.strideW || problem.padH != problem.padW)
    {
        return std::nullopt;
    }
    const auto geometry = deriveGeometry(problem);
    if(!geometry || !directCapabilitiesMatch(instance.compiled, problem, *geometry)
       || problem.dilationH != 1 || problem.dilationW != 1)
    {
        return std::nullopt;
    }
    const auto& g = *geometry;
    const DirectProblem direct{problem.n,
                               problem.hi,
                               problem.wi,
                               problem.groups,
                               g.cpg,
                               g.kpg,
                               problem.y,
                               problem.x,
                               problem.padH,
                               problem.strideH,
                               g.ho,
                               g.wo};
    std::optional<DirectLaunch> launch;
    switch(instance.variant)
    {
    case DirectVariant::WGRAD:
        launch = directWgradLaunch(direct, instance.knobs);
        break;
    case DirectVariant::TILED:
        launch = tiledLaunch(direct, instance.knobs);
        break;
    case DirectVariant::DW:
        launch = dwLaunch(direct, instance.knobs);
        break;
    case DirectVariant::DW_COLS:
        launch = dwColsLaunch(direct, instance.knobs);
        break;
    case DirectVariant::DW_MFMA:
        launch = dwMfmaLaunch(direct, instance.knobs);
        break;
    case DirectVariant::GATHER:
        launch = gatherLaunch(direct, instance.knobs);
        break;
    case DirectVariant::GROUPED_MFMA:
        launch = groupedMfmaLaunch(direct, instance.knobs);
        break;
    default:
        break;
    }
    if(!launch || !gridFits(launch->grid, launch->block))
    {
        return std::nullopt;
    }
    Plan plan{problem, g, {}, {}, false, 1};
    if(!isFoldedVariant(instance.variant))
    {
        plan.zeroDw = true;
        plan.steps.push_back(Step{PlanKernel::SELF,
                                  Abi::DIRECT_WGRAD,
                                  launch->grid,
                                  launch->block,
                                  {Buffer::DY, Buffer::X, Buffer::DW},
                                  directWgradValues(problem, g)});
        return plan;
    }
    const auto taps = foldTaps(instance.variant, direct.kh, direct.kw, direct.cpg);
    const std::array<int64_t, 3> foldGrid{ceilDiv(launch->foldChannels * taps, FOLD_COLUMNS), 1, 1};
    if(!gridFits(foldGrid, FOLD_THREADS))
    {
        return std::nullopt;
    }
    plan.steps.push_back(Step{PlanKernel::SELF,
                              Abi::DIRECT_WGRAD,
                              launch->grid,
                              launch->block,
                              {Buffer::DY, Buffer::X, Buffer::WS},
                              directWgradValues(problem, g)});
    plan.steps.push_back(Step{
        PlanKernel::FOLD,
        Abi::FOLD,
        foldGrid,
        FOLD_THREADS,
        {Buffer::WS, Buffer::NONE, Buffer::DW},
        {static_cast<int32_t>(launch->partitions), static_cast<int32_t>(launch->foldChannels)}});
    plan.workspaces.push_back({Buffer::WS, 0, launch->workspaceBytes});
    return plan;
}

/// Why gfx950_conv_wgrad_aot_plan refuses @p instance on every problem, verbatim, or
/// std::nullopt when only the problem can decide: a helper (launched only by its
/// pipeline).
inline std::optional<std::string_view> staticRefusal(const AotInstance& instance)
{
    if(instance.family == KernelFamily::HELPER)
    {
        return HELPER_REFUSAL;
    }
    return std::nullopt;
}

/// gfx950_conv_wgrad_aot_plan: the launch of @p instance on @p problem, or std::nullopt
/// when it does not compute it (a dtype other than the instance's, a helper, or the
/// family's own rules).
inline std::optional<Plan> instancePlan(const Problem& problem, const AotInstance& instance)
{
    if(problem.dtype != instance.dtype || staticRefusal(instance))
    {
        return std::nullopt;
    }
    if(instance.implicit)
    {
        return implicitPlan(problem, *instance.implicit);
    }
    if(instance.direct)
    {
        return directPlan(problem, *instance.direct);
    }
    return std::nullopt;
}

// ---- the deterministic fallback order --------------------------------------------------

/// Deterministic order among the instances serving one problem, used only when nothing is
/// benchmarked, recorded or trained; higher is better. The adapter has no ranking of its
/// own, so this follows the dgrad pack's: the direct pipelines rank above implicit GEMM,
/// the shape-specialized partial-slot pipelines (depthwise, small-C, gather, grouped MFMA:
/// each serves only the narrow regime it was built to win) in [3, 3.5], the generic
/// atomic ones (direct_wgrad, _tiled) in [2, 2.5); implicit GEMM in [0, 1.01) by the
/// fraction of the launched tile area that is real dW times how well the grid (split-K
/// slices included) fills a device of 256 compute units, with wider vectors breaking ties.
inline double planScore(const AotInstance& instance, const Plan& plan)
{
    constexpr double CU_COUNT = 256.0;
    const auto& compute = plan.steps.front();
    const auto blocks = static_cast<double>(compute.grid[0]) * static_cast<double>(compute.grid[1])
                        * static_cast<double>(compute.grid[2]);
    const auto fill = blocks >= CU_COUNT ? 1.0 : blocks / CU_COUNT;
    if(instance.direct)
    {
        return (isFoldedVariant(instance.direct->variant) ? 3.0 : 2.0) + 0.49 * fill;
    }
    const auto& gemm = *instance.implicit;
    const auto& g = plan.geometry;
    const auto useful = static_cast<double>(g.kpg)
                        * static_cast<double>(plan.launched.y * plan.launched.x * g.cpg);
    const auto launched = static_cast<double>(compute.grid[0] * gemm.tileN)
                          * static_cast<double>(compute.grid[1] * gemm.tileM);
    const auto widths = static_cast<double>(gemm.vectorA + gemm.vectorB + gemm.vectorC) / 24.0;
    return (launched > 0.0 ? useful / launched : 0.0) * fill + 1e-3 * widths;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_wgrad
