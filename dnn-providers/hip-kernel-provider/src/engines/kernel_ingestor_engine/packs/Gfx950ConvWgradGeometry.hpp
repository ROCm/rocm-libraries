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
#include <string_view>
#include <utility>

/// @file Gfx950ConvWgradGeometry.hpp
/// The launch contract of hipkernel:Gfx950ConvWgradAot and
/// hipkernel:Gfx950ConvWgradTwoStageAot: the problem these engines match, the geometry it
/// implies, and the runtime kernarg values of rocKE's wgrad implicit-GEMM and workspace-
/// reduce kernels. Ports kernels/common/conv_abi.py's `conv_arg_names(direction="wgrad")`
/// and conv_wgrad_workspace_reduce.py's kernel ABI; the name lists there are the load-
/// bearing contract (hkp_pack records the compiled kernel's own parameter names), so this
/// file mirrors their ORDER exactly rather than restating it independently.
namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_wgrad
{

/// X is [N, Hi, Wi, C] NHWC, dY is [N, Ho, Wo, K] NHWK, dW is [K, Y, X, C / groups] KYXC.
/// C and K are totals across all groups.
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
};

/// What the problem implies: the forward-style dY extents, the per-group channel counts
/// and the whole tensors' byte sizes (dY, X, dW), in the launch ABI's pointer order.
struct Geometry
{
    int64_t ho;
    int64_t wo;
    int64_t cpg;
    int64_t kpg;
    std::array<int64_t, 3> tensorBytes; // dY, X, dW
};

struct LaunchGeometry
{
    unsigned int gridX;
    unsigned int gridY;
    unsigned int gridZ;
    unsigned int blockX;
};

constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();

/// The forward output extent: floor((in + 2*pad - dilation*(filter-1) - 1) / stride) + 1.
/// X plays the forward input's role and dY the forward output's, the same relationship
/// Gfx950ConvFwd's X -> O uses, so wgrad's Ho/Wo derive from Hi/Wi/Y/X exactly as a
/// forward conv's would. nullopt for any operand outside signed 32-bit descriptor
/// arithmetic, or a non-positive numerator (an output extent that would be <= 0).
inline std::optional<int64_t>
    wgradOutputExtent(int64_t input, int64_t filter, int64_t stride, int64_t pad, int64_t dilation)
{
    if(input <= 0 || input > I32_MAX || filter <= 0 || filter > I32_MAX || stride <= 0
       || stride > I32_MAX || pad < 0 || pad > I32_MAX || dilation <= 0 || dilation > I32_MAX)
    {
        return std::nullopt;
    }
    const auto padded = input + 2 * pad;
    const auto effectiveFilter = dilation * (filter - 1) + 1;
    if(padded > I32_MAX || effectiveFilter > I32_MAX || padded < effectiveFilter)
    {
        return std::nullopt;
    }
    return (padded - effectiveFilter) / stride + 1;
}

/// The product of @p factors when it is in [0, I32_MAX], else std::nullopt; every factor
/// must be nonnegative. Checked per step, so no intermediate product overflows.
inline std::optional<int64_t> boundedProduct(std::initializer_list<int64_t> factors)
{
    int64_t product = 1;
    for(const auto factor : factors)
    {
        if(factor < 0 || (factor != 0 && product > I32_MAX / factor))
        {
            return std::nullopt;
        }
        product *= factor;
    }
    return product;
}

/// The product of @p factors when it is below 2^62, else std::nullopt: the byte counts of
/// a whole tensor (which a single image already bounded to i32 can still overflow once
/// multiplied by a large batch).
inline std::optional<int64_t> wideProduct(std::initializer_list<int64_t> factors)
{
    constexpr int64_t LIMIT = int64_t{1} << 62;
    int64_t product = 1;
    for(const auto factor : factors)
    {
        if(factor < 0 || (factor != 0 && product >= LIMIT / factor))
        {
            return std::nullopt;
        }
        product *= factor;
    }
    return product;
}

/// _problem_error (Gfx950ConvWgradProblem): positive extents, groups in [1, 65535]
/// dividing C and K, a positive dY, one image of dY/X/dW within the signed 32-bit kernel
/// ABI, and every whole tensor below 2^62 bytes.
inline std::optional<Geometry> deriveGeometry(const Problem& problem)
{
    if(problem.n <= 0 || problem.n > I32_MAX || problem.c <= 0 || problem.c > I32_MAX
       || problem.k <= 0 || problem.k > I32_MAX || problem.groups < 1
       || problem.groups > 65535 || problem.c % problem.groups != 0
       || problem.k % problem.groups != 0)
    {
        return std::nullopt;
    }
    const auto ho
        = wgradOutputExtent(problem.hi, problem.y, problem.strideH, problem.padH, problem.dilationH);
    const auto wo
        = wgradOutputExtent(problem.wi, problem.x, problem.strideW, problem.padW, problem.dilationW);
    if(!ho || !wo)
    {
        return std::nullopt;
    }
    const auto cpg = problem.c / problem.groups;
    const auto kpg = problem.k / problem.groups;
    const auto dyImage = boundedProduct({*ho, *wo, problem.k, 2});
    const auto xImage = boundedProduct({problem.hi, problem.wi, problem.c, 2});
    // dW holds C / groups channels per output channel; a total-C count would let a
    // grouped kernel read past the gradient allocation.
    const auto dwBytes = boundedProduct({problem.k, cpg, problem.y, problem.x, 2});
    if(!dyImage || !xImage || !dwBytes)
    {
        return std::nullopt;
    }
    const auto dyBytes = wideProduct({problem.n, *dyImage});
    const auto xBytes = wideProduct({problem.n, *xImage});
    if(!dyBytes || !xBytes)
    {
        return std::nullopt;
    }
    return Geometry{*ho, *wo, cpg, kpg, {*dyBytes, *xBytes, *dwBytes}};
}

// ---------------------------------------------------------------------------------------
// Magic division (unmerge decode), identical algorithm to Gfx950ConvFwd/Dgrad's.
// ---------------------------------------------------------------------------------------

struct MagicDivision
{
    int32_t multiplier;
    int32_t shift;
};

inline std::optional<MagicDivision> magicDivision(int64_t divisor)
{
    if(divisor < 1 || divisor > I32_MAX)
    {
        return std::nullopt;
    }
    int32_t shift = 0;
    while((int64_t{1} << shift) < divisor)
    {
        ++shift;
    }
    const auto d = static_cast<uint64_t>(divisor);
    const uint64_t multiplier = ((((uint64_t{1} << shift) - d) << 32) / d) + 1;
    return MagicDivision{static_cast<int32_t>(static_cast<uint32_t>(multiplier)), shift};
}

/// rocKE's ConvArgs refuses a reduction extent at or above this (24-bit decode multiplies).
constexpr int64_t MUL24_REDUCTION_LIMIT = int64_t{1} << 23;

/// _wgrad_arg_names(is_3d=False, two_stage=False, streamk=None): dY(a), X(b), dW(d),
/// dY_bytes, X_bytes, dW_bytes, the 2D dims block, p_wg_M/N/K, the three stride triples,
/// the two magic pairs, the grid pair, then ks/ks_count. 42 scalar fields after the three
/// pointers (conv_abi.py).
constexpr std::array<std::string_view, 42> IMPLICIT_GEMM_KERNARG_NAMES{"p_N",
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

/// Two-stage adds the f32 scratch pair right before ks/ks_count (conv_abi.py:
/// `if two_stage or streamk == "workspace": out += [("ws_ptr","f32*"),("ws_bytes","i32")]`).
constexpr std::array<std::string_view, 44> IMPLICIT_GEMM_TWO_STAGE_KERNARG_NAMES{"p_N",
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
                                                                                 "ws_ptr",
                                                                                 "ws_bytes",
                                                                                 "ks",
                                                                                 "ks_count"};

/// conv_wgrad_workspace_reduce.py's Stage 2 ABI: ws_ptr, dw_ptr, wg_M, wg_N, ws_bytes,
/// dw_bytes, groups. ws_replicas is compile-time (baked into the kernel), not a kernarg.
constexpr std::array<std::string_view, 5> STAGE2_SCALAR_KERNARG_NAMES{
    "wg_M", "wg_N", "ws_bytes", "dw_bytes", "groups"};

using ImplicitGemmKernargs = std::array<int32_t, IMPLICIT_GEMM_KERNARG_NAMES.size()>;
using ImplicitGemmTwoStageKernargs = std::array<int32_t, IMPLICIT_GEMM_TWO_STAGE_KERNARG_NAMES.size()>;

/// The split-K slice width and degree. An unsplit launch passes ks_count = 1 and
/// ks = wg_K rounded up to tile_k; a split launch's ks is wg_K split ks_count ways and
/// rounded up to a whole number of tile_k tiles (conv_abi.py's _wgrad_arg_names docstring).
inline std::pair<int64_t, int64_t> splitKSlice(int64_t wgK, int64_t ksCount, int64_t tileK)
{
    const auto perSlice = (wgK + ksCount - 1) / ksCount;
    const auto ks = ((perSlice + tileK - 1) / tileK) * tileK;
    return {ks, ksCount};
}

// ---------------------------------------------------------------------------------------
// Split-K degree selection (runtime): ports rocke.helpers.split_k.select_split_k_wgrad and
// kernels.common.conv_implicit_gemm_wgrad.wgrad_atomic_epilogue_available /
// wgrad_atomic_store_vec, gfx950-only (this engine never serves another arch). The KMD's
// own split_k field is a BUILD-time capability flag only -- whether the binary even
// contains the atomic cshuffle epilogue (conv_implicit_gemm_wgrad.py emits it exactly when
// the builder's own spec.split_k > 1) -- never the launch degree: ks/ks_count are always-
// present runtime kernargs (conv_abi.py), and the real cost model
// (rocke.helpers.split_k.select_split_k_wgrad) picks the degree from the actual problem
// shape, not from a baked knob. See Gfx950ConvWgradAotNative.cpp's implicitGemmPlan, the
// caller that composes these two functions with the gridDim.z clamp
// (conv_split_k.py's _resolve_wgrad/_rule).
// ---------------------------------------------------------------------------------------

/// gfx950's wgrad split-K constants (rocke.helpers.split_k's per-arch tables, inlined: this
/// engine is gfx950-only, so there is no second arch to look up).
constexpr int64_t GFX950_NUM_CUS = 256;                 // _ARCH_NUM_CUS["gfx950"]
constexpr int64_t GFX950_NUM_XCDS = 8;                  // _ARCH_NUM_XCDS["gfx950"]
constexpr int64_t WGRAD_ASSUMED_WAVES_PER_CU = 2;       // WGRAD_ASSUMED_WAVES_PER_CU
constexpr int64_t WGRAD_TARGET_WAVES_PER_CU = 16;       // WGRAD_TARGET_WAVES_PER_CU
constexpr int64_t MAX_GRID_DIM_Z = 65535;               // MAX_GRID_DIM_Z

/// conv_implicit_gemm_wgrad.py's _wg_M/_wg_N/_wg_K, 2D only (no Z/Do factor -- this whole
/// header already assumes 2D, see deriveGeometry). wgN/wgK can overflow the same way
/// implicitGemmLeadingKernargs's own wg_N/wg_K already guard against; wgM (== kpg) cannot,
/// once deriveGeometry has bounded kpg to I32_MAX.
inline int64_t wgradGemmM(const Geometry& geometry)
{
    return geometry.kpg;
}

inline std::optional<int64_t> wgradGemmN(const Problem& problem, const Geometry& geometry)
{
    return boundedProduct({problem.y, problem.x, geometry.cpg});
}

inline std::optional<int64_t> wgradGemmK(const Problem& problem, const Geometry& geometry)
{
    return boundedProduct({problem.n, geometry.ho, geometry.wo});
}

/// wgrad_atomic_store_vec(vector_size_c=None): the packed-atomic cshuffle epilogue's own
/// default dW store-vector width, from the per-group channel count alone -- never the real
/// instance's vector_size_c knob (the cost model always asks with vector_size_c=None; see
/// wgrad_atomic_epilogue_available's docstring in conv_implicit_gemm_wgrad.py). Mirrors
/// WgradConvSpec.default_vector_sizes's vec_c at split_k == 1: the widest of {8,4,2,1} (dW
/// is always 16-bit on the path this guards) that divides the channel count.
inline int64_t wgradAtomicStoreVecDefault(const Problem& problem, const Geometry& geometry)
{
    const auto channels = problem.groups > 1 ? geometry.cpg : problem.c;
    for(const int64_t width : {int64_t{8}, int64_t{4}, int64_t{2}, int64_t{1}})
    {
        if(channels % width == 0)
        {
            return width;
        }
    }
    return 1;
}

/// wgrad_atomic_epilogue_available(dtype_d in {"bf16","fp16"}, vector_size_c=None,
/// wave_size=64): whether a 16-bit dW can reach split-K through the packed-atomic cshuffle
/// epilogue at all -- an even dW row length (wg_N = Y*X*cpg) and an even default store-
/// vector width (the packed <2 x dtype> atomic needs a partner element in every pair).
/// wave_size is always 64 here (matchProblem already requires gfx950 wave64 MFMA), so the
/// wave_size==32 special case (store vector forced to 2, WMMA's own packed-pair epilogue) is
/// not ported: this engine never builds or dispatches a WMMA instance. dtype_d not in
/// {bf16,fp16} (an fp32 dW) trivially returns true in the Python, but matchProblem's
/// isDeviceTensor already restricts every graph this engine serves to HALF/BFLOAT16, so that
/// branch is unreachable here and likewise not ported.
inline bool wgradAtomicEpilogueAvailable(const Problem& problem, const Geometry& geometry)
{
    const auto wgN = wgradGemmN(problem, geometry);
    if(!wgN)
    {
        return false;
    }
    const auto storeVec = wgradAtomicStoreVecDefault(problem, geometry);
    return (*wgN % 2 == 0) && (storeVec % 2 == 0);
}

/// select_split_k_wgrad's two CK-formula branches (rocke.helpers.split_k), gfx950-only. The
/// raw split-K degree for a wgrad GEMM -- NOT yet clamped to the gridDim.z budget (the
/// caller's job: conv_split_k.py's `_rule` clamps with z_cap = MAX_GRID_DIM_Z // grid_groups
/// after this returns) and not yet gated on whether the atomic epilogue the degree would
/// ride on is even reachable (wgradAtomicEpilogueAvailable, also the caller's job). wgM/wgN/
/// wgK are the per-group GEMM extents (group_merge == 1 always in this engine's catalog, so
/// the "merged tile" case select_split_k_wgrad guards for groups > 1 collapses to the plain
/// per-group GEMM); groups is the problem's raw group count (conv_split_k.py's
/// resolve_conv_split_k passes grid_groups = groups // group_merge, which is just groups
/// here); blockSize is the instance's own compiled launch block size (warp_m * warp_n *
/// wave_size), needed only by the groups > 1 branch's waves-per-block accounting.
inline int64_t selectSplitKDegreeWgrad(int64_t wgM,
                                        int64_t wgN,
                                        int64_t wgK,
                                        int64_t tileM,
                                        int64_t tileN,
                                        int64_t groups,
                                        int64_t blockSize)
{
    const auto mTiles = (wgM + tileM - 1) / tileM;
    const auto nTiles = (wgN + tileN - 1) / tileN;
    const auto baseGrid = mTiles * nTiles;
    if(baseGrid <= 0)
    {
        return 1;
    }
    if(groups <= 1)
    {
        // CK formula: floor(max_capacity / base_grid), clamped to wg_K.
        const auto maxCapacity = WGRAD_ASSUMED_WAVES_PER_CU * GFX950_NUM_CUS;
        const auto splitK = std::max<int64_t>(1, maxCapacity / baseGrid);
        return std::min(splitK, wgK);
    }
    // Grouped CK formula: a blocks-per-CU capacity (block-size-invariant, unlike the
    // ungrouped branch's wave-count-only capacity), snapped so base_grid * split_k lands on
    // an XCD-count multiple.
    constexpr int64_t WAVE_SIZE = 64; // MFMA wave64 only; this engine never serves wave32.
    const auto wavesPerBlock = std::max<int64_t>(1, blockSize / WAVE_SIZE);
    const auto blocksPerCu = std::max<int64_t>(1, WGRAD_TARGET_WAVES_PER_CU / wavesPerBlock);
    const auto capacity = blocksPerCu * GFX950_NUM_CUS;
    const auto gridNoSplit = baseGrid * groups;
    auto splitK = std::max<int64_t>(1, capacity / gridNoSplit);
    // Clamp BEFORE snapping (conv_split_k.py's comment: snapping an unclamped value can land
    // off a multiple of the step and undo the alignment).
    const auto cap = std::max<int64_t>(1, std::min(wgK, MAX_GRID_DIM_Z / groups));
    splitK = std::min(splitK, cap);
    const auto step
        = GFX950_NUM_XCDS > 1 ? GFX950_NUM_XCDS / std::gcd(baseGrid, GFX950_NUM_XCDS) : 1;
    if(step > 1)
    {
        const auto snapped = (splitK / step) * step;
        const auto snappedOrStep = snapped == 0 ? step : snapped;
        if(snappedOrStep <= cap)
        {
            splitK = snappedOrStep;
        }
    }
    return std::max<int64_t>(1, std::min(splitK, cap));
}

/// The leading 40 scalar fields shared by the single-stage and two-stage implicit-GEMM
/// ABIs (everything before ks/ks_count, and before the two-stage ws_ptr/ws_bytes pair).
/// Mirrors ConvArgs.to_launch_values(direction="wgrad") for a 2D problem: packed NHWC
/// X, KYXC dW with C / groups channels, and NHWK dY strides in elements, one magic pair
/// per unmerge divisor, and the tile counts @p tileM / @p tileN imply. std::nullopt when
/// the reduction extent (wg_N = Y*X*cpg) breaks the kernel's 24-bit decode.
inline std::optional<std::array<int32_t, 40>> implicitGemmLeadingKernargs(
    const Problem& problem, const Geometry& geometry, int64_t tileM, int64_t tileN)
{
    const auto wgM = geometry.kpg;
    const auto wgN = boundedProduct({problem.y, problem.x, geometry.cpg});
    const auto wgK = boundedProduct({problem.n, geometry.ho, geometry.wo});
    if(!wgN || !wgK || *wgN >= MUL24_REDUCTION_LIMIT || tileM <= 0 || tileN <= 0)
    {
        return std::nullopt;
    }
    const auto ho = magicDivision(geometry.ho);
    const auto wo = magicDivision(geometry.wo);
    const auto xDiv = magicDivision(problem.x);
    const auto cpgDiv = magicDivision(geometry.cpg);
    if(!ho || !wo || !xDiv || !cpgDiv)
    {
        return std::nullopt;
    }
    const auto numPidM = (wgM + tileM - 1) / tileM;
    const auto numPidN = (*wgN + tileN - 1) / tileN;
    if(numPidM > 65535 || numPidN > 65535)
    {
        return std::nullopt;
    }
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return std::array<int32_t, 40>{i32(problem.n),
                                   i32(problem.hi),
                                   i32(problem.wi),
                                   i32(problem.c),
                                   i32(problem.k),
                                   i32(problem.y),
                                   i32(problem.x),
                                   i32(problem.strideH),
                                   i32(problem.strideW),
                                   i32(problem.padH),
                                   i32(problem.padW),
                                   i32(problem.dilationH),
                                   i32(problem.dilationW),
                                   i32(problem.groups),
                                   i32(geometry.ho),
                                   i32(geometry.wo),
                                   i32(geometry.cpg),
                                   i32(geometry.kpg),
                                   i32(wgM),
                                   i32(*wgN),
                                   i32(*wgK),
                                   i32(geometry.ho * geometry.wo * problem.k),
                                   i32(geometry.wo * problem.k),
                                   i32(problem.k),
                                   i32(problem.hi * problem.wi * problem.c),
                                   i32(problem.wi * problem.c),
                                   i32(problem.c),
                                   i32(*wgN),
                                   i32(problem.x * geometry.cpg),
                                   i32(geometry.cpg),
                                   ho->multiplier,
                                   ho->shift,
                                   wo->multiplier,
                                   wo->shift,
                                   xDiv->multiplier,
                                   xDiv->shift,
                                   cpgDiv->multiplier,
                                   cpgDiv->shift,
                                   i32(numPidM),
                                   i32(numPidN)};
}

/// Single-launch implicit-GEMM kernargs: the 40 leading fields plus ks/ks_count
/// (split_k == 1: an unsplit launch, ks_count = 1, ks = wg_K rounded up to tile_k).
inline std::optional<ImplicitGemmKernargs> implicitGemmKernargs(const Problem& problem,
                                                                 const Geometry& geometry,
                                                                 int64_t tileM,
                                                                 int64_t tileN,
                                                                 int64_t tileK,
                                                                 int64_t splitK)
{
    const auto leading = implicitGemmLeadingKernargs(problem, geometry, tileM, tileN);
    if(!leading)
    {
        return std::nullopt;
    }
    const auto wgK = boundedProduct({problem.n, geometry.ho, geometry.wo});
    if(!wgK)
    {
        return std::nullopt;
    }
    const auto [ks, ksCount] = splitKSlice(*wgK, splitK, tileK);
    ImplicitGemmKernargs out{};
    std::copy(leading->begin(), leading->end(), out.begin());
    out[40] = static_cast<int32_t>(ks);
    out[41] = static_cast<int32_t>(ksCount);
    return out;
}

/// Two-stage implicit-GEMM kernargs: the 40 leading fields, the f32 scratch pair
/// (ws_bytes = groups * ws_replicas * wg_M * wg_N * 4, the Stage 2 reduce's own ABI
/// agrees on this layout) and ks/ks_count for the split degree.
inline std::optional<ImplicitGemmTwoStageKernargs> implicitGemmTwoStageKernargs(
    const Problem& problem,
    const Geometry& geometry,
    int64_t tileM,
    int64_t tileN,
    int64_t tileK,
    int64_t splitK,
    int64_t wsReplicas)
{
    const auto leading = implicitGemmLeadingKernargs(problem, geometry, tileM, tileN);
    if(!leading || splitK <= 1)
    {
        return std::nullopt;
    }
    const auto wgK = boundedProduct({problem.n, geometry.ho, geometry.wo});
    const auto wgN = boundedProduct({problem.y, problem.x, geometry.cpg});
    const auto wsBytes
        = boundedProduct({problem.groups, wsReplicas, geometry.kpg, wgN.value_or(0), 4});
    if(!wgK || !wgN || !wsBytes)
    {
        return std::nullopt;
    }
    const auto [ks, ksCount] = splitKSlice(*wgK, splitK, tileK);
    ImplicitGemmTwoStageKernargs out{};
    std::copy(leading->begin(), leading->end(), out.begin());
    out[40] = 0; // ws_ptr: not a scalar; filled in by the dispatcher, not this value array.
    out[41] = static_cast<int32_t>(*wsBytes);
    out[42] = static_cast<int32_t>(ks);
    out[43] = static_cast<int32_t>(ksCount);
    return out;
}

/// Stage 2 (workspace-reduce) launch grid: (ceil(wg_N/tile_n), ceil(wg_M/tile_m), groups),
/// block = tile_m * tile_n (one thread per output element), per
/// conv_wgrad_workspace_reduce.py's build_conv_wgrad_workspace_reduce.
inline std::optional<LaunchGeometry> stage2LaunchGeometry(
    const Geometry& geometry, const Problem& problem, int64_t tileM, int64_t tileN)
{
    if(tileM <= 0 || tileN <= 0 || tileM * tileN > 1024 || problem.groups > 65535)
    {
        return std::nullopt;
    }
    const auto wgN = boundedProduct({problem.y, problem.x, geometry.cpg});
    if(!wgN)
    {
        return std::nullopt;
    }
    const auto gridX = (*wgN + tileN - 1) / tileN;
    const auto gridY = (geometry.kpg + tileM - 1) / tileM;
    if(gridX > 65535 || gridY > 65535)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          static_cast<unsigned int>(problem.groups),
                          static_cast<unsigned int>(tileM * tileN)};
}

/// Implicit-GEMM launch grid: (num_pid_n, num_pid_m, groups * ks_count) -- block_id_z
/// decodes group = z / ks_count, slice = z % ks_count (conv_implicit_gemm_wgrad.py).
inline std::optional<LaunchGeometry> implicitGemmLaunchGeometry(
    const Problem& problem, const Geometry& geometry, int64_t tileM, int64_t tileN, int64_t ksCount, int64_t blockSize)
{
    if(tileM <= 0 || tileN <= 0 || blockSize <= 0 || blockSize > 1024 || ksCount < 1)
    {
        return std::nullopt;
    }
    const auto wgN = boundedProduct({problem.y, problem.x, geometry.cpg});
    if(!wgN)
    {
        return std::nullopt;
    }
    const auto numPidM = (geometry.kpg + tileM - 1) / tileM;
    const auto numPidN = (*wgN + tileN - 1) / tileN;
    const auto gridZ = problem.groups * ksCount;
    if(numPidM > 65535 || numPidN > 65535 || gridZ > 65535)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(numPidN),
                          static_cast<unsigned int>(numPidM),
                          static_cast<unsigned int>(gridZ),
                          static_cast<unsigned int>(blockSize)};
}

// ---------------------------------------------------------------------------------------
// Direct family (kernel_family 1): seven host-launched pipelines sharing one 20-value
// kernarg ABI (conv_abi.py's conv_direct_arg_names(direction="wgrad")) and, for five of
// them, a shared "fold" helper that sums a partial-slot scratch into dW
// (kernels/common/conv_direct_dw_wgrad.py's DwWgradFoldSpec). Every formula below is
// ported verbatim from the rocKE spec class it names, not re-derived.
// ---------------------------------------------------------------------------------------

inline int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

/// The 14 scalar kernargs every direct-family kernel and the fold helper's producer share,
/// after the three pointers (dY, X, D) and three byte counts: conv_abi.py's
/// conv_direct_arg_names(direction="wgrad"). D is dW for the two non-folded variants, the
/// partial-slot scratch for the five folded ones -- the ABI itself does not change.
using DirectKernargs = std::array<int32_t, 14>;

inline DirectKernargs directKernargs(const Problem& problem, const Geometry& geometry)
{
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return DirectKernargs{i32(problem.n),
                          i32(problem.hi),
                          i32(problem.wi),
                          i32(geometry.ho),
                          i32(geometry.wo),
                          i32(problem.groups),
                          i32(problem.c),
                          i32(problem.k),
                          i32(geometry.ho * geometry.wo * problem.k),
                          i32(geometry.wo * problem.k),
                          i32(problem.k),
                          i32(problem.hi * problem.wi * problem.c),
                          i32(problem.wi * problem.c),
                          i32(problem.c)};
}

/// A workspace-scratch byte count in [0, I32_MAX] built from already-small factors (every
/// factor here is itself a grid/tile count or a channel/tap count, never a whole-tensor
/// byte size), or nullopt on overflow -- same discipline as boundedProduct.
inline std::optional<int64_t> foldWorkspaceBytes(int64_t partitions, int64_t taps, int64_t channels)
{
    return boundedProduct({partitions, taps, channels, 4});
}

constexpr int64_t DIRECT_GRID_MAX = 65535;

inline bool gridFits(const LaunchGeometry& g)
{
    return g.gridX <= DIRECT_GRID_MAX && g.gridY <= DIRECT_GRID_MAX && g.gridZ <= DIRECT_GRID_MAX
           && g.blockX > 0 && g.blockX <= 1024;
}

/// direct_wgrad (DirectConvWgradSpec): one block owns every (r, s) tap and a single wo
/// tile; by is an INPUT-row block (n_ho_blocks() sizes on H, not Ho -- the kernel's row
/// loop walks hi, and Ho == H only when 2*PAD == KH-1).
inline std::optional<LaunchGeometry> directWgradLaunchGeometry(const Problem& problem,
                                                               const Geometry& geometry,
                                                               int64_t wavesC,
                                                               int64_t wavesK,
                                                               int64_t wavesQ,
                                                               int64_t hoPerBlock,
                                                               int64_t mfmaK)
{
    if(wavesC <= 0 || wavesK <= 0 || wavesQ <= 0 || hoPerBlock <= 0 || mfmaK <= 0)
    {
        return std::nullopt;
    }
    const auto blockK = 16 * wavesK;
    const auto blockC = 16 * wavesC;
    const auto threads = wavesK * wavesC * wavesQ * 64;
    const auto nHoBlocks = ceilDiv(problem.hi, hoPerBlock);
    const auto nWoTiles = ceilDiv(geometry.wo, mfmaK);
    const auto nQBlocks = ceilDiv(nWoTiles, wavesQ);
    const LaunchGeometry launch{
        static_cast<unsigned int>(problem.groups * ceilDiv(geometry.kpg, blockK) * ceilDiv(geometry.cpg, blockC)),
        static_cast<unsigned int>(nHoBlocks),
        static_cast<unsigned int>(problem.n * nQBlocks),
        static_cast<unsigned int>(threads)};
    return gridFits(launch) ? std::optional(launch) : std::nullopt;
}

/// direct_wgrad_tiled (DirectWgradTiledSpec): a persistent 1-D grid; wave_k16/wave_c16 are
/// not AOT knobs (every instance in the catalog bakes the spec's own default, 2), so 0
/// (the KMD "omitted" sentinel) always means 2, never "unconstrained" the way it does for
/// implicit-GEMM's vector widths.
inline std::optional<LaunchGeometry> directWgradTiledLaunchGeometry(const Problem& problem,
                                                                     const Geometry& geometry,
                                                                     int64_t wavesK,
                                                                     int64_t wavesC,
                                                                     int64_t waveK16,
                                                                     int64_t waveC16,
                                                                     int64_t qGroups,
                                                                     int64_t unfoldN,
                                                                     int64_t rowsPerSeg,
                                                                     int64_t targetBlocks)
{
    if(wavesK <= 0 || wavesC <= 0 || qGroups <= 0 || unfoldN <= 0 || 32 % unfoldN != 0
       || rowsPerSeg <= 0 || targetBlocks <= 0)
    {
        return std::nullopt;
    }
    const auto waveK = 16 * (waveK16 > 0 ? waveK16 : 2);
    const auto waveC = 16 * (waveC16 > 0 ? waveC16 : 2);
    const auto blockK = waveK * wavesK;
    const auto blockC = waveC * wavesC;
    const auto threads = wavesK * wavesC * qGroups * 64;
    const auto nChannelTiles
        = problem.groups * ceilDiv(geometry.kpg, blockK) * ceilDiv(geometry.cpg, blockC);
    const auto stripW = 32 / unfoldN;
    const auto nSegments
        = ceilDiv(problem.n, unfoldN) * ceilDiv(geometry.wo, stripW) * ceilDiv(problem.hi, rowsPerSeg);
    const auto nSplits
        = std::min(ceilDiv(nSegments, qGroups), ceilDiv(targetBlocks, nChannelTiles));
    const LaunchGeometry launch{static_cast<unsigned int>(nChannelTiles * nSplits),
                                1,
                                1,
                                static_cast<unsigned int>(threads)};
    return gridFits(launch) ? std::optional(launch) : std::nullopt;
}

/// A folded direct variant's launch plus what its fold pass needs: the direct kernel's own
/// grid/block, the partial-slot scratch's byte size, and the two runtime scalars
/// (parts, channels) the fold kernel reads.
struct FoldedDirectLaunch
{
    LaunchGeometry direct;
    int64_t workspaceBytes;
    int64_t foldParts;
    int64_t foldChannels;
};

/// direct_wgrad_dw (DwWgradLdsSpec): depthwise (cpg == kpg == 1 required by the caller),
/// one partial slot per (channel tile, partition).
inline std::optional<FoldedDirectLaunch> directWgradDwLaunchGeometry(const Problem& problem,
                                                                     const Geometry& geometry,
                                                                     int64_t lanesC,
                                                                     int64_t lanesW,
                                                                     int64_t colsPerLane,
                                                                     int64_t prefetch,
                                                                     int64_t rowsPerSeg,
                                                                     int64_t targetBlocks)
{
    if(lanesC <= 0 || lanesW <= 0 || colsPerLane <= 0 || prefetch < 0 || rowsPerSeg <= 0
       || targetBlocks <= 0 || problem.strideH <= 0)
    {
        return std::nullopt;
    }
    const auto blockC = 8 * lanesC;
    const auto blockWo = lanesW * colsPerLane;
    const auto threads = lanesC * lanesW;
    const auto unroll = prefetch + ceilDiv(problem.y, problem.strideH);
    const auto rps = ceilDiv(rowsPerSeg, unroll) * unroll;
    const auto nChannelTiles = ceilDiv(problem.groups, blockC);
    const auto nUnits = problem.n * ceilDiv(geometry.wo, blockWo) * ceilDiv(geometry.ho, rps);
    const auto nPartitions = std::min(nUnits, ceilDiv(targetBlocks, nChannelTiles));
    const LaunchGeometry launch{static_cast<unsigned int>(nChannelTiles),
                                static_cast<unsigned int>(nPartitions),
                                1,
                                static_cast<unsigned int>(threads)};
    const auto wsBytes = foldWorkspaceBytes(nPartitions, problem.y * problem.x, problem.groups);
    if(!gridFits(launch) || !wsBytes)
    {
        return std::nullopt;
    }
    return FoldedDirectLaunch{launch, *wsBytes, nPartitions, problem.groups};
}

/// direct_wgrad_dw_cols (DwWgradColsSpec): small-C depthwise, one block row per filter row.
inline std::optional<FoldedDirectLaunch> directWgradDwColsLaunchGeometry(
    const Problem& problem, const Geometry& geometry, int64_t strip, int64_t threads, int64_t targetBlocks)
{
    if(strip <= 0 || threads <= 0 || targetBlocks <= 0)
    {
        return std::nullopt;
    }
    const auto nUnits = ceilDiv(problem.n * geometry.ho * (geometry.wo / strip), threads);
    const auto nPartitions = std::min(nUnits, ceilDiv(targetBlocks, problem.y));
    const LaunchGeometry launch{
        static_cast<unsigned int>(problem.y), static_cast<unsigned int>(nPartitions), 1, static_cast<unsigned int>(threads)};
    const auto wsBytes = foldWorkspaceBytes(nPartitions, problem.y * problem.x, problem.groups);
    if(!gridFits(launch) || !wsBytes)
    {
        return std::nullopt;
    }
    return FoldedDirectLaunch{launch, *wsBytes, nPartitions, problem.groups};
}

/// direct_wgrad_dw_mfma (DwWgradMfmaSpec): depthwise, 64 channels/block, optionally
/// unfold_n images packed into one column tile -- a build-time-valid packing can still
/// overrun a runtime problem's Wo/Wi, so image_pitch() is re-checked here, not only at
/// packaging time.
inline std::optional<FoldedDirectLaunch> directWgradDwMfmaLaunchGeometry(const Problem& problem,
                                                                         const Geometry& geometry,
                                                                         int64_t qTiles,
                                                                         int64_t rowsPerChunk,
                                                                         int64_t targetBlocks,
                                                                         int64_t unfoldN)
{
    if(qTiles <= 0 || rowsPerChunk <= 0 || targetBlocks <= 0 || unfoldN <= 0)
    {
        return std::nullopt;
    }
    const auto blockQ = 4 * qTiles;
    if(unfoldN > 1)
    {
        const auto imagePitch = std::max({geometry.wo, geometry.wo + problem.x - 1 - problem.padW, problem.padW + problem.wi});
        if(unfoldN * imagePitch > blockQ)
        {
            return std::nullopt;
        }
    }
    const auto nChannelTiles = ceilDiv(problem.groups, 64);
    const auto nQ = ceilDiv(geometry.wo, blockQ);
    const auto nUnits = ceilDiv(problem.n, unfoldN) * ceilDiv(problem.hi, rowsPerChunk);
    const auto nZ = std::min(nUnits, std::max(int64_t{1}, ceilDiv(targetBlocks, nChannelTiles * nQ)));
    const auto nPartitions = nQ * nZ;
    const LaunchGeometry launch{static_cast<unsigned int>(nChannelTiles),
                                static_cast<unsigned int>(nQ),
                                static_cast<unsigned int>(nZ),
                                256};
    const auto wsBytes = foldWorkspaceBytes(nPartitions, problem.y * problem.x, problem.groups);
    if(!gridFits(launch) || !wsBytes)
    {
        return std::nullopt;
    }
    return FoldedDirectLaunch{launch, *wsBytes, nPartitions, problem.groups};
}

/// direct_wgrad_grouped_mfma (GroupedWgradMfmaSpec): cpg == kpg in {4, 8} (checked by the
/// caller against the instance's baked cpg/kpg), 64 total channels per block regardless of
/// how many groups that spans.
inline std::optional<FoldedDirectLaunch> directWgradGroupedMfmaLaunchGeometry(
    const Problem& problem,
    const Geometry& geometry,
    int64_t wavesQ,
    int64_t waveQ4,
    int64_t rowsPerChunk,
    int64_t targetBlocks)
{
    if(wavesQ <= 0 || rowsPerChunk <= 0 || targetBlocks <= 0)
    {
        return std::nullopt;
    }
    const auto waveQ4Eff = waveQ4 > 0 ? waveQ4 : 1;
    const auto threads = wavesQ * 64;
    const auto blockQ = 4 * waveQ4Eff * wavesQ;
    const auto nChannelTiles = ceilDiv(problem.c, 64);
    const auto nQ = ceilDiv(geometry.wo, blockQ);
    const auto nUnits = problem.n * ceilDiv(problem.hi, rowsPerChunk);
    const auto nZ = std::min(nUnits, std::max(int64_t{1}, ceilDiv(targetBlocks, nChannelTiles * nQ)));
    const auto nPartitions = nQ * nZ;
    const LaunchGeometry launch{static_cast<unsigned int>(nChannelTiles),
                                static_cast<unsigned int>(nQ),
                                static_cast<unsigned int>(nZ),
                                static_cast<unsigned int>(threads)};
    const auto foldTaps = problem.y * problem.x * geometry.cpg;
    const auto wsBytes = foldWorkspaceBytes(nPartitions, foldTaps, problem.k);
    if(!gridFits(launch) || !wsBytes)
    {
        return std::nullopt;
    }
    return FoldedDirectLaunch{launch, *wsBytes, nPartitions, problem.k};
}

/// direct_wgrad_gather (WgradGatherSpec): dense only (groups == 1, checked by the caller),
/// C baked exactly (against the instance's cpg), K runtime.
inline std::optional<FoldedDirectLaunch> directWgradGatherLaunchGeometry(const Problem& problem,
                                                                         const Geometry& geometry,
                                                                         int64_t waveK16,
                                                                         int64_t wavesK,
                                                                         int64_t prefetch,
                                                                         int64_t rowsPerSeg,
                                                                         int64_t targetBlocks)
{
    if(wavesK <= 0 || prefetch < 0 || rowsPerSeg <= 0 || targetBlocks <= 0 || problem.strideH <= 0)
    {
        return std::nullopt;
    }
    const auto waveK16Eff = waveK16 > 0 ? waveK16 : 2;
    const auto blockK = 16 * waveK16Eff * wavesK;
    const auto threads = wavesK * 64;
    const auto nKTiles = ceilDiv(geometry.kpg, blockK);
    const auto unroll = prefetch + ceilDiv(problem.y, problem.strideH);
    const auto rps = ceilDiv(rowsPerSeg, unroll) * unroll;
    const auto nUnits = problem.n * ceilDiv(geometry.wo, 32) * ceilDiv(geometry.ho, rps);
    const auto nPartitions = std::min(nUnits, ceilDiv(targetBlocks, nKTiles));
    const LaunchGeometry launch{static_cast<unsigned int>(nKTiles),
                                static_cast<unsigned int>(nPartitions),
                                1,
                                static_cast<unsigned int>(threads)};
    const auto foldTaps = problem.y * problem.x * geometry.cpg;
    const auto wsBytes = foldWorkspaceBytes(nPartitions, foldTaps, geometry.kpg);
    if(!gridFits(launch) || !wsBytes)
    {
        return std::nullopt;
    }
    return FoldedDirectLaunch{launch, *wsBytes, nPartitions, geometry.kpg};
}

/// direct_wgrad_fold (DwWgradFoldSpec): one block owns 64 (channel, tap) output slots; 16
/// waves split the partition axis and meet in LDS. taps is baked into the binary; parts
/// and channels are runtime kernargs the producing kernel's own launch already computed
/// (FoldedDirectLaunch::foldParts/foldChannels).
inline std::optional<LaunchGeometry> directWgradFoldLaunchGeometry(int64_t channels, int64_t taps)
{
    if(channels <= 0 || taps <= 0)
    {
        return std::nullopt;
    }
    const auto product = boundedProduct({channels, taps});
    if(!product)
    {
        return std::nullopt;
    }
    const LaunchGeometry launch{static_cast<unsigned int>(ceilDiv(*product, 64)), 1, 1, 1024};
    return gridFits(launch) ? std::optional(launch) : std::nullopt;
}

/// direct_wgrad_cast (DirectWgradCastSpec): a flat elementwise narrowing cast,
/// D[i] = cast(A[i]) for i in [0, count). Mirrors DirectWgradCastSpec.launch_geometry()
/// exactly: one thread per element, no reduction, no transpose.
inline std::optional<LaunchGeometry> directWgradCastLaunchGeometry(int64_t count, int64_t threads)
{
    if(count <= 0 || threads <= 0 || threads > 1024)
    {
        return std::nullopt;
    }
    const LaunchGeometry launch{
        static_cast<unsigned int>(ceilDiv(count, threads)), 1, 1, static_cast<unsigned int>(threads)};
    return gridFits(launch) ? std::optional(launch) : std::nullopt;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_wgrad
