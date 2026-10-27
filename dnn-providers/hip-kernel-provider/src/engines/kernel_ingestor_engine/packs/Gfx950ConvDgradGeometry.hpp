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
#include <vector>

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"

/// @file Gfx950ConvDgradGeometry.hpp
/// The launch contract of hipkernel:Gfx950ConvDgradAot, the shape-generic backward-data
/// (dgrad) AOT-instance engine: the problem it matches, the tilde sub-GEMM table its
/// implicit-GEMM kernels read, the kernel arguments and grids of every family, the MFMA
/// pipeline's plan, and the served predicates. Each function mirrors the rocKE adapter
/// rocke/library/builders/common/convolution_backward_data.py, named beside it; the unit
/// test TestGfx950ConvDgradGeometry.cpp compares them with golden vectors that adapter
/// produced.
namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad
{

// Pure helpers shared with the forward engine, unchanged.
using gfx950_conv_fwd::bufferRangesDoNotOverlap;
using gfx950_conv_fwd::ceilDiv;
using gfx950_conv_fwd::LaunchGeometry;
using gfx950_conv_fwd::magicDivision;
using gfx950_conv_fwd::MagicDivision;
using gfx950_conv_fwd::tensorByteSize;

constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();
/// The hardware cap of grid y and z, and of every axis of the direct compute grids.
constexpr int64_t MAX_GRID_DIM = 65535;
/// A dispatch's x extent in work-items is 32-bit.
constexpr int64_t MAX_DISPATCH_ITEMS = (int64_t{1} << 32) - 1;
constexpr int64_t WAVE_SIZE = 64;
constexpr int64_t MAX_BLOCK_THREADS = 1024;

/// Record and magic-table widths of the implicit-GEMM sub_gemm_buf, in i32
/// (SUB_GEMM_RECORD_FIELDS, SUB_GEMM_MAGIC_FIELDS in convolution_backward_data.py).
constexpr int64_t SUB_GEMM_RECORD_FIELDS = 22;
constexpr int64_t SUB_GEMM_MAGIC_FIELDS = 8;
/// The MFMA pipeline's ws_coa starts at ws_t's size rounded up to this.
constexpr int64_t MFMA_WORKSPACE_ALIGNMENT = 256;

/// One backward-data problem as the engine matches it (Gfx950ConvDgradProblem). Hi and
/// Wi are the dX extents, which the graph states: several input extents give one output
/// extent under a stride. C and K are totals across all groups. dY is [N, Ho, Wo, K]
/// NHWK, W is [K, Y, X, C / groups] KYXC and dX is [N, Hi, Wi, C] NHWC.
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

/// What the problem implies: the dY extents, the per-group channel counts and the
/// tensor byte sizes in the ABI order (dY, W, dX).
struct Geometry
{
    int64_t ho;
    int64_t wo;
    int64_t cpg;
    int64_t kpg;
    std::array<int32_t, 3> tensorBytes;
};

/// The forward output extent of one spatial axis, which dY must have:
/// (in + 2 pad - dil (filter - 1) - 1) / stride + 1. The numerator is checked BEFORE
/// dividing: Python's `//` floors, so a negative numerator gives an extent <= 0 there and
/// the problem is refused, while C++ `/` truncates toward zero and would give 1 (Hi = 1,
/// Y = 2, p = 0, s = 2). std::nullopt for any operand outside signed 32-bit descriptor
/// arithmetic (_problem_error).
inline std::optional<int64_t>
    dgradOutputExtent(int64_t input, int64_t filter, int64_t stride, int64_t pad, int64_t dilation)
{
    if(input <= 0 || input > I32_MAX || filter <= 0 || filter > I32_MAX || stride <= 0
       || stride > I32_MAX || pad < 0 || pad > I32_MAX || dilation <= 0 || dilation > I32_MAX)
    {
        return std::nullopt;
    }
    const auto padded = input + 2 * pad;
    const auto effectiveFilter = dilation * (filter - 1) + 1;
    if(padded > I32_MAX || effectiveFilter > I32_MAX)
    {
        return std::nullopt;
    }
    const auto numerator = padded - effectiveFilter;
    if(numerator < 0)
    {
        return std::nullopt;
    }
    return numerator / stride + 1;
}

/// _problem_error: positive extents, groups in [1, 65535] dividing C and K, a positive
/// dY, and every tensor's byte count within the signed 32-bit kernel ABI.
inline std::optional<Geometry> deriveGeometry(const Problem& problem)
{
    if(problem.groups < 1 || problem.groups > MAX_GRID_DIM || problem.c <= 0 || problem.k <= 0
       || problem.c % problem.groups != 0 || problem.k % problem.groups != 0)
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
    const auto dyBytes = tensorByteSize({problem.n, problem.k, *ho, *wo});
    // W holds C / groups channels per filter; a total-C count would let a grouped kernel
    // read past the filter allocation.
    const auto wBytes = tensorByteSize({problem.k, cpg, problem.y, problem.x});
    const auto dxBytes = tensorByteSize({problem.n, problem.c, problem.hi, problem.wi});
    if(!dyBytes || !wBytes || !dxBytes)
    {
        return std::nullopt;
    }
    return Geometry{*ho, *wo, cpg, kpg, {*dyBytes, *wBytes, *dxBytes}};
}

/// ceil(a / b) for any a and b > 0, as Python's -(-a // b).
inline int64_t ceilDivSigned(int64_t a, int64_t b)
{
    return a >= 0 ? ceilDiv(a, b) : -((-a) / b);
}

// ---------------------------------------------------------------------------------------
// The tilde decomposition and the sub-GEMM table (conv_implicit_gemm_dgrad.py)
// ---------------------------------------------------------------------------------------

/// compute_tilde: a strided or dilated backward-data convolution decomposes into
/// y_tilde * x_tilde independent sub-GEMMs, each over a subset of filter taps.
struct TildeDecomposition
{
    int64_t gcdH;
    int64_t gcdW;
    int64_t yTilde;
    int64_t xTilde;
    int64_t yDot;
    int64_t xDot;
    int64_t hTilde;
    int64_t wTilde;
};

inline TildeDecomposition computeTilde(const Problem& p, const Geometry& g)
{
    TildeDecomposition t{};
    t.gcdH = std::gcd(p.strideH, p.dilationH);
    t.gcdW = std::gcd(p.strideW, p.dilationW);
    t.yTilde = p.strideH / t.gcdH;
    t.xTilde = p.strideW / t.gcdW;
    t.yDot = ceilDiv(p.y, t.yTilde);
    t.xDot = ceilDiv(p.x, t.xTilde);
    t.hTilde = p.y > 1 ? g.ho + ceilDiv(p.dilationH * (p.y - 1), p.strideH) : g.ho;
    t.wTilde = p.x > 1 ? g.wo + ceilDiv(p.dilationW * (p.x - 1), p.strideW) : g.wo;
    return t;
}

/// One non-empty sub-GEMM (SubGemmParams), every field already checked to fit i32.
struct SubGemm
{
    int64_t iYTilde;
    int64_t iXTilde;
    int64_t yDotSlice;
    int64_t xDotSlice;
    int64_t hTildeSliceBegin;
    int64_t hTildeSlice;
    int64_t wTildeSliceBegin;
    int64_t wTildeSlice;
    int64_t gemmM;
    int64_t gemmN;
    int64_t gemmK;
    int64_t blockStart;
    int64_t blockEnd;
    int64_t aEmbedHCoeff;
    int64_t aEmbedWCoeff;
    int64_t bYStride;
    int64_t bYOffset;
    int64_t bXStride;
    int64_t bXOffset;
    int64_t dHStride;
    int64_t dHOffset;
    int64_t dWStride;
    int64_t dWOffset;
    int64_t gemmKPadded;
};

inline bool fitsI32(int64_t value)
{
    return value >= std::numeric_limits<int32_t>::min() && value <= I32_MAX;
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

/// enumerate_sub_gemms (split_k 1): the non-empty sub-GEMMs in (i_ytilde, i_xtilde)
/// row-major order, with cumulative block ranges over (gemm_m / tile_m) x (gemm_n /
/// tile_n) tiles. Ported from the PYTHON reference, per group: gemm_n = C / groups and
/// gemm_k = y_dot_slice * x_dot_slice * K / groups. rocKE's C++ twin
/// rocke_enumerate_sub_gemms reads the total C and K instead, which is wrong for
/// groups > 1 (rocKE uses only its record count); it is deliberately not what this
/// follows. std::nullopt when a tile is not positive or any record field leaves i32.
inline std::optional<std::vector<SubGemm>> enumerateSubGemms(
    const Problem& p, const Geometry& g, int64_t tileM, int64_t tileN, int64_t tileK)
{
    if(tileM <= 0 || tileN <= 0 || tileK <= 0)
    {
        return std::nullopt;
    }
    const auto tilde = computeTilde(p, g);
    std::vector<SubGemm> subGemms;
    int64_t cumulativeTiles = 0;
    for(int64_t iYt = 0; iYt < tilde.yTilde; ++iYt)
    {
        for(int64_t iXt = 0; iXt < tilde.xTilde; ++iXt)
        {
            const auto yDotSlice = ceilDivSigned(p.y - iYt, tilde.yTilde);
            const auto xDotSlice = ceilDivSigned(p.x - iXt, tilde.xTilde);
            if(yDotSlice <= 0 || xDotSlice <= 0)
            {
                continue;
            }
            const auto hBegin
                = std::max<int64_t>(0, p.padH - p.dilationH * (tilde.yTilde - 1)) / p.strideH;
            const auto hEnd
                = std::min(tilde.hTilde, ceilDivSigned(p.padH + p.hi - 1, p.strideH) + 1);
            const auto wBegin
                = std::max<int64_t>(0, p.padW - p.dilationW * (tilde.xTilde - 1)) / p.strideW;
            const auto wEnd
                = std::min(tilde.wTilde, ceilDivSigned(p.padW + p.wi - 1, p.strideW) + 1);
            const auto hSlice = hEnd - hBegin;
            const auto wSlice = wEnd - wBegin;
            if(hSlice <= 0 || wSlice <= 0)
            {
                continue;
            }
            const auto gemmM = boundedProduct({p.n, hSlice, wSlice});
            const auto gemmK = boundedProduct({yDotSlice, xDotSlice, g.kpg});
            if(!gemmM || !gemmK)
            {
                return std::nullopt;
            }
            const auto gemmN = g.cpg;
            const auto tiles = ceilDiv(*gemmM, tileM) * ceilDiv(gemmN, tileN);
            SubGemm sg{};
            sg.iYTilde = iYt;
            sg.iXTilde = iXt;
            sg.yDotSlice = yDotSlice;
            sg.xDotSlice = xDotSlice;
            sg.hTildeSliceBegin = hBegin;
            sg.hTildeSlice = hSlice;
            sg.wTildeSliceBegin = wBegin;
            sg.wTildeSlice = wSlice;
            sg.gemmM = *gemmM;
            sg.gemmN = gemmN;
            sg.gemmK = *gemmK;
            sg.blockStart = cumulativeTiles;
            sg.blockEnd = cumulativeTiles + tiles;
            cumulativeTiles = sg.blockEnd;
            sg.aEmbedHCoeff = -(p.dilationH / tilde.gcdH);
            sg.aEmbedWCoeff = -(p.dilationW / tilde.gcdW);
            sg.bYStride = tilde.yTilde;
            sg.bYOffset = iYt;
            sg.bXStride = tilde.xTilde;
            sg.bXOffset = iXt;
            sg.dHStride = p.strideH;
            sg.dHOffset = p.dilationH * iYt + p.strideH * hBegin - p.padH;
            sg.dWStride = p.strideW;
            sg.dWOffset = p.dilationW * iXt + p.strideW * wBegin - p.padW;
            sg.gemmKPadded = ceilDiv(*gemmK, tileK) * tileK;
            for(const auto value : {sg.blockStart,
                                    sg.blockEnd,
                                    sg.hTildeSliceBegin,
                                    sg.wTildeSliceBegin,
                                    sg.dHOffset,
                                    sg.dWOffset,
                                    sg.gemmKPadded,
                                    sg.bYStride,
                                    sg.bXStride,
                                    sg.dHStride,
                                    sg.dWStride})
            {
                if(!fitsI32(value))
                {
                    return std::nullopt;
                }
            }
            subGemms.push_back(sg);
        }
    }
    return subGemms;
}

/// pack_sub_gemm_buffer: the n 22-i32 records (SubGemmParams.pack_i32 with fields 1 and 2
/// the record's own tile counts for this tile), then one 8-i32 magic entry per record --
/// (multiplier, shift) for kpg, x_dot_slice, h_tilde_slice * w_tilde_slice and
/// w_tilde_slice. Records first, then the magic entries; never interleaved. A
/// non-positive or out-of-range divisor is a corrupt record and refuses (std::nullopt),
/// as rocKE's own packer does.
inline std::optional<std::vector<int32_t>>
    packSubGemmBuffer(const std::vector<SubGemm>& subGemms, int64_t tileM, int64_t tileN)
{
    if(tileM <= 0 || tileN <= 0)
    {
        return std::nullopt;
    }
    std::vector<int32_t> buffer;
    buffer.reserve(subGemms.size()
                   * static_cast<size_t>(SUB_GEMM_RECORD_FIELDS + SUB_GEMM_MAGIC_FIELDS));
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    for(const auto& sg : subGemms)
    {
        const std::array<int64_t, SUB_GEMM_RECORD_FIELDS> record{sg.blockStart,
                                                                 ceilDiv(sg.gemmM, tileM),
                                                                 ceilDiv(sg.gemmN, tileN),
                                                                 sg.gemmM,
                                                                 sg.gemmK,
                                                                 sg.hTildeSlice,
                                                                 sg.wTildeSlice,
                                                                 sg.hTildeSliceBegin,
                                                                 sg.wTildeSliceBegin,
                                                                 sg.yDotSlice,
                                                                 sg.xDotSlice,
                                                                 sg.aEmbedHCoeff,
                                                                 sg.aEmbedWCoeff,
                                                                 sg.bYStride,
                                                                 sg.bYOffset,
                                                                 sg.bXStride,
                                                                 sg.bXOffset,
                                                                 sg.dHStride,
                                                                 sg.dHOffset,
                                                                 sg.dWStride,
                                                                 sg.dWOffset,
                                                                 sg.gemmKPadded};
        for(const auto value : record)
        {
            if(!fitsI32(value))
            {
                return std::nullopt;
            }
            buffer.push_back(i32(value));
        }
    }
    for(const auto& sg : subGemms)
    {
        const auto taps = sg.yDotSlice * sg.xDotSlice;
        if(taps <= 0 || sg.gemmK % taps != 0)
        {
            return std::nullopt;
        }
        const auto slice = boundedProduct({sg.hTildeSlice, sg.wTildeSlice});
        if(!slice)
        {
            return std::nullopt;
        }
        for(const auto divisor : {sg.gemmK / taps, sg.xDotSlice, *slice, sg.wTildeSlice})
        {
            const auto magic = magicDivision(divisor);
            if(!magic)
            {
                return std::nullopt;
            }
            buffer.push_back(magic->multiplier);
            buffer.push_back(magic->shift);
        }
    }
    return buffer;
}

// ---------------------------------------------------------------------------------------
// Implicit GEMM (kernel_family 0)
// ---------------------------------------------------------------------------------------

/// One implicit-GEMM instance: the build-time fields of a rocKE dgrad KernelIdentity that
/// bound which problems its binary computes and how it launches (Gfx950ConvDgradAotSpec).
struct ImplicitInstance
{
    int64_t tileM;
    int64_t tileN;
    int64_t tileK;
    int64_t warpM;
    int64_t warpN;
    int64_t vectorA; ///< dY load width, along K / groups.
    int64_t vectorB; ///< W load width, along C / groups.
    int64_t vectorC; ///< dX store width, along C / groups.
    int64_t maxSubGemms; ///< Records the kernel's table walk supports.
    bool grouped; ///< Built with the grouped code path, which also serves groups == 1.
};

/// The launch of an implicit-GEMM instance on one problem: the flat grid, the packed
/// sub-GEMM table and what launch() does around the kernel.
struct ImplicitLaunch
{
    LaunchGeometry geometry;
    std::vector<int32_t> table;
    int64_t numSubGemms;
    /// needs_zeroed_dx: a strided problem leaves the dX rows and columns no tap reaches
    /// unwritten, so launch() zero-fills dX first.
    bool zeroDx;
    /// The table's device copy, (22 + 8) * num_sub_gemms i32.
    size_t workspaceBytes;
};

/// The launch of @p instance on @p problem, or std::nullopt when the binary cannot compute
/// it. Mirrors _implicit_serves in convolution_backward_data.py (the dgrad branch of
/// rocKE's KernelCache.supports_problem plus the launch bounds), in its order: the dY
/// width divides K / groups and the W and dX widths C / groups; a grouped problem needs a
/// grouped binary; 1 <= non-empty sub-GEMMs <= max_sub_gemms; depthwise (groups > 1,
/// C / groups == 1) is refused; the flat tile grid fits one dispatch; N * Hi * Wi stays
/// below 2^31. A packer failure refuses too. Grid (flat tiles, groups, 1), block
/// warp_m * warp_n * 64.
inline std::optional<ImplicitLaunch> implicitInstanceLaunch(const Problem& problem,
                                                            const Geometry& geometry,
                                                            const ImplicitInstance& instance)
{
    if(instance.vectorA <= 0 || instance.vectorB <= 0 || instance.vectorC <= 0
       || instance.warpM <= 0 || instance.warpN <= 0 || instance.maxSubGemms <= 0)
    {
        return std::nullopt;
    }
    if(geometry.kpg % instance.vectorA != 0 || geometry.cpg % instance.vectorB != 0
       || geometry.cpg % instance.vectorC != 0)
    {
        return std::nullopt;
    }
    if(problem.groups > 1 && !instance.grouped)
    {
        return std::nullopt;
    }
    const auto subGemms
        = enumerateSubGemms(problem, geometry, instance.tileM, instance.tileN, instance.tileK);
    if(!subGemms || subGemms->empty()
       || static_cast<int64_t>(subGemms->size()) > instance.maxSubGemms)
    {
        return std::nullopt;
    }
    if(problem.groups > 1 && geometry.cpg == 1)
    {
        return std::nullopt;
    }
    if(problem.groups > MAX_GRID_DIM || instance.warpM > MAX_BLOCK_THREADS / WAVE_SIZE
       || instance.warpN > MAX_BLOCK_THREADS / (WAVE_SIZE * instance.warpM))
    {
        return std::nullopt;
    }
    const auto block = instance.warpM * instance.warpN * WAVE_SIZE;
    const auto flat = subGemms->back().blockEnd;
    if(flat < 1 || flat > I32_MAX || flat * block > MAX_DISPATCH_ITEMS)
    {
        return std::nullopt;
    }
    const auto gemmM = boundedProduct({problem.n, problem.hi, problem.wi});
    if(!gemmM)
    {
        return std::nullopt;
    }
    auto table = packSubGemmBuffer(*subGemms, instance.tileM, instance.tileN);
    if(!table)
    {
        return std::nullopt;
    }
    const auto numSubGemms = static_cast<int64_t>(subGemms->size());
    return ImplicitLaunch{LaunchGeometry{static_cast<unsigned int>(flat),
                                         static_cast<unsigned int>(problem.groups),
                                         1,
                                         static_cast<unsigned int>(block)},
                          std::move(*table),
                          numSubGemms,
                          problem.strideH > 1 || problem.strideW > 1,
                          static_cast<size_t>(numSubGemms)
                              * static_cast<size_t>(SUB_GEMM_RECORD_FIELDS + SUB_GEMM_MAGIC_FIELDS)
                              * sizeof(int32_t)};
}

/// The by-value i32 block of the 44-argument dgrad implicit-GEMM ABI, after
/// (dY*, W*, dX*, dY_bytes, W_bytes, dX_bytes) and before (sub_gemm_buf*, num_sub_gemms):
/// _dgrad_arg_names in rocke/library/kernels/common/conv_abi.py. The kernel reads them
/// positionally; the names are compared against the packaged symbol's recorded arguments.
constexpr std::array<std::string_view, 36> IMPLICIT_GEMM_KERNARG_NAMES{"p_N",
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
                                                                       "p_dg_M",
                                                                       "p_dg_N",
                                                                       "p_dg_K",
                                                                       "p_dY_stride_n",
                                                                       "p_dY_stride_ho",
                                                                       "p_dY_stride_wo",
                                                                       "p_W_stride_k",
                                                                       "p_W_stride_y",
                                                                       "p_W_stride_x",
                                                                       "p_dX_stride_n",
                                                                       "p_dX_stride_hi",
                                                                       "p_dX_stride_wi",
                                                                       "p_magic_m_Hi_mult",
                                                                       "p_magic_m_Hi_shift",
                                                                       "p_magic_m_Wi_mult",
                                                                       "p_magic_m_Wi_shift",
                                                                       "p_num_pid_m",
                                                                       "p_num_pid_n"};

using ImplicitGemmKernargs = std::array<int32_t, IMPLICIT_GEMM_KERNARG_NAMES.size()>;

/// The values for IMPLICIT_GEMM_KERNARG_NAMES: ConvArgs.to_launch_values for a dgrad
/// ConvProblem. The GEMM is M = N * Hi * Wi, N = C / groups, K = Y * X * K / groups; the
/// strides are packed NHWK dY, KYXC W and NHWC dX in elements; the magic pairs divide by
/// Hi and Wi; num_pid_m / num_pid_n are the whole problem's tile counts for the
/// instance's own tile. std::nullopt when a value leaves i32 or a tile is not positive.
inline std::optional<ImplicitGemmKernargs>
    implicitGemmKernargs(const Problem& p, const Geometry& g, int64_t tileM, int64_t tileN)
{
    const auto gemmM = boundedProduct({p.n, p.hi, p.wi});
    const auto gemmK = boundedProduct({p.y, p.x, g.kpg});
    const auto hi = magicDivision(p.hi);
    const auto wi = magicDivision(p.wi);
    if(tileM <= 0 || tileN <= 0 || !gemmM || !gemmK || !hi || !wi)
    {
        return std::nullopt;
    }
    // Every stride below is bounded by its tensor's element count, which deriveGeometry
    // kept within i32.
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return ImplicitGemmKernargs{i32(p.n),
                                i32(p.hi),
                                i32(p.wi),
                                i32(p.c),
                                i32(p.k),
                                i32(p.y),
                                i32(p.x),
                                i32(p.strideH),
                                i32(p.strideW),
                                i32(p.padH),
                                i32(p.padW),
                                i32(p.dilationH),
                                i32(p.dilationW),
                                i32(p.groups),
                                i32(g.ho),
                                i32(g.wo),
                                i32(g.cpg),
                                i32(g.kpg),
                                i32(*gemmM),
                                i32(g.cpg),
                                i32(*gemmK),
                                i32(g.ho * g.wo * p.k),
                                i32(g.wo * p.k),
                                i32(p.k),
                                i32(p.y * p.x * g.cpg),
                                i32(p.x * g.cpg),
                                i32(g.cpg),
                                i32(p.hi * p.wi * p.c),
                                i32(p.wi * p.c),
                                i32(p.c),
                                hi->multiplier,
                                hi->shift,
                                wi->multiplier,
                                wi->shift,
                                i32(ceilDiv(*gemmM, tileM)),
                                i32(ceilDiv(g.cpg, tileN))};
}

/// Deterministic order among the implicit-GEMM instances serving one problem, used only
/// when nothing is benchmarked, recorded or trained; higher is better, in [0, 1.01). The
/// fraction of the launched tile area that is real output (Σ gemm_m * cpg over the tiles
/// launched), times how well the grid fills a device of CU_COUNT compute units, with
/// wider vectors breaking ties: forward's aotInstanceScore on the sub-GEMM grid.
inline double aotInstanceScore(const Problem& problem,
                               const Geometry& geometry,
                               const ImplicitInstance& instance,
                               const ImplicitLaunch& launch)
{
    constexpr double CU_COUNT = 256.0;
    const auto subGemms
        = enumerateSubGemms(problem, geometry, instance.tileM, instance.tileN, instance.tileK);
    if(!subGemms || subGemms->empty())
    {
        return 0.0;
    }
    double useful = 0.0;
    double launched = 0.0;
    for(const auto& sg : *subGemms)
    {
        useful += static_cast<double>(sg.gemmM) * static_cast<double>(sg.gemmN);
        launched += static_cast<double>(ceilDiv(sg.gemmM, instance.tileM) * instance.tileM)
                    * static_cast<double>(ceilDiv(sg.gemmN, instance.tileN) * instance.tileN);
    }
    const auto tiles
        = static_cast<double>(launch.geometry.gridX) * static_cast<double>(launch.geometry.gridY);
    const auto fill = tiles >= CU_COUNT ? 1.0 : tiles / CU_COUNT;
    const auto widths
        = static_cast<double>(instance.vectorA + instance.vectorB + instance.vectorC) / 24.0;
    return (launched > 0.0 ? useful / launched : 0.0) * fill + 1e-3 * widths;
}

// ---------------------------------------------------------------------------------------
// Direct kernels (kernel_family 1) and the MFMA helpers (kernel_family 2)
// ---------------------------------------------------------------------------------------

/// The kernel_family metadata value (KERNEL_FAMILY_* in convolution_backward_data.py).
enum class KernelFamily : int64_t
{
    IMPLICIT_GEMM = 0,
    DIRECT = 1,
    /// The MFMA pipeline's weight transforms: packaged so the pipeline can load them, never
    /// admitted by the kernel matcher.
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

/// The direct_variant metadata value; each belongs to one family.
enum class DirectVariant
{
    NONE, ///< "none": implicit GEMM.
    DEPTHWISE, ///< "depthwise_dgrad": DirectDepthwiseDgradStreamSpec.
    MFMA, ///< "grouped_dgrad_mfma": the transposed forward DirectConvSpec pipeline.
    TRANSPOSE, ///< "transpose": DirectTransposeWeightsDgradSpec (helper).
    REORGANIZE, ///< "reorganize": DirectReorganizeWeightsSpec (helper).
};

inline std::optional<DirectVariant> parseDirectVariant(std::string_view value)
{
    if(value == "none")
    {
        return DirectVariant::NONE;
    }
    if(value == "depthwise_dgrad")
    {
        return DirectVariant::DEPTHWISE;
    }
    if(value == "grouped_dgrad_mfma")
    {
        return DirectVariant::MFMA;
    }
    if(value == "transpose")
    {
        return DirectVariant::TRANSPOSE;
    }
    if(value == "reorganize")
    {
        return DirectVariant::REORGANIZE;
    }
    return std::nullopt;
}

inline bool isHelperVariant(DirectVariant variant)
{
    return variant == DirectVariant::TRANSPOSE || variant == DirectVariant::REORGANIZE;
}

/// What a direct or helper binary compiles in.
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

/// One direct or helper instance (the direct fields of Gfx950ConvDgradAotSpec). Each
/// variant reads only its own knobs; the rest stay 0.
struct DirectInstance
{
    DirectVariant variant;
    DirectCapabilities compiled;
    /// depthwise: dX columns per block and waves per block (one channel per lane).
    int64_t blockW = 0;
    int64_t blockWaves = 0;
    /// MFMA: the compute pass's row tile (the knobs-JSON block_h), output columns and
    /// groups per block.
    int64_t blockH = 0;
    int64_t blockQ = 0;
    int64_t blockGroups = 0;
    /// MFMA: 32-wide K atoms (also the reorganize helper's layout), waves along K and Q,
    /// and the compile-only persistent grid and runtime K loop.
    int64_t foldK32 = 0;
    int64_t wavesK = 0;
    int64_t wavesQ = 0;
    int64_t persistentGrid = 0;
    int64_t runtimeKLoop = 0;
};

/// The depthwise kernels' 20-argument dgrad direct ABI after (A*, B*, D*, A_bytes,
/// B_bytes, D_bytes) with A = dY, B = W, D = dX: conv_direct_arg_names(direction="dgrad")
/// in conv_abi.py.
constexpr std::array<std::string_view, 14> DEPTHWISE_KERNARG_NAMES{"p_N",
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
                                                                   "p_dX_stride_n",
                                                                   "p_dX_stride_hi",
                                                                   "p_dX_stride_wi"};

using DepthwiseKernargs = std::array<int32_t, DEPTHWISE_KERNARG_NAMES.size()>;

/// The values for DEPTHWISE_KERNARG_NAMES: ConvArgs.to_launch_values for the dgrad
/// DirectConvProblem. Bounded by the tensors' element counts.
inline DepthwiseKernargs depthwiseKernargs(const Problem& p, const Geometry& g)
{
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return DepthwiseKernargs{i32(p.n),
                             i32(p.hi),
                             i32(p.wi),
                             i32(g.ho),
                             i32(g.wo),
                             i32(p.groups),
                             i32(p.c),
                             i32(p.k),
                             i32(g.ho * g.wo * p.k),
                             i32(g.wo * p.k),
                             i32(p.k),
                             i32(p.hi * p.wi * p.c),
                             i32(p.wi * p.c),
                             i32(p.c)};
}

/// The MFMA compute pass runs the FORWARD direct ABI (the 20 names of forward's
/// DIRECT_KERNARG_NAMES, A/D strides) on the transposed problem.
using gfx950_conv_fwd::DIRECT_KERNARG_NAMES;
using MfmaComputeKernargs = std::array<int32_t, DIRECT_KERNARG_NAMES.size()>;

/// The compiled filter, stride, padding, dilation and channel counts must all be the
/// problem's (_direct_serves); the direct kernels compile one stride and one padding.
inline bool directCapabilitiesMatch(const DirectCapabilities& compiled,
                                    const Problem& problem,
                                    const Geometry& geometry)
{
    return problem.strideH == problem.strideW && problem.padH == problem.padW
           && compiled.filterH == problem.y && compiled.filterW == problem.x
           && compiled.strideH == problem.strideH && compiled.strideW == problem.strideW
           && compiled.padH == problem.padH && compiled.padW == problem.padW
           && compiled.dilationH == problem.dilationH && compiled.dilationW == problem.dilationW
           && compiled.cpg == geometry.cpg && compiled.kpg == geometry.kpg;
}

/// The depthwise launch: exact capability equality (KH == KW, one stride and padding,
/// dilation 1, cpg == kpg == 1, so C == K == groups), then rocKE's
/// (ceil(Wi / block_w), ceil(groups / (64 * block_waves)), N) grid of 64 * block_waves
/// threads with x a signed 32-bit extent and y, z at most 65535. One launch, no workspace,
/// no memset: the kernel writes every dX element.
inline std::optional<LaunchGeometry> depthwiseLaunch(const Problem& problem,
                                                     const Geometry& geometry,
                                                     const DirectInstance& instance)
{
    const auto& compiled = instance.compiled;
    if(instance.variant != DirectVariant::DEPTHWISE || compiled.filterH != compiled.filterW
       || compiled.dilationH != 1 || compiled.dilationW != 1 || compiled.cpg != 1
       || compiled.kpg != 1 || !directCapabilitiesMatch(compiled, problem, geometry)
       || instance.blockW < 1 || instance.blockW > I32_MAX || instance.blockWaves < 1
       || instance.blockWaves > MAX_BLOCK_THREADS / WAVE_SIZE)
    {
        return std::nullopt;
    }
    const auto threads = instance.blockWaves * WAVE_SIZE;
    const auto gridX = ceilDiv(problem.wi, instance.blockW);
    const auto gridY = ceilDiv(problem.groups, threads);
    if(gridX < 1 || gridX > I32_MAX || gridY < 1 || gridY > MAX_GRID_DIM || problem.n < 1
       || problem.n > MAX_GRID_DIM)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          static_cast<unsigned int>(problem.n),
                          static_cast<unsigned int>(threads)};
}

/// direct_kernel_sweep._uses_coalesced_weights: the compute pass reads the reorganized
/// (coalesced) weights, so the pipeline runs the reorganize helper.
inline bool usesCoalescedWeights(const DirectInstance& instance)
{
    return instance.wavesK > 1 || instance.runtimeKLoop != 0;
}

/// The K-atom width of the MFMA compute pass and its reorganized weights.
inline int64_t mfmaKAtom(const DirectInstance& instance)
{
    return instance.foldK32 != 0 ? 32 : 16;
}

/// _mfma_knobs_reason for a problem with @p kpg: with fold_k32, K / groups is a multiple
/// of 32 and there are kpg / 32 atoms, otherwise ceil(kpg / 16); the atoms split evenly
/// over waves_k. True when the knobs fit.
inline bool mfmaKnobsFit(const DirectInstance& instance, int64_t kpg)
{
    if(instance.wavesK < 1 || kpg < 1)
    {
        return false;
    }
    int64_t atoms = 0;
    if(instance.foldK32 != 0)
    {
        if(kpg % 32 != 0)
        {
            return false;
        }
        atoms = kpg / 32;
    }
    else
    {
        atoms = ceilDiv(kpg, 16);
    }
    return atoms > 0 && atoms % instance.wavesK == 0;
}

/// The MFMA pipeline on one problem: direct_kernel_sweep.plan_for's steps with the
/// scratch laid out in one workspace (gfx950_conv_dgrad_aot_mfma_plan).
struct MfmaPlan
{
    /// W -> ws_t: (groups * KH * KW, ceil(kpg / 64), cpg) x 64.
    LaunchGeometry transpose;
    /// ws_t -> ws_coa, when usesCoalescedWeights: (groups * KH * KW * ceil(kpg / k_atom)
    /// * ceil(cpg / 16), 1, 1) x 64.
    std::optional<LaunchGeometry> reorganize;
    /// (dY, ws, dX): (ceil(Wo' / block_q), groups / block_groups, N * ceil(Ho' / block_h))
    /// x block_groups * waves_q * waves_k * 64.
    LaunchGeometry compute;
    /// ws_t at offset 0, C * KH * KW * kpg * 2 bytes.
    int64_t wsTBytes;
    /// ws_coa at ws_t rounded up to 256, when reorganizing; 0 otherwise.
    int64_t wsCoaOffset;
    int64_t wsCoaBytes;
    /// The end of the last buffer.
    int64_t workspaceBytes;
    /// The transposed problem's output extents (stride 1, PAD' = KH - 1 - PAD).
    int64_t hoT;
    int64_t woT;
};

/// The (x, y, z) grid of @p threads, or std::nullopt when an axis is below 1, x above
/// @p xCap or its work-items past a 32-bit dispatch, or y or z above 65535.
inline std::optional<LaunchGeometry>
    boundedGrid(int64_t gridX, int64_t gridY, int64_t gridZ, int64_t threads, int64_t xCap)
{
    if(gridX < 1 || gridY < 1 || gridZ < 1 || gridX > xCap || gridY > MAX_GRID_DIM
       || gridZ > MAX_GRID_DIM || threads < 1 || threads > MAX_BLOCK_THREADS
       || gridX * threads > MAX_DISPATCH_ITEMS)
    {
        return std::nullopt;
    }
    return LaunchGeometry{static_cast<unsigned int>(gridX),
                          static_cast<unsigned int>(gridY),
                          static_cast<unsigned int>(gridZ),
                          static_cast<unsigned int>(threads)};
}

/// The MFMA instance's plan on @p problem, or std::nullopt when it does not serve it.
/// Mirrors _direct_serves for the MFMA variant, in order:
///   1. exact capability -- 3x3, one padding of 1, stride 1, dilation 1, and the
///      instance's (C / groups, K / groups);
///   2. _mfma_knobs_reason;
///   3. an instance using coalesced weights (waves_k > 1 or runtime_k_loop) serves
///      groups == 1 only: rocKE's build_direct_reorganize_weights adds the group offset to
///      the per-group K axis of ws_t a second time, so every group but the first gets
///      wrong weights (_coalesced_groups_reason);
///   4. the run-time half of rocKE's DirectConvSpec validator, groups % block_groups;
///   5. every launch fits: the compute grid at most 65535 on each axis, the transforms'
///      x a signed 32-bit extent (and a 32-bit dispatch), y and z 65535, and each
///      workspace a signed 32-bit size.
inline std::optional<MfmaPlan>
    mfmaLaunchPlan(const Problem& problem, const Geometry& geometry, const DirectInstance& instance)
{
    const auto& compiled = instance.compiled;
    if(instance.variant != DirectVariant::MFMA || compiled.filterH != 3 || compiled.filterW != 3
       || compiled.strideH != 1 || compiled.strideW != 1 || compiled.padH != 1 || compiled.padW != 1
       || compiled.dilationH != 1 || compiled.dilationW != 1
       || !directCapabilitiesMatch(compiled, problem, geometry))
    {
        return std::nullopt;
    }
    if(!mfmaKnobsFit(instance, geometry.kpg))
    {
        return std::nullopt;
    }
    const bool coalesced = usesCoalescedWeights(instance);
    if(coalesced && problem.groups != 1)
    {
        return std::nullopt;
    }
    if(instance.blockGroups < 1 || instance.blockH < 1 || instance.blockQ < 1 || instance.wavesQ < 1
       || problem.groups % instance.blockGroups != 0)
    {
        return std::nullopt;
    }
    const auto kh = compiled.filterH;
    const auto kw = compiled.filterW;
    // The compute pass is a forward convolution of dY with the transposed weights:
    // H' = Ho, W' = Wo, PAD' = KH - 1 - PAD, stride 1.
    const auto padT = kh - 1 - compiled.padH;
    const auto hoT = gfx950_conv_fwd::outputExtent(geometry.ho, kh, 1, padT, 1);
    const auto woT = gfx950_conv_fwd::outputExtent(geometry.wo, kw, 1, kw - 1 - compiled.padW, 1);
    if(!hoT || !woT)
    {
        return std::nullopt;
    }
    const auto taps = problem.groups * kh * kw;
    const auto transpose = boundedGrid(taps, ceilDiv(geometry.kpg, 64), geometry.cpg, 64, I32_MAX);
    std::optional<LaunchGeometry> reorganize;
    const auto kAtom = mfmaKAtom(instance);
    if(coalesced)
    {
        reorganize = boundedGrid(
            taps * ceilDiv(geometry.kpg, kAtom) * ceilDiv(geometry.cpg, 16), 1, 1, 64, I32_MAX);
        if(!reorganize)
        {
            return std::nullopt;
        }
    }
    const auto computeThreads
        = instance.blockGroups * instance.wavesQ * instance.wavesK * WAVE_SIZE;
    const auto compute = boundedGrid(ceilDiv(*woT, instance.blockQ),
                                     problem.groups / instance.blockGroups,
                                     problem.n * ceilDiv(*hoT, instance.blockH),
                                     computeThreads,
                                     MAX_GRID_DIM);
    if(!transpose || !compute)
    {
        return std::nullopt;
    }
    // ws_t holds C * KH * KW * kpg elements; ws_coa groups * KH * KW * ceil(kpg / k_atom)
    // * ceil(cpg / 16) atoms of 64 lanes, 8 (fold_k32) or 4 elements each.
    const auto wsT = boundedProduct({problem.c, kh, kw, geometry.kpg, 2});
    if(!wsT)
    {
        return std::nullopt;
    }
    int64_t wsCoaOffset = 0;
    int64_t wsCoa = 0;
    int64_t total = *wsT;
    if(coalesced)
    {
        const auto bytes = boundedProduct({problem.groups,
                                           kh,
                                           kw,
                                           ceilDiv(geometry.kpg, kAtom),
                                           ceilDiv(geometry.cpg, 16),
                                           64,
                                           instance.foldK32 != 0 ? 8 : 4,
                                           2});
        if(!bytes)
        {
            return std::nullopt;
        }
        wsCoa = *bytes;
        wsCoaOffset = ceilDiv(*wsT, MFMA_WORKSPACE_ALIGNMENT) * MFMA_WORKSPACE_ALIGNMENT;
        total = wsCoaOffset + wsCoa;
    }
    return MfmaPlan{*transpose, reorganize, *compute, *wsT, wsCoaOffset, wsCoa, total, *hoT, *woT};
}

/// The compute pass's values for DIRECT_KERNARG_NAMES: ConvArgs.to_launch_values for the
/// forward DirectConvProblem of the transposed problem -- H = Ho, W = Wo, Ho', Wo', the
/// total input channels K and output channels C, dY's NHWK strides as A and dX's NHWC
/// strides (over Ho' x Wo') as D.
inline MfmaComputeKernargs
    mfmaComputeKernargs(const Problem& p, const Geometry& g, const MfmaPlan& plan)
{
    const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
    return MfmaComputeKernargs{i32(p.n),
                               i32(g.ho),
                               i32(g.wo),
                               i32(plan.hoT),
                               i32(plan.woT),
                               i32(p.groups),
                               i32(p.k),
                               i32(p.c),
                               i32(g.ho * g.wo * p.k),
                               i32(g.wo * p.k),
                               i32(p.k),
                               i32(plan.hoT * plan.woT * p.c),
                               i32(plan.woT * p.c),
                               i32(p.c)};
}

/// The served predicate of a direct instance, and its single (depthwise) or compute
/// (MFMA) launch: depthwiseLaunch or mfmaLaunchPlan. A helper serves nothing on its own.
inline std::optional<LaunchGeometry> directInstanceLaunch(const Problem& problem,
                                                          const Geometry& geometry,
                                                          const DirectInstance& instance)
{
    switch(instance.variant)
    {
    case DirectVariant::DEPTHWISE:
        return depthwiseLaunch(problem, geometry, instance);
    case DirectVariant::MFMA:
    {
        const auto plan = mfmaLaunchPlan(problem, geometry, instance);
        return plan ? std::optional<LaunchGeometry>(plan->compute) : std::nullopt;
    }
    case DirectVariant::NONE:
    case DirectVariant::TRANSPOSE:
    case DirectVariant::REORGANIZE:
    default:
        return std::nullopt;
    }
}

/// Deterministic order of a direct instance that serves the problem, used only when
/// nothing is benchmarked, recorded or trained. Depthwise ranks above every other family
/// (implicit GEMM refuses grouped depthwise anyway), in [3, 3.5]; MFMA above implicit GEMM
/// (whose N tile is mostly idle at these few channels per group), in [2, 2.5). Within a
/// family the fraction of launched lanes that hold real output breaks ties.
inline double directInstanceScore(const Problem& problem,
                                  const Geometry& geometry,
                                  const DirectInstance& instance,
                                  const LaunchGeometry& launch)
{
    if(instance.variant == DirectVariant::DEPTHWISE)
    {
        const auto columns
            = static_cast<double>(problem.wi)
              / (static_cast<double>(launch.gridX) * static_cast<double>(instance.blockW));
        const auto channels
            = static_cast<double>(problem.groups)
              / (static_cast<double>(launch.gridY) * static_cast<double>(launch.blockX));
        return 3.0 + 0.5 * columns * channels;
    }
    if(instance.variant == DirectVariant::MFMA)
    {
        const auto columns
            = static_cast<double>(geometry.wo)
              / (static_cast<double>(launch.gridX) * static_cast<double>(instance.blockQ));
        const auto rows
            = static_cast<double>(geometry.ho * problem.n)
              / (static_cast<double>(launch.gridZ) * static_cast<double>(instance.blockH));
        return 2.0 + 0.49 * columns * rows;
    }
    return -1.0;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad
