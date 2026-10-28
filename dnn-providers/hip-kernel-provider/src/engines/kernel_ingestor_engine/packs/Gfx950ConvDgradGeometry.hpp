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

#include "engines/kernel_ingestor_engine/packs/Gfx950ConvFwdGeometry.hpp"

/// @file Gfx950ConvDgradGeometry.hpp
/// The launch contract of hipkernel:Gfx950ConvDgradAot, the shape-generic backward-data
/// (dgrad) AOT-instance engine: the instances its catalog describes, the problem it
/// matches, and the launch plan of one instance on one problem -- the served predicate,
/// the kernels it launches with their grids and arguments, the workspace and the dX
/// zero-fill. It ports gfx950_conv_dgrad_aot_plan of the rocKE adapter
/// rocke/library/builders/common/convolution_backward_data.py, which is rocKE's own offer
/// (identity_supports_problem, direct_spec_for_problem) plus the launch bounds; the unit
/// tests compare the two plan for plan by digest (serializePlan / planDigest) over the
/// shipped catalog.
namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad
{

// Pure helpers shared with the forward engine, unchanged.
using gfx950_conv_fwd::ceilDiv;
using gfx950_conv_fwd::LaunchGeometry;
using gfx950_conv_fwd::magicDivision;
using gfx950_conv_fwd::MagicDivision;

constexpr int64_t I32_MAX = std::numeric_limits<int32_t>::max();
constexpr int64_t I32_MIN = std::numeric_limits<int32_t>::min();
/// The hardware cap of grid y and z, and of every axis of the DirectConvSpec compute grids.
constexpr int64_t MAX_GRID_DIM = 65535;
/// A dispatch's x extent in work-items is 32-bit.
constexpr int64_t MAX_DISPATCH_ITEMS = (int64_t{1} << 32) - 1;
constexpr int64_t WAVE_SIZE = 64;
constexpr int64_t MAX_BLOCK_THREADS = 1024;
/// gfx950's LDS per workgroup (ArchTarget.lds_capacity_bytes).
constexpr int64_t LDS_CAPACITY_BYTES = 163840;

/// Record and magic-table widths of the implicit-GEMM sub_gemm_buf, in i32
/// (SUB_GEMM_RECORD_FIELDS, SUB_GEMM_MAGIC_FIELDS in convolution_backward_data.py).
constexpr int64_t SUB_GEMM_RECORD_FIELDS = 22;
constexpr int64_t SUB_GEMM_MAGIC_FIELDS = 8;
/// identity_supports_problem's bound for an identity that recorded no max_sub_gemms.
constexpr int64_t DEFAULT_MAX_SUB_GEMMS = 64;
/// The MFMA pipelines' ws_coa starts at ws_t's size rounded up to this.
constexpr int64_t MFMA_WORKSPACE_ALIGNMENT = 256;
/// conv_direct_grouped: the persistent grid's blocks and the rounds each walks.
constexpr int64_t DIRECT_PERSISTENT_BLOCKS = 256;
constexpr int64_t DIRECT_PERSISTENT_ROUNDS = 64;
/// conv_direct_grouped.XCD_TILES_MIN_ROW_BANDS.
constexpr int64_t XCD_TILES_MIN_ROW_BANDS = 8;
/// direct_variants: the VGPR budgets of the MFMA pipelines' preloaded weights.
constexpr int64_t MFMA_PRELOAD_VGPR_BUDGET = 384;
constexpr int64_t RTC_WIDE_Q_PRELOAD_VGPR_BUDGET = 216;
/// conv_direct_grouped: the windowed depthwise kernel's sentinel addressing range and the
/// channels one MFMA wave of it owns.
constexpr int64_t DW_DGRAD_WIN_MAX_TENSOR_BYTES = (int64_t{1} << 30) - 1;
constexpr int64_t DW_DGRAD_MFMA_CH = 8;

inline bool fitsI32(int64_t value)
{
    return value >= I32_MIN && value <= I32_MAX;
}

/// ceil(a / b) for any a and b > 0, as Python's -(-a // b).
inline int64_t ceilDivSigned(int64_t a, int64_t b)
{
    return a >= 0 ? ceilDiv(a, b) : -((-a) / b);
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

/// The product of @p factors when it is below 2^62, else std::nullopt (nonnegative
/// factors): the byte counts and offsets of a whole tensor.
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

/// The compiled pointer arguments are noalias. Check complete storage ranges, including
/// partially overlapping views, before passing them to the kernel.
inline bool bufferRangesDoNotOverlap(const std::array<uintptr_t, 3>& addresses,
                                     const std::array<int64_t, 3>& bytes)
{
    for(size_t i = 0; i < addresses.size(); ++i)
    {
        if(bytes[i] <= 0
           || addresses[i]
                  > std::numeric_limits<uintptr_t>::max() - static_cast<uintptr_t>(bytes[i]))
        {
            return false;
        }
        for(size_t j = 0; j < i; ++j)
        {
            // Subtract ordered addresses so even untrusted pointers cannot overflow.
            if(addresses[i] >= addresses[j]
                   ? addresses[i] - addresses[j] < static_cast<uintptr_t>(bytes[j])
                   : addresses[j] - addresses[i] < static_cast<uintptr_t>(bytes[i]))
            {
                return false;
            }
        }
    }
    return true;
}

// ---------------------------------------------------------------------------------------
// The problem
// ---------------------------------------------------------------------------------------

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

/// What the problem implies: the dY extents, the per-group channel counts, the whole
/// tensors' byte sizes (dY, W, dX) and one image's of dY and dX. A batch whose dY or dX
/// exceeds the 32-bit kernel ABI runs as batch slices (batchChunks), so only an image and
/// the filter have to fit it.
struct Geometry
{
    int64_t ho;
    int64_t wo;
    int64_t cpg;
    int64_t kpg;
    std::array<int64_t, 3> tensorBytes;
    int64_t dyImageBytes;
    int64_t dxImageBytes;
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
/// dY, one image of dY and dX and the filter within the signed 32-bit kernel ABI, and
/// every tensor below 2^62 bytes.
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
    const auto dyImage = boundedProduct({*ho, *wo, problem.k, 2});
    // W holds C / groups channels per filter; a total-C count would let a grouped kernel
    // read past the filter allocation.
    const auto wBytes = boundedProduct({problem.k, cpg, problem.y, problem.x, 2});
    const auto dxImage = boundedProduct({problem.hi, problem.wi, problem.c, 2});
    if(!dyImage || !wBytes || !dxImage)
    {
        return std::nullopt;
    }
    const auto dyBytes = wideProduct({problem.n, *dyImage});
    const auto dxBytes = wideProduct({problem.n, *dxImage});
    if(!dyBytes || !dxBytes)
    {
        return std::nullopt;
    }
    return Geometry{*ho, *wo, cpg, kpg, {*dyBytes, *wBytes, *dxBytes}, *dyImage, *dxImage};
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
    t.hTilde = p.y > 1 ? g.ho + ceilDiv(p.dilationH * (p.y - 1), p.strideH) : g.ho;
    t.wTilde = p.x > 1 ? g.wo + ceilDiv(p.dilationW * (p.x - 1), p.strideW) : g.wo;
    return t;
}

/// One non-empty sub-GEMM (SubGemmParams), every field already checked to fit i32.
struct SubGemm
{
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

/// is_patch_problem: every dX pixel takes at most one filter tap -- a 2-D ungrouped
/// convolution with no padding or dilation whose stride is at least its filter.
inline bool isPatchProblem(const Problem& p)
{
    return p.groups == 1 && p.padH == 0 && p.padW == 0 && p.dilationH == 1 && p.dilationW == 1
           && p.strideH >= p.y && p.strideW >= p.x;
}

/// patch_sub_gemms: the one record a patch kernel runs @p p with -- the pointwise
/// problem over (N, Ho, Wo) with Y * X * C channels, its tap strides the filter's and
/// its dX strides the problem's.
inline std::optional<std::vector<SubGemm>>
    patchSubGemms(const Problem& p, const Geometry& g, int64_t tileM, int64_t tileN, int64_t tileK)
{
    const auto channels = boundedProduct({p.y, p.x, p.c});
    if(!isPatchProblem(p) || !channels)
    {
        return std::nullopt;
    }
    const Problem pointwise{p.n, *channels, p.k, g.ho, g.wo, 1, 1, 1, 1, 0, 0, 1, 1, 1};
    const Geometry pointwiseGeometry{g.ho, g.wo, *channels, p.k, {}, 0, 0};
    auto subGemms = enumerateSubGemms(pointwise, pointwiseGeometry, tileM, tileN, tileK);
    if(!subGemms || subGemms->size() != 1)
    {
        return std::nullopt;
    }
    auto& sg = subGemms->front();
    sg.bYStride = p.y;
    sg.bXStride = p.x;
    sg.dHStride = p.strideH;
    sg.dWStride = p.strideW;
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
            buffer.push_back(static_cast<int32_t>(value));
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

/// width_fold: the largest f fold_width can fold the width by (1: none) -- f divides the
/// filter width, the W stride, the input width and the W padding, on a 2-D ungrouped
/// undilated-along-W problem.
inline int64_t widthFold(const Problem& p)
{
    if(p.groups != 1 || p.dilationW != 1)
    {
        return 1;
    }
    return std::gcd(std::gcd(p.x, p.strideW), std::gcd(p.wi, p.padW));
}

/// gfx950_conv_dgrad_aot_launch_problem: the problem the implicit-GEMM kernels run @p p
/// as -- its width folded into the channels when it folds (fold_width: f adjacent W
/// positions become f * C channels, X / f taps, stride sW / f, padding pW / f, width
/// Wi / f; the same bytes in all three tensors).
inline Problem implicitLaunchProblem(const Problem& p)
{
    const auto f = widthFold(p);
    if(f <= 1)
    {
        return p;
    }
    Problem folded = p;
    folded.wi = p.wi / f;
    folded.c = p.c * f;
    folded.x = p.x / f;
    folded.strideW = p.strideW / f;
    folded.padW = p.padW / f;
    return folded;
}

/// gfx950_conv_dgrad_aot_batch_chunks: the (n0, n) batch slices one implicit-GEMM launch
/// each covers, one slice when dY and dX fit the 32-bit kernel ABI whole. std::nullopt
/// when one image does not.
inline std::optional<std::vector<std::pair<int64_t, int64_t>>> batchChunks(const Problem& p,
                                                                           const Geometry& g)
{
    const auto perImage = std::max(g.dyImageBytes, g.dxImageBytes);
    if(perImage > I32_MAX)
    {
        return std::nullopt;
    }
    if(perImage * p.n <= I32_MAX)
    {
        return std::vector<std::pair<int64_t, int64_t>>{{0, p.n}};
    }
    const auto chunks = ceilDiv(p.n, I32_MAX / perImage);
    const auto step = ceilDiv(p.n, chunks);
    std::vector<std::pair<int64_t, int64_t>> out;
    for(int64_t n0 = 0; n0 < p.n; n0 += step)
    {
        out.emplace_back(n0, std::min(step, p.n - n0));
    }
    return out;
}

/// Whether the (offset, stride, count, width) runs -- width positions from each of
/// offset + stride * t, t < count -- cover every position of [0, extent).
struct CoverageRun
{
    int64_t offset;
    int64_t stride;
    int64_t count;
    int64_t width;
};

inline bool coversExtent(int64_t extent, const std::vector<CoverageRun>& runs)
{
    std::vector<char> hit(static_cast<size_t>(extent), 0);
    for(const auto& run : runs)
    {
        for(int64_t t = 0; t < run.count; ++t)
        {
            const auto start = run.offset + run.stride * t;
            const auto first = std::max<int64_t>(start, 0);
            const auto last = std::min(start + run.width, extent);
            for(auto pos = first; pos < last; ++pos)
            {
                hit[static_cast<size_t>(pos)] = 1;
            }
        }
    }
    return std::all_of(hit.begin(), hit.end(), [](char value) { return value != 0; });
}

/// _dx_fully_written: whether the implicit-GEMM kernel stores every dX element. A tilde
/// record stores the grid of dX rows d_h_offset + d_h_stride * t (t < h_tilde_slice) by
/// columns d_w_offset + d_w_stride * t (t < w_tilde_slice); which records exist depends
/// on each axis' tilde residue alone, so the records cover dX exactly when the union of
/// their rows covers [0, Hi) and of their columns [0, Wi). A patch record stores a whole
/// Y x X patch from each of its pixels.
inline bool dxFullyWritten(const Problem& p, const std::vector<SubGemm>& subGemms, bool patch)
{
    std::vector<CoverageRun> rows;
    std::vector<CoverageRun> cols;
    for(const auto& sg : subGemms)
    {
        rows.push_back({sg.dHOffset, sg.dHStride, sg.hTildeSlice, patch ? p.y : 1});
        cols.push_back({sg.dWOffset, sg.dWStride, sg.wTildeSlice, patch ? p.x : 1});
    }
    return coversExtent(p.hi, rows) && coversExtent(p.wi, cols);
}

// ---------------------------------------------------------------------------------------
// Instances
// ---------------------------------------------------------------------------------------

/// The kernel_family metadata value (KERNEL_FAMILY_* in convolution_backward_data.py).
enum class KernelFamily : int64_t
{
    IMPLICIT_GEMM = 0,
    DIRECT = 1,
    /// The MFMA pipelines' weight transforms: packaged so the pipelines can load them,
    /// never admitted by the kernel matcher.
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
    DEPTHWISE_WIN, ///< "depthwise_dgrad_win": DirectDepthwiseDgradWindowedSpec.
    MFMA, ///< "grouped_dgrad_mfma": weight transforms, then a forward DirectConvSpec.
    MFMA_RTC, ///< "dgrad_mfma_rtc": the same with runtime channel counts.
    FUSED, ///< "grouped_dgrad_fused": a forward DirectConvSpec reading the original W.
    FOUR_C, ///< "grouped_dgrad_4c": a forward DirectConv4cSpec reading the original W.
    TRANSPOSE, ///< "transpose": DirectTransposeWeightsDgradSpec (helper).
    REORGANIZE, ///< "reorganize": DirectReorganizeWeightsSpec (helper).
    TRANSPOSE_RTC, ///< "transpose_rtc": the runtime-channel transpose (helper).
    REORGANIZE_RTC, ///< "reorganize_rtc": the runtime-channel reorganize (helper).
};

inline std::optional<DirectVariant> parseDirectVariant(std::string_view value)
{
    static constexpr std::array<std::pair<std::string_view, DirectVariant>, 11> NAMES{{
        {"none", DirectVariant::NONE},
        {"depthwise_dgrad", DirectVariant::DEPTHWISE},
        {"depthwise_dgrad_win", DirectVariant::DEPTHWISE_WIN},
        {"grouped_dgrad_mfma", DirectVariant::MFMA},
        {"dgrad_mfma_rtc", DirectVariant::MFMA_RTC},
        {"grouped_dgrad_fused", DirectVariant::FUSED},
        {"grouped_dgrad_4c", DirectVariant::FOUR_C},
        {"transpose", DirectVariant::TRANSPOSE},
        {"reorganize", DirectVariant::REORGANIZE},
        {"transpose_rtc", DirectVariant::TRANSPOSE_RTC},
        {"reorganize_rtc", DirectVariant::REORGANIZE_RTC},
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

inline bool isHelperVariant(DirectVariant variant)
{
    return variant == DirectVariant::TRANSPOSE || variant == DirectVariant::REORGANIZE
           || variant == DirectVariant::TRANSPOSE_RTC || variant == DirectVariant::REORGANIZE_RTC;
}

inline bool isMfmaVariant(DirectVariant variant)
{
    return variant == DirectVariant::MFMA || variant == DirectVariant::MFMA_RTC;
}

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
    int64_t vectorB; ///< W load width, along C / groups (Y * X * C for a patch kernel).
    int64_t vectorC; ///< dX store width.
    int64_t maxSubGemms; ///< Records the kernel's table walk supports.
    bool grouped; ///< Built with the grouped code path, which also serves groups == 1.
    /// Geometry the binary was compiled for, or 0 where it reads the value from its
    /// sub-GEMM records at run time. A stated extent serves only that extent: the code
    /// indexes with it, so another problem would compute wrong dX rather than fail.
    int64_t filterH = 0;
    int64_t filterW = 0;
    int64_t strideH = 0;
    int64_t strideW = 0;
    int64_t dilationH = 0;
    int64_t dilationW = 0;
    /// The dY halo mode of the tap-outer K loop: the output must be the input's size and
    /// at most haloMaxWo wide.
    int64_t dyHalo = 0;
    int64_t haloMaxWo = 0;
    /// A patch embedding as one GEMM over the filter's (y, x, c) (patchSubGemms).
    bool patch = false;
    /// The stride-1 tap-outer K loop: every K tile inside one tap (kpg % tile_k == 0).
    bool tapOuterK = false;
};

/// What a direct or helper binary compiles in. 0 cpg / kpg: runtime channel counts
/// (the runtime-channel pipeline and its transforms).
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

/// The knobs of a direct or helper instance, as the descriptor states them: 0 where the
/// instance left a knob to rocKE's default (the effective* helpers below apply it).
struct DirectKnobs
{
    int64_t blockW = 0;
    int64_t blockWaves = 0;
    int64_t blockH = 0;
    int64_t blockQ = 0;
    int64_t blockGroups = 0;
    int64_t foldK32 = 0;
    int64_t wavesK = 0;
    int64_t wavesQ = 0;
    int64_t persistentGrid = 0;
    int64_t runtimeKLoop = 0;
    int64_t chPerLane = 0;
    int64_t dot2 = 0;
    int64_t mfma = 0;
    int64_t prefetchRows = 0;
    int64_t wFold = 0;
    int64_t kAtoms = 0;
    int64_t mTiles = 0;
    int64_t dgradWeightsLds = 0;
    int64_t ldsOnlySync = 0;
    int64_t ldsPad = 0;
    int64_t rowExit = 0;
    int64_t stageOut = 0;
    int64_t wavesPerEu = 0;
    int64_t xcdTiles = 0;
    int64_t stageRows = 0;
};

/// The knob fields in KMD order (_KNOB_DEFAULTS in convolution_backward_data.py), each
/// with its member.
constexpr std::array<std::pair<std::string_view, int64_t DirectKnobs::*>, 25> DIRECT_KNOB_FIELDS{{
    {"block_w", &DirectKnobs::blockW},
    {"block_waves", &DirectKnobs::blockWaves},
    {"block_h", &DirectKnobs::blockH},
    {"block_q", &DirectKnobs::blockQ},
    {"block_groups", &DirectKnobs::blockGroups},
    {"fold_k32", &DirectKnobs::foldK32},
    {"waves_k", &DirectKnobs::wavesK},
    {"waves_q", &DirectKnobs::wavesQ},
    {"persistent_grid", &DirectKnobs::persistentGrid},
    {"runtime_k_loop", &DirectKnobs::runtimeKLoop},
    {"ch_per_lane", &DirectKnobs::chPerLane},
    {"dot2", &DirectKnobs::dot2},
    {"mfma", &DirectKnobs::mfma},
    {"prefetch_rows", &DirectKnobs::prefetchRows},
    {"w_fold", &DirectKnobs::wFold},
    {"k_atoms", &DirectKnobs::kAtoms},
    {"m_tiles", &DirectKnobs::mTiles},
    {"dgrad_weights_lds", &DirectKnobs::dgradWeightsLds},
    {"lds_only_sync", &DirectKnobs::ldsOnlySync},
    {"lds_pad", &DirectKnobs::ldsPad},
    {"row_exit", &DirectKnobs::rowExit},
    {"stage_out", &DirectKnobs::stageOut},
    {"waves_per_eu", &DirectKnobs::wavesPerEu},
    {"xcd_tiles", &DirectKnobs::xcdTiles},
    {"stage_rows", &DirectKnobs::stageRows},
}};

/// The 0/1 knobs (_BOOL_KNOBS).
constexpr std::array<std::string_view, 11> DIRECT_FLAG_KNOBS{"fold_k32",
                                                             "persistent_grid",
                                                             "runtime_k_loop",
                                                             "dot2",
                                                             "mfma",
                                                             "dgrad_weights_lds",
                                                             "lds_only_sync",
                                                             "row_exit",
                                                             "stage_out",
                                                             "xcd_tiles",
                                                             "stage_rows"};

/// Whether @p variant states @p knob (AOT_DGRAD_DIRECT_KNOBS); every other knob holds 0.
/// A transpose states fold_k32 too, always 0 (the helper lookup keys on it).
inline bool directVariantUses(DirectVariant variant, std::string_view knob)
{
    const auto among = [&](std::initializer_list<std::string_view> knobs) {
        return std::find(knobs.begin(), knobs.end(), knob) != knobs.end();
    };
    const std::initializer_list<std::string_view> mfmaKnobs{"block_groups",
                                                            "block_h",
                                                            "block_q",
                                                            "fold_k32",
                                                            "persistent_grid",
                                                            "runtime_k_loop",
                                                            "waves_k",
                                                            "waves_q"};
    switch(variant)
    {
    case DirectVariant::DEPTHWISE:
        return among({"block_w", "block_waves"});
    case DirectVariant::DEPTHWISE_WIN:
        return among({"block_h",
                      "block_w",
                      "block_waves",
                      "ch_per_lane",
                      "dot2",
                      "mfma",
                      "prefetch_rows",
                      "w_fold"});
    case DirectVariant::MFMA:
        return among(mfmaKnobs);
    case DirectVariant::MFMA_RTC:
        return among(mfmaKnobs) || among({"k_atoms", "m_tiles"});
    case DirectVariant::FUSED:
        return among({"block_groups",
                      "block_h",
                      "block_q",
                      "dgrad_weights_lds",
                      "fold_k32",
                      "lds_only_sync",
                      "lds_pad",
                      "prefetch_rows",
                      "row_exit",
                      "stage_out",
                      "waves_per_eu",
                      "xcd_tiles"});
    case DirectVariant::FOUR_C:
        return among({"block_groups", "block_q", "dgrad_weights_lds", "stage_rows"});
    case DirectVariant::TRANSPOSE:
    case DirectVariant::TRANSPOSE_RTC:
    case DirectVariant::REORGANIZE:
    case DirectVariant::REORGANIZE_RTC:
        return knob == "fold_k32";
    case DirectVariant::NONE:
    default:
        return false;
    }
}

/// One direct or helper instance (the direct fields of Gfx950ConvDgradAotSpec).
struct DirectInstance
{
    DirectVariant variant;
    DirectCapabilities compiled;
    DirectKnobs knobs{};
};

/// A knob's value, or rocKE's spec default where the instance left it at 0.
inline int64_t knobOr(int64_t value, int64_t fallback)
{
    return value != 0 ? value : fallback;
}

// ---------------------------------------------------------------------------------------
// Reading an instance from its metadata
// ---------------------------------------------------------------------------------------

/// The implicit-GEMM tuning fields; a direct or helper kernel holds 0 in each.
constexpr std::array<std::string_view, 8> GEMM_TUNING_FIELDS{
    "tile_m", "tile_n", "tile_k", "warp_m", "warp_n", "warp_tile_m", "warp_tile_n", "warp_tile_k"};
/// The implicit-GEMM fields after the tuning; a direct or helper kernel holds 0 in each.
constexpr std::array<std::string_view, 6> GEMM_ONLY_FIELDS{
    "vector_size_a", "vector_size_b", "vector_size_c", "lds_k_outer", "max_sub_gemms", "grouped"};
/// The implicit-GEMM tuning flags and values past the widths; a direct or helper kernel
/// holds 0 in each.
constexpr std::array<std::string_view, 7> GEMM_FORM_FIELDS{"dy_halo",
                                                           "dy_halo_setprio",
                                                           "dy_halo_kouter_pad",
                                                           "halo_max_wo",
                                                           "patch",
                                                           "tap_outer_k",
                                                           "uniform_k_decode"};
/// What a direct or helper binary compiles in, in DirectCapabilities order.
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
/// The capabilities an implicit-GEMM binary may state, in ImplicitInstance order, and the
/// ones it must leave at 0. Together these partition DIRECT_CAPABILITY_FIELDS.
constexpr std::array<std::string_view, 6> IMPLICIT_BAKEABLE_FIELDS{
    "filter_h", "filter_w", "stride_h", "stride_w", "dilation_h", "dilation_w"};
constexpr std::array<std::string_view, 4> IMPLICIT_RUNTIME_ONLY_FIELDS{
    "pad_h", "pad_w", "cpg", "kpg"};

/// One instance of either family.
struct AotInstance
{
    KernelFamily family;
    std::optional<ImplicitInstance> implicit;
    std::optional<DirectInstance> direct;
};

/// The instance a kernel's metadata describes (supports_gfx950_conv_dgrad_aot), when it is
/// a complete instance of one family with a reviewed launch contract. @p metadata answers
/// intField(name) -> std::optional<int64_t> and stringField(name) -> const std::string*.
/// kernel_family is read first, so no family's checks run on another's placeholders; a
/// missing or malformed field refuses the kernel.
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
    if(!equalsString("layout", "NHWC") || !equalsInt("wave_size", WAVE_SIZE))
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
    const auto isZero = [&](std::string_view field) { return equalsInt(field, 0); };
    const auto intIn = [&](std::string_view field, int64_t low, int64_t high) {
        const auto value = metadata.intField(field);
        return value && *value >= low && *value <= high ? value : std::nullopt;
    };

    if(*family == KernelFamily::IMPLICIT_GEMM)
    {
        const auto* pipeline = metadata.stringField("pipeline");
        const auto* epilogue = metadata.stringField("epilogue");
        if(*variant != DirectVariant::NONE || pipeline == nullptr || epilogue == nullptr
           || (*pipeline != "mem" && *pipeline != "compv3" && *pipeline != "compv4")
           || (*epilogue != "default" && *epilogue != "cshuffle")
           || !std::all_of(
               IMPLICIT_RUNTIME_ONLY_FIELDS.begin(), IMPLICIT_RUNTIME_ONLY_FIELDS.end(), isZero)
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
        for(size_t i = 0; i < vectors.size(); ++i)
        {
            const auto value = metadata.intField(GEMM_ONLY_FIELDS[i]);
            if(!value || (*value != 1 && *value != 2 && *value != 4 && *value != 8))
            {
                return std::nullopt;
            }
            vectors[i] = *value;
        }
        std::array<int64_t, IMPLICIT_BAKEABLE_FIELDS.size()> baked{};
        for(size_t i = 0; i < IMPLICIT_BAKEABLE_FIELDS.size(); ++i)
        {
            // 0 is "runtime"; anything else is an extent the binary indexes with.
            const auto value = intIn(IMPLICIT_BAKEABLE_FIELDS[i], 0, I32_MAX);
            if(!value)
            {
                return std::nullopt;
            }
            baked[i] = *value;
        }
        // Stated per axis pair, as the builder writes them.
        if((baked[0] == 0) != (baked[1] == 0) || (baked[2] == 0) != (baked[3] == 0)
           || (baked[4] == 0) != (baked[5] == 0))
        {
            return std::nullopt;
        }
        const auto ldsKOuter = intIn("lds_k_outer", 0, 1);
        const auto grouped = intIn("grouped", 0, 1);
        const auto maxSubGemms = intIn("max_sub_gemms", 1, I32_MAX);
        const auto dyHalo = intIn("dy_halo", 0, I32_MAX);
        const auto haloMaxWo = intIn("halo_max_wo", 0, I32_MAX);
        const auto setprio = intIn("dy_halo_setprio", 0, I32_MAX);
        const auto kouterPad = intIn("dy_halo_kouter_pad", 0, I32_MAX);
        const auto patch = intIn("patch", 0, 1);
        const auto tapOuterK = intIn("tap_outer_k", 0, 1);
        const auto uniformKDecode = intIn("uniform_k_decode", 0, 1);
        if(!ldsKOuter || !grouped || !maxSubGemms || !dyHalo || !haloMaxWo || !setprio || !kouterPad
           || !patch || !tapOuterK
           || !uniformKDecode
           // A halo binary is sized for the widest output it serves.
           || ((*dyHalo != 0) != (*haloMaxWo != 0)))
        {
            return std::nullopt;
        }
        ImplicitInstance instance{tuning[0],
                                  tuning[1],
                                  tuning[2],
                                  tuning[3],
                                  tuning[4],
                                  vectors[0],
                                  vectors[1],
                                  vectors[2],
                                  *maxSubGemms,
                                  *grouped == 1};
        instance.filterH = baked[0];
        instance.filterW = baked[1];
        instance.strideH = baked[2];
        instance.strideW = baked[3];
        instance.dilationH = baked[4];
        instance.dilationW = baked[5];
        instance.dyHalo = *dyHalo;
        instance.haloMaxWo = *haloMaxWo;
        instance.patch = *patch == 1;
        instance.tapOuterK = *tapOuterK == 1;
        if(instance.warpM > MAX_BLOCK_THREADS / WAVE_SIZE
           || instance.warpN > MAX_BLOCK_THREADS / (WAVE_SIZE * instance.warpM))
        {
            return std::nullopt;
        }
        return AotInstance{*family, instance, std::nullopt};
    }

    const bool helper = isHelperVariant(*variant);
    if(*variant == DirectVariant::NONE || (*family == KernelFamily::HELPER) != helper
       || !equalsString("pipeline", "direct") || !equalsString("epilogue", "direct")
       || !std::all_of(GEMM_TUNING_FIELDS.begin(), GEMM_TUNING_FIELDS.end(), isZero)
       || !std::all_of(GEMM_ONLY_FIELDS.begin(), GEMM_ONLY_FIELDS.end(), isZero)
       || !std::all_of(GEMM_FORM_FIELDS.begin(), GEMM_FORM_FIELDS.end(), isZero))
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
    const DirectCapabilities capabilities{compiled[0],
                                          compiled[1],
                                          compiled[2],
                                          compiled[3],
                                          compiled[4],
                                          compiled[5],
                                          compiled[6],
                                          compiled[7],
                                          compiled[8],
                                          compiled[9]};
    const bool runtimeChannels = *variant == DirectVariant::MFMA_RTC
                                 || *variant == DirectVariant::TRANSPOSE_RTC
                                 || *variant == DirectVariant::REORGANIZE_RTC;
    if(capabilities.filterH < 1 || capabilities.filterH != capabilities.filterW
       || capabilities.dilationH != 1
       || capabilities.dilationW != 1
       // 0 channel counts are runtime, and only the runtime-channel kernels take them.
       || (capabilities.cpg == 0) != (capabilities.kpg == 0)
       || (capabilities.cpg == 0) != runtimeChannels)
    {
        return std::nullopt;
    }
    if(helper)
    {
        // A helper identity zeroes stride and padding: its code reads neither.
        if(capabilities.strideH != 0 || capabilities.strideW != 0 || capabilities.padH != 0
           || capabilities.padW != 0)
        {
            return std::nullopt;
        }
    }
    else if(capabilities.strideH < 1 || capabilities.strideH != capabilities.strideW
            || capabilities.padH != capabilities.padW)
    {
        return std::nullopt;
    }
    if((*variant == DirectVariant::DEPTHWISE || *variant == DirectVariant::DEPTHWISE_WIN)
       && (capabilities.cpg != 1 || capabilities.kpg != 1))
    {
        return std::nullopt;
    }
    DirectKnobs knobs{};
    for(const auto& [field, member] : DIRECT_KNOB_FIELDS)
    {
        const auto value = metadata.intField(field);
        if(!value)
        {
            return std::nullopt;
        }
        const bool flag = std::find(DIRECT_FLAG_KNOBS.begin(), DIRECT_FLAG_KNOBS.end(), field)
                          != DIRECT_FLAG_KNOBS.end();
        if(!directVariantUses(*variant, field))
        {
            if(*value != 0)
            {
                return std::nullopt;
            }
        }
        else if(flag ? (*value != 0 && *value != 1) : (*value < 0 || *value > I32_MAX))
        {
            return std::nullopt;
        }
        knobs.*member = *value;
    }
    // A transpose writes the plain layout; only a reorganize helper is keyed on fold_k32.
    if((*variant == DirectVariant::TRANSPOSE || *variant == DirectVariant::TRANSPOSE_RTC)
       && knobs.foldK32 != 0)
    {
        return std::nullopt;
    }
    return AotInstance{*family, std::nullopt, DirectInstance{*variant, capabilities, knobs}};
}

// ---------------------------------------------------------------------------------------
// The launch plan (gfx950_conv_dgrad_aot_plan)
// ---------------------------------------------------------------------------------------

/// The kernel a step launches: the instance itself or a helper of its pack
/// (PLAN_KERNEL_CODES).
enum class PlanKernel : int64_t
{
    SELF = 0,
    TRANSPOSE = 1,
    REORGANIZE = 2,
    TRANSPOSE_RTC = 3,
    REORGANIZE_RTC = 4,
};

/// The ABI a step launches with (PLAN_ABI_CODES).
enum class Abi : int64_t
{
    /// (dY*, W*, dX*, dY_bytes, W_bytes, dX_bytes), 36 i32, sub_gemm_buf*, num_sub_gemms.
    IMPLICIT = 0,
    /// (A*, B*, D*, A_bytes, B_bytes, D_bytes) then 14 i32: the depthwise kernels.
    DIRECT_DGRAD = 1,
    /// The forward direct block on the transposed problem.
    DIRECT_FWD = 2,
    /// (A*, D*, A_bytes, D_bytes).
    TRANSFORM = 3,
    /// The same, then p_cpg and p_kpg.
    TRANSFORM_RTC = 4,
};

/// The buffers a step binds (PLAN_BUFFER_CODES).
enum class Buffer : int64_t
{
    NONE = -1,
    DY = 0,
    W = 1,
    DX = 2,
    SUB_GEMM = 3,
    WS_T = 4,
    WS_COA = 5,
};

/// A window of a buffer: a batch-split launch binds only its images.
struct View
{
    int64_t offset;
    int64_t bytes;
};

/// One launch of a plan (Gfx950ConvDgradStep). values are the by-value i32 arguments
/// after the operands and their byte sizes, in ABI order (for implicit GEMM the 36
/// problem values then num_sub_gemms). An implicit-GEMM step's sub_gemm_buf is its table,
/// uploaded at tableOffset bytes into the SUB_GEMM region.
struct Step
{
    PlanKernel kernel;
    Abi abi;
    std::array<int64_t, 3> grid;
    int64_t block;
    std::array<Buffer, 3> buffers;
    std::array<std::optional<View>, 3> views;
    std::vector<int32_t> values;
    std::vector<int32_t> table;
    int64_t tableOffset = 0;
};

/// A step over whole buffers and without a table.
inline Step makeStep(PlanKernel kernel,
                     Abi abi,
                     const std::array<int64_t, 3>& grid,
                     int64_t block,
                     const std::array<Buffer, 3>& buffers,
                     std::vector<int32_t> values)
{
    Step step{};
    step.kernel = kernel;
    step.abi = abi;
    step.grid = grid;
    step.block = block;
    step.buffers = buffers;
    step.values = std::move(values);
    return step;
}

struct Workspace
{
    Buffer name;
    int64_t offset;
    int64_t bytes;
};

/// Everything the engine does to run one instance on one problem (Gfx950ConvDgradPlan):
/// zero dX when zeroDx, upload every step's table, then launch the steps. launched is the
/// problem the kernels see (the width-folded one for implicit GEMM).
struct Plan
{
    Problem launched;
    Geometry geometry;
    std::vector<Step> steps;
    std::vector<Workspace> workspaces;
    bool zeroDx = false;

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

/// Gfx950ConvDgradPlan.serialize: the plan as one integer sequence.
inline std::vector<int64_t> serializePlan(const Plan& plan)
{
    std::vector<int64_t> out{
        plan.zeroDx ? 1 : 0, plan.workspaceBytes(), static_cast<int64_t>(plan.workspaces.size())};
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
        for(const auto& view : step.views)
        {
            out.push_back(view ? view->offset : -1);
            out.push_back(view ? view->bytes : -1);
        }
        out.push_back(static_cast<int64_t>(step.values.size()));
        out.insert(out.end(), step.values.begin(), step.values.end());
        out.push_back(step.tableOffset);
        out.push_back(static_cast<int64_t>(step.table.size()));
        out.insert(out.end(), step.table.begin(), step.table.end());
    }
    return out;
}

/// Gfx950ConvDgradPlan.digest: FNV-1a 64 over serializePlan, each value as 8
/// little-endian bytes.
inline uint64_t planDigest(const Plan& plan)
{
    uint64_t hash = 0xCBF29CE484222325ULL;
    for(const auto value : serializePlan(plan))
    {
        auto bits = static_cast<uint64_t>(value);
        for(int byte = 0; byte < 8; ++byte)
        {
            hash ^= (bits >> (8 * byte)) & 0xFFU;
            hash *= 0x100000001B3ULL;
        }
    }
    return hash;
}

/// _grid_reason: a grid of @p block-thread workgroups launches when every axis is
/// positive, x is at most @p xCap and a 32-bit dispatch, and y and z at most 65535.
inline bool gridFits(const std::array<int64_t, 3>& grid, int64_t block, int64_t xCap)
{
    return block >= 1 && block <= MAX_BLOCK_THREADS && grid[0] >= 1 && grid[1] >= 1 && grid[2] >= 1
           && grid[0] <= xCap && grid[0] * block <= MAX_DISPATCH_ITEMS && grid[1] <= MAX_GRID_DIM
           && grid[2] <= MAX_GRID_DIM;
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

/// The values for IMPLICIT_GEMM_KERNARG_NAMES: ConvArgs.to_launch_values for a dgrad
/// ConvProblem. The GEMM is M = N * Hi * Wi, N = C / groups, K = Y * X * K / groups; the
/// strides are packed NHWK dY, KYXC W and NHWC dX in elements; the magic pairs divide by
/// Hi and Wi; num_pid_m / num_pid_n are the whole problem's tile counts for the
/// instance's own tile. std::nullopt when a value leaves i32.
inline std::optional<std::vector<int32_t>>
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
    const std::array<int64_t, 36> values{p.n,
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
                                         *gemmM,
                                         g.cpg,
                                         *gemmK,
                                         g.ho * g.wo * p.k,
                                         g.wo * p.k,
                                         p.k,
                                         p.y * p.x * g.cpg,
                                         p.x * g.cpg,
                                         g.cpg,
                                         p.hi * p.wi * p.c,
                                         p.wi * p.c,
                                         p.c,
                                         hi->multiplier,
                                         hi->shift,
                                         wi->multiplier,
                                         wi->shift,
                                         ceilDiv(*gemmM, tileM),
                                         ceilDiv(g.cpg, tileN)};
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

/// The 14 values after (A*, B*, D*, A_bytes, B_bytes, D_bytes) of the direct dgrad ABI
/// (conv_direct_arg_names(direction="dgrad")): ConvArgs for the dgrad DirectConvProblem.
constexpr std::array<std::string_view, 14> DIRECT_DGRAD_KERNARG_NAMES{"p_N",
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

/// The forward direct ABI (A, D strides) the fused, 4c and MFMA compute passes take on the
/// transposed problem.
using gfx950_conv_fwd::DIRECT_KERNARG_NAMES;

/// directValues: the 14 values of the direct dgrad block (DIRECT_DGRAD_KERNARG_NAMES)
/// for the dgrad problem -- its extents, then dY's NHWK strides and dX's NHWC strides.
inline std::vector<int32_t> directDgradValues(const Problem& p, const Geometry& g)
{
    // Bounded by the tensors' element counts, which the direct plans keep within i32.
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

/// The forward problem the fused, 4c and MFMA compute passes run
/// (dgrad_4c_transposed_problem, make_dgrad_fprop_spec): H = Ho, W = Wo, stride 1,
/// PAD' = KH - 1 - PAD, cpg' = kpg and kpg' = cpg.
struct TransposedProblem
{
    int64_t n;
    int64_t h;
    int64_t w;
    int64_t ho;
    int64_t wo;
    int64_t groups;
    int64_t cpg; ///< The original kpg.
    int64_t kpg; ///< The original cpg.
    int64_t kh;
    int64_t kw;
    int64_t pad;
};

inline std::optional<TransposedProblem> transposedProblem(const Problem& p, const Geometry& g)
{
    const auto padT = p.y - 1 - p.padH;
    if(padT < 0)
    {
        return std::nullopt;
    }
    const auto hoT = dgradOutputExtent(g.ho, p.y, 1, padT, 1);
    const auto woT = dgradOutputExtent(g.wo, p.x, 1, padT, 1);
    if(!hoT || !woT)
    {
        return std::nullopt;
    }
    return TransposedProblem{p.n, g.ho, g.wo, *hoT, *woT, p.groups, g.kpg, g.cpg, p.y, p.x, padT};
}

/// The forward direct block (DIRECT_KERNARG_NAMES) of the transposed problem: its
/// extents, its input's (dY's) NHWC strides over H x W with K channels, then its
/// output's (dX's) NHWK strides over Ho' x Wo' with C channels.
inline std::vector<int32_t> transposedValues(const TransposedProblem& t)
{
    const auto totalC = t.groups * t.cpg;
    const auto totalK = t.groups * t.kpg;
    const std::array<int64_t, 14> values{t.n,
                                         t.h,
                                         t.w,
                                         t.ho,
                                         t.wo,
                                         t.groups,
                                         totalC,
                                         totalK,
                                         t.h * t.w * totalC,
                                         t.w * totalC,
                                         totalC,
                                         t.ho * t.wo * totalK,
                                         t.wo * totalK,
                                         totalK};
    return std::vector<int32_t>(values.begin(), values.end());
}

// ---- implicit GEMM -------------------------------------------------------------------

/// identity_supports_problem for an implicit-GEMM dgrad binary on the launched chunk
/// @p p, then its records (dgrad_sub_gemms): std::nullopt when the binary does not
/// compute @p p.
inline std::optional<std::vector<SubGemm>>
    implicitSubGemms(const Problem& p, const Geometry& g, const ImplicitInstance& instance)
{
    // The operands' contiguous per-group extents: dY along kpg, W along cpg (the filter's
    // Y * X * C for a patch kernel) and dX along cpg (a filter row's run of X * C, which
    // starts at a multiple of sW * C and of the row Wi * C).
    std::array<int64_t, 3> extents{g.kpg, g.cpg, g.cpg};
    if(instance.patch)
    {
        if(!isPatchProblem(p))
        {
            return std::nullopt;
        }
        extents
            = {g.kpg, p.y * p.x * p.c, std::gcd(std::gcd(p.x * p.c, p.strideW * p.c), p.wi * p.c)};
    }
    const std::array<int64_t, 3> widths{instance.vectorA, instance.vectorB, instance.vectorC};
    for(size_t i = 0; i < extents.size(); ++i)
    {
        if(widths[i] != 0 && extents[i] != 0 && extents[i] % widths[i] != 0)
        {
            return std::nullopt;
        }
    }
    if(p.groups > 1 && !instance.grouped)
    {
        return std::nullopt;
    }
    for(const auto& [baked, actual] : {std::pair{instance.filterH, p.y},
                                       std::pair{instance.filterW, p.x},
                                       std::pair{instance.strideH, p.strideH},
                                       std::pair{instance.strideW, p.strideW},
                                       std::pair{instance.dilationH, p.dilationH},
                                       std::pair{instance.dilationW, p.dilationW}})
    {
        if(baked != 0 && baked != actual)
        {
            return std::nullopt;
        }
    }
    auto subGemms = instance.patch
                        ? patchSubGemms(p, g, instance.tileM, instance.tileN, instance.tileK)
                        : enumerateSubGemms(p, g, instance.tileM, instance.tileN, instance.tileK);
    const auto bound = instance.maxSubGemms != 0 ? instance.maxSubGemms : DEFAULT_MAX_SUB_GEMMS;
    if(!subGemms || static_cast<int64_t>(subGemms->size()) > bound)
    {
        return std::nullopt;
    }
    if(p.groups > 1 && g.cpg == 1)
    {
        return std::nullopt;
    }
    if(instance.tapOuterK
       && (p.strideH != 1 || p.strideW != 1 || p.dilationH != 1 || p.dilationW != 1
           || g.kpg % instance.tileK != 0))
    {
        return std::nullopt;
    }
    if(instance.dyHalo != 0 && (g.ho != p.hi || g.wo != p.wi || g.wo > instance.haloMaxWo))
    {
        return std::nullopt;
    }
    if(subGemms->empty())
    {
        return std::nullopt;
    }
    return subGemms;
}

/// _implicit_plan: the launch of an implicit-GEMM instance on @p problem -- the width
/// folded, the batch split where dY or dX exceeds the 32-bit ABI, each slice's records,
/// table, grid (flat tiles, groups, 1) of warp_m * warp_n * 64 threads and kernargs, and a
/// zero-filled dX when the records leave some of it unwritten.
inline std::optional<Plan> implicitPlan(const Problem& problem, const ImplicitInstance& instance)
{
    const auto launched = implicitLaunchProblem(problem);
    const auto geometry = deriveGeometry(launched);
    if(!geometry)
    {
        return std::nullopt;
    }
    const auto chunks = batchChunks(launched, *geometry);
    if(!chunks)
    {
        return std::nullopt;
    }
    const auto threads = instance.warpM * instance.warpN * WAVE_SIZE;
    Plan plan{launched, *geometry, {}, {}, false};
    int64_t tableBytes = 0;
    for(const auto& [n0, n] : *chunks)
    {
        auto chunk = launched;
        chunk.n = n;
        const auto chunkGeometry = deriveGeometry(chunk);
        if(!chunkGeometry)
        {
            return std::nullopt;
        }
        const auto subGemms = implicitSubGemms(chunk, *chunkGeometry, instance);
        if(!subGemms)
        {
            return std::nullopt;
        }
        const std::array<int64_t, 3> grid{subGemms->back().blockEnd, chunk.groups, 1};
        if(!gridFits(grid, threads, I32_MAX) || !boundedProduct({chunk.n, chunk.hi, chunk.wi}))
        {
            return std::nullopt;
        }
        auto table = packSubGemmBuffer(*subGemms, instance.tileM, instance.tileN);
        auto values = implicitGemmKernargs(chunk, *chunkGeometry, instance.tileM, instance.tileN);
        if(!table || !values)
        {
            return std::nullopt;
        }
        values->push_back(static_cast<int32_t>(subGemms->size()));
        auto step = makeStep(PlanKernel::SELF,
                             Abi::IMPLICIT,
                             grid,
                             threads,
                             {Buffer::DY, Buffer::W, Buffer::DX},
                             std::move(*values));
        step.tableOffset = tableBytes;
        if(chunks->size() > 1)
        {
            step.views = {View{n0 * geometry->dyImageBytes, n * geometry->dyImageBytes},
                          std::nullopt,
                          View{n0 * geometry->dxImageBytes, n * geometry->dxImageBytes}};
        }
        tableBytes += static_cast<int64_t>(table->size() * sizeof(int32_t));
        step.table = std::move(*table);
        plan.steps.push_back(std::move(step));
        plan.zeroDx = plan.zeroDx || !dxFullyWritten(chunk, *subGemms, instance.patch);
    }
    plan.workspaces.push_back({Buffer::SUB_GEMM, 0, tableBytes});
    return plan;
}

// ---- direct pipelines ----------------------------------------------------------------

/// identity_supports_problem for a direct binary: its filter, stride, padding and
/// dilation are the problem's (the direct kernels compile one stride and one padding),
/// and so are its channel counts where it bakes them.
inline bool directCapabilitiesMatch(const DirectCapabilities& compiled,
                                    const Problem& problem,
                                    const Geometry& geometry)
{
    return problem.strideH == problem.strideW && problem.padH == problem.padW
           && compiled.filterH == problem.y && compiled.filterW == problem.x
           && compiled.strideH == problem.strideH && compiled.strideW == problem.strideW
           && compiled.padH == problem.padH && compiled.padW == problem.padW
           && compiled.dilationH == problem.dilationH && compiled.dilationW == problem.dilationW
           && (compiled.cpg == 0 || compiled.cpg == geometry.cpg)
           && (compiled.kpg == 0 || compiled.kpg == geometry.kpg);
}

/// direct_uses_coalesced_weights: the compute pass reads the reorganized weights, so the
/// pipeline runs the reorganize transform.
inline bool usesCoalescedWeights(const DirectKnobs& knobs)
{
    return knobOr(knobs.wavesK, 1) > 1 || knobs.runtimeKLoop != 0;
}

/// direct_mfma_knobs_reason on the original problem (per-group channels @p cpg, @p kpg):
/// true when the knobs fit.
inline bool mfmaKnobsFit(const DirectInstance& instance, int64_t cpg, int64_t kpg)
{
    const auto& knobs = instance.knobs;
    const auto wavesK = knobOr(knobs.wavesK, 1);
    const auto wavesQ = knobOr(knobs.wavesQ, 1);
    const auto blockQ = knobOr(knobs.blockQ, 16);
    int64_t atoms = 0;
    if(knobs.foldK32 != 0)
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
    if(atoms == 0 || atoms % wavesK != 0)
    {
        return false;
    }
    if(wavesK > 1)
    {
        const auto kh = instance.compiled.filterH;
        const auto kw = instance.compiled.filterW;
        const auto mTiles = ceilDiv(cpg, 16);
        const auto weights = kh * kw * (atoms / wavesK) * mTiles * (knobs.foldK32 != 0 ? 4 : 2);
        const auto qSubtiles = blockQ / wavesQ / 16;
        const auto accumulators = qSubtiles * mTiles * kh * 4;
        if(weights + accumulators > MFMA_PRELOAD_VGPR_BUDGET)
        {
            return false;
        }
        if(instance.variant == DirectVariant::MFMA_RTC && qSubtiles > 1)
        {
            // dgrad_rtc_vgprs
            const auto loadVec = knobs.foldK32 != 0 ? 8 : 4;
            const auto acc = (blockQ / wavesQ / 16) * knobs.mTiles * kh * 4;
            const auto vgprs
                = acc + kh * kw * (knobs.kAtoms / wavesK) * knobs.mTiles * (loadVec / 2);
            if(vgprs > RTC_WIDE_Q_PRELOAD_VGPR_BUDGET)
            {
                return false;
            }
        }
    }
    return true;
}

/// The DirectConvSpec compute pass of the fused and MFMA pipelines on the transposed
/// problem @p t (make_dgrad_fprop_spec): the run-time half of DirectConvSpec.validate and
/// is_valid_spec -- groups % block_groups, the xcd_tiles row bands, a runtime-channel
/// binary's tile counts and channel alignment, and the staging LDS budget, which follows
/// a runtime cpg -- then its grid (persistent, or (ceil(Wo' / block_q), groups /
/// block_groups, N * ceil(H' / block_h))) of block_groups * waves_q * waves_k * 64
/// threads, held to 65535 on every axis.
inline std::optional<std::pair<std::array<int64_t, 3>, int64_t>>
    directConvSpecLaunch(const TransposedProblem& t, const DirectInstance& instance)
{
    const auto& knobs = instance.knobs;
    const auto blockQ = knobOr(knobs.blockQ, 16);
    const auto blockGroups = knobOr(knobs.blockGroups, 1);
    const auto blockH = knobOr(knobs.blockH, 16);
    const auto wavesQ = knobOr(knobs.wavesQ, 1);
    const auto wavesK = knobOr(knobs.wavesK, 1);
    const bool runtimeChannels = instance.variant == DirectVariant::MFMA_RTC;
    if(t.groups % blockGroups != 0)
    {
        return std::nullopt;
    }
    if(t.cpg % 4 != 0 || t.cpg < 4 || t.kpg < 1 || t.kpg % 4 != 0
       || (knobs.foldK32 != 0 && t.cpg % 32 != 0) || ceilDiv(t.cpg, 16) % wavesK != 0)
    {
        return std::nullopt;
    }
    if(runtimeChannels)
    {
        const auto kAtoms = knobs.foldK32 != 0 ? t.cpg / 32 : ceilDiv(t.cpg, 16);
        if(kAtoms != knobs.kAtoms || ceilDiv(t.kpg, 16) != knobs.mTiles)
        {
            return std::nullopt;
        }
    }
    const auto threads = blockGroups * wavesQ * wavesK * WAVE_SIZE;
    // The A/B staging ring of build_direct_conv (double-buffered) and the waves_k > 1
    // cross-wave reduction scratch.
    const auto loadVec = knobs.foldK32 != 0 ? 8 : 4;
    const auto ldsW = (blockQ - 1) + t.kw;
    const auto chunks = ldsW * blockGroups * (t.cpg / loadVec);
    const auto passes = ceilDiv(chunks, threads);
    const auto staging = passes * threads * loadVec * 2 * 2;
    const auto reduction = wavesK > 1 ? wavesQ * wavesK * WAVE_SIZE * 4 * 4 : 0;
    if(staging + reduction > LDS_CAPACITY_BYTES)
    {
        return std::nullopt;
    }
    const std::array<int64_t, 3> tiles{
        ceilDiv(t.wo, blockQ), t.groups / blockGroups, t.n * ceilDiv(t.h, blockH)};
    if(knobs.xcdTiles != 0 && tiles[2] < XCD_TILES_MIN_ROW_BANDS)
    {
        return std::nullopt;
    }
    std::array<int64_t, 3> grid = tiles;
    if(knobs.persistentGrid != 0)
    {
        // _persistent_cells_reason: the blocks walk at most persistent_rounds cells each.
        const auto cells = wideProduct({tiles[0], tiles[1], tiles[2]});
        if(!cells || *cells > DIRECT_PERSISTENT_ROUNDS * DIRECT_PERSISTENT_BLOCKS)
        {
            return std::nullopt;
        }
        grid = {DIRECT_PERSISTENT_BLOCKS, 1, 1};
    }
    if(!gridFits(grid, threads, MAX_GRID_DIM))
    {
        return std::nullopt;
    }
    return std::pair{grid, threads};
}

/// _direct_plan: rocKE's offer for a direct instance and its plan_for steps.
///   * depthwise_dgrad: one direct-dgrad launch, (ceil(Wi / block_w), ceil(groups /
///     (64 * block_waves)), N) of 64 * block_waves threads.
///   * depthwise_dgrad_win: groups a multiple of ch_per_lane (of 8 in the MFMA form) and
///     dY and dX within the kernel's 2^30-byte sentinel range, then one direct-dgrad
///     launch of DirectDepthwiseDgradWindowedSpec.grid().
///   * grouped_dgrad_fused: one forward launch of the DirectConvSpec compute pass
///     (directConvSpecLaunch) over the transposed problem, on the original W.
///   * grouped_dgrad_4c: groups % block_groups, then one forward launch,
///     (ceil(Wo' / block_q), groups / block_groups, N) of block_groups / 16 * waves_q * 64
///     threads (waves_q = block_q / 4 for the row-staged form).
///   * grouped_dgrad_mfma / dgrad_mfma_rtc: mfmaKnobsFit; with coalesced weights
///     groups == 1 only (rocKE's build_direct_reorganize_weights adds the group offset to
///     the per-group K axis of ws_t a second time, _coalesced_groups_reason); the
///     transpose W -> ws_t ((groups * KH * KW, ceil(kpg / 64), cpg) of 64), the reorganize
///     ws_t -> ws_coa when coalesced ((groups * KH * KW * ceil(kpg / k_atom) * ceil(cpg /
///     16), 1, 1) of 64), then the compute pass on (dY, ws, dX). ws_t is C * KH * KW * kpg
///     * 2 bytes at offset 0, ws_coa groups * KH * KW * ceil(kpg / k_atom) * ceil(cpg /
///     16) * 64 * (8 | 4) * 2 bytes at ws_t rounded up to 256. The runtime-channel
///     transforms take the per-group cpg and kpg.
/// Every tensor and workspace stays within the 32-bit kernel ABI.
inline std::optional<Plan> directPlan(const Problem& problem, const DirectInstance& instance)
{
    const auto geometry = deriveGeometry(problem);
    if(!geometry || isHelperVariant(instance.variant)
       || !directCapabilitiesMatch(instance.compiled, problem, *geometry))
    {
        return std::nullopt;
    }
    const auto& g = *geometry;
    const auto& knobs = instance.knobs;
    if(std::any_of(g.tensorBytes.begin(), g.tensorBytes.end(), [](int64_t bytes) {
           return bytes > I32_MAX;
       }))
    {
        return std::nullopt;
    }
    Plan plan{problem, g, {}, {}, false};
    const std::array<Buffer, 3> tensors{Buffer::DY, Buffer::W, Buffer::DX};
    switch(instance.variant)
    {
    case DirectVariant::DEPTHWISE:
    {
        const auto blockW = knobOr(knobs.blockW, 8);
        const auto threads = knobOr(knobs.blockWaves, 1) * WAVE_SIZE;
        const std::array<int64_t, 3> grid{
            ceilDiv(problem.wi, blockW), ceilDiv(problem.groups, threads), problem.n};
        if(!gridFits(grid, threads, I32_MAX))
        {
            return std::nullopt;
        }
        plan.steps.push_back(makeStep(PlanKernel::SELF,
                                      Abi::DIRECT_DGRAD,
                                      grid,
                                      threads,
                                      tensors,
                                      directDgradValues(problem, g)));
        return plan;
    }
    case DirectVariant::DEPTHWISE_WIN:
    {
        const bool mfma = knobs.mfma != 0;
        const auto chPerLane = knobOr(knobs.chPerLane, 1);
        const auto blockWaves = knobOr(knobs.blockWaves, 1);
        const auto wFold = knobOr(knobs.wFold, 1);
        const auto threads = blockWaves * WAVE_SIZE;
        if(problem.groups % chPerLane != 0 || (mfma && problem.groups % DW_DGRAD_MFMA_CH != 0)
           || std::max(g.tensorBytes[0], g.tensorBytes[2]) > DW_DGRAD_WIN_MAX_TENSOR_BYTES)
        {
            return std::nullopt;
        }
        const auto tileW = mfma ? 32 / wFold : knobOr(knobs.blockW, 8);
        const auto blockCh = mfma ? blockWaves * DW_DGRAD_MFMA_CH : threads * chPerLane;
        const auto imageTiles = mfma ? ceilDiv(problem.n, wFold) : problem.n;
        const std::array<int64_t, 3> grid{ceilDiv(problem.wi, tileW),
                                          ceilDiv(problem.groups, blockCh),
                                          imageTiles
                                              * ceilDiv(problem.hi, knobOr(knobs.blockH, 8))};
        if(!gridFits(grid, threads, I32_MAX))
        {
            return std::nullopt;
        }
        plan.steps.push_back(makeStep(PlanKernel::SELF,
                                      Abi::DIRECT_DGRAD,
                                      grid,
                                      threads,
                                      tensors,
                                      directDgradValues(problem, g)));
        return plan;
    }
    case DirectVariant::FUSED:
    {
        const auto t = transposedProblem(problem, g);
        const auto launch = t ? directConvSpecLaunch(*t, instance) : std::nullopt;
        if(!launch)
        {
            return std::nullopt;
        }
        plan.steps.push_back(makeStep(PlanKernel::SELF,
                                      Abi::DIRECT_FWD,
                                      launch->first,
                                      launch->second,
                                      tensors,
                                      transposedValues(*t)));
        return plan;
    }
    case DirectVariant::FOUR_C:
    {
        const auto t = transposedProblem(problem, g);
        const auto blockGroups = knobs.blockGroups;
        const auto blockQ = knobs.blockQ;
        if(!t || blockGroups < 16 || blockQ < 4 || problem.groups % blockGroups != 0)
        {
            return std::nullopt;
        }
        const auto wavesQ = knobs.stageRows != 0 ? blockQ / 4 : 1;
        const auto threads = blockGroups / 16 * wavesQ * WAVE_SIZE;
        const std::array<int64_t, 3> grid{
            ceilDiv(t->wo, blockQ), problem.groups / blockGroups, problem.n};
        if(!gridFits(grid, threads, I32_MAX))
        {
            return std::nullopt;
        }
        plan.steps.push_back(makeStep(
            PlanKernel::SELF, Abi::DIRECT_FWD, grid, threads, tensors, transposedValues(*t)));
        return plan;
    }
    case DirectVariant::MFMA:
    case DirectVariant::MFMA_RTC:
    {
        const bool runtimeChannels = instance.variant == DirectVariant::MFMA_RTC;
        if(!mfmaKnobsFit(instance, g.cpg, g.kpg))
        {
            return std::nullopt;
        }
        const auto t = transposedProblem(problem, g);
        const auto compute = t ? directConvSpecLaunch(*t, instance) : std::nullopt;
        const bool coalesced = usesCoalescedWeights(knobs);
        if(!compute || (coalesced && problem.groups != 1))
        {
            return std::nullopt;
        }
        const auto kh = problem.y;
        const auto kw = problem.x;
        const auto taps = problem.groups * kh * kw;
        const auto wsT = boundedProduct({problem.c, kh, kw, g.kpg, 2});
        if(!wsT)
        {
            return std::nullopt;
        }
        plan.workspaces.push_back({Buffer::WS_T, 0, *wsT});
        std::vector<int32_t> scalars;
        if(runtimeChannels)
        {
            scalars = {static_cast<int32_t>(g.cpg), static_cast<int32_t>(g.kpg)};
        }
        const auto transformAbi = runtimeChannels ? Abi::TRANSFORM_RTC : Abi::TRANSFORM;
        const std::array<int64_t, 3> transposeGrid{taps, ceilDiv(g.kpg, 64), g.cpg};
        if(!gridFits(transposeGrid, 64, I32_MAX))
        {
            return std::nullopt;
        }
        plan.steps.push_back(
            makeStep(runtimeChannels ? PlanKernel::TRANSPOSE_RTC : PlanKernel::TRANSPOSE,
                     transformAbi,
                     transposeGrid,
                     64,
                     {Buffer::W, Buffer::NONE, Buffer::WS_T},
                     scalars));
        auto weights = Buffer::WS_T;
        if(coalesced)
        {
            const auto kAtom = knobs.foldK32 != 0 ? 32 : 16;
            const auto blocks = wideProduct({taps, ceilDiv(g.kpg, kAtom), ceilDiv(g.cpg, 16)});
            const auto wsCoa = boundedProduct({problem.groups,
                                               kh,
                                               kw,
                                               ceilDiv(g.kpg, kAtom),
                                               ceilDiv(g.cpg, 16),
                                               64,
                                               knobs.foldK32 != 0 ? 8 : 4,
                                               2});
            if(!blocks || !wsCoa || !gridFits({*blocks, 1, 1}, 64, I32_MAX))
            {
                return std::nullopt;
            }
            plan.workspaces.push_back(
                {Buffer::WS_COA,
                 ceilDiv(*wsT, MFMA_WORKSPACE_ALIGNMENT) * MFMA_WORKSPACE_ALIGNMENT,
                 *wsCoa});
            plan.steps.push_back(
                makeStep(runtimeChannels ? PlanKernel::REORGANIZE_RTC : PlanKernel::REORGANIZE,
                         transformAbi,
                         {*blocks, 1, 1},
                         64,
                         {Buffer::WS_T, Buffer::NONE, Buffer::WS_COA},
                         scalars));
            weights = Buffer::WS_COA;
        }
        plan.steps.push_back(makeStep(PlanKernel::SELF,
                                      Abi::DIRECT_FWD,
                                      compute->first,
                                      compute->second,
                                      {Buffer::DY, weights, Buffer::DX},
                                      transposedValues(*t)));
        return plan;
    }
    case DirectVariant::NONE:
    case DirectVariant::TRANSPOSE:
    case DirectVariant::REORGANIZE:
    case DirectVariant::TRANSPOSE_RTC:
    case DirectVariant::REORGANIZE_RTC:
    default:
        return std::nullopt;
    }
}

/// gfx950_conv_dgrad_aot_plan: the launch of @p instance on @p problem, or std::nullopt
/// when it does not compute it.
inline std::optional<Plan> instancePlan(const Problem& problem, const AotInstance& instance)
{
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
/// benchmarked, recorded or trained; higher is better. The direct pipelines rank above
/// implicit GEMM (whose N tile is mostly idle at these few channels per group): the
/// single-kernel ones (windowed and streaming depthwise, fused, 4c) in [3, 3.5], the MFMA
/// pipelines in [2, 2.5); implicit GEMM in [0, 1.01) by the fraction of the launched tile
/// area that is real output times how well the grid fills a device of 256 compute units,
/// with wider vectors breaking ties.
inline double planScore(const AotInstance& instance, const Plan& plan)
{
    constexpr double CU_COUNT = 256.0;
    const auto& compute = plan.steps.back();
    const auto blocks = static_cast<double>(compute.grid[0]) * static_cast<double>(compute.grid[1])
                        * static_cast<double>(compute.grid[2]);
    const auto fill = blocks >= CU_COUNT ? 1.0 : blocks / CU_COUNT;
    if(instance.direct)
    {
        return (isMfmaVariant(instance.direct->variant) ? 2.0 : 3.0) + 0.49 * fill;
    }
    const auto& gemm = *instance.implicit;
    double useful = 0.0;
    double launched = 0.0;
    // A patch kernel's GEMM N is the filter's Y * X * C.
    const auto gemmN
        = gemm.patch ? plan.launched.y * plan.launched.x * plan.launched.c : plan.geometry.cpg;
    for(const auto& step : plan.steps)
    {
        const auto records = step.table.size()
                             / static_cast<size_t>(SUB_GEMM_RECORD_FIELDS + SUB_GEMM_MAGIC_FIELDS);
        for(size_t r = 0; r < records; ++r)
        {
            const auto* record = step.table.data() + r * SUB_GEMM_RECORD_FIELDS;
            const auto gemmM = static_cast<double>(record[3]);
            useful += gemmM * static_cast<double>(gemmN);
            launched += static_cast<double>(record[1]) * static_cast<double>(gemm.tileM)
                        * static_cast<double>(record[2]) * static_cast<double>(gemm.tileN);
        }
    }
    const auto widths = static_cast<double>(gemm.vectorA + gemm.vectorB + gemm.vectorC) / 24.0;
    return (launched > 0.0 ? useful / launched : 0.0) * fill + 1e-3 * widths;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::gfx950_conv_dgrad
