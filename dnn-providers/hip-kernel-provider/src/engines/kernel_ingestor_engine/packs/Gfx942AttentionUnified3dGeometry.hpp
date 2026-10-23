// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <initializer_list>
#include <optional>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx942AttentionUnified3dGeometry.hpp
 * @brief The split rule, tile rule, launch geometry and workspace layout gfx942 unified
 *        3D (split-KV) strided decode restates from rocKE's Python launch path.
 *
 * rocKE launches the segment kernel and then the reduce kernel from Python; the ingestor
 * engine relaunches the same binaries from C++. Nothing compares the two halves, and a
 * mismatch does not fault -- the kernels compute something else. Every function below
 * cites the Python it restates (paths relative to `rocke/library`, or to
 * `rocke/platform/python` for `rocke/helpers`).
 *
 * Scope: the problem `dispatch/attention/common.py::_problem` builds for a strided
 * single-token decode request: `total_q = num_seqs = batch`, `max_seqlen_q = 1`,
 * fp16/bf16 (no fp8 K/V), no sliding window, `target_ctas = 0`. The strided entry of
 * `run_unified_attention_torch` (`kernels/common/attention_unified.py`) ALSO accepts
 * sliding window, softcap and `target_ctas` (`dispatch/attention/strided_decode.py`
 * lists `sliding_window`), but this header restates `_num_segments` only for
 * `sliding_window == 0` and `target_ctas == 0`. With a sliding window rocKE returns the
 * raw `select_3d` count instead (e.g. B4/Hq32/Hkv8/D128/block16/Skv8192 at 304 CUs:
 * 32 there vs 38 here). The engine MUST refuse sliding-window graphs (v1 decision) and
 * must not pass `target_ctas`.
 *
 * Header-only and dependency-light on purpose: pure functions, testable with no HIP
 * context and nothing to mock.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes of one CDNA wave64 CTA: both kernels launch one wave
/// (`_get_3d_pipeline`: `wave_size = 64`, `seg_waves = 1` off gfx1250).
inline constexpr int64_t GFX942_UNIFIED_3D_WAVE_LANES = 64;

/// Workspace region alignment, in bytes.
inline constexpr int64_t GFX942_UNIFIED_3D_WORKSPACE_ALIGNMENT = 256;

/// The fixed CU count the D128 decode pre-bump clamp targets, whatever the live count
/// (`kernels/common/attention_unified.py::_PRE_BUMP_CUS`).
inline constexpr int64_t GFX942_PRE_BUMP_CUS = 120;

/**
 * @brief The KV tile override the gfx942 3D segment kernel is built with, or none.
 *
 * Restates `kernels/common/attention_unified.py::_gfx942_3d_tile_size_override`:
 *
 *     if not (arch == "gfx942" and problem.head_size >= 128 and problem.block_size >= 32):
 *         return None
 *     return problem.block_size // 2
 *
 * A compiled variant records the resolved tile (`tile_size_override` or `block_size`,
 * `kernels/gfx942/attention_tiled_3d.py::resolved_tile_size_override`), so the engine
 * needs this to name the variant rocKE would run.
 */
inline std::optional<int64_t> gfx942TileSizeOverride(int64_t headSize, int64_t blockSize)
{
    if(headSize < 128 || blockSize < 32)
    {
        return std::nullopt;
    }
    return blockSize / 2;
}

/// The KV tile in tokens: the override, else one block per tile
/// (`kernels/gfx942/attention_tiled_3d.py::tile_size`).
inline int64_t gfx942ResolvedTileSize(int64_t headSize, int64_t blockSize)
{
    return gfx942TileSizeOverride(headSize, blockSize).value_or(blockSize);
}

namespace detail
{

/// `rocke/helpers/attention.py::next_power_of_2` (and `_next_power_of_2`).
inline int64_t gfx942NextPowerOf2(int64_t x)
{
    int64_t p = 1;
    while(p < x)
    {
        p <<= 1;
    }
    return p;
}

/// The AITER split formula of `rocke/helpers/attention.py::select_3d_config` at target
/// @p targetCtas, with `num_2d = UnifiedAttentionProblem.total_num_q_blocks_upper_bound
/// * num_kv_heads` (`select_3d` and `_pre_bump_segments` evaluate the same expression):
///
///     block_m = 16 if qpk <= 16 else next_power_of_2(qpk)
///     num_2d = (total_q // (block_m // qpk) + num_seqs) * num_kv_heads
///     segments = min(next_power_of_2(ceil(target / num_2d)), 128)
///     segments = max(segments, 16 if block_size <= 16 else 8)
inline int64_t gfx942AiterSegments(
    int64_t batch, int64_t hq, int64_t hkv, int64_t blockSize, int64_t targetCtas)
{
    const int64_t qpk = hq / hkv;
    const int64_t blockM = qpk <= 16 ? 16 : gfx942NextPowerOf2(qpk);
    const int64_t blockQ = blockM / qpk;
    // total_q == num_seqs == batch for single-token decode.
    const int64_t num2d = (batch / blockQ + batch) * hkv;
    int64_t segments = gfx942NextPowerOf2((targetCtas + num2d - 1) / num2d);
    segments = segments < 128 ? segments : 128;
    const int64_t minSegments = blockSize <= 16 ? 16 : 8;
    return segments > minSegments ? segments : minSegments;
}

} // namespace detail

/**
 * @brief The split-KV segment count rocKE runs for one strided decode problem.
 *
 * Restates `kernels/common/attention_unified.py::_num_segments(problem, "gfx942")` for
 * the problem in this file's scope (sliding_window 0, target_ctas 0, no fp8 K/V,
 * max_seqlen_q 1). Its inputs, and nothing else, are what that function reads there:
 *
 *  1. Decode split rule, when `_gfx942_3d_decode_ctas_per_segment` admits the problem
 *     (head_size 64 or 128, block_size 16, and `ctas * 16 >= num_cus or
 *     max_seqlen_k < 8192`, with `ctas = num_seqs * num_kv_heads`):
 *
 *         per_cu = 6 if head_size == 64 else 4
 *         cap = 64 if max_seqlen_k < 8192 else 128
 *         return max(2, min(per_cu * num_cus // ctas, cap))
 *
 *  2. Otherwise `segments` = `select_3d()` at target `num_cus * 4`
 *     (`_effective_target_ctas`), `pre_bump` = the same formula at
 *     `_PRE_BUMP_CUS * 4 = 480` (`_pre_bump_segments`), and the measured decode clamps:
 *
 *         if max_seqlen_k <= 2048:
 *             D64 -> min(segments, 32); D128 -> min(segments, 16);
 *             D256 -> segments; other D -> min(segments, 64)
 *         elif head_size == 128 and max_seqlen_k < 32768:
 *             min(segments, pre_bump)
 *         else: segments
 *
 * The dtype reaches `select_3d_config` only as `element_size`, which it ignores.
 *
 * @param numCus The CU count `_resolve_num_cus` yields (live count on a gfx942 box, not
 *        floored; 120 off-box).
 */
inline int64_t gfx942SplitRuleSegments(int64_t batch,
                                       int64_t hq,
                                       int64_t hkv,
                                       int64_t headSize,
                                       int64_t blockSize,
                                       int64_t seqlenK,
                                       int64_t numCus)
{
    if(batch <= 0 || hq <= 0 || hkv <= 0 || headSize <= 0 || blockSize <= 0 || seqlenK < 0
       || numCus <= 0 || hq % hkv != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 unified 3d: split rule needs positive batch, heads, head size, block "
            "size and CU count, non-negative seqlen_k, and num_query_heads divisible by "
            "num_kv_heads");
    }

    // _gfx942_3d_decode_ctas_per_segment
    const int64_t ctas = batch * hkv;
    if((headSize == 64 || headSize == 128) && blockSize == 16
       && (ctas * 16 >= numCus || seqlenK < 8192))
    {
        const int64_t perCu = headSize == 64 ? 6 : 4;
        const int64_t cap = seqlenK < 8192 ? 64 : 128;
        const int64_t split = perCu * numCus / ctas;
        const int64_t capped = split < cap ? split : cap;
        return capped > 2 ? capped : 2;
    }

    const int64_t segments = detail::gfx942AiterSegments(batch, hq, hkv, blockSize, numCus * 4);
    if(seqlenK <= 2048)
    {
        if(headSize == 64)
        {
            return segments < 32 ? segments : 32;
        }
        if(headSize == 128)
        {
            return segments < 16 ? segments : 16;
        }
        if(headSize != 256)
        {
            return segments < 64 ? segments : 64;
        }
        return segments;
    }
    if(headSize == 128 && seqlenK < 32768)
    {
        // _pre_bump_segments: target _PRE_BUMP_CUS * 4 = 480.
        const int64_t preBump
            = detail::gfx942AiterSegments(batch, hq, hkv, blockSize, GFX942_PRE_BUMP_CUS * 4);
        return segments < preBump ? segments : preBump;
    }
    return segments;
}

/// One launch: grid and a 1-D block.
struct Gfx942AttentionUnified3dLaunch
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx942AttentionUnified3dLaunch& a,
                           const Gfx942AttentionUnified3dLaunch& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/// The segment launch and the reduce launch, in launch order.
struct Gfx942AttentionUnified3dGeometry
{
    Gfx942AttentionUnified3dLaunch segment;
    Gfx942AttentionUnified3dLaunch reduce;
};

/**
 * @brief The two launches of one strided decode problem.
 *
 * Restates `kernels/common/attention_unified.py::_get_3d_pipeline`:
 *
 *     seg grid = (_num_q_blocks(problem, block_q, decode_grid), num_kv_heads, num_segments)
 *     red grid = (total_q, num_query_heads, 1)
 *     block    = (64, 1, 1) for both on gfx942
 *
 * Strided decode always runs the decode grid (`_strided_3d_specs_from_problem` sets
 * `use_decode_grid = _use_decode_grid(problem)`, true for all-decode on gfx942), so the
 * q-block axis is `num_seqs` (`_num_q_blocks`). `total_q == batch`.
 */
inline Gfx942AttentionUnified3dGeometry
    gfx942AttentionUnified3dGeometry(int64_t batch, int64_t hq, int64_t hkv, int64_t numSegments)
{
    if(batch <= 0 || hq <= 0 || hkv <= 0 || numSegments <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 unified 3d: geometry needs positive batch, heads and segment count");
    }
    Gfx942AttentionUnified3dGeometry g;
    g.segment = {static_cast<unsigned>(batch),
                 static_cast<unsigned>(hkv),
                 static_cast<unsigned>(numSegments),
                 static_cast<unsigned>(GFX942_UNIFIED_3D_WAVE_LANES)};
    g.reduce = {static_cast<unsigned>(batch),
                static_cast<unsigned>(hq),
                1u,
                static_cast<unsigned>(GFX942_UNIFIED_3D_WAVE_LANES)};
    return g;
}

/// Byte offsets of each workspace region, and the total size.
struct Gfx942AttentionUnified3dWorkspace
{
    int64_t segmOutput = 0;
    int64_t segmMax = 0;
    int64_t segmExpsum = 0;
    int64_t sequsedK = 0;
    int64_t cuSeqlensQ = 0;
    int64_t totalBytes = 0;
};

inline int64_t gfx942AlignWorkspace(int64_t bytes)
{
    return (bytes + GFX942_UNIFIED_3D_WORKSPACE_ALIGNMENT - 1)
           / GFX942_UNIFIED_3D_WORKSPACE_ALIGNMENT * GFX942_UNIFIED_3D_WORKSPACE_ALIGNMENT;
}

/**
 * @brief One workspace buffer holding everything the two kernels read or write that the
 *        graph does not provide.
 *
 * The f32 split-KV scratch restates
 * `kernels/common/attention_unified.py::_attention_3d_workspace_specs`:
 * `segm_output[total_q, Hq, nseg, D]`, `segm_max[total_q, Hq, nseg]`,
 * `segm_expsum[total_q, Hq, nseg]`. The strided entry of `run_unified_attention_torch`
 * also takes caller-owned device metadata the engine must materialize: `seqused_k`, a
 * contiguous int32 `[B]` of KV lengths, and `cu_seqlens_q`, a contiguous int32 `[B+1]`
 * of query offsets `[0, 1, ..., B]`. Each region starts on a 256-byte boundary; the
 * total is rounded up to one too.
 */
inline Gfx942AttentionUnified3dWorkspace gfx942AttentionUnified3dWorkspace(int64_t batch,
                                                                           int64_t hq,
                                                                           int64_t numSegments,
                                                                           int64_t headSize)
{
    if(batch <= 0 || hq <= 0 || numSegments <= 0 || headSize <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 unified 3d: workspace needs positive batch, heads, segments, head size");
    }
    constexpr int64_t F32_BYTES = 4;
    constexpr int64_t I32_BYTES = 4;
    const int64_t rows = batch * hq * numSegments; // total_q == batch
    Gfx942AttentionUnified3dWorkspace w;
    w.segmOutput = 0;
    w.segmMax = gfx942AlignWorkspace(w.segmOutput + rows * headSize * F32_BYTES);
    w.segmExpsum = gfx942AlignWorkspace(w.segmMax + rows * F32_BYTES);
    w.sequsedK = gfx942AlignWorkspace(w.segmExpsum + rows * F32_BYTES);
    w.cuSeqlensQ = gfx942AlignWorkspace(w.sequsedK + batch * I32_BYTES);
    w.totalBytes = gfx942AlignWorkspace(w.cuSeqlensQ + (batch + 1) * I32_BYTES);
    return w;
}

/// One past the largest element offset a signed 32-bit index reaches (2^31).
constexpr int64_t GFX942_UNIFIED_3D_I32_INDEX_LIMIT = int64_t{1} << 31;

/**
 * @brief Does every tensor the segment and reduce kernels index with signed 32-bit element
 *        offsets hold fewer than 2^31 elements?
 *
 * The kernels compute these flat element offsets in i32 (`kernels/gfx942/attention_tiled_3d.py`,
 * `TensorDescriptor.naive(...).offset` in `rocke/helpers/transforms.py`, lowered to an i32
 * `getelementptr`, which sign-extends):
 *
 *  - `segm_output[token, head, seg, dim]`: `((token * Hq + head) * nseg + seg) * D + dim`,
 *    B * Hq * nseg * D elements (segment store and reduce load);
 *  - `segm_max` / `segm_expsum[token, head, seg]`: B * Hq * nseg elements;
 *  - Q and O `[token, head, dim]`: `(token * Hq + head) * D + dim`, B * Hq * D elements.
 *
 * At 2^31 elements or more the last offsets wrap negative and the stores land before the
 * buffer, silently. K and V are addressed through i64 bases and buffer resources instead
 * (stridedKvRefusal bounds them). False for non-positive inputs. Total: never throws.
 */
inline bool
    gfx942Unified3dIndicesFitInt32(int64_t batch, int64_t hq, int64_t numSegments, int64_t headSize)
{
    if(batch <= 0 || hq <= 0 || numSegments <= 0 || headSize <= 0)
    {
        return false;
    }
    const auto fits = [](std::initializer_list<int64_t> factors) {
        int64_t elements = 1;
        for(const int64_t factor : factors)
        {
            if(__builtin_mul_overflow(elements, factor, &elements))
            {
                return false;
            }
        }
        return elements < GFX942_UNIFIED_3D_I32_INDEX_LIMIT;
    };
    return fits({batch, hq, numSegments, headSize}) // segm_output
           && fits({batch, hq, numSegments}) // segm_max, segm_expsum
           && fits({batch, hq, headSize}); // Q, O
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
