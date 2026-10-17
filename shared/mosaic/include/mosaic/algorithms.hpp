// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>

#include "mosaic/layout.hpp"
#include "mosaic/tensor.hpp"

// ===================================================================
// AMD-specific spatial scheduling algorithms.
//
// These are NOT part of CuTe layout algebra. They use mosaic Layouts as
// inputs but encode AMD GPU concepts (XCDs, chiplets, grid quantization).
// Recovered from the previous transformations.hpp implementation.
// ===================================================================

namespace mosaic {

// -------------------------------------------------------------------
// chiplet_transform — remap workgroup IDs so that consecutive IDs land
// on the same XCD (chiplet). Useful for L2 cache locality when the
// scheduler default-distributes WGs round-robin across XCDs.
// -------------------------------------------------------------------
MOSAIC_HOST_DEVICE inline std::size_t chiplet_transform(std::size_t workgroup_id,
                                                        std::size_t num_workgroups,
                                                        std::size_t num_xcds) {
    std::size_t index = workgroup_id;
    std::size_t xcd = index % num_xcds;
    std::size_t pos_in_xcd = index / num_xcds;

    std::size_t min_per_xcd = num_workgroups / num_xcds;
    std::size_t extra_wgs = num_workgroups % num_xcds;

    // Total WGs assigned to all XCDs before this one.
    std::size_t offset = xcd * min_per_xcd + std::min(xcd, extra_wgs);

    return offset + pos_in_xcd;
}

// -------------------------------------------------------------------
// chiplet_transform_chunk — chunked variant. WGs are grouped into
// chunks of `chunk_size` consecutive IDs, and chunks are distributed
// across XCDs round-robin. Intermediate between fully-strided and
// fully-contiguous assignment.
// -------------------------------------------------------------------
MOSAIC_HOST_DEVICE inline std::size_t chiplet_transform_chunk(std::size_t workgroup_id,
                                                              std::size_t num_workgroups,
                                                              std::size_t chunk_size,
                                                              std::size_t num_xcds) {
    std::size_t index = workgroup_id;

    // Cap chunk_size so it doesn't exceed WGs-per-XCD.
    chunk_size = std::min(chunk_size, num_workgroups / num_xcds);

    std::size_t xcd = index % num_xcds;

    // Tail WGs that don't fit into a complete chunk-distributed cycle keep
    // their original ID.
    if (index > (num_workgroups / (num_xcds * chunk_size)) * (num_xcds * chunk_size)) {
        return index;
    }

    std::size_t local_idx = index / num_xcds;
    std::size_t chunk_idx = local_idx / chunk_size;
    std::size_t pos_in_chunk = local_idx % chunk_size;

    return chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk;
}

// Alias used by the GEMM example.
MOSAIC_HOST_DEVICE inline std::size_t chiplet_transform_chunked(std::size_t workgroup_id,
                                                                std::size_t num_workgroups,
                                                                std::size_t num_xcds,
                                                                std::size_t chunk_size) {
    return chiplet_transform_chunk(workgroup_id, num_workgroups, chunk_size, num_xcds);
}

MOSAIC_HOST_DEVICE inline void wgm(int workgroup_id, int grid_height, int grid_width, int W,
                                   int& row, int& col) {
    const int num_wgid_in_group = W * grid_width;
    int group_id = workgroup_id / num_wgid_in_group;
    int first_row = group_id * W;
    int group_size_m = min(grid_height - first_row, W);
    row = first_row + ((workgroup_id % num_wgid_in_group) % group_size_m);
    col = (workgroup_id % num_wgid_in_group) / group_size_m;
}

#if defined(__HIP_DEVICE_COMPILE__)
__device__ int get_hw_xcd_id() {
    int xcd;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID, 0, 16)" : "=s"(xcd));
    return xcd;
}

__device__ inline int get_hw_cu_id() {
    int cu;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID, 8, 4)" : "=s"(cu));
    return cu;
}
#endif

MOSAIC_HOST_DEVICE inline int chiplet_swizzle_hw_aligned(int workgroup_id, int num_workgroups,
                                                         int num_xcds) {
#if defined(__HIP_DEVICE_COMPILE__)
    int hw_xcd = get_hw_xcd_id();
    int s = ((hw_xcd - (workgroup_id % num_xcds)) % num_xcds + num_xcds) % num_xcds;
    int aligned = (workgroup_id + s) % num_workgroups;
    int xcd = aligned % num_xcds;
    int per_xcd = num_workgroups / num_xcds;
    int extra = num_workgroups % num_xcds;
    int pos = aligned / num_xcds;
    return xcd * per_xcd + ((xcd < extra) ? xcd : extra) + pos;
#else
    // On host, just use chiplet_transform, ignoring hw alignment.
    return chiplet_transform(workgroup_id, num_workgroups, num_xcds);
#endif
}

// -------------------------------------------------------------------
// gemm_grid_transform — apply a tiled layout to remap workgroup IDs
// across an MxN output grid, handling the boundary strips when the
// grid isn't perfectly tiled by the layout.
//
// `L` permutes IDs in the quantized (qy x qx) interior region. The
// non-quantized right strip (width = grid_width - qx) and bottom strip
// (height = grid_height - qy) are kept in default row-major order.
//
// Caller computes `qy`, `qx` as the quantized extents the layout covers.
// For a typical hierarchical layout `make_layout(outer, mid, inner)`
// where each mode is (Y, X), qy = product of all Y dims, qx = product
// of all X dims (truncated to fit the grid).
// -------------------------------------------------------------------
template <class Layout>
MOSAIC_HOST_DEVICE inline std::size_t gemm_grid_transform(std::size_t index, Layout const& L,
                                                          std::size_t grid_height,
                                                          std::size_t grid_width, std::size_t qy,
                                                          std::size_t qx) {
    const std::size_t total_quantized = qx * qy;
    const std::size_t non_quantized_x = grid_width - qx;  // right strip width
    const std::size_t x_strip_size = non_quantized_x * qy;
    const std::size_t y_strip_start = total_quantized + x_strip_size;

    // Interior region: apply the layout's permutation, then rebase to
    // the full grid_width.
    if (index < total_quantized) {
        const std::size_t local = static_cast<std::size_t>(L(index));
        const std::size_t y = local / qx;
        const std::size_t x = local % qx;
        return y * grid_width + x;
    }

    // Right vertical strip (x in [qx, grid_width)).
    if (non_quantized_x && index < y_strip_start) {
        const std::size_t rel = index - total_quantized;
        const std::size_t y = rel / non_quantized_x;
        const std::size_t x = qx + (rel % non_quantized_x);
        return y * grid_width + x;
    }

    // Bottom horizontal strip (y in [qy, grid_height)).
    const std::size_t rel = index - y_strip_start;
    const std::size_t y = qy + (rel / grid_width);
    const std::size_t x = rel % grid_width;
    return y * grid_width + x;
}

// clear(t) — zero every element of a tensor.
template <class Engine, class Layout>
MOSAIC_HOST_DEVICE constexpr void clear(Tensor<Engine, Layout>& t) {
    using value_type = typename Tensor<Engine, Layout>::value_type;
    auto const N = t.size();
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        t(i) = value_type{};
    }
}

// -------------------------------------------------------------------
// gemm(tiled_mma, a, b, c) — per-thread MFMA dispatcher.
//
// Walks the per-thread register fragments
//
//   a: shape (VPT_A, MMA_M, MMA_K)
//   b: shape (VPT_B, MMA_N, MMA_K)
//   c: shape (VPT_C, MMA_M, MMA_N)
//
// over (ki, mi, ni). For each iteration: pack a(_,mi,ki) + b(_,ni,ki)
// + c(_,mi,ni) into the atom's register vector types, call
// Atom::mma(a_reg, b_reg, c_reg), unpack back into c(_,mi,ni).
//
// `tiled_mma` is a tag-style arg — only its types matter. Mirrors the
// CUTLASS `gemm(MMA, ...)` overload at
// https://github.com/NVIDIA/cutlass/blob/main/include/cute/algorithm/gemm.hpp
//
// Available only under HIP since the atom's mma() and register-vector
// type aliases (ARegType / BRegType / CRegType) are AMD-specific.
// -------------------------------------------------------------------

// -------------------------------------------------------------------
// copy(src, dst)
// copy(copy_atom, src, dst)
// copy(tiled_copy, src, dst)
//
// Elementwise scalar copy from `src` to `dst`. All three overloads
// dispatch to the same loop — the atom / TiledCopy args are
// placeholders for the vectorized lowering pass (B6's traits carry
// `copy_issue()` for the per-issue intrinsic; the dispatcher will
// switch on the atom's lowering kind later). Correctness only.
//
// Requires `src.size() == dst.size()`. The loop walks the flat tensor
// domain — works for any pair of compatible-shape tensors regardless
// of which engine each is on (gmem, LDS, register).
// -------------------------------------------------------------------

template <class SrcEngine, class SrcL, class DstEngine, class DstL>
MOSAIC_HOST_DEVICE inline void copy(Tensor<SrcEngine, SrcL> const& src,
                                    Tensor<DstEngine, DstL>& dst) {
    auto const N = src.size();
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        dst(i) = src(i);
    }
}

template <class CopyAtomOrTiled, class SrcEngine, class SrcL, class DstEngine, class DstL>
MOSAIC_HOST_DEVICE inline void copy(CopyAtomOrTiled, Tensor<SrcEngine, SrcL> const& src,
                                    Tensor<DstEngine, DstL>& dst) {
    auto const N = src.size();
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        dst(i) = src(i);
    }
}

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)

template <class TiledMma, class FragA, class FragB, class FragC>
__device__ inline void gemm(TiledMma, FragA const& a, FragB const& b, FragC& c) {
    using Atom = typename TiledMma::Atom;
    constexpr int MMA_M = TiledMma::MMA_M;
    constexpr int MMA_N = TiledMma::MMA_N;
    constexpr int MMA_K = TiledMma::MMA_K;
    constexpr int VPT_A = Atom::VPT_A;
    constexpr int VPT_B = Atom::VPT_B;
    constexpr int VPT_C = Atom::VPT_C;
    using ARegType = typename Atom::ARegType;
    using BRegType = typename Atom::BRegType;
    using CRegType = typename Atom::CRegType;

#pragma unroll
    for (int ki = 0; ki < MMA_K; ++ki) {
#pragma unroll
        for (int mi = 0; mi < MMA_M; ++mi) {
            ARegType a_reg;
#pragma unroll
            for (int v = 0; v < VPT_A; ++v) {
                a_reg[v] = a(v, mi, ki);
            }
#pragma unroll
            for (int ni = 0; ni < MMA_N; ++ni) {
                BRegType b_reg;
#pragma unroll
                for (int v = 0; v < VPT_B; ++v) {
                    b_reg[v] = b(v, ni, ki);
                }
                CRegType c_reg;
#pragma unroll
                for (int v = 0; v < VPT_C; ++v) {
                    c_reg[v] = c(v, mi, ni);
                }
                c_reg = Atom::mma(a_reg, b_reg, c_reg);
#pragma unroll
                for (int v = 0; v < VPT_C; ++v) {
                    c(v, mi, ni) = c_reg[v];
                }
            }
        }
    }
}

#endif  // __HIPCC__

}  // namespace mosaic
