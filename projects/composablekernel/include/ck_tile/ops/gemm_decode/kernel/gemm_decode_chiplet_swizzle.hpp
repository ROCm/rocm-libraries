// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// XCD-aware workgroup remap for multi-die GPUs (MI300/MI355).
//
// The CP dispatches workgroups round-robin across XCDs by linear wgid:
// HW wgid % num_xcds ==> destination XCD. With the natural launch grid
// where consecutive (m, n_block) pairs share the same B row, that puts
// the B-sharing pairs on different XCDs and forces every XCD's L2 slice
// to fetch the same row independently.
//
// `remap_wgid` is the inverse permutation: it maps a HW wgid to a
// logical wgid such that `chunk_size` consecutive *logical* wgids end up
// running on the same XCD, after the CP's round-robin shuffle. The
// kernel then unflattens the logical wgid back into (m_block, n_block)
// and proceeds normally. Reference: ROCm Blogs "Deep Dive Into 4-Wave
// Interleave FP8 GEMM", Appendix A.
//
// Tail handling: when `num_workgroups` is not a multiple of
// `num_xcds * chunk_size` we leave the trailing wgids unswizzled. This
// keeps the bijection complete and avoids overlapping (m, n) coverage at
// the boundary - the tail just falls back to the default round-robin
// pattern, which only matters for the last few hundred CTAs of the
// launch.
//
// All inputs are runtime values; on the host the kernel can pass
// num_xcds and chunk_size as launch arguments, but for the gemm_decode
// universal/blockscale kernels we keep them as compile-time template
// parameters on `GemmDecodeProblem` so the divisions/mods fold to
// constants.
struct GemmDecodeChipletSwizzle
{
    CK_TILE_HOST_DEVICE static constexpr index_t
    remap_wgid(index_t wgid, index_t num_workgroups, index_t num_xcds, index_t chunk_size)
    {
        const index_t block = num_xcds * chunk_size;
        const index_t limit = (num_workgroups / block) * block;
        if(wgid >= limit)
            return wgid;

        const index_t xcd             = wgid % num_xcds;
        const index_t local           = wgid / num_xcds;
        const index_t chunk_idx       = local / chunk_size;
        const index_t offset_in_chunk = local % chunk_size;

        return chunk_idx * block + xcd * chunk_size + offset_in_chunk;
    }
};

} // namespace ck_tile
