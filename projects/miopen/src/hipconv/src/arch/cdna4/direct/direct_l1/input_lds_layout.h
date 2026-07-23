#pragma once

#include "config.h"
#include "mathutil.h"
#include "swizzle.h"

#include <hip/hip_runtime.h>

namespace hipconv::cdna4
{

// LDS layout for the workgroup's double-buffered input tile.
//
// The tile holds 2(tic/toc) x block_size_h x block_size_w x block_size_c8 uint4
// (8 fp16), each 8 C-channels for one (w, h). The w x c8 plane is swizzled so the
// MFMA 16x32 operand read hits 16 distinct bank-bases per phase. Two addressing
// modes: coordinate-based (wc8, swizzle applied; used by InputLoaderLds reads) and
// slot-based (slot/w_slot; used by InputLoader's wave-uniform writes, where lane L
// lands at slot L and its fetch mapping makes that slot hold the swizzled position).
// All units are uint4 (16 bytes).
template <direct_l1::Config cfg>
struct InputLdsLayout
{
    static constexpr int block_size_p  = cfg.block_p();
    static constexpr int block_size_q  = cfg.block_q();
    static constexpr int block_size_c  = 64;
    static constexpr int block_size_h  = block_size_p + cfg.kh - 1;
    static constexpr int block_size_c8 = block_size_c / 8;

    // Batch-unfold column geometry: n_unfold images of w_unfold columns, adjacent.
    //
    // Each image needs its own kw-1 halo, so per-image span is w_unfold + kw - 1.
    // Standard tile (n_unfold == 1): w_unfold == block_size_q.
    static constexpr int n_unfold     = cfg.unfold_n;
    static constexpr int w_unfold     = cfg.w_unfold();
    static constexpr int w_per_image  = w_unfold + cfg.kw - 1;
    static constexpr int block_size_w = n_unfold * w_per_image;

    // Wave64B128: the read is bank-conflict-free only under the additive rotation.
    //
    // direct_l1 reads with a wave64 ds_read_b128 in 4 scrambled 16-lane phases; the
    // default Wave32B128 shift reintroduces conflicts for this C(64) tile.
    using Swizzle = SwizzleT<block_size_c, SwizzleReadModel::Wave64B128>;

    static constexpr int w_stride = block_size_c8;
    static constexpr int h_stride = block_size_w * w_stride;

    // Data footprint of one buffer (tic or toc): the full block_size_h rows.
    static constexpr int tic_data = block_size_h * h_stride;

    // Rows the input loader writes per step (step 0 writes the bottom ones).
    //
    // Same formula as InputLoader::rows_per_step; recomputed so the pad below can
    // keep those rows clear of the writer's staging region.
    static constexpr int rows_per_step = divup(block_size_h, cfg.kw * 2);

    // uint4 the writer stages at the TOP of the dead (tic) buffer.
    //
    // Sized for the MAX stage (K64/wave, 16384 fp16); smaller stages use a top
    // sub-region. A static_assert in kernel.h checks this is >= the actual stage.
    static constexpr int writer_stage_uint4 = 16384 / 8;

    // Ideal inter-buffer pad (uint4) for the cross-round overlap.
    //
    // The writer's staging (top writer_stage_uint4 slots) must not alias the next
    // round's step-0 prefetch (bottom rows_per_step rows), which requires
    //   tic_pad >= writer_stage_uint4 - (block_size_h - rows_per_step)*h_stride.
    // A smaller block_h leaves less slack, so the pad grows; floored at 16.
    static constexpr int overlap_pad =
        maximum(16, writer_stage_uint4 - (block_size_h - rows_per_step) * h_stride);

    // LDS bytes a given per-buffer stride (uint4) costs: two buffers, 16 B each.
    static constexpr int lds_bytes_for(int stride) { return 2 * stride * 16; }

    // Whether the writer/step-0-prefetch overlap fits the LDS budget.
    //
    // A small kw inflates overlap_pad past 160 KiB for tight filters; when it does,
    // the kernel drops the pad and serializes the two instead (see kernel.h).
    // Drives both that barrier and the pad selection, so they cannot disagree.
    static constexpr bool overlap_safe =
        lds_bytes_for(tic_data + overlap_pad) <= direct_l1::LDS_BYTES_PER_CU;

    // Inter-buffer pad (uint4): overlap_pad if it fits, else the 16 floor.
    //
    // Sits between the buffers, so it is part of the pitch (tic_stride) but not the
    // data size (tic_data).
    static constexpr int tic_pad    = overlap_safe ? overlap_pad : 16;
    static constexpr int tic_stride = tic_data + tic_pad;

    int offset;

    __device__ __host__ constexpr InputLdsLayout(int offset_in = 0) : offset(offset_in) {}

    __device__ __host__ auto step(int delta) const { return InputLdsLayout(offset + delta); }

    __device__ __host__ auto tic(int x) const { return step(x * tic_stride); }
    __device__ __host__ auto h(int x) const { return step(x * h_stride); }

    // Combined (w, c8) step through the swizzle (w in 1-element units, c8 a C-strip).
    __device__ __host__ auto wc8(int w, int c8) const { return step(Swizzle::offset_uint4(w, c8)); }

    // Unswizzled slot-based steps for buffer_load_lds writes.
    __device__ __host__ auto w_slot(int x) const { return step(x * w_stride); }
    __device__ __host__ auto slot(int x) const { return step(x); }

    // Pitch of one buffer (data + pad), uint4; total occupancy is 2 * tic_size.
    static constexpr int tic_size() { return tic_stride; }
};

} // namespace hipconv::cdna4
