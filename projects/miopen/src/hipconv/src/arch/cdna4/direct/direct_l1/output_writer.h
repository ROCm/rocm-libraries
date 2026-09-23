#pragma once

// Per-wave output writer for the direct_l1 kernel.
//
// K-leading layout (weights = srcA, input = srcB; see kernel.h mma): for a fixed
// (p, q16, k16) tile, lane L holds Q = q16*16 + (L%16) (one pixel) and K =
// k16*16 + (L/16)*4 + a (a = 0..3, 4 contiguous channels).
//
// Two store paths:
//
// 1. LDS-STAGED WIDE STORE (wave_k16 == 4, and every batch-unfold tile). A pixel's
//    channels are spread across 16 lanes, so no lane can issue a coalesced wide
//    store directly. Each wave stages one P-row through a private LDS chunk (no
//    __syncthreads: disjoint regions) and reads it back transposed, so consecutive
//    lanes now hold consecutive channels of one pixel, then stores b64 (32 wide
//    stores/wave vs 128 narrow). The chunk's XOR swizzle is bank-conflict-free for
//    both b64 opcodes (see stage_off()); b64 is chosen over b128 for a regular swizzle.
//
// 2. NARROW STORE (K96/K64, the 4x4/5x5 K128 tiles, standalone K32). Each lane
//    stores its 4-contiguous-K run as one dwordx2 straight to global NHWC. Not
//    cross-lane coalesced, but correct for any symmetric variant.

#include "config.h"
#include "layer_pars.h"
#include "memory.h"
#include "packed_ops.h"
#include "types.h"
#include "wave_compute_index.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <type_traits>

namespace hipconv::cdna4::direct_l1
{

template <Config cfg, typename datatype_t>
class OutputWriter
{
    static constexpr int wave_p   = cfg.wave_p;
    static constexpr int wave_q16 = cfg.wave_q16;
    static constexpr int Kwg_k16  = cfg.wave_k16;
    static constexpr int Kwg      = cfg.wave_k();
    static constexpr int block_k  = cfg.block_k();

    // Unfold needs a staged Kwg (Kwg_k16 in {4, 2}).
    //
    // store_staged (full-K) and store_partial_k (the rare straddling wave) both decode
    // the (n_local, w) pixel, but emit() does not.
    static_assert(cfg.unfold_n == 1 || (Kwg_k16 == 4 || Kwg_k16 == 2),
                  "batch-unfold (unfold_n > 1) requires a staged-store Kwg "
                  "(wave_k16 in {4, 2}); emit() does not handle the unfold pixel decode");

    // Buffer-descriptor data format (same constant the input/weights loaders use).
    static constexpr int kRsrcDataFormat = 1 << 15;

    // SLC (bit 1): mark the output stores streaming / non-temporal.
    //
    // The output is write-once, so on CDNA4 an SLC store miss does not insert at MRU
    // and cannot evict the weights/input working set a partner wave is loading.
    // Measured inert at the profiled size (n=128 3x3 K128: L2 hit rate unchanged,
    // the set already fits), retained as upside-only insurance for L2-spilling problems.
    static constexpr int kStoreCachePolicy = 1 << 1;

public:
    // Per-wave staging footprint: one P-row chunk, always 1024 fp16.
    //
    // The stage_off swizzle uses a fixed pitch of 16 blocks (kb^q spreads over
    // [0,16) since q is 0..15), so the slot is 16 pixels x 16 slots x 4 fp16 even at
    // K32 (8 real blocks, scattered). Only the b64 write/read count per row varies
    // with Kwg. Double-buffered (tic/toc) so the look-ahead write overlaps the read.
    static constexpr int kStageBlocksPerRow = 16; // fixed swizzle pitch (q stride)
    static constexpr int kStageRowFp16      = kStageBlocksPerRow * 4; // 64: swizzled row width
    static constexpr int kStageWaveFp16     = 16 * kStageRowFp16; // 1024: one wave's 16-pixel row
    // All waves in the workgroup stage into disjoint slots of one buffer.
    static constexpr int kStageBufFp16  = cfg.num_waves() * kStageWaveFp16;
    static constexpr int kNumStageBufs  = 2; // tic/toc
    static constexpr int stage_lds_fp16 = kNumStageBufs * kStageBufFp16;

    __device__ OutputWriter(const OutputPars& pars,
                            datatype_t* out,
                            datatype_t* stage_lds,
                            int block_n,
                            int block_p,
                            int block_q,
                            int block_k_idx,
                            int block_g_idx,
                            int wave,
                            int wave_k_idx)
        : pars_(pars)
        , out_(out)
        // Per-wave staging slot: the global wave id picks a disjoint 1024-fp16 region.
        //
        // K partitions and waves never alias, so no __syncthreads is needed.
        , stage_(stage_lds +
                 __builtin_amdgcn_readfirstlane((wave_k_idx * (cfg.waves_p * cfg.waves_q) + wave) *
                                                kStageWaveFp16))
        // readfirstlane pins these wave-uniform bases to SGPRs.
        //
        // Else the threadIdx-derived values stay in VGPRs across the whole loop and
        // spill under K128's pressure. Lane-within-wave mappings are recomputed from
        // mbcnt instead (see volatile_lane_id()), so they need not survive the loop.
        , wave_p_base_(__builtin_amdgcn_readfirstlane(block_p + WaveComputeIndex<cfg>(wave).p()))
        , wave_q_base_(__builtin_amdgcn_readfirstlane(block_q + WaveComputeIndex<cfg>(wave).q()))
        , wg_k_base_(__builtin_amdgcn_readfirstlane(block_k_idx * block_k + wave_k_idx * Kwg))
        // Global image index this q-wave's first packed image starts at (unfold only).
        //
        // 0 for the standard tile; the per-lane n_local is added in the store loop.
        , wave_img_base_(__builtin_amdgcn_readfirstlane(
              block_n * cfg.unfold_n + WaveComputeIndex<cfg>(wave).wave_image_base()))
        // This wave's image base relative to the folded output base (see the fold in
        // the body). Standard tile: 0. Unfold: the q-wave's first packed image within
        // the tile's nu-image window (block_n's contribution is folded into the base).
        , img_rel_base_(cfg.unfold_n == 1 ? 0 : WaveComputeIndex<cfg>(wave).wave_image_base())
        , block_g_idx_(block_g_idx)
    {
        // Fold this tile's image origin into the 64-bit base so the 32-bit store
        // offsets only span the tile's images.
        //
        // The whole output may exceed 4 GiB (NUM_RECORDS is 32-bit); one image (nu
        // images under batch-unfold) fits. The store loops address relative to the
        // fold via img_rel_base_, while N/Wo/Ho guards stay absolute. The hardware
        // bounds-checks every store against NUM_RECORDS, dropping offsets past the
        // window. img0 stays a constructor local so its 64-bit live range does not
        // span the round and spill the writer (see img_rel_base_).
        const size_t img_stride = static_cast<size_t>(pars_.Ho) * pars_.Wo * pars_.K_per_pixel();
        const size_t img0       = static_cast<size_t>(block_n) * cfg.unfold_n;

        // large_tensor: also fold this wave's absolute output-row origin into the base.
        //
        // Baseline addresses the row with p_out * row_stride, which overflows int32
        // once one image exceeds 2 GiB (p_out up to Ho, row_stride = Wo*K). Fold the
        // wave's row origin (wave_p_base_) here; the store loops then carry only the
        // wave-relative row p (0..wave_p), while p_out < Ho / q_out < Wo guards stay
        // absolute. Per-wave so no extra relative member is needed (store row == p).
        const size_t row_stride = static_cast<size_t>(pars_.Wo) * pars_.K_per_pixel();
        const size_t row_off =
            cfg.large_tensor ? static_cast<size_t>(wave_p_base_) * row_stride : 0;
        datatype_t* tile_base = out_ + img0 * img_stride + row_off;

        // Window spans the tile's images (baseline) or this wave's wave_p rows
        // (large_tensor), clamped to the bytes remaining to the tensor end. Guards
        // already drop stores past the real Ho/Wo/K, so the clamp only bounds
        // NUM_RECORDS to the allocation (the C(64)-style tail over-read cannot occur on
        // the exact-K store, but the clamp keeps a padded last row inside the tensor).
        const size_t window_span =
            cfg.large_tensor ? static_cast<size_t>(wave_p) * row_stride * sizeof(datatype_t)
                             : static_cast<size_t>(cfg.unfold_n) * img_stride * sizeof(datatype_t);
        const size_t remaining_bytes =
            (static_cast<size_t>(pars_.N) * img_stride - img0 * img_stride - row_off) *
            sizeof(datatype_t);
        const int window_bytes =
            static_cast<int>(window_span < remaining_bytes ? window_span : remaining_bytes);
        out_rsrc_ = __builtin_amdgcn_make_buffer_rsrc(tile_base, 0, window_bytes, kRsrcDataFormat);
    }

    // Write one wave's accumulators to global.
    //
    // Caller must drain the final iteration's loads first. K partitions write
    // disjoint K ranges to disjoint addresses, so no serialization between them.
    __device__ void write(const fp32x4_t (&acc)[wave_p][wave_q16][Kwg_k16])
    {
        if constexpr(!cfg.k_divisible)
        {
            // Arbitrary-K: the padded last k-block can run past real K.
            //
            // This wave owns [wg_k_base_, wg_k_base_ + Kwg); the terms are
            // wave-uniform, so the select is scalar.
            const int k_real = pars_.K_per_group;
            if(wg_k_base_ >= k_real)
                return; // Entire wave is padding; write nothing.
            if(wg_k_base_ + Kwg > k_real)
            {
                // Real-K boundary straddles this wave: per-channel guarded store.
                //
                // At most one wave per problem, so its lower throughput is irrelevant.
                store_partial_k(acc);
                return;
            }
            // Fall through: this wave is entirely within real K, use the fast path.
        }

        // Full-K wave: staged wide store for wave_k16==4 and all unfold tiles.
        //
        // Unfold must use it (only it does the (n_local, w) decode). Non-unfold
        // K32/K96/K64-wave tiles keep the narrow fallback (standalone staging deferred).
        if constexpr(Kwg_k16 == 4 || cfg.unfold_n > 1)
            store_staged(acc);
        else
            store_impl<Kwg_k16>(acc);
    }

private:
    using vec4_t = decltype(packed_convert<datatype_t>(fp32x4_t{}));

    // Raw store-payload type for the buffer_store builtins: vec4_t (4 fp16) = b64.
    using u32x2_t = __attribute__((ext_vector_type(2))) uint32_t;

    // buffer_store helpers: byte offset = elem_off * sizeof(elem), soffset 0.
    __device__ void store_b64(int elem_off, vec4_t r) const
    {
        __builtin_amdgcn_raw_buffer_store_b64(__builtin_bit_cast(u32x2_t, r),
                                              out_rsrc_,
                                              elem_off * static_cast<int>(sizeof(datatype_t)),
                                              0,
                                              kStoreCachePolicy);
    }
    __device__ void store_b16(int elem_off, datatype_t v) const
    {
        __builtin_amdgcn_raw_buffer_store_b16(__builtin_bit_cast(uint16_t, v),
                                              out_rsrc_,
                                              elem_off * static_cast<int>(sizeof(datatype_t)),
                                              0,
                                              kStoreCachePolicy);
    }

    template <int K16>
    __device__ void store_impl(const fp32x4_t (&acc)[wave_p][wave_q16][Kwg_k16])
    {
        // Whole-tile bounds test, wave-uniform.
        //
        // A fully-in-bounds footprint skips the per-pixel q_out >= Wo guard (and its
        // save-exec/branch/restore-exec); only the ragged last block takes the checked path.
        const bool full_tile =
            (wave_p_base_ + wave_p <= pars_.Ho) && (wave_q_base_ + wave_q16 * 16 <= pars_.Wo);
        if(full_tile)
            emit<K16, false>(acc);
        else
            emit<K16, true>(acc);
    }

    // Narrow store of the wave's accumulators to global NHWC.
    //
    // Each lane's 4 contiguous K (a=0..3) go out as one dwordx2 per (p, q16, k16).
    // Consecutive lanes hold different pixels, so these are not cross-lane coalesced;
    // this is the correct fallback for all symmetric variants. int32 addressing
    // (outputs >= 2 GiB rejected upstream).
    template <int K16, bool CheckBounds>
    __device__ void emit(const fp32x4_t (&acc)[wave_p][wave_q16][Kwg_k16])
    {
        const int lane_idx = volatile_lane_id();
        const int q_lane   = lane_idx % 16;       // Q within the q16 group
        const int k_bank   = (lane_idx / 16) * 4; // K offset of this lane's bank

        const int Kpp        = pars_.K_per_pixel();
        const int row_stride = pars_.Wo * Kpp;
        // Lane- and wave-uniform parts of the K offset (independent of p/q16).
        //
        // The tile's image origin is folded into the base pointer (standard tile), so
        // no per-image term remains here.
        const int lane_k_base = block_g_idx_ * pars_.K_per_group + wg_k_base_ + k_bank;
        const int n_base      = lane_k_base;

#pragma unroll
        for(int p = 0; p < wave_p; ++p)
        {
            const int p_out = wave_p_base_ + p;
            if constexpr(CheckBounds)
                if(p_out >= pars_.Ho)
                    continue;
            // large_tensor folded wave_p_base_ into the base, so the store row is the
            // wave-relative p; the p_out >= Ho guard above stays absolute.
            const int p_row  = cfg.large_tensor ? p : p_out;
            const int p_base = n_base + p_row * row_stride;
#pragma unroll
            for(int q16 = 0; q16 < wave_q16; ++q16)
            {
                const int q_out = wave_q_base_ + q16 * 16 + q_lane;
                if constexpr(CheckBounds)
                    if(q_out >= pars_.Wo)
                        continue;
                const int px_off = p_base + q_out * Kpp;
#pragma unroll
                for(int k16 = 0; k16 < K16; ++k16)
                {
                    // One dwordx2: 4 contiguous K = k16*16 + k_bank + {0..3}.
                    const vec4_t r = packed_convert<datatype_t>(acc[p][q16][k16]);
                    store_b64(px_off + k16 * 16, r);
                }
            }
        }
    }

    // Narrow store for the wave straddling the real-K boundary (arbitrary-K path).
    //
    // Same lane mapping as emit(), but each channel goes out as a scalar fp16 guarded
    // by k_out < K_per_group, so padding is never written. P and Q always
    // bounds-checked (this wave is rare; no full-tile fast path). Unfold tiles decode
    // the packed (image, column) here too, using the pre-transpose write-side pixel
    // (q16*16 + q_lane), unlike store_staged's post-transpose read-side index.
    __device__ void store_partial_k(const fp32x4_t (&acc)[wave_p][wave_q16][Kwg_k16])
    {
        const int lane_idx = volatile_lane_id();
        const int q_lane   = lane_idx % 16;       // Q within the q16 group
        const int k_bank   = (lane_idx / 16) * 4; // K offset of this lane's bank

        const int Kpp        = pars_.K_per_pixel();
        const int row_stride = pars_.Wo * Kpp;
        // size_t: Ho * row_stride can exceed int32 for a >2 GiB image (used only in
        // the unfold n_local term; large_tensor 3x1 is unfold_n == 1, so it is 0 there).
        const size_t img_stride = static_cast<size_t>(pars_.Ho) * row_stride;
        const int k_real        = pars_.K_per_group;
        // K offset of this lane's channels within the group (a = 0..3 added below).
        const int lane_k_in_group = wg_k_base_ + k_bank;
        const int lane_k_base     = block_g_idx_ * pars_.K_per_group + lane_k_in_group;
        // Image base relative to the folded base pointer (0 for the standard tile).
        //
        // Guards below stay absolute against N.
        const int n_base = static_cast<int>(img_rel_base_ * img_stride) + lane_k_base;

#pragma unroll
        for(int p = 0; p < wave_p; ++p)
        {
            const int p_out = wave_p_base_ + p;
            if(p_out >= pars_.Ho)
                continue;
            // large_tensor folded wave_p_base_ into the base: store row is wave-relative.
            const int p_row  = cfg.large_tensor ? p : p_out;
            const int p_base = n_base + p_row * row_stride;
#pragma unroll
            for(int q16 = 0; q16 < wave_q16; ++q16)
            {
                const int pixel = q16 * 16 + q_lane;
                int px_off;
                if constexpr(cfg.unfold_n == 1)
                {
                    const int q_out = wave_q_base_ + pixel;
                    if(q_out >= pars_.Wo)
                        continue;
                    px_off = p_base + q_out * Kpp;
                }
                else
                {
                    constexpr int w_unfold = cfg.w_unfold();
                    const int n_local      = pixel / w_unfold;
                    const int w_out        = pixel % w_unfold;
                    if(!(wave_img_base_ + n_local < pars_.N && w_out < pars_.Wo))
                        continue;
                    // img_stride is size_t for the large_tensor overflow guard; unfold
                    // configs are never > 2 GiB, so the image term fits int here.
                    px_off = p_base + static_cast<int>(n_local * img_stride) + w_out * Kpp;
                }
#pragma unroll
                for(int k16 = 0; k16 < Kwg_k16; ++k16)
                {
                    const vec4_t r = packed_convert<datatype_t>(acc[p][q16][k16]);
#pragma unroll
                    for(int a = 0; a < 4; ++a)
                    {
                        // Channel within the group; skip if it is K-padding.
                        if(lane_k_in_group + k16 * 16 + a < k_real)
                            store_b16(px_off + k16 * 16 + a, r[a]);
                    }
                }
            }
        }
    }

    // LDS chunk offset (fp16) for staging position (q, k) within one P-row.
    //
    // 8-byte-block XOR swizzle: bank = (k/4) ^ q; offset = (q*16 + bank)*4 + (k%4).
    // Both sides are b64, so the conflict period is per-block. Bank-conflict-free for
    // both opcodes (verified against the MI350X conflict map):
    //   ds_write_b64: a write phase fixes kb, varies q over 0..15, so kb^q is a
    //     bijection -> 16 distinct banks.
    //   ds_read_b64: a read phase covers one even + one odd pixel, so q&1 splits the
    //     32 lanes into two half-bank ranges and kb^q is a bijection in each -> 64 banks.
    __device__ static int stage_off(int q, int k)
    {
        const int blk  = k >> 2;
        const int win  = k & 3;
        const int bank = blk ^ q;
        return (q * kStageBlocksPerRow + bank) * 4 + win;
    }

    // LDS-staged wide store, one P-row at a time.
    //
    // Per row: ds_write_b64 the 4-K runs, ds_read_b64 the transpose, store_b64.
    // Double-buffered (tic/toc) with the write running one row ahead so its LDS
    // latency overlaps the read. The shared lgkmcnt FIFO orders it: each row issues
    // its read, then the NEXT row's write, then drains only the reads; the next
    // row's top drain completes that write before it is read. The write is
    // unconditional (only the global store is bounds-guarded), so the per-row event
    // count is compile-time constant.
    __device__ void store_staged(const fp32x4_t (&acc)[wave_p][wave_q16][Kwg_k16])
    {
        // Pin wave_q16 == 1: the staging loop has no q16 dimension.
        static_assert(wave_q16 == 1, "store_staged does not handle wave_q16 > 1");
        constexpr int K16 = Kwg_k16; // b64 writes per row (4 at K64, 2 at K32)
        constexpr int lpp = Kwg / 4; // lanes per pixel (16 at K64, 8 at K32)
        constexpr int rd_passes =
            16 / (64 / lpp); // 16 pixels / (64/lpp) per pass (4 at K64, 2 at K32)

        // volatile_lane_id() cannot be hoisted out of the round loop.
        //
        // It and its stage_off swizzle chain materialize here in the low-pressure
        // writer tail rather than spanning the round and spilling at K128's ceiling.
        const int lane_idx = volatile_lane_id();
        // Write side: lane holds Q = L%16, K = k16*16 + (L/16)*4 + {0..3}.
        const int w_q     = lane_idx % 16;
        const int w_kbank = (lane_idx / 16) * 4;
        // Read side: lane holds 4 contiguous K of one pixel (b64).
        const int r_pixel0 = lane_idx / lpp;       // pixel within a 4-pixel pass
        const int r_kbase  = (lane_idx % lpp) * 4; // 4-K block this lane reads

        const int Kpp        = pars_.K_per_pixel();
        const int row_stride = pars_.Wo * Kpp;
        // size_t: Ho * row_stride can exceed int32 for a >2 GiB image (used only in
        // the unfold n_local term; large_tensor 3x1 is unfold_n == 1, so it is 0 there).
        const size_t img_stride = static_cast<size_t>(pars_.Ho) * row_stride;
        const int k_base        = block_g_idx_ * pars_.K_per_group + wg_k_base_;
        // Image base relative to the folded base pointer (0 for the standard tile;
        // see store_partial_k). Unfold adds the per-pixel n_local in the loop below.
        const int n_base = static_cast<int>(img_rel_base_ * img_stride) + k_base;

        // Stage one P-row's accumulators into buffer `buf`; unconditional.
        auto stage_write = [&](int p, int buf) {
            datatype_t* base = stage_ + buf * kStageBufFp16;
#pragma unroll
            for(int k16 = 0; k16 < K16; ++k16)
            {
                const vec4_t r = packed_convert<datatype_t>(acc[p][0][k16]);
                *reinterpret_cast<vec4_t*>(base + stage_off(w_q, k16 * 16 + w_kbank)) = r;
            }
        };

        // Prime the pipeline: stage row 0 into buffer 0.
        stage_write(0, 0);

#pragma unroll
        for(int p = 0; p < wave_p; ++p)
        {
            const int buf       = p & 1;
            datatype_t* rd_base = stage_ + buf * kStageBufFp16;

            // Drain this row's stage write before reading it back.
            wait_lgkmcnt_all();

            const int p_out = wave_p_base_ + p;
            const bool p_in = p_out < pars_.Ho;
            // large_tensor folded wave_p_base_ into the base: store row is wave-relative.
            const int p_row  = cfg.large_tensor ? p : p_out;
            const int p_base = n_base + p_row * row_stride;

            // Read this row's transposed (4-contiguous-K-per-lane) values (b64).
            vec4_t v[rd_passes];
#pragma unroll
            for(int rp = 0; rp < rd_passes; ++rp)
                v[rp] = *reinterpret_cast<const vec4_t*>(
                    rd_base + stage_off(rp * (64 / lpp) + r_pixel0, r_kbase));

            // Look-ahead: stage the next row so its write overlaps this row's read.
            const bool has_next = (p + 1 < wave_p);
            if(has_next)
                stage_write(p + 1, (p + 1) & 1);

            // Drain only the reads, leaving the look-ahead write in flight.
            //
            // The reads were issued first, so the FIFO retires them before the K16
            // write events; wait_lgkmcnt<K16> stops there. Final row: no look-ahead,
            // so drain everything.
            if(has_next)
                wait_lgkmcnt<K16>();
            else
                wait_lgkmcnt_all();

#pragma unroll
            for(int rp = 0; rp < rd_passes; ++rp)
            {
                // The wave-relative pixel index within this P-row (0..15).
                const int pixel = rp * (64 / lpp) + r_pixel0;
                if constexpr(cfg.unfold_n == 1)
                {
                    const int q_out = wave_q_base_ + pixel;
                    if(p_in && q_out < pars_.Wo)
                        store_b64(p_base + q_out * Kpp + r_kbase, v[rp]);
                }
                else
                {
                    // Unfold: split the pixel into its packed image and column.
                    //
                    // The image adds img_stride and selects the global batch index
                    // (wave_img_base_ + n_local), guarded against the padded batch tail.
                    constexpr int w_unfold = cfg.w_unfold();
                    const int n_local      = pixel / w_unfold;
                    const int w_out        = pixel % w_unfold;
                    const bool n_in        = wave_img_base_ + n_local < pars_.N;
                    // img_stride is size_t for the large_tensor overflow guard; unfold
                    // configs are never > 2 GiB, so the image term fits int here.
                    if(p_in && n_in && w_out < pars_.Wo)
                        store_b64(p_base + static_cast<int>(n_local * img_stride) + w_out * Kpp +
                                      r_kbase,
                                  v[rp]);
                }
            }
        }
    }

    // Lane within wave, from mbcnt on the exec mask (not threadIdx.x).
    //
    // Reading threadIdx.x here would keep v0 live across the whole loop and spill it
    // under K128's pressure; mbcnt has no such dependence. Emitted as volatile asm so
    // LICM cannot hoist it to round setup, where its live range would spill and each
    // reload's s_waitcnt vmcnt would drain the cross-round prefetch (see store_staged).
    __device__ static int volatile_lane_id()
    {
        int result;
        asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                     "v_mbcnt_hi_u32_b32 %0, -1, %0"
                     : "=v"(result));
        return result;
    }

    const OutputPars& pars_;
    datatype_t* out_;
    __amdgpu_buffer_rsrc_t out_rsrc_;
    datatype_t* stage_;
    int wave_p_base_;
    int wave_q_base_;
    int wg_k_base_;
    int wave_img_base_;
    int img_rel_base_; // this wave's image base relative to the folded out_rsrc_ base
    int block_g_idx_;
};

} // namespace hipconv::cdna4::direct_l1
