// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

namespace ck_tile {

/// @file
/// Weight-streaming bf16 GEMM for decode M (8 to 64 rows) as a megakernel phase, gfx1250:
///   Y[M, N] = epilogue(X[M, K] * W[N, K]^T)
/// X and W bf16 with K contiguous, Y bf16 or fp32 with N contiguous. At these M every byte of W is
/// read once per step and the phase is bound by HBM bandwidth on W.
///
/// Operand swap: v_wmma_f32_16x16x32_bf16 takes 16 weight rows as A and 16 rows of X (tokens) as
/// B, so the streamed operand is dense whatever M is, and a lane's C holds 8 consecutive output
/// features of one token: C element (feature 8 h + v, token t) is in lane t + 16 h, element v.
///
/// Work item = (n block of kNRows = 16 kNTiles features, K split). The waves of a workgroup share
/// the n block and interleave over the split's 64-wide K windows (wave v takes windows v, v + 8,
/// ...), so one step of the workgroup reads 8 consecutive windows, 1 KB of each weight row. The
/// waves' sums are added through LDS in wave order; with k_splits > 1 each item writes a partial
/// and the last of an n block's items adds the partials in split order. Both orders are fixed, so
/// the result is bit-identical from launch to launch and does not depend on the grid.
/// (Not implemented, by design: fp32 atomics into a zeroed Y. That drops the partial buffer and
/// the ticket, but the result depends on arrival order and the epilogue cannot run on the sum.)
///
/// Data path: a lane copies its own 16-byte pieces of W and X (straight in WMMA operand order)
/// into a per-wave LDS ring with global_load_async_to_lds_b128, then reads them back with
/// ds_load_b128. Only the lane that wrote a piece reads it, so the ring needs no barrier, only
/// the wave's own ASYNCcnt. The ring keeps kStages windows of W in flight per wave without
/// holding VGPRs (8 waves x kStages x kNTiles x 2 KB per workgroup). The async loads are inline
/// asm and every stage issues a fixed number of them (stages past the wave's last window re-load
/// a valid window into an unread slot), so the waits are exact compile-time s_wait_asynccnt
/// values; with the builtin + asyncmark the compiler drained the counter to 0 in the loop.
///
/// Cross-phase prefetch (kPrefetch): W does not depend on activations, so Prefetch() issues the
/// first kStages windows of W of a work item into the ring and, optionally, L2 prefetches
/// (global_prefetch_b8) for the next l2_prefetch_windows windows. The megakernel calls it before
/// it waits on the item's dependency, so the weights stream in while the producer finishes; Run()
/// then issues X (which does depend on the producer) and carries on.
///
/// Cache policy (gfx12 TH / scope fields of the async loads): W non-temporal (kCpolW, read once
/// per step; ISA "th:TH_LOAD_NT"), X regular temporal at device scope (kCpolX: every workgroup
/// re-reads X, and X may have been written by an earlier phase of the same launch; ISA
/// "scope:SCOPE_DEV"). L2 prefetches are global_prefetch_b8 with the default policy.
///
/// Counters: slot = one n block of Y columns (kSlotRows = kNRows), and RowRange() of an item is
/// the range of X columns (K) it reads. So a SmallMGemmPhase that consumes the Y of another one
/// may use after_rows: a split waits only for the producer's n blocks that cover its K range.
/// Do not pair after_rows with a consumer whose RowRange means rows of Y (use after_all).
///
/// Scratch for k_splits > 1: partial (PartialSize floats) and tickets (NumTickets words, zeroed
/// once at allocation; the last arriver resets its ticket, so launches need no reset). k_splits
/// must stay the same for every launch that shares the tickets.
namespace small_m {

inline constexpr index_t kLanes      = 32; // wave32
inline constexpr index_t kTileRows   = 16; // rows of a WMMA operand (features or tokens)
inline constexpr index_t kWindowK    = 64; // K per ring stage: two WMMAs, one 128-byte line a row
inline constexpr index_t kChunks     = 4;  // 16-byte pieces a lane loads per 16-row tile and window
inline constexpr index_t kChunkElems = 8;  // bf16 per piece
inline constexpr index_t kPieceBytes = kLanes * 16; // one piece of every lane of a wave

/// gfx12 cache policy (TH and scope fields of a load): regular temporal, non-temporal, high
/// priority temporal, last use; scope bits CU 0, SE 8, device 16, system 24.
inline constexpr index_t kCpolRT        = 0;
inline constexpr index_t kCpolNT        = 1;
inline constexpr index_t kCpolHT        = 2;
inline constexpr index_t kCpolLU        = 3;
inline constexpr index_t kCpolScopeDev  = 16;
inline constexpr index_t kCpolDefaultW  = kCpolNT;
inline constexpr index_t kCpolDefaultX  = kCpolRT | kCpolScopeDev;

/// K offset inside a window of the 8 elements of piece @p c of lane half @p half: a lane loads
/// the contiguous 64 bytes [64 half, 64 half + 64) of the row's 128-byte window.
CK_TILE_HOST_DEVICE constexpr index_t piece_k(index_t half, index_t c)
{
    return 32 * half + kChunkElems * c;
}

/// Piece holding element @p e (0..15) of the operand of WMMA @p j (0..1): 2 j + e / 8. So element
/// e of lane half h of WMMA j is K 32 h + 16 j + e of the window, on A and on B alike. The WMMA
/// pairs A and B elements by (lane half, element), so any K order shared by A and B is valid.
CK_TILE_HOST_DEVICE constexpr index_t wmma_piece(index_t j, index_t e)
{
    return 2 * j + e / kChunkElems;
}

/// Windows [w0, w1) of split @p s of @p k_splits over @p windows windows.
CK_TILE_HOST_DEVICE constexpr index_t split_begin(index_t windows, index_t k_splits, index_t s)
{
    return static_cast<index_t>(static_cast<long_index_t>(s) * windows / k_splits);
}

/// Windows of a split that wave @p wave of @p waves handles: w0 + wave, w0 + wave + waves, ...
CK_TILE_HOST_DEVICE constexpr index_t
wave_windows(index_t w0, index_t w1, index_t wave, index_t waves)
{
    return w1 - w0 > wave ? (w1 - w0 - wave + waves - 1) / waves : 0;
}

/// Round to bf16, nearest even (ck_tile's default float -> bf16 conversion truncates); the host
/// reference uses the same function. Returns the bits.
CK_TILE_HOST_DEVICE constexpr uint16_t bf16_rne_bits(float f)
{
    const uint32_t u = bit_cast<uint32_t>(f);
    return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

CK_TILE_HOST_DEVICE constexpr float bf16_bits_to_float(uint16_t b)
{
    return bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

/// Epilogue: Apply(args, m, n, v) gets the final fp32 sums v[0..8) of Y[m, n .. n + 8) (after
/// split-K) and may change them in place; the phase then converts to the output type and stores.
/// Norms are not epilogues here (see ops/megakernel/epilogue); this is for cheap element-wise ops.
struct EpiPassThrough
{
    struct Args
    {
    };
    CK_TILE_HOST_DEVICE static void Apply(const Args&, index_t, index_t, float (&)[8]) {}
};

/// v = alpha * v.
struct EpiScale
{
    struct Args
    {
        float alpha;
    };
    CK_TILE_HOST_DEVICE static void Apply(const Args& a, index_t, index_t, float (&v)[8])
    {
        for(index_t i = 0; i < 8; ++i)
            v[i] *= a.alpha;
    }
};

/// v = alpha * v + residual[m, n .. n + 8); residual bf16, row stride ld, 16-byte aligned rows.
struct EpiResidualAdd
{
    struct Args
    {
        const bf16_t* residual;
        index_t ld;
        float alpha;
    };
    CK_TILE_HOST_DEVICE static void Apply(const Args& a, index_t m, index_t n, float (&v)[8])
    {
        uint16_t r[8];
        __builtin_memcpy(r, a.residual + static_cast<long_index_t>(m) * a.ld + n, sizeof(r));
        for(index_t i = 0; i < 8; ++i)
            v[i] = a.alpha * v[i] + bf16_bits_to_float(r[i]);
    }
};

template <typename T>
struct is_out_type : std::bool_constant<std::is_same_v<T, bf16_t> || std::is_same_v<T, float>>
{
};

} // namespace small_m

/// @tparam kMaxM_     rows of X the instantiation covers (16 or more, a multiple of 16); runtime
///                    M may be smaller (rows beyond M load a copy of row M - 1 and are dropped)
/// @tparam kNTiles_   16-row weight tiles per work item (kNRows = 16 kNTiles_)
/// @tparam kStages_   ring depth in 64-wide K windows per wave
/// @tparam OutType_   bf16_t (rounded to nearest even) or float
/// @tparam Epilogue_  see small_m::EpiPassThrough
/// @tparam kPrefetch_ provide Prefetch() (kHasPrefetch): W of an item is issued before the wait
/// @tparam kWaves_    waves per workgroup
template <index_t kMaxM_       = 16,
          index_t kNTiles_     = 2,
          index_t kStages_     = 4,
          typename OutType_    = bf16_t,
          typename Epilogue_   = small_m::EpiPassThrough,
          bool kPrefetch_      = true,
          index_t kWaves_      = 8,
          index_t kCpolW_      = small_m::kCpolDefaultW,
          index_t kCpolX_      = small_m::kCpolDefaultX>
struct SmallMGemmPhase
{
    using OutType  = OutType_;
    using Epilogue = Epilogue_;
    using EpiArgs  = typename Epilogue::Args;

    static constexpr index_t kMaxM    = kMaxM_;
    static constexpr index_t kNTiles  = kNTiles_;
    static constexpr index_t kStages  = kStages_;
    static constexpr index_t kWaves   = kWaves_;
    static constexpr index_t kCpolW   = kCpolW_;
    static constexpr index_t kCpolX   = kCpolX_;
    static constexpr bool kHasPrefetch = kPrefetch_;

    // Follows the compile pass (host 64-lane waves, gfx1250 32): the megakernel derives the
    // launch size from kBlockSize / get_warp_size(), the device code uses kWaves.
    static constexpr index_t kBlockSize = kWaves * get_warp_size();

    static constexpr index_t kMTiles   = kMaxM / small_m::kTileRows;
    static constexpr index_t kNRows    = kNTiles * small_m::kTileRows;
    static constexpr index_t kSlotRows = kNRows;

    static constexpr index_t kPiecesW    = kNTiles * small_m::kChunks;
    static constexpr index_t kPiecesX    = kMTiles * small_m::kChunks;
    static constexpr index_t kPieces     = kPiecesW + kPiecesX;
    static constexpr index_t kStageBytes = kPieces * small_m::kPieceBytes;
    static constexpr index_t kRingBytes  = kWaves * kStages * kStageBytes;
    // Wave partials for the in-workgroup reduction, padded against bank conflicts; aliases the
    // ring (a barrier separates the two).
    static constexpr index_t kRedStride = kNRows + 4; // floats; a multiple of 4
    static constexpr index_t kRedBytes  = kWaves * kMaxM * kRedStride * 4;
    static constexpr index_t kFlagOffset = max(kRingBytes, kRedBytes);

    static_assert(kMaxM >= 16 && kMaxM % 16 == 0, "kMaxM must be a multiple of 16");
    static_assert(kNTiles >= 1 && kStages >= 1 && kWaves >= 1, "bad tiling");
    static_assert(small_m::is_out_type<OutType>::value, "OutType must be bf16_t or float");
    static_assert((kNRows / 8) * kMaxM >= 1, "epilogue groups");

    using barrier_pipeline = void;
    using Wmma             = WarpGemmAttributeWmmaImpl_f32_16x16x32_bf16_bf16;
    using AVec             = typename Wmma::AVecType;
    using BVec             = typename Wmma::BVecType;

    struct Args
    {
        const bf16_t* x;   // [M, stride_x]
        const bf16_t* w;   // [N, stride_w]
        OutType* y;        // [M, ld_y]
        float* partial;    // PartialSize floats; unused when k_splits == 1
        uint32_t* tickets; // NumTickets words, zeroed once; unused when k_splits == 1
        index_t M;
        index_t N;
        index_t K;
        index_t stride_x;
        index_t stride_w;
        index_t ld_y;
        index_t k_splits;
        index_t l2_prefetch_windows; // per wave, after the ring; Prefetch() only
        EpiArgs epi;
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kFlagOffset + 16; }

    CK_TILE_HOST_DEVICE static index_t NumNBlocks(const Args& a) { return a.N / kNRows; }
    CK_TILE_HOST_DEVICE static index_t NumWindows(const Args& a) { return a.K / small_m::kWindowK; }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return NumNBlocks(a) * a.k_splits; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return NumNBlocks(a); }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t) { return a.k_splits; }

    /// Items of one n block are adjacent (n block major), so the first slots complete first.
    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work)
    {
        return amd_wave_read_first_lane(work / a.k_splits);
    }

    /// Columns of X (K) the item reads; see the counter note in the file comment.
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t s  = work - SlotOf(a, work) * a.k_splits;
        const index_t nw = NumWindows(a);
        return make_tuple(small_m::split_begin(nw, a.k_splits, s) * small_m::kWindowK,
                          small_m::split_begin(nw, a.k_splits, s + 1) * small_m::kWindowK);
    }

    CK_TILE_HOST static index_t PartialSize(const Args& a)
    {
        return a.k_splits > 1 ? a.k_splits * a.M * a.N : 0;
    }
    CK_TILE_HOST static index_t NumTickets(const Args& a)
    {
        return a.k_splits > 1 ? NumNBlocks(a) : 0;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        const auto aligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 == 0; };
        return a.M >= 1 && a.M <= kMaxM && a.N >= kNRows && a.N % kNRows == 0 &&
               a.K >= small_m::kWindowK && a.K % small_m::kWindowK == 0 && a.stride_x >= a.K &&
               a.stride_x % 8 == 0 && a.stride_w >= a.K && a.stride_w % 8 == 0 && a.ld_y >= a.N &&
               a.ld_y % 8 == 0 && aligned(a.x) && aligned(a.w) && aligned(a.y) &&
               a.k_splits >= 1 && a.k_splits <= NumWindows(a) && a.l2_prefetch_windows >= 0 &&
               // 32-bit byte offsets of the async loads (from the item's first row / from x)
               2 * static_cast<long_index_t>(kNRows) * a.stride_w < (long_index_t{1} << 31) &&
               2 * static_cast<long_index_t>(kMaxM) * a.stride_x < (long_index_t{1} << 31) &&
               (a.k_splits == 1 ||
                (a.partial != nullptr && aligned(a.partial) && a.tickets != nullptr));
    }

    CK_TILE_DEVICE static auto InitBarriers() { return 0; }

    private:
    /// Where a wave's item starts: uniform values shared by Prefetch and Run.
    struct ItemPos
    {
        index_t nb;     // n block
        index_t split;  // K split
        index_t n0;     // first weight row
        index_t w0;     // first window of the split
        index_t nw;     // windows of this wave
        index_t wave;
    };

    CK_TILE_DEVICE static ItemPos Locate(const Args& a, index_t work)
    {
        ItemPos p;
        p.nb          = amd_wave_read_first_lane(work / a.k_splits);
        p.split       = amd_wave_read_first_lane(work - p.nb * a.k_splits);
        p.n0          = p.nb * kNRows;
        const index_t windows = NumWindows(a);
        p.w0          = small_m::split_begin(windows, a.k_splits, p.split);
        const index_t w1 = small_m::split_begin(windows, a.k_splits, p.split + 1);
        p.wave        = get_warp_id();
        p.nw          = amd_wave_read_first_lane(small_m::wave_windows(p.w0, w1, p.wave, kWaves));
        return p;
    }

    /// The window of stage @p i of this wave.
    CK_TILE_DEVICE static index_t WindowOf(const ItemPos& p, index_t i)
    {
        return p.w0 + p.wave + i * kWaves;
    }

    /// LDS address of this lane's 16 bytes of piece 0 of ring slot 0; piece q of slot s is at
    /// + s * kStageBytes + q * kPieceBytes.
    CK_TILE_DEVICE static CK_TILE_LDS_ADDR char* LaneRing(void* smem, const ItemPos& p)
    {
        return reinterpret_cast<CK_TILE_LDS_ADDR char*>(smem) + p.wave * kStages * kStageBytes +
               get_lane_id() * 16;
    }

    CK_TILE_DEVICE static uint32_t LdsAddr(const CK_TILE_LDS_ADDR char* ptr)
    {
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(ptr));
    }

    /// One global_load_async_to_lds_b128: 16 bytes from sbase + voff to LDS address lds. Inline
    /// asm on purpose: the compiler does not track these loads, so it adds no waits of its own
    /// (with the builtin + asyncmark it drained the counter to 0 in the loop); the phase counts
    /// them itself (WaitAsync) and every group has a fixed instruction count. The immediate offset
    /// stays 0: ck_tile's async-load code treats it as added to the LDS address as well, so the
    /// per-piece offsets go into voff and lds explicitly.
    template <index_t kCpol>
    CK_TILE_DEVICE static void
    AsyncPiece(uint32_t lds, uint32_t voff, const void* sbase)
    {
#if defined(__gfx125__)
#define CK_TILE_SMALL_M_ASYNC(POLICY)                                                    \
    asm volatile("global_load_async_to_lds_b128 %0, %1, %2" POLICY                       \
                 :                                                                       \
                 : "v"(lds), "v"(voff), "s"(sbase)                                       \
                 : "memory")
        if constexpr(kCpol == small_m::kCpolRT)
            CK_TILE_SMALL_M_ASYNC("");
        else if constexpr(kCpol == small_m::kCpolNT)
            CK_TILE_SMALL_M_ASYNC(" th:TH_LOAD_NT");
        else if constexpr(kCpol == small_m::kCpolHT)
            CK_TILE_SMALL_M_ASYNC(" th:TH_LOAD_HT");
        else if constexpr(kCpol == small_m::kCpolLU)
            CK_TILE_SMALL_M_ASYNC(" th:TH_LOAD_LU");
        else if constexpr(kCpol == (small_m::kCpolRT | small_m::kCpolScopeDev))
            CK_TILE_SMALL_M_ASYNC(" scope:SCOPE_DEV");
        else if constexpr(kCpol == (small_m::kCpolNT | small_m::kCpolScopeDev))
            CK_TILE_SMALL_M_ASYNC(" th:TH_LOAD_NT scope:SCOPE_DEV");
        else if constexpr(kCpol == (small_m::kCpolHT | small_m::kCpolScopeDev))
            CK_TILE_SMALL_M_ASYNC(" th:TH_LOAD_HT scope:SCOPE_DEV");
        else
            static_assert(kCpol < 0, "unsupported cache policy");
#undef CK_TILE_SMALL_M_ASYNC
#else
        ignore = lds;
        ignore = voff;
        ignore = sbase;
#endif
    }

    /// Waits until at most @p N async loads of this wave are outstanding; orders LDS accesses.
    template <index_t N>
    CK_TILE_DEVICE static void WaitAsync()
    {
#if defined(__gfx125__)
        asm volatile("s_wait_asynccnt %0" ::"n"(N) : "memory");
#endif
    }

    /// Waits for this wave's LDS reads before a ring slot is refilled. The intrinsic (not asm),
    /// so the compiler's own waitcnt tracking sees that the fragments have landed.
    CK_TILE_DEVICE static void WaitLds()
    {
#if defined(__gfx125__)
        llvm_amdgcn_s_wait_dscnt(0);
        asm volatile("" ::: "memory");
#endif
    }

    /// The window stage @p i loads: past the wave's last window it re-loads the split's first
    /// window (in range, likely in L2) into a slot nobody reads, so every group issues the same
    /// number of loads and WaitAsync counts stay compile-time constants.
    CK_TILE_DEVICE static index_t StageWindow(const ItemPos& p, index_t i)
    {
        return i < p.nw ? WindowOf(p, i) : p.w0;
    }

    static constexpr index_t kAsyncW = kPiecesW; // loads per W group
    static constexpr index_t kAsyncX = kPiecesX; // loads per X group
    static constexpr index_t kAsync  = kPieces;

    /// W pieces (kAsyncW loads) of stage @p i into ring slot i % kStages.
    CK_TILE_DEVICE static void
    IssueW(const Args& a, const ItemPos& p, CK_TILE_LDS_ADDR char* ring, index_t i)
    {
        const index_t lane = get_lane_id();
        const index_t lr   = lane % small_m::kTileRows;
        const index_t half = lane / small_m::kTileRows;
        // Uniform base: the item's first weight row; the lane part fits 32 bits (kNRows rows).
        const bf16_t* base = a.w + static_cast<long_index_t>(p.n0) * a.stride_w;
        const uint32_t dst = LdsAddr(ring + (i % kStages) * kStageBytes);
        static_for<0, kNTiles, 1>{}([&](auto t) {
            const uint32_t voff = static_cast<uint32_t>(
                2 * ((t * small_m::kTileRows + lr) * a.stride_w +
                     StageWindow(p, i) * small_m::kWindowK + small_m::piece_k(half, 0)));
            static_for<0, small_m::kChunks, 1>{}([&](auto c) {
                AsyncPiece<kCpolW>(dst + (t * small_m::kChunks + c) * small_m::kPieceBytes,
                                   voff + c * small_m::kChunkElems * 2,
                                   base);
            });
        });
    }

    /// X pieces (kAsyncX loads) of stage @p i. Token rows past M load row M - 1 (dropped later).
    CK_TILE_DEVICE static void
    IssueX(const Args& a, const ItemPos& p, CK_TILE_LDS_ADDR char* ring, index_t i)
    {
        const index_t lane = get_lane_id();
        const index_t lr   = lane % small_m::kTileRows;
        const index_t half = lane / small_m::kTileRows;
        const index_t k    = StageWindow(p, i) * small_m::kWindowK + small_m::piece_k(half, 0);
        const uint32_t dst = LdsAddr(ring + (i % kStages) * kStageBytes);
        static_for<0, kMTiles, 1>{}([&](auto mt) {
            const index_t row   = min(mt * small_m::kTileRows + lr, a.M - 1);
            const uint32_t voff = static_cast<uint32_t>(2 * (row * a.stride_x + k));
            static_for<0, small_m::kChunks, 1>{}([&](auto c) {
                AsyncPiece<kCpolX>(
                    dst + (kPiecesW + mt * small_m::kChunks + c) * small_m::kPieceBytes,
                    voff + c * small_m::kChunkElems * 2,
                    a.x);
            });
        });
    }

    struct Frags
    {
        int32x4_t q[kPieces];
    };

    CK_TILE_DEVICE static void
    LoadFrags(Frags& f, const CK_TILE_LDS_ADDR char* ring, index_t slot)
    {
        const CK_TILE_LDS_ADDR char* src = ring + slot * kStageBytes;
        static_for<0, kPieces, 1>{}([&](auto q) {
            f.q[q] = *reinterpret_cast<const CK_TILE_LDS_ADDR int32x4_t*>(
                src + q * small_m::kPieceBytes);
        });
    }

    template <typename Vec>
    CK_TILE_DEVICE static Vec Operand(const int32x4_t& lo, const int32x4_t& hi)
    {
        const int32x8_t v = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
        return bit_cast<Vec>(v);
    }

    CK_TILE_DEVICE static void Compute(const Frags& f, fp32x8_t (&acc)[kNTiles][kMTiles])
    {
        static_for<0, 2, 1>{}([&](auto j) {
            static_for<0, kNTiles, 1>{}([&](auto t) {
                constexpr index_t qa = decltype(t)::value * small_m::kChunks +
                                       small_m::wmma_piece(decltype(j)::value, 0);
                const AVec av        = Operand<AVec>(f.q[qa], f.q[qa + 1]);
                static_for<0, kMTiles, 1>{}([&](auto mt) {
                    constexpr index_t qb = kPiecesW + decltype(mt)::value * small_m::kChunks +
                                           small_m::wmma_piece(decltype(j)::value, 0);
                    const BVec bv = Operand<BVec>(f.q[qb], f.q[qb + 1]);
                    Wmma{}(acc[t][mt], av, bv);
                });
            });
        });
    }

    /// Epilogue, conversion to OutType, one 16-byte (bf16) or 32-byte (fp32) store.
    CK_TILE_DEVICE static void
    Finish(const Args& a, index_t m, index_t n, float (&v)[8])
    {
        Epilogue::Apply(a.epi, m, n, v);
        void* dst = a.y + static_cast<long_index_t>(m) * a.ld_y + n;
        if constexpr(std::is_same_v<OutType, float>)
        {
            fp32x4_t* d = static_cast<fp32x4_t*>(dst);
            d[0]        = fp32x4_t{v[0], v[1], v[2], v[3]};
            d[1]        = fp32x4_t{v[4], v[5], v[6], v[7]};
        }
        else
        {
            uint32x4_t o;
            static_for<0, 4, 1>{}([&](auto i) {
                o[i.value] = static_cast<uint32_t>(small_m::bf16_rne_bits(v[2 * i.value])) |
                             (static_cast<uint32_t>(small_m::bf16_rne_bits(v[2 * i.value + 1]))
                              << 16);
            });
            *static_cast<uint32x4_t*>(dst) = o;
        }
    }

    public:
    /// Issues W of the item's first kStages windows into this wave's ring slots, plus L2
    /// prefetches for the next l2_prefetch_windows windows. Per wave; the megakernel calls it
    /// right before it waits on the item's dependency, then calls Run().
    CK_TILE_DEVICE static void Prefetch(const Args& a, index_t work, void* smem)
    {
        const ItemPos p                   = Locate(a, work);
        CK_TILE_LDS_ADDR char* const ring = LaneRing(smem, p);
        static_for<0, kStages, 1>{}([&](auto i) { IssueW(a, p, ring, i.value); });

        // L2: lane half h takes windows kStages + 2 i + h; one global_prefetch_b8 per 128-byte
        // line (one window of one weight row).
        const index_t lane = get_lane_id();
        const index_t lr   = lane % small_m::kTileRows;
        const index_t half = lane / small_m::kTileRows;
        const index_t end  = min(p.nw, kStages + a.l2_prefetch_windows);
        for(index_t i = kStages + half; i < end; i += 2)
        {
            const bf16_t* src = a.w + static_cast<long_index_t>(p.n0 + lr) * a.stride_w +
                                WindowOf(p, i) * small_m::kWindowK;
            static_for<0, kNTiles, 1>{}([&](auto t) {
                amd_global_prefetch(src +
                                    static_cast<long_index_t>(t * small_m::kTileRows) * a.stride_w);
            });
        }
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token&)
    {
        const ItemPos p                   = Locate(a, work);
        CK_TILE_LDS_ADDR char* const ring = LaneRing(smem, p);

        // Issue order: W of stages 0..kStages-1 (here, or in Prefetch before the wait), X of the
        // same stages, then W + X of one further stage per step. Each group has a fixed size, so
        // the async loads still allowed in flight when stage i is needed are constants:
        //   stage 0:                    the X of stages 1..kStages-1
        //   stage i + 1 <= kStages - 1: the X of stages i+2..kStages-1, stages kStages..kStages+i
        //   stage i + 1 >= kStages:     the kStages - 1 stages after it
        if constexpr(!kHasPrefetch)
        {
            static_for<0, kStages, 1>{}([&](auto i) { IssueW(a, p, ring, i.value); });
        }
        static_for<0, kStages, 1>{}([&](auto i) { IssueX(a, p, ring, i.value); });

        fp32x8_t acc[kNTiles][kMTiles];
        static_for<0, kNTiles, 1>{}([&](auto t) {
            static_for<0, kMTiles, 1>{}([&](auto mt) { acc[t][mt] = fp32x8_t{0.f}; });
        });

        // Stage i: frags(i) are loading into cur. Refill slot i % kStages with stage i + kStages
        // once they have landed, wait for stage i + 1, start loading it into nxt, compute stage i.
        const auto step = [&](index_t i, Frags& cur, Frags& nxt, auto wait) {
            WaitLds();
            IssueW(a, p, ring, i + kStages);
            IssueX(a, p, ring, i + kStages);
            wait();
            if(i + 1 < p.nw)
                LoadFrags(nxt, ring, (i + 1) % kStages);
            if(i < p.nw)
                Compute(cur, acc);
        };

        Frags fa, fb;
        WaitAsync<(kStages - 1) * kAsyncX>();
        if(p.nw > 0)
            LoadFrags(fa, ring, 0);
        // Prologue: the first kStages - 1 steps, whose wait counts differ.
        static_for<0, kStages - 1, 1>{}([&](auto ic) {
            constexpr index_t i = decltype(ic)::value;
            const auto wait     = [] {
                WaitAsync<(kStages - 2 - i) * kAsyncX + (i + 1) * kAsync>();
            };
            if constexpr(i % 2 == 0)
                step(i, fa, fb, wait);
            else
                step(i, fb, fa, wait);
        });
        // Steady state from step kStages - 1, two steps an iteration (no register copies).
        const auto wait = [] { WaitAsync<(kStages - 1) * kAsync>(); };
        constexpr bool kOddStart = (kStages - 1) % 2 == 1;
        Frags& f0 = kOddStart ? fb : fa;
        Frags& f1 = kOddStart ? fa : fb;
        for(index_t i = kStages - 1; i < p.nw; i += 2)
        {
            step(i, f0, f1, wait);
            if(i + 1 < p.nw)
                step(i + 1, f1, f0, wait);
        }
        // The trailing groups (re-loads past the last window) must land before the ring is reused.
        WaitAsync<0>();

        // In-workgroup reduction: every wave's C into LDS (rows past M are written and ignored),
        // then 8 outputs a thread, adding the waves in order.
        // Rows of kRedStride floats, addressed in 4-float vectors.
        constexpr index_t kRow4 = kRedStride / 4;
        CK_TILE_LDS_ADDR fp32x4_t* red = reinterpret_cast<CK_TILE_LDS_ADDR fp32x4_t*>(smem);
        block_sync_lds(); // the ring is no longer read
        {
            const index_t lane = get_lane_id();
            const index_t tok  = lane % small_m::kTileRows;
            const index_t half = lane / small_m::kTileRows;
            static_for<0, kNTiles, 1>{}([&](auto t) {
                static_for<0, kMTiles, 1>{}([&](auto mt) {
                    CK_TILE_LDS_ADDR fp32x4_t* dst =
                        red + (p.wave * kMaxM + mt * small_m::kTileRows + tok) * kRow4 +
                        (t * small_m::kTileRows + 8 * half) / 4;
                    const fp32x8_t& v = acc[t][mt];
                    dst[0]            = fp32x4_t{v[0], v[1], v[2], v[3]};
                    dst[1]            = fp32x4_t{v[4], v[5], v[6], v[7]};
                });
            });
        }
        block_sync_lds();

        constexpr index_t kGroupsPerRow = kNRows / 8;
        const index_t groups            = a.M * kGroupsPerRow;
        const bool direct               = a.k_splits == 1;
        const index_t n_blocks          = NumNBlocks(a);
        // Partials of an item: [m][kNRows] at partial[((split * n_blocks + nb) * M) * kNRows].
        // Addressed in 4-float vectors.
        const auto chunk = [&](index_t split) {
            return static_cast<fp32x4_t*>(static_cast<void*>(
                a.partial + (static_cast<long_index_t>(split) * n_blocks + p.nb) * a.M * kNRows));
        };
        for(index_t g = threadIdx.x; g < groups; g += kBlockSize)
        {
            const index_t m = g / kGroupsPerRow;
            const index_t c = (g - m * kGroupsPerRow) * 8;
            float v[8];
            static_for<0, 8, 1>{}([&](auto e) { v[e.value] = 0.f; });
            static_for<0, kWaves, 1>{}([&](auto wv) {
                const CK_TILE_LDS_ADDR fp32x4_t* src = red + (wv * kMaxM + m) * kRow4 + c / 4;
                const fp32x4_t lo                    = src[0];
                const fp32x4_t hi                    = src[1];
                static_for<0, 4, 1>{}([&](auto e) {
                    v[e.value] += lo[e.value];
                    v[e.value + 4] += hi[e.value];
                });
            });
            if(direct)
            {
                Finish(a, m, p.n0 + c, v);
            }
            else
            {
                fp32x4_t* dst = chunk(p.split) + (m * kNRows + c) / 4;
                dst[0]        = fp32x4_t{v[0], v[1], v[2], v[3]};
                dst[1]        = fp32x4_t{v[4], v[5], v[6], v[7]};
            }
        }
        if(direct)
            return;

        // Split-K: publish the partial, take a ticket; the last of the n block's k_splits items
        // adds the partials in split order and runs the epilogue.
        CK_TILE_LDS_ADDR uint32_t* flag = reinterpret_cast<CK_TILE_LDS_ADDR uint32_t*>(
            reinterpret_cast<CK_TILE_LDS_ADDR char*>(smem) + kFlagOffset);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            uint32_t* ticket = a.tickets + p.nb;
            const uint32_t prev =
                __scoped_atomic_fetch_add(ticket, 1u, __ATOMIC_ACQ_REL, __MEMORY_SCOPE_DEVICE);
            const bool last = prev == static_cast<uint32_t>(a.k_splits - 1);
            if(last)
                __scoped_atomic_store_n(ticket, 0u, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            *flag = last ? 1u : 0u;
        }
        block_sync_lds();
        if(amd_wave_read_first_lane(*flag) == 0)
            return;
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");

        for(index_t g = threadIdx.x; g < groups; g += kBlockSize)
        {
            const index_t m = g / kGroupsPerRow;
            const index_t c = (g - m * kGroupsPerRow) * 8;
            float v[8];
            static_for<0, 8, 1>{}([&](auto e) { v[e.value] = 0.f; });
            for(index_t s = 0; s < a.k_splits; ++s)
            {
                const fp32x4_t* src = chunk(s) + (m * kNRows + c) / 4;
                const fp32x4_t lo   = src[0];
                const fp32x4_t hi   = src[1];
                static_for<0, 4, 1>{}([&](auto e) {
                    v[e.value] += lo[e.value];
                    v[e.value + 4] += hi[e.value];
                });
            }
            Finish(a, m, p.n0 + c, v);
        }
    }
};

} // namespace ck_tile
