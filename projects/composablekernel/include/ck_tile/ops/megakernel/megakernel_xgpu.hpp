// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/megakernel_sync.hpp"

namespace ck_tile {

/// @file
/// Cross-GPU (tensor parallel over XGMI peer memory) synchronisation for megakernel phases:
/// system-scope release / acquire helpers, a cross-rank barrier, and fence-free (Lamport style)
/// all-reduce and all-gather phases for decode-sized rows (8-64 rows, a few KB to 16 KB each).
/// Peer mapping (IPC handles, hipDeviceEnablePeerAccess) stays with the host: the phases take the
/// peers' exchange buffers as a device table of pointers that are valid on this device.
///
/// Memory scopes on gfx1250 (what the compiler emits, checked in the ISA):
///   relaxed load / store, SYSTEM scope  global_load / global_store ... scope:SCOPE_SYS
///   acquire load                        + s_wait_loadcnt 0, global_inv scope:SCOPE_SYS
///   release store / fetch_add           global_wb scope:SCOPE_SYS, s_wait_storecnt 0, then the op
///   fence(acquire | release, system)    global_inv / global_wb scope:SCOPE_SYS (+ wait)
/// The exchange packets use buffer_load_b128 / buffer_store_b128 with cache policy scope:SCOPE_SYS
/// (stores also th:TH_STORE_NT): 16-byte accesses that bypass the non-coherent caches without a
/// fence. 128-bit atomics do not exist; volatile 128-bit accesses become flat ops with a wait after
/// each, which would serialise the polls.
///
/// Exchange buffer (one per rank, peers map each other's): kBuffers epoch buffers of `world`
/// slots of slot_bytes each, buffer b / slot q at (b * world + q) * slot_bytes. Rank r writes its
/// data into slot r of buffer (seq mod kBuffers) of every rank, including itself, then reads
/// slots 0..world-1 of its own buffer. Allocate it uncached / fine-grained (hipExtMallocWithFlags
/// hipDeviceMallocUncached, as AITER's custom all-reduce does) so no L2 line of it is ever dirty or
/// stale, and fill it once with Codec::kFillWord (hipMemsetD32); it is never reset by the host.
///
/// Sequence numbers: one per use of an exchange, seq = epoch * uses_per_launch + use_index, with
/// epoch the Megakernel launch epoch (word 0 of its counter buffer, + 1). Every rank runs the
/// same launches and uses, so the sequence agrees across ranks with no host reset, and a graph can
/// replay the launch. kBuffers is a power of two so seq mod kBuffers stays continuous when seq
/// wraps at 2^32.
///
/// Two packet protocols (Codec):
///  - xgpu_sentinel_codec<bf16_t | float>: the data itself is the flag. The empty value is -0
///    (0x8000 / 0x80000000); writers turn -0 into +0 (a sum or a copy cannot tell them apart), so a
///    slot holds data exactly when no element is -0. Readers poll every element, then write -0 back
///    (the reset), so the slot is empty for its next use. Full bandwidth, one reset store per read.
///  - xgpu_ll_codec: raw bytes (fp8 / fp4 activations, e8m0 scales, anything). Each 16-byte packet
///    carries 8 data bytes as two {data, flag} 8-byte halves with flag = the sequence number; a
///    reader waits for both flags. No reset and no forbidden value, half the bandwidth. Assumes the
///    8-byte aligned halves of a 128-bit store land atomically (a 64-bit atomic store compiles to
///    the same global_store_b64 path); the sentinel codec only needs 2- / 4-byte atomicity.
///
/// Ordering without fences. A reader's reset of a cell must land before any peer writes that cell
/// again, kBuffers uses later; a writer must not overwrite a cell a peer is still reading. Both
/// follow from causality as long as, on each rank, the same workgroup handles the same cell in
/// every use of an exchange within a launch (all its stages have the same NumWork; the push and
/// consume stages of a split use too; launches are ordered by the kernel boundary, and ranks may
/// use different grids) and a workgroup waits for its own stores
/// (s_wait_storecnt 0, normally long done) before each push. With kBuffers = 2: a peer writes cell
/// c of use s + 2 only after it read c of use s + 1, which this rank pushed after its reset of c in
/// use s completed. Uses with different shapes need separate exchanges unless a data dependency
/// (the next push waits on a counter of a stage that ran after the reset) orders them. A larger
/// kBuffers only adds slack.
///
/// Waits are bounded: after spin_limit unsuccessful polls (0 = unbounded) the wave executes
/// s_trap, so a missing peer aborts the launch instead of hanging the GPU.

// ---------------------------------------------------------------------------------------------
// System-scope primitives

/// Loads, stores and fences at system scope, for flags shared with peers or the host.
struct xgpu_sys
{
    CK_TILE_DEVICE static uint32_t load(const uint32_t* p)
    {
        return __scoped_atomic_load_n(
            const_cast<uint32_t*>(p), __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }

    CK_TILE_DEVICE static uint32_t load_acquire(const uint32_t* p)
    {
        return __scoped_atomic_load_n(
            const_cast<uint32_t*>(p), __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SYSTEM);
    }

    CK_TILE_DEVICE static void store(uint32_t* p, uint32_t v)
    {
        __scoped_atomic_store_n(p, v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
    }

    CK_TILE_DEVICE static void store_release(uint32_t* p, uint32_t v)
    {
        __scoped_atomic_store_n(p, v, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }

    CK_TILE_DEVICE static uint32_t fetch_add_release(uint32_t* p, uint32_t v)
    {
        return __scoped_atomic_fetch_add(p, v, __ATOMIC_RELEASE, __MEMORY_SCOPE_SYSTEM);
    }

    /// Writes back this wave's dirty lines and waits for its stores: global_wb SCOPE_SYS.
    CK_TILE_DEVICE static void fence_release() { __builtin_amdgcn_fence(__ATOMIC_RELEASE, ""); }

    /// Invalidates this wave's non-coherent cache lines: global_inv SCOPE_SYS.
    CK_TILE_DEVICE static void fence_acquire() { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, ""); }

    /// Waits until this wave's stores are acknowledged (no cache write-back).
    CK_TILE_DEVICE static void wait_stores() { buffer_store_fence(0); }

    /// Backoff between polls: short at first, then longer so a waiting wave adds little traffic.
    CK_TILE_DEVICE static void backoff(uint32_t polls)
    {
        if(polls < 64)
            __builtin_amdgcn_s_sleep(1);
        else
            __builtin_amdgcn_s_sleep(8);
    }

    /// Aborts the launch (s_trap): a peer did not arrive within the spin limit.
    [[noreturn]] CK_TILE_DEVICE static void timeout() { __builtin_trap(); }

    /// Spins until @p flag reaches @p target (modulo 2^32), then acquires at system scope.
    /// Traps after @p spin_limit polls (0 = no limit).
    CK_TILE_DEVICE static void wait(const uint32_t* flag, uint32_t target, uint32_t spin_limit)
    {
        uint32_t polls = 0;
        while(!megakernel_counter::reached(__builtin_amdgcn_readfirstlane(xgpu_sys::load(flag)),
                                           target))
        {
            if(spin_limit != 0 && ++polls > spin_limit)
                timeout();
            backoff(polls);
        }
        fence_acquire();
    }
};

/// Barrier across the ranks, collective over the calling workgroup. @p flags[p] is rank p's flag
/// array of kMaxWorld words (fine-grained memory, zeroed once); one workgroup per rank uses a given
/// array, with @p seq increasing by one per barrier. Releases this workgroup's writes to the peers
/// (system scope) and acquires theirs. Costly (a full L2 write-back): for hand-offs to the host or
/// between launches, not for the per-layer exchanges, which use the phases below.
template <index_t kMaxWorld>
CK_TILE_DEVICE void xgpu_rank_barrier(uint32_t* const (&flags)[kMaxWorld],
                                      index_t rank,
                                      index_t world,
                                      uint32_t seq,
                                      uint32_t spin_limit)
{
    xgpu_sys::fence_release();
    __syncthreads();
    const index_t t = get_thread_id();
    if(t < world)
    {
        // Constant indices only (a runtime index into a kernel argument array goes to scratch).
        uint32_t* mine = flags[0];
        uint32_t* to   = flags[0];
        static_for<1, kMaxWorld, 1>{}([&](auto p) {
            mine = p.value == rank ? flags[p.value] : mine;
            to   = p.value == t ? flags[p.value] : to;
        });
        xgpu_sys::store_release(to + rank, seq);
        uint32_t polls = 0;
        while(!megakernel_counter::reached(xgpu_sys::load(mine + t), seq))
        {
            if(spin_limit != 0 && ++polls > spin_limit)
                xgpu_sys::timeout();
            xgpu_sys::backoff(polls);
        }
    }
    __syncthreads();
    xgpu_sys::fence_acquire();
}

/// The rank barrier as a megakernel stage: one work item (workgroup 0's), so give it an after_all
/// dependency on the stage whose completion it should publish, and make later stages depend on it
/// (after_all) to hold them until every rank got there. seq = the launch sequence of the use.
template <index_t kNumWarps, index_t kMaxWorld = 8>
struct XgpuRankBarrierPhase
{
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();
    static constexpr index_t kSlotRows  = 1;
    using barrier_pipeline              = void;
    CK_TILE_DEVICE static auto InitBarriers() {}

    struct Args
    {
        uint32_t* flags[kMaxWorld]; ///< flags[p] = rank p's kMaxWorld-word flag array
        index_t rank;
        index_t world;
        const uint32_t* launch_state; ///< the Megakernel counter buffer
        uint32_t uses_per_launch;     ///< barriers on these flags per launch
        uint32_t use_index;
        uint32_t spin_limit;
    };

    CK_TILE_HOST_DEVICE static index_t NumWork(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args&) { return 1; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args&, index_t) { return 1; }
    CK_TILE_DEVICE static index_t SlotOf(const Args&, index_t) { return 0; }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args&, index_t)
    {
        return make_tuple(0, 1);
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        bool ok = a.world >= 1 && a.world <= kMaxWorld && a.rank >= 0 && a.rank < a.world &&
                  a.launch_state != nullptr && a.uses_per_launch >= 1 &&
                  a.use_index < a.uses_per_launch;
        for(index_t p = 0; ok && p < a.world; ++p)
            ok = a.flags[p] != nullptr;
        return ok;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t, void*, const Token&)
    {
        const uint32_t epoch = megakernel_counter::load(a.launch_state) + 1;
        xgpu_rank_barrier<kMaxWorld>(
            a.flags, a.rank, a.world, epoch * a.uses_per_launch + a.use_index, a.spin_limit);
    }
};

// ---------------------------------------------------------------------------------------------
// Packets and codecs (host and device: the CPU emulation runs the same code)

/// The 16 bytes one lane moves per access.
struct xgpu_vec
{
    uint32_t w[4];
};

/// Data is its own flag; the empty value is -0. T is bf16_t or float.
template <typename T>
struct xgpu_sentinel_codec
{
    static_assert(std::is_same_v<T, bf16_t> || std::is_same_v<T, float>,
                  "the sentinel codec handles bf16 and fp32 data");
    using DataType                       = T;
    static constexpr index_t kElemBytes  = sizeof(T);
    static constexpr index_t kElems      = 16 / kElemBytes; ///< elements per packet
    static constexpr index_t kDataBytes  = 16;              ///< payload bytes per packet
    static constexpr bool kNeedsReset    = true;
    static constexpr uint32_t kSentinel  = kElemBytes == 2 ? 0x8000u : 0x80000000u;
    static constexpr uint32_t kFillWord  = kElemBytes == 2 ? 0x80008000u : 0x80000000u;
    static constexpr uint32_t kHalfShift = 16;

    /// Raw bits of element @p i.
    CK_TILE_HOST_DEVICE static uint32_t bits(const xgpu_vec& v, index_t i)
    {
        if constexpr(kElemBytes == 4)
            return v.w[i];
        else
            return (v.w[i / 2] >> (kHalfShift * (i % 2))) & 0xFFFFu;
    }

    CK_TILE_HOST_DEVICE static float to_float(uint32_t b)
    {
        return bit_cast<float>(kElemBytes == 2 ? b << kHalfShift : b);
    }

    /// -0 -> +0, so no written element equals the sentinel.
    CK_TILE_HOST_DEVICE static xgpu_vec encode(xgpu_vec v, uint32_t /*seq*/)
    {
        for(index_t i = 0; i < 4; ++i)
        {
            if constexpr(kElemBytes == 4)
            {
                v.w[i] = v.w[i] == kSentinel ? 0u : v.w[i];
            }
            else
            {
                const uint32_t lo = v.w[i] & 0xFFFFu;
                const uint32_t hi = v.w[i] >> kHalfShift;
                v.w[i] = (lo == kSentinel ? 0u : lo) | ((hi == kSentinel ? 0u : hi) << kHalfShift);
            }
        }
        return v;
    }

    CK_TILE_HOST_DEVICE static xgpu_vec decode(const xgpu_vec& v) { return v; }

    /// True while any element still holds the sentinel.
    CK_TILE_HOST_DEVICE static bool pending(const xgpu_vec& v, uint32_t /*seq*/)
    {
        bool p = false;
        for(index_t i = 0; i < 4; ++i)
        {
            if constexpr(kElemBytes == 4)
                p = p || v.w[i] == kSentinel;
            else
                p = p || (v.w[i] & 0xFFFFu) == kSentinel || (v.w[i] >> kHalfShift) == kSentinel;
        }
        return p;
    }

    CK_TILE_HOST_DEVICE static constexpr xgpu_vec empty()
    {
        return xgpu_vec{{kFillWord, kFillWord, kFillWord, kFillWord}};
    }
};

/// Raw bytes: packet {d0, flag, d1, flag}, 8 data bytes per 16-byte packet, flag = sequence
/// number. Sequence 0 (a wrap) is flagged 1: buffer 0 only ever holds even flags (kBuffers is
/// even), so a stale flag never matches, and the zero fill never matches either.
struct xgpu_ll_codec
{
    static constexpr index_t kDataBytes = 8;
    static constexpr bool kNeedsReset   = false;
    static constexpr uint32_t kFillWord = 0;

    CK_TILE_HOST_DEVICE static uint32_t flag(uint32_t seq) { return seq == 0 ? 1u : seq; }

    /// @p v carries the data in w[0], w[1].
    CK_TILE_HOST_DEVICE static xgpu_vec encode(const xgpu_vec& v, uint32_t seq)
    {
        const uint32_t f = flag(seq);
        return xgpu_vec{{v.w[0], f, v.w[1], f}};
    }

    /// Data back in w[0], w[1].
    CK_TILE_HOST_DEVICE static xgpu_vec decode(const xgpu_vec& v)
    {
        return xgpu_vec{{v.w[0], v.w[2], 0u, 0u}};
    }

    CK_TILE_HOST_DEVICE static bool pending(const xgpu_vec& v, uint32_t seq)
    {
        const uint32_t f = flag(seq);
        return v.w[1] != f || v.w[3] != f;
    }

    CK_TILE_HOST_DEVICE static constexpr xgpu_vec empty() { return xgpu_vec{{0u, 0u, 0u, 0u}}; }
};

/// One rank's view of an exchange: where the peers' buffers are and which use this is.
template <index_t kMaxWorld_>
struct xgpu_exchange
{
    static constexpr index_t kMaxWorld = kMaxWorld_;

    /// Device-visible table of pointers (kMaxWorld entries, the first `world` used): peers[p] =
    /// rank p's exchange buffer, valid on this device (16-byte aligned). A table rather than kernel
    /// arguments: a megakernel with several exchange stages would otherwise keep 8 pointers per
    /// stage live in SGPRs for the whole launch (SGPR spills); a table entry is one scalar load
    /// when it is needed.
    void* const* peers;
    index_t rank;                 ///< this rank, in [0, world)
    index_t world;                ///< 1..kMaxWorld (1 = self loop)
    uint32_t slot_bytes;          ///< bytes per (epoch buffer, rank) slot, a multiple of 16
    uint32_t buffer_bytes;        ///< kBuffers * world * slot_bytes (< 4 GiB: 32-bit offsets)
    const uint32_t* launch_state; ///< the Megakernel counter buffer (word 0: last epoch)
    uint32_t uses_per_launch;     ///< uses of this exchange per launch (same on every rank)
    uint32_t use_index;           ///< which of them this phase is
    uint32_t spin_limit;          ///< polls before s_trap, 0 = unbounded

    CK_TILE_HOST_DEVICE uint32_t seq(uint32_t epoch) const
    {
        return epoch * uses_per_launch + use_index;
    }

    /// Rank @p p's exchange buffer. On the device the table is read as constant memory (scalar
    /// load; it must not change during a launch) and the result marked as a global pointer, so
    /// accesses derived from it are global rather than flat.
    CK_TILE_HOST_DEVICE void* peer(index_t p) const
    {
#if defined(__HIP_DEVICE_COMPILE__)
        using const_table_t = void* const __attribute__((address_space(4)))*;
        void* v             = c_style_pointer_cast<const_table_t>(peers)[p];
        return c_style_pointer_cast<void*>(c_style_pointer_cast<CK_TILE_GLOBAL_ADDR char*>(v));
#else
        return peers[p];
#endif
    }

    /// This rank's exchange buffer.
    CK_TILE_HOST_DEVICE void* own() const { return peer(rank); }
};

/// The packet steps of a use, generic over the memory interface (device: buffer ops at system
/// scope; CPU emulation: atomics). Mem provides store(p, off, v) into rank p's buffer, load(off)
/// and store_own(off, v) on this rank's buffer, and backoff(polls).
template <typename Codec, index_t kMaxWorld>
struct xgpu_packets
{
    /// Writes this rank's packet at @p off (inside its slot) into every rank's buffer.
    template <typename Mem>
    CK_TILE_HOST_DEVICE static void
    push(const Mem& m, index_t world, uint32_t off, const xgpu_vec& data, uint32_t seq)
    {
        const xgpu_vec pkt = Codec::encode(data, seq);
        static_for<0, kMaxWorld, 1>{}([&](auto p) {
            if(p.value < world)
                m.store(p.value, off, pkt);
        });
    }

    /// Loads the packet at @p off (slot 0 of the epoch buffer) of every slot until none is
    /// pending. Returns false after @p spin_limit unsuccessful polls (0 = no limit).
    template <typename Mem>
    CK_TILE_HOST_DEVICE static bool wait(const Mem& m,
                                         index_t world,
                                         uint32_t off,
                                         uint32_t slot_bytes,
                                         uint32_t seq,
                                         uint32_t spin_limit,
                                         xgpu_vec (&x)[kMaxWorld])
    {
        // All loads first, so the round trips overlap; then re-poll only what is missing.
        static_for<0, kMaxWorld, 1>{}([&](auto q) {
            if(q.value < world)
                x[q.value] = m.load(off + q.value * slot_bytes);
        });
        uint32_t polls = 0;
        bool ok        = true;
        static_for<0, kMaxWorld, 1>{}([&](auto q) {
            if(q.value < world)
            {
                while(ok && Codec::pending(x[q.value], seq))
                {
                    if(spin_limit != 0 && ++polls > spin_limit)
                        ok = false;
                    else
                    {
                        m.backoff(polls);
                        x[q.value] = m.load(off + q.value * slot_bytes);
                    }
                }
            }
        });
        return ok;
    }

    /// Empties the packet at @p off of every slot (sentinel codec only).
    template <typename Mem>
    CK_TILE_HOST_DEVICE static void
    reset(const Mem& m, index_t world, uint32_t off, uint32_t slot_bytes)
    {
        if constexpr(Codec::kNeedsReset)
        {
            static_for<0, kMaxWorld, 1>{}([&](auto q) {
                if(q.value < world)
                    m.store_own(off + q.value * slot_bytes, Codec::empty());
            });
        }
        else
        {
            ignore = m;
            ignore = world;
            ignore = off;
            ignore = slot_bytes;
        }
    }
};

/// Device memory interface of xgpu_packets: 16-byte buffer ops with scope:SCOPE_SYS.
template <index_t kMaxWorld>
struct xgpu_device_mem
{
    // gfx12 cache policy: TH in bits [2:0], scope in [4:3].
    static constexpr index_t kScopeSys = 3 << 3;
    static constexpr index_t kThNt     = 1;

    const xgpu_exchange<kMaxWorld>& x;

    CK_TILE_DEVICE void store(index_t p, uint32_t off, const xgpu_vec& v) const
    {
        store_to(x.peer(p), off, v);
    }

    CK_TILE_DEVICE void store_own(uint32_t off, const xgpu_vec& v) const
    {
        store_to(x.own(), off, v);
    }

    CK_TILE_DEVICE xgpu_vec load(uint32_t off) const
    {
        return bit_cast<xgpu_vec>(
            llvm_amdgcn_raw_buffer_load_i32x4(make_wave_buffer_resource(x.own(), x.buffer_bytes),
                                              static_cast<index_t>(off),
                                              0,
                                              kScopeSys));
    }

    CK_TILE_DEVICE void store_to(const void* base, uint32_t off, const xgpu_vec& v) const
    {
        llvm_amdgcn_raw_buffer_store_i32x4(bit_cast<int32x4_t>(v),
                                           make_wave_buffer_resource(base, x.buffer_bytes),
                                           static_cast<index_t>(off),
                                           0,
                                           kScopeSys | kThNt);
    }

    CK_TILE_DEVICE void backoff(uint32_t polls) const { xgpu_sys::backoff(polls); }
};

/// Which part of a use a stage runs. kFused pushes then consumes each item; kPush and kConsume
/// split them into two stages (same NumWork), so every push of a workgroup is issued before its
/// first wait.
enum class xgpu_part
{
    kFused,
    kPush,
    kConsume,
};

namespace impl {
template <index_t kNumWarps, index_t kMaxWorld, index_t kBuffers>
struct xgpu_phase_base
{
    static_assert(kBuffers >= 2 && (kBuffers & (kBuffers - 1)) == 0,
                  "kBuffers must be a power of two >= 2 (wrap-safe sequence numbers)");
    static_assert(kMaxWorld >= 1 && kMaxWorld <= 16, "kMaxWorld out of range");

    using Exchange = xgpu_exchange<kMaxWorld>;

    /// As GemmPhase: waves times the wave size of the compile pass (host passes assume 64).
    static constexpr index_t kBlockSize = kNumWarps * get_warp_size();
    /// Lanes of one workgroup on the target (wave32): one packet per lane per item. Fixed, so the
    /// host and device passes agree on NumWork.
    static constexpr index_t kLanes    = kNumWarps * 32;
    static constexpr index_t kSlotRows = 1;
    using barrier_pipeline             = void;
    CK_TILE_DEVICE static auto InitBarriers() {}

    CK_TILE_HOST static uint32_t ExchangeBytes(index_t world, uint32_t slot_bytes)
    {
        return static_cast<uint32_t>(kBuffers * world) * slot_bytes;
    }

    /// Offset of slot @p q of the epoch buffer of @p seq.
    CK_TILE_HOST_DEVICE static uint32_t SlotBase(const Exchange& x, uint32_t seq, index_t q)
    {
        return ((seq % kBuffers) * static_cast<uint32_t>(x.world) + static_cast<uint32_t>(q)) *
               x.slot_bytes;
    }

    CK_TILE_DEVICE static uint32_t DeviceSeq(const Exchange& x)
    {
        // Within a launch word 0 still holds the previous epoch (the Megakernel publishes the new
        // one after every workgroup has left).
        return x.seq(megakernel_counter::load(x.launch_state) + 1);
    }

    CK_TILE_HOST static bool IsSupportedExchange(const Exchange& x, uint32_t needed_slot_bytes)
    {
        // The peer table is device memory: only its presence is checked here.
        return x.world >= 1 && x.world <= kMaxWorld && x.rank >= 0 && x.rank < x.world &&
               x.peers != nullptr && x.launch_state != nullptr && x.uses_per_launch >= 1 &&
               x.use_index < x.uses_per_launch && x.slot_bytes % 16 == 0 &&
               x.slot_bytes >= needed_slot_bytes &&
               static_cast<uint64_t>(kBuffers) * x.world * x.slot_bytes <= 0xFFFFFFFFull &&
               x.buffer_bytes == ExchangeBytes(x.world, x.slot_bytes);
    }
};
} // namespace impl

// ---------------------------------------------------------------------------------------------
// All-reduce epilogues

/// Default all-reduce epilogue: stores the fp32 sums as OutT (bf16, round to nearest even; or
/// fp32), row-major.
/// An epilogue of this kind sees N consecutive columns of one row in registers (fused elementwise
/// work: residual add, scaling, quantisation of a column group).
template <typename OutT_ = bf16_t>
struct xgpu_store_epilogue
{
    using OutT                      = remove_cvref_t<OutT_>;
    static constexpr bool kRowPhase = false;

    struct Args
    {
        OutT* out;      ///< [rows, stride]
        index_t stride; ///< elements
    };

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a, index_t width, index_t vec)
    {
        return a.out != nullptr && a.stride >= width && a.stride % vec == 0 &&
               reinterpret_cast<uintptr_t>(a.out) % (vec * sizeof(OutT)) == 0;
    }

    template <index_t N>
    CK_TILE_DEVICE static void Apply(const Args& a, index_t row, index_t col, const float (&v)[N])
    {
        using vec_t = OutT __attribute__((ext_vector_type(N)));
        vec_t o;
        static_for<0, N, 1>{}([&](auto i) {
            if constexpr(std::is_same_v<OutT, bf16_t>)
                o[i.value] = float_to_bf16<bf16_rounding_mode::standard>(v[i.value]);
            else
                o[i.value] = type_convert<OutT>(v[i.value]);
        });
        *c_style_pointer_cast<vec_t*>(a.out + static_cast<long_index_t>(row) * a.stride + col) = o;
    }
};

/// Whole-row epilogue: once every rank's partial of a row has arrived, runs @p RowPhase on that
/// row with in[q] = the exchange slot of rank q (it sums them in rank order). Meant for the post
/// all-reduce norms, e.g. MkResidualNormPhase<8192, W, kRms, bf16_t, 8> (sum, norm, residual
/// gate, next norm in one pass). RowPhase needs Args::in[], num_inputs, in_stride, work item =
/// row, and in-element type equal to the exchange type. It reads the slots a second time after
/// the poll (served from the local HBM; the all-reduce item is then a whole row).
template <typename RowPhase_>
struct xgpu_row_phase_epilogue
{
    using RowPhase                  = remove_cvref_t<RowPhase_>;
    using Args                      = typename RowPhase::Args;
    static constexpr bool kRowPhase = true;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return RowPhase::GetSmemSize(); }
};

// ---------------------------------------------------------------------------------------------
// All-reduce

/// One-shot Lamport all-reduce of a [rows, width] partial (sum over the ranks), for decode M.
/// Work item = (row, tile of kTileCols columns), one 16-byte packet per lane; with a row-phase
/// epilogue, item = row. Each item pushes its tile to every rank, waits for every rank's tile in
/// its own buffer, sums in fp32 in rank order (so every rank gets bit-identical results), hands the
/// sums to the epilogue and resets the slots.
///
/// Dependencies: the push reads `in`, so the stage waits on its producer with after_rows; later
/// stages wait on this one with after_rows (kSlotRows = 1, one slot per row).
template <typename T_,
          index_t kNumWarps_,
          typename Epilogue_ = xgpu_store_epilogue<bf16_t>,
          index_t kMaxWorld_ = 8,
          index_t kBuffers_  = 2,
          xgpu_part kPart_   = xgpu_part::kFused>
struct XgpuAllReducePhase : impl::xgpu_phase_base<kNumWarps_, kMaxWorld_, kBuffers_>
{
    using Base     = impl::xgpu_phase_base<kNumWarps_, kMaxWorld_, kBuffers_>;
    using T        = remove_cvref_t<T_>;
    using Epilogue = remove_cvref_t<Epilogue_>;
    using Codec    = xgpu_sentinel_codec<T>;
    using Packets  = xgpu_packets<Codec, kMaxWorld_>;
    using Base::kBlockSize;
    using Base::kLanes;
    using typename Base::Exchange;

    static constexpr index_t kMaxWorld  = kMaxWorld_;
    static constexpr index_t kBuffers   = kBuffers_;
    static constexpr xgpu_part kPart    = kPart_;
    static constexpr index_t kElems     = Codec::kElems;
    static constexpr index_t kTileCols  = kLanes * kElems;
    static constexpr bool kRowEpilogue  = Epilogue::kRowPhase;
    static constexpr uint32_t kFillWord = Codec::kFillWord;

    struct Args
    {
        Exchange x;
        const T* in;       ///< this rank's partial [rows, in_stride]
        index_t in_stride; ///< elements
        index_t rows;      ///< same on every rank
        index_t width;     ///< multiple of kElems
        typename Epilogue::Args epi;
    };

    CK_TILE_HOST_DEVICE static index_t TilesPerRow(const Args& a)
    {
        return kRowEpilogue ? 1 : integer_divide_ceil(a.width, kTileCols);
    }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return a.rows * TilesPerRow(a); }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return a.rows; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t)
    {
        return TilesPerRow(a);
    }
    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work)
    {
        return work / TilesPerRow(a);
    }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t r = work / TilesPerRow(a);
        return make_tuple(r, r + 1);
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Epilogue::GetSmemSize(); }

    /// Slot size for @p rows x @p width (every use sharing the exchange must fit).
    CK_TILE_HOST static uint32_t SlotBytes(index_t rows, index_t width)
    {
        return static_cast<uint32_t>(rows * width * static_cast<index_t>(sizeof(T)));
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        bool ok = a.rows > 0 && a.width > 0 && a.width % kElems == 0 && a.in_stride % kElems == 0 &&
                  a.in_stride >= a.width && a.in != nullptr &&
                  reinterpret_cast<uintptr_t>(a.in) % 16 == 0 &&
                  Base::IsSupportedExchange(a.x, SlotBytes(a.rows, a.width));
        if constexpr(kRowEpilogue)
        {
            using RowPhase = typename Epilogue::RowPhase;
            static_assert(RowPhase::kBlockSize == kBlockSize, "row phase block size differs");
            static_assert(std::is_same_v<remove_cvref_t<decltype(*a.epi.in[0])>, T>,
                          "row phase input type must be the exchange type");
            constexpr index_t kInputs = sizeof(a.epi.in) / sizeof(a.epi.in[0]);
            // The slots are bound on the device; any non-null pointer passes the host check.
            auto ra = a.epi;
            for(index_t q = 0; q < kInputs; ++q)
                ra.in[q] = a.in;
            ra.num_inputs = a.x.world;
            ra.in_stride  = a.width;
            ok            = ok && a.x.world <= kInputs && RowPhase::IsSupportedArgument(ra) &&
                 RowPhase::NumWork(ra) == a.rows;
        }
        else
        {
            ok = ok && Epilogue::IsSupportedArgument(a.epi, a.width, kElems);
        }
        return ok;
    }

    // Packet steps, shared with the CPU emulation.

    CK_TILE_HOST_DEVICE static uint32_t CellOffset(const Args& a, index_t row, index_t col)
    {
        return static_cast<uint32_t>((row * a.width + col) * static_cast<index_t>(sizeof(T)));
    }

    /// Pushes the packet of this rank's partial at (row, col).
    template <typename Mem>
    CK_TILE_HOST_DEVICE static void
    PushPacket(const Args& a, const Mem& m, uint32_t seq, index_t row, index_t col)
    {
        const xgpu_vec d = m.template load_input<xgpu_vec>(
            a.in + static_cast<long_index_t>(row) * a.in_stride + col);
        Packets::push(
            m, a.x.world, Base::SlotBase(a.x, seq, a.x.rank) + CellOffset(a, row, col), d, seq);
    }

    /// Waits for the packets of every rank at (row, col), sums them in rank order, resets them.
    template <typename Mem>
    CK_TILE_HOST_DEVICE static bool ReducePacket(
        const Args& a, const Mem& m, uint32_t seq, index_t row, index_t col, float (&acc)[kElems])
    {
        const uint32_t off = Base::SlotBase(a.x, seq, 0) + CellOffset(a, row, col);
        xgpu_vec x[kMaxWorld];
        if(!Packets::wait(m, a.x.world, off, a.x.slot_bytes, seq, a.x.spin_limit, x))
            return false;
        static_for<0, kElems, 1>{}([&](auto e) { acc[e.value] = 0.f; });
        static_for<0, kMaxWorld, 1>{}([&](auto q) {
            if(q.value < a.x.world)
            {
                static_for<0, kElems, 1>{}([&](auto e) {
                    acc[e.value] += Codec::to_float(Codec::bits(x[q.value], e.value));
                });
            }
        });
        Packets::reset(m, a.x.world, off, a.x.slot_bytes);
        return true;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token& token)
    {
        const uint32_t seq = Base::DeviceSeq(a.x);
        const device_mem m{a.x};
        if constexpr(kRowEpilogue)
        {
            RunRow(a, m, seq, work, smem, token);
        }
        else
        {
            ignore                   = smem;
            ignore                   = token;
            const index_t tiles      = TilesPerRow(a);
            const index_t row        = work / tiles;
            const index_t col0       = (work % tiles) * kTileCols;
            constexpr index_t kIters = integer_divide_ceil(kLanes, kBlockSize);
            auto for_each_packet     = [&](auto&& f) {
                static_for<0, kIters, 1>{}([&](auto i) {
                    const index_t lane = get_thread_id() + i.value * kBlockSize;
                    const index_t col  = col0 + lane * kElems;
                    if(lane < kLanes && col < a.width)
                        f(col);
                });
            };
            if constexpr(kPart != xgpu_part::kConsume)
            {
                // This workgroup's resets of earlier uses have landed (see the ordering note).
                xgpu_sys::wait_stores();
                for_each_packet([&](index_t col) { PushPacket(a, m, seq, row, col); });
            }
            if constexpr(kPart != xgpu_part::kPush)
            {
                for_each_packet([&](index_t col) {
                    float acc[kElems];
                    if(!ReducePacket(a, m, seq, row, col, acc))
                        xgpu_sys::timeout();
                    Epilogue::Apply(a.epi, row, col, acc);
                });
            }
        }
    }

    private:
    struct device_mem : xgpu_device_mem<kMaxWorld>
    {
        template <typename V, typename P>
        CK_TILE_DEVICE static V load_input(const P* p)
        {
            return bit_cast<V>(*c_style_pointer_cast<const int32x4_t*>(p));
        }
    };

    template <typename Token>
    CK_TILE_DEVICE static void RunRow(const Args& a,
                                      const device_mem& m,
                                      uint32_t seq,
                                      index_t row,
                                      void* smem,
                                      const Token& token)
    {
        using RowPhase        = typename Epilogue::RowPhase;
        const index_t packets = a.width / kElems;
        const uint32_t off0   = Base::SlotBase(a.x, seq, 0) + CellOffset(a, row, 0);
        if constexpr(kPart != xgpu_part::kConsume)
        {
            xgpu_sys::wait_stores();
            for(index_t v = get_thread_id(); v < packets; v += kBlockSize)
                PushPacket(a, m, seq, row, v * kElems);
        }
        if constexpr(kPart != xgpu_part::kPush)
        {
            for(index_t v = get_thread_id(); v < packets; v += kBlockSize)
            {
                xgpu_vec x[kMaxWorld];
                if(!Packets::wait(m,
                                  a.x.world,
                                  off0 + static_cast<uint32_t>(v * 16),
                                  a.x.slot_bytes,
                                  seq,
                                  a.x.spin_limit,
                                  x))
                    xgpu_sys::timeout();
            }
            // The row phase reads the slots with ordinary loads: no wave may read before every
            // packet is in, and none may keep a stale line.
            __syncthreads();
            xgpu_sys::fence_acquire();

            auto ra              = a.epi;
            const char* own      = static_cast<const char*>(a.x.own());
            constexpr index_t kN = sizeof(ra.in) / sizeof(ra.in[0]);
            static_for<0, kN, 1>{}([&](auto q) {
                if(q.value < a.x.world)
                    ra.in[q.value] =
                        reinterpret_cast<const T*>(own + Base::SlotBase(a.x, seq, q.value));
            });
            ra.num_inputs = a.x.world;
            ra.in_stride  = a.width;
            RowPhase::Run(ra, row, smem, token);

            // Every wave has consumed its loads of the row before any slot is emptied.
            __syncthreads();
            for(index_t v = get_thread_id(); v < packets; v += kBlockSize)
                Packets::reset(m, a.x.world, off0 + static_cast<uint32_t>(v * 16), a.x.slot_bytes);
        }
    }
};

// ---------------------------------------------------------------------------------------------
// All-gather

/// All-gather of each rank's [rows, shard_bytes] shard: afterwards every rank holds all shards,
/// rank q's at out + q * out_rank_stride (row-major with out_row_stride). out_rank_stride =
/// shard_bytes, out_row_stride = world * shard_bytes gives [rows, world * shard] (N-sharded GEMM
/// outputs); out_rank_stride = rows * shard_bytes gives rank-major blocks.
/// Codec: xgpu_sentinel_codec<bf16_t | float> for activations (full bandwidth; -0 arrives as +0),
/// xgpu_ll_codec for raw bytes (fp8 / fp4 data, e8m0 scales; bit exact, half bandwidth).
/// Work item = (row, tile of kTileBytes shard bytes), one packet per lane.
template <typename Codec_,
          index_t kNumWarps_,
          index_t kMaxWorld_ = 8,
          index_t kBuffers_  = 2,
          xgpu_part kPart_   = xgpu_part::kFused>
struct XgpuAllGatherPhase : impl::xgpu_phase_base<kNumWarps_, kMaxWorld_, kBuffers_>
{
    using Base    = impl::xgpu_phase_base<kNumWarps_, kMaxWorld_, kBuffers_>;
    using Codec   = remove_cvref_t<Codec_>;
    using Packets = xgpu_packets<Codec, kMaxWorld_>;
    using Base::kBlockSize;
    using Base::kLanes;
    using typename Base::Exchange;

    static constexpr index_t kMaxWorld  = kMaxWorld_;
    static constexpr index_t kBuffers   = kBuffers_;
    static constexpr xgpu_part kPart    = kPart_;
    static constexpr index_t kData      = Codec::kDataBytes; ///< data bytes per packet
    static constexpr index_t kExpand    = 16 / kData;        ///< exchange bytes per data byte
    static constexpr index_t kTileBytes = kLanes * kData;
    static constexpr uint32_t kFillWord = Codec::kFillWord;

    struct Args
    {
        Exchange x;
        const void* in;      ///< this rank's shard [rows, in_stride bytes]
        index_t in_stride;   ///< bytes
        index_t rows;        ///< same on every rank
        index_t shard_bytes; ///< multiple of kData, same on every rank
        void* out;
        long_index_t out_row_stride;  ///< bytes
        long_index_t out_rank_stride; ///< bytes
    };

    CK_TILE_HOST_DEVICE static index_t TilesPerRow(const Args& a)
    {
        return integer_divide_ceil(a.shard_bytes, kTileBytes);
    }
    CK_TILE_HOST_DEVICE static index_t NumWork(const Args& a) { return a.rows * TilesPerRow(a); }
    CK_TILE_HOST_DEVICE static index_t NumSlots(const Args& a) { return a.rows; }
    CK_TILE_HOST_DEVICE static index_t ItemsPerSlot(const Args& a, index_t)
    {
        return TilesPerRow(a);
    }
    CK_TILE_DEVICE static index_t SlotOf(const Args& a, index_t work)
    {
        return work / TilesPerRow(a);
    }
    CK_TILE_DEVICE static tuple<index_t, index_t> RowRange(const Args& a, index_t work)
    {
        const index_t r = work / TilesPerRow(a);
        return make_tuple(r, r + 1);
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return 0; }

    CK_TILE_HOST static uint32_t SlotBytes(index_t rows, index_t shard_bytes)
    {
        return static_cast<uint32_t>(rows * shard_bytes * kExpand);
    }

    CK_TILE_HOST static bool IsSupportedArgument(const Args& a)
    {
        return a.rows > 0 && a.shard_bytes > 0 && a.shard_bytes % kData == 0 &&
               a.in_stride % kData == 0 && a.in_stride >= a.shard_bytes && a.in != nullptr &&
               a.out != nullptr && reinterpret_cast<uintptr_t>(a.in) % kData == 0 &&
               reinterpret_cast<uintptr_t>(a.out) % kData == 0 && a.out_row_stride % kData == 0 &&
               a.out_rank_stride % kData == 0 &&
               Base::IsSupportedExchange(a.x, SlotBytes(a.rows, a.shard_bytes));
    }

    CK_TILE_HOST_DEVICE static uint32_t CellOffset(const Args& a, index_t row, index_t c)
    {
        return static_cast<uint32_t>((row * a.shard_bytes + c) * kExpand);
    }

    CK_TILE_HOST_DEVICE static long_index_t
    OutOffset(const Args& a, index_t q, index_t row, index_t c)
    {
        return q * a.out_rank_stride + row * a.out_row_stride + c;
    }

    /// Pushes this rank's packet at (row, shard byte c).
    template <typename Mem>
    CK_TILE_HOST_DEVICE static void
    PushPacket(const Args& a, const Mem& m, uint32_t seq, index_t row, index_t c)
    {
        const auto* src =
            static_cast<const char*>(a.in) + static_cast<long_index_t>(row) * a.in_stride + c;
        xgpu_vec d;
        if constexpr(kData == 16)
            d = m.template load_input<xgpu_vec>(src);
        else
        {
            const uint32x2_t h = m.template load_input<uint32x2_t>(src);
            d                  = xgpu_vec{{h[0], h[1], 0u, 0u}};
        }
        Packets::push(
            m, a.x.world, Base::SlotBase(a.x, seq, a.x.rank) + CellOffset(a, row, c), d, seq);
    }

    /// Waits for every rank's packet at (row, c), hands rank q's data to store(q, data), resets.
    template <typename Mem, typename Store>
    CK_TILE_HOST_DEVICE static bool
    GatherPacket(const Args& a, const Mem& m, uint32_t seq, index_t row, index_t c, Store&& store)
    {
        const uint32_t off = Base::SlotBase(a.x, seq, 0) + CellOffset(a, row, c);
        xgpu_vec x[kMaxWorld];
        if(!Packets::wait(m, a.x.world, off, a.x.slot_bytes, seq, a.x.spin_limit, x))
            return false;
        static_for<0, kMaxWorld, 1>{}([&](auto q) {
            if(q.value < a.x.world)
                store(q.value, Codec::decode(x[q.value]));
        });
        Packets::reset(m, a.x.world, off, a.x.slot_bytes);
        return true;
    }

    template <typename Token>
    CK_TILE_DEVICE static void Run(const Args& a, index_t work, void* smem, const Token& token)
    {
        ignore             = smem;
        ignore             = token;
        const uint32_t seq = Base::DeviceSeq(a.x);
        const device_mem m{a.x};
        const index_t tiles      = TilesPerRow(a);
        const index_t row        = work / tiles;
        const index_t c0         = (work % tiles) * kTileBytes;
        constexpr index_t kIters = integer_divide_ceil(kLanes, kBlockSize);
        auto for_each_packet     = [&](auto&& f) {
            static_for<0, kIters, 1>{}([&](auto i) {
                const index_t lane = get_thread_id() + i.value * kBlockSize;
                const index_t c    = c0 + lane * kData;
                if(lane < kLanes && c < a.shard_bytes)
                    f(c);
            });
        };
        if constexpr(kPart != xgpu_part::kConsume)
        {
            xgpu_sys::wait_stores();
            for_each_packet([&](index_t c) { PushPacket(a, m, seq, row, c); });
        }
        if constexpr(kPart != xgpu_part::kPush)
        {
            for_each_packet([&](index_t c) {
                char* out     = static_cast<char*>(a.out);
                const bool ok = GatherPacket(a, m, seq, row, c, [&](index_t q, const xgpu_vec& d) {
                    char* dst = out + OutOffset(a, q, row, c);
                    if constexpr(kData == 16)
                        *c_style_pointer_cast<int32x4_t*>(dst) = bit_cast<int32x4_t>(d);
                    else
                        *c_style_pointer_cast<uint32x2_t*>(dst) = uint32x2_t{d.w[0], d.w[1]};
                });
                if(!ok)
                    xgpu_sys::timeout();
            });
        }
    }

    private:
    struct device_mem : xgpu_device_mem<kMaxWorld>
    {
        template <typename V>
        CK_TILE_DEVICE static V load_input(const char* p)
        {
            if constexpr(sizeof(V) == 16)
                return bit_cast<V>(*c_style_pointer_cast<const int32x4_t*>(p));
            else
                return *c_style_pointer_cast<const V*>(p);
        }
    };
};

} // namespace ck_tile
