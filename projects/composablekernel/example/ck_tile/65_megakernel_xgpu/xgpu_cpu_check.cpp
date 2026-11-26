// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host-only check of the cross-GPU exchange protocol of megakernel_xgpu.hpp. CPU threads play the
// ranks and their workgroups and call the header's own packet code (PushPacket / ReducePacket /
// GatherPacket over xgpu_packets) through an atomic memory interface that writes each 16-byte
// packet as two 8-byte halves in random order (torn packets), with random yields. Per launch:
//   stage 0  all-reduce, exchange A use 0            (bf16, sentinel codec, 2 buffers)
//   stage 1  all-reduce of stage 0's output, A use 1 (after_rows on stage 0, emulated counters)
//   stage 2  raw-byte all-gather, exchange B          (LL codec, 2 buffers)
//   stage 3  bf16 all-gather push, exchange C         (sentinel codec, 4 buffers)
//   stage 4  bf16 all-gather consume, exchange C
// Ranks get different workgroup counts, launches are not synchronised across ranks (only the
// launch end within a rank, like a kernel boundary), the launch epoch starts just below 2^32 so
// sequence numbers wrap (LL sequence 0 included), and inputs change every launch. Every rank's
// outputs must match the host reference bit for bit. A last check removes a rank and expects the
// bounded wait to time out instead of hanging.

#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/megakernel/megakernel_xgpu.hpp"

namespace {

using ck_tile::index_t;
using ck_tile::xgpu_vec;
using bf16_t = ck_tile::bf16_t;

constexpr index_t kWarps    = 2; // 64 lanes per item on the target: several tiles per row
constexpr index_t kMaxWorld = 8;

using Ar =
    ck_tile::XgpuAllReducePhase<bf16_t, kWarps, ck_tile::xgpu_store_epilogue<bf16_t>, kMaxWorld, 2>;
using AgB = ck_tile::XgpuAllGatherPhase<ck_tile::xgpu_ll_codec, kWarps, kMaxWorld, 2>;
using AgH = ck_tile::XgpuAllGatherPhase<ck_tile::xgpu_sentinel_codec<bf16_t>, kWarps, kMaxWorld, 4>;
using Exch = ck_tile::xgpu_exchange<kMaxWorld>;

uint16_t bf16_bits(float f)
{
    return ck_tile::bit_cast<uint16_t>(
        ck_tile::float_to_bf16<ck_tile::bf16_rounding_mode::standard>(f));
}
float bf16_to_float(uint16_t b) { return ck_tile::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
uint16_t canon(uint16_t b) { return b == 0x8000u ? 0 : b; }

/// Memory interface of xgpu_packets over host buffers of 64-bit words.
struct HostMem
{
    std::vector<std::vector<uint64_t>>* bufs;
    index_t rank;
    std::mt19937_64* rng;

    void store(index_t p, uint32_t off, const xgpu_vec& v) const
    {
        uint64_t* d       = (*bufs)[static_cast<size_t>(p)].data() + off / 8;
        const uint64_t lo = v.w[0] | (static_cast<uint64_t>(v.w[1]) << 32);
        const uint64_t hi = v.w[2] | (static_cast<uint64_t>(v.w[3]) << 32);
        const uint64_t r  = (*rng)();
        if(r & 1)
        {
            __atomic_store_n(d, lo, __ATOMIC_RELAXED);
            if(r & 6)
                std::this_thread::yield();
            __atomic_store_n(d + 1, hi, __ATOMIC_RELAXED);
        }
        else
        {
            __atomic_store_n(d + 1, hi, __ATOMIC_RELAXED);
            if(r & 6)
                std::this_thread::yield();
            __atomic_store_n(d, lo, __ATOMIC_RELAXED);
        }
    }
    void store_own(uint32_t off, const xgpu_vec& v) const { store(rank, off, v); }
    xgpu_vec load(uint32_t off) const
    {
        const uint64_t* s = (*bufs)[static_cast<size_t>(rank)].data() + off / 8;
        const uint64_t lo = __atomic_load_n(s, __ATOMIC_RELAXED);
        const uint64_t hi = __atomic_load_n(s + 1, __ATOMIC_RELAXED);
        return xgpu_vec{{static_cast<uint32_t>(lo),
                         static_cast<uint32_t>(lo >> 32),
                         static_cast<uint32_t>(hi),
                         static_cast<uint32_t>(hi >> 32)}};
    }
    void backoff(uint32_t) const { std::this_thread::yield(); }
    template <typename V, typename P>
    static V load_input(const P* p)
    {
        V v;
        std::memcpy(&v, p, sizeof(V));
        return v;
    }
};

struct HostExchange
{
    std::vector<std::vector<uint64_t>> bufs;
    std::vector<void*> table;
    uint32_t slot_bytes, buffer_bytes;
    HostExchange(int world, uint32_t slot, uint32_t total, uint32_t fill)
        : bufs(static_cast<size_t>(world),
               std::vector<uint64_t>(total / 8, fill | (static_cast<uint64_t>(fill) << 32))),
          slot_bytes(slot),
          buffer_bytes(total)
    {
        for(auto& b : bufs)
            table.push_back(b.data());
    }
    Exch view(int rank, const uint32_t* state, uint32_t uses, uint32_t use, uint32_t spin)
    {
        Exch x{};
        x.peers           = table.data();
        x.rank            = rank;
        x.world           = static_cast<index_t>(bufs.size());
        x.slot_bytes      = slot_bytes;
        x.buffer_bytes    = buffer_bytes;
        x.launch_state    = state;
        x.uses_per_launch = uses;
        x.use_index       = use;
        x.spin_limit      = spin;
        return x;
    }
};

std::vector<uint16_t> random_bf16(size_t n, uint64_t seed)
{
    std::mt19937_64 g(seed);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<uint16_t> v(n);
    for(auto& x : v)
    {
        const uint64_t r = g() % 16;
        x                = r == 0 ? uint16_t{0x8000} : r == 1 ? uint16_t{0} : bf16_bits(d(g));
    }
    return v;
}

struct Config
{
    int world       = 8;
    int launches    = 40;
    index_t rows    = 3;
    index_t width   = 1088;        // 512-column tiles: two full, one partial
    index_t shardh  = 192;         // bf16 columns per rank (3 tiles of 64 at 1 packet / lane)
    index_t shardb  = 136;         // bytes per rank (LL: 8 per packet)
    uint32_t state0 = 0xFFFFFFFDu; // epochs 0xFFFFFFFE, 0xFFFFFFFF, 0, 1, ...
};

int run_protocol(const Config& c)
{
    const int Wd    = c.world;
    const index_t M = c.rows;
    HostExchange exA(Wd,
                     Ar::SlotBytes(M, c.width),
                     Ar::ExchangeBytes(Wd, Ar::SlotBytes(M, c.width)),
                     Ar::kFillWord);
    HostExchange exB(Wd,
                     AgB::SlotBytes(M, c.shardb),
                     AgB::ExchangeBytes(Wd, AgB::SlotBytes(M, c.shardb)),
                     AgB::kFillWord);
    HostExchange exC(Wd,
                     AgH::SlotBytes(M, c.shardh * 2),
                     AgH::ExchangeBytes(Wd, AgH::SlotBytes(M, c.shardh * 2)),
                     AgH::kFillWord);

    // Inputs per (launch, rank); outputs per (launch, rank).
    const size_t L = static_cast<size_t>(c.launches), R = static_cast<size_t>(Wd);
    std::vector<std::vector<uint16_t>> in0(L * R), out0(L * R), out1(L * R), inh(L * R),
        outh(L * R);
    std::vector<std::vector<uint8_t>> inb(L * R), outb(L * R);
    for(size_t l = 0; l < L; ++l)
    {
        for(size_t r = 0; r < R; ++r)
        {
            const size_t i = l * R + r;
            in0[i]         = random_bf16(size_t(M) * c.width, 17 * i + 1);
            inh[i]         = random_bf16(size_t(M) * c.shardh, 17 * i + 2);
            inb[i].resize(size_t(M) * c.shardb);
            std::mt19937_64 g(17 * i + 3);
            for(auto& b : inb[i])
                b = static_cast<uint8_t>(g());
            out0[i].assign(size_t(M) * c.width, 0xFFFF);
            out1[i].assign(size_t(M) * c.width, 0xFFFF);
            outh[i].assign(size_t(M) * c.shardh * Wd, 0xFFFF);
            outb[i].assign(size_t(M) * c.shardb * Wd, 0xFF);
        }
    }

    std::atomic<int> timeouts{0};
    std::vector<uint32_t> state(R, c.state0);
    // Stage 0 row counters, as the Megakernel keeps them: never reset, target epoch * items.
    const index_t tilesA = (c.width + Ar::kTileCols - 1) / Ar::kTileCols;
    std::vector<std::atomic<uint32_t>> rowdone(R * static_cast<size_t>(M));
    for(size_t k = 0; k < rowdone.size(); ++k)
        rowdone[k].store(c.state0 * static_cast<uint32_t>(tilesA));

    std::vector<std::thread> threads;
    std::vector<std::unique_ptr<std::barrier<>>> launch_end;
    for(int r = 0; r < Wd; ++r)
        launch_end.emplace_back(new std::barrier<>(1 + r % 3));

    for(int r = 0; r < Wd; ++r)
    {
        const int groups = 1 + r % 3;
        for(int g = 0; g < groups; ++g)
        {
            threads.emplace_back([&, r, g, groups]() {
                std::mt19937_64 rng(1000 * r + g);
                const uint32_t spin = 1u << 22;
                for(size_t l = 0; l < L; ++l)
                {
                    const size_t i = l * R + static_cast<size_t>(r);
                    // Launch start: every workgroup reads the previous epoch.
                    const uint32_t epoch =
                        __atomic_load_n(&state[static_cast<size_t>(r)], __ATOMIC_ACQUIRE) + 1;
                    auto items = [&](index_t n, auto&& f) {
                        for(index_t w = g; w < n; w += groups)
                            f(w);
                    };

                    // Stages 0 and 1: all-reduce twice on exchange A.
                    for(uint32_t use = 0; use < 2; ++use)
                    {
                        const std::vector<uint16_t>& src = use == 0 ? in0[i] : out0[i];
                        std::vector<uint16_t>& dst       = use == 0 ? out0[i] : out1[i];
                        Ar::Args a{};
                        a.x         = exA.view(r, &state[static_cast<size_t>(r)], 2, use, spin);
                        a.in        = reinterpret_cast<const bf16_t*>(src.data());
                        a.in_stride = c.width;
                        a.rows      = M;
                        a.width     = c.width;
                        const HostMem m{&exA.bufs, r, &rng};
                        const uint32_t seq = a.x.seq(epoch);
                        items(Ar::NumWork(a), [&](index_t w) {
                            const index_t row = w / tilesA, col0 = (w % tilesA) * Ar::kTileCols;
                            if(use == 1) // after_rows<0>
                            {
                                const uint32_t target = epoch * static_cast<uint32_t>(tilesA);
                                while(!ck_tile::megakernel_counter::reached(
                                    rowdone[static_cast<size_t>(r) * static_cast<size_t>(M) +
                                            static_cast<size_t>(row)]
                                        .load(std::memory_order_acquire),
                                    target))
                                    std::this_thread::yield();
                            }
                            for(index_t lane = 0; lane < Ar::kLanes; ++lane)
                            {
                                const index_t col = col0 + lane * Ar::kElems;
                                if(col < c.width)
                                    Ar::PushPacket(a, m, seq, row, col);
                            }
                            for(index_t lane = 0; lane < Ar::kLanes; ++lane)
                            {
                                const index_t col = col0 + lane * Ar::kElems;
                                if(col >= c.width)
                                    continue;
                                float acc[Ar::kElems];
                                if(!Ar::ReducePacket(a, m, seq, row, col, acc))
                                {
                                    ++timeouts;
                                    return;
                                }
                                for(index_t e = 0; e < Ar::kElems; ++e)
                                    dst[size_t(row) * c.width + col + e] = bf16_bits(acc[e]);
                            }
                            if(use == 0)
                                rowdone[static_cast<size_t>(r) * static_cast<size_t>(M) +
                                        static_cast<size_t>(row)]
                                    .fetch_add(1, std::memory_order_release);
                        });
                    }

                    // Stage 2: LL byte gather into rank-major blocks.
                    {
                        AgB::Args a{};
                        a.x               = exB.view(r, &state[static_cast<size_t>(r)], 1, 0, spin);
                        a.in              = inb[i].data();
                        a.in_stride       = c.shardb;
                        a.rows            = M;
                        a.shard_bytes     = c.shardb;
                        a.out             = outb[i].data();
                        a.out_row_stride  = c.shardb;
                        a.out_rank_stride = static_cast<ck_tile::long_index_t>(M) * c.shardb;
                        const HostMem m{&exB.bufs, r, &rng};
                        const uint32_t seq  = a.x.seq(epoch);
                        const index_t tiles = AgB::TilesPerRow(a);
                        items(AgB::NumWork(a), [&](index_t w) {
                            const index_t row = w / tiles, c0 = (w % tiles) * AgB::kTileBytes;
                            for(index_t lane = 0; lane < AgB::kLanes; ++lane)
                                if(c0 + lane * AgB::kData < c.shardb)
                                    AgB::PushPacket(a, m, seq, row, c0 + lane * AgB::kData);
                            for(index_t lane = 0; lane < AgB::kLanes; ++lane)
                            {
                                const index_t cc = c0 + lane * AgB::kData;
                                if(cc >= c.shardb)
                                    continue;
                                const bool ok = AgB::GatherPacket(
                                    a, m, seq, row, cc, [&](index_t q, const xgpu_vec& d) {
                                        std::memcpy(
                                            outb[i].data() + AgB::OutOffset(a, q, row, cc), d.w, 8);
                                    });
                                if(!ok)
                                {
                                    ++timeouts;
                                    return;
                                }
                            }
                        });
                    }

                    // Stages 3 and 4: bf16 gather, push stage then consume stage.
                    {
                        AgH::Args a{};
                        a.x               = exC.view(r, &state[static_cast<size_t>(r)], 1, 0, spin);
                        a.in              = inh[i].data();
                        a.in_stride       = c.shardh * 2;
                        a.rows            = M;
                        a.shard_bytes     = c.shardh * 2;
                        a.out             = outh[i].data();
                        a.out_row_stride  = static_cast<ck_tile::long_index_t>(Wd) * c.shardh * 2;
                        a.out_rank_stride = c.shardh * 2;
                        const HostMem m{&exC.bufs, r, &rng};
                        const uint32_t seq  = a.x.seq(epoch);
                        const index_t tiles = AgH::TilesPerRow(a);
                        items(AgH::NumWork(a), [&](index_t w) {
                            const index_t row = w / tiles, c0 = (w % tiles) * AgH::kTileBytes;
                            for(index_t lane = 0; lane < AgH::kLanes; ++lane)
                                if(c0 + lane * AgH::kData < a.shard_bytes)
                                    AgH::PushPacket(a, m, seq, row, c0 + lane * AgH::kData);
                        });
                        items(AgH::NumWork(a), [&](index_t w) {
                            const index_t row = w / tiles, c0 = (w % tiles) * AgH::kTileBytes;
                            for(index_t lane = 0; lane < AgH::kLanes; ++lane)
                            {
                                const index_t cc = c0 + lane * AgH::kData;
                                if(cc >= a.shard_bytes)
                                    continue;
                                const bool ok = AgH::GatherPacket(
                                    a, m, seq, row, cc, [&](index_t q, const xgpu_vec& d) {
                                        std::memcpy(reinterpret_cast<char*>(outh[i].data()) +
                                                        AgH::OutOffset(a, q, row, cc),
                                                    d.w,
                                                    16);
                                    });
                                if(!ok)
                                {
                                    ++timeouts;
                                    return;
                                }
                            }
                        });
                    }

                    // Kernel end: the last workgroup publishes the epoch.
                    launch_end[static_cast<size_t>(r)]->arrive_and_wait();
                    if(g == 0)
                        __atomic_store_n(&state[static_cast<size_t>(r)], epoch, __ATOMIC_RELEASE);
                    launch_end[static_cast<size_t>(r)]->arrive_and_wait();
                    if(timeouts.load() != 0)
                        return;
                }
            });
        }
    }
    for(auto& t : threads)
        t.join();
    if(timeouts.load() != 0)
    {
        std::printf("  %d waits timed out\n", timeouts.load());
        return 1;
    }

    int bad     = 0;
    auto report = [&](const char* what, size_t l, size_t r, size_t k, unsigned got, unsigned want) {
        if(bad++ < 5)
            std::printf(
                "  launch %zu rank %zu %s[%zu] = 0x%x, expected 0x%x\n", l, r, what, k, got, want);
    };
    for(size_t l = 0; l < L; ++l)
    {
        const size_t n = size_t(M) * c.width;
        std::vector<uint16_t> ref0(n), ref1(n);
        for(size_t k = 0; k < n; ++k)
        {
            float s = 0.f;
            for(size_t r = 0; r < R; ++r)
                s += bf16_to_float(canon(in0[l * R + r][k]));
            ref0[k]  = bf16_bits(s);
            float s1 = 0.f;
            for(size_t r = 0; r < R; ++r)
                s1 += bf16_to_float(canon(ref0[k]));
            ref1[k] = bf16_bits(s1);
        }
        for(size_t r = 0; r < R; ++r)
        {
            const size_t i = l * R + r;
            for(size_t k = 0; k < n; ++k)
            {
                if(out0[i][k] != ref0[k])
                    report("allreduce0", l, r, k, out0[i][k], ref0[k]);
                if(out1[i][k] != ref1[k])
                    report("allreduce1", l, r, k, out1[i][k], ref1[k]);
            }
            for(size_t q = 0; q < R; ++q)
            {
                for(index_t row = 0; row < M; ++row)
                {
                    for(index_t j = 0; j < c.shardb; ++j)
                    {
                        const size_t k =
                            (q * size_t(M) + size_t(row)) * size_t(c.shardb) + size_t(j);
                        const uint8_t want = inb[l * R + q][size_t(row) * c.shardb + j];
                        if(outb[i][k] != want)
                            report("gather_bytes", l, r, k, outb[i][k], want);
                    }
                    for(index_t j = 0; j < c.shardh; ++j)
                    {
                        const size_t k =
                            size_t(row) * c.shardh * R + q * size_t(c.shardh) + size_t(j);
                        const uint16_t want = canon(inh[l * R + q][size_t(row) * c.shardh + j]);
                        if(outh[i][k] != want)
                            report("gather_bf16", l, r, k, outh[i][k], want);
                    }
                }
            }
        }
    }
    // After the last use every sentinel slot is empty again.
    for(const HostExchange* e : {&exA, &exC})
        for(const auto& b : e->bufs)
            for(uint64_t w : b)
                if(w != (e == &exA ? Ar::kFillWord : AgH::kFillWord) * 0x100000001ull)
                {
                    if(bad++ < 5)
                        std::printf("  exchange word not reset: 0x%llx\n",
                                    static_cast<unsigned long long>(w));
                    break;
                }
    std::printf("  %d world, %d launches, epochs 0x%08x.., %d mismatches\n",
                c.world,
                c.launches,
                c.state0 + 1,
                bad);
    return bad == 0 ? 0 : 1;
}

/// A rank that never arrives: the bounded wait must give up.
int run_missing_peer()
{
    HostExchange ex(
        2, Ar::SlotBytes(1, 512), Ar::ExchangeBytes(2, Ar::SlotBytes(1, 512)), Ar::kFillWord);
    std::vector<uint16_t> in(512, 0x3F80);
    uint32_t st = 0;
    Ar::Args a{};
    a.x         = ex.view(0, &st, 1, 0, 1000);
    a.in        = reinterpret_cast<const bf16_t*>(in.data());
    a.in_stride = 512;
    a.rows      = 1;
    a.width     = 512;
    std::mt19937_64 rng(5);
    const HostMem m{&ex.bufs, 0, &rng};
    Ar::PushPacket(a, m, a.x.seq(1), 0, 0);
    float acc[Ar::kElems];
    const bool ok = Ar::ReducePacket(a, m, a.x.seq(1), 0, 0, acc);
    std::printf("  missing peer: wait %s\n", ok ? "returned (WRONG)" : "timed out");
    return ok ? 1 : 0;
}

/// Codec unit checks: -0 never survives encode; LL flags of sequence 0.
int run_codecs()
{
    using C = ck_tile::xgpu_sentinel_codec<bf16_t>;
    int bad = 0;
    const xgpu_vec v{{0x80000000u, 0x00008000u, 0x80008000u, 0x3F80BF80u}};
    const xgpu_vec e = C::encode(v, 0);
    bad += C::pending(e, 0) ? 1 : 0;
    bad += e.w[3] != 0x3F80BF80u ? 1 : 0;
    bad += (e.w[0] | e.w[1] | e.w[2]) != 0 ? 1 : 0;
    bad += !C::pending(C::empty(), 0) ? 1 : 0;
    xgpu_vec part = C::empty();
    part.w[0]     = 0x12341234u;
    bad += !C::pending(part, 0) ? 1 : 0;
    using F = ck_tile::xgpu_sentinel_codec<float>;
    bad += F::pending(F::encode(xgpu_vec{{0x80000000u, 1u, 2u, 3u}}, 0), 0) ? 1 : 0;
    using LL = ck_tile::xgpu_ll_codec;
    bad += LL::pending(LL::encode(xgpu_vec{{7u, 9u, 0u, 0u}}, 0), 0) ? 1 : 0;
    bad += !LL::pending(LL::empty(), 0) ? 1 : 0;
    bad += !LL::pending(LL::encode(xgpu_vec{{7u, 9u, 0u, 0u}}, 2), 4) ? 1 : 0;
    const xgpu_vec d = LL::decode(LL::encode(xgpu_vec{{7u, 9u, 0u, 0u}}, 5));
    bad += (d.w[0] != 7u || d.w[1] != 9u) ? 1 : 0;
    std::printf("  codecs: %d failures\n", bad);
    return bad;
}

} // namespace

int main()
{
    int failures = run_codecs();
    Config c;
    failures += run_protocol(c);
    c.world = 1; // self loop
    failures += run_protocol(c);
    c.world  = 3;
    c.state0 = 0;
    failures += run_protocol(c);
    failures += run_missing_peer();
    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
