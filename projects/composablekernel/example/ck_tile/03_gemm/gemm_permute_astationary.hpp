// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A-stationary fused GEMM + permute for gfx950: the Y[h,b,t,d] of gemm_permute.hpp, computed by a
// hand-written kernel built from CK-Tile parts.
// - One workgroup (4 waves) per (b, 128-row t tile). Its A tile [128, K] is loaded into LDS once;
//   that is the only workgroup barrier.
// - Each wave walks [128, 64] output tiles of one head: A is read from LDS, the weight is streamed
//   from global memory straight into MFMA B registers (no LDS), loaded two K steps ahead.
// - The weight is the original W, repacked on every call into the ck_tile::shuffle_b_permuteN
//   layout of the head-major GEMM (repack_weight).
// - 16x16x32 MFMA, C not transposed: N-repeat r and MFMA column j hold tile column 4*j + r, so a
//   lane owns 4 consecutive d of 4 rows and stores them straight from the accumulators into the
//   interleaved Y (128 contiguous bytes per row and wave); no LDS shuffle. Accumulators are double
//   buffered: tile i - 1 is converted to BF16 (type_convert, round to nearest even) and stored
//   while the MFMAs of tile i's first K steps run.
// The gfx1250 kernel is in gemm_permute_astationary_gfx1250.hpp; repack_weight and launch below
// serve both.
#pragma once

#include "gemm_permute.hpp"
#if defined(CK_USE_GFX1250)
#include "gemm_permute_astationary_gfx1250.hpp"
#endif

namespace gemm_permute::astationary {

#if !defined(CK_USE_GFX1250)
// Wave tile j (j = wave + Waves * i) is head j / Tw, columns (j % Tw) * WN.
inline constexpr int BM = 128, BK = 32, WN = 64, Waves = 4, Threads = Waves * 64;
inline constexpr int Tw = D / WN, KBlocks = K / BK, WaveTiles = N / WN / Waves;
inline constexpr int MRep = BM / 16, NRep = WN / 16, WT = 16;
// B prefetch ring (loads run BRing - 1 K steps ahead); the epilogue of the previous tile is spread
// over the first Spread K steps, PerStep row blocks each.
inline constexpr int BRing = 3, Spread = 4, PerStep = MRep / Spread;
static_assert(K % BK == 0 && BK == 32 && D % WN == 0 && Tw == Waves && NRep == 4);
static_assert(KBlocks % BRing == 0 && MRep % Spread == 0 && Spread <= KBlocks &&
              WaveTiles % 2 == 0);

// Weight layout (ck_tile::shuffle_b_permuteN of a 16x16x32, N tile D, 4 N-warp GEMM): flat row
// R = tile * 16 + r * 4 + w holds GEMM columns tile * D + w * 64 + 4 * j + r (j = MFMA column), and
// its 32-wide K step s is the 1 KiB chunk Wf[R * K * 16 + s * 512 + lane * 8 .. + 8], one 16 B
// B-operand load per lane. With head-major GEMM columns (h * D + d) row h * 16 + r * 4 + w is
// N-repeat r of wave tile (h, w).
__host__ __device__ constexpr int flat_column(int R, int j)
{
    return R / (NRep * Tw) * D + R % Tw * WN + j * NRep + R / Tw % NRep;
}

// A prologue and A LDS layout: the CompV3 problem of the 128x256x32 2x2-warp tile.
using LoadShape = ck_tile::TileGemmShape<ck_tile::sequence<BM, 256, BK>,
                                         ck_tile::sequence<2, 2, 1>,
                                         ck_tile::sequence<32, 32, 16>>;
using LoadTraits =
    ck_tile::TileGemmUniversalTraits<false, false, false, false, Row, Row, Row, true>;
using LoadProblem = ck_tile::UniversalGemmPipelineProblem<Bf16, Bf16, float, LoadShape, LoadTraits>;
using LoadPolicy  = ck_tile::GemmPipelineAgBgCrDefaultPolicy;

// Per-wave block GEMM (A and B in registers), C not transposed.
using WarpGemm = ck_tile::WarpGemmDispatcher<Bf16, Bf16, float, 16, 16, 32, false>;
struct WavePolicy
{
    template <typename>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(WarpGemm{}, 1, 1);
    }
    static constexpr ck_tile::index_t KSubTileNum = 1;
    static constexpr bool PackMNIter              = false;
};
struct WaveProblem
{
    using ADataType      = Bf16;
    using BDataType      = Bf16;
    using CDataType      = float;
    using BlockGemmShape = ck_tile::TileGemmShape<ck_tile::sequence<BM, WN, BK>,
                                                  ck_tile::sequence<1, 1, 1>,
                                                  ck_tile::sequence<16, 16, 32>>;
    static constexpr ck_tile::index_t kBlockSize    = 64;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
};
using WaveGemm = ck_tile::BlockGemmARegBRegCRegV1<WaveProblem, WavePolicy>;

__global__ void __launch_bounds__(Threads) kernel(const Bf16* __restrict__ a,
                                                  const Bf16* __restrict__ wf,
                                                  Bf16* __restrict__ y,
                                                  int time,
                                                  int batch)
{
    using namespace ck_tile;
    using Part                    = array<index_t, 2>;
    constexpr index_t ABlockBytes = LoadPolicy::GetSmemSizeA<LoadProblem>();
    alignas(16) __shared__ char lds[KBlocks * ABlockBytes];

    const int time_tiles = (time + BM - 1) / BM;
    const int tile_m     = static_cast<int>(blockIdx.x);
    const int b = tile_m / time_tiles, t0 = tile_m % time_tiles * BM;
    const int rows     = ck_tile::min(BM, time - t0);
    const int64_t m0   = int64_t{t0} * batch + b;
    const index_t warp = get_warp_id();
    const index_t lane = get_lane_id();

    auto* const a_lds_base        = reinterpret_cast<Bf16*>(lds);
    constexpr index_t ABlockElems = ABlockBytes / sizeof(Bf16);
    auto a_lds                    = make_tensor_view<address_space_enum::lds>(
        a_lds_base, LoadPolicy::MakeALdsBlockDescriptor<LoadProblem>());

    // A prologue: rows past `rows` are zero-filled by the buffer loads.
    {
        const Part part{warp, lane};
        auto av = make_naive_tensor_view<address_space_enum::global>(
            a + m0 * K,
            make_tuple(rows, K),
            make_tuple(batch * K, 1),
            number<LoadPolicy::GetVectorSizeA<LoadProblem>()>{},
            number<1>{});
        auto a_dram  = make_tile_window(av,
                                       make_tuple(number<BM>{}, number<BK>{}),
                                        {0, 0},
                                       LoadPolicy::MakeADramTileDistribution<LoadProblem>(),
                                       part);
        using ATile  = decltype(load_tile(a_dram));
        auto a_store = make_tile_window(a_lds,
                                        make_tuple(number<BM>{}, number<BK>{}),
                                        {0, 0},
                                        ATile::get_tile_distribution(),
                                        part);
        ATile at[KBlocks];
        static_for<0, KBlocks, 1>{}([&](auto i) {
            at[i] = load_tile(a_dram);
            move_tile_window(a_dram, {0, BK});
        });
        static_for<0, KBlocks, 1>{}([&](auto i) {
            a_store.set_bottom_tensor_view_data_ptr(a_lds_base + i * ABlockElems);
            store_tile(a_store, at[i]);
        });
    }
    block_sync_lds();

    auto a_read =
        make_tile_window(a_lds,
                         make_tuple(number<BM>{}, number<BK>{}),
                         {0, 0},
                         make_static_tile_distribution(WaveGemm::MakeABlockDistributionEncode()),
                         Part{0, lane});
    using ATile = decltype(load_tile(a_read));
    auto load_a = [&](index_t kb) {
        a_read.set_bottom_tensor_view_data_ptr(a_lds_base + kb * ABlockElems);
        return load_tile(a_read);
    };

    // B: wave tile j = (h, w), K step s, N-repeat r reads the 1 KiB chunk of flat row
    // h * 16 + r * 4 + w. Tiles past the last one are clamped (loaded, never used).
    using BTile     = decltype(make_static_distributed_tensor<Bf16>(
        make_static_tile_distribution(WaveGemm::MakeBBlockDistributionEncode())));
    using BWarpDstr = typename WarpGemm::BWarpDstr;
    constexpr auto b_warp_y_lengths =
        to_sequence(BWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto b_warp_y_zeros = uniform_sequence_gen_t<BWarpDstr::NDimY, 0>{};
    const Bf16* const wf_lane     = wf + lane * 8;
    auto load_b                   = [&](index_t i, auto s) {
        constexpr index_t ti = s / KBlocks, ks = s % KBlocks;
        const index_t j    = ck_tile::min(warp + Waves * (i + ti), N / WN - 1);
        const index_t row0 = j / Tw * (NRep * Tw) + j % Tw;
        BTile t;
        static_for<0, NRep, 1>{}([&](auto r) {
            typename WarpGemm::BWarpTensor w;
            const auto* p = wf_lane + (int64_t{row0 + r * Tw} * K + ks * BK) * WT;
            __builtin_memcpy(&w.get_thread_buffer(), __builtin_assume_aligned(p, 16), 16);
            t.set_y_sliced_thread_data(merge_sequences(sequence<r, 0>{}, b_warp_y_zeros),
                                       merge_sequences(sequence<1, 1>{}, b_warp_y_lengths),
                                       w.get_thread_buffer());
        });
        return t;
    };

    // Y store of wave tile i: buffer resource over rows [0, rows) of the [rows, D]-strided window;
    // rows past `rows` lie beyond num_records, so their stores are dropped. Lane (g = lane / 16,
    // q = lane % 16) stores row 16*mi + 4*g + e, columns 4*q .. 4*q + 3 (N-repeats 0..3).
    auto y_res = [&](index_t i) {
        const index_t j = warp + Waves * i, h = j / Tw, d0 = j % Tw * WN;
        return make_wave_buffer_resource(y + ((int64_t{h} * batch + b) * time + t0) * D + d0,
                                         rows * D * 2);
    };
    const index_t lane_off = ((lane / 16) * 4 * D + (lane % 16) * 4) * sizeof(Bf16);
    using CWarpDstr        = typename WarpGemm::CWarpDstr;
    constexpr auto c_warp_y_lengths =
        to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
    constexpr auto c_warp_y_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};
    auto store_rows               = [&](const int32x4_t& res, const auto& c, auto mi0, auto mi1) {
        static_for<decltype(mi0)::value, decltype(mi1)::value, 1>{}([&](auto mi) {
            thread_buffer<Bf16, 4> v[4];
            static_for<0, NRep, 1>{}([&](auto r) {
                const auto blk =
                    c.get_y_sliced_thread_data(merge_sequences(sequence<mi, r>{}, c_warp_y_zeros),
                                               merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                static_for<0, 4, 1>{}([&](auto e) { v[e][r] = type_convert<Bf16>(blk[e]); });
            });
            static_for<0, 4, 1>{}([&](auto e) {
                amd_buffer_store_impl<Bf16, 4, amd_buffer_coherence_enum::DEVICE_NT1>(
                    v[e], res, lane_off + (index_t(mi) * 16 + index_t(e)) * D * 2, 0);
            });
        });
    };

    using CTile = decltype(make_static_distributed_tensor<float>(
        make_static_tile_distribution(WaveGemm::MakeCBlockDistributionEncode())));
    BTile bt[BRing];
    static_for<0, BRing - 1, 1>{}([&](auto s) { bt[s] = load_b(0, s); });
    ATile a_cur = load_a(0);
    // One wave tile: K steps of loads + MFMAs; piece(k) runs inside step k's MFMA region.
    auto run_tile = [&](index_t i, CTile& c, auto&& piece) {
        tile_elementwise_inout([](auto& v) { v = 0; }, c);
        static_for<0, KBlocks, 1>{}([&](auto k) {
            // bt holds steps k .. k + BRing - 2 (ring slot = step % BRing).
            constexpr index_t ahead = k + BRing - 1;
            bt[ahead % BRing]       = load_b(i, number<ahead>{});
            const ATile a_next      = load_a((k + 1) % KBlocks);
            // Keep the next step's loads ahead of this step's MFMAs.
            __builtin_amdgcn_sched_barrier(0);
            WaveGemm{}(c, a_cur, bt[k % BRing]);
            piece(k);
            __builtin_amdgcn_sched_barrier(0);
            a_cur = a_next;
        });
    };
    // Tile i accumulates into acc[i % 2] while tile i - 1 is converted and stored from the other
    // buffer, 8 VALU ops per store interleaved with the MFMAs of tile i's first Spread K steps.
    // Tile -1 stores through a 0-byte resource, so all of them are dropped.
    CTile acc[2];
    tile_elementwise_inout([](auto& v) { v = 0; }, acc[1]);
    int32x4_t res_prev = make_wave_buffer_resource(y, 0);
    auto pieces        = [&](const CTile& prev) {
        return [&](auto k) {
            if constexpr(k < Spread)
            {
                store_rows(res_prev, prev, number<k * PerStep>{}, number<(k + 1) * PerStep>{});
                static_for<0, PerStep * 4, 1>{}([&](auto) {
                    __builtin_amdgcn_sched_group_barrier(0x008, 32 / (PerStep * 4), 0); // MFMA
                    __builtin_amdgcn_sched_group_barrier(0x002, 8, 0); // VALU
                    __builtin_amdgcn_sched_group_barrier(0x040, 1, 0); // VMEM write
                });
            }
        };
    };
    for(index_t i = 0; i < WaveTiles; i += 2)
    {
        run_tile(i, acc[0], pieces(acc[1]));
        res_prev = y_res(i);
        run_tile(i + 1, acc[1], pieces(acc[0]));
        res_prev = y_res(i + 1);
    }
    store_rows(res_prev, acc[1], number<0>{}, number<MRep>{});
}
#endif

// W[k, n] (row-major [K, N], n = d*H + h) -> flat layout, one thread per 16 B chunk (8 consecutive
// k). head_major: GEMM column h*D + d (this kernel); otherwise n (the weight-preshuffle GEMM).
__global__ void __launch_bounds__(256)
    repack_kernel(const Bf16* __restrict__ w, Bf16* __restrict__ wf, bool head_major)
{
    const int c = blockIdx.x * 256 + threadIdx.x;
    const int j = c % WT, k8 = c / WT % (K / 8), R = c / (WT * (K / 8));
    const int col = flat_column(R, j);
    const int n   = head_major ? col % D * H + col / D : col;
    alignas(16) Bf16 v[8];
#pragma unroll
    for(int e = 0; e < 8; ++e)
        v[e] = w[int64_t{k8 * 8 + e} * N + n];
    __builtin_memcpy(__builtin_assume_aligned(wf + int64_t{c} * 8, 16), v, sizeof(v));
}

inline hipError_t repack_weight(const Bf16* w, Bf16* wf, hipStream_t stream, bool head_major = true)
{
    static_assert(K * N / 8 % 256 == 0);
    repack_kernel<<<K * N / 8 / 256, 256, 0, stream>>>(w, wf, head_major);
    return hipGetLastError();
}

// a: A[t,b,k], wf: repack_weight(W), y: Y[h,b,t,d]; dense, 16-byte aligned, non-overlapping.
inline hipError_t
launch(const Bf16* a, const Bf16* wf, Bf16* y, int time, int batch, hipStream_t stream)
{
    const auto addresses = reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(wf) |
                           reinterpret_cast<uintptr_t>(y);
    // The A view of a workgroup spans BM rows of stride B*K with 32-bit byte offsets.
    if((addresses & 15) || time < 1 || batch < 1 ||
       int64_t{BM} * batch * K * sizeof(Bf16) > INT32_MAX)
    {
        std::fprintf(stderr, "gemm_permute_astationary: T=%d B=%d is not supported\n", time, batch);
        return hipErrorInvalidValue;
    }
    kernel<<<(time + BM - 1) / BM * batch, Threads, 0, stream>>>(a, wf, y, time, batch);
    return hipGetLastError();
}

} // namespace gemm_permute::astationary
