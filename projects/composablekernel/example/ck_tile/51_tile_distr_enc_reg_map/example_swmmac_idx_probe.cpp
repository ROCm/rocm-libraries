// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Empirical probe for swmmac_f16_16x16x64_f16 sparsity-index encoding.
//
// Background:
//   * The LLVM intrinsic for swmmac_f16_16x16x64_f16 takes an i16 index (one i16 per lane).
//   * Codegen picks index_key:0 (lower 16 bits of the i32 src) or index_key:1 (upper 16 bits)
//     based on whether the IR is `(idx >> 16)`.
//   * So per lane the HW sees only 16 bits of information per swmmac op; with 32 lanes that
//     gives 32*16 = 512 bits of sparse-index data total.
//
// What this probe does:
//   * For a single fixed (m, k, n) placement, we sweep 16 different nibble values across
//     all 8 nibble positions of a uniform sparse_idx, separately for index_key=0 and
//     index_key=1, AND we vary WHICH lane "owns" the swept value while all other lanes are
//     fixed at the default 0x88888888.
//   * The (lane, key, nibble_pos) combination that actually changes the output C(m,n)
//     reveals where (m, k, n) maps in the HW per-lane index space.
//
// Output: a table per (m, k, n) listing every (lane, key, nibble_pos) that produced any
//   variation in C as we swept its nibble value.

#include <cstdio>
#include <cstdint>
#include <hip/hip_runtime.h>

#include "ck_tile/core/arch/arch.hpp"
#include "ck_tile/core/arch/mma/amdgcn_mma.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_register_mapper.hpp"
#include "ck_tile/core/arch/mma/utility/tile_distribution_encoding_calculator.hpp"
#include "ck_tile/host/hip_check_error.hpp"

using namespace ck_tile;
using namespace ck_tile::core::arch;
using namespace mma;

using F16        = fp16_t;
using F32        = fp32_t;
using Target1250 = decltype(make_amdgcn_gfx12_target<amdgcn_target_id::GFX1250>());

using MmaOp = amdgcn_mma<F16,
                         F16,
                         F16,
                         16u,
                         16u,
                         64u,
                         DefaultSparseMfmaCtrlFlags,
                         Target1250,
                         MmaOpFamily::SPARSE>;

// Number of (nibble_pos, nibble_value) combos we sweep. We always sweep all 16 values; the host
// chooses the nibble position.
// static constexpr int kNumNibbleValues = 16;

struct ProbeArgs
{
    int m;
    int k;        // unpacked K
    int n;
};

// One sweep "case": configures which lane carries the swept value, which i16 half it lives in
// (key 0 = lower, 1 = upper), and which nibble position within that 16-bit half is varied.
struct SweepCase
{
    int target_lane;  // lane that gets the perturbed sparse_idx
    int key;          // 0 → use lower 16 bits, 1 → use upper 16 bits (idx >> 16)
    int nib_pos;      // 0..3 within the chosen 16-bit half
};

// kernel sweeps `nv` ∈ 0..15 (=blockIdx.x) for the nibble value. Other parameters are fixed.
// d_idx_per_lane is laid out as [16 nibble values × 32 lanes].
__global__ void probe_kernel(ProbeArgs args,
                              SweepCase sweep,
                              const uint32_t* d_idx_per_lane,
                              float* out_c_mn,
                              int* out_a_count,
                              int* out_b_count,
                              uint32_t* out_idx_seen)
{
    using ARegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>;
    using BRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding>;
    using CRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding>;

    using AVecType = typename MmaOp::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    constexpr int a_vec_size = vector_traits<AVecType>::vector_size;
    constexpr int b_vec_size = vector_traits<BVecType>::vector_size;
    constexpr int c_vec_size = vector_traits<CVecType>::vector_size;

    const int lane         = threadIdx.x;
    const uint32_t nib_val = blockIdx.x; // 0..15 (unused inside kernel — index already baked
                                         //         into d_idx_per_lane on host)

    // Per-lane sparse_idx loaded from global memory (forces VGPR allocation).
    const uint32_t sparse_idx = d_idx_per_lane[nib_val * 32 + lane];

    AVecType a_frag{};
    int a_count_local = 0;
    for(int v = 0; v < a_vec_size; ++v)
    {
        auto c = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
        if(c[0] == args.m && c[1] == args.k / 2)
        {
            a_frag[v] = type_convert<F16>(1.0f);
            ++a_count_local;
        }
    }

    BVecType b_frag{};
    int b_count_local = 0;
    for(int v = 0; v < b_vec_size; ++v)
    {
        auto c = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
        if(c[0] == args.n && c[1] == args.k)
        {
            b_frag[v] = type_convert<F16>(1.0f);
            ++b_count_local;
        }
    }

    // Call the intrinsic DIRECTLY so we can pick index_key explicitly.
    // index_key:0 ← pass sparse_idx as-is.   index_key:1 ← pass (sparse_idx >> 16).
    CVecType c_frag{};
    if(sweep.key == 0)
    {
        c_frag = {__builtin_amdgcn_swmmac_f16_16x16x64_f16(
            0, a_frag, 0, b_frag, c_frag, static_cast<int32_t>(sparse_idx), 0, 0)};
    }
    else
    {
        c_frag = {__builtin_amdgcn_swmmac_f16_16x16x64_f16(
            0, a_frag, 0, b_frag, c_frag, static_cast<int32_t>(sparse_idx >> 16), 0, 0)};
    }

    for(int v = 0; v < c_vec_size; ++v)
    {
        auto c = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);
        if(c[0] == args.m && c[1] == args.n)
        {
            out_c_mn[nib_val] = type_convert<float>(c_frag[v]);
        }
    }

    const int a_total = __reduce_add_sync(__activemask(), a_count_local);
    const int b_total = __reduce_add_sync(__activemask(), b_count_local);
    if(lane == 0)
    {
        out_a_count[nib_val]  = a_total;
        out_b_count[nib_val]  = b_total;
        out_idx_seen[nib_val] = sparse_idx;
    }
}

// Run one sweep configuration and return whether C(m,n) varied (and the 16 sampled values).
static bool run_sweep(const ProbeArgs& args,
                      const SweepCase& sweep,
                      float (&h_results)[16])
{
    constexpr int kNumNibbles = 16;
    int h_a_count[kNumNibbles];
    int h_b_count[kNumNibbles];
    uint32_t h_idx_seen[kNumNibbles];
    uint32_t h_idx_per_lane[kNumNibbles * 32];

    // Build per-lane sparse_idx for each nibble value. Only `sweep.target_lane` gets perturbed;
    // all other lanes hold the default 0x88888888.
    const uint32_t base_idx     = 0x88888888u;
    const int abs_nib_pos       = (sweep.key == 1 ? 4 : 0) + sweep.nib_pos; // bit pos in i32
    const uint32_t perturb_mask = 0xFu << (abs_nib_pos * 4);

    for(int nv = 0; nv < kNumNibbles; ++nv)
    {
        const uint32_t perturbed =
            (base_idx & ~perturb_mask) | ((static_cast<uint32_t>(nv) & 0xFu) << (abs_nib_pos * 4));
        for(int l = 0; l < 32; ++l)
            h_idx_per_lane[nv * 32 + l] = (l == sweep.target_lane) ? perturbed : base_idx;
        h_results[nv] = -999.f;
    }

    float* d_results        = nullptr;
    int* d_a_count          = nullptr;
    int* d_b_count          = nullptr;
    uint32_t* d_idx_seen    = nullptr;
    uint32_t* d_idx_per_lane = nullptr;
    HIP_CHECK_ERROR(hipMalloc(&d_results, sizeof(h_results)));
    HIP_CHECK_ERROR(hipMalloc(&d_a_count, sizeof(h_a_count)));
    HIP_CHECK_ERROR(hipMalloc(&d_b_count, sizeof(h_b_count)));
    HIP_CHECK_ERROR(hipMalloc(&d_idx_seen, sizeof(h_idx_seen)));
    HIP_CHECK_ERROR(hipMalloc(&d_idx_per_lane, sizeof(h_idx_per_lane)));
    HIP_CHECK_ERROR(
        hipMemcpy(d_idx_per_lane, h_idx_per_lane, sizeof(h_idx_per_lane), hipMemcpyHostToDevice));

    hipLaunchKernelGGL(probe_kernel,
                       dim3(kNumNibbles),
                       dim3(MmaOp::WaveSize),
                       0,
                       0,
                       args,
                       sweep,
                       d_idx_per_lane,
                       d_results,
                       d_a_count,
                       d_b_count,
                       d_idx_seen);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    HIP_CHECK_ERROR(hipMemcpy(h_results, d_results, sizeof(h_results), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipMemcpy(h_a_count, d_a_count, sizeof(h_a_count), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipMemcpy(h_b_count, d_b_count, sizeof(h_b_count), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipMemcpy(h_idx_seen, d_idx_seen, sizeof(h_idx_seen), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipFree(d_results));
    HIP_CHECK_ERROR(hipFree(d_a_count));
    HIP_CHECK_ERROR(hipFree(d_b_count));
    HIP_CHECK_ERROR(hipFree(d_idx_seen));
    HIP_CHECK_ERROR(hipFree(d_idx_per_lane));

    // Did C vary across the 16 nibble values? Compare via raw bit pattern to keep -Werror happy.
    bool varied = false;
    uint32_t bits0;
    __builtin_memcpy(&bits0, &h_results[0], sizeof(bits0));
    for(int nv = 1; nv < kNumNibbles; ++nv)
    {
        uint32_t bits_nv;
        __builtin_memcpy(&bits_nv, &h_results[nv], sizeof(bits_nv));
        if(bits_nv != bits0)
        {
            varied = true;
            break;
        }
    }
    return varied;
}

static void run_probe(int m, int k_unpacked, int n)
{
    using ARegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>;

    // Find (lane, v) of A's `1` for diagnostic reporting.
    int probe_lane = -1;
    int probe_v    = -1;
    for(int lane = 0; lane < static_cast<int>(ARegMap::num_lanes) && probe_v < 0; ++lane)
    {
        for(int v = 0; v < static_cast<int>(ARegMap::num_vector_items); ++v)
        {
            auto c = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
            if(c[0] == m && c[1] == k_unpacked / 2)
            {
                probe_lane = lane;
                probe_v    = v;
                break;
            }
        }
    }

    printf("\n=========================================================================\n");
    printf("=== probe m=%d k=%d n=%d (A `1` at lane=%d v=%d packed_col=%d) ===\n",
           m,
           k_unpacked,
           n,
           probe_lane,
           probe_v,
           k_unpacked / 2);
    printf("=========================================================================\n");

    ProbeArgs args{m, k_unpacked, n};

    // For every (lane, key, nibble_pos), sweep the nibble and report any combo where C varies.
    bool any_hit = false;
    for(int target_lane = 0; target_lane < 32; ++target_lane)
    {
        for(int key = 0; key <= 1; ++key)
        {
            for(int nib_pos = 0; nib_pos < 4; ++nib_pos)
            {
                SweepCase sweep{target_lane, key, nib_pos};
                float h_results[16];
                if(run_sweep(args, sweep, h_results))
                {
                    any_hit = true;
                    printf("  VARIES: lane=%2d key=%d nib_pos=%d  C[",
                           target_lane,
                           key,
                           nib_pos);
                    for(int nv = 0; nv < 16; ++nv)
                        printf("%s%.0f", nv == 0 ? "" : ",", static_cast<double>(h_results[nv]));
                    printf("]\n");
                }
            }
        }
    }
    if(!any_hit)
    {
        printf("  *** No (lane, key, nib_pos) combination changed C(%d,%d). ***\n", m, n);
    }
}

// =====================================================================
// PHASE 2: end-to-end self-test of the test-side sparse_idx construction
//          + the chained MmaOp::exec defined in sparse_gfx12.hpp.
// =====================================================================

__global__ void selftest_kernel(int m, int k, int n, float* out_c_mn)
{
    using ARegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::AWarpDstrEncoding>;
    using BRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::BWarpDstrEncoding>;
    using CRegMap = TileDistrEncRegMap<typename TileDistrEncCalc<MmaOp>::CWarpDstrEncoding>;

    using AVecType = typename MmaOp::AVecType;
    using BVecType = typename MmaOp::BVecType;
    using CVecType = typename MmaOp::CVecType;

    constexpr int a_vec_size = vector_traits<AVecType>::vector_size;
    constexpr int b_vec_size = vector_traits<BVecType>::vector_size;
    constexpr int c_vec_size = vector_traits<CVecType>::vector_size;

    const int lane = threadIdx.x;

    AVecType a_frag{};
    for(int v = 0; v < a_vec_size; ++v)
    {
        auto c = ARegMap::calc_matrix_indices_from_lane_vector(lane, v);
        if(c[0] == m && c[1] == k / 2)
            a_frag[v] = type_convert<F16>(1.0f);
    }

    BVecType b_frag{};
    for(int v = 0; v < b_vec_size; ++v)
    {
        auto c = BRegMap::calc_matrix_indices_from_lane_vector(lane, v);
        if(c[0] == n && c[1] == k)
            b_frag[v] = type_convert<F16>(1.0f);
    }

    const int pcol     = k / static_cast<int>(MmaOp::kCompressionRatio);
    const int key      = (pcol >> 4) & 1;
    const int idx_lane = m + 16 * ((pcol >> 3) & 1);
    const int nib_pos  = (pcol >> 1) & 3;
    const int sub      = pcol & 1;
    const uint32_t k_off = static_cast<uint32_t>(k) & 3u;

    uint32_t sparse_idx = 0x88888888u;
    if(lane == idx_lane)
    {
        const uint32_t other = (k_off + 1u) & 3u;
        const uint32_t nibble =
            (sub == 0) ? ((other << 2) | k_off) : ((k_off << 2) | other);
        const int shift = (key * 16) + nib_pos * 4;
        sparse_idx      = (sparse_idx & ~(0xFu << shift)) | (nibble << shift);
    }

    CVecType c_frag{};
    c_frag = MmaOp::exec(a_frag, b_frag, c_frag, static_cast<int32_t>(sparse_idx));

    for(int v = 0; v < c_vec_size; ++v)
    {
        auto c = CRegMap::calc_matrix_indices_from_lane_vector(lane, v);
        if(c[0] == m && c[1] == n)
            *out_c_mn = type_convert<float>(c_frag[v]);
    }
}

static void selftest_e2e()
{
    printf("\n=========================================================================\n");
    printf("PHASE 2: end-to-end self-test (test-side per-lane sparse_idx + chained exec)\n");
    printf("=========================================================================\n");

    struct Case { int m, k, n; };
    Case cases[] = {
        {0, 0,  0},  {0, 1,  0},  {0, 2,  0},  {0, 3,  0},
        {0, 16, 0},  {0, 17, 0},  {0, 18, 0},  {0, 19, 0},
        {0, 20, 0},  {0, 21, 0},  {0, 22, 0},  {0, 23, 0},
        {0, 32, 0},  {0, 33, 0},  {0, 34, 0},  {0, 35, 0},
        {0, 48, 0},  {0, 49, 0},  {0, 50, 0},  {0, 51, 0},
        {3, 17, 7},  {5, 35, 5},  {15, 60, 15},
    };

    int num_pass = 0;
    int num_fail = 0;
    for(const auto& cs : cases)
    {
        float h_c  = -999.f;
        float* d_c = nullptr;
        HIP_CHECK_ERROR(hipMalloc(&d_c, sizeof(float)));
        HIP_CHECK_ERROR(hipMemcpy(d_c, &h_c, sizeof(float), hipMemcpyHostToDevice));
        hipLaunchKernelGGL(
            selftest_kernel, dim3(1), dim3(MmaOp::WaveSize), 0, 0, cs.m, cs.k, cs.n, d_c);
        HIP_CHECK_ERROR(hipDeviceSynchronize());
        HIP_CHECK_ERROR(hipMemcpy(&h_c, d_c, sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK_ERROR(hipFree(d_c));

        const bool pass    = (h_c > 0.99f && h_c < 1.01f);
        const char* status = pass ? "PASS" : "FAIL";
        if(pass)
            ++num_pass;
        else
            ++num_fail;
        printf("  [%s] m=%2d k=%2d n=%2d  C(m,n) = %7.3f\n",
               status,
               cs.m,
               cs.k,
               cs.n,
               static_cast<double>(h_c));
    }
    printf("Self-test: %d passed, %d failed\n", num_pass, num_fail);
}

int main()
{
    // -----------------------------------------------------------------------
    // PHASE 1: per-lane single-call sweep (already validated).
    // -----------------------------------------------------------------------
    run_probe(/*m=*/0, /*k=*/16, /*n=*/0);  // pcol=8  -> lane 16 key=0 nib_pos=0
    run_probe(/*m=*/0, /*k=*/17, /*n=*/0);  // pcol=8  -> same nibble, idx0 path
    run_probe(/*m=*/0, /*k=*/18, /*n=*/0);  // pcol=9  -> ?
    run_probe(/*m=*/0, /*k=*/19, /*n=*/0);  // pcol=9  -> ?
    run_probe(/*m=*/0, /*k=*/33, /*n=*/0);  // pcol=16 -> lane  0 key=1 nib_pos=0

    // -----------------------------------------------------------------------
    // Confirm K-group → nib_pos mapping inside lane 16, key=0.
    // Each K group of 4 (pcols {2p, 2p+1}) should land in one nibble.
    // Expected: pcol=10/11 (k=20..23) -> nib_pos=1, pcol=12/13 -> nib_pos=2,
    //           pcol=14/15 -> nib_pos=3.
    // -----------------------------------------------------------------------
    run_probe(/*m=*/0, /*k=*/20, /*n=*/0);
    run_probe(/*m=*/0, /*k=*/24, /*n=*/0);
    run_probe(/*m=*/0, /*k=*/28, /*n=*/0);

    // -----------------------------------------------------------------------
    // Confirm lane mapping. Row m selects lane (m mod 16), packed_col>=8 sets
    // the lane/16 bit. Try changing the row.
    // Expected pcol=8, m=5 -> lane 21 key=0 nib_pos=0.
    // -----------------------------------------------------------------------
    run_probe(/*m=*/5,  /*k=*/16, /*n=*/0);
    run_probe(/*m=*/15, /*k=*/16, /*n=*/0);

    // -----------------------------------------------------------------------
    // Confirm upper-half-K (key=1) lane mapping for higher k.
    // Expected: pcol=16 m=0 -> lane  0 key=1 nib_pos=0
    //           pcol=24 m=0 -> lane 16 key=1 nib_pos=0
    //           pcol=20 m=0 -> lane  0 key=1 nib_pos=1
    //           pcol=28 m=0 -> lane 16 key=1 nib_pos=1
    // -----------------------------------------------------------------------
    run_probe(/*m=*/0, /*k=*/32, /*n=*/0);  // pcol=16 even
    run_probe(/*m=*/0, /*k=*/48, /*n=*/0);  // pcol=24
    run_probe(/*m=*/0, /*k=*/40, /*n=*/0);  // pcol=20
    run_probe(/*m=*/0, /*k=*/56, /*n=*/0);  // pcol=28

    // -----------------------------------------------------------------------
    // Confirm B placement direction doesn't change the answer (vary n).
    // -----------------------------------------------------------------------
    run_probe(/*m=*/0, /*k=*/16, /*n=*/5);
    run_probe(/*m=*/3, /*k=*/17, /*n=*/7);
    // -----------------------------------------------------------------------
    // INVESTIGATE: high-half k mod 4 == 3 cases that the test fix can't satisfy.
    // Sweep low-pcol-half k to also cover sub=0 vs sub=1 with k_off=3.
    // -----------------------------------------------------------------------
    run_probe(/*m=*/0, /*k=*/35, /*n=*/0);  // pcol=17, sub=1, k_off=3
    run_probe(/*m=*/0, /*k=*/3,  /*n=*/0);  // pcol=1,  sub=1, k_off=3 (low half, passes)
    run_probe(/*m=*/0, /*k=*/34, /*n=*/0);  // pcol=17, sub=1, k_off=2 (high half, passes)

    // -----------------------------------------------------------------------
    // PHASE 2: end-to-end self-test of the test fix + chained MmaOp::exec.
    // For each (m, k, n) we build the per-lane sparse_idx exactly the way the
    // test does, call MmaOp::exec (which now chains both index_key halves),
    // and check that C[m, n] == 1.0.
    // -----------------------------------------------------------------------
    selftest_e2e();

    return 0;
}
