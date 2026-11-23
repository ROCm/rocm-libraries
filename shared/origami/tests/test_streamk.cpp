/*******************************************************************************
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <vector>
#include "common.hpp"
#include "origami/gemm_intermediates.hpp"

namespace {

using origami::streamk_hybrid_defaults_t;

// Builds a problem with exactly `tiles` output tiles for the given macrotile
// (a single row of `tiles` tile columns), so callers get precise control over
// the tile count fed into select_hybrid_mode's gates.
inline origami::problem_t make_problem_with_tile_count(size_t mt_m,
                                                       size_t mt_n,
                                                       size_t tiles,
                                                       size_t batch = 1) {
  return make_problem(/*m=*/mt_m,
                      /*n=*/mt_n * tiles,
                      /*k=*/64,
                      origami::transpose_t::T,
                      origami::transpose_t::N,
                      batch);
}

}  // namespace

TEST_CASE("Origami streamk: select_hybrid_mode tile-count gate is inclusive of the threshold",
          "[origami][streamk][hybrid]") {
  // Even with a cotenant and occupancy low enough to otherwise force dynamic
  // unconditionally, a grid at or below MIN_TILES_FOR_DYNAMIC stays static_.
  auto hardware = make_hardware(950);
  auto config   = make_config(1, 1, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto at_gate =
      make_problem_with_tile_count(1, 1, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC);
  REQUIRE(origami::streamk::select_hybrid_mode(at_gate, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::static_);

  auto above_gate =
      make_problem_with_tile_count(1, 1, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);
  REQUIRE(origami::streamk::select_hybrid_mode(above_gate, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode requires a cotenant to go dynamic",
          "[origami][streamk][hybrid]") {
  // Large grid and low occupancy alone aren't enough: with no cotenant
  // holding any CU away from this kernel, static_ is already optimal.
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto problem =
      make_problem_with_tile_count(128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, /*sm_count_target=*/0) ==
          origami::hybrid_mode_t::static_);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware, config, /*sm_count_target=*/hardware.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode low occupancy goes dynamic unconditionally",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto problem =
      make_problem_with_tile_count(128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  for (int occupancy = 1;
       occupancy <= streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC;
       ++occupancy) {
    DYNAMIC_SECTION("occupancy=" << occupancy) {
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy);
      REQUIRE(origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU / 2) ==
              origami::hybrid_mode_t::dynamic);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode falls back to tiles_per_cu "
          "once occupancy alone isn't decisive",
          "[origami][streamk][hybrid]") {
  // Occupancy above MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC, and occupancy
  // reported as unknown (<= 0), both defer to the tiles_per_cu threshold.
  auto hardware      = make_hardware(950);
  auto available_cus = hardware.N_CU / 2;
  auto small = make_problem_with_tile_count(128, 128, static_cast<size_t>(available_cus * 8.0));
  auto big   = make_problem_with_tile_count(128, 128, static_cast<size_t>(available_cus * 9.0));

  for (int occupancy :
       {0, streamk_hybrid_defaults_t::MAX_OCCUPANCY_FOR_UNCONDITIONAL_DYNAMIC + 1}) {
    DYNAMIC_SECTION("occupancy=" << occupancy) {
      auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, occupancy);
      REQUIRE(origami::streamk::select_hybrid_mode(small, hardware, config, available_cus) ==
              origami::hybrid_mode_t::static_);
      REQUIRE(origami::streamk::select_hybrid_mode(big, hardware, config, available_cus) ==
              origami::hybrid_mode_t::dynamic);
    }
  }
}

TEST_CASE("Origami streamk: select_hybrid_mode non-gfx950 always static",
          "[origami][streamk][hybrid]") {
  // Large grid, cotenant present, low occupancy: would select dynamic on
  // gfx950, but the architecture guard forces static_ elsewhere.
  auto config = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto problem =
      make_problem_with_tile_count(128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC + 1);

  auto hardware_gfx950 = make_hardware(950);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware_gfx950, config, hardware_gfx950.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);

  auto hardware_gfx942 = make_hardware(942);
  REQUIRE(origami::streamk::select_hybrid_mode(
              problem, hardware_gfx942, config, hardware_gfx942.N_CU / 2) ==
          origami::hybrid_mode_t::static_);
}

TEST_CASE("Origami streamk: select_hybrid_mode batch multiplies tiles, crossing the gate",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32, 16, 16, 16, false, 1, /*occupancy=*/1);
  auto base     = make_problem_with_tile_count(
      128, 128, streamk_hybrid_defaults_t::MIN_TILES_FOR_DYNAMIC, /*batch=*/1);
  REQUIRE(origami::streamk::select_hybrid_mode(base, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::static_);

  auto base_b4  = base;
  base_b4.batch = 4;
  REQUIRE(origami::streamk::select_hybrid_mode(base_b4, hardware, config, hardware.N_CU / 2) ==
          origami::hybrid_mode_t::dynamic);
}

TEST_CASE("Origami streamk: select_hybrid_mode sm_count_target=0 uses N_CU",
          "[origami][streamk][hybrid]") {
  auto hardware = make_hardware(950);
  auto config   = make_config(128, 128, 32);
  auto problem  = make_problem(4096, 4096, 64);
  auto a        = origami::streamk::select_hybrid_mode(problem, hardware, config, 0);
  auto b        = origami::streamk::select_hybrid_mode(problem, hardware, config, hardware.N_CU);
  REQUIRE(a == b);
}

TEST_CASE("Origami streamk: stream_k=0 uses one WG per output tile", "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(/*m=*/512, /*n=*/512, /*k=*/8192);
  auto config   = make_config(256,
                            256,
                            64,
                            32,
                            32,
                            8,
                            false,
                            1,
                            /*occupancy=*/1,
                            /*non_temporal_a=*/0,
                            /*non_temporal_b=*/0,
                            /*stream_k=*/0);

  const size_t tiles = origami::streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  auto [reduction, num_wgs, num_active_cus, num_timesteps, split_factor] =
      origami::gemm::compute_launch_parameters(problem, hardware, config, config.grid_selection);

  REQUIRE(num_wgs == tiles);
  REQUIRE(split_factor == 1);
  REQUIRE(reduction == origami::reduction_t::none);
  REQUIRE(num_active_cus > 0);
  REQUIRE(num_timesteps >= 1);
}

TEST_CASE("Origami streamk: stream_k=5 K-splits when tiles << CUs", "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);
  auto config   = make_config(256,
                            256,
                            64,
                            32,
                            32,
                            8,
                            false,
                            1,
                            /*occupancy=*/1,
                            /*non_temporal_a=*/0,
                            /*non_temporal_b=*/0,
                            /*stream_k=*/5);

  const size_t tiles = origami::streamk::compute_number_of_output_tiles(
      config.mt.m, config.mt.n, problem.size.m, problem.size.n, problem.batch);

  auto [reduction, num_wgs, num_active_cus, num_timesteps, split_factor] =
      origami::gemm::compute_launch_parameters(problem, hardware, config, config.grid_selection);

  REQUIRE(num_wgs > tiles);
  REQUIRE(split_factor > 1);
  REQUIRE(split_factor == origami::math::safe_ceil_div(num_wgs, tiles));
  REQUIRE(reduction != origami::reduction_t::none);
  REQUIRE(num_active_cus > 0);
  REQUIRE(num_timesteps >= 1);
}

TEST_CASE("Origami streamk: stream_k=0 vs 5 launch divergence on underloaded grid",
          "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);

  auto config_dp            = make_config(256,
                               256,
                               64,
                               32,
                               32,
                               8,
                               false,
                               1,
                               /*occupancy=*/1,
                               /*non_temporal_a=*/0,
                               /*non_temporal_b=*/0,
                               /*stream_k=*/0);
  auto config_sk5           = config_dp;
  config_sk5.stream_k       = 5;
  config_sk5.grid_selection = origami::grid_selection_t::k_split_aware;

  const auto launch = [&](const origami::config_t& c) {
    return origami::gemm::compute_launch_parameters(problem, hardware, c, c.grid_selection);
  };

  const auto result0 = launch(config_dp);
  const auto result5 = launch(config_sk5);

  REQUIRE(std::get<1>(result0) < std::get<1>(result5));
  REQUIRE(std::get<4>(result0) == 1);
  REQUIRE(std::get<4>(result5) > 1);
  REQUIRE(std::get<0>(result0) == origami::reduction_t::none);
  REQUIRE(std::get<0>(result5) != origami::reduction_t::none);
}

TEST_CASE("Origami streamk: context_t reflects stream_k flag", "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);
  auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

  config.stream_k = 0;
  origami::gemm::context_t ctx0(problem, hardware, config);
  REQUIRE(ctx0.splitting_factor == 1);
  REQUIRE(ctx0.num_wgs == ctx0.num_output_tiles);

  config.stream_k       = 5;
  config.grid_selection = origami::grid_selection_t::k_split_aware;
  origami::gemm::context_t ctx5(problem, hardware, config);
  REQUIRE(ctx5.splitting_factor > 1);
  REQUIRE(ctx5.num_wgs > ctx0.num_wgs);
}

TEST_CASE("Origami streamk: compute_total_latency differs for stream_k=0 vs 5",
          "[origami][streamk][flag]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(512, 512, 8192);
  auto config   = make_config(256, 256, 64, 32, 32, 8, false, 1);

  config.stream_k       = 0;
  const double latency0 = origami::gemm::compute_total_latency(problem, hardware, config);

  config.stream_k       = 5;
  config.grid_selection = origami::grid_selection_t::k_split_aware;
  const double latency5 = origami::gemm::compute_total_latency(problem, hardware, config);

  REQUIRE(latency0 != latency5);
  REQUIRE(latency0 < std::numeric_limits<double>::max());
  REQUIRE(latency5 < std::numeric_limits<double>::max());
}

TEST_CASE("Origami streamk: grid_k_split_aware partial-tile correction", "[streamk]") {
  // Verifies that grid_k_split_aware returns the correct grid (SK or DP) for
  // shapes where the partial-tile/DP-efficiency correction logic determines the
  // outcome.  These cases were identified during the WorkStealing investigation.
  //
  //   ipt      = iters_per_tile = floor(K / MT_K)
  //   DP_eff   = tiles / (ceil(tiles/cu_count) * cu_count)
  //
  // Helper: build a problem+config from (tiles, ipt, cu_count, batch) and call
  // grid_k_split_aware.  MT_K=32; K = ipt*32 so floor(K/MT_K) == ipt exactly.
  // M=16, N=16*tiles/batch gives exactly `tiles` output tiles for batch==1 and
  // tiles/batch tiles-per-batch for batched shapes.  cu_budget=0 (no cap).

  auto run = [](size_t tiles, size_t ipt, size_t cu_count, size_t batch = 1) {
    constexpr size_t MT_M = 16, MT_N = 16, MT_K = 32;
    const size_t K = std::max<size_t>(ipt, 1) * MT_K;
    auto p = make_problem(MT_M, MT_N * tiles / batch, K,
                          origami::transpose_t::T, origami::transpose_t::N, batch);
    auto c = make_config(MT_M, MT_N, MT_K);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
    p.num_cus = cu_count;
    return origami::streamk::select_grid_size(p, make_hardware(950), c,
                                              origami::grid_selection_t::k_split_aware);
  };

  // ---- Group A: ipt == 1 (tile-streaming, no K-split) --------------------
  // Force DP when sk_grid > 2*cu_count.  All three cases pass cu_count below
  // N_CU, which caps sk_grid at one WG per CU, so all three keep SK.

  // [16, 884736, 32] and tile 16x256x32: tiles=3456, ipt=1.
  // sk_grid=224 is one WG per CU, below the threshold → keep SK.  A CU budget
  // with nothing else running makes DP look faster here, but with a co-tenant
  // on the other CUs SK is the same or faster.
  REQUIRE(run(3456, 1, 224) == 224);

  // [12, 33554432, 32] and tile 16x448x32: tiles=74899, ipt=1.
  // sk_grid=224 is one WG per CU → keep SK; SK tile-streaming (sequential
  // B-panel reuse) wins.  Measured: forcing DP +462µs slower.
  REQUIRE(run(74899, 1, 224) == 224);

  // tiles=400, ipt=1: sk_grid=224 is one WG per CU → keep SK.
  // Few tiles per WG; SK setup amortization beats a shallow DP wave.
  REQUIRE(run(400, 1, 224) == 224);

  // ---- Group B: batched shapes → always keep SK (cross-tile A/B reuse) -----
  // DYNAMIC_GRID=4 sweep confirmed: batched tiny GEMMs (e.g. tf32 N,N,8192,32,25,25)
  // see no improvement from k_split_aware grid — the SK grid is not the issue.
  // The batch>1 gate correctly preserves SK tile-streaming for A/B L1/L2 reuse.

  // [16, 96, 32] batch=9216 and tile 16x128x32: tiles=9216, ipt=1.
  // Batched: SK preserves cross-tile A/B reuse across the batch dimension.
  REQUIRE(run(9216, 1, 224, 9216) == 224);

  // Batched K-splitting shape: batch=4096, ipt=4.
  // batch>1 gate fires before DP_eff check → always keep SK.
  // First argument is not the tile count here: 16*384/4096 truncates to one tile
  // column, so this shape has 4096 tiles.  Safe only because the gate returns first.
  REQUIRE(run(384, 4, 224, 4096) == 224);

  // ---- Group C: ipt > 1, DP_eff < 80% → keep SK -------------------------
  // tiles barely above cu_count: DP last wave too thin, SK utilization wins.

  // [4096, 4096, 4096] and tile 256x256x32: tiles=256, ipt=128.
  // DP_eff = 256 / (ceil(256/224)*224) = 256/448 = 57% < 80% → SK preserved.
  REQUIRE(run(256, 128, 224) == 224);

  // [2048, 8192, 5640] and tile 256x256x32: tiles=256, ipt=176. DP_eff=57%.
  // Same low-efficiency regime; SK avoids the thin last DP wave.
  REQUIRE(run(256, 176, 224) == 224);

  // ---- Group D: ipt > 1, DP_eff >= 80%, partial tiles → force DP ---------
  // Confirmed by DYNAMIC_GRID=4 sweep: grid_k_split_aware returns tiles for
  // all these shapes, recovering to base latency.

  // [160, 107005, 160] and tile 160x256x32: tiles=419, ipt=5, DP_eff=93%.
  // floor(419*5/224)=9, 9%5=4 → partial tile → DP.
  // Measured: SK 44µs, DP 30µs (L2 hit rate 40% vs 73%).
  REQUIRE(run(419, 5, 224) == 419);

  // [160, 73390, 320] and tile 160x192x64: tiles=382, ipt=5, DP_eff=85%.
  // Measured: SK 36.9µs, DP 26.3µs ≈ base 26.6µs.
  REQUIRE(run(382, 5, 224) == 382);

  // [160, 73390, 320] and tile 160x192x64: tiles=382, ipt=5, cu_count=256 (gfx950 full).
  // sk_grid=224 < cu_count=256 → SK leaves 32 CUs idle → force DP.
  // Measured: DYNAMIC_GRID=4 sweep confirmed recovery to base latency.
  REQUIRE(run(382, 5, 256) == 382);

  // [160, 73390, 320] and tile 160x192x64: tiles=382, ipt=5, cu_count=224 (cotenant).
  // DP_eff: 382/448=85% >= 80% → DP.
  REQUIRE(run(382, 5, 224) == 382);

  // [256, 98304, 128] and tile 256x256x32: tiles=384, ipt=4, DP_eff=85.7%.
  // Measured: SK 65µs, DP 47.8µs ≈ base 46.9µs.
  REQUIRE(run(384, 4, 224) == 384);

  // [2048, 9216, 1480] and tile 256x192x64: tiles=384, ipt=23, DP_eff=85.7%.
  // Measured: SK 98.1µs, DP 78.8µs ≈ base 78.3µs.
  REQUIRE(run(384, 23, 224) == 384);

  // [128, 98304, 256] and tile 128x256x32: tiles=384, ipt=8, DP_eff=85.7%.
  // Measured: SK 48.6µs, DP 40.5µs ≈ base 40.9µs.
  REQUIRE(run(384, 8, 224) == 384);

  // [9984, 2048, 32768] and tile 192x256x64: tiles=416, ipt=512, DP_eff=92%.
  // floor(416*512/224)=951, 951%512=439 → partial tile → DP.
  // Measured: keeping SK here regresses 1.5x (SK 1857µs vs DP 1235µs).
  REQUIRE(run(416, 512, 224) == 416);

  // ---- Group E: DP_eff < 80% → keep SK (tiles barely above cu_count) ------

  // tiles=300, ipt=5: DP_eff = 300/(2*224)=67% < 80% → SK preserved.
  REQUIRE(run(300, 5, 224) < 300);

  // ---- Edge cases ---------------------------------------------------------

  // ipt=0: degenerate K (no full K-tiles), no-op → keep whatever grid was found.
  REQUIRE(run(419, 0, 224) == 224);

  // [4352, 128, 8192] and tile 256x128x32: tiles=17 < cu_count=224.
  // tiles < cu_count → K-split branch: grid > tiles (multiple K-splits launched).
  REQUIRE(run(17, 256, 224) > 17);
}

TEST_CASE("Origami streamk: ipt <= 1 switches to DP above two waves of workgroups", "[streamk]") {
  // At ipt <= 1 the choice between the SK grid and one workgroup per output
  // tile depends on sk_grid, not the tile count: DP above 2*cu_count, SK at or
  // below.  Helper: one tile row of `tiles` columns, no CU cap and unbounded
  // workspace, so occupancy is the only input that changes sk_grid.

  auto run = [](size_t tiles, size_t k, int occupancy) {
    constexpr size_t MT_M = 16, MT_N = 16, MT_K = 32;
    auto p                      = make_problem(MT_M, MT_N * tiles, k);
    auto c                      = make_config(MT_M, MT_N, MT_K, 16, 16, 16, false, 1, occupancy);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
    p.num_cus                   = 0;  // use all 256 CUs
    return origami::streamk::select_grid_size(
        p, make_hardware(950), c, origami::grid_selection_t::k_split_aware);
  };

  // K=32 gives ipt == 1 throughout; only the occupancy changes.

  // --- at or below two waves: keep SK -------------------------------------
  // occupancy 1 → SK grid 256 = 1 wave → keep SK.
  REQUIRE(run(12288, 32, 1) == 256);
  // occupancy 2 → SK grid 512 = exactly 2 waves → keep SK (threshold is exclusive).
  REQUIRE(run(12288, 32, 2) == 512);

  // --- above two waves: force DP ------------------------------------------
  // occupancy 3 → SK grid 768 = 3 waves → one workgroup per output tile.
  REQUIRE(run(12288, 32, 3) == 12288);
  // occupancy 6 → SK grid 1536 = 6 waves → DP.
  REQUIRE(run(12288, 32, 6) == 12288);

  // K == 1 is not special: it is ipt <= 1, decided by the same threshold.
  REQUIRE(run(12288, 1, 2) == 512);
  REQUIRE(run(12288, 1, 3) == 12288);

  // The tile count is not an input: at occupancy 3 the SK grid stays above 2
  // waves, so DP is chosen even past 32*cu_count, where the old window stopped.
  REQUIRE(run(6144, 32, 3) == 6144);
  REQUIRE(run(24576, 32, 3) == 24576);
  REQUIRE(run(98304, 32, 3) == 98304);

  // ipt >= 2 takes a different branch: K=64 fills its DP waves → keeps SK.
  REQUIRE(run(12288, 64, 3) < 12288);

  // batch > 1 still returns early, ahead of the ipt branch.
  {
    constexpr size_t MT_M = 16, MT_N = 16, MT_K = 32, TILES = 12288, BATCH = 4;
    auto p = make_problem(
        MT_M, MT_N * TILES / BATCH, MT_K, origami::transpose_t::T, origami::transpose_t::N, BATCH);
    auto c                      = make_config(MT_M, MT_N, MT_K, 16, 16, 16, false, 1, 3);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
    p.num_cus                   = 0;
    REQUIRE(origami::streamk::select_grid_size(
                p, make_hardware(950), c, origami::grid_selection_t::k_split_aware) == 768);
  }
}

// ---------------------------------------------------------------------------------------------
// Launch arithmetic (the helpers hipBLASLt resolves its Stream-K launch with).
// ---------------------------------------------------------------------------------------------

namespace {

// Deterministic small-integer generator for the property sweeps.
struct lcg_t {
  uint64_t s;
  uint64_t operator()(uint64_t n) {
    s = s * 6364136223846793005ull + 1442695040888963407ull;
    return n ? (s >> 33) % n : 0;
  }
};

// The ordered chunk lengths folded into each tile when every workgroup w owns
// workgroup_iter_range(w, ...) of the flat (padded) space.
std::vector<std::vector<size_t>> fold_signatures(size_t tiles,
                                                 size_t iters,
                                                 size_t grid,
                                                 bool per_tile,
                                                 size_t k_gran) {
  const size_t padded = origami::streamk::padded_iters_per_tile(iters, k_gran);
  std::vector<std::vector<size_t>> sig(tiles);
  for (size_t w = 0; w < grid; ++w) {
    const auto r = origami::streamk::workgroup_iter_range(w, tiles, iters, grid, per_tile, true,
                                                          k_gran);
    for (size_t s = r.start; s < r.end;) {
      const size_t t = s / padded, e = std::min(r.end, (t + 1) * padded);
      sig[t].push_back(e - s);
      s = e;
    }
  }
  return sig;
}

}  // namespace

TEST_CASE("Origami streamk: padded_iters_per_tile rounds up to whole K units",
          "[origami][streamk][launch]") {
  REQUIRE(origami::streamk::padded_iters_per_tile(7, 0) == 7);
  REQUIRE(origami::streamk::padded_iters_per_tile(7, 1) == 7);
  REQUIRE(origami::streamk::padded_iters_per_tile(7, 4) == 8);
  REQUIRE(origami::streamk::padded_iters_per_tile(8, 4) == 8);
  REQUIRE(origami::streamk::padded_iters_per_tile(0, 4) == 0);
}

TEST_CASE("Origami streamk: static_split partitions the Stream-K iterations",
          "[origami][streamk][launch]") {
  using origami::streamk::static_split;
  // Grid divides tiles: every tile data parallel after the first grid, no extras.
  auto s = static_split(1024, 16, 256, 1, false);
  REQUIRE(s.sk_tiles == 256);
  REQUIRE(s.sk_iters_per_wg == 16);
  REQUIRE(s.extra_iters == 0);
  // Remainder tiles plus full_tiles whole tiles per workgroup go to the Stream-K region.
  s = static_split(1000, 16, 256, 1, false);
  REQUIRE(s.sk_tiles == 256 + 1000 % 256);
  REQUIRE(s.sk_tiles * size_t{16} == s.sk_iters_per_wg * size_t{256} + s.extra_iters);
  // Capped at the tile count; fewer tiles than workgroups splits every tile.
  REQUIRE(static_split(300, 16, 256, 4, false).sk_tiles == 300);
  REQUIRE(static_split(100, 16, 256, 1, false).sk_tiles == 100);
  // Force-DP-only and an empty grid.
  REQUIRE(static_split(1000, 16, 256, 1, true).sk_tiles == 0);
  REQUIRE(static_split(1000, 16, 0, 1, false).sk_iters_per_wg == 0);

  lcg_t rnd{20261116};
  for (int i = 0; i < 20000; ++i) {
    const size_t tiles = 1 + rnd(5000), iters = 1 + rnd(300), grid = 1 + rnd(2048);
    const size_t g = 1 + rnd(4);
    const auto split = static_split(tiles, iters, grid, 1 + static_cast<int>(rnd(3)), false, g);
    const size_t padded = origami::streamk::padded_iters_per_tile(iters, g);
    REQUIRE(split.sk_tiles <= tiles);
    REQUIRE(split.sk_iters_per_wg % g == 0);
    REQUIRE(size_t{split.sk_tiles} * padded ==
            size_t{split.sk_iters_per_wg} * grid + split.extra_iters);
    REQUIRE(split.extra_iters < (split.sk_iters_per_wg + g) * grid);
  }
}

TEST_CASE("Origami streamk: workgroup_iter_range covers the flat space once",
          "[origami][streamk][launch]") {
  lcg_t rnd{7};
  for (int i = 0; i < 3000; ++i) {
    const size_t tiles = 1 + rnd(64), iters = 1 + rnd(64), g = 1 + rnd(3);
    const size_t grid  = 1 + rnd(4 * tiles);
    const bool per_tile = rnd(2) != 0;
    const size_t padded = origami::streamk::padded_iters_per_tile(iters, g);
    size_t next         = 0;
    for (size_t w = 0; w < grid; ++w) {
      const auto r =
          origami::streamk::workgroup_iter_range(w, tiles, iters, grid, per_tile, true, g);
      REQUIRE(r.start == next);
      REQUIRE(r.start % g == 0);
      REQUIRE(r.end >= r.start);
      next = r.end;
    }
    REQUIRE(next == tiles * padded);
  }
  REQUIRE(origami::streamk::workgroup_iter_range(0, 4, 8, 0, false, false).end == 0);
}

TEST_CASE("Origami streamk: static_split_row_uniform agrees with the fold signatures",
          "[origami][streamk][launch]") {
  // With no data-parallel region (sk_tiles == tiles), the split is row-uniform exactly when
  // every tile folds the same ordered chunk lengths.
  lcg_t rnd{11};
  int uniform = 0, mixed = 0;
  for (int i = 0; i < 4000; ++i) {
    const size_t tiles = 1 + rnd(24), iters = 1 + rnd(48), g = 1 + rnd(2);
    const size_t grid  = tiles + rnd(5 * tiles);
    const bool per_tile = rnd(2) != 0;
    const auto split   = origami::streamk::static_split(tiles, iters, grid, 1, false, g);
    if (split.sk_tiles != tiles || split.sk_iters_per_wg == 0) continue;
    const auto sig = fold_signatures(tiles, iters, grid, per_tile, g);
    bool same      = true;
    for (const auto& s : sig) same = same && s == sig[0];
    const bool got =
        origami::streamk::static_split_row_uniform(split, tiles, iters, grid, per_tile, true, g);
    // The function is sufficient: whatever it accepts folds uniformly.
    if (got) REQUIRE(same);
    uniform += got;
    mixed += !same;
  }
  REQUIRE(uniform > 100);
  REQUIRE(mixed > 100);
  // Force-DP-only is always uniform; a mixed DP + Stream-K split is refused.
  REQUIRE(origami::streamk::static_split_row_uniform({}, 1000, 16));
  const auto mixed_split = origami::streamk::static_split(1024, 16, 256, 1, false);
  REQUIRE_FALSE(origami::streamk::static_split_row_uniform(mixed_split, 1024, 16, 256));
}

TEST_CASE("Origami streamk: parallel_reduction_row_uniform", "[origami][streamk][launch]") {
  using origami::reduction_t;
  using origami::streamk::parallel_reduction_row_uniform;
  REQUIRE(parallel_reduction_row_uniform(reduction_t::parallel, 64, 32, 0, true));
  REQUIRE_FALSE(parallel_reduction_row_uniform(reduction_t::parallel, 32, 32, 0, true));
  REQUIRE_FALSE(parallel_reduction_row_uniform(reduction_t::parallel, 96, 64, 0, true));
  REQUIRE_FALSE(parallel_reduction_row_uniform(reduction_t::parallel, 64, 32, 1, true));
  REQUIRE_FALSE(parallel_reduction_row_uniform(reduction_t::parallel, 64, 32, 0, false));
  REQUIRE_FALSE(parallel_reduction_row_uniform(reduction_t::tree, 64, 32, 0, true));
  REQUIRE_FALSE(parallel_reduction_row_uniform(reduction_t::parallel, 64, 0, 0, true));
}

TEST_CASE("Origami streamk: dynamic_items", "[origami][streamk][launch]") {
  // No overrides: whole tiles only, a nominal split of 2.
  auto d = origami::streamk::dynamic_items(1000, 17);
  REQUIRE(d.sk_tiles == 0);
  REQUIRE(d.sk_iters_per_wi == 9);
  REQUIRE(d.sk_split == 2);
  REQUIRE(d.total_items == 1000);
  // Overrides split tiles; the split is re-derived after rounding to whole K units.
  d = origami::streamk::dynamic_items(1000, 17, 4, 10, 5);
  REQUIRE(d.sk_iters_per_wi == 4);
  REQUIRE(d.sk_split == 5);
  REQUIRE(d.total_items == 990 + 10 * 5);
  d = origami::streamk::dynamic_items(1000, 17, 4, 10, 3);
  REQUIRE(d.sk_iters_per_wi == 8);
  REQUIRE(d.sk_split == 3);
  // K == 0 is one iteration.
  REQUIRE(origami::streamk::dynamic_items(8, 0).sk_iters_per_wi == 1);
}

TEST_CASE("Origami streamk: reduction, straddling and workspace rules",
          "[origami][streamk][launch]") {
  using origami::reduction_t;
  using origami::streamk::reconcile_reduction;
  REQUIRE(reconcile_reduction(reduction_t::parallel, 64, 32) == reduction_t::parallel);
  REQUIRE(reconcile_reduction(reduction_t::parallel, 63, 32) == reduction_t::tree);
  REQUIRE(reconcile_reduction(reduction_t::parallel, 64, 0) == reduction_t::tree);
  REQUIRE(reconcile_reduction(reduction_t::tree, 1, 32) == reduction_t::tree);

  REQUIRE(origami::streamk::straddles_tiles(16, 300, 1000));
  REQUIRE_FALSE(origami::streamk::straddles_tiles(16, 250, 1000));
  REQUIRE_FALSE(origami::streamk::straddles_tiles(1, 300, 1000));
  REQUIRE_FALSE(origami::streamk::straddles_tiles(16, 0, 1000));

  REQUIRE(origami::streamk::partial_tile_bytes(256, 128, 4, 512) == size_t{256} * 128 * 4 * 512);
}

TEST_CASE("Origami streamk: uniform_grid picks the largest admissible split",
          "[origami][streamk][launch]") {
  using origami::reduction_t;
  using origami::streamk::uniform_grid;
  constexpr size_t big = std::numeric_limits<size_t>::max();
  // Below the tiles: all-full.  At the tiles: unchanged.
  REQUIRE(uniform_grid(512, 8192, 2, reduction_t::tree, 1, false, 0, big, 0) == 8192);
  REQUIRE(uniform_grid(8192, 8192, 2, reduction_t::tree, 1, false, 0, big, 0) == 8192);
  // Tree without per-tile extras needs F | I and I / F >= 8: F = 6 fails 64 % 6, F = 4 fits.
  REQUIRE(uniform_grid(6 * 40, 40, 64, reduction_t::tree, 1, false, 0, big, 0) == 4 * 40);
  // Per-tile extras or parallel reduction drop F | I.
  REQUIRE(uniform_grid(6 * 40, 40, 64, reduction_t::tree, 1, true, 0, big, 0) == 6 * 40);
  REQUIRE(uniform_grid(6 * 40, 40, 64, reduction_t::parallel, 1, false, 0, big, 0) == 6 * 40);
  // Workspace, flag region and cluster width bound F.
  REQUIRE(uniform_grid(4 * 40, 40, 64, reduction_t::tree, 1, false, 1000, 1000 * 80, 0) == 80);
  REQUIRE(uniform_grid(4 * 40, 40, 64, reduction_t::tree, 1, false, 0, big, 100) == 80);
  REQUIRE(uniform_grid(4 * 40, 40, 64, reduction_t::tree, 1, false, 0, big, 79) == 40);
  REQUIRE(uniform_grid(4 * 3, 3, 64, reduction_t::tree, 8, false, 0, big, 0) == 3);
  REQUIRE(uniform_grid(3 * 4, 4, 64, reduction_t::tree, 8, false, 0, big, 0) == 8);

  lcg_t rnd{3};
  for (int i = 0; i < 20000; ++i) {
    const size_t tiles = 1 + rnd(300), I = 1 + rnd(256), grid = 1 + rnd(8 * tiles);
    const auto red     = rnd(2) ? reduction_t::tree : reduction_t::parallel;
    const size_t cw = 1 + rnd(2) * 3, tile_bytes = rnd(4) * 65536, ws = rnd(3) ? big : 1 << 24;
    const size_t flags = rnd(2) ? 0 : 2048;
    const bool per_tile = rnd(2) != 0;
    auto admissible = [&](size_t F) {
      return F == 1 || ((tiles * F) % cw == 0 &&
                        (red == reduction_t::parallel || per_tile || I % F == 0) && I / F >= 8 &&
                        tile_bytes * tiles * F <= ws && (flags == 0 || tiles * F <= flags));
    };
    const size_t got = uniform_grid(grid, tiles, I, red, cw, per_tile, tile_bytes, ws, flags);
    REQUIRE(got % tiles == 0);
    const size_t F = got / tiles;
    REQUIRE(admissible(F));
    if (grid <= tiles) REQUIRE(F == 1);
    for (size_t f = F + 1; f <= grid / tiles; ++f) REQUIRE_FALSE(admissible(f));
  }
}

TEST_CASE("Origami streamk: launch_terms of a launch", "[origami][streamk][launch]") {
  auto p        = make_problem(4096, 4096, 4096);
  p.d_dtype     = origami::data_type_t::BFloat16;
  origami::streamk::launch_t sk;
  sk.valid = true;
  // Not Stream-K: one workgroup per tile, nothing split.
  auto t = origami::streamk::launch_terms(p, sk);
  REQUIRE(t.grid_over_tiles == 1.0);
  REQUIRE(t.num_wgs == 0.0);

  // 300 tiles of 256x256 over 256 workgroups, 64 iterations each, occupancy 1 on 256 CUs.
  sk.is_streamk     = true;
  sk.final_grid     = 256;
  sk.tiles          = 300;
  sk.sk_tiles       = 256 + 300 % 256;
  sk.iters_per_tile = 64;
  sk.reduction      = origami::reduction_t::tree;
  sk.mt_x = sk.mt_y = 256;
  sk.workspace_size_per_elem_c = 4;
  sk.cu_occupancy              = 1;
  sk.cu_count                  = 256;
  const auto pt                = origami::streamk::launch_partials(sk);
  REQUIRE(pt.partials > 0);
  t = origami::streamk::launch_terms(p, sk);
  REQUIRE(t.grid_over_tiles == 256.0 / 300.0);
  REQUIRE(t.split_tiles_frac == static_cast<double>(pt.split_tiles) / 300.0);
  REQUIRE(t.fixup_bytes_over_output ==
          2.0 * static_cast<double>(pt.partials) * 256.0 * 256.0 * 4.0 / (4096.0 * 4096.0 * 2.0));
  REQUIRE(t.num_wgs == 256.0);
  REQUIRE(t.timesteps == 1.0 * 300.0 / 256.0);
  // A CU budget below the device's raises the waves.
  p.num_cus = 128;
  REQUIRE(origami::streamk::launch_terms(p, sk).timesteps == 2.0 * 300.0 / 256.0);
}

TEST_CASE("Origami streamk: context_t launches hipBLASLt's uniform-summation-order grid",
          "[origami][streamk][launch]") {
  // The grid rank_configs models is the one hipBLASLt launches: select_grid_size, then under
  // uniform summation order the row-uniform snap, then parallel reduction only at two or more
  // workgroups per tile.
  const auto hw = make_hardware(950);
  using origami::reduction_t;
  int snapped = 0, unchanged = 0;
  lcg_t rnd{5};
  for (int i = 0; i < 4000; ++i) {
    const size_t mt = size_t{32} << rnd(4), mt_k = size_t{32} << rnd(2);
    auto p = make_problem(1 + rnd(16384), 1 + rnd(16384), 1 + rnd(16384), origami::transpose_t::N,
                          origami::transpose_t::T, rnd(4) ? 1 : 1 + rnd(8));
    auto c = make_config(mt, mt, mt_k, 16, 16, 16, false, 1, 1 + static_cast<int>(rnd(3)), 0, 0,
                         rnd(2) ? 3 : 5);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = 4;
    c.tensile().stream_k_per_tile_extra_iters = rnd(2) != 0;

    const origami::gemm::context_t off(p, hw, c);
    p.uniform_summation_order = true;
    const origami::gemm::context_t on(p, hw, c);

    auto sel             = c;
    sel.reduction_strategy = origami::streamk::select_reduction(p, hw, c, c.grid_selection);
    const size_t g0      = origami::streamk::select_grid_size(p, hw, sel, c.grid_selection);
    const size_t tiles   = on.num_output_tiles;
    const size_t iters   = std::max<size_t>((p.size.k + mt_k - 1) / mt_k, 1);
    const size_t want    = origami::streamk::uniform_grid(
        g0, tiles, iters, sel.reduction_strategy, 1, c.tensile().stream_k_per_tile_extra_iters,
        mt * mt * 4, c.workspace_size,
        sel.reduction_strategy == reduction_t::parallel ? 0 : origami::streamk::FLAG_ELEMENTS);
    REQUIRE(off.num_wgs == g0);
    REQUIRE(on.num_wgs == want);
    REQUIRE(on.reduction_strategy ==
            origami::streamk::reconcile_reduction(sel.reduction_strategy, want, tiles));
    // Never fewer workgroups than tiles under uniform summation order.
    REQUIRE(on.num_wgs >= tiles);
    snapped += on.num_wgs != g0;
    unchanged += on.num_wgs == g0;
  }
  REQUIRE(snapped > 100);
  REQUIRE(unchanged > 100);
}

TEST_CASE("Origami streamk: launch_partials", "[origami][streamk][launch]") {
  // Partial tiles of the static split by brute force (no whole-tile shortcut): distinct cut
  // tiles, every cut a partial.
  auto brute = [](std::uint64_t sk, std::uint64_t iters, std::uint64_t G) {
    origami::streamk::partials_t out;
    if (G == 0 || sk == 0 || iters == 0) return out;
    const std::uint64_t T = sk * iters, W = T / G, E = T % G;
    std::vector<std::uint64_t> cut;
    for (std::uint64_t g = 1; g < G; ++g) {
      const std::uint64_t s = g * W + std::min(g, E);
      if (s % iters) cut.push_back(s / iters);
    }
    out.partials = cut.size();
    std::sort(cut.begin(), cut.end());
    out.split_tiles =
        static_cast<std::uint64_t>(std::unique(cut.begin(), cut.end()) - cut.begin());
    return out;
  };
  using origami::streamk::launch_partials;
  origami::streamk::launch_t base;
  base.valid = base.is_streamk = true;
  base.reduction               = origami::reduction_t::tree;
  base.sk_split                = 1;

  SECTION("static split: hand example and brute force") {
    auto d  = base;
    d.tiles = d.sk_tiles = 10;
    d.iters_per_tile     = 4;
    d.final_grid         = 3;  // T 40, W 13, E 1: starts 14 and 27 cut tiles 3 and 6
    auto pt              = launch_partials(d);
    REQUIRE(pt.partials == 2);
    REQUIRE(pt.split_tiles == 2);
    int with_cuts = 0, shortcut = 0;
    for (std::uint64_t sk : {1, 2, 7, 64, 255, 256, 1000})
      for (std::uint64_t iters : {1, 2, 3, 8, 13, 64})
        for (std::uint64_t G : {1, 2, 3, 100, 256, 512, 999, 1024}) {
          d.tiles = d.sk_tiles = sk;
          d.iters_per_tile     = iters;
          d.final_grid         = G;
          const auto mine = launch_partials(d), ref = brute(sk, iters, G);
          INFO("sk " << sk << " iters " << iters << " G " << G);
          REQUIRE(mine.partials == ref.partials);
          REQUIRE(mine.split_tiles == ref.split_tiles);
          with_cuts += ref.partials > 0;
          shortcut += (sk * iters) % G == 0 && ((sk * iters) / G) % iters == 0;
        }
    REQUIRE(with_cuts > 50);
    REQUIRE(shortcut > 20);
  }
  SECTION("parallel, dynamic, DP-only and non-Stream-K launches") {
    auto d       = base;
    d.tiles      = 40;
    d.sk_tiles   = 3;
    d.sk_split   = 3;
    d.final_grid = 120;
    d.reduction  = origami::reduction_t::parallel;
    auto pt      = launch_partials(d);
    REQUIRE(pt.partials == 120);
    REQUIRE(pt.split_tiles == 40);
    d.is_dynamic = true;
    REQUIRE(launch_partials(d).partials == 0);
    d       = base;
    d.tiles = d.sk_tiles = 10;
    d.iters_per_tile     = 4;
    d.final_grid         = 3;
    d.dp_only            = true;
    REQUIRE(launch_partials(d).partials == 0);
    d.dp_only    = false;
    d.is_streamk = false;
    REQUIRE(launch_partials(d).partials == 0);
  }
}

TEST_CASE("Origami streamk: compute_context_launch packs the context's launch",
          "[origami][streamk][launch]") {
  const auto hw = make_hardware(950);
  int parallel = 0, tree = 0, not_streamk = 0;
  for (const auto& s : std::vector<std::array<size_t, 4>>{
           {4096, 4224, 512, 1}, {512, 512, 16384, 1}, {1024, 1024, 8192, 1}, {3072, 3200, 1024, 2}})
    for (int sk : {0, 3, 5}) {
      auto p = make_problem(s[0], s[1], s[2], origami::transpose_t::N, origami::transpose_t::T, s[3]);
      p.uniform_summation_order = true;
      auto c                      = make_config(128, 128, 64, 16, 16, 32, false, 1, 2, 0, 0, sk);
      c.workspace_size            = std::numeric_limits<size_t>::max();
      c.workspace_size_per_elem_c = 4;
      const origami::gemm::context_t ctx(p, hw, c);
      const auto L = origami::gemm::compute_context_launch(p, hw, c, ctx);
      REQUIRE(L.valid);
      if (sk == 0) {
        REQUIRE_FALSE(L.is_streamk);
        ++not_streamk;
        continue;
      }
      REQUIRE(L.final_grid == ctx.num_wgs);
      REQUIRE(L.tiles == ctx.num_output_tiles);
      REQUIRE(L.reduction == ctx.reduction_strategy);
      REQUIRE(L.iters_per_tile == (s[2] + 63) / 64);
      REQUIRE(L.workspace_size_per_elem_c == 4);
      if (L.reduction == origami::reduction_t::parallel) {
        REQUIRE(L.sk_split == L.final_grid / L.tiles);
        REQUIRE(L.sk_split >= 2);
        ++parallel;
      } else {
        REQUIRE(L.sk_tiles ==
                origami::streamk::static_split(L.tiles, L.iters_per_tile, L.final_grid, 1, false)
                    .sk_tiles);
        ++tree;
      }
    }
  REQUIRE(parallel > 0);
  REQUIRE(tree > 0);
  REQUIRE(not_streamk > 0);
}
