/*******************************************************************************
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 *******************************************************************************/

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include "common.hpp"

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
  // Force DP when tiles in [2*cu_count, 32*cu_count): L2 N-locality wins.

  // [16, 884736, 32] and tile 16x512x32: tiles=1728, ipt=1.
  // SK (224 WGs) was ~2× slower than DP (1728 WGs) due to L2 pollution.
  REQUIRE(run(1728, 1, 224) == 1728);

  // [12, 33554432, 32] and tile 16x448x32: tiles=74899, ipt=1.
  // tiles=74899 ~334 waves >> 32×cu cap → DP grid too large; SK tile-streaming
  // (sequential B-panel reuse) wins.  Measured: forcing DP +462µs slower.
  REQUIRE(run(74899, 1, 224) == 224);

  // tiles=400, ipt=1: tiles < 2*cu_count=448 → keep SK.
  // Few tiles per WG; SK setup amortization beats a shallow DP wave.
  REQUIRE(run(400, 1, 224) == 224);

  // ---- Group B: batched shapes → always keep SK (cross-tile A/B reuse) -----
  // DYNAMIC_GRID=4 sweep confirmed: batched tiny GEMMs (e.g. tf32 N,N,8192,32,25,25)
  // see no improvement from k_split_aware grid — the SK grid is not the issue.
  // The batch>1 gate correctly preserves SK tile-streaming for A/B L1/L2 reuse.

  // [16, 96, 32] batch=9216 and tile 16x128x32: tiles=9216, ipt=1.
  // Batched: SK preserves cross-tile A/B reuse across the batch dimension.
  REQUIRE(run(9216, 1, 224, 9216) == 224);

  // Batched K-splitting shape: batch=4096, tiles=384, ipt=4.
  // batch>1 gate fires before DP_eff check → always keep SK.
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
