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

TEST_CASE("Origami streamk: correct_sk_grid_for_partial_tiles", "[streamk]") {
  // Tests for the universal partial-tile SK grid correction.
  // Each case documents a case where we were picking the wrong SK grid as part 
  // of the WorkStealing investigation.
  // cu_count=224 is representative of a gfx950 device with a cotenant kernel of
  // size 32.
  //
  //   ipt      = iters_per_tile = floor(K / MT_K)
  //   DP_eff   = tiles / (ceil(tiles/cu_count) * cu_count)

  constexpr size_t cu = 224;

  auto correct = [&](size_t sk_grid, size_t tiles, size_t ipt, size_t batch = 1) {
    return origami::streamk::correct_sk_grid_for_partial_tiles(sk_grid, tiles, ipt, cu, batch);
  };

  // ---- Group A: ipt == 1 (tile-streaming, no K-split) --------------------
  // Force DP when tiles >= 2*cu_count: L2 N-locality beats tile-streaming.

  // [16, 884736, 32] and tile 16x512x32: tiles=1728, ipt=1.
  // SK (224 WGs) was ~2× slower than DP (1728 WGs) due to L2 pollution.
  REQUIRE(correct(224, 1728, 1) == 1728);

  // M=12 N=33554432 K=32 → tile 16x448x32: tiles=74899, ipt=1.
  // tiles = 74899 = ~334 waves >> 32*cu upper cap → too large for DP;
  // SK tile-streaming (sequential B) beats a 74899-WG DP grid.  Keep SK.
  // Measured: forcing DP here is 1.4x slower (+462us) than SK.
  REQUIRE(correct(224, 74899, 1) == 224);

  // tiles < 2*cu_count: keep SK (few tiles per WG, setup amortization helps).
  REQUIRE(correct(224, 400, 1) == 224);   // 400 < 2*224=448

  // sk_grid already == tiles: no-op.
  REQUIRE(correct(1728, 1728, 1) == 1728);

  // ---- Group B: batched shapes → always keep SK (cross-tile A/B reuse) -----
  // DYNAMIC_GRID=4 sweep confirmed: batched tiny GEMMs (e.g. tf32 N,N,8192,32,25,25)
  // see no improvement from k_split_aware grid — the SK grid is not the issue.
  // The batch>1 gate correctly preserves SK tile-streaming for A/B L1/L2 reuse.

  // M=16 N=96 K=32 batch=9216 → 16x128x32: tiles=9216, ipt=1.
  REQUIRE(correct(224, 9216, 1, 9216) == 224);

  // Batched K-splitting shape: batch=4096, tiles=384, ipt=4.
  REQUIRE(correct(224, 384, 4, 4096) == 224);

  // ---- Group C: ipt > 1, DP efficiency < 90% → keep SK ------------------
  // tiles barely above cu_count: DP last wave too thin, SK utilization wins.

  // M=4096 N=4096 K=4096 → 256x256x32: tiles=256, ipt=128.
  // DP_eff = 256/(2*224) = 57% < 90% → SK preserved.
  REQUIRE(correct(224, 256, 128) == 224);

  // M=2048 N=8192 K=5640 → 256x256x32: tiles=256, ipt=176. DP_eff=57%.
  REQUIRE(correct(224, 256, 176) == 224);

  // M=4096 N=4096 K=4096 with just 256 tiles — DP_eff=57%, truly needs SK.
  // (covered above)

  // ---- Group D: ipt > 1, DP_eff >= 80%, partial tiles → force DP ---------
  // Confirmed by DYNAMIC_GRID=4 sweep: k_split_aware (which applies the
  // floor/ceil partial-tile check) returns tiles for all these shapes,
  // recovering to base latency. The 80% threshold fires correctly for
  // DP_eff >= 80% while leaving 57% cases in SK (group C above).

  // M=160 N=107005 K=160 → 160x256x32: tiles=419, ipt=5, DP_eff=93%.
  // floor(419*5/224)=9, 9%5=4 → partial tile → DP.
  // Measured: SK 44µs, DP 30µs (L2 hit rate 40% vs 73%).
  REQUIRE(correct(224, 419, 5) == 419);

  // M=160 N=73390 K=320 → 160x192x64: tiles=382, ipt=5, DP_eff=85%.
  // SK "no env" = 36.9µs, DP (DYNAMIC_GRID=4) = 26.3µs ≈ base 26.6µs.
  REQUIRE(correct(224, 382, 5) == 382);

  // gfx950 (256 CUs, full): sk_grid=224 < cu_count=256 — SK leaves 32 CUs idle.
  // DP (382 WGs) is strictly better: all CUs active, no partial-tile overhead.
  // DYNAMIC_GRID=4 sweep confirmed recovery to base latency on gfx950.
  REQUIRE(origami::streamk::correct_sk_grid_for_partial_tiles(224, 382, 5, 256, 1) == 382);

  // gfx950 cotenant (224 CUs): sk_grid=224 == cu_count=224 — falls through to
  // DP_eff check: 382/(2*224)=85% >= 80% → fires → DP. Same outcome.
  REQUIRE(origami::streamk::correct_sk_grid_for_partial_tiles(224, 382, 5, 224, 1) == 382);

  // M=256 N=98304 K=128 → 256x256x32: tiles=384, ipt=4, DP_eff=85.7%.
  // SK "no env" = 65µs, DP (DYNAMIC_GRID=4) = 47.8µs ≈ base 46.9µs.
  REQUIRE(correct(224, 384, 4) == 384);

  // M=2048 N=9216 K=1480 → 256x192x64: tiles=384, ipt=23, DP_eff=85.7%.
  // SK "no env" = 98.1µs, DP (DYNAMIC_GRID=4) = 78.8µs ≈ base 78.3µs.
  REQUIRE(correct(224, 384, 23) == 384);

  // M=128 N=98304 K=256 → 128x256x32: tiles=384, ipt=8, DP_eff=85.7%.
  // SK "no env" = 48.6µs, DP (DYNAMIC_GRID=4) = 40.5µs ≈ base 40.9µs.
  REQUIRE(correct(224, 384, 8) == 384);

  // M=9984 N=2048 K=32768 → 192x256x64: tiles=416, ipt=512, DP_eff=92%.
  // floor(416*512/224)=951, 951%512=439 → partial → DP.
  REQUIRE(correct(224, 416, 512) == 416);

  // ---- Group E: clean SK grid (no boundary crossing) → keep SK -----------

  // tiles=420, ipt=5, sk_grid=210: each CTA gets 420*5/210=10 iters, 10%5=0.
  REQUIRE(correct(210, 420, 5) == 210);

  // ---- Edge cases ---------------------------------------------------------

  REQUIRE(correct(0,   419, 5) == 0);    // sk_grid==0: no-op
  REQUIRE(correct(500, 419, 5) == 500);  // sk_grid>=tiles: already DP
  REQUIRE(correct(224, 419, 0) == 224);  // ipt==0: degenerate, no-op

  // tiles < cu_count: K-splitting needed, guard always skipped.
  // M=4352 N=128 K=8192 → 256x128x32: tiles=17 < 224.
  REQUIRE(correct(224, 17, 256) == 224);
}
