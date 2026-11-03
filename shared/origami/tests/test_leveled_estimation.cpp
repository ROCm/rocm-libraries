// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <limits>
#include <vector>
#include "common.hpp"

// Tests for the leveled estimation cascade that rank_configs() uses when
// ORIGAMI_LEVELED_ESTIMATION is set.

namespace {

constexpr double kRejected = std::numeric_limits<double>::max();

// Cascade tunables, mirrored from estimation_leveled.cpp.
constexpr size_t kMinKeep      = 48;
constexpr double kKeepFraction = 1.0 / 6.0;

// Sets the leveled-estimation option for the scope and restores the prior value on exit.
struct LeveledGuard {
  explicit LeveledGuard(bool enabled) : prev_(origami::runtime_options::get().leveled_estimation) {
    origami::runtime_options::get().leveled_estimation = enabled;
  }
  ~LeveledGuard() { origami::runtime_options::get().leveled_estimation = prev_; }
  bool prev_;
};

std::vector<origami::prediction_result_t> rank(const origami::problem_t& problem,
                                               const origami::hardware_t& hardware,
                                               const std::vector<origami::config_t>& configs,
                                               bool leveled) {
  LeveledGuard guard(leveled);
  return origami::rank_configs(problem, hardware, configs);
}

// A sweep large enough that the keep fraction, not the floor, sets the prune size.
std::vector<origami::config_t> sweep_configs() {
  std::vector<origami::config_t> configs;
  for (size_t mt_m : {16, 32, 64, 96, 128, 192, 256})
    for (size_t mt_n : {16, 32, 64, 96, 128, 192, 256})
      for (size_t mt_k : {32, 64, 128, 256})
        for (int occupancy : {1, 2})
          for (int wgm : {1, 8}) {
            configs.push_back(make_config(mt_m, mt_n, mt_k, 16, 16, 16, false, wgm, occupancy));
            configs.push_back(make_config(mt_m, mt_n, mt_k, 32, 32, 8, false, wgm, occupancy));
          }
  for (size_t i = 0; i < configs.size(); ++i) configs[i].index = i;
  return configs;
}

std::vector<origami::config_t> sample_configs() {
  std::vector<origami::config_t> configs;
  configs.push_back(make_config(256, 256, 32, 32, 32, 8, false, 1, 6, 0, 0));
  configs.push_back(make_config(128, 128, 64, 32, 32, 8, false, 1, 6, 0, 0));
  configs.push_back(make_config(64, 64, 64, 32, 32, 8, false, 1, 6, 0, 0));
  return configs;
}

// Number of configs the single-pass ranking does not reject.
size_t count_accepted(const origami::problem_t& problem,
                      const origami::hardware_t& hardware,
                      const std::vector<origami::config_t>& configs) {
  return std::count_if(configs.begin(), configs.end(), [&](const origami::config_t& config) {
    return origami::gemm::check_lds_capacity(
               hardware, config.mt, problem.a_dtype, problem.b_dtype) &&
           !origami::gemm::fast_reject(problem, hardware, config);
  });
}

size_t expected_keep(size_t accepted) {
  return std::max(kMinKeep, static_cast<size_t>(static_cast<double>(accepted) * kKeepFraction));
}

void require_same_ranking(const std::vector<origami::prediction_result_t>& a,
                          const std::vector<origami::prediction_result_t>& b) {
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE(a[i].config == b[i].config);
    REQUIRE(a[i].config.index == b[i].config.index);
    REQUIRE(a[i].latency == b[i].latency);
  }
}

// Number of leading leveled results scored with the full model. The configs pruned by
// the proxy follow them with the rejected latency.
size_t count_scored(const std::vector<origami::prediction_result_t>& leveled) {
  auto is_rejected = [](const origami::prediction_result_t& r) { return r.latency == kRejected; };
  return std::find_if(leveled.begin(), leveled.end(), is_rejected) - leveled.begin();
}

// The leveled ranking holds the same configs as the single pass. The scored survivors
// come first, in ascending latency, each with the single-pass latency. The pruned configs
// follow, all with the rejected latency.
void require_leveled_ranking(const std::vector<origami::prediction_result_t>& leveled,
                             const std::vector<origami::prediction_result_t>& flat) {
  REQUIRE(leveled.size() == flat.size());

  std::vector<size_t> leveled_indices, flat_indices;
  for (const auto& r : leveled) leveled_indices.push_back(r.config.index);
  for (const auto& r : flat) flat_indices.push_back(r.config.index);
  std::sort(leveled_indices.begin(), leveled_indices.end());
  std::sort(flat_indices.begin(), flat_indices.end());
  REQUIRE(leveled_indices == flat_indices);

  const size_t scored = count_scored(leveled);
  REQUIRE(scored > 0);
  for (size_t i = 0; i < scored; ++i) {
    auto it = std::find_if(flat.begin(), flat.end(), [&](const origami::prediction_result_t& f) {
      return f.config.index == leveled[i].config.index;
    });
    REQUIRE(it != flat.end());
    REQUIRE(it->latency == leveled[i].latency);
    if (i > 0) REQUIRE(leveled[i - 1].latency <= leveled[i].latency);
  }
  for (size_t i = scored; i < leveled.size(); ++i) REQUIRE(leveled[i].latency == kRejected);
}

// The proxy sees only MT, MI, PGR and the data types, so configs that share them have the
// same proxy cost. The test configs carry no Tensile params, so PGR is the default for all.
// The cut must keep or prune each such group as a whole.
void require_whole_proxy_groups(const std::vector<origami::prediction_result_t>& leveled) {
  const size_t scored = count_scored(leveled);
  auto same_group     = [](const origami::config_t& a, const origami::config_t& b) {
    return a.mt == b.mt && a.mi == b.mi;
  };
  for (size_t i = 0; i < scored; ++i)
    for (size_t j = scored; j < leveled.size(); ++j)
      REQUIRE_FALSE(same_group(leveled[i].config, leveled[j].config));
}

std::vector<size_t> scored_indices(const std::vector<origami::prediction_result_t>& leveled) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < count_scored(leveled); ++i) indices.push_back(leveled[i].config.index);
  std::sort(indices.begin(), indices.end());
  return indices;
}

}  // namespace

TEST_CASE("Leveled: ORIGAMI_LEVELED_ESTIMATION parsing", "[origami][leveled]") {
  portable_unsetenv("ORIGAMI_LEVELED_ESTIMATION");
  REQUIRE_FALSE(origami::runtime_options::read_leveled_estimation_from_env());
  portable_setenv("ORIGAMI_LEVELED_ESTIMATION", "1", 1);
  REQUIRE(origami::runtime_options::read_leveled_estimation_from_env());
  portable_setenv("ORIGAMI_LEVELED_ESTIMATION", "0", 1);
  REQUIRE_FALSE(origami::runtime_options::read_leveled_estimation_from_env());
  // Decimal: a leading zero does not switch to octal.
  portable_setenv("ORIGAMI_LEVELED_ESTIMATION", "08", 1);
  REQUIRE(origami::runtime_options::read_leveled_estimation_from_env());
  portable_setenv("ORIGAMI_LEVELED_ESTIMATION", "00", 1);
  REQUIRE_FALSE(origami::runtime_options::read_leveled_estimation_from_env());
  portable_setenv("ORIGAMI_LEVELED_ESTIMATION", "on", 1);
  REQUIRE_FALSE(origami::runtime_options::read_leveled_estimation_from_env());
  portable_unsetenv("ORIGAMI_LEVELED_ESTIMATION");

  const bool prev = origami::runtime_options::get().leveled_estimation;
  origami::runtime_options::get().update_from_env();
  REQUIRE_FALSE(origami::runtime_options::get().leveled_estimation);
  origami::runtime_options::get().leveled_estimation = prev;

  // The explicit-value constructor leaves the cascade off.
  origami::runtime_options explicit_options(false, true, 0.0);
  REQUIRE_FALSE(explicit_options.leveled_estimation);
}

TEST_CASE("Leveled: off by default ranks every accepted config", "[origami][leveled]") {
  // The ranking baselines are the main check that the default path is unchanged;
  // this checks the default path still scores every accepted config.
  auto configs = sweep_configs();
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(8192, 8192, 4096);
      LeveledGuard guard(false);
      auto results = origami::rank_configs(problem, hardware, configs);
      REQUIRE(results.size() == count_accepted(problem, hardware, configs));
      for (const auto& result : results) {
        REQUIRE(result.latency ==
                origami::gemm::compute_total_latency(problem, hardware, result.config));
      }
    }
  }
}

TEST_CASE("Leveled: small config lists match the single pass", "[origami][leveled]") {
  // At most kMinKeep accepted configs are never pruned, so the cascade scores the
  // same set with the same latencies and the ranking is identical.
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();
      require_same_ranking(rank(problem, hardware, configs, true),
                           rank(problem, hardware, configs, false));
    }
  }
}

TEST_CASE("Leveled: whenever nothing is pruned the ranking matches the single pass",
          "[origami][leveled]") {
  // Sweep shapes (including M=1, N=1 and split-K-prone ones), all four transpose
  // combinations, and config lists of every size class: windows of the sweep small
  // enough that the floor keeps them whole, the same windows reversed, and the whole
  // sweep. The windows contain variants that differ only in WGM and occupancy, so latency
  // ties reach the tie-breakers. Whenever the cascade prunes nothing, it must return the
  // single-pass ranking exactly: same order, same configs, same latencies.
  const auto sweep         = sweep_configs();
  constexpr size_t kWindow = 40;
  static_assert(kWindow <= kMinKeep, "windows must never be pruned");

  std::vector<std::vector<origami::config_t>> lists;
  for (size_t begin = 0; begin < sweep.size(); begin += kWindow) {
    const size_t end = std::min(begin + kWindow, sweep.size());
    lists.emplace_back(sweep.begin() + begin, sweep.begin() + end);
    lists.emplace_back(sweep.rbegin() + (sweep.size() - end),
                       sweep.rbegin() + (sweep.size() - begin));
  }
  lists.push_back(sweep);

  const std::vector<std::array<size_t, 4>> shapes = {
      {1, 1, 4096, 1},        {1, 8192, 4096, 1},   {8192, 1, 4096, 1},    {540080, 1, 6768, 1},
      {1, 596864, 11200, 1},  {42720, 1, 18784, 1}, {1, 4096, 262144, 1},  {2, 1024, 512, 1},
      {64, 64, 64, 1},        {96, 224, 240, 1},    {688, 768, 912, 1},    {3760, 1168, 2176, 1},
      {1024, 1024, 16384, 1}, {256, 256, 65536, 1}, {8192, 8192, 4096, 1}, {1, 2048, 1024, 8},
      {2048, 1, 1024, 8},     {128, 128, 8192, 16}};

  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware         = make_hardware(gpu_arch);
      size_t unpruned       = 0;
      size_t unpruned_m1_n1 = 0;
      size_t unpruned_whole = 0;
      for (const auto& [m, n, k, batch] : shapes)
        for (auto ta : {origami::transpose_t::N, origami::transpose_t::T})
          for (auto tb : {origami::transpose_t::N, origami::transpose_t::T}) {
            const auto problem = make_problem(m, n, k, ta, tb, batch);
            for (size_t l = 0; l < lists.size(); ++l) {
              const auto& configs = lists[l];
              auto flat           = rank(problem, hardware, configs, false);
              auto leveled        = rank(problem, hardware, configs, true);

              // Pruned: a config that the single pass scores carries the rejected latency
              // in the leveled ranking, because the proxy cut it.
              bool pruned = false;
              for (const auto& r : leveled) {
                if (r.latency != kRejected) continue;
                auto it = std::find_if(flat.begin(), flat.end(), [&](const auto& f) {
                  return f.config.index == r.config.index;
                });
                if (it != flat.end() && it->latency != kRejected) pruned = true;
              }

              // Lists no larger than the floor are never pruned.
              if (configs.size() <= kMinKeep) REQUIRE_FALSE(pruned);
              if (pruned) continue;

              INFO("shape " << m << "x" << n << "x" << k << "x" << batch << " list " << l
                            << " size " << configs.size());
              require_same_ranking(leveled, flat);
              ++unpruned;
              if (m == 1 || n == 1) ++unpruned_m1_n1;
              if (configs.size() == sweep.size()) ++unpruned_whole;
            }
          }
      // The sweep must exercise the property, including M=1/N=1 shapes and the
      // split-K-prone skip on the whole sweep.
      REQUIRE(unpruned_m1_n1 > 0);
      REQUIRE(unpruned_whole > 0);
      REQUIRE(unpruned > 2 * shapes.size());
    }
  }
}

TEST_CASE("Leveled: split-K-prone shapes skip the prune", "[origami][leveled]") {
  // Few output tiles and a deep K: every config that passes level 0 reaches the full
  // model, so the ranking matches the single pass exactly. This also shows level 0
  // keeps exactly the configs the single pass accepts.
  auto configs = sweep_configs();
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      for (const auto& problem : {make_problem(1024, 1024, 16384),
                                  make_problem(512, 2048, 8192, origami::transpose_t::N),
                                  make_problem(256, 256, 65536)}) {
        auto flat = rank(problem, hardware, configs, false);
        REQUIRE(flat.size() > expected_keep(flat.size()));
        require_same_ranking(rank(problem, hardware, configs, true), flat);
      }
    }
  }
}

TEST_CASE("Leveled: GEMV shapes skip the prune", "[origami][leveled]") {
  // M == 1 or N == 1 with a shallow K, so the split-K-prone skip does not apply: every
  // config that passes level 0 reaches the full model. The same shapes with M or N of 2
  // are pruned, so the GEMV skip is what keeps these lists whole.
  auto configs = sweep_configs();
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      for (const auto& [m, n, batch] : std::vector<std::array<size_t, 3>>{
               {1, 65536, 1}, {65536, 1, 1}, {1, 2048, 8}, {2048, 1, 8}}) {
        INFO("shape " << m << "x" << n << "x1024x" << batch);
        const auto gemv =
            make_problem(m, n, 1024, origami::transpose_t::T, origami::transpose_t::N, batch);
        auto flat = rank(gemv, hardware, configs, false);
        REQUIRE(flat.size() > expected_keep(flat.size()));
        require_same_ranking(rank(gemv, hardware, configs, true), flat);

        const auto not_gemv = make_problem(std::max<size_t>(m, 2),
                                           std::max<size_t>(n, 2),
                                           1024,
                                           origami::transpose_t::T,
                                           origami::transpose_t::N,
                                           batch);
        auto leveled        = rank(not_gemv, hardware, configs, true);
        REQUIRE(count_scored(leveled) < leveled.size());
      }
    }
  }
}

TEST_CASE("Leveled: small-grid shapes keep the single-pass pick", "[origami][leveled]") {
  // A small output grid and a shallow K. Large tiles leave CUs idle or need a K-split
  // with a reduction, and the proxy's data-parallel wave schedule charges them a whole
  // wave each. A proxy that spreads the K-iterations evenly over the CUs instead ranks
  // them first and prunes the single-pass pick.
  auto configs  = sweep_configs();
  auto hardware = make_hardware(950);
  for (const auto& problem :
       {make_problem(688, 768, 912, origami::transpose_t::N), make_problem(688, 768, 912)}) {
    auto flat    = rank(problem, hardware, configs, false);
    auto leveled = rank(problem, hardware, configs, true);
    require_leveled_ranking(leveled, flat);
    REQUIRE(count_scored(leveled) < leveled.size());
    REQUIRE(leveled.front().config.index == flat.front().config.index);
    REQUIRE(leveled.front().latency == flat.front().latency);
  }
}

TEST_CASE("Leveled: prune keeps max(floor, fraction) of the accepted configs",
          "[origami][leveled]") {
  auto configs = sweep_configs();
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      for (const auto& problem :
           {make_problem(8192, 8192, 4096),
            make_problem(4096, 16384, 1024, origami::transpose_t::N),
            make_problem(8192, 8192, 8192, origami::transpose_t::T, origami::transpose_t::T, 4)}) {
        auto flat             = rank(problem, hardware, configs, false);
        auto leveled          = rank(problem, hardware, configs, true);
        const size_t accepted = count_accepted(problem, hardware, configs);
        REQUIRE(flat.size() == accepted);
        require_leveled_ranking(leveled, flat);
        require_whole_proxy_groups(leveled);

        // The cut keeps at least the target; it only grows to finish a proxy-cost tie group.
        const size_t scored = count_scored(leveled);
        REQUIRE(scored >= expected_keep(accepted));
        REQUIRE(scored < flat.size());

        // The cascade scores a subset, so it cannot find a lower latency than the single pass.
        REQUIRE(leveled.front().latency >= flat.front().latency);
      }
    }
  }
}

TEST_CASE("Leveled: kept set does not depend on config order", "[origami][leveled]") {
  // Proxy-cost ties at the cut are kept whole, so reversing the list keeps the same configs.
  auto configs  = sweep_configs();
  auto reversed = configs;
  std::reverse(reversed.begin(), reversed.end());
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(8192, 8192, 4096);
      auto forward  = rank(problem, hardware, configs, true);
      auto backward = rank(problem, hardware, reversed, true);
      REQUIRE(forward.size() == backward.size());
      REQUIRE(scored_indices(forward) == scored_indices(backward));
    }
  }
}

TEST_CASE("Leveled: split-K-prone test uses the problem's CU budget", "[origami][leveled]") {
  // 2560x2560 has 100 reference tiles: fewer than the hardware model's CU count, more
  // than a 64-CU budget. With all CUs the shape is split-K-prone (no prune). With the budget
  // it is not, and the cascade prunes.
  auto configs  = sweep_configs();
  auto hardware = make_hardware(950);
  auto problem  = make_problem(2560, 2560, 8192);

  problem.num_cus = 0;
  require_same_ranking(rank(problem, hardware, configs, true),
                       rank(problem, hardware, configs, false));

  problem.num_cus       = 64;
  auto flat             = rank(problem, hardware, configs, false);
  auto leveled          = rank(problem, hardware, configs, true);
  const size_t accepted = count_accepted(problem, hardware, configs);
  REQUIRE(flat.size() == accepted);
  require_leveled_ranking(leveled, flat);
  require_whole_proxy_groups(leveled);
  REQUIRE(count_scored(leveled) >= expected_keep(accepted));
  REQUIRE(count_scored(leveled) < accepted);

  // A budget at or above the physical CU count is the same as no budget.
  problem.num_cus = 1024;
  require_same_ranking(rank(problem, hardware, configs, true),
                       rank(problem, hardware, configs, false));
}

TEST_CASE("Leveled: lists with a simulation-mode config take the single pass",
          "[origami][leveled]") {
  // The cascade only handles estimation-mode lists. One simulation-mode config sends the
  // whole list through the single pass, so the ranking is identical even for a list
  // large enough to be pruned.
  auto configs = sweep_configs();
  for (int gpu_arch : test_architectures) {
    if (gpu_arch == 1250) continue;  // Formocast not yet supported on gfx1250
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware                      = make_hardware(gpu_arch);
      auto problem                       = make_problem(8192, 8192, 4096);
      auto mixed                         = configs;
      auto& sim                          = mixed.front();
      sim.prediction_mode                = origami::prediction_modes_t::simulation;
      sim.tensile().depth_u              = sim.mt.k;
      sim.tensile().global_split_u       = 1;
      sim.grvw_a                         = 4;
      sim.grvw_b                         = 4;
      sim.gwvw_d                         = 4;
      sim.tensile().wave_num             = 4;
      sim.tensile().wave_group_m         = 2;
      sim.tensile().wave_group_n         = 2;
      sim.tensile().prefetch_global_read = 2;
      auto flat                          = rank(problem, hardware, mixed, false);
      REQUIRE(flat.size() > expected_keep(flat.size()));
      require_same_ranking(rank(problem, hardware, mixed, true), flat);
    }
  }
}

TEST_CASE("Leveled: level 0 applies the ANALYTICAL_GEMM_PICK filter", "[origami][leveled]") {
  // The pick filter rejects every config with another MT before the model runs. Level 0
  // must reject the same configs, so the few that match are ranked like the single pass.
  struct PickGuard {
    origami::dim3_t prev = origami::runtime_options::get().gemm_pick;
    ~PickGuard() { origami::runtime_options::get().gemm_pick = prev; }
  } pick_guard;
  const origami::dim3_t pick                = {128, 128, 64};
  origami::runtime_options::get().gemm_pick = pick;

  auto configs = sweep_configs();
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch) {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(8192, 8192, 4096);
      auto flat     = rank(problem, hardware, configs, false);
      REQUIRE(!flat.empty());
      REQUIRE(flat.size() < configs.size());
      for (const auto& r : flat) REQUIRE(r.config.mt == pick);
      require_same_ranking(rank(problem, hardware, configs, true), flat);
    }
  }
}

TEST_CASE("Leveled: no valid configs falls back like the single pass", "[origami][leveled]") {
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - all configs exceed LDS") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 1024);
      std::vector<origami::config_t> configs;
      if (gpu_arch == 942) {
        configs.push_back(make_config(256, 256, 128, 32, 32, 8, false, 1, 6, 0, 0));
        configs.push_back(make_config(128, 128, 256, 32, 32, 8, false, 1, 6, 0, 0));
      } else {
        configs.push_back(make_config(512, 512, 256, 32, 32, 8, false, 1, 6, 0, 0));
        configs.push_back(make_config(128, 128, 512, 32, 32, 8, false, 1, 6, 0, 0));
      }
      auto flat = rank(problem, hardware, configs, false);
      REQUIRE(flat.size() == configs.size());
      REQUIRE(flat.front().latency == kRejected);
      require_same_ranking(rank(problem, hardware, configs, true), flat);
    }

    DYNAMIC_SECTION("gfx" << gpu_arch << " - all configs fast-rejected") {
      // Small batched problem: every tile smaller than the problem is short-circuited.
      auto hardware = make_hardware(gpu_arch);
      auto problem =
          make_problem(128, 128, 512, origami::transpose_t::T, origami::transpose_t::N, 4);
      std::vector<origami::config_t> configs;
      configs.push_back(make_config(64, 64, 64, 16, 16, 16));
      configs.push_back(make_config(32, 64, 64, 16, 16, 16));
      configs.push_back(make_config(64, 32, 64, 16, 16, 16));
      auto flat = rank(problem, hardware, configs, false);
      REQUIRE(flat.size() == configs.size());
      REQUIRE(flat.front().latency == kRejected);
      require_same_ranking(rank(problem, hardware, configs, true), flat);
    }
  }
}

TEST_CASE("Leveled: empty config list still throws", "[origami][leveled]") {
  auto hardware = make_hardware(950);
  auto problem  = make_problem(1024, 1024, 1024);
  std::vector<origami::config_t> empty_configs;
  LeveledGuard guard(true);
  REQUIRE_THROWS_WITH(origami::rank_configs(problem, hardware, empty_configs),
                      "No configurations provided.");
}
