/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "origami/model.hpp"
#include "common.hpp"

// Tests for the multi-phase ranking pipeline (origami.hpp/cpp).

namespace {

// Pin the tie-break variance to 0 for the duration of a test so ranking is a
// pure latency order (tie-breaks only fire on exactly-equal latencies). The
// prior value is captured and restored on scope exit, so these tests neither
// depend on nor perturb any global ANALYTICAL_GEMM_HEURISTICS_VARIANCE.
struct VarianceGuard {
  VarianceGuard() {
    if (const char* prev = std::getenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE")) {
      had_prev_ = true;
      prev_     = prev;
    }
    portable_setenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE", "0.0", 1);
  }
  ~VarianceGuard() {
    if (had_prev_)
      portable_setenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE", prev_.c_str(), 1);
    else
      portable_unsetenv("ANALYTICAL_GEMM_HEURISTICS_VARIANCE");
  }

  bool had_prev_ = false;
  std::string prev_;
};

// Build a single-phase pipeline running one model fidelity with a prune policy.
origami::ranking_pipeline_t single_phase(origami::model_t model,
                                         origami::target_t target,
                                         origami::prediction_modes_t fidelity,
                                         origami::prune_policy_t prune) {
  origami::ranking_phase_t phase;
  phase.model    = model;
  phase.target   = target;
  phase.fidelity = fidelity;
  phase.prune    = prune;

  origami::ranking_pipeline_t pipeline;
  pipeline.phases.push_back(phase);
  return pipeline;
}

// Three feasible candidate kernels used across several cases.
std::vector<origami::config_t> sample_configs() {
  std::vector<origami::config_t> configs;
  configs.push_back(make_config(256, 256, 32, 32, 32, 8, false, 1, 6, 0, 0));
  configs.push_back(make_config(128, 128, 64, 32, 32, 8, false, 1, 6, 0, 0));
  configs.push_back(make_config(64, 64, 64, 32, 32, 8, false, 1, 6, 0, 0));
  return configs;
}

}  // namespace

TEST_CASE("Pipeline: single-phase matches rank_configs", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - single estimation phase equals rank_configs") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();

      origami::prune_policy_t no_prune;  // kind == none
      auto pipeline = single_phase(origami::model_t::gemm,
                                   origami::target_t::tensilelite,
                                   origami::prediction_modes_t::estimation,
                                   no_prune);

      auto baseline = origami::rank_configs(problem, hardware, configs);
      auto staged   = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);

      REQUIRE(staged.size() == baseline.size());
      for (size_t i = 0; i < staged.size(); ++i) {
        REQUIRE(staged[i].config.mt.m == baseline[i].config.mt.m);
        REQUIRE(staged[i].config.mt.n == baseline[i].config.mt.n);
        REQUIRE(staged[i].config.mt.k == baseline[i].config.mt.k);
        REQUIRE(staged[i].latency == baseline[i].latency);
      }
    }
  }
}

TEST_CASE("Pipeline: top_k pruning keeps the best K survivors", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - top_k prune") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();

      origami::prune_policy_t prune;
      prune.kind  = origami::prune_kind_t::top_k;
      prune.top_k = 2;
      auto pipeline = single_phase(origami::model_t::gemm,
                                   origami::target_t::tensilelite,
                                   origami::prediction_modes_t::estimation,
                                   prune);

      auto baseline = origami::rank_configs(problem, hardware, configs);
      auto staged   = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);

      REQUIRE(staged.size() == 2);
      // The survivors are the two best from the full ranking, in order.
      for (size_t i = 0; i < staged.size(); ++i) {
        REQUIRE(staged[i].config.mt.m == baseline[i].config.mt.m);
        REQUIRE(staged[i].config.mt.n == baseline[i].config.mt.n);
      }
      // Survivors remain sorted by latency (best first).
      REQUIRE(staged[0].latency <= staged[1].latency);
    }
  }
}

TEST_CASE("Pipeline: top_fraction pruning rounds up", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - top_fraction prune") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();  // 3 feasible configs

      origami::prune_policy_t prune;
      prune.kind     = origami::prune_kind_t::top_fraction;
      prune.fraction = 0.5;  // ceil(0.5 * 3) == 2
      auto pipeline  = single_phase(origami::model_t::gemm,
                                   origami::target_t::tensilelite,
                                   origami::prediction_modes_t::estimation,
                                   prune);

      auto staged = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);
      REQUIRE(staged.size() == 2);
    }
  }
}

TEST_CASE("Pipeline: within_fraction_of_best keeps all when generous", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - within_fraction_of_best prune") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();

      // A huge tolerance keeps every feasible config; a tiny one keeps just the best.
      origami::prune_policy_t generous;
      generous.kind     = origami::prune_kind_t::within_fraction_of_best;
      generous.fraction = 100.0;
      auto pipeline_all = single_phase(origami::model_t::gemm,
                                       origami::target_t::tensilelite,
                                       origami::prediction_modes_t::estimation,
                                       generous);
      auto all = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline_all);
      REQUIRE(all.size() == configs.size());

      origami::prune_policy_t tight;
      tight.kind     = origami::prune_kind_t::within_fraction_of_best;
      tight.fraction = 0.0;
      tight.min_keep = 1;
      auto pipeline_best = single_phase(origami::model_t::gemm,
                                        origami::target_t::tensilelite,
                                        origami::prediction_modes_t::estimation,
                                        tight);
      auto best = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline_best);
      REQUIRE(best.size() >= 1);
      REQUIRE(best.front().latency == all.front().latency);
    }
  }
}

TEST_CASE("Pipeline: min_keep floor prevents empty survivors", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - min_keep floor") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();

      origami::prune_policy_t prune;
      prune.kind     = origami::prune_kind_t::top_k;
      prune.top_k    = 0;  // would keep nothing...
      prune.min_keep = 1;  // ...but the floor keeps the single best
      auto pipeline  = single_phase(origami::model_t::gemm,
                                   origami::target_t::tensilelite,
                                   origami::prediction_modes_t::estimation,
                                   prune);

      auto staged = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);
      REQUIRE(staged.size() == 1);
    }
  }
}

TEST_CASE("Pipeline: empty configs throw, empty pipeline is single-pass", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - empty inputs") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);

      origami::prune_policy_t no_prune;
      auto pipeline = single_phase(origami::model_t::gemm,
                                   origami::target_t::tensilelite,
                                   origami::prediction_modes_t::estimation,
                                   no_prune);

      // No configs is always an error, with or without a pipeline.
      std::vector<origami::config_t> empty_configs;
      REQUIRE_THROWS_WITH(
          origami::rank_configs(problem, hardware, empty_configs, origami::model_t::gemm, pipeline),
          "No configurations provided.");

      // An empty pipeline is the default single-pass mode (not an error): it
      // ranks all feasible configs, matching plain rank_configs.
      auto configs = sample_configs();
      origami::ranking_pipeline_t empty_pipeline;
      auto via_empty = origami::rank_configs(problem, hardware, configs,
                                             origami::model_t::gemm, empty_pipeline);
      auto baseline  = origami::rank_configs(problem, hardware, configs);
      REQUIRE(via_empty.size() == baseline.size());
      REQUIRE(via_empty.size() == configs.size());
    }
  }
}

TEST_CASE("Pipeline: infeasible configs are rejected", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - all configs exceed LDS") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 1024);

      std::vector<origami::config_t> invalid_configs;
      if (gpu_arch == 942) {
        invalid_configs.push_back(make_config(256, 256, 128, 32, 32, 8, false, 1, 6, 0, 0));
        invalid_configs.push_back(make_config(128, 128, 256, 32, 32, 8, false, 1, 6, 0, 0));
      } else {  // gfx950 / gfx1250
        invalid_configs.push_back(make_config(512, 512, 256, 32, 32, 8, false, 1, 6, 0, 0));
        invalid_configs.push_back(make_config(128, 128, 512, 32, 32, 8, false, 1, 6, 0, 0));
      }
      // The LDS feasibility gate is skipped for tensilelite (configs are assumed
      // library-validated), so use a target that still runs the capacity check to
      // exercise rejection of LDS-overflow configs.
      for (auto& c : invalid_configs) c.target = origami::target_t::generic;

      origami::prune_policy_t no_prune;
      auto pipeline = single_phase(origami::model_t::gemm,
                                   origami::target_t::generic,
                                   origami::prediction_modes_t::estimation,
                                   no_prune);

      REQUIRE_THROWS_WITH(
          origami::rank_configs(problem, hardware, invalid_configs, origami::model_t::gemm, pipeline),
          "No valid configs found.");
    }
  }
}

TEST_CASE("Pipeline: estimation-then-simulation cascade", "[origami][pipeline][formocast]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    if (gpu_arch == 1250) continue;  // Formocast not yet supported on gfx1250
    DYNAMIC_SECTION("gfx" << gpu_arch << " - cascade narrows then re-ranks with simulation") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(2048, 2048, 2048);

      // Candidates carry the Tensile params the simulation phase reads.
      std::vector<origami::config_t> configs;
      for (auto mt : {std::tuple{128, 128, 32}, std::tuple{256, 256, 32}, std::tuple{64, 64, 32}}) {
        auto [mt_m, mt_n, mt_k] = mt;
        auto config             = make_config(mt_m, mt_n, mt_k, 16, 16, 16, false, 8, 2);
        config.tensile().grvw_a = 4;
        config.tensile().grvw_b = 4;
        config.tensile().gwvw_d = 4;
        config.tensile().depth_u              = mt_k;
        config.tensile().global_split_u       = 1;
        config.tensile().wave_num             = 4;
        config.tensile().wave_group_m         = 2;
        config.tensile().wave_group_n         = 2;
        config.tensile().prefetch_global_read = 2;
        configs.push_back(config);
      }

      auto pipeline =
          origami::make_cascade_pipeline(origami::model_t::gemm, origami::target_t::tensilelite, 2);

      auto staged = origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);

      // Estimation keeps the best 2; simulation re-ranks those survivors.
      REQUIRE(staged.size() == 2);
      REQUIRE(staged[0].latency > 0);
      REQUIRE(staged[0].latency <= staged[1].latency);
    }
  }
}

TEST_CASE("Pipeline: fast-reject predicate drops configs before scoring", "[origami][pipeline]") {
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - primitive reject rule") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();  // MT_M in {256, 128, 64}

      origami::prune_policy_t no_prune;
      origami::ranking_phase_t phase;
      phase.model    = origami::model_t::gemm;
      phase.target   = origami::target_t::tensilelite;
      phase.fidelity = origami::prediction_modes_t::estimation;
      phase.prune    = no_prune;
      // Cheap primitive rule: drop any tile whose MT_M is 256 or larger.
      phase.reject = [](const origami::problem_t&, const origami::hardware_t&,
                        const origami::config_t& c) { return c.mt.m >= 256; };

      origami::ranking_pipeline_t pipeline;
      pipeline.phases.push_back(phase);

      auto staged =
          origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);

      // The 256-row tile is rejected before scoring; the other two survive.
      REQUIRE(staged.size() == 2);
      for (const auto& result : staged) { REQUIRE(result.config.mt.m < 256); }
    }
  }
}

TEST_CASE("Pipeline: estimation phase refines internally, matches single-pass", "[origami][pipeline]") {
  // The estimation model walks its own detail levels internally (with per-config
  // data reuse) and returns full-detail costs. A single no-prune estimation phase
  // must therefore match plain single-pass ranking exactly.
  VarianceGuard guard;
  for (int gpu_arch : test_architectures) {
    DYNAMIC_SECTION("gfx" << gpu_arch << " - internal leveling == single pass") {
      auto hardware = make_hardware(gpu_arch);
      auto problem  = make_problem(1024, 1024, 4096);
      auto configs  = sample_configs();

      origami::prune_policy_t no_prune;
      auto pipeline = single_phase(origami::model_t::gemm,
                                   origami::target_t::tensilelite,
                                   origami::prediction_modes_t::estimation,
                                   no_prune);

      auto baseline = origami::rank_configs(problem, hardware, configs);
      auto staged =
          origami::rank_configs(problem, hardware, configs, origami::model_t::gemm, pipeline);

      REQUIRE(staged.size() == baseline.size());
      for (size_t i = 0; i < staged.size(); ++i) {
        REQUIRE(staged[i].config.mt.m == baseline[i].config.mt.m);
        REQUIRE(staged[i].config.mt.n == baseline[i].config.mt.n);
        REQUIRE(staged[i].config.mt.k == baseline[i].config.mt.k);
        REQUIRE(staged[i].latency == baseline[i].latency);
      }
    }
  }
}

// NOTE: selection-time / scoring-latency benchmarking lives in the standalone
// `origami-bench` target (tests/bench_scoring.cpp), which times both
// score_candidates and the full rank_configs workflow. It is kept out of the
// test suite so timing never runs as part of correctness CTest runs.
