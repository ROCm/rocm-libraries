// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <atomic>
#include <random>
#include <thread>
#include <vector>

namespace tw = tilewright;
using namespace tilewright_test;

TEST_CASE("concurrent loads and ranks agree with a serial run", "[concurrency]") {
  TempDir dir("concurrency");
  ModelSpec spec_a = grid_model(tw::WeightType::Int4);
  for (CellSpec& c : spec_a.cells) c.signatures = {{128, 128, 64, 16, 16, 32, 0, 0}};
  const Image a     = write_model(spec_a);
  const Image b     = write_model(grid_model(tw::WeightType::Bf16));
  const auto path_a = (dir.path() / "a.bin").string();
  const auto path_b = (dir.path() / "b.bin").string();
  write_file(path_a, a.bytes);
  write_file(path_b, b.bytes);

  std::vector<tw::Config> pool;
  for (std::size_t mt : {16, 32, 64, 128, 256})
    for (std::size_t nt : {32, 64, 128}) pool.push_back(make_config(mt, nt, 64));
  std::mt19937_64 rng(5);
  std::vector<tw::Problem> problems;
  std::vector<std::size_t> depth;
  for (int i = 0; i < 48; ++i) {
    problems.push_back(
        make_problem(1 + rng() % 5000, 1 + rng() % 5000, 1 + rng() % 5000, rng() % 3 == 0 ? 4 : 1));
    depth.push_back(rng() % 2 ? 0 : 20);
  }

  const tw::ModelPtr reference = tw::load_model_from_memory(a.bytes.data(), a.bytes.size());
  REQUIRE(reference);
  std::vector<std::vector<tw::Result>> expect;
  for (std::size_t i = 0; i < problems.size(); ++i)
    expect.push_back(tw::rank_configs(*reference, problems[i], test_hardware(), pool, depth[i]));

  // Fresh model and set: dequantization and the per-cell caches are built by
  // the racing threads.
  const tw::ModelPtr held = tw::load_model(path_a);
  REQUIRE(held);
  const tw::ModelPtr shared = tw::load_model_from_memory(a.bytes.data(), a.bytes.size());
  const tw::CandidateSet set(shared, pool);

  std::atomic<int> failures{0};
  std::atomic<int> work{0};
  const unsigned n_threads = std::max(4u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      for (int iter = 0; iter < 120; ++iter) {
        const std::size_t i = (t * 7 + static_cast<std::size_t>(iter)) % problems.size();
        switch ((t + static_cast<unsigned>(iter)) % 5) {
          case 0:
            if (tw::load_model(path_a) != held) ++failures;
            break;
          case 1: {
            const tw::ModelPtr m = tw::load_model(path_b);
            if (!m || tw::describe(*m).weight_type != tw::WeightType::Bf16) ++failures;
            break;
          }
          case 2:
            if (!same_results(set.rank(problems[i], test_hardware(), depth[i]), expect[i]))
              ++failures;
            break;
          case 3:
            if (!same_results(
                    tw::rank_configs(*shared, problems[i], test_hardware(), pool, depth[i]),
                    expect[i]))
              ++failures;
            break;
          default: {
            const tw::ModelPtr m = tw::load_model_from_memory(a.bytes.data(), a.bytes.size());
            if (!m ||
                !same_results(tw::rank_configs(*m, problems[i], test_hardware(), pool, depth[i]),
                              expect[i]))
              ++failures;
            break;
          }
        }
        ++work;
      }
    });
  }
  for (std::thread& th : threads) th.join();
  CHECK(work.load() == static_cast<int>(n_threads) * 120);
  CHECK(failures.load() == 0);
}
