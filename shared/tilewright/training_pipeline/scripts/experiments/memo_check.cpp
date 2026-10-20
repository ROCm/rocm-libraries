// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Pairs of rankings that a re-rank memo (TW_LEVER_MEMO) may or must not
// answer from the first one's scores, for checking that a build with the memo
// ranks as one without it. For every shape of bench_selection_time's inputs
// the program runs each scenario below on the same problem unless noted and
// hashes its second call's results (bench_shipped_rank --hash's form) and its
// first call's; equal hashes in two builds mean equal results. It also prints
// the median wall-clock of each scenario's second call over the shapes, in
// microseconds, which shows whether a memo answered it.
//
//   S01  rank_top(1) alone             S02  rank_top(1); rank_top(1)
//   S03  rank_top(1); rank_top(4)      S04  rank_top(1); rank_top(32)
//   S05  rank_top(1); rank_top(33)     S06  rank_top(1); rank(min_scored 1)
//   S07  rank_top(1); rank(min_scored 0)
//   S08  rank_top(1); rank(min_scored 1) of the next shape
//   S09  rank_top(1); rank_top(1) under the other CU budget (all CUs or half)
//   S10  rank_top(1); rank(min_scored 1) of a second pool: the same kernels
//        rotated by one position
//   S11  rank(min_scored 1); rank(min_scored 1)
//   S12  rank_top(1); prepare(problem); rank(min_scored 1)
//   S13  rank_top(1); rank_top(2); rank_top(2), the third call hashed as the
//        second (S13b hashes the second)
//
// Before each scenario a ranking of another problem (K + 1) runs, so that no
// scenario's first call follows a ranking of its own problem.
//
// usage: memo_check MODEL INPUTS

#include "bench_inputs.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

namespace {

constexpr int kScenarios = 13;

double us_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL INPUTS\n", argv[0]);
    return 2;
  }
  bench_inputs::Inputs ix;
  if (!bench_inputs::load_inputs(argv[2], &ix)) return 1;
  if (ix.shapes.empty() || ix.configs.empty()) {
    std::fprintf(stderr, "no shapes or no kernels\n");
    return 1;
  }
  std::vector<tilewright::Config> rotated(ix.configs.begin() + 1, ix.configs.end());
  rotated.push_back(ix.configs.front());
  for (std::size_t i = 0; i < rotated.size(); ++i) rotated[i].index = i;
  tilewright::PoolOptions options, rotated_options;
  const tilewright::ModelPtr model = bench_inputs::load(argv[1], ix, ix.configs, &options);
  if (!model || !bench_inputs::load(argv[1], ix, rotated, &rotated_options)) return 1;
  const tilewright::CandidateSet set(model, ix.configs, options);
  const tilewright::CandidateSet other_pool(model, rotated, rotated_options);
  set.prepare();
  other_pool.prepare();

  const tilewright::Hardware& hw = ix.hardware;
  std::uint64_t second_hash[kScenarios + 1], first_hash[kScenarios + 1];
  std::fill(second_hash, second_hash + kScenarios + 1, bench_inputs::kFnvBasis);
  std::fill(first_hash, first_hash + kScenarios + 1, bench_inputs::kFnvBasis);
  std::uint64_t s13b_hash = bench_inputs::kFnvBasis;
  std::vector<double> second_us[kScenarios + 1];
  std::vector<tilewright::Result> first, second;
  for (std::size_t si = 0; si < ix.shapes.size(); ++si) {
    const bench_inputs::Shape& sh             = ix.shapes[si];
    const bench_inputs::Shape& nx             = ix.shapes[(si + 1) % ix.shapes.size()];
    const tilewright::Problem p               = bench_inputs::problem_of(ix, sh);
    const tilewright::Problem next            = bench_inputs::problem_of(ix, nx);
    const tilewright::ExecutionContext c      = bench_inputs::context_of(sh);
    const tilewright::ExecutionContext next_c = bench_inputs::context_of(nx);
    tilewright::ExecutionContext other_c;
    other_c.cu_budget            = sh.cu_budget == 0 ? hw.N_CU / 2 : 0;
    tilewright::Problem distinct = p;
    distinct.size.k += 1;
    set.prepare(p, hw);
    set.prepare(next, hw);
    other_pool.prepare(p, hw);
    const auto clear = [&] { set.rank_top(distinct, hw, c, 1, first); };
    const auto timed = [&](int sc, auto&& call) {
      const auto t0 = std::chrono::steady_clock::now();
      call();
      second_us[sc].push_back(us_since(t0));
      bench_inputs::hash_results(second, &second_hash[sc]);
    };
    const auto top_then = [&](int sc, auto&& call) {
      clear();
      set.rank_top(p, hw, c, 1, first);
      bench_inputs::hash_results(first, &first_hash[sc]);
      timed(sc, call);
    };

    clear();
    set.rank_top(p, hw, c, 1, second);
    bench_inputs::hash_results(second, &second_hash[1]);
    bench_inputs::hash_results(second, &first_hash[1]);
    top_then(2, [&] { set.rank_top(p, hw, c, 1, second); });
    top_then(3, [&] { set.rank_top(p, hw, c, 4, second); });
    top_then(4, [&] { set.rank_top(p, hw, c, 32, second); });
    top_then(5, [&] { set.rank_top(p, hw, c, 33, second); });
    top_then(6, [&] { second = set.rank(p, hw, c, 1, nullptr); });
    top_then(7, [&] { second = set.rank(p, hw, c, 0, nullptr); });
    top_then(8, [&] { second = set.rank(next, hw, next_c, 1, nullptr); });
    top_then(9, [&] { set.rank_top(p, hw, other_c, 1, second); });
    top_then(10, [&] { second = other_pool.rank(p, hw, c, 1, nullptr); });
    clear();
    first = set.rank(p, hw, c, 1, nullptr);
    bench_inputs::hash_results(first, &first_hash[11]);
    timed(11, [&] { second = set.rank(p, hw, c, 1, nullptr); });
    top_then(12, [&] {
      set.prepare(p, hw);
      second = set.rank(p, hw, c, 1, nullptr);
    });
    clear();
    set.rank_top(p, hw, c, 1, first);
    bench_inputs::hash_results(first, &first_hash[13]);
    set.rank_top(p, hw, c, 2, first);
    bench_inputs::hash_results(first, &s13b_hash);
    timed(13, [&] { set.rank_top(p, hw, c, 2, second); });
  }
  std::printf("# %zu shapes, %zu kernels, launch '%s'\n",
              ix.shapes.size(),
              ix.configs.size(),
              options.launch ? options.launch->id().c_str() : "");
  for (int sc = 1; sc <= kScenarios; ++sc) {
    double p50 = 0;
    if (!second_us[sc].empty()) {
      std::vector<double> v = second_us[sc];
      std::sort(v.begin(), v.end());
      p50 = v[v.size() / 2];
    }
    std::printf("S%02d %016llx second_p50_us %.3f first %016llx\n",
                sc,
                static_cast<unsigned long long>(second_hash[sc]),
                p50,
                static_cast<unsigned long long>(first_hash[sc]));
  }
  std::printf("S13b %016llx\n", static_cast<unsigned long long>(s13b_hash));
  return 0;
}
