// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TensileLite's re-rank, for checking that two builds rank alike: when its
// predicates reject the top-1 of CandidateSet::rank_top(k = 1), TensileLite
// asks CandidateSet::rank(min_scored = 1) for the same problem right after.
// For every shape of bench_selection_time's inputs (after prepare() once and
// prepare(problem, hardware)), the program makes both calls and prints the
// hashes of their results over the shapes in bench_shipped_rank --hash's form
// (TOPK1 equals bench_shipped_rank --hash --top-k 1 --prepare, RANK equals
// bench_shipped_rank --hash --prepare), then the median wall-clock of the
// second call over the shapes, in microseconds. A build with the re-rank memo
// (TW_LEVER_MEMO) then prints how many second calls the memo answered, from
// the engine's own count.
//
// usage: rerank_check MODEL INPUTS

#include "bench_inputs.hpp"
#include "model_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s MODEL INPUTS\n", argv[0]);
    return 2;
  }
  bench_inputs::Inputs ix;
  if (!bench_inputs::load_inputs(argv[2], &ix)) return 1;
  tilewright::PoolOptions options;
  const tilewright::ModelPtr model = bench_inputs::load(argv[1], ix, ix.configs, &options);
  if (!model) return 1;
  const tilewright::CandidateSet set(model, ix.configs, options);
  set.prepare();
  std::uint64_t top_hash = bench_inputs::kFnvBasis, all_hash = bench_inputs::kFnvBasis;
  std::vector<tilewright::Result> top;
  std::vector<double> second;
#if TW_LEVER_MEMO
  std::uint64_t replays = 0;
#endif
  for (const bench_inputs::Shape& sh : ix.shapes) {
    const tilewright::Problem p          = bench_inputs::problem_of(ix, sh);
    const tilewright::ExecutionContext c = bench_inputs::context_of(sh);
    set.prepare(p, ix.hardware);
    set.rank_top(p, ix.hardware, c, 1, top);
#if TW_LEVER_MEMO
    const std::uint64_t before = tilewright::detail::memo_counts().replays;
#endif
    const auto t0                             = std::chrono::steady_clock::now();
    const std::vector<tilewright::Result> all = set.rank(p, ix.hardware, c, 1, nullptr);
    const auto t1                             = std::chrono::steady_clock::now();
    second.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
#if TW_LEVER_MEMO
    replays += tilewright::detail::memo_counts().replays - before;
#endif
    bench_inputs::hash_results(top, &top_hash);
    bench_inputs::hash_results(all, &all_hash);
  }
  if (second.empty()) {
    std::fprintf(stderr, "no shapes\n");
    return 1;
  }
  std::sort(second.begin(), second.end());
  std::printf("# %zu shapes, %zu kernels, launch '%s'\n",
              ix.shapes.size(),
              ix.configs.size(),
              options.launch ? options.launch->id().c_str() : "");
  std::printf("TOPK1 %016llx\nRANK %016llx\nSECOND_P50_US %.3f\n",
              static_cast<unsigned long long>(top_hash),
              static_cast<unsigned long long>(all_hash),
              second[second.size() / 2]);
#if TW_LEVER_MEMO
  std::printf(
      "# memo replays %llu of %zu\n", static_cast<unsigned long long>(replays), second.size());
#endif
  return 0;
}
