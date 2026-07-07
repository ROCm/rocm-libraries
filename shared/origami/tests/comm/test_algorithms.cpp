/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
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

// Algorithm byte-identity regression. The harness loads
// golden/algorithms_grid.csv (the frozen reference oracle) and verifies every
// schedule_entry_t across 5,936 (algorithm, num_gpus, pid, timestep, my_rank)
// tuples matches the C++ algorithm's link_of() exactly.
//
// CSV columns:
//   algorithm,num_gpus,pid,timestep,my_rank,link_id,peer_rank,direction,
//   is_self,wg_sig
//
// `wg_sig` is a compact opcode chain like "L|X3|R" stored in the golden
// CSV; we recompute the same signature from the C++
// schedule_entry_t::work_graph and string-compare.
#include "test_harness.hpp"

#include "origami/comm/algorithms.hpp"
#include "origami/comm/primitives.hpp"

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace origami::comm;

namespace {

// ─── op_sig: encodes a work-graph op as a compact token ─────────
std::string op_sig(const op_t& op) {
  return std::visit(
      [](const auto& concrete) -> std::string {
        using T = std::decay_t<decltype(concrete)>;
        if constexpr (std::is_same_v<T, load_t>) return "L";
        if constexpr (std::is_same_v<T, store_t>) return concrete.write_through ? "Sw" : "S";
        if constexpr (std::is_same_v<T, reduce_t>) return "R";
        if constexpr (std::is_same_v<T, pull_t>) return "P" + std::to_string(concrete.peer);
        if constexpr (std::is_same_v<T, push_t>) return "X" + std::to_string(concrete.peer);
        if constexpr (std::is_same_v<T, signal_t>) return "G" + std::to_string(concrete.peer);
        if constexpr (std::is_same_v<T, wait_t>) return "W" + std::to_string(concrete.peer);
        return "?";
      },
      op);
}

std::string wg_sig(const std::vector<op_t>& ops) {
  std::string out;
  for (std::size_t i = 0; i < ops.size(); ++i) {
    if (i > 0) out += '|';
    out += op_sig(ops[i]);
  }
  return out;
}

std::string direction_str(direction_t d) { return d == direction_t::PUSH ? "push" : "pull"; }

// ─── Algorithm factory ──────────────────────────────────────────
std::unique_ptr<collective_algorithm_t> make_algorithm(const std::string& name, int N) {
  if (name == "RingAllGather") return allgather_algorithm(N);
  if (name == "RingReduceScatter") return reduce_scatter_algorithm(N);
  if (name == "RingBroadcast") return broadcast_algorithm(N);
  if (name == "AROneShot") return allreduce_one_shot_algorithm(N);
  if (name == "ARTwoShot") return allreduce_two_shot_algorithm(N);
  if (name == "ARRing") return allreduce_ring_algorithm(N);
  if (name == "AllToAll") return alltoall_algorithm(N);
  if (name == "PidPartitioned") return std::make_unique<pid_partitioned_algorithm_t>(N);
  if (name == "RingFixed") return std::make_unique<ring_fixed_algorithm_t>(N);
  return nullptr;
}

// Cache algorithms by (name, num_gpus) so we don't rebuild 5k times.
const collective_algorithm_t& cached_algorithm(const std::string& name, int N) {
  static std::unordered_map<std::string, std::unique_ptr<collective_algorithm_t>> cache;
  const std::string key = name + ":" + std::to_string(N);
  auto it               = cache.find(key);
  if (it == cache.end()) { it = cache.emplace(key, make_algorithm(name, N)).first; }
  return *it->second;
}

// ─── tiny CSV splitter ──────────────────────────────────────────
std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

}  // namespace

// ─── Test: load CSV and compare every row ───────────────────────
TEST(algorithms_match_golden_grid) {
  const char* path = "golden/algorithms_grid.csv";  // run from build/tests/
  std::ifstream in{path};
  if (!in) {
    // CI invocations sometimes set cwd elsewhere; try the source-tree path.
    in.open("../../tests/golden/algorithms_grid.csv");
  }
  CHECK(in.is_open());
  if (!in.is_open()) return;

  std::string line;
  std::getline(in, line);  // header

  std::size_t rows = 0, mismatches = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() < 10) continue;
    const std::string algorithm = f[0];
    const int num_gpus          = std::stoi(f[1]);
    const int pid               = std::stoi(f[2]);
    const int timestep          = std::stoi(f[3]);
    const int my_rank           = std::stoi(f[4]);
    const int exp_link          = std::stoi(f[5]);
    const int exp_peer          = std::stoi(f[6]);
    const std::string exp_dir   = f[7];
    const bool exp_self         = (std::stoi(f[8]) != 0);
    const std::string exp_sig   = f[9];

    const auto& L = cached_algorithm(algorithm, num_gpus);
    const auto se = L.link_of(pid, timestep, my_rank);

    ++rows;
    const std::string got_sig = wg_sig(se.work_graph);
    if (se.link_id != exp_link || se.peer_rank != exp_peer ||
        direction_str(se.direction) != exp_dir || se.is_self != exp_self || got_sig != exp_sig) {
      if (mismatches < 10) {
        std::fprintf(stderr,
                     "  MISMATCH %s N=%d pid=%d ts=%d r=%d:\n"
                     "    expected: link=%d peer=%d dir=%s self=%d wg=%s\n"
                     "    got     : link=%d peer=%d dir=%s self=%d wg=%s\n",
                     algorithm.c_str(),
                     num_gpus,
                     pid,
                     timestep,
                     my_rank,
                     exp_link,
                     exp_peer,
                     exp_dir.c_str(),
                     int(exp_self),
                     exp_sig.c_str(),
                     se.link_id,
                     se.peer_rank,
                     direction_str(se.direction).c_str(),
                     int(se.is_self),
                     got_sig.c_str());
      }
      ++mismatches;
    }
  }
  std::printf("  algorithms_grid: %zu rows, %zu mismatches\n", rows, mismatches);
  CHECK(rows > 5000);
  CHECK(mismatches == 0);
}

// ─── Spot-checks on chunks_per_timestep + num_timesteps ─────────
TEST(algorithm_num_timesteps_matches_reference) {
  // RingAllGather: N-1
  CHECK(allgather_algorithm(8)->num_timesteps() == 7);
  CHECK(reduce_scatter_algorithm(8)->num_timesteps() == 7);
  CHECK(broadcast_algorithm(8)->num_timesteps() == 7);
  // OneShot AR: N-1 (all_to_same_algorithm_t skips self)
  CHECK(allreduce_one_shot_algorithm(8)->num_timesteps() == 7);
  // TwoShot AR: 2N-1
  CHECK(allreduce_two_shot_algorithm(8)->num_timesteps() == 15);
  // Ring AR: 2(N-1)
  CHECK(allreduce_ring_algorithm(8)->num_timesteps() == 14);
  // AllToAll (PidStaggered): N timesteps
  CHECK(alltoall_algorithm(8)->num_timesteps() == 8);
  // PidPartitioned: 1
  pid_partitioned_algorithm_t pp{8};
  CHECK(pp.num_timesteps() == 1);
}

TEST(algorithm_chunks_per_timestep_matches_reference) {
  CHECK(allgather_algorithm(8)->chunks_per_timestep() == 1);
  CHECK(reduce_scatter_algorithm(8)->chunks_per_timestep() == 1);
  CHECK(broadcast_algorithm(8)->chunks_per_timestep() == 8);
  CHECK(allreduce_two_shot_algorithm(8)->chunks_per_timestep() == 8);
  CHECK(allreduce_ring_algorithm(8)->chunks_per_timestep() == 8);
  CHECK(alltoall_algorithm(8)->chunks_per_timestep() == 8);
  CHECK(allreduce_one_shot_algorithm(8)->chunks_per_timestep() == 1);  // AllToSame default
  pid_partitioned_algorithm_t pp{8};
  CHECK(pp.chunks_per_timestep() == 1);
}

// ─── wgs_per_active_link conservation: sum equals num_wgs for rings ────
TEST(ring_wgs_per_active_link_conserves_num_wgs) {
  for (int N : {2, 4, 8}) {
    for (int nch : {1, 2, 3, 7, 8, 16, 32}) {
      auto L  = allgather_algorithm(N);
      auto al = L->wgs_per_active_link(/*timestep=*/0, /*num_wgs=*/nch);
      int sum = 0;
      for (int v : al) sum += v;
      CHECK(sum == nch);
    }
  }
}

// ─── resolve_algorithm: (collective, algorithm) validity ───────────
TEST(resolve_algorithm_selects_all_reduce_variants) {
  // all_reduce is the only collective with a real menu. automatic == two_shot.
  CHECK(resolve_algorithm(primitive_t::all_reduce, algorithm_t::automatic, 8)->num_timesteps() ==
        15);  // 2N-1
  CHECK(resolve_algorithm(primitive_t::all_reduce, algorithm_t::two_shot, 8)->num_timesteps() ==
        15);
  CHECK(resolve_algorithm(primitive_t::all_reduce, algorithm_t::one_shot, 8)->num_timesteps() ==
        7);  // N-1
  CHECK(resolve_algorithm(primitive_t::all_reduce, algorithm_t::ring, 8)->num_timesteps() ==
        14);  // 2(N-1)
}

TEST(resolve_algorithm_accepts_canonical_and_explicit_names) {
  // Single-algorithm collectives accept automatic and their one explicit name.
  CHECK(resolve_algorithm(primitive_t::all_gather, algorithm_t::automatic, 8) != nullptr);
  CHECK(resolve_algorithm(primitive_t::all_gather, algorithm_t::ring, 8) != nullptr);
  CHECK(resolve_algorithm(primitive_t::all_to_all, algorithm_t::automatic, 8) != nullptr);
  CHECK(resolve_algorithm(primitive_t::all_to_all, algorithm_t::direct, 8) != nullptr);
}

TEST(resolve_algorithm_rejects_invalid_pairs) {
  // An algorithm not defined for the collective is rejected, not silently
  // mispriced — this is the structural validity rule.
  CHECK_THROWS_AS(resolve_algorithm(primitive_t::all_gather, algorithm_t::two_shot, 8),
                  std::invalid_argument);
  CHECK_THROWS_AS(resolve_algorithm(primitive_t::all_to_all, algorithm_t::ring, 8),
                  std::invalid_argument);
  CHECK_THROWS_AS(resolve_algorithm(primitive_t::broadcast, algorithm_t::one_shot, 8),
                  std::invalid_argument);
}

// ─── floor_mod sanity ──────────────────────────────────────────────
TEST(floor_mod_handles_negatives) {
  CHECK(floor_mod(0, 8) == 0);
  CHECK(floor_mod(-1, 8) == 7);
  CHECK(floor_mod(-9, 8) == 7);
  CHECK(floor_mod(7, 8) == 7);
  CHECK(floor_mod(8, 8) == 0);
}

ORIGAMI_TEST_MAIN()
