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

// Latency-engine byte-identity regression.
//
// Two corpora:
//   golden/iter_times.csv       (630 rows) -- compute_iter_times outputs
//                                             across 7 work patterns x 5
//                                             num_wgs x 6 primitives x 3
//                                             bw_per_wg regimes.
//   golden/wg_tile_latency.csv  (960 rows) -- compute_wg_tile_latency
//                                             outputs across 8 work graphs
//                                             x 6 sizes x 4 num_wgs
//                                             x 5 primitives.
//
// Tolerance: 1e-12 (absolute), the typical FP-reorder noise floor for
// these expressions. The golden values are stored at full precision so
// the gold IS the exact bit pattern the reference produces.
#include "test_harness.hpp"
#include "test_system.hpp"

#include "origami/comm/hardware.hpp"
#include "origami/comm/heuristics.hpp"
#include "origami/comm/latency.hpp"
#include "origami/comm/primitives.hpp"
#include "origami/comm/types.hpp"

#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace origami::comm;

namespace {

constexpr double kAbsTol = 1e-12;

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

// Work-graph factory keyed by string (matches the golden corpus' work graphs).
std::vector<op_t> make_graph(const std::string& name) {
  if (name == "ag_step") return {load_t{}, store_t{}, push_t{/*peer=*/1}};
  if (name == "rs_step") return {load_t{}, reduce_t{}, store_t{}, push_t{/*peer=*/1}};
  if (name == "bc_step") return {load_t{}, store_t{}, push_t{/*peer=*/1}};
  if (name == "ar_ring_rs")
    return {load_t{},
            wait_t{/*peer=*/0},
            pull_t{/*peer=*/0},
            reduce_t{},
            store_t{},
            signal_t{/*peer=*/1}};
  if (name == "ar_ring_ag")
    return {wait_t{/*peer=*/0}, pull_t{/*peer=*/0}, store_t{}, signal_t{/*peer=*/1}};
  if (name == "ar_one_shot_remote") return {pull_t{/*peer=*/1}, reduce_t{}};
  if (name == "ar_one_shot_self") return {load_t{}, reduce_t{}};
  if (name == "ar_two_shot_bcast") return {load_t{}, push_t{/*peer=*/1}};
  return {};
}

// functional_unit_work_t factory (must match the work kinds in the golden corpus).
functional_unit_work_t make_work(const std::string& kind) {
  functional_unit_work_t w{};
  auto fill_full_read = [&] {
    w.vmem_read_instrs = 8;
    w.tcp_read_cl      = 8;
    w.l2_read_cl       = 8;
    w.mall_read_cl     = 8;
    w.hbm_read_cl      = 8;
  };
  auto fill_full_write = [&] {
    w.vmem_write_instrs = 8;
    w.tcp_write_cl      = 8;
    w.l2_write_cl       = 8;
    w.mall_write_cl     = 8;
    w.hbm_write_cl      = 8;
  };
  if (kind == "load") {
    fill_full_read();
  } else if (kind == "store") {
    fill_full_write();
  } else if (kind == "load_store") {
    fill_full_read();
    fill_full_write();
  } else if (kind == "pull") {
    w.vmem_read_instrs = 8;
    w.tcp_read_cl      = 8;
    w.l2_read_cl       = 8;
    w.xgmi_read_cl     = 8;
  } else if (kind == "push") {
    fill_full_read();
    w.xgmi_write_cl = 8;
  } else if (kind == "reduce") {
    w.valu_ops = 256;
  } else if (kind == "ring_ag_step") {
    fill_full_read();
    fill_full_write();
    w.xgmi_write_cl = 8;
  }
  return w;
}

}  // namespace

// ─── iter_times grid ────────────────────────────────────────────
TEST(iter_times_match_golden_grid) {
  std::ifstream in{"golden/iter_times.csv"};
  if (!in) in.open("../../tests/golden/iter_times.csv");
  CHECK(in.is_open());
  if (!in.is_open()) return;

  std::string line;
  std::getline(in, line);  // header

  std::size_t rows = 0, mismatches = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() < 14) continue;

    const std::string kind = f[0];
    const int num_wgs      = std::stoi(f[1]);
    const std::string prim = f[2];
    const int active       = std::stoi(f[3]);
    const double bw        = std::stod(f[4]);

    const auto work = make_work(kind);
    const std::optional<primitive_t> prim_enum =
        prim.empty() ? std::nullopt : std::optional<primitive_t>{primitive_from_name(prim)};
    const auto times =
        compute_iter_times(work, MI300X_SYSTEM, bw, active, DEFAULT_HEURISTICS, prim_enum);

    struct {
      const char* name;
      double got;
      double exp;
    } cells[] = {
        {"vmem", times.vmem, std::stod(f[5])},
        {"tcp", times.tcp, std::stod(f[6])},
        {"l2", times.l2, std::stod(f[7])},
        {"mall", times.mall, std::stod(f[8])},
        {"hbm_read", times.hbm_read, std::stod(f[9])},
        {"hbm_write", times.hbm_write, std::stod(f[10])},
        {"xgmi_read", times.xgmi_read, std::stod(f[11])},
        {"xgmi_write", times.xgmi_write, std::stod(f[12])},
        {"valu", times.valu, std::stod(f[13])},
    };
    ++rows;
    bool row_mismatch = false;
    for (const auto& c : cells) {
      if (std::fabs(c.got - c.exp) > kAbsTol) {
        if (mismatches < 5) {
          std::fprintf(stderr,
                       "  MISMATCH iter_times kind=%s nw=%d prim=%s bw=%.6g  "
                       "field=%s: got=%.17g exp=%.17g (delta=%g)\n",
                       kind.c_str(),
                       num_wgs,
                       prim.c_str(),
                       bw,
                       c.name,
                       c.got,
                       c.exp,
                       c.got - c.exp);
        }
        row_mismatch = true;
      }
    }
    if (row_mismatch) ++mismatches;
  }
  std::printf("  iter_times: %zu rows, %zu mismatches\n", rows, mismatches);
  CHECK(rows > 600);
  CHECK(mismatches == 0);
}

// ─── wg_tile_latency grid ───────────────────────────────────────
TEST(wg_tile_latency_match_golden_grid) {
  std::ifstream in{"golden/wg_tile_latency.csv"};
  if (!in) in.open("../../tests/golden/wg_tile_latency.csv");
  CHECK(in.is_open());
  if (!in.is_open()) return;

  std::string line;
  std::getline(in, line);  // header

  std::size_t rows = 0, mismatches = 0;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() < 23) continue;

    const std::string graph = f[0];
    const std::size_t wg_cl = std::stoull(f[1]);
    const std::size_t wg_el = std::stoull(f[2]);
    const int num_wgs       = std::stoi(f[3]);
    const int active        = std::stoi(f[4]);
    const double bw         = std::stod(f[5]);
    const std::string prim  = f[6];
    const std::optional<primitive_t> prim_enum =
        prim.empty() ? std::nullopt : std::optional<primitive_t>{primitive_from_name(prim)};

    const auto ops = make_graph(graph);

    comm_config_t cfg{};
    cfg.num_wgs          = num_wgs;
    cfg.load_width       = load_width_t::DWORDX16;
    cfg.vgprs_for_data   = 128;
    cfg.min_bytes_per_wg = 16384;

    const auto out = compute_wg_tile_latency(
        ops,
        wg_tile_geometry_t{wg_cl, wg_el, std::nullopt},
        bw,
        /*active_cus=*/active,
        latency_context_t{cfg, MI300X_SYSTEM, DEFAULT_HEURISTICS, prim_enum});

    struct {
      const char* name;
      double got;
      double exp;
    } cells[] = {
        {"T_total", out.T_total_cycles, std::stod(f[7])},
        {"T_wlt", out.T_wlt_cycles, std::stod(f[8])},
        {"T_prologue", out.T_prologue_cycles, std::stod(f[9])},
        {"T_epilogue", out.T_epilogue_cycles, std::stod(f[10])},
        {"T_sync", out.T_sync_cycles, std::stod(f[11])},
        {"T_vmem", out.T_vmem_cycles, std::stod(f[13])},
        {"T_tcp", out.T_tcp_cycles, std::stod(f[14])},
        {"T_l2", out.T_l2_cycles, std::stod(f[15])},
        {"T_mall", out.T_mall_cycles, std::stod(f[16])},
        {"T_hbm_read", out.T_hbm_read_cycles, std::stod(f[17])},
        {"T_hbm_write", out.T_hbm_write_cycles, std::stod(f[18])},
        {"T_xgmi_read", out.T_xgmi_read_cycles, std::stod(f[19])},
        {"T_xgmi_write", out.T_xgmi_write_cycles, std::stod(f[20])},
        {"T_valu", out.T_valu_cycles, std::stod(f[21])},
    };
    const std::size_t exp_num_iters  = std::stoull(f[12]);
    const std::string exp_bottleneck = f[22];

    ++rows;
    bool row_mismatch = false;
    for (const auto& c : cells) {
      if (std::fabs(c.got - c.exp) > kAbsTol) {
        if (mismatches < 5) {
          std::fprintf(stderr,
                       "  MISMATCH wg_tile_latency graph=%s wg_cl=%zu nw=%d prim=%s  "
                       "field=%s: got=%.17g exp=%.17g (delta=%g)\n",
                       graph.c_str(),
                       wg_cl,
                       num_wgs,
                       prim.c_str(),
                       c.name,
                       c.got,
                       c.exp,
                       c.got - c.exp);
        }
        row_mismatch = true;
      }
    }
    if (out.num_iters != exp_num_iters) {
      if (mismatches < 5) {
        std::fprintf(stderr,
                     "  MISMATCH wg_tile_latency graph=%s wg_cl=%zu nw=%d prim=%s  "
                     "num_iters: got=%zu exp=%zu\n",
                     graph.c_str(),
                     wg_cl,
                     num_wgs,
                     prim.c_str(),
                     out.num_iters,
                     exp_num_iters);
      }
      row_mismatch = true;
    }
    if (out.bottleneck != exp_bottleneck) {
      if (mismatches < 5) {
        std::fprintf(stderr,
                     "  MISMATCH wg_tile_latency graph=%s wg_cl=%zu nw=%d prim=%s  "
                     "bottleneck: got=%s exp=%s\n",
                     graph.c_str(),
                     wg_cl,
                     num_wgs,
                     prim.c_str(),
                     out.bottleneck.c_str(),
                     exp_bottleneck.c_str());
      }
      row_mismatch = true;
    }
    if (row_mismatch) ++mismatches;
  }
  std::printf("  wg_tile_latency: %zu rows, %zu mismatches\n", rows, mismatches);
  CHECK(rows > 900);
  CHECK(mismatches == 0);
}

// ─── Smoke: iter_times structural identity ──────────────────────
TEST(iter_times_load_path_full_levels) {
  const auto times =
      compute_iter_times(make_work("load"), MI300X_SYSTEM, MI300X_COMM.link_bw, /*active_cus=*/16);
  // Read path is populated, write path is zero.
  CHECK(times.vmem > 0.0);
  CHECK(times.tcp > 0.0);
  CHECK(times.l2 > 0.0);
  CHECK(times.mall > 0.0);
  CHECK(times.hbm_read > 0.0);
  CHECK(times.hbm_write == 0.0);
  CHECK(times.xgmi_read == 0.0);
  CHECK(times.xgmi_write == 0.0);
}

ORIGAMI_TEST_MAIN()
