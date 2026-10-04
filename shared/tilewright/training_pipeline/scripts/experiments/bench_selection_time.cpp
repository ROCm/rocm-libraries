// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU wall-clock of tilewright's ranking call over one library's kernel pool.
// The inputs file comes from bench_selection_time.py. For every shape the
// program reports the median of --iters calls (after --warmup calls) of
// CandidateSet::rank and of the uncached rank_configs, then a summary over the
// shapes.

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Shape {
  std::size_t m, n, k, b;
};

struct Inputs {
  tilewright::Problem problem;
  tilewright::Hardware hardware;
  std::vector<Shape> shapes;
  std::vector<tilewright::Config> configs;
};

bool next_line(std::ifstream& in, std::istringstream& out) {
  std::string line;
  while (std::getline(in, line)) {
    const auto first = line.find_first_not_of(" \t\r");
    if (first == std::string::npos || line[first] == '#') continue;
    out.clear();
    out.str(line);
    return true;
  }
  return false;
}

bool expect(std::istringstream& s, const char* tag) {
  std::string got;
  s >> got;
  if (got == tag) return true;
  std::fprintf(stderr, "inputs: expected %s, got '%s'\n", tag, got.c_str());
  return false;
}

bool load_inputs(const std::string& path, Inputs* ix) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  std::istringstream s;
  int a, b, c, d, mi, ta, tb;
  if (!next_line(in, s) || !expect(s, "PROBLEM") || !(s >> a >> b >> c >> d >> mi >> ta >> tb))
    return false;
  ix->problem.a_dtype     = static_cast<tilewright::DataType>(a);
  ix->problem.b_dtype     = static_cast<tilewright::DataType>(b);
  ix->problem.c_dtype     = static_cast<tilewright::DataType>(c);
  ix->problem.d_dtype     = static_cast<tilewright::DataType>(d);
  ix->problem.mi_dtype    = static_cast<tilewright::DataType>(mi);
  ix->problem.a_transpose = static_cast<tilewright::Transpose>(ta);
  ix->problem.b_transpose = static_cast<tilewright::Transpose>(tb);
  if (!next_line(in, s) || !expect(s, "HARDWARE") ||
      !(s >> ix->hardware.N_CU >> ix->hardware.lds_capacity >> ix->hardware.L2_capacity))
    return false;
  std::size_t n = 0;
  if (!next_line(in, s) || !expect(s, "SHAPES") || !(s >> n)) return false;
  for (std::size_t i = 0; i < n; ++i) {
    Shape sh{};
    if (!next_line(in, s) || !(s >> sh.m >> sh.n >> sh.k >> sh.b)) return false;
    ix->shapes.push_back(sh);
  }
  if (!next_line(in, s) || !expect(s, "CONFIGS") || !(s >> n)) return false;
  for (std::size_t i = 0; i < n; ++i) {
    tilewright::Config cfg;
    if (!next_line(in, s) || !(s >> cfg.mt.m >> cfg.mt.n >> cfg.mt.k >> cfg.mi.m >> cfg.mi.n >>
                               cfg.mi.k >> cfg.occupancy >> cfg.cache_hints_a >>
                               cfg.cache_hints_b >> cfg.grvw_a >> cfg.grvw_b >> cfg.gwvw_d))
      return false;
    cfg.index = i;
    ix->configs.push_back(cfg);
  }
  return true;
}

template <typename F>
double median_us(std::size_t warmup, std::size_t iters, F&& call) {
  for (std::size_t i = 0; i < warmup; ++i) call();
  std::vector<double> us;
  us.reserve(iters);
  for (std::size_t i = 0; i < iters; ++i) {
    const auto t0     = std::chrono::steady_clock::now();
    const auto result = call();
    const auto t1     = std::chrono::steady_clock::now();
    asm volatile("" : : "g"(result.data()) : "memory");
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.size() == 1) return sorted.front();
  const double pos      = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lo  = static_cast<std::size_t>(pos);
  const std::size_t hi  = std::min(lo + 1, sorted.size() - 1);
  const double fraction = pos - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}

void print_row(const char* name, std::vector<double> v) {
  if (v.empty()) return;
  std::sort(v.begin(), v.end());
  double sum = 0;
  for (double x : v) sum += x;
  std::printf("%-14s %6zu %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
              name,
              v.size(),
              v.front(),
              percentile(v, 0.5),
              sum / static_cast<double>(v.size()),
              percentile(v, 0.9),
              percentile(v, 0.99),
              v.back());
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model FILE --inputs FILE [--iters 200] [--warmup 50] "
               "[--min-scored 1]\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, inputs_path;
  std::size_t iters = 200, warmup = 50, min_scored = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (i + 1 >= argc) {
      usage(argv[0]);
      return 2;
    }
    const std::string v = argv[++i];
    if (a == "--model")
      model_path = v;
    else if (a == "--inputs")
      inputs_path = v;
    else if (a == "--iters")
      iters = std::stoul(v);
    else if (a == "--warmup")
      warmup = std::stoul(v);
    else if (a == "--min-scored")
      min_scored = std::stoul(v);
    else {
      usage(argv[0]);
      return 2;
    }
  }
  if (model_path.empty() || inputs_path.empty() || iters == 0) {
    usage(argv[0]);
    return 2;
  }
  std::string error;
  const tilewright::ModelPtr model = tilewright::load_model(model_path, &error);
  if (!model) {
    std::fprintf(stderr, "cannot load %s: %s\n", model_path.c_str(), error.c_str());
    return 1;
  }
  Inputs ix;
  if (!load_inputs(inputs_path, &ix)) return 1;

  const tilewright::CandidateSet set(model, ix.configs);
  std::vector<double> cached, uncached;
  std::size_t unscored = 0;
  for (const Shape& sh : ix.shapes) {
    tilewright::Problem p = ix.problem;
    p.size                = tilewright::Dim3{sh.m, sh.n, sh.k};
    p.batch               = sh.b;
    const auto first      = set.rank(p, ix.hardware, min_scored);
    if (first.empty() || !first.front().scored) ++unscored;
    cached.push_back(
        median_us(warmup, iters, [&] { return set.rank(p, ix.hardware, min_scored); }));
    uncached.push_back(median_us(warmup, iters, [&] {
      return tilewright::rank_configs(*model, p, ix.hardware, ix.configs, min_scored);
    }));
  }
  std::printf("# %zu shapes, %zu kernels, min_scored %zu, median of %zu calls after %zu\n",
              ix.shapes.size(),
              ix.configs.size(),
              min_scored,
              iters,
              warmup);
  std::printf("# microseconds per call; %zu shapes had no scored kernel\n", unscored);
  std::printf("%-14s %6s %10s %10s %10s %10s %10s %10s\n",
              "path",
              "shapes",
              "min",
              "median",
              "mean",
              "p90",
              "p99",
              "max");
  print_row("candidate_set", cached);
  print_row("rank_configs", uncached);
  return 0;
}
