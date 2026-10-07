// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU wall-clock of one warm CandidateSet::rank call over a candidate pool,
// with hipBLASLt's launch model (hipblaslt_launch_rule.hpp) when the model
// records one, as TensileLite ranks. Inputs are bench_selection_time's format
// (lib/shipped_subset.py bench_inputs). For every shape the program takes the
// median of --iters calls after --warmup calls, then prints a summary over
// the shapes:
//
//   candidate_set  <shapes> <min> <p50> <mean> <p90> <p99> <max>   (us)
//
// --top-k K times rank_top into a reused buffer instead of rank; --prepare
// calls prepare() once and prepare(problem, hardware) per shape first. A shape
// line may carry a fifth field, the context's cu_budget.
//
// With a build that has TILEWRIGHT_STAGE_TIMERS, --stages also prints, per
// ranking phase, its mean nanoseconds per call over all timed calls, its share
// and its p50 / p99 over the shapes' means.
//
// --hash prints instead an FNV-1a 64 hash of every shape's ranking (config
// index, scored flag and score bits of every result, rank or rank_top as
// selected) and of each shape's alone with --hash-each, for checking that
// two builds rank alike.
//
// --dump-plans N prints the launch plan of every config on the first N shapes
// instead, under --dump-context (default: exclusive use), for --dump-arch
// (default: the model's launch model; no model file is needed with it):
// PLAN <shape> <config> <dynamic> <grid_wgs> <reduction> <launch_key>.

#include "hipblaslt_launch_rule.hpp"
#include "tilewright/model.hpp"
#include "tilewright/stage_timers.hpp"
#include "tilewright/types.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Shape {
  std::size_t m, n, k, b, cu_budget;
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

bool parse_attribute(const std::string& token, tilewright::Attribute* out) {
  const std::size_t eq = token.find('=');
  if (eq == 0 || eq == std::string::npos || eq + 1 == token.size()) return false;
  const char* value = token.c_str() + eq + 1;
  char* end         = nullptr;
  errno             = 0;
  const long long v = std::strtoll(value, &end, 10);
  if (*end != '\0' || errno == ERANGE) return false;
  out->name  = token.substr(0, eq);
  out->value = static_cast<std::int64_t>(v);
  return true;
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
    if (!(s >> sh.cu_budget)) sh.cu_budget = 0;
    ix->shapes.push_back(sh);
  }
  if (!next_line(in, s) || !expect(s, "CONFIGS") || !(s >> n)) return false;
  for (std::size_t i = 0; i < n; ++i) {
    tilewright::Config cfg;
    if (!next_line(in, s) || !(s >> cfg.mt.m >> cfg.mt.n >> cfg.mt.k >> cfg.mi.m >> cfg.mi.n >>
                               cfg.mi.k >> cfg.occupancy >> cfg.cache_hints_a >>
                               cfg.cache_hints_b >> cfg.grvw_a >> cfg.grvw_b >> cfg.gwvw_d))
      return false;
    for (std::string token; s >> token;) {
      tilewright::Attribute attribute;
      if (!parse_attribute(token, &attribute)) {
        std::fprintf(stderr, "inputs: bad attribute '%s'\n", token.c_str());
        return false;
      }
      cfg.attributes.push_back(attribute);
    }
    cfg.index = i;
    ix->configs.push_back(cfg);
  }
  return true;
}

tilewright::Problem problem_of(const Inputs& ix, const Shape& sh) {
  tilewright::Problem p = ix.problem;
  p.size                = {sh.m, sh.n, sh.k};
  p.batch               = sh.b;
  return p;
}

// "hipblaslt:<arch>:<version>" -> arch; empty when the id is not that rule's.
std::string hipblaslt_arch(const std::string& id) {
  const std::string head = "hipblaslt:";
  const std::string tail = ":" + std::to_string(hipblaslt_launch::kRuleVersion);
  if (id.size() <= head.size() + tail.size() || id.compare(0, head.size(), head) != 0 ||
      id.compare(id.size() - tail.size(), tail.size(), tail) != 0)
    return {};
  return id.substr(head.size(), id.size() - head.size() - tail.size());
}

constexpr std::uint64_t kFnvBasis = 0xcbf29ce484222325ull;

std::uint64_t fnv(std::uint64_t h, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xFF;
    h *= 0x100000001b3ull;
  }
  return h;
}

double percentile(const std::vector<double>& sorted, double q) {
  if (sorted.size() == 1) return sorted.front();
  const double pos      = q * static_cast<double>(sorted.size() - 1);
  const std::size_t lo  = static_cast<std::size_t>(pos);
  const std::size_t hi  = std::min(lo + 1, sorted.size() - 1);
  const double fraction = pos - static_cast<double>(lo);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * fraction;
}

int dump_plans(const Inputs& ix,
               std::size_t n_shapes,
               const std::string& arch,
               const tilewright::ExecutionContext& context) {
  const hipblaslt_launch::Launch rule(arch, ix.hardware.N_CU, ix.configs);
  for (std::size_t si = 0; si < std::min(n_shapes, ix.shapes.size()); ++si) {
    const auto p = problem_of(ix, ix.shapes[si]);
    for (const auto& c : ix.configs) {
      const auto plan = rule.plan(c, p, context);
      std::printf("PLAN %zu %zu %d %zu %d %llu\n",
                  si,
                  c.index,
                  plan.dynamic_queue ? 1 : 0,
                  plan.grid_wgs,
                  static_cast<int>(plan.partial_reduction),
                  static_cast<unsigned long long>(rule.launch_key(c)));
    }
  }
  return 0;
}

void usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s --model FILE --inputs FILE [--iters 200] [--warmup 50] "
               "[--min-scored 1] [--top-k K] [--prepare] [--stages] [--hash [--hash-each]] "
               "[--dump-plans N [--dump-context CU_BUDGET,SCHEDULE] [--dump-arch ARCH]]\n",
               argv0);
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_path, inputs_path;
  std::size_t iters = 200, warmup = 50, min_scored = 1, dump = 0, top_k = 0;
  bool prepare = false, stages = false, hash = false, hash_each = false;
  tilewright::ExecutionContext dump_context{};
  std::string dump_arch;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    bool* flag          = a == "--prepare"     ? &prepare
                          : a == "--stages"    ? &stages
                          : a == "--hash"      ? &hash
                          : a == "--hash-each" ? &hash_each
                                               : nullptr;
    if (flag != nullptr) {
      *flag = true;
      continue;
    }
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
    else if (a == "--top-k")
      top_k = std::stoul(v);
    else if (a == "--dump-plans")
      dump = std::stoul(v);
    else if (a == "--dump-arch")
      dump_arch = v;
    else if (a == "--dump-context") {
      const auto comma = v.find(',');
      if (comma == std::string::npos) {
        usage(argv[0]);
        return 2;
      }
      dump_context.cu_budget = std::stoul(v.substr(0, comma));
      dump_context.schedule  = static_cast<tilewright::Schedule>(std::stoi(v.substr(comma + 1)));
    } else {
      usage(argv[0]);
      return 2;
    }
  }
  if ((model_path.empty() && dump_arch.empty()) || inputs_path.empty() || iters == 0) {
    usage(argv[0]);
    return 2;
  }
  hash = hash || hash_each;
  Inputs ix;
  if (!load_inputs(inputs_path, &ix)) return 1;
  if (dump > 0 && !dump_arch.empty()) return dump_plans(ix, dump, dump_arch, dump_context);
  std::string error;
  const tilewright::ModelPtr model = tilewright::load_model(model_path, &error);
  if (!model) {
    std::fprintf(stderr, "cannot load %s: %s\n", model_path.c_str(), error.c_str());
    return 1;
  }
  const std::string recorded = tilewright::describe(*model).provenance.launch_model;
  std::shared_ptr<const hipblaslt_launch::Launch> launch;
  if (!recorded.empty()) {
    const std::string arch = hipblaslt_arch(recorded);
    if (arch.empty()) {
      std::fprintf(
          stderr, "the model records launch model '%s', not this rule's\n", recorded.c_str());
      return 1;
    }
    launch = std::make_shared<hipblaslt_launch::Launch>(arch, ix.hardware.N_CU, ix.configs);
  }

  if (dump > 0) return dump_plans(ix, dump, launch ? hipblaslt_arch(recorded) : "", dump_context);

  tilewright::PoolOptions options;
  options.launch = launch;
  const tilewright::CandidateSet set(model, ix.configs, options);
  if (prepare) set.prepare();
  std::vector<tilewright::Result> out;
  const auto call = [&](const tilewright::Problem& p, const tilewright::ExecutionContext& ctx) {
    if (top_k > 0)
      set.rank_top(p, ix.hardware, ctx, top_k, out);
    else
      out = set.rank(p, ix.hardware, ctx, min_scored, nullptr);
    return out.data();
  };
  const char* what         = top_k > 0 ? "rank_top k" : "min_scored";
  const std::size_t what_n = top_k > 0 ? top_k : min_scored;

  if (hash) {
    std::uint64_t all = kFnvBasis;
    for (std::size_t si = 0; si < ix.shapes.size(); ++si) {
      const Shape& sh = ix.shapes[si];
      tilewright::ExecutionContext ctx;
      ctx.cu_budget = sh.cu_budget;
      const auto p  = problem_of(ix, sh);
      if (prepare) set.prepare(p, ix.hardware);
      call(p, ctx);
      std::uint64_t one = kFnvBasis;
      for (const tilewright::Result& r : out) {
        std::uint64_t bits;
        std::memcpy(&bits, &r.score, sizeof bits);
        for (std::uint64_t v : {static_cast<std::uint64_t>(r.config_index),
                                static_cast<std::uint64_t>(r.scored ? 1 : 0),
                                bits}) {
          one = fnv(one, v);
          all = fnv(all, v);
        }
      }
      all = fnv(all, ~std::uint64_t{0});
      if (hash_each) std::printf("SHAPE %zu %016llx\n", si, static_cast<unsigned long long>(one));
    }
    std::printf("# %zu shapes, %zu kernels, %s %zu, launch '%s'\nHASH %016llx\n",
                ix.shapes.size(),
                ix.configs.size(),
                what,
                what_n,
                recorded.c_str(),
                static_cast<unsigned long long>(all));
    return 0;
  }

  const bool timed_stages = stages && tilewright::stage_timers_enabled();
  if (stages && !timed_stages) std::fprintf(stderr, "--stages: this build has no stage timers\n");
  std::vector<double> per_shape;
  per_shape.reserve(ix.shapes.size());
  std::vector<std::vector<double>> stage_shape(tilewright::kStageCount);
  std::vector<double> stage_total(tilewright::kStageCount, 0.0);
  std::uint64_t stage_calls = 0;
  const double tick_ns      = timed_stages ? tilewright::stage_tick_ns() : 0.0;
  for (const auto& sh : ix.shapes) {
    tilewright::ExecutionContext ctx;
    ctx.cu_budget = sh.cu_budget;
    const auto p  = problem_of(ix, sh);
    if (prepare) set.prepare(p, ix.hardware);
    for (std::size_t i = 0; i < warmup; ++i) call(p, ctx);
    tilewright::reset_stage_times();
    std::vector<double> us;
    us.reserve(iters);
    for (std::size_t i = 0; i < iters; ++i) {
      const auto t0      = std::chrono::steady_clock::now();
      const auto* result = call(p, ctx);
      const auto t1      = std::chrono::steady_clock::now();
      asm volatile("" : : "g"(result) : "memory");
      us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    std::sort(us.begin(), us.end());
    per_shape.push_back(us[us.size() / 2]);
    if (timed_stages) {
      const tilewright::StageTimes t = tilewright::stage_times();
      if (t.calls > 0) {
        stage_calls += t.calls;
        for (std::size_t s = 0; s < tilewright::kStageCount; ++s) {
          const double ns = static_cast<double>(t.ticks[s]) * tick_ns;
          stage_total[s] += ns;
          stage_shape[s].push_back(ns / static_cast<double>(t.calls));
        }
      }
    }
  }
  std::sort(per_shape.begin(), per_shape.end());
  double sum = 0;
  for (double x : per_shape) sum += x;
  std::printf("# %zu shapes, %zu kernels, %s %zu, launch '%s', median of %zu calls after %zu\n",
              per_shape.size(),
              ix.configs.size(),
              what,
              what_n,
              recorded.c_str(),
              iters,
              warmup);
  std::printf("%-14s %6zu %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f\n",
              "candidate_set",
              per_shape.size(),
              per_shape.front(),
              percentile(per_shape, 0.5),
              sum / static_cast<double>(per_shape.size()),
              percentile(per_shape, 0.9),
              percentile(per_shape, 0.99),
              per_shape.back());
  if (timed_stages && stage_calls > 0) {
    double total = 0;
    for (double ns : stage_total) total += ns;
    std::printf("# stage            mean_ns    share     p50_ns     p99_ns  (%llu calls)\n",
                static_cast<unsigned long long>(stage_calls));
    for (std::size_t s = 0; s < tilewright::kStageCount; ++s) {
      std::vector<double> v = stage_shape[s];
      std::sort(v.begin(), v.end());
      const double mean = stage_total[s] / static_cast<double>(stage_calls);
      std::printf("STAGE %-16s %9.1f %7.1f%% %10.1f %10.1f\n",
                  tilewright::stage_name(static_cast<tilewright::Stage>(s)),
                  mean,
                  total > 0 ? 100.0 * stage_total[s] / total : 0.0,
                  v.empty() ? 0.0 : percentile(v, 0.5),
                  v.empty() ? 0.0 : percentile(v, 0.99));
    }
    std::printf("STAGE %-16s %9.1f\n", "total", total / static_cast<double>(stage_calls));
  }
  return 0;
}
