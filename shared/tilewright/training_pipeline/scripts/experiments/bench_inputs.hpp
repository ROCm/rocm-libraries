// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// What rerank_check and memo_check share with bench_shipped_rank: its inputs
// (bench_selection_time's format), its candidate set (hipBLASLt's launch
// model when the model records one) and its --hash of a ranking's results.

#pragma once

#include "hipblaslt_launch_rule.hpp"
#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace bench_inputs {

struct Shape {
  std::size_t m, n, k, b, cu_budget;
};

struct Inputs {
  tilewright::Problem problem;
  tilewright::Hardware hardware;
  std::vector<Shape> shapes;
  std::vector<tilewright::Config> configs;
};

inline bool next_line(std::ifstream& in, std::istringstream& out) {
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

inline bool expect(std::istringstream& s, const char* tag) {
  std::string got;
  s >> got;
  if (got == tag) return true;
  std::fprintf(stderr, "inputs: expected %s, got '%s'\n", tag, got.c_str());
  return false;
}

inline bool parse_attribute(const std::string& token, tilewright::Attribute* out) {
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

inline bool load_inputs(const std::string& path, Inputs* ix) {
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

inline tilewright::Problem problem_of(const Inputs& ix, const Shape& sh) {
  tilewright::Problem p = ix.problem;
  p.size                = {sh.m, sh.n, sh.k};
  p.batch               = sh.b;
  return p;
}

inline tilewright::ExecutionContext context_of(const Shape& sh) {
  tilewright::ExecutionContext ctx;
  ctx.cu_budget = sh.cu_budget;
  return ctx;
}

// The model at `path` and pool options for `configs`: hipBLASLt's launch
// model when the model records it, as TensileLite ranks. Prints the reason
// and returns a null model when either cannot be had.
inline tilewright::ModelPtr load(const std::string& path,
                                 const Inputs& ix,
                                 const std::vector<tilewright::Config>& configs,
                                 tilewright::PoolOptions* options) {
  std::string error;
  tilewright::ModelPtr model = tilewright::load_model(path, &error);
  if (!model) {
    std::fprintf(stderr, "cannot load %s: %s\n", path.c_str(), error.c_str());
    return nullptr;
  }
  const std::string id = tilewright::describe(*model).provenance.launch_model;
  if (id.empty()) return model;
  const std::string head = "hipblaslt:";
  const std::string tail = ":" + std::to_string(hipblaslt_launch::kRuleVersion);
  if (id.size() <= head.size() + tail.size() || id.compare(0, head.size(), head) != 0 ||
      id.compare(id.size() - tail.size(), tail.size(), tail) != 0) {
    std::fprintf(stderr, "the model records launch model '%s', not this rule's\n", id.c_str());
    return nullptr;
  }
  const std::string arch = id.substr(head.size(), id.size() - head.size() - tail.size());
  options->launch = std::make_shared<hipblaslt_launch::Launch>(arch, ix.hardware.N_CU, configs);
  return model;
}

constexpr std::uint64_t kFnvBasis = 0xcbf29ce484222325ull;

inline std::uint64_t fnv(std::uint64_t h, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) {
    h ^= (v >> (8 * i)) & 0xFF;
    h *= 0x100000001b3ull;
  }
  return h;
}

// Adds one ranking's results to `*hash` as bench_shipped_rank --hash does:
// every result's config index, scored flag and score bits, then a separator.
inline void hash_results(const std::vector<tilewright::Result>& results, std::uint64_t* hash) {
  for (const tilewright::Result& r : results) {
    std::uint64_t bits;
    std::memcpy(&bits, &r.score, sizeof bits);
    for (std::uint64_t v : {static_cast<std::uint64_t>(r.config_index),
                            static_cast<std::uint64_t>(r.scored ? 1 : 0),
                            bits})
      *hash = fnv(*hash, v);
  }
  *hash = fnv(*hash, ~std::uint64_t{0});
}

}  // namespace bench_inputs
