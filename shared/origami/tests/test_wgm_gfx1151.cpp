// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Measured WorkGroupMapping choices on gfx1151 (40 CUs, one XCD) against
// origami::select_workgroup_mapping.
//
// data/wgm_gfx1151_measured.inc (provenance and columns in its header) is generated from the
// measurement tables.  data/wgm_gfx1151_known_gaps.inc follows the selector; after a selector
// change regenerate it and tighten the thresholds of the summary test:
//   ORIGAMI_WGM_GAPS_OUT=<this dir>/data/wgm_gfx1151_known_gaps.inc origami-tests "[.wgm-regen]"
// "[.wgm-dump]" writes the pick per row (ORIGAMI_WGM_PICKS_OUT) for comparison with a reference
// implementation of the rule; "[.wgm-legacy]" summarises the selection before the rule.

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include "common.hpp"

namespace {

enum : unsigned {
  PAD      = 1u << 0,   // lda/ldb padded
  PROD     = 1u << 1,   // production-like: padded, or natural strides that do not alias
  IDENT    = 1u << 2,   // nwg0 == 1 or nwg1 == 1
  AUX      = 1u << 3,   // kernel writes the extra output E
  OUT1K    = 1u << 4,   // output row bytes % 1024 == 0
  OUT4K    = 1u << 5,   // output row bytes % 4096 == 0
  OUTRS    = 1u << 6,   // ldc = ldd != m
  SHORT    = 1u << 7,   // best time < 100 us: out of scope, not asserted
  NT       = 1u << 8,   // transA = N, transB = T: auto WGM stays off (pinned WGM), not asserted
  STORE    = 1u << 9,   // store-dominated: output tile bytes >= 0.15 x (A + B tile bytes)
  ARTIFACT = 1u << 10,  // K < 16 with padded lda/ldb: not asserted
  MAGIC    = 1u << 11,  // landscape measured with the magic-number WGM remap
};

struct wgm_case_t {
  const char* id;
  size_t m, n, k, batch;
  origami::transpose_t trans_a, trans_b;
  origami::data_type_t a, b, c, d;
  size_t lda, ldb, ldc, ldd;
  size_t mt0, mt1, depth_u;
  int gsu;
  bool wmma;
  int occupancy;   // measured, workgroups per WGP (resident = N_CU / 2 x occupancy); 0 = unknown
  int pinned_wgm;  // WGM of the shipped kernel
  unsigned flags;
  int best_w;
  double best_us;
  const char* landscape;  // "W:regret[*] ...", regret in 0.1 % over the best class, '*' = accepted

  size_t nwg0() const { return origami::math::safe_ceil_div(m, mt0); }
  size_t nwg1() const { return origami::math::safe_ceil_div(n, mt1); }
  bool asserted() const { return !(flags & (SHORT | NT | ARTIFACT)); }
};

namespace rows {
constexpr auto N    = origami::transpose_t::N;
constexpr auto T    = origami::transpose_t::T;
constexpr auto f32  = origami::data_type_t::Float;
constexpr auto f64  = origami::data_type_t::Double;
constexpr auto f16  = origami::data_type_t::Half;
constexpr auto bf16 = origami::data_type_t::BFloat16;
constexpr auto i8   = origami::data_type_t::Int8;
constexpr auto i32  = origami::data_type_t::Int32;
constexpr auto i4   = origami::data_type_t::Int4;

#define WGM_CASE(...) wgm_case_t{__VA_ARGS__},
const std::vector<wgm_case_t> cases = {
#include "data/wgm_gfx1151_measured.inc"
};
#undef WGM_CASE

// Cases whose pick is currently outside the accepted set.
#define WGM_KNOWN_GAP(id) id,
const std::set<std::string_view> known_gaps = {
#include "data/wgm_gfx1151_known_gaps.inc"
};
#undef WGM_KNOWN_GAP
}  // namespace rows

const origami::hardware_t& gfx1151_hardware() {
  // 40 CUs (20 WGPs), 64 KB LDS, 2 MB L2, the 1900 MHz the measurements were pinned to.
  static const auto hw = origami::hardware_t::get_hardware_for_arch(
      origami::hardware_t::architecture_t::gfx1151, 40, 65536, 512 * 1024, 2 << 20, 1900000);
  return hw;
}

origami::problem_t make_wgm_problem(const wgm_case_t& r) {
  origami::problem_t problem;
  problem.size        = {r.m, r.n, r.k};
  problem.batch       = r.batch;
  problem.a_transpose = r.trans_a;
  problem.b_transpose = r.trans_b;
  problem.a_dtype     = r.a;
  problem.b_dtype     = r.b;
  problem.c_dtype     = r.c;
  problem.d_dtype     = r.d;
  problem.mi_dtype    = r.b;  // A is Int4 for the W4A16 kernels
  problem.ld_a        = r.lda;
  problem.ld_b        = r.ldb;
  problem.ld_c        = r.ldc;
  problem.ld_d        = r.ldd;
  problem.use_e       = (r.flags & AUX) != 0;  // E keeps its natural stride in the measurements
  return problem;
}

// The single place that asks Origami for a WGM: the non-persistent TensileLite path.
// workgroup_mapping carries the pinned WGM, which the selection returns for NT rows only.
int pick_wgm(const wgm_case_t& r) {
  origami::config_t config;
  config.mt = {r.mt0, r.mt1, r.depth_u};
  if (r.wmma) config.mi = {16, 16, 16};  // mi stays {0, 0, 0} for the non-WMMA kernels
  if (r.occupancy > 0) config.occupancy = r.occupancy;
  config.workgroup_mapping        = r.pinned_wgm;
  config.stream_k                 = 0;
  config.grid_selection           = origami::grid_selection_t::data_parallel;
  config.tensile().global_split_u = static_cast<std::int16_t>(r.gsu);

  return origami::select_workgroup_mapping(make_wgm_problem(r), gfx1151_hardware(), config, 0).wgm;
}

// The selection without TensileLite params: what every gfx1151 kernel got before the rule.
int pick_wgm_legacy(const wgm_case_t& r) {
  origami::config_t config;
  config.mt = {r.mt0, r.mt1, r.depth_u};
  return origami::select_workgroup_mapping(make_wgm_problem(r), gfx1151_hardware(), config, 0).wgm;
}

// Key of the WGM equivalence class, as in the generator (canon()).
int wgm_class(const wgm_case_t& r, int w) {
  if (w == 0 || w == 1) return 1;
  const int n0 = static_cast<int>(r.nwg0()), n1 = static_cast<int>(r.nwg1());
  if (n0 <= 1 || n1 <= 1) return w > 1 ? 2 : -2;
  return w > 1 ? std::min(w, n1) : -std::min(-w, n0);
}

struct class_result_t {
  int regret;  // 0.1 %
  bool accepted;
};

std::map<int, class_result_t> parse_landscape(const wgm_case_t& r) {
  std::map<int, class_result_t> out;
  std::istringstream in(r.landscape);
  std::string tok;
  while (in >> tok) {
    const auto colon                     = tok.find(':');
    const bool accepted                  = tok.back() == '*';
    out[std::stoi(tok.substr(0, colon))] = {std::stoi(tok.substr(colon + 1)), accepted};
  }
  return out;
}

struct outcome_t {
  int pick;
  bool measured;
  bool accepted;
  int regret;  // 0.1 %, 0 if not measured
};

outcome_t evaluate(const wgm_case_t& r, int (*picker)(const wgm_case_t&) = pick_wgm) {
  const int pick       = picker(r);
  const auto landscape = parse_landscape(r);
  const auto it        = landscape.find(wgm_class(r, pick));
  if (it == landscape.end()) return {pick, false, false, 0};
  return {pick, true, it->second.accepted, it->second.regret};
}

std::string describe(const wgm_case_t& r, const outcome_t& o) {
  std::ostringstream s;
  s << r.id << ": grid " << r.nwg0() << "x" << r.nwg1() << " (MT" << r.mt0 << "x" << r.mt1 << "x"
    << r.depth_u << ", GSU " << r.gsu << ", batch " << r.batch << "), picked WGM " << o.pick
    << " (class " << wgm_class(r, o.pick) << "), ";
  if (o.measured)
    s << "regret " << o.regret / 10.0 << " %";
  else
    s << "class not measured";
  s << "; accepted {";
  const char* sep = "";
  for (const auto& [w, c] : parse_landscape(r)) {
    if (!c.accepted) continue;
    s << sep << w;
    sep = ", ";
  }
  s << "}, best " << r.best_w << " at " << r.best_us << " us; landscape [W:0.1%] " << r.landscape;
  return s.str();
}

using group_t = bool (*)(const wgm_case_t&);
// Asserted groups; store-dominated overlaps the other three.
bool production(const wgm_case_t& r) {
  return r.asserted() && (r.flags & PROD) && !(r.flags & OUTRS);
}
bool restrided(const wgm_case_t& r) {
  return r.asserted() && (r.flags & PROD) && (r.flags & OUTRS);
}
bool aliased(const wgm_case_t& r) { return r.asserted() && !(r.flags & PROD); }
bool store_dominated(const wgm_case_t& r) { return r.asserted() && (r.flags & STORE); }
// Reported only.
bool short_kernel(const wgm_case_t& r) { return (r.flags & SHORT) != 0; }
bool nt(const wgm_case_t& r) { return (r.flags & NT) != 0; }
bool artifact(const wgm_case_t& r) { return (r.flags & ARTIFACT) != 0; }

// Every case outside the known gaps is accepted; every known gap still fails.
void check_group(group_t group) {
  size_t checked = 0;
  for (const auto& r : rows::cases) {
    if (!group(r)) continue;
    ++checked;
    const auto o   = evaluate(r);
    const bool gap = rows::known_gaps.count(r.id) != 0;
    INFO(describe(r, o));
    if (gap) {
      INFO("listed in wgm_gfx1151_known_gaps.inc but accepted now: regenerate the list");
      CHECK_FALSE(o.accepted);
    } else {
      CHECK(o.accepted);
    }
  }
  REQUIRE(checked > 0);
}

struct summary_t {
  size_t cases = 0, outside = 0, unmeasured = 0;
  long regret    = 0;  // 0.1 %, measured picks only
  int max_regret = 0;
};

summary_t summarize(group_t group, int (*picker)(const wgm_case_t&) = pick_wgm) {
  summary_t s;
  for (const auto& r : rows::cases) {
    if (!group(r)) continue;
    const auto o = evaluate(r, picker);
    ++s.cases;
    if (!o.measured) {
      ++s.unmeasured;
      continue;
    }
    if (!o.accepted) ++s.outside;
    s.regret += o.regret;
    s.max_regret = std::max(s.max_regret, o.regret);
  }
  return s;
}

std::string to_string(const summary_t& s) {
  std::ostringstream o;
  o << s.cases << " cases, " << s.outside << " outside the accepted set, " << s.unmeasured
    << " picks in an unmeasured class, total regret " << s.regret / 10.0 << " %, max "
    << s.max_regret / 10.0 << " %";
  return o.str();
}

}  // namespace

TEST_CASE("WGM gfx1151: table is well formed", "[wgm][gfx1151]") {
  std::set<std::string_view> ids;
  for (const auto& r : rows::cases) {
    INFO(r.id);
    REQUIRE(ids.insert(r.id).second);
    const auto landscape = parse_landscape(r);
    REQUIRE(landscape.count(1) == 1);
    const auto best = landscape.find(wgm_class(r, r.best_w));
    REQUIRE(best != landscape.end());
    REQUIRE(best->second.regret == 0);
    REQUIRE(best->second.accepted);
    for (const auto& [w, c] : landscape) {
      REQUIRE(wgm_class(r, w) == w);
      REQUIRE(c.regret >= 0);
      if (c.accepted) REQUIRE(c.regret <= 30);  // the 3 % tolerance cap
    }
    REQUIRE(((r.flags & IDENT) != 0) == (r.nwg0() == 1 || r.nwg1() == 1));
    REQUIRE(((r.flags & OUTRS) != 0) == (r.ldc != r.m || r.ldd != r.m));
    REQUIRE(((r.flags & SHORT) != 0) == (r.best_us < 100.0));
    if (r.flags & ARTIFACT) REQUIRE((r.k < 16 && (r.flags & PAD)));
    REQUIRE(((r.flags & NT) != 0) ==
            (r.trans_a == origami::transpose_t::N && r.trans_b == origami::transpose_t::T));
  }
  std::set<std::string_view> asserted_ids;
  for (const auto& r : rows::cases)
    if (r.asserted()) asserted_ids.insert(r.id);
  for (const auto& id : rows::known_gaps) {
    INFO(id);
    REQUIRE(asserted_ids.count(id) == 1);
  }
  REQUIRE(gfx1151_hardware().N_CU == 40);
  REQUIRE(gfx1151_hardware().NUM_XCD == 1);
}

TEST_CASE("WGM gfx1151: production-like cases with natural output stride pick an accepted WGM",
          "[wgm][gfx1151]") {
  check_group(production);
}

TEST_CASE("WGM gfx1151: production-like cases with re-strided output pick an accepted WGM",
          "[wgm][gfx1151]") {
  check_group(restrided);
}

TEST_CASE("WGM gfx1151: unpadded aliased-input cases pick an accepted WGM", "[wgm][gfx1151]") {
  check_group(aliased);
}

TEST_CASE("WGM gfx1151: store-dominated cases pick an accepted WGM", "[wgm][gfx1151][store]") {
  check_group(store_dominated);
}

TEST_CASE("WGM gfx1151: NT returns the pinned WGM", "[wgm][gfx1151]") {
  size_t checked = 0;
  for (const auto& r : rows::cases) {
    if (!nt(r)) continue;
    ++checked;
    INFO(r.id);
    CHECK(pick_wgm(r) == r.pinned_wgm);
  }
  REQUIRE(checked > 0);
}

TEST_CASE("WGM gfx1151: summary of the picks against the measured landscapes", "[wgm][gfx1151]") {
  const auto prod  = summarize(production);
  const auto rest  = summarize(restrided);
  const auto alias = summarize(aliased);
  const auto store = summarize(store_dominated);
  WARN("production-like, natural output stride: " << to_string(prod));
  WARN("production-like, re-strided output:     " << to_string(rest));
  WARN("unpadded aliased inputs:                " << to_string(alias));
  WARN("store-dominated (overlaps the others):  " << to_string(store));
  WARN("not asserted, short kernels (< 100 us): " << to_string(summarize(short_kernel)));
  WARN("not asserted, NT:                       " << to_string(summarize(nt)));
  WARN("not asserted, K < 16 padded artifacts:  " << to_string(summarize(artifact)));

  // Thresholds = the selector at the time the known-gap list was generated; tighten when it
  // improves.
  CHECK(prod.outside <= 13);
  CHECK(prod.unmeasured == 0);
  CHECK(prod.regret <= 382);
  CHECK(rest.outside <= 11);
  CHECK(rest.unmeasured == 0);
  CHECK(rest.regret <= 378);
  CHECK(alias.outside <= 31);
  CHECK(alias.unmeasured == 0);
  CHECK(alias.regret <= 5033);
  CHECK(store.outside <= 9);
  CHECK(store.unmeasured == 0);
  CHECK(store.regret <= 244);
  CHECK(prod.outside + prod.unmeasured + rest.outside + rest.unmeasured + alias.outside +
            alias.unmeasured ==
        rows::known_gaps.size());
}

// Hidden: the same summary for the selection before the gfx1151 rule.
TEST_CASE("WGM gfx1151: summary of the legacy selection", "[.wgm-legacy]") {
  WARN("production-like, natural output stride: "
       << to_string(summarize(production, pick_wgm_legacy)));
  WARN("production-like, re-strided output:     "
       << to_string(summarize(restrided, pick_wgm_legacy)));
  WARN(
      "unpadded aliased inputs:                " << to_string(summarize(aliased, pick_wgm_legacy)));
  WARN("store-dominated (overlaps the others):  "
       << to_string(summarize(store_dominated, pick_wgm_legacy)));
  WARN("not asserted, short kernels (< 100 us): "
       << to_string(summarize(short_kernel, pick_wgm_legacy)));
  WARN("not asserted, NT:                       " << to_string(summarize(nt, pick_wgm_legacy)));
  WARN("not asserted, K < 16 padded artifacts:  "
       << to_string(summarize(artifact, pick_wgm_legacy)));
}

// Hidden: writes "id pick" per row for the cross-check against the reference implementation.
TEST_CASE("WGM gfx1151: dump the picks", "[.wgm-dump]") {
  const char* path = std::getenv("ORIGAMI_WGM_PICKS_OUT");
  REQUIRE(path != nullptr);
  std::ofstream out(path);
  REQUIRE(out.good());
  for (const auto& r : rows::cases) out << r.id << " " << pick_wgm(r) << "\n";
}

// Hidden: writes the known-gap list for the current selector.
TEST_CASE("WGM gfx1151: regenerate the known-gap list", "[.wgm-regen]") {
  std::ostringstream body;
  body << "// Copyright Advanced Micro Devices, Inc., or its affiliates.\n"
          "// SPDX-License-Identifier: MIT\n"
          "//\n"
          "// Generated by origami-tests \"[.wgm-regen]\" (see test_wgm_gfx1151.cpp); do not edit "
          "by\n"
          "// hand.  Asserted cases (not SHORT, NT or ARTIFACT) of wgm_gfx1151_measured.inc whose "
          "WGM\n"
          "// from origami::select_workgroup_mapping is outside the accepted set.\n"
          "// Trailing comment: pick, regret, accepted classes.\n";
  size_t gaps = 0;
  for (const auto& r : rows::cases) {
    if (!r.asserted()) continue;
    const auto o = evaluate(r);
    if (o.accepted) continue;
    ++gaps;
    body << "WGM_KNOWN_GAP(\"" << r.id << "\")  // picked " << o.pick << ", ";
    if (o.measured)
      body << "+" << o.regret / 10.0 << " %";
    else
      body << "class not measured";
    body << ", accepted {";
    const char* sep = "";
    for (const auto& [w, c] : parse_landscape(r)) {
      if (!c.accepted) continue;
      body << sep << w;
      sep = ", ";
    }
    body << "}\n";
  }
  const char* path = std::getenv("ORIGAMI_WGM_GAPS_OUT");
  if (path != nullptr) {
    std::ofstream out(path);
    REQUIRE(out.good());
    out << body.str();
  } else {
    std::cout << body.str();
  }
  WARN(gaps << " known gaps");
}
