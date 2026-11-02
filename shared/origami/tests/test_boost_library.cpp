// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Library stamp of Origami Boost models: structural values, name digests, the canonical
// manifest, the diff decision and loading.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "origami/nn/boost/load_validate.hpp"
#include "origami/nn/boost/rank.hpp"
#include "origami/nn/boost/sha256.hpp"
#include "origami/nn/features/gemm_boost.hpp"

using namespace origami;
using namespace origami::nn::boost;
namespace features = origami::nn::features::gemm_boost;
namespace fs = std::filesystem;

namespace {

const fs::path kData = ORIGAMI_BOOST_TEST_DATA;

std::vector<std::uint8_t> read_bytes(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

// A model file with its MANI section replaced by @p text.
std::vector<std::uint8_t> with_manifest(const std::vector<std::uint8_t>& file, const std::string& text) {
  std::vector<std::uint8_t> b(file.begin(), file.begin() + 24);
  for (std::size_t o = 24; o + 12 <= file.size() - 32;) {
    std::uint64_t len;
    std::memcpy(&len, &file[o + 4], 8);
    if (std::memcmp(&file[o], "MANI", 4) != 0)
      b.insert(b.end(), file.begin() + static_cast<std::ptrdiff_t>(o),
               file.begin() + static_cast<std::ptrdiff_t>(o + 12 + len));
    o += 12 + len;
  }
  b.insert(b.end(), {'M', 'A', 'N', 'I'});
  const std::uint64_t len = text.size();
  const auto*         lp  = reinterpret_cast<const std::uint8_t*>(&len);
  b.insert(b.end(), lp, lp + 8);
  b.insert(b.end(), text.begin(), text.end());
  const std::uint64_t body_len = b.size() - 24;
  std::memcpy(&b[16], &body_len, 8);
  const std::string hex = sha256_hex(b.data() + 24, body_len);
  for (std::size_t i = 0; i < 32; ++i)
    b.push_back(static_cast<std::uint8_t>(std::stoi(hex.substr(2 * i, 2), nullptr, 16)));
  return b;
}

struct spec_t {
  std::string name;
  size_t      mt_m, mt_n, mt_k, mi_m, mi_n, mi_k;
  bool        dtl_a = false, one_lds = false;
  int         ulds_a = 1, ulds_b = 1;
};

library_kernel_t kernel(const spec_t& s, std::int64_t index) {
  library_kernel_t k;
  k.name     = s.name;
  k.index    = index;
  config_t& c = k.config;
  c.mt        = {s.mt_m, s.mt_n, s.mt_k};
  c.mi        = {s.mi_m, s.mi_n, s.mi_k};
  c.occupancy = 1;
  c.stream_k  = 3;
  auto& t     = c.tensile();
  t.direct_to_lds_a     = s.dtl_a;
  t.one_lds_buffer      = s.one_lds;
  t.unroll_major_lds_a  = s.ulds_a;
  t.unroll_major_lds_b  = s.ulds_b;
  t.wave_group_m        = 2;
  t.wave_group_n        = 2;
  k.fields.num_loads_coalesced_a = 1;
  k.fields.num_loads_coalesced_b = 1;
  return k;
}

// The training library: 16x16x32 MFMA, MT_K 64/128, no DTL, one LDS mode, two LDS buffers.
std::vector<library_kernel_t> training_kernels() {
  std::vector<library_kernel_t> v;
  const size_t tiles[][3] = {{256, 256, 64}, {256, 128, 64}, {128, 256, 64}, {128, 128, 64},
                             {256, 224, 64}, {192, 256, 64}, {128, 128, 128}, {64, 128, 128}};
  for (std::size_t i = 0; i < std::size(tiles); ++i)
    v.push_back(kernel({"K" + std::to_string(i), tiles[i][0], tiles[i][1], tiles[i][2], 16, 16, 32},
                       static_cast<std::int64_t>(100 + i)));
  return v;
}

int field(const char* name) {
  const auto& f = structural_fields();
  for (std::size_t i = 0; i < f.size(); ++i)
    if (std::string(f[i].name) == name) return static_cast<int>(i);
  return -1;
}

std::shared_ptr<const ensemble_t> model_on(const std::vector<library_kernel_t>& training,
                                           const std::string& dtype = "bf16") {
  const auto entries = canonical_manifest(training);
  const auto bytes   = with_manifest(read_bytes(kData / "boost_model_bf16.orr"),
                                     manifest_section_text(dtype, entries, seen_structural_values(entries)));
  status_t    st;
  std::string detail;
  auto        m = ensemble_t::load_bytes(bytes.data(), bytes.size(), st, detail);
  INFO(detail);
  REQUIRE(m);
  return m;
}

}  // namespace

TEST_CASE("Origami Boost library: structural field table and values", "[nn][boost]") {
  const auto& f = structural_fields();
  REQUIRE(f.size() == 25);
  std::vector<std::string> gated;
  for (const auto& x : f)
    if (x.gate) gated.push_back(x.name);
  REQUIRE(gated == std::vector<std::string>{"dtl", "mfma", "mi", "lds_mode", "one_lds_buffer", "mt_k"});

  auto k = kernel({"A", 256, 128, 64, 16, 16, 32, true, true, 0, 1}, 7);
  const auto v = structural_values(k.config, k.fields);
  REQUIRE(v.size() == f.size());
  CHECK(v[field("dtl")] == "10");
  CHECK(v[field("mfma")] == "1");
  CHECK(v[field("mi")] == "16x16x32");
  CHECK(v[field("lds_mode")] == "0,1,0,0");
  CHECK(v[field("one_lds_buffer")] == "1");
  CHECK(v[field("mt_k")] == "64");
  CHECK(v[field("mt_m")] == "256");
  CHECK(v[field("wave_group")] == "2x2");
  CHECK(v[field("nlc")] == "1,1");

  // hipBLASLt gives non-MFMA kernels the instruction 1x1x64.
  auto dot2 = kernel({"D", 1, 4, 256, 1, 1, 64}, 8);
  const auto vd = structural_values(dot2.config, dot2.fields);
  CHECK(vd[field("mfma")] == "0");
  CHECK(vd[field("mi")] == "1x1x64");
}

TEST_CASE("Origami Boost library: name digests and canonical manifest", "[nn][boost]") {
  // 64-bit digest: the first 16 hex digits of the name's sha256.
  REQUIRE(name_digest("abc") == "ba7816bf8f01cfea");
  auto ks  = training_kernels();
  auto dup = ks[2];
  dup.index = 3;  // listed twice (same name and values), as in some library rows
  ks.push_back(dup);
  std::shuffle(ks.begin(), ks.end(), std::mt19937(7));
  const auto m = canonical_manifest(ks);
  REQUIRE(m.size() == 8);
  REQUIRE(std::is_sorted(m.begin(), m.end(), [](const auto& a, const auto& b) {
    return std::tie(a.digest, a.values) < std::tie(b.digest, b.values);
  }));
  const auto it = std::find_if(m.begin(), m.end(), [](const auto& e) { return e.name == "K2"; });
  REQUIRE(it != m.end());
  REQUIRE(it->digest == name_digest("K2"));
  REQUIRE(it->index == 3);  // smallest index kept
  const auto again = canonical_manifest(training_kernels());
  REQUIRE(again.size() == m.size());
  for (std::size_t i = 0; i < m.size(); ++i) REQUIRE(again[i].values == m[i].values);
  // The model stores digests, not names.
  const auto text = manifest_section_text("bf16", m, seen_structural_values(m));
  REQUIRE(text.find("K2") == std::string::npos);
  REQUIRE(text.find("\t" + name_digest("K2") + "\t") != std::string::npos);
}

TEST_CASE("Origami Boost library: diff decisions", "[nn][boost]") {
  if (!fs::is_regular_file(kData / "boost_model_bf16.orr")) SKIP("fixtures not found");
  const auto train = training_kernels();
  const auto model = model_on(train);
  REQUIRE(model->library_manifest().size() == 8);

  auto decide = [&](std::vector<library_kernel_t> lib) { return diff_library(*model, canonical_manifest(lib)); };

  SECTION("identical library in another order and with other indices: accept") {
    auto lib = train;
    std::reverse(lib.begin(), lib.end());
    for (auto& k : lib) k.index += 5000;
    const auto d = decide(lib);
    CHECK(d.decision == library_decision_t::accept);
    CHECK(d.kept == 8);
    CHECK(d.removed == 0);
    CHECK(d.added == 0);
  }
  SECTION("removed kernels: accept") {
    auto lib = train;
    lib.erase(lib.begin(), lib.begin() + 3);
    const auto d = decide(lib);
    CHECK(d.decision == library_decision_t::accept);
    CHECK(d.kept == 5);
    CHECK(d.removed == 3);
  }
  SECTION("added kernels with seen structure: accept with additions") {
    auto lib = train;
    lib.push_back(kernel({"NEW1", 128, 192, 64, 16, 16, 32}, 900));  // new MT_N: ungated
    lib.push_back(kernel({"NEW2", 64, 64, 128, 16, 16, 32}, 901));
    const auto d = decide(lib);
    CHECK(d.decision == library_decision_t::accept_with_additions);
    CHECK(d.added == 2);
    CHECK(d.new_values.empty());
    REQUIRE(d.new_ungated_values.count("mt_n"));
    CHECK(d.new_ungated_values.at("mt_n") == std::set<std::string>{"192", "64"});
    CHECK(d.summary().rfind("ACCEPT-WITH-ADDITIONS: 8 of 8 training kernels present, 0 removed, 2 added", 0) == 0);
  }
  auto refuses = [&](const spec_t& s, const char* fname, const std::string& value) {
    auto lib = train;
    lib.push_back(kernel(s, 950));
    const auto d = decide(lib);
    INFO(d.summary());
    CHECK(d.decision == library_decision_t::retrain_required);
    REQUIRE(d.new_values.count(fname));
    CHECK(d.new_values.at(fname).count(value));
    CHECK(d.affected == std::vector<std::string>{s.name});
  };
  SECTION("new DTL setting: retrain") { refuses({"DTL", 256, 256, 64, 16, 16, 32, true}, "dtl", "10"); }
  SECTION("new MFMA type: retrain") { refuses({"MI32", 256, 256, 64, 32, 32, 16}, "mi", "32x32x16"); }
  SECTION("non-MFMA kernel: retrain") {
    refuses({"DOT2", 1, 4, 64, 1, 1, 64}, "mfma", "0");
    refuses({"DOT2", 1, 4, 64, 1, 1, 64}, "mi", "1x1x64");
  }
  SECTION("new LDS mode (TLDS0): retrain") {
    refuses({"TLDS0", 256, 256, 64, 16, 16, 32, false, false, 0, 0}, "lds_mode", "0,0,0,0");
  }
  SECTION("one LDS buffer: retrain") { refuses({"OLB", 256, 256, 64, 16, 16, 32, false, true}, "one_lds_buffer", "1"); }
  SECTION("new MT_K: retrain") { refuses({"MTK32", 256, 256, 32, 16, 16, 32}, "mt_k", "32"); }
  SECTION("a training kernel whose values changed counts as removed and added") {
    auto lib = train;
    lib[0].config.tensile().direct_to_lds_a = true;
    const auto d = decide(lib);
    CHECK(d.decision == library_decision_t::retrain_required);
    CHECK(d.removed == 1);
    CHECK(d.added == 1);
    CHECK(d.affected == std::vector<std::string>{"K0"});
  }
  SECTION("summary and JSON name the new values") {
    auto lib = train;
    lib.push_back(kernel({"DTL", 256, 256, 64, 16, 16, 32, true}, 950));
    lib.push_back(kernel({"MI32", 256, 256, 32, 32, 32, 16}, 951));
    const auto d = decide(lib);
    CHECK(d.summary() ==
          "RETRAIN-REQUIRED: 8 of 8 training kernels present, 0 removed, 2 added; 2 added kernels "
          "carry structural values never seen in training: dtl {10}; mi {32x32x16}; mt_k {32}");
    const auto j = d.json();
    CHECK(j.find("\"decision\": \"RETRAIN-REQUIRED\"") != std::string::npos);
    CHECK(j.find("\"new_structural_values\": {\"dtl\": [\"10\"], \"mi\": [\"32x32x16\"], \"mt_k\": [\"32\"]}") !=
          std::string::npos);
    CHECK(j.find("\"affected_kernels\": [\"DTL\", \"MI32\"]") != std::string::npos);
  }
}

TEST_CASE("Origami Boost library: loading and create()", "[nn][boost]") {
  if (!fs::is_regular_file(kData / "boost_model_bf16.orr")) SKIP("fixtures not found");
  const auto file    = read_bytes(kData / "boost_model_bf16.orr");
  const auto train   = training_kernels();
  const auto entries = canonical_manifest(train);
  const auto text    = manifest_section_text("bf16", entries, seen_structural_values(entries));
  status_t    st;
  std::string detail;

  auto model = model_on(train);
  const build_identity_t id{model->head_string("origami_commit"), model->head_string("origami_source_hash"),
                            model->head_string("feature_schema_id"), model->head_string("feature_schema_hash")};

  SECTION("matching library: ok, detail reports the diff") {
    const auto lib = canonical_manifest(train);
    auto       b   = booster_t::create(model, {"gfx950", "bf16", &lib}, id, st, detail);
    INFO(detail);
    REQUIRE(b);
    CHECK(detail.find("kernel library ACCEPT: 8 of 8 training kernels present") != std::string::npos);
  }
  SECTION("a library without some training kernels is accepted") {
    auto lib = canonical_manifest(train);
    lib.erase(lib.begin());
    REQUIRE(booster_t::create(model, {"gfx950", "bf16", &lib}, id, st, detail));
  }
  SECTION("new structural value: refused with the values") {
    auto ks = train;
    ks.push_back(kernel({"TLDS0", 256, 256, 64, 16, 16, 32, false, false, 0, 0}, 960));
    const auto lib = canonical_manifest(ks);
    REQUIRE_FALSE(booster_t::create(model, {"gfx950", "bf16", &lib}, id, st, detail));
    CHECK(st == status_t::incompatible);
    CHECK(detail.find("RETRAIN-REQUIRED") != std::string::npos);
    CHECK(detail.find("lds_mode {0,0,0,0}") != std::string::npos);
  }
  SECTION("arch, dtype and source checks come first") {
    const auto lib = canonical_manifest(train);
    CHECK_FALSE(booster_t::create(model, {"gfx942", "bf16", &lib}, id, st, detail));
    CHECK_FALSE(booster_t::create(model, {"gfx950", "tf32", &lib}, id, st, detail));
    auto other                = id;
    other.origami_source_hash = "sha256:other";
    CHECK_FALSE(booster_t::create(model, {"gfx950", "bf16", &lib}, other, st, detail));
    CHECK(detail.find("origami source mismatch") != std::string::npos);
  }
  SECTION("malformed or foreign manifests") {
    auto bad = [&](const std::string& t, status_t want) {
      const auto b = with_manifest(file, t);
      CHECK_FALSE(ensemble_t::load_bytes(b.data(), b.size(), st, detail));
      CHECK(st == want);
    };
    bad("manifest\t2\n", status_t::unsupported_format);
    std::string other_table = text;
    other_table.replace(other_table.find("field\tmt_k\t1"), 12, "field\tmt_k\t0");
    bad(other_table, status_t::unsupported_format);
    std::string wrong_dtype = text;
    wrong_dtype.replace(wrong_dtype.find("dtype\tbf16"), 10, "dtype\ttf32");
    bad(wrong_dtype, status_t::bad_file);
    bad(text.substr(0, text.find("kernel\t")), status_t::bad_file);  // no kernels
    std::string bad_digest = text;
    const auto  at         = bad_digest.find("kernel\t");
    bad_digest.replace(bad_digest.find('\t', bad_digest.find('\t', at) + 1) + 1, 1, "Z");
    bad(bad_digest, status_t::bad_file);
  }
  SECTION("a file without a MANI section is refused") {
    std::vector<std::uint8_t> b(file.begin(), file.begin() + 24);
    for (std::size_t o = 24; o + 12 <= file.size() - 32;) {
      std::uint64_t len;
      std::memcpy(&len, &file[o + 4], 8);
      if (std::memcmp(&file[o], "MANI", 4) != 0)
        b.insert(b.end(), file.begin() + static_cast<std::ptrdiff_t>(o),
                 file.begin() + static_cast<std::ptrdiff_t>(o + 12 + len));
      o += 12 + len;
    }
    const std::uint64_t body_len = b.size() - 24;
    std::memcpy(&b[16], &body_len, 8);
    const std::string hex = sha256_hex(b.data() + 24, body_len);
    for (std::size_t i = 0; i < 32; ++i)
      b.push_back(static_cast<std::uint8_t>(std::stoi(hex.substr(2 * i, 2), nullptr, 16)));
    CHECK_FALSE(ensemble_t::load_bytes(b.data(), b.size(), st, detail));
    CHECK(st == status_t::bad_file);
    CHECK(detail == "missing required section");
  }
}
