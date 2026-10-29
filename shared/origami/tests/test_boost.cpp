// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "common.hpp"
#include "origami/gemm_intermediates.hpp"
#include "origami/nn/boost/load_validate.hpp"
#include "origami/nn/boost/rank.hpp"
#include "origami/nn/boost/sha256.hpp"
#include "origami/nn/features/gemm_boost.hpp"
#include "origami/nn/nn.hpp"

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

void require_data() {
  if (!fs::is_directory(kData)) SKIP("Origami Boost fixtures not found at " << kData);
}

// Recompute the sha256 trailer after editing the body of a model file.
void reseal(std::vector<std::uint8_t>& b) {
  const std::string hex = sha256_hex(b.data() + 24, b.size() - 24 - 32);
  for (std::size_t i = 0; i < 32; ++i)
    b[b.size() - 32 + i] = static_cast<std::uint8_t>(std::stoi(hex.substr(2 * i, 2), nullptr, 16));
}

std::size_t find_tag(const std::vector<std::uint8_t>& b, const char* tag) {
  const auto it = std::search(b.begin() + 24, b.end(), tag, tag + std::strlen(tag));
  REQUIRE(it != b.end());
  return static_cast<std::size_t>(it - b.begin());
}

std::shared_ptr<const ensemble_t> load(const std::vector<std::uint8_t>& b, status_t& st,
                                       std::string& detail) {
  return ensemble_t::load_bytes(b.data(), b.size(), st, detail);
}

std::shared_ptr<const ensemble_t> fixture(const char* name) {
  status_t    st;
  std::string detail;
  auto        m = ensemble_t::load_file((kData / name).string(), st, detail);
  INFO(detail);
  REQUIRE(m);
  return m;
}

build_identity_t identity_of(const ensemble_t& m) {
  return {m.head_string("origami_commit"), m.head_string("origami_source_hash"),
          m.head_string("feature_schema_id"), m.head_string("feature_schema_hash")};
}

// The model's own stamps, on its training library.
deployment_t target_of(const ensemble_t& m) {
  return {m.head_string("arch"), m.head_string("dtype"), &m.library_manifest()};
}

struct csv_t {
  std::vector<std::string>         header;
  std::vector<std::vector<double>> rows;
};

csv_t read_csv(const fs::path& p) {
  csv_t         c;
  std::ifstream f(p);
  std::string   line;
  auto          split = [](const std::string& s) {
    std::vector<std::string> out;
    std::stringstream        ss(s);
    std::string              item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
  };
  if (std::getline(f, line)) c.header = split(line);
  while (std::getline(f, line)) {
    std::vector<double> r;
    for (const auto& v : split(line)) r.push_back(std::strtod(v.c_str(), nullptr));
    c.rows.push_back(std::move(r));
  }
  return c;
}

hardware_t gfx950() {
  return hardware_t::get_hardware_for_arch(hardware_t::architecture_t::gfx950, 256, 163840,
                                           512 * 1024, 4 * 1024 * 1024, 2'400'000, std::nullopt);
}

// A/B/C/D Float with XFloat32 math: what hipBLASLt passes for compute f32_fast_xf32.
problem_t make_tf32_problem(size_t m, size_t n, size_t k, transpose_t ta, transpose_t tb,
                            size_t batch = 1, bool xf32 = true) {
  problem_t p = make_problem(m, n, k, ta, tb, batch);
  p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = data_type_t::Float;
  p.mi_dtype = xf32 ? data_type_t::XFloat32 : data_type_t::Float;
  return p;
}

// F8BS: A/B Float8 (OCP E4M3) into Float8 MI with BFloat16 C/D, what hipBLASLt passes for f8_r with
// a bf16 output.
problem_t make_fp8_problem(size_t m, size_t n, size_t k, transpose_t ta, transpose_t tb,
                           size_t batch = 1) {
  problem_t p = make_problem(m, n, k, ta, tb, batch);
  p.a_dtype = p.b_dtype = p.mi_dtype = data_type_t::Float8;
  p.c_dtype = p.d_dtype = data_type_t::BFloat16;
  return p;
}

struct candidate_set_t {
  std::vector<config_t>    configs;
  std::vector<candidate_t> candidates;
};

// A plausible static Stream-K launch for a synthetic config: one workgroup per resident slot,
// at most one per tile (no hipBLASLt needed).
streamk_launch_t synthetic_launch(const problem_t& p, const config_t& c) {
  streamk_launch_t sk;
  sk.valid      = true;
  sk.is_streamk = c.stream_k != 0;
  sk.mt_x       = c.mt.m;
  sk.mt_y       = c.mt.n;
  sk.workspace_size_per_elem_c = 4;
  sk.cu_occupancy              = c.occupancy;
  sk.cu_count                  = 256;
  if (!sk.is_streamk) return sk;
  auto ceil_div     = [](std::uint64_t a, std::uint64_t b) { return (a + b - 1) / b; };
  sk.tiles          = ceil_div(p.size.m, c.mt.m) * ceil_div(p.size.n, c.mt.n) * std::max<size_t>(p.batch, 1);
  sk.iters_per_tile = std::max<std::uint64_t>(ceil_div(p.size.k, c.mt.k), 1);
  sk.final_grid     = std::min<std::uint64_t>(sk.tiles, 256 * static_cast<std::uint64_t>(std::max(c.occupancy, 1)));
  sk.sk_tiles       = sk.tiles;
  sk.sk_split       = 1;
  sk.reduction      = reduction_t::tree;
  return sk;
}

// Ten configs, ranked by Origami the way hipBLASLt does it.
candidate_set_t ranked_candidates(const problem_t& p, const hardware_t& hw, bool with_fields) {
  struct spec_t {
    size_t mt_m, mt_n, mt_k, mi_m, mi_n, mi_k;
    int    wg_m, wg_n, occ, sk;
  };
  const spec_t specs[] = {{256, 256, 64, 16, 16, 32, 2, 2, 1, 3}, {256, 128, 64, 16, 16, 32, 2, 2, 1, 3},
                          {128, 256, 64, 16, 16, 32, 2, 2, 1, 3}, {128, 128, 64, 16, 16, 32, 2, 2, 2, 3},
                          {256, 224, 64, 16, 16, 32, 2, 2, 1, 0}, {192, 256, 64, 16, 16, 32, 2, 2, 1, 0},
                          {128, 128, 128, 16, 16, 32, 2, 2, 1, 0}, {64, 128, 128, 16, 16, 32, 1, 2, 2, 3},
                          {256, 64, 128, 32, 32, 16, 2, 1, 2, 3}, {128, 64, 64, 32, 32, 16, 2, 2, 2, 0},
                          {64, 64, 256, 16, 16, 32, 2, 2, 2, 0},  {32, 64, 256, 16, 16, 32, 1, 2, 2, 3}};
  std::vector<config_t> all;
  for (std::size_t i = 0; i < std::size(specs); ++i) {
    const auto& s = specs[i];
    config_t    c = make_config(s.mt_m, s.mt_n, s.mt_k, s.mi_m, s.mi_n, s.mi_k, false, 8, s.occ, 0,
                                0, s.sk);
    c.workspace_size            = std::numeric_limits<size_t>::max();
    c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
    c.index                     = i;
    auto& tp                    = c.tensile();
    tp.wave_group_m             = s.wg_m;
    tp.wave_group_n             = s.wg_n;
    tp.prefetch_global_read     = 2;
    all.push_back(c);
  }
  memoize_non_temporal_availability(all);
  const auto      ranked = rank_configs(p, hw, all);
  candidate_set_t out;
  for (std::size_t i = 0; i < ranked.size() && out.configs.size() < 10; ++i)
    out.configs.push_back(ranked[i].config);
  for (std::size_t i = 0; i < out.configs.size(); ++i) {
    candidate_t cand;
    cand.config          = &out.configs[i];
    cand.origami_latency = ranked[i].latency;
    if (with_fields) {
      cand.fields.num_loads_coalesced_a = 1;
      cand.fields.num_loads_coalesced_b = 1;
      cand.fields.streamk               = synthetic_launch(p, out.configs[i]);
    }
    out.candidates.push_back(cand);
  }
  return out;
}

}  // namespace

TEST_CASE("Origami Boost: sha256 matches FIPS 180-2 vectors", "[nn][boost]") {
  REQUIRE(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  REQUIRE(sha256_hex("abc", 3) ==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const char* two_blocks = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
  REQUIRE(sha256_hex(two_blocks, std::strlen(two_blocks)) ==
          "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  const std::string million(1000000, 'a');
  REQUIRE(sha256_hex(million.data(), million.size()) ==
          "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("Origami Boost: model file loads", "[nn][boost]") {
  require_data();
  const auto m = fixture("boost_model_bf16.orr");
  REQUIRE(m->num_trees() == 300);
  REQUIRE(m->features().size() == features::schema().columns.size());
  for (std::size_t j = 0; j < m->features().size(); ++j)
    REQUIRE(m->features()[j].name == features::schema().columns[j]);
  REQUIRE(m->head_string("arch") == "gfx950");
  REQUIRE(m->head_string("dtype") == "bf16");
  REQUIRE(m->top_n() == 10);
  REQUIRE(m->ranges().size() == 4);
  const char* const names[] = {"M", "N", "K", "batch"};
  for (std::size_t i = 0; i < 4; ++i) REQUIRE(m->ranges()[i].name == names[i]);
  REQUIRE_FALSE(m->library_manifest().empty());
  REQUIRE(m->seen_values().size() == structural_fields().size());
  const auto t = fixture("boost_model_tf32.orr");
  REQUIRE(t->num_trees() == 40);
  REQUIRE(t->head_string("dtype") == "tf32");
  const auto f = fixture("boost_model_fp8.orr");
  REQUIRE(f->num_trees() == 40);
  REQUIRE(f->head_string("dtype") == "fp8");
  REQUIRE(f->head_string("d_dtypes") == "bf16");
}

TEST_CASE("Origami Boost: malformed files are rejected", "[nn][boost]") {
  require_data();
  const auto  good = read_bytes(kData / "boost_model_bf16.orr");
  status_t    st;
  std::string detail;
  auto first_node = [&](const std::vector<std::uint8_t>& b) {
    return find_tag(b, "TREE") + 4 + 8 + 4 + 4 + 4;  // first internal node of tree 0
  };
  // Offset of RNGE entry @p name (its u32 name length), and the RNGE section offset.
  auto range_entry = [&](const std::vector<std::uint8_t>& b, const std::string& name) {
    const std::size_t sec = find_tag(b, "RNGE");
    std::uint32_t     n;
    std::memcpy(&n, &b[sec + 12], 4);
    std::size_t o = sec + 16;
    for (std::uint32_t i = 0; i < n; ++i) {
      std::uint32_t len;
      std::memcpy(&len, &b[o], 4);
      if (std::string(b.begin() + o + 4, b.begin() + o + 4 + len) == name) return o;
      o += 4 + len + 16;
    }
    FAIL("no RNGE entry " << name);
    return o;
  };
  // Add @p delta entries and @p bytes bytes to RNGE (count, section length, body length).
  auto resize_ranges = [&](std::vector<std::uint8_t>& b, int delta, std::int64_t bytes) {
    const std::size_t sec = find_tag(b, "RNGE");
    std::uint32_t     n;
    std::uint64_t     len, body;
    std::memcpy(&n, &b[sec + 12], 4);
    std::memcpy(&len, &b[sec + 4], 8);
    std::memcpy(&body, &b[16], 8);
    n += delta;
    len += bytes;
    body += bytes;
    std::memcpy(&b[sec + 12], &n, 4);
    std::memcpy(&b[sec + 4], &len, 8);
    std::memcpy(&b[16], &body, 8);
  };

  SECTION("missing file") {
    REQUIRE_FALSE(ensemble_t::load_file((kData / "no_such.orr").string(), st, detail));
    REQUIRE(st == status_t::model_missing);
  }
  SECTION("bad magic (including the earlier ORIGAMLR format)") {
    auto b = good;
    std::memcpy(b.data(), "ORIGAMLR", 8);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail == "bad magic");
  }
  SECTION("checksum") {
    auto b = good;
    b[b.size() / 2] ^= 0x5A;
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail == "checksum mismatch");
  }
  SECTION("truncated") {
    auto b = good;
    b.resize(b.size() - 7);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
  }
  SECTION("empty") {
    std::vector<std::uint8_t> b;
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
  }
  SECTION("unknown format major") {
    auto b = good;
    b[8]   = 2;
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::unsupported_format);
  }
  SECTION("categorical split") {
    auto b = good;
    b[first_node(b) + 4] |= 1;
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail.find("categorical") != std::string::npos);
  }
  SECTION("zero-as-missing split") {
    auto b = good;
    b[first_node(b) + 4] = static_cast<std::uint8_t>((b[first_node(b) + 4] & 3) | (1 << 2));
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::unsupported_format);
    REQUIRE(detail.find("zero-as-missing") != std::string::npos);
  }
  SECTION("feature index out of range") {
    auto               b   = good;
    const std::int32_t bad = 100000;
    std::memcpy(&b[first_node(b)], &bad, 4);
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
  }
  SECTION("unknown section is skipped") {
    auto                      b     = good;
    std::vector<std::uint8_t> extra = {'X', 'T', 'R', 'A', 3, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3};
    b.insert(b.end() - 32, extra.begin(), extra.end());
    std::uint64_t body = b.size() - 24 - 32;
    std::memcpy(&b[16], &body, 8);
    reseal(b);
    REQUIRE(load(b, st, detail));
    REQUIRE(st == status_t::ok);
  }
  SECTION("a num_cus range") {
    auto                      b = good;
    std::vector<std::uint8_t> e(27);
    const std::uint32_t       len = 7;
    const double              cus = 256.0;
    std::memcpy(&e[0], &len, 4);
    std::memcpy(&e[4], "num_cus", 7);
    std::memcpy(&e[11], &cus, 8);
    std::memcpy(&e[19], &cus, 8);
    const std::size_t at = range_entry(b, "batch") + 4 + 5 + 16;
    b.insert(b.begin() + at, e.begin(), e.end());
    resize_ranges(b, 1, 27);
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail == "RNGE entry num_cus: only M, N, K, batch");
  }
  SECTION("an unknown range name") {
    auto b                         = good;
    b[range_entry(b, "batch") + 5] = 'x';
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail == "RNGE entry bxtch: only M, N, K, batch");
  }
  SECTION("a range twice") {
    auto b                     = good;
    b[range_entry(b, "N") + 4] = 'M';
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail == "RNGE entry M appears twice");
  }
  SECTION("a range missing") {
    auto              b  = good;
    const std::size_t at = range_entry(b, "K");
    b.erase(b.begin() + at, b.begin() + at + 4 + 1 + 16);
    resize_ranges(b, -1, -21);
    reseal(b);
    REQUIRE_FALSE(load(b, st, detail));
    REQUIRE(st == status_t::bad_file);
    REQUIRE(detail == "RNGE lacks K");
  }
}

TEST_CASE("Origami Boost: compatibility checks", "[nn][boost]") {
  require_data();
  status_t    st;
  std::string detail;
  const auto  m  = fixture("boost_model_bf16.orr");
  const auto  id = identity_of(*m);
  const auto  tg = target_of(*m);

  REQUIRE(booster_t::create(m, tg, id, st, detail));
  REQUIRE(st == status_t::ok);

  auto rejects = [&](deployment_t t, build_identity_t i, const char* what) {
    INFO(what);
    REQUIRE_FALSE(booster_t::create(m, t, i, st, detail));
    REQUIRE(st == status_t::incompatible);
    REQUIRE(detail.find(what) != std::string::npos);
  };
  SECTION("schema mismatch") {
    auto i                = id;
    i.feature_schema_hash = "sha256:other";
    rejects(tg, i, "feature schema");
  }
  SECTION("uncommitted tree with identical sources is accepted") {
    auto i           = id;
    i.origami_commit = "b926a6d757d2b617a0f8aaeab514dfcf35237979";
    REQUIRE(booster_t::create(m, tg, i, st, detail));
    REQUIRE(st == status_t::ok);
    REQUIRE(detail.find("differs") != std::string::npos);
  }
  SECTION("same commit, different source hash is rejected") {
    auto i                = id;
    i.origami_source_hash = "sha256:other";
    REQUIRE(i.origami_commit == m->head_string("origami_commit"));
    rejects(tg, i, "origami source");
  }
  SECTION("arch mismatch") {
    auto t = tg;
    t.arch = "gfx942";
    rejects(t, id, "arch");
  }
  SECTION("dtype mismatch") {
    auto t  = tg;
    t.dtype = "tf32";
    rejects(t, id, "dtype");
  }
  SECTION("no kernel library manifest") {
    auto t            = tg;
    t.library_kernels = nullptr;
    rejects(t, id, "kernel library");
  }
  SECTION("feature outside the schema") {
    auto              b   = read_bytes(kData / "boost_model_bf16.orr");
    const std::size_t off = find_tag(b, "FEAT") + 4 + 8 + 4 + 4;  // first name's bytes
    b[off]                = 'Z';
    reseal(b);
    const auto mz = load(b, st, detail);
    REQUIRE(mz);
    REQUIRE_FALSE(booster_t::create(mz, tg, id, st, detail));
    REQUIRE(st == status_t::missing_feature);
    REQUIRE(detail.find("is not in schema") != std::string::npos);
  }
  SECTION("build identity carries the compiled schema") {
    const auto b = build_identity();
    REQUIRE(b.feature_schema_hash ==
            "sha256:5435d0bc46ede3603afe949d528c25277909fb630ad21fd825cd6cb7fe12aded");
    REQUIRE(b.feature_schema_hash == features::schema().hash);
    REQUIRE(features::schema().columns.size() == 161);
    REQUIRE(features::schema().columns == features::slot_names());
    REQUIRE(b.origami_source_hash.rfind("sha256:", 0) == 0);
  }
}

TEST_CASE("Origami Boost: evaluator matches LightGBM", "[nn][boost]") {
  require_data();
  const auto m   = fixture("boost_model_bf16.orr");
  const auto csv = read_csv(kData / "eval_rows.csv");
  REQUIRE(csv.header.size() == 1 + m->features().size());
  for (std::size_t j = 0; j < m->features().size(); ++j)
    REQUIRE(csv.header[1 + j] == m->features()[j].name);
  REQUIRE(csv.rows.size() >= 100);
  double      max_diff = 0.0;
  std::size_t nonfinite = 0;
  std::vector<double> batch;
  for (const auto& r : csv.rows) {
    std::vector<double> x(r.begin() + 1, r.end());
    nonfinite += std::count_if(x.begin(), x.end(), [](double v) { return !std::isfinite(v); });
    m->prepare(x.data());
    max_diff = std::max(max_diff, std::abs(m->evaluate(x.data()) - r[0]));
    batch.insert(batch.end(), x.begin(), x.end());
  }
  std::vector<double> out(csv.rows.size());
  m->evaluate_batch(batch.data(), out.size(), out.data());
  for (std::size_t i = 0; i < out.size(); ++i) REQUIRE(out[i] == m->evaluate(&batch[i * m->features().size()]));
  std::cout << "[boost] evaluator vs LightGBM over " << csv.rows.size() << " rows (" << nonfinite
            << " non-finite inputs): max |diff| " << max_diff << std::endl;
  REQUIRE(nonfinite > 0);
  REQUIRE(max_diff <= 1e-12);
}

TEST_CASE("Origami Boost: ranks by origami * exp(yhat) and falls back", "[nn][boost]") {
  require_data();
  status_t    st;
  std::string detail;
  const auto  m = fixture("boost_model_bf16.orr");
  const auto  b = booster_t::create(m, target_of(*m), identity_of(*m), st, detail);
  REQUIRE(b);
  const auto hw = gfx950();
  const auto p  = make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::N);

  SECTION("order") {
    auto cs = ranked_candidates(p, hw, true);
    REQUIRE(cs.candidates.size() == 10);
    const auto res = b->rank(p, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::ok);
    REQUIRE(res.yhat.size() == 10);
    std::vector<double> row(m->features().size()), key(10);
    for (std::size_t i = 0; i < 10; ++i) {
      REQUIRE(b->model_features(p, hw, cs.candidates[i], row.data()));
      m->prepare(row.data());
      REQUIRE(m->evaluate(row.data()) == res.yhat[i]);
      REQUIRE(res.corrected[i] == cs.candidates[i].origami_latency * std::exp(res.yhat[i]));
      key[i] = cs.candidates[i].origami_latency * std::exp(res.yhat[i]);
    }
    std::vector<std::size_t> want(10);
    std::iota(want.begin(), want.end(), std::size_t{0});
    std::stable_sort(want.begin(), want.end(), [&](auto x, auto y) { return key[x] < key[y]; });
    REQUIRE(res.order == want);
  }
  SECTION("only the first top_n are ranked") {
    auto cs = ranked_candidates(p, hw, true);
    cs.candidates.push_back(cs.candidates.back());
    const auto res = b->rank(p, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::ok);
    REQUIRE(res.order.size() == 11);
    REQUIRE(res.order.back() == 10);
  }
  SECTION("out of range") {
    const auto big = make_problem(100000, 4096, 4096, transpose_t::N, transpose_t::N);
    auto       cs  = ranked_candidates(big, hw, true);
    const auto res = b->rank(big, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::out_of_range);
    REQUIRE(res.detail.find("M=") != std::string::npos);
    for (std::size_t i = 0; i < res.order.size(); ++i) REQUIRE(res.order[i] == i);
    REQUIRE(b->check_problem(big, hw) == status_t::out_of_range);
  }
  SECTION("the default CU budget ranks") {
    // sm_count_target 0 resolves to the physical count (resolve_num_cus), as in Origami.
    REQUIRE(hw.N_CU == 256);
    auto pc = p;
    for (size_t budget : {size_t{0}, size_t{256}, size_t{300}}) {
      pc.num_cus = budget;
      INFO("num_cus " << budget);
      REQUIRE(b->check_problem(pc, hw) == status_t::ok);
      auto       cs  = ranked_candidates(pc, hw, true);
      const auto res = b->rank(pc, hw, cs.candidates.data(), cs.candidates.size());
      REQUIRE(res.status == status_t::ok);
    }
  }
  SECTION("a CU budget below the physical count ranks") {
    // A model has no CU range; the budget reaches it through the features.
    auto pc    = p;
    pc.num_cus = 224;
    REQUIRE(b->check_problem(pc, hw) == status_t::ok);
    auto       cs  = ranked_candidates(pc, hw, true);
    const auto res = b->rank(pc, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::ok);
  }
  SECTION("default budget features equal the explicit physical count") {
    auto cs = ranked_candidates(p, hw, true);
    auto p0 = p, p256 = p;
    p0.num_cus   = 0;
    p256.num_cus = 256;
    std::vector<double> x(features::slot_names().size()), y(x.size());
    for (const auto& c : cs.candidates) {
      REQUIRE(features::extract(p0, hw, c, x.data()));
      REQUIRE(features::extract(p256, hw, c, y.data()));
      for (std::size_t j = 0; j < x.size(); ++j)
        REQUIRE(((x[j] == y[j]) || (std::isnan(x[j]) && std::isnan(y[j]))));
    }
  }
  SECTION("d dtype not trained") {
    auto pf    = p;
    pf.d_dtype = data_type_t::Float;
    REQUIRE(b->check_problem(pf, hw) == status_t::out_of_range);
  }
  SECTION("kernel fields not provided") {
    auto       cs  = ranked_candidates(p, hw, false);
    const auto res = b->rank(p, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::missing_feature);
    REQUIRE(res.detail.find("kernel field") != std::string::npos);
  }
  SECTION("Stream-K launch not provided") {
    auto cs = ranked_candidates(p, hw, true);
    cs.candidates[4].fields.streamk.valid = false;
    const auto res = b->rank(p, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::missing_feature);
    REQUIRE(res.detail.find("candidate 4") != std::string::npos);
    REQUIRE(res.detail.find("sk_") != std::string::npos);
    for (std::size_t i = 0; i < res.order.size(); ++i) REQUIRE(res.order[i] == i);
  }
  SECTION("rejected candidate") {
    auto cs                          = ranked_candidates(p, hw, true);
    cs.candidates[3].origami_latency = std::numeric_limits<double>::max();
    const auto res = b->rank(p, hw, cs.candidates.data(), cs.candidates.size());
    REQUIRE(res.status == status_t::missing_feature);
  }
  SECTION("too few candidates") {
    auto       cs  = ranked_candidates(p, hw, true);
    const auto res = b->rank(p, hw, cs.candidates.data(), 1);
    REQUIRE(res.status == status_t::too_few_candidates);
    REQUIRE(res.order.size() == 1);
  }
}

TEST_CASE("Origami Boost: TF32 model slot, stamps and dtype gating", "[nn][boost]") {
  require_data();
  status_t    st;
  std::string detail;
  const auto  mt = fixture("boost_model_tf32.orr");
  const auto  mb = fixture("boost_model_bf16.orr");
  const auto  id = identity_of(*mt);
  const deployment_t tf32_slot{"gfx950", "tf32", &mt->library_manifest()};
  const deployment_t bf16_slot{"gfx950", "bf16", &mb->library_manifest()};

  SECTION("TF32 model in the TF32 slot is accepted") {
    REQUIRE(booster_t::create(mt, tf32_slot, id, st, detail));
    REQUIRE(st == status_t::ok);
  }
  SECTION("TF32 model in the BF16 slot is rejected") {
    REQUIRE_FALSE(booster_t::create(mt, bf16_slot, id, st, detail));
    REQUIRE(detail.find("dtype mismatch") != std::string::npos);
  }
  SECTION("BF16 model in the TF32 slot is rejected") {
    REQUIRE_FALSE(booster_t::create(mb, tf32_slot, id, st, detail));
    REQUIRE(detail.find("dtype mismatch") != std::string::npos);
  }
  SECTION("TF32 model keeps its own stamps") {
    auto i2                = id;
    i2.origami_source_hash = "sha256:other";
    REQUIRE_FALSE(booster_t::create(mt, tf32_slot, i2, st, detail));
    REQUIRE(detail.find("origami source") != std::string::npos);
    REQUIRE_FALSE(booster_t::create(mt, {"gfx942", "tf32", &mt->library_manifest()}, id, st, detail));
    REQUIRE(detail.find("arch") != std::string::npos);
  }
  SECTION("dtype gating: xf32 only, never strict f32 or bf16") {
    const auto bt = booster_t::create(mt, tf32_slot, id, st, detail);
    const auto bb = booster_t::create(mb, bf16_slot, id, st, detail);
    REQUIRE(bt);
    REQUIRE(bb);
    const auto hw     = gfx950();
    const auto xf32   = make_tf32_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T);
    const auto strict = make_tf32_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T, 1, false);
    const auto bf16   = make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T);
    REQUIRE(bt->check_problem(xf32, hw) == status_t::ok);
    REQUIRE(bt->check_problem(strict, hw, &detail) == status_t::out_of_range);
    REQUIRE(detail.find("not tf32") != std::string::npos);
    REQUIRE(bt->check_problem(bf16, hw) == status_t::out_of_range);
    REQUIRE(bb->check_problem(xf32, hw, &detail) == status_t::out_of_range);
    REQUIRE(detail.find("not bf16") != std::string::npos);
    REQUIRE(bb->check_problem(bf16, hw) == status_t::ok);
  }
  SECTION("TF32 ranking orders by origami * exp(yhat)") {
    const auto bt = booster_t::create(mt, tf32_slot, id, st, detail);
    REQUIRE(bt);
    const auto hw = gfx950();
    const auto p  = make_tf32_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T);
    auto       cs = ranked_candidates(p, hw, true);
    const auto res = bt->rank(p, hw, cs.candidates.data(), cs.candidates.size());
    INFO(res.detail);
    REQUIRE(res.status == status_t::ok);
    std::vector<double> key(res.yhat.size());
    for (std::size_t i = 0; i < key.size(); ++i)
      key[i] = std::log(cs.candidates[i].origami_latency) + res.yhat[i];
    std::vector<std::size_t> want(key.size());
    std::iota(want.begin(), want.end(), std::size_t{0});
    std::stable_sort(want.begin(), want.end(), [&](auto x, auto y) { return key[x] < key[y]; });
    REQUIRE(std::equal(want.begin(), want.end(), res.order.begin()));
  }
}

TEST_CASE("Origami Boost: FP8 model slot, stamps and dtype gating", "[nn][boost]") {
  require_data();
  status_t    st;
  std::string detail;
  const auto  mf = fixture("boost_model_fp8.orr");
  const auto  mb = fixture("boost_model_bf16.orr");
  const auto  id = identity_of(*mf);
  const deployment_t fp8_slot{"gfx950", "fp8", &mf->library_manifest()};
  const deployment_t bf16_slot{"gfx950", "bf16", &mb->library_manifest()};
  const auto         hw = gfx950();
  const auto         f8 = make_fp8_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T);

  SECTION("FP8 model only in the FP8 slot") {
    REQUIRE(booster_t::create(mf, fp8_slot, id, st, detail));
    REQUIRE(st == status_t::ok);
    REQUIRE_FALSE(booster_t::create(mf, bf16_slot, id, st, detail));
    REQUIRE(detail.find("dtype mismatch") != std::string::npos);
    REQUIRE_FALSE(booster_t::create(mb, fp8_slot, id, st, detail));
    REQUIRE(detail.find("dtype mismatch") != std::string::npos);
    auto i2                = id;
    i2.origami_source_hash = "sha256:other";
    REQUIRE_FALSE(booster_t::create(mf, fp8_slot, i2, st, detail));
    REQUIRE(detail.find("origami source") != std::string::npos);
  }
  SECTION("dtype gating: F8BS only") {
    const auto bf = booster_t::create(mf, fp8_slot, id, st, detail);
    const auto bb = booster_t::create(mb, bf16_slot, id, st, detail);
    REQUIRE(bf);
    REQUIRE(bb);
    REQUIRE(bf->check_problem(f8, hw) == status_t::ok);
    auto f32_out = f8;
    f32_out.c_dtype = f32_out.d_dtype = data_type_t::Float;
    auto mixed      = f8;
    mixed.b_dtype   = data_type_t::BFloat8;
    auto fnuz       = f8;
    fnuz.a_dtype = fnuz.b_dtype = data_type_t::Float8_fnuz;
    auto mx         = f8;
    mx.a_mx_block_size = mx.b_mx_block_size = 32;
    auto bf16_math  = f8;
    bf16_math.mi_dtype = data_type_t::BFloat16;
    for (const auto& p : {f32_out, mixed, fnuz, mx, bf16_math}) {
      REQUIRE(bf->check_problem(p, hw, &detail) == status_t::out_of_range);
      REQUIRE(detail.find("not fp8") != std::string::npos);
    }
    REQUIRE(bf->check_problem(make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::T), hw) ==
            status_t::out_of_range);
    REQUIRE(bb->check_problem(f8, hw, &detail) == status_t::out_of_range);
    REQUIRE(detail.find("not bf16") != std::string::npos);
  }
  SECTION("FP8 ranking orders by origami * exp(yhat)") {
    const auto bf = booster_t::create(mf, fp8_slot, id, st, detail);
    REQUIRE(bf);
    auto       cs  = ranked_candidates(f8, hw, true);
    const auto res = bf->rank(f8, hw, cs.candidates.data(), cs.candidates.size());
    INFO(res.detail);
    REQUIRE(res.status == status_t::ok);
    std::vector<double> key(res.yhat.size());
    for (std::size_t i = 0; i < key.size(); ++i)
      key[i] = std::log(cs.candidates[i].origami_latency) + res.yhat[i];
    std::vector<std::size_t> want(key.size());
    std::iota(want.begin(), want.end(), std::size_t{0});
    std::stable_sort(want.begin(), want.end(), [&](auto x, auto y) { return key[x] < key[y]; });
    REQUIRE(std::equal(want.begin(), want.end(), res.order.begin()));
  }
  SECTION("features take 1-byte A/B and the 2-byte D") {
    // 320 tiles of 256x256 on 256 workgroups: Stream-K partial tiles.
    const auto f8p  = make_fp8_problem(5120, 4096, 4096, transpose_t::N, transpose_t::T);
    const auto bf16 = make_problem(5120, 4096, 4096, transpose_t::N, transpose_t::T);
    const auto cs   = ranked_candidates(bf16, hw, true);
    auto       at   = [](const char* name) {
      return static_cast<std::size_t>(features::slot_index(name));
    };
    std::vector<double> x8(features::slot_names().size()), x16(x8.size());
    int                 with_partials = 0;
    for (const auto& c : cs.candidates) {
      candidate_t c8 = c, c16 = c;
      c8.origami_latency  = gemm::compute_total_latency(f8p, hw, *c.config);
      c16.origami_latency = gemm::compute_total_latency(bf16, hw, *c.config);
      REQUIRE(features::extract(f8p, hw, c8, x8.data()));
      REQUIRE(features::extract(bf16, hw, c16, x16.data()));
      REQUIRE(x8[at("d_lds_bytes_eff")] == x16[at("d_lds_bytes_eff")] / 2);
      REQUIRE(x8[at("f_grvw_bytes_a")] == x16[at("f_grvw_bytes_a")] / 2);
      REQUIRE(x8[at("f_gwvw_bytes_d")] == x16[at("f_gwvw_bytes_d")]);
      REQUIRE(x8[at("sk_fixup_bytes_over_output")] == x16[at("sk_fixup_bytes_over_output")]);
      with_partials += x8[at("sk_fixup_bytes_over_output")] > 0;
    }
    REQUIRE(with_partials > 0);
  }
}

namespace {

// Feature-parity table: problem and library fields per row, then pred_cycles and the schema
// columns as the training feature builder computed them.
struct parity_table_t {
  std::map<std::string, double> hw;
  std::vector<std::string>      header;
  std::vector<std::vector<double>> rows;
  std::size_t col(const std::string& name) const {
    const auto it = std::find(header.begin(), header.end(), name);
    REQUIRE(it != header.end());
    return static_cast<std::size_t>(it - header.begin());
  }
};

parity_table_t read_parity(const fs::path& p) {
  parity_table_t t;
  std::ifstream  f(p);
  std::string    line;
  REQUIRE(std::getline(f, line));
  REQUIRE(line.rfind("# hw ", 0) == 0);
  // Flat JSON object of numbers (string values are skipped).
  for (std::size_t at = line.find('"'); at != std::string::npos; at = line.find('"', at + 1)) {
    const std::size_t end = line.find('"', at + 1);
    if (end == std::string::npos || line.compare(end + 1, 2, ": ") != 0) break;
    const std::string key   = line.substr(at + 1, end - at - 1);
    const char*       begin = line.c_str() + end + 3;
    char*             stop  = nullptr;
    const double      value = std::strtod(begin, &stop);
    if (stop != begin) t.hw[key] = value;
    at = end;
    if (*begin == '"') at = line.find('"', end + 4);
  }
  for (const char* key : {"N_CU", "lds_capacity", "rf_capacity", "L2_capacity", "compute_clock_khz"})
    REQUIRE(t.hw.count(key) == 1);
  REQUIRE(std::getline(f, line));
  std::stringstream hs(line);
  for (std::string item; std::getline(hs, item, ',');) t.header.push_back(item);
  while (std::getline(f, line)) {
    std::vector<double> r;
    std::stringstream   ss(line);
    for (std::string item; std::getline(ss, item, ',');) r.push_back(std::strtod(item.c_str(), nullptr));
    REQUIRE(r.size() == t.header.size());
    t.rows.push_back(std::move(r));
  }
  return t;
}

// hardware_t from the table's header: the arch constants, then every field the table reports
// (a device's get_hardware_for_device values when the table came from one).
hardware_t hardware_from_table(const parity_table_t& t) {
  auto has = [&](const char* k) { return t.hw.count(k) == 1; };
  auto sz  = [&](const char* k) { return static_cast<size_t>(t.hw.at(k)); };
  auto hw  = hardware_t::get_hardware_for_arch(
      hardware_t::architecture_t::gfx950, sz("N_CU"), sz("lds_capacity"), sz("rf_capacity"),
      sz("L2_capacity"), static_cast<int>(t.hw.at("compute_clock_khz")),
      has("pci_chip_id") ? std::make_optional(static_cast<int>(t.hw.at("pci_chip_id")))
                         : std::nullopt);
  if (has("CU_per_L2")) hw.CU_per_L2 = sz("CU_per_L2");
  if (has("NUM_XCD")) hw.NUM_XCD = sz("NUM_XCD");
  if (has("compute_clock_ghz")) hw.compute_clock_ghz = t.hw.at("compute_clock_ghz");
  if (has("parallel_mi_cu")) hw.parallel_mi_cu = sz("parallel_mi_cu");
  if (has("mem1_perf_ratio")) hw.mem1_perf_ratio = t.hw.at("mem1_perf_ratio");
  if (has("mem2_perf_ratio")) hw.mem2_perf_ratio = t.hw.at("mem2_perf_ratio");
  if (has("mem3_perf_ratio")) hw.mem3_perf_ratio = t.hw.at("mem3_perf_ratio");
  if (has("mem_bw_coef0"))
    hw.mem_bw_per_wg_coefficients = {t.hw.at("mem_bw_coef0"), t.hw.at("mem_bw_coef1"),
                                     t.hw.at("mem_bw_coef2")};
  if (has("wavefront_size")) REQUIRE(hw.wavefront_size() == sz("wavefront_size"));
  return hw;
}

data_type_t dtype_code(double v) {
  return v == 3.0   ? data_type_t::Float8
         : v == 2.0 ? data_type_t::XFloat32
         : v == 1.0 ? data_type_t::Float
                    : data_type_t::BFloat16;
}

// problem_t of a table row: BF16 unless the table carries dtype codes (0 BFloat16, 1 Float,
// 2 XFloat32, 3 Float8) for A/B, C/D and the MI input type.
problem_t problem_from_row(const parity_table_t& t, const std::vector<double>& r) {
  auto      v = [&](const char* name) { return r[t.col(name)]; };
  problem_t p = make_problem(static_cast<size_t>(v("M")), static_cast<size_t>(v("N")),
                             static_cast<size_t>(v("K")),
                             v("transA") != 0 ? transpose_t::T : transpose_t::N,
                             v("transB") != 0 ? transpose_t::T : transpose_t::N,
                             static_cast<size_t>(v("batch")));
  p.num_cus = static_cast<size_t>(v("num_cus"));
  if (std::find(t.header.begin(), t.header.end(), "ab_dtype") != t.header.end()) {
    p.a_dtype = p.b_dtype = dtype_code(v("ab_dtype"));
    p.c_dtype = p.d_dtype = dtype_code(v("cd_dtype"));
    p.mi_dtype            = dtype_code(v("mi_dtype"));
  }
  return p;
}

// origami::config_t as Serialization/PredictionLibrary.hpp builds it from a solution record.
config_t config_from_record(const parity_table_t& t, const std::vector<double>& r) {
  auto v = [&](const char* name) { return static_cast<long>(r[t.col(name)]); };
  config_t c;
  c.mt = {static_cast<size_t>(v("mt_m")), static_cast<size_t>(v("mt_n")),
          static_cast<size_t>(v("mt_k"))};
  if (v("mi_m") == 0 && v("mi_n") == 0 && v("mi_k") == 0) c.mi = {1, 1, 64};
  else c.mi = {static_cast<size_t>(v("mi_m")), static_cast<size_t>(v("mi_n")),
               static_cast<size_t>(v("mi_k"))};
  c.hand_optimized_main_loop  = v("cms") > 0;
  c.subtile                   = v("subtile") != 0;
  c.occupancy                 = std::max(static_cast<int>(v("occupancy")), 1);
  c.workgroup_mapping         = static_cast<int>(v("wgm"));
  c.cache_hints_a             = static_cast<int>(v("nt_a"));
  c.cache_hints_b             = static_cast<int>(v("nt_b"));
  c.cache_hints_d             = static_cast<int>(v("nt_d"));
  c.non_temporal_a_available  = v("nt_tile_a") != 0;  // per macro tile (memoize_non_temporal_availability)
  c.non_temporal_b_available  = v("nt_tile_b") != 0;
  c.workspace_size            = std::numeric_limits<size_t>::max();
  c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
  c.stream_k                  = static_cast<int>(v("stream_k_mode"));
  c.grvw_a                    = static_cast<size_t>(v("grvw_a"));
  c.grvw_b                    = static_cast<size_t>(v("grvw_b"));
  c.gwvw_d                    = static_cast<size_t>(v("gwvw_d"));
  c.vector_width_a            = static_cast<int>(v("vw_a"));
  c.vector_width_b            = static_cast<int>(v("vw_b"));
  c.cluster_dim = {static_cast<size_t>(v("cd_m")), static_cast<size_t>(v("cd_n")),
                   static_cast<size_t>(v("cd_k"))};
  tensile_params_t tp;
  tp.local_split_u        = static_cast<int>(v("lsu"));
  tp.direct_to_vgpr_a     = v("dtv_a") != 0;
  tp.direct_to_vgpr_b     = v("dtv_b") != 0;
  tp.direct_to_lds_a      = v("dtl_a") != 0;
  tp.direct_to_lds_b      = v("dtl_b") != 0;
  tp.wavefront_size       = static_cast<int>(v("wavefront_size"));
  tp.wave_group_m         = static_cast<int>(v("miwg_m"));
  tp.wave_group_n         = static_cast<int>(v("miwg_n"));
  tp.prefetch_global_read = static_cast<int>(v("pgr"));
  tp.one_lds_buffer       = v("one_lds_buffer") != 0;
  tp.source_swap          = v("source_swap") != 0;
  tp.unroll_major_lds_a   = static_cast<int>(v("ulds_a"));
  tp.unroll_major_lds_b   = static_cast<int>(v("ulds_b"));
  tp.lds_transpose_read_a = v("ldstr_a") != 0;
  tp.lds_transpose_read_b = v("ldstr_b") != 0;
  c.backend               = tp;
  return c;
}

// kernel_fields_t of a row: the coalesced-load counts and the Stream-K launch (ski_* columns,
// reduction coded as reduction_t, -1 = none).
kernel_fields_t fields_from_row(const parity_table_t& t, const std::vector<double>& r) {
  auto            v = [&](const char* name) { return r[t.col(name)]; };
  auto            u = [&](const char* name) { return static_cast<std::uint64_t>(v(name)); };
  kernel_fields_t kf;
  kf.num_loads_coalesced_a = static_cast<int>(v("nlc_a"));
  kf.num_loads_coalesced_b = static_cast<int>(v("nlc_b"));
  auto& sk                 = kf.streamk;
  sk.valid                 = true;
  sk.is_streamk            = v("ski_streamk") != 0;
  sk.final_grid            = u("ski_final_grid");
  sk.tiles                 = u("ski_tiles");
  sk.sk_tiles              = u("ski_sk_tiles");
  sk.sk_split              = u("ski_sk_split");
  sk.reduction             = v("ski_reduction") < 0
                                 ? reduction_t::none
                                 : static_cast<reduction_t>(static_cast<std::uint32_t>(v("ski_reduction")));
  sk.is_dynamic                = v("ski_is_dynamic") != 0;
  sk.dp_only                   = v("ski_dp_only") != 0;
  sk.iters_per_tile            = u("ski_iters_per_tile");
  sk.mt_x                      = u("ski_mt_x");
  sk.mt_y                      = u("ski_mt_y");
  sk.workspace_size_per_elem_c = u("ski_ws_per_elem_c");
  sk.cu_occupancy              = static_cast<int>(v("ski_cu_occupancy"));
  sk.cu_count                  = u("ski_cu_count");
  return kf;
}

}  // namespace

TEST_CASE("Origami Boost: Stream-K launch columns (schema v4)", "[nn][boost]") {
  // Partial tiles of the static split by brute force (no whole-tile shortcut), as
  // Schema v4's streamk_partials states it: distinct cut tiles, every cut a partial.
  auto brute = [](std::uint64_t sk, std::uint64_t iters, std::uint64_t G) {
    streamk_partials_t out;
    if (G == 0 || sk == 0 || iters == 0) return out;
    const std::uint64_t T = sk * iters, W = T / G, E = T % G;
    std::vector<std::uint64_t> cut;
    for (std::uint64_t g = 1; g < G; ++g) {
      const std::uint64_t s = g * W + std::min(g, E);
      if (s % iters) cut.push_back(s / iters);
    }
    out.partials = cut.size();
    std::sort(cut.begin(), cut.end());
    out.split_tiles =
        static_cast<std::uint64_t>(std::unique(cut.begin(), cut.end()) - cut.begin());
    return out;
  };
  streamk_launch_t base;
  base.valid = base.is_streamk = true;
  base.reduction               = reduction_t::tree;
  base.sk_split                = 1;

  SECTION("static split: hand example and brute force") {
    auto d = base;
    d.tiles = d.sk_tiles = 10;
    d.iters_per_tile     = 4;
    d.final_grid         = 3;  // T 40, W 13, E 1: starts 14 and 27 cut tiles 3 and 6
    auto pt              = streamk_partials(d);
    REQUIRE(pt.partials == 2);
    REQUIRE(pt.split_tiles == 2);
    int with_cuts = 0, shortcut = 0;
    for (std::uint64_t sk : {1, 2, 7, 64, 255, 256, 1000})
      for (std::uint64_t iters : {1, 2, 3, 8, 13, 64})
        for (std::uint64_t G : {1, 2, 3, 100, 256, 512, 999, 1024}) {
          d.tiles = d.sk_tiles = sk;
          d.iters_per_tile     = iters;
          d.final_grid         = G;
          const auto mine = streamk_partials(d), ref = brute(sk, iters, G);
          INFO("sk " << sk << " iters " << iters << " G " << G);
          REQUIRE(mine.partials == ref.partials);
          REQUIRE(mine.split_tiles == ref.split_tiles);
          with_cuts += ref.partials > 0;
          shortcut += (sk * iters) % G == 0 && ((sk * iters) / G) % iters == 0;
        }
    REQUIRE(with_cuts > 50);
    REQUIRE(shortcut > 20);
  }
  SECTION("parallel, dynamic, DP-only and non-Stream-K launches") {
    auto d       = base;
    d.tiles      = 40;
    d.sk_tiles   = 3;
    d.sk_split   = 3;
    d.final_grid = 120;
    d.reduction  = reduction_t::parallel;
    auto pt      = streamk_partials(d);
    REQUIRE(pt.partials == 120);
    REQUIRE(pt.split_tiles == 40);
    d.is_dynamic = true;
    REQUIRE(streamk_partials(d).partials == 0);
    d            = base;
    d.tiles = d.sk_tiles = 10;
    d.iters_per_tile     = 4;
    d.final_grid         = 3;
    d.dp_only            = true;
    REQUIRE(streamk_partials(d).partials == 0);
    d.dp_only    = false;
    d.is_streamk = false;
    REQUIRE(streamk_partials(d).partials == 0);
  }
  SECTION("columns") {
    const auto hw = gfx950();
    auto       cs = ranked_candidates(make_problem(4096, 4096, 4096, transpose_t::N, transpose_t::N),
                                      hw, true);
    const auto p  = make_problem(1000, 3000, 4096, transpose_t::N, transpose_t::T, 2);
    const auto ix = [](const char* name) {
      const int i = features::slot_index(name);
      REQUIRE(i >= 0);
      return static_cast<std::size_t>(i);
    };
    REQUIRE(features::slot_names().size() == 161);
    REQUIRE(features::schema().columns.back() == "sk_timesteps_launched");
    std::vector<double>       f(features::slot_names().size());
    std::vector<std::uint8_t> avail(f.size());
    candidate_t               cand = cs.candidates[0];

    cand.fields.streamk.valid      = true;
    cand.fields.streamk.is_streamk = false;
    REQUIRE(features::extract(p, hw, cand, f.data(), nullptr, avail.data()));
    REQUIRE(f[ix("sk_grid_over_tiles")] == 1.0);
    for (const char* c : {"sk_split_tiles_frac", "sk_fixup_bytes_over_output",
                          "sk_num_wgs_launched", "sk_timesteps_launched"})
      REQUIRE(f[ix(c)] == 0.0);
    REQUIRE(std::all_of(avail.begin(), avail.end(), [](auto a) { return a == 1; }));

    cand.fields.streamk.valid = false;
    REQUIRE(features::extract(p, hw, cand, f.data(), nullptr, avail.data()));
    for (std::size_t j = 0; j < f.size(); ++j)
      REQUIRE(avail[j] == (features::slot_names()[j].rfind("sk_", 0) == 0 ? 0 : 1));

    // Static launch with cuts; bf16 D (2 bytes), occupancy 2.
    auto& sk          = cand.fields.streamk;
    sk                = base;
    sk.tiles          = 10;
    sk.sk_tiles       = 10;
    sk.iters_per_tile = 4;
    sk.final_grid     = 3;
    sk.mt_x           = 256;
    sk.mt_y           = 128;
    sk.workspace_size_per_elem_c = 4;
    sk.cu_occupancy              = 2;
    sk.cu_count                  = 256;
    REQUIRE(features::extract(p, hw, cand, f.data()));
    REQUIRE(f[ix("sk_grid_over_tiles")] == 3.0 / 10.0);
    REQUIRE(f[ix("sk_split_tiles_frac")] == 2.0 / 10.0);
    REQUIRE(f[ix("sk_fixup_bytes_over_output")] ==
            ((2.0 * 2.0) * (256.0 * 128.0 * 4.0)) / (1000.0 * 3000.0 * 2.0 * 2.0));
    REQUIRE(f[ix("sk_num_wgs_launched")] == 3.0);
    REQUIRE(f[ix("sk_timesteps_launched")] == 1.0 * 10.0 / 3.0);
    // Waves count resident slots on the CU budget: smCountTarget if 0 < it < N_CU.
    sk.final_grid = 1000;
    sk.tiles = sk.sk_tiles = 1000;
    for (auto [budget, slots] : std::vector<std::pair<size_t, double>>{
             {0, 512.0}, {224, 448.0}, {256, 512.0}, {300, 512.0}}) {
      auto pb    = p;
      pb.num_cus = budget;
      REQUIRE(features::extract(pb, hw, cand, f.data()));
      INFO("budget " << budget);
      REQUIRE(f[ix("sk_timesteps_launched")] == std::ceil(1000.0 / slots) * 1000.0 / 1000.0);
    }
  }
}

TEST_CASE("Origami Boost: features match the training feature builder", "[nn][boost]") {
  // parity_reference_v4.csv / _v4_tf32.csv / _v4_fp8.csv: samples of the v4 reference vectors at
  // this Origami (device hardware, recorded Origami top 10s, default CU budget, per-macro-tile
  // non-temporal availability per config, each candidate's Stream-K launch from hipBLASLt's
  // computeStreamKDecisions).
  fs::path path;
  SECTION("reference sample, BF16") { path = kData / "parity_reference_v4.csv"; }
  SECTION("reference sample, TF32") { path = kData / "parity_reference_v4_tf32.csv"; }
  SECTION("reference sample, FP8") { path = kData / "parity_reference_v4_fp8.csv"; }
  REQUIRE(fs::is_regular_file(path));
  const auto               t  = read_parity(path);
  const auto               hw = hardware_from_table(t);
  const std::size_t        f0 = t.col("f_active_cus");
  std::vector<std::string> names(t.header.begin() + static_cast<std::ptrdiff_t>(f0), t.header.end());
  REQUIRE(names == features::schema().columns);

  std::vector<double> max_rel(names.size(), 0.0);
  std::vector<bool>   integral(names.size(), true);
  double              max_pred_rel = 0.0;
  std::vector<double> out(features::slot_names().size());
  for (const auto& r : t.rows) {
    auto         v    = [&](const char* name) { return r[t.col(name)]; };
    problem_t    p    = problem_from_row(t, r);
    const auto   c    = config_from_record(t, r);
    const double pred = gemm::compute_total_latency(p, hw, c);
    max_pred_rel      = std::max(max_pred_rel, std::abs(pred - v("pred_cycles")) / v("pred_cycles"));
    candidate_t cand;
    cand.config          = &c;
    cand.origami_latency = pred;
    cand.fields          = fields_from_row(t, r);
    REQUIRE(features::extract(p, hw, cand, out.data()));
    for (std::size_t j = 0; j < names.size(); ++j) {
      const double x    = out[j];
      const double mine = std::isfinite(x) ? x : 0.0;  // schema: non-finite -> 0
      const double ref  = r[f0 + j];
      if (ref != std::floor(ref) || mine != std::floor(mine)) integral[j] = false;
      const double rel = mine == ref ? 0.0 : std::abs(mine - ref) / std::max(std::abs(ref), 1e-300);
      max_rel[j]       = std::max(max_rel[j], rel);
    }
  }
  int failing = 0;
  std::cout << "[boost parity] " << t.rows.size() << " rows from " << path.filename()
            << " (schema " << features::schema().id << "); pred_cycles max rel diff "
            << max_pred_rel << "\n";
  for (std::size_t j = 0; j < names.size(); ++j) {
    const bool ok = integral[j] ? max_rel[j] == 0.0 : max_rel[j] <= 1e-6;
    if (!ok) {
      ++failing;
      std::cout << "  MISMATCH " << names[j] << (integral[j] ? " (integer)" : "") << " max rel "
                << max_rel[j] << "\n";
    }
  }
  std::size_t worst = 0, n_int = 0;
  for (std::size_t j = 0; j < names.size(); ++j) {
    n_int += integral[j];
    if (!integral[j] && (integral[worst] || max_rel[j] > max_rel[worst])) worst = j;
  }
  std::cout << "  " << names.size() - failing << "/" << names.size() << " columns match (" << n_int
            << " integer-valued, exact; worst float column " << names[worst] << " max rel "
            << max_rel[worst] << ")" << std::endl;
  REQUIRE(max_pred_rel <= 1e-12);
  REQUIRE(failing == 0);
}

TEST_CASE("Origami Boost: tile-latency terms rebuild compute_tile_latency", "[nn][boost]") {
  // The breakdown columns (v2_*) map compute_gemm_intermediates' tile terms. On the reference
  // configs and on a sweep that exercises every term, the terms must add up to Origami's own
  // compute_tile_latency, and the epilogue column (the residual of the tile total) must equal
  // the epilogue term.
  auto check = [](const problem_t& p, const hardware_t& hw, const config_t& c, double& worst,
                  double& worst_epi, std::array<int, 5>& exercised) {
    const double total = gemm::compute_total_latency(p, hw, c);
    if (!(total < std::numeric_limits<double>::max())) return false;
    const auto   I       = gemm::compute_gemm_intermediates(p, hw, c, total);
    const auto&  t       = I.tile;
    const double origami = gemm::compute_tile_latency(p, hw, c, I.context);
    REQUIRE(t.L_tile_total == origami);
    const double rebuilt =
        (t.L_prologue + t.L_mainloop + t.L_epilogue + t.L_lsu_reduce) * t.weight_tile_total;
    worst     = std::max(worst, std::abs(rebuilt - origami) / origami);
    worst_epi = std::max(worst_epi, std::abs(t.L_epilogue - I.L_epilogue_residual) / origami);
    exercised[0] += t.compute_utilization < 1.0;
    exercised[1] += t.L_tail > 0.0;
    exercised[2] += t.L_single_lds_buffer > 0.0;
    exercised[3] += t.L_lsu_reduce > 0.0;
    exercised[4] += t.weight_tile_total < 1.0;
    return true;
  };
  auto report = [](const char* what, int n, double worst, double worst_epi,
                   const std::array<int, 5>& ex) {
    std::cout << "[boost breakdown] " << what << ": " << n << " configs, max rel |terms - "
              << "compute_tile_latency| " << worst << ", epilogue term vs column " << worst_epi
              << "; non-trivial: compute_utilization " << ex[0] << ", L_tail " << ex[1]
              << ", L_single_lds_buffer " << ex[2] << ", L_lsu_reduce " << ex[3]
              << ", tile-weight discount " << ex[4] << std::endl;
  };

  SECTION("reference vectors") {
    for (const fs::path path : {kData / "parity_reference_v4.csv", kData / "parity_reference_v4_tf32.csv",
                                kData / "parity_reference_v4_fp8.csv"}) {
      REQUIRE(fs::is_regular_file(path));
      const auto         t  = read_parity(path);
      const auto         hw = hardware_from_table(t);
      double             worst = 0.0, worst_epi = 0.0;
      std::array<int, 5> ex{};
      int                n = 0;
      for (const auto& row : t.rows) {
        problem_t  p = problem_from_row(t, row);
        const auto c = config_from_record(t, row);
        n += check(p, hw, c, worst, worst_epi, ex);
      }
      report(path.filename().c_str(), n, worst, worst_epi, ex);
      REQUIRE(n > 0);
      REQUIRE(worst <= 1e-12);
      REQUIRE(worst_epi <= 1e-12);
    }
  }
  SECTION("synthetic sweep exercising every helper") {
    const auto         hw = gfx950();
    double             worst = 0.0, worst_epi = 0.0;
    std::array<int, 5> ex{}, ex_tf32{};
    int                n = 0, n_tf32 = 0;
    double             worst_tf32 = 0.0, worst_epi_tf32 = 0.0;
    const std::vector<std::array<size_t, 4>> problems = {
        {4096, 4096, 4096, 1}, {8192, 8192, 4127, 1}, {7, 4096, 2050, 1},
        {4096, 9, 777, 1},     {1024, 1024, 8192, 4}, {300, 5000, 2560, 2}};
    const std::vector<std::array<size_t, 6>> tiles = {
        {256, 256, 64, 16, 16, 32}, {128, 128, 64, 16, 16, 32}, {256, 128, 128, 32, 32, 16},
        {64, 64, 256, 16, 16, 32},  {128, 256, 32, 16, 16, 32}, {256, 256, 32, 16, 16, 32}};
    for (bool tf32 : {false, true})
    for (const auto& pr : problems)
      for (auto ta : {transpose_t::N, transpose_t::T})
        for (const auto& tl : tiles)
          for (int lsu : {1, 2, 4})
            for (bool one_lds : {false, true})
              for (int occ : {1, 2}) {
                problem_t p = tf32 ? make_tf32_problem(pr[0], pr[1], pr[2], ta, transpose_t::N, pr[3])
                                   : make_problem(pr[0], pr[1], pr[2], ta, transpose_t::N, pr[3]);
                config_t  c = make_config(tl[0], tl[1], tl[2], tl[3], tl[4], tl[5], false, 8, occ,
                                          0, 0, 0);
                c.workspace_size            = std::numeric_limits<size_t>::max();
                c.workspace_size_per_elem_c = std::numeric_limits<size_t>::max();
                auto& tp                    = c.tensile();
                tp.local_split_u            = lsu;
                tp.one_lds_buffer           = one_lds;
                tp.prefetch_global_read     = 2;
                tp.wave_group_m             = 2;
                tp.wave_group_n             = 2;
                c.gwvw_d                    = static_cast<size_t>(lsu == 4 ? 8 : 4);
                if (tf32)
                  n_tf32 += check(p, hw, c, worst_tf32, worst_epi_tf32, ex_tf32);
                else
                  n += check(p, hw, c, worst, worst_epi, ex);
              }
    report("synthetic bf16", n, worst, worst_epi, ex);
    report("synthetic tf32", n_tf32, worst_tf32, worst_epi_tf32, ex_tf32);
    REQUIRE(n > 100);
    REQUIRE(n_tf32 > 100);
    REQUIRE(worst <= 1e-12);
    REQUIRE(worst_epi <= 1e-12);
    REQUIRE(worst_tf32 <= 1e-12);
    REQUIRE(worst_epi_tf32 <= 1e-12);
    for (int i = 0; i < 4; ++i) REQUIRE(ex[i] > 0);
    for (int e : ex_tf32) REQUIRE(e > 0);
    REQUIRE(ex[4] == 0);  // BF16 never takes the tile-weight discount
  }
}

namespace {

const fs::path kShipped     = ORIGAMI_BOOST_SHIPPED_MODELS;
const fs::path kOrigamiRoot = ORIGAMI_SOURCE_ROOT;
const fs::path kDigests     = kData / "feature_digests.txt";

constexpr const char* kParitySamples[] = {"parity_reference_v4.csv", "parity_reference_v4_tf32.csv",
                                          "parity_reference_v4_fp8.csv"};

// "<sample>\t<column>" -> sha256 of the column over the sample's rows, as the model reads it
// (non-finite -> 0, -0 -> +0; binary64, little-endian, row order). Column pred_cycles is Origami's
// latency, which the re-rank multiplies by exp(yhat).
using digests_t = std::map<std::string, std::string>;

digests_t feature_digests() {
  digests_t   out;
  const auto& columns = features::schema().columns;
  for (const char* sample : kParitySamples) {
    const auto t  = read_parity(kData / sample);
    const auto hw = hardware_from_table(t);
    std::vector<std::vector<std::uint8_t>> bytes(columns.size() + 1);
    std::vector<double>                    row(features::slot_names().size());
    auto put = [](std::vector<std::uint8_t>& b, double x) {
      if (!std::isfinite(x) || x == 0.0) x = 0.0;
      std::uint8_t raw[sizeof(double)];
      std::memcpy(raw, &x, sizeof raw);
      b.insert(b.end(), raw, raw + sizeof raw);
    };
    for (const auto& r : t.rows) {
      const problem_t p = problem_from_row(t, r);
      const auto      c = config_from_record(t, r);
      candidate_t     cand;
      cand.config          = &c;
      cand.origami_latency = gemm::compute_total_latency(p, hw, c);
      cand.fields          = fields_from_row(t, r);
      REQUIRE(features::extract(p, hw, cand, row.data()));
      put(bytes[0], cand.origami_latency);
      for (std::size_t j = 0; j < columns.size(); ++j)
        put(bytes[j + 1], row[static_cast<std::size_t>(features::slot_index(columns[j]))]);
    }
    for (std::size_t j = 0; j <= columns.size(); ++j)
      out[std::string(sample) + "\t" + (j == 0 ? "pred_cycles" : columns[j - 1])] =
          sha256_hex(bytes[j].data(), bytes[j].size());
  }
  return out;
}

void write_digests(const fs::path& path, const digests_t& d) {
  std::ofstream f(path);
  f << "# Origami Boost features the shipped models (data/nn/boost) were validated with:\n"
       "# per parity sample and column, sha256 of the column as the model reads it\n"
       "# (test_boost.cpp, feature_digests). Recorded by origami-tests \"[shipped]\" with\n"
       "# ORIGAMI_BOOST_RECORD_FEATURE_DIGESTS=<this file> when the models are retrained;\n"
       "# a re-stamp leaves it unchanged.\n"
    << "feature_schema_hash\t" << features::schema().hash << "\n";
  for (const auto& [key, hash] : d) f << key << "\t" << hash << "\n";
  REQUIRE(f.good());
}

bool read_digests(const fs::path& path, std::string& schema_hash, digests_t& d) {
  std::ifstream f(path);
  if (!f) return false;
  for (std::string line; std::getline(f, line);) {
    if (line.empty() || line[0] == '#') continue;
    const auto tab = line.rfind('\t');
    if (tab == std::string::npos) return false;
    if (line.rfind("feature_schema_hash\t", 0) == 0) schema_hash = line.substr(tab + 1);
    else d[line.substr(0, tab)] = line.substr(tab + 1);
  }
  return true;
}

}  // namespace

// Every shipped model (data/nn/boost/<arch>/*.orr) must pass the checks this build of Origami
// makes on its own: readable file, arch, dtype, feature schema and Origami source hash (the kernel
// library is checked by tensilelite-tests' OrigamiBoostLibrary.ShippedModelsAcceptBuiltLibrary).
// On a source-hash mismatch the parity samples' features decide what the fix is: bit-identical,
// re-stamp; changed, retrain.
TEST_CASE("Origami Boost: shipped models load in this build", "[nn][boost][shipped]") {
  if (!fs::is_directory(kShipped)) SKIP("no shipped Origami Boost models at " << kShipped);
  require_data();
  if (const char* record = std::getenv("ORIGAMI_BOOST_RECORD_FEATURE_DIGESTS")) {
    write_digests(record, feature_digests());
    WARN("recorded the feature digests in " << record);
  }

  const auto build = build_identity();
  struct stale_t {
    std::string model, stamp;
  };
  std::vector<stale_t> stale;
  std::size_t          models = 0;
  std::vector<fs::path> arch_dirs;
  for (const auto& e : fs::directory_iterator(kShipped))
    if (e.is_directory()) arch_dirs.push_back(e.path());
  std::sort(arch_dirs.begin(), arch_dirs.end());
  for (const auto& dir : arch_dirs) {
    const std::string arch = dir.filename().string();
    std::set<std::string> indexed;
    std::ifstream         index(dir / "origami_nn_index");
    for (std::string line; std::getline(index, line);) {
      std::istringstream ls(line.substr(0, line.find('#')));
      std::string        stem, backend, file;
      if (ls >> stem >> backend >> file && backend == "boost") indexed.insert(file);
    }
    for (const auto& file : indexed) {
      INFO(arch << "/origami_nn_index names " << file);
      CHECK(fs::is_regular_file(dir / file));
    }
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir))
      if (e.path().extension() == ".orr") files.push_back(e.path());
    std::sort(files.begin(), files.end());
    for (const auto& path : files) {
      ++models;
      const std::string name = path.filename().string();
      INFO("shipped model " << path.string());
      CHECK(indexed.count(name) == 1);
      status_t    st;
      std::string detail;
      const auto  m = ensemble_t::load_file(path.string(), st, detail);
      if (!m) FAIL("cannot be read (" << to_string(st) << "): " << detail);
      const std::string dtype = m->head_string("dtype");
      CHECK(m->head_string("arch") == arch);
      CHECK((dtype == "bf16" || dtype == "tf32" || dtype == "fp8"));
      CHECK(name == "origami_boost_" + arch + "_" + dtype + ".orr");
      if (m->head_string("feature_schema_hash") != build.feature_schema_hash)
        FAIL_CHECK("trained on feature schema " << m->head_string("feature_schema_hash")
                   << ", this build computes " << build.feature_schema_hash
                   << " (features/gemm_boost): the model must be retrained");
      // Everything else booster_t::create checks, the source stamp aside.
      auto id                = build;
      id.origami_source_hash = m->head_string("origami_source_hash");
      const deployment_t own{arch, dtype, &m->library_manifest()};
      CHECK(booster_t::create(m, own, id, st, detail));
      INFO(detail);
      CHECK(st == status_t::ok);
      if (m->head_string("origami_source_hash") != build.origami_source_hash)
        stale.push_back({path.string(), m->head_string("origami_source_hash")});
    }
  }
  REQUIRE(models > 0);

  const auto  now = feature_digests();
  std::string recorded_schema;
  digests_t   recorded;
  REQUIRE(read_digests(kDigests, recorded_schema, recorded));
  std::vector<std::string> changed;
  for (const auto& [key, hash] : now) {
    const auto it = recorded.find(key);
    if (it == recorded.end() || it->second != hash) changed.push_back(key);
  }
  if (recorded_schema != features::schema().hash || recorded.size() != now.size())
    changed.insert(changed.begin(), "(feature schema or column set)");
  std::ostringstream which;
  for (std::size_t i = 0; i < changed.size() && i < 8; ++i) {
    auto key = changed[i];
    std::replace(key.begin(), key.end(), '\t', ' ');
    which << "\n      " << key;
  }
  if (changed.size() > 8) which << "\n      ... " << changed.size() - 8 << " more";

  if (stale.empty()) {
    // The stamps match. Digests that differ here are a platform's libm or an extractor change the
    // feature-parity test bounds; only where they match does the re-stamp decision below hold.
    if (!changed.empty())
      WARN("feature digests differ from " << kDigests.string() << " on " << changed.size()
                                          << " columns:" << which.str());
    return;
  }

  std::ostringstream msg;
  msg << "Origami Boost: the Origami sources changed. This build's source hash is "
      << build.origami_source_hash << "; " << stale.size() << " shipped model(s) are stamped for "
      << "other sources and every build of this tree would refuse them (hipBLASLt falls back to "
      << "stock Origami ranking):";
  for (const auto& s : stale) msg << "\n      " << s.model << " (stamped " << s.stamp << ")";
  if (changed.empty()) {
    msg << "\n\nThe model features on the parity samples are BIT-IDENTICAL to the ones the models "
           "were validated with (pred_cycles and all "
        << features::schema().columns.size() << " columns of " << std::size(kParitySamples)
        << " samples), so the models are still valid: only their stamp is out of date. Re-stamp "
           "them, re-run this test and commit the models with your change:\n\n"
        << "    python3 " << (kOrigamiRoot / "tools" / "origami_boost_restamp.py").string() << "\n"
        << "    origami-tests \"[shipped]\"\n"
        << "    git add " << kShipped.string() << "\n\n"
        << "(shared/origami/README.md, \"Shipped Origami Boost models\")";
  } else {
    msg << "\n\nThe model features CHANGED on the parity samples (" << changed.size()
        << " column(s) differ from " << kDigests.filename().string() << "):" << which.str()
        << "\n\nThe shipped models were trained on the old features and must be RETRAINED for "
           "this change (OrigamiML, the Origami Boost training pipeline, outside this "
           "repository); a re-stamp would ship a model that no longer matches its inputs. "
           "Contact the Origami Boost owners (@ROCm/origami-reviewers; models: Ali Yazdani, "
           "@aliry95amd) before landing the change. Docs: shared/origami/README.md, \"Shipped "
           "Origami Boost models\", and shared/origami/data/nn/README.md.";
  }
  FAIL(msg.str());
}
