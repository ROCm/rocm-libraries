// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "features.hpp"
#include "model_impl.hpp"
#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tw = tilewright;
using namespace tilewright_test;

namespace {

constexpr std::size_t kAll = static_cast<std::size_t>(-1);

struct Random {
  std::mt19937_64 rng;

  explicit Random(std::uint64_t seed) : rng(seed) {}

  std::size_t below(std::size_t n) { return static_cast<std::size_t>(rng() % n); }

  template <typename T>
  T pick(std::initializer_list<T> values) {
    return *(values.begin() + below(values.size()));
  }

  // `n` configs drawn from `distinct` random kernels.
  std::vector<tw::Config> pool(std::size_t distinct, std::size_t n) {
    std::vector<tw::Config> kinds;
    for (std::size_t i = 0; i < distinct; ++i) {
      const std::size_t mt_m = pick<std::size_t>({16, 32, 64, 128, 256});
      const std::size_t mt_n = pick<std::size_t>({16, 32, 64, 128, 256});
      const std::size_t mt_k = pick<std::size_t>({16, 32, 64, 128});
      tw::Config c           = make_config(mt_m, mt_n, mt_k);
      c.mi                   = below(4) == 0 ? tw::Dim3{1, 1, 64} : tw::Dim3{16, 16, 32};
      c.cache_hints_a        = pick<int>({0, 0, 1, 4});
      c.occupancy            = pick<int>({-1, 1, 2, 4});
      c.grvw_a               = pick<std::size_t>({1, 2, 8});
      kinds.push_back(c);
    }
    std::vector<tw::Config> out;
    for (std::size_t i = 0; i < n; ++i) {
      tw::Config c = kinds[below(distinct)];
      c.index      = i;
      out.push_back(c);
    }
    return out;
  }

  tw::Problem problem() {
    const tw::DataType dts[] = {
        tw::DataType::BFloat16, tw::DataType::Half, tw::DataType::Float, tw::DataType::Float8};
    const std::size_t m     = 1 + below(9000);
    const std::size_t n     = 1 + below(9000);
    const std::size_t k     = 1 + below(9000);
    const std::size_t batch = pick<std::size_t>({1, 1, 2, 8});
    const tw::DataType dt   = dts[below(4)];
    const tw::Transpose ta  = below(2) ? tw::Transpose::T : tw::Transpose::N;
    const tw::Transpose tb  = below(2) ? tw::Transpose::T : tw::Transpose::N;
    return make_problem(m, n, k, batch, dt, ta, tb);
  }
};

std::vector<AttrSpec> sample_attributes() {
  return {{"param_a", 0, 1, {}},
          {"param_b", 8, 0, {}},
          {"param_c", 0, 2, {}},
          {"param_d", -1, 3, {-1, 1, 2, 4, 8}}};
}

ModelSpec v3_grid(std::vector<AttrSpec> attributes = sample_attributes()) {
  ModelSpec spec       = as_v3(grid_model(), std::move(attributes));
  spec.schedules       = 0x7;
  spec.min_cu_fraction = 1.0 / 64.0;
  return spec;
}

// MLREC_v3 serving flag: the pool is grouped by launch key.
constexpr std::uint32_t kLaunchKeys = 0x10;

ModelSpec with_launch_keys(ModelSpec spec) {
  spec.v3_serving_flags |= kLaunchKeys;
  return spec;
}

ModelSpec v3_one_cell(const std::string& label, std::vector<AttrSpec> attributes = {}) {
  ModelSpec spec = as_v3(ModelSpec{}, std::move(attributes));
  CellSpec c;
  c.label = label;
  spec.cells.push_back(c);
  return spec;
}

std::string load_error(const Image& image) {
  std::string err;
  const tw::ModelPtr m = tw::load_model_from_memory(image.bytes.data(), image.bytes.size(), &err);
  if (m) return std::string();
  return err.empty() ? std::string("<no message>") : err;
}

bool mentions(const std::string& text, const std::string& what) {
  return text.find(what) != std::string::npos;
}

std::string spec_error(const ModelSpec& spec) { return load_error(write_model(spec)); }

tw::Config with_attributes(tw::Config c, std::vector<tw::Attribute> attributes) {
  c.attributes = std::move(attributes);
  return c;
}

tw::ExecutionContext context(std::size_t cu_budget, tw::Schedule schedule) {
  tw::ExecutionContext ctx;
  ctx.cu_budget = cu_budget;
  ctx.schedule  = schedule;
  return ctx;
}

// FNV-1a 64 over the view key: the signature, then the spec's attributes
// (defaults for absent ones), each as 8 little-endian bytes.
std::uint64_t key_hash(const tw::Config& c, const std::vector<AttrSpec>& spec) {
  std::vector<std::int64_t> v = {static_cast<std::int64_t>(c.mt.m),
                                 static_cast<std::int64_t>(c.mt.n),
                                 static_cast<std::int64_t>(c.mt.k),
                                 static_cast<std::int64_t>(c.mi.m),
                                 static_cast<std::int64_t>(c.mi.n),
                                 static_cast<std::int64_t>(c.mi.k),
                                 c.cache_hints_a,
                                 c.cache_hints_b};
  const auto value            = [&](const std::string& name, std::int64_t def) {
    for (const tw::Attribute& a : c.attributes)
      if (a.name == name) return a.value;
    return def;
  };
  for (const AttrSpec& a : spec) v.push_back(value(a.name, a.default_value));
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (std::int64_t x : v)
    for (int b = 0; b < 8; ++b)
      h = (h ^ static_cast<unsigned char>(static_cast<std::uint64_t>(x) >> (8 * b))) *
          0x100000001B3ull;
  return h;
}

// Every config ranked on its own, in pool order of the results' scores.
std::vector<tw::Result> rank_singly(const tw::Model& model,
                                    const tw::Problem& p,
                                    const tw::ExecutionContext& ctx,
                                    const std::vector<tw::Config>& pool,
                                    const tw::LaunchModel* launch = nullptr,
                                    std::size_t min_scored        = kAll) {
  tw::detail::PoolKeys k = singleton_keys(pool);
  k.attribute_count      = model.attribute_reads.size();
  k.attribute_values.resize(pool.size() * k.attribute_count);
  for (std::size_t i = 0; i < pool.size(); ++i)
    tw::detail::read_attributes(model, pool[i], k.attribute_values.data() + i * k.attribute_count);
  return tw::detail::rank_keys(model,
                               p,
                               test_hardware(),
                               ctx,
                               pool,
                               k,
                               min_scored,
                               tw::TieBreak::PoolOrder,
                               nullptr,
                               launch);
}

}  // namespace

TEST_CASE("v3 models load and describe themselves", "[v3][loader]") {
  ModelSpec spec           = v3_grid();
  spec.v3_serving_flags    = 0x05;
  spec.v3_cascade_keys     = 32;
  spec.min_cu_fraction     = 0.25;
  spec.schedules           = 0x3;
  const tw::ModelPtr m     = load_spec(spec);
  const tw::ModelInfo info = tw::describe(*m);
  CHECK(info.format_version == 3);
  CHECK(info.feature_catalog_hash == kV3Hash);
  CHECK(info.feature_spec_hash == spec_hash(sample_attributes()));
  CHECK(info.query_dim == kV3Q);
  CHECK(info.item_dim == kV3I + 8);
  CHECK(info.inter_dim == kV3X);
  CHECK(info.min_cu_fraction == 0.25);
  CHECK(info.max_cu_fraction == 1.0);
  CHECK(info.schedules == 0x3);
  CHECK(info.cascade_keys == 32);
  CHECK(info.whitelist_tier);
  CHECK_FALSE(info.launch_keys);
  CHECK(info.small_batched_rule);
  CHECK_FALSE(info.dot2_rule);
  CHECK(info.non_temporal_rule);
  CHECK_FALSE(info.clip);
  CHECK(info.uncovered_base_labels == 0);
  CHECK(tw::attribute_names(*m) ==
        std::vector<std::string>{"param_a", "param_b", "param_c", "param_d"});
  const std::vector<std::string> hashes = tw::feature_catalog_hashes();
  CHECK(hashes == std::vector<std::string>{tw::feature_catalog_hash(), kV3Hash});

  const tw::ModelInfo plain = tw::describe(*load_spec(v3_one_cell("Large|Large|LargeK|Bnone")));
  CHECK(plain.item_dim == kV3I);
  CHECK_FALSE(plain.small_batched_rule);
  CHECK_FALSE(plain.dot2_rule);
  CHECK_FALSE(plain.non_temporal_rule);
  CHECK(plain.cascade_keys == 0);
  CHECK(plain.uncovered_base_labels == 95);
  CHECK(tw::describe(*load_spec(with_launch_keys(v3_grid()))).launch_keys);

  const tw::ModelInfo v2 = tw::describe(*load_spec(grid_model()));
  CHECK(v2.format_version == 2);
  CHECK(v2.feature_spec_hash == 0);
  CHECK(v2.item_dim == 12);
  CHECK(v2.small_batched_rule);
  CHECK(v2.dot2_rule);
  CHECK(v2.non_temporal_rule);
}

// Frozen: lib/mlrec.py computes the same values (training_pipeline/tests/
// test_v3.py). A new value means the v3 features changed, which needs every
// v3 model to be trained again.
TEST_CASE("the v3 feature spec hash covers names, the spec and golden features", "[v3][catalog]") {
  CHECK(spec_hash({}) == 0x640b5d2e92a9f47full);
  CHECK(spec_hash(sample_attributes()) == 0x40d53a0bf908f951ull);

  const std::vector<AttrSpec> spec = sample_attributes();
  std::vector<std::vector<AttrSpec>> variants(5, spec);
  std::swap(variants[0][0], variants[0][1]);
  variants[1][1].default_value = 9;
  variants[2][0].transform     = 0;
  variants[3][3].values.back() = 16;
  variants[4][2].name          = "param_e";
  for (const std::vector<AttrSpec>& v : variants) CHECK(spec_hash(v) != spec_hash(spec));

  const std::string text = tw::detail::feature_spec_text(attribute_features(spec));
  CHECK(text.rfind(std::string(kV3Hash) + "\nversion\t3\nquery\tlog2_m,log2_n,", 0) == 0);
  CHECK(mentions(text, ",schedule_auto\nitem\tlog2_mt_m,"));
  CHECK(mentions(text,
                 ",ntb_norm,attr_log2p1_param_a,attr_param_b,attr_nz_param_c,attr_param_d_eq_-1,"
                 "attr_param_d_eq_1,"));
  CHECK(mentions(text, ",reduction_atomic\nparam_a\t0\tlog2p1\n"));
  CHECK(mentions(text, "\nparam_d\t-1\tonehot\t-1,1,2,4,8\ngolden\t0\t"));

  // Golden cases 0 (the default launch, exclusive use) and 1 (a dynamic queue
  // with a parallel reduction, under a CU budget and the dynamic schedule) are
  // the features compute_features gives for their inputs on a model with the
  // golden arch constants.
  ModelSpec golden         = v3_one_cell("Large|Large|LargeK|Bnone", spec);
  golden.parallel_mi_cu    = 4.0;
  golden.mi_default_cycles = 64.0;
  golden.mi_table          = {{16, 16, 32, static_cast<int>(tw::DataType::BFloat16), 16.0},
                              {16, 16, 128, static_cast<int>(tw::DataType::Float8), 16.0},
                              {16, 16, 4, static_cast<int>(tw::DataType::XFloat32), 32.0}};
  const tw::ModelPtr m     = load_spec(golden);
  const auto golden_line   = [&](int index, const tw::Features& f) {
    std::string line          = "golden\t" + std::to_string(index) + "\t";
    std::vector<float> values = f.query;
    values.insert(values.end(), f.item.begin(), f.item.end());
    values.insert(values.end(), f.interaction.begin(), f.interaction.end());
    for (std::size_t i = 0; i < values.size(); ++i) {
      std::uint32_t bits;
      std::memcpy(&bits, &values[i], sizeof bits);
      char hex[16];
      std::snprintf(hex, sizeof hex, "%s%08x", i > 0 ? "," : "", static_cast<unsigned>(bits));
      line += hex;
    }
    return line + "\n";
  };
  tw::Config c0         = make_config(256, 256, 64, 16, 16, 32, 0, 0);
  c0.attributes         = {{"param_a", -4}, {"param_b", 3}, {"param_c", 10}, {"param_d", 4}};
  const tw::Features f0 = tw::compute_features(*m,
                                               make_problem(4096, 4096, 4096),
                                               c0,
                                               tw::Hardware{256, 163840, 4194304},
                                               tw::ExecutionContext{});
  tw::Problem p1 =
      make_problem(777, 12345, 1000, 3, tw::DataType::Float8, tw::Transpose::N, tw::Transpose::T);
  p1.c_dtype    = tw::DataType::BFloat16;
  p1.d_dtype    = tw::DataType::BFloat16;
  tw::Config c1 = make_config(128, 64, 128, 16, 16, 128, 4, 0);
  c1.attributes = {{"param_a", -1}, {"param_b", 6}, {"param_c", -4}, {"param_d", 8}};
  const tw::Features f1 =
      tw::compute_features(*m,
                           p1,
                           c1,
                           tw::Hardware{256, 163840, 4194304},
                           tw::ExecutionContext{64, tw::Schedule::Dynamic},
                           tw::LaunchPlan{true, 192, tw::PartialReduction::Parallel});
  CHECK(mentions(text, golden_line(0, f0) + golden_line(1, f1) + "golden\t2\t"));

  // A file whose hash covers the spec alone, as before this hash covered the
  // features, is rejected.
  std::string legacy = std::string(kV3Hash) + "\n";
  for (const AttrSpec& a : spec) {
    static const char* const kTransforms[] = {"identity", "log2p1", "nonzero", "onehot"};
    legacy += a.name + "\t" + std::to_string(a.default_value) + "\t" + kTransforms[a.transform];
    if (a.transform == 3) {
      legacy += "\t";
      for (std::size_t i = 0; i < a.values.size(); ++i)
        legacy += (i > 0 ? "," : "") + std::to_string(a.values[i]);
    }
    legacy += "\n";
  }
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (unsigned char ch : legacy) h = (h ^ ch) * 0x100000001B3ull;
  ModelSpec old          = v3_one_cell("Large|Large|LargeK|Bnone", spec);
  old.spec_hash_override = h;
  CHECK(mentions(spec_error(old), "feature_spec_hash does not match"));
}

TEST_CASE("a v3 image truncated at any byte or cut at any field is rejected", "[v3][loader]") {
  ModelSpec spec                = v3_one_cell("Large|Large|LargeK|Bnone", sample_attributes());
  spec.clip                     = 1;
  spec.cells[0].signatures      = {{256, 256, 64, 16, 16, 32, 0, 0}};
  spec.cells[0].view_key_hashes = {1, 2, 3};
  const Image image             = write_model(spec);
  REQUIRE(load_error(image).empty());
  for (std::size_t n = 0; n < image.bytes.size(); ++n) {
    INFO("size " << n);
    std::string err;
    REQUIRE_FALSE(tw::load_model_from_memory(image.bytes.data(), n, &err));
    REQUIRE_FALSE(err.empty());
  }
  const std::size_t trailer = image.at.at("trailer");
  for (const auto& field : image.at) {
    if (field.second <= image.header_size || field.second >= trailer) continue;
    for (std::size_t extra : {0, 1}) {
      const std::size_t offset = field.second + extra;
      INFO(field.first << " +" << extra);
      Image cut = image;
      cut.bytes.resize(offset);
      cut.bytes.insert(cut.bytes.end(), {'M', 'L', 'R', 'E', 'C', 'E', 'N', 'D'});
      put_u64(cut, cut.at.at("payload_size"), offset - image.header_size);
      refresh_crc(cut);
      CHECK_FALSE(load_error(cut).empty());
    }
  }
  Image corrupt = image;
  corrupt.bytes[image.at.at("cell0.q_w0") + 3] ^= 0x10;
  CHECK(mentions(load_error(corrupt), "payload CRC mismatch"));
}

TEST_CASE("v3 header defects are rejected", "[v3][loader]") {
  const ModelSpec good = v3_one_cell("Large|Large|LargeK|Bnone", sample_attributes());
  REQUIRE(spec_error(good).empty());
  const auto with = [&](void (*edit)(ModelSpec&)) {
    ModelSpec s = good;
    edit(s);
    return spec_error(s);
  };
  CHECK(mentions(with([](ModelSpec& s) { s.feature_hash = "e7fe4b524851e895"; }),
                 "MLREC_v3 catalog"));
  CHECK(mentions(with([](ModelSpec& s) { s.feature_hash = "0000000000000000"; }),
                 "MLREC_v3 catalog"));
  CHECK(mentions(with([](ModelSpec& s) { s.spec_hash_override = 12345; }), "feature_spec_hash"));
  CHECK(mentions(with([](ModelSpec& s) { s.i_dim += 1; }), "feature dims"));
  CHECK(mentions(with([](ModelSpec& s) { s.q_dim = 55; }), "feature dims"));
  CHECK(mentions(with([](ModelSpec& s) { s.x_dim = 37; }), "feature dims"));
  CHECK(mentions(with([](ModelSpec& s) { s.v3_serving_flags = 0x200; }), "unknown serving flags"));
  CHECK(mentions(with([](ModelSpec& s) { s.v3_serving_flags = 0x20; }),
                 "without the non-temporal rule"));
  CHECK(mentions(with([](ModelSpec& s) { s.v3_serving_flags = 0x80; }), "no trained key"));
  CHECK(mentions(with([](ModelSpec& s) {
                   s.v3_serving_flags = 0x80;
                   s.trained_keys     = {7, 5};
                 }),
                 "not strictly ascending"));
  CHECK(mentions(with([](ModelSpec& s) {
                   s.v3_serving_flags = 0x80;
                   s.trained_keys     = {5, 5};
                 }),
                 "not strictly ascending"));
  CHECK(mentions(with([](ModelSpec& s) { s.min_cu_fraction = 0.0; }), "CU fraction"));
  CHECK(mentions(with([](ModelSpec& s) { s.min_cu_fraction = -0.5; }), "CU fraction"));
  CHECK(mentions(with([](ModelSpec& s) {
                   s.min_cu_fraction = 0.75;
                   s.max_cu_fraction = 0.5;
                 }),
                 "CU fraction"));
  CHECK(mentions(with([](ModelSpec& s) { s.max_cu_fraction = 1.5; }), "CU fraction"));
  CHECK(mentions(
      with([](ModelSpec& s) { s.max_cu_fraction = std::numeric_limits<double>::quiet_NaN(); }),
      "non-finite max_cu_fraction"));
  CHECK(mentions(with([](ModelSpec& s) { s.schedules = 0; }), "schedule mask"));
  CHECK(mentions(with([](ModelSpec& s) { s.schedules = 0x9; }), "schedule mask"));
  CHECK(mentions(with([](ModelSpec& s) { s.clip = 2; }), "clip flag"));
  CHECK(mentions(with([](ModelSpec& s) { s.attributes[0].transform = 4; }),
                 "unknown attribute transform"));
  CHECK(mentions(with([](ModelSpec& s) {
                   s.attributes[3].values.clear();
                   s.i_dim = kV3I + attribute_dims(s.attributes);
                 }),
                 "one-hot value count"));
  CHECK(mentions(with([](ModelSpec& s) { s.attributes[3].values = {1, 2, 1, 4, 8}; }),
                 "repeats a one-hot value"));
  CHECK(mentions(with([](ModelSpec& s) { s.attributes[1].name = "param_a"; }),
                 "duplicate attribute 'param_a'"));
  CHECK(mentions(with([](ModelSpec& s) { s.attributes[1].name = ""; }), "empty attribute name"));
  CHECK(mentions(with([](ModelSpec& s) { s.attributes[1].name = "has space"; }),
                 "non-printable attribute name"));
  CHECK(mentions(with([](ModelSpec& s) {
                   s.attributes.assign(65, AttrSpec{});
                   for (std::size_t i = 0; i < 65; ++i)
                     s.attributes[i].name = "a" + std::to_string(i);
                   s.i_dim = kV3I + 65;
                 }),
                 "n_attributes 65"));

  Image reserved     = write_model(good);
  reserved.bytes[30] = 1;
  CHECK(mentions(load_error(reserved), "reserved header bytes"));

  ModelSpec many = good;
  many.attributes.clear();
  for (std::size_t i = 0; i < 64; ++i)
    many.attributes.push_back({"a" + std::to_string(i), 0, 0, {}});
  many.i_dim = kV3I + 64;
  CHECK(spec_error(many).empty());
}

TEST_CASE("v3 provenance is optional and describe() reports it", "[v3][loader][provenance]") {
  ModelSpec spec            = v3_one_cell("Large|Large|LargeK|Bnone", sample_attributes());
  const tw::ModelPtr m      = load_spec(spec);
  const tw::Provenance none = tw::describe(*m).provenance;
  CHECK(none.config_id.empty());
  CHECK(none.training_data_hash.empty());
  CHECK(none.source_revision.empty());
  CHECK(none.date.empty());
  CHECK(tw::describe(*load_spec(grid_model())).provenance.date.empty());

  spec.provenance          = {{"date", "2000-01-01T12:00:00Z"}, {"config_id", "gfx950_bbs_tn"}};
  const tw::Provenance two = tw::describe(*load_spec(spec)).provenance;
  CHECK(two.config_id == "gfx950_bbs_tn");
  CHECK(two.date == "2000-01-01T12:00:00Z");
  CHECK(two.training_data_hash.empty());
  CHECK(two.source_revision.empty());
  spec.provenance           = {{"training_data_hash", "sha256:0123abcd"},
                               {"source_revision", "0123abc-dirty"},
                               {"config_id", "c"},
                               {"date", "2000-01-01"},
                               {"launch_model", "rule:gfxtest:3"}};
  const tw::ModelPtr all    = load_spec(spec);
  const tw::Provenance five = tw::describe(*all).provenance;
  CHECK(five.config_id == "c");
  CHECK(five.training_data_hash == "sha256:0123abcd");
  CHECK(five.source_revision == "0123abc-dirty");
  CHECK(five.date == "2000-01-01");
  CHECK(five.launch_model == "rule:gfxtest:3");
  CHECK(none.launch_model.empty());

  const std::vector<tw::Config> pool = {make_config(128, 128, 64), make_config(64, 64, 64)};
  const tw::Problem p                = make_problem(4096, 4096, 4096);
  CHECK(same_results(tw::rank_configs(*all, p, test_hardware(), pool, kAll),
                     tw::rank_configs(*m, p, test_hardware(), pool, kAll)));
}

TEST_CASE("v3 provenance defects are rejected", "[v3][loader][provenance]") {
  const ModelSpec good = v3_one_cell("Large|Large|LargeK|Bnone", sample_attributes());
  const auto with      = [&](std::vector<std::pair<std::string, std::string>> provenance,
                        std::vector<unsigned char> extra = {}) {
    ModelSpec s    = good;
    s.provenance   = std::move(provenance);
    s.header_extra = std::move(extra);
    return spec_error(s);
  };
  CHECK(with({{"config_id", "a"}}).empty());
  CHECK(mentions(with({{"trainer", "a"}}), "unknown provenance key 'trainer'"));
  CHECK(mentions(with({{"date", "a"}, {"date", "b"}}), "duplicate provenance key 'date'"));
  CHECK(mentions(with({{"date", ""}}), "empty provenance value"));
  CHECK(mentions(with({{"", "a"}}), "empty provenance key"));
  CHECK(mentions(with({{"config_id", "a b"}}), "non-printable provenance value"));
  CHECK(mentions(with({{"date", "a"},
                       {"config_id", "b"},
                       {"source_revision", "c"},
                       {"training_data_hash", "d"},
                       {"launch_model", "e"},
                       {"date", "f"}}),
                 "provenance count 6"));
  CHECK(mentions(with({}, {0, 0}), "provenance count 0"));
  CHECK(mentions(with({{"date", "a"}}, {0}), "header_size does not match"));
  CHECK(mentions(with({}, {1}), "truncated header: provenance count"));

  // The block cut short anywhere after its count, with header_size following.
  ModelSpec spec   = good;
  spec.provenance  = {{"config_id", "gfx950"}, {"date", "2000-01-01"}};
  const Image full = write_model(spec);
  REQUIRE(load_error(full).empty());
  const std::size_t start = full.at.at("n_provenance");
  for (std::size_t cut = start + 1; cut < full.header_size; ++cut) {
    INFO("header cut at " << cut);
    Image image = full;
    image.bytes.erase(image.bytes.begin() + static_cast<long>(cut),
                      image.bytes.begin() + static_cast<long>(full.header_size));
    put_u32(image, image.at.at("header_size"), static_cast<std::uint32_t>(cut));
    CHECK_FALSE(load_error(image).empty());
  }
  Image without = full;
  without.bytes.erase(without.bytes.begin() + static_cast<long>(start),
                      without.bytes.begin() + static_cast<long>(full.header_size));
  put_u32(without, without.at.at("header_size"), static_cast<std::uint32_t>(start));
  CHECK(load_error(without).empty());
}

TEST_CASE("v3 cell bounds and keyed whitelists are validated", "[v3][loader]") {
  ModelSpec spec = v3_one_cell("Large|Large|LargeK|Bnone", sample_attributes());
  spec.clip      = 1;
  REQUIRE(spec_error(spec).empty());

  const Image good = write_model(spec);
  Image nan        = good;
  put_u32(nan, good.at.at("cell0.q_lo") + 8, 0x7FC00000u);
  refresh_crc(nan);
  CHECK(mentions(load_error(nan), "NaN in query bounds"));

  Image inverted = good;
  put_f32(inverted, good.at.at("cell0.i_lo") + 4, 2.0f);
  put_f32(inverted, good.at.at("cell0.i_lo") + 4 * (spec.i_dim + 1), 1.0f);
  refresh_crc(inverted);
  CHECK(mentions(load_error(inverted), "lower bound above upper in item bounds"));

  Image equal = good;
  put_f32(equal, good.at.at("cell0.x_lo"), 3.0f);
  put_f32(equal, good.at.at("cell0.x_lo") + 4 * spec.x_dim, 3.0f);
  refresh_crc(equal);
  CHECK(load_error(equal).empty());

  Image keys = good;
  put_u32(keys, good.at.at("cell0.n_vk"), 0x10000000u);
  refresh_crc(keys);
  CHECK(mentions(load_error(keys), "view key count 268435456 exceeds the payload"));
}

TEST_CASE("v3 features follow the usable CUs and the schedule", "[v3][features]") {
  const tw::ModelPtr v3 = load_spec(v3_grid());
  const tw::ModelPtr v2 = load_spec(grid_model());
  const tw::Hardware hw = test_hardware();
  tw::Hardware small    = hw;
  small.N_CU            = 16;
  const tw::Problem p   = make_problem(1000, 2000, 512);
  const tw::Config c    = make_config(128, 128, 64, 16, 16, 32, 0, 0, 2);

  // With 16 usable CUs, v2's formulas on a 16-CU device.
  const tw::Features budget =
      tw::compute_features(*v3, p, c, hw, context(16, tw::Schedule::Dynamic));
  const tw::Features ref = tw::compute_features(*v2, p, c, small);
  REQUIRE(budget.query.size() == kV3Q);
  REQUIRE(budget.item.size() == kV3I + 8);
  REQUIRE(budget.interaction.size() == kV3X);
  // v2's features without occupancy_norm, the vector widths and bw_occ.
  const auto same_as_v2 = [](const tw::Features& a, const tw::Features& b) {
    for (std::size_t j = 0; j < 55; ++j) CHECK(a.query[j] == b.query[j]);
    for (std::size_t j = 0; j < 8; ++j) CHECK(a.item[j] == b.item[j]);
    for (std::size_t j = 0; j < 30; ++j) CHECK(a.interaction[j] == b.interaction[j]);
    for (std::size_t j = 30; j < 33; ++j) CHECK(a.interaction[j] == b.interaction[j + 2]);
    CHECK(a.interaction[33] == b.interaction[36]);
  };
  same_as_v2(budget, ref);

  // log2_enum_waves_128 = log2(ceil(8 * 16 tiles / 16 CUs)); enum_wave_eff_128 = 128 / (8 * 16).
  CHECK(budget.query[51] == 3.0f);
  CHECK(budget.query[53] == 1.0f);
  CHECK(budget.query[55] == 0.25f);
  CHECK(budget.query[56] == 0.0f);
  CHECK(budget.query[57] == 1.0f);
  CHECK(budget.query[58] == 0.0f);
  CHECK(budget.interaction[33] == 1.0f);  // min(128 tiles, 16) / 16

  const tw::Features excl = tw::compute_features(*v3, p, c, hw);
  const tw::Features full = tw::compute_features(*v2, p, c, hw);
  same_as_v2(excl, full);
  CHECK(excl.query[51] == 1.0f);
  CHECK(excl.query[55] == 1.0f);
  CHECK(excl.query[56] == 1.0f);
  CHECK(excl.query[57] == 0.0f);
  CHECK(excl.interaction[33] == 1.0f);  // min(128, 64) / 64
  CHECK(tw::compute_features(*v3, p, c, hw, context(64, tw::Schedule::Auto)).query[58] == 1.0f);
  CHECK(tw::compute_features(*v3, p, c, hw, context(500, tw::Schedule::Default)).query ==
        excl.query);
}

TEST_CASE("count log2 gives the bits of std::log2", "[v3][features]") {
  const auto want        = [](double x) { return static_cast<float>(std::log2(std::max(x, 1.0))); };
  const double inf       = std::numeric_limits<double>::infinity();
  std::vector<double> xs = {-inf,
                            -1.0,
                            0.0,
                            -0.0,
                            0.5,
                            1.0,
                            1.5,
                            2047.0,
                            2047.5,
                            2048.0,
                            2049.0,
                            1e12,
                            inf,
                            std::numeric_limits<double>::quiet_NaN()};
  for (int n = 0; n < 2100; ++n) xs.push_back(n);
  for (const double x : xs) {
    const float got = tw::detail::count_log2(x), w = want(x);
    CHECK(std::memcmp(&got, &w, sizeof got) == 0);
  }
}

TEST_CASE("v3 item features apply the attribute spec", "[v3][features]") {
  const tw::ModelPtr m = load_spec(v3_grid());
  const tw::Problem p  = make_problem(1000, 2000, 512);
  const tw::Config c   = make_config(128, 128, 64);
  const auto item      = [&](std::vector<tw::Attribute> attributes) {
    const std::vector<float> f =
        tw::compute_features(*m, p, with_attributes(c, std::move(attributes)), test_hardware())
            .item;
    return std::vector<float>(f.begin() + 8, f.end());
  };
  CHECK(item({{"param_a", 32}, {"param_c", 1}, {"param_d", 4}}) ==
        std::vector<float>{static_cast<float>(std::log2(33.0)), 8, 1, 0, 0, 0, 1, 0});
  CHECK(item({}) == std::vector<float>{0, 8, 0, 1, 0, 0, 0, 0});
  CHECK(item({{"param_a", -3},
              {"param_b", -16},
              {"param_c", -1},
              {"param_d", 16},
              {"unrelated", 9}}) == std::vector<float>{0, -16, 1, 0, 0, 0, 0, 0});
  CHECK(item({{"param_a", 3}}).front() == 2.0f);
}

// is_dynamic, log2_grid, log2_grid_per_cu, log2_wg_waves, wg_wave_eff,
// log2_tiles_per_wg, uneven_work and the reduction_* flags of `wgs`
// workgroups over `tiles` output tiles on `n_eff` usable CUs.
std::vector<float> launch_values(bool dynamic,
                                 double wgs,
                                 double tiles,
                                 double n_eff,
                                 tw::PartialReduction r = tw::PartialReduction::None) {
  const double waves = std::ceil(wgs / n_eff);
  return {dynamic ? 1.0f : 0.0f,
          static_cast<float>(std::log2(std::max(wgs, 1.0))),
          static_cast<float>(std::log2(std::max(wgs / n_eff, 0.001))),
          static_cast<float>(std::log2(std::max(waves, 1.0))),
          static_cast<float>(wgs / (waves * n_eff)),
          static_cast<float>(std::log2(std::max(tiles / wgs, 0.001))),
          std::fmod(tiles, wgs) != 0.0 ? 1.0f : 0.0f,
          r == tw::PartialReduction::Tree ? 1.0f : 0.0f,
          r == tw::PartialReduction::Parallel ? 1.0f : 0.0f,
          r == tw::PartialReduction::Atomic ? 1.0f : 0.0f};
}

std::vector<float> launch_part(const tw::Features& f) {
  return std::vector<float>(f.interaction.begin() + 34, f.interaction.end());
}

// Records the configs it is asked to plan, and plans them with `fn`.
struct RecordingLaunch final : tw::LaunchModel {
  mutable std::vector<std::size_t> asked;
  tw::LaunchPlan (*fn)(const tw::Config&) = nullptr;

  tw::LaunchPlan plan(const tw::Config& c,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    asked.push_back(c.index);
    return fn != nullptr ? fn(c) : tw::LaunchPlan{};
  }
};

struct ThrowingLaunch final : tw::LaunchModel {
  tw::LaunchPlan plan(const tw::Config&,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    throw std::runtime_error("no plan");
  }
};

tw::LaunchPlan index_plan(const tw::Config& c) {
  return {c.index % 2 == 0, 7 + 31 * c.index, tw::PartialReduction::Tree};
}

// Six signatures, then a variant of each that differs in every other Config
// field: indices s and s + 6 share a view key.
std::vector<tw::Config> variant_pool() {
  std::vector<tw::Config> pool;
  for (std::size_t mt : {32, 64, 128, 256}) pool.push_back(make_config(mt, 128, 64));
  pool.push_back(make_config(128, 64, 32, 16, 16, 32, 1, 0));
  pool.push_back(make_config(64, 256, 64, 32, 32, 16));
  for (std::size_t s = 0; s < 6; ++s) {
    tw::Config v = pool[s];
    v.occupancy  = 4;
    v.grvw_a     = 1;
    v.grvw_b     = 2;
    v.gwvw_d     = 1;
    v.attributes = {{"unread", 3}};
    pool.push_back(v);
  }
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  return pool;
}

TEST_CASE("v3 launch features read the integration's launch plan", "[v3][launch]") {
  const tw::ModelPtr m  = load_spec(v3_grid());
  const tw::Hardware hw = test_hardware();
  const tw::Problem p   = make_problem(4096, 4096, 512);  // 1024 tiles of 128 x 128
  const tw::Config c    = make_config(128, 128, 64);
  const auto launch     = [&](const tw::LaunchPlan& plan, const tw::ExecutionContext& ctx) {
    return launch_part(tw::compute_features(*m, p, c, hw, ctx, plan));
  };
  const tw::ExecutionContext excl;
  using R = tw::PartialReduction;

  // Without a plan: static, one workgroup per output tile.
  CHECK(launch_part(tw::compute_features(*m, p, c, hw)) == launch_values(false, 1024, 1024, 64));
  CHECK(launch(tw::LaunchPlan{}, excl) == launch_values(false, 1024, 1024, 64));
  CHECK(launch({false, 1024, R::None}, excl) == launch_values(false, 1024, 1024, 64));
  CHECK(launch({true, 192, R::None}, excl) == launch_values(true, 192, 1024, 64));
  CHECK(launch({false, 320, R::Tree}, excl) == launch_values(false, 320, 1024, 64, R::Tree));
  CHECK(launch({false, 4096, R::Parallel}, excl) ==
        launch_values(false, 4096, 1024, 64, R::Parallel));
  CHECK(launch({false, 100, R::Atomic}, excl) == launch_values(false, 100, 1024, 64, R::Atomic));
  // Per-CU values use the usable CUs; the batch multiplies the tiles.
  CHECK(launch({false, 320, R::Tree}, context(16, tw::Schedule::Dynamic)) ==
        launch_values(false, 320, 1024, 16, R::Tree));
  CHECK(launch_part(tw::compute_features(*m, make_problem(4096, 4096, 512, 3), c, hw)) ==
        launch_values(false, 3072, 3072, 64));
  // The plan changes only the launch features.
  const tw::Features a = tw::compute_features(*m, p, c, hw, excl, {true, 192, R::Atomic});
  const tw::Features b = tw::compute_features(*m, p, c, hw);
  CHECK(a.query == b.query);
  CHECK(a.item == b.item);
  CHECK(std::equal(a.interaction.begin(), a.interaction.begin() + 34, b.interaction.begin()));

  // v2 features ignore it.
  const tw::ModelPtr v2 = load_spec(grid_model());
  CHECK(tw::compute_features(*v2, p, c, hw, excl, {true, 7, R::Atomic}).interaction ==
        tw::compute_features(*v2, p, c, hw).interaction);
}

TEST_CASE("v3 configs with one signature are one candidate", "[v3][signature]") {
  const tw::Hardware hw              = test_hardware();
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const std::vector<tw::Config> pool = variant_pool();
  const tw::ModelPtr m               = load_spec(v3_grid({}));
  CHECK(tw::attribute_names(*m).empty());

  const tw::detail::PoolKeys keys = tw::detail::pool_keys(*m, pool);
  CHECK(keys.first.size() == 6);
  for (std::size_t s = 0; s < 6; ++s) {
    CHECK(keys.key_of[s] == keys.key_of[s + 6]);
    const tw::Features a = tw::compute_features(*m, p, pool[s], hw);
    const tw::Features b = tw::compute_features(*m, p, pool[s + 6], hw);
    CHECK(a.query == b.query);
    CHECK(a.item == b.item);
    CHECK(a.interaction == b.interaction);
  }
  const auto score_of = [](const std::vector<tw::Result>& r, std::size_t i) {
    for (const tw::Result& x : r)
      if (x.config_index == i) return x.score;
    return std::numeric_limits<double>::quiet_NaN();
  };
  const std::vector<tw::Result> r = tw::CandidateSet(m, pool).rank(p, hw, kAll);
  REQUIRE(scored_indices(r).size() == pool.size());
  for (std::size_t s = 0; s < 6; ++s) CHECK(score_of(r, s) == score_of(r, s + 6));
  CHECK(same_results(r, rank_singly(*m, p, tw::ExecutionContext{}, pool)));

  // A whitelisted view key lists every variant of its signature.
  ModelSpec keyed = v3_grid({});
  for (CellSpec& cell : keyed.cells) cell.view_key_hashes = {key_hash(pool[8], {})};
  CHECK(scored_indices(tw::CandidateSet(load_spec(keyed), pool).rank(p, hw)) ==
        std::vector<std::size_t>{2, 8});

  // A declared attribute joins the view key; v2 keys hold every Config field.
  std::vector<tw::Config> split = pool;
  split[6].attributes           = {{"param_a", 1}};
  CHECK(tw::detail::pool_keys(*load_spec(v3_grid({{"param_a", 0, 0, {}}})), split).first.size() ==
        7);
  CHECK(tw::detail::pool_keys(*load_spec(grid_model()), pool).first.size() == 12);
}

TEST_CASE("a launch model plans each fully scored view key once, for its first config",
          "[v3][launch]") {
  const tw::Hardware hw              = test_hardware();
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const std::vector<tw::Config> pool = variant_pool();
  const tw::ModelPtr m               = load_spec(v3_grid({}));
  const auto with = [&](const tw::ModelPtr& model, std::shared_ptr<const tw::LaunchModel> launch) {
    return tw::CandidateSet(model, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, launch});
  };
  const std::vector<tw::Result> plain = tw::CandidateSet(m, pool).rank(p, hw, kAll);

  // Present with default plans: the result without a launch model.
  auto rec                          = std::make_shared<RecordingLaunch>();
  const tw::CandidateSet set        = with(m, rec);
  const std::vector<tw::Result> got = set.rank(p, hw, kAll);
  CHECK(same_results(got, plain));
  std::vector<std::size_t> asked = rec->asked;
  std::sort(asked.begin(), asked.end());
  CHECK(asked == std::vector<std::size_t>{0, 1, 2, 3, 4, 5});

  // Its plans change the scores, and the variants still score with their
  // signature's first config.
  auto planned                         = std::make_shared<RecordingLaunch>();
  planned->fn                          = &index_plan;
  const std::vector<tw::Result> scored = with(m, planned).rank(p, hw, kAll);
  CHECK_FALSE(same_results(scored, plain));
  for (std::size_t i : planned->asked) CHECK(i < 6);
  for (std::size_t s = 0; s < 6; ++s) {
    double a = 0, b = 0;
    for (const tw::Result& x : scored) {
      if (x.config_index == s) a = x.score;
      if (x.config_index == s + 6) b = x.score;
    }
    CHECK(a == b);
  }

  // A cascade asks only for the keys it keeps.
  ModelSpec cascade                 = v3_grid({});
  cascade.v3_cascade_keys           = 2;
  auto kept                         = std::make_shared<RecordingLaunch>();
  const std::vector<tw::Result> top = with(load_spec(cascade), kept).rank(p, hw);
  CHECK(scored_indices(top).size() == 4);
  REQUIRE(kept->asked.size() == 2);
  for (std::size_t i : kept->asked) {
    const std::vector<std::size_t> s = scored_indices(top);
    CHECK(std::find(s.begin(), s.end(), i) != s.end());
    CHECK(std::find(s.begin(), s.end(), i + 6) != s.end());
  }

  // Never asked by v2 models or for an unlisted schedule; its errors propagate.
  auto v2 = std::make_shared<RecordingLaunch>();
  with(load_spec(grid_model()), v2).rank(p, hw, kAll);
  CHECK(v2->asked.empty());
  ModelSpec narrow            = v3_grid({});
  narrow.schedules            = 0x1;
  auto unsupported            = std::make_shared<RecordingLaunch>();
  const tw::CandidateSet nset = with(load_spec(narrow), unsupported);
  CHECK(scored_indices(nset.rank(p, hw, context(0, tw::Schedule::Dynamic))).empty());
  CHECK(unsupported->asked.empty());
  CHECK_THROWS_AS(with(m, std::make_shared<ThrowingLaunch>()).rank(p, hw), std::runtime_error);
}

// Records the configs a ranking asks it to prefetch and to plan, in call
// order: (true, index) for prefetch(), (false, index) for plan(). The record
// is reserved up front so that prefetch(), which must not throw, never
// allocates.
struct PrefetchLaunch final : tw::LaunchModel {
  static constexpr std::size_t kRoom = 1024;
  mutable std::vector<std::pair<bool, std::size_t>> calls;

  PrefetchLaunch() { calls.reserve(kRoom); }

  tw::LaunchPlan plan(const tw::Config& c,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    calls.emplace_back(false, c.index);
    return index_plan(c);
  }
  void prefetch(const tw::Config& c) const noexcept override {
    if (calls.size() < kRoom) calls.emplace_back(true, c.index);
  }

  // The configs of the prefetch() calls, or of the plan() calls, ascending.
  std::vector<std::size_t> asked(bool prefetched) const {
    std::vector<std::size_t> out;
    for (const auto& [f, i] : calls)
      if (f == prefetched) out.push_back(i);
    std::sort(out.begin(), out.end());
    return out;
  }
  // Whether each plan() call for a config of `configs` came after a
  // prefetch() call for it.
  bool prefetched_first(const std::vector<std::size_t>& configs) const {
    for (auto at = calls.begin(); at != calls.end(); ++at) {
      const auto& [f, i] = *at;
      if (f || std::find(configs.begin(), configs.end(), i) == configs.end()) continue;
      if (std::find(calls.begin(), at, std::make_pair(true, i)) == at) return false;
    }
    return true;
  }
};

TEST_CASE("a ranking asks the launch model to prefetch the whitelist tier it will plan",
          "[v3][launch]") {
  const tw::Hardware hw              = test_hardware();
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const tw::ExecutionContext excl    = {};
  const std::vector<tw::Config> pool = variant_pool();
  // Every cell whitelists signatures 1 and 2; configs 7 and 8 are variants of
  // them, and every config is feasible for `p`.
  ModelSpec keyed = v3_grid({});
  for (CellSpec& cell : keyed.cells)
    cell.view_key_hashes = {key_hash(pool[1], {}), key_hash(pool[2], {})};
  const tw::ModelPtr m = load_spec(keyed);
  const auto with = [&](const tw::ModelPtr& model, std::shared_ptr<const tw::LaunchModel> launch) {
    return tw::CandidateSet(model, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, launch});
  };
  // The same plans without the hook. Every ranking below is of a new
  // candidate set, so the re-rank memo answers none of them.
  const auto unhooked = [&](const tw::ModelPtr& model, std::size_t min_scored) {
    auto rec = std::make_shared<RecordingLaunch>();
    rec->fn  = &index_plan;
    return with(model, rec).rank(p, hw, excl, min_scored);
  };
  // The first configs of the whitelisted keys, which the whitelist tier plans
  // and, with TW_LEVER_PREFETCH, asks to prefetch first.
  const std::vector<std::size_t> tier = {1, 2};
#if TW_LEVER_PREFETCH
  const std::vector<std::size_t> prefetched = tier;
#else
  const std::vector<std::size_t> prefetched;
#endif

  // The whitelist tier alone: unprepared, prepared, and through rank_top.
  for (int mode = 0; mode < 3; ++mode) {
    auto launch                = std::make_shared<PrefetchLaunch>();
    const tw::CandidateSet set = with(m, launch);
    if (mode == 1) set.prepare();
    if (mode == 2) {
      CHECK(same_results(set.rank_top(p, hw, excl, 1), scored_prefix(unhooked(m, 1), 1)));
    } else {
      const std::vector<tw::Result> r = set.rank(p, hw, excl, 0);
      std::vector<std::size_t> scored = scored_indices(r);
      std::sort(scored.begin(), scored.end());
      CHECK(scored == std::vector<std::size_t>{1, 2, 7, 8});
      CHECK(same_results(r, unhooked(m, 0)));
    }
    CHECK(launch->asked(true) == prefetched);
    CHECK(launch->asked(false) == tier);
    CHECK(launch->prefetched_first(prefetched));
  }

  // Both tiers: the second, every other feasible key, is planned only.
  {
    auto launch                     = std::make_shared<PrefetchLaunch>();
    const std::vector<tw::Result> r = with(m, launch).rank(p, hw, excl, kAll);
    CHECK(scored_indices(r).size() == pool.size());
    CHECK(same_results(r, unhooked(m, kAll)));
    CHECK(launch->asked(true) == prefetched);
    CHECK(launch->asked(false) == std::vector<std::size_t>{0, 1, 2, 3, 4, 5});
    CHECK(launch->prefetched_first(prefetched));
  }

  // Nothing is prefetched without a whitelist, and nothing is asked for by a
  // v2 model or for an unlisted schedule.
  {
    const tw::ModelPtr open = load_spec(v3_grid({}));
    auto launch             = std::make_shared<PrefetchLaunch>();
    CHECK(same_results(with(open, launch).rank(p, hw, excl, 0), unhooked(open, 0)));
    CHECK(launch->asked(true).empty());
    CHECK(launch->asked(false) == std::vector<std::size_t>{0, 1, 2, 3, 4, 5});

    auto v2 = std::make_shared<PrefetchLaunch>();
    with(load_spec(grid_model()), v2).rank(p, hw, kAll);
    CHECK(v2->calls.empty());

    ModelSpec narrow = keyed;
    narrow.schedules = 0x1;
    auto unsupported = std::make_shared<PrefetchLaunch>();
    CHECK(scored_indices(
              with(load_spec(narrow), unsupported).rank(p, hw, context(0, tw::Schedule::Dynamic)))
              .empty());
    CHECK(unsupported->calls.empty());
  }
}

// Launches by occupancy, its launch key.
struct OccupancyLaunch final : tw::LaunchModel {
  mutable std::vector<std::size_t> asked;
  mutable std::size_t keys_asked = 0;
  bool throw_key                 = false;
  bool zero_key                  = false;

  tw::LaunchPlan plan(const tw::Config& c,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    asked.push_back(c.index);
    return {c.occupancy > 2,
            static_cast<std::size_t>(57 * c.occupancy + 3),
            tw::PartialReduction::Tree};
  }
  std::uint64_t launch_key(const tw::Config& c) const override {
    ++keys_asked;
    if (throw_key) throw std::runtime_error("no key");
    return zero_key ? 0 : static_cast<std::uint64_t>(c.occupancy);
  }
};

TEST_CASE("launch keys split a signature into its launch classes", "[v3][launch]") {
  const tw::Hardware hw              = test_hardware();
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const std::vector<tw::Config> pool = variant_pool();
  const tw::ModelPtr m               = load_spec(with_launch_keys(v3_grid({})));
  auto keyed                         = std::make_shared<OccupancyLaunch>();
  const tw::PoolOptions options{tw::TieBreak::PoolOrder, keyed};
  REQUIRE(pool[0].occupancy != pool[6].occupancy);

  CHECK(tw::detail::pool_keys(*m, pool).first.size() == 6);
  CHECK(tw::detail::pool_keys(*m, pool, std::make_shared<RecordingLaunch>().get()).first.size() ==
        6);
  const tw::detail::PoolKeys keys = tw::detail::pool_keys(*m, pool, keyed.get());
  CHECK(keys.first.size() == 12);
  // v2 view keys already hold occupancy; launch keys never reach them.
  const tw::ModelPtr v2 = load_spec(grid_model());
  CHECK(tw::detail::pool_keys(*v2, pool, keyed.get()).key_of ==
        tw::detail::pool_keys(*v2, pool).key_of);

  // Each launch class is planned on its own, so grouping is exact.
  const tw::CandidateSet set(m, pool, options);
  const std::vector<tw::Result> got = set.rank(p, hw, kAll);
  std::vector<std::size_t> asked    = keyed->asked;
  std::sort(asked.begin(), asked.end());
  std::vector<std::size_t> all(pool.size());
  for (std::size_t i = 0; i < all.size(); ++i) all[i] = i;
  CHECK(asked == all);
  CHECK(same_results(got, rank_singly(*m, p, tw::ExecutionContext{}, pool, keyed.get())));
  CHECK_FALSE(same_results(got, tw::CandidateSet(m, pool).rank(p, hw, kAll)));

  // Configs that launch alike stay one candidate.
  std::vector<tw::Config> twice = pool;
  for (std::size_t i = 0; i < pool.size(); ++i) {
    tw::Config c = pool[i];
    c.grvw_a     = 4;
    c.index      = pool.size() + i;
    twice.push_back(c);
  }
  CHECK(tw::detail::pool_keys(*m, twice, keyed.get()).first.size() == 12);
  CHECK(same_results(tw::CandidateSet(m, twice, options).rank(p, hw, kAll),
                     rank_singly(*m, p, tw::ExecutionContext{}, twice, keyed.get())));

  // A whitelisted view key admits every launch class of its signature.
  ModelSpec keyed_wl = with_launch_keys(v3_grid({}));
  for (CellSpec& cell : keyed_wl.cells) cell.view_key_hashes = {key_hash(pool[8], {})};
  std::vector<std::size_t> admitted =
      scored_indices(tw::CandidateSet(load_spec(keyed_wl), pool, options).rank(p, hw));
  std::sort(admitted.begin(), admitted.end());
  CHECK(admitted == std::vector<std::size_t>{2, 8});

  // A cascade keeps view keys, with every launch class of each, and extends
  // by whole view keys for min_scored.
  ModelSpec cascade       = with_launch_keys(v3_grid({}));
  cascade.v3_cascade_keys = 3;
  const tw::ModelPtr cm   = load_spec(cascade);
  const auto scored_set   = [](const std::vector<tw::Result>& r) {
    std::vector<std::size_t> s = scored_indices(r);
    std::sort(s.begin(), s.end());
    return s;
  };
  for (const std::size_t min_scored : {std::size_t{0}, std::size_t{7}}) {
    keyed->asked.clear();
    const std::vector<tw::Result> top = tw::CandidateSet(cm, pool, options).rank(p, hw, min_scored);
    CHECK(scored_set(top) == scored_set(tw::CandidateSet(cm, pool).rank(p, hw, min_scored)));
    CHECK(scored_indices(top).size() == (min_scored == 0 ? 6u : 8u));
    CHECK(keyed->asked.size() == scored_indices(top).size());
    CHECK(same_results(
        top, tw::rank_configs(*cm, p, hw, tw::ExecutionContext{}, pool, options, min_scored)));
  }

  // Key errors propagate when the pool is grouped.
  auto throwing       = std::make_shared<OccupancyLaunch>();
  throwing->throw_key = true;
  CHECK_THROWS_AS(tw::CandidateSet(m, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, throwing}),
                  std::runtime_error);
}

TEST_CASE("rank_configs with pool options ranks like a CandidateSet", "[v3][launch]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  ModelSpec cascade                  = v3_grid({});
  cascade.v3_cascade_keys            = 4;
  for (const tw::ModelPtr& m : {load_spec(v3_grid({})),
                                load_spec(with_launch_keys(v3_grid({}))),
                                load_spec(cascade),
                                load_spec(with_launch_keys(cascade)),
                                load_spec(grid_model())}) {
    for (const tw::TieBreak tb : {tw::TieBreak::PoolOrder, tw::TieBreak::Prior}) {
      const tw::PoolOptions options{tb, std::make_shared<OccupancyLaunch>()};
      const tw::CandidateSet set(m, pool, options);
      std::vector<double> prior(pool.size());
      for (std::size_t i = 0; i < prior.size(); ++i) prior[i] = static_cast<double>((7 * i) % 5);
      for (const tw::Problem& p : {make_problem(1000, 2000, 512), make_problem(64, 64, 64, 4)}) {
        for (const tw::ExecutionContext& ctx :
             {tw::ExecutionContext{}, context(16, tw::Schedule::Dynamic)}) {
          for (std::size_t min_scored : {std::size_t{0}, std::size_t{3}, kAll}) {
            const std::vector<tw::Result> a = set.rank(p, hw, ctx, min_scored, &prior);
            CHECK(same_results(
                a, tw::rank_configs(*m, p, hw, ctx, pool, options, min_scored, &prior)));
          }
        }
      }
    }
  }
  const tw::ModelPtr m = load_spec(with_launch_keys(v3_grid({})));
  const tw::Problem p  = make_problem(1000, 2000, 512);
  const tw::PoolOptions none{};
  CHECK(same_results(tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, none, kAll),
                     tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, kAll)));
  const tw::PoolOptions launched{tw::TieBreak::PoolOrder, std::make_shared<OccupancyLaunch>()};
  CHECK_FALSE(
      same_results(tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, launched, kAll),
                   tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, kAll)));
}

TEST_CASE("a cascade keeps the earliest of the keys its embedding term ties", "[v3][cascade]") {
  // Values outside the one-hot's give equal item features to keys that the
  // attribute keeps apart.
  ModelSpec spec       = v3_grid({{"x", 0, 3, {1, 2}}});
  spec.v3_cascade_keys = 4;
  const tw::ModelPtr m = load_spec(spec);
  std::vector<tw::Config> pool;
  for (std::int64_t x = 10; x < 50; ++x) {
    pool.push_back(make_config(128, 128, 64));
    pool.back().attributes = {{"x", x}};
  }
  for (std::size_t mt : {32, 64, 256}) pool.push_back(make_config(mt, 128, 64));
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  REQUIRE(tw::detail::pool_keys(*m, pool).first.size() == pool.size());
  const tw::Hardware hw = test_hardware();
  const tw::CandidateSet set(m, pool);
  for (const tw::Problem& p : {make_problem(1000, 2000, 512),
                               make_problem(64, 64, 64, 4),
                               make_problem(4096, 4096, 4096)}) {
    for (std::size_t min_scored :
         {std::size_t{0}, std::size_t{1}, std::size_t{4}, std::size_t{9}}) {
      const std::vector<tw::Result> r = set.rank(p, hw, min_scored);
      CHECK(same_results(r, tw::rank_configs(*m, p, hw, pool, min_scored)));
      std::vector<std::size_t> tied;
      for (const tw::Result& x : r)
        if (x.scored && x.config_index < 40) tied.push_back(x.config_index);
      std::sort(tied.begin(), tied.end());
      for (std::size_t i = 0; i < tied.size(); ++i) CHECK(tied[i] == i);
    }
  }
}

TEST_CASE("launch plans reach rank_top, the tier-2 cascade, a bounded cache and prepare",
          "[v3][launch]") {
  const tw::Hardware hw              = test_hardware();
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const std::vector<tw::Config> pool = variant_pool();
  ModelSpec spec                     = v3_grid({});
  spec.v3_cascade_keys               = 2;
  for (CellSpec& cell : spec.cells) cell.view_key_hashes = {key_hash(pool[8], {})};
  const tw::ModelPtr m  = load_spec(spec);
  const auto asked_once = [](const RecordingLaunch& r) {
    std::vector<std::size_t> a = r.asked;
    std::sort(a.begin(), a.end());
    return std::adjacent_find(a.begin(), a.end()) == a.end() &&
           std::all_of(a.begin(), a.end(), [](std::size_t i) { return i < 6; });
  };
  auto rec = std::make_shared<RecordingLaunch>();
  rec->fn  = &index_plan;
  const tw::CandidateSet set(m, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, rec});

  // Tier 1 is the whitelisted signature 2; three configs need two cascade
  // keys of tier 2. Each scored key is planned once, by its first config.
  const std::vector<tw::Result> r       = set.rank(p, hw, 3);
  const std::vector<std::size_t> scored = scored_indices(r);
  REQUIRE(scored.size() == 6);
  CHECK(scored[0] + scored[1] == 2 + 8);
  CHECK(rec->asked.size() == 3);
  CHECK(asked_once(*rec));
  for (std::size_t i : rec->asked)
    CHECK(std::count(scored.begin(), scored.end(), i) +
              std::count(scored.begin(), scored.end(), i + 6) ==
          2);
  CHECK(same_results(r,
                     tw::detail::rank_keys(*m,
                                           p,
                                           hw,
                                           tw::ExecutionContext{},
                                           pool,
                                           tw::detail::pool_keys(*m, pool),
                                           3,
                                           tw::TieBreak::PoolOrder,
                                           nullptr,
                                           rec.get())));

  // rank_top plans the same keys and returns the scored prefix.
  for (std::size_t k : {1, 3, 7, 12}) {
    rec->asked.clear();
    const std::vector<tw::Result> full   = set.rank(p, hw, k);
    const std::vector<std::size_t> plans = rec->asked;
    rec->asked.clear();
    CHECK(same_results(set.rank_top(p, hw, {}, k), scored_prefix(full, k)));
#if TW_LEVER_MEMO && !TW_LEVER_MEMO_CHECK
    // The re-rank memo without its plan check answers rank_top(1) from the
    // whitelist tier the rank before it scored, and asks for no plan.
    CHECK(rec->asked == (k == 1 ? std::vector<std::size_t>{} : plans));
#else
    CHECK(rec->asked == plans);
#endif
    CHECK(asked_once(*rec));
  }

  // prepare() asks for no plan; with a cache that keeps nothing, the results
  // and the plans asked for stay the same.
  const std::size_t saved = tw::cache_limit();
  for (std::size_t limit : {std::size_t{1}, saved}) {
    tw::set_cache_limit(limit);
    auto fresh = std::make_shared<RecordingLaunch>();
    fresh->fn  = &index_plan;
    const tw::CandidateSet other(m, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, fresh});
    other.prepare(p, hw);
    other.prepare();
    CHECK(fresh->asked.empty());
    for (int round = 0; round < 2; ++round) {
      fresh->asked.clear();
      CHECK(same_results(other.rank(p, hw, 3), r));
      CHECK(fresh->asked.size() == 3);
    }
  }
  tw::set_cache_limit(saved);
}

TEST_CASE("launch keys hold through rank_top, both cascades, the epoch and a bounded cache",
          "[v3][launch]") {
  const tw::Hardware hw              = test_hardware();
  const std::vector<tw::Config> pool = variant_pool();
  std::vector<tw::Config> other      = pool;
  other.resize(5);
  const auto scored_set = [](const std::vector<tw::Result>& r) {
    std::vector<std::size_t> s = scored_indices(r);
    std::sort(s.begin(), s.end());
    return s;
  };
  ModelSpec tier1       = with_launch_keys(v3_grid({}));
  tier1.v3_cascade_keys = 2;
  ModelSpec tier2       = tier1;
  for (CellSpec& cell : tier2.cells) cell.view_key_hashes = {key_hash(pool[8], {})};
  const std::vector<tw::Problem> problems = {make_problem(1000, 2000, 512),
                                             make_problem(64, 64, 64, 4)};
  const std::vector<std::size_t> depths   = {0, 1, 3, 7, kAll};
  const std::size_t saved                 = tw::cache_limit();
  for (const ModelSpec& spec : {tier1, tier2}) {
    const tw::ModelPtr m = load_spec(spec);
    const tw::PoolOptions keyed{tw::TieBreak::PoolOrder, std::make_shared<OccupancyLaunch>()};
    REQUIRE(tw::detail::pool_keys(*m, pool, keyed.launch.get()).views() == 6);
    const tw::CandidateSet set(m, pool, keyed), plain(m, pool), small(m, other, keyed);
    std::vector<std::vector<tw::Result>> expect, expect_small;
    for (const tw::Problem& p : problems)
      for (std::size_t d : depths) {
        const std::vector<tw::Result> r = set.rank(p, hw, d);
        // The cascade width counts view keys, so the configs scored are those
        // of the same view keys as without launch keys.
        CHECK(scored_set(r) == scored_set(plain.rank(p, hw, d)));
        CHECK(same_results(r, tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, keyed, d)));
        for (std::size_t k : {std::size_t{1}, std::size_t{3}, std::size_t{12}, std::size_t{32}})
          CHECK(same_results(set.rank_top(p, hw, {}, k), scored_prefix(set.rank(p, hw, k), k)));
        expect.push_back(r);
        expect_small.push_back(small.rank(p, hw, d));
      }

    // Interleaved pools, an epoch passing its largest value and caches that
    // keep nothing or are prepared leave every ranking unchanged.
    for (std::size_t limit : {std::size_t{1}, saved}) {
      tw::set_cache_limit(limit);
      const tw::CandidateSet fresh(m, pool, keyed);
      if (limit == saved) fresh.prepare(problems[0], hw);
      for (std::uint32_t start : {0x7FFFFFFFu - 3u, 0x7FFFFFFFu}) {
        tw::detail::set_rank_epoch(start);
        for (int round = 0; round < 2; ++round) {
          std::size_t i = 0;
          for (const tw::Problem& p : problems)
            for (std::size_t d : depths) {
              INFO("limit " << limit << " start " << start << " round " << round << " i " << i);
              CHECK(same_results(fresh.rank(p, hw, d), expect[i]));
              CHECK(same_results(small.rank(p, hw, d), expect_small[i]));
              CHECK(same_results(fresh.rank_top(p, hw, {}, d), scored_prefix(expect[i], d)));
              ++i;
            }
        }
      }
    }
    tw::set_cache_limit(saved);
  }
}

TEST_CASE("a model that does not serve launch keys ranks as if every key were 0", "[v3][launch]") {
  const tw::Hardware hw                   = test_hardware();
  const std::vector<tw::Config> pool      = variant_pool();
  const std::vector<tw::Problem> problems = {make_problem(1000, 2000, 512),
                                             make_problem(64, 64, 64, 4)};
  const std::vector<std::size_t> depths   = {0, 1, 3, 7, kAll};
  ModelSpec cascade                       = v3_grid({});
  cascade.v3_cascade_keys                 = 2;
  ModelSpec whitelisted                   = cascade;
  for (CellSpec& cell : whitelisted.cells) cell.view_key_hashes = {key_hash(pool[8], {})};
  for (const ModelSpec& spec : {v3_grid({}), cascade, whitelisted}) {
    const tw::ModelPtr m     = load_spec(spec);
    const tw::ModelPtr split = load_spec(with_launch_keys(spec));
    auto keyed               = std::make_shared<OccupancyLaunch>();
    auto zero                = std::make_shared<OccupancyLaunch>();
    zero->zero_key           = true;
    auto throwing            = std::make_shared<OccupancyLaunch>();
    throwing->throw_key      = true;
    const tw::PoolOptions with_keys{tw::TieBreak::PoolOrder, keyed};
    const tw::PoolOptions with_zero{tw::TieBreak::PoolOrder, zero};
    CHECK(tw::detail::pool_keys(*m, pool, keyed.get()).key_of ==
          tw::detail::pool_keys(*m, pool).key_of);
    const OccupancyLaunch occupancy;
    CHECK(tw::detail::pool_keys(*split, pool, &occupancy).first.size() == 12);
    const tw::CandidateSet set(m, pool, with_keys), plain(m, pool, with_zero),
        unkeyed(m, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, throwing});
    const tw::CandidateSet grouped(
        split, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, std::make_shared<OccupancyLaunch>()});
    bool split_differs = false;
    for (const tw::Problem& p : problems)
      for (std::size_t d : depths) {
        const std::vector<tw::Result> want = plain.rank(p, hw, d);
        CHECK(same_results(set.rank(p, hw, d), want));
        CHECK(same_results(unkeyed.rank(p, hw, d), want));
        CHECK(same_results(set.rank_top(p, hw, {}, d), plain.rank_top(p, hw, {}, d)));
        CHECK(same_results(tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, with_keys, d),
                           want));
        split_differs |= !same_results(grouped.rank(p, hw, d), want);
      }
    CHECK(split_differs);
    CHECK(keyed->keys_asked == 0);
    CHECK(throwing->keys_asked == 0);
  }
}

// A launch model with a fixed id and the default plans.
struct NamedLaunch final : tw::LaunchModel {
  explicit NamedLaunch(std::string name) : name(std::move(name)) {}
  std::string name;
  tw::LaunchPlan plan(const tw::Config&,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    return {};
  }
  std::string id() const override { return name; }
};

struct UnnamedLaunch final : tw::LaunchModel {
  tw::LaunchPlan plan(const tw::Config&,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    return {};
  }
};

TEST_CASE("a v3 model that records its launch model ranks only with it",
          "[v3][launch][provenance]") {
  ModelSpec spec                     = v3_grid({});
  const tw::ModelPtr plain           = load_spec(spec);
  spec.provenance                    = {{"launch_model", "rule:gfxtest:3"}};
  const tw::ModelPtr named           = load_spec(spec);
  const std::vector<tw::Config> pool = variant_pool();
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const tw::Hardware hw              = test_hardware();
  const auto set = [&](const tw::ModelPtr& m, std::shared_ptr<const tw::LaunchModel> launch) {
    return tw::CandidateSet(m, pool, tw::PoolOptions{tw::TieBreak::PoolOrder, std::move(launch)});
  };
  const auto error = [&](std::shared_ptr<const tw::LaunchModel> launch) {
    try {
      set(named, std::move(launch));
    } catch (const std::invalid_argument& e) { return std::string(e.what()); }
    return std::string();
  };

  CHECK(same_results(set(named, std::make_shared<NamedLaunch>("rule:gfxtest:3")).rank(p, hw, kAll),
                     set(plain, nullptr).rank(p, hw, kAll)));
  CHECK(mentions(error(nullptr), "launch model 'rule:gfxtest:3' and the candidate set has none"));
  CHECK(mentions(error(std::make_shared<NamedLaunch>("rule:gfxtest:4")),
                 "launch model 'rule:gfxtest:3' and the candidate set's launch model is "
                 "'rule:gfxtest:4'"));
  CHECK(mentions(error(std::make_shared<UnnamedLaunch>()),
                 "launch model 'rule:gfxtest:3' and the candidate set's launch model is ''"));
  CHECK(UnnamedLaunch().id().empty());

  // rank_configs with pool options refuses the same launch models.
  const auto ad_hoc = [&](const tw::ModelPtr& m, std::shared_ptr<const tw::LaunchModel> launch) {
    return tw::rank_configs(*m,
                            p,
                            hw,
                            tw::ExecutionContext{},
                            pool,
                            tw::PoolOptions{tw::TieBreak::PoolOrder, std::move(launch)},
                            kAll);
  };
  const auto ad_hoc_error = [&](std::shared_ptr<const tw::LaunchModel> launch) {
    try {
      ad_hoc(named, std::move(launch));
    } catch (const std::invalid_argument& e) { return std::string(e.what()); }
    return std::string();
  };
  CHECK(same_results(ad_hoc(named, std::make_shared<NamedLaunch>("rule:gfxtest:3")),
                     set(plain, nullptr).rank(p, hw, kAll)));
  CHECK(ad_hoc_error(nullptr) == error(nullptr));
  CHECK(ad_hoc_error(std::make_shared<NamedLaunch>("rule:gfxtest:4")) ==
        error(std::make_shared<NamedLaunch>("rule:gfxtest:4")));
  CHECK(ad_hoc_error(std::make_shared<UnnamedLaunch>()) ==
        error(std::make_shared<UnnamedLaunch>()));
  CHECK_FALSE(ad_hoc_error(nullptr).empty());
  for (const std::shared_ptr<const tw::LaunchModel>& launch :
       {std::shared_ptr<const tw::LaunchModel>(),
        std::shared_ptr<const tw::LaunchModel>(std::make_shared<NamedLaunch>("rule:gfxtest:4"))})
    CHECK(same_results(ad_hoc(plain, launch), set(plain, nullptr).rank(p, hw, kAll)));

  // A model that records none takes any launch model, or none.
  const std::vector<std::shared_ptr<const tw::LaunchModel>> any = {
      nullptr, std::make_shared<UnnamedLaunch>(), std::make_shared<NamedLaunch>("rule:gfxtest:4")};
  for (const std::shared_ptr<const tw::LaunchModel>& launch : any)
    CHECK(
        same_results(set(plain, launch).rank(p, hw, kAll), set(plain, nullptr).rank(p, hw, kAll)));
}

TEST_CASE("v3 models declare the contexts they support", "[v3][context]") {
  ModelSpec spec        = v3_grid();
  spec.min_cu_fraction  = 0.25;
  spec.schedules        = 0x3;
  const tw::ModelPtr m  = load_spec(spec);
  const tw::Hardware hw = test_hardware();
  CHECK(tw::supports(*m, {}, hw));
  CHECK(tw::supports(*m, context(16, tw::Schedule::Default), hw));
  CHECK(tw::supports(*m, context(32, tw::Schedule::Dynamic), hw));
  CHECK(tw::supports(*m, context(64, tw::Schedule::Default), hw));
  CHECK(tw::supports(*m, context(1000, tw::Schedule::Dynamic), hw));
  CHECK_FALSE(tw::supports(*m, context(15, tw::Schedule::Default), hw));
  CHECK_FALSE(tw::supports(*m, context(1, tw::Schedule::Default), hw));
  CHECK_FALSE(tw::supports(*m, context(0, tw::Schedule::Auto), hw));
  CHECK_FALSE(tw::supports(*m, context(0, static_cast<tw::Schedule>(3)), hw));
  CHECK_FALSE(tw::supports(*m, {}, tw::Hardware{0, 65536, 1u << 20}));

  ModelSpec partial       = v3_grid();
  partial.min_cu_fraction = 0.25;
  partial.max_cu_fraction = 0.5;
  partial.schedules       = 0x4;
  const tw::ModelPtr q    = load_spec(partial);
  CHECK_FALSE(tw::supports(*q, context(0, tw::Schedule::Auto), hw));
  CHECK(tw::supports(*q, context(32, tw::Schedule::Auto), hw));
  CHECK_FALSE(tw::supports(*q, context(33, tw::Schedule::Auto), hw));

  const std::vector<tw::Config> pool = {make_config(128, 128, 64), make_config(64, 64, 64)};
  const tw::Problem p                = make_problem(1000, 2000, 512);
  const tw::CandidateSet set(m, pool);
  const std::vector<tw::Result> r16 = set.rank(p, hw, context(16, tw::Schedule::Default));
  const std::vector<tw::Result> r64 = set.rank(p, hw);
  CHECK(scored_indices(r16).size() == 2);
  CHECK(scored_indices(r64).size() == 2);
  CHECK_FALSE(same_results(r16, r64));
  // A budget below the range is scored as the smallest budget in it, bit
  // for bit, with its own schedule; a schedule the model does not list
  // leaves every config unscored.
  CHECK(tw::scoring_context(*m, context(15, tw::Schedule::Default), hw)->cu_budget == 16);
  CHECK(tw::scoring_context(*m, context(1, tw::Schedule::Dynamic), hw)->cu_budget == 16);
  CHECK(tw::scoring_context(*m, context(1, tw::Schedule::Dynamic), hw)->schedule ==
        tw::Schedule::Dynamic);
  CHECK(tw::scoring_context(*m, context(64, tw::Schedule::Default), hw)->cu_budget == 64);
  CHECK_FALSE(tw::scoring_context(*m, context(0, tw::Schedule::Auto), hw));
  CHECK_FALSE(tw::scoring_context(*m, {}, tw::Hardware{0, 65536, 1u << 20}));
  CHECK(same_results(set.rank(p, hw, context(15, tw::Schedule::Default)), r16));
  CHECK(same_results(set.rank(p, hw, context(1, tw::Schedule::Default)), r16));
  CHECK(scored_indices(set.rank(p, hw, context(0, tw::Schedule::Auto))).empty());
  // Above the range, the largest budget in it.
  const tw::CandidateSet partial_set(q, pool);
  const std::vector<tw::Result> r32 = partial_set.rank(p, hw, context(32, tw::Schedule::Auto));
  CHECK(scored_indices(r32).size() == 2);
  CHECK(tw::scoring_context(*q, context(0, tw::Schedule::Auto), hw)->cu_budget == 32);
  CHECK(same_results(partial_set.rank(p, hw, context(33, tw::Schedule::Auto)), r32));
  CHECK(same_results(partial_set.rank(p, hw, context(0, tw::Schedule::Auto)), r32));
  CHECK(scored_indices(partial_set.rank(p, hw, context(32, tw::Schedule::Default))).empty());
}

TEST_CASE("a CU floor raises a context with fewer usable CUs to it, or a smaller device whole",
          "[v3][context]") {
  ModelSpec plain       = v3_grid();
  plain.min_cu_fraction = 0.25;
  plain.schedules       = 0x3;
  ModelSpec floored     = plain;
  floored.v3_serving_flags |= 0x100;
  floored.min_effective_cus    = 16;
  const tw::ModelPtr m         = load_spec(floored);
  const tw::ModelPtr unfloored = load_spec(plain);
  CHECK(tw::describe(*m).min_effective_cus == 16);
  CHECK(tw::describe(*unfloored).min_effective_cus == 0);

  const auto device = [](std::size_t n_cu) { return tw::Hardware{n_cu, 65536, 4u << 20}; };
  for (std::size_t n_cu : {64u, 32u, 17u, 16u, 15u, 12u, 1u}) {
    const tw::Hardware hw = device(n_cu);
    for (std::size_t budget = 0; budget <= 80; ++budget)
      for (tw::Schedule schedule :
           {tw::Schedule::Default, tw::Schedule::Dynamic, tw::Schedule::Auto}) {
        const tw::ExecutionContext ctx = context(budget, schedule);
        const std::size_t usable       = budget > 0 && budget < n_cu ? budget : n_cu;
        INFO("device " << n_cu << " budget " << budget << " schedule "
                       << static_cast<int>(schedule));
        CHECK(tw::supports(*m, ctx, hw) == (tw::supports(*unfloored, ctx, hw) && usable >= 16));
      }
  }
  // The floor reaches below the fraction range only on a smaller device.
  CHECK(tw::supports(*m, context(16, tw::Schedule::Default), device(64)));
  CHECK_FALSE(tw::supports(*m, context(15, tw::Schedule::Default), device(64)));
  CHECK(tw::supports(*m, context(16, tw::Schedule::Dynamic), device(32)));
  CHECK_FALSE(tw::supports(*m, context(8, tw::Schedule::Default), device(32)));
  CHECK(tw::supports(*unfloored, context(8, tw::Schedule::Default), device(32)));
  CHECK(tw::supports(*m, {}, device(16)));
  CHECK_FALSE(tw::supports(*m, {}, device(15)));
  CHECK(tw::supports(*unfloored, {}, device(15)));
  CHECK_FALSE(tw::supports(*m, context(0, tw::Schedule::Auto), device(64)));

  // A supported context ranks as the model without the floor does, bit for
  // bit; any other ranks as the floor does on that device (the whole device
  // when it has fewer CUs than the floor), never unscored.
  std::vector<tw::Config> pool = {make_config(128, 128, 64),
                                  make_config(64, 64, 64),
                                  make_config(256, 128, 64),
                                  make_config(128, 128, 64, 16, 16, 32, 4, 0)};
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  const tw::Problem p = make_problem(1000, 2000, 512);
  const tw::CandidateSet set(m, pool), unfloored_set(unfloored, pool);
  std::size_t scored = 0, raised = 0;
  for (std::size_t n_cu : {64u, 32u, 16u, 15u})
    for (std::size_t budget : {0u, 8u, 15u, 16u, 20u, 31u, 48u})
      for (tw::Schedule schedule : {tw::Schedule::Default, tw::Schedule::Dynamic}) {
        const tw::Hardware hw          = device(n_cu);
        const tw::ExecutionContext ctx = context(budget, schedule);
        INFO("device " << n_cu << " budget " << budget);
        const std::vector<tw::Result> r = set.rank(p, hw, ctx, kAll);
        CHECK(same_results(r, tw::rank_configs(*m, p, hw, ctx, pool, kAll)));
        CHECK(same_results(set.rank_top(p, hw, ctx, 2), scored_prefix(set.rank(p, hw, ctx, 2), 2)));
        CHECK_FALSE(scored_indices(r).empty());
        const std::optional<tw::ExecutionContext> at = tw::scoring_context(*m, ctx, hw);
        REQUIRE(at.has_value());
        CHECK(at->schedule == schedule);
        if (tw::supports(*m, ctx, hw)) {
          ++scored;
          CHECK(at->cu_budget == budget);
          CHECK(same_results(r, unfloored_set.rank(p, hw, ctx, kAll)));
        } else {
          ++raised;
          // The floor of 16 or the fraction 0.25 of the device, whichever
          // is more, and at most the whole device.
          const std::size_t usable = budget > 0 && budget < n_cu ? budget : n_cu;
          const std::size_t floor =
              std::max<std::size_t>((n_cu + 3) / 4, std::min<std::size_t>(16, n_cu));
          CHECK(usable <= floor);
          CHECK(at->cu_budget == (floor < n_cu ? floor : 0));
          CHECK(same_results(r, set.rank(p, hw, *at, kAll)));
          CHECK(same_results(r, unfloored_set.rank(p, hw, *at, kAll)));
        }
      }
  CHECK(scored > 0);
  CHECK(raised > 0);
}

// Plans every config alike, with a grid that depends on the budget of the
// context it is asked for, and records those budgets.
struct BudgetLaunch final : tw::LaunchModel {
  mutable std::vector<std::size_t> budgets;
  tw::LaunchPlan plan(const tw::Config&,
                      const tw::Problem&,
                      const tw::ExecutionContext& context) const override {
    budgets.push_back(context.cu_budget);
    return {false, 3 + context.cu_budget, tw::PartialReduction::None};
  }
};

TEST_CASE("a CU budget outside a model's range ranks as the nearest budget it covers",
          "[v3][context]") {
  ModelSpec spec                     = v3_grid();
  spec.min_cu_fraction               = 0.25;
  spec.schedules                     = 0x3;
  const tw::ModelPtr m               = load_spec(spec);
  const std::vector<tw::Config> pool = variant_pool();
  const tw::Problem p                = make_problem(4096, 2048, 512);
  // A quarter of 255, 38 or 6 CUs is not a whole number of CUs: the fewest
  // usable CUs at fraction 0.25 is the next one above it there.
  for (std::size_t n_cu : {256u, 255u, 128u, 64u, 38u, 32u, 8u, 6u, 3u}) {
    const tw::Hardware hw{n_cu, 65536, 4u << 20};
    const std::size_t lo = (n_cu + 3) / 4;  // the fewest usable CUs at fraction 0.25
    const auto launch    = std::make_shared<BudgetLaunch>();
    const tw::PoolOptions options{tw::TieBreak::PoolOrder, launch};
    const tw::CandidateSet set(m, pool, options);
    for (std::size_t budget = 0; budget <= n_cu + 2; ++budget)
      for (tw::Schedule schedule : {tw::Schedule::Default, tw::Schedule::Dynamic}) {
        INFO("device " << n_cu << " budget " << budget);
        const tw::ExecutionContext ctx               = context(budget, schedule);
        const std::size_t usable                     = budget > 0 && budget < n_cu ? budget : n_cu;
        const std::optional<tw::ExecutionContext> at = tw::scoring_context(*m, ctx, hw);
        REQUIRE(at.has_value());
        CHECK(at->schedule == schedule);
        if (usable >= lo) {
          // In the range: the context itself, so the ranking is unchanged.
          CHECK(tw::supports(*m, ctx, hw));
          CHECK(at->cu_budget == budget);
        } else {
          CHECK_FALSE(tw::supports(*m, ctx, hw));
          CHECK(at->cu_budget == lo);
          CHECK(tw::supports(*m, *at, hw));
        }
        launch->budgets.clear();
        const std::vector<tw::Result> r = set.rank(p, hw, ctx, kAll);
        CHECK_FALSE(scored_indices(r).empty());
        CHECK_FALSE(launch->budgets.empty());
        // Launches are planned for the scoring context only.
        for (std::size_t b : launch->budgets) CHECK(b == at->cu_budget);
        CHECK(same_results(r, set.rank(p, hw, *at, kAll)));
        CHECK(same_results(r, tw::rank_configs(*m, p, hw, ctx, pool, options, kAll)));
        CHECK(same_results(set.rank_top(p, hw, ctx, 3), scored_prefix(set.rank(p, hw, ctx, 3), 3)));
      }
    // A schedule the model does not list stays unscored, at any budget.
    for (std::size_t budget : {std::size_t{0}, std::size_t{1}, lo, n_cu}) {
      CHECK_FALSE(tw::scoring_context(*m, context(budget, tw::Schedule::Auto), hw));
      CHECK(scored_indices(set.rank(p, hw, context(budget, tw::Schedule::Auto), kAll)).empty());
    }
  }
  CHECK_FALSE(
      tw::scoring_context(*m, context(8, tw::Schedule::Default), tw::Hardware{0, 65536, 4u << 20}));

  // A v2 model covers every CU only: every budget of the default schedule
  // ranks as exclusive use, another schedule not at all.
  const tw::ModelPtr v2 = load_spec(grid_model());
  const tw::Hardware hw = test_hardware();
  const tw::CandidateSet v2_set(v2, pool);
  const std::vector<tw::Result> exclusive = v2_set.rank(p, hw, kAll);
  CHECK_FALSE(scored_indices(exclusive).empty());
  for (std::size_t budget : {1u, 15u, 63u, 64u, 100u}) {
    INFO("budget " << budget);
    const tw::ExecutionContext ctx = context(budget, tw::Schedule::Default);
    CHECK(same_results(v2_set.rank(p, hw, ctx, kAll), exclusive));
    CHECK(tw::scoring_context(*v2, ctx, hw)->cu_budget == (budget >= hw.N_CU ? budget : 0));
  }
  CHECK_FALSE(tw::scoring_context(*v2, context(0, tw::Schedule::Dynamic), hw));
  CHECK(scored_indices(v2_set.rank(p, hw, context(0, tw::Schedule::Dynamic), kAll)).empty());
}

TEST_CASE("v3 soft feasibility rules apply only when the model selects them", "[v3][rules]") {
  const tw::Hardware hw = test_hardware();
  const auto scored     = [&](std::uint32_t flags, const tw::Problem& p, const tw::Config& c) {
    ModelSpec spec        = v3_grid({});
    spec.v3_serving_flags = flags;
    const tw::ModelPtr m  = load_spec(spec);
    return tw::rank_configs(*m, p, hw, {c}, kAll)[0].scored;
  };
  // Dot2 for M > 2.
  const tw::Problem m100 = make_problem(100, 512, 512);
  const tw::Config dot2  = make_config(16, 64, 64, 1, 1, 64);
  CHECK(scored(0, m100, dot2));
  CHECK(scored(0x05, m100, dot2));
  CHECK_FALSE(scored(0x02, m100, dot2));
  CHECK(scored(0x02, make_problem(2, 512, 512), dot2));
  // A small batched problem with a tile that does not cover it.
  const tw::Problem batched = make_problem(64, 64, 256, 4);
  const tw::Config small    = make_config(32, 32, 64);
  CHECK(scored(0, batched, small));
  CHECK_FALSE(scored(0x01, batched, small));
  CHECK(scored(0x01, batched, make_config(64, 64, 64)));
  // A hinted kernel on an aligned problem that is not skinny.
  const tw::Problem square = make_problem(4096, 4096, 4096);
  const tw::Config hinted  = make_config(256, 256, 64, 16, 16, 32, 1, 0);
  CHECK(scored(0, square, hinted));
  CHECK_FALSE(scored(0x04, square, hinted));

  // The LDS fit always applies. The leading free dimension reads the vector
  // widths, which v3 leaves to the integration's kernel predicates.
  CHECK_FALSE(scored(0, square, make_config(256, 256, 256)));
  const tw::Problem lead =
      make_problem(4, 512, 512, 1, tw::DataType::BFloat16, tw::Transpose::N, tw::Transpose::N);
  CHECK(scored(0, lead, make_config(16, 64, 64)));
  const tw::ModelPtr v2 = load_spec(grid_model());
  CHECK_FALSE(tw::rank_configs(*v2, lead, hw, {make_config(16, 64, 64)}, kAll)[0].scored);
  tw::Config narrow = make_config(16, 64, 64);
  narrow.grvw_a     = 4;
  CHECK(tw::rank_configs(*v2, lead, hw, {narrow}, kAll)[0].scored);
}

TEST_CASE("v3 tiers: keyed whitelists, the whitelist switch and the cascade", "[v3][tiers]") {
  const tw::Hardware hw            = test_hardware();
  const tw::Problem p              = make_problem(1000, 2000, 512);
  const std::vector<AttrSpec> attr = sample_attributes();
  std::vector<tw::Config> pool;
  for (std::int64_t s : {0, 8, 16, 32})
    pool.push_back(with_attributes(make_config(128, 128, 64), {{"param_a", s}}));
  pool.push_back(make_config(64, 64, 64));
  pool.push_back(make_config(256, 128, 64));

  // Keyed: only the listed view key; signatures still list whole signatures.
  ModelSpec keyed = v3_grid(attr);
  for (CellSpec& cell : keyed.cells) cell.view_key_hashes = {key_hash(pool[2], attr), 77};
  const tw::ModelPtr km = load_spec(keyed);
  const tw::CandidateSet ks(km, pool);
  CHECK(scored_indices(ks.rank(p, hw)) == std::vector<std::size_t>{2});
  CHECK(scored_indices(tw::rank_configs(*km, p, hw, pool)) == std::vector<std::size_t>{2});
  ModelSpec both = keyed;
  for (CellSpec& cell : both.cells) cell.signatures = {{64, 64, 64, 16, 16, 32, 0, 0}};
  const std::vector<std::size_t> listed =
      scored_indices(tw::rank_configs(*load_spec(both), p, hw, pool));
  CHECK(listed.size() == 2);
  CHECK(std::find(listed.begin(), listed.end(), 2) != listed.end());
  CHECK(std::find(listed.begin(), listed.end(), 4) != listed.end());

  // The switch drops the whitelist tier.
  ModelSpec off         = keyed;
  off.v3_serving_flags  = 0x08;
  const tw::ModelPtr om = load_spec(off);
  CHECK(scored_indices(tw::CandidateSet(om, pool).rank(p, hw)).size() == pool.size());
  CHECK_FALSE(tw::describe(*om).whitelist_tier);

  // The cascade gives the full score to the best keys by the embedding term.
  ModelSpec cascade       = v3_grid(attr);
  cascade.v3_cascade_keys = 2;
  const tw::ModelPtr cm   = load_spec(cascade);
  const tw::CandidateSet cs(cm, pool);
  CHECK(scored_indices(cs.rank(p, hw)).size() == 2);
  CHECK(scored_indices(cs.rank(p, hw, 4)).size() == 4);
  CHECK(same_results(cs.rank(p, hw), tw::rank_configs(*cm, p, hw, pool)));
}

TEST_CASE("v3 models cascade tier 2; v2 models score all of it", "[v3][tiers]") {
  const tw::Hardware hw = test_hardware();
  const tw::Problem p   = make_problem(1000, 2000, 512);
  std::vector<tw::Config> pool;
  for (std::size_t mt_m : {32, 64, 128, 256})
    for (std::size_t mt_n : {32, 64, 128}) pool.push_back(make_config(mt_m, mt_n, 32));
  const tw::Config& c5   = pool[5];
  const Signature listed = {static_cast<int>(c5.mt.m),
                            static_cast<int>(c5.mt.n),
                            static_cast<int>(c5.mt.k),
                            static_cast<int>(c5.mi.m),
                            static_cast<int>(c5.mi.n),
                            static_cast<int>(c5.mi.k),
                            c5.cache_hints_a,
                            c5.cache_hints_b};
  // Without an interaction tower the score is the embedding term, so the
  // cascade keeps the best keys of tier 2 by score.
  const auto embedding_only = [](CellTensors& t) {
    for (std::vector<float>* v : {&t.x_w0, &t.x_b0, &t.x_w2, &t.x_b2})
      std::fill(v->begin(), v->end(), 0.0f);
  };
  ModelSpec spec = v3_grid({});
  for (CellSpec& cell : spec.cells) {
    cell.signatures = {listed};
    cell.edit       = embedding_only;
  }
  ModelSpec narrow        = spec;
  narrow.v3_cascade_keys  = 2;
  const tw::ModelPtr full = load_spec(spec);
  const tw::ModelPtr cm   = load_spec(narrow);
  const tw::CandidateSet set(cm, pool);
  const std::vector<tw::Result> all = tw::rank_configs(*full, p, hw, pool, kAll);
  REQUIRE(scored_indices(all).size() == pool.size());
  REQUIRE(all[0].config_index == 5);
  for (std::size_t min_scored : {std::size_t{0},
                                 std::size_t{1},
                                 std::size_t{2},
                                 std::size_t{4},
                                 std::size_t{11},
                                 std::size_t{12},
                                 kAll}) {
    INFO("min_scored " << min_scored);
    const std::size_t tier2 =
        min_scored <= 1 ? 0 : std::min<std::size_t>(11, std::max<std::size_t>(2, min_scored - 1));
    const std::vector<tw::Result> r = tw::rank_configs(*cm, p, hw, pool, min_scored);
    std::vector<tw::Result> expect(all.begin(), all.begin() + 1 + static_cast<long>(tier2));
    CHECK(same_results(scored_prefix(r, kAll), expect));
    CHECK(same_results(set.rank(p, hw, min_scored), r));
    CHECK(same_results(set.rank_top(p, hw, {}, min_scored), scored_prefix(r, min_scored)));
  }

  ModelSpec v2 = grid_model();
  for (CellSpec& cell : v2.cells) cell.signatures = {listed};
  v2.cascade_keys = 2;
  CHECK(scored_indices(tw::rank_configs(*load_spec(v2), p, hw, pool, 2)).size() == pool.size());
  CHECK(scored_indices(tw::rank_configs(*load_spec(v2), p, hw, pool, 1)).size() == 1);
}

TEST_CASE("v3 bounds clamp raw features before whitening", "[v3][clip]") {
  const std::vector<AttrSpec> attr = {{"param_a", 0, 1, {}}};
  ModelSpec spec                   = v3_grid(attr);
  spec.clip                        = 1;
  for (CellSpec& cell : spec.cells) {
    cell.i_lo.assign(spec.i_dim, -std::numeric_limits<float>::infinity());
    cell.i_hi.assign(spec.i_dim, std::numeric_limits<float>::infinity());
    cell.i_lo[8] = 0.0f;
    cell.i_hi[8] = 2.0f;
    cell.x_lo.assign(spec.x_dim, -std::numeric_limits<float>::infinity());
    cell.x_hi.assign(spec.x_dim, std::numeric_limits<float>::infinity());
    cell.x_lo[0] = 2.0f;
    cell.x_hi[0] = 7.5f;
  }
  ModelSpec open        = spec;
  open.clip             = 0;
  const tw::ModelPtr cm = load_spec(spec);
  const tw::ModelPtr om = load_spec(open);
  const tw::Hardware hw = test_hardware();
  const tw::Problem p   = make_problem(1000, 2000, 512);

  std::vector<tw::Config> pool;
  for (std::int64_t s : {3, 15, 0})
    pool.push_back(with_attributes(make_config(128, 128, 64), {{"param_a", s}}));
  const std::vector<tw::Result> clipped = tw::rank_configs(*cm, p, hw, pool, kAll);
  const std::vector<tw::Result> plain   = tw::rank_configs(*om, p, hw, pool, kAll);
  const auto score_of                   = [](const std::vector<tw::Result>& r, std::size_t i) {
    for (const tw::Result& x : r)
      if (x.config_index == i) return x.score;
    return std::numeric_limits<double>::quiet_NaN();
  };
  CHECK(score_of(clipped, 0) == score_of(clipped, 1));
  CHECK(score_of(clipped, 0) != score_of(clipped, 2));
  CHECK(score_of(plain, 0) != score_of(plain, 1));
  CHECK(score_of(clipped, 0) == score_of(plain, 0));

  // The batched interaction path clamps as the one-at-a-time path does.
  Random rnd(5);
  const std::vector<tw::Config> big = rnd.pool(12, 40);
  for (const tw::ExecutionContext& ctx :
       {tw::ExecutionContext{}, context(16, tw::Schedule::Dynamic)})
    for (const tw::Problem& q : {p, make_problem(16, 8192, 64), make_problem(300, 300, 3000, 3)}) {
      const std::vector<tw::Result> together =
          tw::rank_configs(*cm, q, hw, ctx, big, kAll, tw::TieBreak::PoolOrder);
      CHECK(same_results(together, rank_singly(*cm, q, ctx, big)));
      CHECK(same_results(together, tw::CandidateSet(cm, big).rank(q, hw, ctx, kAll)));
      for (const tw::Result& x : together) {
        if (!x.scored) continue;
        const tw::Result alone = tw::rank_configs(
            *cm, q, hw, ctx, {big[x.config_index]}, kAll, tw::TieBreak::PoolOrder)[0];
        CHECK(std::memcmp(&alone.score, &x.score, sizeof(double)) == 0);
      }
    }
}

TEST_CASE("v3 models route every problem through the nearest trained base label", "[v3][route]") {
  ModelSpec spec = as_v3(ModelSpec{});
  for (const char* label : {"Tiny|Tiny|TinyK|Bnone", "Large|Large|LargeK|Bany"}) {
    CellSpec c;
    c.label = label;
    spec.cells.push_back(c);
  }
  // A base label in the tree whose leaves have no trained cell.
  spec.splits = {
      {"Mid|Mid|MidK|Bnone", 'M', 300, "Mid|Mid|MidK|Bnone#lo", "Mid|Mid|MidK|Bnone#hi"}};
  const tw::ModelPtr v3 = load_spec(spec);
  ModelSpec as_v2       = spec;
  as_v2.version         = 2;
  as_v2.feature_hash    = "e7fe4b524851e895";
  as_v2.q_dim           = 55;
  as_v2.i_dim           = 12;
  as_v2.x_dim           = 37;
  const tw::ModelPtr v2 = load_spec(as_v2);

  CHECK(tw::route(*v3, make_problem(16, 16, 16)) == 0);
  CHECK(tw::route(*v3, make_problem(64, 16, 16)) == 0);
  CHECK(tw::route(*v3, make_problem(4096, 4096, 512, 2)) == 1);
  CHECK(tw::route(*v3, make_problem(4096, 4096, 4096)) == 1);
  CHECK(tw::route(*v3, make_problem(400, 400, 400)) >= 0);
  CHECK(tw::route(*v2, make_problem(64, 16, 16)) == -1);
  CHECK(tw::route(*v2, make_problem(400, 400, 400)) == -1);
  CHECK(tw::route(*v2, make_problem(16, 16, 16)) == 0);
  CHECK(tw::describe(*v3).uncovered_base_labels == 93);

  const std::vector<tw::Config> pool = {make_config(64, 64, 64), make_config(32, 32, 32)};
  for (const tw::Problem& p :
       {make_problem(400, 400, 400), make_problem(100, 3000, 7), make_problem(9000, 2, 70000, 9)})
    CHECK_FALSE(scored_indices(tw::rank_configs(*v3, p, test_hardware(), pool)).empty());
}

TEST_CASE("v3 cell bounds hold every problem routed to the cell", "[v3][route][prune]") {
  Random rnd(41);
  ModelSpec split = as_v3(ModelSpec{});
  for (const char* label : {"Tiny|Tiny|TinyK|Bnone", "Large|Large|LargeK|Bany"}) {
    CellSpec c;
    c.label = label;
    split.cells.push_back(c);
  }
  split.splits = {
      {"Mid|Mid|MidK|Bnone", 'M', 300, "Mid|Mid|MidK|Bnone#lo", "Mid|Mid|MidK|Bnone#hi"}};
  for (const ModelSpec& spec : {v3_grid(), split}) {
    const tw::ModelPtr m                             = load_spec(spec);
    const std::vector<tw::detail::CellBounds> bounds = tw::detail::cell_bounds(*m);
    REQUIRE(bounds.size() == m->cells.size());
    for (int i = 0; i < 3000; ++i) {
      tw::Problem p = rnd.problem();
      if (i % 3 == 0) p.size = {1 + rnd.below(40), 1 + rnd.below(600), 1 + rnd.below(600)};
      const int cell = tw::detail::route_cell(*m, p);
      if (cell < 0) continue;
      const tw::detail::CellBounds& b = bounds[static_cast<std::size_t>(cell)];
      REQUIRE(b.reached);
      CHECK(b.m_lo <= p.size.m);
      CHECK(p.size.m <= b.m_hi);
      CHECK(b.n_lo <= p.size.n);
      CHECK(p.size.n <= b.n_hi);
      CHECK(p.size.k <= b.k_hi);
      if (p.batch == 1) CHECK(b.batch_one);
    }
  }
}

TEST_CASE("pruning and the feasibility shortcuts agree with the rules", "[v3][rules][prune]") {
  Random rnd(43);
  const tw::Hardware hw = test_hardware();
  const auto in         = [&](std::uint64_t lo, std::uint64_t hi) {
    return lo + static_cast<std::uint64_t>(rnd.below(static_cast<std::size_t>(hi - lo + 1)));
  };
  for (int round = 0; round < 300; ++round) {
    tw::detail::SoftRules soft;
    soft.small_batched_one_tile = rnd.below(2) != 0;
    soft.dot2_small_m           = rnd.below(2) != 0;
    soft.non_temporal           = rnd.below(4) != 0;
    soft.leading_free           = rnd.below(2) != 0;
    soft.non_temporal_family    = rnd.below(2) != 0;
    tw::detail::CellBounds b;
    b.reached                    = true;
    b.m_lo                       = rnd.pick<std::uint64_t>({1, 2, 3, 33, 129, 513});
    b.m_hi                       = b.m_lo + rnd.pick<std::uint64_t>({0, 31, 200, 5000});
    b.n_lo                       = rnd.pick<std::uint64_t>({1, 2, 33, 129, 513});
    b.n_hi                       = b.n_lo + rnd.pick<std::uint64_t>({0, 31, 200, 5000});
    b.k_hi                       = rnd.pick<std::uint64_t>({32, 512, 1023, 9000});
    b.k_lo                       = std::min(b.k_hi, rnd.pick<std::uint64_t>({1, 33, 513}));
    const int batches            = static_cast<int>(rnd.below(3));
    b.batch_one                  = batches != 1;
    b.batch_many                 = batches != 0;
    std::vector<tw::Config> pool = rnd.pool(12, 12);
    for (tw::Config& c : pool) c.cache_hints_b = rnd.pick<int>({0, 0, 1, 4});
    const std::vector<std::uint8_t> family = tw::detail::family_bits(pool);
    std::vector<tw::detail::KernelShape> shapes;
    bool nt_a = false, nt_b = false;
    for (std::size_t i = 0; i < pool.size(); ++i) {
      shapes.push_back(tw::detail::kernel_shape(pool[i]));
      shapes.back().family = family[i];
      nt_a |= pool[i].cache_hints_a == 4;
      nt_b |= pool[i].cache_hints_b == 4;
    }
    tw::detail::PlainBounds plain, hinted_bounds;
    for (const tw::detail::KernelShape& k : shapes)
      (k.cache_hints_a == 0 && k.cache_hints_b == 0 ? plain : hinted_bounds).include(k);
    for (int i = 0; i < 20; ++i) {
      tw::Problem p = rnd.problem();
      p.size        = {in(b.m_lo, b.m_hi), in(b.n_lo, b.n_hi), in(b.k_lo, b.k_hi)};
      if (rnd.below(2) != 0) p.size.k = 64 * in(1, std::max<std::uint64_t>(b.k_hi / 64, 1));
      p.size.k = std::min<std::uint64_t>(std::max(p.size.k, b.k_lo), b.k_hi);
      p.batch  = batches == 0   ? 1
                 : batches == 1 ? rnd.pick<std::size_t>({2, 8})
                                : rnd.pick<std::size_t>({1, 2});
      const tw::detail::Feasibility rules(p, hw, nt_a, nt_b, soft);
      const bool all_pass = rules.all_pass(plain), hinted = rules.hinted_may_pass();
      const bool all_but_non_temporal        = rules.all_but_non_temporal_pass(plain);
      const bool all_but_non_temporal_hinted = rules.all_but_non_temporal_pass(hinted_bounds);
      const auto non_temporal                = [&](const tw::detail::KernelShape& k) {
        return rules.non_temporal_rule(rules.tile_aligned(k.mt_k % 1024),
                                       rules.b_branch(k.mt_m),
                                       rules.a_branch(k.mt_n),
                                       tw::detail::hint_class(k.cache_hints_a, k.cache_hints_b),
                                       k.family);
      };
      if (all_pass) CHECK(all_but_non_temporal);
      for (const tw::detail::KernelShape& k : shapes) {
        const bool ok = rules(k);
        if (tw::detail::never_feasible(b, k, nt_a, nt_b, soft)) CHECK_FALSE(ok);
        if (k.cache_hints_a == 0 && k.cache_hints_b == 0) {
          if (all_pass) CHECK(ok);
          if (all_but_non_temporal) CHECK(non_temporal(k) == ok);
        } else {
          if (!hinted) CHECK_FALSE(ok);
          if (all_but_non_temporal_hinted) CHECK(non_temporal(k) == ok);
        }
      }
    }
  }
  tw::detail::CellBounds none;
  CHECK_FALSE(
      tw::detail::never_feasible(none,
                                 tw::detail::kernel_shape(make_config(16, 16, 16, 1, 1, 64)),
                                 false,
                                 false,
                                 tw::detail::SoftRules{}));
}

TEST_CASE("v3 pools prune keys without changing a ranking", "[v3][pool][prune]") {
  Random rnd(47);
  ModelSpec spec        = v3_grid({});
  spec.v3_serving_flags = 0x07;
  spec.v3_cascade_keys  = 4;
  for (CellSpec& cell : spec.cells)
    if (cell.seed % 4 == 0) cell.signatures = {{64, 64, 64, 16, 16, 32, 0, 4}};
  for (const bool keyed : {false, true}) {
    const tw::ModelPtr m = load_spec(keyed ? with_launch_keys(spec) : spec);
    const tw::PoolOptions options{tw::TieBreak::PoolOrder, std::make_shared<OccupancyLaunch>()};
    for (int round = 0; round < 6; ++round) {
      std::vector<tw::Config> pool;
      for (const tw::Config& c : rnd.pool(4 + rnd.below(12), 12 + rnd.below(30))) {
        for (int v = 0; v < 3; ++v) {
          tw::Config x    = c;
          x.cache_hints_a = v == 1 ? 4 : 0;
          x.cache_hints_b = v == 2 ? 4 : 0;
          x.occupancy     = 1 + rnd.below(3);
          x.index         = pool.size();
          pool.push_back(x);
        }
      }
      pool.push_back(make_config(64, 64, 64, 16, 16, 32, 0, 4));
      pool.back().index = pool.size() - 1;
      const tw::CandidateSet set(m, pool, options);
      if (round % 2 == 0) set.prepare();
      for (int i = 0; i < 40; ++i) {
        tw::Problem p = rnd.problem();
        switch (i % 4) {
          case 0:
            p.size = {1 + rnd.below(64), 64 * (1 + rnd.below(100)), 64 * (1 + rnd.below(60))};
            break;
          case 1:
            p.size = {64 * (1 + rnd.below(100)), 1 + rnd.below(64), 64 * (1 + rnd.below(60))};
            break;
          case 2:
            p.size  = {1 + rnd.below(256), 1 + rnd.below(256), 1 + rnd.below(1023)};
            p.batch = 2 + rnd.below(6);
            break;
          default: break;
        }
        const tw::ExecutionContext ctx =
            context(rnd.pick<std::size_t>({0, 32}), tw::Schedule::Default);
        for (std::size_t min_scored : {std::size_t{0}, std::size_t{1}, std::size_t{9}, kAll}) {
          const std::vector<tw::Result> a = set.rank(p, test_hardware(), ctx, min_scored);
          CHECK(same_results(
              a, tw::rank_configs(*m, p, test_hardware(), ctx, pool, options, min_scored)));
          CHECK(same_results(set.rank_top(p, test_hardware(), ctx, min_scored),
                             scored_prefix(a, min_scored)));
        }
      }
    }
  }
}

TEST_CASE("v3 pools rank alike through every entry point", "[v3][pool]") {
  Random rnd(31);
  ModelSpec spec       = v3_grid();
  spec.v3_cascade_keys = 5;
  for (CellSpec& cell : spec.cells)
    if (cell.seed % 3 == 0) cell.signatures = {{128, 128, 64, 16, 16, 32, 0, 0}};
  const tw::ModelPtr m = load_spec(spec);
  for (int round = 0; round < 6; ++round) {
    std::vector<tw::Config> pool = rnd.pool(2 + rnd.below(10), 1 + rnd.below(50));
    // Whitelisted configs, so that tier 2 forms.
    const std::size_t listed = 1 + rnd.below(3);
    for (std::size_t i = 0; i < listed; ++i) pool.push_back(make_config(128, 128, 64));
    for (tw::Config& c : pool)
      c.attributes = {{"param_a", static_cast<std::int64_t>(rnd.below(3)) * 16},
                      {"param_d", rnd.pick<std::int64_t>({-1, 1, 8, 16})},
                      {"unread", static_cast<std::int64_t>(rnd.below(3))}};
    const tw::CandidateSet set(m, pool);
    for (int i = 0; i < 20; ++i) {
      const tw::Problem p = rnd.problem();
      const tw::ExecutionContext ctx =
          context(rnd.pick<std::size_t>({0, 8, 32, 64}),
                  rnd.pick<tw::Schedule>(
                      {tw::Schedule::Default, tw::Schedule::Dynamic, tw::Schedule::Auto}));
      for (std::size_t min_scored :
           {std::size_t{0}, std::size_t{3}, std::size_t{7}, std::size_t{20}, kAll}) {
        const std::vector<tw::Result> a = set.rank(p, test_hardware(), ctx, min_scored);
        CHECK(same_results(
            a,
            tw::rank_configs(
                *m, p, test_hardware(), ctx, pool, min_scored, tw::TieBreak::PoolOrder)));
        CHECK(same_results(set.rank_top(p, test_hardware(), ctx, min_scored),
                           scored_prefix(a, min_scored)));
      }
    }
  }
}

namespace {

// MLREC_v3 serving flags of the non-temporal rule, the whitelist switch, the
// family-wise rule, the bank and the trained-key gate.
constexpr std::uint32_t kNonTemporal = 0x04;
constexpr std::uint32_t kNoWhitelist = 0x08;
constexpr std::uint32_t kFamilyRule  = 0x20;
constexpr std::uint32_t kBank        = 0x40;
constexpr std::uint32_t kGate        = 0x80;

// The non-temporal rule as lib/feasibility.py states it, refined family-wise
// over `pool` when `family` is set: per config, whether it passes.
std::vector<bool> reference_non_temporal(const tw::Problem& p,
                                         const std::vector<tw::Config>& pool,
                                         bool family) {
  const auto bits = [](tw::DataType d) -> std::uint64_t {
    switch (d) {
      case tw::DataType::Float:
      case tw::DataType::XFloat32: return 32;
      case tw::DataType::Float8:
      case tw::DataType::Float8_fnuz:
      case tw::DataType::BFloat8:
      case tw::DataType::BFloat8_fnuz: return 8;
      default: return 16;
    }
  };
  const std::uint64_t M = p.size.m, N = p.size.n, K = p.size.k;
  const std::uint64_t a = bits(p.a_dtype), b = bits(p.b_dtype);
  const bool a_trans = p.a_transpose == tw::Transpose::T;
  const bool b_trans = p.b_transpose == tw::Transpose::T;
  bool nt_a = false, nt_b = false;
  for (const tw::Config& c : pool) {
    nt_a = nt_a || c.cache_hints_a == 4;
    nt_b = nt_b || c.cache_hints_b == 4;
  }
  std::vector<bool> pass;
  for (const tw::Config& c : pool) {
    const bool aligned = K * a % 1024 == 0 && c.mt.k * a % 1024 == 0;
    const bool to_b = M <= 2 * c.mt.m && !b_trans && N * b / std::max<std::uint64_t>(M * a, 1) > 5;
    const bool to_a =
        !to_b && N <= 2 * c.mt.n && a_trans && M * a / std::max<std::uint64_t>(N * b, 1) > 5;
    const bool hinted = c.cache_hints_a != 0 || c.cache_hints_b != 0;
    pass.push_back(aligned ? !((to_b && nt_b && c.cache_hints_b != 4) ||
                               (to_a && nt_a && c.cache_hints_a != 4) || (!to_b && !to_a && hinted))
                           : !hinted);
  }
  const bool region =
      p.batch > 1 && K <= 512 && ((M <= 32 && N > 32) || (M > 32 && M <= 128 && N <= 32));
  if (!family || !region) return pass;
  const auto same_family = [](const tw::Config& x, const tw::Config& y) {
    return x.mt.m == y.mt.m && x.mt.n == y.mt.n && x.mt.k == y.mt.k && x.mi.m == y.mi.m &&
           x.mi.n == y.mi.n && x.mi.k == y.mi.k;
  };
  std::vector<bool> out = pass;
  for (std::size_t i = 0; i < pool.size(); ++i) {
    bool any = false, plain = false;
    std::pair<int, int> lowest{std::numeric_limits<int>::max(), 0};
    for (std::size_t j = 0; j < pool.size(); ++j) {
      if (!same_family(pool[i], pool[j])) continue;
      any    = any || pass[j];
      plain  = plain || (pool[j].cache_hints_a == 0 && pool[j].cache_hints_b == 0);
      lowest = std::min(lowest, std::make_pair(pool[j].cache_hints_a, pool[j].cache_hints_b));
    }
    const std::pair<int, int> kept = plain ? std::make_pair(0, 0) : lowest;
    if (!any && std::make_pair(pool[i].cache_hints_a, pool[i].cache_hints_b) == kept) out[i] = true;
  }
  return out;
}

bool lds_fits(const tw::Problem& p, const tw::Config& c) {
  const tw::Hardware hw = test_hardware();
  return tw::detail::check_lds_capacity(hw, c.mt, p.a_dtype, p.b_dtype);
}

// The configs a v3 model with `flags` (no cascade) scores for `p`, ascending,
// after checking that every entry point agrees.
std::vector<std::size_t> scored_under(std::uint32_t flags,
                                      const tw::Problem& p,
                                      const std::vector<tw::Config>& pool,
                                      std::vector<std::uint64_t> trained = {}) {
  ModelSpec spec        = v3_grid({});
  spec.v3_serving_flags = flags;
  spec.trained_keys     = std::move(trained);
  const tw::ModelPtr m  = load_spec(spec);
  const tw::Hardware hw = test_hardware();
  const tw::ExecutionContext ctx;
  const std::vector<tw::Result> r = tw::rank_configs(*m, p, hw, ctx, pool, kAll);
  const tw::CandidateSet lazy(m, pool), ready(m, pool);
  ready.prepare();
  CHECK(same_results(lazy.rank(p, hw, ctx, kAll), r));
  CHECK(same_results(ready.rank(p, hw, ctx, kAll), r));
  CHECK(same_results(ready.rank_top(p, hw, ctx, 3), scored_prefix(r, 3)));
  std::vector<std::size_t> out = scored_indices(r);
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace

TEST_CASE("v3 models describe the family-wise rule, the bank and the trained-key gate",
          "[v3][loader]") {
  ModelSpec spec            = v3_one_cell("Large|Large|LargeK|Bnone", sample_attributes());
  const tw::ModelInfo plain = tw::describe(*load_spec(spec));
  CHECK_FALSE(plain.non_temporal_family_rule);
  CHECK_FALSE(plain.bank);
  CHECK_FALSE(plain.trained_key_gate);
  CHECK(plain.trained_keys == 0);

  spec.v3_serving_flags   = kNonTemporal | kFamilyRule | kBank | kGate;
  spec.trained_keys       = {3, 9, 0xFFFFFFFFFFFFFFFFull};
  spec.provenance         = {{"config_id", "cfg"}};
  const tw::ModelPtr m    = load_spec(spec);
  const tw::ModelInfo all = tw::describe(*m);
  CHECK(all.non_temporal_rule);
  CHECK(all.non_temporal_family_rule);
  CHECK(all.bank);
  CHECK(all.trained_key_gate);
  CHECK(all.trained_keys == 3);
  CHECK(all.provenance.config_id == "cfg");
  CHECK(m->trained_keys == std::vector<std::uint64_t>{3, 9, 0xFFFFFFFFFFFFFFFFull});

  Image image = write_model(spec);
  put_u32(image, image.at.at("n_trained_keys"), 1000);
  CHECK(mentions(load_error(image), "exceeds the header"));
  // Without the gate's flag the section is not read, so its bytes are an
  // error where the header ends.
  image = write_model(spec);
  put_u32(image, image.at.at("v3_serving_flags"), kNonTemporal);
  CHECK_FALSE(load_error(image).empty());
}

TEST_CASE("the family-wise rule keeps one member of a family the rule rejects whole",
          "[v3][rules][family]") {
  // Families: H has only hinted members, Q a hint-free member and no NTB4
  // member, R an NTB4 member; P gives the pool both NT4 variants.
  std::vector<tw::Config> pool = {make_config(8, 512, 32, 4, 4, 4, 3, 1),  // 0 H
                                  make_config(8, 512, 32, 4, 4, 4, 1, 1),  // 1 H, its lowest hints
                                  make_config(16, 128, 32),                // 2 P
                                  make_config(16, 128, 32, 16, 16, 32, 4, 0),   // 3 P
                                  make_config(16, 128, 32, 16, 16, 32, 0, 4),   // 4 P
                                  make_config(16, 128, 64),                     // 5 Q
                                  make_config(16, 128, 64, 16, 16, 32, 4, 0),   // 6 Q
                                  make_config(16, 128, 64, 16, 16, 16),         // 7 R
                                  make_config(16, 128, 64, 16, 16, 16, 0, 4)};  // 8 R
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  const std::uint32_t today = kNonTemporal | kNoWhitelist, family = today | kFamilyRule;
  const auto check = [&](const tw::Problem& p,
                         std::vector<std::size_t> today_scored,
                         std::vector<std::size_t> family_scored) {
    CHECK(scored_under(today, p, pool) == today_scored);
    CHECK(scored_under(family, p, pool) == family_scored);
  };
  // K not 128-byte aligned: every hinted kernel is rejected, so H keeps its
  // lowest-hint member inside the region (M <= 32 < N, batched, K <= 512).
  check(make_problem(2, 1792, 4, 4096), {2, 5, 7}, {1, 2, 5, 7});
  check(make_problem(1280, 53, 48, 4096), {2, 5, 7}, {2, 5, 7});
  check(make_problem(2, 1792, 4, 1), {2, 5, 7}, {2, 5, 7});
  check(make_problem(64, 20, 4, 8), {2, 5, 7}, {1, 2, 5, 7});
  // Aligned K and a skinny B: the rule requires NTB4 members. Q has none and
  // keeps its hint-free member; R keeps its NTB4 member; the 32-deep tiles
  // stay unaligned.
  check(make_problem(16, 1792, 64, 8), {2, 8}, {1, 2, 5, 8});
  check(make_problem(16, 1792, 1024, 8), {2, 8}, {2, 8});
  // Without a family of hinted members only, an unaligned K keeps nothing.
  pool.erase(pool.begin(), pool.begin() + 2);
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  check(make_problem(2, 1792, 4, 4096), {0, 3, 5}, {0, 3, 5});
}

TEST_CASE("the family-wise rule matches a reference on random pools and problems",
          "[v3][rules][family]") {
  Random rnd(53);
  const tw::Hardware hw = test_hardware();
  for (int round = 0; round < 12; ++round) {
    std::vector<tw::Config> pool;
    for (int f = 0; f < 6 + static_cast<int>(rnd.below(6)); ++f) {
      const tw::Config base = make_config(rnd.pick<std::size_t>({4, 16, 64, 128}),
                                          rnd.pick<std::size_t>({16, 64, 256, 512}),
                                          rnd.pick<std::size_t>({32, 64, 128}),
                                          16,
                                          16,
                                          rnd.pick<std::size_t>({16, 32}));
      for (int v = 0; v < 1 + static_cast<int>(rnd.below(4)); ++v) {
        tw::Config c    = base;
        c.cache_hints_a = rnd.pick<int>({0, 0, 1, 3, 4});
        c.cache_hints_b = rnd.pick<int>({0, 0, 1, 4, 7});
        c.index         = pool.size();
        pool.push_back(c);
      }
    }
    for (const bool family : {false, true}) {
      const std::uint32_t flags = kNonTemporal | kNoWhitelist | (family ? kFamilyRule : 0);
      ModelSpec spec            = v3_grid({});
      spec.v3_serving_flags     = flags;
      const tw::ModelPtr m      = load_spec(spec);
      const tw::CandidateSet set(m, pool);
      if (round % 2 == 0) set.prepare();
      for (int i = 0; i < 60; ++i) {
        tw::Problem p = rnd.problem();
        if (i % 2 == 0) {
          p.size  = {1 + rnd.below(160), 1 + rnd.below(3000), 1 + rnd.below(600)};
          p.batch = rnd.pick<std::size_t>({1, 2, 64});
          if (rnd.below(2) != 0) std::swap(p.size.m, p.size.n);
          if (rnd.below(2) != 0) p.size.k = 64 * (1 + rnd.below(9));
        }
        const std::vector<bool> nt = reference_non_temporal(p, pool, family);
        std::vector<std::size_t> want;
        for (std::size_t c = 0; c < pool.size(); ++c)
          if (nt[c] && lds_fits(p, pool[c])) want.push_back(c);
        const std::vector<tw::Result> r = set.rank(p, hw, tw::ExecutionContext{}, kAll);
        std::vector<std::size_t> got    = scored_indices(r);
        std::sort(got.begin(), got.end());
        CHECK(got == want);
        CHECK(same_results(r, tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, kAll)));
      }
    }
  }
}

TEST_CASE("the trained-key gate scores trained keys while one is feasible", "[v3][gate]") {
  const std::vector<AttrSpec> attr = sample_attributes();
  const tw::Hardware hw            = test_hardware();
  std::vector<tw::Config> pool;
  pool.push_back(with_attributes(make_config(128, 256, 64), {{"param_a", 0}}));   // 0 trained
  pool.push_back(with_attributes(make_config(128, 256, 64), {{"param_a", 16}}));  // 1
  pool.push_back(make_config(256, 256, 64));                                      // 2 trained
  pool.push_back(make_config(64, 64, 64));                                        // 3
  pool.push_back(make_config(64, 64, 64));                                        // 4
  pool.push_back(make_config(256, 256, 64));                                      // 5 trained
  for (std::size_t i = 0; i < pool.size(); ++i) pool[i].index = i;
  std::vector<std::uint64_t> trained = {key_hash(pool[0], attr), key_hash(pool[2], attr)};
  std::sort(trained.begin(), trained.end());

  ModelSpec open        = v3_grid(attr);
  open.v3_serving_flags = kNoWhitelist;
  ModelSpec gated       = open;
  gated.v3_serving_flags |= kGate;
  gated.trained_keys    = trained;
  const tw::ModelPtr om = load_spec(open), gm = load_spec(gated);
  const tw::CandidateSet os(om, pool), gs(gm, pool), ready(gm, pool);
  ready.prepare();
  CHECK(gs.untrained_keys() == 2);
  CHECK(gs.untrained_configs() == 3);
  CHECK(os.untrained_keys() == 0);
  CHECK(os.untrained_configs() == 0);

  // A trained key fits: only trained keys are scored, as without the gate.
  const tw::Problem bf16           = make_problem(1000, 2000, 512);
  const std::vector<tw::Result> og = os.rank(bf16, hw, kAll);
  const std::vector<tw::Result> gr = gs.rank(bf16, hw, kAll);
  std::vector<std::size_t> scored  = scored_indices(gr);
  std::sort(scored.begin(), scored.end());
  CHECK(scored == std::vector<std::size_t>{0, 2, 5});
  for (const tw::Result& r : gr) {
    if (!r.scored) continue;
    const auto same = std::find_if(og.begin(), og.end(), [&](const tw::Result& o) {
      return o.config_index == r.config_index;
    });
    REQUIRE(same != og.end());
    CHECK(std::memcmp(&same->score, &r.score, sizeof(double)) == 0);
  }
  CHECK(same_results(ready.rank(bf16, hw, kAll), gr));
  CHECK(same_results(tw::rank_configs(*gm, bf16, hw, pool, kAll), gr));

  // In fp32 no trained tile fits the LDS but the untrained 64x64x64 does, so
  // every feasible key is scored.
  const tw::Problem fp32                  = make_problem(1000, 2000, 512, 1, tw::DataType::Float);
  const std::vector<std::size_t> fallback = scored_indices(gs.rank(fp32, hw, kAll));
  CHECK(fallback == scored_indices(os.rank(fp32, hw, kAll)));
  CHECK_FALSE(fallback.empty());
  CHECK(same_results(ready.rank(fp32, hw, kAll), gs.rank(fp32, hw, kAll)));
  CHECK(same_results(tw::rank_configs(*gm, fp32, hw, pool, kAll), gs.rank(fp32, hw, kAll)));

  // A gate that lists every key ranks like no gate.
  ModelSpec every = gated;
  every.trained_keys.clear();
  for (const tw::Config& c : pool) every.trained_keys.push_back(key_hash(c, attr));
  std::sort(every.trained_keys.begin(), every.trained_keys.end());
  every.trained_keys.erase(std::unique(every.trained_keys.begin(), every.trained_keys.end()),
                           every.trained_keys.end());
  const tw::CandidateSet es(load_spec(every), pool);
  CHECK(es.untrained_keys() == 0);
  for (const tw::Problem& p : {bf16, fp32})
    for (std::size_t min_scored : {std::size_t{0}, std::size_t{1}, kAll})
      CHECK(same_results(es.rank(p, hw, min_scored), os.rank(p, hw, min_scored)));
}

TEST_CASE("the trained-key gate holds through whitelists, tier 2, the cascade and launch keys",
          "[v3][gate]") {
  Random rnd(59);
  const std::vector<AttrSpec> attr = sample_attributes();
  const tw::Hardware hw            = test_hardware();
  for (int round = 0; round < 8; ++round) {
    std::vector<tw::Config> pool = rnd.pool(6 + rnd.below(10), 20 + rnd.below(30));
    for (tw::Config& c : pool) {
      c.attributes = {{"param_a", static_cast<std::int64_t>(rnd.below(2)) * 16}};
      c.occupancy  = 1 + static_cast<int>(rnd.below(3));
    }
    std::vector<std::uint64_t> all, trained;
    for (const tw::Config& c : pool) all.push_back(key_hash(c, attr));
    std::sort(all.begin(), all.end());
    all.erase(std::unique(all.begin(), all.end()), all.end());
    for (std::uint64_t h : all)
      if (rnd.below(3) != 0) trained.push_back(h);
    if (trained.empty()) trained.push_back(all.front());
    ModelSpec spec        = v3_grid(attr);
    spec.v3_cascade_keys  = rnd.pick<std::uint32_t>({0, 2, 5});
    spec.v3_serving_flags = kGate | (rnd.below(2) != 0 ? kNoWhitelist : 0) |
                            (rnd.below(2) != 0 ? kNonTemporal : 0) |
                            (rnd.below(2) != 0 ? kLaunchKeys : 0);
    spec.trained_keys = trained;
    for (CellSpec& cell : spec.cells)
      if (cell.seed % 2 == 0) cell.view_key_hashes = {all[rnd.below(all.size())]};
    const tw::ModelPtr m = load_spec(spec);
    const tw::PoolOptions options{tw::TieBreak::PoolOrder, std::make_shared<OccupancyLaunch>()};
    const tw::CandidateSet set(m, pool, options);
    if (round % 2 == 0) set.prepare();
    for (int i = 0; i < 30; ++i) {
      const tw::Problem p = rnd.problem();
      for (std::size_t min_scored : {std::size_t{0}, std::size_t{1}, std::size_t{9}, kAll}) {
        const std::vector<tw::Result> r = set.rank(p, hw, tw::ExecutionContext{}, min_scored);
        CHECK(same_results(
            r, tw::rank_configs(*m, p, hw, tw::ExecutionContext{}, pool, options, min_scored)));
        CHECK(same_results(set.rank_top(p, hw, tw::ExecutionContext{}, min_scored),
                           scored_prefix(r, min_scored)));
        const std::vector<std::size_t> scored = scored_indices(r);
        const bool any_trained = std::any_of(scored.begin(), scored.end(), [&](std::size_t c) {
          return std::binary_search(trained.begin(), trained.end(), key_hash(pool[c], attr));
        });
        for (std::size_t c : scored)
          if (any_trained)
            CHECK(std::binary_search(trained.begin(), trained.end(), key_hash(pool[c], attr)));
      }
    }
  }
}

TEST_CASE("the bank flag never changes a ranking", "[v3][rules]") {
  Random rnd(61);
  ModelSpec spec        = v3_grid({});
  spec.v3_serving_flags = kNonTemporal;
  spec.v3_cascade_keys  = 3;
  ModelSpec bank        = spec;
  bank.v3_serving_flags |= kBank;
  const std::vector<tw::Config> pool = rnd.pool(10, 40);
  const tw::CandidateSet a(load_spec(spec), pool), b(load_spec(bank), pool);
  for (int i = 0; i < 40; ++i) {
    const tw::Problem p = rnd.problem();
    CHECK(same_results(a.rank(p, test_hardware(), 1), b.rank(p, test_hardware(), 1)));
  }
}

TEST_CASE("the family-wise rule ranks alike through the cascade, launch keys and the cache",
          "[v3][rules][family][pool]") {
  Random rnd(67);
  for (const bool keyed : {false, true}) {
    ModelSpec spec        = v3_grid({});
    spec.v3_serving_flags = kNonTemporal | kFamilyRule | (keyed ? kLaunchKeys : 0u);
    spec.v3_cascade_keys  = 3;
    const tw::ModelPtr m  = load_spec(spec);
    const tw::PoolOptions options{tw::TieBreak::PoolOrder, std::make_shared<OccupancyLaunch>()};
    for (int round = 0; round < 6; ++round) {
      std::vector<tw::Config> pool;
      for (int f = 0; f < 10 + static_cast<int>(rnd.below(10)); ++f) {
        const tw::Config base  = make_config(rnd.pick<std::size_t>({4, 8, 16, 32}),
                                            rnd.pick<std::size_t>({64, 128, 256}),
                                            rnd.pick<std::size_t>({32, 64}),
                                            rnd.pick<std::size_t>({4, 16}),
                                            rnd.pick<std::size_t>({4, 16}),
                                            rnd.pick<std::size_t>({4, 32}));
        const bool hinted_only = rnd.below(2) != 0;
        for (int v = 0; v < 1 + static_cast<int>(rnd.below(3)); ++v) {
          tw::Config c    = base;
          c.cache_hints_a = hinted_only ? rnd.pick<int>({1, 3}) : rnd.pick<int>({0, 4});
          c.cache_hints_b = hinted_only ? rnd.pick<int>({1, 7}) : rnd.pick<int>({0, 4});
          c.occupancy     = 1 + static_cast<int>(rnd.below(3));
          c.index         = pool.size();
          pool.push_back(c);
        }
      }
      const tw::CandidateSet set(m, pool, options);
      if (round % 2 == 0) set.prepare();
      for (int i = 0; i < 30; ++i) {
        tw::Problem p = make_problem(1 + rnd.below(32),
                                     33 + rnd.below(3000),
                                     1 + rnd.below(512),
                                     rnd.pick<std::size_t>({2, 8, 64}));
        if (i % 3 == 1) std::swap(p.size.m, p.size.n);
        if (i % 3 == 2) p.size.k = 64 * (1 + rnd.below(8));
        for (std::size_t min_scored : {std::size_t{1}, std::size_t{5}, kAll}) {
          const std::vector<tw::Result> a =
              set.rank(p, test_hardware(), tw::ExecutionContext{}, min_scored);
          CHECK(same_results(
              a,
              tw::rank_configs(
                  *m, p, test_hardware(), tw::ExecutionContext{}, pool, options, min_scored)));
          CHECK(same_results(set.rank_top(p, test_hardware(), tw::ExecutionContext{}, min_scored),
                             scored_prefix(a, min_scored)));
        }
      }
    }
  }
}
