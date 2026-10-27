// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "model_impl.hpp"
#include "model_writer.hpp"
#include "test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>

namespace tw = tilewright;
namespace fs = std::filesystem;
using namespace tilewright_test;

namespace {

ModelSpec small_spec(tw::WeightType wt = tw::WeightType::Fp32) {
  ModelSpec spec;
  spec.weight_type       = wt;
  spec.mi_table          = {{16, 16, 32, static_cast<int>(tw::DataType::BFloat16), 16.0},
                            {32, 32, 16, static_cast<int>(tw::DataType::BFloat16), 32.0}};
  const std::string base = "Large|Large|LargeK|Bnone";
  spec.splits            = {{base, 'M', 4096, base + "#M<=4096", base + "#M>4096"}};
  CellSpec a;
  a.label      = base;
  a.seed       = 1;
  a.signatures = {{256, 256, 64, 16, 16, 32, 0, 0}};
  CellSpec b;
  b.label     = base + "#M<=4096";
  b.seed      = 2;
  b.embed_dim = 5;
  CellSpec c;
  c.label        = base + "#M>4096";
  c.seed         = 3;
  c.inter_hidden = 3;
  spec.cells     = {a, b, c};
  return spec;
}

// Error message of loading `bytes`, or "" when the model loads.
std::string load_error(const std::vector<unsigned char>& bytes, std::size_t size) {
  std::string err;
  tw::ModelPtr m = tw::load_model_from_memory(bytes.data(), size, &err);
  if (m) return std::string();
  return err.empty() ? std::string("<no message>") : err;
}

std::string load_error(const Image& image) { return load_error(image.bytes, image.bytes.size()); }

bool mentions(const std::string& text, const std::string& what) {
  return text.find(what) != std::string::npos;
}

// The image cut at `offset` inside the payload, re-closed with a consistent
// header and trailer so the payload parser itself sees the short record.
Image cut_payload(const Image& image, std::size_t offset) {
  Image cut = image;
  cut.bytes.resize(offset);
  cut.bytes.insert(cut.bytes.end(), {'M', 'L', 'R', 'E', 'C', 'E', 'N', 'D'});
  put_u64(cut, cut.at.at("payload_size"), offset - image.header_size);
  refresh_crc(cut);
  return cut;
}

}  // namespace

TEST_CASE("crc32 is CRC-32/ISO-HDLC", "[loader]") {
  const char* check = "123456789";
  CHECK(tw::detail::crc32(reinterpret_cast<const unsigned char*>(check), 9) == 0xCBF43926u);
  CHECK(tw::detail::crc32(nullptr, 0) == 0u);
  std::vector<unsigned char> data(300);
  for (std::size_t i = 0; i < data.size(); ++i) data[i] = static_cast<unsigned char>(i * 151 + 7);
  for (std::size_t n = 0; n <= data.size(); ++n) {
    std::uint32_t crc = 0xFFFFFFFFu;
    for (std::size_t i = 0; i < n; ++i) {
      crc ^= data[i];
      for (int b = 0; b < 8; ++b) crc = (crc & 1u) ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
    }
    REQUIRE(tw::detail::crc32(data.data(), n) == (crc ^ 0xFFFFFFFFu));
  }
}

TEST_CASE("feature_catalog_hash is the trained catalog", "[loader]") {
  CHECK(std::string(tw::feature_catalog_hash()) == "e7fe4b524851e895");
}

TEST_CASE("well-formed models load for every weight type", "[loader]") {
  for (tw::WeightType wt :
       {tw::WeightType::Fp32, tw::WeightType::Bf16, tw::WeightType::Int8, tw::WeightType::Int4}) {
    const Image image = write_model(small_spec(wt));
    std::string err   = "stale";
    tw::ModelPtr m    = tw::load_model_from_memory(image.bytes.data(), image.bytes.size(), &err);
    REQUIRE(m);
    CHECK(err.empty());
    const tw::ModelInfo info = tw::describe(*m);
    CHECK(info.arch == "gfxtest");
    CHECK(info.feature_catalog_hash == tw::feature_catalog_hash());
    CHECK(info.weight_type == wt);
    CHECK(info.n_cells == 3);
    CHECK(info.n_splits == 1);
    CHECK(tw::cell_label(*m, 1) == "Large|Large|LargeK|Bnone#M<=4096");
    CHECK(tw::cell_label(*m, 3).empty());
    CHECK(tw::cell_label(*m, -1).empty());
  }
}

TEST_CASE("an image truncated at any byte is rejected", "[loader]") {
  const Image image = write_model(small_spec(tw::WeightType::Int4));
  for (std::size_t n = 0; n < image.bytes.size(); ++n) {
    INFO("size " << n);
    REQUIRE_FALSE(load_error(image.bytes, n).empty());
  }
  CHECK(load_error(image).empty());
}

TEST_CASE("a payload cut at any field is rejected", "[loader]") {
  const Image image         = write_model(small_spec(tw::WeightType::Int8));
  const std::size_t trailer = image.at.at("trailer");
  for (const auto& field : image.at) {
    if (field.second <= image.header_size || field.second >= trailer) continue;
    for (std::size_t extra : {0, 1}) {
      const std::size_t offset = field.second + extra;
      INFO(field.first << " +" << extra);
      CHECK_FALSE(load_error(cut_payload(image, offset)).empty());
    }
  }
}

TEST_CASE("structural header defects are rejected", "[loader]") {
  const Image good = write_model(small_spec());

  SECTION("MLREC_v1") {
    Image v1 = good;
    std::memcpy(v1.bytes.data(), "MLREC_v1", 8);
    const std::string err = load_error(v1);
    CHECK(mentions(err, "MLREC_v1"));
    CHECK(mentions(err, "convert"));
  }
  SECTION("Git LFS pointer") {
    const std::string ptr =
        "version https://git-lfs.github.com/spec/v1\noid sha256:0000\nsize 12345\n";
    const std::vector<unsigned char> bytes(ptr.begin(), ptr.end());
    CHECK(mentions(load_error(bytes, bytes.size()), "LFS"));
  }
  SECTION("bad magic") {
    Image bad = good;
    bad.bytes[3] ^= 0x20;
    CHECK(mentions(load_error(bad), "magic"));
  }
  SECTION("empty") { CHECK_FALSE(load_error(std::vector<unsigned char>(), 0).empty()); }
  SECTION("endian marker") {
    Image bad = good;
    put_u32(bad, bad.at.at("endian"), 0x04030201u);
    CHECK(mentions(load_error(bad), "endian"));
  }
  SECTION("header_size") {
    Image bad = good;
    put_u32(bad, bad.at.at("header_size"), static_cast<std::uint32_t>(good.header_size + 4));
    CHECK_FALSE(load_error(bad).empty());
    put_u32(bad, bad.at.at("header_size"), 8);
    CHECK_FALSE(load_error(bad).empty());
  }
  SECTION("payload_size") {
    Image bad = good;
    put_u64(bad, bad.at.at("payload_size"), std::numeric_limits<std::uint64_t>::max());
    CHECK(mentions(load_error(bad), "file size"));
  }
  SECTION("trailer") {
    Image bad        = good;
    bad.bytes.back() = 'X';
    CHECK(mentions(load_error(bad), "MLRECEND"));
    Image missing = good;
    missing.bytes.resize(missing.bytes.size() - 8);
    CHECK(mentions(load_error(missing), "file size"));
    Image extra = good;
    extra.bytes.push_back(0);
    CHECK(mentions(load_error(extra), "file size"));
  }
  SECTION("weight dtype and serving flags") {
    Image bad                            = good;
    bad.bytes[bad.at.at("weight_dtype")] = 4;
    CHECK(mentions(load_error(bad), "weight_dtype"));
    Image res                             = good;
    res.bytes[res.at.at("serving_flags")] = 2;
    CHECK(mentions(load_error(res), "serving flags"));
  }
  SECTION("feature dims") {
    for (const char* dim : {"q_dim", "i_dim", "x_dim"}) {
      Image bad = good;
      put_u32(bad, bad.at.at(dim), 54);
      CHECK(mentions(load_error(bad), "dims"));
    }
  }
  SECTION("feature hash") {
    ModelSpec spec    = small_spec();
    spec.feature_hash = "0123456789abcdef";
    CHECK(mentions(load_error(write_model(spec)), "hash"));
    spec.feature_hash = "";
    CHECK(mentions(load_error(write_model(spec)), "hash"));
  }
  SECTION("arch") {
    ModelSpec spec = small_spec();
    spec.arch      = "";
    CHECK(mentions(load_error(write_model(spec)), "arch"));
    spec.arch = "gfx 950";
    CHECK(mentions(load_error(write_model(spec)), "arch"));
  }
  SECTION("no cells") {
    ModelSpec spec = small_spec();
    spec.cells.clear();
    spec.splits.clear();
    CHECK(mentions(load_error(write_model(spec)), "no cells"));
  }
}

TEST_CASE("arch constants must be finite and positive", "[loader]") {
  const Image good = write_model(small_spec());
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (const char* field :
       {"parallel_mi_cu", "bw_c0", "bw_c1", "bw_c2", "mi_default", "mi0.cycles"})
    for (double v : {nan, inf, -inf}) {
      Image bad = good;
      put_f64(bad, bad.at.at(field), v);
      INFO(field << " = " << v);
      CHECK(mentions(load_error(bad), "non-finite"));
    }
  for (const char* field : {"parallel_mi_cu", "mi_default", "mi0.cycles"})
    for (double v : {0.0, -1.0}) {
      Image bad = good;
      put_f64(bad, bad.at.at(field), v);
      INFO(field << " = " << v);
      CHECK(mentions(load_error(bad), "positive"));
    }
}

TEST_CASE("the MI table is validated", "[loader]") {
  SECTION("count larger than the header") {
    Image bad = write_model(small_spec());
    put_u32(bad, bad.at.at("n_mi"), 0xFFFFFFFFu);
    CHECK(mentions(load_error(bad), "n_mi"));
  }
  SECTION("count above the table limit") {
    ModelSpec spec = small_spec();
    for (std::uint32_t i = 0; i < 1100; ++i)
      spec.mi_table.push_back({i + 100, 16, 32, static_cast<int>(tw::DataType::Half), 8.0});
    CHECK(mentions(load_error(write_model(spec)), "exceeds 1024"));
    spec.mi_table.resize(1024);
    CHECK(load_error(write_model(spec)).empty());
  }
  SECTION("count smaller than the header") {
    Image bad = write_model(small_spec());
    put_u32(bad, bad.at.at("n_mi"), 1);
    CHECK(mentions(load_error(bad), "header_size"));
  }
  SECTION("dtype") {
    for (int dt : {-1, 23, 99, static_cast<int>(tw::DataType::Float8_fnuz)}) {
      ModelSpec spec         = small_spec();
      spec.mi_table[0].dtype = dt;
      INFO("dtype " << dt);
      CHECK(mentions(load_error(write_model(spec)), "MI table dtype"));
    }
  }
  SECTION("duplicate entry") {
    ModelSpec spec = small_spec();
    spec.mi_table.push_back(spec.mi_table[0]);
    CHECK(mentions(load_error(write_model(spec)), "duplicate MI"));
  }
}

TEST_CASE("payload corruption is caught by the CRC", "[loader]") {
  const Image good = write_model(small_spec());
  Image bad        = good;
  bad.bytes[good.at.at("cell1.q_w2") + 5] ^= 0x01;
  CHECK(mentions(load_error(bad), "CRC"));
  put_u32(bad, bad.at.at("crc"), 0);
  CHECK(mentions(load_error(bad), "CRC"));
}

TEST_CASE("counts are bounded by the bytes that remain", "[loader]") {
  const Image good = write_model(small_spec());
  SECTION("n_cells") {
    Image bad = good;
    put_u32(bad, bad.at.at("n_cells"), 0xFFFFFFFFu);
    CHECK(mentions(load_error(bad), "n_cells"));
  }
  SECTION("n_splits") {
    Image bad = good;
    put_u32(bad, bad.at.at("n_splits"), 0xFFFFFFFFu);
    CHECK(mentions(load_error(bad), "n_splits"));
  }
  SECTION("signature count") {
    Image bad = good;
    put_u32(bad, bad.at.at("cell0.n_sk"), 0x7FFFFFFFu);
    refresh_crc(bad);
    CHECK(mentions(load_error(bad), "signature count"));
  }
  SECTION("label length") {
    Image bad = good;
    put_u16(bad, bad.at.at("cell2.label"), 0xFFFF);
    refresh_crc(bad);
    CHECK(mentions(load_error(bad), "truncated"));
  }
  SECTION("layer widths") {
    for (const char* dim : {"cell0.embed_dim", "cell0.hidden_dim", "cell0.inter_hidden"}) {
      for (std::uint32_t v : {0u, 0x10001u, 0xFFFFFFFFu}) {
        Image bad = good;
        put_u32(bad, bad.at.at(dim), v);
        refresh_crc(bad);
        INFO(dim << " = " << v);
        CHECK(mentions(load_error(bad), "layer width"));
      }
      Image wide = good;
      put_u32(wide, wide.at.at(dim), 0x10000u);
      refresh_crc(wide);
      CHECK(mentions(load_error(wide), "truncated"));
    }
  }
  SECTION("trailing payload bytes") {
    Image bad = good;
    bad.bytes.insert(bad.bytes.begin() + static_cast<long>(bad.at.at("trailer")), 4, 0);
    put_u64(bad, bad.at.at("payload_size"), bad.bytes.size() - 8 - bad.header_size);
    refresh_crc(bad);
    CHECK(mentions(load_error(bad), "trailing"));
  }
}

TEST_CASE("non-finite payload values are rejected", "[loader]") {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  SECTION("fp32 fields") {
    const Image good = write_model(small_spec(tw::WeightType::Fp32));
    const std::pair<const char*, std::size_t> fields[] = {{"cell0.temperature", 0},
                                                          {"cell1.q_mean", 2},
                                                          {"cell1.q_std", 54},
                                                          {"cell2.i_mean", 11},
                                                          {"cell0.x_std", 0},
                                                          {"cell0.q_w0", 3},
                                                          {"cell1.q_b2", 1},
                                                          {"cell2.i_w2", 7},
                                                          {"cell0.x_w2", 2},
                                                          {"cell2.x_b2", 0}};
    for (const auto& field : fields)
      for (float v : {nan, inf, -inf}) {
        Image bad = good;
        put_f32(bad, bad.at.at(field.first) + 4 * field.second, v);
        refresh_crc(bad);
        INFO(field.first << "[" << field.second << "] = " << v);
        CHECK(mentions(load_error(bad), "non-finite"));
      }
  }
  SECTION("bf16 weights") {
    const Image good = write_model(small_spec(tw::WeightType::Bf16));
    for (const std::uint16_t bits : std::initializer_list<std::uint16_t>{0x7F80, 0xFF80, 0x7FC1}) {
      Image bad = good;
      put_u16(bad, bad.at.at("cell1.i_w0") + 6, bits);
      refresh_crc(bad);
      CHECK(mentions(load_error(bad), "non-finite"));
    }
  }
  SECTION("quantization scales") {
    for (tw::WeightType wt : {tw::WeightType::Int8, tw::WeightType::Int4}) {
      const Image good = write_model(small_spec(wt));
      for (float v : {nan, inf, FLT_MAX, -FLT_MAX / 2}) {
        Image bad = good;
        put_f32(bad, bad.at.at("cell2.q_w4"), v);
        refresh_crc(bad);
        INFO("scale " << v);
        CHECK(mentions(load_error(bad), "quantization scale"));
      }
    }
  }
}

TEST_CASE("labels, axes and the split tree are validated", "[loader]") {
  const std::string base = "Large|Large|LargeK|Bnone";
  SECTION("duplicate cell label") {
    ModelSpec spec      = small_spec();
    spec.cells[2].label = spec.cells[0].label;
    CHECK(mentions(load_error(write_model(spec)), "duplicate cell label"));
  }
  SECTION("duplicate split parent") {
    ModelSpec spec = small_spec();
    spec.splits.push_back({base, 'N', 100, base + "#N<=100", base + "#N>100"});
    CHECK(mentions(load_error(write_model(spec)), "duplicate split parent"));
  }
  SECTION("cycle") {
    ModelSpec spec = small_spec();
    spec.splits.push_back({base + "#M<=4096", 'K', 10, base, base + "#M<=4096#K>10"});
    CHECK(mentions(load_error(write_model(spec)), "cycle"));
  }
  SECTION("self loop") {
    ModelSpec spec    = small_spec();
    spec.splits[0].hi = base;
    CHECK(mentions(load_error(write_model(spec)), "cycle"));
  }
  SECTION("axis") {
    for (char axis : {'X', 'm', '\0'}) {
      ModelSpec spec      = small_spec();
      spec.splits[0].axis = axis;
      CHECK(mentions(load_error(write_model(spec)), "axis"));
    }
    Image bad                               = write_model(small_spec());
    bad.bytes[bad.at.at("split0.axis") + 1] = 'N';
    refresh_crc(bad);
    CHECK(mentions(load_error(bad), "axis"));
  }
  SECTION("empty or non-printable labels") {
    ModelSpec spec      = small_spec();
    spec.cells[1].label = "";
    CHECK(mentions(load_error(write_model(spec)), "empty cell label"));
    spec.cells[1].label = "bad\nlabel";
    CHECK(mentions(load_error(write_model(spec)), "non-printable cell label"));
    ModelSpec split    = small_spec();
    split.splits[0].lo = "";
    CHECK(mentions(load_error(write_model(split)), "empty split lo label"));
  }
}

TEST_CASE("the v3 CU floor is validated", "[loader][v3]") {
  constexpr std::uint32_t kGate = 0x80, kFloor = 0x100;
  ModelSpec spec         = as_v3(small_spec());
  spec.v3_serving_flags  = kFloor;
  spec.min_effective_cus = 32;
  const Image good       = write_model(spec);
  REQUIRE(load_error(good).empty());
  const tw::ModelPtr m = tw::load_model_from_memory(good.bytes.data(), good.bytes.size());
  REQUIRE(m);
  CHECK(tw::describe(*m).min_effective_cus == 32);
  CHECK(tw::describe(*load_spec(as_v3(small_spec()))).min_effective_cus == 0);
  CHECK(tw::describe(*load_spec(small_spec())).min_effective_cus == 0);

  SECTION("zero") {
    ModelSpec zero         = spec;
    zero.min_effective_cus = 0;
    CHECK(mentions(load_error(write_model(zero)), "CU floor must be at least 1"));
  }
  SECTION("the flag without the field") {
    Image bad = good;
    bad.bytes.erase(bad.bytes.begin() + static_cast<long>(bad.at.at("min_effective_cus")),
                    bad.bytes.begin() + static_cast<long>(bad.at.at("min_effective_cus") + 4));
    put_u32(bad, bad.at.at("header_size"), static_cast<std::uint32_t>(good.header_size - 4));
    CHECK(mentions(load_error(bad), "truncated header: CU floor"));
  }
  SECTION("the field without the flag") {
    Image bad = good;
    put_u32(bad, bad.at.at("v3_serving_flags"), 0);
    CHECK_FALSE(load_error(bad).empty());
  }
  SECTION("an unknown flag beside it") {
    Image bad = good;
    put_u32(bad, bad.at.at("v3_serving_flags"), kFloor | 0x200);
    CHECK(mentions(load_error(bad), "unknown serving flags"));
  }
  SECTION("truncated at any byte of the header") {
    for (std::size_t n = 0; n <= good.header_size; ++n) {
      INFO("size " << n);
      REQUIRE_FALSE(load_error(good.bytes, n).empty());
    }
  }
  SECTION("after the trained keys and before the provenance") {
    ModelSpec all         = spec;
    all.v3_serving_flags  = kGate | kFloor;
    all.trained_keys      = {4, 8};
    all.min_effective_cus = 0xFFFFFFFFu;
    all.provenance        = {{"config_id", "cfg"}};
    const Image image     = write_model(all);
    REQUIRE(image.at.at("trained_key1") < image.at.at("min_effective_cus"));
    REQUIRE(image.at.at("min_effective_cus") < image.at.at("n_provenance"));
    const tw::ModelPtr loaded = tw::load_model_from_memory(image.bytes.data(), image.bytes.size());
    REQUIRE(loaded);
    const tw::ModelInfo info = tw::describe(*loaded);
    CHECK(info.min_effective_cus == 0xFFFFFFFFu);
    CHECK(info.trained_keys == 2);
    CHECK(info.provenance.config_id == "cfg");
    Image swapped = image;
    put_u32(swapped, swapped.at.at("v3_serving_flags"), kFloor);
    CHECK_FALSE(load_error(swapped).empty());
  }
}

TEST_CASE("load_model deduplicates by canonical path", "[loader][registry]") {
  TempDir dir("dedup");
  const Image image   = write_model(small_spec(tw::WeightType::Int4));
  const fs::path file = dir.path() / "model.tilewright.bin";
  write_file(file, image.bytes);
  fs::create_directories(dir.path() / "sub");
  std::error_code ec;
  fs::create_symlink(file, dir.path() / "link.bin", ec);

  std::string err;
  tw::ModelPtr a = tw::load_model(file.string(), &err);
  REQUIRE(a);
  CHECK(err.empty());
  CHECK(tw::load_model(file.string()) == a);
  CHECK(tw::load_model((dir.path() / "." / "model.tilewright.bin").string()) == a);
  CHECK(tw::load_model((dir.path() / "sub" / ".." / "model.tilewright.bin").string()) == a);
  if (!ec) CHECK(tw::load_model((dir.path() / "link.bin").string()) == a);

  tw::ModelPtr other = tw::load_model_from_memory(image.bytes.data(), image.bytes.size());
  REQUIRE(other);
  CHECK(other != a);
}

TEST_CASE("load_model reports file errors", "[loader][registry]") {
  TempDir dir("errors");
  std::string err;
  CHECK_FALSE(tw::load_model((dir.path() / "missing.bin").string(), &err));
  CHECK(mentions(err, "missing.bin"));
  CHECK_FALSE(tw::load_model(dir.path().string(), &err));
  CHECK(mentions(err, "regular file"));

  Image v1 = write_model(small_spec());
  std::memcpy(v1.bytes.data(), "MLREC_v1", 8);
  write_file(dir.path() / "v1.bin", v1.bytes);
  CHECK_FALSE(tw::load_model((dir.path() / "v1.bin").string(), &err));
  CHECK(mentions(err, "convert"));

  const std::string ptr = "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n";
  write_file(dir.path() / "pointer.bin", std::vector<unsigned char>(ptr.begin(), ptr.end()));
  CHECK_FALSE(tw::load_model((dir.path() / "pointer.bin").string(), &err));
  CHECK(mentions(err, "LFS"));

  Image big = write_model(small_spec());
  put_u64(big, big.at.at("payload_size"), 1ull << 40);
  write_file(dir.path() / "big.bin", big.bytes);
  CHECK_FALSE(tw::load_model((dir.path() / "big.bin").string(), &err));
  CHECK(mentions(err, "file size"));

  CHECK_FALSE(tw::load_model_from_memory(nullptr, 10, &err));
  CHECK_FALSE(err.empty());
}

TEST_CASE("load_model_by_index", "[loader][registry]") {
  TempDir dir("index");
  const Image image = write_model(small_spec(tw::WeightType::Int8));
  write_file(dir.path() / "a.tilewright.bin", image.bytes);
  Image broken = image;
  broken.bytes[broken.at.at("cell0.q_w0") + 9] ^= 0x10;
  write_file(dir.path() / "broken.tilewright.bin", broken.bytes);
  write_text(dir.path() / "tilewright_index",
             "# comment line\r\n"
             "\r\n"
             "StemA\ta.tilewright.bin   # trailing comment\r\n"
             "  StemB   broken.tilewright.bin\n"
             "StemC missing.tilewright.bin\n"
             "lonely\n"
             "StemA broken.tilewright.bin\n");
  const std::string d = dir.path().string();

  std::string err = "stale";
  tw::ModelPtr a  = tw::load_model_by_index("StemA", d, &err);
  REQUIRE(a);
  CHECK(err.empty());
  CHECK(a == tw::load_model((dir.path() / "a.tilewright.bin").string()));

  err = "stale";
  CHECK_FALSE(tw::load_model_by_index("StemX", d, &err));
  CHECK(err.empty());
  CHECK_FALSE(tw::load_model_by_index("lonely", d, &err));
  CHECK(err.empty());
  CHECK_FALSE(tw::load_model_by_index("StemA", (dir.path() / "nowhere").string(), &err));
  CHECK(err.empty());
  CHECK_FALSE(tw::load_model_by_index("StemA", "", &err));
  CHECK(err.empty());

  CHECK_FALSE(tw::load_model_by_index("StemB", d, &err));
  CHECK(mentions(err, "CRC"));
  CHECK_FALSE(tw::load_model_by_index("StemC", d, &err));
  CHECK(mentions(err, "missing.tilewright.bin"));
  for (std::uint32_t chip : {0x1234u, 0xFFFFu}) {
    err = "stale";
    CHECK(tw::load_model_by_index("StemA", d, chip, &err) == a);
    CHECK(err.empty());
  }

  // An index that cannot be read to its end is no index to choose a row from.
  TempDir odd("index-dir");
  fs::create_directories(odd.path() / "tilewright_index");
  err = "stale";
  CHECK_FALSE(tw::load_model_by_index("StemA", odd.path().string(), 0x1234, &err));
#if defined(__GLIBCXX__) && !defined(_WIN32)
  // libstdc++ opens a directory and reports the failed read as badbit.
  CHECK(mentions(err, "cannot read index"));
#else
  if (!err.empty()) CHECK(mentions(err, "cannot"));
#endif
}

TEST_CASE("load_model_by_index prefers the row of the device's PCI chip ID", "[loader][registry]") {
  TempDir dir("index-chip");
  ModelSpec b = small_spec(tw::WeightType::Int8);
  ModelSpec c = small_spec(tw::WeightType::Int8);
  b.arch      = "gfxtestb";
  c.arch      = "gfxtestc";
  write_file(dir.path() / "any.bin", write_model(small_spec(tw::WeightType::Int8)).bytes);
  write_file(dir.path() / "b.bin", write_model(b).bytes);
  write_file(dir.path() / "c.bin", write_model(c).bytes);
  write_text(dir.path() / "tilewright_index",
             "# <stem> <model file> [chip=0x<PCI chip ID>]\n"
             "Stem\tb.bin\tchip=0x1234\n"
             "Stem\tany.bin\n"
             "Stem c.bin chip=0xABCD   # upper-case digits\r\n"
             "Stem\tc.bin\tchip=0x1234\n"
             "Stem\tb.bin\n"
             "OnlyChip\tb.bin\tchip=0x00ab\n"
             "Other\tany.bin\tchip=0xzz\n");
  const std::string d = dir.path().string();
  const auto arch     = [&](const std::string& stem, std::uint32_t chip) {
    std::string err      = "stale";
    const tw::ModelPtr m = tw::load_model_by_index(stem, d, chip, &err);
    INFO(stem << " chip " << chip << ": " << err);
    CHECK(err.empty());
    return m ? tw::describe(*m).arch : std::string("<none>");
  };
  // The first row naming the chip ID wins wherever it stands, else the first
  // row without one; another stem's malformed row does not matter.
  CHECK(arch("Stem", 0) == "gfxtest");
  CHECK(arch("Stem", 0x1234) == "gfxtestb");
  CHECK(arch("Stem", 0xABCD) == "gfxtestc");
  CHECK(arch("Stem", 0x9999) == "gfxtest");
  CHECK(arch("Stem", 0x11234) == "gfxtest");
  CHECK(arch("OnlyChip", 0xAB) == "gfxtestb");
  CHECK(arch("OnlyChip", 0) == "<none>");
  CHECK(arch("OnlyChip", 0x1234) == "<none>");

  std::string err = "stale";
  CHECK(tw::load_model_by_index("Stem", d, &err) ==
        tw::load_model((dir.path() / "any.bin").string()));
  CHECK(tw::load_model_by_index("Stem", d, 0x1234, &err) ==
        tw::load_model((dir.path() / "b.bin").string()));
  CHECK_FALSE(tw::load_model_by_index("OnlyChip", d, &err));
  CHECK(err.empty());
  CHECK_FALSE(tw::load_model_by_index("Stem", (dir.path() / "nowhere").string(), 0x1234, &err));
  CHECK(err.empty());
  CHECK_FALSE(tw::load_model_by_index("Stem", "", 0x1234, &err));
  CHECK(err.empty());

  write_text(dir.path() / "tilewright_index", "Stem\tb.bin\tchip=0x1234\nStem\tmissing.bin\n");
  CHECK(arch("Stem", 0x1234) == "gfxtestb");
  CHECK_FALSE(tw::load_model_by_index("Stem", d, 0x4321, &err));
  CHECK(mentions(err, "missing.bin"));
}

TEST_CASE("a malformed tilewright_index row of a stem makes its lookup fail",
          "[loader][registry]") {
  TempDir dir("index-malformed");
  write_file(dir.path() / "m.bin", write_model(small_spec()).bytes);
  const std::string d = dir.path().string();
  for (const char* row : {"Stem m.bin chip=0x",
                          "Stem m.bin chip=0x0",
                          "Stem m.bin chip=0x0000",
                          "Stem m.bin chip=0x12345",
                          "Stem m.bin chip=0x12g4",
                          "Stem m.bin chip=0X1234",
                          "Stem m.bin chip=1234",
                          "Stem m.bin chip=+0x12",
                          "Stem m.bin 0x1234",
                          "Stem m.bin id=0x1234",
                          "Stem m.bin extra",
                          "Stem\tm.bin\tchip=0x1234\textra",
                          "Stem m.bin chip=0x1234 chip=0x1234"}) {
    INFO(row);
    for (const std::string& before : {std::string(), std::string("Stem\tm.bin\n")}) {
      write_text(dir.path() / "tilewright_index", before + row + "\nStem m.bin chip=0x1234\n");
      const std::string line = before.empty() ? ":1:" : ":2:";
      for (std::uint32_t chip : {0u, 0x1234u, 0x4321u}) {
        std::string err;
        CHECK_FALSE(tw::load_model_by_index("Stem", d, chip, &err));
        CHECK(mentions(err, "tilewright_index" + line + " malformed row for Stem"));
      }
      std::string err;
      CHECK_FALSE(tw::load_model_by_index("Stem", d, &err));
      CHECK(mentions(err, "malformed row for Stem"));
      CHECK(tw::load_model_by_index("Other", d, &err) == nullptr);
      CHECK(err.empty());
    }
  }

  std::string err;
  write_text(dir.path() / "tilewright_index",
             "Stem m.bin chip=0x1234\nStem m.bin\nStem m.bin chip=0x1234 # a comment\n"
             "Stem m.bin chip=0xzz\n");
  for (std::uint32_t chip : {0u, 0x1234u}) {
    CHECK_FALSE(tw::load_model_by_index("Stem", d, chip, &err));
    CHECK(mentions(err, "tilewright_index:4: malformed row for Stem: 'chip=0xzz'"));
  }

  write_text(dir.path() / "tilewright_index", "Stem m.bin chip=0x\x01\x7f\n");
  CHECK_FALSE(tw::load_model_by_index("Stem", d, &err));
  CHECK(mentions(err, "'chip=0x\?\?'"));

  // Rows of one field are skipped, as are malformed rows of other stems.
  write_text(dir.path() / "tilewright_index", "Other m.bin chip=0xzz\nStem\nStem m.bin\n");
  err = "stale";
  CHECK(tw::load_model_by_index("Stem", d, 0x1234, &err));
  CHECK(err.empty());
}
