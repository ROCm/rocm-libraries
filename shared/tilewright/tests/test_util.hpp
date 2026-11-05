// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "model_impl.hpp"
#include "model_writer.hpp"
#include "tilewright/model.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilewright_test {

// A fresh directory under the system temp directory, removed on destruction.
class TempDir {
 public:
  explicit TempDir(const std::string& tag) {
    std::random_device rd;
    path_ = std::filesystem::temp_directory_path() /
            ("tilewright-test-" + tag + "-" + std::to_string(rd()) + std::to_string(rd()));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  TempDir(const TempDir&)            = delete;
  TempDir& operator=(const TempDir&) = delete;

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

inline void write_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!f) throw std::runtime_error("cannot write " + path.string());
}

inline void write_text(const std::filesystem::path& path, const std::string& text) {
  write_file(path, std::vector<unsigned char>(text.begin(), text.end()));
}

inline tilewright::ModelPtr load_spec(const ModelSpec& spec) {
  const Image image = write_model(spec);
  std::string error;
  tilewright::ModelPtr model =
      tilewright::load_model_from_memory(image.bytes.data(), image.bytes.size(), &error);
  if (!model) throw std::runtime_error("synthetic model rejected: " + error);
  return model;
}

inline tilewright::Problem make_problem(std::size_t m,
                                        std::size_t n,
                                        std::size_t k,
                                        std::size_t batch        = 1,
                                        tilewright::DataType dt  = tilewright::DataType::BFloat16,
                                        tilewright::Transpose ta = tilewright::Transpose::T,
                                        tilewright::Transpose tb = tilewright::Transpose::N) {
  tilewright::Problem p;
  p.size        = {m, n, k};
  p.batch       = batch;
  p.a_transpose = ta;
  p.b_transpose = tb;
  p.a_dtype = p.b_dtype = p.c_dtype = p.d_dtype = p.mi_dtype = dt;
  return p;
}

inline tilewright::Config make_config(std::size_t mt_m,
                                      std::size_t mt_n,
                                      std::size_t mt_k,
                                      std::size_t mi_m = 16,
                                      std::size_t mi_n = 16,
                                      std::size_t mi_k = 32,
                                      int cha          = 0,
                                      int chb          = 0,
                                      int occupancy    = 1) {
  tilewright::Config c;
  c.mt            = {mt_m, mt_n, mt_k};
  c.mi            = {mi_m, mi_n, mi_k};
  c.occupancy     = occupancy;
  c.cache_hints_a = cha;
  c.cache_hints_b = chb;
  c.grvw_a        = 8;
  c.grvw_b        = 8;
  c.gwvw_d        = 4;
  return c;
}

inline const tilewright::Hardware& test_hardware() {
  static const tilewright::Hardware hw{64, 65536, 4u << 20};
  return hw;
}

// Index, scored flag and score bits all equal.
inline bool same_results(const std::vector<tilewright::Result>& a,
                         const std::vector<tilewright::Result>& b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i].config_index != b[i].config_index || a[i].scored != b[i].scored) return false;
    if (std::memcmp(&a[i].score, &b[i].score, sizeof(double)) != 0) return false;
  }
  return true;
}

// The scored results among the first k of `r`.
inline std::vector<tilewright::Result> scored_prefix(const std::vector<tilewright::Result>& r,
                                                     std::size_t k) {
  std::vector<tilewright::Result> out;
  for (const tilewright::Result& x : r) {
    if (!x.scored || out.size() == k) break;
    out.push_back(x);
  }
  return out;
}

inline std::vector<std::size_t> scored_indices(const std::vector<tilewright::Result>& r) {
  std::vector<std::size_t> out;
  for (const tilewright::Result& x : r)
    if (x.scored) out.push_back(x.config_index);
  return out;
}

// A grouping with one group per config, so that detail::rank_keys scores every
// config on its own.
inline tilewright::detail::PoolKeys singleton_keys(const std::vector<tilewright::Config>& pool) {
  tilewright::detail::PoolKeys k;
  const std::size_t n = pool.size();
  k.key_of.resize(n);
  k.first.resize(n);
  k.members.resize(n);
  k.member_begin.resize(n + 1);
  const std::vector<std::uint8_t> family = tilewright::detail::family_bits(pool);
  for (std::size_t i = 0; i < n; ++i) {
    k.key_of[i] = k.first[i] = k.members[i] = k.member_begin[i] = i;
    k.shape.push_back(tilewright::detail::kernel_shape(pool[i]));
    k.shape.back().family = family[i];
    k.nt_a |= pool[i].cache_hints_a == 4;
    k.nt_b |= pool[i].cache_hints_b == 4;
  }
  k.member_begin[n] = n;
  return k;
}

// Whether an alignment veto of K above `k_min` and `align_bits`-bit lines
// applies to `p`, computed apart from the engine: a layout with exactly one
// K-contiguous operand (A transposed with B transposed, or neither), K above
// `k_min`, and K elements of that operand short of whole lines at the widths
// of the non-temporal rule.
inline bool veto_applies(const tilewright::Problem& p,
                         std::uint64_t k_min      = 512,
                         std::uint64_t align_bits = 1024) {
  const bool a = p.a_transpose == tilewright::Transpose::T;
  const bool b = p.b_transpose == tilewright::Transpose::N;
  if (a == b || p.size.k <= k_min) return false;
  const tilewright::DataType dt = a ? p.a_dtype : p.b_dtype;
  std::uint64_t bits            = 16;
  if (dt == tilewright::DataType::Float || dt == tilewright::DataType::XFloat32) bits = 32;
  if (dt == tilewright::DataType::Float8 || dt == tilewright::DataType::Float8_fnuz ||
      dt == tilewright::DataType::BFloat8 || dt == tilewright::DataType::BFloat8_fnuz)
    bits = 8;
  return p.size.k * bits % align_bits != 0;
}

// Whether that veto names `c` for `p`: its cache hint on the operand the veto
// reads is the non-temporal one.
inline bool vetoed(const tilewright::Problem& p,
                   const tilewright::Config& c,
                   std::uint64_t k_min      = 512,
                   std::uint64_t align_bits = 1024) {
  return veto_applies(p, k_min, align_bits) &&
         (p.a_transpose == tilewright::Transpose::T ? c.cache_hints_a : c.cache_hints_b) == 4;
}

// What a model with that veto must rank, from `rank(configs)`, the same model
// without it ranking a list of configs as the caller does: the pool without
// the configs the veto names, mapped back to the pool, with every other
// config unscored in input order; or the whole pool's ranking when the rest
// leaves nothing scored.
template <class Rank>
std::vector<tilewright::Result> expected_with_veto(const tilewright::Problem& p,
                                                   const std::vector<tilewright::Config>& pool,
                                                   Rank rank,
                                                   std::uint64_t k_min      = 512,
                                                   std::uint64_t align_bits = 1024) {
  std::vector<std::size_t> kept;
  std::vector<tilewright::Config> rest;
  for (std::size_t i = 0; i < pool.size(); ++i)
    if (!vetoed(p, pool[i], k_min, align_bits)) {
      kept.push_back(i);
      rest.push_back(pool[i]);
    }
  if (rest.size() == pool.size() || rest.empty()) return rank(pool);
  const std::vector<tilewright::Result> sub = rank(rest);
  if (sub.empty() || !sub.front().scored) return rank(pool);
  std::vector<tilewright::Result> out;
  std::vector<bool> scored(pool.size(), false);
  for (const tilewright::Result& r : sub) {
    if (!r.scored) break;
    out.push_back(tilewright::Result{kept[r.config_index], r.score, true});
    scored[kept[r.config_index]] = true;
  }
  for (std::size_t i = 0; i < pool.size(); ++i)
    if (!scored[i]) out.push_back(tilewright::Result{i, 0.0, false});
  return out;
}

// Where an MLREC_v3 image's serving flags are (`flags_at`, their value
// `flags`) and where the alignment veto's fields are, or would go (`veto_at`:
// after the trained keys and the CU floor, as training_pipeline/lib/mlrec.py
// writes them); false when `image` is not an MLREC_v3 file whose header can
// be walked.
inline bool alignment_veto_offsets(const std::vector<unsigned char>& image,
                                   std::size_t* flags_at,
                                   std::uint32_t* flags,
                                   std::size_t* veto_at) {
  constexpr std::size_t kFixedHeader = 52;
  if (image.size() < kFixedHeader || std::memcmp(image.data(), "MLREC_v3", 8) != 0) return false;
  std::size_t at   = kFixedHeader;
  const auto has   = [&](std::size_t n) { return at <= image.size() && n <= image.size() - at; };
  const auto get16 = [&](std::size_t o) {
    return static_cast<std::uint32_t>(image[o]) | static_cast<std::uint32_t>(image[o + 1]) << 8;
  };
  const auto get32      = [&](std::size_t o) { return get16(o) | get16(o + 2) << 16; };
  const auto skip_label = [&] {
    if (!has(2)) return false;
    at += 2 + get16(at);
    return true;
  };
  // feature_catalog_hash, feature_spec_hash, arch, the two arch constants and
  // the MI table of 24-byte rows.
  if (!skip_label()) return false;
  at += 8;
  if (!skip_label()) return false;
  at += 16;
  if (!has(4)) return false;
  at += 4 + 24 * static_cast<std::size_t>(get32(at));
  if (!has(4)) return false;
  *flags_at = at;
  *flags    = get32(at);
  // serving flags, cascade width, CU fraction range, schedules, clip.
  at += 4 + 4 + 16 + 1 + 1;
  if (!has(2)) return false;
  const std::uint32_t n_attributes = get16(at);
  at += 2;
  for (std::uint32_t a = 0; a < n_attributes; ++a) {
    if (!skip_label()) return false;
    at += 8;
    if (!has(1)) return false;
    const unsigned transform = image[at++];
    if (transform == 3) {
      if (!has(2)) return false;
      at += 2 + 8 * static_cast<std::size_t>(get16(at));
    }
  }
  if ((*flags & 0x80) != 0) {
    if (!has(4)) return false;
    at += 4 + 8 * static_cast<std::size_t>(get32(at));
  }
  if ((*flags & 0x100) != 0) at += 4;
  if (!has((*flags & 0x200) != 0 ? 8 : 0)) return false;
  *veto_at = at;
  return true;
}

inline void le32_put(std::vector<unsigned char>& out, std::size_t o, std::uint32_t v) {
  for (std::size_t b = 0; b < 4; ++b) out[o + b] = static_cast<unsigned char>(v >> (8 * b));
}

inline std::uint32_t le32_get(const std::vector<unsigned char>& in, std::size_t o) {
  std::uint32_t v = 0;
  for (std::size_t b = 0; b < 4; ++b) v |= static_cast<std::uint32_t>(in[o + b]) << (8 * b);
  return v;
}

// An MLREC_v3 image with the alignment veto added (the serving flag and its
// fields, as training_pipeline/lib/mlrec.py's with_alignment_veto writes
// them); empty when `image` is not an MLREC_v3 file whose header can be
// walked, or already declares the veto.
inline std::vector<unsigned char> with_alignment_veto(const std::vector<unsigned char>& image,
                                                      std::uint32_t k_min,
                                                      std::uint32_t align_bits) {
  std::size_t flags_at = 0, at = 0;
  std::uint32_t flags = 0;
  if (!alignment_veto_offsets(image, &flags_at, &flags, &at) || (flags & 0x200) != 0) return {};
  std::vector<unsigned char> out(image.begin(), image.begin() + static_cast<long>(at));
  out.resize(out.size() + 8);
  le32_put(out, at, k_min);
  le32_put(out, at + 4, align_bits);
  out.insert(out.end(), image.begin() + static_cast<long>(at), image.end());
  le32_put(out, flags_at, flags | 0x200);
  le32_put(out, 12, le32_get(image, 12) + 8);
  return out;
}

// An MLREC_v3 image without the alignment veto (its serving flag and fields
// removed, as lib/mlrec.py's without_alignment_veto removes them): the image
// itself when it declares none; empty when it is not an MLREC_v3 file whose
// header can be walked.
inline std::vector<unsigned char> without_alignment_veto(const std::vector<unsigned char>& image) {
  std::size_t flags_at = 0, at = 0;
  std::uint32_t flags = 0;
  if (!alignment_veto_offsets(image, &flags_at, &flags, &at)) return {};
  if ((flags & 0x200) == 0) return image;
  std::vector<unsigned char> out(image.begin(), image.begin() + static_cast<long>(at));
  out.insert(out.end(), image.begin() + static_cast<long>(at + 8), image.end());
  le32_put(out, flags_at, flags & ~std::uint32_t{0x200});
  le32_put(out, 12, le32_get(image, 12) - 8);
  return out;
}

}  // namespace tilewright_test
