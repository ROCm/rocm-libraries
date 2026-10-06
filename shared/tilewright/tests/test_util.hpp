// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "model_impl.hpp"
#include "model_writer.hpp"
#include "tilewright/model.hpp"

#include <cstddef>
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
  for (std::size_t i = 0; i < n; ++i) {
    k.key_of[i] = k.first[i] = k.members[i] = k.member_begin[i] = i;
    k.nt_a |= pool[i].cache_hints_a == 4;
    k.nt_b |= pool[i].cache_hints_b == 4;
  }
  k.member_begin[n] = n;
  return k;
}

}  // namespace tilewright_test
