// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami::nn::twrec::detail {

struct CellModel {
  std::string label;
  std::uint32_t embed_dim     = 0;
  std::uint32_t hidden_dim    = 0;
  std::uint32_t inter_hidden  = 0;
  float temperature           = 1.0f;
  std::vector<float> q_mean, q_std;
  std::vector<float> i_mean, i_std;
  std::vector<float> x_mean, x_std;
  std::vector<float> q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  std::vector<float> i_w0, i_b0, i_w2, i_b2;
  std::vector<float> x_w0, x_b0, x_w2, x_b2;
  std::vector<std::array<int, 8>> smart_k_signatures;
};

struct SplitRule {
  std::string cell;
  char axis     = 'M';
  int threshold = 0;
  std::string lo_label;
  std::string hi_label;
};

struct LoadedModel {
  std::string feature_names_hash;
  std::string arch;
  std::uint32_t q_dim = 0;
  std::uint32_t i_dim = 0;
  std::uint32_t x_dim = 0;
  std::vector<CellModel> cells;
  std::unordered_map<std::string, std::size_t> cell_index;
  std::unordered_map<std::string, SplitRule> splits;
};

}  // namespace origami::nn::twrec::detail
