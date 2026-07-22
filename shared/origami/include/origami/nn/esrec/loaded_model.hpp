// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/esrec/fallback.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace origami::nn::esrec::detail {

struct HardwareConstants {
  int   n_cu       = 256;
  float peak_flops = 2.3e15f;
  float mem_bw     = 8e12f;
  float l1_size    = 32768.0f;
  float l2_size    = 4194304.0f;
  float l3_size    = 268435456.0f;
  float wave_size  = 64.0f;
  float dtype_size = 2.0f;
  float acc_size   = 4.0f;
};

struct EncoderModel {
  std::vector<float> scaler_mean;
  std::vector<float> scaler_scale;
  std::vector<std::vector<float>> weights;
  std::vector<std::vector<float>> bias;
  std::vector<float>              proj_weights;
  std::vector<float>              proj_bias;
};

struct LoadedModel {
  std::string arch;
  std::string problem_stem;
  std::string weights_hash;
  std::uint32_t input_dim  = 0;
  std::uint32_t embed_dim  = 0;
  bool          is_nt      = false;

  HardwareConstants hw;
  EncoderModel      encoder;
  std::vector<std::vector<float>>              centroids;
  std::vector<std::vector<std::vector<float>>>   embeddings;
  std::vector<std::vector<int>>                  cluster_indices;
  std::unordered_map<int, std::vector<float>>    solution_by_index;

  FallbackEvaluator fallback;
};

}  // namespace origami::nn::esrec::detail
