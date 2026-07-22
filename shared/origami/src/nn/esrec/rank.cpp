// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/esrec/rank.hpp"

#include "origami/nn/esrec/encoder.hpp"
#include "origami/nn/esrec/fallback.hpp"
#include "origami/nn/features/gemm_embedding_similarity.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace origami::nn::esrec {
namespace {

float dot(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return 0.0f;
  float sum = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace

std::vector<rank_entry_t> rank_configs(const detail::LoadedModel& model,
                                       const problem_t& problem,
                                       const hardware_t& hardware,
                                       const std::vector<config_t>& configs,
                                       const inference_options_t& /*options*/) {
  std::vector<rank_entry_t> empty;
  empty.resize(configs.size());
  for (std::size_t i = 0; i < configs.size(); ++i) {
    empty[i].config_index = i;
  }

  if (problem.batch != 1 || configs.empty()) {
    return empty;
  }

  const float m = static_cast<float>(problem.size.m);
  const float n = static_cast<float>(problem.size.n);
  const float k = static_cast<float>(problem.size.k);

  if (!model.fallback.empty()) {
    const int category = detail::classify_gemm(m, n, k, static_cast<float>(problem.batch));
    if (model.fallback.matches_pre_model(m, n, k, category)) {
      return empty;
    }
  }

  std::vector<float> raw_features(model.input_dim);
  features::gemm_embedding_similarity::build_query(
      problem, model.hw, model.is_nt, raw_features.data(), raw_features.size());

  const std::vector<float> query_embedding = encode_query(model.encoder, raw_features);

  std::vector<rank_entry_t> scored;
  scored.reserve(configs.size());

  for (std::size_t i = 0; i < configs.size(); ++i) {
    const int solution_index = static_cast<int>(configs[i].index);
    const auto it            = model.solution_by_index.find(solution_index);
    if (it == model.solution_by_index.end()) continue;

    rank_entry_t entry;
    entry.config_index = i;
    entry.score        = dot(query_embedding, it->second);
    entry.scored       = true;
    scored.push_back(entry);
  }

  if (scored.empty()) {
    return empty;
  }

  std::sort(scored.begin(), scored.end(), [](const rank_entry_t& a, const rank_entry_t& b) {
    return a.score > b.score;
  });

  if (!model.fallback.empty()) {
    const int category = detail::classify_gemm(m, n, k, static_cast<float>(problem.batch));
    if (model.fallback.matches_post_model(m, n, k, category, scored.front().score)) {
      return empty;
    }
  }

  std::vector<bool> used(configs.size(), false);
  std::vector<rank_entry_t> ranked;
  ranked.reserve(configs.size());
  for (const rank_entry_t& entry : scored) {
    ranked.push_back(entry);
    used[entry.config_index] = true;
  }
  for (std::size_t i = 0; i < configs.size(); ++i) {
    if (!used[i]) ranked.push_back(rank_entry_t{i, 0.0, false});
  }
  return ranked;
}

}  // namespace origami::nn::esrec
