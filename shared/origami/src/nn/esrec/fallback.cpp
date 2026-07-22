// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/esrec/fallback.hpp"

#include <yaml-cpp/yaml.h>

#include <cmath>
#include <limits>

namespace origami::nn::esrec::detail {
namespace {

bool open_open_contains(float value, float lower, float upper) {
  return value > lower && value < upper;
}

bool read_range_intervals(const YAML::Node& node, RangeIntervals* out) {
  if (!node || !node.IsSequence()) return false;
  if (node.size() % 2 != 0) return false;
  out->intervals.clear();
  for (std::size_t i = 0; i < node.size(); i += 2) {
    out->intervals.emplace_back(node[i].as<float>(), node[i + 1].as<float>());
  }
  return true;
}

bool read_categories(const YAML::Node& node, std::vector<int>* out) {
  if (!node || !node.IsSequence()) return false;
  out->clear();
  out->reserve(node.size());
  for (const auto& v : node) out->push_back(v.as<int>());
  return true;
}

}  // namespace

bool RangeIntervals::matches(float value) const {
  if (intervals.empty()) return true;
  for (const auto& [lo, hi] : intervals) {
    if (open_open_contains(value, lo, hi)) return true;
  }
  return false;
}

bool PreModelRule::matches(float m_val, float n_val, float k_val, int category) const {
  if (!m.matches(m_val)) return false;
  if (!n.matches(n_val)) return false;
  if (!k.matches(k_val)) return false;
  if (categories.empty()) return true;
  for (int cat : categories) {
    if (cat == category) return true;
  }
  return false;
}

bool PostModelRule::matches(float m_val,
                            float n_val,
                            float k_val,
                            int category,
                            float top_score) const {
  if (!PreModelRule::matches(m_val, n_val, k_val, category)) return false;
  return score.matches(top_score);
}

bool FallbackEvaluator::matches_pre_model(float m, float n, float k, int category) const {
  for (const auto& rule : pre_rules_) {
    if (rule.matches(m, n, k, category)) return true;
  }
  return false;
}

bool FallbackEvaluator::matches_post_model(float m,
                                           float n,
                                           float k,
                                           int category,
                                           float top_score) const {
  for (const auto& rule : post_rules_) {
    if (rule.matches(m, n, k, category, top_score)) return true;
  }
  return false;
}

bool FallbackEvaluator::load_from_yaml(const YAML::Node& fallback_node) {
  pre_rules_.clear();
  post_rules_.clear();
  if (!fallback_node || !fallback_node.IsMap()) return true;

  const YAML::Node pre = fallback_node["pre_model_features"];
  if (pre && pre.IsSequence()) {
    for (const auto& node : pre) {
      PreModelRule rule;
      rule.rule_id = node["rule_id"].as<int>();
      if (!read_range_intervals(node["m"], &rule.m)) return false;
      if (!read_range_intervals(node["n"], &rule.n)) return false;
      if (!read_range_intervals(node["k"], &rule.k)) return false;
      if (!read_categories(node["cats"], &rule.categories)) return false;
      pre_rules_.push_back(std::move(rule));
    }
  }

  const YAML::Node post = fallback_node["post_model_features"];
  if (post && post.IsSequence()) {
    for (const auto& node : post) {
      PostModelRule rule;
      rule.rule_id = node["rule_id"].as<int>();
      if (!read_range_intervals(node["m"], &rule.m)) return false;
      if (!read_range_intervals(node["n"], &rule.n)) return false;
      if (!read_range_intervals(node["k"], &rule.k)) return false;
      if (!read_range_intervals(node["score"], &rule.score)) return false;
      if (!read_categories(node["cats"], &rule.categories)) return false;
      post_rules_.push_back(std::move(rule));
    }
  }

  return true;
}

int classify_gemm(float m, float n, float k, float batch_count) {
  struct CategoryRule {
    int   cat;
    float m_min, m_max;
    float n_min, n_max;
    float k_min, k_max;
    float b_min, b_max;
  };

  const float INF = std::numeric_limits<float>::infinity();
  const CategoryRule categories[] = {
      {1, 2.0f, 1024.0f, 2.0f, 1024.0f, 2.0f, 1024.0f, 1.0f, 1.0f},
      {3, 4094.0f, 8192.0f, 4096.0f, 8192.0f, 4096.0f, 8192.0f, 1.0f, 1.0f},
      {2, 2.0f, 8192.0f, 2.0f, 8192.0f, 2.0f, 8192.0f, 1.0f, 1.0f},
      {5, 8193.0f, INF, 2.0f, 128.0f, 2.0f, 128.0f, 1.0f, 1.0f},
      {4, 8193.0f, INF, 2.0f, 8192.0f, 2.0f, 8192.0f, 1.0f, 1.0f},
      {7, 2.0f, 128.0f, 8193.0f, INF, 2.0f, 128.0f, 1.0f, 1.0f},
      {6, 2.0f, 8192.0f, 8193.0f, INF, 2.0f, 8192.0f, 1.0f, 1.0f},
      {9, 2.0f, 128.0f, 2.0f, 128.0f, 8193.0f, INF, 1.0f, 1.0f},
      {8, 2.0f, 8192.0f, 2.0f, 8192.0f, 8193.0f, INF, 1.0f, 1.0f},
      {10, 8193.0f, INF, 8193.0f, INF, 2.0f, 8192.0f, 1.0f, 1.0f},
      {11, 2.0f, 8192.0f, 8193.0f, INF, 8193.0f, INF, 1.0f, 1.0f},
      {12, 8193.0f, INF, 2.0f, 8192.0f, 8193.0f, INF, 1.0f, 1.0f},
      {13, 8193.0f, INF, 8193.0f, INF, 8193.0f, INF, 1.0f, 1.0f},
      {14, 1.0f, 1.0f, 1.0f, INF, 1.0f, INF, 1.0f, 1.0f},
      {15, 1.0f, INF, 1.0f, 1.0f, 1.0f, INF, 1.0f, 1.0f},
      {16, 1.0f, INF, 1.0f, INF, 1.0f, 1.0f, 1.0f, 1.0f},
  };

  for (const auto& rule : categories) {
    if (m >= rule.m_min && m <= rule.m_max && n >= rule.n_min && n <= rule.n_max &&
        k >= rule.k_min && k <= rule.k_max && batch_count >= rule.b_min &&
        batch_count <= rule.b_max) {
      return rule.cat;
    }
  }
  return -1;
}

}  // namespace origami::nn::esrec::detail
