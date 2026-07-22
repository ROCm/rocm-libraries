// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace YAML {
class Node;
}

namespace origami::nn::esrec::detail {

struct RangeIntervals {
  std::vector<std::pair<float, float>> intervals;

  bool matches(float value) const;
};

struct PreModelRule {
  int               rule_id = 0;
  RangeIntervals    m;
  RangeIntervals    n;
  RangeIntervals    k;
  std::vector<int>  categories;

  bool matches(float m_val, float n_val, float k_val, int category) const;
};

struct PostModelRule : PreModelRule {
  RangeIntervals score;

  bool matches(float m_val, float n_val, float k_val, int category, float top_score) const;
};

/// Compiled fallback evaluator (rules loaded once; evaluated via functions).
class FallbackEvaluator {
 public:
  bool empty() const { return pre_rules_.empty() && post_rules_.empty(); }

  bool matches_pre_model(float m, float n, float k, int category) const;
  bool matches_post_model(float m, float n, float k, int category, float top_score) const;

  bool load_from_yaml(const YAML::Node& fallback_node);

 private:
  std::vector<PreModelRule>  pre_rules_;
  std::vector<PostModelRule> post_rules_;
};

int classify_gemm(float m, float n, float k, float batch_count);

}  // namespace origami::nn::esrec::detail
