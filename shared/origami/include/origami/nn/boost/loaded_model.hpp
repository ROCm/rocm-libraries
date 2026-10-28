// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "origami/nn/boost/library_stamp.hpp"
#include "origami/nn/boost/types.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami::nn::boost {

namespace detail {
struct forest_t;
}

/// Parsed model file (orr_loader.cpp). Holds no compatibility verdict; use booster_t to run it.
class ORIGAMI_EXPORT ensemble_t {
 public:
  struct feature_t {
    std::string name;
  };
  struct range_t {
    std::string name;
    double      min = 0.0;
    double      max = 0.0;
  };

  /// Load from a file. On failure returns nullptr and sets @p status / @p detail.
  static std::shared_ptr<const ensemble_t> load_file(const std::string& path,
                                                     status_t&          status,
                                                     std::string&       detail);
  /// Load from memory (tests, embedded models).
  static std::shared_ptr<const ensemble_t> load_bytes(const void*  data,
                                                      std::size_t  size,
                                                      status_t&    status,
                                                      std::string& detail);

  /// Ensemble output for @p n rows in model column order, stored contiguously (row i at
  /// rows + i * features().size()). Rows must have been through prepare().
  void evaluate_batch(const double* rows, std::size_t n, double* out) const;
  double evaluate(const double* row) const;

  /// Non-finite values to 0 (as training's nan_to_num), in place.
  void prepare(double* row) const;

  const std::vector<feature_t>& features() const { return features_; }
  /// Training ranges: M, N, K and batch, each once (the loader refuses any other entry).
  const std::vector<range_t>&   ranges() const { return ranges_; }
  std::size_t                   num_trees() const { return tree_root_.size(); }
  std::size_t                   top_n() const { return top_n_; }

  /// HEAD string value ("" if absent).
  std::string head_string(const std::string& key) const;

  /// The training library's canonical manifest (MANI section).
  const std::vector<manifest_entry_t>& library_manifest() const { return manifest_; }
  /// Per structural field (structural_fields() order), the values seen in training.
  const std::vector<std::set<std::string>>& seen_values() const { return seen_values_; }

 private:
  ensemble_t() = default;
  friend struct model_parser;

  std::vector<std::pair<std::string, std::string>> head_strings_;
  std::vector<feature_t>                          features_;
  std::vector<range_t>                            ranges_;
  std::vector<double>                             values_;  // leaf value per node position
  std::vector<std::int32_t>                       tree_root_;
  std::vector<std::uint32_t>                      tree_depth_;
  std::shared_ptr<const detail::forest_t>         forest_;  // quantized trees
  std::size_t                                     top_n_ = 10;
  std::vector<manifest_entry_t>                   manifest_;
  std::vector<std::set<std::string>>              seen_values_;
};

}  // namespace origami::nn::boost
