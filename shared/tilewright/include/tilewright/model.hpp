// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "tilewright/types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace tilewright {

class Model;
using ModelPtr = std::shared_ptr<const Model>;

struct ModelInfo {
  std::string arch;
  std::string feature_catalog_hash;
  WeightType weight_type = WeightType::Fp32;
  std::size_t n_cells    = 0;
  std::size_t n_splits   = 0;
};

// Feature vectors the scorer computes for one (problem, config) pair, before
// whitening. compute_features returns empty vectors for invalid hardware.
struct Features {
  std::vector<float> query;
  std::vector<float> item;
  std::vector<float> interaction;
};

// Hash of the feature catalog this engine computes. Model files carrying a
// different hash are rejected at load.
const char* feature_catalog_hash() noexcept;

// Loads a model file. Returns nullptr on any error and stores the reason in
// *error when error is non-null. Loading a path that resolves to an already
// loaded file returns the same model.
ModelPtr load_model(const std::string& path, std::string* error = nullptr) noexcept;

// Loads a model from an in-memory image of a model file. Not cached.
ModelPtr load_model_from_memory(const void* data,
                                std::size_t size,
                                std::string* error = nullptr) noexcept;

// Looks up `logic_stem` in `<dir>/tilewright_index` and loads the model it
// lists. Returns nullptr with *error left empty when the index file or the stem
// is absent, and nullptr with *error set when the listed model fails to load.
ModelPtr load_model_by_index(const std::string& logic_stem,
                             const std::string& dir,
                             std::string* error = nullptr) noexcept;

ModelInfo describe(const Model& model);

// Index of the trained cell that scores `problem` (the split-tree leaf, or its
// nearest trained ancestor), or -1 when there is none.
int route(const Model& model, const Problem& problem) noexcept;

// Label of cell `cell` (as returned by route), or an empty string.
std::string cell_label(const Model& model, int cell);

Features compute_features(const Model& model,
                          const Problem& problem,
                          const Config& config,
                          const Hardware& hardware);

// A caller's fixed pool of candidate kernels, ranked against one model.
// Per-cell data that depends only on the pool (item embeddings, whitelist
// membership) is computed on the first rank call that reaches the cell and
// reused afterwards. Safe to share between threads. The constructor throws
// std::invalid_argument for a null model.
class CandidateSet {
 public:
  CandidateSet(ModelPtr model, std::vector<Config> configs);
  ~CandidateSet();

  CandidateSet(const CandidateSet&)            = delete;
  CandidateSet& operator=(const CandidateSet&) = delete;

  const Model& model() const noexcept;
  const std::vector<Config>& configs() const noexcept;

  // Returns one Result per config. Scored configs come first: survivors of the
  // cell's whitelist (every feasible config when the whitelist keeps none)
  // best-first, followed (only when min_scored exceeds their count) by the
  // remaining feasible configs best-first; ties keep input order. Unscored
  // configs, including those whose score is not finite, follow in input order.
  // Invalid hardware (a zero field) or a problem with no trained cell returns
  // every config unscored.
  std::vector<Result> rank(const Problem& problem,
                           const Hardware& hardware,
                           std::size_t min_scored = 0) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Same result as CandidateSet::rank for an ad-hoc config list, without caching.
std::vector<Result> rank_configs(const Model& model,
                                 const Problem& problem,
                                 const Hardware& hardware,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored = 0);

}  // namespace tilewright
