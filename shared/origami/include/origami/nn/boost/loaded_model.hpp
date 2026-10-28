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

#include "origami/hardware.hpp"
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
  std::vector<std::int32_t>                       tree_root_;
  std::vector<std::uint32_t>                      tree_depth_;
  std::shared_ptr<const detail::forest_t>         forest_;  // quantized trees
  std::size_t                                     top_n_ = 10;
  std::vector<manifest_entry_t>                   manifest_;
  std::vector<std::set<std::string>>              seen_values_;
};

/// A model that passed every compatibility check (load_validate.cpp): the registry's boost_v1
/// payload and the only way to rank with it (rank.cpp).
class ORIGAMI_EXPORT booster_t {
 public:
  /**
   * @brief Load @p path and verify it against @p target and @p identity.
   *
   * Checks, in order: file format, arch, dtype, feature schema hash, Origami source hash, the
   * kernel library (diff_library on target.library_kernels, refused on retrain_required), that
   * every model column belongs to the schema and that the extractor computes it. The Origami
   * commit is not checked; on success @p detail reports both commits and the library diff.
   */
  static std::shared_ptr<const booster_t> create(const std::string&      path,
                                                 const deployment_t&     target,
                                                 const build_identity_t& identity,
                                                 status_t&               status,
                                                 std::string&            detail);
  /// Same, for an already parsed model.
  static std::shared_ptr<const booster_t> create(std::shared_ptr<const ensemble_t> model,
                                                 const deployment_t&               target,
                                                 const build_identity_t&           identity,
                                                 status_t&                         status,
                                                 std::string&                      detail);

  /**
   * @brief Re-order Origami's top candidates.
   *
   * @param candidates Origami's ranked candidates (best first). Only the first top_n are
   *        considered; the caller passes distinct kernels.
   */
  boost_result_t rank(const problem_t&   problem,
                      const hardware_t&  hardware,
                      const candidate_t* candidates,
                      std::size_t        num_candidates) const;

  /// Problem-level check only (dtype class, D dtype, M/N/K/batch training ranges; cheap). Any
  /// CU budget is accepted: the features see it through resolve_num_cus(problem.num_cus, N_CU).
  status_t check_problem(const problem_t&  problem,
                         const hardware_t& hardware,
                         std::string*      detail = nullptr) const;

  /// Model-ordered feature row for one candidate. False (with reason) if not computable.
  bool model_features(const problem_t&   problem,
                      const hardware_t&  hardware,
                      const candidate_t& candidate,
                      double*            row,
                      std::string*       why = nullptr) const;

  const ensemble_t& model() const { return *model_; }

 private:
  friend bool rerank_configs(const booster_t&,
                             const problem_t&,
                             const hardware_t&,
                             std::vector<prediction_result_t>&,
                             rerank_source_t*);
  booster_t() = default;

  /// The model's dtype class, read from HEAD once (check_problem runs on every call).
  enum class dtype_class_t { bf16, tf32, fp8, unknown };
  /// A training range with the problem dimension it bounds, in the model's RNGE order.
  struct range_check_t {
    const ensemble_t::range_t* range;
    int                        dim;  // 0 M, 1 N, 2 K, 3 batch
  };

  /// rank() after check_problem passed, into @p res (its vectors keep their capacity across
  /// calls). @p corrected: also fill res.corrected (rerank_configs does not read it).
  void rank_checked(const problem_t&   problem,
                    const hardware_t&  hardware,
                    const candidate_t* candidates,
                    std::size_t        num_candidates,
                    bool               corrected,
                    boost_result_t&    res) const;

  std::shared_ptr<const ensemble_t> model_;
  std::vector<std::size_t>          slot_;  // model column -> extractor slot
  std::vector<data_type_t>          d_dtypes_;
  dtype_class_t                     dtype_class_ = dtype_class_t::unknown;
  std::vector<range_check_t>        range_checks_;
};

}  // namespace origami::nn::boost
