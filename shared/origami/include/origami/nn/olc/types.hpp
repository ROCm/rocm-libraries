// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "origami/origami_export.h"
#include "origami/types.hpp"

/**
 * @file types.hpp
 * @brief Origami Learned Correction (OLC, backend olc_v1): a learned correction that re-orders
 * Origami's top candidates.
 *
 * A per-arch, per-dtype model file (ORIOLC format, `.orr`) holds a gradient-boosted tree
 * ensemble that predicts yhat ~ log(measured_us) - log(origami_cycles) for one candidate kernel.
 * The corrector computes each candidate's feature vector from Origami's own intermediates
 * (features/gemm_olc), evaluates the trees, and orders the candidates by
 * origami_cycles * exp(yhat), keeping Origami's order on ties. It never ranks by yhat alone, so
 * it runs after the analytical ranking (inference_mode_t::analytical_rerank).
 *
 * A model is tied to one Origami source state, one feature schema and its training kernel
 * library, by manifest diff: the library in use may drop kernels and add kernels of known
 * structure (library_stamp.hpp). corrector_t::create refuses anything else. Every per-call failure
 * returns Origami's order unchanged with a reason.
 */
namespace origami::nn::olc {

/// Outcome of loading a model or ranking a call. Anything but ok means "use stock order".
enum class status_t : int {
  ok = 0,
  model_missing,       ///< no model file at the path
  bad_file,            ///< unreadable, bad magic, bad checksum, malformed section
  unsupported_format,  ///< format version, structural-field table or tree form this reader lacks
  incompatible,        ///< arch / dtype / Origami / schema / kernel-library mismatch
  out_of_range,        ///< problem outside the model's dtype class or training ranges
  missing_feature,     ///< a model input the extractor cannot compute for this candidate
  too_few_candidates,  ///< fewer than two usable candidates
};

ORIGAMI_EXPORT const char* to_string(status_t status);

/// The feature schema the extractor was written against: ordered column names and their hash.
struct feature_schema_t {
  std::string              id;
  std::string              hash;
  std::vector<std::string> columns;
};

/// What this build of Origami is. A model must carry the same stamps.
struct build_identity_t {
  std::string origami_commit;       ///< last commit touching the model sources (logged, not checked)
  std::string origami_source_hash;  ///< sha256 over the model sources (configure time)
  std::string feature_schema_id;
  std::string feature_schema_hash;
};

struct manifest_entry_t;

/// Where the model is used. Supplied by the caller (e.g. hipBLASLt when it loads a library).
struct deployment_t {
  std::string arch;   ///< e.g. "gfx950"
  std::string dtype;  ///< problem class: "bf16", "tf32" or "fp8"
  /// Canonical manifest of the library's PredictionMatching rows for this dtype class
  /// (@see canonical_manifest()).
  const std::vector<manifest_entry_t>* library_kernels = nullptr;
};

/**
 * @brief Kernel fields the caller knows that config_t does not carry.
 *
 * Filled from the solution's library record (hipBLASLt: ContractionSolution::sizeMapping),
 * never from the kernel name. -1 means "not provided"; a model that needs a field the caller
 * did not provide falls back with missing_feature.
 */
struct kernel_fields_t {
  int num_loads_coalesced_a = -1;  ///< sizeMapping.NumLoadsCoalescedA
  int num_loads_coalesced_b = -1;  ///< sizeMapping.NumLoadsCoalescedB
};

/// One candidate in Origami's ranked order.
struct candidate_t {
  /// The config rank_configs saw, with non_temporal_{a,b}_available set by
  /// memoize_non_temporal_availability over the full candidate list.
  const config_t* config          = nullptr;
  double          origami_latency = 0.0;  ///< rank_configs latency (cycles)
  kernel_fields_t fields{};
};

/// Per-call outcome. On any status but ok, order is 0..n-1 (Origami's order).
struct olc_result_t {
  status_t            status = status_t::ok;
  std::string         detail;
  std::vector<size_t> order;      ///< candidate indices, best first
  std::vector<double> yhat;       ///< per candidate (input order), empty on fallback
  std::vector<double> corrected;  ///< origami_latency * exp(yhat), input order
};

/**
 * @brief The caller's side of a re-rank (inference_mode_t::analytical_rerank), passed in
 * rank_options_t::nn.rerank_source.
 *
 * rank_configs hands the analytical results to prepare() first, once per call, whatever the re-rank
 * then decides. The backend walks the results best first and offers each config until the model's
 * top_n are accepted; the accepted ones are its candidates. On success they move to the front of
 * the results in the model's order and every other result keeps its position relative to the
 * rest; on any refusal the results stay as prepare() left them.
 */
class rerank_source_t {
 public:
  virtual ~rerank_source_t() = default;

  /// The caller's own adjustment of the analytical order, made before the re-rank sees it.
  /// Default: none.
  virtual void prepare(std::vector<prediction_result_t>& results) { (void)results; }

  /// False skips @p config (the caller would not use it). True makes it a candidate, with the
  /// kernel fields config_t does not carry in @p fields.
  virtual bool candidate(const config_t& config, kernel_fields_t& fields) = 0;

  /// The call's outcome. With ok, @p order lists the accepted candidates best first (indices in
  /// acceptance order); otherwise it is empty and @p detail says why.
  virtual void outcome(status_t                   status,
                       const std::string&         detail,
                       const std::vector<size_t>& order) {
    (void)status;
    (void)detail;
    (void)order;
  }
};

}  // namespace origami::nn::olc
