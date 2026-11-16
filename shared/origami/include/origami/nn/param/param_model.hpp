// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace origami::nn::param {

/// A parameter-predictor model (msgpack contract v1): a small MLP over features of the problem
/// size (M, N, K, batch) with one softmax head per kernel parameter (MT0, MT1, DU, GSU, WGM, SU,
/// SUM, SUS, PGR, TLDS, LDSB, DTVB). A candidate kernel scores
///   sum_h head_weight[h] * log p_h(the kernel's own value of parameter h),
/// a value the model has no class for scores unk_logp. The values are read from the candidate's
/// config_t, so the model is not tied to one catalog's row order.
struct model_t;

/// Whether this build can read param models (Origami built with msgpack support).
bool supported();

/// Load and validate a model file; nullptr with *error set when it is refused.
std::shared_ptr<const model_t> load(const std::string& path, std::string* error);

/// The logic-file key the model was trained for, e.g. gfx1151_Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs.
const std::string& problem_key(const model_t& model);

/// One-line description (features, layers, heads, test vectors) for diagnostics.
std::string summary(const model_t& model);

/// Recompute the model's embedded test vectors (features to 1e-4, top-10 order up to score ties).
bool self_test(const model_t& model, std::string* error);

/// Value-based score of one candidate; false when its parameters cannot be read.
bool score(const model_t& model,
           const std::vector<std::vector<double>>& logp,
           const config_t& config,
           double* out);

/// Per-head log-probabilities for a problem size.
std::vector<std::vector<double>> head_logp(const model_t& model,
                                           std::uint64_t m,
                                           std::uint64_t n,
                                           std::uint64_t k,
                                           std::uint64_t batch);

/// Every config, best first (latency = -score, ties keep the input order); configs whose
/// parameters cannot be read get a NaN latency and come last.
std::vector<prediction_result_t> rank(const model_t& model,
                                      const problem_t& problem,
                                      const std::vector<config_t>& configs);

}  // namespace origami::nn::param
