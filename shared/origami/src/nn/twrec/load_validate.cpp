// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/twrec/load_validate.hpp"

#include <cmath>

namespace origami::nn::twrec::detail {

WeightCounts cell_weight_counts(std::uint32_t qd,
                                std::uint32_t id,
                                std::uint32_t xd,
                                std::uint32_t hd,
                                std::uint32_t ed,
                                std::uint32_t ih) {
  return WeightCounts{std::size_t(hd) * qd,
                      hd,
                      std::size_t(hd) * hd,
                      hd,
                      std::size_t(ed) * hd,
                      ed,
                      std::size_t(hd) * id,
                      hd,
                      std::size_t(ed) * hd,
                      ed,
                      std::size_t(ih) * xd,
                      ih,
                      ih,
                      1};
}

bool validate_string_len(const std::string& s) { return s.size() <= kMaxStrLen; }

bool validate_feature_dims(std::uint32_t q_dim, std::uint32_t i_dim, std::uint32_t x_dim) {
  if (q_dim == 0 || i_dim == 0 || x_dim == 0) return false;
  if (q_dim > kMaxQDim || i_dim > kMaxIDim || x_dim > kMaxXDim) return false;
  return true;
}

bool validate_cell_hyperparams(const CellModel& cm) {
  if (!validate_string_len(cm.label)) return false;
  if (!std::isfinite(cm.temperature)) return false;
  return valid_hyperparam_tier(cm.embed_dim, cm.hidden_dim, cm.inter_hidden);
}

bool validate_cell_weights(const CellModel& cm,
                           std::uint32_t q_dim,
                           std::uint32_t i_dim,
                           std::uint32_t x_dim) {
  const auto wc = cell_weight_counts(
      q_dim, i_dim, x_dim, cm.hidden_dim, cm.embed_dim, cm.inter_hidden);
  const auto ok = [&](std::size_t got, std::size_t want) { return got == want; };
  return ok(cm.q_mean.size(), q_dim) && ok(cm.q_std.size(), q_dim) &&
         ok(cm.i_mean.size(), i_dim) && ok(cm.i_std.size(), i_dim) &&
         ok(cm.x_mean.size(), x_dim) && ok(cm.x_std.size(), x_dim) &&
         ok(cm.q_w0.size(), wc.q_w0) && ok(cm.q_b0.size(), wc.q_b0) &&
         ok(cm.q_w2.size(), wc.q_w2) && ok(cm.q_b2.size(), wc.q_b2) &&
         ok(cm.q_w4.size(), wc.q_w4) && ok(cm.q_b4.size(), wc.q_b4) &&
         ok(cm.i_w0.size(), wc.i_w0) && ok(cm.i_b0.size(), wc.i_b0) &&
         ok(cm.i_w2.size(), wc.i_w2) && ok(cm.i_b2.size(), wc.i_b2) &&
         ok(cm.x_w0.size(), wc.x_w0) && ok(cm.x_b0.size(), wc.x_b0) &&
         ok(cm.x_w2.size(), wc.x_w2) && ok(cm.x_b2.size(), wc.x_b2);
}

bool validate_split_rule(const SplitRule& rule) {
  if (!validate_string_len(rule.cell)) return false;
  if (!validate_string_len(rule.lo_label)) return false;
  if (!validate_string_len(rule.hi_label)) return false;
  if (!valid_split_axis(rule.axis)) return false;
  return true;
}

bool finalize_loaded_model(LoadedModel* model) {
  if (!model) return false;
  if (!validate_feature_dims(model->q_dim, model->i_dim, model->x_dim)) return false;
  if (model->cells.size() > kMaxCells) return false;
  if (model->splits.size() > kMaxSplits) return false;
  if (!validate_string_len(model->arch)) return false;
  if (!validate_string_len(model->feature_names_hash)) return false;

  for (const auto& kv : model->splits) {
    if (!validate_split_rule(kv.second)) return false;
  }

  model->cell_index.clear();
  for (std::size_t i = 0; i < model->cells.size(); ++i) {
    const CellModel& cm = model->cells[i];
    if (!validate_cell_hyperparams(cm)) return false;
    if (!validate_cell_weights(cm, model->q_dim, model->i_dim, model->x_dim)) return false;
    if (cm.smart_k_signatures.size() > kMaxSmartK) return false;
    auto [it, inserted] = model->cell_index.emplace(cm.label, i);
    if (!inserted) return false;
  }
  return true;
}

}  // namespace origami::nn::twrec::detail
