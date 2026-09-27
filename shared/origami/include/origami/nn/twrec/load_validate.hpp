// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/twrec/load_limits.hpp"
#include "origami/nn/twrec/loaded_model.hpp"

#include <cstddef>
#include <string>

namespace origami::nn::twrec::detail {

struct WeightCounts {
  std::size_t q_w0, q_b0, q_w2, q_b2, q_w4, q_b4;
  std::size_t i_w0, i_b0, i_w2, i_b2;
  std::size_t x_w0, x_b0, x_w2, x_b2;
};

WeightCounts cell_weight_counts(std::uint32_t qd,
                                std::uint32_t id,
                                std::uint32_t xd,
                                std::uint32_t hd,
                                std::uint32_t ed,
                                std::uint32_t ih);

bool validate_feature_dims(std::uint32_t q_dim, std::uint32_t i_dim, std::uint32_t x_dim);

bool validate_string_len(const std::string& s);

bool validate_cell_hyperparams(const CellModel& cm);

bool validate_cell_weights(const CellModel& cm,
                           std::uint32_t q_dim,
                           std::uint32_t i_dim,
                           std::uint32_t x_dim);

bool validate_split_rule(const SplitRule& rule);

bool finalize_loaded_model(LoadedModel* model);

}  // namespace origami::nn::twrec::detail
