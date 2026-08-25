// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/hardware.hpp"
#include "origami/nn/twrec/loaded_model.hpp"
#include "origami/nn/types.hpp"
#include "origami/types.hpp"

#include <cstddef>
#include <vector>

namespace origami::nn::twrec {

struct rank_entry_t {
  std::size_t config_index = 0;
  double      score        = 0.0;
  bool        scored       = false;
};

std::vector<rank_entry_t> rank_configs(const detail::LoadedModel& model,
                                       const problem_t& problem,
                                       const hardware_t& hardware,
                                       const std::vector<config_t>& configs,
                                       const inference_options_t& options);

}  // namespace origami::nn::twrec
