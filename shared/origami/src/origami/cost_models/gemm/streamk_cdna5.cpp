// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <cstddef>

#include "origami/gemm.hpp"
#include "origami/types.hpp"

#include "model.hpp"

namespace origami {
namespace gemm {

// Workgroups per cluster (Cs along M by Ck along N).
size_t cdna5_model::cluster_size(const config_t& config) {
  return std::max<size_t>(config.cluster_dim.m, 1) * std::max<size_t>(config.cluster_dim.n, 1);
}

}  // namespace gemm
}  // namespace origami
