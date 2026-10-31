// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "origami/nn/boost/library_stamp.hpp"
#include "origami/nn/boost/loaded_model.hpp"
#include "origami/nn/boost/types.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

namespace origami::nn::boost {

/// This build's stamps: the configure-time Origami source hash and commit, and the compiled
/// feature schema (features/gemm_boost.hpp).
ORIGAMI_EXPORT build_identity_t build_identity();

}  // namespace origami::nn::boost

// The compatibility checks booster_t::create runs, in its order. Each returns false with a
// one-line reason in @p detail.
namespace origami::nn::boost::detail {

/// arch, dtype, feature schema hash, Origami source hash (status incompatible).
bool validate_stamps(const ensemble_t&       model,
                     const deployment_t&     target,
                     const build_identity_t& identity,
                     std::string&            detail);

/// The kernel library in use: refused without a manifest or on retrain_required (status
/// incompatible).
bool validate_library(const ensemble_t&   model,
                      const deployment_t& target,
                      library_diff_t&     diff,
                      std::string&        detail);

/// Every model column is in the compiled schema and computed by the extractor; @p slot maps model
/// columns to extractor slots (status missing_feature).
bool validate_columns(const ensemble_t& model, std::vector<std::size_t>& slot, std::string& detail);

/// The trained D dtypes (HEAD d_dtypes; status bad_file).
bool validate_d_dtypes(const ensemble_t&         model,
                       std::vector<data_type_t>& d_dtypes,
                       std::string&              detail);

}  // namespace origami::nn::boost::detail
