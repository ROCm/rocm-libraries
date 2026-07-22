// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "origami/nn/esrec/loaded_model.hpp"
#include "origami/origami_export.h"
#include "origami/types.hpp"

#include <cstddef>

namespace origami::nn::features::gemm_embedding_similarity {

constexpr const char* catalog_id         = "gemm_embedding_similarity";
constexpr const char* feature_names_hash = "embedding_similarity_v1";

/// Build raw GEMM feature vector before StandardScaler (TN: 141, NT: 192).
void ORIGAMI_EXPORT build_query(const problem_t& problem,
                                const esrec::detail::HardwareConstants& hw,
                                bool is_nt,
                                float* out,
                                std::size_t out_dim);

}  // namespace origami::nn::features::gemm_embedding_similarity
