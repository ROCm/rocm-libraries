// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/types.hpp"

namespace origami::nn {

bool library_models_t::has(backend_id_t backend) const {
  return get(backend) >= 0;
}

model_handle_t library_models_t::get(backend_id_t backend) const {
  switch (backend) {
    case backend_id_t::tilewright_v1:
      return tilewright;
    case backend_id_t::embedding_similarity_v1:
      return embedding_similarity;
  }
  return invalid_handle;
}

}  // namespace origami::nn
