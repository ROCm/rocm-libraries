// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/detail/recommender.hpp"

#include "origami/nn/nn.hpp"
#include "origami/nn/detail/param_store.hpp"
#include "origami/nn/param/param_model.hpp"

#include <cmath>

#include <optional>
#include <vector>

namespace origami::nn::detail {
namespace {

model_handle_t first_valid(model_handle_t a, model_handle_t b, model_handle_t c) {
  if (a >= 0) return a;
  if (b >= 0) return b;
  return c;
}

}  // namespace

std::optional<std::vector<prediction_result_t>> try_rank_with_model(
    model_handle_t handle,
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    const inference_options_t& options) {
  (void)hardware;
  (void)options;
  const model_info_t* info = handle < 0 ? nullptr : nn::model_info(handle);
  if (info == nullptr) return std::nullopt;
  switch (info->backend) {
    case backend_id_t::param_v1: {
      auto model = param_payload(handle);
      if (!model) return std::nullopt;
      auto ranked = param::rank(*model, problem, configs);
      // Nothing scored (no candidate's parameters readable): let the caller fall back.
      if (ranked.empty() || !std::isfinite(ranked.front().latency)) return std::nullopt;
      return ranked;
    }
    case backend_id_t::tilewright_v1:
    case backend_id_t::embedding_similarity_v1:
      break;  // not built in
  }
  return std::nullopt;
}

model_handle_t resolve_model_handle(const rank_options_t& options) {
  switch (options.nn_backend) {
    case nn_backend_t::tilewright:
      return first_valid(options.nn_model,
                         options.library_models ? options.library_models->tilewright
                                                : invalid_handle,
                         nn::default_model(backend_id_t::tilewright_v1));
    case nn_backend_t::embedding_similarity:
      return first_valid(options.nn_model,
                         options.library_models ? options.library_models->embedding_similarity
                                                : invalid_handle,
                         nn::default_model(backend_id_t::embedding_similarity_v1));
    case nn_backend_t::param:
      return first_valid(options.nn_model,
                         options.library_models ? options.library_models->param : invalid_handle,
                         nn::default_model(backend_id_t::param_v1));
    case nn_backend_t::auto_select:
      if (options.nn_model >= 0) return options.nn_model;
      if (options.library_models) {
        if (options.library_models->tilewright >= 0) return options.library_models->tilewright;
        if (options.library_models->embedding_similarity >= 0)
          return options.library_models->embedding_similarity;
        return options.library_models->param;
      }
      return nn::default_model(backend_id_t::tilewright_v1);
  }
  return invalid_handle;
}

}  // namespace origami::nn::detail
