// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/detail/recommender.hpp"

#include "origami/nn/olc/rank.hpp"
#include "origami/nn/detail/model_store.hpp"
#include "origami/nn/nn.hpp"

#include <optional>
#include <vector>

namespace origami::nn::detail {
namespace {

model_handle_t first_valid(model_handle_t a, model_handle_t b, model_handle_t c) {
  if (a >= 0) return a;
  if (b >= 0) return b;
  return c;
}

class OlcReranker final : public IReranker {
 public:
  OlcReranker(model_handle_t handle, const olc::corrector_t& corrector)
      : handle_(handle), corrector_(corrector) {}

  model_info_t info() const override {
    if (const model_info_t* registered = nn::model_info(handle_)) {
      return *registered;
    }
    model_info_t fallback;
    fallback.backend = backend_id_t::olc_v1;
    return fallback;
  }

  bool rerank(const problem_t& problem,
              const hardware_t& hardware,
              std::vector<prediction_result_t>& results,
              const inference_options_t& options) override {
    return olc::rerank_configs(corrector_, problem, hardware, results, options.rerank_source);
  }

 private:
  model_handle_t handle_;
  const olc::corrector_t& corrector_;
};

}  // namespace

std::optional<std::vector<prediction_result_t>> try_rank_with_model(
    model_handle_t handle,
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    const inference_options_t& options) {
  (void)problem;
  (void)hardware;
  (void)configs;
  (void)options;
  if (handle < 0 || nn::model_info(handle) == nullptr) return std::nullopt;
  // No full-ranking backend (an IRecommender for inference_mode_t::nn) is built in yet.
  return std::nullopt;
}

bool try_rerank_with_model(model_handle_t handle,
                           const problem_t& problem,
                           const hardware_t& hardware,
                           std::vector<prediction_result_t>& results,
                           const inference_options_t& options) {
  if (options.rerank_source) options.rerank_source->prepare(results);
  if (handle < 0) return false;

  const olc::corrector_t* corrector = olc_payload(handle);
  if (corrector == nullptr) return false;

  OlcReranker reranker(handle, *corrector);
  return reranker.rerank(problem, hardware, results, options);
}

model_handle_t resolve_rerank_handle(const rank_options_t& options) {
  // olc_v1 is the re-rank backend, so auto_select resolves to it.
  if (options.nn_backend != nn_backend_t::auto_select && options.nn_backend != nn_backend_t::olc)
    return invalid_handle;
  if (options.nn_model >= 0) return options.nn_model;
  if (options.library_models && options.library_models->olc >= 0)
    return options.library_models->olc;
  return nn::default_model(backend_id_t::olc_v1);
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
    case nn_backend_t::olc:
      return resolve_rerank_handle(options);
    case nn_backend_t::auto_select:
      if (options.nn_model >= 0) return options.nn_model;
      if (options.library_models) {
        if (options.library_models->tilewright >= 0) return options.library_models->tilewright;
        return options.library_models->embedding_similarity;
      }
      return nn::default_model(backend_id_t::tilewright_v1);
  }
  return invalid_handle;
}

}  // namespace origami::nn::detail
