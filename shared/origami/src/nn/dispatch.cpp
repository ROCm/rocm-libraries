// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/detail/recommender.hpp"

#include "origami/nn/detail/model_store.hpp"
#include "origami/nn/nn.hpp"
#include "origami/nn/twrec/rank.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace origami::nn::detail {
namespace {

model_handle_t first_valid(model_handle_t a, model_handle_t b, model_handle_t c) {
  if (a >= 0) return a;
  if (b >= 0) return b;
  return c;
}

bool any_scored(const std::vector<prediction_result_t>& results) {
  for (const auto& r : results) {
    if (std::isfinite(r.latency)) return true;
  }
  return false;
}

class TilewrightRecommender final : public IRecommender {
 public:
  explicit TilewrightRecommender(model_handle_t handle) : handle_(handle) {}

  model_info_t info() const override {
    if (const model_info_t* registered = nn::model_info(handle_)) {
      return *registered;
    }
    model_info_t fallback;
    fallback.backend = backend_id_t::tilewright_v1;
    return fallback;
  }

  std::vector<prediction_result_t> rank(const problem_t& problem,
                                        const hardware_t& hardware,
                                        const std::vector<config_t>& configs,
                                        const inference_options_t& options) override {
    const twrec::detail::LoadedModel* model = model_payload(handle_);
    if (model == nullptr) return {};

    const std::vector<twrec::rank_entry_t> tw_results =
        twrec::rank_configs(*model, problem, hardware, configs, options);

    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<prediction_result_t> results;
    results.reserve(tw_results.size());
    for (const twrec::rank_entry_t& r : tw_results) {
      const double latency = r.scored ? -r.score : kNaN;
      results.push_back(prediction_result_t{latency, configs[r.config_index]});
    }
    return results;
  }

 private:
  model_handle_t handle_;
};

}  // namespace

std::optional<std::vector<prediction_result_t>> try_rank_with_model(
    model_handle_t handle,
    const problem_t& problem,
    const hardware_t& hardware,
    const std::vector<config_t>& configs,
    const inference_options_t& options) {
  if (handle < 0) return std::nullopt;

  const model_info_t* info = nn::model_info(handle);
  if (info == nullptr || info->backend != backend_id_t::tilewright_v1) {
    return std::nullopt;
  }

  TilewrightRecommender recommender(handle);
  auto results = recommender.rank(problem, hardware, configs, options);
  if (!any_scored(results)) return std::nullopt;
  return results;
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

rank_options_t resolve_rank_options(rank_options_t options) {
  if (const char* mode = std::getenv("ORIGAMI_INFERENCE_MODE")) {
    if (options.inference == inference_mode_t::analytical) {
      if (std::string(mode) == "nn") {
        options.inference = inference_mode_t::nn;
      } else if (std::string(mode) == "nn_fallback") {
        options.inference = inference_mode_t::nn_fallback;
      }
    }
  }

  if (const char* backend = std::getenv("ORIGAMI_NN_BACKEND")) {
    if (options.nn_backend == nn_backend_t::auto_select) {
      const std::string value(backend);
      if (value == "tilewright") {
        options.nn_backend = nn_backend_t::tilewright;
      } else if (value == "embedding_similarity") {
        options.nn_backend = nn_backend_t::embedding_similarity;
      }
    }
  }
  return options;
}

}  // namespace origami::nn::detail
