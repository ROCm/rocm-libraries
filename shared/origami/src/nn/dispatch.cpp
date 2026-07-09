// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "origami/nn/detail/recommender.hpp"

#include "origami/nn/nn.hpp"

#if defined(ORIGAMI_ENABLE_NN_TILEWRIGHT) && ORIGAMI_ENABLE_NN_TILEWRIGHT

#  include "origami/hardware.hpp"
#  include "origami/types.hpp"
#  include "tilewright/model.hpp"
#  include "tilewright/types.hpp"

#  include <cmath>
#  include <cstdlib>
#  include <limits>

namespace origami::nn::detail {
namespace {

tilewright::DataType to_tilewright(data_type_t dt) {
  return static_cast<tilewright::DataType>(static_cast<int>(dt));
}

tilewright::Transpose to_tilewright(transpose_t t) {
  return static_cast<tilewright::Transpose>(static_cast<int>(t));
}

tilewright::Problem to_tilewright(const problem_t& p) {
  tilewright::Problem mp;
  mp.size        = {p.size.m, p.size.n, p.size.k};
  mp.batch       = p.batch;
  mp.a_transpose = to_tilewright(p.a_transpose);
  mp.b_transpose = to_tilewright(p.b_transpose);
  mp.a_dtype     = to_tilewright(p.a_dtype);
  mp.b_dtype     = to_tilewright(p.b_dtype);
  mp.c_dtype     = to_tilewright(p.c_dtype);
  mp.d_dtype     = to_tilewright(p.d_dtype);
  mp.mi_dtype    = to_tilewright(p.mi_dtype);
  return mp;
}

tilewright::Config to_tilewright(const config_t& c) {
  tilewright::Config mc;
  mc.mt            = {c.mt.m, c.mt.n, c.mt.k};
  mc.mi            = {c.mi.m, c.mi.n, c.mi.k};
  mc.occupancy     = c.occupancy;
  mc.cache_hints_a = c.cache_hints_a;
  mc.cache_hints_b = c.cache_hints_b;
  mc.grvw_a        = c.grvw_a;
  mc.grvw_b        = c.grvw_b;
  mc.gwvw_d        = c.gwvw_d;
  mc.index         = c.index;
  return mc;
}

tilewright::Hardware to_tilewright(const hardware_t& h) {
  tilewright::Hardware mh;
  mh.N_CU                       = h.N_CU;
  mh.lds_capacity               = h.lds_capacity;
  mh.L2_capacity                = h.L2_capacity;
  mh.parallel_mi_cu             = h.parallel_mi_cu;
  mh.mem_bw_per_wg_coefficients = h.mem_bw_per_wg_coefficients;
  return mh;
}

bool any_scored(const std::vector<prediction_result_t>& results) {
  for (const auto& r : results) {
    if (std::isfinite(r.latency)) return true;
  }
  return false;
}

model_handle_t first_valid(model_handle_t a, model_handle_t b, model_handle_t c) {
  if (a >= 0) return a;
  if (b >= 0) return b;
  return c;
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
    const tilewright::Problem mp  = to_tilewright(problem);
    const tilewright::Hardware mh = to_tilewright(hardware);

    std::vector<tilewright::Config> tw_configs;
    tw_configs.reserve(configs.size());
    for (const auto& c : configs) {
      tw_configs.push_back(to_tilewright(c));
    }

    const std::vector<tilewright::Result> tw_results =
        tilewright::rank_configs(handle_, mp, mh, tw_configs, options.min_scored);

    const double kNaN = std::numeric_limits<double>::quiet_NaN();
    std::vector<prediction_result_t> results;
    results.reserve(tw_results.size());
    for (const tilewright::Result& r : tw_results) {
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

#else

namespace origami::nn::detail {

std::optional<std::vector<prediction_result_t>> try_rank_with_model(
    model_handle_t,
    const problem_t&,
    const hardware_t&,
    const std::vector<config_t>&,
    const inference_options_t&) {
  return std::nullopt;
}

model_handle_t resolve_model_handle(const rank_options_t&) { return invalid_handle; }

rank_options_t resolve_rank_options(rank_options_t options) { return options; }

}  // namespace origami::nn::detail

#endif
