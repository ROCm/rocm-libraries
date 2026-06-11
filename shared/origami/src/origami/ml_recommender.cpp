// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// origami::ml_recommender -- THIN ADAPTER over the standalone `mosaic` library
// =============================================================================
//
// The MLREC_v6 engine (the reconstructed GRID-aware "split-tree, per-cell
// two-tower MLP" kernel recommender) was extracted verbatim into the
// framework-agnostic `mosaic` library (shared/mosaic). This translation unit
// is now a *thin adapter*: it converts origami's types into mosaic's neutral
// types (field-by-field), calls into mosaic, and converts the results back
// into origami::prediction_result_t.
//
// The inference math, .bin parsing, feature math, whitening, smart-K filter
// and argmax tie-break all live in mosaic/src/mosaic/model.cpp -- unchanged --
// so the C++ pick stays byte-identical to the Python deployed picks.
//
// Symbol contract preserved for the rest of origami / its bindings / its tests
// (see python/src/origami/bindings.cpp, src/origami/origami.cpp,
// tests/test_ml_recommender.cpp):
//   origami::ml_recommender::load_weights / weights_loaded
//   origami::ml_recommender::rank_configs (3-arg and 4-arg)
//   origami::ml_recommender::route_cluster_for_problem / cluster_uses_ml
//   origami::rank_configs (4-arg analytical+ML dispatch)
// =============================================================================

#include "origami/ml_recommender.hpp"

#include "origami/origami.hpp"
#include "origami/types.hpp"

#include "mosaic/model.hpp"
#include "mosaic/types.hpp"

#include <limits>
#include <vector>

namespace origami {
namespace ml_recommender {

namespace {

// ── origami -> mosaic type conversions ─────────────────────────────────────
// The mosaic enums mirror origami's enum declaration ORDER, so the enum
// conversions are exact static_casts. Each struct conversion copies every
// field the mosaic engine (feature builders + routing + feasibility + LDS
// gate) reads, plus a few carried-through knobs.

inline mosaic::DataType to_mosaic(data_type_t dt) {
  return static_cast<mosaic::DataType>(static_cast<int>(dt));
}

inline mosaic::Transpose to_mosaic(transpose_t t) {
  return static_cast<mosaic::Transpose>(static_cast<int>(t));
}

inline mosaic::PredictionMode to_mosaic(prediction_modes_t m) {
  return static_cast<mosaic::PredictionMode>(static_cast<std::uint32_t>(m));
}

mosaic::Problem to_mosaic(const problem_t& p) {
  mosaic::Problem mp;
  mp.size        = {p.size.m, p.size.n, p.size.k};
  mp.batch       = p.batch;
  mp.a_transpose = to_mosaic(p.a_transpose);
  mp.b_transpose = to_mosaic(p.b_transpose);
  mp.a_dtype     = to_mosaic(p.a_dtype);
  mp.b_dtype     = to_mosaic(p.b_dtype);
  mp.c_dtype     = to_mosaic(p.c_dtype);
  mp.d_dtype     = to_mosaic(p.d_dtype);
  mp.mi_dtype    = to_mosaic(p.mi_dtype);
  return mp;
}

mosaic::Config to_mosaic(const config_t& c) {
  mosaic::Config mc;
  mc.mt              = {c.mt.m, c.mt.n, c.mt.k};
  mc.mi              = {c.mi.m, c.mi.n, c.mi.k};
  mc.occupancy       = c.occupancy;
  mc.cache_hints_a   = c.cache_hints_a;
  mc.cache_hints_b   = c.cache_hints_b;
  mc.grvw_a          = c.grvw_a;
  mc.grvw_b          = c.grvw_b;
  mc.gwvw_d          = c.gwvw_d;
  mc.vector_width_a  = c.vector_width_a;
  mc.vector_width_b  = c.vector_width_b;
  mc.index           = c.index;
  mc.prediction_mode = to_mosaic(c.prediction_mode);
  // Tensile-derived knobs (carried for completeness; not read by the scorer).
  if (c.has_tensile_params()) {
    const tensile_params_t& t = c.tensile();
    mc.depth_u              = t.depth_u;
    mc.global_split_u       = t.global_split_u;
    mc.local_split_u        = t.local_split_u;
    mc.prefetch_global_read = t.prefetch_global_read;
  }
  return mc;
}

mosaic::ConfigML to_mosaic(const config_ml_t& m) {
  mosaic::ConfigML mm;
  mm.cache_hints_c         = m.cache_hints_c;
  mm.cache_hints_d         = m.cache_hints_d;
  mm.cache_hints_e         = m.cache_hints_e;
  mm.prefetch_global_read  = m.prefetch_global_read;
  mm.prefetch_local_read   = m.prefetch_local_read;
  mm.lds_read_vector_width = m.lds_read_vector_width;
  mm.local_split_u         = m.local_split_u;
  mm.lds_pad_a             = m.lds_pad_a;
  mm.lds_pad_b             = m.lds_pad_b;
  mm.lds_buffer_pad_a      = m.lds_buffer_pad_a;
  mm.lds_buffer_pad_b      = m.lds_buffer_pad_b;
  return mm;
}

mosaic::Hardware to_mosaic(const hardware_t& h) {
  mosaic::Hardware mh;
  mh.N_CU                      = h.N_CU;
  mh.lds_capacity              = h.lds_capacity;
  mh.L2_capacity               = h.L2_capacity;
  mh.parallel_mi_cu            = h.parallel_mi_cu;
  mh.mem_bw_per_wg_coefficients = h.mem_bw_per_wg_coefficients;
  return mh;
}

}  // namespace

bool load_weights(const std::string& bin_path) {
  return mosaic::load_weights(bin_path);
}

bool weights_loaded() { return mosaic::weights_loaded(); }

bool cluster_uses_ml(int) { return true; }

int route_cluster_for_problem(const problem_t& problem) {
  return mosaic::route(to_mosaic(problem));
}

// ── the deployed ranker -- delegate to mosaic, then map results back ───────
std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs,
                                              const std::vector<config_ml_t>* configs_ml) {
  const mosaic::Problem  mp = to_mosaic(problem);
  const mosaic::Hardware mh = to_mosaic(hardware);

  std::vector<mosaic::Config> mcfgs;
  mcfgs.reserve(configs.size());
  for (const config_t& c : configs) mcfgs.push_back(to_mosaic(c));

  std::vector<mosaic::ConfigML> mml;
  const std::vector<mosaic::ConfigML>* mmlp = nullptr;
  if (configs_ml) {
    mml.reserve(configs_ml->size());
    for (const config_ml_t& m : *configs_ml) mml.push_back(to_mosaic(m));
    mmlp = &mml;
  }

  const std::vector<mosaic::Result> res = mosaic::rank_configs(mp, mh, mcfgs, mmlp);

  // mosaic returns survivors first (descending score, scored == true), then
  // filtered-out configs (scored == false). origami stores latency = -score
  // for survivors and NaN for the rest -- byte-identical to the prior runtime.
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<prediction_result_t> result;
  result.reserve(res.size());
  for (const mosaic::Result& r : res) {
    const double latency = r.scored ? -r.score : kNaN;
    result.push_back(prediction_result_t{latency, configs[r.config_index]});
  }
  return result;
}

// 3-arg overload: no per-config ML features (analytical defaults).
std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs) {
  return origami::ml_recommender::rank_configs(problem, hardware, configs, nullptr);
}

}  // namespace ml_recommender

// ── top-level analytical+ML dispatch (4-arg origami::rank_configs) ─────────
// PredictionLibrary.hpp stamps prediction_mode = ml_recommender on the configs
// when Debug::Instance().useMLRecommender() (TENSILE_ML_RECOMMENDER=1) is true;
// this overload forwards those to the mosaic grid ML path and everything else
// to the analytical 3-arg origami::rank_configs (defined in gemm.cpp).
std::vector<prediction_result_t> rank_configs(const problem_t& problem,
                                              const hardware_t& hardware,
                                              const std::vector<config_t>& configs,
                                              const std::vector<config_ml_t>* configs_ml) {
  bool use_ml = !configs.empty() &&
                configs.front().prediction_mode == prediction_modes_t::ml_recommender;
  if (use_ml) return ml_recommender::rank_configs(problem, hardware, configs, configs_ml);
  return rank_configs(problem, hardware, configs);
}

}  // namespace origami
