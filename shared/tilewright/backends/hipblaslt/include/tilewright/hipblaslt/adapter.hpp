// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// tilewright hipBLASLt backend -- the framework-specific glue
// =============================================================================
//
// The tilewright ENGINE (shared/tilewright/include/tilewright, src/tilewright) is generic and
// depends on no GEMM framework. This backend is the hipBLASLt-specific binding:
// it converts the GEMM problem/config/hardware structs that hipBLASLt/TensileLite
// already construct (origami's `problem_t`/`config_t`/`hardware_t`) into tilewright's
// neutral types, calls the tilewright engine, and maps the ranking back into
// `origami::prediction_result_t` so the caller's downstream code is unchanged.
//
// This is the ONLY tilewright translation unit allowed to reference framework
// (origami) headers -- the engine core stays framework-agnostic. Future
// backends live alongside this one under backends/<framework>/.
//
// Header-only: callers (e.g. TensileLite's ProblemPredictionLibrary) include
// this and link `roc::tilewright`. Gating (TENSILE_USE_TILEWRIGHT) is the caller's job.
// =============================================================================

#pragma once

#include "origami/origami.hpp"
#include "origami/types.hpp"

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

#include <limits>
#include <string>
#include <vector>

namespace tilewright {
namespace hipblaslt {

namespace detail {

// tilewright enums mirror origami's declaration order, so these are exact casts.
inline tilewright::DataType to_tilewright(origami::data_type_t dt) {
  return static_cast<tilewright::DataType>(static_cast<int>(dt));
}

inline tilewright::Transpose to_tilewright(origami::transpose_t t) {
  return static_cast<tilewright::Transpose>(static_cast<int>(t));
}

inline tilewright::Problem to_tilewright(const origami::problem_t& p) {
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

inline tilewright::Config to_tilewright(const origami::config_t& c) {
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

inline tilewright::Hardware to_tilewright(const origami::hardware_t& h) {
  tilewright::Hardware mh;
  mh.N_CU                       = h.N_CU;
  mh.lds_capacity               = h.lds_capacity;
  mh.L2_capacity                = h.L2_capacity;
  mh.parallel_mi_cu             = h.parallel_mi_cu;
  mh.mem_bw_per_wg_coefficients = h.mem_bw_per_wg_coefficients;
  return mh;
}

}  // namespace detail

// Resolve (load+register) a per-library model for a Tensile library-logic file
// stem (filename without directory/extension) via the colocated "tilewright_index".
// Returns a handle >= 0, or -1 if no index/stem match (the caller then uses the
// base analytical path -- there is no global fallback model).
inline int load_model_for_logic(const std::string& logic_stem, const std::string& hint_dir = "") {
  return tilewright::load_model_by_index(logic_stem, hint_dir);
}

// Build a tilewright::Config from an origami kernel config. Call once per solution
// (at library load) to build a config list, then pass it to rank_configs.
inline tilewright::Config make_config(const origami::config_t& base) {
  return detail::to_tilewright(base);
}

// Rank pre-built tilewright configs for a problem with the two-tower scorer.
// `tilewright_configs` (built via make_config) and `origami_configs` are
// index-aligned: the former drives scoring, the latter is used only to populate
// the returned prediction_result_t. Returns one result per config: survivors
// first in ascending latency (latency = -score), filtered-out configs last with
// NaN latency -- matching the legacy origami ML runtime's contract.
inline std::vector<origami::prediction_result_t> rank_configs(
    const origami::problem_t& problem,
    const origami::hardware_t& hardware,
    const std::vector<tilewright::Config>& tilewright_configs,
    const std::vector<origami::config_t>& origami_configs,
    std::size_t min_scored = 0) {
  const tilewright::Problem mp  = detail::to_tilewright(problem);
  const tilewright::Hardware mh = detail::to_tilewright(hardware);

  const std::vector<tilewright::Result> res =
      tilewright::rank_configs(mp, mh, tilewright_configs, min_scored);

  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<origami::prediction_result_t> result;
  result.reserve(res.size());
  for (const tilewright::Result& r : res) {
    const double latency = r.scored ? -r.score : kNaN;
    result.push_back(origami::prediction_result_t{latency, origami_configs[r.config_index]});
  }
  return result;
}

// Handle-based overload: rank against the per-library model `handle`. A valid
// model handle (>= 0) is required; callers that lack a per-library model should
// use the analytical path instead of this overload. When `handle` < 0 every
// config is returned unscored (NaN) so the caller falls through to analytical.
inline std::vector<origami::prediction_result_t> rank_configs(
    int handle,
    const origami::problem_t& problem,
    const origami::hardware_t& hardware,
    const std::vector<tilewright::Config>& tilewright_configs,
    const std::vector<origami::config_t>& origami_configs,
    std::size_t min_scored = 0) {
  const tilewright::Problem mp  = detail::to_tilewright(problem);
  const tilewright::Hardware mh = detail::to_tilewright(hardware);

  // tilewright::rank_configs already returns one unscored (NaN) result per config
  // when handle < 0, so no special-casing here -- the result is always
  // index-aligned with the inputs (size == configs).
  const std::vector<tilewright::Result> res =
      tilewright::rank_configs(handle, mp, mh, tilewright_configs, min_scored);

  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<origami::prediction_result_t> result;
  result.reserve(res.size());
  for (const tilewright::Result& r : res) {
    const double latency = r.scored ? -r.score : kNaN;
    result.push_back(origami::prediction_result_t{latency, origami_configs[r.config_index]});
  }
  return result;
}

// Convenience overload: rank straight from origami configs (base params only,
// ML features defaulted). Builds the tilewright config list on the fly.
inline std::vector<origami::prediction_result_t> rank_configs(
    const origami::problem_t& problem,
    const origami::hardware_t& hardware,
    const std::vector<origami::config_t>& configs,
    std::size_t min_scored = 0) {
  std::vector<tilewright::Config> mcfgs;
  mcfgs.reserve(configs.size());
  for (const origami::config_t& c : configs) mcfgs.push_back(detail::to_tilewright(c));
  return rank_configs(problem, hardware, mcfgs, configs, min_scored);
}

}  // namespace hipblaslt
}  // namespace tilewright
