// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// mosaic hipBLASLt backend -- the framework-specific glue
// =============================================================================
//
// The mosaic ENGINE (shared/mosaic/include/mosaic, src/mosaic) is generic and
// depends on no GEMM framework. This backend is the hipBLASLt-specific binding:
// it converts the GEMM problem/config/hardware structs that hipBLASLt/TensileLite
// already construct (origami's `problem_t`/`config_t`/`hardware_t`) into mosaic's
// neutral types, calls the mosaic engine, and maps the ranking back into
// `origami::prediction_result_t` so the caller's downstream code is unchanged.
//
// This is the ONLY mosaic translation unit allowed to reference framework
// (origami) headers -- the engine core stays framework-agnostic. Future
// backends live alongside this one under backends/<framework>/.
//
// Header-only: callers (e.g. TensileLite's ProblemPredictionLibrary) include
// this and link `roc::mosaic`. Gating (TENSILE_USE_MOSAIC) is the caller's job.
// =============================================================================

#pragma once

#include "origami/origami.hpp"
#include "origami/types.hpp"

#include "mosaic/model.hpp"
#include "mosaic/types.hpp"

#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace mosaic {
namespace hipblaslt {

namespace detail {

// The mosaic enums mirror origami's enum declaration ORDER, so these are exact
// static_casts.
inline mosaic::DataType to_mosaic(origami::data_type_t dt) {
  return static_cast<mosaic::DataType>(static_cast<int>(dt));
}

inline mosaic::Transpose to_mosaic(origami::transpose_t t) {
  return static_cast<mosaic::Transpose>(static_cast<int>(t));
}

// Read an integer that follows a `<tok><digits>` marker in a Tensile kernel
// name (e.g. "_LRVW8_" -> 8). Returns `dflt` when the token is absent. Mirrors
// the Python pipeline's parse_ml_kernel_params_from_name (lib/dat.py) so the
// deployed C++ features match the trained model byte-for-byte.
inline int kernel_name_int(const std::string& name, const char* tok, int dflt) {
  const std::size_t p = name.find(tok);
  if (p == std::string::npos) return dflt;
  std::size_t i = p + std::char_traits<char>::length(tok);
  if (i >= name.size() || name[i] < '0' || name[i] > '9') return dflt;
  return static_cast<int>(std::strtol(name.c_str() + i, nullptr, 10));
}

inline mosaic::Problem to_mosaic(const origami::problem_t& p) {
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

inline mosaic::Config to_mosaic(const origami::config_t& c) {
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
  // Tensile-derived knobs (carried for completeness; not read by the scorer).
  if (c.has_tensile_params()) {
    const origami::tensile_params_t& t = c.tensile();
    mc.depth_u              = t.depth_u;
    mc.global_split_u       = t.global_split_u;
    mc.local_split_u        = t.local_split_u;
    mc.prefetch_global_read = t.prefetch_global_read;
  }
  return mc;
}

inline mosaic::Hardware to_mosaic(const origami::hardware_t& h) {
  mosaic::Hardware mh;
  mh.N_CU                       = h.N_CU;
  mh.lds_capacity               = h.lds_capacity;
  mh.L2_capacity                = h.L2_capacity;
  mh.parallel_mi_cu             = h.parallel_mi_cu;
  mh.mem_bw_per_wg_coefficients = h.mem_bw_per_wg_coefficients;
  return mh;
}

}  // namespace detail

// True once mosaic has a model loaded (eagerly, lazily, or via load_weights()).
inline bool weights_loaded() { return mosaic::weights_loaded(); }

// Explicitly load mosaic weights from a .bin (normally auto-discovered).
inline bool load_weights(const std::string& bin_path) {
  return mosaic::load_weights(bin_path);
}

// Route a GEMM problem to its mosaic leaf cell index (or -1).
inline int route(const origami::problem_t& problem) {
  return mosaic::route(detail::to_mosaic(problem));
}

// Resolve (load+register) a per-library mosaic model for a Tensile library-logic
// file stem (filename without directory/extension) via the colocated
// "mosaic_index". Returns a handle id >= 0, or -1 if no index/stem match (the
// caller then uses the analytical path -- there is no global fallback model).
inline int load_model_for_logic(const std::string& logic_stem) {
  return mosaic::load_model_by_index(logic_stem);
}

// Route a GEMM problem against a specific model handle. Returns -1 (no pick)
// for an invalid handle (< 0).
inline int route(int handle, const origami::problem_t& p) {
  return handle >= 0 ? mosaic::route(handle, detail::to_mosaic(p)) : -1;
}

// Build a full mosaic::Config (base params + ML features) from a GEMM kernel.
// The base params come from the origami config_t; the ML features come from the
// Tensile kernel name (codegen tokens _NTC/_NTE/_PLR/_LRVW/_LPA/_LPB/_LBSPP*)
// plus the 3 SizeMapping fields hipBLASLt actually carries (NonTemporalD ->
// cache_hints_d, PrefetchGlobalRead, LocalSplitU). The tokens + defaults mirror
// the training pipeline's lib/dat.py so the deployed C++ features match the
// trained model. Call once per solution (at library load) to build a config
// list, then pass it to rank_configs.
inline mosaic::Config make_config(const origami::config_t& base,
                                  const std::string& kernel_name,
                                  int non_temporal_d,
                                  int prefetch_global_read,
                                  int local_split_u) {
  mosaic::Config c = detail::to_mosaic(base);
  c.cache_hints_c         = detail::kernel_name_int(kernel_name, "_NTC", 0);
  c.cache_hints_d         = non_temporal_d;
  c.cache_hints_e         = detail::kernel_name_int(kernel_name, "_NTE", 0);
  c.prefetch_global_read  = prefetch_global_read;
  c.prefetch_local_read   = detail::kernel_name_int(kernel_name, "_PLR", 1);
  c.lds_read_vector_width = detail::kernel_name_int(kernel_name, "_LRVW", 1);
  c.local_split_u         = local_split_u;
  c.lds_pad_a             = detail::kernel_name_int(kernel_name, "_LPA", 0);
  c.lds_pad_b             = detail::kernel_name_int(kernel_name, "_LPB", 0);
  c.lds_buffer_pad_a      = detail::kernel_name_int(kernel_name, "_LBSPPA", 0);
  c.lds_buffer_pad_b      = detail::kernel_name_int(kernel_name, "_LBSPPB", 0);
  return c;
}

// Rank pre-built mosaic configs for a problem with the two-tower scorer.
// `mosaic_configs` (built via make_config) and `origami_configs` are
// index-aligned: the former drives scoring, the latter is used only to populate
// the returned prediction_result_t. Returns one result per config: survivors
// first in ascending latency (latency = -score), filtered-out configs last with
// NaN latency -- matching the legacy origami ML runtime's contract.
inline std::vector<origami::prediction_result_t> rank_configs(
    const origami::problem_t& problem,
    const origami::hardware_t& hardware,
    const std::vector<mosaic::Config>& mosaic_configs,
    const std::vector<origami::config_t>& origami_configs) {
  const mosaic::Problem  mp = detail::to_mosaic(problem);
  const mosaic::Hardware mh = detail::to_mosaic(hardware);

  const std::vector<mosaic::Result> res = mosaic::rank_configs(mp, mh, mosaic_configs);

  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<origami::prediction_result_t> result;
  result.reserve(res.size());
  for (const mosaic::Result& r : res) {
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
    const std::vector<mosaic::Config>& mosaic_configs,
    const std::vector<origami::config_t>& origami_configs) {
  const mosaic::Problem  mp = detail::to_mosaic(problem);
  const mosaic::Hardware mh = detail::to_mosaic(hardware);

  const std::vector<mosaic::Result> res =
      handle >= 0 ? mosaic::rank_configs(handle, mp, mh, mosaic_configs)
                  : std::vector<mosaic::Result>{};

  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  std::vector<origami::prediction_result_t> result;
  result.reserve(res.size());
  for (const mosaic::Result& r : res) {
    const double latency = r.scored ? -r.score : kNaN;
    result.push_back(origami::prediction_result_t{latency, origami_configs[r.config_index]});
  }
  return result;
}

// Convenience overload: rank straight from origami configs (base params only,
// ML features defaulted). Builds the mosaic config list on the fly.
inline std::vector<origami::prediction_result_t> rank_configs(
    const origami::problem_t& problem,
    const origami::hardware_t& hardware,
    const std::vector<origami::config_t>& configs) {
  std::vector<mosaic::Config> mcfgs;
  mcfgs.reserve(configs.size());
  for (const origami::config_t& c : configs) mcfgs.push_back(detail::to_mosaic(c));
  return rank_configs(problem, hardware, mcfgs, configs);
}

}  // namespace hipblaslt
}  // namespace mosaic
