// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// mosaic -- public, framework-neutral kernel-recommender API
// =============================================================================
//
// mosaic is the standalone, framework-agnostic GRID-aware ("split-tree,
// per-cell two-tower MLP") GEMM kernel recommender. It depends on NO GEMM
// framework: callers (e.g. a per-framework backend adapter) convert their own
// types into mosaic's neutral types (see mosaic/types.hpp) and call the entry
// points below.
//
// Weights are loaded lazily by a process-wide singleton via the MOSAIC_WEIGHTS
// environment override or auto-discovery relative to the library; see model.cpp.
// =============================================================================

#pragma once

#include "mosaic/types.hpp"

#include <string>
#include <vector>

namespace mosaic {

// Explicitly load weights from a .bin path (MLREC_v6 format). Replaces any
// previously loaded model on success. Returns false on any I/O or format
// error. Normally callers rely on the lazy singleton instead.
bool load_weights(const std::string& bin_path);

// True once a model has been successfully loaded (eagerly, lazily, or via
// load_weights()).
bool weights_loaded();

// Route a problem to its leaf model-cell index, lazily loading weights if
// needed. Returns the cells[] index, or -1 when no weights are loaded or no
// trained ancestor cell exists. (Honors the ML_FORCE_CLUSTER override.)
int route(const Problem& p);

// Rank candidate configs for a problem using the per-cell two-tower scorer.
// Each Config carries its own ML features (cache hints, prefetch, LDS, ...).
// Mirrors the deployed `compute_deployed_top1_picks`: LDS gate ->
// feasibility filter -> optional smart-K signature filter (two-pass with
// fallback) -> two-tower score -> argmax (first-max wins).
//
// The returned vector covers EVERY input config: survivors come first, in
// descending score order (stable, so ties keep input order -> element 0 is the
// first-max pick), each with scored == true; filtered-out configs follow in
// ascending input order with scored == false.
std::vector<Result> rank_configs(const Problem& p, const Hardware& hw,
                                 const std::vector<Config>& configs);

}  // namespace mosaic
