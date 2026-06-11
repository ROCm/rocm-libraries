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

// Explicitly load weights from a .bin path (MLREC_v1 format). Replaces any
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

// ── multi-model handle API ─────────────────────────────────────────────────
// The singleton API above is one process-wide model (mosaic_weights.bin). The
// handle API lets a process hold MANY models at once -- one per Tensile
// library-logic file -- each selected by an opaque integer handle. The
// singleton functions are unchanged and remain fully supported for back-compat.

// Load+register a model from a .bin path (MLREC_v1). Returns a handle id >= 0,
// or -1 on any I/O/format error. Dedups by path: loading the same path twice
// returns the same handle (and does not reparse). Thread-safe.
int load_model(const std::string& bin_path);

// True if `handle` refers to a successfully loaded model.
bool model_loaded(int handle);

// Route a problem against the model `handle`. Returns the cells[] index, or -1
// when the handle is invalid or no trained ancestor cell exists.
int route(int handle, const Problem& p);

// Rank candidate configs against the model `handle` (same scoring contract as
// the singleton rank_configs). When `handle` is < 0 or invalid, every config is
// returned unscored (the same all-NaN fallback as "no weights loaded").
std::vector<Result> rank_configs(int handle, const Problem& p, const Hardware& hw,
                                 const std::vector<Config>& configs);

// Resolve a model for a Tensile library-logic file stem (the logic filename
// without directory or extension) via a "mosaic_index" file colocated with the
// weights. Returns a handle id >= 0 on success, or -1 if the index file or the
// stem is absent. The index dir is found with the same discovery used by the
// singleton's lazy auto-load.
int load_model_by_index(const std::string& logic_stem);

}  // namespace mosaic
