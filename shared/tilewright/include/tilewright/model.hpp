// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// =============================================================================
// tilewright -- public, framework-neutral kernel-recommender API
// =============================================================================
//
// tilewright is the standalone, framework-agnostic GRID-aware ("split-tree,
// per-cell two-tower MLP") GEMM kernel recommender. It depends on NO GEMM
// framework: callers (e.g. a per-framework backend adapter) convert their own
// types into tilewright's neutral types (see tilewright/types.hpp) and call the entry
// points below.
//
// Per-library weights ship co-located with the framework's kernel library and
// are discovered the same way the kernels are (the handle API, keyed by the
// logic-file directory). A process-wide singleton (relative to the loaded
// library) also exists for standalone use; see model.cpp.
// =============================================================================

#pragma once

#include "tilewright/types.hpp"

#include <string>
#include <vector>

namespace tilewright {

// ── singleton API: one process-wide model ──────────────────────────────────

// Load weights from a .bin path (MLREC_v1), replacing any loaded model.
// Returns false on I/O/format error. Callers normally rely on the lazy load.
bool load_weights(const std::string& bin_path);

// True once a model has been loaded (eagerly, lazily, or via load_weights()).
bool weights_loaded();

// Route a problem to its leaf cell index (cells[]), lazily loading weights.
// Returns -1 when no weights are loaded or no trained ancestor cell exists.
int route(const Problem& p);

// Rank candidate configs with the per-cell two-tower scorer. Returns one Result
// per input config: scored survivors first in descending score order (stable,
// so element 0 is the first-max pick), then filtered-out configs (scored=false)
// in input order. `min_scored` requests ranking depth beyond the per-cell
// smart_K whitelist without changing the whitelist tier. See CLAUDE.md.
std::vector<Result> rank_configs(const Problem& p,
                                 const Hardware& hw,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored = 0);

// ── handle API: many models at once, one per library-logic file ─────────────

// Load+register a model from a .bin path. Returns a handle >= 0, or -1 on
// error. Dedups by path (same path -> same handle, no reparse). Thread-safe.
int load_model(const std::string& bin_path);

// Same contract as the singleton rank_configs, against model `handle`. An
// invalid handle (< 0) returns every config unscored (all-NaN fallback).
std::vector<Result> rank_configs(int handle,
                                 const Problem& p,
                                 const Hardware& hw,
                                 const std::vector<Config>& configs,
                                 std::size_t min_scored = 0);

// Resolve a model for a library-logic file stem via a "tilewright_index" colocated
// with the weights. Returns a handle >= 0, or -1 if the index or stem is
// absent. `hint_dir` is checked first (pass the .dat's directory so weights are
// found next to the library); empty falls back to the env + library + cwd +
// /opt/rocm search. See CLAUDE.md for the discovery order.
int load_model_by_index(const std::string& logic_stem, const std::string& hint_dir = "");

}  // namespace tilewright
