# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this directory.

## Overview

tilewright ranks candidate GEMM kernels for a problem with a trained model: routing to a cell of a split tree over a 96-cell (M, N, K, batch) grid, an LDS gate and feasibility filter, a per-cell kernel whitelist, and a per-cell two-tower MLP score. hipBLASLt's TensileLite calls it (when `TENSILE_USE_TILEWRIGHT` is set) to order a Tensile library's kernels. `README.md` describes the algorithm, the MLREC_v2 and MLREC_v3 file formats, the API and the environment variables; read it before changing behaviour.

tilewright is framework-independent: it must not include Tensile, TensileLite or Origami headers, link Origami, or import the Origami Python module. TensileLite owns the glue between Tensile types and tilewright.

## Repository layout

| Path | Purpose |
|------|---------|
| `include/tilewright/types.hpp`, `model.hpp` | Public API used by TensileLite and the training pipeline. Keep it stable. |
| `src/tilewright/format.cpp` | MLREC_v2 and MLREC_v3 loader (strict validation, CRC-32) and per-cell lazy dequantization |
| `src/tilewright/features.cpp` | Feature catalogs and their registry (v2: query 55, item 12, interaction 37; v3: query 59, item 8 plus the attribute spec, interaction 44; view keys; supported contexts), LDS gate, feasibility rules |
| `src/tilewright/kernels.cpp` | fp32 `linear` / `linear_relu` / `dot` in scalar, AVX2 and AVX-512 variants |
| `src/tilewright/rank.cpp` | Routing, context check, view-key grouping, tiers, whitening, scoring, tie-breaks, attribute validation, `CandidateSet`, `rank_configs` |
| `src/tilewright/registry.cpp` | File loading, canonical-path deduplication, `tilewright_index` (rows per library stem and PCI chip ID), environment knobs |
| `src/tilewright/cache.cpp` | The process-wide cache limit over fp32 cell weights and candidate-set caches (least recently used released first) |
| `python/` | nanobind module `tilewright` (`src/bindings.cpp`) and pytest suite |
| `tests/` | Catch2 suite; `model_writer.*` writes synthetic MLREC_v2 and MLREC_v3 models |
| `training_pipeline/` | Pipeline that trains and writes the models (has its own README) |

## Build and test

```bash
cd shared/tilewright
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTILEWRIGHT_ENABLE_PYTHON=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

- Tests are on by default for a standalone build. Catch2 3 comes from `find_package` or, with `TILEWRIGHT_ENABLE_FETCH=ON` (default), from FetchContent.
- The suite writes its own synthetic models. Real-model tests load every shipped model: hipBLASLt keeps them, with their `tilewright_index` and blocklists, in `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/<arch>/<subtree>/Tilewright/` (`TILEWRIGHT_TEST_WEIGHTS_DIR` names another root).
- Run the suite under sanitizers after touching the loader, caches or threading: configure with `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"` (and separately `-fsanitize=thread`).
- Python bindings: `pip install shared/tilewright/python`, or build with `-DTILEWRIGHT_ENABLE_PYTHON=ON` and run the `tilewright-python` ctest entry.
- The selection-time options `TILEWRIGHT_LEVER_PREFETCH` and `TILEWRIGHT_LEVER_MEMO` are on by default and never change a pick; configure with `-DTILEWRIGHT_LEVER_PREFETCH=OFF` or `-DTILEWRIGHT_LEVER_MEMO=OFF` to build without one (Invariants, Selection-time options).

## Invariants

- **Scores are part of the model contract.** A shipped model must rank identically on every host and compiler. Do not add `-ffast-math`, FP contraction or `-march` flags, do not change the summation order of the kernels, and keep the scalar, AVX2 and AVX-512 variants bitwise identical (the `[kernels]` test checks this). Feature math stays in double and rounds once to float.
- **Feature catalogs are frozen.** Feature formulas, order and names are what the models were trained on; the v2 hash (`feature_catalog_hash()`) is `e7fe4b524851e895`, the v3 hash `b8b9f4a518add96f`, and an MLREC_v3 file's `feature_spec_hash` also covers the v3 feature names and the features of golden inputs under fixed launch plans and execution contexts (`640b5d2e92a9f47f` for the empty spec; `tests/test_v3.cpp` and the pipeline's `tests/test_v3.py` pin it). Changing any of them requires retraining every model of that catalog and a new hash; `training_pipeline/lib/features.py` must stay bit-exact with `features.cpp` (the pipeline's parity tests check it). v3 features use no fitted constants: only instruction facts (MI cycles) stored in the model, and the integration's launch plan.
- **v3 is framework-agnostic.** A v3 kernel is its signature (`mt`, `mi`, `cache_hints_a/b`) plus the attribute spec, and its features read nothing else; never add a framework-specific parameter (vector widths, occupancy, Tensile knobs) to v3 features, the view key or the default spec, which is empty. A spec may only name parameters every GEMM kernel framework defines. Launch behaviour enters only through the `LaunchModel` (`PoolOptions::launch`): `launch_key` (an opaque `u64`; equal signature, spec attributes and key must mean equal plans) splits a view key into launch classes only for a model whose serving flags ask for launch keys (others rank bit-identically to every key being 0 and never ask for one), and `plan` is asked once per fully scored group for its first config in pool order; whitelist hashes and the cascade width stay per view key; the engine holds no framework's launch rule (hipBLASLt's is `training_pipeline/lib/hipblaslt_launch.py`, which TensileLite's adapter must match).
- **The model file is untrusted input.** The loader validates every count against the bytes that remain before allocating, checks the CRC, dims, hash, finiteness (by bit tests), labels, split tree, optional header fields (the trained keys, the CU floor, the provenance) and trailer, and never throws out of the API. Add a negative test in `tests/test_loader.cpp` for every new field or rule.
- **The ranking contract** (tier 1 whitelist, fallback to all feasible kernels, tier 2 only when `min_scored` exceeds tier 1, stable best-first order, unscored kernels last in input order) is relied on by TensileLite and by the training pipeline's evaluation. `CandidateSet::rank` must equal `rank_configs` bit for bit, and `CandidateSet::rank_top(..., k, ...)` the scored results among the first `k` of `rank(..., k, ...)`.
- **Selection-time options** (`TW_LEVER_PREFETCH`, `TW_LEVER_MEMO` in `rank.cpp`, on by default and off when defined to 0, so test them with `#if`, never `#ifdef`; README, Building and testing) never change a result, so never a pick, and with both off the engine compiles exactly as before they existed, so keep their code inside their `#if` blocks. The re-rank memo is keyed on every field of `Problem`, `Hardware` and `ExecutionContext` (`same_ranking_inputs`) and on the candidate set, model, launch model and cell, and replays what `rank_in_cell` writes: a new input of a ranking, or a change to how a whitelist tier is ordered or written, must reach `same_ranking_inputs` or `replay_memo` too. A field added to one of those structs or to `LaunchPlan` (`same_plans`) stops their structured bindings from compiling: compare it there, and give it its own change in the per-field tests of `tests/test_levers.cpp`, which change one compared field at a time. Whatever the CMake options say, the `tilewright-levers` and `tilewright-levers-nocheck` ctest entries run the suite with both options, with and without the memo's plan check (`TW_LEVER_MEMO_CHECK`), `tilewright-levers-off` runs it with neither, and `tilewright-levers-parity` requires `rerank_check` and `memo_check` to hash the shipped models' rankings alike in builds with neither option, each alone and both, and the memo to answer some of them; keep every one when a default changes. A test must read the values the engine compiled: the CMake options define the macros PUBLIC, to 1 or 0, and `model_impl.hpp`, not `rank.cpp`, sets their defaults and the plan check's, so never define one for the engine alone. No test can see the prefetch: time ranking calls with cold caches after changing it. A bench that times repeated rankings of one problem is built without the memo, which can answer every other one of them.
- **View keys.** A pool is grouped by the catalog's view key and each group is filtered and scored once; the result must equal scoring every config on its own, bit for bit and with ties across groups in input order (`tests/test_pool.cpp` compares against `detail::rank_keys` with one group per config). A catalog's view key must contain every `Config` field its features, the feasibility rules or the whitelist read. `index` and `attributes` never change a v2 result; a v3 view key is the signature followed by the spec's attributes, and a v3 pool of a model with launch keys groups by view key plus launch key (`PoolKeys::view_of` maps each group to its view key).
- **Execution contexts.** A model scores only the contexts it was trained for (`FeatureCatalog::supports`; v2: every CU and `Schedule::Default`; v3: the CU-fraction range and schedules its file declares and, when the file has one, at least its CU floor of usable CUs on any device); any other context returns every config unscored. Do not widen this without models trained for the context. Under its declared contexts a v3 model scores every problem, through the nearest trained base cell when its own has none. A file without the CU floor must rank exactly as before the floor existed.
- **Index rows.** `load_model_by_index` reads every row of the stem: its first row naming the device's PCI chip ID (`chip=0x` and one to four hex digits, not all zero) wins, else its first row without one. A row of the stem is malformed when its third field has any other form or it has a fourth field; a malformed row fails the lookup with an error and is never skipped or read as a row without a chip ID. Lines of fewer than two fields are skipped and rows of two fields keep their meaning, as before the chip ID. `training_pipeline/lib/tilewright_index.py` reads rows the same way (its tests compare it with the engine), and the pipeline never drops a malformed row from an index it rewrites: stage06 refuses to deploy into such an index, and stage 7's library keeps another stem's malformed row as it stands.
- **Feasibility rules.** Hard rules must be validity rules (the kernel cannot run, or TensileLite's own predicates reject it): the LDS fit and, for v2, the leading-free-dimension rule (v3 leaves it to the integration, since it reads vector widths). Selection heuristics (small batched one tile, Dot2 M < 3, non-temporal) are soft: v2 keeps its behaviour, v3 applies them only when its serving flags say so.
- **Constant item features.** An item feature with a stored std below 1e-3 whitens to 0 when `|x - mean| > 2 * std` (float arithmetic); every other value, and every query or interaction feature, whitens to `(x - mean) / std` (std below 1e-6 counts as 1). The training pipeline (`training_pipeline/stages/stage05_train.py`, `whiten_features`) applies the identical rule; change both together and re-check that shipped models rank as before.
- **Tie-breaks** (`TieBreak::PoolOrder` keeps input order; `TieBreak::Prior` orders equal scores by ascending prior, non-finite last, then input order) only reorder configs with equal scores within a tier.
- **`DataType` values are file-format keys.** Never reorder or insert members. They follow Origami's enumeration, which differs from Tensile's (Tensile has no `Int4`), so callers convert by name.
- **stderr output** is limited to the `TILEWRIGHT_DIAG` and `TILEWRIGHT_PICK_LOG` lines; the training pipeline parses both formats (the pick line's `top1_index` is the pool position of the top config).

## Gotchas

- Environment knobs are read once per process; tests that need them run as separate ctest entries (`tilewright-env-*`).
- MLREC_v1 files are rejected. Convert them with `training_pipeline/scripts/convert_mlrec_v1_to_v2.py`; never edit `.bin` files by hand.
- `load_model` deduplicates by canonical path and returns the cached model while it is alive; use `load_model_from_memory` to get a fresh model from bytes.
- The Python module raises `ValueError` for invalid model data, invalid attribute names and a prior of the wrong length, and `TypeError` for wrong argument types; ranking edge cases (no cell, invalid hardware, unsupported context) return unscored results instead of raising.
- `Config` has a `std::vector<Attribute>` member, so brace-initializing it positionally leaves `attributes` out (and warns under `-Wextra`); assign fields by name.

## License headers

New source files start with the short SPDX header (after a `#!` line, if any):

```cpp
// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
```

```python
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
```

## Formatting

CI runs these on `shared/tilewright/**`: clang-format 18.1.4 with this directory's `.clang-format`, black 25.12.0, cmake-lint with the repository's `.cmake-format.py`, check-yaml, trailing-whitespace, end-of-file-fixer, and `bandit -t B506`.

Comments state constraints the code cannot show. Do not add history notes, change logs or comments that narrate the next line. Committed files must not contain measured performance numbers, hardware specifications of unreleased devices, host names or user paths.
