# AI Agent Guidance

This file provides guidance for AI coding agents when working with code in this directory.

## Overview

tilewright ranks candidate GEMM kernels for a problem with a trained model: routing to a cell of a split tree over a 96-cell (M, N, K, batch) grid, an LDS gate and feasibility filter, a per-cell kernel whitelist, and a per-cell two-tower MLP score. hipBLASLt's TensileLite calls it (when `TENSILE_USE_TILEWRIGHT` is set) to order a Tensile library's kernels. `README.md` describes the algorithm, the MLREC_v2 file format, the API and the environment variables; read it before changing behaviour.

tilewright is framework-independent: it must not include Tensile, TensileLite or Origami headers, link Origami, or import the Origami Python module. TensileLite owns the glue between Tensile types and tilewright.

## Repository layout

| Path | Purpose |
|------|---------|
| `include/tilewright/types.hpp`, `model.hpp` | Public API used by TensileLite and the training pipeline. Keep it stable. |
| `src/tilewright/format.cpp` | MLREC_v2 loader (strict validation, CRC-32) and per-cell lazy dequantization |
| `src/tilewright/features.cpp` | Feature catalogs and their registry (only v2: query 55, item 12, interaction 37; view key; supported contexts), LDS gate, feasibility rules |
| `src/tilewright/kernels.cpp` | fp32 `linear` / `linear_relu` / `dot` in scalar, AVX2 and AVX-512 variants |
| `src/tilewright/rank.cpp` | Routing, context check, view-key grouping, tiers, whitening, scoring, tie-breaks, attribute validation, `CandidateSet`, `rank_configs` |
| `src/tilewright/registry.cpp` | File loading, canonical-path deduplication, `tilewright_index`, environment knobs |
| `python/` | nanobind module `tilewright` (`src/bindings.cpp`) and pytest suite |
| `tests/` | Catch2 suite; `model_writer.*` writes synthetic MLREC_v2 models |
| `weights/hipblaslt/<arch>/<arch>/` | Shipped models (`*.tilewright.bin`) and `tilewright_index` files |
| `training_pipeline/` | Pipeline that trains and writes the models (has its own README) |

## Build and test

```bash
cd shared/tilewright
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTILEWRIGHT_ENABLE_PYTHON=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

- Tests are on by default for a standalone build. Catch2 3 comes from `find_package` or, with `TILEWRIGHT_ENABLE_FETCH=ON` (default), from FetchContent.
- The suite writes its own synthetic models. Real-model tests load every shipped `weights/**/*.tilewright.bin`.
- Run the suite under sanitizers after touching the loader, caches or threading: configure with `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"` (and separately `-fsanitize=thread`).
- Python bindings: `pip install shared/tilewright/python`, or build with `-DTILEWRIGHT_ENABLE_PYTHON=ON` and run the `tilewright-python` ctest entry.

## Invariants

- **Scores are part of the model contract.** A shipped model must rank identically on every host and compiler. Do not add `-ffast-math`, FP contraction or `-march` flags, do not change the summation order of the kernels, and keep the scalar, AVX2 and AVX-512 variants bitwise identical (the `[kernels]` test checks this). Feature math stays in double and rounds once to float.
- **The feature catalog is frozen.** Feature formulas, order and names are what the models were trained on; `feature_catalog_hash()` is `e7fe4b524851e895`. Changing any of them requires retraining every model and a new hash.
- **The model file is untrusted input.** The loader validates every count against the bytes that remain before allocating, checks the CRC, dims, hash, finiteness (by bit tests), labels, split tree and trailer, and never throws out of the API. Add a negative test in `tests/test_loader.cpp` for every new field or rule.
- **The ranking contract** (tier 1 whitelist, fallback to all feasible kernels, tier 2 only when `min_scored` exceeds tier 1, stable best-first order, unscored kernels last in input order) is relied on by TensileLite and by the training pipeline's evaluation. `CandidateSet::rank` must equal `rank_configs` bit for bit.
- **View keys.** A pool is grouped by the catalog's view key and each group is filtered and scored once; the result must equal scoring every config on its own, bit for bit and with ties across groups in input order (`tests/test_pool.cpp` compares against `detail::rank_keys` with one group per config). A catalog's view key must contain every `Config` field its features, the feasibility rules or the whitelist read. `index` and `attributes` never change a v2 result.
- **Execution contexts.** A model scores only the contexts it was trained for (`FeatureCatalog::supports`; v2: every CU and `Schedule::Default`); any other context returns every config unscored so that TensileLite falls back to Origami. Do not widen this without models trained for the context.
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
