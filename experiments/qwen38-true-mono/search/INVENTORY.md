# Search, tuning and selection: rocm-libraries vs copy-rocm-lib (inventory)

Scope: every search / tuning / selection / sweep capability of rocKE in `dev/rocm-libraries` (branch
`users/vanantha/rocke/hiptensor-support`, read-only) against `dev/copy-rocm-lib` (`users/vanantha/rocke-monokernels`
working tree), what was copied into copy-rocm-lib (additive only), and what was deferred and why. Paths are relative
to the rocKE root `dnn-providers/hip-kernel-provider/rocke/` unless they start with `experiments/`.

Method: index blob hashes of the whole rocKE subtree in both trees (2,828 vs 2,403 tracked files; 491 missing in
copy-rocm-lib, 445 differ), filtered for search / tune / sweep / select / heuristic / replay / LightGBM names, then a
working-tree pass with CRLF normalized (same result), then a per-file diff of every differing hit (direction and whether
the delta is search-related). The tune-engine harness (`dev/hiptensor-rocke-rfc/tune-engine/`, `BUILD.md`, `OFFLINE.md`,
`LIVE.md`) and the earlier scratch tree it came from (`dev/scratch/rocke-tune`) were checked against rocm-libraries: the
rocm-libraries copy of the engine is the later one (wording-only differences, plus one functional one noted below).

Status words: **same** (identical after CRLF normalization), **missing** (not in copy-rocm-lib), **differs** (both
have it; the column says which side carries what), **copy-only**.

## 1. The search engine (`platform/python/rocke/dispatch/tuning/search/`)

| file | what it does | in copy-rocm-lib |
|---|---|---|
| `__init__.py` | public API | missing -> **copied** |
| `__main__.py` | CLI: `strategies`, `demo`, `replay`, `tune --factory`, `show` | missing -> **copied** |
| `params.py` | `Param` (ordinal / categorical / bool, gated), `SearchSpace` (genomes, keys, distance, moves, repair, stratified sampling, `from_configs` member index, `from_axes`) | missing -> **copied** |
| `knobspace.py` | a dispatcher `KnobSpace` for one request as a search space (`config_key` identity) | missing -> **copied** |
| `trial.py` | `Trial`, `Observation`, `Status`, `Fidelity`, `Budget` / `BudgetMeter` | missing -> **copied** |
| `strategy.py` | ask/tell `Strategy` base (seeded, JSON state, noise-aware gain threshold), `RandomSearch`, `Exhaustive` | missing -> **copied** |
| `local.py` | `HillClimb`: first improvement, combination re-sweep of near-neutral single moves, ordinal two-step, lookback stack, restarts | missing -> **copied** |
| `de.py` | mixed-variable DE: rand/1, current-to-best/1, current-to-pbest/1 + archive; jDE / SHADE | missing -> **copied** |
| `halving.py` | successive halving with racing, Hyperband over a fidelity ladder | missing -> **copied** |
| `stats.py` | signed-rank, paired / Welch t, Holm, `Race`, `NoiseModel` (per-rep and per-call noise) | missing -> **copied** |
| `surrogate.py` | priors (`CallablePrior`, `TablePrior`, `LightGBMPrior`, `PriorScale`), `TPE`, `GPEI` (Matern 5/2, prior mean) | missing -> **copied** |
| `portfolio.py` | discounted Thompson-sampling `Portfolio`, `WarmStart` | missing -> **copied** |
| `measure.py` | `Backend` protocol, `Evaluator` (mandatory numeric check per artifact, spill gate, timeouts, interleaved batches, persistent cache keyed by toolchain + machine), `CallableBackend`, `SyntheticBackend`, `TableBackend` | missing -> **copied** |
| `db.py` | `TunedRecord`, `TuningDB` (locked merge, significance-gated serving, neighbour / other-arch warm starts), `TraceWriter` | missing -> **copied** |
| `driver.py` | `run_search`: budget, stall stop, checkpoint / resume, interleaved confirmation vs the default | missing -> **copied** |
| `api.py` | `make_strategy` (+ `auto`, `FAMILY_DEFAULTS`), `tune`, `tune_candidate`, `search_configs`, `SpecBackend`, `record_result`, `default_ladder` | missing -> **copied** |
| `replay.py` | offline replay of recorded sweep CSVs | missing -> **copied** |
| `synthetic.py` | `gemm_like`, `sphere` test problems | missing -> **copied** |

Functional difference between the two engine copies: the earlier scratch copy also put `ROCKE_CODEOBJ_BACKEND` into the
compile-cache fingerprint (`measure._CODEGEN_ENV`); rocm-libraries dropped it (that branch has no direct code-object
backend). copy-rocm-lib HAS the Rust direct backend, so the copied file is left verbatim and the mono tuner's backends
add `ROCKE_CODEOBJ_BACKEND` / `ROCKE_DIRECT_LEVERS` through `Backend.environment()` (which enters the same fingerprint).
Merge item M5 below.

## 2. Hooks of the engine in existing files (not copied: merge items)

These four files exist in copy-rocm-lib with other (newer) content, so the engine's hooks were NOT written into them.
Each hook is a unified diff under `library/kernels/gfx950/qwen38_mono/tune/vendor/merge/` against copy-rocm-lib's
current file, verified on alola (job 68222892): the patches apply with `patch -p1` to a scratch copy, and then all 303
engine tests, the 4 library attention search tests (23 subtests), the platform dispatch suite (507), the Autotuner tests
(8) and the tuned-family library tests (47) pass. Without them, 6 engine tests and 6 attention search tests fail, all on
the missing hooks (the remaining 297 pass).

| merge item | file | rocm-libraries has | copy-rocm-lib has | patch |
|---|---|---|---|---|
| M1 | `platform/python/rocke/dispatch/core.py` | `KernelCandidate.search_space(request) -> (knob_space, base)` field (+9) | no field | `01-core-search_space.patch` |
| M2 | `platform/python/rocke/dispatch/tuning/candidate.py` | `make_tuned_candidate` provides `search_space` | newer file (`route_auto`, `auto_knobs`, `routed()`), no `search_space` | `02-candidate-search_space.patch` (the hook re-applied onto the newer structure) |
| M3 | `platform/python/rocke/helpers/autotune.py` | `Autotuner(strategy=, budget=, check_fn=, seed=, search_options=)` search mode, `("search", ...)` cache keys | no search mode (strict subset) | `03-autotune-search-mode.patch` (the rocm-libraries file) |
| M4 | `platform/python/rocke/sweep.py` | `search_instances(specs, measure=, check=, strategy=, budget=)`; also a GEMM `levers=` argument of `sweep_instances` | neither | `04-sweep-search_instances.patch` (only `search_instances`; the `levers=` change belongs to the GEMM lane) |
| M5 | `.../tuning/search/measure.py` (copied file) | `_CODEGEN_ENV` without `ROCKE_CODEOBJ_BACKEND` | same, copied verbatim | add `ROCKE_CODEOBJ_BACKEND` and `ROCKE_DIRECT_LEVERS` to `_CODEGEN_ENV` once the direct backend lands in the platform (the mono tuner covers it through `environment()` meanwhile) |
| M6 | `platform/python/rocke/dispatch/tuning/__init__.py` | docstring line pointing at `.search` | docstring lists `.waves` instead | one docstring line, no code |
| M7 | `platform/python/rocke/dispatch/ARCHITECTURE.md` (section 14) and `platform/dsl_docs/autotune/overview.md` ("Search engine", +580 lines, incl. the C++ port design note) | the engine's docs | neither | docs only: copy the sections when M1-M4 land; until then read them in rocm-libraries |

## 3. Learned selection (LightGBM + Origami)

| path | what | in copy-rocm-lib |
|---|---|---|
| `platform/python/rocke/heuristics/` (train, predict, evaluate, feature engines, data pipeline, `search.py`, docs) | LightGBM training / prediction toolkit (GEMM, grouped conv) | **same**, except `gen_gemm_sweep_data.py` differs by 5 lines (GEMM sweep-data generator) |
| `platform/python/rocke/dispatch/gemm/selector.py` + `selector_models/*.model.txt, *.meta.json` (20 files) | the GEMM dispatcher's learned selector: candidate generation, feature vector, a pure-Python LightGBM text-model walker (`TreeEnsemble`, no LightGBM at run time), per-class decision margins | **missing, deferred**: GEMM-specific and wired into existing GEMM files (`common.py`, `plan.py`, `fp16_rcr.py`, `bf16_rcr.py`, `g5_*`, `skinny.py`) that differ in copy-rocm-lib; landing it is the GEMM lane's merge. The mono tuner does not route GEMMs. `TreeEnsemble` is the piece worth reusing for a Python-only selector (merge item M8) |
| `platform/python/rocke/dispatch/gemm/skinny_select.py`, `gemm/autotune.py` | skinny-GEMM config selector, GEMM autotune glue | missing, deferred (same reason) |
| `platform/cpp/instances/common/gemm_dispatch_selector.cpp` | C++ scorer of the GEMM selector | missing, deferred (C++ engine work is out of scope of this lane) |
| `platform/tests/dispatch/dispatch_tests/gemm/test_selector*.py`, `test_rilv_selectors.py`, `platform/tests/instances/test_gemm_r2_nt_tune.py` | their tests | missing, deferred with them |
| `library/tests/test_selector_parity.py` | Python / C++ selector parity | differs (rocm-libraries +711 lines: GEMM selector parity) |
| `platform/cpp/instances/common/attention_unified_selectors.cpp` | attention selector tables | differs (small, both sides; attention lane) |

## 4. Tuned-space machinery, sweeps, autotune

| path | what | in copy-rocm-lib |
|---|---|---|
| `dispatch/tuning/{axes,walk,identity,space,spec,testing}.py` | `KnobAxis`, sweep levels / pruned walk / sampler, `config_key` / `tuning_id`, `KnobSpace`, protocols, conformance kit | **same** |
| `dispatch/tuning/candidate.py` | `make_tuned_candidate`, pin resolution | differs: copy-rocm-lib newer (`route_auto`), rocm-libraries has the search hook (M2) |
| `dispatch/tuning/waves.py` | `WavesPerEuSpace` (waves-per-EU outer knob) | **copy-only** |
| `helpers/autotune.py` | in-process `Autotuner` with disk cache | differs (M3) |
| `sweep.py`, `sweep_bench` | offline cartesian sweeps | differs (M4, GEMM `levers=`) |
| `library/builders/gfx950/gdn/tune.py`, `library/builders/gfx950/gdn/sweep.py`, `library/tests/test_gdn_decode_tune.py`, `test_gdn_decode_sweep.py`, `library/tests/dispatch/gdn/test_tuning_contract.py` | GDN decode tuning | differs both ways; `sweep.py` and the GDN contract test are copy-only (copy-rocm-lib is the newer GDN work) |
| `library/dispatch/attention/tuning_specs.py`, `library/tests/dispatch/attention/test_tuning_space.py` | attention tuned specs | differs (small, attention lane) |
| `library/tests/dispatch/attention/test_tuning_search.py` | attention search tests | missing -> **copied** (needs M1/M2 to pass) |
| `library/benchmarks/common/direct_kernel_sweep.py` | direct-codegen kernel sweep | differs (rocm-libraries +312 lines; not search-engine) |
| `library/benchmarks/common/split_k_cost_fit.py`, MLA live benchmark, `tensor_network_mono/` | benchmark helpers of other lanes | missing, out of scope |
| `platform/python/rocke/benchmark/gemm/fp16_rcr_sweep.py` (+ test) | GEMM sweep driver | differs (GEMM lane) |
| `dispatch/families/tensor_gett_tuned.json` | hipTensor GETT tuned table | missing, out of scope (hipTensor) |
| `platform/dsl_docs/optimization/optimization_runbook.md` | the runbook (Step 0, the 10-step loop, lever catalog) | differs (rocm-libraries +29 / copy +7 lines; read either; the method is the same) |

## 5. Harnesses outside the repos (method sources, not copied)

| where | what | used for |
|---|---|---|
| `hiptensor-rocke-rfc/tune-engine/offline/` (`data.py`, `run.py`, `defaults.py`, `selector_oof.py`) | offline replay of recorded GEMM / conv sweeps, held-out family defaults | the replay design of `tune/replay.py` (round tables, recorded grids) |
| `hiptensor-rocke-rfc/tune-engine/live/` (`live_gemm.py`, `live_permute.py`, `common.py`, `clocks.py`, `jobs/clk_sampler.sh`) | live GPU search harness, per-GPU SCLK / power sampling by PCI BDF | `tune/gate.ClockSampler` (BDF-matched sysfs, amd-smi fallback) |
| `hiptensor-rocke-rfc/tune-engine/fix/` (`mutate_auto.py`, `live_stats.py`, `defaults_cv.py`) | automated mutation testing of the engine, paired statistics, CV of family defaults | deferred: mutation-testing the mono tuner (merge item M9) |
| `hackathon-prep/qwen38/gdn-b3/climb_b3.py` | the B3 in-situ climb with signed lookback (DESIGN 9.2) | `tune/strategies.SignedLookback` (same rules; grid searched model-guided instead of exhaustively) |
| `hackathon-prep/qwen38/sched/ab_sched.py`, `sched_rig.py`, `coz.py`, `blame.py` | in-situ layer A/B with controls, Coz causal profile, blame graph | `tune/gate.py` (rounds, controls, win rule); the RSI loop of the guide |

## 6. What was copied (additive only; every file new in copy-rocm-lib, LF endings)

- `platform/python/rocke/dispatch/tuning/search/` (18 files, verbatim, CRLF -> LF).
- `platform/tests/dispatch/dispatch_tests/tuning/test_search_{edge_cases,failure_handling,integration,replay,space,stats,strategies}.py` (7).
- `library/tests/dispatch/attention/test_tuning_search.py` (1).
- No existing copy-rocm-lib file was modified; every existing build is byte-identical (no emitter, helper or kernel
  file was touched; the new package is imported by nothing that already exists). The `tuning/__init__.py` of
  copy-rocm-lib does not import `.search`, which is reached as a subpackage.

Validation on alola (CPUONLY, `~/ckdsl-venv`): torch-free import OK; 297 / 303 engine tests pass as-is, 303 / 303 with
M1-M4 applied to a scratch copy (job 68222892); the existing qwen38 library tests touching the knob owners
(`test_qwen38_moe.py`, `test_qwen38_layer_framework.py`, `test_qwen38_stage_fn.py`: 342 passed) are unchanged.

## 7. New in copy-rocm-lib (the mono tuner, not a copy)

`library/kernels/gfx950/qwen38_mono/tune/` (`knobs`, `catalog`, `strategies`, `gate`, `objective`, `tdb`, `evaluate`,
`live_toy`, `replay`, `experiments`, `__main__`), `vendor/merge/` (M1-M4), `library/tests/test_qwen38_tune.py` (29 CPU
tests). How to use it: `GUIDE.md` next to this file. The climber's lookback follows PLAN decision 13: it never
prunes a lever or a level from solo results (GUIDE section 6 step 3; validation in 7.4).

## 8. Open merge items (for the WP1 platform owner)

| id | item | action |
|---|---|---|
| M1-M4 | engine hooks in `core.py`, `candidate.py`, `autotune.py`, `sweep.py` | apply `tune/vendor/merge/0[1-4]-*.patch` (verified: section 2) |
| M5 | `ROCKE_CODEOBJ_BACKEND` / `ROCKE_DIRECT_LEVERS` in the engine's cache fingerprint | one-line change in `measure._CODEGEN_ENV` when the direct backend is in the platform |
| M6, M7 | docs (`tuning/__init__.py` docstring, `ARCHITECTURE.md` section 14, `autotune/overview.md` search section) | copy from rocm-libraries with M1-M4 |
| M8 | GEMM learned selector (`gemm/selector.py`, models, C++ scorer, tests) | GEMM lane's merge; `TreeEnsemble` could back the mono `tdb.Selector` where `lightgbm` is not installed |
| M9 | mutation testing of `tune/` (the tune-engine's `mutate_auto.py` method) | run it over the mono tuner once the tracks start using it |
| M10 | toy chain numerics check (`benchmarks/gfx950/qwen38_mono/toy_chain_bench.check_state`, `toy_chain.toy_reference`) | at depth >= 8 (P = 160) a fault confined to an EARLIER launch of a chain vanishes mod 2^32 (each level multiplies `r_prev` by 12880 = 2^4 x 805), so the chained check is blind to it; only the last launch's mailboxes are a real check. Owner's choice: an odd per-level multiplier in the recurrence (changes the reference and the kernel), or compare every launch's mailboxes. The tuner already plants its mutants in the last launch (GUIDE 7.3) |
| M11 | `toy_chain.best_spec(8, 8)` | the live search found faster configurations in two seeds (GUIDE 7.2: no NUMA copies, `step_first`, `poll_max=16`); confirm on a fresh allocation and the other rocke node before updating `best_spec` |
| M12 | `Q38Transport` (one cluster job per round from the login side) | implemented and unit-tested through a fake transport; not yet run against the cluster |

**Status 2026-10-31 (late).** M1-M4 (`git apply --check -R` of `tune/vendor/merge/0[1-4]-*.patch` succeeds) and the
M6 / M7 docs (`tuning/__init__.py` docstring, `ARCHITECTURE.md`, `autotune/overview.md`) are now in the copy-rocm-lib
tree. They were applied there at 11:24-11:41 local time, outside this lane: this lane edits only its own new files.
On that tree, locally (CPU): `test_tuning_search.py` 4 / 4 with 23 subtests, the engine's `tuning/` search tests
279 / 279, `test_qwen38_tune.py` 41 / 41. The mono tuner now also follows PLAN decision 15 (order-dependent
combinations, GUIDE section 6 step 5 and section 7.5).
