---
name: hipdnn-uhd
description: "Generate, evaluate, validate and install hipDNN Universal Heuristic Descriptors (UHDs) for an engine on the local machine: shape sets the local GPU's engine serves (hipdnn_corpus_gen), evaluation slices, collections and stored datasets (uhd_gen), training per role (catalog ranker L2, engine predictor L1), the held-out judgement, Stage V on the local GPU and installation with uhd_gen promote, triage of failed runs, and readying a new engine for training. Use when asked to generate, train, refresh, judge, validate, ship or install heuristics for a hipDNN engine, to manage UHD training data, or to diagnose a UHD run."
argument-hint: "[generate|data|evaluate|validate|triage|onboard] <engine | model dir> [build=<dir>] [store=<dir>]"
allowed-tools: Bash, Read, Write, Grep, Glob
---

# hipDNN UHD (local)

One engine, one device, everything on the local machine: the user's GPU, checkout and build,
with hipDNN's own tools (`hipdnn_corpus_gen`, `hipdnn_bench`, `hipdnn_uhd_features`,
`uhd_gen`) and this skill's `scripts/`. Read the page for the ask; each names the next.

| Ask | Page |
|---|---|
| generate, train or refresh an engine's heuristics | [generate.md](generate.md) |
| what data exists, the device key, the store, the slice; convert measurements; retrain or size from stored data | [data.md](data.md) |
| is a model good enough | [evaluate.md](evaluate.md), with [reference.md](reference.md) for the report's fields |
| validate, ship, promote or install a model | [validate.md](validate.md) |
| a step failed, hangs, or produced a result that looks wrong | [triage.md](triage.md) |
| an engine that has never been trained | [onboarding.md](onboarding.md), with [corpus-facets.md](corpus-facets.md) |

The lifecycle: [onboarding.md](onboarding.md) once per engine → [generate.md](generate.md)
(shape set, slice, collect, store, train; L2 first, then L1 collected under it) →
[evaluate.md](evaluate.md) → [validate.md](validate.md) → install on the user's word.

## Scripts

Standard library plus `uhd_gen`; run as `python3 $UHD/scripts/<script>`, `UHD` being this
skill's installed directory.

| Script | Does |
|---|---|
| [`eval_slice.py`](scripts/eval_slice.py) | a set's evaluation slice; the union `--eval-benchmarks` reads; a regime-even sample; the slice's stored rows |
| [`candidate_tree.py`](scripts/candidate_tree.py) | a real copy of the build's descriptor tree with candidate models installed |
| [`engine_run.py`](scripts/engine_run.py) | used by the three below: one `hipdnn_bench` call that reads only the tree named for it, with an empty cache of its own |
| [`validate_l1.py`](scripts/validate_l1.py) | per graph: measured, incumbent prediction, candidate prediction |
| [`validate_l2.py`](scripts/validate_l2.py) | per graph: the engine run under the build's tree and the candidate tree, with the runtime's refusals |
| [`decide.py`](scripts/decide.py) | the Stage V gates from those rows |

## Rules for every page

- **One device.** Every set, collection, dataset and model is one product of one arch, keyed
  `<arch>-<product>` ([data.md](data.md)). Never merge two devices' rows; with several
  products present, ask which one.
- **Nothing changes the checkout** until the user says to install a validated model; commit
  only when asked, and never push.
- Quote every warning and refusal verbatim in the report.
