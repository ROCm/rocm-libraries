# RFC 0023: Integration Gauntlet (Exhaustive Integration Test Generation)

- Contributors: Adam Dickin
- **Status**: Planned
- **Status notes**: Proposed. Nothing is merged to `develop`. S1–S5, S7–S9, S11–S13 (S12 for
  hip-kernel-provider only), S15 (measured on gfx950) and the hip-kernel-provider spaces of S17 are
  built on an unmerged branch; [§17.1](#171-stories) gives the state of each story, and each section
  says what is not yet built. S6 (typed decline reasons) is deferred. S10, S14, S16 and S17b (except
  its SDPA backward space) are not started.
- **Last verified**: 2026-11-02 against the implementation branch `users/addickin/gauntlet_domains_align`
  (merged with `develop` `79b4ae8503b`)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Scope](#3-scope)
4. [Design Overview](#4-design-overview)
5. [Space Declarations](#5-space-declarations)
6. [Pairwise Generator](#6-pairwise-generator)
7. [Gauntlet Fuzz Mode](#7-gauntlet-fuzz-mode)
8. [Rejection Probes](#8-rejection-probes)
9. [Outcome Model and Report](#9-outcome-model-and-report)
10. [Catalog Sweep](#10-catalog-sweep)
11. [Declared Alignment](#11-declared-alignment)
12. [Input Value Domains](#12-input-value-domains)
13. [Corpus Intake](#13-corpus-intake)
14. [Oracle Cost and Coverage](#14-oracle-cost-and-coverage)
15. [Tiers](#15-tiers)
16. [Gating an Engine Integration](#16-gating-an-engine-integration)
17. [Story Breakdown and Ticket Reconciliation](#17-story-breakdown-and-ticket-reconciliation)
18. [Alternatives Considered](#18-alternatives-considered)
19. [Risks](#19-risks)
20. [Open Questions](#20-open-questions)
21. [Glossary](#21-glossary)

## 1. Executive Summary

The per-op bundles of [RFC 0022](0022_PerOpTestPlans.md) run in CI. They are sized to fit CI. This
RFC adds the **Integration Gauntlet**: a large, generated, slow gate that runs outside CI. Every new
engine integration must survive it, including integrations that an agent wrote.

The gauntlet passes only when every engine, on every graph it is given, does one of two things: it
**declines the graph**, or it **accepts the graph and produces a correct
result**. It must never accept a graph it cannot handle.

The design has five parts:

- **Space declarations.** Each op declares its axes, legality rules and value domains once, in a
  `space.json` beside a `graph.template.json`. The RFC 0022 plan points at it.
- **Generators.** A seeded pairwise generator writes concrete bundles for the CI levels. A fuzz
  sampler draws random legal samples for the gauntlet. Both read the same declaration.
- **Concrete graphs only.** Every generator writes ordinary template-sweep bundles
  ([RFC 0011](0011_BundleTestFramework.md)). The harness runs only concrete graphs. A fuzz sample
  is a sweep case that was written a moment ago.
- **One outcome per case.** The harness writes one record per case: `PASS`, `DECLINED` (with the
  engine's reason), `WRONG`, `CRASH`, `UNCLAIMED_PASS` or `UNEXERCISED`. Reports give fractions,
  never "green".
- **Catalog and alignment.** A catalog sweep runs every kernel of a descriptor engine, not just the
  one the heuristic picks. Declared-alignment probes place each pointer at exactly the declared
  alignment, and engines read the declared alignment when they accept a graph.

## 2. Problem Statement

The integration suite (`dnn-providers/integration-tests`) runs concrete, checked-in graphs on one
engine at a time, against an oracle. It enforces support claims ([RFC 0015](0015_EngineSupportClaims.md)).
Seven gaps keep it from gating a new engine:

| # | Gap | Evidence | Section |
|---|---|---|---|
| G1 | **No catalog mode.** A plan builds the heuristic's choice only. A descriptor engine can match 10 to 1000 kernels to one graph. Only the top one is ever run. | The harness calls `build_plans()` with its default policy, `HEURISTICS_CHOICE` (`Graph.hpp`) | [§10](#10-catalog-sweep) |
| G2 | **No sweep generator.** Sweeps are captured from C++ tests or written by hand. | `generator: capture-bundles` in sweep metadata; RFC 0022 §2 | [§5](#5-space-declarations), [§6](#6-pairwise-generator) |
| G3 | **Declines are not recorded.** A decline on an unclaimed graph is a `GTEST_SKIP`. A probe that was declined cleanly looks the same as one that never ran. | `IntegrationBundleVerificationHarness::engineDidNotRun()` returns `skipped(...)` | [§9](#9-outcome-model-and-report) |
| G4 | **No corpus intake.** dnn-benchmarking graphs need normalization and deduplication. | `migration-scripts/import_graph.py` handles one graph at a time | [§13](#13-corpus-intake) |
| G5 | **Oracle cost and coverage.** The GPU reference runs one thread per output element. Some ops have no reference. | `gpu-ref/kernels/*` | [§14](#14-oracle-cost-and-coverage) |
| G6 | **No fuzz mode or one-step replay.** Seeds exist for input data only. The only replay is `--gtest_filter`. | `InputFillRecipes` global seed | [§7](#7-gauntlet-fuzz-mode) |
| G7 | **Engines ignore declared alignment.** The backend checks the pointer against the declared alignment at execute. No engine reads the declared alignment when it accepts a graph. | `EnginePluginResourceManager.cpp` execute-time check | [§11](#11-declared-alignment) |

## 3. Scope

### 3.1 In scope

- The space declaration format, and how an RFC 0022 plan declares its exhaustive mode with it.
- The pairwise generator, the fuzz sampler, the rejection-probe library and the corpus intake, as
  producers of concrete bundles.
- The outcome model, the per-case outcome report and fraction-based reporting.
- The catalog sweep and kernel-identity recording.
- Declared-alignment testing and the engine contract for declared alignment.
- Input value-domain control.
- The oracle cost model.
- The tier model (Enablement, Hazard, Gauntlet) and its mapping to the RFC 0022 levels.
- The follow-on story breakdown, and the reconciliation with ALMIOPEN-2470 and ALMIOPEN-2827.

### 3.2 Out of scope

| Item | Where it belongs |
|---|---|
| Running the gauntlet or fuzz mode as PR or nightly CI signal | Not planned. The gauntlet runs on demand. |
| Coverage-guided or mutation fuzzing (libFuzzer, AFL) | Not planned. This is randomized differential testing against an oracle. |
| Writing the space declaration of every op | One story per op, after the op's RFC 0022 plan exists |
| Kernel-internal testing | The team that owns the kernel |
| Fuzzing the descriptor loader and matcher | [RFC 0017](0017_UniversalKernelDescriptor.md), [RFC 0018](0018_UniversalMatchDescriptor.md) |
| Which cases each CI level holds | [RFC 0022](0022_PerOpTestPlans.md) |
| A service that schedules gauntlet runs onto idle machines | Not planned |

## 4. Design Overview

```mermaid
flowchart LR
  PLAN[RFC 0022 plan<br/>exhaustive-mode section] --> SPACE[space.json<br/>+ graph.template.json]
  SPACE --> PW[Pairwise generator]
  SPACE --> FZ[Fuzz sampler]
  SPACE --> PR[Probe library]
  CORPUS[dnn-benchmarking corpus] --> IN[Corpus intake]
  PW --> B[Concrete sweep bundles<br/>sweep.json, purpose per case]
  FZ --> B
  PR --> B
  IN --> B
  B --> H[hipdnn_integration_tests<br/>engine lane]
  H --> OR[Outcome report<br/>JSON Lines]
  CAT[Catalog sweep] --> OR
  OR --> REP[Fractions per engine,<br/>kernel and outcome]
  OR -- WRONG / CRASH --> FRZ[Freeze: import_graph.py<br/>purpose regression]
  FRZ --> B
```

Three rules hold everywhere:

1. **One declaration per op.** The generators, the sampler and the probes read the same
   `space.json`. Nothing keeps a second copy of the legality rules (RFC 0022 §14).
2. **The harness runs concrete graphs only.** A generator writes a bundle; the harness runs it. A
   fuzz sample, a pairwise case and a corpus graph reach the engine by the same path as a quick
   bundle, with the same oracles, tolerances and claim rules.
3. **Every case has an expected outcome and a recorded outcome.** The case carries a `purpose`
   (RFC 0022 §9). The run records what happened. A report compares the two.

## 5. Space Declarations

### 5.1 Location: with the tool, pointing at a sweep template

Spaces ship with the fuzz tool, one file per op, plus one per distinct region of an op
(`<Op>[.<Name>].space.json`):

```
dnn-providers/integration-tests/gauntlet/
  run_gauntlet_fuzz.py
  space.py, sampler.py
  spaces/
    SdpaFwd.space.json              ASM_SDPA region
    SdpaFwd.Gfx950Dense.space.json  rocKE gfx950 dense attention region
    SdpaFwd.Flash2Region, SdpaFwd.Gfx1151, SdpaFwd.Gfx1151Decode, SdpaBackward   (.space.json each)
    RMSNorm.space.json
    Layernorm, LayernormBackward, Batchnorm, BatchnormInference, BatchnormBackward,
    RMSNormBackward, ResampleFwd   (.space.json each)
    Matmul.Align.space.json         probe-only: one fixed graph, alignment and placement only
    SdpaFwd.Flash2.space.json       probe-only: one fixed fp16 graph for HIP_FLASH2
    Matmul.Stride, SdpaFwd.AsmStride, SdpaFwd.Gfx950DenseStride   padded-stride probes (§11.3)
    SdpaFwd.AsmProbe, SdpaBackward.AsmProbe, SdpaFwd.Gfx950DenseProbe, SdpaFwd.Gfx1151DecodeProbe, SdpaFwd.Gfx1151Probe, SdpaFwd.Flash2Probe,
    Layernorm.MlopsProbe, LayernormBackward.MlopsProbe, RMSNorm.MlopsProbe,
    Batchnorm.MlopsProbe, BatchnormBackward.MlopsProbe, Matmul.Probe   rejection probes (§8)
```

A **probe-only space** has no shape axes. It holds one fixed graph and varies only the alignment and
placement axes (§11), to show where an engine's alignment boundary is. A **rejection probe** (§8)
draws one `family` per case, each a graph an engine should decline, and a **padded-stride probe**
(§11.3) adds a `pad` axis to a small fixed shape set.

A space does not carry a graph. It names an existing template-sweep topology by path:

```json
"template": "../../integration-test-bundles/quick/SdpaFwd/Default/graph.template.json"
```

One template therefore serves three producers: captured or hand-written sweep rows, generated rows
(pairwise, §6), and fuzz samples (§7). Every producer emits rows in `sweep.json`'s own case format
(`{id, values, metadata}`), so a fuzz sample, a pairwise row and a captured row are the same kind of
object. Bringing a fuzz finding into the suite is appending one row to the `sweep.json` beside the
referenced template. No template is copied, so a space cannot drift from the topology CI runs; a
moved or deleted template is a load error that names the space.

Spaces live outside `integration-test-bundles/`, so bundle discovery and the bundle scripts never
see them and the CI levels never run them. The RFC 0022 plan of the op names its spaces in its
exhaustive-mode section.

### 5.2 Format

```json
{
  "version": 1,
  "operation": "SdpaFwd",
  "template": "../../integration-test-bundles/quick/SdpaFwd/Default/graph.template.json",
  "axes": {
    "dtype":  {"values": ["bfloat16", "half"]},
    "layout": {"values": ["bshd", "bhsd"]},
    "b":      {"values": [1, 2, 3, 4]},
    "hkv":    {"values": [1, 2, 4, 8]},
    "group":  {"values": [1, 2, 4, 8]},
    "sq":     {"range": [1, 2048], "boundaries": [1, 63, 64, 65, 127, 128, 129, 512]},
    "skv":    {"range": [1, 2048], "boundaries": [1, 64, 128, 512, 1024]},
    "d":      {"values": [64, 128, 192]},
    "causal": {"values": [false, true]}
  },
  "derive": {
    "hq":       "hkv * group",
    "dims_q":   "[b, hq, sq, d]",
    "dims_kv":  "[b, hkv, skv, d]",
    "stride_q": "[sq*hq*d, d, hq*d, 1] if layout == 'bshd' else [hq*sq*d, sq*d, d, 1]",
    "stride_kv": "[skv*hkv*d, d, hkv*d, 1] if layout == 'bshd' else [hkv*skv*d, skv*d, d, 1]"
  },
  "constraints": [
    {"rule": "not causal or sq <= skv", "reason": "bottom-right causal mask needs sq <= skv"},
    {"rule": "b*hq*sq*skv*d <= 2**31", "reason": "GPU reference cost ceiling (section 14)"}
  ],
  "case": {
    "values": {
      "io_data_type": "${dtype}",
      "tensors": [
        {"uid": 0, "dims": "${dims_kv}", "strides": "${stride_kv}", "data_type": "${dtype}"},
        {"uid": 1, "dims": "${dims_q}",  "strides": "${stride_q}",  "data_type": "${dtype}"},
        {"uid": 2, "dims": "${dims_q}",  "strides": "${stride_q}",  "data_type": "${dtype}"},
        {"uid": 3, "dims": "${dims_kv}", "strides": "${stride_kv}", "data_type": "${dtype}"}
      ],
      "attributes": {
        "attributes__attn_scale_value": "${1.0 / d ** 0.5}",
        "attributes__diagonal_alignment": "${'BOTTOM_RIGHT' if causal else 'TOP_LEFT'}",
        "attributes__right_bound": "${0 if causal else None}"
      }
    },
    "metadata": {"purpose": "probe"}
  }
}
```

The driver adds the rest of the case metadata itself: `format_version`, `generator`
(`gauntlet-fuzz`), `seed` (the data seed, in `[0, 2^32)`) and `notes` (space, shape seed and axis
values). The case `id` is `s<shape seed>`.

| Field | Meaning |
|---|---|
| `description`, `engine` | Free text. `engine` names the engine the space targets; it is informational and does not choose the engine of a run. The description says what the space draws and why each constraint exists. |
| `axes` | Each axis is either a `values` list, or an integer `range` `[lo, hi]` with optional `boundaries`. A sampler draws a boundary value with probability ½ and a uniform value otherwise, so powers of two, ±1 and degenerate values appear often. An axis marked `"late": true` draws from a stream of its own (shape seed, attempt, axis name), so an axis added to an existing space does not change the shapes that earlier seeds produced. Late axes come last. |
| `derive` | Values computed from axes, in declaration order. A derived value can use earlier derived values. |
| `constraints` | Rules a sample must meet. A sample that breaks one is discarded and redrawn, up to 5000 draws per seed; a space with no legal draw in that budget is a load error that names each rule and how often it rejected. Each rule has a reason. There are three kinds: legality rules copied from the frontend validation of the op (RFC 0022 §6), the cost ceilings of §14, and accuracy bounds that keep a value domain inside what the oracle can grade (§14). |
| `case` | The sweep case to write: `values`, `metadata` and optional `tensor_patches` (RFC 0011). A string `"${expr}"` is replaced by the value of `expr`, keeping its JSON type. Strings that do not start with `${` and end with `}` are copied. The metadata can carry `inputs` (§12), `validator` (§12), `placement` and `placement_uid` (§11.3). |
| `expect` | Optional. What a probe case must do: `declined` or `declined_or_pass`, or an expression over the sample's axes that renders to one of them (§8). A probe space without `expect` expects `declined_or_pass`; a case whose `purpose` is not `probe` has no expectation. |

Expressions are a small, safe subset of Python: literals (numbers, strings, `True`, `False`,
`None`), lists, arithmetic (`+ - * / // % **`), comparisons, `and`/`or`/`not`, `x if c else y`,
indexing, and the functions `min`, `max`, `len`, `abs`. Names are axes and derived values. Anything
else (calls to other functions, attribute access, comprehensions, lambdas) is a load error. One
evaluator (Python, standard library only) serves every generator.

### 5.3 What a plan declares

The exhaustive-mode section of an RFC 0022 plan lists the op's spaces and, for each, the axes that
are core, the shape ranges, and the value domains that matter. The space file is the machine form of
that section. A plan review checks that the space and the plan agree.

## 6. Pairwise Generator

Built for the fuzz driver (`gauntlet/pairwise.py`, `run_gauntlet_fuzz.py --pairwise`). Completeness
is proven for every space with a varying graph: `unproven` is empty for each of the 14 shape spaces
(the SDPA forward, backward, Flash2 region and gfx1151 spaces and the normalization, batch
normalization and Resample spaces). The cap on the search stays as a fallback. The settled
`illegal` and `dead` sets of every shipped space are pinned (`gauntlet/pairwise_pins.json`, pairwise
seed 1): the exact tuples for the three SDPA forward spaces whose engines run on a device, a count and
a digest for every other space. A change to a space's axes, derived values or constraints shows up as
a re-pin in the same commit. Rows are not yet written into CI level directories.

The pairwise generator reads a space and builds a set of rows in which every value of every axis
meets every value of every other axis at least once (all-pairs), in a row where **both values
change the rendered case** (a value is *live* in a row when moving it to another level of its axis
gives a legal row that renders differently). Pairs where one value is inert do not count.

- **Levels.** A `values` axis contributes its values; a `range` axis contributes its declared
  boundaries inside the range plus one interior value drawn from the seed. Range ends count only
  when declared as boundaries.
- **Construction.** Seeded greedy (AETG-style): start each row from the first uncovered pair, draw
  several completions that fill the remaining axes by uncovered-pair gain, keep the legal one with
  the best gain, and credit only its live pairs. Each row is checked whole against the constraints.
- **Impossible pairs.** A pair no completion covers is settled by an exact search. With the pair
  fixed, the search finds which axes each expression of the space *effectively* reads (the branch of
  `d3 if ndim == 5 else 1` that a fixed `ndim` rules out reads no `d3`; an entry whose `when` is false
  reads none of its body), and takes the closure of those over the constraints, the derived values that
  can fail and the liveness checks. Every other axis cannot change whether the pair is legal or live
  and is filled from a legal base row. Over the relevant axes it runs a depth-first search with
  forward checking: once every axis of a check but one is set, the check cuts that axis's levels at
  once, and the axis with the fewest levels left is set next. The axes left split into the connected
  components of the checks that read two of them. Components share no check, so a row exists exactly
  when every component has a completion, and each is searched alone. The pair is reported `illegal`
  (no legal row holds both values) or `dead` (legal rows exist, but one value is inert in each), never
  dropped. Past the node cap it is `unproven`, and is never counted covered.
- **Repair round.** After the run, pairs that were only ever declined get new rows built from levels
  that did pass (`--pairwise-repair`), so a pair is not counted as reached only through declines.
- **Report.** `summary.json` gains `pair_coverage` per space: `tuples`, `illegal`, `dead`, `drawn`,
  `passed`, `declined_only`, `never_passed`, `repair_rows` and the first uncovered pairs.
- **Determinism.** Same space, same seed, same rows. Row ids are `pw<k>`; rows carry
  `metadata.generator = "gauntlet-pairwise"` and the seed, and replay by name like fuzz samples.
- **Output.** `--generate-only` writes the rows as sweeps without running them. Placing them in a
  level directory, with claims, is the remaining step toward the I1 bundle generator of RFC 0022 §15.
- **Device runs.** On gfx950 (2026-11-02) `--pairwise` on `SdpaFwd/Gfx950Dense` ran 542 rows (376
  `PASS`, 166 `DECLINED`, no `WRONG`) over 3389 tuples with 0 `illegal`, 307 `dead` and 0 `unproven`;
  the 3082 tuples drawn equal tuples − illegal − dead. `SdpaBackward/Default` was all `DECLINED` on
  gfx950 (ASM_SDPA ranks no backward engine there), so that run shows coverage only: 339 rows over 4025
  tuples, 54 `dead`, 3971 drawn.

## 7. Gauntlet Fuzz Mode

### 7.1 What it does

A **fuzz run** draws random legal samples from one or more spaces, writes each sample as a case of a
concrete sweep, runs the sweep through the integration binary on one engine lane, and records an
outcome for each sample. It is the gauntlet's open-ended stream: it reaches three-way interactions
that pairwise does not cover.

The driver is `dnn-providers/integration-tests/gauntlet/run_gauntlet_fuzz.py` (Python 3, standard
library only). It does not change what the binary runs; it only writes bundles and reads outcomes.

### 7.2 Control

Environment variables control a run, so it scales without code or bundle changes. Each has a
command-line flag that overrides it.

| Variable | Flag | Meaning | Default |
|---|---|---|---|
| `HIPDNN_FUZZ_SAMPLES` | `--samples` | Samples per space | 32 |
| `HIPDNN_FUZZ_SEED` | `--seed` | Shape seed of the first sample | random, printed |
| `HIPDNN_FUZZ_DATA_SEED` | `--data-seed` | Input-data seed for every sample | derived from each shape seed |
| `HIPDNN_FUZZ_FILTER` | `--filter` | Regex on `<Op>/<Space>` | all spaces |
| `HIPDNN_FUZZ_TIME_BUDGET` | `--time-budget` | Wall-clock limit, e.g. `30m`, `4h` | none |
| `HIPDNN_FUZZ_CASE_TIMEOUT` | `--case-timeout` | Longest one case may run before it counts as a hang | `10m` |

The binary, test article, engine and test config are flags (`--binary`, `--test-article`,
`--test-engine`, `--test-config`), the same values a CTest lane passes. `--binary-arg` passes any
other binary flag. `--spaces-dir` runs the spaces of another directory, so a space can be edited for
triage without touching the tree. `--work-dir`, `--batch-size` and `--summary` set the output
directory, the samples per binary invocation and the summary path.

Before the first batch the driver lists the binary's tests on an empty bundle root with the run's
flags. A bad engine name, a bad plugin, a missing config or no GPU stops the run with exit status 2
before any sample runs, instead of reporting every sample as a load crash.

### 7.3 Seeds and replay

- Sample *i* of a run has shape seed `S + i`, where `S` is the run's seed. One sample is fully
  determined by its shape seed: the sampler seeds a fresh generator with it.
- The data seed is separate. By default it is derived from the shape seed; `--data-seed` fixes it.
  It becomes the case's `metadata.seed`, which the harness already uses for input fills. A failing
  geometry can run again with new data, and the reverse (as in cuDNN's `random_config.py`).
- The case id is `s<shape seed>`. For every `WRONG` and `CRASH` the driver writes `replay.sh`
  (§7.5), one command of this form:

  ```
  run_gauntlet_fuzz.py --filter '^SdpaFwd/Gfx950Dense$' --seed 913377 --samples 1 --data-seed 4021 ...
  ```

  That command writes the same graph with the same data and runs it alone. Device-filled inputs
  differ from host-filled ones, so the replay uses the fill path of the original run.

### 7.4 Every sample gets an outcome

The driver writes the samples of a batch under `<work>/bundles/` (the `--golden-data-dir` root) and
runs the binary with `--outcome-report` (§9). The harness writes a `started` record before it runs
a case and a `finished` record when it ends.

| What happened | Driver records |
|---|---|
| A `finished` record | Its outcome |
| The binary died with a case `started` and not `finished` | `CRASH` for that case; the rest of the batch runs again |
| A case ran longer than the case timeout | The driver kills the binary; `CRASH` (hang) for that case; the rest runs again |
| A written case is absent from `--gtest_list_tests` (the harness skipped it at load) | `UNEXERCISED` with the load message |
| The binary died before any case started | The batch is split in halves and rerun until the case that kills registration is isolated; it is `UNEXERCISED` (load crash) |

A crash costs one case, not the run. `SIGINT` stops after the current batch and still writes the
report.

### 7.5 Freezing a finding

For each `WRONG` or `CRASH` sample the driver writes `<work>/failures/<case>/` before the run
continues, outside the bundle root so later batches never load it:

- `<case>.json`, the expanded concrete graph, with UIDs already in canonical (by-name) order, so an
  import does not renumber tensors and change their per-tensor input seeds;
- `<case>.meta.json`, its metadata (`format_version`, `seed`, `purpose: regression`,
  `reference_source` naming the space and shape seed);
- `repro.json`: the space and the hashes of the space and template files, the shape seed, the axis
  values, the data seed, the fill path, engine, arch, ROCm and git versions, flags and the
  `HIPDNN_*` environment;
- `outcome.json` (the outcome record) and `log_tail.txt` (the end of the binary's log);
- `replay.sh` (§7.3), and `replay.txt`, which holds the same command, the fill path
  (`HIPDNN_TEST_HOST_INPUT_FILL=1` forces the host fill) and the freeze command below.

`migration-scripts/import_graph.py --graph failures/<case>/<case>.json --bundle-dir
integration-test-bundles --tier <level>` adds it to a checked-in sweep, with duplicate detection. A
tensor alignment that differs from the template is kept as a `tensor_patches` entry, and
`placement_uid` is remapped with the UIDs. The path from finding to regression case is mechanical.

### 7.6 Report

At the end the driver prints, per engine and space, the count of each outcome and the decline
reasons, and writes `summary.json`. For the `alignment`, `placement` and `domain` axes the summary
also counts outcomes per axis value (`axis_outcomes`), which is how an alignment boundary is read.
The exit status is non-zero when any sample is `WRONG` or `CRASH`. `UNEXERCISED` samples are listed
with their reasons; they never count as green.

The driver writes one sweep per input data type, under
`bundles/gauntlet/<Op>/<Space>/<tag>_cases/` (`<tag>` is `fp32`, `fp16` or `bfp16`). The tag is
then part of the GTest suite name (`gauntlet_RMSNorm_Default_fp16_cases.s123`), so the dtype globs
of an engine's TOML tolerances and skips apply to fuzz cases as they do to quick bundles.

### 7.7 Shrinking

`--shrink N` reduces each `WRONG` or `CRASH` sample's axis values (earlier levels of a `values`
axis, smaller values of a `range` axis), up to `N` candidate runs, keeping a candidate only while its
outcome stays the same. The data seed and case id stay, so the engine TOML grades every candidate
alike. The failure folder then carries the shrunk sample next to the original, with its own
`replay.sh`.

### 7.8 Not yet built

- A CTest lane label (`gauntlet`) per provider engine (S16). It must stay out of every level that
  the external YAML catch-alls select. Until then the driver is the entry point.
- A space for any op outside hip-kernel-provider's HIP_MLOPS and SDPA regions (S17b).

## 8. Rejection Probes

Built, as the probe library below and the alignment probes of §11: every space has `alignment` and
`placement` axes, and probe-only spaces (§5.1) fix a graph and vary only those axes.

Probes live only in gauntlet spaces and run only through the driver; their cases are never frozen into a
checked-in bundle and never checked into a CI level. A rejection that CI must enforce is an RFC 0022
`expected_rejection` bundle (I10), which is never executed and fails the lane when an engine accepts
it. Keeping probes out of the bundle tree also keeps them out of the support-matrix render, which
counts every checked-in case.

A **probe** is a legal graph whose expected outcome is "decline, or accept and be
correct" (RFC 0022 §9). Probes test the integration contract: an engine that accepts a graph it
cannot run is the failure the gauntlet exists to catch.

| Probe class | Example |
|---|---|
| Unsupported attribute | SDPA dropout, paging, ALiBi on an engine that has none |
| Awkward combination | Conv with X and W in different dtypes |
| Declared alignment below the kernel's need | A tensor that declares 2-byte alignment (§11) |
| Strided or non-dense | Leading stride not a multiple of the vector width |
| Hazard shape | int32 index overflow; dims not divisible by 4 |

Illegal graphs, which the frontend rejects, are frontend unit tests (RFC 0022 §6), not probes.

**The probe library.** One probe-only space per engine family, in four classes: unsupported
attributes, mixed dtypes, int32 index overflow and dimensions not divisible by 4. A space draws a
`family` per case, each family one hazard on a small fixed graph. Non-dense strides are the
padded-stride probes of §11.3.

| Space | Engine | Families |
|---|---|---|
| `SdpaFwd/AsmProbe` | ASM_SDPA forward | Dropout, ALiBi, padding, sequence-length mode, additive mask, paging, sink, block mask, fp8 scales on a bf16 graph, statistics, a device scale tensor, mixed Q/K/V/O dtypes, fp16/fp32, odd dimensions, overflow |
| `SdpaBackward/AsmProbe` | ASM_SDPA backward | The backward attribute set, statistics dtype and rank, mixed dtypes, grouped-query heads, head sizes the registry lacks, odd dimensions, overflow. gfx942 only: every case is `DECLINED` at the device gate on gfx950 |
| `SdpaFwd/Gfx950DenseProbe` | gfx950 dense attention pack | Attributes the kernel cannot apply, mixed dtypes, head sizes and head counts the catalog does not ship, odd dimensions, scale limits, overflow |
| `SdpaFwd/Gfx1151DecodeProbe` | gfx1151 RDNA WMMA decode engine | Graphs the dispatcher routes to prefill, head sizes outside the decode buckets, attributes the engine binds no operand for, mixed dtypes, bias, statistics and alignment gates, stride gates, bounds past int32, overflow. Authored on a gfx1151 device but not run yet |
| `SdpaFwd/Gfx1151Probe` | gfx1151 RDNA WMMA prefill engine | Graphs the dispatcher routes to decode (short queries, few query CTAs, long keys a mask reduces to few visible keys), head sizes that are not a multiple of 8 or above 256, attributes the engine binds no operand for, mixed dtypes, bias, statistics and alignment gates, stride gates, scale gates, bounds past int32, overflow. Authored on a gfx1151 device but not run yet |
| `SdpaFwd/Flash2Probe` | HIP_FLASH2 | Attribute declines, mixed dtypes, odd dimensions, 32-bit dimension limits. gfx942 only; not run on a device in this work |
| `Layernorm/MlopsProbe`, `LayernormBackward/MlopsProbe`, `RMSNorm/MlopsProbe` | HIP_MLOPS | Mixed dtypes of scale, bias, statistics and output, an io dtype outside float/half/bfloat16, an inconsistent `normalized_dim_count`, ranks 2, 3 and 6, declared alignment 1 and 2, odd sizes, overflow |
| `Batchnorm/MlopsProbe`, `BatchnormBackward/MlopsProbe` | HIP_MLOPS | Non-float scale, bias and statistics, io dtypes outside float/half/bfloat16, `y`, `dy` or `dx` in another dtype than `x`, half running statistics, declared alignment 1 and 2, odd sizes, overflow |
| `Matmul/Probe` | hipBLASLt (MIOpen declines every matmul) | Mixed A/B/C dtypes, transposed operands and result, odd M/N/K, rank 2, batch broadcast and permuted-batch strides, declared alignment below 16, overflow of a dimension, a leading dimension or a batch stride past 32 bits |

**Expectation.** Each case has an `expect`, which the space renders per case from its axes:
`declined` (the engine must not take it) or `declined_or_pass` (it may take it and then must be
correct). A probe space without `expect` expects `declined_or_pass`. `declined` is met by
`DECLINED` only; `declined_or_pass` by `DECLINED`, `PASS` and `UNCLAIMED_PASS`. `WRONG`, `CRASH` and
`UNEXERCISED` always violate it: an accept that nothing verified is no evidence. The driver records
`expect`, `expectation` (`met` or `violated`) and `violation` on each outcome record, and the summary
gives each space `expectations: {met, violated, violations}`. A violation is a finding to triage; it
does not change the exit status, which stays driven by `WRONG` and `CRASH`.

**Accepted above the ceiling.** An overflow probe is larger than the oracle can afford (§14), so it
runs at `enforcement_level: buildable` (RFC 0015). `DECLINED` meets its `declined` expectation. If the
engine accepts and builds it, the binary stops before any comparison and the driver records it
`UNEXERCISED` with the reason `accepted above the oracle ceiling`, a violation: an accepted overflow
probe is never `PASS`.

**Typed reason (deferred).** Today a decline is "not in the ranked list" or a late `EngineNotApplicableError`
message. The outcome report records the message. A deferred story (S6) would add a reason code to the
plugin's decline (an enum in the plugin SDK: `UNSUPPORTED_DTYPE`, `UNSUPPORTED_LAYOUT`,
`UNSUPPORTED_ATTRIBUTE`, `ALIGNMENT`, `SHAPE_LIMIT`, `NO_KERNEL`, `OTHER`), so reports can group
declines without parsing text.

Probes are cheap: a decline costs one query.

## 9. Outcome Model and Report

### 9.1 Outcomes

Each case on each engine gets exactly one outcome:

| Outcome | Meaning | Gate |
|---|---|---|
| `PASS` | Accepted and verified against an oracle. For a `placement: "under"` case (§11.3): the backend rejected the misaligned pointer with `BAD_PARAM`, and nothing was compared | pass |
| `DECLINED` | The engine resolved the support query and is absent from the ranked list, or declined later with `EngineNotApplicableError` | pass for a probe; FAIL for a claim (the run already fails it as `CLAIM_BROKEN`, RFC 0015) |
| `WRONG` | Accepted, ran, and the output did not match the oracle | FAIL |
| `CRASH` | In the ranked list, then failed to build or execute (including a late `NOT_SUPPORTED`), hung, or killed the process | FAIL |
| `UNCLAIMED_PASS` | `PASS` on a claim-bearing graph whose sidecar does not claim this cell | report (the claim is too narrow) |
| `UNEXERCISED` | Nothing verified it: a skip rule, no oracle, an oracle or harness error | report; never green |

### 9.2 Derivation

The outcome is derived from facts the harness already decides once per test
(`VerificationOutcome`, `SupportVerdict`), plus one new field: the **stage** at which a failed or
skipped test stopped.

| Harness facts | Outcome |
|---|---|
| status `PASSED`, depth ≥ required, no other gtest failure, verdict `UNCLAIMED_SUPPORT` | `UNCLAIMED_PASS` |
| status `PASSED`, depth ≥ required, no other gtest failure | `PASS` |
| the support query resolved and the engine under test is absent (status `SKIPPED`, or `FAILED` with stage `claim` and verdict `CLAIM_BROKEN`) | `DECLINED` |
| status `FAILED`, origin `COMPARISON` | `WRONG` |
| status `FAILED`, origin `ENGINE`, stage `build` or `execute` | `CRASH` |
| stage `graph` (`from_binary` failed: the frontend rejected the graph) | `UNEXERCISED` |
| support query unresolved (`QUERY_ERRORED`), origin `ORACLE` or `HARNESS`, a skip rule, unverifiable, a pass with another gtest failure | `UNEXERCISED` |
| `started` with no `finished` | `CRASH` (driver, §7.4) |

Two rules keep the classes honest:

- **Acceptance is the ranked list.** An engine that is in the ranked list and then returns
  `NOT_SUPPORTED` while building plans has accepted the graph and failed it: `CRASH`. Applicability
  must be decided up front. The harness's existing path for a late `EngineNotApplicableError` stays
  a decline.
- **Unknown is not declined.** When the support query does not resolve, nobody knows whether the
  engine declined. That is `UNEXERCISED`, never `DECLINED`.

### 9.3 Report format

`hipdnn_integration_tests --outcome-report <file>` (or `HIPDNN_TEST_OUTCOME_REPORT`) appends JSON
Lines. Each bundle test writes a `started` record before it runs and a `finished` record when it
ends. Records are flushed one at a time, so a crash keeps every earlier line.

```json
{"event":"started","test":"gauntlet_SdpaFwd_Gfx950Dense_bfp16_cases.s913377","bundle":".../sweep.json#s913377"}
{"event":"finished","test":"gauntlet_SdpaFwd_Gfx950Dense_bfp16_cases.s913377","bundle":".../sweep.json#s913377",
 "engine":"ASM_SDPA_ENGINE","arch":"gfx950","platform":"linux","purpose":"probe",
 "outcome":"DECLINED","stage":"query","depth":"not-reached","origin":"none","verifier":"none",
 "reason":"Engine ASM_SDPA_ENGINE does not support this graph (1 output tensor(s), 0 ranked engine(s))"}
```

`purpose` comes from the case metadata (`claim`, `probe`, `regression`; absent when not set). The
loader rejects any other value. This is the tag half of RFC 0022's I2. Probes are never checked in
(§8), so the support-matrix render never sees one. `stage` is where the test stopped
(`graph`, `claim`, `query`, `build`, `execute`, `compare`, `oracle`, `setup`, or `none` for a pass).
A case that sets `inputs`, `validator`, `placement` or `placement_uid` echoes them into its
`finished` record, and a pass that verified a rejection adds `"expected_status":"BAD_PARAM"`. The
record also carries `candidate` when the case forces one (§10.3), and `reference_ms` and
`reference_lane` when a reference graded the case (§14).

### 9.4 Fractions, not green

A report states, per engine and per space or bundle:

> *N of M cases passed; D declined (by reason); W wrong; C crashed; U unexercised (by reason).*

When kernels are recorded (§10.3) the report adds the kernels dispatched per engine. A run whose
cases are all `DECLINED` or `UNEXERCISED` says so; it is not a pass of the engine.

## 10. Catalog Sweep

Built for the engines that enumerate their catalog (the kernel-ingestor engines of
hip-kernel-provider); every other engine is a one-entry catalog. Running every MIOpen solver is not
built (S10).

### 10.1 What exists

The catalog enumeration API already exists for descriptor engines (RFC 0019 §13.2):
`HIPDNN_ATTR_ENGINE_CANDIDATES_EXT` pages the matched catalog of one engine for one graph, and
`hipdnn_frontend::detail::getEngineCandidates()` decodes it into `EngineCandidatePage`. Each
`EngineCandidate` has a stable `id` (the kernel identity), its `kernelFeatures`, and an
`EngineVariant` whose complete knob tuple selects exactly that kernel. Engines that cannot enumerate
answer `NOT_APPLICABLE`, not an empty page. Only the kernel-ingestor engines enumerate. A knob
tuple builds through `create_execution_plan_ext(engineId, knobs)` on the normal compiled-plan
path. `Graph::add_engine_variants()` is not used: it only feeds `autotune()`, and
`create_execution_plans()` refuses after it.

### 10.2 Sweep mode

The sweep lives in the driver, not in a looping binary mode: the driver writes each candidate as a
concrete case, so the harness still runs concrete graphs only and every candidate gets the normal
per-case isolation, crash recovery and failure folder. It costs one listing call per graph.

1. **Listing.** `hipdnn_integration_tests --test-engine E --list-candidates <file>` loads each
   registered case, writes the candidate page of the engine under test (every page of
   `getEngineCandidates`) as JSON Lines, and builds and executes nothing. A record's `status` is
   `ok` (the engine enumerated), `not_enumerable` (the engine cannot enumerate, or does not take the
   graph) or `error` (the graph did not load or the enumeration broke).
2. **Forcing.** A case may carry `metadata.engine_variant = {id, knobs}`. The harness then builds
   exactly that candidate with `create_execution_plan_ext(engineId, knobs)` instead of
   `create_execution_plans()`, and the `finished` record gets `candidate` (§10.3).
3. **Driver.** `run_gauntlet_fuzz.py --catalog` (it needs `--test-engine`; it works with fuzz,
   `--pairwise` and `--row` runs) lists the candidates of each drawn graph and writes one case per
   candidate, `<graph case id>c<k>`, `k` being the candidate's index in the listing. The catalog is
   sorted by kernel id, so `k` is stable. A candidate case has the graph's axes and data seed plus the
   `engine_variant`, so each candidate gets its own graph object, its own NaN-sentinel outputs and its
   own oracle run. A candidate failure's `replay.sh` calls `--row` on a row file that carries the
   `engine_variant`, so a replay forces the same candidate.
4. **Shards.** `--catalog-shard i/n` keeps the candidates whose index satisfies `k % n == i`. Running
   the shards `0/n` .. `(n-1)/n` runs every candidate of every graph exactly once, none lost and none
   repeated, which is what a runtime-compilation engine needs, since every candidate compiles.
5. **Reconciliation.** `summary.json` has a `catalog` block: per graph the catalog size and an outcome
   per candidate id, and per engine the totals. `catalog.reconciled` is true when every selected
   candidate of every graph has exactly one outcome. A graph whose listing ended in `error`, listed no
   candidates, or killed or hung the binary is `UNEXERCISED` with the reason, never dropped; a
   candidate is never skipped.

Statuses map onto §9: a candidate that fails to build or execute is `CRASH`; a numeric mismatch is
`WRONG`. Isolation holds by construction: a case is one graph, its outputs are NaN-filled before the
engine runs, and a reference buffer is never shared between cases.

Engines without enumeration (MIOpen, hipBLASLt) are a one-entry catalog: a `not_enumerable` graph
becomes the case `<graph>c0` with `engine_variant {"id": "default"}`, the engine's own pick. Running
every MIOpen solver needs a provider API for "run solution by ID"; that is a separate story (S10).

On gfx950 (2026-11-02) a smoke sweep of six `SdpaFwd/Gfx950Dense` graphs with the dense pack listed
two graphs with 2 and 3 candidates, each of which passed, and four graphs the pack does not take
(`not_enumerable`, `DECLINED`); `candidate_mismatch` was empty and the run reconciled.

### 10.3 Kernel identity in every run

Built for the engines that log the kernel they select. The SDPA engines of hip-kernel-provider
(ASM_SDPA forward and backward, HIP_FLASH2, the attention packs) emit one `kernel: <engine>
<kernel>` log line per plan. The harness captures these lines per case and the `finished` record
gains `kernels`, the distinct kernels the case ran. The driver's summary counts cases per kernel for
each engine, and `kernel_unreported` counts cases that executed without a kernel line. Engines that
log no kernel (HIP_MLOPS) leave `kernels` absent.

For descriptor engines the catalog sweep (§10.2) forces a candidate, and the `finished` record's
`candidate` field names it: the `id` of the case's `engine_variant`, or `"default"` for the engine's
own pick; the field is absent when the case forces nothing. The driver checks that it equals the id
it forced (a mismatch is `candidate_mismatch`, which fails the run, because the force did not reach
the engine), and, where the record has `kernels`, that the part of each after its last `/` equals
the id too (`kernel_mismatch`, listed and printed, not a failure: a log-format difference is not an
engine bug). A sweep case therefore records exactly which catalog entry ran.

## 11. Declared Alignment

### 11.1 The contract

`TensorAttributes::set_alignment(n)` declares the byte alignment of a tensor's pointer (default 16).
The graph carries it. The backend already enforces it at execute: a pointer less aligned than
declared is `BAD_PARAM` (`EnginePluginResourceManager.cpp`, `executeOpGraph`).

A non-default alignment also raises the minimum engine plugin API version of the graph to 1.3.0
(`K_TENSOR_ATTRIBUTE_ALIGNMENT_MIN_VERSION` in `PluginVersionConstants.hpp`). A plugin below 1.3.0
is never offered such a graph; it is assumed to need 16. The same 1.3.0 floor also admits graphs
with ragged tensors, and a raise from below 1.1.0 also admits override-shape graphs. An engine
whose plugin raises its version to receive non-default alignments must therefore decline ragged
and override-shape graphs explicitly if it does not support them.

The missing half was: no engine read the declared alignment when it decided to accept a graph. A
kernel that assumes 16-byte vector loads accepted a graph that declares 2-byte alignment, and the
harness could not see it because every buffer it allocates is aligned to at least 256 bytes.

### 11.2 Engine fix

Each engine reads each tensor's declared alignment in its applicability check. When the declared
alignment is below what the kernel needs, it declines, or selects a variant that handles it. Until
typed decline reasons exist (S6), the decline carries no `ALIGNMENT` code; the outcome report shows
it as an engine absent from the ranked list.

Built for hip-kernel-provider, which now reports engine plugin API 1.3.0. Each rule below is
confirmed by probes: below the threshold `DECLINED`, at and above it `PASS`, with exact placement.

| Engine | Rule | Source of the need |
|---|---|---|
| HIP_MLOPS BatchNorm (inference, forward training, backward) | Folds the declared alignment into the vector-width or variant choice; declines when no variant fits. Per-channel fp32 tensors need 4 bytes. | Vector loads and fixed 4- and 2-wide accesses |
| HIP_MLOPS LayerNorm, RMSNorm, Resample (forward and backward) | Declines when the declared alignment is below, or not a multiple of, the element size | Found by fuzzing: garbage output at 1 and 2 bytes with exact placement |
| ASM_SDPA forward and backward | Declines below 16 bytes (8 for fp8 K and V) | Attention kernel documentation; the code objects have no source |
| HIP_FLASH2 | Q and O need 8, K and V need 2 | Kernel source |
| Ingestor packs (gfx950 dense attention, gfx1151 WMMA attention and decode) | Q, K, V and O need 16, LSE 4, bias its element size | Attention kernel documentation |

For the ingestor packs the check lives in each pack's native matcher, not in a
`$tensor.alignment` criterion of the match descriptor ([RFC 0018](0018_UniversalMatchDescriptor.md)).
The native matcher already decides applicability for these packs, and one criterion per tensor
would duplicate the rule in every descriptor. A descriptor criterion stays the route for packs
without a native matcher.

MIOpen (API 1.2.0) and hipBLASLt (1.0.0) are not changed. The version gate offers them only
graphs at the default alignment, which they accept, and probes confirm they decline every other
value. They need their own S12 story only if a user needs a non-default alignment on them.

### 11.3 Harness placement

A case asks for placement in its metadata. Placement applies to the engine run only; the reference
run is never placed.

| `metadata.placement` | Engine run's device pointers |
|---|---|
| `default` | Where the allocator put them |
| `exact` | Every device pointer is aligned to its tensor's declared *N* and not to *2N* |
| `under` | The pointer of `metadata.placement_uid` is aligned to *N/2* and not to *N*. The case passes only if the engine accepts the graph and the backend then rejects that uid with `BAD_PARAM` |

The harness copies each input into a shadow buffer at the placed offset, fills each output shadow
with the NaN sentinel and copies outputs back after execute. Runtime pass-by-value scalars,
sub-byte packed inputs and host-only runs are left in place. A `placement_uid` that cannot be
placed makes the case `UNEXERCISED` at stage `setup`.

Every space has an `alignment` axis (1 to 256 bytes, rendered as a `tensor_patches` entry per
device tensor) and a `placement` axis. The tests this gives:

- probes for each declared value from 1 to 256, each with its pointer at exactly that alignment;
- one case per space and sample with a pointer less aligned than declared, which must return
  `BAD_PARAM`.

**The strided cross.** Three padded-stride probe spaces cross the declared alignment and the
placement with a leading stride that is not a multiple of the vector width: `Matmul/Stride`
(hipBLASLt and MIOpen), `SdpaFwd/AsmStride` (ASM_SDPA) and `SdpaFwd/Gfx950DenseStride` (the gfx950
dense pack). Each adds a late `pad` axis (elements added to the leading stride) and a `pad_on` axis
that names the tensors it applies to; strides enter the case only through `derive`, so the shipped
spaces draw what they always did. The harness allocates, fills, places and compares non-packed
tensors: the allocation, the output sentinel and the placement shadows cover the element space (the
offset of the last element plus one), a host fill and the comparison touch the elements and not the
gaps, and unit tests cover gap bytes and the element space of padded layouts.

On gfx950 (2026-11-02) every padded probe was `DECLINED` or `PASS`, none `WRONG`:

| Space | Result |
|---|---|
| `SdpaFwd/AsmStride` | `PASS` for every pad at declared alignment 16 or more; `DECLINED` below |
| `SdpaFwd/Gfx950DenseStride` | `DECLINED` for every pad above 0 |
| `Matmul/Stride` | hipBLASLt `PASS` for every pad at alignment 16, `DECLINED` at the other alignments; MIOpen `DECLINED` everywhere |

The row-stride divisibility gap that was expected in ASM_SDPA did not reproduce, so no engine check
was added. A host fill leaves the gap bytes of a padded tensor as uninitialised memory, while a
device fill writes finite values there, so an engine that wrongly read a gap would give a
non-reproducible `WRONG` on the host path. Poisoning the gaps deterministically is open.

## 12. Input Value Domains

`InputFillRecipes` set a range per op and dtype. They could not exclude zero, and they never
injected non-finite or subnormal values. A case's `metadata.inputs` now selects a fill kind per
tensor uid:

| Kind | Every element is | Use |
|---|---|---|
| `free(lo, hi)` | Uniform over `[lo, hi]` (today's fill) | Default ranges |
| `fixed(value)` | `value` | Constant inputs, degenerate reductions |
| `nonzero(lo, hi)` | A `free` draw that is not zero in the data type | Divisions, normalization |
| `zeros` | Zero | Degenerate reductions |
| `large(scale)` | Magnitude in `[top/2, top]`, `top = min(scale, largest finite)`, random sign | Overflow and accumulation |
| `subnormal` | A subnormal of the data type | Kernels that flush or take shortcuts |
| `special(nan \| inf \| -inf, fraction)` | That fraction of elements non-finite, the rest `free` | Non-finite propagation |

A host fill draws from a per-uid generator, so it is deterministic per seed. The `nonzero`, `zeros`,
`large`, `subnormal` and `special` kinds of float, half, bfloat16 and the fp8 formats are also filled
on the device for inputs of at least 2^14 elements, by hipRTC kernels (`gpu-ref/kernels/common`) that
shape the draws of the filler's rocRAND generator; smaller tensors, other types, sub-byte types and
`fixed` fills use the host. Each kind keeps its value contract on the device, including the planted
non-finite element that space rules rely on. Host and device fills are not bit-identical: a failure
reproduces bit for bit only on the fill path of the run that found it, which the report and
`replay.sh` name, and `HIPDNN_TEST_HOST_INPUT_FILL=1` forces the host fill. The parse is strict: an
unknown kind, an entry with neither `kind` nor `seed`, or a malformed field is a metadata error. A
kind the data type cannot hold (`subnormal` on an integer, `inf` on most fp8 formats) makes the case
`UNEXERCISED`.

Every shape space has a `domain` axis (`default`, `narrow`, `wide`, `positive`, `constant`,
`nonzero`, `zeros`, `large`, `subnormal`, `nan`, `inf`) that renders these recipes for the space's
primary data tensors.

**Non-finite grading.** `metadata.validator: "match_nonfinite"` grades NaN against NaN and Inf
against Inf of the same sign by position, and grades finite elements with the resolved tolerance.
A reference with no finite element fails. With the mask on, engine outputs are prefilled with a
NaN of a distinctive payload, and an element still holding it fails, so an output the engine never
wrote is caught. The mask applies to float, half, bfloat16 and double outputs; other outputs keep
strict grading. It exists on the host only, so it forces host-side comparison.

## 13. Corpus Intake

Not yet built.

The dnn-benchmarking headline, microbench and model bundles become gauntlet input:

1. Unpack each graph and normalize it: canonical UIDs by tensor name, canonical JSON
   (`bundle_utils.canonical_uid_map_by_name`, `canon`).
2. Deduplicate by skeleton hash plus exact expansion (`import_graph.py` already does both for one
   graph). A batch mode imports a tarball.
3. Write them to `gauntlet/corpus/<Op>/`, never to the CI levels. dnn-benchmarking says it is too
   early to gate CI on it.
4. Each graph carries `purpose: probe` and its source in `reference_source`.

A corpus graph that exceeds the oracle ceiling (§14) is recorded `UNEXERCISED`, not dropped.

## 14. Oracle Cost and Coverage

The GPU reference runs one thread per output element and walks the full reduction in that thread. Its
cost grows as outputs × reduction length. A case may spend 60 s on its oracle. Each space states its
ceiling as a cost constraint, and `gauntlet/measure_oracle_cost.py` measures what that ceiling costs:
per space it runs a ladder of explicit rows that sweep the cost expression from small to the ceiling
through the driver's `--row` path (by default 8 rungs, each a quarter of the next), fits
`reference_ms = a + b · cost` by least squares over the rungs whose reference ran, and extrapolates the
cost at which the reference takes 60 s. The numbers come from two fields of the `finished` record
(§9.3): `reference_ms`, the wall time of the reference run that graded the case, and
`reference_lane`, `gpu` or `cpu`, which counts a GPU to CPU fallback as `cpu`. The driver copies both
to `outcomes.jsonl`.

Measured on gfx950 on 2026-11-02, every current ceiling stays:

| Space | Cost driver | Ceiling | Lane | At the ceiling | Cost at 60 s | Fit |
|---|---|---|---|---|---|---|
| `SdpaFwd/Default` (ASM_SDPA) | `b·hq·sq·skv·dqk` | 2^31 | GPU | 13.6 s, extrapolated | 9.6e9 | 2 rungs only (274 ms at 2^17, 276 ms at 2^19); the engine declined the larger rungs |
| `SdpaFwd/Gfx950Dense` | `b·hq·sq·skv·d` | 2^31 | GPU | 458 ms, measured | 7.1e11 | 6 rungs, r² 0.9995 |
| `Batchnorm/Default` | `n·c·hw` | 2^30 | GPU | 184 ms, measured | 3.7e12 | 8 rungs |
| `BatchnormInference/Default` | `n·c·hw` | 2^30 | GPU | 159 ms, measured | 1.7e13 | 8 rungs |
| `BatchnormBackward/Default` | `n·c·hw` | 2^24 | CPU | 446 ms, measured | 2.3e9 | 8 rungs, r² 0.9992 |
| `Layernorm/Default` | `n·c·d·h·w` | 2^30 | GPU | 171 ms, measured | 5.1e12 | 8 rungs |
| `LayernormBackward/Default` | `n·c·d·h·w` | 2^30 | GPU | 359 ms, measured | 3.5e12 | 8 rungs |
| `RMSNorm/Default` | `n·c·d·h·w` | 2^30 | GPU | 163 ms, measured | 8.5e12 | 8 rungs |
| `RMSNormBackward/Default` | `n·c·d·h·w` | 2^30 | GPU | 1.17 s, measured | 7.5e10 | 8 rungs, r² 0.99998 |
| `ResampleFwd/Default`, input | `n·c·d·h·w` | 2^30 | CPU | 2.4 s, extrapolated from 92 ms at 2^25 | 2.7e10 | 6 rungs, r² 0.9998 |
| `ResampleFwd/Default`, pooled work | `n·c·od·oh·ow·wvol` | 2^30 | GPU | 196 ms, extrapolated from 169 ms at 1.1e8 | 2.1e12 | 6 rungs, flat series (r² 0.49) |
| `SdpaBackward/Default` | `b·hq·sq·skv·(d+d)` | 2^29 | CPU | about 24 s (an earlier gfx942 measurement) | about 1.3e9 | linear above 2^26 at about 45 ns per cost unit |

Every ceiling measured on gfx950 predicts at most 13.6 s at the ceiling (the ASM_SDPA figure; every
other row is 2.4 s or less), far under the 60 s budget, so none is raised: a higher one would only
make fuzz cases slower, and the engine limits the spaces cite (the `int` template parameters of the
kernels, work-group limits) are reached first. A GPU-lane case carries a fixed cost of 0.15 to 0.35 s
per case (the constant of the fit). ASM_SDPA, with two measured points, and the two ResampleFwd rows,
extrapolated from rungs below the ceiling, are the least anchored. The `SdpaBackward` row is the
earlier gfx942 measurement (its CPU reference is serial); it cannot be re-measured on gfx950, where
ASM_SDPA backward ranks no engine, and its 2^29 ceiling stays.

**Lane.** The gauntlet's oracle chain takes the GPU reference wherever the GPU graph executor has an
applicable plan for the op and falls back to the CPU reference. That is why the norm spaces ran on the
GPU lane although `ReferenceOpCoverage` lists only convolution and SDPA for the GPU: those lists decide
which checked-in bundles are registered for reference validation, not which lane the gauntlet chain
tries. BatchNorm backward has no GPU plan, so its oracle is the scalar CPU reference, which is why its
ceiling is lower. The lane of Resample depends on the row: the two ladders of `ResampleFwd` ran the
CPU and the GPU lane respectively.

**Not measured.**

| Space | Reason |
|---|---|
| `SdpaFwd/Flash2Region` | HIP_FLASH2 runs on gfx942 only; no such device in the measurement |
| `SdpaFwd/Gfx1151`, `SdpaFwd/Gfx1151Decode` | The engines run on gfx1151 only; no such device in the measurement |
| `SdpaBackward/Default` on gfx950 | ASM_SDPA backward ranks no engine on gfx950, so no reference runs |
| Probe spaces (`*Probe`, `*Stride`, `Matmul/Align`, `SdpaFwd/Flash2`) | They state no cost ceiling: they draw fixed or small graphs, and an overflow probe runs at `buildable` (§8) |
| Convolution, Matmul | No shape space yet (S17b). The starting estimates, 2^33 for `n·k·p·q·c·r·s/g` and for `m·n·k`, stand unmeasured |

Rules:

- A space states its ceiling as a constraint, so a generator never writes a case the oracle cannot
  afford.
- Gauntlet huge-shape and int32-overflow cases above the ceiling run at `enforcement_level:
  buildable` (RFC 0015) or against golden data. They never pass without an oracle.
- An op with no reference is `UNEXERCISED` at `full` until a reference or golden data exists; its
  gauntlet runs at `buildable`.
- When the CPU reference's own rounding error exceeds the tolerance for a value domain (for
  example sequential fp32 sums over large `n·hw` in BatchNorm backward, or bf16 rounding of P before
  P·V in SDPA), the space bounds that domain with an accuracy constraint and cites the reason. Each
  such constraint is a known blind spot. The fix is a per-output tolerance in the engine's TOML:
  `[[validator_overrides]]` accepts `atol` and `rtol` for the `allclose` and
  `allclose_matching_infinities` validators, resolved per tensor label, and those values win over
  `[[tolerance_overrides]]` for the tensor (later entries win among validator entries). The entries
  are scoped to gauntlet suites (`*gauntlet*<Op>*`), so checked-in CI bundles keep the common
  tolerance. They replaced the BatchNorm backward DX bound for half and bfloat16 and the DSCALE and
  DBIAS bound, and the SDPA bound on wide and large V for ASM_SDPA and the gfx950 dense pack (the
  output is graded by `allclose`). Removing the DSCALE and DBIAS bound exposed wrong DSCALE and
  DBIAS from the engine at large `n·hw`, which the bound had hidden; the finding stays open. Two
  bounds that name the CPU reference's own sequential sums stay, because no tolerance can lift them.

## 15. Tiers

| Gauntlet tier | When | Content | RFC 0022 level |
|---|---|---|---|
| **Enablement** | Pre-commit and PR | Pairwise layers 1–4 at small shapes; one executed case per feature × engine; `expected_rejection` bundles for rejections CI must enforce (RFC 0022 I10) | quick, PR |
| **Hazard** | Nightly and release | Adds hazard shapes (except extremes) and more pairwise seeds | nightly, full |
| **Gauntlet** | A new integration, or on demand | Every kernel of the catalog × the levels above × rejection probes × huge and int32-overflow shapes × the corpus × fuzz samples | not a level; days |

Enablement and Hazard are concrete, checked-in bundles produced by the generators. The Gauntlet tier
runs outside CI.

## 16. Gating an Engine Integration

A new engine (or a large change to one, including one an agent wrote) is gated as follows:

1. Its claims and bundles pass every CI level.
2. The gauntlet runs on each claimed arch: the catalog sweep over every space's pairwise set, the
   probe library, and a fuzz run of at least the agreed sample count per space.
3. The report shows **zero `WRONG` and zero `CRASH`**, every `DECLINED` has a reason, and every
   `UNEXERCISED` has a reason a reviewer accepted.
4. The reviewer checks the PR for weakened tolerances, removed probes, narrowed spaces and inputs that
   never reach the route under test (the AITER `validate-kernel-pr` idea). Tolerances live in
   provider TOML and space files, so the diff shows any change.

## 17. Story Breakdown and Ticket Reconciliation

### 17.1 Stories

| # | Story | Depends on | Gap | State |
|---|---|---|---|---|
| S1 | Outcome report (`--outcome-report`, six outcomes, `purpose` metadata) | — | G3 | Built |
| S2 | Space declaration format and expression evaluator | — | G2 | Built |
| S3 | Fuzz driver: sampling, seeds, batches, crash recovery, freeze output, summary | S1, S2 | G6 | Built |
| S4 | First fuzz spaces: SDPA forward, RMSNorm forward | S3 | G6 | Built |
| S5 | Pairwise generator | S2 | G2 | Built; completeness proven for every space with a varying graph, illegal and dead sets pinned (§6) |
| S6 | Typed decline reasons in the plugin SDK | S1 | G3 | Deferred: changes every engine; last in the backlog |
| S7 | Rejection-probe library per op | S2 | G3 | Built: ten probe spaces with per-case `expect`, gauntlet-only; the gfx942-only spaces not run on a device (§8) |
| S8 | Catalog sweep mode | S1 | G1 | Built: `--list-candidates`, `engine_variant` forcing, `--catalog`, `--catalog-shard`, reconciliation (§10) |
| S9 | Kernel identity in outcome records | — | G1 | Built: `kernels` for engines that log their kernel, `candidate` for forced candidates (§10.3) |
| S10 | MIOpen "run solution by ID" provider API | S8 | G1 | Not started |
| S11 | Exact-alignment placement in the harness | — | G7 | Built, including the strided cross (§11.3) |
| S12 | Engines honor declared alignment (one story per engine) | S11 | G7 | hip-kernel-provider built; MIOpen and hipBLASLt gated by version (§11.2) |
| S13 | Input value-domain control | S2 | — | Built, with per-output tolerances (§14) and device fills (§12) |
| S14 | Corpus batch intake | — | G4 | Not started |
| S15 | Oracle cost measurement and per-op ceilings | — | G5 | Built for gfx950: measured, every ceiling kept (§14); Flash2, gfx1151 and SdpaBackward on gfx950 unmeasured |
| S16 | Gauntlet CTest lane label per provider, excluded from every CI level | S3 | G6 | Not started |
| S17a | Spaces for the remaining hip-kernel-provider ops (LayerNorm, BatchNorm, RMSNorm backward, Resample) | S2 | G2 | Built |
| S17b | Spaces for convolution, matmul, pointwise, reduction and SDPA backward | S2, RFC 0022 plans | G2 | SDPA backward space built; others not started |

"Built" means on an unmerged branch. Besides S1–S4, S7–S9, S11–S13, S15 and S17a, the branch carries
shrinking (§7.7) and the SDPA backward space. S6 is deferred; S10, S14, S16 and S17b (except its SDPA
backward space) are not started.

### 17.2 ALMIOPEN-2470 (fuzz mode)

Re-parent under this RFC's epic. Its stories map to S2 (format and seed contract), S3 (generator,
loop, decline classification, artifacts), S4 (first op family), S16 (category and check targets) and
documentation. Its design changes in two ways: sampling happens in a driver that writes concrete
sweeps rather than inside the binary, and failure artifacts go through `import_graph.py`.

### 17.3 ALMIOPEN-2827 (knob catalog sweep)

Re-parent as S8. Its new public catalog attribute (`HIPDNN_ATTR_ENGINE_KNOB_CATALOG_EXT`) is not
needed: the candidate enumeration of RFC 0019 §13.2 already returns the matched catalog with a
kernel identity and an enrollable knob tuple per entry. The sweep mode, its statuses, its isolation
rules and its reconciliation line carry over.

## 18. Alternatives Considered

| Alternative | Why rejected |
|---|---|
| **Sample inside the binary** (register N sampled cases at discovery) | A crash ends the process and loses every later sample. The pairwise generator, which writes checked-in sweeps, would need the same evaluator in C++ and Python. Sampling in a driver keeps one evaluator and leaves the binary running concrete graphs only. |
| **One looping test per fuzz bundle** | Samples are not filterable, a crash ends the loop, and the per-case claim, tolerance and oracle logic would need a second home. |
| **Full cross products** | Too large (RFC 0022 §10). Pairwise plus fuzz plus targeted products covers two-way faults and samples the rest. |
| **A new public knob-catalog API** (ALMIOPEN-2827 as written) | The candidate enumeration API already exists and carries kernel identity. |
| **Negative claims** ("engine X must decline graph Y") | Couples tests to engine limits. A probe's `DECLINED` with a reason is recorded without a contract. |
| **Expression language in C++** | Only the generators evaluate expressions, and they are Python like the rest of the bundle tooling. |

## 19. Risks

| Risk | Mitigation |
|---|---|
| Fuzz noise: declines dominate and hide defects | Outcomes separate `DECLINED` from `WRONG`/`CRASH`; the gate counts only the latter. Spaces bias toward claimed regions with boundary values. |
| Oracle wrong for a sampled feature | Oracle prerequisites per op (RFC 0022 §11); a `WRONG` is triaged against a second oracle before it is filed. |
| Tolerance edges on random data | Same tolerances as the CI levels; a tolerance-only `WRONG` is reported with max error so triage can tell it apart. |
| A space drifts from frontend validation | Illegal samples fail in graph build; the driver counts them as `UNEXERCISED` with the frontend message, which flags the drift. |
| Gauntlet runs leak into CI | Spaces live outside the bundle tree (§5.1); no CTest lane runs the driver until S16, which must prove exclusion. |
| Long runs lose results | Outcome lines are flushed per case; crash recovery reruns the remainder. |
| The plugin API raise for alignment (§11.1) changes which graphs a shipping provider receives: ragged and override-shape graphs reach it too | Every hip-kernel-provider engine declines ragged graphs, and the gfx950 dense pack declines override-shape graphs, explicitly. Graphs at the default alignment reach the provider exactly as before. |
| Accuracy constraints narrow a space to what the oracle can grade, hiding defects in that region | Each one cites its reason (§14); per-output TOML tolerance overrides replace them. |

## 20. Open Questions

1. **Oracle ceilings.** Measured on gfx950 (§14); the gfx942 and gfx1151 spaces and any further arch
   remain unmeasured.
2. **Default sample count** for the engine-integration gate (§16).
3. **Decline reason codes.** Is the §8 enum enough, and does it belong in the plugin ABI or in the
   decline message format?
4. **Kernel identity for non-descriptor engines.** Should MIOpen report its solver ID through the
   same `kernel` field?
5. **Where fuzz findings are triaged.** One Jira component per provider, or a shared gauntlet board?
6. **MIOpen and hipBLASLt alignment.** Should they raise to plugin API 1.3.0 and check the declared
   alignment, or keep receiving only default-alignment graphs?
7. **Non-finite semantics per op.** The HIP Resample maxpool never selects NaN, while the CPU
   reference keeps a leading NaN. Which one is the contract? The `nan` domain avoids the case today.

## 21. Glossary

- **Gauntlet.** The on-demand exhaustive gate of this RFC.
- **Space.** A `space.json` plus `graph.template.json`: an op's axes, derived values, legality and case template.
- **Sample.** One legal point of a space, written as a concrete sweep case.
- **Shape seed, data seed.** The seed that picks a sample's graph; the seed that fills its inputs.
- **Probe, claim, regression.** Case purposes (RFC 0022 §9).
- **Outcome.** One of `PASS`, `DECLINED`, `WRONG`, `CRASH`, `UNCLAIMED_PASS`, `UNEXERCISED` (§9).
- **Catalog.** The kernels a descriptor engine matches to a graph (RFC 0019).
- **Freeze.** Turning a fuzz finding into a checked-in regression case.
