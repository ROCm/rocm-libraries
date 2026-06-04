# RFC 0013: hipDNN — Engine Support Claims in Per-Engine TOML

- Contributors: TBD
- **Status**: Draft
- **Implementation Version**: TBD
- **Jira**: [ALMIOPEN-1918](https://amd-hub.atlassian.net/browse/ALMIOPEN-1918)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Regressions Captured](#3-regressions-captured)
4. [Current System Overview](#4-current-system-overview)
5. [TOML Schema](#5-toml-schema)
6. [Failure Detection](#6-failure-detection)
7. [Condensation Heuristic](#7-condensation-heuristic)
8. [Auto-Generation Tool](#8-auto-generation-tool)
9. [Sharding](#9-sharding)
10. [Workflow and CI](#10-workflow-and-ci)
11. [Alternatives Considered](#11-alternatives-considered)
12. [Risks](#12-risks)
13. [Open Questions and Future Work](#13-open-questions-and-future-work)
14. [Documentation](#14-documentation)
15. [Glossary](#15-glossary)

## 1. Executive Summary

This RFC adds **structured engine-support claims** to each per-engine integration-test config. Within this per-engine config, claims are scoped per-asic and live in a machine-managed sidecar (`<EngineName>.supported.toml`) paired with the hand-edited main TOML. Each claim asserts a cross-product of `(op_chain, dtype_combo, layout)` tuples the engine must support on the named arch. The verifier runs after `RUN_ALL_TESTS()` and fails the build when a claimed test loses engine support.

Claims are exact-string lists, not globs — no wildcards. Per-matcher, `op_chains` strings come from the existing `describeGraph()` (extended with per-node `:variant` tags for mode-bearing nodes and `[flag,…]` tags for shape-bearing nodes — see §5.4); `dtype_combos` are TOML inline tables mirroring the support-matrix display (`{io, output?, compute, intermediate?}`); `layouts` are an enumeration. This eliminates an entire class of test-name glob hazards (platform-divergent matchers, char-class collisions, fixture-naming fragility, `TEST_F`/`TYPED_TEST` format misclassification, `DISABLED_` prefix).

The auto-generation tool (`--write-support-claims`, embedded in `hipdnn_integration_tests`, C++-only) observes runtime support and rewrites the sidecar for the current asic. Engineer reviews via `git diff`.

Three principles govern the design and recur throughout this document:

1. **Schema mirrors dispatch.** The matcher key must record what the engine actually dispatches on, not what is convenient to type. Adding a dimension to the matcher key is cheap; recovering from a silent collision is expensive. This is why the dtype dimension uses named-field `dtype_combos` (compute and intermediate are first-class) and why nodes carry variant tags.
2. **Conflicts surface gaps.** The support-conflict refusal in §7 — triggered when the same graph signature is observed as both supported and unsupported in one run — is the only mechanism that discovers the schema is too coarse for the engine's actual behaviour. It is deliberately not softened — its volume is what drives schema extension.
3. **Variants are demand-driven.** Speculative variant tags and combo fields create matcher-set noise — extra rows that look like dispatch-relevant distinctions but aren't. Add them only when an observed partition (or a support conflict) demonstrates the bare node type or combo shape is too coarse.

## 2. Problem Statement

The integration suite already records which engines support which graphs (`SupportMatrixCollector`) and emits a markdown matrix via `--generate-support-matrix`, but the output is advisory — nothing fails if a row flips between runs. When MIOpen's solver coverage shifts and a previously-supported conv shape becomes unsupported on gfx942, the integration test cleanly `GTEST_SKIP`s (no engine supports the graph) and CI passes.

We need a CI gate that fails the moment a supported test stops being supported on the asics we've claimed — without forcing exhaustive TOML maintenance for every other test or asic the engineer isn't actively working on. The existing `[[test_skips]]` mechanism conflates "engine doesn't support" with "engine supports but is broken"; this RFC separates them.

The `--generate-support-matrix` path still exists for ad-hoc inspection, but the preferred way to produce committed matrix docs is the offline renderer (§14): it reads the sidecars directly and needs no hardware run.

## 3. Regressions Captured

This section is the scope statement. Reviewers should anchor here on what the system detects and what it deliberately doesn't.

### 3.1 Detected

| Regression | How it surfaces |
|------------|-----------------|
| A previously-supported test silently loses engine support on a claimed asic. | Observed `(op_chain, dtype_combo, layout)` is in some matcher's cross-product; engine returned no support → **FAIL**. |
| An issue occurs before the test runs. | Test errored before it could record its graph properties → **FAIL** with "errored before record; fix the error first." |
| A matcher is too wide and claims support the engine doesn't have. | Some test in the matcher's cross-product runs and the engine returns no support → **FAIL** pointing at the specific `(op_chain, dtype_combo, layout)` triple. Engineer narrows the matcher or adds a `[[test_skips]]` for known issues. |
| A matcher matches zero observed tests. | Something has gone wrong with the test suite and we are finding no tests for a particular matcher.  If this occurs and is not a bug, we need to revise the matchers for an asic so they are up to date. |
| The engine claims support for a test that no matcher covers, then the test fails. | The engine states it supports a graph then fails verification.  This is the current behaviour of the test suite today. |

**Worked example — the over-claim shape variants exist to express.** Without shape variants, a matmul-backed engine that only supports 1×1 convolution (1×1 conv ≡ batched GEMM) has no way to say so: its only op_chain token is `ConvFprop`, which claims *all* conv shapes. The engine would either over-claim (Rule A failures on the next 3×3 run) or be excluded entirely. Shape-variant tags (`ConvFprop[1x1]`) let it claim exactly the shapes it dispatches. This is principle 1 (schema mirrors dispatch) in concrete form — the shape axis is part of what the engine dispatches on, so it has to be part of the matcher key.

### 3.2 Deliberately not detected

| Non-regression | Rationale |
|----------------|-----------|
| Engine starts supporting a graph no matcher covers. | If we were to fail this case, we will likely blow up CI constantly when we miss updating a particular asic that gets extra support.  For this case, we should log a warning that the claimed support != actual support along with steps to fix the issue  |
| Entire `INSTANTIATE_TEST_SUITE_P` is deleted; remaining tests still satisfy the matcher. | Catalog shrinkage is a code-review concern. The PR that deletes the catalog is the right place to catch it. |
| Asic has no `[[supported]]` block yet. | New-asic bring-up: verifier treats absence as "not enforced." Engineer runs `--write-support-claims` once on the new hardware. |
| Tests pre-skipped via `[[test_skips]]`. | Excluded from claim evaluation. Skip is the explicit "don't enforce" knob. |

The narrowness is the design's load-bearing property. It exchanges symmetric capability tracking for low-maintenance contracts engineers can keep current.

## 4. Current System Overview

### 4.1 What the support matrix already records

`SupportMatrixCollector` is a thread-safe singleton populated by `IntegrationGraphVerificationHarness::verifyGraph`:

```cpp
SupportMatrixCollector::get().recordGraphSupport(
    graph.graph_attributes.get_name(),
    describeGraph(graph),                  // e.g. "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD [io=fp32, compute=fp32]"
    testName,
    status.is_good() ? engineIds : {},
    _testCaseNote,
    _testCaseLayout);
```

`describeGraph()` walks graph nodes and serializes a structured op chain plus dtype tags. The layout is set by fixtures via `setTestCaseLayout(...)`. Recording happens **after** `[[test_skips]]` has short-circuited any pre-skipped tests in `SetUp()`. The structured data this RFC needs is already produced — the implementation will route it through a typed channel (`describeGraphStructured()`) instead of re-parsing the string.

`TestSettings` parses the per-engine TOML via `tomlplusplus`; this RFC extends it with sidecar discovery and `[[supported]]` parsing.

## 5. TOML Schema

The schema spans two paired files per engine. The main file is hand-edited; the sidecar is wholly machine-managed and rewritten in full by `--write-support-claims`. Both are loaded together and unioned at parse time.

### 5.1 Main file (hand-edited)

```toml
# MIOPEN_ENGINE.toml — never touched by --write-support-claims

[meta]
version = 1
engine  = "MIOPEN_ENGINE"        # required when [[supported]] is in use

[[tolerance_overrides]]
filters = ["Full/*BatchnormBackwardCalcStats3d*.Correctness/0"]
atol = 0.002
rtol = 0.012

[[test_skips]]
archs   = ["gfx90a"]
filters = ["*ConvFwdBiasActiv*"]
reason  = "Engine returns wrong results on gfx90a — ROCm/rocm-libraries#6979"
```

`[meta] engine` is checked at load against the loaded plugin's engine name (from `getEngineInfo`). Mismatch → refuse to enforce. The main TOML's `[meta].version` is a separate, unrelated version stream from the sidecar's (§5.3 schema versioning).

### 5.2 Sidecar file (machine-managed)

```toml
# MIOPEN_ENGINE.supported.toml — updates should be made by --write-support-claims.
# Do not hand-edit.

[meta]
version = 1
engine  = "MIOPEN_ENGINE"

[[supported]]
arch     = "gfx1151"
platform = "windows"

# Conv across all directions × all observed shape variants. When support is
# uniform across variants — the common case — the condenser folds every
# variant into one matcher with a long op_chains list, so sidecar size barely
# grows. (Shape variants serialize as bracket-only tags: see §5.4.)
[[supported.matchers]]
op_chains = [
    "ConvFprop", "ConvFprop[1x1]", "ConvFprop[1x1,multi_batch]",
    "ConvFprop[grouped,padding]", "ConvFprop[padding,stride]",
    "ConvFprop[padding,dilation]", "ConvFprop[non_square,padding]",
    "ConvDgrad", "ConvDgrad[1x1]", "ConvWgrad", "ConvWgrad[1x1]",
]
dtype_combos = [
    {io="bf16", compute="fp32", intermediate="bf16"},
    {io="fp16", compute="fp32", intermediate="fp16"},
    {io="fp32", compute="fp32", intermediate="fp32"},
]
layouts = ["NCDHW", "NCHW", "NDHWC", "NHWC"]

# Conv + Bias + Activation (CBA), including a mixed-precision asymmetric combo.
# Mode-bearing Pointwise variants keep the leading colon (Pointwise:MODE).
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:SIGMOID",
]
dtype_combos = [
    {io="fp16", compute="fp32"},
    {io="fp32", compute="fp32"},
    {io="bf16", compute="fp32"},
    {io="fp16", output="fp32", compute="fp32"},   # asymmetric (mixed precision)
]
layouts = ["NCHW", "NHWC"]

# Batchnorm with the multi_batch shape tag (N>1 hits a different solver path).
[[supported.matchers]]
op_chains = [
    "Batchnorm", "Batchnorm[multi_batch]",
    "Batchnorm[multi_batch] + Pointwise:RELU_FWD[upper_clip]",
]
dtype_combos = [{io="fp32", compute="fp32", intermediate="fp32"}]
layouts = ["NCHW", "NHWC"]

# ─── gfx10 (no CK fusion kernels — list only plain Conv here) ─────────────
[[supported]]
arch = "gfx10"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad"]
dtype_combos = [
    {io="fp16", compute="fp32", intermediate="fp16"},
    {io="fp32", compute="fp32", intermediate="fp32"},
    {io="bf16", compute="fp32", intermediate="bf16"},
]
layouts = ["NCHW", "NHWC"]
```

### 5.3 Field semantics

| Field          | Required | Meaning |
|----------------|----------|---------|
| `arch`         | yes      | Exact match against `archTokenOf(gcnArchName)` — the prefix before the first `:` of the raw `gcnArchName` (e.g. `gfx942:sramecc+:xnack-` → `gfx942`). Substring matching is rejected — it would collide families (`"gfx10"` would match `gfx1030`, `gfx1100`). One block per asic. |
| `platform`     | no       | Exact match against `"windows"` or `"linux"`. Default = any. |
| `op_chains`    | yes      | Exact match against `describeGraphStructured(graph).opChain`. See §5.4 for variant-tag rules. |
| `dtype_combos` | yes      | Non-empty array of inline tables with named fields. See §5.4. |
| `layouts`      | yes      | Exact match against the test's `setTestCaseLayout(...)` value. |

A test **matches** a matcher iff its observed `(op_chain, io, output, compute, intermediate, layout)` 6-tuple matches some combo's `(io, effectiveOutput, compute, intermediate)` AND its `op_chain ∈ op_chains` AND its `layout ∈ layouts`. Output dtype is normalized to `io` for symmetric records on both sides (`effectiveOutput = output ?? io`). A test is **claimed** iff it matches ≥1 matcher in the asic's block. No wildcards anywhere — the schema rejects `*` and `?`.

**Schema versioning.** The sidecar carries `[meta] version`, starting at `1`. Any semantic change to the schema — a new `dtype_combos` key, a change to the variant-tag set or flag order, a change to the op_chain serialization rule — bumps the version. A reader refuses a sidecar whose version it does not understand (loud, not silent) rather than mis-parsing it; the engineer regenerates via `--write-support-claims`. The sidecar version is a separate stream from the main TOML's `[meta] version`.

**Loader rules.**

- All required fields (`arch`, `op_chains`, `dtype_combos`, `layouts`) must be present and non-empty.
- All string values are matched exact-string. No `*`, no `?`, no fnmatch.
- `dtype_combos` entries must be inline tables.
- Unknown keys inside a combo are rejected (catches typos / silent schema drift). Unknown *top-level* keys are logged-and-ignored for forward compatibility.
- Duplicate combos (compared by all four fields, with `output` normalized to `io`) are rejected.

### 5.4 Per-node variant tags and `dtype_combos`

Two extensibility points let the schema record what the engine actually dispatches on without inventing new syntax.

**Per-node variant tags in op_chains.** `describeNodeVariant()` returns a stable per-node string when the bare node type isn't enough to partition graphs the engine dispatches differently. Today:

| Node | Tag kind | Tag values | Why |
|------|----------|------------|-----|
| `Pointwise` | mode + `[flags]` | `MODE` then a `[flag,…]` list of the set optional params, emitted in fixed order: `elu_alpha`, `lower_clip`, `lower_slope`, `upper_clip`, `softplus_beta`, `swish_beta`. Example: `RELU_FWD[lower_clip,upper_clip]`. | Different solvers per (mode, params) — plain ReLU / ReLU6 / clamp / leaky-ReLU all use mode `RELU_FWD`. |
| `Reduction` | mode | `ADD`, `MAX`, … when set. | Different solvers per reduction op. |
| `ConvFprop` / `ConvDgrad` / `ConvWgrad` | `[flags]` only | Shape flags emitted in fixed dimension order: `1x1`, `grouped`, `multi_batch`, `non_square`, `padding`, `stride`, `dilation`. Example: `ConvFprop[1x1,multi_batch]`. | Conv engines partition by shape — dedicated 1×1 solver paths, GEMM-backed engines that handle *only* 1×1, engines that skip grouped/dilated. |
| `Batchnorm` / `BatchnormBackward` / `BatchnormInference` / `BatchnormInferenceVarianceExt` | `[flags]` only | `multi_batch` (N>1). Example: `BatchnormInference[multi_batch]`. | N=1 vs N>1 hits different solver paths; the partition axis for single-batch-inference engines. |

**Flag order is fixed, not alphabetical (stability contract).** Both flag lists emit in a fixed order that is part of the frozen op_chain format, *not* alphabetical:

- Conv flags: `1x1, grouped, multi_batch, non_square, padding, stride, dilation`. (Alphabetical would begin `1x1, dilation, grouped, …` — different.)
- Pointwise flags: `elu_alpha, lower_clip, lower_slope, upper_clip, softplus_beta, swish_beta`. (Alphabetical would place `softplus_beta`/`swish_beta` before `upper_clip` — different.)

Any change to either order changes existing op_chain strings and is therefore a `[meta] version` bump plus a coordinated sidecar regen.

**Serialization rule — colon vs bare brackets (part of the frozen contract).** Two tag shapes coexist and serialize differently:

- **Mode-bearing tags** (Pointwise, Reduction) attach with a leading colon: `Pointwise:RELU_FWD`, `Reduction:ADD`, `Pointwise:RELU_FWD[upper_clip]`.
- **Bracket-only tags** (Conv, Batchnorm shape flags) attach directly with no colon: `ConvFprop[1x1,grouped]`, `Batchnorm[multi_batch]`.

A colon means "this node has a mode/op"; bare brackets mean "this node has shape flags but no mode." A reader (and the matcher loader) can tell them apart without a node-type lookup. The implementation rule: `describeNodeVariant()` returns a string that already begins with `[` for bracket-only tags, and `describeGraph` appends a colon only when the variant string doesn't already begin with `[`.

**Variants are demand-driven.** A variant tag (or combo field) earns its place only when an observed partition or a support conflict (§7) demonstrates the bare node type is too coarse. Adding speculative variants creates matcher-set noise — extra entries that look like dispatch-relevant distinctions but aren't, because the engine treats them identically. The support-conflict refusal in §7 is the canonical signal that a new variant is needed; absent that signal, leave the node type alone.

**`dtype_combos` fields.**

| Field          | Required | Meaning |
|----------------|----------|---------|
| `io`           | yes      | Input dtype. Also the output dtype when `output` is omitted (symmetric — the common case). |
| `output`       | no       | Output dtype. Set only when it differs from `io` (mixed-precision graphs). |
| `compute`      | yes      | Compute / accumulation dtype. |
| `intermediate` | no       | Intermediate dtype. Set only when the graph specifies it via `graph.set_intermediate_data_type(...)`. |

The named-field (inline-table) shape is deliberate:

- **Mirrors the support-matrix display.** The existing markdown shows `[io=bf16, compute=fp32, intermediate=fp32]` per entry. The schema uses the same shape, so the support matrix can be rendered directly from the sidecar — single source of truth (§14).
- **Captures what the engine actually dispatches on.** A simpler dtype dimension that collapsed `compute` and `intermediate` to record-side only would hide them from the matcher key. Combos make them first-class matcher-key fields: if the engine dispatches differently per compute dtype, a support conflict surfaces it instead of silently mixing.
- **Extensible by adding keys, not by inventing syntax.** A future dispatch dimension (weight dtype, accumulator, layout-variant, etc.) is just a new optional key. No new parsing convention; existing combos still load.

### 5.5 Variant data sources

The shape tags are computed from graph node attributes, not from test-case metadata, so the describer stays the single source of truth. (The test-case `note` field that previously fed the support-matrix "Notes" column is no longer load-bearing for claims; it still drives human-readable test-case notes.) Sources:

- **Conv:** `attributes.get_x()/get_w()` (fprop), `get_dx()/get_w()` (dgrad), `get_x()/get_dw()` (wgrad) for shapes; `get_pre_padding()`, `get_post_padding()`, `get_stride()`, `get_dilation()` for the rest.
  - `1x1` = all spatial filter dims == 1; `grouped` = in-channels / filter-in-channels > 1; `multi_batch` = N > 1; `non_square` = spatial input dims differ; `padding` / `stride` / `dilation` = any non-trivial value.
- **Batchnorm family:** `attributes.get_x()` → N > 1 ⇒ `multi_batch`.

This replaces the per-test-case `generateNote()` tag derivation in `*Common.hpp` as the authority for what a graph *is*.

## 6. Failure Detection

### 6.1 The five rules

- **Rule A — claim broken.** Claimed test with empty `engineIds` → **FAIL**.
- **Rule B — issue before test runs.** Walk `UnitTest::GetInstance` post-`RUN_ALL_TESTS`; any test that uses `IntegrationGraphVerificationHarness`, is non-skipped, and has no `recordGraphSupport` entry → **FAIL**. The harness self-registers each test in its `SetUp()` (before any `ASSERT_*`) so the verifier can distinguish "expected to record but didn't" from utility tests that legitimately never call `verifyGraph`. Requires moving `recordGraphSupport` to the first statement of `verifyGraph` so crashes after that point still produce records. Scoped to harness-using tests only.
- **Rule C — zero-coverage matcher.** Matcher with no observed tests in its cross-product → **FAIL** in full unfiltered CI runs; informational otherwise (§6.2).
- **Rule D — engine over-claim** *(note on existing test failure, not a new failure)*. Failed test + engine returned support + no matcher → annotate the test failure to point at `get_ranked_engine_ids`.
- **Rule E — unclaimed gain** *(warning, not a failure)*. Passing test + engine returned support + no matcher → log "claimed support != actual support" with fix steps.

### 6.2 Local partial runs

Rules A, B, D, E fire only over observed records — partial runs are checked against the partial set they actually exercised. Rule C is the one rule that depends on completeness; the verifier detects "full CI mode" by checking `--gtest_filter` is unset, no shard env vars are set, and `--enforce-support-claims` is in effect. Outside that mode Rule C downgrades to informational stderr.

### 6.3 Example failure output

Findings carry the full dispatch signature in the `dtype_combo` field (the label matches the schema field name), and name the matcher's source location:

```
[SUPPORT CLAIMS] arch=gfx942 platform=linux engine=MIOPEN_ENGINE: 3 failures, 1 note, 1 warning.

  CLAIM BROKEN (Rule A):
    Smoke/IntegrationGpuConvFwdBiasActiv.Correctness/0
      observed: op_chain="ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD"
                dtype_combo="{io=fp32, compute=fp32, intermediate=fp32}" layout="NCHW"
      claim source: [[supported]]#0/[[supported.matchers]]#3 in MIOPEN_ENGINE.supported.toml
      engine returned no support for this graph
      Action: narrow op_chains/dtype_combos to exclude this tuple, add a
              [[test_skips]] if it's broken-but-supported, or fix the engine.
      see docs/support-claims-failures.md#rule-a-claim-broken

  ISSUE BEFORE THE TEST RUNS (Rule B):
    Smoke/IntegrationGpuMatmul.Correctness/fp32_NCHW_32x32x32
      status: FAILED, no SupportMatrixCollector record — likely SetUp() failure
      Action: fix the underlying test failure first.

  ZERO-COVERAGE MATCHER (Rule C):
    block #4 in MIOPEN_ENGINE.supported.toml for arch=gfx942
      op_chains[0] = "ConvFprop + Pointwise:LEAKY_RELU"  -- 0 observed tests
      Action: regenerate via --write-support-claims, or hand-edit.

  ENGINE OVER-CLAIM (Rule D, note on existing test failure):
    Smoke/IntegrationGpuSdpaForward.Correctness/bf16_seq128_head16
      test FAILED; engine returned support: [MIOPEN_ENGINE]; no matcher covers this graph.
      Note: engine over-claimed via get_ranked_engine_ids. Add a [[test_skips]]
            or tighten the engine's applicability logic.

  UNCLAIMED GAIN (Rule E, warning):
    Smoke/IntegrationGpuLayernormForward.Correctness/fp16_NCHW_32x512x768
      observed: op_chain="LayerNorm" dtype_combo="{io=fp16, compute=fp32}" layout="NCHW"
                engine returned support: [MIOPEN_ENGINE].
      Action: if intentional, add to MIOPEN_ENGINE.supported.toml under
              [[supported]] arch="gfx942".
```

Asymmetric records show the output dtype, e.g. `{io=fp16, output=fp32, compute=fp32}`. Failures are grouped by `(matcher, op_chain)` when many tests share the same cross-product. Full lists are written to `support_claim_failures.txt` for CI artifact capture.

### 6.4 Verifier preconditions (refuse-to-run)

The verifier refuses `--enforce-support-claims` if any of:

- More than one distinct **plugin** is loaded, counted by `pluginName` from `getEngineInfo` (**not** by engine count). One plugin exposing several engines is fine — e.g. `miopen_provider_plugin` exposes both `MIOPEN_ENGINE` and `MIOPEN_ENGINE_DETERMINISTIC`; the verifier scopes per-engine via `supportingEngines.find(_engineName)`, and the condenser filters records by engine name too. Multi-engine attribution across distinct plugins is the deferred case (§9, §13).
- Loaded plugin's engine name doesn't match `[meta] engine`.
- `GTEST_TOTAL_SHARDS > 1` or `GTEST_SHARD_INDEX` is set (§9).
- `--gtest_break_on_failure` is set (aborts before the verifier runs).
- `--gtest_repeat` is set with N > 1 (record dedup conflict).
- The build is debug (`PrintToStringParamName` is non-deterministic in some cases; auto-gen and verifier must agree on param strings).

A `TestEventListener` for the verifier registers **before** any other listener so it owns `OnTestProgramEnd` when later listeners short-circuit.

## 7. Condensation Heuristic

The auto-gen tool sees up to 10K+ observed records per asic and must produce a small safe matcher set.

This section is the one place that uses set notation, because the algorithm is genuinely set-algebra. Two sets, defined over the observed records: the **supported set** `S` = `(op_chain, dtype_combo, layout)` tuples observed with non-empty engine support, and the **unsupported set** `U` = tuples observed with empty support. The emitted matcher set must satisfy:

1. **Coverage**: every tuple in `S` lies in the cross-product of some emitted matcher.
2. **Safety**: no tuple in `U` lies in any emitted matcher's cross-product. (A matcher over-claiming a `U` tuple would fail Rule A on the very next run.)
3. **Minimality (soft)**: fewer matchers preferred.

**Algorithm.** For each `op_chain` in `S`, compute a **rectangle cover** over its `(dtype_combo, layout)` cells — a set of rectangles `R_1, R_2, …` whose union covers every cell in `S_op` and intersects no cell in `U_op`. A single op_chain may need multiple rectangles when its supported cells are interleaved with unsupported ones.

Implementation: greedy maximum-rectangle cover. Repeatedly pick the largest safe rectangle covering remaining `S_op` cells, remove the covered cells, repeat until empty. Tie-break order for determinism: cells covered descending → rectangle area ascending (prefer smaller, to minimize Rule A risk from unobserved cells inside the rectangle) → lex order on `(combos, layouts)`. After per-op covers are computed, group by rectangle: each emitted matcher carries one rectangle and the set of op_chains whose cover includes it. Pure set operations over `std::set` / `std::map`; no trie, no token-splitting, no globbing.

**Worked example — anti-diagonal.** With `U = {(combo_a, NCHW), (combo_b, NHWC)}` and `S = {(combo_a, NHWC), (combo_b, NCHW)}`, the correct cover is two rectangles: `{combo_a}×{NHWC} ∪ {combo_b}×{NCHW}`. A naive one-rectangle-per-op_chain axis-shrink would silently drop half of `S`.

**Worked example — gfx10 with a CBA carve-out.** Engine doesn't have CK fusion kernels: every `ConvFprop + Pointwise:*` tuple is in `U`. Plain `ConvFprop` / `ConvDgrad` / `ConvWgrad` are in `S` with full combo/layout coverage and zero overlap with `U` → they share the same rectangle (all combos × all layouts), grouped into one matcher. The `ConvFprop + Pointwise:*` op_chains have empty covers (every observation is in `U`), so no matcher covers them; CBA tests run, return empty support, harness `GTEST_SKIP`s — no claim fires.

**Mixed-fixture case** (some cells for an op_chain in `S`, some in `U`): the cover splits into the largest safe sub-rectangles that exclude the `U` cells. Coverage of the supported combinations is preserved; the excluded combinations are listed on stderr ("supported: NCHW fp16/fp32/bf16; unsupported: NHWC fp16/fp32 — investigate") so the engineer can decide whether to add a `[[test_skips]]`.

### 7.1 Support-conflict safety invariant

A tuple `(op_chain, dtype_combo, layout)` that appears in both `S` and `U` across a single observation set is a **support conflict**: the same graph signature was observed as both supported and unsupported, which means the schema is too coarse to partition graphs by the engine's dispatch. No tuple **MAY** be a support conflict.

The condenser detects support conflicts and **refuses to write the sidecar**, producing a diagnostic that lists every conflicting tuple along with the test cases that voted each way. The engineer's fix is one of:

1. **Extend a per-node variant tag** (§5.4) when the divergence is node-structural (e.g. a new Pointwise param, or a shape axis, not yet captured). Bump `[meta].version`. Regenerate.
2. **Extend the `dtype_combos` field set** (§5.4) when the divergence is a dispatch dimension the schema doesn't yet capture (e.g. a new optional dtype field). Bump `[meta].version`. Regenerate.
3. **Add a `[[test_skips]]` entry** when the divergence is genuinely test-shape-dependent and not worth a schema extension. Regenerate; skipped tests don't produce records and the conflict disappears.

The conflict refusal is the canonical feedback loop for granularity gaps. Without it, schema drift is silent.

### 7.2 Variant cardinality is self-limiting

Shape variants multiply the op_chain axis (one bare `ConvFprop` becomes up to ~9 observed variants on the current fixture set). When engine support is uniform across variants — the common case — the rectangle-cover step folds them all into a single matcher with a long `op_chains` list, so sidecar size barely grows. Cardinality only expands when support genuinely partitions by shape, which is exactly when the extra rows carry information. The variant set is therefore self-limiting: uniform support collapses, divergent support is the signal worth recording.

This is principle 3 (variants are demand-driven) operationalized. For example, a Batchnorm FULL_TRAINING vs WITH_BATCH_STATS topology variant would land in the same rectangle for every dtype/layout — MIOpen dispatches both identically — so it should not be added. `multi_batch` belongs because it partitions support. A variant earns its place only when an observed partition (or a support conflict) demonstrates the bare node is too coarse.

## 8. Auto-Generation Tool

### 8.1 CLI

```bash
./hipdnn_integration_tests \
    --test-config dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --write-support-claims
```

Runs the full suite, observes per-test support, condenses to safe matchers (§7), and wholesale-rewrites the `[[supported]]` block(s) for the current `(arch, platform)` in `<EngineName>.supported.toml`. Other asics' blocks and the main TOML are untouched.

### 8.2 Atomic write

The tool generates the new sidecar via `tomlplusplus` in memory, writes to `<sidecar>.tmp.<pid>` with `O_WRONLY | O_CREAT | O_EXCL`, `fsync`s, and `rename`s atomically. On failure the tmp is unlinked; the original sidecar is never partially overwritten.

Platform notes: on Windows the replace uses `MoveFileExA` with `MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH` to get atomic replace + flush. On POSIX, the close+fsync+rename sequence is sufficient when source and target share a filesystem (always true here — both live in the config dir).

### 8.3 Preconditions

Refuses to run if any of:

- `--gtest_filter` set (partial baseline).
- `GTEST_TOTAL_SHARDS > 1` or shard env vars set (§9).
- More than one distinct plugin loaded (§6.4 — counted by `pluginName`).
- Debug build (`PrintToStringParamName` determinism).
- Sidecar's mount is read-only.
- Another process holds `flock(LOCK_EX)` on the sidecar.
- Existing sidecar has a `[[supported]]` block for this `(arch, platform)` whose matchers have zero observed coverage (would silently drop a previously valid claim — investigate before regenerating).
- Any observed `(op_chain, dtype_combo, layout)` tuple is a support conflict — observed as both supported and unsupported across the run (§7.1). The condenser dumps the full conflict list to `support_claim_conflicts.txt` and refuses to write the sidecar.

### 8.4 Build integration

`--write-support-claims` is a CLI flag on the `hipdnn_integration_tests` binary. CI never runs the tool — engineer-driven only; auto-applying it would silently rewrite the contract.

## 9. Sharding

This RFC does not solve sharding in v1. GoogleTest sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`) and external runners split a single binary's test population across processes; each process sees only its slice. Running the in-process verifier per shard would silently misbehave — a matcher whose cross-product happens to fall in a shard that didn't draw it would fire the "matched no observed tests" path despite the regression being real elsewhere.

v1 refuses `--enforce-support-claims` when any sharding env var is detected, with a clear stderr message. CI configurations that want enforcement run an unsharded job.

Intended v2 path: each shard writes its `SupportMatrixCollector` records to `support_records_shard_<N>.json`; a single `hipdnn_integration_tests --verify-claims-from <dir>` reads all shards, unions them, and runs the verifier once. The reduce phase can promote Rule C's zero-coverage warning to a hard error because in the union of all shards every matcher should have coverage. The full Full-tier integration suite eventually needs sharding for CI wall-clock.

## 10. Workflow and CI

**Day-to-day**: CI runs `--enforce-support-claims` on pre-submit (Smoke) and post-submit (Standard/Comprehensive/Full) on each target asic, unsharded. A claim-broken failure surfaces in the standard test report; the engineer either fixes the code, narrows the matcher, regenerates via `--write-support-claims`, or adds a `[[test_skips]]` entry with a reason.

**Bootstrap (new engine or new asic)**: engineer runs `./hipdnn_integration_tests --test-config <path-to-engine-toml> --write-support-claims` on the target hardware (Full tier, no filter, no sharding, release build), reviews the resulting sidecar diff, stages both files, and commits.

If the first regen reports support conflicts, follow the diagnostic's pointer to the offending node type or dispatch dimension; extend `describeNodeVariant()` or the `dtype_combos` field set; bump `[meta].version`; regenerate. Repeat until the regen lands clean (exit 0, no conflicts). Round-trip verify by immediately re-running with `--enforce-support-claims`: a healthy sidecar produces 0 failures / 0 notes / 0 warnings.

**Staged rollout for existing engines**: ship the loader and verifier with enforcement OFF by default → per-provider PRs add `[meta] engine` and a sidecar asynchronously → flip enforcement ON by default in a coordinated PR.

**HIPBLASLT bring-up gap.** The cross-engine `hipdnn_integration_tests` suite exercises no matmul graphs today, so a `--write-support-claims` run for `HIPBLASLT_ENGINE` would yield zero matchers. Its sidecar would be a legal `[meta]`-only bring-up placeholder until matmul fixtures are routed through the support-claims path (a v2 item — relates to the §13 multi-engine / coverage-completeness discussion).

## 11. Alternatives Considered

### 11.1 Flat per-instance test-name list

`tests = ["Suite/Fixture.Case/Param", ...]` enumerating every supported instance. Doesn't scale (10K+ instances per asic); swamps PR diffs; unreviewable. **Rejected** in favour of patterns/matchers.

### 11.2 Separate `[[unsupported]]` section with mandatory reason

Doubles the maintenance load (every op family classified positively *or* negatively); forces TOML updates whenever engine capability grows. **Rejected** — absence from `[[supported]]` is the implicit unsupported state.

### 11.3 Default-deny on unclaimed gain (or pure silence)

Three options for the engine-supports-something-no-matcher-covers case: fail, silent, or warn. Fail forces TOML updates on every new test family (PR churn, discourages adding tests). Silent lets the TOML drift from reality without anyone noticing. **Warning adopted** as the middle ground — Rule E in §6.1.

### 11.4 Glob patterns over test names

Earlier drafts used `patterns = ["*ConvFwd*"]` matched against GTest test names via `globMatch`. Rejected after review found: `globMatch` is platform-divergent (`fnmatch` vs `PathMatchSpecA`); GTest parameter strings can contain `[` and `]`; matching depends on naming convention (PascalCase, `TEST_P` vs `TEST_F` vs `TYPED_TEST`); `DISABLED_` prefix matches but tests never run. Structured matchers eliminate all four. **Rejected** in favour of structured selectors.

### 11.5 Multi-asic `[[supported]]` blocks

`archs = ["gfx942", "gfx90a"]` to share patterns. Couples updates that should be independent — engineer fixing gfx942 has to know the gfx90a story. **Rejected** in favour of one block per asic; sharing is by duplication.

### 11.6 Flat / arrow-string dtype lists

Simpler dtype dimensions were considered: a flat `io_dtypes = ["fp16", "fp32"]` list, and an arrow-string `io_dtype_pairs = ["fp16->fp32", ...]` form. The flat list collapses input vs output and hides compute and intermediate entirely from the matcher key, so it can't represent mixed-precision graphs or solvers that dispatch on compute dtype. The arrow-string form carries input/output but is positional, needs a parser convention for the arrow, and can't carry compute/intermediate without inventing further syntax. **Rejected** in favour of the named-field `dtype_combos` inline tables (§5.4): native TOML, named fields, mirrors the support-matrix display, extensible by adding keys.

## 12. Risks

| Risk | Mitigation |
|------|------------|
| A pattern accidentally over-matches and CI fails on every run. | The auto-gen tool's safety check (§7) prevents this for generated matchers. For hand-edited matchers, the failure is immediate and the report names the specific triple. |
| `describeGraph` output format changes, breaking existing `op_chain` strings. | `describeGraph` and its dependencies become a stability contract once this RFC ships. The `op_chain` string depends on `graph.visit()` traversal order, `to_string(NodeType)` mappings, the Pointwise/Reduction mode appendix, the `describeNodeVariant()` shape/mode tags, the fixed flag orders, and the colon-vs-bracket serialization rule (§5.4). Any change to graph flattening, node visit order, enum-to-string mapping, mode formatting, variant tag set, flag order, or the colon/bracket rule is a `[meta] version` bump plus a coordinated sidecar regen. |
| A variant tag or `dtype_combos` field changes shape. | The stability contract extends to `describeNodeVariant()` per node type — adding or modifying a variant tag (or changing flag order) bumps `[meta].version` and triggers coordinated regen. Likewise, adding a new key to `dtype_combos` (or making an optional key required) is a version bump. The support-conflict refusal in §7 is the system's primary signal that a new variant or combo field is needed — when it fires, that's the prompt to extend. |
| `[[test_skips]]` over-broad filter silently shadows a `[[supported]]` matcher. | Load-time warning when any matcher's cross-product is fully covered by a same-arch `[[test_skips]]` filter. |
| Test-naming-convention drift breaking the matcher. | Structured matchers don't depend on test naming — they match on `describeGraph` output. Naming convention is irrelevant to the verifier. |
| Auto-gen wholesale-replace blows away hand-curated content. | Sidecar split (§5): hand-curated `[[supported]]` blocks belong in the main file. The tool only touches the sidecar. |
| First run on a new asic fails because no block exists. | Verifier treats absence as **not enforced**. Engineer runs `--write-support-claims` once and commits. |

## 13. Open Questions and Future Work

- **Multi-engine attribution.** v1 refuses builds with more than one distinct **plugin** loaded (counted by `pluginName` — §6.4). One plugin exposing multiple engines (the MIOpen case) works today; multiple distinct plugins is the deferred case. The eventual fix requires either per-engine support bitmaps in `recordGraphSupport` (queries every loaded engine before picking one — Nx probe cost) or sourcing the verifier from RFC 0006's pre-filter manifest. Tracked separately.
- **Sharding reduce phase.** Per-shard JSON output mode + `--verify-claims-from <dir>` aggregator + ctest fixture (§9). Implementation deferred.
- **Detecting obsolete `[[test_skips]]`.** A `detect_obsolete = true` flag could trigger an offline check that builds each skipped graph and queries engine support without executing. Requires harness API changes for graph-build-without-execute.
- **Wildcards in matchers.** Rejected for v1. If a real engine (e.g. compiler-based) needs the "commit to all" stance, add explicit wildcard semantics in v2 with documentation that it commits to current AND future enum values.

## 14. Documentation

Alongside the implementation, this RFC commits to maintainer documentation in three artifacts under `dnn-providers/integration-tests/docs/`:

1. **Schema reference** (`support-claims-schema.md`) — field-by-field reference for `[meta]`, `[[supported]]`, `[[supported.matchers]]`, including the variant-tag table, the fixed flag orders, and the colon-vs-bracket serialization rule, with common shapes for new asics and op families.
2. **Failure-mode runbook** (`support-claims-failures.md`) — one section per Rule (A–E). Each answers: what triggered this, what the message means, ranked likely root causes, and step-by-step remediation.
3. **Bring-up guide** (`support-claims-bringup.md`) — adding a new asic or engine end-to-end: regen via the binary, review the diff, iterate on support conflicts, what to look for, what to commit. Cross-links the schema and the runbook.

**Sidecar as source of truth for the support matrix.** The `dtype_combos` schema mirrors the markdown display format (`[io=…, compute=…, intermediate=…]`). The render path can therefore be `sidecars → SupportMatrix.md` rather than two independently-maintained derivations of the same support data. The matrix becomes a *view* of the sidecars, not a separate state.

**`render_support_matrix.py`.** The implementation will include a renderer at `dnn-providers/integration-tests/tools/render_support_matrix.py` that reads one or more `<ENGINE>.supported.toml` sidecars and emits a single combined markdown matrix; each sidecar's `[meta].engine` becomes a column. Rows are the union of every `(op_chain, dtype_combo, layout)` tuple across sidecars; each engine column shows ✅ + layouts or `—`. Two styles:

- `zoom` (default) — a Google-Maps-style overview table (op family × engine, coverage fractions like `✅ 30/30`, `🟡 27/30`, `—`) with collapsible `<details>` per family expanding to per-variant layouts and then to full per-(variant, dtype) rows. Requires a `<details>`-aware viewer (GitHub, VS Code preview).
- `classic` — one row per (op_chain, dtype_combo); flat, lookup-friendly.

The dtype display (`[io=…, compute=…, intermediate=…]`) will be byte-identical to `SupportMatrixCollector`'s, so the toml-rendered matrix and the test-time matrix are interchangeable. This offline path is the preferred way to produce committed matrix docs without re-running on hardware (§2).

**Runtime linking — load-bearing.** Every verifier failure message ends with a stable anchor pointing into the failure-mode runbook, e.g. `see docs/support-claims-failures.md#rule-c-zero-coverage-matcher`. A maintainer reading a CI log can click straight to the entry that addresses what they're looking at. Anchor names and verifier output are co-designed; renaming an anchor without updating the verifier fails CI lint.

Docs land in the same PR as the implementation. PRs that change schema or verifier behaviour without updating the docs fail review.

## 15. Glossary

- **Matcher.** A `[[supported.matchers]]` entry. Claims that the cross-product of its `op_chains × dtype_combos × layouts` is fully supported by the engine.
- **Claimed.** A test whose `(op_chain, dtype_combo, layout)` lies in some matcher's cross-product for the current `(arch, platform)`.
- **Claim broken.** An observed claimed test with empty `engineIds`. Rule A.
- **Zero-coverage matcher.** A matcher whose cross-product matches zero observed tests in a full unfiltered CI run. Rule C.
- **Engine over-claim.** A test that fails while the engine returned support and no matcher covers it. Rule D (note, not a new failure).
- **Unclaimed gain.** A test that passes while the engine returned support and no matcher covers it. Rule E (warning).
- **Sidecar.** The machine-managed `<EngineName>.supported.toml` file containing `[[supported]]` blocks, paired with the hand-edited main `<EngineName>.toml`.
- **Variant tag.** A per-node label returned by `describeNodeVariant()` and appended to the node's serialization in op_chain strings. Partitions graphs the engine dispatches differently despite sharing a node type. Examples: `Pointwise:RELU_FWD[upper_clip]`, `Reduction:ADD`.
- **Shape variant.** A bracket-only `describeNodeVariant()` tag derived from a Conv or Batchnorm node's tensor shapes / conv params (`1x1`, `grouped`, `multi_batch`, `non_square`, `padding`, `stride`, `dilation`). Attaches to the node name without a leading colon: `ConvFprop[1x1,grouped]`.
- **Bracket-only vs mode-bearing tag.** A serialization distinction in op_chain: mode-bearing tags (Pointwise, Reduction) use `Node:MODE[flags]`; bracket-only tags (Conv, Batchnorm shape) use `Node[flags]`. The presence of the colon signals whether the node has a mode/op.
- **Dtype combo.** One entry in a matcher's `dtype_combos` list, an inline TOML table with named fields: `io` (required input dtype), `output` (optional, defaults to `io` for symmetric), `compute` (required), `intermediate` (optional). One entry per distinct dispatch signature.
- **Support conflict.** A tuple `(op_chain, dtype_combo, layout)` observed as both supported (by at least one test case) and unsupported (by at least one other) in the same run. Violates §7.1's safety invariant; the condenser refuses to write the sidecar and surfaces the offending test cases for engineer review. (In §7's set notation: a tuple in both the supported set `S` and the unsupported set `U`.)
- **Rectangle cover.** The output of §7's condensation algorithm for one op_chain: a set of `(dtype_combo_subset, layout_subset)` rectangles whose union covers every supported cell while avoiding every unsupported cell. Each rectangle becomes one emitted matcher.
