# hipDNN - Engine Support Claims in Per-Engine TOML

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
14. [Glossary](#14-glossary)

## 1. Executive Summary

This RFC adds **structured engine-support claims** to each per-engine integration-test config. Within this per-engine config, claims are scoped per-asic and live in a machine-managed sidecar (`<EngineName>.supported.toml`) paired with the hand-edited main TOML. Each claim asserts a cross-product of `(op_chain, io_dtype, layout)` tuples the engine must support on the named arch. The verifier runs after `RUN_ALL_TESTS()` and fails the build when a claimed test loses engine support.

Claims are exact-string lists, not globs — no wildcards. The `op_chain` strings are produced by the existing `describeGraph()`; `io_dtypes` and `layouts` are enumerations. This eliminates an entire class of test-name glob hazards (platform-divergent matchers, char-class collisions, fixture-naming fragility, `TEST_F`/`TYPED_TEST` format misclassification, `DISABLED_` prefix).

The auto-generation tool (`--write-support-claims`, embedded in `hipdnn_integration_tests`, C++-only) observes runtime support and rewrites the sidecar for the current asic. Engineer reviews via `git diff`.

## 2. Problem Statement

The integration suite already records which engines support which graphs (`SupportMatrixCollector`) and emits a markdown matrix via `--generate-support-matrix`, but the output is advisory — nothing fails if a row flips between runs. When MIOpen's solver coverage shifts and a previously-supported conv shape becomes unsupported on gfx942, the integration test cleanly `GTEST_SKIP`s (no engine supports the graph) and CI passes.

We need a CI gate that fails the moment a supported test stops being supported on the asics we've claimed — without forcing exhaustive TOML maintenance for every other test or asic the engineer isn't actively working on. The existing `[[test_skips]]` mechanism conflates "engine doesn't support" with "engine supports but is broken"; this RFC separates them.

## 3. Regressions Captured

This section is the scope statement. Reviewers should anchor here on what the system detects and what it deliberately doesn't.

### 3.1 Detected

| Regression | How it surfaces |
|------------|-----------------|
| A previously-supported test silently loses engine support on a claimed asic. | Observed `(op_chain, io_dtype, layout)` is in some matcher's cross-product; engine returned no support → **FAIL**. |
| An issue occurs before the test runs. | Test errored before it could record its graph properties → **FAIL** with "errored before record; fix the error first." |
| A matcher is too wide and claims support the engine doesn't have. | Some test in the matcher's cross-product runs and the engine returns no support → **FAIL** pointing at the specific `(op_chain, io_dtype, layout)` triple. Engineer narrows the matcher or adds a `[[test_skips]]` for known issues. |
| A matcher matches zero observed tests. | Something has gone wrong with the test suite and we are finding no tests for a particular matcher.  If this occurs and is not a bug, we need to revise the matchers for an asic so they are up to date. |
| The engine claims support for a test that no matcher covers, then the test fails. | The engine states it supports a graph then fails verification.  This is the current behaviour of the test suite today. |

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

`describeGraph()` walks graph nodes and serializes a structured op chain plus dtype tags. The layout is set by fixtures via `setTestCaseLayout(...)`. Recording happens **after** `[[test_skips]]` has short-circuited any pre-skipped tests in `SetUp()`. The structured data this RFC needs is already produced — implementation routes it through a typed channel instead of re-parsing the string.

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

`[meta] engine` is checked at load against the loaded plugin's engine name (from `getEngineInfo`). Mismatch → refuse to enforce.

### 5.2 Sidecar file (machine-managed)

```toml
# MIOPEN_ENGINE.supported.toml — wholesale rewritten by --write-support-claims.
# Do not hand-edit.

[meta]
version = 1
engine  = "MIOPEN_ENGINE"

# ─── gfx942 ──────────────────────────────────────────────────────────────

[[supported]]
arch = "gfx942"

# Plain Conv (no fusion)
[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# Conv + Activation (no bias) — every observed activation mode
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:SIGMOID",
    "ConvFprop + Pointwise:TANH",
    "ConvFprop + Pointwise:GELU_FWD",
    "ConvFprop + Pointwise:ELU",
    "ConvFprop + Pointwise:SWISH",
    "ConvFprop + Pointwise:SOFTPLUS",
]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# Conv + Bias + Activation (CBA)
[[supported.matchers]]
op_chains = [
    "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:SIGMOID",
    "ConvFprop + Pointwise:ADD + Pointwise:TANH",
    "ConvFprop + Pointwise:ADD + Pointwise:GELU_FWD",
    "ConvFprop + Pointwise:ADD + Pointwise:ELU",
    "ConvFprop + Pointwise:ADD + Pointwise:SWISH",
    "ConvFprop + Pointwise:ADD + Pointwise:SOFTPLUS",
]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# ─── gfx90a (no CBA — see test_skips in main file) ───────────────────────

[[supported]]
arch = "gfx90a"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad",
             "Batchnorm", "BatchnormInference", "BatchnormBackward"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# ─── gfx10 (no CK fusion kernels) ────────────────────────────────────────

[[supported]]
arch = "gfx10"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad",
             "Batchnorm", "BatchnormInference"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]
```

### 5.3 Field semantics

- **`arch`** (required, string). Compared against `TestConfig::getCurrentArch()` (raw `gcnArchName` like `gfx942:sramecc+:xnack-`) by tokenizing at the first `:` and exact-matching the prefix. Substring matching is rejected — it would collide families (`"gfx10"` would match `gfx1030`, `gfx1100`). One block per asic.
- **`platform`** (optional, string, default = any). Exact match against `"windows"` or `"linux"`.
- **`op_chains` / `io_dtypes` / `layouts`** (required, non-empty arrays of strings). Each value is matched exactly against the corresponding field of the observed test's `(op_chain, io_dtype, layout)` tuple. No wildcards anywhere — schema rejects `*`.

A test **matches** a matcher iff its `op_chain ∈ op_chains` AND `io_dtype ∈ io_dtypes` AND `layout ∈ layouts`. A test is **claimed** iff it matches ≥1 matcher in the asic's block.

Schema versioning: `[meta] version = 1`. Unknown keys are logged-and-ignored for forward compatibility; semantic changes bump the version, and v1 readers refuse v2 files loudly.

## 6. Failure Detection

### 6.1 The five rules

- **Rule A — claim broken.** Claimed test with empty `engineIds` → **FAIL**.
- **Rule B — issue before test runs.** Walk `UnitTest::GetInstance` post-`RUN_ALL_TESTS`; any registered, non-skipped test with no `recordGraphSupport` entry → **FAIL**. Requires moving `recordGraphSupport` to the first statement of `verifyGraph` so crashes after that point still produce records.
- **Rule C — zero-coverage matcher.** Matcher with no observed tests in its cross-product → **FAIL** in full unfiltered CI runs; informational otherwise (§6.2).
- **Rule D — engine over-claim** *(note on existing test failure, not a new failure)*. Failed test + engine returned support + no matcher → annotate the test failure to point at `get_ranked_engine_ids`.
- **Rule E — unclaimed gain** *(warning, not a failure)*. Passing test + engine returned support + no matcher → log "claimed support != actual support" with fix steps.

### 6.2 Local partial runs

Rules A, B, D, E fire only over observed records — partial runs are checked against the partial set they actually exercised. Rule C is the one rule that depends on completeness; the verifier detects "full CI mode" by checking `--gtest_filter` is unset, no shard env vars are set, and `--enforce-support-claims` is in effect. Outside that mode Rule C downgrades to informational stderr.

### 6.3 Example failure output

```
[SUPPORT CLAIMS] arch=gfx942 platform=linux engine=MIOPEN_ENGINE: 3 failures, 1 note, 1 warning.

  CLAIM BROKEN (Rule A):
    Smoke/IntegrationGpuConvFwdBiasActiv.Correctness/NCHW_1x16x16x16_1x16x3x3_relu
      observed: op_chain="ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD"
                io_dtype="fp32" layout="NCHW"
      engine returned no support for this graph
      Action: narrow op_chains to exclude this tuple, add a [[test_skips]] if
              it's broken-but-supported, or fix the engine.

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
      observed: ("LayerNorm","fp16","NCHW"); engine returned support: [MIOPEN_ENGINE].
      Action: if intentional, add to MIOPEN_ENGINE.supported.toml under
              [[supported]] arch="gfx942".
```

Failures are grouped by `(matcher, op_chain)` when many tests share the same cross-product. Full lists are written to `support_claim_failures.txt` for CI artifact capture.

### 6.4 Verifier preconditions (refuse-to-run)

The verifier refuses `--enforce-support-claims` if any of:

- More than one plugin is loaded (multi-engine deferred — §9, §13).
- Loaded plugin's engine name doesn't match `[meta] engine`.
- `GTEST_TOTAL_SHARDS > 1` or `GTEST_SHARD_INDEX` is set (§9).
- `--gtest_break_on_failure` is set (aborts before the verifier runs).
- `--gtest_repeat` is set with N > 1 (record dedup conflict).
- The build is debug (`PrintToStringParamName` is non-deterministic in some cases; auto-gen and verifier must agree on param strings).

A `TestEventListener` for the verifier registers **before** any other listener so it owns `OnTestProgramEnd` when later listeners short-circuit.

## 7. Condensation Heuristic

The auto-gen tool sees up to 10K+ observed records per asic and must produce a small safe matcher set.

Let `S` = observed `(op_chain, io_dtype, layout)` tuples with non-empty support; `U` = tuples with empty support. The emitted matcher set must satisfy:

1. **Coverage**: every tuple in `S` lies in the cross-product of some emitted matcher.
2. **Safety**: no tuple in `U` lies in any emitted matcher's cross-product. (A matcher over-claiming a `U` tuple would fail Rule A on the very next run.)
3. **Minimality (soft)**: fewer matchers preferred.

Algorithm: group `S` by `(io_dtypes, layouts)` rectangle; for each group, emit one matcher with that rectangle and the set of `op_chains` present. If a candidate matcher's cross-product would include a tuple in `U`, split the matcher — drop one of `op_chains[i]`, `io_dtypes[j]`, or `layouts[k]` until safe. Prefer dropping `op_chains` (most fine-grained). Pure set-grouping over `std::set` / `std::map`; no trie, no token-splitting, no globbing.

**Worked example — gfx10 with a CBA carve-out.** Engine doesn't have CK fusion kernels: every `ConvFprop + Pointwise:*` tuple is in `U`. Plain `ConvFprop` / `ConvDgrad` / `ConvWgrad` are in `S` with full dtype/layout coverage and zero overlap with `U` → one safe matcher with all three op_chains and the full rectangle. The `ConvFprop + Pointwise:*` op_chains are entirely in `U`, so no matcher covers them; CBA tests run, return empty support, harness `GTEST_SKIP`s — no claim fires.

Mixed-fixture case (some tuples for an op_chain in `S`, some in `U`): tool emits no matcher for that op_chain and lists the conflict on stderr ("supported: NCHW fp16/fp32/bf16; unsupported: NHWC fp16/fp32 — investigate"). Engineer hand-edits the main TOML if a `[[test_skips]]` is warranted.

## 8. Auto-Generation Tool

### 8.1 CLI

```bash
./hipdnn_integration_tests \
    --test-config dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --write-support-claims
```

Runs the full suite, observes per-test support, condenses to safe matchers (§7), and wholesale-rewrites the `[[supported]]` block(s) for the current `(arch, platform)` in `<EngineName>.supported.toml`. Other asics' blocks and the main TOML are untouched.

### 8.2 Atomic write

The tool generates the new sidecar via `tomlplusplus` in memory, writes to `<sidecar>.tmp.<pid>` with `O_WRONLY | O_CREAT | O_EXCL`, `fsync`s, and `rename`s atomically (using `MoveFileExA` on Windows). On failure the tmp is unlinked; the original sidecar is never partially overwritten.

### 8.3 Preconditions

Refuses to run if any of:

- `--gtest_filter` set (partial baseline).
- `GTEST_TOTAL_SHARDS > 1` or shard env vars set (§9).
- More than one plugin loaded.
- Debug build (`PrintToStringParamName` determinism).
- Sidecar's mount is read-only.
- Another process holds `flock(LOCK_EX)` on the sidecar.
- Existing sidecar has a `[[supported]]` block for this `(arch, platform)` whose matchers have zero observed coverage (would silently drop a previously valid claim — investigate before regenerating).

### 8.4 Build integration

`--write-support-claims` is a CLI flag on the `hipdnn_integration_tests` binary. CI never runs the tool — engineer-driven only; auto-applying it would silently rewrite the contract.

## 9. Sharding

This RFC does not solve sharding in v1. GoogleTest sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX`) and external runners split a single binary's test population across processes; each process sees only its slice. Running the in-process verifier per shard would silently misbehave — a matcher whose cross-product happens to fall in a shard that didn't draw it would fire the "matched no observed tests" path despite the regression being real elsewhere.

v1 refuses `--enforce-support-claims` when any sharding env var is detected, with a clear stderr message. CI configurations that want enforcement run an unsharded job.

Intended v2 path: each shard writes its `SupportMatrixCollector` records to `support_records_shard_<N>.json`; a single `hipdnn_integration_tests --verify-claims-from <dir>` reads all shards, unions them, and runs the verifier once. The reduce phase can promote Rule C's zero-coverage warning to a hard error because in the union of all shards every matcher should have coverage. The full Full-tier integration suite eventually needs sharding for CI wall-clock;

## 10. Workflow and CI

**Day-to-day**: CI runs `--enforce-support-claims` on pre-submit (Smoke) and post-submit (Standard/Comprehensive/Full) on each target asic, unsharded. A claim-broken failure surfaces in the standard test report; the engineer either fixes the code, narrows the matcher, regenerates via `--write-support-claims`, or adds a `[[test_skips]]` entry with a reason.

**Bootstrap (new engine or new asic)**: engineer runs `./hipdnn_integration_tests --test-config <path-to-engine-toml> --write-support-claims` on the target hardware (Full tier, no filter, no sharding, release build), reviews the resulting sidecar diff, stages both files, and commits.

**Staged rollout for existing engines**: ship the loader and verifier with enforcement OFF by default → per-provider PRs add `[meta] engine` and a sidecar asynchronously → flip enforcement ON by default in a coordinated PR.

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

## 12. Risks

| Risk | Mitigation |
|------|------------|
| A pattern accidentally over-matches and CI fails on every run. | The auto-gen tool's safety check (§7) prevents this for generated matchers. For hand-edited matchers, the failure is immediate and the report names the specific triple. |
| `describeGraph` output format changes, breaking existing `op_chain` strings. | `describeGraph` becomes a stability contract once this RFC ships. Any format change is a `[meta] version` bump and a coordinated sidecar regen. |
| `[[test_skips]]` over-broad filter silently shadows a `[[supported]]` matcher. | Load-time warning when any matcher's cross-product is fully covered by a same-arch `[[test_skips]]` filter. |
| Test-naming-convention drift breaking the matcher. | Structured matchers don't depend on test naming — they match on `describeGraph` output. Naming convention is irrelevant to the verifier. |
| Auto-gen wholesale-replace blows away hand-curated content. | Sidecar split (§5): hand-curated `[[supported]]` blocks belong in the main file. The tool only touches the sidecar. |
| First run on a new asic fails because no block exists. | Verifier treats absence as **not enforced**. Engineer runs `--write-support-claims` once and commits. |

## 13. Open Questions and Future Work

- **Multi-engine attribution.** v1 refuses multi-engine builds. The eventual fix requires either per-engine support bitmaps in `recordGraphSupport` (queries every loaded engine before picking one — Nx probe cost) or sourcing the verifier from RFC 0006's pre-filter manifest. Tracked separately.
- **Sharding reduce phase.** Per-shard JSON output mode + `--verify-claims-from <dir>` aggregator + ctest fixture (§9). Implementation deferred.
- **Detecting obsolete `[[test_skips]]`.** A `detect_obsolete = true` flag could trigger an offline check that builds each skipped graph and queries engine support without executing. Requires harness API changes for graph-build-without-execute.
- **Wildcards in matchers.** Rejected for v1. If a real engine (e.g. compiler-based) needs the "commit to all" stance, add explicit wildcard semantics in v2 with documentation that it commits to current AND future enum values.

## 14. Glossary

- **Matcher.** A `[[supported.matchers]]` entry. Claims that the cross-product of its `op_chains × io_dtypes × layouts` is fully supported by the engine.
- **Claimed.** A test whose `(op_chain, io_dtype, layout)` lies in some matcher's cross-product for the current `(arch, platform)`.
- **Claim broken.** An observed claimed test with empty `engineIds`. Rule A.
- **Zero-coverage matcher.** A matcher whose cross-product matches zero observed tests in a full unfiltered CI run. Rule C.
- **Engine over-claim.** A test that fails while the engine returned support and no matcher covers it. Rule D (note, not a new failure).
- **Unclaimed gain.** A test that passes while the engine returned support and no matcher covers it. Rule E (warning).
- **Sidecar.** The machine-managed `<EngineName>.supported.toml` file containing `[[supported]]` blocks, paired with the hand-edited main `<EngineName>.toml`.
