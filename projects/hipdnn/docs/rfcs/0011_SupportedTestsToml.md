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
5. [Proposed Design](#5-proposed-design)
6. [TOML Schema](#6-toml-schema)
7. [Failure Detection](#7-failure-detection)
8. [Condensation Heuristic](#8-condensation-heuristic)
9. [Auto-Generation Tool](#9-auto-generation-tool)
10. [Sharding](#10-sharding)
11. [Workflow](#11-workflow)
12. [CI Integration](#12-ci-integration)
13. [Alternatives Considered](#13-alternatives-considered)
14. [Risks](#14-risks)
15. [Open Questions and Future Work](#15-open-questions-and-future-work)
16. [Glossary](#16-glossary)

## 1. Executive Summary

This RFC proposes adding **structured engine-support claims**, scoped per-asic, to each per-engine integration-test config. Claims live in a hand-edited main TOML plus a machine-managed sidecar; both are loaded together. Each claim asserts a cross-product of `(op_chain, io_dtype, layout)` tuples the engine must support on the named arch. At the end of each integration-test run the verifier compares actual engine support against the claims and fails on one delta:

- **Supported claim broken**: a test whose `(op_chain, io_dtype, layout)` properties match a `[[supported.matchers]]` entry for the current arch, but the engine returned no support. The contract regressed.

Tests not covered by any matcher are unenforced — adding a new op or test family that the engine happens to support is silent and doesn't force a TOML update. The TOML records *intentional* contracts, not capability inventory.

Claims are **structured, not glob-based**. The matcher's `op_chains` is an exact list of strings (e.g. `"ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD"`) produced by the existing `describeGraph()`; `io_dtypes` and `layouts` are exact enumerations. No wildcards anywhere. This kills five classes of test-name-glob hazard at once: cross-platform `fnmatch`/`PathMatchSpecA` divergence, char-class collisions in parameter strings, fixture-naming-convention fragility, `TEST_F`/`TYPED_TEST` format misclassification, and `DISABLED_` prefix interaction.

Each engine TOML pairs with a single-engine assumption: the verifier refuses `--enforce-support-claims` when multiple plugins are loaded simultaneously (the multi-engine RFC 0006 case is out of v1 scope). Sharded runs are also refused under v1 with an explicit pointer at the future per-shard-reduce design (§10).

The auto-generation tool (`--write-support-claims`) is C++-only, embedded in `hipdnn_integration_tests`. It writes the sidecar `<EngineName>.supported.toml` for the current asic from observed runtime support, using a simple set-grouping algorithm (§8). Engineer reviews via `git diff`.

## 2. Problem Statement

The integration suite already records which engines support which graphs (`SupportMatrixCollector` in `dnn-providers/integration-tests/src/harness/SupportMatrixCollector.hpp`) and emits a markdown matrix on demand via `--generate-support-matrix`. But this output is advisory only — nothing fails if a row flips between runs.

When MIOpen's solver coverage shifts and a previously-supported conv shape becomes unsupported on gfx942, the integration test cleanly `GTEST_SKIP`s (see `IntegrationGraphVerificationHarness::verifyGraph` — the harness skips when no engine supports the graph) and CI passes. We need a CI gate that fails the moment a supported test stops being supported on the asics where we've claimed support — without forcing exhaustive TOML maintenance for every other test or asic the engineer isn't actively working on.

The existing `[[test_skips]]` mechanism already conflates two distinct things: tests the engine *doesn't* support (no kernel) and tests the engine *does* support but that are currently broken. This RFC separates those: support claims live in their own section; `[[test_skips]]` stays for the "engine supports but broken" case with its required `reason`.

## 3. Regressions Captured

This section is the scope statement. Reviewers should anchor here on what the system detects and what it deliberately doesn't.

### 3.1 Detected

| Regression | How it surfaces |
|------------|-----------------|
| A previously-supported test silently loses engine support on a claimed asic. | Observed `(op_chain, io_dtype, layout)` is in some matcher's cross-product; engine returned no support → **FAIL**. |
| A test errors before reaching `recordGraphSupport` (SetUp crash, ASSERT in graph construction, HIP init failure). | Walk of `UnitTest::GetInstance()` finds registered tests with status != PASSED/SKIPPED and no recorded support → **FAIL** with "errored before record; cannot verify claim — fix the error first" (§7.3). Conservative: we treat unknown as broken. |
| Engineer claims a matcher whose cross-product is wider than the engine actually supports. | Some test in the cross-product runs and returns empty support → **FAIL** pointing at the specific `(op_chain, io_dtype, layout)` triple. Engineer narrows the matcher or adds a `[[test_skips]]`. |
| A pre-existing claim's cross-product matches zero observed tests when `--write-support-claims` is run. | Tool refuses to wholesale-replace; engineer must investigate (catalog shrank? filter applied? wrong build?). Prevents silent loss of valid claims (§9.5). |

### 3.2 Deliberately not detected

| Non-regression | Rationale |
|----------------|-----------|
| Engine starts supporting a graph no matcher covers. | Avoiding this forces TOML updates on every new op or test family. Trade made consciously — TOML is intentional contracts, not capability inventory. |
| Entire `INSTANTIATE_TEST_SUITE_P` is deleted; remaining tests still satisfy the matcher. | Catalog shrinkage is a code-review concern. The PR that deletes the catalog is the right place to catch it. |
| Asic has no `[[supported]]` block yet. | New-asic bring-up: verifier treats absence as "not enforced." Engineer runs `--write-support-claims` once on the new hardware. |
| Numerical correctness drift. | Out of scope — existing test assertions and `[[tolerance_overrides]]` handle this. |
| Tests pre-skipped via `[[test_skips]]`. | Excluded from claim evaluation. Skip is the explicit "don't enforce" knob. |

The narrowness is the design's load-bearing property. It exchanges symmetric capability tracking for low-maintenance contracts engineers can keep current.

## 4. Current System Overview

### 4.1 What the suite already records

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

`describeGraph()` (`src/harness/GraphDescription.hpp`) walks graph nodes via `to_string(NodeType)` (`src/harness/NodeTypeNames.hpp`) and serializes a structured op chain plus dtype tags. The layout is set by fixtures via `setTestCaseLayout(...)`. The data structured selectors need is already produced; this RFC routes it through a typed channel rather than parsing the stringified form.

Recording happens **after** `[[test_skips]]` has already short-circuited the test in `SetUp()`, so the records correspond to tests not pre-skipped on this arch/platform.

### 4.2 TOML config today

`TestSettings` (`src/harness/TestSettings.hpp`) parses the per-engine TOML using `tomlplusplus`. Current sections: `[meta] version`, `[[tolerance_overrides]]`, `[[test_skips]]`. CLI plumbing in `src/main.cpp` (`--test-config`) stores a single `TestSettings` instance via `TestConfig::initialize`.

### 4.3 Where the new check fits

| Component | Existing role | New role |
|-----------|--------------|----------|
| `describeGraph` | Returns a flat string. | Refactored to return a typed `GraphDescription { op_chain, io_dtype, compute_dtype, intermediate_dtype }`; existing string form derived for the markdown matrix. |
| `recordGraphSupport` | Captures the stringified description. | Captures the typed form *plus* the layout *plus* engine IDs. |
| `TestSettings` | Parses tolerance + skips. | Also parses sidecar `<Engine>.supported.toml`; exposes `findMatchingMatcher(graphProps, arch, platform)`. |
| `SupportMatrixCollector` | Singleton record store. | Same; consumed by the verifier and auto-gen tool. |
| `main.cpp` | Owns final exit code. | Runs `SupportClaimVerifier` after `RUN_ALL_TESTS()`; hosts `--write-support-claims`. |

## 5. Proposed Design

### 5.1 Goals

1. **Per-asic contracts.** A developer working on gfx942 updates only the gfx942 block. No need to know other asics.
2. **Structured selectors, no wildcards.** Match on parsed graph properties, not test-name strings. Eliminates the entire class of glob-portability and naming-convention issues.
3. **Per-engine TOML, single engine assumed in v1.** Verifier refuses if multiple engines are loaded. Multi-engine attribution is RFC 0006 follow-up work.
4. **Tiny by construction.** Each matcher claims a cross-product; engines with broad uniform support use few matchers. Auto-gen output is in the dozens of lines per asic.
5. **Per-test failure granularity.** The verifier checks every observed test against matchers and reports per-test failures.
6. **Low maintenance ceiling.** Only claim-broken triggers a failure. Adding new tests or new ops doesn't force TOML edits anywhere.
7. **Graceful local degradation.** Partial runs (filter / fewer binaries / one tier) are checked over only what was observed; absent claims are not enforced; `--write-support-claims` refuses anything other than a full unsharded run.
8. **Auto-gen via the test binary itself.** `--write-support-claims` is a single C++ command. No Python, no JSON sidecar, no cross-language pipeline. Writes a separate `.supported.toml` sidecar; the hand-edited main TOML is never touched.
9. **Backward compatible.** Engines without a `[[supported]]` block (or without a sidecar) behave exactly as today.

### 5.2 Non-goals

- **Unclaimed-gain detection.** §3.2.
- **Catalog-shrink detection.** §3.2.
- **`[[unsupported]]` section.** Absence from `[[supported]]` is the implicit unsupported state.
- **Multi-asic blocks.** Each `[[supported]]` block names exactly one asic via `arch = "<single>"`. Sharing across asics is by duplication.
- **Multi-engine claims.** Each plugin's TOML is independent; runtime multi-engine is out of v1 scope (§5.3, §10).
- **Wildcards in matchers.** `op_chains`, `io_dtypes`, `layouts` are exact-string lists; no `*` or glob semantics. Wildcards would be claims about untested values.
- **Changes to `--generate-support-matrix`.** Markdown matrix is unchanged.

### 5.3 Data flow

```
                ┌───────────────────────────────────────────────────────┐
                │  MIOPEN_ENGINE.toml  (committed, hand-edited)         │
                │  [meta] engine = "MIOPEN_ENGINE"                      │
                │  [[tolerance_overrides]]                              │
                │  [[test_skips]]                                       │
                └───────────────────────┬───────────────────────────────┘
                                        │ load + merge
                ┌───────────────────────┴───────────────────────────────┐
                │  MIOPEN_ENGINE.supported.toml  (committed, machine)   │
                │  [[supported]]   arch = "gfx942"                      │
                │  [[supported.matchers]]   op_chains=[...] …           │
                │  [[supported]]   arch = "gfx90a"   …                  │
                └───────────────────────┬───────────────────────────────┘
                                        ▼
   test run ──► SupportMatrixCollector ──► SupportClaimVerifier ──► fail report
                (records typed graph                                       │
                 properties + engineIds)                                   ▼
                                                              fail run on any
                                                              broken matcher

   `hipdnn_integration_tests --write-support-claims` (offline):
       runs suite → groups observed-supported tuples → safety-checks against
       observed-unsupported → atomically rewrites .supported.toml for the
       current asic → engineer reviews `git diff`
```

## 6. TOML Schema

### 6.1 Main file (hand-edited)

```toml
# MIOPEN_ENGINE.toml — never touched by --write-support-claims

[meta]
version = 1
engine  = "MIOPEN_ENGINE"        # required; verifier checks the running plugin matches

[[tolerance_overrides]]
filters = ["Full/*BatchnormBackwardCalcStats3d*.Correctness/0"]
atol = 0.002
rtol = 0.012

[[test_skips]]
archs   = ["gfx90a"]
filters = ["*ConvFwdBiasActiv*"]
reason  = "Engine returns wrong results on gfx90a — ROCm/rocm-libraries#6979"
```

`[meta] engine` is new and required for any TOML using `[[supported]]`. The verifier compares it against the loaded plugin's engine name reported by `hipdnnGetEngineCount_ext` / `getEngineInfo`. Mismatch → refuse to enforce.

### 6.2 Sidecar file (machine-managed)

```toml
# MIOPEN_ENGINE.supported.toml — wholesale rewritten by --write-support-claims.
# Do not hand-edit. Both files MUST be committed together (CI lints this).

[meta]
version = 1
engine  = "MIOPEN_ENGINE"        # must match main file's [meta] engine

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

# Batchnorm variants
[[supported.matchers]]
op_chains = [
    "Batchnorm", "BatchnormInference", "BatchnormBackward",
    "BatchnormInferenceVarianceExt",
]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# Matmul
[[supported.matchers]]
op_chains = ["Matmul"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# ─── gfx90a (no CBA — wrong results; see test_skips in main file) ────────

[[supported]]
arch = "gfx90a"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad",
             "Batchnorm", "BatchnormInference", "BatchnormBackward"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# Note: no Conv+Pointwise matcher. CBA tests on gfx90a are pre-skipped
# via [[test_skips]] in the main file and never reach the verifier.

# ─── gfx10 (no CK fusion kernels) ────────────────────────────────────────

[[supported]]
arch = "gfx10"

[[supported.matchers]]
op_chains = ["ConvFprop", "ConvDgrad", "ConvWgrad",
             "Batchnorm", "BatchnormInference"]
io_dtypes = ["fp16", "fp32", "bf16"]
layouts   = ["NCHW", "NHWC"]

# No Conv+Pointwise matcher. CBA tests run; engine returns empty support;
# harness GTEST_SKIPs. No matcher claims them → §3.2 default-silent.
```

### 6.3 Field semantics

- **`arch`** (required, string). Matched against `TestConfig::getCurrentArch()` (raw `gcnArchName` like `"gfx942:sramecc+:xnack-"`). The verifier tokenizes the raw arch at the first `:` and compares the prefix exact-equal to the matcher's `arch`. **Substring matching is NOT used** (that would collide families: `"gfx10"` would match `gfx1030`, `gfx1100`, etc.). Singular by design — one block per asic.
- **`platform`** (optional, string, default = any). Exact match against `TestConfig::getCurrentPlatform()` (`"windows"` or `"linux"`).
- **`[[supported.matchers]]`** (required, ≥1 per block). Each matcher claims a cross-product:
  - **`op_chains`** (required, non-empty array of strings). Each string is an exact match against the observed test's `describeGraph().op_chain_string()`. No wildcards.
  - **`io_dtypes`** (required, non-empty array of strings). Each is a dtype name as printed by `to_string(DataType)` (e.g. `"fp16"`, `"fp32"`, `"bf16"`). No wildcards.
  - **`layouts`** (required, non-empty array of strings). Each is a layout label as passed by fixtures to `setTestCaseLayout` (e.g. `"NCHW"`, `"NHWC"`, `"NCDHW"`). No wildcards.

Schema rejects any wildcard, any whole-element `*`, and any empty array.

### 6.4 Matching algorithm

For each observed test (registered, not pre-skipped, reached `recordGraphSupport`), the verifier picks the single `[[supported]]` block whose `arch` (tokenized at `:`) and `platform` match the current run. At most one block applies per `(arch, platform)`; load-time check rejects duplicates.

A test **matches** a `[[supported.matchers]]` entry iff:
- its `op_chain` is in the entry's `op_chains`,
- AND its `io_dtype` is in the entry's `io_dtypes`,
- AND its `layout` is in the entry's `layouts`.

The test is **claimed** iff it matches ≥1 matcher in the block.

### 6.5 Schema versioning policy

`[meta] version` follows these rules:

- **Unknown keys** in `[meta]`, `[[supported]]`, or `[[supported.matchers]]` are ignored with an info log. This permits forward-compatible additions in minor revisions.
- **Field semantics never change in v1**; if the semantics of `op_chains` / `io_dtypes` / `layouts` ever change, the version bumps to 2 and v1 readers refuse to load v2 files (loud error, not silent).
- **Mixed-version repository state** (some engine TOMLs at v1, others at v2 during rollout) is supported by: v2 readers load v1 files normally; v1 readers refuse v2. New TOML version rollout is a coordinated PR across the affected engine TOMLs.
- `version` defaults to `1` if absent (backward compat for existing files without `[meta]`).

## 7. Failure Detection

### 7.1 The two rules

**Rule A — claim broken.** For each observed test whose `(op_chain, io_dtype, layout)` is claimed by ≥1 matcher: if `supportingEngineIds` is empty → **FAIL**.

**Rule B — errored before record.** Post-`RUN_ALL_TESTS()`, walk `UnitTest::GetInstance()`. For any registered test with status != `PASSED` and != `SKIPPED` and no record in `SupportMatrixCollector` → **FAIL** with "test errored before `recordGraphSupport`; verifier cannot determine if claim was broken — fix the underlying error first."

Rule B is conservative. A test that crashed in `SetUp()` or threw in graph construction has unknown graph properties; we treat unknown as broken so a real regression that manifests as a crash isn't silently passed over. The cost is occasional false-positive verifier failures when unrelated infrastructure breaks — but the underlying test failure will be louder than the verifier's message, so the engineer fixes the test first and the verifier noise resolves with it.

`recordGraphSupport` is moved to the **very first statement** of `verifyGraph` (before any `ASSERT_*`) so that anything that gets a graph constructed gets a record. Tests that fail before producing a graph fall under Rule B.

### 7.2 Local partial runs

Tests not observed (filtered out, in an unbuilt binary, in an unselected tier) contribute nothing. The verifier diffs only over `recordGraphSupport` entries. CI runs full → enforces all claimed observations; local filtered runs → enforces what they ran. A matcher with zero observed coverage in this run produces an informational stderr line but does not fail (it might be in a tier not exercised).

### 7.3 Example failure output

```
[SUPPORT CLAIMS] arch=gfx942 platform=linux engine=MIOPEN_ENGINE: 2 failures.

  CLAIM BROKEN:
    Smoke/IntegrationGpuConvFwdBiasActiv.Correctness/NCHW_1x16x16x16_1x16x3x3_relu
      observed: op_chain="ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD"
                io_dtype="fp32" layout="NCHW"
      matched by [[supported.matchers]] op_chains[2] = "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD"
      engine returned no support for this graph
      Action: narrow op_chains to exclude this tuple, add a [[test_skips]] if
              it's broken-but-supported, or fix the engine.

  ERRORED BEFORE RECORD:
    Smoke/IntegrationGpuMatmul.Correctness/fp32_NCHW_32x32x32
      status: FAILED
      no SupportMatrixCollector record — likely SetUp() failure
      Action: fix the underlying test failure; verifier cannot determine if
              the support claim was broken until the test runs to recordGraphSupport.
```

Failures are grouped by `(matcher, op_chain)` when many tests share the same matcher cross-product to avoid drowning the log; the first three offending param strings are listed inline with a `--verbose-claim-failures` flag for the full list. Full lists also written to `support_claim_failures.txt` for CI artifact capture.

### 7.4 Implementation sketch

```cpp
struct GraphDescription
{
    std::string op_chain;          // e.g. "ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD"
    std::string io_dtype;          // e.g. "fp32"
    std::string compute_dtype;
    std::string intermediate_dtype;
};

struct ClaimFailure
{
    enum Kind { ClaimBroken, ErroredBeforeRecord } kind;
    std::string testName;
    GraphDescription graphDesc;    // empty for ErroredBeforeRecord
    std::string layout;
    std::string matchedReason;     // human-readable; for grouping
};

class SupportClaimVerifier
{
public:
    std::vector<ClaimFailure> diff(const TestSettings& settings,
                                   std::string_view archRaw,
                                   std::string_view platform,
                                   std::span<const GraphSupportRecord> records,
                                   const ::testing::UnitTest& unitTest) const;

    static void writeReport(std::ostream& out,
                            std::span<const ClaimFailure> failures,
                            bool verbose);
};
```

Wired into `main.cpp` after `RUN_ALL_TESTS()`:

```cpp
const int gtestResult = RUN_ALL_TESTS();
int verifyResult = 0;
if(TestConfig::get().enforceSupportClaims())
{
    auto failures = SupportClaimVerifier{}.diff(/* ... */);
    if(!failures.empty())
    {
        SupportClaimVerifier::writeReport(std::cerr, failures, /*verbose=*/false);
        verifyResult = 1;
    }
}
return gtestResult | verifyResult;
```

A `TestEventListener` for the verifier registers **before** any other listener so it owns `OnTestProgramEnd` even when later listeners short-circuit. Synthetic `EXPECT_FAIL` entries are emitted for xUnit fidelity.

### 7.5 Verifier preconditions (refuse-to-run)

The verifier refuses `--enforce-support-claims` if any of:

- More than one plugin is loaded (multi-engine v1 unsupported).
- The loaded plugin's engine name doesn't match `[meta] engine` in the TOML.
- `GTEST_TOTAL_SHARDS > 1` or `GTEST_SHARD_INDEX` is set (see §10).
- `--gtest_break_on_failure` is set (it aborts before the verifier runs).
- `--gtest_repeat` is set with N > 1 (record dedup policy is "one record per test"; repeats would conflict).
- The build is debug (`PrintToStringParamName` is non-deterministic vs release in some cases; auto-gen and verifier must use the same param strings).

Each refusal prints a clear message naming the offending env var or flag.

### 7.6 Why fail-on-broken-claim only, not symmetric

A previous draft proposed failing on "unclaimed gain" too. That forces TOML updates on every new test family or op the engine happens to support, which discourages adding tests and bloats every unrelated PR. The TOML is a contract, not a capability inventory. Symmetric detection is rejected (§13.5).

## 8. Condensation Heuristic

The auto-gen tool sees up to 10K+ observed test records per asic. The algorithm condenses them to a small set of safe `[[supported.matchers]]` entries.

### 8.1 Invariants

Let:
- `S` = set of observed `(op_chain, io_dtype, layout)` tuples with non-empty supportingEngineIds for the current `(arch, platform)`.
- `U` = set of observed `(op_chain, io_dtype, layout)` tuples with empty supportingEngineIds.

Emitted matcher set `M` must satisfy:

1. **Coverage**: every tuple in `S` lies in the cross-product of some `m ∈ M`.
2. **Safety**: no tuple in `U` lies in the cross-product of any `m ∈ M`.
3. **Minimality (soft)**: fewer matchers preferred.

Safety is hard: a matcher that includes an `(op_chain, io_dtype, layout)` tuple where the engine returned empty would fail CI on the very next run. The tool rejects such matchers.

### 8.2 Algorithm

The algorithm is set-grouping, not pattern-coalescing:

1. Group `S` by `(io_dtypes, layouts)` rectangle. Two tuples `(op_a, d, l)` and `(op_b, d, l)` go in the same group when their `(d, l)` values are equal *and* extending the group's `op_chains` doesn't create a `(*, d', l')` in `U` for any `d'`/`l'` already in the group.
2. For each group, emit one `[[supported.matchers]]` with that group's `op_chains`, `io_dtypes`, and `layouts`.
3. If a candidate matcher's cross-product would include a tuple in `U`, split it — drop one of `op_chains[i]`, `io_dtypes[j]`, or `layouts[k]` until safe. Prefer dropping op_chains (most fine-grained).

The implementation is straightforward C++ over `std::set` / `std::map` — no trie, no glob coalescing, no token splitting. No external dependencies beyond `tomlplusplus` (already used).

### 8.3 Worked example

Observed-supported on gfx942 (subset, for illustration):

| op_chain | io_dtype | layout |
|----------|----------|--------|
| ConvFprop | fp16 | NCHW |
| ConvFprop | fp16 | NHWC |
| ConvFprop | fp32 | NCHW |
| ConvFprop | fp32 | NHWC |
| ConvFprop | bf16 | NCHW |
| ConvFprop | bf16 | NHWC |
| ConvDgrad | fp16 | NCHW |
| ConvDgrad | fp32 | NCHW |
| ConvWgrad | fp32 | NCHW |
| ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD | fp16 | NCHW |
| (… etc) | | |

Observed-unsupported on gfx942: none in this example.

Grouping:
- All Conv* with io={fp16,fp32,bf16} layouts={NCHW,NHWC} present → one matcher with `op_chains=["ConvFprop","ConvDgrad","ConvWgrad"]`, full dtype/layout cross-product.
- Wait — ConvDgrad lacks (fp16,NHWC), (fp32,NHWC), (bf16,*). ConvWgrad lacks even more. Safety check: emitting one matcher with all three op_chains × full rectangle would include unobserved tuples (might be `U`!). Algorithm splits: ConvFprop matcher gets full rectangle; ConvDgrad and ConvWgrad get narrower matchers reflecting only their observed tuples.

The output is what the schema in §6.2 shows.

### 8.4 Worked example with carve-out

On gfx10 the engine doesn't have CK fusion kernels. Observed-unsupported includes every `ConvFprop + Pointwise:*` tuple. The algorithm:

- Group plain `ConvFprop`/`ConvDgrad`/`ConvWgrad` with full dtype/layout rectangle → safe (no overlap with `U`) → emit.
- Group `ConvFprop + Pointwise:ADD + Pointwise:RELU_FWD` → entire op_chain is in `U` → no matcher emitted. The CBA tests will run, return empty support, harness `GTEST_SKIP`s, and no matcher claims them.

No special handling needed — the safety check naturally produces "no matcher" output for un-supportable op_chains.

### 8.5 Edge cases

| Case | Handling |
|------|----------|
| Test errored before record (Rule B in §7.1). | Tuple is absent from both `S` and `U`. Algorithm proceeds; verifier raises Rule B at run time. |
| Two tests with same `(op_chain, io_dtype, layout)`, one supported, one unsupported. | Algorithm treats as conflict — adds to `U` and emits an stderr note ("inconsistent support for tuple X across observed tests; suspect flake or arch-conditional behavior"). |
| Tuple observed in `S` whose op_chain string is identical to an existing claim's entry but with a different layout. | Algorithm augments the existing matcher's `layouts` list if safe; else creates a new matcher. |
| `op_chain` string contains characters that would be problematic for TOML escaping (`"`, `\`, newlines). | Forbidden in the source: `describeGraph()` produces ASCII strings without escapes. Load-time check rejects pathological strings to be safe. |

## 9. Auto-Generation Tool

### 9.1 CLI

```bash
./hipdnn_integration_tests \
    --test-config dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --write-support-claims
```

Runs the full suite, observes per-test support, condenses to safe matchers (§8), and wholesale-replaces the `[[supported]]` block(s) for the current `(arch, platform)` in `MIOPEN_ENGINE.supported.toml`. Other asics' blocks are untouched. The hand-edited main file is **never** modified.

### 9.2 Sidecar file layout

The sidecar lives alongside the main TOML, named `<EngineName>.supported.toml` (e.g. `MIOPEN_ENGINE.supported.toml`). Both files must be committed together; a CI lint rejects PRs that stage only one.

The loader reads both files at startup (`TestSettings::initialize` follows the sidecar discovery rule: "if `<main>.supported.toml` exists alongside the main TOML, parse it and union its `[[supported]]` blocks"). Discovery is mechanical, no opt-in flag.

Inside the sidecar:
- `[meta] engine` must match the main file's engine; mismatch → load fails loud.
- `[[supported]]` blocks are the only content. Anything else → load fails loud.
- Wholesale replacement is the only write mode; merge-preserve is future work.

### 9.3 Atomic single-stage write

The tool uses tomlplusplus to read the existing sidecar (if any), drops `[[supported]]` blocks matching the current `(arch, platform)`, adds the freshly-generated block, and writes:

1. Generate full sidecar TOML text into memory via tomlplusplus's serializer.
2. Write to `MIOPEN_ENGINE.supported.toml.tmp.<pid>` with `O_WRONLY | O_CREAT | O_EXCL`.
3. `fsync` the temp file.
4. `rename(tmp, final)` — atomic on POSIX, atomic on Windows when `MoveFileExA` with `MOVEFILE_REPLACE_EXISTING` is used (the tool calls the right API per platform).
5. `fsync` the containing directory (POSIX) to persist the rename.

If any step fails, the tmp file is unlinked and the tool exits nonzero with a clear message. The original sidecar is never partially overwritten.

### 9.4 Preconditions (refuse to run)

The tool refuses if any of:

- `--gtest_filter` is set (partial run → bogus baseline).
- `GTEST_TOTAL_SHARDS > 1` or `GTEST_SHARD_INDEX` set (sharded run → partial baseline; §10).
- Multiple plugins are loaded (cannot attribute support per engine; v1 single-engine only).
- The build is debug (`PrintToStringParamName` may produce different param strings vs release; mandate release builds for baseline generation).
- The sidecar's filesystem mount is read-only (e.g. some Docker bind-mount configurations).
- Another process holds a `flock(LOCK_EX)` on the sidecar (concurrent writes refused).
- The sidecar already contains a `[[supported]]` block for the current `(arch, platform)` AND that block has matchers whose cross-products match zero observed tests (would silently drop a previously valid claim → §3.1 last row).

### 9.5 Mixed-fixture stderr output

When an `op_chain` appears in both `S` and `U` for the same `(arch, platform)`, the algorithm can't safely include it. Tool emits to stderr:

```
Conflicting support for op_chain "ConvFprop + Pointwise:ADD + Pointwise:GELU_FWD" on gfx942:
  supported: 3 tests (NCHW fp16/fp32/bf16)
  unsupported: 2 tests (NHWC fp16/fp32)

  Resolution options (apply by hand-editing the main file's [[test_skips]],
  or by re-running --write-support-claims after fixing the engine):
    - Add a [[test_skips]] entry for the unsupported subset, if it's an
      engine-supports-but-broken case
    - Investigate why NHWC is unsupported (engine bug? layout-specific path?)
    - If the unsupported subset is intentional, no action — those tests
      remain default-silent (§3.2)
```

The tool emits no matcher for the conflicting op_chain. Tests in the unsupported subset are silent under §3.2; tests in the supported subset are silent too (no claim). The engineer decides whether to encode the resolution in the main file.

### 9.6 Build integration

Two surfaces:

- **CLI flag** on the integration test binary (`--write-support-claims`).
- **CMake/ctest target** per provider (e.g. `ninja miopen-provider-write-support-claims`) that invokes the binary with the right `--test-config` baked in.

## 10. Sharding

This RFC does not solve sharding in v1. It explicitly refuses enforcement when sharding is detected, but the design needs to accommodate sharding in the future, so the path is documented here.

### 10.1 The problem

GoogleTest's sharding (`GTEST_TOTAL_SHARDS` / `GTEST_SHARD_INDEX` env vars), `ctest -jN`, and external runners like `gtest-parallel` all split a single test binary's test population across multiple processes. Each process sees only its shard's tests. In the proposed in-process verifier:

- Each process loads the same TOML and runs the verifier against its own `SupportMatrixCollector` records.
- A matcher whose cross-product is entirely outside a shard's slice fires the "matched no observed tests" info line but doesn't fail.
- A regression in a shard that didn't draw the relevant tests passes green; the shard that did draw them fails. The CI aggregator may report the failure correctly or may not, depending on how shard results merge.
- More dangerously: matcher safety becomes unverifiable per-shard, because each shard's `U` is a subset of the whole.

### 10.2 v1 behaviour: refuse

`--enforce-support-claims` refuses to run when any sharding env var is detected (`GTEST_TOTAL_SHARDS`, `GTEST_SHARD_INDEX`, plus the gtest-parallel detection markers). Clear stderr message names the env var and points at this section. CI configurations that want enforcement must run an unsharded job.

This is a real cost — the integration suite's Full tier could benefit from sharding — but it's the only honest v1 stance. The alternative (in-process verifier with sharded input) silently misbehaves.

### 10.3 Future design: per-shard records + reduce phase

The intended v2 path:

1. Each test process writes its `SupportMatrixCollector` records to `support_records_shard_<N>.json` (or `.bin` for size) in a configured output directory, instead of running the verifier in-process.
2. After all shards complete, a single `hipdnn_integration_tests --verify-claims-from <dir>` invocation reads every shard's records, unions them, runs the verifier once, and emits the combined diff.
3. CTest invokes the reduce job as a `FIXTURES_CLEANUP`-style dependent test that runs after all shards.

The reduce-phase verifier can also promote the "matched no observed tests" info line to a hard error, because in the union of all shards every matcher's cross-product should have non-zero observed coverage (assuming a full unfiltered run).

Per-shard JSON records also serve the auto-gen tool: a sharded `--write-support-records-shard` mode could collect observations across shards, then a separate `--write-support-claims-from <dir>` invocation does the condensation. Out of v1.

### 10.4 Why not just disable sharding in CI

The hipDNN integration suite at Full-tier maturity is hours of wall time; CI parallelism is non-optional medium-term. Documenting the refuse-to-enforce path and pointing at the reduce design is the honest v1 stance; banning sharding outright would push back on a legitimate CI need.

## 11. Workflow

### 11.1 Day-to-day flow (most PRs)

1. PR author lands a code change.
2. CI runs `hipdnn_integration_tests --enforce-support-claims ...` (single-engine, unsharded job).
3. If a matcher is broken or a test errored, CI fails with the per-test report (§7.3).
4. Author edits:
   - Real regression → fix the code.
   - Matcher too broad → regenerate via `--write-support-claims` for that asic, or narrow by hand.
   - Test errored unrelatedly → fix the test (no TOML change).
   - Engine-supported-but-broken → add a `[[test_skips]]` entry in the main file with a reason.
5. Re-run CI; failure clears.

### 11.2 Bootstrap (new engine or new asic)

1. Engineer runs `ninja <provider>-write-support-claims` on the target hardware (Full tier, no filter, no sharding, release build).
2. Tool generates `[[supported]]` block(s) for the current `(arch, platform)` in the sidecar. Mixed-fixture cases listed on stderr.
3. Engineer reviews the `git diff` of `<EngineName>.supported.toml`. Stages both the main file (if changed) and the sidecar.
4. Commits; first CI run on that asic goes green.

### 11.3 Staged rollout for existing engines

The first PR landing this RFC must not break existing engines that lack TOML claims. Order:

1. **Ship the loader and verifier with enforcement OFF by default**. `--enforce-support-claims` is opt-in; engines without claims are unaffected.
2. **Per-provider PRs add a `[meta] engine` line + sidecar** asynchronously, generated via `--write-support-claims` on each target asic.
3. **Flip enforcement ON by default once all providers have sidecars**. Single coordinated PR.

`.gitattributes` pins `*.supported.toml` to LF line endings to prevent Windows engineers from churning the file with CRLF.

## 12. CI Integration

- **Pre-submit (Smoke):** `--enforce-support-claims` on. Catches claim regressions on the fast tier.
- **Post-submit (Standard/Comprehensive/Full):** `--enforce-support-claims` on. Wider observed set → more tuples evaluated.
- **Unsharded enforcement job per asic.** Until the §10.3 reduce design lands, the enforcement job runs unsharded. Other CI shape (sharded test runs that don't enforce claims) is unaffected.
- **CI never runs `--write-support-claims`.** Engineer-driven tool only; auto-applying in CI would silently rewrite the contract.
- **Sidecar-pair lint.** CI rejects PRs that stage `<E>.toml` or `<E>.supported.toml` without the other when both would be needed.
- **CODEOWNERS for the sidecar.** Recommend each provider's TOML pair has CODEOWNERS coverage so contract changes get reviewed.

## 13. Alternatives Considered

### 13.1 Flat per-instance test-name list

Enumerate every supported test name. Doesn't scale (10K+ instances per asic); swamps PR diffs; unreviewable. **Rejected.**

### 13.2 Per-`(suite, instantiation)` counts and/or digests

Snapshot, not contract; counts decay silently as catalogs grow; breaks on partial local runs. **Rejected.**

### 13.3 Coverage floors / `min_count`

Stale by construction; brittle on partial runs. Catalog-shrink detection is intentionally out of scope (§3.2). **Rejected.**

### 13.4 Separate `[[unsupported]]` section with mandatory `reason`

Doubles the maintenance load (every op family classified positively *or* negatively). Forces TOML updates whenever engine capability grows. **Rejected** — absence from `[[supported]]` is the implicit unsupported state.

### 13.5 Default-deny on unclaimed gain

Tests not matching any matcher that turn out to be supported → FAIL. Forces TOML updates on every new test family the engine happens to support. Bring-up friction; PR churn; discourages adding tests. **Rejected** — the TOML is intentional contracts, not capability inventory.

### 13.6 Multi-asic `[[supported]]` blocks

`archs = ["gfx942", "gfx90a"]` to share patterns. Couples updates that should be independent — engineer fixing gfx942 has to know the gfx90a story. **Rejected** in favour of one block per asic.

### 13.7 Glob patterns over test names

Earlier drafts of this RFC used `patterns = ["*ConvFwd*"]` matched against GTest test names via `globMatch`. Rejected after review found:

- `globMatch` is platform-divergent (`fnmatch` vs `PathMatchSpecA`) — same pattern can match on Linux and miss on Windows because `*` doesn't cross `/`.
- GTest parameter strings can contain `[` and `]` (char-class metacharacters).
- The match is sensitive to test-naming convention (PascalCase, TEST_P vs TEST_F vs TYPED_TEST format differences).
- `DISABLED_` prefix matches but tests never run, silently passing claims.

Structured matchers (op_chain + dtype + layout from `describeGraph`) eliminate all four. **Rejected** in favour of structured selectors.

### 13.8 Wildcards in structured matchers

A draft proposed `op_chains = ["ConvFprop + Pointwise:*"]` with `*` as a Pointwise-mode wildcard. Rejected because `*` would mean "all current AND future pointwise modes" — a claim about the unknown that the engineer can't honestly verify. **Rejected** — exact enumeration only.

### 13.9 Sidecar baseline file per engine

Already adopted (§6.2). Earlier drafts considered keeping the auto-gen output inside the main TOML using a sentinel comment for the machine-managed region; that was rejected because sentinels are fragile to autoformatters, CRLF normalisation, and BOM. Sidecar files have clear human/machine separation, atomic single-writer semantics, and trivial CI lint for stage-pair-ness.

### 13.10 Python-based TOML round-trip

`tomlkit` in a Python helper for comment-preserving rewrites. Rejected because the sidecar approach makes round-trip a non-issue — the sidecar is wholly machine-owned and tomlplusplus's canonical serialiser is fine for it. **Rejected.**

### 13.11 Hash digest only

One SHA over the sorted supported set per asic. Zero diagnostic value; no per-test action list. **Rejected.**

### 13.12 GTest-only failure mode (no run-end verifier)

`ADD_FAILURE()` inside the harness body. A test can only fail if it ran; the post-`RUN_ALL_TESTS` listener catches Rule B cases (errored before record) that GTest-only doesn't. **Partial adoption** — verifier runs at program end *and* installs a `TestEventListener` for xUnit fidelity.

### 13.13 Verifier as standalone binary

Architect-suggested: pull the verifier out into its own binary that consumes (RFC 0006 manifest + TOML). Survives test-binary aborts and fits sharding aggregation naturally. **Rejected for v1**, deferred. v1 keeps verifier in-binary; if/when sharding's reduce phase lands (§10.3) the verifier mode that consumes serialized records can be reused as a standalone path. The architectural rationale is sound but the v1 scope is in-process.

### 13.14 Programmatic skip-list inversion

Replace claims with: every unsupported test must have a matching `[[test_skips]]` entry. Forces engineers to enumerate the complement (larger set); `reason` strings become meaningless for "no kernel." **Rejected.**

## 14. Risks

| Risk | Mitigation |
|------|------------|
| Matcher claims a triple the engine doesn't support; CI fails on every run until fixed. | The auto-gen tool's safety check prevents this for generated matchers (§8.1). For hand-edited matchers, the failure is immediate and the report names the specific triple. |
| A matcher whose cross-product matches no observed test silently passes review. | Informational stderr line at verifier time. The auto-gen tool refuses to wholesale-replace a block whose existing matchers have zero coverage (§9.4 last bullet). |
| First run on a new asic fails because no `[[supported]]` block exists. | Verifier treats absence as **not enforced**. Bring-up is unblocked; engineer runs `--write-support-claims` once and commits. |
| Sidecar gets out of sync with main TOML (e.g. `[meta] engine` mismatch). | Load-time check; loud error. CI lint requires both files staged together. |
| Atomic write fails mid-operation (disk full, crash). | tmp+fsync+rename; if any step fails, original is untouched; tmp is cleaned up on exit. |
| Engineer edits the sidecar by hand and the auto-gen wipes it. | Sidecar's header comment marks it machine-managed; the tool's wholesale-replace is documented. Engineer-written claims belong in the main TOML, not the sidecar (the main file supports `[[supported]]` blocks too; both are unioned). |
| Test-naming convention drifts and the harness can't recover graph properties. | Structured matchers don't depend on test naming — they match on `describeGraph` output. Naming convention is irrelevant to the verifier. (Still recommended via a separate CI lint, but not load-bearing here.) |
| `describeGraph` output format changes between releases, breaking existing matcher op_chain strings. | `describeGraph` becomes a stability contract once this RFC ships. Any change to its output format is a `[meta] version` bump and a coordinated sidecar regen. Documented in §6.5. |
| `[[test_skips]]` over-broad filter silently shadows a `[[supported]]` matcher. | Load-time warning when any `[[supported.matchers]]` cross-product is fully covered by a same-arch `[[test_skips]]` filter. |
| `--gtest_repeat` or `--gtest_break_on_failure` interaction. | Verifier refuses to run when either is set (§7.5). |
| Secret leakage via `testName` strings landing in committed TOML. | The sidecar contains `op_chain`/`dtype`/`layout` strings only — never `testName`. Test names are used internally by the verifier for failure messages, never serialized. |
| Concurrent `--write-support-claims` runs corrupt the sidecar. | `flock(LOCK_EX)` for the duration of read+compute+write. Second concurrent run refused (§9.4). |

## 15. Open Questions and Future Work

- **RFC 0006 multi-engine attribution.** v1 refuses multi-engine builds. The eventual fix requires either (a) per-engine support bitmaps in `recordGraphSupport` (queries every loaded engine before picking one — Nx probe cost) or (b) sourcing the verifier from RFC 0006's pre-filter manifest. Both are non-trivial. Tracked separately.
- **Sharding reduce phase.** §10.3 sketches the v2 design. Implementation requires per-shard JSON output mode, a `--verify-claims-from <dir>` mode, and a ctest fixture wiring it together.
- **Detecting obsolete `[[test_skips]]`.** A `detect_obsolete = true` flag could trigger an offline check: for each skipped test, build the graph and query engine support without executing. Returned support means the skip is stale. Requires harness API changes for graph-build-without-execute.
- **Wildcards in matchers.** §13.8 rejects `*` for v1. If a real engine (e.g. compiler-based, claims "any pointwise mode") needs the "commit to all" stance, add explicit wildcard semantics in v2 with documentation that it commits to current AND future enum values.
- **Merge-preserve auto-gen.** v1 wholesale-replaces matching blocks. Merge-preserving behaviour (keep matchers still validly covering all-supported observations and no unsupported ones; only add new matchers for gaps) is a v2 enhancement if re-run friction becomes a felt cost.
- **Build-constraint scoping.** Support is technically a function of (engine, arch, build flags, ROCm version). v1 factors out arch and platform. A future schema could add `rocm_min_version`, `compiler_flags`, etc. for engines where this matters.
- **Standalone verifier binary.** §13.13 deferred. Natural rework when sharding's reduce phase lands.
- **CODEOWNERS for sidecars.** Convention to recommend; not enforced by this RFC.

## 16. Glossary

- **Engine.** A plugin-provided implementation that can execute a graph. Reported by `hipdnnGetEngineCount_ext` / `getEngineInfo`.
- **Supported (for a test).** The engine returned at least one entry from `Graph::get_ranked_engine_ids()`.
- **Pre-skipped.** A test removed by a matching `[[test_skips]]` entry during `SetUp()`. Excluded from claim evaluation.
- **Observed.** A test that reached `SupportMatrixCollector::recordGraphSupport` during this run (not pre-skipped, not filtered out, in a built binary).
- **Errored before record.** A test that started but failed/crashed before reaching `recordGraphSupport`. Conservatively counts as a verifier failure (Rule B in §7.1).
- **Graph properties.** The structured tuple `(op_chain, io_dtype, layout)` extracted from `describeGraph()` output and `setTestCaseLayout`.
- **Matcher.** A `[[supported.matchers]]` entry. Claims that the cross-product of its `op_chains × io_dtypes × layouts` is fully supported by the engine.
- **Claimed.** A test whose `(op_chain, io_dtype, layout)` lies in some matcher's cross-product for the current `(arch, platform)`.
- **Claim broken.** An observed claimed test with empty `engineIds`. The only positive failure mode (Rule A in §7.1).
- **Sidecar.** The machine-managed `<EngineName>.supported.toml` file containing `[[supported]]` blocks. Always paired with the hand-edited main `<EngineName>.toml`.
