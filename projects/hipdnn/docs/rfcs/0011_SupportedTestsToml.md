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
8. [Pattern Condensation Heuristic](#8-pattern-condensation-heuristic)
9. [Auto-Generation Tool](#9-auto-generation-tool)
10. [Workflow](#10-workflow)
11. [CI Integration](#11-ci-integration)
12. [Alternatives Considered](#12-alternatives-considered)
13. [Risks](#13-risks)
14. [Open Questions and Future Work](#14-open-questions-and-future-work)
15. [Glossary](#15-glossary)

## 1. Executive Summary

This RFC proposes adding a `[[supported]]` section, scoped per-asic, to each per-engine integration-test TOML (e.g. `dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml`). Each block holds glob patterns asserting "every test matching these patterns must be supported by this engine on this arch." At the end of each integration-test run, the harness compares actual engine support against the claims and fails the run on exactly one delta:

- **Supported claim broken**: a test matches a `[[supported]]` pattern for the current arch but the engine returned no support. The contract regressed.

Tests not covered by any `[[supported]]` pattern are not enforced — adding a new op or test family that the engine happens to support is silent and does not require a TOML update. The TOML records *intentional* support contracts, not a capability inventory. Symmetric "engine gained support somewhere unexpected" detection is deliberately out of scope (see §3).

The existing `[[test_skips]]` section keeps its current behaviour and its required `reason` field. Its semantics are clarified: it is for tests the engine **does** support but that are currently broken (wrong results, crashes, bugs). Skipped tests are pre-empted in `SetUp()` and excluded from claim evaluation.

The model is **pattern-based, not enumeration-based**. There are no per-test name lists, no counts, no digests. The TOML stays small — one block per asic per engine, each block typically tens of pattern lines. A bundled auto-generation tool (`--write-support-claims`) implemented entirely in C++ produces or refreshes the block for the current asic by observing runtime support and condensing it to a minimal set of safe patterns (§8). The tool rewrites only the machine-managed region of the TOML, below a sentinel comment. The engineer reviews via `git diff`.

## 2. Problem Statement

The integration suite already has the data it needs to know which engines support which graphs — `SupportMatrixCollector` records `(graphName, graphDescription, testName, supportingEngineIds)` for every test and emits a markdown matrix when `--generate-support-matrix` is passed (see `dnn-providers/integration-tests/src/harness/SupportMatrixCollector.hpp`). But this output is advisory only — nothing fails if a row flips between runs.

When MIOpen's solver coverage shifts and a previously-supported conv shape becomes unsupported on gfx942, the integration test cleanly `GTEST_SKIP`s (the harness skips when no engine supports the graph; see `IntegrationGraphVerificationHarness::verifyGraph`) and CI passes. We need a CI gate that fails the moment a supported test stops being supported on the asics where we've claimed support for it — without forcing exhaustive TOML maintenance for every other test or asic the engineer isn't actively working on.

## 3. Regressions Captured

This section explicitly scopes what the new system detects and what it deliberately doesn't, so reviewers have one place to anchor on intent.

### 3.1 Detected

| Regression | How it surfaces |
|------------|-----------------|
| A previously-supported test silently loses engine support on a claimed asic (solver removed, heuristic narrowed, refactor regression). | `[[supported]]` pattern matches the test; engine returned no support → **FAIL**. |
| A parameterized test instance silently disappears such that the engine was the only one running it. | Same as above — the existing instance(s) the claim covered are still checked. A specific instance going missing without the pattern noticing is caught only if it was the *only* instance the engine supported in that pattern's match set, and another test triggers the regression. See §3.2 for the case this design doesn't address. |
| A code change accidentally flips an engine's `get_ranked_engine_ids` from returning an ID to returning empty for a graph the TOML claims is supported. | `[[supported]]` pattern asserts must-be-supported; engine returns empty → **FAIL**. |
| Engineer claims `*Conv*` then a `ConvFwdBiasActiv` test runs and is unsupported on the claimed asic. | The over-broad pattern matches; engine doesn't support that specific instance → **FAIL**. Forces the engineer to narrow the pattern, add an explicit `[[test_skips]]` if the test is actually broken, or fix the engine. |

### 3.2 Deliberately not detected

| Non-regression | Rationale |
|----------------|-----------|
| Engine starts supporting a test that no `[[supported]]` pattern covers. | Avoiding this would force a TOML update on every PR that adds a new op or test family. Trade made consciously — the TOML records intentional contracts, not capability inventory. |
| Entire `INSTANTIATE_TEST_SUITE_P` is deleted; remaining tests still satisfy the pattern. | Catalog shrinkage is a code-review concern. The PR that deletes the catalog is the right place to catch it; the test-config TOML is the wrong layer. |
| A claimed asic doesn't have a `[[supported]]` block yet. | New-asic bring-up: the verifier treats absence as "not enforced." Engineer runs `--write-support-claims` once on the new hardware to establish the contract. |
| Numerical correctness drift. | Out of scope — existing test assertions and `[[tolerance_overrides]]` handle this. |
| Tests skipped via `[[test_skips]]` because of a known bug. | Excluded from claim evaluation. Skip is the explicit "don't enforce" knob. |

The narrowness of detected vs. not-detected is the design's load-bearing property. It exchanges symmetric capability tracking for low-maintenance per-asic contracts that engineers can actually keep current.

## 4. Current System Overview

### 4.1 What the suite already records

`SupportMatrixCollector` (`src/harness/SupportMatrixCollector.hpp`) is a thread-safe singleton populated by `IntegrationGraphVerificationHarness::verifyGraph`:

```cpp
SupportMatrixCollector::get().recordGraphSupport(
    graph.graph_attributes.get_name(),
    describeGraph(graph),
    testName,                              // "Suite/Fixture.TestCase/Param"
    status.is_good() ? engineIds : {},     // engine IDs that returned support
    _testCaseNote,
    _testCaseLayout);
```

Recording happens **after** `[[test_skips]]` has already short-circuited the test in `SetUp()`. The records correspond to "tests not pre-skipped on this arch/platform" — exactly the population the verifier wants.

### 4.2 TOML config today

`TestSettings` (`src/harness/TestSettings.hpp`) parses the per-engine TOML using `tomlplusplus`. Current sections:

```toml
[meta]
version = 1

[[tolerance_overrides]]
filters = ["..."]
atol = ...
rtol = ...

[[test_skips]]
archs    = ["gfx942"]   # optional substring match against gcnArchName
platforms = ["windows"] # optional
filters  = ["..."]
reason   = "..."
```

CLI plumbing lives in `src/main.cpp` (`--test-config`) and `TestConfig::initialize` stores a single `TestSettings` instance.

### 4.3 Where the new check fits

| Component | Existing role | New role |
|-----------|--------------|----------|
| `TestSettings` | Parses tolerance + skips | Also parses `[[supported]]` blocks |
| `SupportMatrixCollector` | Records and emits markdown | Records feed the verifier and the auto-gen tool |
| `main.cpp` | Owns final exit code | Runs the verifier after `RUN_ALL_TESTS()` and fails on drift; hosts `--write-support-claims` |

## 5. Proposed Design

### 5.1 Goals

1. **Per-asic contracts.** A developer working on gfx942 updates only the gfx942 `[[supported]]` block. No need to know or touch gfx90a, gfx10, gfx11, gfx12 stories.
2. **Single source of truth per engine.** All sections live in the same TOML.
3. **Tiny by construction.** Patterns, not enumerations. One block per asic; coalesced patterns inside (§8).
4. **Per-test failure granularity.** The verifier checks every observed test against patterns and reports per-test failures.
5. **Low maintenance ceiling.** Only claim-broken triggers a failure. Adding new tests or new ops doesn't force TOML edits anywhere.
6. **Graceful local degradation.** Partial runs (filter / fewer binaries / one tier) are checked over only what was observed.
7. **Auto-gen via the test binary itself.** `--write-support-claims` is a single C++ command on `hipdnn_integration_tests`. No Python, no JSON sidecar, no cross-language pipeline.
8. **Backward compatible.** Engines without `[[supported]]` blocks behave exactly as today.

### 5.2 Non-goals

- **Unclaimed-gain detection.** §3.2.
- **Catalog-shrink detection.** §3.2.
- **`[[unsupported]]` section.** Absence from `[[supported]]` is the implicit unsupported state. No section, no schema, no separate diagnostics.
- **Per-test enumeration.** The TOML never lists individual test names.
- **Multi-asic blocks.** Each `[[supported]]` block names exactly one asic via `arch = "<single>"`. Sharing across asics is by duplication. Forces per-asic update granularity.
- **Multi-engine claims.** Each plugin's TOML is independent.
- **Changes to `--generate-support-matrix`.** Markdown matrix is unchanged.

### 5.3 Data flow

```
                ┌──────────────────────────────────────────────┐
                │  per-engine TOML (committed)                 │
                │  [[supported]]   arch="gfx942"  patterns=... │
                │  [[supported]]   arch="gfx90a"  patterns=... │
                │  [[test_skips]]  ...                         │
                └──────────────────────┬───────────────────────┘
                                       │ load
                                       ▼
   test run ──► SupportMatrixCollector ──► SupportClaimVerifier ──► diff report
                (existing)                  (new)                       │
                                                                        ▼
                                                          fail run on any
                                                          claim-broken entry

   `hipdnn_integration_tests --write-support-claims` (offline, current asic):
       runs suite → condenses observed support into safe patterns → rewrites
       only the machine-managed region below the TOML sentinel → engineer
       reviews `git diff`
```

## 6. TOML Schema

```toml
[meta]
version = 1

# Existing.
[[tolerance_overrides]]
filters = ["Full/*BatchnormBackwardCalcStats3d*.Correctness/0"]
atol = 0.002
rtol = 0.012

# Existing — clarified semantics: engine supports the test but it's broken.
[[test_skips]]
archs   = ["gfx90a"]
filters = ["*ConvFwdBiasActiv*"]
reason  = "Engine returns wrong results on gfx90a — ROCm/rocm-libraries#6979"

# ─── auto-generated: do not edit below this line ─────────────────────
# Regenerate per-arch with: ninja miopen-provider-write-support-claims
# ─────────────────────────────────────────────────────────────────────

# NEW: positive support claim, scoped to one asic.
[[supported]]
arch     = "gfx942"
# platform = "linux"            # optional; exact match against current platform
patterns = [
    "*ConvFwd*",
    "*ConvBwd*",
    "*ConvWgrad*",
    "*ConvFwdBiasActiv*",
    "*Batchnorm*",
    "*Pooling*",
    "*Activation*",
    "*Matmul*",
    "*Reduction*",
    "*RMSNorm*",
]

[[supported]]
arch     = "gfx90a"
patterns = [
    "*ConvFwd*",
    "*ConvBwd*",
    "*ConvWgrad*",
    "*Batchnorm*",
    "*Pooling*",
    "*Activation*",
    "*Reduction*",
    "*RMSNorm*",
]

[[supported]]
arch     = "gfx10"
patterns = [
    "*ConvFwd*",
    "*ConvBwd*",
    "*ConvWgrad*",
    "*Batchnorm*",
    "*Pooling*",
    "*Activation*",
]
# Note: gfx11, gfx12 not yet brought up; their tests will be observed
# but not enforced since no [[supported]] block exists for them.
```

### 6.1 Field semantics

- `arch` (**required**, string). Substring matched against `TestConfig::getCurrentArch()` (raw gcnArchName like `"gfx942:sramecc+:xnack-"`). Singular by design — one block per asic. Schema rejects an array.
- `platform` (optional, string, default = any). Exact match against `TestConfig::getCurrentPlatform()` (`"windows"` or `"linux"`).
- `patterns` (**required**, non-empty array of strings). GTest-style globs (`*` wildcard) matched against the fully-qualified GTest test name. Reuses `globMatch` from `TestSettings`.

`[[supported]]` blocks have no `reason` field — the block itself is the contract documentation.

### 6.2 Matching algorithm

At test time, the verifier picks the single `[[supported]]` block whose `arch` substring-matches the current run and whose `platform` matches (if specified). At most one block applies per `(arch, platform)`. The verifier rejects the config at load if multiple blocks match the same `(arch, platform)` — duplication is unambiguous and should be eliminated by hand.

A test is **claimed** if any pattern in the matching block matches its full GTest name.

### 6.3 Why per-asic blocks

Engineers usually have hands-on context for one asic at a time — the one they're testing against. A schema that lets (or forces) a single block to span multiple asics couples updates: a developer fixing gfx942 has to know what's currently true on gfx90a to not damage that asic's block. Single-arch blocks decouple updates entirely — the gfx942 block can be regenerated or hand-edited without anyone needing to think about other asics.

The cost is duplication when two asics genuinely share support stories. In practice the stories diverge enough (RDNA gap, MI300-only ops, etc.) that duplication is small, and the workflow win is the real driver.

### 6.4 Why no `[[unsupported]]` section

An earlier draft had a separate `[[unsupported]]` section for explicitly-asserted-not-supported patterns, with a "documented gap closed" failure when the engine started supporting something it had claimed not to. That capability was traded away to keep the workflow low-friction:

- Documenting every gap doubles the TOML maintenance load (every op family needs to be classified positively *or* negatively).
- The "gap closed" failure forces engineers to update TOMLs reactively whenever an engine grows capability — which is exactly what §3.2 says we want to avoid.

Absence from `[[supported]]` is the unsupported state. No separate section needed.

### 6.5 Why no `reason` field on `[[supported]]`

`[[supported]]` is a positive contract: "this engine works here." There's nothing to explain — the entry itself is the documentation. By contrast `[[test_skips]]` requires a `reason` because every skip is an active issue worth naming, and `[[tolerance_overrides]]` accepts free-form comments (in TOML) because every bumped tolerance reflects a specific investigation.

## 7. Failure Detection

### 7.1 The single rule

For each observed test (registered, not pre-skipped, reached `recordGraphSupport`):

> **If the test matches any `[[supported]]` pattern for the current `(arch, platform)`, the engine must have returned at least one supporting engine ID. Otherwise FAIL.**

Tests not matching any pattern are unenforced — silent regardless of actual support.

### 7.2 Local partial runs

Tests never observed (filtered out, in an unbuilt binary, in an unselected tier) contribute nothing. The verifier diffs only over `recordGraphSupport` entries. CI runs full → enforces all matched tests; local filtered runs → enforces what they ran.

A `[[supported]]` pattern that matches no observed test gets an informational stderr line ("pattern X matched no observed tests") but does **not** fail — partial-run friendliness.

### 7.3 Example failure output

```
[SUPPORT CLAIMS] arch=gfx942 platform=linux: 1 failure.

  SUPPORTED CLAIM BROKEN:
    Smoke/IntegrationGpuConvFwdBiasActiv.Correctness/NCHW_1x16x16x16_1x16x3x3
      matched supported pattern(s): ["*Conv*"]
      engine returned no support for this graph
      Action: narrow the pattern, add a [[test_skips]] entry if the test is
              actually engine-supported-but-broken, or fix the engine.
```

### 7.4 Implementation sketch

```cpp
struct ClaimFailure
{
    std::string testName;
    std::vector<std::string> matchedSupportedPatterns;
};

class SupportClaimVerifier
{
public:
    std::vector<ClaimFailure> diff(const TestSettings& settings,
                                   std::string_view archRaw,
                                   std::string_view platform,
                                   std::span<const GraphSupportRecord> records) const;

    static void writeReport(std::ostream& out, std::span<const ClaimFailure> failures);
};
```

`TestSettings` gains `findMatchingSupportedPatterns(testName, arch, platform)` analogous to `findSkipForTest`, reusing the existing `globMatch`.

Wired into `main.cpp` after `RUN_ALL_TESTS()`:

```cpp
const int gtestResult = RUN_ALL_TESTS();
int verifyResult = 0;
if(TestConfig::get().enforceSupportClaims())
{
    auto failures = SupportClaimVerifier{}.diff(/* ... */);
    if(!failures.empty())
    {
        SupportClaimVerifier::writeReport(std::cerr, failures);
        verifyResult = 1;
    }
}
return gtestResult | verifyResult;
```

A `TestEventListener` also emits a synthetic failure for xUnit fidelity.

## 8. Pattern Condensation Heuristic

The auto-gen tool sees up to 10K+ observed test names per asic at Full-tier scale and must turn them into a small, safe set of glob patterns to store in the TOML. This section defines the algorithm.

### 8.1 Invariants

Let:
- `S` = set of observed-supported test names for the current `(arch, platform)`.
- `U` = set of observed-unsupported test names for the current `(arch, platform)`.

The tool emits a pattern set `P` satisfying:

1. **Coverage**: For every `t ∈ S`, some `p ∈ P` matches `t`. (Every supported test is claimed.)
2. **Safety**: For every `t ∈ U` and every `p ∈ P`, `p` does *not* match `t`. (No pattern over-matches an unsupported test, which would cause spurious claim-broken failures the moment that test ran.)
3. **Minimality (soft)**: `|P|` is small; coalesce patterns when safe.

Safety is the hard constraint. A pattern that over-matches an unsupported test would fail CI on the very next run — broken-by-construction. The tool refuses to emit such a pattern.

### 8.2 Fixture-based grouping

GTest test names have the form `<Instantiation>/<Fixture>.<TestCase>/<Param>`. The `<Fixture>` segment is the natural unit of support — typically a single `INSTANTIATE_TEST_SUITE_P` produces all instances with one fixture name, and that fixture's support story (all-supported, all-unsupported, or mixed) is uniform.

Initial pattern set: one pattern `*<Fixture>*` per fixture, but only for fixtures whose observed instances are entirely in `S`. Mixed-support fixtures get nothing (emitted to stderr for human decision — see §9).

This per-fixture set is correct (covers `S`, doesn't match `U`) but verbose. Step 2 coalesces it.

### 8.3 Camel-case token coalescing

Fixture names follow PascalCase by convention (per hipDNN test-naming rules — see `CLAUDE.md`): `IntegrationGpuConvFwd2dFp32`, `IntegrationGpuBatchnormFwd`, etc. Split each fixture name on uppercase letters to get a token sequence: `[Integration, Gpu, Conv, Fwd, 2d, Fp32]`.

Build a trie keyed by these token sequences, with each leaf labelled with the fixture's support state. At each internal node, the tool tries to emit a single pattern `*<concatenated-prefix>*` covering all descendant leaves. The trial is **safe** iff:

- All descendant leaves are all-supported fixtures (no mixed, no all-unsupported), AND
- No test in `U` matches `*<concatenated-prefix>*` (recheck against the full unsupported set — including instances of fixtures not in this subtree).

Greedy traversal: start at the root, descend until safe emission becomes possible.

### 8.4 Worked example

Observed-supported fixtures on gfx942 (all all-supported):
- `IntegrationGpuConvFwd2dFp32`, `IntegrationGpuConvFwd2dFp16`, `IntegrationGpuConvFwd3dFp32`
- `IntegrationGpuConvBwd2dFp32`, `IntegrationGpuConvBwd2dFp16`
- `IntegrationGpuConvWgrad2dFp32`
- `IntegrationGpuBatchnormFwd`, `IntegrationGpuBatchnormBwd`
- `IntegrationGpuMatmulFp32`

Observed-unsupported fixtures on gfx942 (engine returned empty):
- (none in this scenario)

Trie root `Integration/Gpu/...`:
- Subtree `Conv/Fwd/...` covers `ConvFwd2dFp32`, `ConvFwd2dFp16`, `ConvFwd3dFp32`. All supported. Coalesce to `*ConvFwd*`. Safety check: nothing in `U` matches `*ConvFwd*` (U is empty). Safe. Emit `*ConvFwd*`.
- Subtree `Conv/Bwd/...` → `*ConvBwd*`. Safe. Emit.
- Subtree `Conv/Wgrad/...` → `*ConvWgrad*`. Safe. Emit.
  - *Further coalescing to `*Conv*`?* Would cover `ConvFwd*`, `ConvBwd*`, `ConvWgrad*`. Safety check: nothing in `U` matches `*Conv*`. Safe — but the per-op-family patterns are already the natural granularity engineers reason about, and `*Conv*` is no shorter than three lines. The tool stops at op-family level by default; further coalescing is a hand edit above the sentinel (§9).
- Subtree `Batchnorm/...` → `*Batchnorm*`. Safe. Emit.
- Subtree `Matmul/...` → `*Matmul*`. Safe. Emit.

Final pattern set:
```toml
patterns = ["*ConvFwd*", "*ConvBwd*", "*ConvWgrad*", "*Batchnorm*", "*Matmul*"]
```

Five patterns, ~200 underlying observed tests covered, zero false matches.

### 8.5 Worked example with carve-outs

Now suppose on gfx10 the engine doesn't have CK fusion kernels — `*ConvFwdBiasActiv*` is observed-unsupported.

Observed-supported fixtures on gfx10:
- `IntegrationGpuConvFwd2dFp32`, `IntegrationGpuConvFwd2dFp16`, `IntegrationGpuConvFwd3dFp32`
- `IntegrationGpuConvBwd2dFp32`, `IntegrationGpuConvBwd2dFp16`
- `IntegrationGpuConvWgrad2dFp32`
- `IntegrationGpuBatchnormFwd`

Observed-unsupported fixtures on gfx10:
- `IntegrationGpuConvFwdBiasActiv2dFp32`, `IntegrationGpuConvFwdBiasActiv2dFp16`

Coalescing now hits a safety wall:
- `*ConvFwd*` matches both `ConvFwd2dFp32` (supported) AND `ConvFwdBiasActiv2dFp32` (unsupported). **Unsafe.** Reject.
- `*ConvFwd2d*` matches `ConvFwd2dFp32`, `ConvFwd2dFp16` (supported) but also `ConvFwdBiasActiv2dFp32` (unsupported because BiasActiv comes after `2d`? Actually no — `ConvFwdBiasActiv2dFp32` has `2d` after `BiasActiv`, so `*ConvFwd2d*` does NOT match it). Safe. Emit `*ConvFwd2d*`.
- Hmm, this depends on naming. Different scheme would change the coalescing.

In practice the tool falls back to per-fixture patterns when coalescing isn't safe:
```toml
patterns = [
    "*ConvFwd2dFp32*",
    "*ConvFwd2dFp16*",
    "*ConvFwd3dFp32*",
    "*ConvBwd*",
    "*ConvWgrad*",
    "*Batchnorm*",
]
# Note: no *ConvFwdBiasActiv* claim — engine returned no support, observed-unsupported.
```

Six patterns instead of five, but every one is provably safe.

### 8.6 Edge cases the algorithm handles

| Case | Handling |
|------|----------|
| Fixture `Conv` appears as a substring of another fixture `Conversion` that's unsupported. | Glob `*Conv*` would match `*Conversion*`; safety check rejects the coalesce. Tool falls back to `*ConvFwd*` / `*ConvBwd*` / etc. |
| Mixed-support fixture (some instances supported, some not). | Tool emits no pattern for the fixture. Lists the supported / unsupported test names on stderr for human decision (narrow pattern hand-written above the sentinel, or add `[[test_skips]]`). |
| All observed tests are unsupported (engine doesn't support this asic at all). | Tool emits an empty `patterns = []` block, which the TOML parser rejects at load. Tool falls back to writing no `[[supported]]` block at all for that asic — verifier treats absence as "not enforced." |
| New fixture observed but no token prefix coalesces (single-fixture branch). | Just emits `*<Fixture>*`. No worse than the per-fixture baseline. |
| Existing pattern above the sentinel already covers some fixtures the tool would emit below. | Tool still emits the pattern below — both regions union at load. Engineer can remove the redundant tool-emitted pattern during diff review if they want, but it doesn't break correctness. |

### 8.7 C++ implementation sketch

The algorithm is straightforward to implement in C++ using `std::string` and a simple recursive tree:

```cpp
struct FixtureNode
{
    std::map<std::string, FixtureNode> children;  // token → child
    std::vector<std::string> supportedTests;      // tests at this leaf
    std::vector<std::string> unsupportedTests;    // tests at this leaf
};

// Build trie from observed records, splitting fixture names by camel-case.
FixtureNode buildTrie(std::span<const GraphSupportRecord> records);

// Recursively emit minimal safe patterns.
std::vector<std::string> emitPatterns(const FixtureNode& root,
                                      const std::set<std::string>& globalUnsupported);
```

Camel-case splitting: walk the string, start a new token at every uppercase letter that follows a lowercase letter or digit (handles `Fp32`, `2d`, etc. as separate tokens).

Safety check: for each candidate pattern, run `globMatch(pattern, t)` against every test in `globalUnsupported`; reject on any match.

No external dependencies — the trie and matching all use `std::string` / `std::map`. Reuses the existing `globMatch` from `TestSettings`.

### 8.8 What the algorithm does *not* try to do

- **Cross-fixture super-coalescing** (e.g. `*` matching everything). Too risky against future test additions; the per-op-family granularity (`*ConvFwd*`, `*Matmul*`, etc.) is the design target.
- **Negative lookahead globs** (e.g. `*Conv*` minus `*ConvFwdBiasActiv*`). TOML glob syntax doesn't support it, and adding a separate exclusion list would reintroduce the `[[unsupported]]` section we deliberately removed (§6.4).
- **Reasoning about fixture naming intent.** The algorithm operates purely on observed support sets; it doesn't know that `Conv` and `Conversion` are unrelated. The safety check is what protects against confusion.

## 9. Auto-Generation Tool

### 9.1 CLI

```bash
./hipdnn_integration_tests \
    --test-config dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --write-support-claims
```

Runs the full suite (errors out if `--gtest_filter` is set), observes per-test support via `SupportMatrixCollector`, condenses observed support to safe patterns (§8), and rewrites the `[[supported]]` block for the current `(arch, platform)` in the TOML. Other asics' blocks are untouched.

### 9.2 Sentinel-delimited machine-managed region

The tool divides the TOML into two regions using a sentinel comment line:

```toml
[meta]
version = 1

# Hand-edited content above the sentinel: [[tolerance_overrides]], [[test_skips]],
# any hand-curated [[supported]] blocks with their own explanatory comments.

[[test_skips]]
archs   = ["gfx90a"]
filters = ["*ConvFwdBiasActiv*"]
reason  = "Engine returns wrong results — ROCm/rocm-libraries#6979"

# ─── auto-generated: do not edit below this line ─────────────────────
# Regenerate per-arch with: ninja miopen-provider-write-support-claims
# ─────────────────────────────────────────────────────────────────────

[[supported]]
arch     = "gfx942"
patterns = ["*ConvFwd*", "*ConvBwd*", ...]
```

**Above sentinel**: hand-edited. May contain any section kind with arbitrary comments. **Never modified by the tool.**

**Below sentinel**: machine-managed. Contains only auto-generated `[[supported]]` blocks. Engineer edits here are overwritten on next tool run.

**At load time**: `TestSettings` reads the whole file as one document; both regions contribute. Within `[[supported]]`, no `(arch, platform)` can appear in both regions — the load-time uniqueness check (§6.2) catches duplicates.

### 9.3 Single-stage C++ rewrite

Text-level surgery on the head, `tomlplusplus` round-trip on the tail:

1. Run the full suite via the existing harness; observe per-test support via `SupportMatrixCollector`.
2. Read the TOML file as text. Locate the sentinel line.
3. **Sentinel absent** → append sentinel + freshly-generated `[[supported]]` block for the current `(arch, platform)`. Write file. Done.
4. **Sentinel present** →
   - Take everything from start-of-file through the sentinel block as **head text**, untouched.
   - Parse the below-sentinel region with `tomlplusplus`. Drop any `[[supported]]` block whose `arch` and `platform` exactly match the current run. Keep all other-asic blocks verbatim.
   - Generate the new block for the current `(arch, platform)` from §8's algorithm.
   - Serialise the combined below-sentinel content via `tomlplusplus` in its canonical style.
   - Write `head_text` + `\n` + `serialised_tail`.

The above-sentinel section is preserved byte-for-byte: every multi-paragraph comment, every blank line, every section ordering decision is intact because the tool never round-trips it. The below-sentinel section gets whatever `tomlplusplus` emits (expanded tables, deterministic key order) — that's fine because the sentinel says "do not edit."

Only dependency: `tomlplusplus`, which the project already uses on the read path. No Python, no JSON sidecar, no cross-language data passing.

### 9.4 Mixed-fixture stderr output

```
Fixture IntegrationGpuConvFwdBiasActiv has mixed support (3 supported / 5 unsupported):
  supported:
    - Smoke/.../NCHW_1x16x16x16_1x16x3x3
    - Smoke/.../NHWC_1x16x16x16_1x16x3x3
    - Standard/.../NCHW_2x32x14x14_1x32x1x1
  unsupported:
    - Smoke/.../NCHW_special_case_1
    - Smoke/.../NCHW_special_case_2
    ...
  Decide (above the sentinel so the tool doesn't overwrite):
    - Add a narrower [[supported]] pattern (e.g. *ConvFwdBiasActiv*NCHW_1x*), OR
    - Add [[test_skips]] entries with a reason if the unsupported tests are
      actually engine-supported-but-broken
```

The "above the sentinel" instruction is critical: hand-curated narrow patterns must live in the hand-edited region or they'll be wiped on the next tool run.

### 9.5 Failure modes the tool detects up front

The tool refuses to run (clear error, no file write) if:

- `--gtest_filter` is set — partial-run baseline would be wrong.
- The sentinel line is present but malformed (truncated, duplicated).
- The below-sentinel region contains anything other than `[[supported]]` blocks. Catches the case where someone hand-edited the machine region and the tool would otherwise silently delete their work.

### 9.6 Build integration

Two surfaces:

- **CLI flag** on the integration test binary (`--write-support-claims`) — fits the existing flag family (`--test-config`, `--generate-support-matrix`).
- **CMake/ctest target** per provider (e.g. `ninja miopen-provider-write-support-claims`) — discoverable, idiomatic, runs the binary with the right `--test-config` baked in.

## 10. Workflow

### 10.1 Day-to-day flow (most PRs)

1. PR author lands a code change.
2. CI runs `hipdnn_integration_tests --enforce-support-claims ...`.
3. If a `[[supported]]` claim is broken, CI fails with a report listing every claim-broken test (§7.3).
4. Author edits the TOML:
   - Real regression → fix the code.
   - Pattern is too broad → narrow it (above the sentinel for persistence), or regenerate with `--write-support-claims` for that asic.
   - Test is engine-supported-but-broken → add to `[[test_skips]]` with a reason.
5. Re-run CI; failure clears.

### 10.2 Bootstrap (new engine or new asic)

1. Engineer runs `ninja <provider>-write-support-claims` on the target hardware (the asic they have access to).
2. Tool generates a `[[supported]]` block for the current `(arch, platform)` with patterns produced by §8. Mixed-support fixtures are listed on stderr.
3. Engineer reviews the `git diff` of the machine-managed region. Adds `[[test_skips]]` entries above the sentinel for mixed-fixture tests that are bugs to track. Optionally hand-coalesces patterns above the sentinel.
4. Commits the TOML; first CI run on that asic goes green.

Other asics are unaffected — their blocks (if any) are untouched.

### 10.3 Refresh (existing engine, asic-specific code change)

Same as bootstrap but the engineer reads the `git diff` more carefully for the asic they regenerated — removed or added patterns reflect what the auto-gen sees today vs. what was previously declared. Anything unexpected gets investigated, not auto-accepted. Other asics' blocks are untouched in the diff.

## 11. CI Integration

- **Pre-submit (Smoke):** `--enforce-support-claims`. Catches claim regressions on the fast tier.
- **Post-submit (Standard/Comprehensive/Full):** `--enforce-support-claims`. Wider observed set → more tests evaluated against patterns.
- **Update PRs:** Reviewers expect a TOML diff with any code change that legitimately moves engine support. PRs without the update fail.
- **THEROCK plugin filtering** (RFC 0006 §8.1) is orthogonal: an unbuilt plugin's TOML is not loaded.
- **CI never runs `--write-support-claims`.** The tool is engineer-driven; auto-applying it in CI would silently rewrite the contract.

## 12. Alternatives Considered

### 12.1 Flat per-instance test-name list

`tests = ["Suite/Fixture.Case/Param", ...]` enumerating every supported instance.

- Pros: `git diff` shows exact name changes.
- Cons: Doesn't scale (10K+ instances per asic); swamps PR diffs; unreviewable.

**Rejected** in favour of patterns.

### 12.2 Per-(suite, instantiation) counts and/or digests

One TOML row per `INSTANTIATE_TEST_SUITE_P` recording counts and digests.

- Pros: Compact; catches per-test flips.
- Cons: Snapshot, not contract — counts decay silently as catalogs grow; breaks on partial local runs.

**Rejected** in favour of patterns.

### 12.3 Coverage floors / `min_count`

Per-pattern minimum-instance counts to catch catalog shrinkage.

- Pros: Catches accidental deletion of an `INSTANTIATE_TEST_SUITE_P`.
- Cons: Stale by construction; brittle on partial runs.

**Rejected.** Catalog shrink is intentionally out of scope (§3.2).

### 12.4 Separate `[[unsupported]]` section

Positive negative-claim with optional `reason`, plus a fourth failure bucket ("documented gap closed").

- Pros: Symmetric capability tracking; "gap closed" event surfaces loud.
- Cons: Doubles TOML maintenance — every op family classified positively *or* negatively. Forces TOML updates whenever engine capability grows. Conflicts directly with the low-maintenance goal (§3.2).

**Rejected.** Absence from `[[supported]]` is the unsupported state.

### 12.5 Default-deny on unclaimed gain

Tests not matching any `[[supported]]` pattern that turn out to be supported → FAIL.

- Pros: Keeps the TOML accurate as a complete capability map.
- Cons: Every new test family or op that the engine happens to support forces a TOML update across every asic block. Bring-up friction; PR churn; discourages adding tests.

**Rejected** for low-maintenance reasons. The TOML is intentional contracts, not capability inventory.

### 12.6 Multi-asic `[[supported]]` blocks

`archs = ["gfx942", "gfx90a"]` to share patterns across asics.

- Pros: Less duplication in the file.
- Cons: An engineer fixing gfx942 has to know the current truth on gfx90a to not damage that asic's block. Couples updates that should be independent.

**Rejected** in favour of one block per asic. Sharing is by duplication.

### 12.7 Sidecar baseline file per engine

Put claims in a separate file from skips/tolerances.

- Pros: Cleaner human/machine separation.
- Cons: Two files instead of one; engineer can forget to commit both. Sentinel-delimited single file (§9.2) gives the same human/machine separation without the extra file.

**Rejected.**

### 12.8 Python-based TOML round-trip

`tomlkit` in a Python helper for comment-preserving rewrites.

- Pros: Free comment preservation in the full file.
- Cons: Cross-language data passing; second dependency; two-step process the engineer can forget. The sentinel approach (§9.2) preserves human comments byte-for-byte without needing format-preserving round-trip in the machine region.

**Rejected** in favour of single-language C++.

### 12.9 Hash digest only

One SHA over the entire sorted supported set per asic.

- Pros: Smallest possible file.
- Cons: Zero diagnostic value; no per-test action list; enforces snapshot not contract.

**Rejected.**

### 12.10 Reuse `--generate-support-matrix` markdown as baseline

Diff the markdown matrix between runs.

- Pros: No new format.
- Cons: Markdown is aggregated; regressions inside a group disappear; formatted for humans, diffs poorly.

**Rejected.**

## 13. Risks

| Risk | Mitigation |
|------|------------|
| A pattern accidentally over-matches (e.g. `*Conv*` matches an unrelated `*Conversion*` test). | §8's safety check refuses to emit such patterns from auto-gen; hand-written patterns above the sentinel are the engineer's responsibility, surfaced as a verifier failure if they over-match an observed-unsupported test. |
| A pattern matches no observed test (typo, removed op). | Informational stderr line — visible in CI logs but not a failure (partial runs would spam). Strict mode can be added later. |
| First run on a new asic fails because no `[[supported]]` block exists. | Verifier treats "no matching block for this `(arch, platform)`" as **not enforced**. Bring-up is unblocked; engineer runs `--write-support-claims` once and commits. |
| Auto-gen wholesale-replace blows away patterns on re-run. | Sentinel split (§9.2): hand-curated narrow patterns belong *above* the sentinel and survive every tool run. Only the machine-managed region below is overwritten. |
| Engineer hand-edits the machine-managed region; tool silently wipes it. | Tool refuses to run (§9.5) if the below-sentinel region contains anything other than `[[supported]]` blocks. Forces operator intervention. |
| Glob patterns depend on the test-naming convention. | Project already enforces PascalCase fixture names and `byTag()`; reuses existing `globMatch` from `[[test_skips]]` / `[[tolerance_overrides]]`. |
| Engine grows new capability that no `[[supported]]` pattern covers. | Deliberately silent (§3.2). Engineer who wants to claim the new capability adds a pattern in their next PR; the system doesn't force the update. |

## 14. Open Questions and Future Work

- **Multi-engine claims.** When a build loads multiple engines (RFC 0006 §6), each engine's TOML is independent; the verifier needs to attribute `supportingEngineIds` per file. Likely a `[meta] engine = "MIOPEN_ENGINE"` field. Out of scope here.
- **Detecting obsolete `[[test_skips]]` entries.** A `detect_obsolete = true` flag could trigger an offline check that builds the graph for each skipped test and queries engine support without executing. If support is returned the skip rule has outlived its reason. Adds value once skip-rule rot becomes a real maintenance burden.
- **Coalescing patterns inside the machine region.** §8 stops coalescing at op-family granularity. Aggressive coalescing (e.g. up to `*Conv*` when safe) is straightforward to add when output verbosity is felt.
- **Strict mode for unmatched patterns.** Promote "pattern matched no observed tests" from info to error when no `--gtest_filter` is set.
- **Catalog-shrink detection.** Intentionally out of scope (§3.2). If ever needed, a `[[coverage_floors]]` section with auto-regenerated counts could be added without polluting contract-level claims.

## 15. Glossary

- **Engine.** A plugin-provided implementation that can execute a graph. Reported by `hipdnnGetEngineCount_ext` / `getEngineInfo`.
- **Supported (for a test).** The engine returned at least one entry from `Graph::get_ranked_engine_ids()`.
- **Pre-skipped.** A test removed by a matching `[[test_skips]]` entry during `SetUp()`. Excluded from claim evaluation.
- **Observed.** A test that reached `SupportMatrixCollector::recordGraphSupport` during this run (not pre-skipped, not filtered out, in a built binary).
- **Claimed.** A test matches at least one `[[supported]]` pattern for the current `(arch, platform)`.
- **Supported claim broken.** Observed test is claimed but engine returned no support. The only failure mode.
- **Sentinel.** Marker comment line in the TOML separating the hand-edited region (above) from the machine-managed region (below). The auto-gen tool only touches below.
- **Safe pattern** (auto-gen). A glob that matches no test in the observed-unsupported set for the current `(arch, platform)`. The condensation algorithm (§8) emits only safe patterns.
