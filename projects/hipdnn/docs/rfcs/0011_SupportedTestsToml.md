# hipDNN - Engine Support Claims in Per-Engine TOML

- Contributors: TBD
- **Status**: Draft
- **Implementation Version**: TBD
- **Jira**: [ALMIOPEN-1918](https://amd-hub.atlassian.net/browse/ALMIOPEN-1918)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Current System Overview](#3-current-system-overview)
4. [Proposed Design](#4-proposed-design)
5. [TOML Schema](#5-toml-schema)
6. [Failure Detection](#6-failure-detection)
7. [Auto-Generation Tool](#7-auto-generation-tool)
8. [Workflow](#8-workflow)
9. [CI Integration](#9-ci-integration)
10. [Alternatives Considered](#10-alternatives-considered)
11. [Risks](#11-risks)
12. [Open Questions and Future Work](#12-open-questions-and-future-work)
13. [Glossary](#13-glossary)

## 1. Executive Summary

This RFC proposes adding two new sections to each per-engine integration-test TOML (e.g. `dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml`):

- **`[[supported]]`** — glob patterns asserting *"every test matching these patterns must be supported by this engine on this arch/platform."*
- **`[[unsupported]]`** — glob patterns asserting *"every test matching these patterns must NOT be supported."* Optional `reason` for the cases worth documenting; most entries are just "engine doesn't implement this" and don't need one.

The existing `[[test_skips]]` section keeps its current behaviour and its required `reason` field. Its semantics are clarified: it is for tests the engine **does** support but that are currently broken (wrong results, crashes, bugs). The reason is high-signal — it points at the active issue. Skipped tests are pre-empted in `SetUp()` and excluded from claim evaluation.

At the end of each integration-test run the harness compares actual engine support against the claims and fails on four specific deltas:

- **Supported-claim broken**: a test matches `[[supported]]` but the engine returned no support.
- **Unsupported-claim broken**: a test matches `[[unsupported]]` but the engine returned support (the documented gap closed; remove the entry and probably add a `[[supported]]` pattern).
- **Unclaimed gain**: a test was supported by the engine but no claim covers it (default-deny on undeclared support — the engineer must explicitly account for it).
- **Config conflict**: a test matches both `[[supported]]` and `[[unsupported]]`. Rejected at TOML load.

The model is **pattern-based, not enumeration-based**. There are no per-test name lists, no per-group counts, no digests. The TOML stays tiny — typically tens of lines per engine — while the verifier still checks every individual registered test against the claims. A bundled auto-generation tool (`--write-support-claims`) produces a complete starter classification from observed support so the engineer reviews via `git diff` rather than writing from scratch.

## 2. Problem Statement

The integration suite already has the data it needs to know which engines support which graphs — `SupportMatrixCollector` records `(graphName, graphDescription, testName, supportingEngineIds)` for every test and emits a markdown matrix when `--generate-support-matrix` is passed (see `dnn-providers/integration-tests/src/harness/SupportMatrixCollector.hpp`). But this output is:

1. **Advisory only.** The matrix is regenerated on demand; nothing fails if a row flips between runs.
2. **Markdown.** Optimised for human reading and grouped/aggregated by op, not for diffable regression detection.
3. **Not arch-aware.** Engine support varies across arches.
4. **Conflates two distinct things.** `[[test_skips]]` today covers both "engine has no kernel" and "engine supports but is buggy." The current MIOpen TOML's CBA entry mixes both: gfx90a "returns wrong results" (bug = skip) while RDNA archs "have no CK fusion kernels" (unsupported = unsupported).

As a result the failure modes the team cares about are silent:

- **Supported regression.** MIOpen stops accepting a previously-supported convolution shape on gfx942 → harness `GTEST_SKIP`s and CI passes.
- **Documented gap closes.** MIOpen ships a CK fusion kernel for gfx10 → CBA test now runs and is supported, but the skip rule still hides it; no one knows.
- **Undocumented capability.** A new `SuperCoolOp` test family lands; MIOpen happens to support it; engineer never notices and the TOML never reflects it.

The TOML should encode engine intent on every arch — "supports these, explicitly doesn't support those, skip these because they're broken" — and the verifier should fail the moment reality drifts from that intent in any direction.

## 3. Current System Overview

### 3.1 What the suite already records

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

Recording happens **after** `[[test_skips]]` has already short-circuited the test in `SetUp()`. So the records correspond to "tests not pre-skipped on this arch/platform" — exactly the population the verifier wants.

### 3.2 TOML config today

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

### 3.3 Where the new check fits

| Component | Existing role | New role |
|-----------|--------------|----------|
| `TestSettings` | Parses tolerance + skips | Also parses `[[supported]]` and `[[unsupported]]` |
| `SupportMatrixCollector` | Records and emits markdown | Records also feed the new verifier and the auto-gen tool |
| `main.cpp` | Owns final exit code | Runs the verifier after `RUN_ALL_TESTS()` and fails on drift; hosts `--write-support-claims` |

## 4. Proposed Design

### 4.1 Three orthogonal sections

| Section | Semantic | Engine response | Test runs? | `reason`? |
|---------|----------|-----------------|------------|-----------|
| `[[supported]]` | Engine **must** support matching tests | non-empty `supportingEngineIds` | yes | no — the contract is the documentation |
| `[[unsupported]]` | Engine **must not** support matching tests | empty `supportingEngineIds` | yes (and harness `GTEST_SKIP`s because no engine supports it) | optional — most entries are "engine doesn't implement this"; only document the cases worth it |
| `[[test_skips]]` | Don't run matching tests (broken / buggy / unsafe — engine *does* claim support) | not checked | no — skipped in `SetUp()` | **required** — every skip is an active issue; the reason names it |

The asymmetry is intentional: `[[test_skips]]` reasons are high-signal (every entry points at an active bug or workaround), while most `[[unsupported]]` entries would just say "not implemented." Making `[[unsupported]]` reasons optional keeps the meaningful ones (e.g. the RDNA CBA case with its issue link) prominent rather than drowning them in `reason = "not implemented"` boilerplate.

### 4.2 Goals

1. **Single source of truth per engine.** All three sections live in the same TOML.
2. **Tiny by construction.** Patterns scale with the number of distinct support stories (handful per engine), not the number of test instances (10K+ at Full-tier scale).
3. **Per-test failure granularity.** The verifier checks every registered test against the patterns and reports per-test failures.
4. **Symmetric detection.** Every direction of drift fails: support lost, support gained in a documented gap, support gained without any claim.
5. **No staleness.** Patterns absorb catalog growth and shrinkage automatically.
6. **Graceful local degradation.** Partial runs are checked over only what was observed.
7. **Discoverable auto-gen.** A bundled tool produces a complete starter classification via `git diff`-able TOML edits.
8. **Backward compatible.** Engines without `[[supported]]`/`[[unsupported]]` blocks behave exactly as today.

### 4.3 Non-goals

- **Catalog-shrink detection.** If an `INSTANTIATE_TEST_SUITE_P` is deleted, the verifier won't notice — that's a code-review concern (PR diff is the right place). Deliberately out of scope; see §10.3.
- **Per-test enumeration.** The TOML never lists individual test names.
- **Multi-engine claims.** Each plugin's TOML is independent.
- **Changes to `--generate-support-matrix`.** The markdown matrix is unchanged.

### 4.4 Data flow

```
       ┌───────────────────────────────────────────────────────────┐
       │  per-engine TOML (committed)                              │
       │  [[supported]]    archs=[...]  patterns=[...]             │
       │  [[unsupported]]  archs=[...]  patterns=[...]  reason=... │
       │  [[test_skips]]   archs=[...]  filters=[...]   reason=... │
       └────────────────────────────┬──────────────────────────────┘
                                    │ load
                                    ▼
   test run ──► SupportMatrixCollector ──► SupportClaimVerifier ──► diff report
                (existing)                  (new)                       │
                                                                        ▼
                                                          fail run on any
                                                          claim-broken or
                                                          unclaimed-gain entry

   `hipdnn_integration_tests --write-support-claims` (offline):
       runs suite → rewrites only the machine-managed region below the
                    TOML sentinel comment → engineer reviews `git diff`
```

## 5. TOML Schema

```toml
[meta]
version = 1

# Existing.
[[tolerance_overrides]]
filters = ["..."]
atol = 0.002
rtol = 0.012

# NEW: positive claims. Every observed, non-skipped test matching any pattern
# must have at least one supporting engine on the matching arch/platform.
[[supported]]
archs    = ["gfx942"]            # required; substring match against gcnArchName
# platforms = ["linux"]          # optional; exact match against current platform
patterns = [
    "*ConvFwd*",
    "*ConvBwd*",
    "*ConvWgrad*",
    "*Batchnorm*",
    "*Matmul*",
]

# NEW: negative claims. Every observed, non-skipped test matching any pattern
# must have NO supporting engine. Reason optional — include for the cases worth
# documenting, omit for routine "engine doesn't implement this."
[[unsupported]]
archs    = ["gfx10", "gfx11", "gfx12"]
patterns = ["*ConvFwdBiasActiv*"]
reason   = "No CK fusion kernels on RDNA — ROCm/rocm-libraries#6979"

# Most [[unsupported]] entries look like this — no reason, the absence of
# the op in the engine is its own documentation:
[[unsupported]]
archs    = ["gfx942"]
patterns = ["*Sdpa*", "*Attention*"]

# Existing — now strictly "engine supports it but it's broken / would crash /
# gives wrong answers." Test is pre-empted in SetUp and excluded from claim eval.
[[test_skips]]
archs   = ["gfx90a"]
filters = ["*ConvFwdBiasActiv*"]
reason  = "Engine returns wrong results — ROCm/rocm-libraries#6979"
```

### 5.1 Field semantics

For both `[[supported]]` and `[[unsupported]]`:

- `archs` (**required**). List of substrings matched against `TestConfig::getCurrentArch()` (raw gcnArchName like `"gfx942:sramecc+:xnack-"`). Required because support genuinely differs by arch. Same matching rules as `[[test_skips]]`.
- `platforms` (optional, default = any). Exact match against `TestConfig::getCurrentPlatform()` (`"windows"` or `"linux"`).
- `patterns` (**required**, non-empty). GTest-style globs (`*` wildcard) matched against the fully-qualified GTest test name. Reuses `globMatch` from `TestSettings`.

`[[unsupported]]` accepts an **optional** `reason` string. Use it when the gap has a specific story worth pinning down (e.g. "no CK fusion kernels on RDNA, tracked in #6979"). Omit it when the truth is just "engine doesn't implement this op family" — that's already evident from the absence of a corresponding `[[supported]]` claim, and writing `reason = "not implemented"` on every entry adds noise that buries the meaningful ones. `[[supported]]` has no `reason` field at all — the contract is the documentation.

### 5.2 Matching algorithm

All `[[supported]]` and `[[unsupported]]` blocks whose `archs`/`platforms` match the current run are unioned into two effective pattern sets — `claimedSupportedPatterns` and `claimedUnsupportedPatterns`. A test is *claimed-supported* if any pattern in the supported set matches; *claimed-unsupported* if any pattern in the unsupported set matches.

If any observed test matches both sets simultaneously, the TOML has a conflict; the verifier emits an error and the run fails. (This can also be caught at load time if the pattern sets statically overlap; left as an implementation detail.)

### 5.3 Why patterns, not enumeration

A flat list of fully-qualified test names doesn't scale (10K+ instances at Full-tier maturity). A per-`(suite, instantiation)` summary with counts/digests goes stale (counts decay silently as catalogs grow, break on partial local runs). Patterns sidestep both: the TOML records the *contract* (the engine claims to support this pattern) rather than the *current snapshot*. Catalog growth and shrinkage are absorbed automatically.

### 5.4 Why default-deny on unclaimed-but-supported

When a new test family lands (say `SuperCoolOp`) and the engine happens to support it, "default-not-enforced" means the engineer never finds out — the TOML stops being an accurate capability map. Default-deny forces explicit acknowledgement: add a `[[supported]]` pattern (intentional support, one-line PR), or a `[[test_skips]]` entry if running it is unsafe, or an `[[unsupported]]` entry if the support was accidental and should be locked off. The friction is small; discoverability is the whole point.

### 5.5 Why three sections, not two

An earlier draft folded the "engine doesn't support" case into `[[test_skips]]` (the skip entry's `reason` doubled as documentation of the gap). That conflated two distinct things:

- **Engine supports it but it's broken.** The test would crash or give wrong answers; we skip to keep CI green while we fix it. Skip should fire.
- **Engine has no kernel / no claim of support.** The test is pointless to run because `get_ranked_engine_ids` returns empty; the harness already `GTEST_SKIP`s. We want to *assert* this state so a future "engine started supporting it" event is loud.

These have different remediations (fix the bug vs. add a kernel), different lifetimes (skip is usually transient, unsupported is usually long-term), and different signals on "the engine got better" (a skip becomes obsolete and you may not notice; an `[[unsupported]]` becomes a verifier failure that screams).

Separating them gives the failure output natural language ("a documented gap has closed; remove the `[[unsupported]]` entry") and keeps `[[test_skips]]` reasons honest ("we know we have a bug").

## 6. Failure Detection

### 6.1 Per-test classification

For each observed test (registered, not pre-skipped, reached `recordGraphSupport`):

| `claimed_supported` | `claimed_unsupported` | `actual_support` | Bucket | Severity |
|:---:|:---:|:---:|--------|:--------:|
| ✓ | – | ✓ | Supported claim honoured | pass |
| ✓ | – | ✗ | **Supported claim broken** | FAIL |
| – | ✓ | ✗ | Unsupported claim honoured | pass |
| – | ✓ | ✓ | **Unsupported claim broken** (gap closed) | FAIL |
| – | – | ✗ | Default-unsupported (ok) | pass |
| – | – | ✓ | **Unclaimed gain** | FAIL |
| ✓ | ✓ | * | **Config conflict** | FAIL |

Pre-skipped tests don't reach `recordGraphSupport` and so are naturally excluded from all four failure buckets. `[[test_skips]]` remains the explicit "don't enforce" knob for individual tests.

### 6.2 Local partial runs

Tests never observed (filtered out, in an unbuilt binary, in an unselected tier) contribute nothing. The diff covers only `recordGraphSupport` entries, so a partial run is checked against the partial set it exercised. CI runs full → enforces all; local filtered runs → enforces what they ran.

A `[[supported]]` or `[[unsupported]]` block whose patterns match no observed test gets an informational stderr line ("claim X matched no observed tests") but does **not** fail — partial-run friendliness.

### 6.3 Example failure output

```
[SUPPORT CLAIMS] arch=gfx942 platform=linux: 3 failures.

  SUPPORTED CLAIM BROKEN:
    Smoke/IntegrationGpuConvFwdBiasActiv.Correctness/NCHW_1x16x16x16_1x16x3x3
      matched supported pattern(s): ["*Conv*"]
      engine returned no support for this graph
      Action: narrow the pattern, add a [[test_skips]] (if broken),
              or add an [[unsupported]] entry with reason (if no kernel).

  UNSUPPORTED CLAIM BROKEN (documented gap has closed):
    Standard/IntegrationGpuConvFwdBiasActiv.Correctness/NCHW_2x32x14x14_1x32x1x1
      matched unsupported pattern(s): ["*ConvFwdBiasActiv*"]
        from block: archs=["gfx10","gfx11","gfx12"]  reason="No CK fusion kernels on RDNA"
      engine MIOPEN_ENGINE now returns support
      Action: remove the [[unsupported]] entry, then add a [[supported]]
              pattern covering this test (or narrow the existing one).

  UNCLAIMED GAIN:
    Smoke/IntegrationGpuSuperCoolOpFp32.Correctness/NCHW_1x16x32x32
      engine MIOPEN_ENGINE returned support
      no [[supported]] or [[unsupported]] pattern matches this test
      Action: add a [[supported]] pattern (e.g. "*SuperCoolOp*") if intentional,
              or [[unsupported]] / [[test_skips]] with a reason if not.
```

### 6.4 Implementation sketch

```cpp
struct ClaimFailure
{
    enum Kind {
        SupportedClaimBroken,
        UnsupportedClaimBroken,
        UnclaimedGain,
        ConfigConflict
    } kind;
    std::string testName;
    std::vector<std::string> matchedSupportedPatterns;
    std::vector<std::string> matchedUnsupportedPatterns;
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

`TestSettings` gains `findMatchingSupportedPatterns(testName, arch, platform)` and `findMatchingUnsupportedPatterns(testName, arch, platform)` helpers, both analogous to `findSkipForTest` and reusing the existing `globMatch`.

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

### 6.5 Why fail-on-gain (in either direction)

Symmetric detection isn't aesthetic — it's the property that keeps the TOML honest. If "engine gained support somewhere unexpected" is silent, the TOML drifts out of sync with reality and stops being trustworthy for capability review. One failure shape ("you need to update the TOML, here's what changed") with a small number of resolutions (add `[[supported]]`, add `[[unsupported]]`, add `[[test_skips]]`, narrow a pattern) keeps the engineer's workflow predictable.

## 7. Auto-Generation Tool

The TOML is hand-edited in the small, but writing 50 classified test groups from scratch is tedious and error-prone. `--write-support-claims` is a single C++ command that observes runtime support and writes the `[[supported]]` / `[[unsupported]]` blocks for the current `(arch, platform)` directly into the engine's TOML. The engineer reviews via `git diff`.

### 7.1 CLI

```bash
./hipdnn_integration_tests \
    --test-config dnn-providers/miopen-provider/config/MIOPEN_ENGINE.toml \
    --write-support-claims
```

Errors out if `--gtest_filter` is set — partial-run output would write a bogus baseline.

### 7.2 Sentinel-delimited machine-managed region

The tool divides the TOML into two regions using a sentinel comment line:

```toml
[meta]
version = 1

# Hand-edited content (tolerance overrides, test skips, hand-curated
# [[supported]]/[[unsupported]] blocks with their own explanatory comments)
# lives above the sentinel.

[[test_skips]]
archs   = ["gfx90a"]
filters = ["*ConvFwdBiasActiv*"]
reason  = "Engine returns wrong results — ROCm/rocm-libraries#6979"

# ─── auto-generated: do not edit below this line ─────────────────────
# Regenerate with: ninja miopen-provider-write-support-claims
# ─────────────────────────────────────────────────────────────────────

[[supported]]
archs    = ["gfx942"]
patterns = ["*ConvFwd*", "*ConvBwd*", "*Batchnorm*", "*Matmul*"]

[[unsupported]]
archs    = ["gfx942"]
patterns = ["*Sdpa*", "*Attention*"]

# ... more auto-generated blocks per arch ...
```

**Above sentinel**: hand-edited. May contain any of the four section kinds (`[[tolerance_overrides]]`, `[[test_skips]]`, `[[supported]]`, `[[unsupported]]`) in any order with arbitrary comments. **Never modified by the tool.**

**Below sentinel**: machine-managed. Contains only auto-generated `[[supported]]` and `[[unsupported]]` blocks. Engineer edits here will be overwritten on next tool run.

**At load time**: `TestSettings` reads the whole file as one document; both regions contribute patterns and they're unioned per `(arch, platform)`. The verifier doesn't know or care about the sentinel.

### 7.3 Single-stage C++ rewrite

The tool flow is text-level on the head, TOML-level on the tail:

1. Run the full suite via the existing harness; observe per-test support via `SupportMatrixCollector`.
2. Read the TOML file as text. Locate the sentinel line.
3. **Sentinel absent** → append sentinel + freshly-generated blocks for the current `(arch, platform)`. Write file. Done.
4. **Sentinel present** →
   - Take everything from start-of-file through the sentinel block as **head text**, untouched.
   - Parse the below-sentinel region with `tomlplusplus`. Drop every `[[supported]]` / `[[unsupported]]` block whose `archs`/`platforms` exactly match the current run. Keep all other-arch blocks verbatim.
   - Generate new blocks for the current `(arch, platform)` from observed support (§7.4).
   - Serialise the combined below-sentinel content via `tomlplusplus` in its canonical style.
   - Write `head_text` + `\n` + `serialised_tail`.

The above-sentinel section is preserved byte-for-byte: every multi-paragraph comment, every blank-line choice, every section ordering decision the engineer made is intact because the tool never round-trips it. The below-sentinel section gets whatever `tomlplusplus` emits (expanded tables, deterministic key order) — that's fine because the sentinel explicitly says "do not edit."

Only dependency: `tomlplusplus`, which the project already uses on the read path. No Python, no JSON sidecar, no cross-language data passing.

### 7.4 Per-fixture heuristic

For each fixture (the `<Fixture>` part between `<Instantiation>/` and `.` in GTest names):

| Fixture state | Tool action |
|---------------|-------------|
| All observed tests **supported** | Add `*<Fixture>*` to `[[supported]]` patterns |
| All observed tests **unsupported** | Add `*<Fixture>*` to `[[unsupported]]` patterns (no `reason` — engineer adds one during review if the gap has a specific story) |
| **Mixed** | Write nothing for this fixture; emit to stderr as human-decision item |

Generated `[[unsupported]]` entries deliberately omit `reason`. Most gaps are routine ("engine doesn't implement this op") and don't need annotation; the engineer adds a `reason` during `git diff` review for the meaningful cases. Auto-writing `reason = "TODO"` placeholders was considered and rejected — TODO markers would proliferate across every generated entry and lose meaning quickly.

**Mixed-support stderr output:**

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
  Decide:
    - Narrow [[supported]] pattern to the supported subset (above sentinel), OR
    - Add narrower [[unsupported]] patterns for the unsupported subset (above sentinel, reason optional), OR
    - Add [[test_skips]] entries with a reason if the unsupported tests are actually engine-supported-but-broken
```

The "above sentinel" parentheticals matter: hand-curated narrow patterns belong in the hand-edited region so the tool doesn't overwrite them on the next run.

No coalescing in v1 (e.g. `*Conv*` covering `*ConvFwd*` + `*ConvBwd*` + …). The output is one pattern per fixture; the engineer coalesces in the hand-edited region above the sentinel if desired. Coalescing inside the machine-managed region is a future enhancement.

### 7.3 Heuristic

For each fixture (the `<Fixture>` part between `<Instantiation>/` and `.` in GTest names):

| Fixture state | Tool action |
|---------------|-------------|
| All observed tests **supported** | Add `*<Fixture>*` to `[[supported]]` patterns |
| All observed tests **unsupported** | Add `*<Fixture>*` to `[[unsupported]]` patterns (no `reason` — engineer adds one during review if the gap has a specific story) |
| **Mixed** | Write nothing for this fixture; emit to stderr as human-decision item |

Generated `[[unsupported]]` entries deliberately omit `reason`. Most gaps are routine ("engine doesn't implement this op") and don't need annotation; the engineer adds a `reason` during `git diff` review for the meaningful cases. Auto-writing `reason = "TODO"` placeholders was considered and rejected — TODO markers would proliferate across every generated entry and lose meaning quickly.

**Mixed-support stderr output:**

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
  Decide:
    - Narrow [[supported]] pattern to the supported subset, OR
    - Add narrower [[unsupported]] patterns for the unsupported subset (reason optional), OR
    - Add [[test_skips]] entries with a reason if the unsupported tests are actually engine-supported-but-broken
```

No coalescing in v1 (e.g. `*Conv*` covering `*ConvFwd*` + `*ConvBwd*` + …). The output is one pattern per fixture; the engineer coalesces during diff review if desired. Coalescing is a future enhancement once the basic flow is in production.

### 7.5 Wholesale replace within the machine region

V1 wholesale-replaces matching `[[supported]]`/`[[unsupported]]` blocks in the below-sentinel region rather than merging into existing patterns. A previously machine-coalesced `*Conv*` (if a future coalescing pass produced it) becomes `*ConvFwd*`, `*ConvBwd*`, `*ConvWgrad*` on re-run. Annoying but transparent — `git diff` shows everything.

The hand-edited region above the sentinel is the persistence layer for engineer-coalesced or narrowed patterns: anything the engineer wants to survive across tool runs goes there. Merge-preserving behaviour inside the machine region is a v2 enhancement if it becomes a felt cost.

### 7.6 Failure modes the tool detects up front

The tool refuses to run (clear error, no file write) if:

- `--gtest_filter` is set — partial-run baseline would be wrong.
- The sentinel line is present but malformed (e.g. truncated, duplicated). Operator intervention required.
- The below-sentinel region contains anything other than `[[supported]]` / `[[unsupported]]` blocks. Catches the case where someone hand-edited the machine region and the tool would otherwise silently delete their work.

### 7.7 Build integration

Two surfaces:

- **CLI flag** on the integration test binary (`--write-support-claims`) — fits the existing flag family (`--test-config`, `--generate-support-matrix`).
- **CMake/ctest target** per provider (e.g. `ninja miopen-provider-write-support-claims`) — discoverable, idiomatic, runs the binary with the right `--test-config` baked in.

## 8. Workflow

### 8.1 Day-to-day flow (most PRs)

1. PR author lands a code change.
2. CI runs `hipdnn_integration_tests --enforce-support-claims ...`.
3. If support changed, CI fails with a report listing every claim broken and unclaimed gain (§6.3).
4. Author edits the TOML to reflect intent:
   - Intentional new support → add pattern to `[[supported]]`.
   - Intentional removed support → narrow a pattern; if it's a doc'd gap, add to `[[unsupported]]` with a reason.
   - Engine started supporting a documented gap → remove the `[[unsupported]]` entry; add to `[[supported]]`.
   - Accidental change → fix the code.
5. Re-run CI; failures clear.

### 8.2 Bootstrap (new engine or new arch)

1. Engineer runs `ninja <provider>-write-support-claims` on the target hardware.
2. Tool generates `[[supported]]` and `[[unsupported]]` blocks for the current `(arch, platform)`, with mixed-support fixtures listed on stderr.
3. Engineer reviews the `git diff` of the machine-managed region. Adds `reason` strings to `[[unsupported]]` entries that deserve documentation (most won't) — these go in the hand-edited region *above* the sentinel so they survive the next tool run. Coalesces patterns above the sentinel if desired. Adds `[[test_skips]]` for mixed-fixture tests that are bugs to track.
4. Commits the TOML; first CI run on that arch goes green.

### 8.3 Refresh (existing engine, large code change)

Same as bootstrap but the engineer reads the `git diff` more carefully — removed patterns and replaced blocks reflect what the auto-gen sees today vs. what was previously declared. Anything that looks unexpected gets investigated, not auto-accepted.

## 9. CI Integration

- **Pre-submit (Smoke):** `--enforce-support-claims`. Catches claim regressions on the fast tier.
- **Post-submit (Standard/Comprehensive/Full):** `--enforce-support-claims`. Wider observed set → more tests evaluated against patterns.
- **Update PRs:** Reviewers expect a TOML diff with any code change that legitimately moves engine support. PRs without the update fail.
- **THEROCK plugin filtering** (RFC 0006 §8.1) is orthogonal: an unbuilt plugin's TOML is not loaded, so no verification runs.
- **CI never runs `--write-support-claims`.** The tool is engineer-driven; auto-applying it in CI would silently rewrite the contract.

## 10. Alternatives Considered

### 10.1 Flat per-instance test-name list

`tests = ["Suite/Fixture.Case/Param", ...]` enumerating every supported instance.

- Pros: `git diff` shows exact name changes; no glob semantics.
- Cons: Doesn't scale (10K+ instances per arch at Full-tier maturity); swamps PR diffs with mechanical noise; unreviewable.

**Rejected** in favour of patterns.

### 10.2 Per-(suite, instantiation) counts and/or digests

One TOML row per `INSTANTIATE_TEST_SUITE_P` recording `total`, `supported`, and a digest over the per-test support flags.

- Pros: Compact (one row per instantiation), catches per-test flips counts alone miss.
- Cons: Auto-generated count/digest is *snapshot*, not *contract* — goes stale silently as catalogs grow; breaks on partial local runs (60 tests built locally vs. 240 in CI).

**Rejected** in favour of patterns.

### 10.3 Coverage floors / `min_count`

Per-pattern minimum-instance counts to catch catalog shrinkage.

- Pros: Catches accidental deletion of an `INSTANTIATE_TEST_SUITE_P`.
- Cons: Stale by construction (forget to bump after adding tests → silent on later deletions); brittle on partial runs.

**Rejected.** Catalog shrink is intentionally out of scope — it's a code-review concern.

### 10.4 Sidecar baseline file per engine

Put claims in a generated sidecar (`MIOPEN_ENGINE.support.toml`) separate from human-edited skips/tolerances.

- Pros: Cleaner separation of "human" vs. "machine" content.
- Cons: With patterns there *is no* machine-managed snapshot — patterns are hand-edited contracts. Sidecar argument only applied to auto-regenerated content.

**Rejected.**

### 10.5 Single section overloading `[[test_skips]]`

An earlier draft proposed using `[[test_skips]]` for both "broken bug to fix" and "engine has no kernel," with `reason` strings disambiguating.

- Pros: One mechanism.
- Cons: Conflates two engineering concerns with different remediation paths and different lifetimes. Specifically loses the "engine gained capability in a documented gap" signal — a skip silently goes stale when the engine starts working, and no one notices. Three sections give that signal naturally.

**Rejected** in favour of the three-section split (`[[supported]]` / `[[unsupported]]` / `[[test_skips]]`).

### 10.6 Hash digest only

One SHA over the entire sorted supported set per arch. Fail when the hash drifts.

- Pros: Smallest possible file.
- Cons: Zero diagnostic value; no per-test action list; enforces snapshot not contract.

**Rejected.**

### 10.7 Reuse `--generate-support-matrix` markdown as baseline

Diff the markdown matrix between runs.

- Pros: No new format.
- Cons: Markdown is aggregated (op groupings, layout unions); regressions inside a group disappear; formatted for humans, diffs poorly.

**Rejected.** Markdown stays the human output; claims are the machine output.

### 10.8 GTest-only failure mode (no run-end verifier)

`ADD_FAILURE()` from inside the harness body when actual support doesn't match the claims.

- Pros: Per-test xUnit granularity.
- Cons: A test can only fail if it ran. Aggregated reporting and the "config conflict" check are cleaner end-of-run.

**Partial adoption**: this RFC keeps the run-end verifier as primary *and* installs a `TestEventListener` so the diff surfaces as a synthetic gtest failure for xUnit fidelity.

### 10.9 Programmatic skip-list inversion

Replace claims with: every unsupported test must have a matching `[[test_skips]]` entry. Then "everything else is expected supported" is implicit.

- Pros: Single config concept.
- Cons: Forces engineers to enumerate the *complement* (typically larger set); `reason` strings become meaningless for "no engine support, no specific reason."

**Rejected.**

## 11. Risks

| Risk | Mitigation |
|------|------------|
| A pattern accidentally matches more than intended (e.g. `*Conv*` matches an unrelated `*ConvolutionAware*` fixture). | Verifier failure points at the over-broad pattern; engineer narrows it or adds a carve-out. |
| Pattern that no longer matches anything silently passes. | Informational "pattern X matched no observed tests" line — visible in CI logs but not a failure (partial runs would spam). Strict mode can be added later. |
| First run on a new arch fails (no block yet). | Verifier treats "no matching `[[supported]]`/`[[unsupported]]` for this `(arch, platform)`" as **not enforced**. Bring-up is unblocked; engineer runs `--write-support-claims` once and commits. |
| `[[supported]]` and `[[unsupported]]` patterns overlap for a real test → config conflict failure. | Verifier reports the test name and both matched patterns; engineer narrows one side. Possible to detect statically at TOML load for known-overlap globs; deferred. |
| `[[unsupported]]` block accumulates as a wall of patterns with no documentation. | Most entries don't need reasons (the absence of the op is its own documentation). For the cases that *do* need reasons, code review is the gate — reviewers ask "why is this an explicit gap?" when the pattern looks non-obvious. A future verifier check could flag patterns that hide complex truth (e.g. a pattern that matches what `[[supported]]` *almost* covers). |
| Auto-gen wholesale-replace blows away patterns on re-run. | The sentinel split (§7.2) is the mitigation: hand-coalesced or hand-curated patterns belong *above* the sentinel and survive every tool run. Only the machine-managed region below the sentinel is overwritten, and that region's contents are documented as ephemeral. |
| Engineer hand-edits the machine-managed region below the sentinel; tool silently wipes it. | Tool refuses to run (§7.6) if the below-sentinel region contains anything other than `[[supported]]` / `[[unsupported]]` blocks, or if the sentinel is malformed. Forces operator intervention. |
| Glob patterns depend on test-naming convention. | Project already enforces `byTag()` and naming rules (see hipDNN CLAUDE.md); reuses existing `globMatch` from `[[test_skips]]` / `[[tolerance_overrides]]`. |

## 12. Open Questions and Future Work

- **Multi-engine claims.** Each plugin's TOML is independent; needs a `[meta] engine = "MIOPEN_ENGINE"` field so the verifier attributes support correctly when multiple engines are loaded. Out of scope here; follow-up to RFC 0006 §6.
- **Detecting obsolete skip rules.** A `detect_obsolete = true` flag on `[[test_skips]]` could trigger an offline check: build the graph for each skipped test, query engine support without executing. If support is returned the skip rule has outlived its reason. Adds value once skip-rule rot becomes a real maintenance burden.
- **Merge-preserve auto-gen.** §7.4 — preserve hand-coalesced patterns across re-runs.
- **Pattern coalescing in the auto-gen tool.** §7.3 — collapse `*ConvFwd*` + `*ConvBwd*` + `*ConvWgrad*` into `*Conv*` when safe. Defer until the per-fixture output proves verbose in practice.
- **Catalog-shrink detection.** Intentionally out of scope (§4.3). If ever felt, a `[[coverage_floors]]` section with auto-regenerated counts could be added without polluting contract-level claims.
- **Strict mode for unmatched patterns.** Promote "pattern matched no observed tests" from info to error when no `--gtest_filter` is set.

## 13. Glossary

- **Engine.** A plugin-provided implementation that can execute a graph. Reported by `hipdnnGetEngineCount_ext` / `getEngineInfo`.
- **Supported (for a test).** The engine returned at least one entry from `Graph::get_ranked_engine_ids()`.
- **Pre-skipped.** A test removed by a matching `[[test_skips]]` entry during `SetUp()`. Excluded from claim evaluation.
- **Observed.** A test that reached `SupportMatrixCollector::recordGraphSupport` during this run (not pre-skipped, not filtered out, in a built binary).
- **Supported claim.** A pattern in a `[[supported]]` block; asserts "every observed test matching this must be supported."
- **Unsupported claim.** A pattern in an `[[unsupported]]` block; asserts "every observed test matching this must NOT be supported." `reason` is optional — included for gaps with a specific story (issue link, deliberate carve-out), omitted for routine "engine doesn't implement this."
- **Supported claim broken.** Observed test matches a supported claim but engine returned no support. FAIL.
- **Unsupported claim broken.** Observed test matches an unsupported claim but engine returned support. FAIL — a documented gap has closed.
- **Unclaimed gain.** Observed test is supported but no claim covers it. FAIL.
- **Config conflict.** Observed test matches both a supported and an unsupported pattern. FAIL.
