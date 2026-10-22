# RFC 0015: Per-Graph Engine Support Claims

- Contributors: Adam Dickin
- **Status**: In progress
- **Status notes**: Claim loader, enforcement, write tool, pre-commit verifier, and CI log harvest shipped (#10244, #10554, #10594, #12338, #12675). The committed-data matrix renderer and failure runbook (§11) are not built. The harvest coverage report (§12.4) is not built.
- **Last verified**: 2026-10-15 against `develop` (2501d0b28cc)

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Scope](#3-scope)
4. [Design Overview](#4-design-overview)
5. [Claim Schema](#5-claim-schema)
6. [Enforcement Levels](#6-enforcement-levels)
7. [Outcome States and Guards](#7-outcome-states-and-guards)
8. [Required Harness Capability](#8-required-harness-capability)
9. [Write Tool and Contract](#9-write-tool-and-contract)
10. [Oversized and Golden-Infeasible Graphs](#10-oversized-and-golden-infeasible-graphs)
11. [Support Matrix Generation and Documentation](#11-support-matrix-generation-and-documentation)
12. [CI Support-Observation Harvest](#12-ci-support-observation-harvest)
13. [Alternatives Considered](#13-alternatives-considered)
14. [Risks](#14-risks)
15. [Open Questions and Future Work](#15-open-questions-and-future-work)
16. [Glossary](#16-glossary)

## 1. Executive Summary

This RFC adds a **per-graph engine-support claim** to the bundle test framework
defined in [RFC 0011](0011_BundleTestFramework.md). Each golden bundle
gains an optional machine-managed companion — `{Name}.support.json` for a single-graph bundle,
or a co-located per-case `support.json` for a template-sweep bundle (§5.4) — that records, per
claimed graph and per `(engine, arch, platform)`, whether the engine is expected to support that
exact graph.
At test time the runner queries live engine support for the graph and fails the build when
a graph the bundle **claims** as supported is no longer supported on a claimed
`(arch, platform)`.

Because every claimed graph is an on-disk bundle, the claim key **is that graph**: the verdict
lives next to the graph it pins — the bundle's `{Name}.json` for a single-graph bundle, or the
sweep's `cases[].id` expanded from `graph.template.json` (§5.4) for a template-sweep case, both
conforming to the [`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema that carries exact
dims, strides, per-tensor dtypes, layout, and op parameters — and enforcement re-queries that
same graph. There is no separate fingerprint, no pattern language, no glob matching, and no
region/condensation engine. A regression is a claimed engine flipping from supported to
*declined* (dropping out of the bundle's live support query) for a claimed
`(engine, arch, platform)`.

A claimed graph comes in two on-disk forms, both defined by RFC 0011: a **single-graph bundle**
(one `{Name}.json`) and a **template-sweep case** (one `cases[].id` expanded from a shared
`graph.template.json` via `sweep.json`). The claim mechanism is identical for both — the key is
always the exact graph, and for a sweep that graph is the expanded case, disambiguated by its
`cases[].id` (§5.4). A single-graph bundle carries a `{Name}.support.json`; a template-sweep
bundle carries one co-located `support.json` whose claims are keyed per case id.

Enforcement is layered as a **three-rung ladder**, each rung a strictly stronger signal than
the one below and each building on it:

1. **`applicability`** — the engine *accepts* the graph (the support query returns it). No
   build, no execution, no comparison.
2. **`buildable`** — the engine additionally *compiles an execution plan* for the graph
   (plans are created and built) without executing it.
3. **`full`** — the engine additionally *executes* the graph and its output is *numerically
   verified* against a reference.

A bundle declares how far up the ladder it is enforced via `enforcement_level` (§6); a higher
rung implies all the rungs beneath it, so a `full` bundle that stops being *accepted* fails at
the applicability rung. The lower two rungs are the home for graphs whose golden data is
infeasible to generate or compare (very large tensors, long-sequence attention), and for cheap
"this shape must stay accepted / must still build" declarations that need no reference data at
all.

The claims are produced by an engineer-driven tool (`--write-support-claims`) that observes
live support on the target hardware and writes each bundle's verdict for the current
`(arch, platform)`. CI never writes claims; it only enforces them.

Three principles govern the design:

1. **The graph is the key.** A claim attaches to an exact, committed graph. The verdict
   never depends on test names, parameter strings, or a serialized description of the
   graph — only on the graph that is already on disk.
2. **Claims are intentional contracts, not capability inventories.** A graph is gated only
   if someone authored a bundle for it. Absence of a claim means "not enforced," not
   "unsupported." This keeps maintenance bounded and the contract reviewable.
3. **Acceptance, buildability, and correctness are separable.** "The engine must accept this
   graph," "the engine must compile a plan for it," and "the engine must produce correct
   numbers for it" are three distinct assertions of increasing strength. The framework lets a
   bundle assert any prefix of that ladder without paying for the rungs above it.

## 2. Problem Statement

The bundle test framework ([RFC 0011](0011_BundleTestFramework.md)) validates
engine outputs against pre-computed data and **skips** a bundle when the engine under test
reports that it does not support the graph (the engine execution step maps "unsupported" to
`GTEST_SKIP`). This is correct for bring-up — a graph no engine supports yet should not
fail — but it creates a silent regression channel:

> When an engine's solver/kernel coverage shifts and a previously-supported graph becomes
> unsupported on a given arch, the bundle cleanly `GTEST_SKIP`s, the suite is green, and
> nobody is notified that coverage was lost.

The same gap exists for graphs whose golden data is infeasible to generate (multi-GB
tensors, long-sequence attention): they cannot be `full` bundles at all today, so there is
no way to assert even that the engine still *accepts* them.

We need a CI gate that fails the moment a graph we have explicitly claimed stops being
supported on an arch we have claimed — without forcing exhaustive per-arch maintenance for
every graph and arch an engineer is not actively working on, and without requiring golden
data for graphs that are too large to validate numerically.

## 3. Scope

This section is the contract boundary. Reviewers should anchor here.

### 3.1 Detected

| Regression | How it surfaces |
|------------|-----------------|
| A graph that a bundle claims `supported` loses engine support for a claimed `(engine, arch, platform)`. | The claimed engine is *declined* — absent from the resolved support query's ranked list (`GRAPH_NOT_SUPPORTED` when it is the sole/last engine) → **FAIL** (replaces the silent `GTEST_SKIP`). See §5.2 / §7.1 for how "declined" is distinguished from "errored". |
| A graph with a `supported` claim hits an unresolved query (build failure, infrastructure fault) before its verdict can be evaluated. | The claim could not be assessed → **FAIL** ("errored before the claim could be checked; fix the error first"). Distinct from the *declined* verdict above, which is the real-regression case. |

### 3.2 Deliberately not detected

| Non-regression | Rationale |
|----------------|-----------|
| A whole bundle is deleted, removing its claim. | Catalog shrinkage is a code-review concern; the per-tier floor (§7.2, once available) catches whole-bundle loss, and version-control history records it. |
| A claim is eroded *within* a surviving bundle — `support.json` deleted, or a verdict downgraded from `supported` to absent. | **Not** caught by the bundle-count floor (the bundle still discovers and queries). This is accepted declared-graphs-only tail-risk, mitigated only by code review and version-control history. |
| An arch has no claim for a graph yet. | New-arch bring-up: absence is "not enforced." The engineer runs `--write-support-claims` once on the new hardware. |

**The narrowness is the load-bearing property.** This framework trades symmetric,
population-wide capability tracking for low-maintenance, exactly-reviewable contracts that
attach to graphs an engineer deliberately chose to pin. A graph regressing while it has no
bundle — or whose claim was explicitly removed — is *out of contract by construction*, not a
missed detection.

## 4. Design Overview

The unit of claim is the **claimed graph** from RFC 0011, which exists in two on-disk forms.
A **single-graph bundle** is a directory with a graph definition (`{Name}.json`) and, for
`full` bundles, one `.tensor{uid}.bin` per tensor. A **template-sweep bundle** is a directory
with a shared `graph.template.json` and a `sweep.json` whose `cases[]` each expand to one graph
(RFC 0011). This RFC adds one optional machine-managed file to each kind:

```
{Name}/
  {Name}.json              # graph definition (RFC 0011, unchanged)
  {Name}.tensor{uid}.bin   # tensor data (RFC 0011, full bundles only)
  {Name}.meta.json         # provenance + tolerance (RFC 0011, human + tooling)
  {Name}.support.json      # NEW: machine-managed engine-support claim
```

For a template-sweep bundle the single sidecar is co-located with the sweep files and keyed by
case id (§5.4):

```
{TopologyName}/
  graph.template.json      # shared topology (RFC 0011, unchanged)
  sweep.json               # case matrix + inline per-case meta (RFC 0011, human + tooling)
  golden/{CaseId}/...      # per-case tensor data (RFC 0011, full cases only)
  support.json             # NEW: machine-managed claims, keyed by cases[].id
```

The support sidecar is **machine-written**: `--write-support-claims` (§9) adds claims and never
removes one, since retracting a claim is a deliberate, reviewed change. It is kept separate from
the human-authored `meta.json` (and, for a sweep, from `sweep.json`). The runner reads it at test
time and compares its recorded verdict against a live
engine support query for the exact graph — the expanded case, for a sweep.

Claim evaluation is built on RFC 0011's existing machinery:

- **Discovery must exclude the new sidecar.** Bundle discovery (`discoverBundles` in
  `dnn-providers/integration-tests/src/harness/bundle/BundleDiscovery.hpp`) registers one test per
  single-graph `{Name}.json` and one per template-sweep `cases[].id`, already skipping companion
  `meta.json` and treating `graph.template.json` + `sweep.json` as a sweep root. The two sidecar
  kinds are excluded by different mechanisms. A single-graph `{Name}.support.json` sits in an
  ordinary bundle dir, so it reaches `isGraphFile()` and — because its final dotted segment
  `support` is not yet a companion kind — would register as a spurious second test; the fix is to
  add `"support"` to `companionKinds()` (currently `{"meta"}`), the companion-suffix exclusion list
  `isGraphFile()` consults. A sweep's bare `support.json` needs no code change: `discoverBundles`
  already skips every `.json` under a sweep root (the `isDescendantOf(sweepDir)` guard), and
  `discoverSweepCases()` looks only at `sweep.json` and `graph.template.json`, so the sidecar is
  never a discovery candidate — the claim loader reads it later. *Acceptance:* a single-graph dir
  with `{Name}.json` + `{Name}.support.json` registers exactly one test; a sweep dir registers
  exactly one per `cases[].id`; the sidecar is never loaded as a graph.
- **The claim key is the on-disk graph**, not a content hash. For a single-graph bundle it is
  that bundle (`{Name}.json` + co-located `{Name}.support.json`); for a template sweep it is the
  **expanded case** — the `cases[].id` applied to `graph.template.json`, plus that case's verdict
  in the co-located `support.json` (§5.4). RFC 0011 already treats each `cases[].id` as its own
  logical test/graph (keyed `sweep.json#caseId`), so no new identity mechanism is needed:
  enforcement re-expands the case and re-queries it, and two byte-identical graphs are two claims.
- **The verdict** is produced offline on hardware and committed; enforcement is a pure
  comparison at test time.

## 5. Claim Schema

A single-graph bundle's `{Name}.support.json` lists, per engine and arch, the **platforms on
which the graph is claimed supported** (template sweeps wrap this same per-engine map under
per-case groups — §5.4):

```json
{
  "version": 1,
  "claims": {
    "MIOPEN_ENGINE": {
      "gfx942": ["linux", "windows"],
      "gfx90a": ["linux"]
    }
  }
}
```

### 5.1 Field semantics

| Field | Required | Meaning |
|-------|----------|---------|
| `version` | yes | Schema version of the support file. A reader refuses a version it does not understand (loud, not silent) and the engineer regenerates. |
| `claims` | yes | Map of engine name → arch → **array of supported platforms**. |
| engine key | — | Exact engine name as reported by `getEngineInfo`. |
| arch key | — | Exact match against the arch token — the prefix before the first `:` of the raw `gcnArchName` (e.g. `gfx942:sramecc+:xnack-` → `gfx942`).
| platform entry | — | Exact `"linux"` or `"windows"` in the arch's array. **Presence = a `supported` claim** for that `(engine, arch, platform)`; there is no *unsupported* value — an omitted platform is simply *not enforced* (never *asserted-declined*). |

### 5.2 Verdict meaning

The support query (`get_ranked_engine_ids`) returns a status plus a ranked id list spanning
**all loaded engines**. A verdict is always *per engine*, decided in two steps:

1. **Did the query resolve?** — keyed on the status code. A *resolved* query is status `OK`
   **or** `GRAPH_NOT_SUPPORTED`: in both cases the harness successfully determined support (an
   empty list under `GRAPH_NOT_SUPPORTED` is the legitimate "no engine accepts it" answer, not a
   failure). Any *other* error status (backend error, OOM, device lost, build failure) is
   *unresolved*.
2. **For a resolved query, is the claimed engine in the list?** — keyed on membership. Map each
   ranked id to its engine name (§8); the claimed engine is *supported* if its id is present,
   *declined* if absent.

This yields three per-engine outcomes:

- **supported** — query resolved and the engine's id is in the ranked list.
- **declined** — query resolved and the engine's id is absent. `GRAPH_NOT_SUPPORTED` (empty
  list) is the special case where *every* loaded engine is declined; with several engines
  loaded, one engine can be declined while another is supported under a single `OK` status.
- **unknown** — the query did not resolve (a non-decline error status): no per-engine verdict
  is available for any engine.

The status code separates *resolved* from *unknown*; membership separates *supported* from
*declined* within a resolved query (see §8 for why list-emptiness alone is not the
discriminator — an empty list and an error status are different things, and a non-empty list
must still be attributed per engine).

Enforcement reads only the engine under test (`--test-engine`). Claims for other engines are not
checked in that run (§7.3).

Claim semantics:

- A claim (an `(engine, arch, platform)` **listed** as supported) → the engine **must** return
  *supported*. A *declined* result → **claim broken FAIL** (§7). An *unknown* result →
  **errored-before-assert FAIL** (§7.1) — the claim could not be checked; fix the error.
- A **missing** `(engine, arch, platform)` (platform not listed, arch/engine absent, or no file)
  → **not enforced** for that combination. This is *not* an assertion that the engine declines
  it — the combination simply is not gated. Capability appearing on an unlisted combination of a
  **claim-bearing** bundle is surfaced by the end-of-run "unclaimed support" summary (§7.1) and,
  across runs, by the CI harvest (§12), which proposes adding the claim; neither is a failure.

### 5.3 Loader rules

- All keys are matched exact-string. No `*`, no `?`, no fnmatch.
- Malformed entries are rejected: an arch value that is not an array, or a platform token other
  than `"linux"`/`"windows"` (catches typos / schema drift).
- An empty `claims` map is legal (a bundle with no current claims).
- The file is optional; a bundle with no `support.json` is simply not support-gated.

### 5.4 Template-sweep claim files

A template-sweep bundle holds many cases (`cases[].id`) that share one `graph.template.json`
but differ by dtype, dims, strides, layout, or attributes (RFC 0011). Support can differ case to
case, but many cases usually share the same footprint across ASICs, so a sweep's claims are
organized **engine-first, then grouped by shared support footprint** rather than repeating a
full arch/platform matrix per case. The sweep's machine-owned sidecar is a single
**`support.json`** co-located with `sweep.json` (bare name, matching the sweep directory's
`graph.template.json` / `sweep.json` convention), shaped as:

```json
{
  "version": 1,
  "claims": {
    "MIOPEN_ENGINE": [
      {
        "cases": ["small_fp16_nchw", "small_fp32_nchw", "small_fp32_ncdhw"],
        "support": {
          "gfx942": ["linux", "windows"],
          "gfx90a": ["linux"]
        }
      },
      {
        "cases": ["big_fp8_nhwc"],
        "support": {
          "gfx942": ["linux"]
        }
      }
    ]
  }
}
```

- **`claims` is keyed by engine** (canonical JSON sorts these keys — §9.2); each engine maps to an
  **array of claim groups**. A group is `{ cases, support }`: `cases` lists `cases[].id`s and
  `support` is the `arch -> [supported platforms]` map — identical in shape and meaning to a
  single-graph `{Name}.support.json`'s per-engine map (§5.1–5.2) — applied to **every** case in
  the group. Listing a platform claims *supported*; omitting it means *not enforced*. Resolution
  (§5.2) and outcomes (§7) are evaluated **per `(case, engine, arch, platform)`**.
- **The grouping is the compression the sweep buys.** All cases whose support footprint is
  identical share one group — the "common set supported across many ASICs" is one array entry,
  and cases whose footprint differs (the outliers) fall into their own entries. Two cases group
  iff their `support` map is identical.
- **Every case is claimed at most once per engine.** A `cases[].id` may appear in **exactly one**
  group within a given engine's array; the same id in two groups for one engine is a
  **pre-commit error** (ambiguous claim). The same id may appear under different engines.
- **Case ids must exist in the sibling `sweep.json`.** A `cases` entry naming a case absent from
  `sweep.json` is a **pre-commit error** (a renamed or deleted case must not leave an orphaned
  claim). A `sweep.json` case named in **no** group for an engine is simply **not support-gated**
  for that engine — the per-case analog of a single-graph bundle with no `support.json`.
- **Canonical ordering (for idempotent writes).** Within a group `cases` is sorted
  lexicographically, and the groups within an engine's array are ordered by their first case id.
  With canonical-JSON key sorting this makes `--write-support-claims` output deterministic, so a
  re-run with unchanged observations is a zero diff (§9.2).
- The file, any engine array, and any group are optional. An empty `claims` map is legal.
- All other loader rules from §5.3 hold unchanged (exact-string keys, platform arrays of
  `"linux"`/`"windows"` only, rejected unknown `version`).

The claim key is still the exact graph (§4) — for a sweep, the case expanded from
`graph.template.json` by its `cases[].id`, whose verdict is read from the group listing it. A
`support.json` entry only annotates a case `sweep.json` already defines; it never adds, removes,
or renames one.

## 6. Enforcement Levels

A bundle's enforcement level is selected by a new **`enforcement_level`** field, defaulting to
`full`. For a single-graph bundle it lives in the companion **`meta.json`** (RFC 0011's schema is
extended to carry it). For a template-sweep case it lives in that case's **inline metadata**
(`cases[].metadata.enforcement_level` in `sweep.json`, RFC 0011's per-case metadata), so each
case in one sweep may sit at a different rung. The name is deliberately *not* `verification`, to
avoid colliding with RFC 0011's runtime `--verification-mode` CLI flag — they are orthogonal axes
(see §6.1).

The three levels are a **stacking ladder**: each runs every check of the level below it and
then one more, so a higher level is a strictly stronger signal. Each level maps directly onto
a stop-point in the existing golden harness
(`IntegrationGraphVerificationHarness::verifyGraph`).

| Level | Runs | Stops after | `.bin` data |
|-------|------|-------------|-------------|
| `applicability` | Build the graph, query engine support, assert the verdict against `support.json`. **No plan build, no input generation, no execution, no comparison.** | `get_ranked_engine_ids` (engine *accepts* the graph) | ignored |
| `buildable` | Everything `applicability` does, **plus** create and build the engine's execution plans (`create_execution_plans` → `check_support` → `build_plans`) — the plan *compiles*. Still **no input generation, no execution, no comparison.** | `build_plans` (engine *compiles a plan*) | ignored |
| `full` (default) | Everything `buildable` does, **plus** execute the engine and numeric-verify. Reference outputs come from committed golden `.bin` when present; when absent, RFC 0011 computes a runtime **GPU or CPU reference** instead (graph-only behaviour) — `.bin` is therefore not required for a `full` bundle. | numeric comparison (engine *runs correctly*) | optional |

In every level, if a `support.json` claims `supported` and the engine returns *declined* at
that level's stop-point, this is a **claim-broken FAIL** instead of RFC 0011's silent skip
(§7). Because the levels stack, the failure is attributed to the **lowest rung that broke**: a
`full` bundle whose engine no longer *accepts* the graph fails at the applicability rung (the
message names the rung), not at numeric comparison.

`applicability` and `buildable` are distinct **run modes**, not reference sources: neither is
`golden`, `gpu`, or `cpu` (§6.1). Their purpose is to check whether an engine *accepts* or can
*compile* a graph without running it — the home for graphs with very large tensor sizes where
executing or comparing outputs is infeasible, and for cheap "this shape must remain
accepted / must still build" declarations. The runner branches **before input generation** and
stops at the rung's stop-point. The level is chosen explicitly by `enforcement_level`, never
inferred from the presence or absence of `.bin` files (a `full` bundle whose `.bin` files have
not yet been generated is still `full` and falls back to a runtime reference; it is not a lower
rung). `.bin` files on an `applicability` or `buildable` bundle are simply ignored.

This makes the support gate available to graphs that can never have golden data, and lets an
engineer pin "this shape must remain accepted" or "this shape must still compile" with a
bundle that costs nothing to execute.

### 6.1 Composition with RFC 0011 `--verification-mode`

These are **two orthogonal axes** and must not be read off the same list of words:

- **`--verification-mode`** (run-wide CLI flag, values `auto` / `golden` / `gpu` / `cpu`) selects
  *which reference source supplies expected outputs* for numeric comparison.
- **`enforcement_level`** (per-bundle `meta.json` field, values `applicability` / `buildable` /
  `full`) selects *how far up the ladder this bundle is checked*. `applicability` and
  `buildable` are their own run modes — neither produces or compares reference outputs, so no
  `--verification-mode` value applies to them.

**Support-claim enforcement runs for every claim-bearing bundle regardless of
`--verification-mode`** — it depends only on the live support query (and, for `buildable`, the
plan build), not on any reference source. The grid below shows the two axes crossed:

| `enforcement_level` ↓ \ `--verification-mode` → | `auto` / `golden` / `gpu` / `cpu` |
|---|---|
| **`full`** | Support claim always evaluated. The verification-mode value governs only the *numeric* comparison's reference source; it never gates the support or build checks. |
| **`buildable`** | Support claim always evaluated; the verification-mode value is irrelevant (no comparison happens). Like `applicability`, **never** subject to RFC 0011's "golden mode skips bundles with no `.bin`" rule. |
| **`applicability`** | Support claim always evaluated; the verification-mode value is irrelevant (no comparison happens). The bundle is **never** subject to RFC 0011's "golden mode skips bundles with no `.bin`" rule — an applicability bundle typically has no `.bin`, and must not be skipped in the very mode (`golden`, used by Nightly/Weekly CI) meant to enforce it. |

The `buildable` and `applicability` cells are a normative override of RFC 0011's golden-mode
no-`.bin` SKIP for claim-bearing bundles below `full`, and must be stated in the RFC 0011
verification-modes section when that schema extension lands.

### 6.2 Acceptance criteria

- [ ] `enforcement_level` ∈ {`applicability`, `buildable`, `full`} is read from `meta.json`
      (single-graph) or `cases[].metadata` (sweep case); default is `full` when absent **for
      non-claim-bearing bundles/cases**, and a **hard pre-commit error** when a claim exists
      (a `{Name}.support.json`, or a `support.json` `cases` entry) but `enforcement_level` is
      missing or invalid (a typo must never silently flip how far up the ladder a claim-bearing
      graph is checked).
- [ ] A claim-bearing bundle at any level returning *declined* at that level's stop-point FAILs
      (not skips) under every `--verification-mode`, and the failure names the rung that broke.
- [ ] A `buildable` bundle queries support and builds plans but never generates inputs,
      executes, or compares; a plan-build failure on a `supported` claim FAILs.
- [ ] A `buildable` or `applicability` bundle is evaluated under `golden` mode and is never
      skipped for lacking `.bin`.
- [ ] An `applicability` bundle never builds plans, generates inputs, executes, or compares.

## 7. Outcome States and Guards

### 7.1 Per-graph outcomes

Only the engine under test (`--test-engine`) is evaluated. Its claim for the run's
`(arch, platform)` is checked against the support query, using the per-engine verdict of §5.2
(status code → *resolved* vs *unknown*; membership → *supported* vs *declined*):

- **Claim broken.** A `supported` claim whose query *resolved* but whose engine is *declined*
  (its id absent from the ranked list — `GRAPH_NOT_SUPPORTED` when it is the only/last engine)
  → **FAIL**. The message names the bundle, the `(engine, arch, platform)`, and the reason.
  This is the real-regression case.
- **Errored before assert.** A `supported` claim whose query did **not** resolve (build
  failure, backend error, OOM, device lost), or whose graph build itself crashes → **FAIL**
  ("fix the underlying error first"). Distinct from *declined*: the verdict could not be
  evaluated, so this is not a coverage-loss verdict but it still must not pass green.
- **Support unknown.** For an unclaimed `(engine, arch, platform)`, an unresolved query →
  **note**, nothing evaluated. Support is *unknown*, not *declined*; it must never be treated as
  a verdict.
- **Unclaimed support (informational).** For a **claim-bearing** bundle/sweep (one that carries a
  `support.json`), a graph the engine *supported* on an `(engine, arch, platform)` the file does
  **not** list → listed under `unclaimed_support` in the end-of-run `SUPPORT CLAIM SUMMARY`,
  never a FAIL and never a SKIP change. The §12 CI harvest reads the same list from CI logs and
  proposes the claims to add. A graph with no `support.json` at all is out of scope — surfacing
  every unclaimed graph would be the rejected default-deny noise (§13.4).

The split between *declined* and *unknown* is the whole reason §5.2/§8 key on the status code
for the first step: collapsing an error status into "declined" would force a choice between
flaky CI (every error → FAIL) and missed regressions (every error → note). A *resolved* query
(`OK` or `GRAPH_NOT_SUPPORTED`) is the only one that produces a verdict; within it, an engine's
absence from the ranked list is its *declined* signal. When several engines are loaded, the
verdict reads only the engine under test's membership, so another engine's `declined` does not
affect it. Claims for other engines are not checked in this run; each engine's own lane checks
them.

### 7.2 Run-level guards

- **Empty-query guard (v1, self-contained).** If enforcement is requested and **zero**
  support queries were observed in the whole run (no GPU, plugin failed to load, over-narrow
  filter), the run **FAILs** loudly rather than passing green. This needs no completeness
  reasoning — zero queries is unambiguous — and is independent of sharding. This is the
  actionable v1 floor.
- **Empty-observation floor (blocked-by RFC 0011).** A per-tier minimum discovered/queried
  count would catch partial catalog loss, but it depends on the per-tier baseline mechanism
  that RFC 0011 lists only as deferred CI-integration work (no schema, no source for the
  numbers). Until that lands, this floor is **not** a v1 guard; the empty-query guard above
  is. When the RFC 0011 baseline exists, a tier that enforces but assesses fewer bundles than
  its baseline becomes a failure.

### 7.3 Engine selection, multi-plugin runs, and preconditions

A run selects which engine(s) it drives through two harness flags — `--test-article` (the
plugin shared library to load) and `--test-engine` (an engine within it). Three modes result.
They determine whether the run enforces claims and how it generates them:

| Mode | Flags | Engines exercised | Enforce | Generate |
|------|-------|-------------------|---------|----------|
| **A — auto-select** | neither (default plugin config) | hipDNN's heuristic picks the top-ranked engine **per graph** | no | **no** |
| **B — plugin, all engines** | `--test-article <plugin>` only | as mode A; a write run reads ranked-list membership for **every** loaded engine from one support query per graph | no | **yes** |
| **C — plugin + engine** | `--test-article <plugin>` + `--test-engine <engine>` | exactly that one engine | yes | **yes** |

- **Enforcement needs `--test-engine` (mode C).** One lane tests one engine. The run checks only
  the engine under test's claims, by its ranked-list membership (§5.2, §8). Without
  `--test-engine` the run checks no claims: the default enforcement drops to warning-only, and an
  explicit `--enforce-support-claims` is an error. There is **no refuse-to-run** on multiple
  plugins; other loaded engines are not checked, so one engine's `declined` never affects the
  engine under test's verdict.
- **Deep (`buildable`/`full`) checks follow the engine under test.** Building and executing pin
  the engine under test as the preferred engine. Mode A and mode B let the backend pick, which
  can differ graph to graph, so they cannot attribute a `buildable`/`full` verdict to a fixed
  engine.
- **Generation records ranked-list membership.** `--write-support-claims` needs `--test-article`,
  so it runs in mode B or C. It opens each graph, records which engines are in the ranked list,
  and skips the test; it does not build plans or execute. Mode C records the named engine; mode B
  records every loaded engine from the same query, so a single mode-B run refreshes a whole
  plugin's engines on the current ASIC. Each engine's keys are disjoint (§9.2), and one run
  writes them all, so the §9.2 "one writer at a time" property holds. Mode A is run-only.
- **Not yet built:** mode B as one full pass per engine, run in sequence and pinning each engine
  in turn, so a single invocation enforces claims and checks every rung for all of a plugin's
  engines (see the §9.4 all-engines helper).

A claim for an engine other than the engine under test, or for an engine that is **not
loaded**, is *not enforced* in this run (the empty-query guard of §7.2 still catches a run that
observed *nothing*). The claim check is scoped **per-arch**: a brand-new arch with no claims yet
for the engine under test is the not-enforced bring-up case (§3.2), not a refuse-to-run.

### 7.4 Acceptance criteria

- [ ] A `supported` claim whose engine is *declined* (absent from the ranked list of a resolved
      query) → claim-broken FAIL.
- [ ] An unresolved query (non-decline error status) on a `supported` claim →
      errored-before-assert FAIL.
- [ ] Unresolved query on an unclaimed `(engine, arch, platform)` → note, not FAIL.
- [ ] Enforcement with zero observed queries → FAIL (empty-query guard).
- [ ] New-arch run with no claims for the engine under test on that arch → not-enforced, not a
      refuse-to-run.
- [ ] A run with **two or more plugins loaded** checks only the engine under test's claims (no
      refuse-to-run); another loaded engine's *declined* does not change its verdict.
- [ ] A claim for an engine not loaded in the current run is not enforced.
- [ ] A claim-bearing bundle whose engine *supports* an unlisted `(engine, arch, platform)` is
      listed in the end-of-run unclaimed-support summary, not FAILed.

## 8. Required Harness Capability

The `applicability` and `buildable` levels and the "errored before assert" outcome require the
harness to **build a graph, query engine support, and (for `buildable`) compile its plans —
all without executing it or supplying tensor data**:

> Given a `{Name}.json`, produce a finalized `hipdnn_frontend::graph::Graph` **with no tensor
> data**, call `get_ranked_engine_ids`, map the result to supported / declined / unknown
> (§5.2), and — for `buildable` — additionally run `create_execution_plans` → `check_support`
> → `build_plans` for the claimed engine, stopping before any execution.

Every step of this is an **existing frontend capability** — no spike is required before the
`applicability` or `buildable` levels are committed:

1. **Status code + id list together give the verdict.** `get_ranked_engine_ids` returns an
   `Error` plus an id vector. `ErrorCode::GRAPH_NOT_SUPPORTED` (`frontend/.../Error.hpp`)
   signals "well-formed but no engine accepts it," distinct from `HIPDNN_BACKEND_ERROR` and the
   other codes. The status code is the first discriminator — `OK`/`GRAPH_NOT_SUPPORTED` mean the
   query *resolved*, any other code means *unknown* — and the **id list is the second**: a
   resolved query's list, mapped to engine names (point 5), says which engines are supported and
   (by absence) which are declined. The existing live-compute harness collapses these because it
   only ever asked about one preferred engine; this RFC reads the code *and* the membership.
   *This is why §5.2/§7.1 do not key on list-emptiness alone — an empty list and an error status
   are different things, and a non-empty list must still be attributed per engine.*
2. **Bundle JSON deserializes into a queryable graph today.** The frontend already exposes
   `Graph::from_json(handle, json)` (`frontend/.../Graph.hpp`), built on
   `detail::deserializeAndUnpackJsonGraph` (`frontend/.../detail/GraphUnpacker.hpp`), which
   parses a bundle's `{Name}.json`, reconstructs the backend operation-graph descriptor, and
   unpacks it into frontend nodes — no new bundle-JSON→graph bridge is needed.
3. **Finalize needs metadata, not tensor data.** `get_ranked_engine_ids` requires a *finalized*
   descriptor (`hasReadyGraphDesc()` = valid **and** finalized), so the graph must be built
   (`from_json` with a handle finalizes it). That finalize step — including the id/type
   inference done during finalization — needs only the graph's shape/dtype **metadata**, which
   the JSON already carries; it does **not** need host/device tensor buffers, which are
   consumed only by `graph.execute()`. So "going as far as finalize" is cheap and data-free:
   the applicability path runs `from_json(handle, json)` → `get_ranked_engine_ids()` with no
   tensor allocation, no execution, and no golden-data pull.
4. **Plan build is data-free too.** The `buildable` rung extends the path with
   `create_execution_plans()` → `check_support()` → `build_plans()` — the exact sequence the
   golden harness already runs at `IntegrationGraphVerificationHarness::verifyGraph` *before*
   it generates inputs or executes. These steps compile the engine's plan from the finalized
   descriptor and need no host/device tensor buffers; input generation and `execute()` come
   strictly after. The `buildable` path therefore stops cleanly at `build_plans()` with no
   tensor allocation and no golden-data pull, exactly like `applicability` plus a compile.
5. **Per-engine attribution is available.** The harness can already map a ranked engine id to
   its name (`hipdnn_data_sdk::utilities::getEngineNameFromId` / `engineNameToId`) and
   enumerate every loaded engine by name (`hipdnnGetEngineCount_ext` + `getEngineInfo`). A
   claim for a specific `(engine, arch, platform)` is therefore checked by testing whether that
   engine's id is present in the ranked list — the same membership test the golden harness
   already performs for a single `--test-engine` (`std::find(engineIds, targetEngineId)`).
   Enforcement runs this test for the engine under test only; a mode-B write run runs it for
   every loaded engine (§7.3).

## 9. Write Tool and Contract

### 9.1 CLI

```bash
# Mode C — one engine of one plugin
./hipdnn_integration_tests \
    --golden-data-dir dnn-providers/integration-tests/integration-test-bundles \
    --test-article <plugin-lib> --test-engine MIOPEN_ENGINE \
    --write-support-claims

# Mode B — every loaded engine, from one support query per graph
./hipdnn_integration_tests \
    --golden-data-dir dnn-providers/integration-tests/integration-test-bundles \
    --test-article <plugin-lib> \
    --write-support-claims
```

Discovers bundles (single-graph and template-sweep cases alike), observes live support on the
**current** hardware, and writes each claimed graph's verdict for the exercised
`(engine, arch, platform)` only — into `{Name}.support.json` for a single-graph bundle, or into
the sweep `support.json`'s engine array under the group covering that `cases[].id` (§5.4) for a
sweep case. Every verdict the run does not observe — other engines, arches, platforms, and sweep
cases — is left unchanged (§9.2 details how a sweep's grouped array is preserved when only some
cells change). The graph JSON (`{Name}.json` / `graph.template.json`), `sweep.json`, `.bin`
files, and `meta.json` are never modified.

Generation requires `--test-article`, so it runs in **mode B or C** (§7.3), never mode A. It
records ranked-list membership only; it does not build plans or execute. Mode C records the
engine named by `--test-engine` (`TestConfig::getEngineName` → membership test). Mode B omits
`--test-engine` and records every loaded engine from the same support query, refreshing the
whole plugin on one ASIC in a single invocation. Either way each engine writes disjoint keys, so
results merge cleanly (§9.2); broad multi-ASIC coverage can instead accumulate from CI runs
(§12) without a dedicated session.

This engineer-driven write is one of two ways claims get refreshed; the other is the CI
harvest (§12), where routine CI runs print the unclaimed support they see and an engineer runs
a script that adds those claims. Both share the same observe-then-write split — only the actor
and cadence differ.

### 9.2 Write contract

- **Machine-owned, engineer-driven.** CI never runs this tool — auto-applying it would
  silently rewrite the contract. The engineer runs it on target hardware, reviews the
  `git diff`, and commits.
- **Surgical, idempotent writes.** For a single-graph `{Name}.support.json` the write updates only
  the current `(engine, arch, platform)` keys, leaving siblings byte-identical. A sweep's grouped
  array (§5.4) can't be patched in place — a changed cell may move a case between groups — so the
  tool **flattens, overlays, then re-groups canonically**: it expands `support.json` to
  per-`(engine, case, arch, platform)` cells, adds the cells this run observed as supported, keeps
  the rest, and re-derives the canonical grouped form (§5.4). Both paths re-emit canonical JSON
  (sorted keys, fixed number formatting, stable newline), so an unchanged run is a **zero diff**
  and untouched engines stay byte-identical. Because engines, arches, and platforms are disjoint
  cells, separate per-engine/per-arch regenerations merge cleanly; only one binary runs at a time,
  so no multi-writer coordination is needed.
- **Add-only.** A write never removes a claim, including one the run observed as declined. A
  regression and a deliberate withdrawal reach the tool as the same observation, and nothing in
  the run can tell them apart, so retracting a claim is a deliberate, reviewed change. Until it
  lands, the enforcing run reports the disagreement as `CLAIM_BROKEN`.
- **Change churn is bounded and safe.** The zero-diff guarantee is for an *unchanged* run; when a
  support value actually changes, canonical regrouping may merge or split groups, so the diff can
  be larger than the one changed cell. Canonical formatting keeps this granular — one case id per
  line, so a membership change reads as per-case line adds/removes — and any genuine regression (a
  claimed `supported` becoming `declined`) surfaces independently as an enforcement FAIL (§7.1), so
  diff churn can never silently mask a lost claim.
- **Written at end of the run.** The tool writes each claimed graph's support sidecar directly
  after the test run completes. If a target file cannot be opened for writing (e.g. it is locked), the
  tool reports a clear error naming the file rather than silently dropping the verdict; it does
  not need the tmp-file/atomic-rename machinery a concurrent writer would.
- **DVC-independent.** Querying support needs only the graph (`.json`), never the
  (potentially multi-GB) `.bin` files. The write, `applicability`, and `buildable` passes must
  not trigger a golden-data pull.
- **Empty-write guard.** The tool refuses to overwrite an existing `supported` verdict with
  an absent/empty observation set (no GPU, plugin load failure), so a degenerate run cannot
  null out committed claims.

### 9.3 Validation (pre-commit)

The RFC 0011 pre-commit bundle verifier is extended to:

- **Require an explicit `enforcement_level` on every claim-bearing graph.** A single-graph
  bundle that carries a `{Name}.support.json`, or a sweep case that has an entry in the sweep's
  `support.json`, must declare `enforcement_level` ∈ {`applicability`, `buildable`, `full`} (in
  `meta.json` or `cases[].metadata` respectively); missing or invalid is a **hard error** (never
  silently defaulted), so a typo cannot flip how far up the ladder the graph is checked. RFC
  0011's existing `.bin` data-integrity checks are unchanged and apply per its own rules (e.g. a
  bundle/case that ships `.bin` files must ship them for every tensor UID); this RFC adds no new
  `.bin` presence/absence requirement.
- Validate every support sidecar against the schema (§5, §5.4): known `version`, well-formed
  engine/arch keys, and platform arrays containing only `"linux"`/`"windows"`. For a sweep
  `support.json` additionally: `claims` maps each engine to an **array** of `{ cases, support }`
  groups; every `cases` id resolves to a real `cases[].id` in the sibling `sweep.json` (no
  orphaned claims); and **no case id appears in more than one group for the same engine** (no
  ambiguous claim). A bad `version`, malformed key/platform, orphaned case id, or a case claimed
  twice for one engine is a pre-commit FAIL (not a runtime surprise).

### 9.4 Test registration (CMake)

External-provider integration tests are registered by a CMake helper. The miopen-provider (and
other providers) currently call `add_external_integration_test_target`
(`dnn-providers/integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake`), which pins a single
engine — **mode C**:

```cmake
add_external_integration_test_target(
    TARGET_NAME   <ctest name>
    PLUGIN_TARGET <plugin lib target>   # → --test-article $<TARGET_FILE:...>
    ENGINE_NAME   <engine>              # → --test-engine
    TEST_CONFIG   <config file>)        # → --test-config
```

Not yet built: a **second variant for mode B** that registers one ctest running every engine
the plugin exposes, sequentially:

```cmake
add_external_integration_test_target_all_engines(   # name provisional
    TARGET_NAME     <ctest name>
    PLUGIN_TARGET   <plugin lib target>
    TEST_CONFIG_DIR <directory of test configs>)
```

It **omits `ENGINE_NAME`** (the run enumerates the plugin's engines itself, §7.3 mode B) and
takes a **config directory** rather than a single `TEST_CONFIG` file, so each enumerated engine
can resolve its own config from that directory. Because the engines run sequentially within one
binary invocation, this variant both enforces and **generates** support claims for all of a
plugin's engines in a single ctest, honouring the §9.2 one-writer-at-a-time property. The
single-engine helper stays; the two coexist (a provider registers whichever fits).

### 9.5 Acceptance criteria

- [ ] Re-running `--write-support-claims` with no support change produces a zero git diff
      (canonical-JSON idempotency).
- [ ] A write run with an empty/absent observation set does not null an existing `supported`
      verdict (empty-write guard).
- [ ] A claim-bearing bundle with a missing/invalid `enforcement_level` fails pre-commit.
- [ ] A mode-B run (`--test-article` only) records every loaded engine's ranked-list membership
      and writes a verdict for each; a mode-C run writes only the named engine's keys; a mode-A
      run writes nothing.
- [ ] Not yet built: the all-engines CMake helper registers one ctest that exercises every
      engine the plugin exposes, sequentially.
- [ ] A sweep case's verdict is written into the group covering that `cases[].id` under the
      correct engine, other engines' arrays are left byte-identical, and the emitted grouping is
      canonical (cases sorted, groups ordered by first case id).
- [ ] A sweep `support.json` fails pre-commit if a `cases` id has no matching `sweep.json` case,
      or if a case id appears in more than one group for the same engine.

## 10. Oversized and Golden-Infeasible Graphs

An `applicability` (or `buildable`) bundle for a very large graph primarily exercises the
**graph builder/finalize path** (integer-overflow cleanliness of dim/stride math) — and, for
`buildable`, the engine's plan-compile path — and only secondarily the engine's support
decision. A builder bug and an engine regression would be indistinguishable from a single FAIL.
Therefore:

- Choose the **smallest** shape that is golden-infeasible *for compute or storage cost
  reasons*, with dims that remain comfortably representable, rather than the largest
  representable shape.
- If integer-overflow *handling* is itself the property under test, express it as an
  explicitly-labelled builder test, not as a support claim.

This keeps an `applicability`/`buildable` FAIL attributable to the engine, which is the thing
the claim is meant to gate.

## 11. Support Matrix Generation and Documentation

The harness has one support matrix: the runtime `SupportMatrixCollector`.
`--generate-support-matrix[=file]` writes it as markdown (default `support_matrix.md`). It
records C++ graph tests only, not bundles, so it does not show claims. Claim behaviour and the
`SUPPORT CLAIM SUMMARY` keys are documented in
`dnn-providers/integration-tests/docs/support-claim-enforcement.md`.

### 11.1 Matrix from committed claims

Not yet built.

**Generated from committed data, not a hardware run.** A renderer generates an engine support
matrix **purely from the data checked into the repo** — each single-graph bundle's
`{Name}.json`, or a template-sweep case (`graph.template.json` + the `cases[].id` from
`sweep.json`), for the row identity (operation, layout, dtype, scenario), and its support
sidecar (`{Name}.support.json`, or the sweep `support.json` group covering the case, §5.4) for
the per-`(engine, arch, platform)` verdict. Because every claimed graph carries its verdict on
disk (§4), the committed bundles and their `support.json` sidecars are a complete,
hardware-free description of claimed support. A contributor can regenerate or diff the matrix
from a plain checkout with no GPU, no plugin, and no DVC pull (only `.json`/`.support.json` are
needed, and both live in git — §9.2).

**Renderer.** `dnn-providers/integration-tests/scripts/render_support_matrix.py` discovers every
support sidecar under the golden-data tree — each `{Name}.support.json` and each sweep
`support.json` — joins each to its graph (`{Name}.json`, or the expanded `cases[].id` from
`graph.template.json` + `sweep.json`), and emits one combined markdown matrix: engine columns,
one row per claimed graph keyed by `(operation, layout, dtype, scenario)`, each cell showing the
verdict per `(arch, platform)`.
The matrix is a **view** of the committed claims, not a second state to maintain.

**One collection, then a projection.** The renderer resolves each sidecar into a flat list of
claim units once, then renders that list as `dnn-providers/integration-tests/SUPPORT_MATRIX.md`:
a human document that aggregates cases into `(operation, layout, dtype, scenario)` rows and
annotates each row with an HTML comment naming the contributing bundle path, its tier, and its
case count — coordinates back to the claim, not the claim itself. Collection is kept separate
from rendering, so a richer consumer — a browsable or zoomable view of the matrix — needs only a
new projection of the same units rather than a scraper of the rendered tables. Enumerating every case id inline in the markdown is available (`--max-case-ids`) but off by
default: it multiplies the document size to no visual effect, and a large enough markdown file
stops rendering on GitHub at all.

**Sidecar as source of truth.** The `support.json` sidecars are authoritative; the matrix is
derived. The full matrix is **generated on demand and not committed** — a contributor who wants
it runs the renderer against their checkout. This is deliberate: the rendered document is large
and grows with every bundle and every gfx target added, which makes it both a large diff on every
sidecar edit and close enough to the size at which GitHub stops rendering markdown to be a
liability. Not committing it also removes its drift problem outright: there is no full second
copy to go stale. The one committed projection, the overview below, is checked instead.

The cost is that the full document is not in a pull request diff, and a reader cannot browse
it on the web without a checkout. A compact overview recovers most of that:
`support_features/feature_matrix.py` writes
`dnn-providers/integration-tests/SUPPORT_MATRIX_OVERVIEW.md`, the share of each op's API
options claimed per GPU family, engine and target, small enough to render on GitHub. The
overview **is committed**, so a claim change shows in the diff as a changed percentage, and it
is regenerated in the same commit as the sidecars it reflects — `harvest_support_claims.py
--write` (§12.3) does so itself. Reviewers who want the per-graph detail regenerate the full
matrix locally; `render_support_matrix.py --overview-only` gives its per-target graph counts
alone.

**Freshness.** Three checks keep the sidecars and the overview in step:

- The `render-support-matrix` pre-commit hook renders the full matrix and discards it, failing
  if any sidecar cannot be resolved: a malformed or orphaned sidecar, or a claim on an unknown
  case.
- The `check-support-matrix-overview` pre-commit hook runs `feature_matrix.py --check`,
  failing if the committed overview differs from a fresh render.
- Both hooks see only the files pre-commit passes them, and the top-level `exclude` in
  `.pre-commit-config.yaml` hides `integration-test-bundles/` from every hook, so a commit
  that changes only sidecars triggers neither. The `hipdnn_support_matrix_renderer_python_tests`
  and `hipdnn_support_features_python_tests` ctests cover that case: they run in
  `hipdnn-superbuild-ci` on every change under `dnn-providers/` and render the real tree, and
  the second fails if the committed overview is stale.

Claim *correctness* is enforced separately by `verify-support-claims` (§9.3) and by the
integration tests themselves (§4).

Two clarifications. (1) The matrix row key `(operation, layout, dtype, scenario)` is
*coarser* than the actual claim key (the on-disk graph, §4): two bundles — or two sweep cases —
sharing that 4-tuple but differing in exact shape are distinct claims that would collapse to
one matrix row. This is a presentation simplification, not a contract — the per-bundle
`{Name}.support.json` and per-case sweep `support.json` entries remain the source of truth; the
matrix should make the bundle path (and case id, for a sweep) reachable per row so the full
claim set is recoverable. (2) The runtime `SupportMatrixCollector` and this renderer read
different sources — live C++ graph tests versus committed sidecars — and must not be conflated
into divergent matrices.

### 11.2 Failure runbook

Not yet built.

Every claim-broken / errored-before-assert message ends with a stable anchor into a
failure-mode runbook (`docs/support-claims-failures.md`), one section per outcome state (§7),
each answering: what triggered it, what it means, ranked likely causes, and step-by-step
remediation. Anchor names and verifier output are co-designed; renaming an anchor without
updating the verifier fails CI lint.

## 12. CI Support-Observation Harvest

§9's `--write-support-claims` tool refreshes claims by running the suite on one target ASIC.
But CI is where the broadest matrix of ASICs and platforms actually runs, nightly — and every
run that reads claims already queries support for each claim-bearing bundle (§7). Rather than
require a separate engineer-driven hardware session per ASIC, each CI run prints the support
that no claim records, and a script harvests those prints from the CI logs into proposed
`support.json` updates.

This preserves the §9.2 invariant that **CI never writes claims**: CI only prints what it
observes (read-only with respect to the contract); turning that output into committed claims is
an offline, engineer-reviewed step.

### 12.1 The `SUPPORT CLAIM SUMMARY` block

After its tests, a run prints one block to stdout. It skips the block only when it found no
graph with a sidecar and recorded no verdict:

```text
==== SUPPORT CLAIM SUMMARY (ENFORCING) ====
{
  "support_claim_summary": {
    ...
    "mode": "enforcing",
    "run": { "arch": "gfx942", "engine": "MIOPEN_ENGINE", "platform": "linux" },
    "schema_version": 1,
    "unclaimed_support": [
      {
        "bundle": "integration-test-bundles/quick/ConvFwd/sweep.json",
        "cases": ["case_a", "case_b"],
        "reached": "verified",
        "required": "verified"
      }
    ],
    ...
  }
}
```

The header and `mode` name the claim mode: `ENFORCING` / `enforcing`, or
`WARNING ONLY -- NOT ENFORCED` / `warning_only`. Both modes report the same results. The
document also carries coverage counters, verdict counts, `claim_failures`, and `failed_in_use`
(see `support-claim-enforcement.md` for every key). An `unclaimed_support` entry never changes
pass/fail.

An `unclaimed_support` entry follows these rules:

- Only a bundle or sweep that has a `support.json` sidecar can produce an entry.
- The entry is for the **engine under test** only. One lane tests one engine, so other loaded
  engines get no entry.
- The query resolved, the engine is in the ranked list (*supported*, §5.2), and the sidecar
  does not claim this `(engine, arch, platform)` (or sweep case). An *unknown* result gives no
  entry.
- `bundle` starts at the bundle root's folder name. A sweep lists its cases under `cases`.
  `engine`, `arch`, and `platform` appear only when they differ from `run`.
- `reached` and `required` give the depth the graph reached and the depth its bundle requires
  (`not-reached`, `applicable`, `buildable`, `executed`, `verified`).

### 12.2 Shard- and ASIC-tolerant by construction

The harvest turns each entry into one cell per case, `(bundle, case, arch, platform, engine)`.
Each cell is a self-contained fact, so:

- **Sharding.** One job log can hold several blocks, one per test process; the script reads
  every block and unions the cells. Unlike enforcement's empty-observation floor (§7.2),
  harvesting needs no completeness guarantee from a single process — the union across jobs and
  runs is what matters.
- **Multiple ASICs.** Jobs on different ASICs give disjoint `(arch, platform)` keys, so their
  cells never conflict — the same disjoint-key property as the §9.2 writes.
- **Multiple engines.** Each lane tests one engine, so each provider's jobs give that engine's
  cells (disjoint `engine` keys). The default `--jobs` pattern, `Test \w+provider\b`, reads
  every provider's jobs, so one CI run gives support data for several engines.

### 12.3 Harvesting into claims (offline, reviewed)

`dnn-providers/integration-tests/scripts/harvest_support_claims.py` reads CI job logs and
prints the claim cells to add:

- **Input.** Each positional `RUN` is a job log or a directory of `*.log` job logs. Each
  `--run ID` fetches a GitHub workflow run with `gh` and reads its jobs whose name matches
  `--jobs` and that succeeded or failed. With no input, the script fetches the latest completed
  scheduled run of the develop nightly (`therock-multi-arch-ci-nightly.yml`). A log whose text
  appears in two runs is refused, so no run counts twice.
- **Block checks.** The script rejects a whole block when no gtest `ran.` line comes before it,
  its JSON does not parse, `schema_version` is not 1, `mode` is unknown, the run arch,
  platform, or engine is invalid, `counters_consistent` is not `true`, or a `harness_defects`
  count is not 0. A log with a rejected block contributes nothing, and the exit status is 1. A
  log with no block is listed under `Missing`; its engines are not harvested.
- **Cell checks.** A cell is kept only when its gtest printed an `OK` line in the same log
  stream and `reached` is at or above `required`. A cell whose gtest failed, was skipped, or
  printed no `OK` line is dropped, as is a cell with a missing depth.
- **Across runs.** A cell is claimed when it is kept in at least `--min-runs` runs (default 1)
  and dropped in none. One drop in any run vetoes it.
- **Add only.** The harvest only adds claims; it never removes a claim. A claim that the
  engine stops meeting is already a CI **FAIL** (§7.1), so it surfaces loudly, not through the
  harvest. `claim_failures` and `failed_in_use` are listed for a human; they never become
  claims.
- **Output is a reviewed change, not an auto-commit.** By default the script prints a table
  (`--format json` for JSON). With `--arch` and `--platform` it adds the claims for that one
  arch/platform to the current sidecars and names each sidecar that would change; `--write`
  (which needs both) writes them. Sidecars must be up to date: the script fetches the tip of
  develop with git and refuses a sidecar that lacks any claim its develop version has. Nothing
  is written if a log was rejected or a sidecar was refused. Sidecars are rendered in the
  canonical form that `verify_support_claims.py` checks (§9.3), so the diff is minimal. An
  engineer reviews and commits the change, keeping the §9.2 "CI never writes the contract"
  invariant intact.

### 12.4 Full vs incremental refresh

> *Does a full refresh of an ASIC require a full run?*

A **complete** refresh of an ASIC requires that **every claim-bearing bundle has been observed
at least once on that ASIC** — completeness is "every bundle observed ≥ 1×," however achieved.
Two ways to get there:

- **One dedicated full run** — `--write-support-claims` over the entire suite, unsharded and
  unfiltered, on that hardware, in a single pass (§9).
- **Accumulated CI runs** — the union of nightly runs' kept cells, once they have collectively
  covered every tier and shard for that ASIC. No dedicated hardware session is required; the
  coverage falls out of routine CI over time. Because the harvest only adds claims, this path
  finds gains only.

So a full refresh still requires full *coverage*, but not necessarily a single full *run*: the
harvest model lets that coverage accumulate from CI instead of a manual session.

Not yet built: a per-ASIC coverage report (which claim-bearing bundles have and have not been
observed) so an engineer knows when an ASIC's picture is complete. The script lists only logs
with no block (`Missing`).

### 12.5 Acceptance criteria

- [ ] A run with a claim-bearing graph prints one `SUPPORT CLAIM SUMMARY` block whose
      `unclaimed_support` lists each unclaimed cell the engine under test supports, with
      `reached` and `required`; the block does not alter pass/fail.
- [ ] The harvest script unions cells across jobs and runs, keeps only cells whose gtest passed
      at the required depth, and only adds claims; it never removes a committed claim and
      ignores *unknown* results.
- [ ] One dropped cell in any run vetoes that claim; `--min-runs` sets how many runs must keep
      it.
- [ ] The script writes sidecars only with `--write` for one `--arch`/`--platform`, refuses
      sidecars behind develop, and never commits.
- [ ] Not yet built: the script reports per-ASIC observation coverage (observed vs total
      claim-bearing bundles).

## 13. Alternatives Considered

### 13.1 Condensed region matchers over a structured graph signature

A claim system could record **patterns** that match many graphs at once — e.g. a set of
matchers over `(op-structure, dtype-map, layout)` regions, produced by a condensation
algorithm that folds many observations into a minimal safe set and refuses to write when the
same signature is seen both supported and unsupported. This generalizes: it can claim graph
shapes that were never individually tested, so a regression on an un-enumerated shape still
fails.

**Rejected for v1.** Region matching exists to compensate for a test population that has no
catalog — graphs generated transiently at runtime that cannot be pointed at individually. In
this framework every claimed graph is an on-disk bundle, so the catalog already exists and
is exactly the set of bundles. There are no un-enumerated-but-run graphs to generalize to,
which removes the only thing region matching buys. It would also reintroduce a structured
signature schema, a stability contract for that schema, a condensation heuristic, and a
conflict-refusal mechanism — substantial machinery whose entire purpose is moot once the
graph is the key. The cost is the declared-graphs-only scope (§3.2), accepted deliberately.
Forward-looking "all graphs of this shape, including future ones, must stay supported"
promises are the one capability region matching would add; they are deferred to §15.

### 13.2 Claims embedded in `meta.json`

Support verdicts could live in the existing `meta.json` rather than a separate file.
**Rejected:** `meta.json` is human-authored (provenance, calibrated tolerance). Mixing a
machine-rewritten, per-arch, hardware-observed verdict into a hand-edited file invites diff
noise and accidental clobbering. A dedicated machine-owned `support.json` keeps the
ownership boundary clean (the same split RFC 0011 uses between graph data and metadata).

### 13.3 A single central support index instead of per-bundle files

One index file (per engine, or global) keyed by bundle identity would reduce file count and
shrink the support-matrix render to a single read. **Rejected for v1** in favour of
per-bundle co-location, which keeps a bundle self-contained (a customer-submitted bundle
carries its own support story) and makes a regression a one-file diff. If per-file write
churn proves unsustainable at scale, a per-engine index keyed by bundle identity is the
fallback (§15) — still with no region matching.

### 13.4 Default-deny on unclaimed graphs

Failing when an engine supports a graph that no bundle claims would force a bundle for every
new capability and churn CI constantly. **Rejected** in favour of the end-of-run "unclaimed
support" summary (§7.1): absence of a claim is "not enforced," and capability growth is
surfaced, not punished.

### 13.5 Glob/pattern matching over test names

Matching claims against GTest test names via globs is platform-divergent, sensitive to
parameter-string formatting and test-macro style, and matches `DISABLED_` tests that never
run. **Rejected** — the claim attaches to the graph, never to a test name.

## 14. Risks

| Risk | Mitigation |
|------|------------|
| A graph regresses while it has no bundle, so it is not gated. | Accepted by design (§3.2, principle 2). `applicability` bundles make pinning a shape cheap; the support matrix (§11) makes coverage visible so gaps are noticeable. |
| Per-arch write runs produce noisy diffs across many `support.json` files. | Surgical, canonical-JSON, disjoint-key writes (§9.2) keep diffs minimal and idempotent. Fallback: per-engine index (§15). |
| `graph.fbs` / serialization evolution changes how a graph deserializes, shifting the identity a claim attaches to. | The claim attaches to the deserialized graph object, not raw bytes; FlatBuffers schema changes must stay backwards compatible (same constraint RFC 0011 already carries). |
| `applicability`/`buildable` FAIL is ambiguous between a builder bug and an engine regression for oversized shapes. | Shape-selection rule in §10 (smallest golden-infeasible shape, overflow handling as a separate labelled test). |
| The support query cannot distinguish "declined" from "errored". | Two-step verdict (§5.2, §8): the status code separates a *resolved* query (`OK`/`GRAPH_NOT_SUPPORTED`) from an *unknown* error; within a resolved query, ranked-list membership separates *supported* from *declined* per engine. Only a resolved query yields a verdict; all other errors are "unknown". |
| With multiple plugins loaded, a declined engine is mis-attributed or taints another engine's verdict. | Enforcement checks only the engine under test, by its ranked-list membership (§5.2, §7.3, §8); another engine's `declined` never affects it. |
| A claim is eroded within a surviving bundle (support.json deleted / verdict downgraded). | Accepted declared-graphs-only tail-risk (§3.2); mitigated by code review and version-control history rather than a dedicated guard. |
| A no-GPU or plugin-load-failure run silently passes or nulls claims. | Empty-query guard and empty-write guard (§7.2, §9.2). |
| CI harvest (§12) proposes a wrong/over-broad claim. | The harvest only adds claims, ignores *unknown* results, keeps a cell only when its gtest passed at the required depth, lets one drop in any run veto a cell, and never commits — an engineer reviews every change (§12.3). |
| Stale or spoofed CI logs corrupt the harvest. | The script rejects any block that fails its checks, refuses a log that appears in two runs, and refuses sidecars behind develop (§12.3). Logs are input to a reviewed change, never directly the contract. |

## 15. Open Questions and Future Work

- **Forward-looking region promises.** A claim of the form "every graph of this shape,
  including ones not yet authored, must stay supported" would require a pattern/region
  mechanism (§13.1) with explicit wildcard semantics. Deferred; add only if a concrete need
  appears.
- **Gating the runtime-fixture suite.** Graphs exercised only by runtime test fixtures
  (without bundles) are not support-gated. The migration path is to author `applicability`
  bundles for the shapes worth pinning. Whether to systematically convert fixture shapes to
  bundles is future work.
- **Automated harvest PR bot.** §12.3 keeps harvest engineer-reviewed. A gated bot that opens
  the proposed `support.json` PR automatically from nightly CI logs (still merged by a human)
  is a possible next step.
- **Harvesting the full runtime population.** If runtime-fixture graphs also reported
  unclaimed support (not just bundle-backed ones), the harvest could surface support data for
  graphs that do not yet have bundles, informing which `applicability` bundles are worth
  authoring. Depends on the fixture path printing the same `unclaimed_support` entries.
- **Per-engine index.** If per-bundle `support.json` write churn is unsustainable at the
  full bundle population, move claims to a per-engine index keyed by bundle identity — still
  no region matching, same verdict semantics.
- **Sharding.** Enforcement assumes an unsharded run so the empty-observation floor (§7.2) is
  meaningful. A per-shard record + reduce phase is future work, shared with the golden
  framework's own sharding story.
- **Deep enforcement across engines.** Enforcement checks one engine per run (mode C). Running
  all of a plugin's engines in one invocation — sequential mode-B passes (§7.3, §9.4), then
  concurrent passes — is future work, bounded by the harness's one-preferred-engine-per-pass
  model.

## 16. Glossary

- **Bundle.** A golden data directory as defined by
  [RFC 0011](0011_BundleTestFramework.md), in one of two forms: a **single-graph bundle**
  (`{Name}.json` + optional `.tensor{uid}.bin` + `meta.json`) or a **template-sweep bundle**
  (`graph.template.json` + `sweep.json` + `golden/{CaseId}/...`). The unit a support claim
  attaches to; for a sweep the claim attaches per `cases[].id`.
- **Support claim.** A support-sidecar entry listing an `(engine, arch, platform)` as *supported*
  for a claimed graph — under `{Name}.support.json` for a single-graph bundle, or in the sweep
  `support.json`'s engine group that lists the case id (§5.4). There is no *unsupported* claim;
  an unlisted combination is simply not enforced.
- **Claimed graph.** A graph listed as `supported` for the current `(engine, arch, platform)` —
  a single-graph bundle, or a template-sweep case (§4, §5.4).
- **Claim broken.** A `supported`-claimed `(engine, arch, platform)` whose live support query
  *resolved* but whose engine came back *declined*. The headline failure.
- **Resolved query.** A support query whose status is `OK` or `GRAPH_NOT_SUPPORTED` — the
  harness successfully determined support. The discriminator that separates a usable verdict
  from an *unknown* error (§5.2).
- **Declined.** A resolved query in which the claimed engine's id is **absent** from the ranked
  list (well-formed graph, that engine does not accept it). `GRAPH_NOT_SUPPORTED` is the special
  case where every loaded engine is declined. The per-engine signal treated as a support
  verdict.
- **Support unknown.** A support query that did **not** resolve (any non-decline error status);
  excluded from claim evaluation and surfaced as a note. The resolved/unknown split (status
  code) plus per-engine membership (declined/supported) is what makes the gate neither blind nor
  flaky (§5.2, §7.1, §8).
- **Enforcement level.** How far up the three-rung ladder a bundle is checked:
  `applicability` (query support only), `buildable` (additionally compile plans), or `full`
  (additionally execute + numeric verify). Selected by the bundle's `meta.json`
  `enforcement_level` field (named to avoid collision with RFC 0011's runtime
  `--verification-mode`). Each rung includes every check of the rung below it.
- **Applicability bundle.** A bundle whose `enforcement_level` is `applicability`: validated
  only for engine acceptance, never built, executed, or numerically compared. Typically
  graph-only, but may carry `.bin` files (they are simply ignored in this mode). The home for
  golden-infeasible graphs.
- **Buildable bundle.** A bundle whose `enforcement_level` is `buildable`: validated for engine
  acceptance *and* plan compilation (`create_execution_plans` → `check_support` →
  `build_plans`), but never executed or numerically compared. A stronger signal than
  `applicability` for a graph whose golden data is infeasible. `.bin` files, if present, are
  ignored.
