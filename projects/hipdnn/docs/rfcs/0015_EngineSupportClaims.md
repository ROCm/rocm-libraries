# RFC 0015: Per-Graph Engine Support Claims

- Contributors: Adam Dickin

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

This RFC adds a **per-graph engine-support claim** to the golden reference validation
framework defined in [RFC 0011](0011_GoldenReferenceValidation.md). Each golden bundle
gains an optional machine-managed companion, `{Name}.support.json`, that records — per
`(engine, arch, platform)` — whether the engine is expected to support that exact graph.
At test time the runner queries live engine support for the graph and fails the build when
a graph the bundle **claims** as supported is no longer supported on a claimed
`(arch, platform)`.

Because every claimed graph is an on-disk bundle, the claim key **is that bundle**: the
verdict lives next to the bundle's `{Name}.json` (which conforms to the
[`graph.fbs`](../../flatbuffers_sdk/schemas/graph.fbs) schema and carries exact dims,
strides, per-tensor dtypes, layout, and op parameters), and enforcement re-queries that
same graph. There is no separate fingerprint, no pattern language, no glob matching, and no
region/condensation engine. A regression is a claimed engine flipping from supported to
*declined* (dropping out of the bundle's live support query) for a claimed
`(engine, arch, platform)`.

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

The golden reference framework ([RFC 0011](0011_GoldenReferenceValidation.md)) validates
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
| A graph claimed `unsupported` unexpectedly becomes supported. | Live support query returns supported → **note** ("claimed support != actual support"), not a failure. |

### 3.2 Deliberately not detected

| Non-regression | Rationale |
|----------------|-----------|
| A whole bundle is deleted, removing its claim. | Catalog shrinkage is a code-review concern; the per-tier floor (§7.2, once available) catches whole-bundle loss, and version-control history records it. |
| A claim is eroded *within* a surviving bundle — `support.json` deleted, or a verdict downgraded from `supported` to absent. | **Not** caught by the bundle-count floor (the bundle still discovers and queries). This is accepted declared-graphs-only tail-risk, mitigated only by code review and version-control history. |
| A stale `unsupported` claim masks a later re-loss of support. | If support is granted then lost again while the claim still reads `unsupported`, the loss is by design not gated until `--write-support-claims` is re-run. Consistent with principle 2. |
| An arch has no claim for a graph yet. | New-arch bring-up: absence is "not enforced." The engineer runs `--write-support-claims` once on the new hardware. |

**The narrowness is the load-bearing property.** This framework trades symmetric,
population-wide capability tracking for low-maintenance, exactly-reviewable contracts that
attach to graphs an engineer deliberately chose to pin. A graph regressing while it has no
bundle — or whose claim was explicitly removed — is *out of contract by construction*, not a
missed detection.

## 4. Design Overview

The unit of claim is the **golden bundle** from RFC 0011. A bundle is a directory with a
graph definition (`{Name}.json`) and, for `full` bundles, one `.tensor{uid}.bin` per
tensor. This RFC adds one optional file:

```
{Name}/
  {Name}.json              # graph definition (RFC 0011, unchanged)
  {Name}.tensor{uid}.bin   # tensor data (RFC 0011, full bundles only)
  {Name}.meta.json         # provenance + tolerance (RFC 0011, human + tooling)
  {Name}.support.json      # NEW: machine-managed engine-support claim
```

`{Name}.support.json` is **machine-owned**: it is written wholesale by
`--write-support-claims` (§9) and is never hand-edited, keeping it cleanly separate from the
human-authored `meta.json`. The runner reads it at test time and compares its recorded
verdict against a live engine support query for the exact graph the bundle contains.

Claim evaluation is built on RFC 0011's existing machinery:

- **Discovery must exclude the new sidecar.** Bundle discovery (`scanBundleJsonFiles` in
  `test_sdk/.../FileUtilities.hpp`) recursively scans for `*.json` and registers one test per
  match, already skipping the `meta.json` sidecar by an exact-filename filter. The new
  `support.json` is a third `*.json` in the bundle directory, so the filter **must be extended
  to skip it too** — otherwise a spurious test registers per claim-bearing bundle and tries to
  `loadGraphAndTensors()` on a support file. A `support.json` annotates a bundle that is
  already discovered; it must never create or remove a test. *Acceptance criterion:* a bundle
  directory containing `{Name}.json` + `{Name}.support.json` registers exactly one test, and
  the support sidecar is never loaded as a graph.
- **The claim key is the on-disk bundle**, not a content hash. RFC 0011 derives only a
  *suite name* from `(operation, layout, dtype)` and disambiguates scenarios by the bundle
  directory name; it does not produce a unique per-graph identity. This RFC therefore keys a
  claim to a specific bundle (its `{Name}.json` plus the co-located `support.json`), and
  enforcement re-queries *that bundle's* graph. Two byte-identical graphs in two bundles are
  intentionally two claims. No new naming, pattern, or fingerprint mechanism is introduced —
  the bundle's location is the key, and the graph it holds is what gets re-queried.
- **The verdict** is produced offline on hardware and committed; enforcement is a pure
  comparison at test time.

## 5. Claim Schema

`{Name}.support.json` maps `engine -> arch -> platform -> verdict`:

```json
{
  "version": 1,
  "claims": {
    "MIOPEN_ENGINE": {
      "gfx942":  { "linux": "supported", "windows": "supported" },
      "gfx90a":  { "linux": "unsupported" }
    }
  }
}
```

### 5.1 Field semantics

| Field | Required | Meaning |
|-------|----------|---------|
| `version` | yes | Schema version of the support file. A reader refuses a version it does not understand (loud, not silent) and the engineer regenerates. |
| `claims` | yes | Map of engine name → arch → platform → verdict. |
| engine key | — | Exact engine name as reported by `getEngineInfo`. |
| arch key | — | Exact match against the arch token — the prefix before the first `:` of the raw `gcnArchName` (e.g. `gfx942:sramecc+:xnack-` → `gfx942`).
| platform key | — | Exact `"linux"` or `"windows"`. |
| verdict value | — | `"supported"` or `"unsupported"`. |

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

Verdict semantics:

- `supported` claim → the engine **must** return *supported*. A *declined* result → **claim
  broken FAIL** (§7). An *unknown* result → **errored-before-assert FAIL** (§7.1) — the claim
  could not be checked; fix the error.
- `unsupported` claim → the engine is expected to return *declined*. Returning *supported*
  emits a **note** (§7), never a failure — capability growth must not break CI. An *unknown*
  result → note, claim left unevaluated.
- A missing `(engine, arch, platform)` entry → **not enforced** for that combination.

### 5.3 Loader rules

- All keys are matched exact-string. No `*`, no `?`, no fnmatch.
- Unknown verdict values are rejected (catches typos / schema drift).
- An empty `claims` map is legal (a bundle with no current claims).
- The file is optional; a bundle with no `support.json` is simply not support-gated.

## 6. Enforcement Levels

A bundle's enforcement level is selected by a new **`enforcement_level`** field in
`meta.json` (RFC 0011's companion schema is extended to carry it), defaulting to `full`. The
name is deliberately *not* `verification`, to avoid colliding with RFC 0011's runtime
`--verification-mode` CLI flag — they are orthogonal axes (see §6.1).

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

- [ ] `meta.json` carries `enforcement_level` ∈ {`applicability`, `buildable`, `full`};
      default is `full` when absent **for non-claim-bearing bundles**, and a **hard pre-commit
      error** when a `support.json` exists but `enforcement_level` is missing or invalid (a typo
      must never silently flip how far up the ladder a claim-bearing bundle is checked).
- [ ] A claim-bearing bundle at any level returning *declined* at that level's stop-point FAILs
      (not skips) under every `--verification-mode`, and the failure names the rung that broke.
- [ ] A `buildable` bundle queries support and builds plans but never generates inputs,
      executes, or compares; a plan-build failure on a `supported` claim FAILs.
- [ ] A `buildable` or `applicability` bundle is evaluated under `golden` mode and is never
      skipped for lacking `.bin`.
- [ ] An `applicability` bundle never builds plans, generates inputs, executes, or compares.

## 7. Outcome States and Guards

### 7.1 Per-graph outcomes

Each claimed `(engine, arch, platform)` is evaluated **independently** from the same support
query, using the per-engine verdict of §5.2 (status code → *resolved* vs *unknown*; membership
→ *supported* vs *declined*):

- **Claim broken.** A `supported` claim whose query *resolved* but whose engine is *declined*
  (its id absent from the ranked list — `GRAPH_NOT_SUPPORTED` when it is the only/last engine)
  → **FAIL**. The message names the bundle, the `(engine, arch, platform)`, and the support file
  location, and links to the failure runbook (§11). This is the real-regression case.
- **Errored before assert.** A `supported` claim whose query did **not** resolve (build
  failure, backend error, OOM, device lost), or whose graph build itself crashes → **FAIL**
  ("fix the underlying error first"). Distinct from *declined*: the verdict could not be
  evaluated, so this is not a coverage-loss verdict but it still must not pass green.
- **Support unknown.** For an `unsupported` (or unclaimed) graph, an unresolved query → **note**,
  claim left unevaluated. Support is *unknown*, not *declined*; it must never be treated as a
  verdict.
- **Unexpected support.** An `unsupported` claim whose engine is *supported* (present in the
  list) → **note** ("claimed support != actual support; update the claim if intentional").

The split between *declined* and *unknown* is the whole reason §5.2/§8 key on the status code
for the first step: collapsing an error status into "declined" would force a choice between
flaky CI (every error → FAIL) and missed regressions (every error → note). A *resolved* query
(`OK` or `GRAPH_NOT_SUPPORTED`) is the only one that produces a verdict; within it, an engine's
absence from the ranked list is its *declined* signal. When several engines are loaded, each
claimed engine gets its own verdict from the one query — one engine's `declined` does not taint
another's `supported`.

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

A run selects which engine(s) it drives through two existing harness flags — `--test-article`
(the plugin shared library to load) and `--test-engine` (an engine within it). Three modes
result, and they determine both how deep enforcement reaches and whether the run can generate
claims:

| Mode | Flags | Engines exercised | Enforce | Generate |
|------|-------|-------------------|---------|----------|
| **A — auto-select** | neither (default plugin config) | hipDNN's heuristic picks the top-ranked engine **per graph** (existing behaviour) | yes | **no** |
| **B — plugin, all engines** | `--test-article <plugin>` only | **every** engine the plugin exposes, run **sequentially** — one full pass per engine | yes | **yes** |
| **C — plugin + engine** | `--test-article <plugin>` + `--test-engine <engine>` | exactly that one engine | yes | **yes** |

- **Support (applicability) enforcement is multi-engine in every mode.** The support query spans
  all loaded engines, so each claimed `(engine, arch, platform)` is attributed by ranked-list
  membership (§5.2, §8) regardless of mode. There is **no refuse-to-run** on multiple plugins —
  for the support gate, N plugins are simply N engines, each evaluated independently; one
  engine's `declined` never taints another's `supported`.
- **Deep (`buildable`/`full`) checks follow the executed engine.** Building and executing pin a
  single preferred engine per pass. Mode A runs whichever engine the heuristic auto-selects —
  which can differ graph to graph — so it cannot attribute a `buildable`/`full` verdict to a
  fixed engine and therefore **does not generate**. Mode B enumerates the plugin's engines
  (`hipdnnGetEngineCount_ext` + `getEngineInfo`) and runs the suite once per engine in sequence,
  pinning each in turn, so every rung is attributable. Mode C pins the one named engine.
- **Generation needs a pinned engine, which modes B and C provide.** `--write-support-claims`
  runs in mode B (refresh every engine the plugin returns, sequentially, in one invocation) or
  mode C (refresh one engine). Each pass writes only its own engine's disjoint keys (§9.2), and
  because the plugin's engines run **sequentially** the §9.2 "one writer at a time" property
  holds — a single mode-B run can refresh a whole plugin's engines on the current ASIC. Mode A
  is run-only.

A claim for an engine that is **not loaded** in the current run is *not enforced* (no query
observed it; the empty-query guard of §7.2 still catches a run that observed *nothing*). The
claim check is scoped **per-arch**: enforcement proceeds whenever a loaded engine has *any*
claim for the current `(arch, platform)`; a brand-new arch with no claims yet is the
not-enforced bring-up case (§3.2), not a refuse-to-run.

### 7.4 Acceptance criteria

- [ ] A `supported` claim whose engine is *declined* (absent from the ranked list of a resolved
      query) → claim-broken FAIL.
- [ ] An unresolved query (non-decline error status) on a `supported` claim →
      errored-before-assert FAIL.
- [ ] Unresolved query on an `unsupported`/unclaimed graph → note, not FAIL.
- [ ] *supported* (present in list) on an `unsupported` claim → note, not FAIL.
- [ ] Enforcement with zero observed queries → FAIL (empty-query guard).
- [ ] New-arch run with no claims for the loaded engine on that arch → not-enforced, not a
      refuse-to-run.
- [ ] A run with **two or more plugins loaded** enforces each engine's claims independently
      from one query (no refuse-to-run); one engine *declined* while another is *supported*
      yields a FAIL for the first and a pass for the second.
- [ ] A claim for an engine not loaded in the current run is not enforced.

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
   Extending it to N engines is iterating that test per loaded engine; see §7.3.

## 9. Write Tool and Contract

### 9.1 CLI

```bash
# Mode C — one engine of one plugin
./hipdnn_integration_tests \
    --golden-data-dir dnn-providers/integration-tests/integration_test_bundles \
    --test-article <plugin-lib> --test-engine MIOPEN_ENGINE \
    --write-support-claims

# Mode B — every engine the plugin exposes, run sequentially
./hipdnn_integration_tests \
    --golden-data-dir dnn-providers/integration-tests/integration_test_bundles \
    --test-article <plugin-lib> \
    --write-support-claims
```

Discovers bundles, observes live support on the **current** hardware, and writes each bundle's
`support.json` verdict for the exercised `(engine, arch, platform)` only. Every other key in the
file — other engines, arches, and platforms — is left byte-untouched. The main bundle, its
graph, `.bin` files, and `meta.json` are never modified.

Generation requires a **pinned engine**, so it runs in **mode B or C** (§7.3), never mode A: a
`buildable`/`full` verdict means compiling (and executing) one specific engine's plan, and even
the `applicability` rung is written per-engine so a run's scope is unambiguous. Mode C pins one
engine via `--test-engine` (the existing selector drives the harness's per-engine support check,
`TestConfig::getEngineName` → membership test). Mode B omits `--test-engine`: the run enumerates
the plugin's engines and writes each one's keys in sequence, refreshing the whole plugin on one
ASIC in a single invocation. Either way each engine writes disjoint keys, so results merge
cleanly (§9.2); broad multi-ASIC coverage can instead accumulate from CI observations (§12)
without a dedicated session.

This engineer-driven write is one of two ways claims get refreshed; the other is the CI
observation harvest (§12), where routine enforcement runs *emit* what they observe and an
offline consumer proposes claim updates. Both share the same observe-then-write split — only
the actor and cadence differ.

### 9.2 Write contract

- **Machine-owned, engineer-driven.** CI never runs this tool — auto-applying it would
  silently rewrite the contract. The engineer runs it on target hardware, reviews the
  `git diff`, and commits.
- **Surgical, idempotent writes.** A write reads the current `support.json`, updates only the
  current `(engine, arch, platform)` keys, and re-emits the whole file with canonical JSON
  (sorted keys, fixed number formatting, stable newline) so sibling keys are byte-identical
  and re-running with no change produces a zero diff. Because different engines, arches, and
  platforms write disjoint keys, per-engine and per-arch regeneration on separate checkouts
  produces git diffs that merge cleanly. No multi-writer coordination is needed: only one
  integration-test binary runs at a time, so concurrent writes from multiple threads/processes
  are not a scenario.
- **Written at end of sweep.** The tool writes each bundle's `support.json` directly after the
  test sweep completes. If a target file cannot be opened for writing (e.g. it is locked), the
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

- **Require an explicit `enforcement_level` on claim-bearing bundles.** A bundle that carries a
  `support.json` must declare `enforcement_level` ∈ {`applicability`, `buildable`, `full`};
  missing or invalid is a **hard error** (never silently defaulted), so a typo cannot flip how
  far up the ladder the bundle is checked. RFC 0011's existing `.bin` data-integrity
  checks are unchanged and apply per its own rules (e.g. a bundle that ships `.bin` files must
  ship them for every tensor UID); this RFC adds no new `.bin` presence/absence requirement.
- Validate `support.json` against the schema (§5): known `version`, well-formed
  `engine/arch/platform` keys, verdict values in the allowed set. A bad `version` or unknown
  verdict is a pre-commit FAIL (not a runtime surprise).

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

This RFC adds a **second variant for mode B** that registers one ctest running every engine the
plugin exposes, sequentially:

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
existing single-engine helper is unchanged; the two coexist (a provider registers whichever
fits).

### 9.5 Acceptance criteria

- [ ] Re-running `--write-support-claims` with no support change produces a zero git diff
      (canonical-JSON idempotency).
- [ ] A write run with an empty/absent observation set does not null an existing `supported`
      verdict (empty-write guard).
- [ ] A claim-bearing bundle with a missing/invalid `enforcement_level` fails pre-commit.
- [ ] A mode-B run (`--test-article` only) enumerates the plugin's engines and writes a verdict
      for each; a mode-C run writes only the named engine's keys; a mode-A run writes nothing.
- [ ] The all-engines CMake helper registers one ctest that exercises every engine the plugin
      exposes, sequentially.

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

**Generated from committed data, not a hardware run.** The engine support matrix is generated
**purely from the data checked into the repo** — each bundle's `{Name}.json` (for the row
identity: operation, layout, dtype, scenario) and its `{Name}.support.json` (for the
per-`(engine, arch, platform)` verdict). This **replaces** generating the matrix from a
runtime `SupportMatrixCollector` populated during a live test run: because every claimed
graph already carries its verdict on disk (§4), the committed bundles and their `support.json`
sidecars are a complete, hardware-free description of claimed support. A contributor can
regenerate or diff the matrix from a plain checkout with no GPU, no plugin, and no DVC pull
(only `.json`/`.support.json` are needed, and both live in git — §9.2).

**Renderer.** `dnn-providers/integration-tests/tools/render_support_matrix.py` discovers every
`{Name}.support.json` under the golden-data tree, joins each to its co-located `{Name}.json`,
and emits one combined markdown matrix: engine columns, one row per bundle keyed by
`(operation, layout, dtype, scenario)`, each cell showing the verdict per `(arch, platform)`.
The matrix is a **view** of the committed claims, not a second state to maintain.

**Sidecar as source of truth.** The `support.json` sidecars are authoritative; the matrix is
derived. The matrix is regenerated (and its freshness checked) by a CI lint that fails if the
committed matrix differs from a re-render of the current sidecars — so the committed doc can
never silently drift from the claims.

Two clarifications. (1) The matrix row key `(operation, layout, dtype, scenario)` is
*coarser* than the actual claim key (the on-disk bundle / exact graph, §4): two bundles
sharing that 4-tuple but differing in exact shape are distinct claims that would collapse to
one matrix row. This is a presentation simplification, not a contract — the per-bundle
`support.json` files remain the source of truth; the matrix should make the bundle path
reachable per row so the full claim set is recoverable. (2) A runtime `SupportMatrixCollector`
already exists on the live-compute harness; this renderer reads the committed `support.json`
sidecars instead, a deliberately separate source of truth — the two must not be conflated
into divergent matrices.

**Runbook linking.** Every claim-broken / errored-before-assert message ends with a stable
anchor into a failure-mode runbook (`docs/support-claims-failures.md`), one section per
outcome state (§7), each answering: what triggered it, what it means, ranked likely causes,
and step-by-step remediation. Anchor names and verifier output are co-designed; renaming an
anchor without updating the verifier fails CI lint. Schema and bring-up docs
(`support-claims-schema.md`, `support-claims-bringup.md`) land in the same PR as the
implementation; PRs that change schema or verifier behaviour without updating the docs fail
review.

## 12. CI Support-Observation Harvest

§9's `--write-support-claims` tool refreshes claims by running the suite on one target ASIC.
But CI is where the broadest matrix of ASICs and platforms actually runs, nightly — and every
enforcement run already queries support for each claim-bearing bundle (§7). Rather than require
a separate engineer-driven hardware session per ASIC, this section lets CI **emit the support
it observes** as structured data, so an offline consumer can harvest newly-supported graphs and
propose `support.json` updates. Nightly CI thereby becomes a continuous stream of support
observations across every ASIC it runs.

This preserves the §9.2 invariant that **CI never writes claims**: CI only *emits observations*
(read-only with respect to the contract); turning observations into committed claims is an
offline, engineer-reviewed step.

> **Provisional.** The concrete near-term intent is simply to **print these observations into
> the CI logs** so we can collect them after the fact, because we do not yet fully know what
> artifact storage / data egress the CI environment will allow. The structured-record and
> JSONL-artifact design below is the direction we want; the exact transport may change once we
> learn what is actually retrievable from CI. Treat §12 as the target shape, not a frozen
> contract.

### 12.1 Emitting observations

An enforcement run (`--enforce-support-claims`) gains a side output,
`--emit-support-observations=<path>`, that writes one record per observed `(bundle, engine)`
regardless of verdict. Emission is a pure side effect — it never changes enforcement
pass/fail. The observed verdict is the per-engine **support** signal (ranked-list membership,
§5.2) — the applicability-level "engine accepts this graph" fact — and is derived for **every
loaded engine** from the single query, *independent of the bundle's `enforcement_level`*. So a
nightly run with several plugins loaded emits a support observation for each loaded engine in
one pass, even though the deeper `buildable`/`full` pass-fail (surfaced as a CI FAIL, §7.1) is
attributed only to the engine whose plan was built/executed. Each record carries what the
offline consumer needs to attribute and apply it:

```jsonl
{"schema":"support_observation/1","bundle":"quick/ConvFwd/nhwc/fp16/resnet50_layer3","engine":"MIOPEN_ENGINE","arch":"gfx942","platform":"linux","verdict":"supported","enforcement_level":"full","rocm_version":"6.4.0","commit":"<sha>","run_id":"<ci-run>","timestamp":"<iso8601>"}
```

- `verdict` ∈ `supported | declined | unknown`, using the per-engine mapping of §5.2
  (status code → resolved/unknown; ranked-list membership → supported/declined).
- `bundle` is the repo-relative bundle path — the same on-disk key a claim attaches to (§4) —
  so the consumer can locate the exact `support.json` to update.
- Provenance (`rocm_version`, `commit`, `run_id`, `timestamp`) lets the consumer break ties
  (prefer the newest observation) and audit where a claim came from.

**Two transports.** The transport we can rely on first is the **CI log**: each record is
emitted to stdout behind a stable tag — `[[HIPDNN_SUPPORT_OBS]] {json}`, one per line — so it
can be scraped from a build log after the fact with no artifact-storage dependency. Where the CI
job *can* retain artifacts, the same records are additionally written as a **JSONL artifact**
(`support_observations.jsonl`) for direct upload. Which of these is primary depends on what the
CI environment supports (see the provisional note above); both are append-only and union-safe.

### 12.2 Shard- and ASIC-tolerant by construction

Each record is a self-contained per-`(bundle, engine, arch, platform)` fact, so:

- **Sharding.** Each shard emits only the bundles it ran; the consumer unions all shards'
  records. Unlike enforcement's empty-observation floor (§7.2), harvesting needs no
  completeness guarantee from a single process — the union across shards/runs is what matters.
- **Multiple ASICs.** Different ASICs' nightly runs write disjoint `(arch, platform)` keys, so
  their observations never conflict — the same disjoint-key property as the §9.2 writes.
- **Multiple engines.** A multi-plugin run emits one record per loaded engine (disjoint
  `engine` keys), so the harvest gains support data for several engines from a single run — the
  same disjoint-key property, on the engine axis.

### 12.3 Harvesting into claims (offline, reviewed)

An offline consumer (`tools/harvest_support_observations.py`) ingests observation records from
one or more CI runs and proposes `support.json` updates:

- **Newly-supported graphs (the stream).** A bundle observed `supported` whose `support.json`
  has no entry — or an `unsupported` entry — for that `(engine, arch, platform)` → propose
  setting it to `supported`. This is the continuous gain stream from nightly CI: as an engine
  picks up coverage on an ASIC, the next nightly run observes it and the harvest surfaces it.
- **Monotonic by policy — harvest only adds/upgrades, never silently removes.** A
  `supported → declined` flip is either a real regression (already a CI **FAIL** under §7.1, so
  it surfaces loudly, not via silent harvest) or needs human judgement; the harvest never
  auto-downgrades a committed `supported` verdict. `unknown` observations are ignored (not a
  verdict, §5.2).
- **Output is a reviewed change, not an auto-commit.** The consumer emits a proposed
  `support.json` diff (or opens a gated bot PR). An engineer reviews and merges it, keeping the
  §9.2 "CI never writes the contract" invariant intact. The surgical canonical-JSON write
  format (§9.2) makes the proposed diff minimal and reviewable.

### 12.4 Full vs incremental refresh

> *Does a full refresh of an ASIC require a full run?*

A **complete** refresh of an ASIC requires that **every claim-bearing bundle has been observed
at least once on that ASIC** — completeness is "every bundle observed ≥ 1×," however achieved.
Two ways to get there:

- **One dedicated full run** — `--write-support-claims` (or `--enforce-support-claims
  --emit-support-observations`) over the entire suite, unsharded and unfiltered, on that
  hardware, in a single pass (§9).
- **Accumulated CI observations** — the union of nightly enforcement runs' emitted
  observations, once they have collectively covered every tier and shard for that ASIC. No
  dedicated hardware session is required; the coverage falls out of routine CI over time.

So a full refresh still requires full *coverage*, but not necessarily a single full *run*: the
harvest model lets that coverage accumulate from CI instead of a manual session. The consumer
reports per-ASIC observation coverage (which bundles have and have not been observed) so an
engineer knows when an ASIC's picture is complete.

### 12.5 Acceptance criteria

- [ ] `--emit-support-observations` writes one JSONL record per observed `(bundle, engine)`
      with verdict + provenance, and does not alter enforcement pass/fail.
- [ ] The same records are emitted to stdout under the `[[HIPDNN_SUPPORT_OBS]]` tag.
- [ ] The harvest consumer unions records across shards/runs and proposes only additions or
      upgrades to `supported`; it never auto-downgrades a committed `supported` verdict and
      ignores `unknown`.
- [ ] The harvest output is a reviewable diff / gated PR, never a direct CI commit.
- [ ] The consumer reports per-ASIC observation coverage (observed vs total claim-bearing
      bundles).

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
new capability and churn CI constantly. **Rejected** in favour of the note in §7.1
("unexpected support"): absence of a claim is "not enforced," and capability growth is
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
| With multiple plugins loaded, a declined engine is mis-attributed or taints another engine's verdict. | Per-engine attribution by ranked-list membership (§5.2, §7.3, §8): each claimed engine's verdict is read independently from the one query; one engine's `declined` never affects another's. |
| A claim is eroded within a surviving bundle (support.json deleted / verdict downgraded). | Accepted declared-graphs-only tail-risk (§3.2); mitigated by code review and version-control history rather than a dedicated guard. |
| A no-GPU or plugin-load-failure run silently passes or nulls claims. | Empty-query guard and empty-write guard (§7.2, §9.2). |
| CI harvest (§12) proposes a wrong/over-broad `supported` upgrade. | Harvest is monotonic (adds/upgrades only), ignores `unknown`, and never auto-commits — every proposed change is an engineer-reviewed diff/PR (§12.3). Provenance fields let the reviewer audit which run/commit/ROCm version produced an observation. |
| Stale or spoofed observation logs corrupt the harvest. | Observations carry `commit`/`run_id`/`rocm_version`/`timestamp`; the consumer prefers newest and the result is reviewed before merge. Logs are inputs to a proposal, never directly the contract. |

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
  the proposed `support.json` PR automatically from nightly observation artifacts (still
  merged by a human) is a natural follow-on once the harvest consumer is proven.
- **Harvesting the full runtime population.** If runtime-fixture graphs also emit observations
  (not just bundle-backed ones), the harvest could surface support data for graphs that do not
  yet have bundles, informing which `applicability` bundles are worth authoring. Depends on the
  fixture path emitting the same observation records.
- **Per-engine index.** If per-bundle `support.json` write churn is unsustainable at the
  full bundle population, move claims to a per-engine index keyed by bundle identity — still
  no region matching, same verdict semantics.
- **Sharding.** Enforcement assumes an unsharded run so the empty-observation floor (§7.2) is
  meaningful. A per-shard record + reduce phase is future work, shared with the golden
  framework's own sharding story.
- **Concurrent deep enforcement across engines.** Multi-engine *support* (applicability)
  enforcement is in scope from one query (§7.3), and `buildable`/`full` coverage of all of a
  plugin's engines is in scope **sequentially** via mode B (one pinned pass per engine). What is
  *not* yet done is running those per-engine passes **concurrently** — each still pins one
  preferred engine, so a mode-B run is as long as the sum of its engines. Parallelising the
  per-engine passes (and the cross-plugin case) is future work, bounded by the harness's
  one-preferred-engine-per-pass model.

## 16. Glossary

- **Bundle.** A golden data directory (`{Name}.json` + optional `.tensor{uid}.bin` +
  `meta.json`) as defined by [RFC 0011](0011_GoldenReferenceValidation.md). The unit a
  support claim attaches to.
- **Support claim.** A `{Name}.support.json` entry asserting, per `(engine, arch,
  platform)`, that the engine is expected to support (or not support) the bundle's graph.
- **Claimed graph.** A graph whose bundle has a `supported` verdict for the current
  `(engine, arch, platform)`.
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
