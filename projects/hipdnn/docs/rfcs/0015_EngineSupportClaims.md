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
region/condensation engine. A regression is a single bundle's live support query flipping
from supported to `GRAPH_NOT_SUPPORTED`.

A second enforcement level, **applicability**, lets a graph-only bundle assert support
*without executing or comparing outputs*. This is the home for graphs whose golden data is
infeasible to generate or compare (very large tensors, long-sequence attention), and for
cheap "this shape must stay supported" declarations that need no reference data at all.

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
3. **Applicability is separable from correctness.** "The engine must accept this graph" and
   "the engine must produce correct numbers for this graph" are distinct assertions. The
   framework lets a bundle assert the first without paying for the second.

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
| A graph that a bundle claims `supported` loses engine support on a claimed `(arch, platform)`. | Live support query returns the *declined* verdict (`GRAPH_NOT_SUPPORTED`) for the claimed graph → **FAIL** (replaces the silent `GTEST_SKIP`). See §5.2 / §7.1 for how "declined" is distinguished from "errored". |
| A graph with a `supported` claim hits a non-decline error (build failure, infrastructure fault) before its verdict can be evaluated. | The claim could not be assessed → **FAIL** ("errored before the claim could be checked; fix the error first"). Distinct from the *declined* verdict above, which is the real-regression case. |
| A graph claimed `unsupported` unexpectedly becomes supported. | Live support query returns supported → **note** ("claimed support != actual support"), not a failure. |

### 3.2 Deliberately not detected

| Non-regression | Rationale |
|----------------|-----------|
| A graph with **no bundle** loses or gains support. | The contract is declared-graphs-only (principle 2). A graph nobody authored a bundle for is not part of the contract. To gate a graph, give it a bundle — an `applicability` bundle is enough. |
| A whole bundle is deleted, removing its claim. | Catalog shrinkage is a code-review concern; the per-tier floor (§7.2, once available) catches whole-bundle loss, and version-control history records it. |
| A claim is eroded *within* a surviving bundle — `support.json` deleted, or a verdict downgraded from `supported` to absent. | **Not** caught by the bundle-count floor (the bundle still discovers and queries). This is part of the accepted declared-graphs-only tail-risk; the claim-diff guard in §9.3 is the mitigation, not the floor. |
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

- **Discovery** is unchanged: `discoverGoldenBundles` recursively registers one test per
  bundle. A `support.json` does not create or remove tests; it annotates the bundle that is
  already discovered.
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
| arch key | — | Exact match against `archTokenOf(gcnArchName)` — the prefix before the first `:` of the raw `gcnArchName` (e.g. `gfx942:sramecc+:xnack-` → `gfx942`). Substring matching is rejected; it would collide families. **`archTokenOf` is net-new** (today only `currentDeviceArchRaw()` exists, which returns the raw string and whose comment recommends *substring* matching); this RFC introduces the exact-prefix helper and the loader must use it, not substring matching. |
| platform key | — | Exact `"linux"` or `"windows"`. |
| verdict value | — | `"supported"` or `"unsupported"`. |

### 5.2 Verdict meaning

The support query (`get_ranked_engine_ids`) returns a status plus a ranked id list. This RFC
reads it as three outcomes, **keyed on the specific status code** (see §8 for why
status-code keying, not list-emptiness, is the correct discriminator):

- **supported** — good status with a non-empty ranked list.
- **declined** — the dedicated *graph-not-supported* status (`GRAPH_NOT_SUPPORTED`): the
  graph is well-formed but no engine accepts it. This is the clean "engine doesn't support
  this" verdict.
- **unknown** — any *other* error status (backend error, OOM, device lost, build failure):
  the query could not produce a support verdict.

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

| Level | Needs `.bin`? | Action |
|-------|---------------|--------|
| `full` (default) | yes (golden) or runtime-computed | Build the graph, execute the engine, numeric-verify against golden data. If a `support.json` claims `supported` and the engine returns *declined*, this is a **claim-broken FAIL** instead of RFC 0011's silent skip. |
| `applicability` | no | Build the graph, query engine support, assert the verdict against `support.json`. **No input generation, no execution, no reference, no comparison.** |

`applicability` is a third behaviour of RFC 0011's graph-only bundles. RFC 0011 today
defines a graph-only bundle as one that still generates inputs, runs the engine *and* a
reference, and compares at runtime; an `applicability` bundle branches **before input
generation** and stops after the support query. The level is chosen explicitly by
`enforcement_level`, never inferred from the mere absence of `.bin` files (a `full` bundle
whose `.bin` files have not yet been generated is still `full`, not `applicability`).

This makes the support gate available to graphs that can never have golden data, and lets an
engineer pin "this shape must remain supported" with a graph-only bundle that costs nothing
to store or run.

### 6.1 Composition with RFC 0011 `--verification-mode`

`enforcement_level` (per-bundle: *what kind of check this bundle gets*) and RFC 0011's
`--verification-mode` (run-wide: *which reference source supplies expected outputs* —
`auto`/`golden`/`gpu`/`cpu`) are independent. **Support-claim enforcement runs for every
claim-bearing bundle regardless of `--verification-mode`** — it depends only on the live
support query, not on any reference source. In particular:

| `--verification-mode` | `full` claim-bearing bundle | `applicability` bundle |
|-----------------------|-----------------------------|------------------------|
| `auto` / `golden` / `gpu` / `cpu` | Support claim is always evaluated. The reference-source selection governs only the *numeric* comparison; it never gates the support check. | Support claim is always evaluated. The bundle is **never** subject to RFC 0011's "golden mode skips bundles with no `.bin`" rule — an applicability bundle has no `.bin` by definition and must not be skipped in the very mode (`golden`, used by Nightly/Weekly CI) meant to enforce it. |

This is a normative override of RFC 0011's golden-mode no-`.bin` SKIP for claim-bearing
applicability bundles, and must be stated in the RFC 0011 verification-modes section when
that schema extension lands.

### 6.2 Acceptance criteria

- [ ] `meta.json` carries `enforcement_level` ∈ {`full`, `applicability`}; default is `full`
      when absent **for non-claim-bearing bundles**, and a **hard pre-commit error** when a
      `support.json` exists but `enforcement_level` is missing or invalid (a typo must never
      silently turn an `applicability` bundle into a `full` bundle missing all `.bin`).
- [ ] A `full` claim-bearing bundle returning *declined* FAILs (not skips) under every
      `--verification-mode`.
- [ ] An `applicability` bundle is evaluated under `golden` mode and is never SKIPped for
      lacking `.bin`.
- [ ] An `applicability` bundle never generates inputs, executes, or compares.

## 7. Outcome States and Guards

### 7.1 Per-graph outcomes

The discriminator is the support query's **status code** (§5.2, §8), not list-emptiness:

- **Claim broken.** A `supported` claim whose query returns the *declined* status
  (`GRAPH_NOT_SUPPORTED`) → **FAIL**. The message names the bundle, the
  `(engine, arch, platform)`, and the support file location, and links to the failure
  runbook (§11). This is the real-regression case.
- **Errored before assert.** A `supported` claim whose query returns any *other* error status
  (build failure, backend error, OOM, device lost), or whose graph build itself crashes →
  **FAIL** ("fix the underlying error first"). Distinct from *declined*: the claim could not
  be evaluated, so this is not a coverage-loss verdict but it still must not pass green.
- **Support unknown.** For an `unsupported` (or unclaimed) graph, a non-decline error status
  → **note**, claim left unevaluated. Support is *unknown*, not *declined*; it must never be
  treated as a verdict.
- **Unexpected support.** An `unsupported` claim whose engine returns *supported* → **note**
  ("claimed support != actual support; update the claim if intentional").

The split between *declined* and *unknown* is the whole reason §5.2/§8 key on the specific
status code: collapsing both into "is_bad" would force a choice between flaky CI (every error
→ FAIL) and missed regressions (every error → note). `GRAPH_NOT_SUPPORTED` is the engine's
contractual signal for "well-formed but unrunnable," and is the only error status treated as
a verdict.

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

### 7.3 Preconditions (refuse-to-run)

Enforcement refuses if more than one distinct plugin is loaded (multi-engine attribution is
out of scope here). The single-plugin engine-name check is scoped **per-arch**: enforcement
proceeds whenever the loaded engine has *any* claim for the current `(arch, platform)`; a
brand-new arch with no claims yet is the not-enforced bring-up case (§3.2), not a
refuse-to-run.

### 7.4 Acceptance criteria

- [ ] *declined* (`GRAPH_NOT_SUPPORTED`) on a `supported` claim → claim-broken FAIL.
- [ ] Any non-decline error on a `supported` claim → errored-before-assert FAIL.
- [ ] Non-decline error on an `unsupported`/unclaimed graph → note, not FAIL.
- [ ] *supported* on an `unsupported` claim → note, not FAIL.
- [ ] Enforcement with zero observed queries → FAIL (empty-query guard).
- [ ] New-arch run with no claims for the loaded engine on that arch → not-enforced, not a
      refuse-to-run.

## 8. Required Harness Capability

The `applicability` level and the "errored before assert" outcome both require the harness
to **build a graph and query engine support without executing it**:

> Given a `{Name}.json`, produce a queryable `hipdnn_frontend::graph::Graph`, finalize it
> (`validate` → `build_operation_graph`) **with no tensor data**, and call
> `get_ranked_engine_ids`, mapping the result to supported / declined / unknown by status
> code (§5.2).

This is a first-class requirement of this RFC, not an optional optimization. It rests on
three points; the first is verified, the latter two are **ASSUMPTIONS that require a frontend
spike before the `applicability` level is committed**:

1. **Status-code tri-state is available (verified).** `get_ranked_engine_ids` returns a
   status plus an id vector; the dedicated `GRAPH_NOT_SUPPORTED` code signals "well-formed but
   no engine accepts it," distinct from other error codes. The existing live-compute harness
   collapses these into a single branch only because it never needed to distinguish them;
   this RFC reads the specific code instead. *This is why §5.2/§7.1 key on the status code,
   not on list-emptiness — there is no "clean empty list" distinct from an error status.*
2. **Build-without-execute path (ASSUMPTION).** RFC 0011's `loadGraphAndTensors` yields a
   *FlatBuffers* graph buffer, not a `frontend::graph::Graph`; `get_ranked_engine_ids` is a
   method on the latter, reached only after the frontend builder runs. A bundle-JSON →
   `frontend::graph::Graph` bridge is **new work** (the existing `GraphUnpacker` unpacks a
   post-build backend descriptor, not a bundle's JSON). Note the live-compute harness already
   queries support *before* tensor generation (and has a `skipGraphValidation` early-exit that
   is essentially applicability today), so the *capability* exists on that path — it is the
   bundle-JSON-to-queryable-graph bridge that must be built for the golden path.
3. **Finalize succeeds with no tensor data (ASSUMPTION).** Finalize performs id/type
   inference ("tensor Ids and types may be inferred during graph finalization"), which may
   require shape/dtype completeness even without host/device buffers. The §9.3
   dtype-resolution pre-commit check is the hedge. If finalize hard-requires tensors, the
   `applicability` path allocates **metadata-only tensors** (no host/device data); treat this
   as the default path until the spike proves bare finalize works.

## 9. Write Tool and Contract

### 9.1 CLI

```bash
./hipdnn_integration_tests \
    --golden-data-dir dnn-providers/integration-tests/golden_reference_data \
    --write-support-claims
```

Discovers bundles, observes live engine support for each on the **current** hardware, and
writes each bundle's `support.json` verdict for the current `(arch, platform)` only. Other
arches' and platforms' verdicts in the same file are left untouched. The main bundle, its
graph, `.bin` files, and `meta.json` are never modified.

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
  and re-running with no change produces a zero diff. **Single-writer per working copy.** The
  write unit is the whole file (read-modify-rewrite), so two writers racing against the *same*
  checkout are last-writer-wins at file granularity and would silently drop a sibling arch's
  verdict. Concurrent per-arch regeneration must therefore run on **disjoint working copies**;
  the resulting property is "sequential edits / separate checkouts produce disjoint git diffs
  that merge cleanly," not "concurrent in-place edits are safe." (A future per-engine index,
  §15, would need an explicit advisory lock or re-read-merge instead.)
- **Atomic.** The file is generated in memory, written to `{Name}.support.json.tmp.<pid>`,
  flushed, and atomically renamed over the target (`MoveFileEx` with replace+write-through on
  Windows; close+fsync+rename on POSIX, **plus fsync of the containing directory** so the
  rename itself is durable). On failure the tmp is unlinked and the original is never
  partially overwritten. Stray `.tmp.<pid>` files from a killed process are gitignored and
  rejected by the pre-commit verifier (§9.3).
- **DVC-independent.** Querying support needs only the graph (`.json`), never the
  (potentially multi-GB) `.bin` files. The write and `applicability` passes must not trigger
  a golden-data pull.
- **Empty-write guard.** The tool refuses to overwrite an existing `supported` verdict with
  an absent/empty observation set (no GPU, plugin load failure), so a degenerate run cannot
  null out committed claims.

### 9.3 Validation (pre-commit)

The RFC 0011 pre-commit bundle verifier is extended to:

- **Read `enforcement_level` first**, then branch: apply RFC 0011's existing "every tensor
  UID has a `.bin`" check (Data Integrity #6) only when `enforcement_level: full`, and apply
  the no-`.bin` requirement only when `enforcement_level: applicability`. A missing/invalid
  `enforcement_level` on a claim-bearing bundle is a **hard error** (never defaulted to
  `full`), so a typo cannot silently reclassify an applicability bundle as a full bundle
  missing all its `.bin`.
- Validate `support.json` against the schema (§5): known `version`, well-formed
  `engine/arch/platform` keys, verdict values in the allowed set. A bad `version` or unknown
  verdict is a pre-commit FAIL (not a runtime surprise).
- Require that an `applicability` bundle has **no** `.bin` files; reject stray
  `.support.json.tmp.<pid>` artifacts.
- Require that a claim-bearing bundle's `{Name}.json` is **fully dtype-resolved** — every
  operand tensor carries an explicit `data_type` rather than relying on context fill — so the
  on-disk graph the claim attaches to is unambiguous and matches what the engine dispatches
  on at finalize time. (Spike dependency: confirm the check can detect an unresolved operand
  *before* finalize, since some inference happens at finalize time — §8 assumption 3.)
- **Claim-diff guard.** Flag any commit that deletes a `support.json` or downgrades an
  existing `supported` verdict to absent (the canonical-JSON surgical-write property makes
  this a cheap diff to detect). This is the mitigation for within-bundle claim erosion
  (§3.2), which the per-tier floor does not catch.

### 9.4 Acceptance criteria

- [ ] Re-running `--write-support-claims` with no support change produces a zero git diff
      (canonical-JSON idempotency).
- [ ] A write run with an empty/absent observation set does not null an existing `supported`
      verdict (empty-write guard).
- [ ] The verifier branches on `enforcement_level` before the `.bin` checks; a claim-bearing
      bundle with missing/invalid `enforcement_level` fails pre-commit.
- [ ] A commit deleting a `support.json` or downgrading a `supported` verdict is flagged by
      the claim-diff guard.

## 10. Oversized and Golden-Infeasible Graphs

An `applicability` bundle for a very large graph primarily exercises the **graph
builder/finalize path** (integer-overflow cleanliness of dim/stride math), and only
secondarily the engine's support decision. A builder bug and an engine regression would be
indistinguishable from a single FAIL. Therefore:

- Choose the **smallest** shape that is golden-infeasible *for compute or storage cost
  reasons*, with dims that remain comfortably representable, rather than the largest
  representable shape.
- If integer-overflow *handling* is itself the property under test, express it as an
  explicitly-labelled builder test, not as a support claim.

This keeps an `applicability` FAIL attributable to the engine, which is the thing the claim
is meant to gate.

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

### 12.1 Emitting observations

An enforcement run (`--enforce-support-claims`) gains a side output,
`--emit-support-observations=<path>`, that writes one record per observed `(bundle, engine)`
regardless of verdict. Emission is a pure side effect — it never changes enforcement
pass/fail. Each record carries what the offline consumer needs to attribute and apply it:

```jsonl
{"schema":"support_observation/1","bundle":"quick/ConvFwd/nhwc/fp16/resnet50_layer3","engine":"MIOPEN_ENGINE","arch":"gfx942","platform":"linux","verdict":"supported","enforcement_level":"full","rocm_version":"6.4.0","commit":"<sha>","run_id":"<ci-run>","timestamp":"<iso8601>"}
```

- `verdict` ∈ `supported | declined | unknown`, using the status-code mapping of §5.2.
- `bundle` is the repo-relative bundle path — the same on-disk key a claim attaches to (§4) —
  so the consumer can locate the exact `support.json` to update.
- Provenance (`rocm_version`, `commit`, `run_id`, `timestamp`) lets the consumer break ties
  (prefer the newest observation) and audit where a claim came from.

**Two transports.** The primary output is a **JSONL artifact** the CI job uploads
(`support_observations.jsonl`). Where only console logs are retained, the same records are also
emitted to stdout behind a stable tag — `[[HIPDNN_SUPPORT_OBS]] {json}`, one per line — so they
can be scraped from a build log after the fact. Both are append-only and union-safe.

### 12.2 Shard- and ASIC-tolerant by construction

Each record is a self-contained per-`(bundle, engine, arch, platform)` fact, so:

- **Sharding.** Each shard emits only the bundles it ran; the consumer unions all shards'
  records. Unlike enforcement's empty-observation floor (§7.2), harvesting needs no
  completeness guarantee from a single process — the union across shards/runs is what matters.
- **Multiple ASICs.** Different ASICs' nightly runs write disjoint `(arch, platform)` keys, so
  their observations never conflict — the same disjoint-key property as the §9.2 writes.

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
| `graph.fbs` / serialization evolution changes how a graph deserializes, shifting the identity a claim attaches to. | The claim attaches to the deserialized graph object, not raw bytes; FlatBuffers schema changes must stay backwards compatible (same constraint RFC 0011 already carries). The pre-commit dtype-resolution check (§9.3) guards the one ambiguity that matters for dispatch. |
| `applicability` FAIL is ambiguous between a builder bug and an engine regression for oversized shapes. | Shape-selection rule in §10 (smallest golden-infeasible shape, overflow handling as a separate labelled test). |
| Build-without-execute / bundle-JSON→`frontend::graph::Graph` bridge / bare finalize may not work as assumed. | Marked as ASSUMPTIONS requiring a frontend spike (§8) before the `applicability` level is committed; metadata-only tensor allocation is the fallback. The live-compute harness already queries support before tensor generation, so the capability is plausible, but the golden-path bridge is new work. |
| The support query cannot distinguish "declined" from "errored". | Keyed on the specific `GRAPH_NOT_SUPPORTED` status code, not list-emptiness (§5.2, §8); only that code is treated as a verdict, all other errors are "unknown". |
| A claim is eroded within a surviving bundle (support.json deleted / verdict downgraded). | Claim-diff guard in §9.3 (the per-tier floor does not catch this); acknowledged in §3.2 as accepted tail-risk. |
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
- **Multi-engine attribution.** Enforcing claims for several distinct plugins in one run
  requires per-engine support attribution; deferred (§7.3 refuses it for now).

## 16. Glossary

- **Bundle.** A golden data directory (`{Name}.json` + optional `.tensor{uid}.bin` +
  `meta.json`) as defined by [RFC 0011](0011_GoldenReferenceValidation.md). The unit a
  support claim attaches to.
- **Support claim.** A `{Name}.support.json` entry asserting, per `(engine, arch,
  platform)`, that the engine is expected to support (or not support) the bundle's graph.
- **Claimed graph.** A graph whose bundle has a `supported` verdict for the current
  `(engine, arch, platform)`.
- **Claim broken.** A `supported`-claimed graph whose live support query returns the
  *declined* status (`GRAPH_NOT_SUPPORTED`). The headline failure.
- **Declined.** The dedicated `GRAPH_NOT_SUPPORTED` status: graph well-formed but no engine
  accepts it. The only error status treated as a support verdict.
- **Support unknown.** A support query that returned any *other* (non-decline) error status;
  excluded from claim evaluation and surfaced as a note. The declined/unknown split (keyed on
  the specific status code) is what makes the gate neither blind nor flaky (§5.2, §7.1, §8).
- **Enforcement level.** `full` (execute + numeric verify) or `applicability` (build + query
  support only), selected by the bundle's `meta.json` `enforcement_level` field (named to
  avoid collision with RFC 0011's runtime `--verification-mode`).
- **Applicability bundle.** A graph-only bundle (no `.bin`) validated only for engine
  support, never executed or numerically compared. The home for golden-infeasible graphs.
