---
name: hipdnn-pr-quality
description: "hipDNN supplements to the ROCm PR quality base skill. Use for hipDNN PR author, review, or pre-merge gating (target branch develop; product paths under projects/hipdnn/** and the providers under dnn-providers/**). Adds and tightens base rules; never relaxes a base MUST."
argument-hint: "[author | review | pre-merge] [PR URL | branch:<name> | local]"
extends: rocm-pr-quality
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# hipDNN PR Quality (overlay)

## Dependency (mandatory — do this first)

Read and apply the `rocm-pr-quality` base skill before anything below. It lives in `ROCm/TheRock`
at `skills/rocm-pr-quality/` (`SKILL.md` + `reference.md`). Since `rocm-libraries` is a submodule of
TheRock, a normal TheRock checkout already has the base present alongside this overlay.

The supplements here only **ADD** rules or **TIGHTEN** thresholds. They never relax a base MUST-rule.
On any conflict, the base MUST-rule wins.

> Packaging note: hipDNN's `tools/ai/install-skills.py` snapshots a single skill directory into the
> active host's skills dir. It does **not** carry the `rocm-pr-quality` base along, because the base
> lives in TheRock outside `tools/ai/skills/`. When this overlay is installed in isolation, make the
> base reachable too (install from a TheRock checkout, or point at `skills/rocm-pr-quality/`). Closing
> that gap in the installer is follow-up work, not part of this overlay.

______________________________________________________________________

## Scope

- **Target branch:** `develop`.
- **Product paths:** `projects/hipdnn/**` (the hipDNN component root) and the providers under
  `dnn-providers/**`.
- Paths below are written relative to the repo root; in a standalone hipDNN checkout the
  `projects/hipdnn/` paths are relative to the checkout root.
- Changes outside these (docs, repo tooling) follow the base bar only.

______________________________________________________________________

## PR-policy gate

hipDNN PRs are gated by the **Libraries PR Bot**, the automated check every PR must clear before it
can be reviewed. Treat it as authoritative: conform the PR to it before author or pre-merge sign-off,
it overrides this skill's waivers and self-evident exemptions, and the skill never works around it.

Do **not** rely on a copy of the bot's rules here, and do not assume its layout. Bot logic and its
location change over time, so any copy or hardcoded path here goes stale and starts lying. Instead,
locate the bot in the repo and read its live policy at run time, then conform to whatever it currently
enforces. As a starting hint it has lived in the rocm-libraries tree under a `libraries_pr_bot`-style
directory with an accompanying CI workflow and a contributor-facing FAQ, but confirm that by
investigating rather than trusting this hint; in a standalone hipDNN checkout it is in the parent
rocm-libraries tree, not under `projects/hipdnn/`.

When you advise the author to do something solely to clear the gate that the contributing guide does
not state, name it as a gate requirement so they know where it came from. This overlay only points at
the gate; it does not restate the gate's rules, so when the bot changes, the overlay needs no change.

______________________________________________________________________

## Supplements

### Scoping of base rules (adds — bind scope buckets to hipDNN paths)

Bind the base scope buckets to hipDNN's component layout:

- **Frontend:** `projects/hipdnn/frontend/`, `projects/hipdnn/python/`, public frontend headers,
  graph/node/attribute wrappers, public C++/Python API.
- **Backend:** `projects/hipdnn/backend/`, descriptors, engines, plugin loading, pack/unpack logic,
  the backend C API.
- **Data & FlatBuffers SDK:** `projects/hipdnn/data_sdk/`, `projects/hipdnn/flatbuffers_sdk/`, `.fbs`
  schemas, generated-object wrappers.
- **Plugin SDK:** `projects/hipdnn/plugin_sdk/`, plugin interfaces, ABI/API contracts.
- **Providers:** `dnn-providers/`, provider registration, applicability, execution, workspace,
  external library calls.
- **Build/infra:** `CMakeLists.txt`, `cmake/`, `CMakePresets.json`, CI, packaging, scripts.
- **Tests:** unit/integration tests, test SDK helpers, GTest fixtures, generated test data.
- **Docs/tools:** documentation, RFCs, codegen, developer tooling.

### Adds — change classes (bind to hipDNN paths)

On top of the base classes, tag hipDNN PRs with:

- `provider-op` — a provider adds or extends an op / support surface (`dnn-providers/**`).
- `schema/fbs` — a `.fbs` schema or generated-object-wrapper change (serialization compatibility).
- `plugin-abi` — a change to a `plugin_sdk/` interface or ABI/API contract.
- `cudnn-compat` — a public-API change whose contract is meant to track an equivalent cuDNN API.

### Adds — PR body format (the hipDNN author-assist template)

For hipDNN author-assist, use this section order instead of the base's generic template. New PRs are
**draft by default**; open ready-for-review only when the user explicitly asks. Do not render empty
`N/A` fields.

```markdown
## Summary
<1–3 sentences: purpose, motivation, what it enables. Put the tracking reference here.>

## Risk Assessment
<Risk level (1–5) and a one-paragraph rationale.>

## ASIC Coverage
<Blast radius and the ASICs that must be verified before merge. State whether passing PR CI is
sufficient, a specific-ASIC run is required, or a full multi-arch sweep is required, and why. Omit
only for docs/comments/skill-only changes with no ASIC impact.>

## Testing Summary
- <Testing category and what it covers.>

## Testing Checklist
- [x] <Test group> - `<command>` - Status: Passed
- [ ] <Multi-arch sweep, only if the blast radius requires one> - TheRock multi-arch CI - ASICs: <families> - Status: Pending
- [ ] PR CI - GitHub PR checks - Status: Pending

## Technical Changes
- <Top-level technical what/why change.>
```

Checklist discipline: `[x]` only for validation that actually passed; `[ ]` for pending/not-run/failed.
Represent each required-but-not-yet-passed ASIC verification as its own unchecked gate.

### Adds — resource-ownership review checks (ASAN is in CI)

hipDNN CI runs sanitizer-enabled tests; a leak fails the build. Treat ownership ambiguity as a
substantive finding, not a nit.

- Owning raw pointers are wrapped in RAII **immediately** after acquisition; avoid manual `delete`
  (fragile across assertions, exceptions, early returns).
- FlatBuffers `UnPack()` (backend/data_sdk) returns owning raw pointers — prefer the generated helper
  (e.g. `UnPackGraph()` → `std::unique_ptr<GraphT>`), or wrap manually `std::unique_ptr<T>(table->UnPack())`.
- `getAttribute()` with `HIPDNN_TYPE_BACKEND_DESCRIPTOR` allocates a fresh descriptor; ownership
  transfers to the caller — wrap in `std::unique_ptr<HipdnnBackendDescriptor>` immediately.
- Check provider handles, workspace buffers, streams, and external-library resources for correct
  lifetime and failure-path cleanup.

### Adds — provider-behavior review checks (`provider-op`)

- Provider applicability predicates match the implementation's actual support.
- Registration uses the correct op type, tensor layouts, data types, compute types, behavior notes.
- Workspace-size calculation, stream usage, async behavior, and external-library API calls are correct.
- Unsupported cases fail predictably instead of dispatching to a partial/invalid implementation.

### Adds — cuDNN compatibility review checks (`cudnn-compat`)

For a hipDNN API meant to track a cuDNN API, compare signature, parameter semantics, defaults, status
behavior, ownership/lifetime, and documented constraints. Prefer source-level comparison against the
public cuDNN frontend repo; use NVIDIA's published cuDNN docs as supporting reference. If no
authoritative source is reachable, flag the point for human verification rather than relying on memory.
A public-API change that silently diverges from the equivalent cuDNN behavior is a finding unless the
divergence is explicit, documented, and intentional.

### Adds — serialization compatibility (`schema/fbs`)

A `.fbs` schema change must keep backward/forward compatibility (field ordering/IDs, defaults) or
declare and justify the break. Check the generated-object-wrapper behavior, not just the schema text.

### Adds — ASIC / multi-arch coverage (binds the base blast-radius rule)

Judge required coverage from what the diff changes, not the file path, then reconcile against what the
PR actually tested and claimed (the PR body's `## ASIC Coverage` section).

- ASIC-independent wiring/plumbing that does not change kernel selection, support surface, or default
  behavior → passing PR CI is sufficient.
- Frontend/default-setting or dispatch-behavior changes that existing tests exercise → a multi-arch
  run is warranted.
- Provider changes that add/extend ops or support surface → a full multi-arch sweep, unless the ops
  are arch-scoped, in which case only those archs.
- Arch-specific changes (e.g. a gfx950-only path) → only the affected ASICs.
- Newly enabling a generic integration suite (e.g. activating a suite in a provider lane) → a full
  sweep across all supported GFX families, not just the routine-CI default set.
- Docs/comments/skill-only → no ASIC coverage needed.

Discover families and labels rather than hardcoding; they drift. They are discoverable from the
in-repo multi-arch workflow (`.github/workflows/therock-multi-arch-ci.yml`, `workflow_dispatch` inputs
`linux_amdgpu_families` / `linux_test_labels`) and TheRock's GPU-family matrix. The workflow's default
`linux_amdgpu_families` is a routine-CI subset, and a family may build but skip tests when its runner
is disabled — treat a build-only-but-untested arch as **uncovered**. When coverage is short, the gap
can be closed by launching a TheRock multi-arch integration CI run on the required archs; say so, so
the recommendation is actionable. Do not over-escalate: an arch-independent change covered by passing
PR CI needs no sweep, and saying so is a valid outcome.

### Tightens — pre-merge stale-base on high-coupling hipDNN files

Make the base pre-merge stale-base check concrete for hipDNN. High-coupling files:
public headers under `backend/include/`, `frontend/include/hipdnn_frontend/`, `plugin_sdk/include/`;
`.fbs` schemas under `flatbuffers_sdk/`; provider registration/dispatch under `dnn-providers/**`; and
build wiring (`CMakeLists.txt`, `cmake/`, `CMakePresets.json`). Overlap with the base branch on any of
these since the PR diverged → **strong-recommend** rebase + re-run, because these break combinations
that neither PR's own CI can see. Schema/ABI overlap (`.fbs`, `plugin_sdk/` contracts) → treat as
**mandatory** rebase + re-run.

### Adds — build/test execution belongs to the build skills

This overlay does not run builds or tests. To actually configure/build or run/triage tests as part of
author or pre-merge work, use the hipDNN build skills (`$hipdnn-superbuild` to configure/build with
providers, `$hipdnn-superbuild-test` to run/filter tests against an existing build). Use the standalone
`projects/hipdnn` build (`ninja check` / `ninja unit-check`) for hipDNN-only changes.

______________________________________________________________________

## Severity mapping

The base review tiers map onto hipDNN's familiar labels: `BLOCKING` ≈ Critical,
`IMPORTANT` ≈ Major, `SUGGESTION` ≈ Minor, `FUTURE WORK` ≈ Suggestion. Use the base tiers in output;
the equivalence is for readers used to the old `hipdnn-review` labels.

______________________________________________________________________

## What the overlay cannot do

Drop the regression-test-on-defect rule (M1), allow product-code changes with no test/flag/waiver
(M2), allow disabling tests to green CI (M3), or skip work tracking/linking (M4/M5). Those are base
MUSTs; this overlay can only make them stricter or bind them to hipDNN paths.

______________________________________________________________________

## What this overlay does NOT replace

This overlay covers the **PR lifecycle** (author / review / pre-merge) and supersedes the old
`pr-summary` and `hipdnn-review` skills. It deliberately does **not** cover, and these hipDNN skills
remain on their own:

- **RFC / design-doc review** — `rfc-review`, `rfc-review-compatibility`, `rfc-review-ops`,
  `rfc-review-security`, `rfc-backlog`. RFCs propose; PRs implement. The base is PR-scoped, so the RFC
  family is out of scope here.
- **Build & test execution** — `hipdnn-superbuild`, `hipdnn-superbuild-test`. This overlay points at
  them but does not run builds or tests itself.
- **Codegen** — the DescriptorGenerator `hipdnn-codegen` skill.

Two further gaps a reviewer should know about: this overlay is advisory and never posts to GitHub/Jira
without explicit human approval, and (per the packaging note above) hipDNN's installer does not yet
pull the `rocm-pr-quality` base along with the overlay.
