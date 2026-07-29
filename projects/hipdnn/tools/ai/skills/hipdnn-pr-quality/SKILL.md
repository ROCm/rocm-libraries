---
name: hipdnn-pr-quality
description: "hipDNN supplements to the ROCm PR quality base skill. Use for hipDNN PR author, review, or pre-merge gating (target branch develop; product paths under projects/hipdnn/** and the providers under dnn-providers/**). Adds and tightens base rules; never relaxes a base MUST."
argument-hint: "[author | review | pre-merge] [PR URL | branch:<name> | local]"
extends: rocm-pr-quality
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# hipDNN PR Quality (overlay)

## Dependency (mandatory — refresh and apply the base first)

This overlay sits on top of the `rocm-pr-quality` base skill, which is **owned by ROCm and lives in
`ROCm/TheRock`** at `skills/rocm-pr-quality/` (`SKILL.md` + `reference.md`). ROCm sets the PR-quality
floor there and can change it at any time, so this overlay never bundles or pins an old copy of the
base. It always pulls the current one.

Before doing anything else in an action, refresh the base from TheRock's `main` at **user scope**, then
read and apply it. The base is foundational for any ROCm work, so it belongs once per developer at user
scope and kept up to date — not vendored per repo:

```
gh skill install ROCm/TheRock rocm-pr-quality@main --scope user --force
```

Run this **every time** you invoke this skill, so you are always on the latest ROCm policy rather than
a stale local copy. (`gh skill` is a preview feature; `--force` reinstalls in place, and
`gh skill update rocm-pr-quality` is the lighter equivalent once it is installed.) The `@main` suffix
matters: a bare name resolves to the latest tagged release, which predates the skill, so `@main` is
what gives the canonical, current base. `ROCm/TheRock` is public, so this resolves from anywhere,
including a standalone hipDNN clone. Swap `--scope user` for `--agent <host>` only if your agent reads
skills from a non-default location. To deliberately pin a known-good base instead of latest (break
glass), replace `@main` with `--pin <tag-or-sha>`.

**Degrade gracefully — do not let the refresh block the work:**

1. Run the refresh above. If it succeeds, use the freshly pulled base.
2. If the refresh fails (offline, no `gh`, the preview command changed) **but a `rocm-pr-quality` base
   is already available** — a prior install, or a TheRock checkout containing `skills/rocm-pr-quality/`
   — just use that existing copy and move on. A failed refresh when you already have a base is fine.
3. If the refresh fails **and** you have no base at all, obtain it another way before proceeding: fetch
   it directly from `https://github.com/ROCm/TheRock` (the canonical source, pointed to from that
   repo's `CONTRIBUTING.md`) at `skills/rocm-pr-quality/`, or read it from a local TheRock clone if one
   exists on the machine.
4. Only if no base can be reached by any of these should you stop and tell the user the
   `rocm-pr-quality` base is unavailable.

The supplements here only **ADD** rules or **TIGHTEN** thresholds. They never relax a base MUST-rule.
On any conflict, the base MUST-rule wins.

______________________________________________________________________

## Scope

- **Target branch:** `develop`.
- **Product paths:** `projects/hipdnn/**` (the hipDNN component root) and the providers under
  `dnn-providers/**`.
- Paths below are written relative to the repo root; in a standalone hipDNN checkout the
  `projects/hipdnn/` paths are relative to the checkout root.
- Changes outside these (docs, repo tooling) follow the base bar only.

______________________________________________________________________

## PR-policy gate (binds the base machine-gate rule)

The base's "conform to the machine gate" principle applies here unchanged (it already covers that the
gate is authoritative, outranks this skill's waivers and exemptions, is never worked around, and that
gate-only requirements get flagged to the author). This overlay only binds that rule to hipDNN's gate:
the **Libraries PR Bot**. Conform the PR to it before author or pre-merge sign-off.

Locate the bot and read its live policy at run time rather than trusting a copy — bot logic and its
location drift, so any rule restated or path hardcoded here goes stale. As a starting hint it has
lived in the rocm-libraries tree under a `libraries_pr_bot/` directory (a policy file, a CI workflow,
and a contributor FAQ); confirm by investigating. In a standalone hipDNN checkout it is in the parent
rocm-libraries tree, not under `projects/hipdnn/`.

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
<1–3 sentences: purpose, motivation, what it enables.>

JIRA ID : <key>   <!-- tracking reference on its own line, exactly as the PR bot expects -->

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

The base blast-radius/coverage rule applies as-is (judge from what the diff changes, not the path; map
arch-independent → standard CI, behavior-shifting → multi-arch, arch-scoped → only those archs,
support-surface-expanding → full sweep; don't over-escalate). This overlay only binds it to hipDNN and
adds the hipDNN-specific cases:

- Reconcile required coverage against the PR body's `## ASIC Coverage` section, where hipDNN records it.
- Provider op / support-surface changes (`dnn-providers/**`) → full multi-arch sweep, unless the op is
  arch-scoped, in which case only those archs.
- Newly enabling a generic integration suite (e.g. activating a suite in a provider lane) → a full
  sweep across **all** supported GFX families, not just the routine-CI default subset.
- A family that builds but skips tests (its runner is disabled) is **uncovered**, not covered.

Discover families and labels rather than hardcoding (they drift): from the in-repo multi-arch workflow
(`.github/workflows/therock-multi-arch-ci.yml`, `workflow_dispatch` inputs `linux_amdgpu_families` /
`linux_test_labels`) and TheRock's GPU-family matrix. When coverage is short, say so and note it can
be closed by launching a TheRock multi-arch integration CI run on the required archs, so the
recommendation is actionable.

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
`pr-summary` and `hipdnn-review` skills, which now remain only as deprecated stubs that redirect here.
It deliberately does **not** cover, and these hipDNN skills remain on their own:

- **RFC / design-doc review** — `rfc-review`, `rfc-review-compatibility`, `rfc-review-ops`,
  `rfc-review-security`, `rfc-backlog`. RFCs propose; PRs implement. The base is PR-scoped, so the RFC
  family is out of scope here.
- **Build & test execution** — `hipdnn-superbuild`, `hipdnn-superbuild-test`. This overlay points at
  them but does not run builds or tests itself.
- **Codegen** — the DescriptorGenerator `hipdnn-codegen` skill.

One thing a reviewer should know: this overlay is advisory and never posts to GitHub/Jira without
explicit human approval. Its one hard dependency, the `rocm-pr-quality` base, is intentionally external
— ROCm owns it in TheRock — and the skill refreshes it from TheRock's `main` at user scope on every run
(see the Dependency section).
