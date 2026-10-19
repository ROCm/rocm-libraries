---
name: github-ci-pr-quality
description: "GitHub CI supplements to the ROCm PR quality base skill. Use when authoring, reviewing, or merge-gating a rocm-libraries PR that touches .github/ (workflows, actions, CI scripts). Adds and tightens base rules."
argument-hint: "[author | review | pre-merge] [PR URL | branch:<name> | local]"
extends: rocm-pr-quality
allowed-tools: Bash, Read, Grep, Glob, Task, WebFetch
---

# GitHub CI PR Quality (overlay)

## Dependency (mandatory: do this first)

Read and apply the `rocm-pr-quality` base skill before anything below. It lives in `ROCm/TheRock`, not in this repo: [`SKILL.md`](https://github.com/ROCm/TheRock/blob/main/skills/rocm-pr-quality/SKILL.md) and [`reference.md`](https://github.com/ROCm/TheRock/blob/main/skills/rocm-pr-quality/reference.md). The `extends` and `argument-hint` keys above are repo conventions that not every agent runtime interprets, which is why this section tells you to load the base explicitly. See the "PR-quality skills" section of the root [`AGENTS.md`](../../AGENTS.md) for how overlays are layered.

The supplements here only **ADD** rules or **TIGHTEN** thresholds. They never relax a base MUST-rule. On any conflict, the base MUST-rule wins. Where `.github/` CI changes also touch a component's code, apply that component's overlay too (for example `hipblaslt-pr-quality`).

---

## Scope

- **Target branch:** `develop`.
- **Paths:** `.github/workflows/**`, `.github/actions/**`, `.github/scripts/**`, `.github/instructions/**`, `.github/CODEOWNERS`, `.github/labeler.yml`, `.github/repos-config.json`, and any file a workflow runs (for example `tools/libraries_pr_bot/**`).
- Component CI that lives outside `.github/` (a project's own `.github/`, `tox.ini`, `test_categories.yaml`) is in scope when the PR also changes how a root workflow calls it.
- A PR **outside** `.github/` that moves, renames, or deletes a file that CI refers to also gets the reference check below. That covers a `paths:` filter, a `COMPONENTS` entry, a `CODEOWNERS` line, a tox or `test_categories.yaml` path, and a script a workflow runs. Such a PR touches no CI file, so nothing else will flag it.
- Other changes follow the base bar only.

---

## Required cross-read: the project's TESTING.md

A CI change is a testing change. Before reviewing or authoring one, find out which projects it affects and read what those projects say about their own gates.

1. **Find the affected projects.** Use the workflow's `paths:` filters, the `COMPONENTS` map in `.github/scripts/component_ci.py`, `.github/scripts/therock_matrix.py`, and `.github/repos-config.json`. A change to a shared workflow or action (`ci-env`, `component-ci.yml`, `therock-*`, a reusable `workflow_call` file) affects every project it serves: read all of them, or say explicitly which you sampled and why.
1. **Find each project's testing document.** Discover it, do not assume it: `git ls-files '*TESTING.md' '*testing.md'`, then take the nearest one above the project root (and any nested one, such as `projects/hipblaslt/tensilelite/TESTING.md`). If a project has none, say so in the PR and use its README or CONTRIBUTING instead.
1. **Read the CI sections.** At minimum "Pre-submit / CI Gates", "Validation Gates and Ownership", and any "Where these tests actually run" table.
1. **Check the PR against them.** The PR description must state which lane it adds, changes, or removes, and whether that lane gates a merge. If a lane or a gate changes, the matching TESTING.md is updated in the same PR, or the PR gives a reason it is not. A CI change that contradicts the project's TESTING.md, or silently makes it stale, is `IMPORTANT`; one that removes a documented gate is `BLOCKING`.
1. **Respect who owns the gate.** Many gates run in Math CI, which is configured outside this repo. A PR here cannot add, remove, or verify one of those; it may only describe it. Do not claim a gate exists or is required without a source (a workflow file, a TESTING.md table, a branch-protection setting you can cite).

---

## Supplements

### Adds: change classes (bind to CI paths)

On top of the base classes, tag CI PRs with one or more of:

- `ci-new-workflow`: adds a workflow file or a new job.
- `ci-runner-change`: changes `runs-on`, a runner label, or a container image.
- `ci-gate-change`: adds, removes, or changes what a required or gating check does or when it runs.
- `ci-shared-state`: changes `.github/actions/ci-env`, a reusable workflow, a `therock-*` workflow, or anything every project inherits.
- `ci-script`: changes a Python/Bash script a workflow runs.

When ambiguous, pick the stricter class. `build/ci` from the base still applies.

### Adds: where a CI test belongs

Do not add a workflow file for each test. The default is to put a test where its siblings already run.

- **Extend before adding.** A new check goes into an existing workflow when one fits. For CPU-only component checks, the established pattern is a `component-ci-<name>.yml` called from `component-ci.yml`, with its trigger declared in the `COMPONENTS` map in `component_ci.py`. Several unit-level checks for the same component belong in one lane.
- **A new workflow file needs a written reason** in the PR description: why no existing lane fits, what it costs per run, who owns it. "The test is new" is not a reason. Missing reason on a `ci-new-workflow` PR is `IMPORTANT`; a new file whose only content is one small test run is `BLOCKING` unless the author shows why it cannot join an existing lane.
- **Reuse the project's own test entry point.** The workflow calls the project's documented command (for example a `tox` environment or a `ctest` target from its TESTING.md), not a re-implementation in YAML.
- **Say what it gates.** State whether the check is required. A new check that fails the PR needs the same care as a gating test: stable, deterministic, and owned.

### Adds: runner sizing and run cost

- **Right-size the runner.** Light CPU-only work (unit tests, linters, small script tests) belongs on a small runner. `ubuntu-24.04` (GitHub-hosted) is the verified pattern: `hostnumerics-ci.yml`, `shared-ctest-ci.yml`, and `pre-commit.yml` use it. Do not default to the `linux-runner` output of `.github/actions/ci-env` (a large prod pool) for unit-level work. Using it for light work with no stated reason is `IMPORTANT`; for a new workflow it is `BLOCKING` when the job's real work is seconds to a few minutes.
- **Discover, do not hardcode, the pools.** Runner labels and sizes change. Read `ci-env` and the `runs-on` values in existing workflows for what is in use, and confirm any other label with DevOps. This skill does not list the small and medium pool labels.
- **Pay for checkout and setup only when needed.** Use `sparse-checkout` for the paths the job reads, shallow `fetch-depth` unless history is needed, and no container image unless the job needs the toolchain in it. A job whose setup dwarfs its work is a finding. Example: [PR #12185](https://github.com/ROCm/rocm-libraries/pull/12185) added a CPU-only check on the prod runner in a container; the reviewer measured about 3 minutes in total for 7 seconds of tests, 2m53s of it checking out the repository.
- **Separate build from test** and keep GPU runners for jobs that need a GPU (see the TheRock GitHub Actions style guide).
- **Cap the blast of a bad run.** Set `timeout-minutes` and a `concurrency` group with `cancel-in-progress: true` on pull-request workflows.

### Adds: trigger scope

- Run a workflow only when relevant paths change (a `paths:` filter, or the `changes` job and `component_ci.py`). Include the workflow's own file in the filter so editing it re-runs it.
- **Every pattern must match something.** For each entry you add or edit in a `paths:` filter, `COMPONENTS`, or `CODEOWNERS`, confirm at least one tracked file matches it (`git ls-files`, with the same glob rules; `component_ci.py` uses `fnmatch`, where `*` also crosses `/`). A pattern that matches nothing never fires, and nothing reports it.
- **Moves and deletes update CI in the same PR.** When a PR renames, moves, or deletes files, search `.github/`, tox files, and `CODEOWNERS` for the old paths and update them in that PR. Also check open PRs that add references to those paths; neither PR's own CI can see the other.
- A required check must report on every PR. A job skipped by a path filter can leave a required status pending. Show how the check reports when it is skipped, or do not make it required.
- `pull_request_target` workflows must never check out or execute PR head code (see the warning in `libraries-pr-bot.yml`). Any new `pull_request_target` or secret-using trigger is `BLOCKING` until that is shown.

### Adds: the tested tree (what the job checks out)

Every lane should test the same thing: the PR merged into its base branch (`refs/pull/<N>/merge`). On a `pull_request` event, `github.sha` is that merge commit and `actions/checkout` with no `ref:` checks it out. In a `workflow_call` workflow, `github.*` is the caller's, so a called workflow inherits the same tree. Today the PR-triggered workflows rely on this default; none of them spells it out.

- **Keep the default.** A PR-triggered job checks out with no `ref:`. An explicit `ref:` (or `github.event.pull_request.head.sha`) tests a different tree from the other lanes and needs a stated reason in the PR. That is `IMPORTANT`.
- **Know what depends on it.** `component_ci.py` computes changes with `git diff HEAD^`, which means "what this PR changed" only because `HEAD^` is the base tip of a merge commit. If a checkout moves to the PR head, `HEAD^` becomes the PR's previous commit and the path filters go quietly wrong. Anything that uses `HEAD^`, a shallow `fetch-depth`, or `github.sha` as the external repo ref (as `therock-multi-arch-ci.yml` does) is relying on the merge commit.
- **Mixing views needs care.** A step that diffs `pull_request.base.sha` against `pull_request.head.sha` while the checkout is the merge commit must use a merge-base style diff (`base...head`). A plain two-dot diff of a PR that is behind its base reports the base's own changes as the PR's. The `.ambr` guard in `component-ci-tensilelite-coverage.yml` is the model: `check_snapshot_diff.py` computes the merge-base itself before diffing.
- **A test job never runs on `pull_request_target`.** That trigger checks out the base branch, so a test lane there silently tests the wrong tree (and see the security rule above). It is for bots and labelers that never run PR code.
- **Know the limits of the merge ref.** It is fixed when the event fires. Nothing re-runs when the base moves, and "Re-run jobs" reuses the same commit, so an old green PR tested an old base. A PR with merge conflicts has no merge ref and its `pull_request` workflows do not start. State these in a finding instead of treating a green check on a stale PR as current.
- **Pinned inputs are a separate choice.** The TheRock ref (`therock-ref`, the `@<sha>` on reusable workflows) is pinned on purpose and comes from the merged tree. Do not confuse it with the tree under test.

### Adds: shared CI state (`ci-shared-state`)

`.github/actions/ci-env` is the single source of truth for values every workflow inherits: `therock-ref`, `kpack-ref`, `baseline-run-id`, `docker-image`, `linux-runner`, `windows-runner`. Changes there, and to reusable or `therock-*` workflows, move every project at once.

- The PR must say what moves for whom, and how it was checked across projects (not only the PR's own CI).
- Emptying, removing, or changing the meaning of a value is `BLOCKING` without an impact statement. A value that downstream jobs treat as "reuse a prior build" must not silently become empty.
- Automated bump PRs are held to the same rule as human ones. A green PR with the right title can still change shared state in a way its own CI cannot see.
- When a reusable workflow's interface changes, inspect every caller. Update callers in the same PR only where the change breaks them or changes their intended behavior; an added optional input with a behavior-preserving default needs none. A caller left on a breaking interface is `BLOCKING`. Callers outside this repo, or pinned to a version, cannot be edited here: state the migration obligation in the PR instead.

### Defers to the style guides

Follow the TheRock [GitHub Actions style guide](https://github.com/ROCm/TheRock/blob/main/docs/development/style_guides/github_actions_style_guide.md) and cite its section for a style finding: commit-SHA pinning of actions, pinned runner image labels, least-privilege `permissions`, no template injection, `persist-credentials: false`, and Python scripts over non-trivial inline Bash. When authoring, run `pre-commit run --files <changed files>` (the hooks can rewrite files; commit the result) and, where available, `actionlint`. When reviewing, do not modify the PR's checkout (the base review contract is read-only): run the hooks in a throwaway checkout of the PR head, or read the PR's own pre-commit check result, and report any formatter diff as a finding. Treat a style-guide gap as `SUGGESTION` unless it is a security item, which is `IMPORTANT` at minimum.

### Tightens: stale-base on high-coupling CI files

Make the base stale-base check concrete. High-coupling files: `.github/actions/ci-env/**`, `.github/workflows/component-ci.yml`, `.github/scripts/component_ci.py`, `.github/scripts/ci_utils.py`, `.github/scripts/therock_matrix.py`, `.github/scripts/therock_configure_ci.py`, and the `therock-*` workflows. Overlap with the base branch on any of these since the PR diverged means a **mandatory** rebase and re-run (base default is strong-recommend). Two CI PRs can each be green and still break each other in combination. The base helper (`pr_quality_check.py`) has no profile for this overlay and its `base` profile has an empty high-coupling list, so it will not flag this overlap for you: apply this list by hand.

### Tightens: approvals

`/.github/` is owned by `@ROCm/rocm-libraries-reviewers` in `CODEOWNERS`. A CI PR needs that owner's approval, plus the owner of any component-specific workflow named in `CODEOWNERS` (for example hipDNN and MIOpen workflows). Do not state an approval count the repo has not configured.

### Adds: review checklist

When reviewing a CI PR, report each of these as pass, warn, or fail with a link or `file:line`:

| Check | Severity if it fails |
| --- | --- |
| Affected projects identified and their TESTING.md read; PR names the lane and whether it gates | `IMPORTANT` |
| New workflow file justified; not a one-test-one-workflow addition | `IMPORTANT` (`BLOCKING` for a single small test) |
| Runner sized for the work; `ci-env` prod runner not used by default | `IMPORTANT` (`BLOCKING` for a new light workflow) |
| Checkout and setup cost proportionate (sparse, shallow, no needless container) | `SUGGESTION` (`IMPORTANT` if setup dominates) |
| Trigger scope, own file in `paths:`, required check reports when skipped | `IMPORTANT` |
| Every new or edited path pattern matches an existing file; a PR that moves or deletes files updated all CI references to them | `IMPORTANT` (`BLOCKING` when the stale entry was a gating check's only trigger) |
| PR jobs check out the default merge ref; any explicit `ref:` or `head.sha` is justified; a `base...head` diff is used where views mix | `IMPORTANT` |
| Shared-state change has an impact statement; no value emptied silently | `BLOCKING` |
| `pull_request_target` or secrets do not run PR head code | `BLOCKING` |
| `timeout-minutes` and `concurrency` set on PR workflows | `SUGGESTION` |
| Style-guide items (SHA pins, permissions, injection, inline Bash) | `SUGGESTION` (`IMPORTANT` for security) |
| TESTING.md updated when a lane or gate changed | `IMPORTANT` |

---

## What the overlay cannot do

Drop the regression/test obligations (M1/M2), allow disabling tests to green CI (M3), or skip work tracking and linking (M4/M5). Those are base MUSTs; this overlay can only make them stricter or bind them to CI paths. It is also advisory: it does not approve, block, or enforce. Enforcement (CODEOWNERS, a lint on runner labels, the Libraries PR Bot) is a separate, DevOps-owned lane, and a machine gate outranks this skill.

---

## Worked examples this overlay is designed to catch

[PR #12185](https://github.com/ROCm/rocm-libraries/pull/12185) added `component-ci-library-uniqueness.yml` for a CPU-only uniqueness check, wired through `component-ci.yml` on the `ci-env` prod runner. A reviewer later measured 3 minutes in total for 7 seconds of tests and asked for a smaller runner or an existing workflow. The overlay flags this up front: `ci-new-workflow` with no stated reason, a prod runner for light work, and a checkout and container cost out of proportion to the work.

[PR #13623](https://github.com/ROCm/rocm-libraries/pull/13623) (an automated TheRock reference bump) changed `baseline-run-id` in `ci-env` from a run ID to an empty string. `therock-multi-arch-ci.yml` treats an empty baseline as "no prebuilt stages" and turns stage reuse off (see its `prebuilt_stages` and `baseline_run_id` inputs), so every pull request rebuilt every stage, not only the stages its diff touched. The bump was reverted in [#13698](https://github.com/ROCm/rocm-libraries/pull/13698), and commit `67a16e4ea5c` records the effect. The bump's own checks were fine; the cost landed on every other PR. That is the `ci-shared-state` case: a value emptied without an impact statement, visible only by looking across projects.

[PR #12185](https://github.com/ROCm/rocm-libraries/pull/12185) also added trigger paths under the old `tensilelite/Tensile/` package directory. [PR #12150](https://github.com/ROCm/rocm-libraries/pull/12150) renamed that package to `tensilelite/tensilelite/` the same day and missed those references. The lane's trigger pointed at a file that no longer existed until [#13399](https://github.com/ROCm/rocm-libraries/pull/13399) ("correct paths missed in #12150") fixed it. Each PR was correct when reviewed, and the rename touched no `.github/` file. The reference check at the move is what catches it.

---

## Worked review output (retrospective example)

This shows the shape the base skill asks for (overall assessment, the four answers, severity-ordered findings, blocking items) for one CI PR. It is a retrospective run on [PR #12185](https://github.com/ROCm/rocm-libraries/pull/12185), which is already merged, so it is an illustration and not a review that gated anything. It was scoped to the `.github/` changes; the SolutionUID code belongs to the hipBLASLt overlay and is not re-reviewed here.

**Evidence checked:** [the job run](https://github.com/ROCm/rocm-libraries/actions/runs/37805607146/job/113409341093) (runner `aws-linux-scale-rocm-prod`; 3:10 total, checkout 2:53, container init 0:06, test step 0:07); the new workflow and its wiring in `component-ci.yml` and `component_ci.py`; and `projects/hipblaslt/TESTING.md` and `tensilelite/TESTING.md`, neither of which mentions the new lane.

**Overall assessment:** `CHANGES REQUESTED` (two `BLOCKING` items, both resolvable by a written justification or by folding the check into an existing lane).

**The four questions**

1. *What lands:* a new `component-ci-library-uniqueness.yml`, wired through `component-ci.yml` and a `COMPONENTS` entry, running one tox environment (one test file) in a container on the `ci-env` prod runner.
1. *Appropriate test level:* a CPU-only, unit-level check. It belongs in an existing lightweight lane or on a small runner.
1. *Omitted tests or flags, and waivers:* nothing omitted and no waiver declared, so none to adjudicate.
1. *Adjacent lanes to consider:* an existing tensilelite lane (the CPU-only coverage lane) might host it. The PR does not say whether that was considered.

**Findings (most severe first)**

| Finding | Severity | Evidence |
| --- | --- | --- |
| New workflow file for one small test, no reason given | `BLOCKING` unless justified | `ci-new-workflow`: one tox environment; the PR body does not say why it cannot join an existing lane. |
| Prod runner for light work | `BLOCKING` | `ci-runner-change`: `linux-runner` from `ci-env` passed straight through; the real work was 7 seconds. |
| Checkout and setup dominate the job | `IMPORTANT` | 2m53s of 3m10s, no `sparse-checkout`. Sparse checkout is the likely fix; that is a hypothesis, not measured. |
| TESTING.md not updated; gating unstated | `IMPORTANT` | The PR calls the lane a gate, but it rolls up into `Component CI Summary`, which `tensilelite/TESTING.md` says is not a required check. If that still holds, the lane cannot block a merge. |
| No `timeout-minutes` | `SUGGESTION` | Minor. |

**Blocking items:** justify the new workflow file in the PR description or move the check into an existing lane; use a right-sized runner or state why the prod runner is needed.

**Passed:** the workflow's own file is in the path filter, the summary job includes the lane, actions are pinned to SHAs, and `permissions` is `contents: read`.

**Limits of this review:** it did not re-run the job, did not check the small-runner labels (the skill deliberately names none; confirm them with DevOps), and judged only the `.github/` surface.
