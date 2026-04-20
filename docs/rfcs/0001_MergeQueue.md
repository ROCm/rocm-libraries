# Federated Merge Queues for rocm-libraries

- Contributors: Samuel Reeder
- **Status**: First draft
- **Target Branch**: `develop` (only)

## 1. Executive Summary

This RFC proposes a custom, in-repo merge queue for rocm-libraries that serializes merges across coupled components on `develop`. Each opted-in component gets its own FIFO queue. A PR enters every queue of every component it depends on, and merges only when it is at the head of all of them. Authors trigger the queue with a `/merge` PR comment; an in-repo workflow squash-merges PRs as they reach the front.

Initial scope is the hipDNN ecosystem: `hipdnn` core, four providers (`miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider`), and `integration-tests`. The queue is opt-in — other rocm-libraries subprojects are unaffected — starts optional (no branch-protection changes), and is designed so any project can join later by adding two centralized config entries.

## 2. Problem Statement

- **Provider/core coupling.** A change to hipDNN core can break any provider; a change to a provider can silently regress hipDNN integration. Today, nothing serializes these merges, so two PRs touching coupled code can both pass CI in isolation and break `develop` together.
- **CI capacity.** Parallel un-rebased merges thrash `develop` and force needless re-runs of expensive provider builds.
- **Developer overhead.** A reviewer should not have to manually coordinate merge order across components.

We want a system that serializes safely by default, is cheap to opt into, and stays out of contributors' way.

## 3. Goals and Non-Goals

### Goals
- One FIFO queue per component: `hipdnn`, four providers, `integration-tests`.
- Cross-blocking that follows dependencies (see [§4.2](#42-path--queue-mapping)). A PR is blocked by every PR ahead of it in any queue it belongs to.
- Opt-in per project. Subprojects outside the hipDNN ecosystem are unaffected unless they opt in.
- `develop` is the only managed target branch.
- Start optional; provide a clear path to "required" via branch protection.
- External contributors can use the queue with the same safeguards as anyone else.

### Non-Goals
- Replacing existing CI workflows.
- Other rocm-libraries projects in v1.
- Batched merges (`batch_size > 1`) — deferred.
- Release or feature branches — `develop` only.

## 4. Design

### 4.1 Queues

Six FIFO queues, one per component:

- `hipdnn`
- `miopen-provider`
- `hipblaslt-provider`
- `hip-kernel-provider`
- `fusilli-provider`
- `integration-tests`

Each PR enters zero or more queues based on the paths it touches. A PR merges only when at the head of **every** queue it belongs to.

### 4.2 Path → Queue mapping

The mapping encodes a single rule: a PR enters the queue of every component whose state it depends on, so it is blocked by anything ahead of it in any of those queues.

| Path changed                          | Queues entered                                                                                          | Why                                                                                                                                                  |
|---------------------------------------|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `projects/hipdnn/**`                  | `hipdnn`, `miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider`, `integration-tests` (all 6) | Core touches every consumer's interface. Blocks anything queued behind it; blocked by anything ahead of it in any queue.                              |
| `dnn-providers/<provider>/**`         | `<provider>`, `integration-tests`                                                                       | A provider depends on its own queue and the integration suite. Does **not** enter `hipdnn`; it blocks core only implicitly because core enters its queue. Blocks other providers only indirectly via shared `integration-tests` membership; never directly. |
| `dnn-providers/integration-tests/**`  | `integration-tests`, `miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `fusilli-provider` | Integration tests depend on every provider, and providers depend on the integration suite to stay green. Does **not** enter `hipdnn`; core does not depend on integration tests. |
| Anything else                         | none                                                                                                    | Project not opted in.                                                                                                                                |

A PR editing both `projects/hipdnn/api/foo.h` and `dnn-providers/miopen-provider/src/bar.cpp` enters all six queues — core's row is a superset of every provider's row.

#### Membership at a glance

```
                                       QUEUES
                            ┌─────┬─────┬─────┬─────┬─────┬──────┐
                            │ hip │ mio │ hbl │ hkp │ fus │ intT │
PR touches...               │ dnn │ pen │ ast │     │     │      │
────────────────────────────┼─────┼─────┼─────┼─────┼─────┼──────┤
projects/hipdnn/**          │  ●  │  ●  │  ●  │  ●  │  ●  │  ●   │
dnn-providers/miopen-…/**   │  ·  │  ●  │  ·  │  ·  │  ·  │  ●   │
dnn-providers/hipblaslt-…   │  ·  │  ·  │  ●  │  ·  │  ·  │  ●   │
dnn-providers/hip-kernel-…  │  ·  │  ·  │  ·  │  ●  │  ·  │  ●   │
dnn-providers/fusilli-…     │  ·  │  ·  │  ·  │  ·  │  ●  │  ●   │
dnn-providers/integration-… │  ·  │  ●  │  ●  │  ●  │  ●  │  ●   │
────────────────────────────┴─────┴─────┴─────┴─────┴─────┴──────┘
   ● = PR enters this queue     · = PR does not enter this queue

   hipdnn = hipdnn (core)        hkp  = hip-kernel-provider
   mio    = miopen-provider      fus  = fusilli-provider
   hbl    = hipblaslt-provider   intT = integration-tests
```

Two patterns to read off the matrix:

- The `hipdnn` column has exactly one mark — only core PRs ever enter the core queue. Providers and integration tests block core *implicitly*, by sitting ahead of core PRs in the queues that core also enters (every other column).
- The `projects/hipdnn/**` row is the only fully-marked row — core touches every consumer, so a core PR is gated by every other in-flight PR in the ecosystem.

#### Worked example: four PRs enqueued in order

PRs of four distinct types enter the queue in order — A (core), B (miopen-provider), C (hipblaslt-provider), D (integration-tests). Heads shown leftmost; `→` is "behind".

```
T₀ — all four enqueued

  hipdnn              │ A
  miopen-provider     │ A → B       → D
  hipblaslt-provider  │ A      → C  → D
  hip-kernel-provider │ A           → D
  fusilli-provider    │ A           → D
  integration-tests   │ A → B  → C  → D

  A is at the head of every queue → A merges next.

T₁ — after A merges

  hipdnn              │ (empty)
  miopen-provider     │ B → D
  hipblaslt-provider  │ C → D
  hip-kernel-provider │ D
  fusilli-provider    │ D
  integration-tests   │ B → C → D

  B is at the head of {miopen-provider, integration-tests} → B merges next.
  C is at the head of hipblaslt-provider but blocked by B in integration-tests.
  D is blocked behind B and C.

T₂ — after B merges

  miopen-provider     │ D
  hipblaslt-provider  │ C → D
  hip-kernel-provider │ D
  fusilli-provider    │ D
  integration-tests   │ C → D

  C is at the head of {hipblaslt-provider, integration-tests} → C merges next.
  D is still blocked by C.

T₃ — after C merges

  miopen-provider     │ D
  hipblaslt-provider  │ D
  hip-kernel-provider │ D
  fusilli-provider    │ D
  integration-tests   │ D

  D is at the head of all five queues it belongs to → D merges.
```

Two takeaways:

- **B and C never run in parallel** even though they touch different providers — they share `integration-tests`, so the integration suite serializes them. This is the bidirectional integration-tests ↔ providers link doing its job.
- **If a second core PR E were enqueued at T₂**, it would join the tail of *every* queue — including the ones C and D still sit in — and would have to wait for both to clear before merging. Core is never allowed to overtake an in-flight provider or integration-tests PR.


### 4.3 Opt-in

Two centralized edits add a project to the queue:

1. Append a path → queue list to the path/queue config (`PATH_TO_QUEUES`).
2. Append a corresponding scope block to the Mergify config (for CI status visibility).

There is no per-project file. Removing both entries opts a project out cleanly. The initial opt-in set is hipDNN core + four providers + integration-tests. Other rocm-libraries projects are explicitly out of scope for v1.

### 4.4 PR lifecycle

1. An authorized user (PR author or any user with write/maintain/admin on the repo, per [§4.6](#46-permissions)) comments `/merge`.
2. The command handler validates eligibility (see [§4.6](#46-permissions)), computes the queue set from the PR's changed paths, applies a `mq:queued` label and one `mq:<queue>` label per queue, and posts a single status comment with a hidden JSON metadata marker.
3. A processor runs every 3 minutes. For each queue, it picks the head PR (oldest `enqueued_at`). If a PR is at the head of *every* queue it belongs to, the processor labels it `mq:active`, merges `develop` into the PR branch, and waits one cycle for CI to run against the freshly-merged tip.
4. On the next cycle:
   - **CI green** → squash-merge.
   - **CI still pending** → PR keeps `mq:active`, stays at the head, and is retried each cycle until checks settle.
   - **CI red** → eject with a comment naming the failure; the author re-enqueues with `/merge` after fixing.
   - **New commits pushed by a non-bot user while queued** → eject (the queue's "what we tested" guarantee no longer holds).
5. `/dequeue` removes a PR. The PR author or any write-access user can dequeue.

Labels:

- `mq:queued` — waiting at some position in one or more queues.
- `mq:active` — at the head of all its queues; CI cycle in progress.
- `mq:<queue>` — membership marker, one per queue the PR belongs to.

### 4.5 Cadence and concurrency

- 3-minute processor cron. Single concurrency group (see [§4.5](#45-cadence-and-concurrency)); no overlap, no cancellation.
- One PR processed per queue per cycle. No batching.
- The processor commits as `github-actions[bot]` so its merge of `develop` into the PR branch does not trigger the new-commit ejector in step 4.

### 4.6 Permissions

- `/merge`: PR author **or** any user with write/maintain/admin on the repo. The PR-author allowance is what makes the queue usable for external contributors on their own PRs (see [§5](#5-open-source-contributor-policy)).
- `/dequeue`: PR author or write-access user.
- The squash-merge itself is performed via GitHub's merge API by the bot, so branch-protection rules apply (see [§5](#5-open-source-contributor-policy)).

### 4.7 Processing model

Queue state is stored entirely in PR metadata — labels and a hidden JSON comment posted by the command handler at enqueue time — with no external database. The processor reconstructs the full queue picture on every cycle by scanning all open PRs with `mq:queued` or `mq:active` labels.

**Per-cycle algorithm:**

1. **Discover.** List all open PRs targeting `develop` that carry an `mq:queued` or `mq:active` label. For each, parse the metadata comment to recover `enqueued_at` and the set of queues the PR belongs to.
2. **Build queues.** For each queue, sort member PRs by `enqueued_at` (oldest first). The result is six independent FIFO lists.
3. **Identify ready PRs.** A PR is *ready* if it is at the head of every queue it belongs to. Scan all queues; collect the set of ready PRs.
4. **Activate.** For each ready PR not yet labeled `mq:active`:
   - Merge `develop` into the PR branch. If the merge produces conflicts, eject the PR with a comment explaining the conflict.
   - Label the PR `mq:active` and remove `mq:queued`.
   - Skip further evaluation this cycle — CI needs to run against the merged tip before the next step applies.
5. **Evaluate active PRs.** For each PR already labeled `mq:active`:
   - All required checks passed → squash-merge via the GitHub merge API.
   - Any required check failed → eject with a comment naming the failed check.
   - Checks still pending → no action; the PR stays active and is retried next cycle.
   - A non-bot user has pushed new commits since activation → eject.

**Consistency guarantees.** The single-concurrency-group constraint ([§4.5](#45-cadence-and-concurrency)) ensures no two processor runs overlap. Because state is reconstructed from PR metadata on every cycle, the processor is stateless and crash-safe — a failed run simply retries on the next 3-minute tick with no stale in-memory state.

**Conflict handling.** If merging `develop` into an active PR's branch produces a merge conflict, the PR is ejected immediately. The eject comment instructs the author to resolve conflicts locally, push, and re-enqueue with `/merge`.

## 5. Open-Source Contributor Policy

**Policy.** External contributors may use `/merge` on their own PRs. The safeguards that apply to any merge — required reviewer approvals and required CI checks — apply unchanged.

**Why this works without extra queue logic.** `develop` today requires a CODEOWNER approval and a passing TheRock CI run. GitHub's merge API enforces these rules on every merge call — including the queue's. If a CODEOWNER hasn't approved or TheRock CI hasn't passed, GitHub rejects the squash. The queue piggybacks on branch protection; it cannot bypass it. Branch protection is the safety net, the queue is the serializer. These protections apply from Phase 1 onward (see [§6](#6-rollout)); the queue inherits them, it does not relax them.

**Defense-in-depth.** The command handler should also reject `/merge` on a PR that has not yet been approved, so an unapproved PR doesn't sit in the queue burning processor cycles only to fail at the squash. This is a small check at enqueue time, not a replacement for branch protection. (See Phase 2 in [§6](#6-rollout).)

**Latency.** No priority lane in v1. Uniform 3-minute poll for everyone. Equal latency is simpler and avoids second-tier optics. A priority lane is listed in [§9](#9-future-work).

## 6. Rollout

| Phase | What                                                                                                                                                                                                                                  |
|-------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1     | Optional. hipDNN-ecosystem opt-in only. No branch-protection changes.                                                                                                                                                                 |
| 2     | Configure `develop` branch protection: required reviewer approvals + required CI checks. Enable the at-enqueue approval pre-check. Add a non-blocking status check that flags non-queue merges so reviewers can nudge contributors.   |
| 3     | Add a required status check `merge-queue/managed` on `develop`. The queue bot posts it on every PR — pending until queue squash-merge for PRs touching opted-in paths, immediate success ("skipped — no managed paths") for all others. Path scoping lives in the bot, so branch protection stays one rule and other subprojects feel no friction. Repository admins (gardeners) retain bypass via GitHub's standard admin-override; this is an expected escape hatch for emergencies, not a routine path. (See [§7](#7-risks) for related concerns.) |
| 4     | Extend opt-in to other rocm-libraries subprojects on request.                                                                                                                                                                         |

## 7. Risks

- **3-minute poll** is slow at low load. Acceptable; revisit if it pinches.
- **Head-of-line stalls.** Cross-queue blocking means a slow core PR at the front of every queue holds up everything else. Mitigated by `/dequeue` (see [§4.4](#44-pr-lifecycle)) and reviewer discipline on core PRs.
- **Maintenance debt.** Custom in-repo Python is a maintenance cost compared to a hosted service. Accepted; revisit if the scripts grow beyond a single maintainer's head.
- **Surprise auto-eject** when an author pushes new commits mid-queue. The eject comment must be explicit about why and how to re-enqueue.

## 8. Open Questions

- Should the at-enqueue approval pre-check ship in Phase 1 or wait for Phase 2?

## 9. Future Work

Batch merging (`batch_size > 1`), a priority lane for short-running PRs, a dashboard listing all queue contents, auto-rebase on conflicts, and extending opt-in beyond the hipDNN ecosystem.

## Appendix A: Prototype reference

A working prototype implementing a near-cousin of this design exists on `fork/develop`. The configuration must be reshaped to match this RFC before adoption — specifically, the provider rows in `PATH_TO_QUEUES` must add `integration-tests`, the `dnn-providers/integration-tests/` row must add the four providers, and `merge_queue_command.py` must drop its hard write-access gate to allow PR authors per [§4.6](#46-permissions). The files involved:

- `.github/workflows/merge-queue-{command,process,status}.yml`
- `.github/workflows/mergify-scopes.yml`
- `.github/scripts/merge_queue.py`
- `.github/scripts/merge_queue_command.py`
- `.github/scripts/merge_queue_config.py` — the path → queue table from [§4.2](#42-path--queue-mapping) lives here
- `.github/scripts/merge_queue_process.py`
- `.mergify.yml` — scopes for CI status visibility
