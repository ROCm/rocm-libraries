<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Merge procedure (the merge queue)

## Engine

Gate and batch-gate jobs, the auto-merge guard and the merge plan run while the merge queue is enabled
(`features.merge_queue.enabled`), in prototype too, where nothing goes out (each arm, hold, comment and issue is kept
as an unsent item) and the guard only alerts. Each confirmed hold requests changes (`features.merge_queue.auto_hold`),
and
arming (`features.merge_queue.auto_merge` `enabled`) runs only for PRs <user_login> cleared (`rk clear N`).

**Engine:** `rkmon_lib/merge.py`, run by every tick while the merge queue is enabled. Run `rk queue-advance` or
`rk gate N` by hand only on <user_login>'s instruction.

**Gates G5–G7 come from check status lines:**
- `merge`, `build`, `byteid`, `byteid_all`, `libparity`, `tests` and `installed` give G6;
- `golden` and `dg` give G5;
- `compile` gives G7.

They are recorded for the exact head and develop SHA of the merge that was checked, with develop's rocKE tree id. Review
checks (`workers/prN/checks/`) record them too, so a PR reviewed on today's develop needs no extra gate job. They stay
fresh while develop's rocKE tree is unchanged: develop moves outside rocKE cannot change a rocKE gate result.

## Two tiers: candidates and ready

| Stage | Gates | What happens |
|---|---|---|
| **Candidate** | G1 (our 🟢/🟡 at head), G2 (approvals) and G8 (no stack, no hold label, not frozen) pass; CI, the fresh merge or the ABI turn may still be pending or failing | Its merge analysis starts early: a gate job (`ANALYSIS_START #N`) when one of the parallel builds is free. Develop sanity, the head batch and review checks go first; a candidate that has waited `analysis_max_wait_min` (90) goes next anyway. Never in the first 5 minutes of a draft of the local review loop on this machine, and not while a worker on a huge PR (`huge_pr_lines`, 8000) runs, unless it waited 90 minutes or fixes develop. A failure is held for triage like any gate failure |
| **Ready** | Every static gate: G1–G4, G8, G9 | Batched, validated, proposed or armed (below) |

**On the dashboards** the ready tier is the **Mergeable Candidates**: PRs that passed CI and can merge now (the
Pipeline box and the card above the Merge plan). A candidate that is not ready is not listed there: one whose CI still
runs, fails only on known CI infrastructure at its head (the review's `CI_INFRA` record), or fails while develop's own
CI fails too is counted as "+N waiting on CI"; one whose CI fails because of the PR is in Needs Work; one with green
CI and another gate open is counted as "+N waiting on another gate".

**Hold labels.** A PR with a label like `DO NOT MERGE: documentation`, `on hold`, `ON_HOLD`, `WIP` or
`Not ready to Review` fails G8 until the label is removed (`hold_label_patterns`). It is
neither a candidate nor an approval candidate.

**GitHub's own block.** The develop ruleset also requires resolved review threads, and an extra approval for changes
pushed after the approvals (`required_review_thread_resolution`, `require_extra_approval_for_unattributed_changes`).
Stale approvals are not dismissed.
- G2 reads the review threads: an unresolved thread keeps it pending, and `rk approvals` names who is next.
- G3 checks the ruleset's required checks by name (Math CI Summary, pre-commit, Multi-Arch CI Summary, Component CI
  Summary), at the PR's head. GitHub's rollup only lists checks that reported. A required check that never reported
  leaves the PR `BLOCKED` with a green rollup. G3 then stays PENDING with "required check not
  passed yet: X (missing)", and the author or a maintainer has to re-run CI or push.
- The names come from the ruleset once a day; each head's results are cached until it changes.
- The board blames "GitHub's own rules" only when every required check passed and nothing else explains `BLOCKED`.

**Auto-merge guard:**
- The tick prints `!! AUTOMERGE_RISK #N` when a PR has auto-merge on and fails a gate GitHub does not enforce itself:
  - G1: our 🔴/🟠 review stands at its head;
  - G5–G7: a failure on the merge with today's develop that triage confirmed as the PR's own (a hold). Untriaged
    failures never count;
  - G8: a hold label (DO NOT MERGE, WIP, on hold, Not ready).

  GitHub itself enforces CI, approvals and conflicts.
- The board lists it under Needs you, with the plain-words reasons.
- While the merge queue is live (the guard turns auto-merge off), the line says to run `rk automerge-off N`. That runs
  `gh pr merge --disable-auto` and then the sticky queue comment: "Hi @author, I turned auto-merge off…", what fails
  and what to do, in plain words. It does so once per head. <user_login>'s own PRs only when
  `features.merge_queue.auto_merge_guard.on_your_pull_requests` is `same_as_others`; auto-merge <user_login> turned on
  themselves is never turned off, on any PR: the line alerts instead (`rules.md`, Auto-merge guard, with its wave-1
  exception).
- While the merge queue sends nothing (prototype, or the monitor in standby) the guard only alerts
  (`rk automerge-off` refuses: the auto-merge guard only alerts): it is <user_login>'s call.

**The dashboard explains the gates** in a collapsible "What the gates mean (G1–G9)" panel in the Mergeable Candidates
card: what each gate checks, when it passes, and who fixes a failure.

**What the tick does to the head batch** (`rk batches`):
- **Single PR with gates that are not fresh:** a gate job (`rk gate N`) refreshes `wt<N>m` with `mkwt.py` and runs
  the checks as a gate (`workers/prN/gate/`).
- **Several PRs:** a batch gate job merges the members in order onto develop in `wtbatch`, checking every step when
  goldens or ABI are involved. The result is in the batch's folder in `workers/`, named for develop's first 8 characters
  and the members (`workers/b9b1e04c2-101-102/result`): `validated`, `conflict #N`, or `fail at step k`.
- **Carry-over:** a validated batch stays valid while develop moves only by its own members, landing in order.

**Tick lines and what to do:**

| Line | Do |
|---|---|
| `PROPOSE A → B` | Tell <user_login> the validated order. <user_login> merges |
| `ARM_READY #N` | Only when every member is cleared. Re-read the batch, then run `rk arm N`. It checks that auto-merge is on, `cleared`, the freeze, that nothing else is armed, that N is the next member of a validated batch, and the live head via the API, then runs `gh pr merge --auto --squash --match-head-commit` |
| `!! DISARM_NEEDED #N` | The queue froze, the head moved, or develop moved by other work. Run `rk disarm N REASON` at once |
| `LANDED #N` | The armed PR merged. Develop sanity follows, and the next member arms only after it is green |
| `!! AUTOMERGE_JUMP #N` | Auto-merge would land #N ahead of the merge plan: it takes an ABI number ahead of an earlier lane member, or makes an R0–R1 PR rebase. Tell <user_login> in one line; it is their call (ask the author to wait, or let it land). Once per head and finding. A top priority or a pinned group is expected first and never alerts. No outward action |
| `QUEUE_NOTICE #N` | Run `rk queue-notice N` to create or edit our single in-place status comment (line 1: `Automated merge-queue status by <AUTOMATED_REVIEWER_NAME>`). It goes out only while the merge queue is live (`features.merge_queue.mode`), which <user_login> sets only once they have told the team; in prototype it is kept as an unsent item |
| `BATCH_FAILED` / `BATCH_BLOCKED` | The member the result names is held (next row). The others re-batch without it |
| `TRIAGE #N` | A gate or batch failure held the PR. Run the failing check on develop alone. If it's the PR's fault: `rk hold N REASON`, then `rk request-changes N FILE` (A held queue head, below). If develop or our tool is at fault: `rk release N`, then fix the tool or open an incident. A develop failure nobody is tracking is a breakage: a develop issue, CI-red or stability (`incident_procedure.md`), after `rk breakages`. Other untracked product issues get a <JIRA_PROJECT> ticket |
| `!! HOLD_REVIEW_NEEDED #N` | A confirmed hold without a change request: GitHub would still let it merge. Run `rk request-changes N FILE` at once |
| `HOLD_CLEARED #N` | The author pushed, or the PR closed. It is re-gated; nothing to do |
| `HOLD_FIXED #N` | The author pushed since the hold, and our change request stands. The fix gets a P1 follow-up review and its merge analysis (a candidate). Nothing else to do |
| `HOLD_REAPPROVED #N` | <user_login> re-approved (or the change request went another way): the hold's record ends |
| `HOLD_STALE #N` | Held for 24 h without a push. Ask <user_login>: keep its place (`rk hold N`) or release it (`rk release N`) |

## A held queue head

A failing PR keeps its position, and the queue moves on to PRs that don't affect it.

A confirmed hold (`rk hold`, `rk request-changes`) and the `TRIAGE` and `HOLD_*` lines run only while the merge
queue is live with `features.merge_queue.auto_hold`; otherwise nothing is held on GitHub, and a gate failure is
reported to <user_login> only.

**When:** a gate or batch gate fails. The tick keeps the PR in status `triage` instead of re-testing it every tick.

**Triage:** confirm whose fault it is before anything reaches the author.
- Run the failing check on develop alone.
- **The PR's fault:** `rk hold N REASON`. Only now can the author hear about it, through the change request below.
- **Develop or our tool:** `rk release N`. Releasing a gate hold clears the recorded gates, so the PR is re-gated after
  the fix.

**Request changes on every confirmed hold,** so the PR cannot merge without the fix:
1. Right after `rk hold`, write the change request and run `rk request-changes N FILE`.
   - Line 1 is `Automated merge-queue hold by <AUTOMATED_REVIEWER_NAME>`.
   - In plain words: what fails on the merge with today's develop, and the exact fix (for example the re-bless
     commands).
   - End with: "Once it's fixed, push and re-request a review from @<user_login>. A held PR needs that explicit
     re-approval before it can merge."
   - The public-repo rules apply.

   It is the only review the toolkit submits. It refuses without a confirmed hold at the head, on a closed or merged PR,
   on <user_login>'s own PR, and when <user_login> already requests changes. The tick then records <user_login>'s own
   change request as the hold's.
2. The author pushes the fix. The hold ends (`HOLD_CLEARED`, then `HOLD_FIXED`), and our change request stands:
   - G2 stays pending, so the fix is a candidate and its merge analysis re-gates it;
   - its follow-up review is a P1 final review.
3. The author re-requests <user_login>'s review. Once our review is current at the new head and the merge analysis
   passes there, the PR is listed under Ready for your approval: "held for …; it needs your explicit re-approval".
   Before that, "what's left" names who is next: the author re-requests, or we re-gate.
4. <user_login> re-approves (`HOLD_REAPPROVED`). We never approve and never dismiss the change request, even after
   `rk release`. A released PR also waits for <user_login>'s re-approval.

**While held:**
- The PR keeps its rank but is neither batched nor armed.
- Any ready PR whose landing would force the held PR to rebase (a textual conflict on today's develop) waits too: the
  "no impact" rule. Its blocker says `waits: ... held #N`. A shared golden shard or DG family is care, not a wait (the
  merge plan: only conflicts batch and hold). The merge plan puts held PRs first,
  so they keep their place there too.
- A conflict only on the ABI counters is exempt. The lane gives the number to the next lane member that may land, and
  the held PR renumbers later.
- `rk holds` lists the held PRs. Each tick's summary shows a `HOLDS:` line.

**End:**
- A push by the author: automatic, and the PR is re-gated.
- `rk release N`.
- After `hold_stale_h` hours (default 24), <user_login> decides whether to keep the PR's place.

**Batch failure:** the member named in the result (`fail at step k (#N)` or `conflict #N`) is held, and the rest
re-batch without it.

GitHub does not re-test PRs against the latest develop, because `strict_required_status_checks_policy` is false. The
gates below make up for that on every PR we propose or arm.

## Gate job for the queue head

While the merge queue is enabled the tick starts it (`rk gate N`) when gates G5–G7 are not fresh for the queue head's
current (head, develop) pair. It is a background job on the merge of the head with today's develop:

| Gate | What the job checks |
|---|---|
| G5 goldens re-blessed | The golden gate as CI's rocke_golden_static (every flavor), compared with develop's run; the differential golden (the CI CTest commands), with no CHANGED or MISSING anchors. A differential that did not run leaves G5 pending. A failure means the PR needs a re-bless: comment to the author with the exact command |
| G6 build + engines | The merge itself; the native build and ctest; byte identity for every touched family at every emission flavor of the tree; the library kernels' C++/Python parity (with `libparity_check`); the tests the checks pick; the installed tests |
| G7 arch coverage | See the next section |

### G7 arch coverage

1. **Affected arches.** Take them from the touched paths. An arch directory means that arch. Core, lowering or IR code
   means every production target.
2. **Compile evidence for every affected arch.** Byte identity or the golden, plus a clang 24 compile with no new spills
   or scratch.
3. **Execution evidence for numerics-relevant changes.** Any one of:
   - a CI lane that actually ran the tests (not skipped);
   - the author's hardware evidence in the PR;
   - our own run on that arch's GPU (the remote GPU tests, `features.review_automation.remote_gpu_tests.enabled`).
4. **A gap** is a 🟠 in the review, unless <user_login> waives it.

The tick reads the job's status lines from `workers/pr<N>/gate/` and records them in `state.json` as `prs.<N>.gates`
(head, develop, results). Execution evidence (step 3) is judged in the review.

**The automated `compile` section is the compile evidence** (`tools/p_emit_compile.py`). Each parity config of a
touched family is compiled for its own target:
- The target comes from the IR's `target-cpu`, or else from the arch the emitter's `run_emit()` resolves.
- A target is never guessed: a guessed target gives false failures.

Rejected configs are compared with develop, using the `wtbase` worktree at the merge's develop SHA:
- A config develop builds but the PR rejects is a **new rejection** and fails G7.
- One develop also rejects is reported, not failed.

This matters because byte identity counts "both engines reject" as agreement.

## ABI lanes (gate G9)

`rocke/abi.h` has two counters, and each is a serialized lane:
- **Binary** `ROCKE_ABI_VERSION`: struct layout, mid-list enumerators, signatures.
- **Wire** `ROCKE_RECIPE_ABI`: constructs an older engine would misread. A wire bump must also register the construct
  in `portable_ir/src/abi.py`.

A family kernel-ABI string (`*_ABI_VERSION = "…/vN"`) whose existing value changes also gets a lane. A new family
string conflicts with nothing.

Numbers are assigned at landing, in the merge plan's order:
- The lane head is the lane member in the earliest merge-plan wave, once it may land (its other gates green and nothing
  ahead of it conflicting). Without a plan: the first lane member whose other gates are green; among the members not
  ready yet, top priorities first.
- G9 passes only for the head, and only when its claim equals develop + 1 with the ` * N: …` history line present and
  no earlier history line dropped.
- The first in line, while it cannot land yet, is pending ("it takes the next number once it can land"). Everyone else
  waits with a tentative number (`rk abi` or the dashboard).
- A stack moves as one unit under its bottom PR, and identical bump hunks carried by several PRs count as one claim.

**What to do:**
- When a PR becomes the lane head with the wrong number, the plan shows `ABI_RENUMBER`. Ask the author once, in the
  review comment, to set the number.
- A PR that changes or removes lines in a C ABI header (`rocke/*.h`, not `instance_*`, `helper_*` or `*_internal.h`)
  without a binary bump gets G9 pending. Check whether it is append-only. If so, run `rk abi-ok N`; if not, raise a 🟠
  "needs a binary ABI bump".

**After an ABI landing,** develop sanity must also confirm that every number has exactly one history line.

## Merge plan

The merge plan groups the PRs that can be merged together without conflicts, and recommends the merge order that
causes the least toil.

Every queue computation (each tick, and `rk status --queue`) plans the near-merge PRs: non-draft PRs into develop
without a hold label, with our 🟢/🟡 or an approval, plus every develop fix and your priority. `rk merge-plan` prints it.
The board has a Merge plan row; both dashboards have a Merge plan section, the HTML one with a conflict grid (Ⓐ marks
ABI pairs), a lane line for the ABI lanes and the rebase advice collapsed under "Rebase advice (N PRs)". The
Mergeable Candidates section above it shows the PRs that passed CI and can merge now, and their gates.

**Conflicts and care.** Each PR is merged with today's develop. Two PRs that share a file have their two merges merged
on develop. Textual conflicts order, batch and hold. **ABI pairs:** two PRs that bump the same ABI counter (the binary
lane, or the recipe lane) conflict even when git merges them cleanly, because the later one renumbers to develop + 1.
They order and separate the waves like a textual conflict, never hold (a held PR keeps its marks but orders nothing and
goes to the back of its lane), and show as Ⓐ in the grid and as a renumber step in the rebase advice. Care is shown and
weighted, and changes neither the order nor the batches: the same golden shard, DG family, CMake list or test-category
file, or new parity config indices (the first to land keeps the index). A PR that conflicts with develop rebases first,
and the plan names the PRs it shares the most files with.

**The ordering rule.** Every row is a wave: a group of PRs that can be merged together in the given order without
causing conflicts, ideally ending with a PR that causes conflicts. A priority PR always goes into the earliest wave
possible, so it is never left behind; priorities that conflict with each other sit in different waves.
1. **Priorities first, wave by wave.** The priorities, in order: pinned groups (`rk group`), the top priorities (a
   develop fix CI sees, then your priorities, then other develop fixes; smaller first), held PRs (they keep their
   place), and ready PRs the plan has kept waiting 24 h. Each goes into the earliest wave where it conflicts (textual,
   ABI, or a pair not checked yet) with nothing already placed. Priorities that conflict with each other land in
   different waves, the higher one first.
2. **A priority that conflicts with develop** is placed the same way, by the files and ABI lanes it shares (no pair is
   merged with a PR that conflicts with develop, so a shared file counts). It ends its wave, "after its rebase": "rebase
   now" in wave 1, else "rebase after wave k-1 lands".
3. **The fill.** Each wave then takes in the best set of the other PRs left that conflict with no member. A PR that
   conflicts with a priority still waiting stays behind it, so no PR makes a priority rebase. Weights: 8 ready (R0) or
   top priority, 4 only approvals or threads left (R1), 2 approved with our 🟠 or a follow-up due (R3), 1 the rest.
4. **Union check.** Wherever a wave's members share files, they are merged on
   develop one after another in landing order. A member that conflicts with the union of those before it moves to a
   later wave ("conflicts with #a + #b together"). A wave not checked to the end keeps the plan partial.
5. **Order in a wave:** first the members whose landing makes no later PR rebase, then the ones that do, so a wave ends
   with the PR that causes the next wave's rebases. Within each part: priorities in their order, then the readiest, then
   the lowest weighted rebase cost.
6. **Hubs last:** a non-priority PR with 5 or more conflicts with the others, the ABI lane aside, joins the waves when
   nothing else is left.
7. The ABI lane head is the lane member in the earliest wave (ABI lanes, above).

A ready priority may wait behind a blocked one it shares files with (a rebase, a 🔴): <user_login> merges such PRs at
their discretion, and the next tick plans again.

**What may land now:** a PR in wave 1 that merges clean with develop, with nothing ahead of it conflicting (✓ on the
dashboards), never one "after its rebase". Only those feed the batches. A PR in a later wave waits, and its advice
says for whom: "wait: #A lands first (conflicts in F), then rebase".

**Ready early.** A top priority in wave k > 1 that is at the Ready stage (G1–G4, G8, G9), while a PR in an earlier wave
is not, gets `!! PRIORITY_READY_EARLY` once per PR, head and wave, and a Needs-you item while that stays true. A top
priority that must rebase first is never ready. Both name what it waits on, and who landing it now makes rebase (textual
conflicts with earlier waves, files shared with a rebasing priority ahead) or renumber (an ABI claim ahead). The plan
does not change for it; merging it is <user_login>'s call, and the next tick plans again.

**Auto-merge:** a PR with auto-merge on may land when CI turns green, whatever the plan says. The plan names the rebases
it would force and the ABI number it takes. The tick prints `!! AUTOMERGE_JUMP #N` when it would take the number ahead
of an earlier lane member or make an R0–R1 PR rebase. A top priority is expected first: no alert.

**Cost:** one merge per PR per develop SHA, and one per pair that shares a file, cached by develop's blobs of the shared
files. With develop unchanged a tick merges nothing; a develop move re-merges each PR (about 30, 1–2 s). At most 200
merges in 30 s per tick by default (`merge_plan.max_merges`, `merge_plan.budget_s`). The union checks share that budget.
What is left shows as partial and counts as a conflict meanwhile. No GitHub call. If no PR can be merged with develop at
all (git failing), the queue orders and batches by head-to-head pairs and `d`, and the board says why.

## Merge batches

There are two types of batch:
- **User groups**, pinned by <user_login> with `rk group A B C`. They come first and keep <user_login>'s order. A user
  group stays one batch even when a member is not ready yet, conflicts with another member, or shares an ABI lane. Its
  notes say what it is waiting for or what is wrong; tell <user_login> rather than splitting it. `rk ungroup N` removes
  it.
- **Auto batches**: ready PRs that may land now in the merge plan, with no textual conflict between them on today's
  develop, at most one claim per ABI lane, and no stacks. They have at most 4 PRs and 12,000 lines. `rk batches` lists
  both.

One **batch gate** job validates them together:
- It merges them in landing order onto today's develop and runs the full gate suite on the result.
- It also checks every prefix when a member touches a golden or an ABI counter, so each intermediate develop state is
  validated too.
- On failure, the member the result names is held (see A held queue head), and the rest re-batch without it.

A validated batch lands back to back, with one develop sanity run after the last member. develop CI still runs on every
push.

**Who merges:**
- An uncleared batch is proposed to <user_login> as an ordered list.
- A batch whose members are all cleared is armed one PR at a time, each with `--match-head-commit`. After each landing,
  check that develop is exactly the validated prefix. If anyone else merged in between, stop and re-gate.

**Goldens:** the representative golden is one shard per family (a JSON file of the family's name in
`rocke_representative_ir_sha256/`) with no stored aggregate counts. PRs that touch different shards never conflict on
it, and two PRs that add cases to the same shard conflict only where their entries meet. The merge plan merges them on
today's develop to know. A shared shard or DG family is care: the second to land may have to re-bless, so a batch with a
golden or ABI member still checks every prefix.

## Propose or arm

These cases apply once the head batch is validated.

- **Not cleared, or auto-merge off** (`features.merge_queue.auto_merge` `hints_only`): the tick prints `PROPOSE`. Tell
  <user_login> "#N is next: every gate is green against develop", with develop's SHA. They merge.
- **Cleared by <user_login> (`rk clear N`):** the tick prints `ARM_READY #N`. Run `rk arm N`, which runs
  `gh pr merge --auto --squash --match-head-commit` with the validated head. Never use `--admin`. Arm only one PR at a
  time. `rk arm` refuses if any of these is false:
  - auto-merge is on (the merge queue live, `features.merge_queue.auto_merge` `enabled`);
  - the PR is cleared;
  - the queue is not frozen;
  - nothing else is armed;
  - the PR is the next member of a validated batch;
  - on GitHub the PR is still open, not a draft, and at the validated head.

## Disarm

Run `rk disarm N REASON` (`gh pr merge --disable-auto`, recorded in the ledger) when any of these happens:
- develop moves by anything other than the batch's own members, so the validation does not carry over;
- the head moves;
- the queue freezes;
- the monitor goes OFF.

The tick prints `!! DISARM_NEEDED #N` for the first three. For the last, the `off` subcommand disarms.

## After a landing

Every merge, ours or anyone's, triggers a develop sanity run on the new develop SHA.
- **Green:** re-gate the next queue head.
- **Red:** go to `incident_procedure.md`.
