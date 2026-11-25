<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Develop guard and incident procedure (`features.merge_queue.develop_guard`)

While the monitor is ON, every develop failure gets this treatment, with no exceptions.

Develop checks run whenever the develop guard is on, in prototype too, and a red run opens an incident and
freezes the queue. What goes out (develop issues, the culprit's notice, fix links, the revert offer:
`rk issue-file`, `rk notice --kind incident`, `rk fix-link`) goes out only while the merge queue is live; in
prototype, and in standby, those commands keep it as an unsent item, and you report the break to <user_login>.

**Routing:**
- A develop breakage is tracked as a **GitHub issue** in the repository
  `features.merge_queue.develop_guard.issue_repository` names (blank: `repository.name`), never as a <JIRA_PROJECT>
  ticket: one source of truth for breakages. Several people keep develop green, and the develop-green team
  force-reverts where necessary.
- Its type, and so its label, says whether CI sees it: CI-red (`features.merge_queue.develop_guard.ci_red_label`)
  when the break makes CI red; stability (`features.merge_queue.develop_guard.stability_label`), assigned to the
  culprit's author, when CI does not see it (a failing test no CI lane runs).
- Someone else may already have filed it. Read the open ones first (`rk breakages`; the tick reports rocKE ones as
  `BREAKAGE_OPEN`).
- The culprit PR's author hears about it through the issue (cc) and one notice on the PR.
- After `revert_offer_after_min`, the revert decision stays with the develop-green team. We offer them evidence (a
  verified local revert) on the issue, with <user_login>'s OK. We never push a revert or open a revert PR unless
  <user_login> asks.

## What the tick does by itself (`rkmon_lib/guard.py`)

**Sanity runs.**
- Every develop move that brings rocKE commits queues a run. A burst of landings is tested once, at its newest SHA.
- Runs take one of the parallel builds, ahead of review checks, in the dedicated worktree `wtsanity`. Each run's
  output is in `scratch/sanity/`, in a folder named by the first 12 characters of its SHA.

**What a run checks:**
- native build plus ctest;
- the golden gate (every flavor), as CI's rocke_golden_static;
- the differential golden's two CTest commands;
- byte identity for the families the landed commits touch, at every emission flavor;
- every family at every flavor, daily or whenever engine core changed;
- the ABI history invariant (`tools/abi_check.py`).

**Tick lines** (ID is the incident's id, SHA a commit's first 10 characters):

| Line | Meaning |
|---|---|
| `SANITY_START SHA` / `SANITY SHA: running` / `SANITY_GREEN SHA` | Progress |
| `!! CRITICAL INCIDENT_OPEN ID` | A sanity run is red, or develop CI is red in a rocKE job. The incident opens and the queue **freezes** (`st.frozen`) |
| `!! CRITICAL CULPRIT ID: #N` | The culprit is identified, either as the only landing since the last green run or by bisect |
| `BISECT_NEEDED ID` | More than one candidate. Run `rk bisect`: it re-runs only the failing sections on each candidate merge, oldest first, and the first red one is the culprit (`CULPRIT … by bisect`) |
| `!! CRITICAL REVERT_OFFER ID` | Open for `revert_offer_after_min` or more, with a culprit |
| `INCIDENT_CLOSED ID` | A later sanity run is green; the queue unfreezes |

## What you do

1. **On `INCIDENT_OPEN`.**
   - Tell <user_login> at once, at the top of the reply: the SHA, the failing sections and the next step.
   - Read the run's `sanity.md` (in its folder under `scratch/sanity/`), or the CI job log for CI-only signals
     (`rk job-log ID`: its failing lines first). Check any summary against the raw log.
   - Separate infrastructure flakes from real breaks. A recurring flake is still tracked, as a CI-red issue: the repo
     files intermittent failures there too. A one-off runner failure is noted with
     `rk incident-note ID summary "…"` and watched.
2. **Attribution judgment.** Bisect names the first merge where the failing check fails *locally*. If a later merge in
   the range is the one that *enabled* that check in CI, the enabling merge is the culprit and the earlier ones are
   contributing. Example: a merge turns on a CTest whose golden was recorded before two earlier merges landed; bisect
   names the first of those, but the enabling merge is the culprit. Record the corrected attribution with
   `rk incident-note ID culprit N` and a `contributing` note.
3. **On `CULPRIT`** (after `rk bisect` if needed):
   - **The develop issue.** Run `rk breakages`.
     - **Already filed** (same failing test, same culprit or fix PR): `rk incident-note ID issue N`. Comment only
       when we add something, such as the bisect result, the enabling-vs-culprit attribution or a local repro:
       `rk issue-comment N FILE --key ID`. If its type is wrong today (CI turned red, or never ran the test):
       `rk issue-relabel N --to stability|ci_red [--assign LOGIN]`.
     - **Not filed:** draft it in the repo's develop issue style.
       - Sections: Context (branch, failing workflow, example run, component, platform/arch, fix candidate); Error;
         Cause; Why it slipped through CI; Proposed fix; Resolution criteria.
       - Labels: the CI-red label with its extra labels when CI is red; otherwise the stability label, with the
         culprit's author's login as `assign:` in the front matter. Then `project: hip-kernel-provider`,
         `rocke: platform`, the platform and arch labels, `code-regression`.
       - cc the culprit's author.
       - Line 1 of the body is `Automated develop-health report by <AUTOMATED_REVIEWER_NAME>`. Public repo: no
         internal links, no absolute perf numbers, bare <JIRA_PROJECT> tags.

       Once the break is proven (develop alone reproduces it) and no issue exists, file it **without asking**:
       `rk issue-file FILE`, with the incident's id as `incident:` in the front matter, which links the incident.
       Then tell <user_login> in one line.
     - **Link the fixes.** Search open PRs for a fix: `gh search prs` on the failing test, the culprit PR, the error
       text, or the area. Name each in the issue ("Fix candidate: #N"), and record it with
       `rk incident-note ID fixed_by N`. Either makes it a develop fix: review P0, merge tier 0. For an issue already
       filed, link a verified fix with `rk fix-link N P`; the tick's `FIX_CANDIDATE` lines name likely ones.
     - **No planned fix** (CI-red issues only): run `rk teams-breakage N` at once. It alerts the <TEAM_CHANNEL> with
       an automated message that links the issue and asks who can take it. It refuses when a fix is named, and for a
       stability issue: the breaking PR's author, assigned to it, takes that one.
   - **Swarm.** Write the notice. Line 1 must be `Automated develop-health notice by <AUTOMATED_REVIEWER_NAME>`.
     Include the failing check, the signature, the evidence, the develop issue's number and a request to fix forward.
     Post it with `rk notice N FILE --kind incident --key ID`; one notice per incident and PR. Then run
     `rk incident-note ID comment URL`, with the notice's address.
4. **On `REVERT_OFFER`.** The develop-green team decides on force reverts.
   - Prepare a local revert of the culprit merge in a scratch worktree (`git revert -m 1` on the merge's SHA, on a
     branch from develop). Check that it builds and that the failing section passes.
   - Ask <user_login> whether to post that evidence on the develop issue (`rk issue-comment N FILE --key ID-revert`).
     Then run `rk incident-note ID revert_offered yes`. Push a revert branch or open a revert PR only if <user_login>
     asks.
5. **On a fix PR.** Record it with `rk incident-note ID fixed_by N`.
   - The toolkit then treats it as a develop fix: review P0 with the shorter settle (`settle_develop_fix_min`), merge
     tier 0, analysis first.
   - A PR that closes an open develop issue (either type), that an open issue names as its fix candidate, or that
     `rk fix-link` linked to one, is picked up automatically (`st.develop_fixes`; the board's "Develop fixes" row).
6. **On `INCIDENT_CLOSED`.** If we filed the develop issue, comment the fix and the green sanity run on it
   (`rk issue-comment N FILE --key ID-closed`). The issue's owner closes it. Write a journal line, and
   tell <user_login> in one line. `rk incident-close ID [FIXED_BY]` closes the incident by hand when needed.

Incidents are kept in `state/incidents.json`, and the weekly report reads them: who broke develop, who fixed it, and
for how long.
