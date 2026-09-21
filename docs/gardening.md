# ROCm Libraries Gardeners

This documents the mechanics of
[gardening](https://github.com/ROCm/TheRock/blob/main/docs/rfcs/RFC0002-MonoRepo-Gardener-Rotations.md)
for the ROCm Libraries. If you haven't read the above doc, please start there.

## Becoming a member

Gardeners will need to be members of the [Compute Library Gardeners team](https://github.com/orgs/ROCm/teams/compute-library-gardeners).
Please contact an owner to become a gardener.

## Communications channel

We will be leveraging a shared Teams channel that contains all gardeners as well as core
infrastructure team members. You will be added to this channel once you become a member.

For anyone who wants to reach a gardener please email:
[rocm-libraries-gardeners](mailto:rocm-libraries-gardeners@amd.com)

## Mechanics of Gardening

Your primary job is to keep the mono-repo shippable. In order to facilitate this we've made
status badges for all relevant CI available here:
https://github.com/ROCm/rocm-libraries?tab=readme-ov-file#monorepo-status-and-ci-health.
Effectively your job is to ensure all status badges are green. All of these status
badges are clickable which will allow you to deep-dive on any failures quickly. If any
CI is missing, please file an issue leveraging the "gardener" tag, ping on the teams chat,
or preferably, add it yourself. You'll probably be tagged to review the PR if someone
else gets to it first.

## Notes on Privileges

Developers will not be able to bypass pre-submit checks in this repository unless an admin or
gardener pushes it through. This is being done intentionally to ensure we keep the quality of
the tree green. This also means that you will be asked to push changes through without
additional context. Your duty is to ensure you keep the tree green (or make it greener) so gardeners will need to understand the context before approving
any of these changes. Changes
that are ok:

- Reverts to fix broken things.
- Fast-forward fixes where reverts are unclear
- Fixes unrelated to code health (docs, etc)

On a case by case basis you should consider critical customer fixes, but these should be considered
as a group and likely admins should be approving the majority of those.

As an example to include an admin: *we have a critical feature but develop is broken and it is unrelated to our changes*

### Pushing through a known infra failure

The most common bypass request is a PR blocked by a CI failure unrelated to the change. Every one of
these must hold:

Precondition | Keypoint
---- | ---------
Run has finished | A verdict read off an in-progress run is not a verdict
Failing check is **required** | If it is advisory there is nothing to bypass; the author can merge normally
Known issue exists | Filed, and you can link it
Unrelated to the diff | The change cannot plausibly cause it
Not specific to this PR | It reproduces elsewhere, or survives a re-run
Nothing new hiding behind it | No second, different failure in the same job

Then ask what the list does not: **would waiting fix this?** A lane broken for weeks with no
assignee will not repair itself, so bypass it. A platform outage, one unlucky runner, or a breakage
already fixed on `develop` clears on its own or with a fresh run, so wait.

When you do push it through:

- **Rationale in the thread first** - the check, the issue it maps to, why the change is not
  to blame.
- **State what you are not vouching for.** An infra classification does not cover numerics,
  performance, or any lane that never ran.
- **Preserve the author's description.** A regenerated commit message drops tracking lines such as
  `JIRA ID` that the policy checks and downstream tooling depend on.
- **Watch the post-submit run.** Pushing it through makes the outcome yours, and that run is where
  the evidence you skipped finally appears.

### When the answer is no

Declining is the common outcome. Make it actionable:

- **File it** - run and job links, failing step, error text - and **assign an owner** rather than
  leaving it unassigned.
- **Route it** - code to the [CODEOWNERS](../.github/CODEOWNERS), CI system to the owning
  [CI team](#ci-teams), fleet to the [SRE rotation](#working-with-the-sre-rotation).
- **Reply where it was asked** - the PR thread, and the gardening channel if it was raised there.
- **Give a dated condition** - "if only `<issue>` remains and no owner has replied by `<date>`, I
  will push it through". "Not yet" reads as stalling.

A bypass offered to you is still your call: it removes the reporter's objection, not the requirement
to keep the tree green.

## Scope of Gardeners and Developers

In scope:
- Gardeners are responsible for ensuring develop (post-submit) checks remain green.
- If a post-submit check is red, the gardeners should review the failing CI system and triage the issue.
- No matter the issue, gardeners should notify the larger gardening team at least once per day about any post-submit failures.
- If the issue is related to a failure in the CI system (not a code change), the gardener should note the issue,
  verify whether existing PRs are facing the same problem, and notify the appropriate CI team, escalating the issue if required.
- If the issue is related to a code change, the gardener should isolate the error message, and notify the
  appropriate component owners with a link to the log (reference the [CODEOWNERS](../.github/CODEOWNERS) file).

Not in scope:
- Gardeners are not responsible for fixing code changes that break post-submit checks.
- Gardeners are not responsible for monitoring the health of every open PR.

Developer responsibilities:
- If developers find CI system failures in their PR (pre-submit) checks they should notify the gardener on rotation and the appropriate CI team.

### First pass triage

Most requests arrive as "my PR is blocked, can someone look?". Steps 1-3 are cheap and often end it.

\# | Step | Keypoint
---- | ------- | ---------
1 | **Get a pointer** | Run URL, job URL, and roughly when. Classify the ask too: *"confirm these are unrelated"* asks for analysis, not for a bypass
2 | **List the required checks** | Only these block a merge, and the set is per repository
3 | **Read the merge state** | Says whether a bypass is even the question
4 | **Confirm it is current** | Never triage an in-flight run; a queued lane flips verdicts
5 | **Find the existing issue** | Search the error text across both repositories, not by label
6 | **Re-run, or re-dispatch** | A re-run replays the *same* merge commit
7 | **Answer in the thread** | The classification, the links behind it, and who owns the next step

```bash
# 2 - required contexts on the base branch (needs no admin rights)
gh api repos/ROCm/rocm-libraries/rulesets --jq '.[] | "\(.id) \(.name) \(.target)"'
gh api repos/ROCm/rocm-libraries/rulesets/<RULESET_ID> \
  --jq '[.rules[] | select(.type=="required_status_checks")
         | .parameters.required_status_checks[].context]'
# 3 - is there anything to bypass?
gh pr view <PR_NUMBER> --repo ROCm/rocm-libraries --json mergeStateStatus,reviewDecision
# 4 - current state of every check
gh pr checks <PR_NUMBER> --repo ROCm/rocm-libraries
# 5 - both repositories, by error text
gh search issues "<error text>" --repo ROCm/rocm-libraries --repo ROCm/TheRock --state open
# 6 - re-run only the failed jobs
gh run rerun --failed <RUN_ID>
```

**Required checks (step 2).** On `develop`, `rocm-libraries` requires `TheRock CI Summary`,
`Math CI Summary` and `pre-commit`, while `rocm-systems` requires `TheRock CI Summary` and
`HIP NVIDIA CI Summary` and no `pre-commit`. The two `gardening.md` files differ by six lines, so
enumerate rather than assume. Everything outside the set - packaging install lanes, coverage
thresholds, aggregates - is advisory: worth an issue, never worth a bypass.

**Merge state (step 3).**

`mergeStateStatus` | `reviewDecision` | What it means
---- | ------- | ---------
`BLOCKED` | `REVIEW_REQUIRED` | Review is missing. Nothing to bypass; route to the [CODEOWNERS](../.github/CODEOWNERS)
`BLOCKED` | `APPROVED` | A required check is red or never reported - the bypass case
`UNSTABLE` | `APPROVED` | Every red is advisory. The author can squash it themselves
`BEHIND` / `DIRTY` | any | The branch needs updating or has conflicts, which is the author's job

A required check that **never dispatched** belongs in row 2, not row 1: it can never report, so
auto-merge never fires and waiting does not help.

**Why search TheRock as well (step 5).** It owns the build, the packaging, and the TheRock-driven
lanes, so a failure surfacing on a PR here is frequently already filed there. Its issues often sit
on a triage board with no labels at all, and its infra labels are `infra`, `infra-timeout`,
`infra-machine`, `test-infra` and `test-flaky` rather than `gardener`. The
[gardener known bugs](https://github.com/ROCm/rocm-libraries/issues?q=is%3Aissue%20state%3Aopen%20label%3Agardener)
list still helps for this repository.

**Re-run versus re-dispatch (step 6).** Pre-submit CI builds the merge of the PR with its base, and
a re-run replays that same merge commit: it clears a flake, but cannot pick up a fix that landed on
`develop` afterwards. That needs a fresh dispatch, and adding then removing a label does it with no
commit and no change to the author's branch - provided the label is not one the CI matrix parses. An
internal gate that ignores label events needs a push.

**Outcomes.**

What you found | What you do
---- | ---------
Nothing red in the required set | Not blocked by CI. Point at the real blocker, and file issues for untracked advisory reds
Every required red is a filed infra issue, unrelated to the diff | The [bypass criteria](#pushing-through-a-known-infra-failure) apply
A required red is real, new, or unexplained | Do not bypass. File it, assign an owner, give a dated condition
No result at all - the job died before running | Fresh run. Recurring across PRs makes it an infra issue
Runners offline, queues growing, jobs stuck across PRs | Hand it to the [SRE rotation](#working-with-the-sre-rotation)

If none of that resolves it, route it with the in-scope rules: CI system failures go to the owning
[CI team](#ci-teams), code failures go to the [CODEOWNERS](../.github/CODEOWNERS).

### Reading the failure

Do not take a red mark at face value.

Symptom | Read it as
---- | ---------
Job died in checkout, setup, or action download | **No result**, not a failure - and not evidence the change is sound either
`Executing the custom container implementation failed` | The runner wrapper reporting a killed step. Read the timestamps above it
Gap equal to the step timeout | An infra timeout
Error immediately above the wrapper message | The real fault
A `Summary` check is red | An aggregate. Count jobs before calling a lane dead
Scattered shards fail while siblings pass | A resource or throttling signature
A whole lane fails identically | Not throttling - look for a real cause

```bash
gh api "repos/ROCm/rocm-libraries/actions/runs/<RUN_ID>/jobs?per_page=100" --paginate \
  --jq '.jobs[] | select(.conclusion != "success") | "\(.conclusion)\t\(.name)"'
```

That endpoint returns only the latest attempt, so a re-run erases the earlier failures. Timestamp
your numbers when you are measuring how far something has spread.

Two arguments are much stronger than "it passed on a re-run":

- **Unreachable diff** - a build guard, a path filter, or an architecture the change does not touch.
  Expand the lane's architecture family first: lanes are named by family, diffs by architecture, and
  the two are often the same thing.
- **A control differing only in the change** - the same job failing at the same time on a branch
  without it, or better, a sibling job in the same run and on the same commit that passed.

### Working with the SRE rotation

The gardener owns the verdict on a PR or post-submit failure; a separate SRE rotation owns the fleet
that produced it. Hand it over when the cause is the fleet rather than one job:

- Jobs queued or stuck beyond roughly ten minutes, across many PRs rather than one.
- Runners offline, or an architecture label with no capacity.
- Checkout or setup steps failing at a rate that is not specific to one workflow.

The AMD-internal
[SRE playbook](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1453823456/TheRock+SRE+Playbook)
lists the runner health report, the dashboards that confirm this in a couple of minutes, the alert
thresholds the rotation works to, and the channel to raise it in.

When one fault is blocking the whole queue, escalating it *is* the work: a bypass per PR costs a
bypass each time and fixes nothing. Bring numbers - how many jobs, over what window, and which step
they share. Attribute to the **first failing step** rather than to the job, or aggregate and
notification jobs will count the same root cause several times over.

### Beyond the Responsibilities

Gardeners should generally aim to be efficient at operating the CI/CD systems and doing first pass triage and routing.
Especially for people new to the role, this will involve more reaching out for help and coordinating resolution, but as experience increases,
it is natural to take a more active role in helping to route and do first pass triage oneself.
While going the extra mile on this is not a requirement of the role, efficient gardeners should aim to develop a proficiency with the
tools and their colleagues such that their judgment reduces the overall toil to the team. Often people who develop these skills find it
more effective to look a little bit more deeply at failures and route for resolution properly in one step.

This kind of investment is deeply valued for the overall health of the team and is encouraged.

### CI Teams

CI | Main primary contact | Team
---- | ------- | ---------
Math CI | eidenyoshida | [ROCm/rocm-math-lib-ci-team](https://github.com/orgs/ROCm/teams/rocm-math-lib-ci-team)
External (Azure) CI | jayhawk-commits | [ROCm/external-ci](https://github.com/orgs/ROCm/teams/external-ci)
TheRock CI | geomin12 | [ROCm/therockinfra](https://github.com/orgs/ROCm/teams/therockinfra)

## Gardener Rotation

[Confluence doc for Gardener Rotation](http://u.amd.com/rocm-libraries-gardeners)

It is the responsibility of the current gardeners to update the table when the gardeners rotate.

### Log

Filling in this section is optional while on rotation. While this level of
organization and tracking is not expected from all members, seeing the incident
history and actions taken in one location can be useful. However, for bugs that you can't immediately address
please file a new GH issue and label it with the "gardener" label.

You can see current list of [gardener known bugs](https://github.com/ROCm/rocm-libraries/issues?q=is%3Aissue%20state%3Aopen%20label%3Agardener)

Date | Library | Issue overview | Link to details | Resolved?
---- | ------- | -------------- | --------------- | ---------
6/30 | | | | ✅
