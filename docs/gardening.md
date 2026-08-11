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

The most common bypass request is a PR blocked by a CI failure that has nothing to do with the
change. Before pushing one of those through, confirm all of the following:

- The run has finished. A verdict read off an in-progress run is not a verdict; a queued lane
  turning green routinely changes the conclusion.
- The failing check is one that actually blocks the merge. See
  [First pass triage](#first-pass-triage) for how to list the required checks. If the red check is
  not in that set there is nothing to bypass, and the answer is that the author can merge normally.
- Every blocking failure matches a known infra issue that is already filed, and you can link it.
- The failing check is unrelated to what the PR actually changes.
- The failure is not specific to this PR: it reproduces on other PRs, or it survives a re-run.
- No new, different failure is hiding behind the known one.

Then ask the question that list does not cover: **would waiting fix this?** A lane that has been
broken for weeks with no assignee will not repair itself, and a bypass is the right call. A platform
outage, a single unlucky runner, or a breakage already fixed on `develop` all clear on their own or
with a fresh run, and a bypass there spends an audit record to save an hour.

Leave the reasoning on the PR *before* you merge: the check that failed, the issue it maps to, and
why the change is not the cause. Say plainly what you are not vouching for, because an infra
classification does not cover numerical correctness, performance, or any lane that never ran.
Writing that boundary down is what keeps the bypass honest, and the note is what saves the next
gardener from re-deriving the same conclusion.

When you merge, preserve the description the author wrote. An override that regenerates the commit
message will drop tracking lines such as `JIRA ID`, which the repository's policy checks and the
downstream tooling depend on. Afterwards keep an eye on the post-submit run for that merge, since
pushing it through makes the outcome yours, and because the post-submit run is where the evidence
you agreed to skip finally appears.

### When the answer is no

Declining is a normal outcome, and most requests end here. A refusal is only useful if it arrives
with the next step attached:

- **File the issue** if nothing tracks the failure yet: the run and job links, the failing step, and
  the error text. Assign it to the team that owns the failing lane rather than leaving it
  unassigned, and label it so the next gardener finds it.
- **Route it.** Code failures go to the [CODEOWNERS](../.github/CODEOWNERS) for the paths involved,
  CI system failures to the owning [CI team](#ci-teams), and fleet-level failures to the SRE
  rotation.
- **Reply where the request was made**, both in the PR thread and in the gardening channel if it was
  raised there, with the classification and the links behind it.
- **Say what would change the answer, with a date.** "If the only remaining failure is still
  `<issue>` and no owner has replied by `<date>`, I will push it through" is something the reporter
  can plan around. "Not yet" is not, and reads as stalling.

A bypass that is offered to you is still your decision. Reporters will sometimes say up front that
they are comfortable with one; that removes their objection, not the requirement to keep the tree
green.

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

Most requests reach a gardener as "my PR is blocked, can someone take a look?". The lists above say
which of those are yours; these steps are the first pass that gets you to that answer quickly.

1. Ask for a precise pointer. A PR link on its own is rarely enough. Ask for the failing run URL,
   the failing job URL, and roughly when the problem was seen. A run that looked stuck when it was
   reported has often finished by the time you look at it. Also read what is being asked: "can you
   confirm these are unrelated?" is a request for a classification, and answering it with an offer
   to bypass skips a step the reporter did not ask you to skip.
2. Establish which checks actually block the merge. A red check is not automatically a blocker. The
   branch ruleset lists the required contexts, and reading it needs no admin rights:

   ```bash
   gh api repos/ROCm/rocm-libraries/rulesets --jq '.[] | "\(.id) \(.name) \(.target)"'
   gh api repos/ROCm/rocm-libraries/rulesets/<RULESET_ID> \
     --jq '[.rules[] | select(.type=="required_status_checks")
            | .parameters.required_status_checks[].context]'
   ```

   On `develop` in this repository that is `TheRock CI Summary`, `Math CI Summary`, and
   `pre-commit`. Everything else — packaging install lanes, coverage thresholds, multi-arch
   aggregates — is advisory. Those are still worth triaging and still worth an issue, but they do
   not stop a merge and never need a bypass. Enumerate rather than remember: `rocm-systems` requires
   `TheRock CI Summary` and `HIP NVIDIA CI Summary` on its `develop` and does not require
   `pre-commit` at all, so an answer does not carry from one repository to the other.
3. Read the merge state before proposing anything. Two fields say whether there is work for a
   gardener here at all:

   ```bash
   gh pr view <PR_NUMBER> --repo ROCm/rocm-libraries --json mergeStateStatus,reviewDecision
   ```

   `mergeStateStatus` | `reviewDecision` | What it means for you
   ---- | ------- | ---------
   `BLOCKED` | `REVIEW_REQUIRED` | Review is missing. Nothing is bypassable yet; route to the [CODEOWNERS](../.github/CODEOWNERS) for the paths the PR touches.
   `BLOCKED` | `APPROVED` | A required check is red or was never reported. This is the case the bypass criteria are written for.
   `UNSTABLE` | `APPROVED` | Every failing check is outside the required set. The author can squash it themselves; say so rather than offering a bypass.
   `BEHIND` / `DIRTY` | any | The branch needs updating or has conflicts, which is the author's to do.

   A required check that was never dispatched belongs in the second row rather than the first: it
   will never report, so auto-merge cannot fire and waiting does not help. Compare the required
   contexts against the check runs actually present on the head commit before assuming a check is
   merely slow.
4. Confirm the current state, and confirm the run has finished.
   `gh pr checks <PR_NUMBER> --repo ROCm/rocm-libraries` shows whether the check is still failing,
   passed on a retry, or never started. While a run is still going, triage only the required checks
   that have already failed and say that the rest is not in yet. If it is green now, say so and ask
   for the earlier run rather than guessing which failure was meant.
5. Search for an existing issue before digging into logs. Many reports turn out to be an
   already-tracked failure, and linking that issue is faster and more useful than a fresh
   investigation. Search on the error text or the failing job name rather than on a label, and
   search [ROCm/TheRock](https://github.com/ROCm/TheRock/issues) as well as this repository: the
   TheRock-driven lanes and the build and packaging code live there, so a failure that surfaces on a
   rocm-libraries PR is often already filed against TheRock.

   ```bash
   gh search issues "<error text>" --repo ROCm/rocm-libraries --repo ROCm/TheRock --state open
   ```

   Searching by label alone will miss things. TheRock issues are often tracked on a triage board
   with no labels at all, and the infrastructure ones use `infra`, `infra-timeout`, `infra-machine`,
   `test-infra`, or `test-flaky` rather than `gardener`. The
   [gardener known bugs](https://github.com/ROCm/rocm-libraries/issues?q=is%3Aissue%20state%3Aopen%20label%3Agardener)
   list is still worth a look for this repository.
6. Re-run or re-dispatch before escalating, and know which one you need. Infra flakes such as host
   timeouts, GPU sanity check hangs, and runner resource errors frequently clear on
   `gh run rerun --failed <RUN_ID>`, which is far cheaper than a hand-off. But pre-submit CI builds
   the merge of the PR with its base, and a re-run replays that *same* merge commit, so it cannot
   pick up a fix that landed on `develop` after the run was created. Moving onto a current base
   needs the workflows dispatched again: adding and removing a label does that without touching the
   author's branch or adding a commit, as long as the label is not one the CI matrix parses. A
   failure that survives a fresh run on a current base, or that reproduces on unrelated PRs, is
   worth an issue.
7. Answer in the thread. Say what you found, link the run, job, or issue you based it on, and name
   who owns the next step. A reply with no links leaves the next gardener to redo the same work.

That is enough to reach one of a small number of outcomes:

What you found | What you do
---- | ---------
Nothing red in the required set | Say the PR is not blocked by CI, and point at whatever is actually blocking it. File issues for the advisory reds if they are untracked.
Every required red maps to a filed infra issue and is unrelated to the diff | The bypass criteria in [Notes on Privileges](#pushing-through-a-known-infra-failure) apply. Post the reasoning, then merge with the author's original description preserved.
A required red is real, is new, or you cannot explain it | Do not bypass. File it, assign an owner, and say what would change your mind.
The job produced no result at all | Dispatch a fresh run. If it recurs across PRs, it is an infrastructure failure, not a PR failure.
Runners offline, queues growing, jobs stuck across many PRs | Hand it to the SRE rotation; see below.

If none of that resolves it, route it with the in-scope rules: CI system failures go to the owning
[CI team](#ci-teams), code failures go to the [CODEOWNERS](../.github/CODEOWNERS).

### Reading the failure

Most misclassifications come from taking a red mark at face value.

**A job that died before it ran anything is no result, not a failure.** A checkout that times out, a
runner that is killed, an action download that fails: none of these say the change is broken, and
none of them say it is sound either. Report them as missing signal rather than as unrelated
failures, so nobody counts them as coverage the change does not have. Wrapper errors are the same
trap one level up — a message about the container implementation failing is the runner reporting
that some step was killed, so read the timestamps above it. A gap of exactly the step timeout is an
infra timeout; an error immediately above it is the real fault.

**A summary check is an aggregate, so count the jobs before declaring a lane dead.**

```bash
gh api "repos/ROCm/rocm-libraries/actions/runs/<RUN_ID>/jobs?per_page=100" --paginate \
  --jq '.jobs[] | select(.conclusion != "success") | "\(.conclusion)\t\(.name)"'
```

A scattered subset of shards failing while their siblings pass is the signature of a resource or
throttling problem; an entire lane failing the same way is not. Note also that this endpoint returns
only the latest attempt, so a re-run erases the earlier failures — take a timestamp with your
numbers if you are measuring how widely something has spread.

When arguing that a failure is not the PR's fault, two things are much stronger than "it passed on a
re-run":

- **The change cannot reach the failing job**, because of a guard in the build files, a path filter
  on the workflow, or an architecture the diff does not touch. Expand the lane's architecture family
  before relying on this: lanes are named by family and diffs by individual architecture, and those
  are frequently the same thing.
- **A control that differs only in the change.** The same job failing at the same time on a branch
  without it, or better, a sibling job in the same run on the same commit that passed. Same run,
  same commit, one green and one timed out is the cleanest control available and costs nothing to
  find.

### Working with the SRE rotation

A separate SRE rotation owns the machines the CI runs on: runner health, queue pressure, offline or
stuck runners, and the hosted build capacity. The division is that the gardener owns the verdict on
a given PR or post-submit failure, and the SRE owns the fleet that produced it. Hand a failure over
when the cause is the fleet rather than any particular job — jobs queued or stuck for more than
about ten minutes across many PRs, runners offline or an architecture label with no capacity, or
checkout and setup steps failing at a rate that is not specific to one workflow. The AMD-internal
[SRE playbook](https://amd.atlassian.net/wiki/spaces/MLSE/pages/1453823456/TheRock+SRE+Playbook)
lists the runner health report and infrastructure dashboards that confirm this in a couple of
minutes, along with the alert thresholds the SRE rotation works to and the channel to raise it in.

When one infrastructure fault is blocking every PR in the queue, escalating it *is* the work.
Pushing them through one at a time costs a bypass each time, teaches nobody anything, and grows
linearly with the queue. Bring numbers instead: how many jobs, over what window, and which step they
share. Attributing failures to the first failing step rather than to the job is what makes those
numbers hold up, since aggregate and notification jobs otherwise count the same root cause several
times over.

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
