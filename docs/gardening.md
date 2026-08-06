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

- The failure matches a known infra issue that is already filed, and you can link it.
- The failing check is unrelated to what the PR actually changes.
- The failure is not specific to this PR: it reproduces on other PRs, or it survives a re-run.
- No new, different failure is hiding behind the known one.

Leave the reasoning on the PR when you push it through: the check that failed, the issue it maps to,
and why the change is not the cause. That note is what saves the next gardener from re-deriving the
same conclusion. Then keep an eye on the post-submit run for that merge, since pushing it through
makes the outcome yours.

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
   reported has often finished by the time you look at it.
2. Confirm the current state before investigating. `gh pr checks <PR_NUMBER> --repo ROCm/rocm-libraries`
   shows whether the check is still failing, passed on a retry, or never started. If it is green
   now, say so and ask for the earlier run rather than guessing which failure was meant.
3. Check the known bugs first. Look through the
   [gardener known bugs](https://github.com/ROCm/rocm-libraries/issues?q=is%3Aissue%20state%3Aopen%20label%3Agardener)
   and the owning CI system's known issues before digging into logs. Many reports turn out to be an
   already-tracked infra failure, and linking that issue is faster and more useful than a fresh
   investigation.
4. Re-run before escalating. Infra flakes such as host timeouts, GPU sanity check hangs, and runner
   resource errors frequently pass on a re-run, which is cheaper than a hand-off. A failure that
   survives a re-run, or that reproduces on unrelated PRs, is worth an issue.
5. Answer in the thread. Say what you found, link the run, job, or issue you based it on, and name
   who owns the next step. A reply with no links leaves the next gardener to redo the same work.

If none of the above resolves it, route it with the in-scope rules: CI system failures go to the
owning [CI team](#ci-teams), code failures go to the [CODEOWNERS](../.github/CODEOWNERS).

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
