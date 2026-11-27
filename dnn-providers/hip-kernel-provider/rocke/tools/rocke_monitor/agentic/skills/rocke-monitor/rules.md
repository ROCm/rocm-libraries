<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE monitor: standing rules

The tick card in `SKILL.md` keeps the short form; this file is the full text. Read it at session start, on `resume`,
after every compaction, and before an outward action you have not taken since then.

## Standing rules (always, no exceptions)

- **Comment only.**
  - Never approve or request changes. The one exception: every confirmed hold gets a change request,
    `rk request-changes N FILE` (below).
  - Never dismiss a review or edit other people's comments.
  - Never change <user_login>'s own review states.
- **A hold requests changes.** A held PR gets a change request, so it cannot merge until its issue is fixed. The author
  clears it by re-requesting a review from the originator, and the PR then needs the originator's explicit
  re-approval.
  1. Right after `rk hold N`, run `rk request-changes N FILE`. The file says what fails, how to fix it, and to
     re-request a review from @<user_login> once fixed.
  2. The author fixes it and re-requests <user_login>'s review. We review and re-gate the fix.
  3. <user_login> re-approves explicitly.

  We never approve or dismiss. <user_login>'s own change request counts. Details: `merge_procedure.md`, "A held queue head".
- **Merging:** see Who merges, below.
- **Verification:** the verifier reviews blind first, then compares with the draft when there is one; every draft
  claim is reproduced (`ran`) or downgraded (`read`) before anything is posted. Trim false findings and add missed
  ones. With no first review pass (`features.review_automation.passes.first_review_pass.model` `default`, its
  default), the verifier reviews alone.
- **Post format:**
  - Line 1 names this deployment's automated reviewer and the model that drafted the review:
    `Automated review by <AUTOMATED_REVIEWER_NAME>: Model - <drafter model>, effort: <effort>`. With a first review
    pass, its model and effort (max for the local review loop, the binding's own `effort` for the tool loop); with
    none, the verifier's model and effort. The footer names the verifier, and credits a first review pass only when
    one drafted.
  - Public repo: no internal links, no absolute performance numbers, bare <JIRA_PROJECT> tags only.
  - What `rk post --premises` stops on, so a slip costs an edit and a re-run: `#` before a number anywhere but the header line
    (write PR numbers bare); `@` before a letter (an @-mention: write decorators without it, `pytest.mark.skipif`); a number
    run into a unit or an `s` ("17s", "the 404s", "5 min", "2 GB": name waits and limits without numbers); "the user";
    "stale"; GPU product names (name the gfx target instead), `/home/` paths, `rk abi` and `rk breakages`;
    ACCEPT or REJECT. The words match whole: "navigation", "instinctively" or "stalemate" pass.
  - PR references (`#NNNNN`, `PR #N`, `/pull/N` links) and dated narratives of particular runs that a PR adds to its
    committed files (code, comments, docs, case studies, runbooks) are a 🟡 Needed-to-merge row with every file:line:
    the repo states generalized facts, not case history. References already on develop are not raised; a PR body may
    name related PRs.
  - Measured ratios only where `platform/AGENTS.md` allows them. Its Compliance rule allows measured relative
    performance results (ratios, percentages, speedups) only in PR descriptions and qualifying benchmark/result files.
    In our comments:
    - describe performance qualitatively ("slower at the measured configurations"), or point at the ratio the PR's
      own description states;
    - counts and structural ratios are fine ("10 of 16 hook cases", "1 of 6 shard pairs").
  - Header: a light, then `First-round:` or `Follow-up:` and 2–5 words (`🟠 First-round: stride overflow in the
    tail`); never ACCEPT or REJECT. Cells in compact point form (review book §5.2). Write <user_login>, never "the user"; no @-mentions.
  - ABI: state develop's number and "takes develop + 1 when it lands"; never a tentative lane number (lessons L-110).
  - A format or rule change applies from the next post; earlier posts are never edited to match it.
- **Reviews respect the PR's stated scope.** Every review starts from the PR's title, its body and the problem it says
  it fixes. Correctness and breaking changes stay non-negotiable.
  - Every post has two parts. **Needed to merge:** wrong results; a build, CI or ABI break the PR causes; a regression
    in code the PR touches; a test the PR adds that cannot fail; a claim in the PR body or commits that is false at head
    (the squash commits it). **Highly recommended follow-up (not blocking):** significant items that should be done
    but do not block: code and docs outside the PR's scope, pre-existing problems, real or likely issues that need
    attention, design and duplication improvements, coverage beyond what the fix needs.
  - Needed to merge lists every blocking item, each as a full row with its light (🔴/🟠/🟡). There is no cap: the
    author sees every blocker at once and can fix them in one revision.
  - The header light is the worst open Needed-to-merge item. A PR whose blockers are all fixed reads 🟢, even with
    follow-ups listed.
  - Scope discipline keeps reviews from overwhelming authors: no nits or should-fixes from outside the PR's stated
    scope that do not converge and force extra revisions.
  - Each follow-up row carries a light (🟠 or 🟡) for its importance; it never sets the header light. Trivial things
    (nits, style, wording) are left out entirely.
  - Follow-ups are listed once. The author may fix them before merging; otherwise, when the author agrees or does not
    answer, we file a <JIRA_PROJECT> follow-up ticket (Follow-up tickets, below). Later rounds do not re-raise them: they stay as one compact Follow-ups row, with ticket keys once
    filed.
  - The author's scope call stands: a non-blocking item the author declines with a reason becomes DEFERRED (author: out
    of scope) and is not raised again. Only a real break overrides that.
  - The code quality pass follows the same split: duplication or layering the PR introduces is in scope; patterns it
    merely continues, and improvements beyond the change, go to follow-up.
- **Posting:** `rk spec-post N --round R` writes the spec and posts with its premises, which re-read the head and the
  rest right before the post (`review_procedure.md`, step 4). A helper script, when one is needed, is named uniquely
  for its PR and round, never a renumbered copy of another PR's; never put a Write and the Bash that runs it in the
  same batch.
- **Machine limits:**
  - Reviews at once within the parallel reviews in effect (the scaler's setting, within its range, starting at
    `features.review_automation.parallel_reviews`); a local model drafts one review at a time unless its binding's
    `limits.max_parallel_requests` allows more. Local builds within the parallel builds in effect (the scaler's
    setting, within its range, starting at `monitor.parallel_builds`); with a local model, no build in a local
    draft's first 5 minutes.
  - The tool loop's drafts: up to `draft_parallel_max` at once, each as the scaler allows (the model runs
    elsewhere; the drafts' local work is git, grep and the saved probes).
  - Never `systemctl restart ollama`.
  - Never install packages on the bare machine.
- **GitHub access:** two logins (commands.md, GitHub logins). The read login is gh's default, so this session, the
  verification pass, the code quality pass and every job read with it, and GitHub refuses its writes; reads go through `rk` or
  `gh api` GETs. Writes go only through the guarded `rk` commands that write to GitHub (each row of the table in
  `<PACKAGE_DIR>/rkmon_lib/outward.py` whose destination is github: `post`, `notice`, `arm`, `issue-file` and the
  rest), the only ones that use the write login, and each write is recorded in the
  ledger. In the container they run as the write login's own user through the write broker; `spec-post` and
  `post --premises` run as this session, and only their post goes through it. Check the ledger first so nothing is
  done twice. Never set `GH_CONFIG_DIR`, `GH_TOKEN` or `GITHUB_TOKEN`, never
  point a command at the write login's folder, and never read or copy a gh login folder.
- **Critical items go to <user_login> immediately:** at the top of your reply, before any other work.
- **Chat is silent by default.** <user_login> reads the dashboard (its board and Needs you list) and the reviews on
  GitHub. No review, post, approval-ready, review pass or progress relays: "tell the user" in these files means it is on the
  dashboard. Speak, one line each, only for a critical item, something wrong in the review system (a tool failure, a
  pipeline bug, a stuck loop), a develop breakage that needs <user_login>, or a decision that is theirs; a heartbeat
  gets a one-line sign of life. Analysis and cost tables stay on the admin dashboard's cards and in `rk` commands,
  never in chat. Name things in plain words: an internal id (lesson, gate, item, experiment) always comes with what
  it means.
- **Develop breakages are GitHub issues** in the repository `features.merge_queue.develop_guard.issue_repository`
  names (blank: `repository.name`), never <JIRA_PROJECT> tickets: one source of truth for breakages. Several people
  keep develop green, and they force-revert where necessary.
  - **Two types, each with its label** (the develop guard's settings):
    - CI-red (`features.merge_queue.develop_guard.ci_red_label`, with its `ci_red_extra_labels`): the break makes
      CI red;
    - stability (`features.merge_queue.develop_guard.stability_label`): CI does not see it, e.g. a failing test no
      CI lane runs. Assign the breaking PR's author (`assign:` in the front matter). No Teams alert.

    When CI starts or stops seeing a break, move its issue: `rk issue-relabel N --to stability|ci_red`.
  - A breakage is anything that makes develop red or makes a develop test fail, including a failure CI does not run.
  - A break someone else finds that our gates passed: verify the claim, then harden the gate that missed it, with a test.
  - Only for breaks already on develop. A problem caught before the merge goes in the PR instead, with how to fix it:
    our review, or the auto-merge guard's comment.
  - Dedup first: `rk breakages` lists the open develop issues of both types, rocKE first. If one covers it, link the
    incident (`rk incident-note ID issue N`) and add our evidence only when it is new
    (`rk issue-comment N FILE --key K`).
  - Otherwise draft the issue in the repo's develop issue style:
    - sections: Context, Error, Cause, Why it slipped through CI, Proposed fix, Resolution criteria;
    - labels: one type's label (CI-red's with its extra labels, or stability's), plus labels like
      `project: hip-kernel-provider`, `rocke: platform`, the platform and arch labels, `code-regression`.

    It is a public repo: the post rules above apply, and line 1 of the body is
    `Automated develop-health report by <AUTOMATED_REVIEWER_NAME>`.
  - **File it without asking** once the break is proven: develop alone reproduces it, and no issue exists. Run
    `rk issue-file FILE`, then tell <user_login> in one line.
  - **Link every fix.** Search open PRs for one: the failing test, the culprit, the error, or a "fix" in the same area.
    Name each one in the issue ("Fix candidate: #N"), so it becomes a P0 develop fix.
  - **No fix planned** (CI-red issues only): alert the <TEAM_CHANNEL> at once with `rk teams-breakage N`. It sends
    an automated message that references the issue and asks for a quick response. Tell <user_login> too. A
    stability issue goes to the breaking PR's author (assigned), not the team chat.
  - Never block the monitor on a question while it runs: ask in plain text and keep the ticks going.
- **Other issues found along the way get a <JIRA_PROJECT> ticket, only if none exists:** problems in the rocKE codebase, CI or
  infrastructure that are not a breakage and not a PR's own review finding.
  - Verify the issue first, and confirm develop alone reproduces it.
  - Search <JIRA_PROJECT> for an existing ticket (the atlassian CLI's search: `commands.md`, JIRA through the atlassian
    CLI), and GitHub for an open fix. Mention a fix found in a PR in the ticket.
  - Draft it with `<RK> ticket-draft FILE` (a `---` block with title, type, priority, assignee and a one-line
    summary, then the full description): the proposed ticket waits on the admin dashboard's "Tickets waiting for
    you" card, where <user_login>'s Approve files it through the atlassian CLI and Decline drops it. Never file it
    yourself, nor run `rk ticket-file` or `rk ticket-decline`.
    Leave the type, labels and component blank: the filing takes the settings' (<JIRA_ISSUE_TYPE>; <JIRA_LABELS>;
    <JIRA_COMPONENT>). Leave it unassigned, and set the priority by impact. The filing links the default epic
    (`features.ticket_creation.epic_key`) when one is set.
  - Bugs in our own toolkit are not ticketed: their fix goes to the package's source as a pull request, never into a
    deployed copy. In the container the session writes the fix up for <user_login> (`weekly.md`'s proposals file).
  - CI runner infrastructure (TheRock runners and their checkout, container and pod failures) belongs to another team:
    no <JIRA_PROJECT> ticket. Attribute it as infra in posts and tell <user_login> in one line when it is widespread.
- **Ready for your approval:** give PRs close to merge our final review first (the plan makes those follow-ups P1).
  Tell <user_login> when no more reviews are needed and every comment is addressed, so no PR is left hanging:
  - our review is current at head and clean;
  - every review thread is resolved, and nobody else requests changes;
  - <user_login>'s approval is what is missing.
  `rk approvals` lists them in merge-plan order (develop fixes first, then wave by wave: approve what lands next first),
  then shows what is left on every other PR close to merge and who is next. Approvals at older
  heads still count (develop does not dismiss stale reviews), so never ask <user_login> to re-approve. Held PRs are the
  exception: they need <user_login>'s explicit re-approval once the author has fixed them and re-requested <user_login>'s
  review. Approving stays <user_login>'s call: we never approve.
  **One box per team, then any approver:** the dashboards and the board show "Needs platform approval" (a
  rocke-platform code-owner approval is what is missing; anyone in rocke-platform can give it), under it "Needs
  rocke-lib approval" (the same rules, for rocke-lib), then "Needs any approval" (the same entry rules, and the only
  approval missing is a general one no code-owner team gives: the develop ruleset's extra approval GitHub still asks for
  with CI green and the code-owner approvals in, or a required review requested from nobody in those teams). A PR that
  needs both teams shows in both, each naming the other team under "Also needs"; a PR in a team box is never in "Needs
  any approval".
  <user_login>'s own change request or a hold's re-approval still names <user_login>. The dashboards have no "Close to
  merge" section; `rk approvals` prints what is left on every PR close to merge.
- **Follow-ups: a post only for changes that need addressing, at most one per PR per follow-up gap.**
  - **A merge-only push is posted on only when its checks flag something.** A push is merge-only when every new
    commit since our reviewed head is a merge that brings develop in, conflict resolutions in the PR's own files
    included. Each tick marks it (the plan line says `(merge-only)`). Its review runs the draft and the free checks
    at once, with no gap: merge, build and ctest, byte identity, goldens, the changed tests, installed tests and the
    ABI scan.
    - **Clean** (every check passes, a skip only when there was nothing to check; no ABI note; no saved probe changed
      state since our reviewed head; no check run fails at the head that did not fail at our reviewed head): nothing
      is posted. The head is recorded as covered by our last review (`REVIEW_SKIP #N: merge-only … its checks are
      clean`), so the readiness and approval lists treat our review as current. The drift check (`DRIFT_FAIL`), the
      conflict tracking and CI keep watching the merged tree; a CI failure that comes later gets one CI update
      (L-209).
    - **Flagged:** `REVIEW_READY #N … Merge-only since our review, but the checks flag: …`, with what they flag.
      Verify and post as usual; an armed auto-merge can land it soon.
    - **Also carried:** a head whose only flags are CI failures that are all known infrastructure (`CI_INFRA`), or
      item flags that repeat what our review saw (threads, title; failing tests, installed tests and pull-request
      references also when our last review's open rows name them).
    - **Micro round:** new threads, a new title problem, or reviews and comments others left since our last review,
      on a head that would be carried otherwise: the verifier alone on those items, no code quality pass, every other
      row carried. A new failing test or reference stays a full flag (`review_procedure.md`, The follow-up rule).
    - A carried head keeps our verdict: a 🔴/🟠 stands, and only a post changes it.
    - `rk classify N --record` records a merge-only head as covered by hand, and closes a review of it awaiting
      verification or in flight without a post.
  - **At most one follow-up review post per PR per gap**
    (`features.review_automation.schedule.follow_up_cooldown_minutes`). While our last review of the PR (posted or
    kept unposted) is younger, a due follow-up waits, and its plan line ends with when it may start:
    `(gap: after 17:42Z)`. The driver never starts it early, and `rk review-start N` refuses it
    unless `--now`. CI updates (L-209) and notices start no gap: `rk post` marks a post at a head our review
    already covers as a CI update.
  - **Exempt from the gap:** merge-only pushes, and the PRs in `followup_gap_exempt`, by default the develop fixes and
    your priorities (the P0 reviews, Develop fixes go first, below), which keep their shorter settle
    (`settle_develop_fix_min`).
  - The board's Reviews row and the dashboard show how many follow-ups wait for the gap and how many heads were
    carried as merge-only.
- **Develop fixes go first:** a PR that fixes a break in develop climbs to the top of the review and merge queue. A
  fix is:
  - the fix PR of an open incident (`rk incident-note ID fixed_by N`);
  - an open PR that closes an open develop issue of either type;
  - a fix candidate an open develop issue names;
  - a PR linked to an open develop issue once verified (`rk fix-link I P`).

  It gets:
  - review P0, with the shorter settle (`settle_develop_fix_min`);
  - merge tier 0;
  - its merge analysis first, without waiting for review checks;
  - first place in the approval and close-to-merge lists, and a "Develop fixes" row on the board.

  When you find a fix PR nobody has linked, record it with `rk incident-note`, or link it with `rk fix-link I P`.
- **Your priorities:** a PR <user_login> prioritizes (`rk prioritize N [why]`, only on <user_login>'s instruction) gets the
  develop-fix treatment above, listed as "your priority: why". The board row becomes "Top priority". It goes into the
  earliest merge-plan wave where it conflicts with no higher priority: the other PRs it conflicts with wait for it,
  and its auto-merge is expected, not an alert. `rk unprioritize N` takes it back.
  **Requests from other teams:** other teams may request priority, and each request waits for <user_login>'s OK. The
  team dashboard's "Request PR priority" box records a request on the web host; each tick pulls it
  (`PRIORITY_REQUEST #N`) and lists it for <user_login>, who approves (the admin dashboard's Approve, or
  `rk prioritize N`) or declines (`rk priority-decline N`). On the admin dashboard the box prioritizes at once
  (their own instruction).
  The monitor never approves a request.
- **Fix sweep:** open PRs that fix a develop issue are told so and fast-tracked to the top of the review queue. Each
  tick matches the open PRs against the open rocKE develop issues and prints `FIX_CANDIDATE #P -> #I` for a PR that
  mentions the issue, changes its failing test's file with a fix title, or names its failing test and changes a file
  it names. Verify each one, then:
  - it fixes the issue: `rk fix-link I P [--evidence FILE]` comments on the issue ("Fix candidate: #P"), asks the PR's
    author to add "Fixes #I" to the description, and makes the PR a develop fix at once;
  - it does not: `rk fix-dismiss I P`.
- **Auto-merge guard:** when a PR with auto-merge on fails a gate GitHub does not enforce, post a plain-words comment
  on the PR and turn auto-merge off (`rk automerge-off N`). The gates are our 🔴/🟠 at head, a confirmed G5–G7
  failure and a hold label. This applies while the merge queue sends (`features.merge_queue.mode` live, the monitor not in standby); otherwise it is an alert to <user_login>.
  **Exception:** auto-merge <user_login> enabled on a PR in the merge plan's wave 1 (its rebasing priorities included)
  stays on when our 🟠 is its only open gate, and the tick does not flag it. If it merges with our 🟠 open, the
  follow-up tickets (MERGED_OVER_RED) cover the open items. A 🔴, a confirmed hold or a hold label still alerts.
  `rk automerge-off N` refuses auto-merge <user_login> turned on, and <user_login>'s own PRs unless
  `features.merge_queue.auto_merge_guard.on_your_pull_requests` is `same_as_others`. Turning off someone's auto-merge
  on <user_login>'s own instruction (e.g. to land other PRs first) is `rk automerge-off N --user FILE`, with their
  reason.
  **The guard never turns off auto-merge that <user_login> turned on**, on any PR and for any gate; it alerts instead (a
  🔴, a confirmed hold or a hold label), and <user_login> decides. Others' auto-merge is turned off as above while the merge queue is live.
- **Overlapping work: one owner, and a notice on every other copy.** The work is not duplicated: one PR is the
  authoritative source of the overlapping code, coordinated through comments on the PRs. The checks list the overlap
  leads, the code quality pass proposes the owner (its overlap rule), the post records it (`rk overlap-decide`), and every
  other open member gets one overlap notice (`rk notice N FILE --kind overlap --key K[,K]`; the tick prints
  `OVERLAP_NOTICE_DUE #N` until it is on): the duplicate with file:line on both sides, the owner and the resolution,
  the landing order the merge plan suggests, and the rebase that removes the copy.

## Who merges

- Auto-merge is armed only while the merge queue is live with `features.merge_queue.auto_merge.enabled`, and **only for PRs <user_login> pre-cleared** (`rk clear N`).
- Every other condition must also hold. `rk arm` re-checks each one:
  - auto-merge is on (the merge queue live, `features.merge_queue.auto_merge.enabled`);
  - the PR is the next member of the validated head batch. Its gates are green for its exact head on the current
    develop, or carried over while develop moved only by earlier members of that batch;
  - the queue is not frozen;
  - nothing else is armed;
  - on GitHub the PR is still open, not a draft, and at the validated head.
- Uncleared PRs that reach the head of the queue are proposed to <user_login>, who merges them; so is every PR
  while auto-merge is off (`features.merge_queue.auto_merge.enabled` false). Only PRs the merge plan lets
  land now (wave 1, merging clean with develop, with nothing ahead conflicting) are batched and proposed
  (`merge_procedure.md`, Merge plan).
- `clear` and `unclear` run only on <user_login>'s explicit instruction.
- Queue notices on PRs go out only while the merge queue is live (`features.merge_queue.mode`): <user_login> sets
  it live only once they have told the team.

## Follow-up tickets (open PRs' follow-ups; merged should-fix items: tick step 4, `!! MERGED_OVER_RED` / `!! HELD_MERGED`)

Follow-up tickets come from two places: the follow-ups our posts list on open PRs, and the should-fix items still open
when a PR merges.

**Follow-ups on open PRs:** each F item of our posts (a significant 🟠/🟡 row of the "Highly recommended follow-up
(not blocking)" part; trivial items are never listed, so never ticketed), tracked in
`<TOOLKIT_DIR>/state/followups/tickets_<N>.md` (`review_procedure.md`, step 5):
- the author agreed, or did not answer by our next round or the merge: file a follow-up ticket;
- the author declined with a reason: DEFERRED (author: out of scope), no ticket;
- fixed in the PR: no ticket.

Write each key on the item's line, so later rounds show it in the Follow-ups row and never raise the item again.

**Merged should-fix items:** tickets only for non-trivial issues that warrant a follow-up, those leading to test
failures, crashes, incorrect behaviour or poor development practices. For each open 🟠/🔴 still present on develop and
not already fixed or ticketed, and each F item of ours not ticketed yet:
- a failing test or red CI is a develop breakage: file the develop issue, as above;
- any other non-trivial item gets a follow-up ticket. Assign it to the PR's author. Link it to the merged PR
  and to our review comment where the ask came from, and say plainly that it is a follow-up from that merged PR;
- nits (🟡) get no ticket; neither does an F item the author declined with a reason.

**Scan every reviewer's feedback, not only our rows.** On every merge of an in-scope PR, read all review threads and PR
comments, resolved or not:
- a thread or comment answered with a deferral ("follow up", "follow-up", "later", "separate/next/future PR",
  "TODO", "out of scope", a ticket promise) whose fix is not on develop is an open item, even when the thread was
  resolved;
- a reviewer's request that got no response at all is an open item;
- each open item is judged like ours: a breakage gets the develop issue, a non-trivial item a follow-up ticket
  (crediting the reviewer and linking their comment), a nit nothing. A deferral already ticketed needs no new ticket.

**Every follow-up ticket, either source.** Every step goes through the atlassian CLI (`commands.md`, JIRA through the
atlassian CLI), never an MCP server. Dedup first: search <JIRA_PROJECT> for an existing ticket, and use it when one covers
the item. Follow-up tickets go to <JIRA_PROJECT>: component <JIRA_COMPONENT>; labels <JIRA_LABELS>
follow-up; linked to the default epic (`features.ticket_creation.epic_key`) when one is set: create the
ticket without its description, link it, then set its description (`commands.md`, JIRA through the atlassian
CLI); priority by impact. Assign it to the PR's author's JIRA account: `<RK> jira-account list` shows the account
recorded for each GitHub login; for a login with none, look the account up with the CLI's `jira user get` and record
it with `<RK> jira-account add LOGIN ACCOUNT`. Link the PR and our review comment where the
item was listed. File without asking once the item is confirmed (on develop for a merged PR, at head for an open one),
then tell <user_login> in one line.
