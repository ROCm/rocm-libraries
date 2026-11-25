<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Review procedure (review automation)

The pipeline runs locally with no outward action until you post:
1. While review automation is enabled (`features.review_automation.enabled`), every `rk tick` advances it, and
   so does the review driver every minute; in prototype, and in standby, `rk post` keeps each verified review
   unposted, as an unsent item too, instead of posting it. Otherwise run `rk review-advance` or `rk review-start N`.
   `rk review-start N` is refused while a review is in flight: queue it with `rk review-queue N`.
2. It picks the review queue's entries first, in order (`rk review-queue`), then P2 follow-ups on near-ready PRs,
   then P3 first rounds, then P4 other follow-ups, smallest first.
3. It skips unsettled heads (younger than `features.review_automation.schedule.push_cooldown_minutes`; after 2+
   pushes in 90 min, until 45 min after the last; a develop fix after 10 min), follow-ups inside the gap after our
   last review on the PR, posted or kept (The follow-up rule, below), PRs larger than
   `features.review_automation.large_prs.in_depth_review_over_lines`, and PRs in `review_skip`. `rk review-start N`
   applies the same settle and gap, and refuses an unsettled head or a follow-up inside the gap unless `--now`. A
   queued entry waits only for an open PR, its own review in flight or awaiting verification, `review_skip`, the
   approval of an in-depth review (Split review) and the settle; without `--again` also for our post at its head and
   the gap. `rk review-start N --again` skips the gap.
4. It runs as many reviews at once as the parallel reviews in effect (the scaler's setting, within its range, starting
   at `features.review_automation.parallel_reviews`; fewer when the reviewer endpoint allows fewer: `rk status` shows
   the range and says why), each PR in its own worktrees and workspace. Their checks share the parallel builds in effect
   (the scaler's setting, within its range, starting at `monitor.parallel_builds`, <CHECKS_LIMIT> asked; `rk status`
   shows the range), which the parallel reviews never change: a draft whose checks find every build busy waits. A
   drafter binding's pool drafts the next picks beside it, and the shadow drafter drafts each live review's PR beside it
   (Stages, below).

## Stages

| Stage | What happens | Tick output |
|---|---|---|
| Mark (every tick, before the plan) | Each PR whose head moved since our review is checked once per head for a **merge-only** push: every new commit is a merge that brings develop in, conflict resolutions in the PR's own files included. Its plan item says so, and no gap keeps it waiting | `PLAN … P1 FOLLOW_UP #N (merge-only)` |
| Gap (other follow-ups) | Waits while our last review on the PR, posted or kept, is under `features.review_automation.schedule.follow_up_cooldown_minutes` (60) minutes old: one follow-up review per PR every hour. CI updates and notices start no gap. Merge-only pushes and the PRs in `followup_gap_exempt` (by default develop fixes and your priorities) never wait | `PLAN … P4 FOLLOW_UP #N (gap: after 17:42Z)` |
| Classify (follow-ups only) | Compares the PR's own added and removed lines at our reviewed head and now. **Rebase-only**, or a rebase plus a golden re-record after a 🟢/🟡 review: records the new head as equivalent, posts nothing and moves to the next item. **Merge-only** or **substantive:** continues | `REVIEW_SKIP #N` / `#N is merge-only …` / `#N changed substantively` |
| Draft | The draft stage config names (`rk drafts`; `commands.md`, The drafter), as a detached job: the drafter binding's tool loop (`tools/draft_run.py`: it stages the PR, drafts, then runs `rk evidence-check`), or the local review loop (`review_agent_run.py N`, one at a time, about 10 min). With a drafter binding, the review adopts its PR's pool draft at the same head, else starts a draft once the scaler allows it. With no drafter (`commands.md`, The drafter) the job only stages the PR (`tools/draft_run.py N --stage-only`: its data and trees), and there is no pool | `REVIEW_START #N` / `REVIEW_WAIT #N: the scaler holds its draft: …` / `REVIEW_START #N (…): staging started` |
| Pool (drafter binding) | Beside the review in flight, each advance starts one more draft for the next picks, up to `draft_parallel_max` (6) at once, while the scaler allows it: the binding's last 5 minutes used under 80% of its token and request budget, no HTTP 429 came, and the drafts running are under the ceiling (host and queue load are recorded only). Every answer is a line of `state/scaler.jsonl`. A merge-only or covered follow-up gets no pool draft, and a pool draft whose PR closed, moved on or is covered by our review is dropped | `DRAFT_START #N` / `DRAFT_WAIT #N: …` / `DRAFT_DROP #N: …` |
| Shadow (`roles.shadow_drafter`) | Beside each live review (the review in flight, unless a merge-only head, and each review awaiting verification), the shadow drafter's binding drafts the same PR at the live review's head, in phases (`commands.md`, The drafter), in a folder in `workers/` named for the binding and the PR (`workers/shadow_glm/pr101/`), through the same ceiling, scaler and limiter. The live review never adopts it and never waits for it | `SHADOW_START #N` / `SHADOW_WAIT #N: …` / `SHADOW_READY #N: …` / `SHADOW_DONE #N: …` / `SHADOW_DROP #N: …` |
| Checks | As one of the parallel builds in effect (the scaler's setting, within its range, starting at `monitor.parallel_builds`; `rk status` shows the range): once the draft job has staged the PR (drafter binding), or, with a local reviewer, at least 5 minutes after its worker started (never in a worker's first 5 min). The generated script covers what the PR touches; see below. The result is `workers/prN/checks/checks.md` | `REVIEW #N: … checks started` |
| Ready | Both jobs have ended. A merge-only head whose checks are clean is recorded as covered and nothing is posted, and so is one whose only flag is CI failures that are all known infrastructure (`CI_INFRA`), or, when the checks wrote their thread and title item lists, threads and title flags our review saw, or test, installed and pull-request-reference flags that repeat what it saw; new threads, a new title problem, or reviews and comments others left since our last review then make a micro round (`micro:` and the items, the follow-up rule below); one they flag otherwise goes to verification with the reason. A drafter binding's REVIEW_READY names the draft's model, tokens, requests, minutes and evidence check, and `evidence_check.md` to read first. The line names the live draft's assessment tag (`qwen` for the local review loop, else the binding's name) and the shadow draft: its folder and tag when it is done, else "not done", plus its `prior_leads.md` (prior items a script settled, with evidence). With no drafter it reads "no drafter: verify without a draft … Verify alone": line 1 and the footer name the verifier, and no first review pass. When our last review is at this head already, the line ends on a hint: `post it with --again`, or while review posting is off `keep it with --again` | `REVIEW_READY #N (…) Inputs: … Shadow <shadow tag>: …` / `REVIEW_SKIP #N: merge-only … its checks are clean` |
| Fail | The draft job ended without making worktrees: the PR goes on `review_skip`, unless its staging met another fetch into the shared clone (the shallow file changed): not the PR's fault, so it is not skipped, and its review starts again (from the front of the review queue when the queue started it). A drafter binding's job that leaves no draft is retried once, then goes to verification without one; the local review loop's is retried once, then the PR is skipped | `REVIEW_FAIL #N` / `REVIEW_READY #N (…): no draft (the drafter failed twice): verify without it` |

**The drafter's smoke gate.** A drafter binding's first 10 drafts are scored on three counts:
- completion: at least 9 of 10 end with a draft within 45 minutes;
- evidence: every 🔴/🟠 claim has a run or a quote (`evidence_check.md`: no 🔴/🟠 claim MISSING);
- tokens: the median draft uses at most 0.5M tokens, and no request is over 60K (`rk drafts` per draft;
  `<WORKING_DIR>/llm_limits/tokens.jsonl` per request).

If the gate fails, `features.review_automation.passes.first_review_pass.model` goes back to the drafter it named
before (`commands.md`, The drafter).

**The follow-up rule** (the full text is in `rules.md`):
- A post only for changes that need addressing. A merge-only push runs its draft and free checks at once; when they
  are clean (every check passes, a skip only when there was nothing to check; no ABI note; no saved probe changed state
  since our reviewed head; no check run fails at the head that did not fail at our reviewed head), nothing is posted
  and the head is recorded as covered by our review. The readiness and approval lists then treat our review as
  current, and the drift check (`DRIFT_FAIL`), the conflict tracking and CI keep watching the merged tree.
  A merge-only head whose only flag is CI failing at the head and not at our reviewed head is carried the same way
  when every failing job is known infrastructure: its log has a signature in config `ci_infra_signatures`, or its
  first error line is the same job's at develop's latest SHA, or on at least 2 other open PRs today. The tick then
  prints `CI_INFRA #N at …: merge-only, and its only flag is known CI infrastructure: …` (each job, with why it is
  known), which needs nothing; a log or check-run list it cannot read leaves the job unknown. Anything else flagged
  reads `REVIEW_READY #N … Merge-only since our review, but the checks flag: …` and the flags: verify and post as
  usual.
  The `threads` and `title` flags read PR-level state, which a develop merge does not change, so on a merge-only head
  they count only for what our review did not see. Checks that write their items (`checks/peer_threads.json`,
  `checks/peer_title.json`) record them for their head when the review finishes. When a merge-only head's
  only flags are these (beside CI failures that are all known infrastructure): every flagged thread (by its first
  comment's link) and the flagged title were flagged at a head our review covers, so the head is carried like a clean
  one (`REVIEW_SKIP … its threads flag repeats what our review saw …`); a new thread or a new title problem makes a
  **micro round**, `REVIEW_READY #N … Merge-only since our review; micro: …. The verifier alone (no code quality pass)
  …` with the items: brief the verifier on those items alone (On REVIEW_READY, step 2). Reviews, review comments and
  comments others (not bots) left since our last review make a micro round too, on a head that would be carried
  otherwise: a head with something to answer is never carried silently.
  The `tests`, `installed` and `prnumbers` flags count the same way for the items they name: the failing test node
  ids (`checks/tests/tests_pr_failed.txt` and `checks/tests/tests_develop_unrun.txt`; the installed tests'
  `checks/installed/results.json`) and the pull-request references (`checks/peer_prnumbers.json`, matched by file and
  text, so a reference on a moved line still matches). Such a flag is settled when every item it names was flagged at
  a head our review covers or appears in our last review's open rows (its findings table, 🔴, 🟠 and 🟡, and the
  status rows still OPEN), read from `pr/ours.md` when staged at that head, else from GitHub. A merge-only head whose
  only flags are settled is carried
  (`REVIEW_SKIP … its tests, installed and prnumbers flags repeat what our review saw …`). A new failing test or a new
  reference makes no micro round: its flag stays a full one, and the head goes to verification as above.
  Checks that wrote no item list leave the flag a full one, as are the item flags beside any other flag (a probe that
  changed state, a CI failure that is not known infrastructure).
  A carried head keeps our verdict; only a post changes it. The clash scan, the mutation sweep, the CTest tier check,
  the moved-reference check, the docs-against-code check, the import probe, the tests that import the changed modules
  and the peer-review leads are leads: they flag no merge-only head. The ABI scan and the peer-review flags (unanswered
  threads, tracker IDs, env docs, test lists, C/C++ getenv, the title, split tautologies, sibling test imports,
  workload names, pull-request references, timings in comments, back-edges, kept ABI strings, over-written arrays,
  NaN-blind tests, optional imports, repeated headings) are check flags like the others.
- At most one follow-up review per PR every hour (`features.review_automation.schedule.follow_up_cooldown_minutes`,
  60; Gap, above). The driver never starts a follow-up early. When the gap ends, the next advance starts it (a tick's,
  at the latest). `rk post` notes a post at a head our review already covers as a CI update (`ci-update` in the
  ledger), and a CI update starts no gap.
- The board's Reviews row and the dashboard show how many follow-ups wait for the gap and how many heads were carried
  as merge-only (last 24 h).

**What the check script covers:**
- the merge state: the head merged with its base branch, the PR's `base.ref` (develop, or the feature branch it
  targets: `worker/mkwt.py` merges it when the driver passes `RKMON_BASE`). For a feature branch the section says so
  (`base branch: <branch> (not develop) …`), and the merge base, the tests at head, the mutation sweep, the CTest
  tiers, the moved references, docs against code, the import probe, "Since our last review" and the ABI scan compare
  with that branch; a line `!! the mkwt log names no base=<branch>` says mkwt.py merged develop instead. The section
  names keep "develop";
- native build plus ctest. The build of the merged tree (`wt<N>m` in the build folder of the checks' slot: core with
  the native tests, and the rocke_engine pybind module) is the review's shared build when the section names it (its
  `shared build:` line names the build, the tree and its commit; also in `checks/shared_build`): the checks keep their
  slot for the review, and the verifier and the code quality reviewer reuse the build instead of building the head
  themselves, running its tests as PR code in slots of their own (their team files say how);
- byte identity for the touched families at every emission flavor of the tree (llvm20/22/23/24 on today's develop),
  plus every family at every flavor when engine core changed;
- the golden gate as CI's rocke_golden_static (every flavor; load errors and drift by family, compared with develop);
- the differential-golden CTest commands, with NEW counts per flavor;
- the PR's own changed tests on CPU at llvm22, plus companion contract tests (CMake registration, arch data);
- the tests the PR does not change that import a Python module it changes (`tools/importer_tests.py`: the modules
  under `platform/python/` and `library/`, tests and conftests aside): on the merge, each failure retried on the base,
  at most 12 files (those importing the most changed modules first, then the smallest) within 10 minutes. The section
  lists the files run and names the files over the cap (`NOT RUN (over the cap, k): …`); a run cut by the time box
  reads `NOT RUN`. A lead;
- clang 24 compile of the touched families with rocKE's own codegen options;
- the saved probes of our open findings, re-run on the merge (head + the develop mkwt merged) and, indented, on the
  head alone; each line gives the verdict, the exit code, the tree's commit and the seconds
  (`N1_x: REPRODUCES rc=1 (tree 9b1e04c2aa, 3 s): …`). The unindented lines are the ones a merge-only head's
  probes are compared on. Re-run a probe by hand only on ERROR or when the head moved;
- CI at head: the check-run counts, then per failing job (latest attempt) its first error line, the failing tests its
  log names, develop's latest completed run of the same job (and whether it fails with the same first error), and a
  known-infrastructure label (a `ci_infra_signatures` signature, or develop failing the same); a re-run attempt is
  marked (`attempt 2 of 2: a re-run`); then the failing tests by name (`failing tests by name (k, …)`), each with the
  failing jobs (shards) whose logs name it, the most first, so a failing test the PR does not change shows as plainly
  as its own; jobs re-run and passing are listed, failures hidden in earlier attempts are listed (`ci_attempts`), and
  TheRock's test-job log lines follow;
- for a follow-up, "Since our last review": `git range-diff` from our reviewed head to the head over their own merge
  bases (each commit the same patch, changed, new or dropped; merge commits counted) and each PR file whose own patch
  changed, with its +/- lines then and now. Its status reads `same patches … (a rebase only | a merge only)` or
  `changed: …`, so a force-pushed re-cut reads at a glance;
- threads since our last review;
- changed tests in the installed tree (CI's `rocke_pytest` layout, L-526): N run of M, each failure's first error
  line, labelled when develop's installed run fails it too; a skip says why;
- both changed-test sections (the source tree and the installed tree) run with the tools CI's test lanes have: cmake
  and ctest on PATH (`paths.cmake`, else the review venv's), PyYAML importable (`paths.pyyaml`, linked in alone), and
  the worktrees check out `shared/ctest` (the CTest category parser the provider-category tests read). Each section
  then:
  - lists every skipped test with its reason under `SKIPPED k` (grouped by reason over 20, every one in the
    section's `skipped` file);
  - prints a `NOT RUN` line for the tests a missing cmake or PyYAML kept from running
    (`NOT RUN: cmake missing (3 tests), PyYAML missing (1 test)`);
  - prints the exact command that re-runs each test file (`rk pr-run --env ENV -C ROOT -- python -B -m pytest … FILE`
    for the source tree: in a PR runner's slot, in that run's environment; add `-k` or a test id after the file);
  - ends on its result line: `result: PASS …` (with `(k skipped: …)` when anything was skipped), `result: NOT RUN: …`
    when a missing tool kept a test from running, or `result: FAIL …` (which names the missing tools too). A result
    that a missing tool made `NOT RUN` is a check flag for a merge-only head, so that head goes to verification.
    Trust a section that names the head and passed; re-run a test file only for a finding the section does not
    settle (a single case, another flavor, GPU);
- overlapping work in other open PRs (`rk overlaps N --section`): the same new name or the same logic in another open
  PR, with the land order and any recorded owner, then copies of a develop pattern: leads for the code quality pass;
- clashes with other open PRs (`rk clash N`): both land orders with every other open PR on develop (conflicting files,
  modify/delete pairs) and the silent clashes that merge clean: a name one PR removes that the other's new lines use, a
  file one moves or deletes that the other includes, imports or names, and a C/C++ function whose parameter list one
  PR changes while the other's new lines call it with an argument count the new list does not take (`silent (arity, a
  lead from a parse of the signatures: build the merge)`: it compiles on neither land order); leads where one changes a
  function the other's new tests call; who ports what. A PR on a feature branch is read against that branch when the
  mirror has it. Heads or blobs not in the mirror read NOT RUN;
- a mutation sweep of the PR's checks (`rk mutants N`): each guard, raise, refusal and assert its Python lines add or
  change (a guard's error exit includes a validator's `return False, "…"`), disabled once in a scratch copy of
  the head and run against the PR's tests and the tests that import the module. A follow-up takes only the checks its
  push since our last review adds or changes; a merge-only round, or a push with no new check, reads "no new checks
  since" and our reviewed head, and a push whose new checks are C/C++ only reads "no new Python checks since" and the
  head, and lists them. At most 25 mutants in 10 minutes; a survivor is a check no test catches; C/C++ checks
  (`throw`, `assert(`, and under an `if` a `return` of an error value or a refusal macro such as `ROCKE_SPEC_FAIL(…)`)
  are listed, not run. Every round's results go to the PR's sweep history
  (`<WORKING_DIR>/cache/mutants/pr<N>.jsonl`), and a later round (a merge-only one too) lists the earlier rounds'
  survivors, timeouts and NOT RUN items still at its head, each with the head of the sweep that found it; they are
  carried, not re-run, so mutate one by hand only to confirm a finding;
- CTest labels and tiers (`rk ctest-labels N`): the merge base and the head, each configured (not built) as
  hip-kernel-provider configures rocKE, with every CTest entry's labels in the build and the installed tree, then a
  test map: each test file the PR adds or changes -> the head's entries that run it (listed, under a listed
  directory, or through the in-tree runner) -> their tiers -> the PR-CI lanes that run them (config `ci_test_tiers`;
  by default TheRock's hipkernelprovider shards, installed tree, `-L standard -LE ex_gpu`). Leads: an entry that loses
  a quick/standard/comprehensive/full tier in one tree (with its `add_test()` move and what the other tree keeps),
  leaves a tree, or comes with no tier; a changed test file no PR-CI lane runs, with why;
- references to moved or deleted files (`rk moved-refs N`): the references the head still makes to an old path or
  module name that resolve to nothing (doc links and backticked paths, includes, imports, CMake and YAML paths), by
  file:line, changelogs aside; then the broken links in the docs the PR changes;
- docs against code (`rk doc-refs N`): the doc lines (.md and .rst in the rocKE tree, changelogs aside) that name a
  function, class, `ROCKE_*` env var or knob, command-line flag or arch the PR removes or renames and its head's code
  has nowhere; and the env vars its code newly reads, the IR ops and the IR dtypes it adds that
  `platform/dsl_docs/reference/` (env_flags.md, op_vocabulary.md, dtype_support.md) does not name. Leads, not
  verdicts;
- an engine import probe (`rk import-probe N`): each Python module the PR changes under `library/` and
  `platform/python/`, imported at the head and the merge base with `rocke_engine` as a stand-in and no other engine
  built, and its cheapest `backend` entry point called with `backend="python"`. Leads: an import or call that works
  at the base and needs an unbuilt engine at the head;
- the ABI scan (a check flag; G9 also waits for `rk abi-ok N` on it): `ROCKE_ABI_VERSION`/`ROCKE_RECIPE_ABI` against
  develop, and a NOTE for each of: a struct layout change in a public or generated (`instance_*`/`helper_*`) header
  of `include/rocke/` with no binary bump (a field added, removed, reordered or retyped; a renamed field is no change);
  an exported `rocke_*` function those headers remove or whose signature they change with no binary bump (one moved to
  another header or kept in a source is still exported); abi.h and `portable_ir/src/abi.py` (`BINARY_ABI`,
  `RECIPE_ABI`) at different numbers; a wire bump with no construct registered in abi.py. Its status line reads
  `ABI fail 2 notes: …` or `ABI pass …`;
- the peer-review checks (`rk peer-checks N`; `rkmon_lib/peer_checks.py`), each its own section ending in a status
  line. Flags:
  - unanswered review threads: the PR's live threads (unresolved, not outdated) that a reviewer other than the PR's
    owners started (Copilot counts, other bots do not) and no owner (its author or a commit author) answered, by
    reviewer, with where and the thread's link. Read from GitHub at the time of the checks: track each in the
    Prior-item rows;
  - tracker IDs of any project in added lines (`<JIRA_PROJECT>-NNN` or another `KEY-NNN`), anywhere in the PR; the
    prefixes of config `tracker_allow` (its row in `<PACKAGE_DIR>/docs/internal_settings.md`) and data files do not
    count. `rule_scan`'s tracker list covers four prefixes only;
  - `ROCKE_*` env vars that non-test code newly reads (Python or C/C++) and `platform/dsl_docs/reference/env_flags.md`
    at the head does not name (the PR's own edit counts, a documented `ROCKE_X_*` family covers its members); a name
    the merge base's code already reads is develop's, not the PR's. Test-only reads are `rule_scan`'s;
  - new `library/tests/**/test_*.py` files that no CMakeLists.txt, .cmake or YAML file under the provider names, by
    file name or by a directory above them (CMake collects some directories whole): no CI lane runs them. The CTest
    labels section says which lane runs a listed one;
  - `getenv` calls in added C/C++ lines of the rocKE tree outside tests;
  - a PR title that is not `type(scope): subject` (platform/AGENTS.md: TheRock PR bot policy; length is not checked);
  - a split tautology: a string literal that contains the separator tested against `split()`'s pieces
    (`"_dg" not in name.split("_")` is always true, so the assertion cannot fail);
  - a test's bare import of a sibling test module (`from test_x import …`, `test_x.py` beside it) when neither the file
    nor a `conftest.py` beside it puts the directory on `sys.path`: pytest's `--import-mode=importlib` does not, so
    the import resolves only when an earlier module happened to add it (`library/tests/conftest.py` adds
    `library/tests`);
  - workload sizes (`M32N48K128`, `M{M}N{N}K{K}`) in an added kernel name, in code or a golden file: name by operation;
  - pull-request references (`#NNNN`, `#NNNN`, `PR #NNNN`, `rocm-libraries#123`, `/pull/N`) in the added lines of any
    file but data files, code and docs alike (case studies, runbooks): <user_login>'s rule, the repo states generalized
    facts, never a ledger. platform/AGENTS.md allows external GitHub links; this rule does not. Each one the PR adds
    is a 🟡 Needed-to-merge row, as is a dated narrative of a particular run (which the verifier finds by reading);
    the PR body may name related PRs;
  - a measured timing or speedup in an added source comment or docstring, tests aside (`~18 s for …`, `3.5% faster`,
    `N GB/s`, `well under a second`, `a few hundred microseconds`; platform/AGENTS.md: source comments stay
    qualitative). A parameter (a timeout, a default, a gap, a bound) is not one;
  - a platform file newly reaching into `library/`: an import of a library package or module, a `sys.path` edit, a
    `../library/…` path (`os.path.join(…, "..", "library", …)` too) at a target the file did not reach at the
    merge base; CMake's install-layout lists (`tests/library/tests/…`) are not edges;
  - a dispatch family's kernarg or signature code lines (`signature=`, `kernargs`, `arg_types`, `KernelArg(`; comments
    and docstrings aside) changed in a file whose kernel ABI string (`abi_version="…/vN"`, `*_ABI*`) no added line
    redefines;
  - a fixed C array written through a running index (`names[n++] = …`) at more sites than it has slots, more so than
    at the merge base (a moved file is compared with its old path), with no bound check of the index in its block;
  - a NaN-blind failure test: `if err >= tol:` or `(err > tol).sum()` with no `isnan`/`isfinite`/`allclose` nearby;
  - a test module that imports an optional package (`ml_dtypes`, `scipy`, `triton`, …) at module level with no
    `pytest.importorskip` of it: collection fails in the CTest lanes that do not install it;
  - a numbered Markdown heading the PR adds whose number repeats at its level (fenced code aside);

  and leads, which never flag: a request field forced to `bool(...)` in non-test library code (Rule 4: refuse, never
  normalize; read whether the field was validated first), torch device queries (`torch.cuda.*`, `torch.version.hip`)
  in non-test library code, and a CMake `file(GLOB …)` without CONFIGURE_DEPENDS (one evaluated at install time may
  mean it); a shared lowerer (`core/lower_*.py`, `cpp/core/lower_*/`) changed with no parity `*_emit.{py,c}` pair in the
  PR (peers ask for a pair on a few of these: whether the change needs one is the verifier's call); a tuple-unpack
  target in an added line that its function never reads (minor);
- `rule_scan` on the diff and the PR body, and an armed auto-merge's stored message vs the live one.

## On REVIEW_READY: verify, check, post

**Review offload** (`offload.mode`, <OFFLOAD_MODE> here; `offload.md`). A line that names the review's kind, then
`audit` (`REVIEW_READY #101 (follow-up, audit)`), is the reviewer model's whole review: audit it as `offload.md`
says, in place of steps 1–3 below. A line that names the kind, then `deep:` and the reasons
(`REVIEW_READY #101 (first-round, deep: …)`), runs steps 1–5 as they stand, with its files as extra inputs. With
`offload.mode` shadow, the line is the usual one: verify as below; the shadow run posts nothing (`rk shadow-report N`
after the post).

1. **Prepare, then read the inputs.** A helper script for PR N takes N as an argument or is written fresh; never make
   one by renumbering another PR's script (the spend scan books such calls to the template's PR). Run
   `rk review-prep N` (add `--round fr|fr<k>|fu<k>` when its default is wrong, and always when you prepare again
   during a round: the verifier writes `notes.md` early, and a follow-up directory with notes counts as a used round).
   It fills the review directory in `paths.tmp`, named for the PR and the round (`v101fu2`; `paths.tmp` is
   `<WORKING_DIR>/tmp`, or the RAM disk's while it is on), which is also the verifier's scratch directory:
   `facts.txt` (what `rk facts N` prints), `classify.txt` (`rk classify N`), `overlap_section.md` (the checks' overlap
   section) and `files.txt` (the PR's files, merge base to head). It prints the summary to read first: our last
   reviews, the probes and notes, the facts' key sections, the classification, the checks' merge and re-probe
   sections and overlap leads, then the head, merge base and commit count.

   **In one call:** `rk dispatch-pack N [--round R]` runs `rk review-prep N` and prints, compactly, what the briefs
   need: the head and state, our last review's flagged rows and findings, the unresolved threads and <user_login>'s
   comments (from the facts); the delta since our last review (the classification, the PR's own changed files, its
   own non-merge commits and the merge count); the checks' flags, leads and shared build; CI's FAIL and RUN lines
   (`rk ci-fails`); the overlap owner decisions that name N; a micro round's items; and last its `lane:` line
   (`needed` or `exempt`, with the reason): the rule for the code quality pass below, applied to the delta. Use it
   instead of `rk review-prep`, `rk ci-fails` and reading the delta by hand (exit 1: an input was not read; the pack
   says which). Then read (paths in the REVIEW_READY line; with no drafter there is no draft, so the first three are
   absent):
   - for a drafter binding's draft, `evidence_check.md` first: each claim PASS, FAIL, NOT RE-RUN or MISSING;
   - the shadow draft's `prior_leads.md`, when the line names it: leads only, for the verifier's prior rows;
   - `review_agent.md`;
   - `checks/checks.md`;
   - `pr/meta.md`;
   - for follow-ups, our last review (`ours.url`) and `classify.txt`.
2. **Dispatch the rocKE Review Verifier** (`.claude/team_members/rocke_review_verifier.md`)
   as a subagent, in the background, so several verifications can run at once. Use the dedicated agent type
   `rocke-review-verifier` with no `model` option (its file in `.claude/agents/` runs `<AUDIT_MODEL>`, which a
   `model` option would override; no workspace CLAUDE.md or memory, a fixed tool list, its own 1M compaction window);
   fall back to `general-purpose` with model `<AUDIT_ALIAS>` only when the type is not listed. Dispatch it with
   `effort: "<VERIFY_EFFORT>"` (`features.review_automation.passes.verification_pass.effort`), and the code quality
   reviewer with `effort: "<QUALITY_EFFORT>"` (`features.review_automation.passes.code_quality_pass.effort`);
   `rk roles` shows both. A pass at the model's default effort is dispatched with no effort; never leave another
   agent's effort unset: it would inherit the session's.
   - **Exception to "paste full content"** (continuous review): give the verifier the input file paths
     to read itself, plus the metadata, the classification and what you spotted outside the draft: the draft's
     findings stay out of the brief, so the blind pass stays blind. Pasting 10–20k tokens per review
     would exhaust the orchestrator's context over a backlog. You still read the verifier's final post and check it
     (step 3).
   - It needs:
   - when there is a draft: the draft (`review_agent.md`), `agent_log.txt` (the drafter's tool calls), for the local
     review loop's draft `checks/lint.md` (`rk draft-lint`'s flags, named by REVIEW_READY) and, for a drafter
     binding's draft, `evidence.jsonl` and `evidence_check.md`: to open after its blind pass;
   - `checks.md`;
   - the PR metadata;
   - the classification (`classify.txt`);
   - in a round without the code quality pass, the checks' overlap section (`overlap_section.md`): the verifier
     checks the cross-PR items itself (another open PR's copy, a land-order renumbering, a filter two PRs must
     reconcile);
   - `facts.txt` (the output of `rk facts N`): the brief names its path, and the verifier prints it whole as its
     first read (lessons L-201);
   - for a follow-up, our follow-ups so far: `<TOOLKIT_DIR>/state/followups/tickets_<N>.md` (step 5), when it
     exists;
   - the exact line 1 the review was staged with, `pr/meta.md`'s `- Line 1:` (made from the passes at the review
     start: the drafter's model and effort, or with no drafter the verifier's);
   - the `rk` CLI's values for the commands card: `<WORKER_USER>` and `<WORKING_DIR>` (filled in at install);
   - its scratch worktrees' prefix under the review directory: `wt_v*` for the verifier, `wt_q*` for the code quality
     reviewer, each made with `rk pr-tree DIR` before `git worktree add` (the session's guard refuses it otherwise;
     lessons L-591);
   - the live drafter's model and where it runs, for the footer (the local review loop's draft: <WORKER_MODEL>,
     local; a drafter binding's: `rk drafts`' drafter line);
   - the live draft's assessment tag (`qwen`), or that the line says "no draft";
   - with no drafter (REVIEW_READY "no drafter … Verify alone"): that there is no drafter, so it reviews alone and
     writes `qwen none` (and `<shadow tag> none` when no shadow draft is named) in block 2, and its footer names the
     verifier with no "first pass by" (`rk post --premises` stops one that has it);
   - the shadow draft (the folder the line names, such as `workers/shadow_glm/pr101/`: `review_agent.md`,
     `evidence_check.md`, `prior_settled.md`) and its tag when the line names it done, else that it is not done yet,
     and the `prior_leads.md` path as leads. A shadow draft done later prints `SHADOW_READY #N`: send its path to the
     verifier while it runs;
   - the shared build, when the checks name one (the pack's `shared build:` line): its path, for reuse;
   - for a **micro round** (REVIEW_READY's `micro:` and its items): that it is one, and the items. The head is
     develop merges only and its checks are clean but for those new threads or title problem, so the verifier covers
     the items alone and carries every other row of our last review, with no code quality pass (its team file says
     how).

   The format is the team file's Output Format (the book's §5 is out of date for it): don't ask for the book. The
   long-command protocol is in both team files; the brief need not repeat it. So is the GitHub rule: the agents' `gh`
   is the read login, and they never set `GH_CONFIG_DIR` or touch a login folder. A brief never names a login folder,
   a token or a credentials path.

   **Blind first, then the drafts** (the team file's What to Check §0). The verifier writes its own findings in
   `notes.md` before it opens any draft (`review_agent.md`, `agent_log.txt`, `evidence.jsonl`, `evidence_check.md`).
   Then it reconciles the live draft, and writes one line per judgement in block 2, starting with the live draft's
   tag: `qwen G2 match N1`, `qwen G3 new N4`, `qwen G5 false` and the reason, `qwen G6 dropped` and the reason,
   `qwen missed N2 🟠`, `qwen prior N3 right` or `wrong` (G2 is the draft's N2; a prior row is named by its first
   cell, or by its id in the draft, `P1`). Once its post is written, it opens the shadow draft and writes the same
   lines with the shadow's tag (`<shadow tag> ...`), or `<shadow tag> none` when the shadow draft is not done by its
   hand-back. A finding only a draft found goes into the post once the verifier has confirmed it itself, as a normal
   N row in its own words. With "no draft", or with no drafter, it reviews without one and writes `qwen none` (and
   `<shadow tag> none` when no shadow draft came). `rk post --premises` records the lines in
   `state/drafter_assessment.jsonl` (the drafters' card on the dashboards and in `rk drafts`, `GLM drafter vs Opus`).

   It returns the post plus verification notes.

   **The code quality pass** (lessons L-227).
   The optional first passes (when `features.review_automation.passes.ci_triage_pass.model` or
   `features.review_automation.passes.first_code_quality_pass.model` is not `default`): when
   `rk review-prep` copied `ci_triage.md` or `quality_leads.md`, name the file in the verifier's brief (CI) or
   the code quality reviewer's (quality) as leads only: each lead's quotes were found at their lines, the agent
   checks each lead itself before it uses one and still does its whole task; a review-prep line saying no first
   pass means the agent works without it.
   When the REVIEW_READY line says `Code quality pass (…): …/quality.md, done` (`roles.quality` names a binding: the
   toolkit ran the code quality pass on it), `rk review-prep` has put its `quality.md` in the review directory: don't
   dispatch the code quality reviewer; tell the verifier the file is final.
   Otherwise, in the same batch, dispatch the rocKE Code Quality Reviewer
   (`.claude/team_members/rocke_code_quality_reviewer.md`)
   as a second background subagent (agent type `rocke-code-quality-reviewer` with no `model` option: its file runs `<DEEP_MODEL>`;
   the same fallback, with model `<DEEP_ALIAS>`), for every first round (docs-only PRs too: one home per fact, claims
   backed by tests) and for every follow-up whose new commits change anything but docs, fixes for our own items, CMake
   and test lists included, since a fix can break what it touches and a CI list decides what runs (lessons L-567,
   L-569). A follow-up gets no code quality pass when its delta is only develop merges or docs, or when its new commits
   change only test files and docs to answer our own open findings (no library, kernel, binding, CMake or CI-list
   change): the verifier checks it against the quality checklist itself, mutates what the new tests claim to pin, and
   writes the Code quality row from the delta and the last review's row. A micro round gets no code quality pass either.
   `rk dispatch-pack`'s last line applies the rule to the classified delta
   (`lane: exempt (only test files and docs changed: if they answer our own open findings; else run the lane)` leaves
   that judgement to you). The code quality reviewer has no call target: it hands back once every changed file and every
   claim is traced (L-569). Give it the head and the merge base (the prep summary's last lines), the diff's file list
   (`files.txt`), the checks' section "Overlapping work in other open PRs" (`overlap_section.md`, or
   `rk overlaps N --section`), the review directory (it writes `quality.md` there) and, for a follow-up, our open
   quality findings and our earlier follow-ups (not raised again). Tell the verifier the code quality reviewer runs in
   parallel. The verifier folds `quality.md` in itself: before it hands back, it waits up to about 20 minutes for the
   code quality reviewer's final file in the review directory (`rk wait-quality`; a fold-in resume re-writes its whole
   context). Send it to the verifier (SendMessage) to fold in only when the verifier handed back with the Code quality
   row `pending`; the verifier returns the final post. CI updates (L-209) need no code quality pass.

   **Head moved mid-verification** (`REVIEW_DROP #N` while N's agents run): stop both agents (`TaskStop`), with no
   "stop and save notes" message: they keep `notes.md` and `quality.md` current as they go. The toolkit re-picks the
   PR: a PR the review queue started goes back to the front of the queue (its line says `back in the review queue`),
   any other comes back with the plan. Brief the next round's agents to start from those files and keep what is still
   true at the new head (L-104). A first round reuses its review directory; a follow-up's next round gets the next one,
   so name the stopped round's files.
3. **Check the verified post yourself:**
   - every "(ran)" has a matching section in `checks.md`;
   - the post starts from the PR's stated scope and has its two parts: **Needed to merge** (wrong results; a build, CI
     or ABI break the PR causes; a regression in code it touches; a test it adds that cannot fail; a claim in its body
     or commits that is false at head) and **Highly recommended follow-up (not blocking)** (significant items that
     should be done but do not block). Correctness and breaking changes are never follow-ups;
   - the header light matches the worst open Needed-to-merge item, prior rows included; follow-ups never set it, so a
     PR whose blockers are all fixed reads 🟢;
   - every real blocker is listed under Needed to merge, each as a full row with its light (no cap);
   - each F row is significant (no nits, style or wording) and carries a 🟠 or 🟡 light for its importance;
   - no nits or should-fixes creep in from outside the PR's stated scope as blockers;
   - earlier follow-ups are not re-raised: they sit in the one Follow-ups row, with ticket keys once filed, and an item
     the author declined with a reason reads DEFERRED (author: out of scope) unless a real break overrides it;
   - an ABI note from G9, if any, is addressed;
   - a follow-up heading says "Follow-up".
   - the verifier's `brief:` line: fix any wrong fact at its source (the facts script or the brief template).
   - the Code quality row is present and not `pending`, and its light reflects the blocking quality findings only;
     spot-check each kept 🟠 quality finding's evidence (both homes of a duplicate, the sibling path in the other
     engine, the missing test) and its tag (introduced by the PR: blocking; continued or beyond the change: follow-up).
   - each drafter's lines in block 2 (the live draft's tag, and the shadow's when a shadow draft came) cover every
     draft finding, every posted finding and every prior row the draft statused, or read `<tag> none`.

   If a fact is missing, run the check yourself before posting.
4. **Post with its premises.** The verified post stays in the review directory, named for the PR and the round
   (`pr101_fu2.md`). Write the spec, one uniquely named JSON file per post (`spec101fu2.json` in `paths.tmp`), with
   `rk spec N --round R [--save "<SAVE>"] [--cost-verifier T --cost-quality T]` (`commands.md`): it reads the review
   directory's `notes.md` (its `premises:`, `FAILOK`, `CI pending:`, `conflict set` and `decide:` lines) and the post,
   and reads the rest live (FAILOK and RUNOK as `rk ci-fails N`, CONFLICTS from `git merge-tree` with the PR's base
   branch, the body's tracker tag, the threads); it anchors the fill on the Bots / CI row's ✓ count ("12 ✓"), adding one
   ("; 12 ✓") to a row with none. Each `CHECK:` line it prints (exit 3) is a difference between the notes and now: read
   it before posting (a new failure, push, thread or conflict set means re-verify; a finished or re-run job means
   refresh the Bots / CI row and run `rk spec` again). Then run `rk post N FILE --premises SPEC`; `--dry-run` runs every
   check and `rk post`'s guards and stops before the post. A re-review at a head we posted on adds `--again` and uses a
   new ROUND.

   **In one call:** `rk spec-post N --round R [rk spec's options] [--again] [--dry-run]` runs `rk spec`, and
   only when it made the spec with no `CHECK:` line, the post with it (`rk post … --premises`); `CHECK:` lines stop it
   before any post (exit 3: read them as above, then run it again). Use it instead of the two commands. It writes the
   spec again from `notes.md` and the post on every run, so a change goes into the notes (or post an edited spec with
   `rk post N FILE --premises SPEC`). The spec's keys:

   | Key | What |
   |---|---|
   | `N`, `ROUND` | the PR and the round (`fr`, `fr2` …, `fu1`, `fu2` …) |
   | `SRC`, `NOTES` | the verified post (the command's file) and the verifier's `notes.md` |
   | `HEAD10`, `DEVBASE` | the head and the base it verified (10 characters each): develop's commit, or, for a PR on a feature branch, that branch's (the notes' `base` line: the branch and its commit) |
   | `CONFLICTS` | the files the merge with the base branch conflicts in, space-separated (`""` when clean) |
   | `FAILOK`, `MAXCANCEL` | the failing check runs it accounted for (`rk ci-fails N` lists them), and the most cancelled ones allowed |
   | `RUNOK` or `RUNANY` | the runs it saw still running (re-check them on the next ticks: L-209), or `true` when the CI row says CI still runs broadly |
   | `THREADS`, `STOP` | the threads, total then resolved (`"6 5"`), and the UTC time it stopped reading |
   | `<JIRA_PROJECT>` | the tag the PR body must still carry; `""` when the body has none (the key is required) |
   | `fill` | `[old, new]`: the Bots / CI ✓ count, filled in at post time (`$ok` in new: every successful run, re-runs included). `old` must be unique in the post: `2 ✓` also matches inside `1,042 ✓`, so anchor it on the next word |
   | `FOOTER` | the footer's start (default `_First-round review` for `fr`, else `_Follow-up review`) |
   | `SAVE` | the `rk save N verify` text (step 5) |
   | `AFTER` | shell lines run once the post is saved, e.g. `$RK overlap-decide …` (step 5) |
   | `COST` | `{"verifier": N, "quality": N}`: each pass's `subagent_tokens` from its task notification (`null` for a pass that did not run); `rk post` records them in `state/review_cost.jsonl` |

   It prints each step and stops at the first change (exit 4), on one last line that says why: what the spec or the
   post has, what is there now, and what to do (`premise: threads changed: now 6 5, the spec has 5 5 (total,
   resolved): stop`):
   - the premises: the live head and open state; the PR's base branch (develop, or the feature branch it targets) has
     not changed the PR's own files since `DEVBASE` (the files from `files.txt` too); the conflict set; CI at the
     latest attempt of each check (failures exactly `FAILOK`, cancels at most `MAXCANCEL`, nothing running beyond
     `RUNOK`); threads; no review, review comment or human comment after `STOP`; the <JIRA_PROJECT> tag in the body;
   - the post: no CRLF; its copy in `paths.posts` (`pr101_fu2.md`), not written yet; the ✓ count filled in; no PR
     number written `#101` outside line 3; no hygiene hit (links, internal names, GPU product names, `/home/` paths,
     "the user", "stale", the words matched whole; `@` before a letter: write decorators without it,
     `pytest.mark.skipif`), measured figure (a number run into a unit or an `s`: "17s", "the 404s", "5 min", "2 GB")
     or verdict word; line 1 (the one the review was staged with); the header; the Code quality row, not pending, and
     the footer (naming the verifier; with no drafter, crediting no first review pass); no review of ours at this
     head, posted or kept (with `--again`: the state's head is the live head).

   Then it writes the copy and posts it through `rk post`, which:
   - refuses a line 1 other than the one the review was staged with, non-public links, absolute performance numbers,
     linked <JIRA_PROJECT> tags, a second post at the same head (unless `--again`) and posting over the hourly rate;
   - records the ledger;
   - posts with the write login (in the container as its own user, through the write broker);
   - copies the post to `<OUTPUTS_DIR>/PR_FIRST_ROUND_N.md` / `PR_FOLLOWUP_N.md`;
   - frees the PR's place among the parallel reviews.

   With review posting off, it keeps the review unposted instead (no comment, no ledger entry), and only some of the
   `AFTER` lines run (`commands.md`, the `post` row).

   Last, it runs `rk save N verify "<SAVE>"` and the `AFTER` lines, archives notes.md to
   `<WORKING_DIR>/lessons/notes/` (`2026-11-24_pr101_fu2.md`; L-215), and records the notes' assessment lines in
   `state/drafter_assessment.jsonl` (`assessment recorded …`, naming any line it did not understand). A step that fails
   there is said in a `!!` line and the rest still runs: finish it by hand.

   A premise stop is a stop, not a re-verify (lessons L-224): when CI only added re-run attempts or finished jobs,
   refresh the Bots / CI row and the spec (`rk spec`), and run it again; a new failure, comment or push means
   re-verify.

   **When develop changed the PR's own files** (`premise: develop changed k of this PR's own files since DEVBASE …`,
   the files on the lines above it; for a PR on a feature branch, `the base <branch> changed …`), run
   `rk spec-post N --round R --devbase-recheck`: it posts nothing, prints develop's diff stat for those files since
   DEVBASE, and re-runs every other premise at today's develop, the conflict set included
   (`devbase-recheck: every other premise is true at …`, with the conflict set, or the next stop). Then the checks
   that need no model, at today's develop:
   - the merge's conflict set (`git merge-tree --write-tree --name-only` with today's develop and the head, in the
     mirror: L-229), against the spec's `CONFLICTS`;
   - the saved probes, if any, on a scratch merge of the head with it (L-134), TREE its folder:
     `rk pr-run -- python3 -B <PACKAGE_DIR>/tools/run_probes.py <WORKING_DIR>/probes/pr<N> TREE`, against the lines
     the verifier relied on.

   Nothing changed: set the develop SHA on the `premises:` line of `notes.md` to today's develop (10 characters) and
   run `rk spec-post` again, which writes the spec from the notes; or set the spec's `DEVBASE` and post it with
   `rk post N FILE --premises SPEC`. Something changed: start a small
   fresh `rocke-review-verifier` with a narrow brief (the verifier's `notes.md`, the post, the changed files with
   develop's commits on them, and what changed); it updates the post, the notes and the `premises:` line. Resume the
   original verifier instead only if it handed back minutes ago: a resume reloads its 300–450k context, often cold
   (about $4.5, against $1.5–2.5 for a fresh re-check).
5. **Record.** The post logs itself in the daily log, with its verdict, findings and link, and records its saves:
   each 🔴/🟠 finding, and each earlier finding of ours now confirmed FIXED.
   - Add what verification corrected, which measures the live drafter's quality: the spec's `SAVE`, the count and a
     short list from the verifier's notes (`"2 qwen claims corrected: N3, N5"`; for a drafter binding's draft, its
     name in place of `qwen`); `rk post --premises` runs `rk save N verify` with it.
   - A post that needed no correction gets `SAVE` `"0 qwen claims corrected"`, so the rate stays honest.
   - With no drafter there is no draft to correct: leave `--save` out, and `SAVE` is the round, a colon and the
     header's light and words (`fu2: 🟠 Follow-up: one blocker left`).
   - Each `decide:` line in the verifier's block 2 (a key, its owner and the members: an owner the post proposed)
     becomes one of the spec's `AFTER` lines, `$RK overlap-decide KEY OWNER --members N,N --note …` (`rk spec` adds
     it), so the reviews of the other members follow it once the post is saved. Local: no GitHub write. A key that is
     not an overlap key is a `CHECK:` line: fix the `decide:` line.
   - Then each other open member of the group owes an overlap notice (the tick prints `OVERLAP_NOTICE_DUE #N` until
     it is on; `rules.md`, Overlapping work). Have the code quality reviewer draft it at both PRs'
     current heads: the duplicate with file:line on both sides, the owner and the resolution, the landing order the
     merge plan suggests (say plainly what each side does when the plan lands the copier first), and the rebase
     steps. Line 1 is `Automated overlap notice by <AUTOMATED_REVIEWER_NAME>`, and the post rules apply. Check it, then
     `rk notice N FILE --kind overlap --key K[,K]` (one comment per PR; keys joined by ",", since a "like/" key itself
     contains "+").
   - **Follow-ups.** Once the post is up, add each new F row to `<TOOLKIT_DIR>/state/followups/tickets_<N>.md`,
     one line per item, such as `F1 · GIST · FILE:LINE · awaiting AUTHOR` (the row, its gist, where, and the PR's
     author), with the post's link. At each later round, read the verifier's `followup:` lines; at the PR's merge,
     read the threads yourself. Then, per item:
     - the author agreed, or has not answered: file a <JIRA_PROJECT> follow-up ticket by `rules.md`, Follow-up
       tickets (dedup against <JIRA_PROJECT> first), and write its key on the item's line;
     - the author declined with a reason: write `DEFERRED (author: out of scope)`; no ticket;
     - fixed in the PR: write `FIXED` and the commit; no ticket.

     The next round's verifier reads the file for the Follow-ups row, so no later post re-raises an item.

## Split review

**When.** The one in-depth review of a PR over `features.review_automation.large_prs.in_depth_review_over_lines`
(`REVIEW_MANUAL #N`), normally its first round.
One verifier cannot read a 30k–500k-line diff in its window: single reviews read their whole diff up to about 1.5k
changed lines, and their 🔴/🟠 recall falls 3–4× above that because the files that hold the misses go unread. So the
verification is split (`rk review-split`).

**Approval first.** An in-depth review costs too much to run unasked: nothing of it starts (no staging, no checks, no
draft, no agents) until <user_login> approves it, whatever starts it. The Needs you list asks them, with the PR's size
and about how many agents the review runs and what they cost (the split's rule, and the spend card's figure the split's
`cost:` line uses): Approve on the admin dashboard or `rk review-approve N`, Decline there or `rk review-skip N`. Until
then no line names it to you, its review-queue entry waits (`waits: your approval`), and `rk review-start N` refuses
it. The approval is theirs alone: the guard refuses `rk review-approve` to you. Approved, it starts as any review too
large for the autopilot: from the plan by hand (`REVIEW_MANUAL #N`), or from the review queue by the driver. A first
round forced on a PR we reviewed (`rk review-start N --kind first-round`) is an in-depth review too: the command refuses
it until they approve it with `rk review-approve N`, then starts it. An approval stands, whatever the head does, until
a review of ours is recorded after it or the PR closes.

**A giant PR's later rounds are light** (their REVIEW_MANUAL and REVIEW_READY lines say so), and need no approval. One
verifier, no code quality pass, no split: it checks each of our open findings at the new head and the delta for breaks,
conflicts, CI and new threads, and carries every other row.

1. **Start and prepare.** `rk review-start N` when the REVIEW_MANUAL line says (one the review queue starts needs
   nothing); at its REVIEW_READY, which says to split it, prepare as On REVIEW_READY's step 1 says.
2. **Split.** `rk review-split N` (same `--round` as the prep; `--max-lines L`, default 8000, within 6,000–10,000;
   `--max-agents K`, default 20). It reads the PR's own diff, merge base with its `base.ref` to head, and writes in
   the review directory:
   - `parts/part<k>.md` and `parts/part<k>.files`: the part's files with their lines (a file over the limit is cut
     into slices by line range), the command that prints its diff, the checklist lines of every area in it, quoted
     from the verifier team file (`paths.verifier_team_file`, or `--team-file`), the code quality checks quoted from
     the code quality reviewer's team file beside it, the files it shares with other parts, and the generated, golden
     and data files near it, summarized with their generator rather than read;
   - `parts/cross.md`: the cross-cutting pass's inputs (the ABI files and public headers, the CMake, CTest and tier
     lists, the hot spots, every link between parts, the sliced files, the checks' clash and overlap sections, all
     summarized data, pure renames and stray files);
   - `parts/plan.json`.

   It prints the parts (lines, files, about how many tokens of diff, areas) and the dispatch plan in waves. One part
   means no split: verify as usual. Each part is filled up to the limit area by area (library kernels and builders
   per family, the library's other packages, platform C++ by directory, the bindings, platform Python by package, the
   Rust crates, the tests, the docs, CMake/CI and test lists, another project by its top directories), so a part may
   hold several areas and a slice may share a part with other files. Files that change together stay in one part: a
   C++ mirror and its Python twin, a header and its source, a test and the module it is named after or imports, a
   CMake or YAML list and the files it names.
3. **Dispatch the parts** as the plan's waves say, in the background: per part, a `rocke-review-verifier` with
   `effort: "<VERIFY_EFFORT>"` and the description the plan prints for the part, as printed. The brief starts with the
   part file's path as printed and names the review directory, the head and merge base, `facts.txt`, `checks.md`, the
   `rk` CLI's values, a scratch prefix of its own (`wt_v<k>*`), and that it is a part review (the team file's Part
   review). No draft, no shadow, no line 1, and no code quality reviewer: the part verifier runs the code quality
   checks its part file quotes.

   The description and the path name the PR and the round: that is how the orchestrator knows the review has its
   verifiers. An agent dispatched without them counts for no review, and the review's REVIEW_READY line is handed back
   while the parts run. At most `--max-agents` agents run at once; other reviews' agents count too, so lower it while they run. Start a
   later wave's parts as agents hand back. On a follow-up the plan gives no verifier to a part where nothing changed
   since our reviewed head: the cross pass carries its prior rows.
4. **The cross pass.** Once every dispatched part has handed back with its `part<k>_findings.md`, one
   `rocke-review-verifier` with `effort: "<VERIFY_EFFORT>"` and the description the plan prints for the cross pass:
   its brief is a normal verification's (On REVIEW_READY's step 2: line 1, the drafter and its tag or that there is no
   drafter, the shadow draft, `facts.txt`, `classify.txt`, the prior review, the scratch prefix) plus the path of
   `parts/cross.md` as printed and that it is the cross pass (the team file's Cross pass). It writes the single post (`pr101_fu2.md`: the PR and the round) and `notes.md`, as any
   verifier does.
5. **Check and post** as On REVIEW_READY's steps 3–5 say. The spec's cost: `--cost-verifier` is the part verifiers'
   and the cross pass's tokens summed; no `--cost-quality`, since no code quality reviewer runs.

**Head moved mid-review** (`REVIEW_DROP #N`): stop every agent of the review. Run `rk review-split N` at the new head,
and brief a part whose files and lines are unchanged to start from its `part<k>_notes.md`.

**Cost scales with the PR's size**: one verifier per part plus the cross pass, so a PR of k parts takes k + 1 agents,
each about one review's cost. The plan's `cost:` line counts them and, with review spend on, gives about what they
cost at the mean review of the spend card's last 7 days. A part of at most 8000 changed lines fits one verifier's window: its diff is about 11 tokens a changed line, it
reads about three times that around the diff (callers, the other engine, the tests), and its fixed start (lessons,
team file, facts, checks) is 60–80K.

## Isolation

Every run (review checks, gate, batch gate, develop sanity, compile baseline) uses a pristine tree and a fresh build:
- `mkwt.py` aborts, resets and git-cleans an existing worktree before checking out;
- `build_native.py` deletes the worktree's folder in `build/` and builds from scratch;
- the checks output directory is wiped at the start;
- every section that reads the native build builds this exact tree first.

Nothing from an earlier review of the same PR can leak into a result.

## Manual controls

| Command | Effect |
|---|---|
| `rk review-status` | The review in flight, with both jobs' log tails; the review queue; the pool's and the shadow drafts |
| `rk review-queue [N …] [--again] [--drop]` | The reviews the driver starts before the plan's, in order (`commands.md`) |
| `rk review-start N --again` | A re-review now at a head we posted on; refused while a review is in flight |
| `rk review-split N [--max-lines L] [--max-agents K]` | A review too large for one verifier cut into parts and a cross pass, with the dispatch plan (Split review); writes only `parts/` in the review directory |
| `rk drafts` / `rk evidence-check N [--dir DIR]` | The drafter, its pool, last drafts, scaler decisions and card / a draft's evidence checked again with no model (`commands.md`) |
| `rk draft-lint N [--dir DIR]` | The draft's flags against its own inputs, in `checks/lint.md`; the draft is unchanged (`commands.md`) |
| `rk review-skip N` / `rk review-unskip N` | Keep PR N out of the autopilot picks, or put it back. A skip also stops its review in flight, keeps an interrupted one from starting again, withdraws one awaiting verification and ends the approval of an in-depth review (it declines one: Split review); stop the agents you started for it yourself (`TaskStop`) |
| `rk classify N [--record]` | Classify a follow-up; with `--record`, it reads the PR's live head and refuses (exit 4, nothing recorded) while that is not the head the tick saw: run `rk tick`, then again. A merge-only or rebase-only result at the live head records the equivalence, with review automation on or off, and a review of that head awaiting verification or in flight is closed without a post |
| `rk jobs [ID]` | Recent jobs, or one job's log |
