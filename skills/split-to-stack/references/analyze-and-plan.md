# Phases 2–6: analyze, classify, group, order, propose

Read by the commit readers, the Inventory, the Planner, and the orchestrator (Phase 6). Also read `concepts.md` and `files.md` if your role lists them. Analysis agents never execute repository code (`common.md`).

## Phase 2: Analyze

**Commit list.** `git rev-list --reverse B..$L` (L is linear on B, with duplicates and commits already in B removed and merge residue folded in by Phase 1). The orchestrator records the list, and what Phase 1 dropped (from `linear.md`), in `commits.md` (the orchestrator owns that file), and splits the list into batches for the commit readers; never more readers in parallel than free agent slots.

**Commit readers** (read-only, parallel batches). For **each** assigned commit run `git show <sha>` (do not classify from subjects alone); for generated and bulk files use `--stat` only (bulk = more than ~500 changed lines in one file, or more than 20 files of one pattern). Write `commits/<full sha>.md` (full 40-character SHA; under ~40 lines): intent, the files and symbols it changes, and notes useful for grouping, dependencies and order. For every new or changed symbol, string, key or file, state **who reads it** (its consumer) and whether the change alters the output of an existing path. Describe what the commit does; do not sort it into pre-named categories or layers, and do not adopt categories suggested in your task. To state dependencies you may read the other commits of the list (`commits.md`) with `git show --stat`; you still write only your own files. For the hook commits (`style: apply pre-commit hook fixes`) give just the file list. A revert pair split across batches: say so in both files. Commit diffs are evidence of intent; the aggregate diff is what gets split.

**Inventory** (read-only except `ledger.tsv`).
1. Decide the generated sets: `.gitattributes` (`linguist-generated`, `-diff`), known generated paths and extensions (for example `.s` under `CustomKernels/`, `Logic/**.yaml`), and generator header markers. Name one set per producing purpose and write the globs in `inventory.md` (the ledger header records them too; the orchestrator copies them into `run.md`). Define named globs before relying on `--attr-generated`: every attribute-marked path that no glob claims falls into one set, `attr-generated`. Hand-authored bulk files (for example copies of an existing config that differ by two lines) are **not** generated: they stay ordinary `F` rows, which the Planner groups by purpose; name a set only if a single purpose covers hundreds of files.
   Deleted, renamed and type-changed generated files become ordinary `F` rows automatically (their `generated` column still names the set). For generated files that must stay with their code (generated test files or fixtures, or files the code cannot build or pass without in either order), pass `--exclude-generated GLOB`; list each exclusion and its reason in `inventory.md`.
2. Run `scripts/build_ledger.py --repo <repo> --base B --tip $L --out $RUN/ledger.tsv --generated-set NAME=GLOB … [--attr-generated] [--exclude-generated GLOB …]`. The tip is always the SHA of L (`$L` from `run.md`; Resume compares it with that). The `commits` column lists L's commits by path, so it is coarse evidence; the commit readers' notes are what the Planner uses for authorship.
   If the script exits with status 3 the diff is empty: everything already landed (for example through squash merges). Write that in `inventory.md` and return `BLOCKED`; the orchestrator reports it and stops before anything is built.
3. Write `inventory.md`: entry counts by kind, files, hand-written and generated changed lines (plus + minus), number of commits (those in the range, and how many have ledger rows), generated sets with file counts and lines, and anything unusual (mode changes, symlinks, submodule pointers, binaries, `whole` rows). Changes introduced and then reverted are not in the aggregate diff, so they are already dropped.

Size is informational: there is no size gate. The numbers go into the Phase 6 proposal.

## Phase 3: Classify

Give each ledger entry a one-sentence purpose and a type from the closed list in `concepts.md`. Classification is by purpose, not by commit or file: a commit or file may hold changes of different logical changes. Record purposes in `plan_assign.tsv`.

For each entry also record, at the start of its `note` column, its **consumer** (who reads what the entry produces) and whether it is a **behavior change** or **new surface** (`concepts.md`), for example `consumer: handwritten custom-kernel launch; new surface`. These attributes, not the language or the directory, decide grouping in Phase 4. A behavior-change label needs evidence: for a changed guard, condition or early return, read the predicate definitions and their inputs and say in the note why the condition can (or cannot) differ (`concepts.md`, "A behavior change must be shown"); otherwise write `behavior change (unverified)`.

If one entry serves more than one purpose, mark it **entangled** (see `concepts.md`): it gets one step per group, with a state for every step that is not `final`. Take intermediate states from L's history where they exist (`scripts/extract_state.py` for hunk entries, written as `stored:<path>` with note `history: <sha>`; `commit:<sha>` for whole-file entries; `git log -L <range>:<file> B..L` finds the commit). A state derived by **subtraction** (the final text with the parts that belong to later groups deleted, adding no line that is not in the final text) is not guessing: it is the normal way to stage a file that serves several groups, and the Planner must use it instead of keeping the file whole; write it as `subtractive:<path>` (`check_plan.py` proves it is the final text with lines removed, and that each subtractive step is contained in the next one) and list what was removed in `plan.md`. Any other hand-written state is `stored:<path>` with note `synthesized`. Only when an intermediate state would need text that exists nowhere in the final file (invented glue code, a rewritten helper) is it guessing: then keep the entry in one group and state both purposes (the proposal tells the user).

Also identify **transition code** (temporary compatibility code a lower branch needs so each branch stands alone, `concepts.md`) and record it in `transitions.tsv`. Do not run anything to find out: if you cannot tell whether a split point builds, mark it *unverified* in `plan.md`.

## Phase 4: Group

Organize the entries into branch groups, written to `groups.tsv` and `plan_assign.tsv`:
- every ledger entry is assigned (entangled: one step per group); each group is one branch with a single purpose;
- group by what changes accomplish, not by which commit they came from. A logical change has one purpose, one consumer and one risk surface (`concepts.md`) and is a fix or feature **plus its test**; a file is not a logical change, and one file can hold the tests of several. Size is a reason to split only when the parts are independently reviewable; never split a logical change to hit a line count (aim for under ~400 hand-written lines per branch only where that holds);
- **contract halves stay together** (`concepts.md`): the producer and the consumer of one value are one group even if they are in different languages, directories or test frameworks;
- **behavior changes to existing paths go in their own group**, below the new-surface group that benefits from them, unless separating them needs invented text (`concepts.md`);
- **a different consumer or a different risk surface means a separate group**, even in the same file or layer. For example a change to generated or serialized output is not grouped with a runtime launch path just because both serve one feature name;
- a dependency alone is not a reason for a separate group (Phase 5): if the lower group has no consumer without the upper one, merge them;
- logical changes spanning several files stay together (a signature change and all callers);
- **tests land in the same group as the behavior they assert.** A test file, or a test body, that asserts behavior from several groups is split by what each part asserts (per test case, per API, per section of a shared body), and each part goes in the group that introduces that behavior, using subtractive states for the intermediate versions of the file (Phase 3). Shared helpers, build lines and includes go with the first group that needs them. Only a part that needs several groups at once goes in the last of them. A trailing test-only group is allowed only for tests that do not depend on the split's code, or tests that cannot be split without invented text; the Planner states why in `plan.md`;
- generated files go into standalone branches unless an exception applies (`concepts.md`);
- formatting-only changes form a `style` group at the bottom;
- for each group set `author`, `trailers`, `scope`, `subject`, and a plain-language `summary` of 1–3 sentences for a reviewer who has not seen the source (no symbol dumps; for generated-file branches add the file count and lines).

**Test plan** (written even if the user later skips tests): for each branch, list what it changes and who consumes it and, from each touched component's `TESTING.md` or `AGENTS.md`, the build target and unit-test commands, plus `pre-commit run --files <the branch's changed files>`. Organize it by branch purpose; the component table is an input, not the structure. The test plan lists commands only; it does not run them and does not predict failures.

## Phase 5: Dependency analysis

Group Y depends on group X if any of these holds. Check each one explicitly:
1. **Code:** Y modifies, calls or overrides code that X adds, removes or changes the signature of.
2. **Symbols and data:** Y uses a type, enum value, constant, struct field, header, Python import, registry or serialization key that X adds, renames or changes.
3. **Build:** Y needs a build-system change in X: CMake targets, source lists, architecture lists, requirements, packaging.
4. **Generated output:** Y contains output produced by a generator or input change in X, consumes generated files added in X, or the reverse. A standalone generated-files branch goes wherever the stack still builds and passes at every step: below its consumer if adding the files alone is inert, above it if the code works without them. If neither order works, the atomic-change exception applies and they stay together.
5. **Tests:** Y's tests assert behavior that X introduces, or X's change breaks a test that Y fixes. Apply this per test part, not per test file: a file with three test cases for three fixes does not depend on all three as a whole, and must not be moved to the end for that reason (see Phase 4).
6. **Config/CLI/env:** Y relies on a flag, env var, YAML key or config option that X adds.

"Different functions in the same file" is only independent if none of 1–6 holds. Phase 8 (if the user chose tests) is the backstop: a failure there means a missed dependency, so fix the order.

A dependency decides the **order** of two groups, not whether they are two groups. Before keeping a dependent pair separate, apply the Phase 4 tests: if they are halves of one contract (`concepts.md`), or the lower group has no consumer without the upper one, merge them. Keep them separate only if both are independently useful and independently revertable.

Output: the dependency graph, used only to compute the **linear order**, a topological sort; `depends_on` in `groups.tsv` lists each group's lower dependencies. Among groups ready at the same time, order by type (`concepts.md`). A dependency cycle means the groups must be merged into one. Group numbers `NN` follow the final order.

**Completeness gate (end of Phase 5, and again after any plan edit or renumbering):** `scripts/check_plan.py --repo <repo> --run $RUN --prefix users/<you>/` must print `PLAN CHECK PASS` (the orchestrator adds `--check-remote origin` before Phase 6). A failure is a planning error: fix it before anything is built. Warnings (an entry inside a transition window) must be read and answered in `plan.md`.

**Grouping critique (written to `plan.md` before Phase 6):** a section `## Grouping critique` answering, from the diff and the plan only:
1. For each pair of adjacent groups: *why can they not be one branch?* Acceptable answers name a different consumer, a different risk surface, or independent usefulness and revertability of both. These are **not** acceptable: "different language", "different directory or component", "different test framework", "it depends on the lower group", "it is small", "the commit reader suggested it".
2. For each group: *what is the consumer of this group alone, and what breaks if it is reverted on its own?*
3. For each group of mixed content: is there a behavior change to an existing path inside a new-surface group (`concepts.md`)?

A Planner whose answer is not acceptable must merge or re-cut the groups and re-run `check_plan.py`. The orchestrator checks that the section exists and that no answer is on the not-acceptable list before Phase 6. For plans with three or more groups the orchestrator also starts a fresh read-only agent to attack the critique (find a pair that should be merged or a group that should be split, citing the ledger entries) and records the result in `run.md`; its findings are shown in the proposal.

The attack agent must also **re-derive, from the code, every behavior-change classification that creates or justifies a group boundary** (including any group of type `fix` and any group that exists because of a behavior change). It reads the definitions the claim depends on (predicates, their callers, the constructors of their inputs, `git log -S` for why a guard exists) and does not accept the classification from `plan.md`, `plan_assign.tsv` or the commit notes. For each one it reports `confirmed`, `refuted` or `unverified` with evidence. A refuted classification sends the entry back to the Planner (dead-code removal goes to a `chore` or the surrounding group); an unverified one cannot stand alone as a group (`concepts.md`).

**Inputs the orchestrator gives the Planner.** Besides the run files, pass any earlier review comments, findings or notes about this branch that bear on grouping (for example a reviewer's remark that a hunk is a no-op), marked "claims to check, not facts".

## Review thread mapping and reply drafts (when `pr.md` exists)

Re-cutting rewrites commits, so old threads become outdated.
- Map every open thread to the ledger entry containing its file and line, and so to the new branch that holds that code. GitHub anchors a thread to the PR head (or an older commit for an outdated thread), not to L: translate the line with `git blame -L <line>,<line> <thread commit> -- <path>` followed by finding that content in L (`git diff -U0 <thread commit> L -- <path>`), and mark threads with `outdated` or no current line as such. Record it in `plan.md` ("3 open threads → branch 02, 1 → branch 04"). A thread whose lines no longer exist in any branch is reported.
- For each open thread write a short reply draft (never posted by default): "This PR was split to keep it reviewable. The code this comment refers to is now in #<N> (`<branch>`), `<path>` (around line <L>)." Plain, factual, no apology, no commit hashes. For threads whose lines no longer exist, say so and name the branch that removed them. Use `<N>` placeholders until the new PR numbers exist; posting is a publishing action (`publish-and-cleanup.md`).
- Every re-cut old PR is retired (no PR takeover): record, for each, where its code went.

## Phase 6: Propose (orchestrator)

Present the plan before anything is built:

1. **Phase 0/1 results:** target SHA `T`, whether S was updated, base `B`, the linearization report (or "no-op: L = S'"), the **hook delta** (files and lines changed by the hook commits, and any hook findings the hooks could not fix), and the size figures from `inventory.md`.
2. **The stack as one table**, bottom first, from `scripts/render_plan.py --run $RUN --prefix users/<you>/`: exactly the columns `#`, `type`, `branch suffix`, `description`. The common prefix `users/<you>/` is stated once in a note, not in the rows. No other columns (no line counts, no position reasons) unless the user asks; dependencies and reasons for position stay in `plan.md`. Each `description` states in one clause **who consumes the change** and whether it changes an existing path (for example "adds new names only, used by handwritten custom kernels" or "changes the serialized descriptor of every generated kernel"). Add one line after the table: "Grouping critique: <the adjacent pairs and why each stays separate>", or the groups that were merged because of it.
3. **Transition code** (`render_plan.py --transitions`) and every non-final state (`stored:`, `subtractive:`, `commit:`) as a before/after diff (`render_plan.py --states --repo <repo>`). Add a **test placement audit**: list every group of type `test` and every file of tests that is not in the group of the code it covers, with the reason; a trailing test-only group with no stated reason is a planning error to fix before presenting.
4. **`gh stack` mapping:** one stack in the same order; its trunk is the target, or the topmost kept branch when the kept PRs are not the user's (`publish-and-cleanup.md`); or the manual PR-base order if gh or gh stack is unavailable. For every old PR in scope: kept, or retired (closed after the new PRs exist), and any auto-merge or queue state. For a single-branch scope, the dependents left unchanged.
5. **Review thread mapping** with the reply drafts, when `pr.md` exists.
6. **Test plan:** one line per branch with its commands, always shown and labelled "runs only if you choose tests" (the choice is made at implementation start in plan-only mode).
7. **Ownership:** every re-cut PR owned by someone else is marked; retiring it (disabling its auto-merge, closing it) needs a separate confirmation at publishing, and the default for such a PR is to leave it open and only draft the comment. If the plan has a single group, say that it is a rename of the source branch, not a split, and recommend abort.

Example (all branches are under `users/<you>/`):

| # | type | branch suffix | description |
|---|---|---|---|
| 01 | fix | timer-overflow | Fixes the timer wrapping after 49 days by switching to a 64-bit counter. |
| 02 | test | skip-flaky-net-test | Skips the flaky network test on Windows. Unrelated to the timer fix. |
| 03 | refactor | gpu-input-struct | Moves the GPU input arguments into one struct so later changes touch a single place. No behavior change. |
| 04 | feat | async-d2d-copies | Copies device-to-device asynchronously on a second stream, using the struct from 03. |

Offer **approve**, **edit** (regroup, rename, reorder), or **abort**. If the user edits, update the plan files, re-run `check_plan.py`, and present again. Record the approved plan verbatim in `run.md` with `approval: user`, and copy the plan files to `$RUN/approved/`.

**Default run:** stop here and tell the user how to continue (approve, then ask to implement). **`--implement`:** present and record the plan with `approval: auto (--implement)`, then continue to Phase 7 without waiting, subject to the stops in SKILL.md.
