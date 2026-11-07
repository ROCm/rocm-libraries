# Run directory, plan files, and scripts (Inventory, Planner, Builder read this file)

## Run directory `$RUN`

`$RUN=$(git rev-parse --path-format=absolute --git-common-dir)/split-to-stack/<key>/` (absolute; `<key>` as in SKILL.md). Markdown files have the headings `## Inputs`, `## Result`, `## Blocked`.

| File | Written by | Content |
|---|---|---|
| `run.md` | orchestrator (it copies what other agents report: generated sets from `inventory.md`, `$L` from `linear.md`, B and S' from `update.md`) | scope and flags, stack kind, current user, the SHAs of S, T, P, B, S' and L, the `$WT` path, decisions (target, kept and re-cut PRs and their owners, update answer, test answer and any opt-out), generated sets, `approval:` (`user` or `auto (--implement)`) and the approved plan verbatim, last completed phase |
| `survey.md`, `pr.md` | Survey | scope discovery, target status, merges, tools, PR state (size, review state, CI, open threads, auto-merge, merge queue) |
| `update.md` | Updater | S' SHA, B, hook commits (SHAs) and findings, a merge-residue and duplicate-patch scan, conflicts and their resolution |
| `linear.md` | Linearizer | L SHA, replay list, dropped commits, rewrite pairs, folded residue, tree-equality proof |
| `commits/<sha>.md` | Commit readers | intent, changed files and symbols |
| `commits.md` | orchestrator | the commit list given to the readers and the commits dropped (already in B, duplicates) |
| `ledger.tsv`, `inventory.md` | Inventory | the ledger; sizes and generated-set notes |
| `groups.tsv`, `plan_assign.tsv`, `transitions.tsv`, `states/` | Planner | the plan |
| `plan.md` | Planner | dependency reasons, entangled sequences, test plan, thread mapping and reply drafts, gate result |
| `build_date` | `build_stack.py` | fixed commit date so rebuilds give identical SHAs for unchanged branches |
| `stack.tsv`, `refs.tsv`, `stack.md` | Builder (`stack.tsv` and `refs.tsv` written by `build_stack.py`) | `stack.tsv`: the last successful build (NN, branch, sha, tree). `refs.tsv`: the SHAs this run last published to branches; `--refs update` moves a branch only if it still points there. Notes in `stack.md` |
| `tests.md` | Builder | per branch: commands, results, classification, fixes, transition proposals |
| `build/` | Builder | build directory (outside `$WT`) |

## Plan files

Tab-separated, UTF-8, one header row. Lines starting with `#` are comments; `# key=value` lines are header metadata. An empty field is written `-`. Fields must not contain tabs or newlines.

### `ledger.tsv` (Inventory, immutable)

Header lines: `# base=<sha>`, `# tip=<sha>` (always `$L`), `# diff-options=<exact git diff options>`, `# generated-sets=NAME=GLOB;…`, `# attr-generated=yes|no`, and `# generated-exclude=GLOB;…` (only when `--exclude-generated` was used). Columns:

| column | meaning |
|---|---|
| `id` | `H<n>` hunk, `F<n>` whole-file change, `G<n>` file of a generated set (only added or modified paths; see `generated`) |
| `path`, `old_path` | `old_path` only for renames |
| `status` | `A` added, `M` modified, `D` deleted, `R` renamed, `T` type change |
| `hunk` | `-a,b +c,d`: the hunk's range in the base and in L (`H` rows only) |
| `plus`, `minus` | lines added and removed (from the hunk, or from `--numstat` for whole-file rows) |
| `binary` | `no`, `yes`, `sub` (submodule), or `whole` (text file with no hunks) |
| `mode` | new file mode when it changed or the file is new; `-` otherwise |
| `generated` | the generated set the path matches; `-` for hand-written code and for paths matching `--exclude-generated`. On a `G` row the file belongs to `@<set>`; on an `F` row (a deleted, renamed or type-changed generated file) it is information and size accounting only, and the row is assigned on its own |
| `commits` | short SHAs of the source commits that touched the path |

A mode-only change is an `F` row with `status M`, `binary no`, and a `mode`; its only valid plan state is `final`. The `commits` column holds full 40-character SHAs. In every TSV an empty field is written `-`; a literal `-` is written `\-`, and a field that starts with a backslash starts with two.

### `groups.tsv` (Planner): one row per branch, in stack order

| column | meaning |
|---|---|
| `NN` | two-digit position `01`..`nn` without gaps; `01` is the bottom |
| `suffix` | branch name after `users/<you>/`; no suffix may be a `/`-prefix of another |
| `type` | one of `fix chore style test ci docs build refactor perf feat` |
| `scope`, `subject` | commit subject is `type(scope): subject` (no parentheses when `scope` is empty) |
| `author` | exactly `Name <email>` (no `<` or `>` in the name): the author of the source commits the group's entries came from (the one with the most lines in the group); empty means the current git identity |
| `trailers` | `|`-separated trailers (write a literal `|` as `\|`): `Co-authored-by:` for other authors, and trailers or issue keys carried over from the source commits for the group's entries (never invented; no `Split-from:`) |
| `depends_on` | comma-separated lower group numbers this group depends on |
| `summary` | the plain-language 1–3 sentence what-and-why for a reviewer; also the commit body. The Phase 6 table is rendered from it. |

### `plan_assign.tsv` (Planner): one row per entry per step

| column | meaning |
|---|---|
| `entry` | a ledger id, or `@<set>` for a whole generated set |
| `step`, `nsteps` | `1`, `1` normally; `1..n` of `n` for an entangled entry |
| `group` | the `NN` this step belongs to; groups must strictly increase with the step |
| `state` | `final` (the text from L), `commit:<sha>` (whole-file entries only), `stored:<path>` (a file under `$RUN`: from history or hand-written), or `subtractive:<path>` (a file under `$RUN` that is the final text with lines removed; `check_plan.py` verifies it). See `concepts.md` |
| `purpose` | one sentence |
| `note` | free text, for example `history: <sha>` or `synthesized` |

### `transitions.tsv` (Planner)

Columns: `id`, `entry` (the related ledger entry, or `-`), `added_in`, `removed_in` (group numbers, `removed_in` above `added_in`), `reason`, `path`, `stored` (path under `$RUN` of the full file content that applies to branches `added_in` up to but not including `removed_in`; it must already include every other change to that file in that window). `path` may be a temporary file that exists at neither end; windows on one path must not overlap.

## Scripts (`scripts/`)

Python 3 and git only. All take `--repo` (any path inside the repository) and `--run $RUN`.

| Script | Used by | What it does |
|---|---|---|
| `build_ledger.py --repo R --base B --tip $L --out $RUN/ledger.tsv [--generated-set NAME=GLOB …] [--attr-generated] [--exclude-generated GLOB …]` | Inventory | writes the ledger; read-only; only added and modified paths become `G` rows (deleted, renamed and type-changed generated files are `F` rows); `--exclude-generated` paths are never generated; exit 3 and no file when the diff is empty (everything already landed) |
| `check_plan.py --repo R --run $RUN --prefix users/<you>/ [--check-remote origin] [--no-regen]` | Planner, orchestrator, Builder | completeness gate: the ledger regenerates byte for byte, every entry assigned once, steps and groups valid, generated sets whole, stored states and transitions valid, every `subtractive:` state is the final text with lines removed (and contained in the next `subtractive:` step), names valid and free; exit 0 = PASS; prints per-group sizes, the number of subtractive states checked, and warnings |
| `render_plan.py --run $RUN --prefix users/<you>/ [--table] [--transitions] [--states --repo R]` | orchestrator | Phase 6 table, transition list, diffs of every non-final state (`stored:`, `subtractive:`, `commit:`) |
| `build_stack.py --repo R --run $RUN --prefix users/<you>/ --refs none\|create\|update` | Builder | builds the commits (chained from B, one per group) from the plan; exit 0 only if `tree(top) == tree(L)`, and records nothing (no `stack.tsv`, no refs) otherwise; a file/directory clash between groups is an error; `--refs create` creates the branches, `--refs update` creates missing ones and moves existing ones only if they still point at the SHA in `refs.tsv`; all refs move in one atomic transaction; refuses branches checked out in a worktree |
| `verify_stack.py --repo R --run $RUN --prefix users/<you>/ [--tip REV] [--refs] [--no-content-check]` | orchestrator (8.1) | independent check (does not use the composition code): branches equal the groups, linear chain on B, no merges, only paths the plan names change between consecutive branches, line counts of simple text files and blobs of simple whole-file entries match the plan (catches a hunk that leaked into a lower branch), `tree(top) == tree(tip)` |
| `extract_state.py --repo R --run $RUN --entry H<n> --commit <full sha> --out states/<file>` | Planner | writes the stored state of a hunk entry as it was at a source commit (the base lines of the hunk's region with that commit's changes applied); fails if a change crosses the region boundary (then write the state by hand) |
| `selftest.py [--stress R B T]` | maintainers | checks every script on a synthetic repo, or on a real range |

How a branch is built (diffs are taken with `--no-relative`, `--inter-hunk-context=0`, `-l0`, `-M50%`, literal pathspecs and the tip's attributes, so user config does not change the ledger): for group k, each file's content is computed from the BASE blob plus the entries assigned to groups up to k (hunks at their recorded base line numbers; whole-file entries from L or a `stored:`, `subtractive:` or `commit:` state), so no patch application, no fuzz, and no hand edits. A rebuild after a plan edit keeps every branch below the first changed one at the same SHA.
