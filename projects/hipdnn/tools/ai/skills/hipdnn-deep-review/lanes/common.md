# Lane contract

Read this file, [../findings.md](../findings.md), [../policy.md](../policy.md), your brief
and inputs. Scouts and `SYNTH` read this file and [orient.md](orient.md) only.

## Variables
| Variable | Meaning |
|---|---|
| `$SKILL`, `$EVIDENCE` | This skill's directory; the review output directory (layout below; `PIN.md` records every value) |
| `$SCRATCH` | Disposable directory; work only in `$SCRATCH/<ID>/` (your lane, scout, `SYNTH`, `VER-<batch>`) |
| `$CLONE`, `$KNOWLEDGE` | The repository clone (read-only); the store of published Orient files |
| `$HEAD`, `$HEAD_SHORT`, `$BASE` | Full SHA of the head; its first 12 hex digits; full SHA of the base (merge base; left side of a range) |
| `$KEY`, `$TARGET`, `$TARGET_TIP` | Key for `refs/review/$KEY/`; the branch the change is measured against; full SHA of that branch's tip at pin time |
| `$REPO`, `$REPO_BASE` | Sparse worktrees of `$CLONE` at `$HEAD` and `$BASE` |
```markdown
$EVIDENCE/LEDGER.md   $EVIDENCE/r<N>/: PIN.md env.sh CONTEXT.md ID-MAP.md REPORT.md COMMENTS.md
  dispatch-context.md files.txt numstat.txt commits.txt sparse.txt drift.txt areas.tsv owners.tsv   (round > 1: range-diff.txt affected.txt)
  pr/ (pr.json body.md inline.json files.json checks.tsv branch.txt pr-text.txt pr-comment.txt pr-inline.txt) | intent.md   orient/<slug>.md
  packets/<LANE>.md  lanes/<LANE>.md  verify/<ID>.md
  evidence subdirectories: verify/<ID>-evidence/  lanes/*-evidence/  lanes/<ID>/
```
Evidence subdirectories hold raw logs, diffs and scratch output. They are local-only: never
shared, published or swept, and no report or comment may depend on a reader opening them.

## Rules

- Read the description (`pr/body.md` or `intent.md`) with its `Known issues` items (`K<n>`). Do not file a known item. File it
  only when the description understates it (worse effect, wider reach or higher tier than stated); the finding then says
  `understated: known-open-item K<n>`.
- Write every output to the absolute `Output:` path of your dispatch, never to a path relative to your working directory.
- Shell blocks begin `. ./env.sh`, which exports the variables above, `GREP` and `fullclone`: run each block as its own
  command in a subshell (`( … )` or `bash -c`) from `$EVIDENCE/r<N>/` (the absolute paths are in `dispatch-context.md`), so a
  `cd` or an `exit` does not outlive it. A block that prints `STOP:` ends the lane with `STOP: <reason>`.
- Never edit `$REPO`, `$REPO_BASE` or `$CLONE`, or write refs into `$CLONE`. To try a fetch,
  `git clone --shared --no-checkout` it into `$SCRATCH/<ID>/clone` and `git remote set-url
  origin https://github.com/<owner>/<repo>.git` there.
- A path outside the sparse set is absent from `$REPO`, and absence there proves nothing.
  Read it with `git -C "$CLONE" show "$HEAD:<path>"`; search it with `git -C "$CLONE" grep
  -n -e <pattern> "$HEAD" -- <pathspec>` (always pass a pathspec; `$BASE` for the base).
- Use GNU grep: `"$GREP"`, which `env.sh` exports (`ggrep` when present, else `/usr/bin/grep`).
- Read beyond the diff when a claim depends on it; look for what is missing. Read the area
  files in your packet first: they describe `$BASE` and prove nothing. If one is wrong, add
  the Handoff `SCOUT: <correction>`.
- Check a failure at `$BASE` before calling it introduced; pre-existing problems are
  `FUTURE WORK` at most. Do not re-raise a ledger `REFUTED` claim unless its code changed.
- Act outside `$SCRATCH` and `$EVIDENCE` only as the dispatch approves; put any other GPU,
  remote or shared-compute action under **Approval requests**.

## Context packet

`packets/<LANE>.md`. Correctness lane: the owned files with `numstat` rows and their areas'
Orient file paths; find callers, contracts and reaching tests yourself and list them under
**Coverage**. A packet marked `Sampled: <generator>; sample <files>` replaces "every changed
line" with the generator plus that sample; record the rest as `sampled`. Concern lane: the
changed files in scope with `numstat` rows, the areas' Orient file paths and the stack
parent. From round 2: the range-diff path, `affected.txt` and your ledger rows.

A merged agent writes one report named after its first lane; findings keep their lane's
prefix. A split lane is `<LANE><k>`.
