# Orient: scout and synthesis

Each `SCOUT<k>` describes one area as it stood at `$BASE`; `SYNTH` merges the area files
into `CONTEXT.md`. Neither files findings. Apply [common.md](common.md) *Rules*, except the
Handoff and ledger rules.

## SCOUT

You receive an area, its slug, the changed paths in it, the PR description (or intent
document) and, if any, a prior area file. Write only `$EVIDENCE/r<N>/orient/<slug>.md`.

Read `$REPO_BASE`, and `git -C "$CLONE" show "$BASE:<path>"` outside the sparse set. Never
read `$REPO`, the diff or the PR's commits: reviewers measure the change against your
account of the base. The changed paths and the description only say where to look.

Describe the area, not the change, in at most 80 lines. Link; do not copy. Every backticked path is relative to the
repository root, never to the area. Write alternatives as `a, b and c`, never as `a/b/c` in backticks.

```markdown
# <Area>
Describes `<area paths>` at `<full $BASE sha>`, <date>. Status: new | refreshed | reused
```

Then, as `##` sections: **Purpose and boundaries**; **Glossary** (term, meaning, defined
at); **Code map** (path, role); **Design as it stands** (entry points, contracts, extension
points such as how a new op, engine or option is added, known limits, where a document and
the code disagree); **Docs and RFCs** (path, what it settles, status); **Tests** (suite,
where, what it covers, how to select it); **Options and guards**; **Consumers**.

**Options and guards.** For the area's changed files: each `option()` and cache variable with its default, each
`#ifdef`, `#if` and CMake `if(...)` guard, and for each change the flag it depends on and that flag's default, so a change
that is dead code with the flag OFF shows. `RUN`, `ROCK` and `AUTH` read this section.

**Consumers.** The in-repo includers of the area's changed public headers (`git -C "$CLONE" grep -l <header name> "$BASE"
-- <pathspec>`) and which project or provider tests exercise them. `RUN` and `AUTH` read this section.

Sources, in order: the area's `README.md` and `docs/`; `projects/hipdnn/docs/`; RFCs under
`projects/hipdnn/docs/rfcs/`; then the code and tests. Check a document against the code
before citing it.

A prior file is current when its `Describes` sha is in `$CLONE` and this prints nothing:

```bash
git -C "$CLONE" diff --stat <prior sha> "$BASE" -- <area paths>
```

Then keep the file, set the sha to `$BASE` and add only what the description makes
relevant. Otherwise re-read what the stat lists and update those sections.

## SYNTH

You receive every `orient/<slug>.md`, the description, `PIN.md`, `files.txt` and
`numstat.txt`. Do not read the diff or the code. Write only `$EVIDENCE/r<N>/CONTEXT.md`, at
most one page:

Start with `# Context: <target> (round <N>)`, then `##` sections: **Purpose** (two
sentences from the description); **Glossary** (merged, one line per term; mark a term two
areas define differently); **Areas** (table: area, relative link to its Orient file, status,
one-line summary); **Where the change lands** (per area: the entry points, contracts and
extension points the stated purpose goes through, and the touch points a change of this
kind updates: registries, CMake lists, docs, tests); **Options and guards** (merged from the area files: flag, default, areas and paths it gates);
**Consumers** (merged: header or surface, consumer project, tests); **Tests that should move** (suite,
area, why); **Questions for reviewers** (contracts the description leaves open,
contradictions between area files).

Derive each statement from the description and the area files; do not guess what the diff
does, and do not repeat the area files.
