---
name: hipdnn-integration-testing
description: "Run, read, and extend hipDNN's cross-provider integration test suite (dnn-providers/integration-tests): bundles and sweeps, golden data and DVC, .support.json claim sidecars, per-engine TOML, tier YAML, CTest lanes. Doc-driven: loads all of its knowledge at run time from the suite's human-readable docs in dnn-providers/integration-tests/docs/ (local checkout first, GitHub develop as fallback). Use when adding or updating bundles or support claims, running or triaging hipdnn_integration_tests, or when a run prints a SUPPORT CLAIM SUMMARY with unclaimed_support, CLAIM_BROKEN or failed_in_use, 'zero tests ran', or an all-skipped result."
argument-hint: "[task: run|read-output|add-bundle|update-claims|file-formats|triage] [engine name]"
allowed-tools: Bash, Read, Grep, Glob, WebFetch
---

# hipDNN Integration Testing (doc-driven)

This skill has no knowledge of the suite itself. Human-readable documents in
the repository describe the file formats, how to run the suite, how to read
its output, and how to add bundles and update support claims. The documents
are written for developers. This skill loads the documents and applies them.
It makes sure that the developer learns what a run asks of them. When the
documents change, the behavior of the skill changes. This file does not need
editing.

## 1. Load the documents — every invocation

The documents are in `dnn-providers/integration-tests/docs/` under the
rocm-libraries root. The entry point is
`dnn-providers/integration-tests/docs/README.md`. Never answer from memory or
from an earlier session. Load the documents again at each invocation.

1. **Find the checkout that the developer is working in.** Call its root
   `<repo-root>`. Try these options in order:
   - a checkout or worktree the developer named;
   - the current directory: `git rev-parse --show-toplevel`;
   - a build directory that the developer uses: its `CMakeCache.txt` records
     the source tree as `CMAKE_HOME_DIRECTORY`.
2. **Local documents first.** If
   `<repo-root>/dnn-providers/integration-tests/docs/README.md` exists, read
   the documents from there. A local copy describes the code that the
   developer builds, including unmerged changes on their branch. Use it
   instead of any remote copy. That index file is the test. If it does not
   exist, the checkout has no usable documents. This is true even if other
   files exist under `docs/` or a `README.md` is next to it. Such a tree is an
   older tree, not a partial source to answer from.
3. **GitHub `develop` otherwise.** If there is no checkout, or if that file
   does not exist in the checkout, fetch the same paths from raw GitHub. Fetch
   the index first:

   ```text
   https://raw.githubusercontent.com/ROCm/rocm-libraries/develop/dnn-providers/integration-tests/docs/README.md
   ```

   Each other document that the index links is relative to that directory.
   Replace `README.md` in the URL with the linked file name. Fetch the text
   verbatim with `curl -fsSL <url>`. Use WebFetch only when there is no shell.
   Ask WebFetch for the complete raw text, because a summary loses the exact
   strings that this skill matches against. Tell the developer that the text
   came from `develop` and can differ from their tree.
4. **Neither reachable.** If neither source is reachable, stop and say so. Do
   not substitute remembered content. Do not answer from other READMEs or
   documents that exist in the checkout.

Say once, in one line, which source you loaded: the checkout path and its branch
(`git -C <repo-root> rev-parse --abbrev-ref HEAD`), or GitHub `develop`.

Read the index first. It defines the terms, maps topics to documents, and
lists the signals that a run can print. Then read every document that the task
touches. If a task crosses areas, read all of the documents. For example,
"add a bundle and get the lane green" touches all of them. The documents link
to tool READMEs elsewhere in the tree and to RFCs. Follow those links the same
way: local first, then the same repository path on GitHub.

## 2. Work from the documents

- Answer and act from the loaded text. Name the document and section that you
  used.
- If the documents are silent, read the source files that they point to (they
  name the headers and scripts). Do not guess. Tell the developer that you
  read the source files.
- If a document and the code disagree, the code is what runs. Follow the code.
  Tell the developer which section is wrong. Offer the correction as a change
  to that document. Never offer it as a change to this skill.
- To build or run in a local superbuild, use the `hipdnn-superbuild-test`
  skill (target discovery, Windows DLL `PATH`). This skill interprets what
  those runs mean.
- **What you are allowed to change.** A request to add a bundle or update
  claims covers every step that the documents prescribe for it. This includes
  running `--write-support-claims` and the confirming run. If you notice
  anything while you do something else, including signals from an unrelated
  run, propose the exact command or edit. Do not act on it.

## 3. After every run: surface the signals

Each time you run the suite, or the developer shows you its output, check the
output against **every row** of the signals table in the index. Report each
signal that is present, with the next step that the documents prescribe.
Report the result the way the running document says. Never report an exit
code alone.

This applies also when the developer asked about something else. Before you do
anything else, give one short line that accounts for the whole table: the rows
present by name, then "absent: all other rows". This line makes sure that a
row that was not checked is not mistaken for a row that was. Some rows need
two parts of the output read together (a header line, or one counter set
beside another). These rows are the easiest to miss. Check them against the
summary blocks themselves, not the pass/fail line.

If the output that you have lacks the blocks that the documents say a run
prints, get the full output the way the running document describes. An
example is a CTest run that shows only pass/fail lines. Do this before you
decide that no signal is present. Tell the developer that you did this.

Call out `unclaimed_support` explicitly each time it appears in a
`SUPPORT CLAIM SUMMARY`. Do this even when the run is green, and even when the
developer asked about something else. Name the engine, arch and platform from
the `run` block of the summary, and the bundles and cases listed. Then give
the complete command that records them. This is the documented authoring
command with every argument filled in for this run and this checkout: test
article, engine, engine config, data directory (where the documented procedure
says it must point) and a filter that selects the listed bundles. Follow it
with the confirming run that the procedure requires. A link to the procedure
alone is not enough.

## 4. Keep this skill hollow

Do not add facts about the suite to this file. If the developer corrects how
the suite works, or you find that the documents are out of date or incomplete,
the fix belongs in `dnn-providers/integration-tests/docs/`. Propose that edit.
