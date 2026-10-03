---
name: hipdnn-integration-testing
description: "Run, read, and extend hipDNN's cross-provider integration test suite (dnn-providers/integration-tests): bundles and sweeps, golden data and DVC, .support.json claim sidecars, per-engine TOML, tier YAML, CTest lanes. Doc-driven: loads all of its knowledge at run time from the suite's human-readable docs in dnn-providers/integration-tests/docs/ (local checkout first, GitHub develop as fallback). Use when adding or updating bundles or support claims, running or triaging hipdnn_integration_tests, or when a run prints a SUPPORT CLAIM SUMMARY with unclaimed_support, CLAIM_BROKEN or failed_in_use, 'zero tests ran', or an all-skipped result."
argument-hint: "[task: run|read-output|add-bundle|update-claims|file-formats|triage] [engine name]"
allowed-tools: Bash, Read, Grep, Glob, WebFetch
---

# hipDNN Integration Testing (doc-driven)

This skill carries no knowledge of the suite itself. File formats, how to run
it, how to read its output, and how to add bundles and update support claims
all live in human-readable documents in the repository, written for
developers. This skill loads those documents, applies them, and makes sure the
developer hears what a run is asking of them. When the documents change, the
skill's behavior changes with them; nothing here needs editing.

## 1. Load the documents — every invocation

The documents live in `dnn-providers/integration-tests/docs/` under the
rocm-libraries root, with `dnn-providers/integration-tests/docs/README.md` as the
entry point. Never answer from memory or from an earlier session: load them
fresh.

1. **Find the checkout the developer is working in**, and call its root
   `<repo-root>`. Try, in order:
   - a checkout or worktree the developer named;
   - the current directory: `git rev-parse --show-toplevel`;
   - a build directory in play: its `CMakeCache.txt` records the source tree
     as `CMAKE_HOME_DIRECTORY`.
2. **Local documents first.** If `<repo-root>/dnn-providers/integration-tests/docs/README.md`
   exists, read the documents from there. A local copy describes the code the
   developer is actually building, including unmerged changes on their branch,
   so it wins over any remote copy.
3. **GitHub `develop` otherwise.** If there is no checkout, or that file does not
   exist in it, fetch the same paths from raw GitHub — the index first:

   ```text
   https://raw.githubusercontent.com/ROCm/rocm-libraries/develop/dnn-providers/integration-tests/docs/README.md
   ```

   Every other document the index links is relative to that directory: replace
   `README.md` in the URL with the linked file name. Fetch verbatim with
   `curl -fsSL <url>`; use WebFetch only when there is no shell, and ask it for
   the complete raw text, since a summary loses the exact strings this skill
   matches against. Tell the developer the text came from `develop` and may not
   match their tree.
4. **Neither reachable** → stop and say so. Do not substitute remembered
   content.

Say once, in a line, which source you loaded (checkout path and branch, or
GitHub `develop`).

Read the index first. It defines the terms, maps topics to documents, and
lists the signals a run can print. Then read every document the task touches;
when a task crosses areas — "add a bundle and get the lane green" touches all
of them — read them all. The documents link to tool READMEs elsewhere in the
tree and to RFCs; follow those the same way, local first, then the same
repository path on GitHub.

## 2. Work from the documents

- Answer and act from the loaded text and name the document and section you
  relied on.
- Where the documents are silent, read the source files they point to (they
  name the headers and scripts) instead of guessing, and say that you did.
- Where a document and the code disagree, the code is what runs. Follow the
  code, tell the developer which section is wrong, and offer the correction as
  a change to that document — never as a change to this skill.
- To build or execute in a local superbuild, use the `hipdnn-superbuild-test`
  skill (target discovery, Windows DLL `PATH`); this skill interprets what those
  runs mean.
- **What you may change.** A request to add a bundle or update claims covers
  every step the documents prescribe for it, including running
  `--write-support-claims` and the confirming run. Anything you notice while
  doing something else — including signals from an unrelated run — gets a
  proposal with the exact command or edit, not an action.

## 3. After every run: surface the signals

Whenever you run the suite, or the developer shows you its output, check the
output against **every row** of the signals table in the index and report each
signal that is present, with the next step the documents prescribe. Report the
result the way the running document says to, never as an exit code alone.

If the output you have lacks the blocks the documents say a run prints — for
example a CTest run that shows only pass/fail lines — obtain the full output
the way the running document describes before concluding that no signal is
present, and say that you did.

Call out `unclaimed_support` explicitly every time it appears in a
`SUPPORT CLAIM SUMMARY`, even when the run is green and even when the developer
asked about something else. Name the engine, arch and platform from the
summary's `run` block and the bundles and cases listed, and point the developer
at the documented procedure for updating the sidecars.

## 4. Keep this skill hollow

Do not add facts about the suite to this file. If the developer corrects how
the suite works, or you find the documents stale or missing something, the fix
belongs in `dnn-providers/integration-tests/docs/`; propose that edit.
