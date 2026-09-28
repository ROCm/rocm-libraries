---
name: hipdnn-integration-testing
description: "Run, read, and extend hipDNN's cross-provider integration test suite (dnn-providers/integration-tests): bundles and template sweeps, golden data and DVC, .support.json claim sidecars, per-engine TOML, tier YAML, CTest lanes. Doc-driven: loads all of its knowledge at run time from the suite's human-readable docs in dnn-providers/integration-tests/docs/ (local checkout first, GitHub develop as fallback). Use when adding or updating bundles or support claims, running or triaging hipdnn_integration_tests, or when a run prints a SUPPORT CLAIM SUMMARY with unclaimed_support, CLAIM_BROKEN or failed_in_use, 'zero tests ran', or an all-skipped result."
argument-hint: "[task: run|read-output|add-bundle|update-claims|file-formats|triage] [engine: MIOPEN_ENGINE|HIPBLASLT_ENGINE|HIP_MLOPS_ENGINE|ASM_SDPA_ENGINE|...]"
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

1. **Local checkout first.** Find the repository root with:

   ```bash
   git rev-parse --show-toplevel
   ```

   Call that `<repo-root>`. If `<repo-root>/dnn-providers/integration-tests/docs/README.md`
   exists, read the documents from there. A local copy describes the code the
   developer is actually building, including unmerged changes on their branch,
   so it wins over any remote copy.
2. **GitHub `develop` otherwise.** With no checkout, or a checkout that predates
   the `docs/` directory, fetch the same paths from raw GitHub — the index first:

   ```text
   https://raw.githubusercontent.com/ROCm/rocm-libraries/develop/dnn-providers/integration-tests/docs/README.md
   ```

   Fetch each further document by replacing `README.md` with the file name the
   index links to (WebFetch, or `curl -fsSL` with that URL). Tell the developer
   the text came from `develop` and may not match their tree.
3. **Neither reachable** → stop and say so. Do not substitute remembered
   content.

Say once, in a line, which source you loaded (checkout path and branch, or
GitHub `develop`).

Read the index first. It maps topics to documents and lists the signals a run
can print. Then read every document the task touches; when a task crosses
areas — "add a bundle and get the lane green" touches all of them — read them
all, they are short. The documents link to tool READMEs elsewhere in the tree
(migration scripts, reference-data scripts) and to RFCs; follow those the same
way, local first, then the same repository path on GitHub.

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
- Do not change bundles, sidecars, golden data or engine TOML — including by
  running `--write-support-claims`, which edits the source tree — unless the
  developer asked for it. Otherwise propose the exact command or edit and let
  them decide.

## 3. After every run: surface the signals

Whenever you run the suite, or the developer shows you its output, check the
output against **every row** of the signals table in the index and report each
signal that is present, with the next step the documents prescribe. Report the
result the way the running document says to (the counts it names), never an
exit code on its own.

Call out `unclaimed_support` explicitly every time it appears in a
`SUPPORT CLAIM SUMMARY`, even when the run is green and even when the developer
asked about something else. It is the one signal that never fails a run, so it
is routinely missed. Name the engine, arch and platform from the summary's
`run` block and the bundles and cases listed, and point the developer at the
documented procedure for updating the sidecars.

## 4. Keep this skill hollow

Do not add facts about the suite to this file. If the developer corrects how
the suite works, or you find the documents stale or missing something, the fix
belongs in `dnn-providers/integration-tests/docs/`; propose that edit.
