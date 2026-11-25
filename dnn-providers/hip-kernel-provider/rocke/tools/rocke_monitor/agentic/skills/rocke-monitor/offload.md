<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Review offload: the reviewer model reviews, you audit

The review offload is `offload.mode` (<OFFLOAD_MODE> here). With it on, the reviewer model (`<WORKER_MODEL>`) does
each review's substance, first rounds included, in four passes of its constrained tool loop
(`<PACKAGE_DIR>/worker/offload_agent.py`):

| Pass | What it writes |
|---|---|
| draft | `draft.md`: its own findings (N1, ...), the checks it ran (D1, ...) and the utility line |
| quality | `quality.md`: the code quality pass, by the code quality reviewer's instructions rendered for it (Q1, ...) |
| regrade | `regrade.md`: every prior item at this head, by the verifier's §3 (P1, ...), and the saved probes it ran |
| assemble | after the checks, `post_draft.md` and `notes_draft.md`: the post in the verifier's Output Format, its `map:` lines |

Every claim carries its evidence in `evidence.jsonl`: a tool run (`runs.jsonl`: the exact command, its exit, a digest
and an excerpt of its output) or a `file:line` quote, checked when it was recorded. Its own probes and mutants ran in a
sandbox (no network, nothing writable but its review folder). Then `rk evidence-check N --offload` re-reads every quote
and re-runs the cheap commands, with no model, and writes `evidence_check.md`: PASS or FAIL per claim, the claims with
no evidence, and your spot-check sample. You audit; you re-derive a review only when it goes deep.

## On an audit's `REVIEW_READY`

The tick's line names the review's kind, then `audit`: `REVIEW_READY #101 (follow-up, audit)`.

1. **Prepare.** Run `rk review-prep N` as for any review (`review_procedure.md`, step 1). Read `evidence_check.md`
   first, then `post_draft.md` and `notes_draft.md`; the other files are there when a claim needs them.
2. **Dispatch the rocKE Review Verifier for the audit:** agent type `rocke-review-verifier` with no `model` option (its
   file runs `<AUDIT_MODEL>`), effort `<AUDIT_EFFORT>`, in the background, with a brief that says **offload audit** and
   names the review folder, the review directory, `evidence_check.md`, `post_draft.md`, `notes_draft.md`,
   `checks/checks.md`, `facts.txt`, `classify.txt` and line 1. Its team file's "Offload audit" section is its checklist.
   Do not dispatch the code quality reviewer: the reviewer model ran the code quality pass, and the audit checks its
   findings.
3. **Escalate to a deep review** when the verifier's `deep:` line or your own reading says so: claims that contradict
   each other or the checks, a draft that does not fit the diff (findings on files the PR does not touch, a summary of
   another change), a 🔴/🟠 whose evidence the audit could not confirm, a failed spot check, or prior items the
   re-grade missed. Say why in one journal line (`rk log`), then follow "Deep review" below.
4. **Check, post and record** as `review_procedure.md` steps 3–5 say, with these changes: the post is the verifier's
   audited post (`pr101_fu2.md`: the PR and the round); its footer reads "review by the reviewer model
   (`<WORKER_MODEL>`), audited by <AUDIT_NAME>"; the spec's `SAVE` is the count and a short list from the verifier's
   notes (`"2 reviewer-model claims corrected: N3, N5"`; `"0 reviewer-model claims corrected"` when none were). A probe
   of the reviewer model's (the review folder's `probes/`) that backs a 🔴/🟠 you post is saved for the next round like
   any other.

## Deep review

The tick's line names the review's kind, then `deep:` and the reasons: `REVIEW_READY #101 (first-round, deep: …)`. The
full review of `review_procedure.md`, steps 1–5: the verifier and the code quality reviewer (the lane), both on
`<DEEP_MODEL>` at effort `<DEEP_EFFORT>`, the footer naming <DEEP_NAME>. The lane's agent file runs `<DEEP_MODEL>`, so
the lane gets no `model` option; the verifier's runs `<AUDIT_MODEL>`: when that is not `<DEEP_MODEL>`, dispatch the
verifier with model `<DEEP_ALIAS>`. Give the verifier the description `Verify #N <round>, deep review` (for the second
follow-up of #101, `Verify #101 fu2, deep review`), so `rk spend` counts its calls as asked for `<DEEP_MODEL>`. Give
both the reviewer model's files as extra inputs (`post_draft.md` or `draft.md`, `quality.md`, `regrade.md`,
`evidence_check.md`): a colleague's draft to verify, never evidence. A review goes deep when:
- its evidence check failed (a quote not at its line, a re-run that differs) or a 🔴/🟠 claim has none;
- a pass wrote nothing (the reviewer model went down mid-review);
- it is a first round over `offload.deep_lines` changed lines or `offload.deep_files` files, or one touching the hot
  spots in `offload.deep_hot_spots` or the paths in `offload.deep_paths` (the ABI, CMake, the hubs);
- the audit escalates it (step 3 above).

## The trial (`offload.mode` shadow)

With `offload.mode` shadow, every review runs as `review_procedure.md` says and posts as always. The offload passes
also run for it, into `workers/pr<N>/offload/`, and post nothing; the finish job checks their evidence and keeps the
run beside the verifiers' notes (`offload/pr101_fu2/` in the lessons folder). After our post,
`rk shadow-report N` compares the two: matched, missed and extra findings with their evidence checks, both header
lights, and whether the offload would have been audited or reviewed in depth. `rk shadow-report` alone writes the
summary over the last `offload.trial_prs` PRs (`<OUTPUTS_DIR>/rocke_shadow/summary.md`). Never post or quote a shadow
file.
