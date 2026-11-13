# Facts (`FACTS`)

You establish the round's PR-level facts once, so no lane has to rediscover them. Lanes read `facts.md`, cite a fact by its ID,
and file only what the fact does not state; `ORCH` files the facts that are findings. State facts only: never judge the code.

Write only `$EVIDENCE/r<N>/facts.md` and `$EVIDENCE/r<N>/facts-evidence/`. Run beside Orient, after *Pin and intake*.

## What to establish

Each fact is a numbered block (`F1`, `F2`, ...):

```markdown
F<n>:      one factual sentence, with the number or name that makes it checkable
Source:    the command, CI job URL, log excerpt as $EVIDENCE/r<N>/facts-evidence/<file>, or path:line at $HEAD
Proposed:  finding <kind> <tier> (a failing check, a stale description, a test no CI run selects) | context only
Concerns:  the lanes that should build on it (RUN, TESTQ, ARCH, ...)
```

1. **CI at `$HEAD`.** From `pr/checks.tsv`: every failed check, and every skipped check that would have run the changed code. For a
   failed test job save the failing log excerpt (suite, test name, failure count, first failure line) and say whether the same job
   passed on the target's latest run. Name the cause in one line only when the log states it. Count checks that executed tests.
2. **Test selection.** For each added or renamed test, the tier filter or label that selects it in PR CI (`test_categories.yaml`,
   `ctest -L`, the job's `--gtest_filter`) and the number of times its name appears in the saved job logs. Zero is a fact.
3. **Description against commits.** Compare `pr/body.md` with every commit message in `commits.txt` and the file list: claims about
   behavior, scope, test counts and results that a later commit changed or that the files contradict.
4. **Head and relatives.** Whether the live PR head equals `$HEAD`; open PRs that touch the same functions or files (search by path);
   the target tip drift summary from `drift.txt`.
5. **Process presence.** Tracker key and links present in the body (presence only; `AUTH` resolves them), checklist items ticked.

## Rules

- A fact needs a source another agent can re-read. Nothing is `[INFERENCE]` here: leave out what you cannot show.
- Do not read code to interpret a failure beyond what the log says; the lanes do that.
- Keep it to one page plus evidence files; link evidence, never paste logs.
