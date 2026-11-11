# The A/B flow (StinkyTofu)

Read this when you verify a change. The goal is performance: does a change make a kernel faster,
slower, or leave it the same? StinkyTofu is compiler-like, so the same input gives the same asm. If the
asm is identical, performance is identical and no GPU run is needed. TensileLite loads the rocisa
binding that embeds StinkyTofu, so TensileLite generates the asm that is compared.

```text
tier 1  CPU  asm A/B, one result per config (TensileLite --cpu-only, baseline vs candidate;
             inputs and rule: compare.md)
               identical  -> same asm, same performance: that config is fine, skip tier 2
               different  -> pattern check, then tier 2 for that config
tier 2  GPU  perf A/B, only for configs where tier 1 found a difference.
             BEFORE it, read projects/hipblaslt/harness/PROFILING.md
             first check that the kernel under test gives correct results (see PROFILING.md),
             then compare perf
             a difference of at least one SD (run-to-run noise) needs a human check
```

Every comparison run records the baseline commit hash.

**Final check (local CI).** Finish the change using the A/B above. Then, before you submit a PR, run
the StinkyTofu tests locally (build, ctest, FileCheck; see `README.md` and `tests/README.md`). They
are fast, so run them locally and do not leave them to CI.

**Tier 1 needs a fresh binding.** TensileLite loads the rocisa binding, not the `invoke build` output.
After every StinkyTofu edit, run `invoke rocisa` in `projects/hipblaslt/tensilelite` (see its
`AGENTS.md`), in each worktree, or the comparison runs against stale output.

**The pattern check** is a filter, not a score. The patterns to check for come from the database
(`entries/`, see `DATABASE.md`), not from this repo. Each entry there has a source and a status. Record
the result as `skipped` (no database), `no entries` (database read, nothing matched), or a hit (see
`DATABASE.md`). A hit on a `confirmed` pattern rejects the candidate before it costs GPU time. A hit on
a `candidate` pattern only warns the developer. The check covers entries of kind `pattern`. Any other
entry that matches the task (a finding, an insight) is mentioned to the developer as context and never
rejects anything. A clean result means only "no known problem". The GPU decides the rest. Different asm
is not a performance win: only tier 2 can show a gain. Identical asm supports only "no performance
change".
