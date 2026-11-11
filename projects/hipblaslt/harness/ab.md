# The A/B flow (hipBLASLt and TensileLite)

Read this when you verify a change. The goal is performance: does a change make a kernel faster,
slower, or leave it the same? If the generated asm is identical, performance is identical and no GPU
run is needed.

```text
tier 1  CPU  asm A/B, one result per config (TensileLite --cpu-only, baseline vs candidate;
             inputs and rule: compare.md in ../../../shared/stinkytofu/harness/)
               identical  -> same asm, same performance: that config is fine, skip tier 2
               different  -> pattern check, then tier 2 for that config
tier 2  GPU  perf A/B, only for configs where tier 1 found a difference.
             BEFORE it, read PROFILING.md
             first check that the kernel under test gives correct results (see PROFILING.md),
             then compare perf
             a difference of at least one SD (run-to-run noise) needs a human check
```

Every comparison run records the baseline commit hash.

**Final check (local CI).** Finish the change using the A/B above. Then, before you submit a PR, run the
checks that CI will run for the paths you changed, as far as they are quick. The full test suites
take hours, so you can leave them to CI on the PR and wait for the result. Commands come from
`../TESTING.md` and the `AGENTS.md` files.

**The pattern check** is a filter, not a score. The patterns to check for come from the database
(`entries/`, see `DATABASE.md` in `../../../shared/stinkytofu/harness/`), not from this repo. Each entry
there has a source and a status. Record the result as `skipped` (no database), `no entries` (database
read, nothing matched), or a hit. A hit on a `confirmed` pattern rejects the candidate before it costs
GPU time. A hit on a `candidate` pattern only warns the developer. The check covers entries of kind
`pattern`. Any other entry that matches the task (a finding, an insight) is mentioned to the developer
as context and never rejects anything. A clean result means only "no known problem". The GPU decides
the rest. Different asm is not a performance win: only tier 2 can show a gain. Identical asm supports
only "no performance change". Pull requests are not benchmarked in CI today (see `../TESTING.md`), so
tier 2 is where a performance claim is actually checked.

**Snapshot tests (characterization goldens).**
- A snapshot of generated asm that fails only because the asm changed may be updated once the perf
  A/B shows a gain. Update only the nodes the change touches, and show the developer the diff.
- A snapshot of an API output guards behavior that callers depend on. If a change alters it, stop
  and ask the developer for a review. Do not update it to get CI green.
