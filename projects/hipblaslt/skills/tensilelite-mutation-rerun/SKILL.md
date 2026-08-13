---
name: tensilelite-mutation-rerun
description: Guide optional, offline TensileLite mutation analysis used to find behavioral coverage and assertion gaps that should be closed with focused characterization tests. Use when planning, running, resuming, or auditing a TensileLite mutmut campaign; inspecting a surviving mutant; verifying that a characterization test detects it; or recording reproducible mutation evidence.
---

# TensileLite Mutation Rerun

Use mutation testing only as an opt-in analysis technique for finding gaps in
TensileLite's characterization coverage. Treat surviving mutants as diagnostic
leads. When a survivor exposes an observable gap, make the durable result a
focused characterization test in the normal pytest suite.

Do not treat mutation scores or reports as CI or release gates. Do not change
production code merely to improve a mutation score. The scripts and Dockerfile
bundled here are executable documentation for maintainers and coding agents;
ordinary TensileLite builds and tests do not depend on them.

## Prepare

1. Read [references/execution.md](references/execution.md) before running a
   campaign. It contains the supported commands, manifest format, and recovery
   rules.
2. Use a Linux environment. On Windows, use WSL or a Linux container; native
   Windows is unsupported. The documented workflow uses the optional image in
   `scripts/Dockerfile`.
3. Start from a clean tracked worktree. Keep one verifier process active at a
   time because mutation is temporarily applied to that worktree.
4. Do not push, edit pull requests, or update external trackers unless the user
   explicitly authorizes that action.

## Choose the next target

When the target is not already specified, read
[references/prioritization.md](references/prioritization.md). Build a fixed,
reviewed candidate universe; use measured coverage and mutation evidence; keep
missing inputs explicitly pending; and record the human rationale for the next
module. Do not present an arbitrary weighted proxy as an objective ranking.

## Run one investigation

1. Define the slice, following
   [references/slice-planning.md](references/slice-planning.md): record the
   source version, target module, selected tests, container, worker limit, and
   output directory before editing any configuration. Assemble the test set
   with [references/covering-set.md](references/covering-set.md) and measure
   its coverage of that exact file. Treat the threshold as a scheduling
   heuristic, not as evidence that the tests detect each mutation. A run whose
   selected tests do not meet the reviewed threshold is deferred, not started.
2. Record the source and container state with `scripts/slice-preflight.sh`.
3. Use `scripts/pyproject-mutmut.sh backup` and `set` to configure the bounded
   campaign.
4. Run mutmut with an explicit worker limit. Review every result that was not
   killed, following [references/survivor-triage.md](references/survivor-triage.md).
   Save the complete set of mutant IDs before grouping work, inspect each one
   with `mutmut show <id>`, and keep exactly one review row per ID.
5. Restore `pyproject.toml` and require `assert-clean` before survivor
   verification.
6. For a meaningful survivor, add a focused characterization test that passes
   on unchanged source and fails on the changed behavior, following
   [references/test-authoring.md](references/test-authoring.md). A new untracked test
   file can be verified directly; when an existing tracked test must change,
   use a separate clean worktree or a deliberate local commit first.
7. Verify that evidence with `scripts/mutmut-verify.sh`. Collection, usage,
   timeout, transport, interruption, and restoration failures are inconclusive;
   they never prove a kill.
8. Run `scripts/tests/run-selftests.sh` after changing any bundled helper.
9. Compare a report against an earlier one only when their inputs are
   compatible, following
   [references/regression-comparison.md](references/regression-comparison.md).
   Require nonempty, unique mutant identities and an explicit decision for
   every missing or new mutant. Comparison does not block any automated GitHub
   check today; the repository has neither a standard report generator nor a
   workflow that consumes one.

## Report the outcome

Follow [references/reporting.md](references/reporting.md). Build the report from
saved mutmut output, the complete review table, verifier output, and the
restoration result, keeping mutmut's original statuses separate from review
decisions. Record the source commit, container image, mutmut version, selected
source and tests, complete result counts, characterization tests added, and any
unresolved survivors or infrastructure failures. End with one explicit state:

- **Unresolved survivor:** the exact mutation has not yet been classified;
  preserve its ID and evidence without calling it a coverage gap.
- **Gap found:** a survivor exposed missing behavioral coverage and a focused
  characterization test was added.
- **No gap demonstrated:** the reviewed mutants were already detected or a
  survivor was shown to be equivalent or otherwise unobservable; record that
  reasoning without treating the score as a project quality gate.
- **Inconclusive:** an environment, collection, timeout, tool, or restoration
  failure prevented trustworthy evidence. The verifier may label some of these
  failures `BAD`; neither label is valid kill evidence.
- **Blocked:** a named dependency or user decision is required.

Preserve unrelated work and never describe incomplete or partially counted
results as successful evidence.
