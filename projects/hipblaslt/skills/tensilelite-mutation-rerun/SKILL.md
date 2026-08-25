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

## Run one investigation

1. Choose one source module and a focused candidate test set.
2. Record the source and container state with `scripts/slice-preflight.sh`.
3. Use `scripts/pyproject-mutmut.sh backup` and `set` to configure the bounded
   campaign.
4. Run mutmut with an explicit worker limit. Inspect every result that was not
   killed with `mutmut show <id>` before deciding whether it exposes a gap.
5. Restore `pyproject.toml` and require `assert-clean` before survivor
   verification.
6. For a meaningful survivor, add a focused characterization test that passes
   on unchanged source and fails on the changed behavior. A new untracked test
   file can be verified directly; when an existing tracked test must change,
   use a separate clean worktree or a deliberate local commit first.
7. Verify that evidence with `scripts/mutmut-verify.sh`. Collection, usage,
   timeout, transport, interruption, and restoration failures are inconclusive;
   they never prove a kill.
8. Run `scripts/tests/run-selftests.sh` after changing any bundled helper.

## Report the outcome

Record the source commit, container image, mutmut version, selected source and
tests, complete result counts, characterization tests added, and any unresolved
survivors or infrastructure failures. End with one explicit state:

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
