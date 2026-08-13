---
name: tensilelite-mutation-rerun
description: Guide reproducible, fail-closed TensileLite mutation-testing reruns with mutmut. Use when Codex is asked to plan, rerun, resume, or audit a TensileLite mutation campaign; select and validate covering tests for a target module; inspect or kill surviving mutants; reproduce AIHPBLAS-3868 results; or hand off mutation evidence. The actual mutmut run requires Linux or WSL, normally through Docker.
---

# TensileLite Mutation Rerun

Treat mutmut as the mutation engine and the repository tooling as the
TensileLite-specific reproducibility contract. Mutmut does not require wrappers
to start; use the wrappers to preserve environment provenance, per-slice
configuration, strict kill semantics, and restoration across reruns.

## Establish the environment

1. Resolve the repository root and the `projects/hipblaslt/tensilelite` source
   directory.
2. Read `projects/hipblaslt/tensilelite/Tensile/Tests/unit/mutation/README.md`
   and the `[tool.mutmut]` table in `projects/hipblaslt/tensilelite/pyproject.toml`.
3. Require Linux or WSL. Mutmut 3.6 exits on native Windows and relies on
   `fork`, Unix resource limits, and Unix signals. A Windows host may drive a
   Linux container or WSL, but do not claim native-Windows support.
4. Identify the existing mutation container and its `/work` mount. Ask for the
   container name if it cannot be discovered safely.
5. Do not push, edit PRs, or update Jira unless the user explicitly authorizes
   that external operation.

## Run one mutation slice

1. **Define and validate the slice.** Read
   [references/slice-planning.md](references/slice-planning.md). Pin the source
   SHA, target module(s), reviewed test selection, container, worker bound, and
   artifact directory before changing configuration. Keep the slice small
   enough to explain and recount independently.
2. **Build and validate a covering set.** Read
   [references/covering-set.md](references/covering-set.md) and stop unless the
   exact target-module row meets the threshold with pytest exit status 0.
3. **Record preflight provenance.** Run `slice-preflight.sh` with the slice ID,
   target module, container, and a slice-specific output directory. Stop on
   dirty tracked source or a missing container.
4. **Back up configuration.** Run `pyproject-mutmut.sh backup`, then set the
   reviewed `only_mutate` and `pytest_add_cli_args_test_selection` values. Never
   change `mutate_only_covered_lines = false` without first resolving the rocisa
   unload/re-import crash documented in `pyproject.toml`.
5. **Prove the clean baseline.** Run the selected tests before mutmut. A failing
   baseline invalidates the slice; do not classify mutants from that run.
6. **Run mutmut with bounded concurrency.** Use `mutmut run --max-children 32`
   unless a reviewed slice record requires a lower value. Capture the command,
   source SHA, container image, mutmut version, exit status, and result counts.
7. **Triage every non-killed result.** Read
   [references/survivor-triage.md](references/survivor-triage.md). Capture the
   complete input ID set before grouping, inspect each mutant with `mutmut show`,
   and maintain a one-row-per-mutant conservation ledger. Treat `survived`,
   `no tests`, `timeout`, and suspicious/infrastructure outcomes separately.
8. **Add behavior-distinguishing tests.** Read
   [references/test-authoring.md](references/test-authoring.md). For each
   proposed assertion, state what source mutation makes it fail. Do not add
   coverage-only assertions, silently skip tests, or idealize behavior that the
   current source does not implement.
9. **Verify claimed kills.** Rerun the named mutant with one child and, when a
   manifest is available, use `mutmut-verify.sh`. Count a kill only when the
   clean node passes, the mutated node returns pytest assertion status 1, and
   source restoration is clean. Collection, usage, internal, timeout, and
   interruption errors are inconclusive, never kills.
10. **Audit equivalent mutations skeptically.** Require a concrete invariant or
    proof over valid inputs. Do not add `pragma: no mutate` merely because a
    distinguishing test is inconvenient.
11. **Restore in all outcomes.** Run `pyproject-mutmut.sh restore` and
    `assert-clean`, then confirm no tracked source mutation remains. Restoration
    is required after success, failure, or interruption.
12. **Write the handoff.** Report the target, exact test selection, source and
    environment provenance, baseline status, total/killed/survived/no-test/
    timeout/inconclusive/equivalent counts, tests added, residual risks, and
    artifact paths. Never inflate a score by dropping inconclusive outcomes.

## Guardrails

- Keep mutation application and source restoration serial.
- Require exact survivor-ledger conservation before reporting results.
- Never use aggregate package `TOTAL` coverage as proof for a target module.
- Never accept coverage from a failing pytest run.
- Never silently substitute the full unit suite for an unproven focused set.
- Keep `--max-children` explicit; host CPU count is not a reproducible setting.
- Do not blanket-update syrupy snapshots. Follow the hipBLASLt golden discipline
  and update only reviewed nodes.
- Preserve unrelated dirty or untracked user files.
- Prefer explicit, committed slice selections over rediscovering them on every
  rerun. Revalidate after the source or test tree changes.

## Required outcome

End with one of these explicit states:

- **Certified:** baseline green, covering threshold met, run accounted for,
  claimed kills verified, residual equivalents justified, and worktree clean.
- **Deferred:** covering threshold not met or the target row is missing.
- **Inconclusive:** environment, collection, timeout, or restoration failure
  prevents a trustworthy score.
- **Blocked:** a specific external dependency or user decision is required.

Do not label an incomplete or partially accounted run certified.
