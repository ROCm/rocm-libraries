# Mutation Production Plan Review Findings

Review target: `work/tensilelite-characterization/PLAN-MUTATION-PRODUCTION.md`

Status: **not ready for end-to-end execution yet**. The plan is strong
conceptually, but the executable workflow sketch has concurrency and harness gaps
that can break the shared worktree or produce false readiness.

## Findings

### 1. High: script sketch violates the single-mutmut-instance rule

The plan correctly says never run two mutmut instances against the shared tree
(`PLAN-MUTATION-PRODUCTION.md:40`, `:257`), but Part B uses
`parallel(slice.map(... mutmut ...))` for Phase 1 (`:406`). That can launch
multiple mutmut runs against the same checkout/cache.

Required fix: replace this with a sequential loop, or one engine agent that runs
the full slice serially and returns all module populations.

### 2. High: Phase 3 says "serial" but the workflow does not enforce serialization

The plan says apply/run/revert must not run concurrently
(`PLAN-MUTATION-PRODUCTION.md:201-204`), but the `pipeline(...)` verifies
survivors independently (`:416-434`). A prompt saying "serial" is not a lock.

Required fix: make verification a single serial runner stage, or split it into
parallel reasoning followed by one verifier agent/script that applies mutants one
at a time.

### 3. High: leak check is cwd-sensitive and currently false-clean from repo root

The done criteria use:

```bash
git status --porcelain -- 'Tensile/*.py' 'Tensile/**/*.py'
```

from `PLAN-MUTATION-PRODUCTION.md:85`. From the workspace root this returns
nothing because `Tensile/` is under `projects/hipblaslt/tensilelite/`. Running
with `git -C projects/hipblaslt/tensilelite ...` does show the existing
`Tensile/Tests/common/config_helpers.py` change.

Required fix: pin cwd for all host git commands or use project-prefixed paths.

### 4. High: the "proven" verification harness is not yet generic for mutmut survivors

The plan relies on generalizing `wf/p7-survivor-kill.sh`
(`PLAN-MUTATION-PRODUCTION.md:191`), but the existing script is hard-coded to
two old P6 mutants and two files (`wf/p7-survivor-kill.sh:16`, `:22`). It also
applies search/replace strings, not arbitrary `mutmut show` diffs
(`wf/p7-survivor-kill.sh:32`).

Required fix: add a generic verifier interface before execution:

- `mutant_id`
- target file
- patch/apply command or normalized diff
- test node
- expected return-code contract
- revert assertion

### 5. Medium: mutmut scoping is underspecified

Phase 0 says `[mutmut] paths_to_mutate = the slice`
(`PLAN-MUTATION-PRODUCTION.md:146`), while Phase 1 says one agent runs mutmut
per module (`:157`). That needs an exact mechanism.

Required fix: choose one of:

- rewrite config per module serially;
- use a supported mutmut CLI override, if available in the installed version;
- run one slice-level mutmut pass and group results by module.

As written, a module agent may mutate the whole slice.

### 6. Medium: artifact schemas are incomplete for consumed outputs

The checklist says every consumed stage has a schema
(`PLAN-MUTATION-PRODUCTION.md:458`), but synthesis agents have no output schemas
(`:438-449`). `VERDICT` also does not require `base_rc`, `mut_rc`, `revert`, or
`notes` despite those being required evidence for kill proof (`:382-386`).

Required fix: make the verification evidence fields required where applicable
and add schemas for synthesis artifacts or make the final assemble step validate
their required headers/fields.

### 7. Medium: concurrent triage can still edit shared source

Distinct test files avoid test-file collisions (`PLAN-MUTATION-PRODUCTION.md:181`),
but `intentionally-unhelpful` survivors add `# pragma: no mutate` to source files
(`:178`). Multiple survivor agents can touch the same module concurrently.

Required fix: have triage produce proposed pragma edits, then apply pragmas in
one serialized source-edit stage.

## Readiness Summary

The slice choices and test directories are real, and the repo has the expected
`coverage-unit` env plus characterization tests. The high-level lifecycle is
complete: prep, execute, triage, verify, synthesize, and CI rollout are all
represented.

The blockers are in execution control. Before approving the workflow card,
revise Part B so mutation execution and mutation application are mechanically
serialized, fix the cwd-sensitive git checks, and add a generic mutmut survivor
verifier. After those changes, the plan should be close to end-to-end runnable.
