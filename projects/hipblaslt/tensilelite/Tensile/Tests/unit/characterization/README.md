# TensileLite characterization tests

This directory holds **characterization tests**: tests that pin down what the TensileLite Python code *does today*, so the behavior can be refactored later with confidence. They are not specification tests — they encode observed behavior, including latent bugs (which are flagged, not silently "fixed" in a test). The goal is a trustworthy net under the codegen / solution-derivation / config / toolchain-helper surface before any refactor touches it.

Companion docs in this directory:
- **`DECISIONS.md`** — why modules are accepted below the coverage bar, which latent bugs are pinned, which mutants are accepted as equivalent, and the few departures from the add-only rule.
- **`_codegen/GPU-MOCK.md`** — the GPU-less (`--cpu-only`) seam that makes the client/perf and device-probe paths exercisable without a GPU, and the synthetic-perf caveat that comes with it.

Everything you need to understand or extend the suite is in this directory. Additional background and the campaign's full per-module provenance are tracked under AIHPBLAS-3871.

## Per-module protocol

Each module was characterized the same way, one atomic commit per module, **add-only** (no edits to the production source under test, so the goldens pin real behavior rather than our scaffolding):

1. **Target.** Pick the next module by coverage-per-effort. Write a short `target.md` (archived) describing the module, the public surface by tier, the determinism strategy, and the before→after coverage goal.
2. **Pin behavior.** Drive the public API with table-driven parametrization; snapshot each output with [syrupy](https://github.com/syrupy-project/syrupy) into `__snapshots__/*.ambr`. Pin raise paths with `pytest.raises` on the raised type/message. Where a module exposes a real latent bug, **pin the bug** (assert the current crash/wrong result) and record it in `DECISIONS.md` rather than working around it.
3. **Determinism.** No RNG / clock / network / global-state leakage into a snapshot. Normalize incidental fields *in the test*, never by changing production code to make a golden stable. Deep-copy shared globals and reset `globalParameters` between tests.
4. **Measure.** Run path-mode coverage (below) for the module; aim ≥95% line. When a module can't reach the bar honestly (real fork/IPC paths, GPU/asm emit, integration-only builders), accept it below the bar and document the ceiling in `DECISIONS.md` — do not fake coverage.
5. **No regression.** Once per batch, run the full `-m unit` suite and confirm it stays green and whole-project coverage does not drop.
6. **Mutation (widening).** Once a module is covered, mutation testing certifies the assertions actually *catch* changes; survivors are triaged and killed (see **Mutation testing** below).

### Directory layout

- One subdirectory per characterized module (e.g. `DataType/`, `LibraryIO/`, `Configuration/`, `TensileLogic/`, …), each with its `test_*_char.py` files and a local `__snapshots__/`.
- `_codegen/` — the codegen record/replay harness (`codegen_harness.py`, `config_harness.py`, `matrix.py`), the per-arch attribution fixtures, and `GPU-MOCK.md`. Codegen goldens use an order-invariant `{basename, err}` digest rather than a full assembly-text hash, because the emitter's text is order-coupled through process-global rocisa scheduler state.
- `conftest.py` — shared fixtures for the characterization suite.

## How to run

These tests require the standard TensileLite dev environment with **`rocisa` built** (`invoke rocisa`, per the top-level `README.md`). The suite imports the compiled `rocisa` extension; it will not collect without it.

Run the full unit suite (characterization tests are marked `-m unit` and collected by the existing `testpaths`):

```bash
pytest -m unit Tensile/Tests/unit
```

### Coverage is path-mode

Always measure coverage with `--cov=Tensile` — a **filesystem path**, never a dotted module name — combined across `-n4` xdist workers:

```bash
pytest -m unit -n4 --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit
```

A dotted `--cov` target (e.g. `--cov=Tensile.Common.DataType`) re-imports `rocisa` and SIGABRTs on duplicate nanobind registration. To read a single module's row, grep the term-missing output (the single-file path prefix does not filter the report):

```bash
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep "Common/DataType.py"
```

Line coverage = `(Stmts - Miss) / Stmts`.

## Snapshot / golden discipline (governance)

These `.ambr` goldens are the safety net for the TensileLite/hipBLASLt consolidation refactor. They only protect you if they change **deliberately, one reviewed diff at a time**. The cardinal rule: never blanket-regenerate.

### When a characterization test fails on your change

1. **Did you intend to change this behavior?**
   - **No →** you found a regression. The golden is doing its job — fix your code; do not touch the `.ambr` files.
   - **Yes →** update only the affected node, then review the diff:

     ```bash
     pytest <node-id> --snapshot-update
     # e.g. Tensile/Tests/unit/characterization/DataType/test_datatype_char.py::test_foo
     ```

     Read every changed line in the `.ambr` diff and explain the behavior change in your PR description. A golden diff is a reviewed behavior change, not a chore.
2. **Never** run a bare, suite-wide `pytest --snapshot-update`. It silently rewrites every golden and destroys the net — a green run that proves nothing. (A CI guard enforcing this is coming under AIHPBLAS-3876; don't wait for it.)
3. After recording, re-run the node **without** `--snapshot-update` twice — it must be byte-identical. Churn means the test isn't deterministic; fix it via the `{basename, err}` digest / canonicalization, not by re-recording.
4. For **stable archs** (gfx908/90a/942) a codegen golden change is a *signal* — treat a digest diff as a suspected compiler/codegen regression and justify it in `DECISIONS.md` (and the PR description) before committing the new golden. Newer, still-churning archs may keep a small number of compiler generations side by side.

### Legitimate bulk regeneration

A real mass update (e.g. an intended change to the snapshot format itself) is allowed, but it must be a **conscious, reviewed act**: do it in its own PR that touches nothing else, and explain why in the description. The forthcoming CI guard (AIHPBLAS-3876) will require an explicit label/marker for such PRs.

### Reviewer checklist

- [ ] Every `.ambr` change is scoped to the node(s) the PR intends to change (not a blanket regeneration), and each golden diff is explained in the PR description.

### Resources

- Project overview & snapshot governance (Confluence): _link pending publish_
- Tracking: AIHPBLAS-3867 (Lock the net), AIHPBLAS-3875 (this policy + helper script), AIHPBLAS-3876 (CI blanket-regeneration guard).

## Mutation testing

The mutation slice is configured in `[tool.mutmut]` in `pyproject.toml` and run through tox:

```bash
tox -e mutation-unit
```

It is currently a narrow **report-only pilot** (five files); widening across the critical modules is planned. Accepted equivalent mutants and every `# pragma: no mutate` are justified in `DECISIONS.md` (a mutant is "killed" only if the suite passes clean, fails on the mutant, and reverts cleanly).
