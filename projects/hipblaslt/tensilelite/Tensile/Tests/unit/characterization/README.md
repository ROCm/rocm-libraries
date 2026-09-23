# TensileLite characterization tests

This directory holds **characterization tests**: tests that pin down what the TensileLite Python code *does today*, so the behavior can be refactored later with confidence. They are not specification tests — they encode observed behavior, including latent bugs (which are flagged, not silently "fixed" in a test). The goal is a trustworthy net under the codegen / solution-derivation / config / toolchain-helper surface before any refactor touches it.

Companion docs in this directory:
- **`adr/`** — Architecture Decision Records: one short, append-only file per genuine decision (Nygard form). See **Architecture Decision Records (ADRs)** below.
- **`DECISIONS.md`** — the running registry: the at-a-glance catalog of modules accepted below the coverage bar, pinned latent bugs, accepted-equivalent mutants, and the few departures from the add-only rule.
- **`_codegen/GPU-MOCK.md`** — the GPU-less (`--cpu-only`) seam that makes the client/perf and device-probe paths exercisable without a GPU, and the synthetic-perf caveat that comes with it.

Everything you need to understand or extend the suite is in this directory. Per-module provenance for each characterized module — the public surface by tier, the determinism strategy, before→after coverage, and the mutation-testing outcome — is recorded in `DECISIONS.md` and the per-module commit history. Any accepted coverage ceiling or pinned-bug rationale gets a full writeup in an ADR under `adr/`, with a short pointer row in `DECISIONS.md`.

## Per-module protocol

Each module was characterized the same way, one atomic commit per module, **add-only** (no edits to the production source under test, so the goldens pin real behavior rather than our scaffolding):

1. **Target.** Pick the next module by coverage-per-effort. Write a short `target.md` (archived) describing the module, the public surface by tier, the determinism strategy, and the before→after coverage goal.
2. **Pin behavior.** Drive the public API with table-driven parametrization; snapshot each output with [syrupy](https://github.com/syrupy-project/syrupy) into `__snapshots__/*.ambr`. Pin raise paths with `pytest.raises` on the raised type/message. Where a module exposes a real latent bug, **pin the bug** (assert the current crash/wrong result), record an **ADR** under `adr/` with a filed defect, and index it in `DECISIONS.md` — rather than working around it.
3. **Determinism.** No RNG / clock / network / global-state leakage into a snapshot. Normalize incidental fields *in the test*, never by changing production code to make a golden stable. Deep-copy shared globals and reset `globalParameters` between tests.
4. **Measure.** Run path-mode coverage (below) for the module; aim ≥95% line. When a module can't reach the bar honestly (real fork/IPC paths, GPU/asm emit, integration-only builders), accept it below the bar: record an **ADR** under `adr/` explaining the structural reason, and index it with a short pointer row in `DECISIONS.md` — do not fake coverage.
5. **No regression.** Once per batch, run the full `-m unit` suite and confirm it stays green and whole-project coverage does not drop.
6. **Mutation (gap discovery).** Once a module is covered, optional mutation
   analysis looks for behavioral changes that its assertions do not detect.
   Survivors are diagnostic leads; confirmed gaps are closed with focused
   characterization tests (see **Mutation testing** below).

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

## Coverage floors + ratchet (CI enforcement)

A **floor** is a minimum level of coverage that CI will not let you fall below. There are floors at
two granularities, and every coverage run enforces both:

- **A whole-project floor**: one fixed percentage for the combined total, summed across every
  measured file.
- **A per-file floor**: one number per file, recorded in a committed baseline, so each file is held
  to its own line.

If the combined total, or any single file, drops below its floor, the run fails.

The **ratchet** is how the floors move: one way, up, and only on purpose. Picture a ratchet wrench;
it only turns one way. Each upward click locks in a new, higher floor you can no longer fall back
below. Raising the floors is a deliberate, occasional maintenance step: a reviewed PR that recomputes
per-file coverage, updates the baseline, and bumps the whole-project number. Everyday coverage gains
never lower anything; the floors advance only when someone clicks them up.

Why per-file floors, and not just a whole-project one? A single project-wide floor can be fooled. One
file can quietly lose its tests while another file's gains prop the average back up, so the combined
total still clears the floor. Per-file floors catch that: a drop in any single file fails the run
even when the overall number looks fine.

### What the coverage number counts: the union of two suites

The tests that run here are really two suites: the **characterization** tests (this directory) and
the **pure unit** tests (the rest of `Tensile/Tests/unit`). Coverage is measured on the **union** of
the two. A line counts as covered if *either* suite reaches it. The floors are measured on that
union, so a per-file floor pins "coverage from characterization or unit, whichever reaches this
line", not characterization alone.

### The characterization → unit migration, and what the floors do (and do not) show

Characterization tests are scaffolding. They pin today's behavior so the code can be refactored
safely, but the long-term goal is real unit tests that make the scaffolding unnecessary. Because
coverage is measured on the union, moving a line's protection from a characterization test to a unit
test does not change the number: the union still covers the line, so no floor moves. That is
deliberate. The per-file floors act as a **safety rail for the migration** rather than an obstacle:
you can retire a characterization test only once a unit test covers the same lines, because if you
remove the net before the replacement exists, coverage drops below the floor and CI stops you.

What the floors do **not** do is measure migration progress. They do not know, or care, whether a
covered line was reached by a characterization test or a unit test. The characterization-vs-unit
**summary card** (rendered by the coverage lane into the GitHub run summary) is what shows that. The
card splits every measurable statement into four buckets that sum to 100%: reached by both suites,
by characterization only, by unit only, or by no test at all. The *characterization-only* count is
the migration debt (lines still protected only by scaffolding, which should fall toward zero as unit
tests take over), and the *no test coverage* count is the untested surface behind the whole-project
floor.

Those buckets say how far the migration has come overall; the card's second table says **where** the
remaining work is. It lists the largest files by measurable statement count (biggest first, whether
or not the PR touched them) with each file's unit-suite and characterization-suite coverage side by
side, plus that file's characterization-only statement count. Ranking by size rather than by worst
percentage is deliberate: a 500-statement file at 60% hides far more untested code than a
20-statement file at 30%, so size is what picks the next refactor target. Read a row as one file's
migration state: a high *Char %* with a low *Unit %* is a file still leaning on scaffolding, and its
*Char only* number is the debt to convert. These percentages are statement-level shares of the
file's own statements, matching the bucket table above, so they run slightly higher than the
branch-inclusive whole-project numbers in the card's headline. The row count is adjustable via
`--top-files N` on `tools/coverage_split_summary.py` (default 20; `0` omits the table).

Both floors live in the `coverage-gate` tox environment, which judges the coverage data that
`coverage-unit` produced:

- **Whole-project floor** (AIHPBLAS-3877). The value lives in exactly one place: `fail_under` under
  `[tool.coverage.report]` in the top-level [`pyproject.toml`](../../../../pyproject.toml). The
  combined-coverage `coverage report` step fails the run when the combined total drops below it. The
  number is not duplicated in `tox.ini` or in CI YAML.
- **Per-file floors** (AIHPBLAS-3878). `coverage-baseline.json` (in this directory) holds each
  file's floor. `tools/coverage_ratchet.py check` fails the run if any file drops below its floor by
  more than the tolerance (see below). The failure message names every file that dropped and prints
  the one command that raises the floors on purpose.

#### Measuring and enforcing are separate envs

```bash
tox -e coverage-unit   # run both suites, union them, write the reports, print the table
tox -e coverage-gate   # judge those reports: fail below either floor
```

The TensileLite coverage GitHub Actions job runs both, in that order, and owns the floors.

The split exists because `coverage-unit` has a second caller. The Math CI Jenkins job
`tensilelite-unit-codecov` runs `tox -e coverage-unit` to produce the `coverage.xml` it uploads to
Codecov under the `TensileLite-Unit` flag. It predates these floors, it does not want them, and its
config lives in `ROCm/rocJenkins`, which this repo cannot change. While enforcement lived inside
`coverage-unit`, a floor miss failed that Jenkins stage too, and because the job runs under `set -e`
it also took the C++ `coverage-cpp` pass and both Codecov uploads down with it. Putting the gate in
its own env means a caller opts in by naming it and cannot trip it by accident.

`coverage-gate` runs no tests and builds no rocisa; it only reads `.coverage` and `coverage.json`.
So it takes seconds, and reproducing a CI floor failure locally does not mean re-running the suite:
run `coverage-unit` once, then re-run `coverage-gate` as often as you like.

#### The tolerance (noise buffer)

The tolerance is how far a file may slip below its floor before the run fails. It exists to absorb
measurement noise rather than real coverage loss, and there is more of that noise than you might
expect:

- **A merge that deletes covered code lowers the ratio** even though nothing became less tested.
  A real example: `develop` removed 7 covered statements from `Contractions.py`, moving it
  88.16% → 88.05% with identical missed statements and identical branch coverage.
- **Worker scheduling and toolchain differences** flip individual branch arcs between otherwise
  identical runs, because the suite runs under `pytest-xdist`.
- **A percentage-point tolerance does not scale with file size.** One branch arc in a 770-unit file
  is worth about 0.13 pp, so a 0.1 pp budget cannot absorb even a single-arc wobble in any file
  smaller than roughly 1000 measurable units.

It is therefore set deliberately wide, **1 pp**, while the gate first lands, so it fails on real
regressions instead of on noise. Both the committed baseline's `tolerance` field and
`DEFAULT_TOLERANCE` in `tools/coverage_ratchet.py` carry that value, and a unit test pins them
together: `check` reads the baseline's value while `update` writes the default, so if they drift
apart the next ratchet click would silently retune the gate. Tighten the buffer once the numbers
prove stable across a few `develop` merges; the whole-project floor is the backstop meanwhile.

Note that `update` rewrites the baseline's `_comment` and `tolerance` from the tool, so record
anything durable about them in `write_baseline` rather than hand-editing the JSON, which the next
regeneration would silently discard.

### Raising the floors (the ratchet)

Raising the floors is the ratchet click: a deliberate, reviewed step, never automatic. Most of the
time it is a genuine rise (new tests pushed coverage up, so you lock the gain in). A per-file drop is
a signal to decide first. If a file lost coverage because a test is missing, that is a real
regression; add the test rather than lowering its floor. Only when a drop is intentional (for
example, code was removed) do you reset that file's floor as part of the same reviewed change. The
tool holds you to that: `update` raises floors on its own, but it will not lower one unless you name
that file with `--allow-lower`.

A floor-raising PR is a small, behavior-neutral maintenance change. It should touch only
`coverage-baseline.json` (the per-file floors) and, when you also lift the whole-project floor,
`fail_under` in `pyproject.toml`. Nothing else.

1. **Get fresh numbers.** Run the lane so it writes a current `coverage.json`:

   ```bash
   tox -e coverage-unit    # writes coverage.json
   ```

2. **Raise the per-file floors.** Ratchet the baseline against that report:

   ```bash
   python Tensile/Tests/unit/characterization/tools/coverage_ratchet.py update --current coverage.json
   ```

   Each file's floor rises to its current coverage (rounded to two decimals), and a file with no
   floor yet is pinned at its current number. Floors never move down here: if any file measured
   lower, `update` writes nothing, names those files, and exits non-zero.

3. **Decide about any drop, then name it.** A refusal is the tool asking which of two situations you
   are in. A missing test is a real regression, so add the test rather than lowering the floor. Only
   when the drop is intentional (code was removed, say) do you lower that file, and you lower it by
   naming it:

   ```bash
   python Tensile/Tests/unit/characterization/tools/coverage_ratchet.py update \
       --current coverage.json \
       --allow-lower=Tensile/Components/Subtile/SubtileGREmit.py
   ```

   One `--allow-lower` per file, and it lowers only the files named. That is what keeps a run made
   to move one floor from quietly resetting all the others to whatever the coverage run on disk
   happened to measure.

4. **Review the baseline diff before you commit it.** Expect rises, plus exactly the reductions you
   named. The diff is the review artifact, so keep it readable, and say in the PR why each lowered
   floor was accepted.

5. **Optionally raise the whole-project floor.** If combined coverage has climbed with room to
   spare, bump `fail_under` in `pyproject.toml` toward the 80% target. Leave a small margin below the
   measured number (a point or two): the per-file `--tolerance` already absorbs run-to-run wobble for
   the per-file floors, but `fail_under` is an exact cutoff, so a floor set right at the current
   number can trip on normal noise.

6. **Commit and open the PR.** Commit the `coverage-baseline.json` (and, if changed, `pyproject.toml`)
   diff with a one-line rationale, for example "raise floors after landing the DataType tests". Never
   widen the tolerance or blank the baseline to go green.

> The committed baseline is **populated** (one entry per measured file) and the per-file floors are
> **active**. They were seeded from a real GPU-less `coverage-unit` run (with `rocisa` built);
> refresh them with the `update` command above whenever an intentional change moves coverage. A
> brand-new source file has no floor yet, so the per-file check ignores it until the next `update`
> records one. During that window the whole-project floor is the backstop.

## Snapshot / golden discipline (governance)

These `.ambr` goldens are the safety net for the TensileLite/hipBLASLt consolidation refactor. They only protect you if they change **deliberately, one reviewed diff at a time**. The cardinal rule: never blanket-regenerate.

An opt-in local **pre-commit hook** runs the unit + characterization tests affected by your staged change and, on a golden mismatch, prints a `--snapshot-update` command scoped to the failing node(s) — so you run the loop below before you ever push. Setup: see the TensileLite [`README.md`](../../../../README.md).

### When a characterization test fails on your change

1. **Did you intend to change this behavior?**
   - **No →** you found a regression. The golden is doing its job — fix your code; do not touch the `.ambr` files.
   - **Yes →** update only the affected node, then review the diff:

     ```bash
     pytest <node-id> --snapshot-update
     # e.g. Tensile/Tests/unit/characterization/DataType/test_datatype_char.py::test_foo
     ```

     Read every changed line in the `.ambr` diff and explain the behavior change in your PR description. If the change pins or flips a known-wrong behavior, record a new ADR under `adr/` (or supersede the existing one). A golden diff is a reviewed behavior change, not a chore.
2. **Never** run a bare, suite-wide `pytest --snapshot-update`. It silently rewrites every golden and destroys the net. That is, one could introduce a bug, update the goldens to make the tests green, and thereby *pin the bug* — rendering the tests useless. A CI guard that disallows bulk updates is planned; don't wait for it.
3. After recording, re-run the node **without** `--snapshot-update` twice — it must be byte-identical. Churn means the test isn't deterministic; fix it via the `{basename, err}` digest / canonicalization, not by re-recording.
4. For **stable archs** (gfx908/90a/942) a codegen golden change is a *signal* — treat a digest diff as a suspected compiler/codegen regression and justify it in an **ADR** (and the PR description) before committing the new golden. Newer, still-churning archs may keep a small number of compiler generations side by side.

### Legitimate bulk regeneration

A real mass update (e.g. an intended change to the snapshot format itself) is allowed, but it must be a **conscious, reviewed act**: do it in its own PR that touches nothing else, and explain why in the description. A planned CI guard will require an explicit label/marker for such PRs.

### Reviewer checklist

- [ ] Every `.ambr` change is scoped to the node(s) the PR intends to change (not a blanket regeneration), and each golden diff is explained in the PR description.
- [ ] A behavior change that pins or flips a known-wrong golden has a matching ADR under `adr/` (new, or superseding the prior one) with a defect link.

## Architecture Decision Records (ADRs)

A genuine decision — **pinning a known-wrong behavior**, **accepting a module below the coverage bar** for a structural reason, or **a departure from the add-only rule** — is recorded as an **ADR**: one short, append-only file under [`adr/`](adr/) in [Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions) form (`Status` / `Context` / `Decision` / `Consequences`), numbered (e.g. `adr/0001-pin-results-only-boolop-crash.md`). When the decision pins a behavior that looks wrong, the ADR carries a `Defect:` tracker link, and that defect tracks the eventual fix. This applies every time one of those three forks comes up, not just the once-per-module cases called out in steps 2 and 4 above — e.g. a one-off add-only departure (deleting verified-dead source) gets an ADR too, even though it isn't part of the per-module loop.

ADRs are **append-only and superseded, never edited in place**. If a later change flips a pinned-bug golden (the defect is fixed), update the golden in that PR and add a new ADR that supersedes the old one (set the old one's `Status:` to `Superseded by adr/NNNN`). Never fold a corrected or revised decision back into an existing ADR's body — write a new one and supersede.

**Accepted-equivalent mutants are the one exception below the ADR bar**: a mutation survivor that's genuinely unkillable is justified with a one-line `# pragma: no mutate` comment plus an entry in `DECISIONS.md`'s **Mutation testing** section (see below) — not an ADR. The rationale for a single equivalent mutant is rarely more than a sentence, so the full Context/Decision/Consequences form would be ceremony without content; if a mutant's justification grows into a real design discussion, that is itself a sign it belongs in an ADR instead.

`DECISIONS.md` is the **running registry**: an at-a-glance catalog with **one short row per decision** — title, `ADR:` link, `Defect:` link if any, and a one-to-two sentence summary — never the full rationale. The rationale (context, alternatives rejected, consequences) lives in the ADR and only the ADR; a catalog row that repeats it has drifted from this contract and should be trimmed back down when noticed. See [`adr/README.md`](adr/README.md) for the ADR format and template.

## Mutation testing

Mutation testing is used only as an optional, offline technique for finding gaps
in characterization coverage. A surviving mutant is not a product failure or a
CI result. When it exposes an observable gap, the lasting result is a focused
characterization test in this suite—not a mutation-score gate or a production
code change made only to improve the score.

The mutation slice is configured in `[tool.mutmut]` in `pyproject.toml` and can
be run explicitly through tox:

```bash
tox -e mutation-unit
```

It is not part of tox's default environment list or the project CI workflow.
The optional maintainer/agent procedure and safety helpers live in the
[`tensilelite-mutation-rerun` skill](../../../../../skills/tensilelite-mutation-rerun/SKILL.md).
Accepted equivalent mutants and every `# pragma: no mutate` are justified in
`DECISIONS.md` (a mutant is "killed" only if the suite passes clean, fails on
the mutant, and reverts cleanly).
