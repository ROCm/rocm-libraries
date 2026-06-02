# Next goal — characterization tests for `Tensile/LibraryIO.py`

Follow-up to the completed `TensileLogic/` suite. Same pattern, next module.
This file is the self-contained kickoff; it assumes the env and conventions
already established (see `prompt.md`, `env/README.md`, `HANDOFF.md`, and the
two completed targets under
`Tensile/Tests/unit/characterization/{Validators,TensileLogic}/`).

## GOAL

Add a characterization-test suite that pins current behaviour of
`Tensile/LibraryIO.py` with **≥95% line coverage on the module**, using syrupy
(reuse `Tensile/Tests/unit/characterization/survey.md`, no new survey),
integrated with the existing pytest/tox setup. Document resisting functions +
a go/no-go on the *next* target after this one.

Achieved when:
- **Baseline (BEFORE) coverage captured FIRST**, before adding any new tests —
  `LibraryIO.py` coverage from the *existing* suite — saved as a committed
  artifact (`coverage-before.txt`, see below).
- `target.md` for LibraryIO (new file in the LibraryIO test dir, per add-only).
- `Tensile/Tests/unit/characterization/LibraryIO/` exists with syrupy snapshots
  driving every reasonably-testable public function (see API inventory).
- Determinism handled: filesystem paths normalised; timestamps/versions in any
  written output normalised or pinned; round-trips compared structurally, not
  via raw bytes where a version string leaks.
- `pytest --cov` (path-mode) reports **≥95% line** on the module.
- **Final (AFTER) coverage captured** (`coverage-after.txt`) with before→after
  delta recorded in `target.md`.
- `resistance.md` (new file in the LibraryIO dir) lists every resisting fn +
  reason + workaround.
- `recommendations.md` (new file in the LibraryIO dir): go/no-go on the
  *following* target (`Common` core types vs a Solution/Problem slice) +
  effort estimate.
- No regression: full `-m unit` still passes (current baseline after the
  TensileLogic suite = **1330 passed / 201 skipped**; this work only adds).
- **The work is committed** (atomic commits; tree clean; no push).
- **The next goal prompt is created**: `next-goal-<target>.md` for the target
  chosen in recommendations.md, grounded in inspection of that module.

## Coverage reports (BEFORE / AFTER) — required deliverable

```
# BEFORE — clean tree (existing tests only):
pytest -m unit --cov=Tensile/LibraryIO --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit \
  | tee Tensile/Tests/unit/characterization/LibraryIO/coverage-before.txt

# AFTER — once the suite is complete:
pytest -m unit --cov=Tensile/LibraryIO --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit \
  | tee Tensile/Tests/unit/characterization/LibraryIO/coverage-after.txt
```

> NOTE: `--cov=Tensile/LibraryIO` will match the single file `LibraryIO.py`
> (coverage treats the path prefix as a match). Confirm the report shows the
> `LibraryIO.py` row; if path-prefix matching is fussy, fall back to
> `--cov=Tensile` and grep the `LibraryIO.py` row out of the table.

## CONSTRAINTS (unchanged, hard)

- **ADD-ONLY**: only add new files. Never modify/delete any existing file.
  Anything needing an edit → new file or document as a limitation. (The two
  prior targets kept their `target.md`/`resistance.md`/`recommendations.md` as
  **new files in the per-target test dir** rather than editing the shared ones
  — do the same.)
- **Commit frequently**, atomic commits. **Never push / no PRs.** Local only.
- rocm-libraries boundary: stay within `projects/hipblaslt/tensilelite`.
- Snapshot structured/normalised forms, never raw nondeterministic blobs
  (paths/timestamps/version strings normalised).
- BOUND: 180 turns / 180 min. AUTONOMY: don't ask for confirmation; document
  blockers and continue.

## ENV (reuse — already built & validated)

- Image `tensilelite-char:dev`; container `tl-char` (worktree mounted at
  /work). rocisa built (needs `LD_LIBRARY_PATH=/opt/rocm/lib`, baked).
- **Install syrupy if the container was recreated** (`pip install syrupy`) —
  otherwise the existing characterization suites error at setup with
  "fixture 'snapshot' not found". (This bit the TensileLogic run; see that
  suite's `resistance.md`.)
- Resume: see `HANDOFF.md` "Resume steps".
- **Coverage gotcha (critical): pass `--cov` a PATH, never a dotted module** —
  a dotted name re-imports rocisa (nanobind) → SIGABRT.
- Line coverage = `(Stmts - Miss)/Stmts` (pyproject sets `branch=True`; the
  bar is **line** — report both).

## TARGET: `Tensile/LibraryIO.py` (1 file, ~808 LOC)

Only one existing unit test touches it (`test_TensileLibLogicToYaml.py`, which
tests the `TensileLibLogicToYaml` *script*, not `LibraryIO` directly, and is
currently **skipped**), so baseline coverage is expected to be low. rocisa is
pulled in transitively (Solution parsing), so it must be built — it is.

### Public API inventory (drive each; note the tier)

| Symbol | LOC area | Tier / nature |
|---|---|---|
| `StrictTypeLoader` (yaml Loader subclass) | 81-125 | A — exercised by every `readYAML`; pin its strict-typing behaviour on a small doc |
| `_fast_yaml_scalar`, `_fast_yaml_str`, `_fast_yaml_flow_list`, `fast_yaml_dump` | 127-229 | A — pure serialisers; snapshot output strings across scalar/str/list inputs (incl. quoting edge cases) |
| `write`, `writeYAML`, `writeJson`, `writeMsgPack` | 230-265 | A/B — round-trip via `read*`; `writeMsgPack` may RESIST (binary, may need msgpack lib) |
| `_writeSolutionsHeader`, `_findBodyOffset`, `writeSolutions` | 266-344 | B — file I/O over `tmp_path`; header/body offset logic; snapshot the written text (normalised) |
| `read`, `readYAML`, `readJson` | 345-368 | A — round-trip readers; snapshot parsed structure |
| `parseSolutionsFile`, `parseSolutionsData` | 369-437 | B/C — reach into Solution parsing + `isaInfoMap`; characterize on a fixture, expect some resistance |
| `getRealDataTypeA`, `getRealDataTypeB` | 438-461 | A — pure dtype mapping; snapshot across DataTypes |
| `LibraryLogic` (NamedTuple) | 462-471 | A — construct + snapshot `_asdict()` |
| `parseLibraryLogicFile`, `parseLibraryLogicData`, `parseLibraryLogicList`, `rawLibraryLogic` | 472-685 | B/C — the (de)serialisation contract; build a minimal serialized-logic fixture (the `data[0..8]` shape: version dict, schedule, arch, devices, problemType, solutions, indexOrder, exactLogic, rangeLogic — see `TensileLogic/test_run_char.py::_write_logic` for the first 6) and snapshot the parsed `LibraryLogic`. Version-incompatible guard (L403-406) → reject path |
| `getCUCount` | 686-705 | B — reads `CU` env / shells out to `rocminfo`; monkeypatch `os.environ["CU"]` for the env path, document the subprocess path as resistance |
| `createLibraryLogic` | 706-808 | B — assembles the serialized tuple from a `logicTuple`; round-trip against `rawLibraryLogic`/`parseLibraryLogicList` |

### Determinism plan (read these first)

1. **Version strings.** Serialized logic embeds `MinimumRequiredVersion`;
   `parseLibraryLogicData` checks `versionIsCompatible` against `__version__`.
   Pin a fixture version that is compatible; for the reject path, snapshot the
   raised error type/message, not a version-coupled blob.
2. **Filesystem.** All `write*`/`read*`/`writeSolutions` tests use `tmp_path`;
   normalise any absolute path in snapshots to a basename.
3. **`getCUCount` subprocess.** Drive the deterministic `CU` env-var path
   (`monkeypatch.setenv("CU", "304")`); the `rocminfo` subprocess fallback is
   environment-dependent → document as resistance (or monkeypatch
   `subprocess.run`).
4. **msgpack.** `writeMsgPack` writes binary; if the msgpack dependency or a
   stable byte layout is awkward, snapshot a round-trip through a reader or
   document as resistance.
5. **Reuse the prior techniques.** syrupy `snapshot`; `tmp_path` fixtures;
   inject heavy collaborators in the module namespace where a function shells
   out or reaches into Solution parsing; snapshot structured returns
   (`_asdict()` for NamedTuples, sorted collections).

### Location & layout (decided, consistent with prior targets)

- Suite: `Tensile/Tests/unit/characterization/LibraryIO/`, files marked
  `pytestmark = pytest.mark.unit`. Collected by the existing
  `testpaths=Tensile/Tests` — no config edit (add-only safe).
- `target.md`, `resistance.md`, `recommendations.md`, `coverage-before.txt`,
  `coverage-after.txt` as **new files in that dir**. Snapshots in
  `LibraryIO/__snapshots__/` via `--snapshot-update`.
- A `conftest.py` with a session-scoped `isa_info_map` fixture (copy the one
  from `TensileLogic/conftest.py`) if any parse path needs caps.

## Suggested commit sequence (atomic)

1. `coverage-before.txt` (baseline) + `target.md`.
2. Pure serialisers (`_fast_yaml_*`, `fast_yaml_dump`, `getRealDataTypeA/B`,
   `LibraryLogic`) — cheapest snapshots.
3. `read*`/`write*` round-trips (YAML/JSON) over `tmp_path`.
4. `writeSolutions` + header/offset helpers.
5. `parseLibraryLogic*` + `rawLibraryLogic` + `createLibraryLogic` (the
   contract) on a serialized-logic fixture; version-incompatible reject.
6. `parseSolutions*`, `getCUCount` (env path) — expect resistance.
7. resistance.md + recommendations.md.
8. Final coverage run + `coverage-after.txt` + no-regression confirmation.
9. Create `next-goal-<target>.md` for the target selected in
   recommendations.md (grounded in inspection) + commit. Mark this checklist
   done.

## Definition of done checklist

- [ ] `coverage-before.txt` captured (baseline) and committed BEFORE new tests.
- [ ] `--cov=Tensile/LibraryIO` (path-mode) ≥95% line.
- [ ] `coverage-after.txt` captured; before→after delta in target.md.
- [ ] Every Tier A/B public fn has snapshot coverage; Tier C documented.
- [ ] Paths/versions/timestamps normalised in snapshots.
- [ ] resistance.md lists each resisting fn (expect: `writeMsgPack`,
      `getCUCount` subprocess path, deep `parseSolutionsData` branches).
- [ ] recommendations.md: go/no-go on the next target (Common vs a
      Solution/Problem slice).
- [ ] full `-m unit` ≥ 1330 passed / 201 skipped, no failures, additive only.
- [ ] All work committed (atomic, no push); tree clean.
- [ ] Next goal prompt `next-goal-<target>.md` created and committed.
