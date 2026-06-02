# Recommendations — after `Tensile/TensileLogic/`

Kept as a **new file** in the `TensileLogic/` test dir rather than editing the
shared `../recommendations.md`, per the add-only rule. Builds on that file's
"GO (conditional)" verdict and cost model.

## Result for this target

`Tensile/TensileLogic/` went from **31.55% line (393 stmts, 269 missing)** to
**99.49% line (393 stmts, 2 missing; +67.94 pts)** (see `coverage-after.txt`).
Seven of eight files reach 100% line;
`ValidChipId.py` retains two unreachable defensive lines (129, 155 — see
`resistance.md`). No regression: the full `-m unit` suite stays green and the
skip count is unchanged; this work is purely additive.

This matched the cost model's cheapest tier ("pure validators / table logic,
~0.5–1 day"). `Run.py`, nominally the stateful outlier, reached 100% line by
**injecting** its collaborators (validators, toolchain/caps builders,
`ParallelMap2`, `_setup`) in the module namespace — confirming that
orchestration code is tractable when its dependencies are import-level and
swappable, without a live toolchain or GPU.

## What worked (additions to the shared list)

- **Injection at the module boundary.** `Run` imports its validators,
  `ParallelMap2`, `makeIsaInfoMap`, and `validateToolchain` as module globals,
  so `monkeypatch.setattr(Run, name, stub)` cleanly decouples the orchestrator
  from the (slow, subprocess-bound, multiprocessing) collaborators. Pin the
  *combination* logic here; pin the collaborators in their own suites.
- **Snapshot state + return for stateful validators.** For
  `ValidWorkGroupMappingXCC`, snapshotting `{returned, reported_failures}`
  after a per-test reset pins the module-global dedup accounting that a
  return-only snapshot would miss.
- **`tmp_path` for file-reading code.** `KnownBugs`, `hasCustomKernel`, and the
  serialized-logic fixtures for `_runChecks` are all driven from `tmp_path`,
  keeping inputs hermetic and paths normalisable.

## Go / no-go on the next target

### Verdict: **GO — `LibraryIO` next** (then `Common` core types)

`LibraryIO.py` (~808 LOC) is the highest-value remaining cheap-ish target: it is
the solution-library **(de)serialisation contract** — exactly the "snapshot the
structured solution/dict round-trip" the original goal pointed at. It has only
one existing unit test (`test_TensileLibLogicToYaml.py`, currently skipped), so
the headroom is large.

| Candidate | Why / why not | Effort to ≥95% line |
|---|---|---|
| **`LibraryIO` (group 5)** ✅ chosen | Structured YAML/JSON round-trips; pure serialisers (`fast_yaml_dump`, `_fast_yaml_scalar/str/flow_list`, `writeYAML/Json`, `read*`) snapshot trivially. `parseLibraryLogicData/List` and `parseSolutionsData` are heavier (they reach into Solution parsing + `isaInfoMap`) — characterize the round-trip on a fixture logic file, normalise paths/timestamps. | ~1–2 days |
| `Common` core types (group 1) | High leverage (everything depends on `DataType`/`ValidParameters`), pure — but a large surface; better second. | ~2–4 days |

**Why LibraryIO over Common now:** it directly extends this target (`Run` calls
`readYAML`; `_runChecks` consumes `data[4]/data[5]` from exactly the serialized
format `LibraryIO` produces), it is a contract that breaks silently when
changed, and its pure serialiser half is near-free coverage. Defer `Common` and
the stateful/GPU groups (2/4/6) as before.

### Effort estimate for `LibraryIO`

~1–2 days. Cheap half (serialisers + `read*`/`write*` round-trips, the
`StrictTypeLoader`, `getRealDataTypeA/B`, `getCUCount`): ~0.5 day of snapshot
tests. Expensive half (`parseLibraryLogicData/List`, `parseSolutionsData`,
`createLibraryLogic`): needs a real serialized-logic fixture and an
`isaInfoMap`, plus path/timestamp normalisation; budget the remaining ~1 day and
expect a few genuinely-resistant branches (msgpack write, version-incompatible
guards) to land in `resistance.md`.

The grounded kickoff for it is `work/tensilelite-characterization/next-goal-libraryio.md`
(repo-root-relative), alongside the prior `next-goal-tensilelogic.md`.
