# Characterization target — `Tensile/LibraryIO.py`

Follow-up to the completed `TensileLogic/` suite (#7). Same pattern, next
module. Kept as a **new file** under the `LibraryIO/` test dir rather than
editing the shared `../target.md`, per the add-only rule. See `../survey.md`
for the syrupy survey (reused, no new survey needed).

## Module under test

One source file, `Tensile/LibraryIO.py` (~808 LOC, 448 stmts). It is the
serialization/deserialization layer for Tensile's two on-disk artifacts:
**solution files** (`writeSolutions` / `parseSolutions*`) and **library-logic
files** (`createLibraryLogic` / `parseLibraryLogic*` / `rawLibraryLogic`),
plus the low-level format primitives (`read*` / `write*`, a hand-rolled fast
YAML emitter `fast_yaml_dump`, and the `StrictTypeLoader` that preserves 0/1
as ints instead of bools). Also two pure dtype mappers (`getRealDataTypeA/B`),
the `LibraryLogic` NamedTuple, and `getCUCount` (env var / `rocminfo`).

| Area | LOC | Public API (tier) |
|---|---|---|
| `StrictTypeLoader` | 81-105 | strict 0/1-vs-bool YAML loader (A, via read) |
| `_fast_yaml_scalar/_str/_flow_list`, `fast_yaml_dump` | 127-228 | pure serialisers (A) |
| `write`, `writeYAML`, `writeJson`, `writeMsgPack` | 230-264 | format dispatch + writers (A/B) |
| `_writeSolutionsHeader`, `_findBodyOffset`, `writeSolutions` | 266-339 | solution-file writer + header/offset (B) |
| `read`, `readYAML`, `readJson` | 345-366 | readers (A) |
| `parseSolutionsFile`, `parseSolutionsData` | 369-436 | solution-file parser (B/C) |
| `getRealDataTypeA/B` | 438-460 | pure dtype mapping (A) |
| `LibraryLogic` (NamedTuple) | 462-470 | return tuple (A) |
| `parseLibraryLogicFile/Data/List`, `rawLibraryLogic` | 472-680 | logic-file parser (B/C) |
| `getCUCount` | 686-704 | CU count from env / rocminfo (B) |
| `createLibraryLogic` | 706-808 | assemble logic tuple for writing (B) |
| **TOTAL** | | **448 stmts** |

## Why this module

- **Closes the serialization loop.** The `Validators/` and `TensileLogic/`
  suites pinned how solutions are *validated*; `LibraryIO` is how they are
  *written and read back*. Round-tripping `createLibraryLogic` →
  `parseLibraryLogicList` and `writeSolutions` → `parseSolutionsData` pins the
  on-disk contract that the rest of the toolchain depends on.
- **Mostly pure / file-I/O.** The format primitives and dtype mappers are pure
  (cheap snapshots). The write/read paths are deterministic over `tmp_path`.
  Only `parseSolutionsData` / `parseLibraryLogicData` (full `Solution`
  construction) and `getCUCount`'s `rocminfo` fallback carry real environmental
  weight — handled by real `isaInfoMap` fixtures (the contract) and
  `monkeypatch` (the subprocess), respectively.

## Determinism handling

1. **Version strings.** Serialized artifacts embed `__version__` /
   `MinimumRequiredVersion`. Snapshots of written headers normalise the version
   token to `<VERSION>`; the version-incompatible reject path snapshots the
   warning behaviour, not a version-coupled blob.
2. **Filesystem.** All write/read tests use `tmp_path`; any absolute path in a
   snapshot is reduced to its basename.
3. **`getCUCount`.** The deterministic `CU` env-var path is driven with
   `monkeypatch.setenv`; the `rocminfo` subprocess fallback is monkeypatched
   (`subprocess.run`) so both branches are pinned without a live GPU.
4. **msgpack.** `writeMsgPack` round-trips through `msgpack.unpack` (binary
   bytes are not snapshotted directly).
5. **Reuse prior techniques.** syrupy `snapshot`; `tmp_path` fixtures; a
   session-scoped `isa_info_map` fixture (copied from `TensileLogic/`) for the
   parse paths; structured returns snapshotted (`_asdict()` for the NamedTuple,
   sorted dict keys, normalised paths).

## Location & coverage command (same rules as prior targets)

Suite at `Tensile/Tests/unit/characterization/LibraryIO/`, marked `-m unit`
(collected by the existing `testpaths=Tensile/Tests`, no config edit). Pass
`--cov` the package dir + grep the LibraryIO row (single-file path-prefix does
not match — see `coverage-before.txt`):

```
pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
  --cov-report=term-missing Tensile/Tests/unit | grep LibraryIO
```

Line coverage = `(Stmts - Miss) / Stmts` (the goal's bar is line; both line
and blended are reported).

## Result (before → after)

| | Stmts | Miss | Line cov | Blended |
|---|---|---|---|---|
| Before | 448 | 301 | 32.81% | 30.45% |
| After  | TBD | TBD | TBD | TBD |

(Filled in at the final coverage step; see `coverage-after.txt` and
`resistance.md` / `recommendations.md`.)
