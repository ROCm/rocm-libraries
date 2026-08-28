# ADR 0011: Remove `TensileClientConfig` as verified dead code

Status:  Accepted
Defect:  none — behavior is intended (cleanup, not a bug fix)
Commit:  e667f3b (PR #7989) — https://github.com/ROCm/rocm-libraries/commit/e667f3bfebd2aa5670534342aa26e5a97f4426ad (the actual removal) — landed on develop via PR #7989's squash merge, 74e4693

## Context
`TensileClientConfig` looked, on first read, like it might be either dead
code or a live tuning entrypoint broken by an earlier refactor, and two
earlier passes over this question reached opposite wrong conclusions before
this one:
- **Wrong take #1** (`944f2ed`, [full SHA](https://github.com/ROCm/rocm-libraries/commit/944f2ed8f00809c4e79993caba4013ec6736d296)): "dead module, skip; assert nothing" —
  called it dead only because its import failed, without checking callers or
  packaging.
- **Wrong take #2:** "live tuning entrypoint, broken by refactor, restore
  it (~2 lines)" — also wrong: there is no caller and it is not shipped, so
  there was nothing live to restore. This conflated it with the genuinely
  live `writeClientConfig*` path.

The corrected analysis, verified by caller/packaging/import inspection:
- **No in-tree caller.** Following `invoke` / the build / QuickTune / the
  tuning docs, the client-config writing done during tuning goes through
  `ClientWriter.writeClientConfig` / `writeClientConfigIni` (driven by
  `bin/Tensile` → `Tensile.py` → `BenchmarkProblems.py`). Nothing calls the
  standalone `TensileClientConfig.main()` / `bin/TensileClientConfig`. The
  two share the "ClientConfig" name but are different code paths — the
  source of wrong take #2's confusion.
- **Not shipped.** `MANIFEST.in` packages only `bin/Tensile` and
  `bin/TensileCreateLibrary`; `[project.scripts]` registers only `Tensile`.
- **Unimportable anyway.** `TensileClientConfig.py:29` still did
  `from .Common import ... assignGlobalParameters,
  restoreDefaultGlobalParameters`, the pre-refactor flat import path. After
  `Tensile.Common` became a package, those functions live in
  `Common/GlobalParameters.py` and are not re-exported by
  `Common/__init__.py` (which only star-imports Constants/Parallel/Types/
  Utilities), so the import raised `ImportError`. Sibling entrypoints
  (`Tensile.py`, `GenerateSummations`, `TensileUpdateLibrary`,
  `TensileRetuneLibrary`) were migrated to `.Common.GlobalParameters`; this
  one was missed. A second latent break existed too: `:176` called
  `assignGlobalParameters(globalParams)` with one argument against the
  current two-argument `(config, isaInfoMap)` signature.

## Decision
Remove `TensileClientConfig` as verified dead code:
- `Tensile/TensileClientConfig.py` (the module)
- `Tensile/bin/TensileClientConfig` (the launcher)
- the `"TensileClientConfig"` entry in `cmake/tensilelite_auto_build.cmake`'s
  `VALID_BINS`

`shared/tensile/Tensile/TensileClientConfig.py` — a separate vendored
full-Tensile tree with different `ClientWriter` signatures — was **not**
touched; it is out of scope for this repo's TensileLite tree.

## Consequences
This is a real source deletion, a deliberate departure from the
characterization pass's add-only rule, committed separately as a cleanup at
the user's explicit direction (2026-06-03). Full `-m unit`
(`Tensile/Tests/unit`) showed **2466 passed / 201 skipped both before and
after** the removal — no regression.

**Process note (why this ADR exists as a retrofit):** this decision went
through two wrong conclusions before landing on the third, correct one, all
originally recorded as edits to a single running `DECISIONS.md` entry with a
"history, do not repeat" note bolted on. That in-place rewriting is exactly
what the ADR's append-only-and-superseded convention exists to prevent — had
this been an ADR from the start, takes #1 and #2 would each have been their
own file, explicitly superseded rather than overwritten. This ADR is
deliberately written as the single, correct, final decision; the wrong
intermediate takes are preserved above only as calibration for future
"is this dead code?" questions, not as a template to follow.
