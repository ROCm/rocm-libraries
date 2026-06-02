# Resisting functions / lines — `Tensile/LibraryIO.py`

Everything reasonably testable is pinned (98.21% line). What remains is either
genuinely unreachable in this environment or intentionally not snapshotted.
New file in the per-target dir per the add-only rule.

## Unreachable LINES (8) — counted as "Miss" but cannot be hit here

| Line(s) | Code | Why unreachable | Workaround tried |
|---|---|---|---|
| 51 | `print2("orjson not installed. Fallback to ujson.")` | Runs only if `import ujson` **succeeds** after orjson fails. ujson is not installed, so the import always raises and the print is skipped. | Can't make a non-existent package importable; blocking orjson (done in `test_import_fallbacks_char.py`) reaches the `import ujson` line but not the success-print. |
| 55 | `print2(... "Fallback to simplejson.")` | Same — simplejson not installed. | As above. |
| 62-63 | `except ImportError: printExit(...)` for `import yaml` | yaml is mandatory and already imported by collaborators; blocking it makes the *module import itself* call `sys.exit`, leaving the shared module half-initialised and crashing the suite. | Deliberately **not** blocked (every other except arm is, via reload). Documented instead. |
| 559, 562 | `solutionState["ProblemType"]['MacDataType{A,B}'] = getRealDataType{A,B}(...)` | Inside `solutionStateToSolution`. By the time it runs, the data-level defaulting (L514-518) has already added `MacDataTypeA/B` to the ProblemType, so the `'... not in ...'` guard is always false. Redundant defensive code. | Would require a ProblemType object missing keys the data-level pass guarantees — not constructible through the public entry points. |
| 565, 570 | `solutionState["ProblemType"]['DataType{A,B}'] = ...['MacDataType{A,B}']` | Same: L520-528 always populate `DataTypeA/B` before the object is built, so the per-solution `'not in'` guard is dead. | As above. |

## Unreachable partial BRANCHES (3) — affect blended %, not line %

| Branch | Meaning | Why unreachable |
|---|---|---|
| 539->541 | logic solution with `KernelLanguage != "Assembly"` | The vendored fixture (and essentially all asm_full logic) is Assembly; a Source-kernel logic solution would need ISA pre-set and is not part of the serialized-logic contract under test. |
| 694->701 | `getCUCount`: empty rocminfo output | `"".strip().split("\n")` is `['']` — always truthy — so the `if lines:` false-arm cannot be taken. |
| (517->.. closed) | data-level MacDataType guards | Both arms now covered (fixture true-arm + `test_parse_..._no_cucount_with_datatypes` false-arm). |

## Functions exercised via injection / round-trip (documented technique, not a gap)

| Function | Technique | Reason |
|---|---|---|
| `parseLibraryLogic{File,Data}` | real vendored fixture + session `assembler`/`isa_info_map` | Full `Solution`/`ProblemType`/`MasterSolutionLibrary.FromOriginalState` construction; snapshots are a **normalised structural summary** (schedule/arch/counts/sorted type-mismatches/selected ProblemType fields), never the live objects. |
| `parseSolutions{File,Data}` | genuine round-trip: parse logic → `writeSolutions` the real solutions → parse back | A standalone solutions file needs solution states with embedded `ProblemType`; generating them from real solutions is the faithful way to get a valid one. |
| custom-kernel branch (L546-555) | `monkeypatch` `getCustomKernelConfig` in the `LibraryIO` namespace | The fixture has no custom-kernel solution; the empty-config case drives the merge branch, a length-3 `MatrixInstruction` config drives the `ValueError` (raised before any `Solution` is built, so MI self-consistency is irrelevant). |
| import fallbacks (L48-58, 69-77, 109-110) | `importlib.reload` with `orjson/ujson/simplejson/msgpack` blocked (`sys.modules[x]=None`) and `CSafeLoader/CSafeDumper` hidden (`delattr`), then reload again to restore | These `except ImportError` arms never run during normal collection because the fast variants are installed. The module is reloaded back to its real bindings in a `finally`, so the rest of the session is unaffected (verified: full `-m unit` = 1443 passed). |
| `getCUCount` rocminfo path | `monkeypatch` `LibraryIO.subprocess.run` | No live GPU/rocminfo in the unit env; both the parse-success and no-match→`printExit` outcomes are pinned deterministically. |
| `writeMsgPack` | binary round-trip via `msgpack.unpack` | Raw msgpack bytes are not snapshotted (binary, not human-reviewable). |

## Determinism notes

- Every embedded Tensile version string (`__version__` / `MinimumRequiredVersion`)
  is normalised to `<VERSION>` in snapshots; the version-incompatible path
  snapshots the *parsed result*, not the warning text.
- All file I/O is over `tmp_path`; the one vendored fixture is referenced by a
  path-relative `Path(__file__).parent`, never an absolute path in a snapshot.
- `assembler.rocm_version` (env-specific) is read by `Solution` but never leaks
  into a snapshot — only structural summaries are captured.
