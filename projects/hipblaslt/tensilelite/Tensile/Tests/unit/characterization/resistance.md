# Resistance log — `Tensile/SolutionStructs/Validators/`

Functions/lines that resisted characterization, with the reason and the
workaround used. Coverage after the suite: module **96.65% line / 95.08%
branch-blended** (239 stmts, 8 missing). All 8 missing lines are provably
unreachable defensive code. Under the **add-only** rule we cannot add a
`# pragma: no cover` (that edits source), so they remain "missing" in the
report and are documented here instead.

## Unreachable lines (cannot be exercised by any input)

### MatrixInstruction.py:197 — 940/941 → 942 ISA remap
```python
assert isa in SUPPORTED_ISA, ...        # L194
if (9, 4, 0) <= isa <= (9, 4, 1):       # L196
    isa = (9, 4, 2)                     # L197  <- unreachable
```
`SUPPORTED_ISA` (verified at runtime) contains `gfx942` but **not** `gfx940`
or `gfx941`. Any `isa` in `[(9,4,0),(9,4,1)]` fails the `assert` at L194
first, so L197 can never run. Dead code pending removal of the 940/941
TODO noted in the source. **Workaround:** none possible without editing
source; documented here.

### MatrixInstruction.py:227-231, 233 — dtype-key fallback
```python
if not _dtype_key_in_tables(miDataTypeKey):     # L225 (reached)
    if macDataTypeA == macDataTypeB:            # L226 (reached, False arm)
        doubled = ca + ca                       # L227 unreachable
        if _dtype_key_in_tables(doubled):       # L228 unreachable
            miDataTypeKey = doubled             # L229 unreachable
        elif _dtype_key_in_tables(ca):          # L230 unreachable
            miDataTypeKey = ca                  # L231 unreachable
    elif _dtype_key_in_tables(cb + ca):         # L232 (reached, False arm)
        miDataTypeKey = cb + ca                 # L233 unreachable
```
- L227-229: only run when `macDataTypeA == macDataTypeB` **and** the key
  `ca+cb` is absent from the tables. But if `macA == macB` then
  `ca+cb == ca+ca == doubled`; the L225 guard already proved that key
  absent, so L228 re-tests the *same* key and is always False. The doubled
  assignment can never fire.
- L230-231: `_dtype_key_in_tables(ca)` with a single char. The MFMA/SMFMA
  tables have **no single-character keys** (all are `HH`, `F8B8`, ...), so
  this is always False.
- L233: requires `ca+cb` absent but `cb+ca` present, i.e. an **asymmetric**
  table key. Every MFMA/SMFMA pair key is symmetric (`F8B8` and `B8F8` both
  exist, etc.), so this never fires.

**Reached** via `test_validate_dtype_key_fallback_mixed_mac` (MacDataTypeA=h,
MacDataTypeB=b → key `HB`): covers L225 (True), L226 (False arm), L232
(False arm). The inner assignments stay unreachable. **Workaround:** none
without source edits; the case pins the *reachable* control flow.

### MXScaleFormat.py:95 — `_mxMatrixLabel` string fallback
```python
def _mxMatrixLabel(dtValue):
    if dtValue in _MX_FP8_LIKE: return "FP8"
    ...
    if dtValue == DataTypeEnum.Float4.value: return "FP4"
    return str(dtValue)                 # L95 unreachable
```
`_mxMatrixLabel` is only ever called while building a rejection reason, which
only happens when a matrix value is in `_MX_ALL`. Every member of `_MX_ALL`
matches one of the explicit branches above, so the `str()` fallback is
unreachable. (`_mxScaleLabel`'s analogous fallback at L106 *is* reachable —
a `None` scale is labeled "None" — and is covered.) **Workaround:** none
without source edits.

## Behavioural hazards handled (did not block, but worth noting)

### `reject()` writes stdout and can raise on LibraryLogic states
`SolutionStructs/Utilities.reject` prints the reason to stdout and, when a
valid `SolutionIndex` is present, raises (the "rejection of a LibraryLogic
is not expected" guard). Both only happen when `printSolutionRejectionReason`
is True. **Workaround:** all reject-path tests pass
`printRejectionReason=False`, so `reject` deterministically sets
`state["Valid"]=False` and returns with no stdout and no raise. Snapshots
capture `{returned, valid}`, never stdout.

### `defaultSolution` is shared global mutable state
`Tensile.Common.GlobalParameters.defaultSolution` is a module-level dict;
the pre-existing `test_MatrixInstructionConversion.py` mutates it in place
(`solution = defaultSolution; solution.update(...)`). To avoid cross-test
contamination, the characterization suite **deep-copies** it
(`copy.deepcopy(defaultSolution)`) in `_consistent(...)` before mutating.
The global itself is never written.

### asmCaps come from the live assembler
`matrixInstructionToMIParameters` reads `isaInfoMap[isa].asmCaps["HasMFMA"/
"HasWMMA"]`, derived once via `makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)`
(shells out to `amdclang++`). These booleans are stable for a given ISA +
toolchain, so snapshots are reproducible in the dev container. A different
compiler version *could* shift a cap and a snapshot; this is the only
environment coupling. **Workaround:** snapshots pin the structured MI-param
dict (not raw blobs); branch-specific cases pick ISAs whose caps are fixed
by hardware definition (e.g. gfx1100 has no MFMA).
