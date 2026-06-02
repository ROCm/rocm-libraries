# Checkpoint / resume — tensilelite characterization (#7)

Status as of this checkpoint. Local-only; nothing pushed. **ADD-ONLY**: only
add new files — never modify/delete any existing file (see prompt.md).

## Done (6 commits on `users/davidd-amd/tensillite-coverage`)

1. env scaffolding — `work/tensilelite-characterization/{prompt.md,env/}`
2. `survey.md` + `target.md` (under `Tensile/Tests/unit/characterization/`)
3. **MXScaleFormat characterization suite** — 33 syrupy snapshots,
   `MXScaleFormat.py` at **98.11%** (only the unreachable L95 fallback left)
4. this checkpoint (bound→180, add-only rule, handoff)

Verified: all commits strictly additive; full `-m unit` suite green
(1186 passed / 201 skipped); path-mode `--cov` works.

## Module coverage standing (target = ≥95% on the module)

| File | Stmts | Now | Note |
|---|---|---|---|
| WorkGroup.py | 7 | 100% (existing tests) | add characterization snapshots for completeness |
| MXScaleFormat.py | 68 | 98.11% | DONE (L95 unreachable → resistance.md) |
| MatrixInstruction.py | 164 | ~69% (existing only) | **the remaining work** |

## Resume steps

1. Start env (image already built: `tensilelite-char:dev`):
   ```sh
   WT=/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage
   docker rm -f tl-char 2>/dev/null
   docker run -d --name tl-char -v "$WT":/work \
     -w /work/projects/hipblaslt/tensilelite tensilelite-char:dev sleep infinity
   docker exec -w /work/projects/hipblaslt/tensilelite tl-char invoke rocisa
   ```
2. Coverage run (path-mode --cov; NEVER dotted — aborts via rocisa double-init):
   ```sh
   docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite \
     -w /work/projects/hipblaslt/tensilelite tl-char \
     pytest -m unit --cov=Tensile/SolutionStructs/Validators \
       --cov-config=pyproject.toml --cov-report=term-missing \
       Tensile/Tests/unit
   ```
3. Generate/refresh snapshots for a new test file:
   `pytest -p no:cacheprovider --snapshot-update <file>`

## Remaining work (in order)

### A. WorkGroup characterization (quick)
New file `Validators/test_workgroup_char.py`, `pytestmark = pytest.mark.unit`.
`validateWorkGroup(solution)` asserts `"WorkGroup" in solution` and
`solution["WorkGroup"] in makeValidWorkGroups()`, returns True. Snapshot:
valid WG (e.g. [16,16,1]) → True; invalid/missing → `pytest.raises(AssertionError)`.

### B. MatrixInstruction characterization (the bulk)
New file `Validators/test_matrixinstruction_char.py`. Build inputs like
`Tensile/Tests/unit/test_MatrixInstructionConversion.py`:
```py
cxxCompiler = validateToolchain("amdclang++")
isaInfoMap = makeIsaInfoMap(SUPPORTED_ISA, cxxCompiler)   # module-level, real asmCaps
```
- **matrixInstructionToMIParameters** (conversion half, snapshot the returned
  dict). Inputs to cover the missing branches:
  - happy CDNA path (gfx942/gfx90a, h dtype, 9-item MI, WorkGroup set)
  - `len(mi)!=9` → `pytest.raises(ValueError)` (L59)
  - `workGroup=None` → skips L91 (branch 87->92)
  - gfx950 isa → `isgfx950` True + MXBlockA/B set → L142/147 duplicateFactor=1
  - non-gfx950 + MXBlockA/B set → L143/148 (`MIInputPerThreadMXSA/B`)
  - Sparse=1 and Sparse=2 → sparseA/sparseB (L137-139,144,149)
  - navi gfx11xx (hasWMMA, not hasMFMA, isa[0]==11) → L133-135
  - F32XdlMathOp enable path → need DataType objs: F32XdlMathOp not single +
    DataType single → L73,98 (`enableF32xdl=True`)
- **validateMIParameters** (assert-heavy). Strategy: build a *consistent*
  solution by running `matrixInstructionToMIParameters` on a valid 9-item MI
  for the ISA/dtype, merge into `defaultSolution`, then validate → happy path.
  Early-return branches to hit: empty MI (`mi4==[]` → True, L238-240),
  `not miEnabled` → False (L259-260), 940/941→942 remap (L196-197), dtype-key
  fallback (L225-233), MFMA/WMMA/SMFMA/SWMMAC reject paths (L280-300) using
  invalid MI4 with `printSolutionRejectionReason=False`.
- **Likely resistance** (document, don't force): deep asserts that require an
  exact self-consistent MIBlock/MIWaveGroup; the `MFMA_BF16_1K` bf16-1k assert
  (L280-281) needs a bf16 MI that is in `validMFMA["B1k"]` but not in the
  normal table — may be hard to construct. Capture in resistance.md with the
  snapshot of the input state that was reachable.

### C. resistance.md
- `MXScaleFormat.py:95` — `_mxMatrixLabel` str-fallback: unreachable, every
  `_MX_ALL` member has an explicit label branch; only callable with a matrix
  value that is both in `_MX_ALL` and unmatched, which cannot occur.
- Any MatrixInstruction asserts/branches that resisted, with reason + the
  workaround used (synthetic vs real isaInfoMap, etc.).
- `reject()` global-ish behavior: raises on states with a valid
  `SolutionIndex` when printing; tests pass `printSolutionRejectionReason=False`
  to stay deterministic (documented, not a blocker).

### D. recommendations.md
Go/no-go on scaling to the rest of tensilelite + per-future-target effort
estimate. Inputs: groups & tiers in prompt.md MODULE MAP; the env gotchas
(rocisa build, path-mode --cov). Note the add-only constraint's impact
(can't register markers / edit testpaths / add pragmas — must work around).

### E. Final no-regression check
`pytest -m unit Tensile/Tests/unit` → must stay 1186 passed / 201 skipped
(plus the new characterization tests passing).

## Add-only gotchas to respect
- `-m unit` marker + `testpaths=Tensile/Tests` already exist → new tests under
  `Tensile/Tests/unit/characterization/` are collected with NO config edit. ✅
- Cannot add a coverage pragma to exclude L95 (would edit source) → document
  in resistance.md instead. ✅
- `--cov-config=pyproject.toml` reuses existing config (read-only) → fine. ✅
