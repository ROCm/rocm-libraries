# Goal-centric execution plan — drive TensileLite coverage to ≥80%

**North star (the goal, non-negotiable):** total line+branch coverage of
`projects/hipblaslt/tensilelite/Tensile` reaches **≥80%**, achieved **only by
adding characterization tests** — *no source changes of any kind*.

This file is the **single source of truth + checkpoint**. To stop/restart: read
this file top-to-bottom, look at the **Progress log** + **Checklist**, take the
last `coverage/master-baseline-<N>.txt` as the current BEFORE, and continue at
the first unchecked item. Nothing else is needed to resume.

---

## 1. Where we are (measured, not assumed)

`coverage/CURRENT-full-package.txt` (path-mode `--cov=Tensile`, `-m unit`,
`Tensile/Tests/unit`, 2026-06-03):

```
TOTAL  54812 stmts  36338 missing  22764 branch  989 partial  →  30.62%
```

Batches A–F (prior effort) drove the **pure / IO / config / toolchain** surface
to 95–100% and deliberately **excluded the codegen/asm/GPU surface (~28k missing
stmts)**. That exclusion is exactly why we sit at 30.62%. **Reaching 80% is
therefore impossible without covering codegen.** The whole of this plan is about
covering codegen *deterministically and CPU-only*, which info.md already proved
is the right lever (direction 1).

### The gap, by yield (top uncovered, missing-stmt count)

| Module | Missing | Cluster |
|---|---:|---|
| `KernelWriterAssembly.py` | 11,942 | **codegen core** |
| `KernelWriter.py` | 5,945 | **codegen core** |
| `SolutionStructs/Solution.py` | 1,926 | derivation (slice 3b) |
| `Components/StreamK.py` | 1,717 | codegen component |
| `Components/GlobalWriteBatch.py` | 1,593 | codegen component |
| `Components/GSU.py` | 1,203 | codegen component |
| `Components/LocalRead.py` | 1,107 | codegen component |
| `LibraryLogic.py` | 874 | orchestration |
| `Components/SIA.py` | 761 | codegen component |
| `Activation.py` | 683 | asm emitters |
| `AsmStoreState.py` | 602 | asm helper |
| `Components/WorkGroupMappingAlgos.py` | 594 | codegen component |
| `Components/ShiftVectorComponents.py` | 533 | codegen component |
| `Components/LraTileAssignment.py` | 471 | codegen component |
| `AsmAddressCalculation.py` | 424 | asm helper |
| `Components/{TensorDataMover,LSU,Subtile/*,PackData,…}` | ~1,800 | codegen components |
| `KernelWriter{Modules,BetaOnly,Conversion}.py` | ~680 | codegen helpers |
| `TensileCreateLibrary/Run.py` | 316 | orchestration |
| `ClientWriter.py` | 270 | orchestration (file IO) |

Codegen-cluster total ≈ **28k of the 36.3k missing**. Covering ~90% of it +
Solution derivation + the orchestration residue clears 80%.

### Arithmetic of the goal (statement view, conservative)

- Covered now: `54812 − 36338 = 18,474` stmts.
- 80% of 54,812 = **43,850** stmts must be covered → need **+25,376** stmts.
- KernelWriterAssembly + KernelWriter alone = **17,887** reachable-by-emit.
- Components cluster ≈ **10,000**. Solution derivation ≈ **1,900**.
- => the emit mechanism + derivation **structurally suffices**; the rest is
  mop-up. (Branch coverage tracks line coverage closely on this emit code.)

---

## 2. The mechanism that unlocks 80% (proven feasible today)

**Deterministic, CPU-only assembly emit.** The assembly *emitter* needs no GPU —
only the *run* does. Verified entry points in this tree:

- `Tensile.TensileCreateLibrary.Run.processKernelSource(kwa, data, outOpts,
  splitGSU, kernel)` → `KernelCodeGenResult`, whose core is
  `kernelWriter.getSourceFileString(kernel)` → **assembly source string**
  (`Run.py:184`, `:192`). Deterministic given a pinned toolchain/ISA.
- `gpu_test_helpers.py` already constructs a real `KernelWriterAssembly`
  (`create_writer`), inits any ISA CPU-only (`init_rocisa('gfx942')`, uses
  `amdclang++`, no GPU), and emits tile asm (`generate_gra_asm`, `generate_lra_asm`,
  `_generate_tile_asm`). Only `run_on_gpu`/`assemble_and_run` are GPU-gated.
- Inputs are **in-tree** (no 5.5GB tuning tree needed to start):
  - direction (1) logic file: `Tensile/Tests/unit/characterization/LibraryIO/data/
    logic_gfx942_HSS_BH.yaml` (222 lines).
  - direction (2) range configs: `Tensile/Tests/common/**` (gemm/gsu/…), each a
    full BenchmarkProblems spec with `TestParameters: marks:[skip-gfx*]` arch
    gating. We add small curated YAMLs (add-only) only to widen ISA/dtype/schedule
    branches not hit by existing configs.

**Golden shape (per info.md acceptance):** snapshot the **canonicalized**
assembly text (a register-numbering + whitespace + temp-label canonicalizer so
benign refactors don't churn goldens). The expensive GPU/toolchain
assemble-and-compare tier is **out of scope** here (GPU-gated); we snapshot the
emitted text, which is the deterministic, host-independent artifact.

---

## 3. Hard rules (carried from the prior effort — do not relax)

1. **ADD-ONLY.** Only create new files under
   `Tensile/Tests/unit/characterization/`. Never modify/delete any existing file
   (source, tests, `pytest.ini`/`tox.ini`/`pyproject.toml`, docs). Anything
   needing an edit (a marker, a pragma, testpaths) is solved with a new file or
   recorded in `resistance.md`. `-m unit` + `testpaths=Tensile/Tests` already
   exist → new suites are auto-collected with zero config edits.
2. **NEVER push / never open a PR.** Local atomic commits only.
3. `--cov` takes the **path** `Tensile` (or a subdir path) — **never a dotted
   module** (rocisa double-import → SIGABRT). Scope full runs to
   `Tensile/Tests/unit` (the `Tests/common/test_config.py` collection landmine).
4. Snapshots generated **in-container** (`--snapshot-update`, root-owned). Use
   `importlib.import_module` for `SolutionStructs` submodules shadowed by the
   package `__init__`.
5. **Pin, don't fix, latent bugs.** If a characterization test reveals a real bug
   (cf. D12/D14/median/dedup), snapshot the *actual current behavior* and record
   it in `DECISIONS.md` — changing source is forbidden by the goal.

---

## 4. Per-atomic-commit + grouping protocol

**Atomic commit = a change you can roll back with zero breakage** (suite still
green, add-only honored). Commit *every* atomic unit; never batch unrelated work.

**Grouping = by code functionality / suite.** One suite dir per functional area
(`_codegen/`, `CodegenKernelWriter/`, `ComponentStreamK/`, `SolutionDerivation2/`,
…). Within a suite, commit in this order (each its own commit, or fewer for tiny
suites):

1. `coverage-before.txt` (the module's row pulled from the latest
   `master-baseline-*.txt`) + `target.md` (what/why/how for this suite).
2. test file(s) + `__snapshots__/` (generated in-container).
3. `coverage-after.txt` + `resistance.md` (genuinely-unreachable / GPU-only /
   pinned-bug lines) + checklist tick in this file.

**Per phase (checkpoint):** run the full `-m unit --cov=Tensile` **once**,
confirm **no regression** (pass count only grows; 201 skipped unchanged),
save a fresh `coverage/master-baseline-<passCount>.txt`, append a **Progress log**
line, and commit the checkpoint. This baseline is the resume anchor.

---

## 5. Phases (ordered by coverage yield → mop-up)

Each phase lists its suites; check items off in §7. Stop/restart safely at any
checkpoint.

### Phase G0 — Codegen emit harness (the shared mechanism)  ⏳
New `Tensile/Tests/unit/characterization/_codegen/`:
- `harness.py`: `emit_kernel_asm(config_or_logic_path, arch, select=...) -> str`
  — parse config → `generateLogicDataAndSolutions` (or `create_writer` for the
  targeted path) → `getSourceFileString` → return source. Plus
  `canonicalize_asm(text)` (strip register numbers→`%v`/`%s`/`%a`, addresses,
  timestamps, temp labels) for stable goldens.
- `conftest.py`: session fixture building `isaInfoMap` once via
  `validateToolchain('amdclang++')` + `makeIsaInfoMap` (cf. existing
  MatrixInstruction tests); `init_rocisa` per-arch.
- `test_harness_smoke.py`: emit ONE small kernel for gfx942, snapshot canonical
  asm. **Gate:** harness reusable, snapshot stable across two runs.
- **Acceptance:** KernelWriterAssembly/KernelWriter coverage jumps measurably
  from this single kernel (proves the mechanism wires through).
- Commit(s): harness+conftest; smoke test+snapshot; checkpoint.

### Phase 1 — KernelWriter / KernelWriterAssembly core emit  ⏳ (largest yield)
New `CodegenKernelWriter/`. Drive a **config matrix** through G0, one snapshot per
cell, grouped into commits by family:
- by ISA: gfx942, gfx90a, gfx950, gfx1100/1101 (navi WMMA), gfx1200/1201.
- by dtype: h/bf16/s/f8/bf8/i8/f64/complex.
- by schedule/flags: PrefetchGlobalRead 0/1/2, GSU>1, StreamK on, SplitU,
  DirectToLds, DirectToVgpr, WaveSplitK, persistent loop.
One commit per ISA-or-dtype family (≈6–10 commits). After the phase, checkpoint.
- **Expected:** the dominant jump (KWA 11.9k + KW 5.9k missing → most covered).
- resistance.md: branches needing combos no in-tree config expresses → either add
  a small curated YAML (add-only) or document as residue for Phase 6.

### Phase 2 — Components/* (codegen components)  ⏳
New per-component suites, ordered by missing-stmt yield:
`ComponentStreamK` (1717) · `ComponentGlobalWriteBatch` (1593) ·
`ComponentGSU` (1203) · `ComponentLocalRead` (1107) · `ComponentSIA` (761) ·
`ComponentWGMapping` (594) · `ComponentShiftVector` (533) ·
`ComponentLraTile` (471) · `ComponentSubtile` (Kernel/SubtileGREmit/Logical) ·
`ComponentMisc` (TensorDataMover, LSU, PackData, CMSValidator, CustomSchedule,
ComputeStoreVgprs, Signature, Priority, SumUnroll, PersistentLoop, MAC_*).
Most are exercised transitively by Phase-1 configs; here we add **targeted
configs** to select the specific component variants (e.g. a StreamK config, a
GSU>1 config, a WMMA config for navi MAC components) and snapshot the component's
emitted module. One commit per component family. Checkpoint at phase end (and a
mid-phase checkpoint after the first 4 heavy components).

### Phase 3 — Asm* helpers + KernelWriter{Modules,BetaOnly,Conversion}  ⏳
New `AsmHelpers/`. `AsmStoreState` (602) + `AsmAddressCalculation` (424) +
`KernelWriterModules` (202) + `KernelWriterBetaOnly` (201) +
`KernelWriterConversion` (279). Largely covered by Phases 1–2; add store-D-style
targeted harness tests (reuse `gpu_test_helpers.create_writer`) for residue
methods. Commit per file-group; checkpoint.

### Phase 4 — Solution.py derivation (slice 3b, deferred from A–F)  ⏳
New `SolutionDerivation2/`. 1,926 missing. Drive
`Solution.assignDerivedParameters` / `getKernels` across the Phase-1 config
matrix; snapshot the **derived solution state** (sorted keys) — also catches the
in-place-mutation gap info.md flags (snapshot post-call state, not just return).
Group commits by derivation area (tile/depthU, GSU, schedule, predicates).
Checkpoint.

### Phase 5 — Orchestration residue (emit-only)  ⏳
New `OrchestrationResidue/`. `LibraryLogic` generateLogic/LogicAnalyzer (874),
`TensileCreateLibrary/Run` end-to-end emit into a tmp dir (316 — call `run()`
with a tiny config + `--no-build`-equiv path, snapshot the written artifacts/file
list), `ClientWriter` file-writers (270), `Activation` asm-emit layer (683).
Group by module; checkpoint.

### Phase 6 — Mop-up to ≥80% + final gate  ⏳
- Full `--cov=Tensile --cov-report=term-missing`; rank remaining missing lines.
- Targeted tests for the highest-yield reachable residue; add curated YAMLs
  (add-only) for branch combos no config expressed.
- Everything genuinely **GPU-only / unreachable** → `resistance.md` with proof.
- **GATE: TOTAL ≥ 80%.** If short, iterate Phase 6 (the loop does not stop until
  the gate passes — see §6). Save final `master-baseline-<N>.txt`, write
  `recommendations.md`, update HANDOFF.md.

---

## 6. Stop condition & restart

**Do not stop until `TOTAL ≥ 80%`** on the canonical run:
```sh
docker exec -e PYTHONPATH=/work/projects/hipblaslt/tensilelite \
  -w /work/projects/hipblaslt/tensilelite tl-char \
  pytest -m unit --cov=Tensile --cov-config=pyproject.toml \
    --cov-report=term-missing Tensile/Tests/unit
```
**Restart from cold:**
```sh
WT=/home/davdixon/projects/rocm-libraries/.claude/worktrees/tensilelite-coverage
docker rm -f tl-char 2>/dev/null
docker run -d --name tl-char -v "$WT":/work \
  -w /work/projects/hipblaslt/tensilelite tensilelite-char:dev sleep infinity
docker exec -w /work/projects/hipblaslt/tensilelite tl-char invoke rocisa
```
Then read this file's **Checklist** + **Progress log**, take the last
`master-baseline-<N>.txt` as BEFORE, resume at the first unchecked item.

**Snapshot refresh:** `pytest -p no:cacheprovider --snapshot-update <suiteDir>`
(in-container).

---

## 7. Checklist (tick as completed; the live resume index)

- [x] **G0** harness + conftest + smoke snapshot; mechanism proven; checkpoint (TOTAL 30.62%→47.82%, 2470 passed)
- [ ] **P1** KernelWriter/Assembly matrix — gfx942 family
- [ ] **P1** gfx90a family
- [ ] **P1** gfx950 family
- [ ] **P1** navi gfx11xx (WMMA) family
- [ ] **P1** gfx12xx family
- [ ] **P1** dtype/schedule sweep (f8/bf8/i8/f64/complex; PGR/GSU/SplitU/DTL/DTV)
- [ ] **P1** checkpoint (fresh master-baseline; no-regression)
- [ ] **P2** ComponentStreamK
- [ ] **P2** ComponentGlobalWriteBatch
- [ ] **P2** ComponentGSU
- [ ] **P2** ComponentLocalRead
- [ ] **P2** mid-phase checkpoint
- [ ] **P2** ComponentSIA
- [ ] **P2** ComponentWGMapping
- [ ] **P2** ComponentShiftVector
- [ ] **P2** ComponentLraTile
- [ ] **P2** ComponentSubtile
- [ ] **P2** ComponentMisc
- [ ] **P2** checkpoint
- [ ] **P3** AsmStoreState + AsmAddressCalculation
- [ ] **P3** KernelWriter{Modules,BetaOnly,Conversion}
- [ ] **P3** checkpoint
- [ ] **P4** Solution derivation slice 3b
- [ ] **P4** checkpoint
- [ ] **P5** LibraryLogic
- [ ] **P5** TensileCreateLibrary/Run end-to-end emit
- [ ] **P5** ClientWriter + Activation emit
- [ ] **P5** checkpoint
- [ ] **P6** mop-up rounds (repeat until gate)
- [ ] **P6** FINAL GATE: TOTAL ≥ 80% ✅ + recommendations.md + HANDOFF update

---

## 8. Progress log
(one line per completed phase/module: `<item> — before% → after% (N tests), commit <sha>`)

- 2026-06-03 — Plan authored. Baseline measured: **TOTAL 30.62%**
  (`coverage/CURRENT-full-package.txt`), full `-m unit` = 2466 passed / 201
  skipped. Codegen-cluster = ~28k of 36.3k missing → the target mass.
- 2026-06-03 — **G0 done.** CPU-only emit harness (`_codegen/`) + canonicalizer +
  smoke golden. One gfx942 kernel lifted **TOTAL 30.62% → 47.82%** (+17.2 pts,
  ~9.6k stmts), full `-m unit` 2470 passed / 201 skipped (no regression).
  Baseline `coverage/master-baseline-G0.txt`. Commits 44a11fbe556 (+ checkpoint).

---

## 9. Risk register (honest)

- **R1 — 80% may require deep branch combos.** Mitigation: Phase 6 loop + curated
  add-only YAMLs; if a residue is provably GPU-only/unreachable, it goes to
  `resistance.md` and we compensate yield elsewhere. The §1 arithmetic shows the
  emit mechanism + derivation structurally exceed the 80% line; risk is in the
  long tail, not the bulk.
- **R2 — golden churn.** Mitigation: the canonicalizer (G0) strips register/addr
  numbering; goldens key on structure, not allocation.
- **R3 — emit raises on some configs** (invalid solution for an ISA).
  Mitigation: snapshot the rejection/`err!=0` path too (that is real covered
  behavior); select valid cells via the configs' own `skip-gfx*` marks.
- **R4 — runtime/time.** Full instrumented run ≈ 2–4 min. Mitigation: per-suite
  fast `--cov=Tensile <SuiteDir>` during iteration; full run only at checkpoints.
- **R5 — a test reveals a real bug.** Pin current behavior + log in DECISIONS.md;
  never edit source (goal forbids it).
