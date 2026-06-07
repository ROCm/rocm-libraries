# Parametric-Chaos Run-2 README

## Repro instructions

### 1. Environment

Container: `tl-char` (docker, mounts worktree as `/work`).  
Project root: `/work/projects/hipblaslt/tensilelite`  
Workflow scripts: `/work/work/tensilelite-characterization/wf/parametric-chaos/`  

### 2. Assembly (re-generate these deliverables)

```bash
docker cp /tmp/assemble_run2.py tl-char:/tmp/assemble_run2.py
docker exec -w /work/projects/hipblaslt/tensilelite tl-char python3 /tmp/assemble_run2.py
```

All deliverables are written to:  
`/work/work/tensilelite-characterization/parametric-chaos/PublicInputSurfaceDeep/`

### 3. Run reified tests (CPU-only, pass-check only)

```bash
TESTDIR=Tensile/Tests/unit/characterization/PublicInputSurfaceDeep
docker exec -w /work/projects/hipblaslt/tensilelite tl-char \
  pytest -p no:cacheprovider -m unit -q $TESTDIR
```

Do NOT use `--cov=Tensile` without specifying a path — see SIGABRT caveat below.

### 4. Branch extraction helper

```bash
WF=/work/work/tensilelite-characterization/wf/parametric-chaos
docker exec -w /work/projects/hipblaslt/tensilelite tl-char \
  python3 $WF/branch_extractor.py --target-file Tensile/BenchmarkProblems.py --output /tmp/branches.jsonl
```

### 5. Constraint harvesting helper

```bash
docker exec -w /work/projects/hipblaslt/tensilelite tl-char \
  python3 $WF/harvest_constraints.py --branch-file /tmp/branches.jsonl --output /tmp/constraints.jsonl
```

---

## Tool versions (in-container)

| tool | version |
|---|---|
| Python | 3.12.10 |
| z3 | 4.16.0.0 |
| crosshair | 0.0.106 |
| hypothesis | 6.155.2 |
| pysmt | 0.9.6 |

---

## Static / solver / runtime split

| classification | count | description |
|---|---|---|
| fully-static | 1 | predicate truth determined solely by CLI args or YAML keys |
| solver-backed-under-assumptions | 7 | z3 confirmed both branches under stated assumptions |
| runtime-dependent | 11 | predicate depends on filesystem, OS type, or env vars |
| derived-local | 1 | predicate derived from object state, no direct public input |

Runtime-dependent branches are explicitly flagged in `validation_report.md`. They are never silently asserted as confirmed unless a live filesystem/OS witness was produced.

---

## Known caveats

- **rocisa nanobind SIGABRT**: `--cov` arg to pytest MUST resolve to a path (directory), not a dotted module. Use `--cov=Tensile` (directory) or omit `--cov` entirely. Never `--cov=Tensile.BenchmarkProblems`.
- **GPU-probe branches**: The LibraryIO.py:701 true-branch calls rocminfo. The CPU-only test environment cannot exercise the rocminfo path; only the predicate is pinned.
- **Windows branches**: All `os.name == 'nt'` branches are confirmed SAT but never exercised in Linux CI. Regressions are dark.
- **Slice coverage gap**: 16 of 20 branches lack formal backward slice fragments; public-input attribution is from solver-phase analysis only.