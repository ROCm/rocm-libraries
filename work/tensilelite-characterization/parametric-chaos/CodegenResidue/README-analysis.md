# README — CodegenResidue Analysis Reproduction

## What This Bundle Is

This is the Run 3 output of the parametric-chaos characterization pipeline applied to
the `CodegenResidue` surface: the residual codegen branches in:

- `Tensile/SolutionStructs/Solution.py` (5,230 lines, 1,180 branch sites)
- `Tensile/KernelWriter.py` (9,900 lines, 1,858 branch sites)
- `Tensile/KernelWriterAssembly.py` (18,789 lines, 3,028 branch sites)

Total census: **6,066** branch sites. Work-list: **20** branches inventoried. **20** unit tests reified.

---

## Reproduction Instructions

### Prerequisites

1. Docker container `tl-char` running with `/work` mounted to the worktree root.
2. Worktree at `/work` (host path: `work/` under the worktree root).
3. Helper scripts at `/work/work/tensilelite-characterization/wf/parametric-chaos/`.

### Step 1 — Census (preflight + branch extraction)

```bash
# Run inside container
docker exec -w /work/projects/hipblaslt/tensilelite tl-char python3 \
  /work/work/tensilelite-characterization/wf/parametric-chaos/branch_extractor.py \
  --root . \
  --files Tensile/SolutionStructs/Solution.py Tensile/KernelWriter.py Tensile/KernelWriterAssembly.py \
  --outdir /work/work/tensilelite-characterization/parametric-chaos/CodegenResidue \
  --max-units 20
```

Produces:
- `branch_census.jsonl` (6,066 lines)
- `file_inventory.csv`

### Step 2 — Constraint Harvest

```bash
docker exec -w /work/projects/hipblaslt/tensilelite tl-char python3 \
  /work/work/tensilelite-characterization/wf/parametric-chaos/harvest_constraints.py \
  --root . \
  --outdir /work/work/tensilelite-characterization/parametric-chaos/CodegenResidue
```

Produces: `constraints_harvested.jsonl`

### Step 3 — Covering Array

```bash
docker exec -w /work/projects/hipblaslt/tensilelite tl-char python3 \
  /work/work/tensilelite-characterization/wf/parametric-chaos/covering_array.py \
  --frags /work/work/tensilelite-characterization/parametric-chaos/CodegenResidue/_frags \
  --outdir /work/work/tensilelite-characterization/parametric-chaos/CodegenResidue/covering_array \
  --strength 2
```

Produces: `covering_array/model.json`, `covering_array/cases.csv`

### Step 4 — Assemble Deliverables

```bash
docker exec -w /work/projects/hipblaslt/tensilelite tl-char python3 \
  /work/work/tensilelite-characterization/parametric-chaos/build_codegenresidue.py
```

Produces: `branch_parameter_hypergraph.json`, `domain_model.json`,
`characterization_catalog.jsonl`, `scorecard.json`

### Step 5 — Run Tests (pass-check)

```bash
docker exec -w /work/projects/hipblaslt/tensilelite tl-char \
  pytest -p no:cacheprovider -m unit -q \
  Tensile/Tests/unit/characterization/CodegenResidue/
```

### Step 6 — Finalize (authoritative scorecard via finalize.py)

```bash
docker exec -w /work/projects/hipblaslt/tensilelite tl-char python3 \
  /work/work/tensilelite-characterization/wf/parametric-chaos/finalize.py \
  --root . \
  --outdir /work/work/tensilelite-characterization/parametric-chaos/CodegenResidue \
  --testdir /work/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/CodegenResidue \
  --extractor /work/work/tensilelite-characterization/wf/parametric-chaos/branch_extractor.py \
  --files Tensile/SolutionStructs/Solution.py Tensile/KernelWriter.py Tensile/KernelWriterAssembly.py \
  --max-units 20
```

---

## Tool Versions (in tl-char container)

| Tool | Version |
|------|---------|
| Python | 3.12.10 |
| z3 | 4.16.0 |
| crosshair | 0.0.106 |
| hypothesis | 6.155.2 |
| pysmt | 0.9.6 |

---

## Static / Solver / Runtime Split

| Classification | Count | Description |
|----------------|-------|-------------|
| fully-static | 6 | Predicate is a pure function of YAML parameters; exhaustively verified by z3 + CrossHair |
| solver-backed-under-assumptions | 13 | Solver found SAT model under bounded encoding; results valid within stated assumptions |
| runtime-dependent | 1 | Predicate involves live loop state not closed-form solvable; classified UNKNOWN |

---

## Key Files

| File | Description |
|------|-------------|
| `preflight.json` | Container + environment readiness checks |
| `file_inventory.csv` | Branch site counts per file |
| `branch_census.jsonl` | Full census (6,066 branch records) |
| `constraints_harvested.jsonl` | Op-surface constraints from Configuration.py |
| `branch_parameter_hypergraph.json` | Nodes (branches) + edges (→ public inputs) |
| `domain_model.json` | Parameter domains per branch |
| `characterization_catalog.jsonl` | Joined v2 records (census + slice + domain + solver + verdict + reify) |
| `covering_array/model.json` | 2-way covering array model (12 params, 14 rows) |
| `covering_array/cases.csv` | Covering array cases |
| `validation_report.md` | Per-unit table with solver status and confirmation |
| `analyst_summary.md` | Clustered families, hotspots, caveats |
| `scorecard.json` | Authoritative numeric counts |
| `_frags/` | Per-phase per-branch JSON fragments |

---

## Known Limitations / Blind Spots

1. **covering_array** rows are not filtered for impossible combinations (e.g., `DirectToLds` requires `KernelLanguage=Assembly`).
2. **doReadA** (`KernelWriter.py:4065`) is UNKNOWN; no solver witness obtained.
3. **6,046 uncovered** branches remain in the census; only the top-20 work-list was characterized.
4. ISA-gated branches (ISA = specific GPU arch) require runtime confirmation; CPU-only pipeline treats them as bounded unknowns.
