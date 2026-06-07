# Analyst Summary — CodegenResidue Characterization (Run 3)

## Overview

Run 3 applied the parametric-chaos pipeline to the CodegenResidue surface:
the codegen-residue branches in `Tensile/KernelWriter.py`,
`Tensile/KernelWriterAssembly.py`, and `Tensile/SolutionStructs/Solution.py`.

The full census of 6,066 branch sites was reduced to a 20-branch work-list
ranked by the v2 prioritization heuristic. Of those 20:

- 19 were confirmed SAT (solver + witnesses + reified tests where applicable)
- 1 was classified UNKNOWN / runtime-dependent (`doReadA` accumulator at KernelWriter.py:4065)
- 15 were reified as unit tests under `Tensile/Tests/unit/characterization/CodegenResidue/`

---

## Clustered Branch Families

### 1. HalfPLR Bit-Extraction Family (KernelWriter.py:4072 and 4152)

Branches `3a433f9e` (HalfPLRA) and `6c1a0094` (HalfPLRB) are siblings.

- `HalfPLRA = bool(HalfPLR & 0x01)` — True when `HalfPLR` in `{1, 3}`
- `HalfPLRB = bool(HalfPLR & 0x02)` — True when `HalfPLR` in `{2, 3}`

Both branches are **fully-static**: they are pure functions of the integer YAML
parameter `HalfPLR` (valid values `{0,1,2,3}`). No ISA probes or GPU state
involved. The solver (`z3`) exhaustively enumerated all 4 values and confirmed
both branches. Both are reified as unit tests.

**Covering implication**: A test suite covering `HalfPLR` in `{0, 1, 2, 3}` exercises
all four (HalfPLRA, HalfPLRB) polarity combinations. Minimum covering set is
`HalfPLR in {0, 1, 2, 3}` — all four are needed to fully differentiate.

### 2. ScheduleIterAlg Equality Family (KernelWriter.py:882 and 951)

Branches `7b6b7c5f` (scheduleIterAlg == 0) and `1c015182` (scheduleIterAlg == 1)
are siblings guarding different schedule paths in `_makeSubIterSchedule`.

- Public inputs: `ScheduleIterAlg`, `KernelLanguage`
- Both confirmed SAT with witnesses; both reified with 24 tests each.

**Covering implication**: `ScheduleIterAlg` needs values `{0, 1}` at minimum.
`KernelLanguage` must be `"Assembly"` for the assembly code path.

### 3. Loop-Body Read Gating Family (KernelWriter.py:4065, 4072, 4145, 4152)

Four branches in `_loopBody` guard local-read operations:

| branch | predicate | classification |
|--------|-----------|---------------|
| `462514797d` | `doReadA` | runtime-dependent (UNKNOWN) |
| `3a433f9e` | `kernel["HalfPLRA"]` | fully-static |
| `4108a067` | `doReadB` (at gate L4145) | solver-backed |
| `6c1a0094` | `kernel["HalfPLRB"]` | fully-static |

`doReadA` and `doReadB` are loop-iteration-local Boolean accumulators. Their
value is a function of iteration index `u`, `iui`, and multiple YAML parameters
(`LoopIters`, `InnerUnroll`, `PrefetchLocalRead`, etc.), but the combination rule
involves a live `u < ...` comparison that cannot be closed-form solved without
fixing `u`. The HalfPLRA/B sub-guards are pure and confirmed.

### 4. Sgpr Preload Family (KernelWriterAssembly.py:2266 and 2267)

Branches `8e5e9525` and `dc455979` guard consecutive sgpr-preload code sections.

- `8e5e9525`: `numSgprPreload > 0` — fully-static over `{PreloadKernArgs, ISA, ProblemType.Batched}`
- `dc455979`: adjacent guard with same driver variables

Both confirmed SAT, both reified as unit tests.

### 5. groOffsetInMacroTile Family (KernelWriterAssembly.py:4250)

Branch `0902ebf1`: `self.states.groOffsetInMacroTile` — a derived integer set to
1 iff `len(PackedC0IndicesX)==1 and len(PackedC1IndicesX)==1 and BufferLoad`.

Confirmed SAT. Not reified to a new test by the Reify frag, but 23 tests were
authored for this branch and the file exists (`test_pchaos_KernelWriterAssembly_L4250_char.py`).

### 6. TailLoop Residue Family (KernelWriterAssembly.py:7089, KernelWriter.py related)

Branch `4480256b` at KWA:7089 guards tail-loop handling based on
`{AssertSummationElementMultiple, DepthU}`. Confirmed SAT.

### 7. DirectToLds Family (KernelWriterAssembly.py:10892)

Branch `c8562779`: `directToLdsM0Update` at L10892, driven by `DirectToLds` parameter.
`DirectToLds` takes values `{0,1,2,3}` (bit field). Confirmed SAT with 9 tests.

### 8. Large-stride Bias/TDMInst Family (KernelWriterAssembly.py:16798)

Branch `aabf0d22`: complex predicate over `{TDMInst, BufferLoad, AssertFree0ElementMultiple, GlobalReadVectorWidthA}`.
Fully-static, confirmed SAT, 7 targeted test cases reified.

---

## Canonical Worked Example: KernelWriter.py:4152 (HalfPLRB)

The `kernel["HalfPLRB"]` branch at line 4152 is the simplest worked example of the
full pipeline:

1. **Census**: branch_extractor emits branch_id `6c1a0094...`
2. **Slice**: public input `HalfPLR` (YAML int in `{0,1,2,3}`); derived symbol `HalfPLRB = bool(HalfPLR & 0x02)`.
3. **Domain**: `HalfPLR` domain `{0, 2}` (minimum covering pair); `HalfPLRB` domain `{False, True}`.
4. **Solve**: z3 encodes `HalfPLR` as 8-bit BitVec bounded to `[0,3]`, derives `HalfPLRB := (HalfPLR & 0x02) != 0`. TRUE-witness: `HalfPLR=2`. FALSE-witness: `HalfPLR=0`.
5. **Verify**: CrossHair finds no counterexample to `__return__ == (half_plr in (2, 3))` under `pre: 0 <= half_plr <= 3`.
6. **Reify**: `test_pchaos_KernelWriter_L4152_char.py` pins actual behavior; all assertions pass.

---

## Prioritized Hotspots

In priority order for follow-up characterization or test investment:

1. **doReadA accumulator** (`KernelWriter.py:4065`): The only UNKNOWN. A bounded symbolic execution with explicit loop unrolling (fixing `LoopIters` to small values) could produce partial witnesses. Currently zero test coverage from the pipeline.
2. **overflowedResources gate** (`KernelWriterAssembly.py:1902`): Classified SAT-bounded; the public inputs list is long (22 YAML + ISA parameters). The covering array covers the principal axes but cross-parameter interactions are unverified.
3. **groOffsetInMacroTile** (`KernelWriterAssembly.py:4250`): Confirmed and 23 tests written, but the Reify frag did not set `reified=True`. The test file is present and passes; update the Reify frag to reflect this.

---

## Caveats and Blind Spots

- **covering_array constraints not wired**: The `covering_array/model.json` notes that impossible parameter combinations (e.g., `DirectToLds` requires `KernelLanguage=Assembly`) are not enforced during row generation. Run-1 covering array rows should be filtered for feasibility before test generation.
- **ISA-gated branches**: Branches `4944b8f5` (`UseSubtileImpl` + ISA gfx950) and `dc455979` (ISA sgpr-preload gate) have runtime ISA dependencies. Solver confirmations are conditional on the ISA bitvector being within enumerated values. GPU-probe branches are out of scope for CPU-only characterization.
- **6,046 uncovered branches**: Only 20 of 6,066 branches were characterized. The 20-branch work-list targets the highest-priority residue. The remaining branches are predominantly intermediate codegen control-flow with low public-input exposure.
- **e85e407e test failure**: The Reify frag for `KernelWriterAssembly.py:7475` records `passed=False`. This gap should be investigated before promoting that test to the gate.
