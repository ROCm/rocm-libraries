# StreamK on gfx1100 — what was asked, what was found

Answers the original brief: *enable Atomic where possible; enable a different StreamK
version, or improve StreamK3; then benchmark all.* Start here; detail is in the linked
reports.

**36 000 GPU measurements in `results/` for the gate work (9000 + 15 000 + 12 000), plus the
earlier SK3/SK4/SK5 sweeps. Zero failed measurements in any campaign.**

---

## The three asks

### 1. Enable Atomic → **impossible on this part, and the gate that blocks it tests the wrong thing**

`StreamKAtomic` cannot run on gfx1100. Three independent blockers, any one sufficient
(`REPORT.md` §12):

- `Solution.py:1686` rejects unless `EnableMatrixInstruction`
- `Solution.py:1713` rejects unless the **input** type is fp32
- gfx1100 has **no fp32 MatrixInstruction** (WMMA is f16/bf16/i8/i4 → fp32 accumulate)

Those are jointly unsatisfiable. True HHS is doubly out: it writes fp16 D, the emitted
instruction is `BufferAtomicAddF32`, and gfx1100 has no `global_atomic_pk_add_f16`.

**The interesting part is that the gate is wrong on its own terms.** The atomic writes **D**,
so the real predicate is the *destination* type — and the machinery already keys on compute
type elsewhere (`KernelWriter.py:7453`). For **HSS** (fp16 in → fp32 out) accumulator,
compute type and D are all fp32 and `HasAtomicAdd` is true. That is the only legal route to
the atomic path on this hardware, and it is adjacent to the target case, not it.

### 2. Enable a different StreamK version → **done: SK4/SK5 now build on RDNA for the first time**

`StreamK: 4` and `5` passed validation but **failed at the assembler on every RDNA target** —
a gfx12-only `th:TH_ATOMIC_RETURN` emitted unconditionally where gfx11 spells it `glc`.
One-line fix; both capability check and helper already existed.
→ `UPSTREAM_PR_1_glc.md`, patch `artifacts/rocisa_glc_fix.patch`

| | before | after |
|---|---|---|
| assembler errors | 6 | **0** |
| solutions surviving KernelWriter | 3/12 | **6/6** |
| SK4 validation | *could not build* | **48/48 PASSED** |

2064 validated SK4 measurements, zero failures. **But SK4 is not worth using**: best SK4
anywhere is **95.0% of best SK3**, and on **like-for-like configurations SK4 costs 10.6%**
(median 89.38% over 2288 matched pairs — identical solution, only the SK mode differing).
Quote 89.4% for *"what does choosing SK4 cost"*, 95.0% for *"how good can SK4 get"*; the
gap between them is best-vs-best letting each mode pick its own parameters. Both recomputed
from raw data over all 46 run directories with the mandatory validation filter applied
(7977 records, **0 FAILED**). It wins in 8 of 35 measured cells, all sharing one
signature (`CLR0` + no prefetch + DepthU 16/32 + ≥1 ms) — prefetch needs to know what comes
next, and work-stealing cannot. SK5 is not a third algorithm; it emits SK3 or SK4 and picks
at runtime.

**Also found and fixed a GPU crash**: `TENSILE_STREAMK_TILES > tiles` wraps a `uint32_t` and
faults (25 launch failures reproduced). The SK3 path already had the clamp; SK4 was missing
it. → `UPSTREAM_PR_2_clamp.md`

### 3. Improve StreamK3 → **done: a size gate worth +1.3% to +2.4% of wall-clock**

Re-analysing the grid census showed StreamK is a per-shape **win** and a wall-clock **loss**
(96.95% geomean vs 102.17% throughput). Cause is pure weighting: sub-0.1 ms shapes are 53%
of the suite by count and **5% by time**, and that is where StreamK wins; ≥1 ms is 10% by
count and **69% by time**, and that is where it loses.

So gate it on size. `ORIGAMI_MN_GATE` in `streamk.cpp` — above `M*N`, return `tiles`, which
is exactly what `grid_data_parallel()` returns. Env-gated, **default 0 is byte-identical to
stock**. → `GATED_POLICY.md` (offline), `GATE_RESULT.md` (measured)

**+1.3% to +2.4% of wall-clock. The range is protocol, not uncertainty** — see the
reconciliation table in `GATE_RESULT.md`, which is the only place to quote gate numbers from.
Reproduced across two library builds and two protocols; survives jackknife; no confound.

The threshold is a **plateau, not a fitted constant**: cuts of 1e6 / 2.867e6 / 1e7 land
within 0.05 pt (15 000 measurements).

---

## The caveat that governs all of it

**A default gfx1100 build ships no StreamK kernels at all.** Across `Logic/asm_full/navi31/`:
**2560 `StreamK: 0`** against **22 `StreamK: 3`**, and all 22 sit in `Experimental/`, which
`tasks.py` excludes by default. The library every measurement used (`~/exp/stock`) is a
purpose-built SK3 catalog — commit `82580dfc726 "prune gfx1100 logic to the SK3 Prediction
catalog only"`.

And **shipping StreamK would not be a win**: 97.91% of the shipped SK0 library without the
gate, 100.25% with it. Parity at best. → `SHIP_TEST.md`

So: the gate is a correct answer to *"given StreamK kernels, when should the launch stay
data-parallel"*. It is not a drop-in win, because there are no StreamK kernels to gate.

---

## What needs your decision

1. **File the two bug fixes upstream?** Both are correctness, both are one line, both are
   independent of any of the above — they matter whatever happens to StreamK. Blast radius
   is nil: SK4/SK5 ship nowhere, and gfx12 behaviour is byte-identical.
2. **Pursue the gate?** It works, but cannot fire on a default build. Worth it only if
   shipping StreamK for navi31 is on the table — and §3 above says that is roughly parity.
3. **Revert the patches in `~/exp/stock`?** All are env-gated and default to stock, so they
   are inert. I have not reverted them; all are saved in `artifacts/` with `PATCHES.md`.

## The transferable findings

Independent of gfx1100, StreamK, or hipBLASLt — these are in `RUNBOOK.md`:

- **Metric choice decided the answer three separate times.** Per-shape geomean and
  wall-clock disagreed *in sign* on StreamK-vs-DP, on the gate, and on shipping StreamK.
  Report both, and say which one the decision rests on.
- **Prefer an inert *partition* to an inert *arm*.** A region where all arms are identical
  *by construction* caught 0.31 pt of arm-position drift that an A/A arm structurally cannot
  see — and that drift had produced a clean, monotone, entirely false ranking.
- **Jackknife every wall-clock comparison.** On this suite the top 10 shapes hold 18–24% of
  kernel time; a verdict flipped sign after dropping five. I stated and retracted two
  headlines before enforcing this.
- **A projection built by recombining measured arms flatters configurations never run** —
  optimistic twice here, by 0.7 pt and on the low-threshold behaviour entirely.
