# Square GEMM: climbing from the tile-pipeline baseline toward matrix peak

A case study in taking a square `fp16` GEMM from the stock CK DSL tile pipeline
to the matrix unit's throughput ceiling on **MI355X / gfx950 (CDNA4)** — one
measured technique at a time, each verified for correctness (`bad=0`) and read
straight off the hardware counters.

The headline result is a single, reproducible **ladder** at one fixed geometry
(256×256×64, 4×4 warps, 8192³, `fp16`):

| # | technique layered on | TF/s | % of matrix peak† | what it removes |
|---|---|---:|---:|---|
| A | plain tile pipeline (VGPR-staged double buffer) | 589 | 21% | — (starting point) |
| B | + direct-to-LDS load | 647 | 23% | the global→VGPR→LDS round-trip |
| C | + depth-2 prefetch, **single-barrier** mainloop | 1026 | 36% | the per-tile barrier bubble |
| D | + 4-way LDS swizzle | 1448 | 51% | 52% → 25% bank conflict |
| E | + **element-granular swizzle** | **1522** | **54%** | 25% → **0%** bank conflict |

†"Matrix peak" is the matrix-issue ceiling the `MfmaUtil` counter measures
against — i.e. the rate at which the MFMA unit can retire `v_mfma_f32_16x16x32`
back-to-back. % of peak is `MfmaUtil` read directly; on this part it implies a
ceiling of ≈2.8 PF/s `fp16`. **2.6× from baseline to E, every rung bit-exact.**

The rest of this document is how each rung was found and proven, the things that
*looked* like levers but weren't, and exactly what stands between 54% and 100%.

## Tools

The whole study is counter- and ISA-driven, never guessed:

* **`rocprofv3`** — `MfmaUtil` (distance to peak), `LDSBankConflict`,
  `OccupancyPercent`, `MemUnitStalled`. These four decide every question.
* **ISA inspection** — `KernelDef` → LLVM IR → `libamd_comgr` HSACO →
  `llvm-objdump` (via `ck_dsl.analysis.analyze_hsaco`); opcode histogram +
  resource counts so the emitted mainloop can be read instruction-for-instruction.
* **Isolated-subprocess sweeps** — every config benched in its own process, so a
  single bad geometry that GPU-faults can't poison the rest of a sweep.
* **LDS round-trip dump** — write a tile through the real load path, read it back
  row-major, diff vs input — used to prove a layout correct *before* chasing a
  perf bug as a correctness bug.

## Hardware / software pin

| | |
|---|---|
| GPU | MI355X / gfx950 (CDNA4), 160 KB LDS/CU, 32 LDS banks |
| ROCm | 7.2 |
| atom | `v_mfma_f32_16x16x32` (CDNA4 wide-K), wave64 |
| shape | square `fp16`, RCR, 4096³ / 8192³ |

---

## Rung A → B: direct-to-LDS

The stock pipeline stages every tile global→VGPR→LDS (a `buffer_load` then a
`ds_write`). The hardware can stream global memory **straight into LDS**
(`buffer_load_dwordx4 ... lds`, via `async_buffer_load_lds_addr`), eliminating
the `ds_write` entirely and the VGPRs that held the staged data. ISA-confirmed:
**0 `ds_write`** in the mainloop. `TraitSpec.direct_to_lds`. Small alone
(589→647) but it's the foundation the prefetch and swizzle both build on.

## Rung B → C: depth-2 prefetch with a single barrier (the biggest jump)

This is where most of the climb happens (647 → 1026, +59%). Two ideas:

**Depth-2 ping-pong.** Two LDS half-buffers; while the MFMAs consume half *p*,
the next K-tile streams into half *p^1*. `TraitSpec.dtl_prefetch`.

**One barrier per K-tile, not two.** A double-buffer has a write-after-read
hazard (this iteration overwrites the half the *previous* iteration just read)
and a read-after-write hazard (this iteration reads the half just loaded). The
natural-but-wrong structure spends a barrier on each. At **1 workgroup per CU**
there is no second workgroup to hide a barrier bubble, so a second barrier
directly halves the time the MFMA unit has to run. The fix is *ordering* — issue
the next-tile async write **after** the single barrier:

```
for each K-tile (parity p, other half p^1):
    s_waitcnt(vmcnt=0, lgkmcnt=0)   # vmcnt0: half(p) has LANDED          (RAW)
                                    # lgkmcnt0: prev reads of half(p^1) DRAINED (WAR)
    s_barrier                       # one WG rendezvous: half(p) visible to all
    load  -> half(p^1)             # async; issued AFTER the drain, so it cannot
                                    #   race the just-finished reads -> no 2nd barrier
    mma   <- half(p)               # matrix work overlaps the async HBM transfer
```

One `s_waitcnt` + one bare `s_barrier` cover **both** hazards because the write
comes after the drain. (`_emit_kloop_prefetch`.) This was found via a real bug:
a depth-2 race that appeared only at 3–4 K-tiles — diagnosed with the LDS dump
(layout correct), a discriminating test matrix (only the prefetch path wrong),
and a K-sweep (the 3–4-tile signature of a race). The first fix added a *second*
barrier and worked but cost the bubble above; collapsing back to one — correctly
— is rung C.

## Rung C → D → E: killing LDS bank conflicts

With the mainloop tight, the counter said **52% `LDSBankConflict`** and 35%
`MfmaUtil` — the MFMA unit was starving on LDS reads. This took two rungs and a
piece of arithmetic.

**Why the conflict exists.** The A tile is row-major `[M][K]` with K = 64 halves
= 128 bytes = **exactly 32 bank-words**. The bank an element hits is
`(row·stride + col)/2 mod 32` — and with stride a multiple of the bank count,
the `row` term **vanishes**:

```
bank = (row·64 + col)·2/4 mod 32 = (row·32 + col/2) mod 32 = (col/2) mod 32
```

So every one of the 16 M-rows in an atom reads the *same* column → the *same*
banks. A stride alias, not a layout accident.

**The tool is a swizzle, not padding.** XOR a function of the row into the
column: `bank = ((col ^ f(row))/2) mod 32`. Now `f(row)` can spread rows across
banks. Two things decide how far it spreads:

* **Granularity.** Swizzle whole `ds_read_b128` slots (8 halves = bit 3), so
  every target stays 16-byte aligned. The LDS write is hardware-contiguous and
  the read recomputes the same XOR, so it's a bijection — **bit-exact** with no
  layout change (XOR is its own inverse: the load fetches `col^f(row)`, the read
  asks for `col^f(row)`, they cancel).
* **Which bits, how many.** The aliasing dimension is `m_in_atom` (0…15) — the
  **low** row bits — and there are `block_k/8 = 8` slots to spread into.

**Rung D — the first swizzle keyed the wrong bits.** A 2-slot toggle on a *high*
row bit (`col ^= ((row>>3)&1)<<4`) cut 52% → **25%** (51% util, 1448 TF). Better,
but stuck: it distinguishes warp groups, not the rows inside an atom, and 2 slots
against a 16-row alias is 4-way = 25% by construction.

**Rung E — element-granular, low-bit, all-slots → 0%.** Deriving the parameters
from the geometry instead of guessing — granularity `L = log2(8) = 3`, use all
slots `W = log2(block_k/8) = 3`, key on the low bits `R = 0`:

```
col ^= (row & 7) << 3          # permute all 8 b128 slots by the low row bits
```

drives `LDSBankConflict` to **0.0%**, lifts `MfmaUtil` to **54%**, and is
**bit-exact** (verified race-free on the K-sweep 1…13 tiles + rectangular +
large). This is now the auto-derived default for `lds_swizzle` (it computes
`L/W/R` from the atom and `block_k`; `CK_SWZ_R/W/L` override for experiments).
**Yes — zero bank conflicts is achievable on this access pattern, and this is the
mechanism.**

## What looked like a lever but wasn't (measured negatives)

* **Higher occupancy.** Hypothesis: smaller tiles → 2+ WG/CU → hide the barrier.
  Measured the opposite — every 2+ WG/CU geometry is far slower, because `tk=32`
  halves MFMAs/tile (barrier cost per FLOP explodes) and smaller M/N kills reuse:

  | tile / warps | WG/CU | TF/s |
  |---|---|---:|
  | **256×256×64 4×4** | **1** | **1522** |
  | 128×128×64 2×2 | 2 | ~870 |
  | 256×256×32 4×4 | 2 | ~810 |
  | 128×128×32 2×2 | 5 | ~430 |

  The big tile at 1 WG/CU wins decisively. Geometry is settled, not a lever.
* **In-phase prefetch-local-read.** Reading the next k-step's fragments before
  the current MFMAs: depth-1 was **neutral** (the backend's schedule hint
  already fills the in-tile shadow); reading a whole step ahead spilled 16 live
  fragments to memory and **halved** throughput. Removed.
* **LDS padding.** Widening the row stride also breaks the alias, but it spends
  LDS the 1-WG/CU tile can't give up (>160 KB) and measured net-negative.
  The *free* fix (swizzle) wins; the *LDS-spending* one loses on occupancy.

## Rung F: the L2 cache lever (`dtl_cache_b=ALL`, +9%)

The direct-to-LDS global loads carry a cache-coherency hint. B defaulted to
`CACHE_STREAM` (SLC set — "one-shot, don't pollute L2"), which is right for a
weight matrix read once. But a **square** GEMM reuses the *whole* B across every
M-tile, so streaming it is exactly wrong: setting `dtl_cache_b=CACHE_ALL` keeps B
L2-resident and recovers **+9%** (0.735→0.80× rocBLAS, measured interleaved,
`bad=0`). Chiplet/XCD grid swizzle (`chiplet_swizzle`, `chunk=32`) and an
`amdgpu-waves-per-eu` hint add ~0.5–1% on top → **~0.81× rocBLAS**.

## What looked like a lever but wasn't — extended (from the moonshot)

A 13-agent workflow mined the rocBLAS Tensile solution for large square and
implemented seven of its techniques as ck_dsl experiments (all `bad=0`, all in
isolated copies). **None beat the cache-tuned baseline.** The decisive one,
**depth-2 global prefetch (PGR2)**, was then built *correctly* end-to-end with a
real register-staging path (see "What the IR path cannot do" below) — it works
and is bit-exact but lands at 0.42× because the `ds_write` it requires can't be
hidden without assembly-level scheduling. `ScheduleIterAlg=3` eliminated all 12
register spills but code bloat cancelled it;
deeper `DepthU` lost the double-buffer. Recon also confirmed rocBLAS uses the
**same geometry we converged on** independently: `DepthU=64`, 256-wide macro-tile,
double-buffered.

## How this compares to hand-tuned assembly (rocBLAS)

rocBLAS GEMM is **Tensile-generated, hand-scheduled GCN assembly** — every
instruction placed, operands register-double-buffered. Our kernel is authored in
a Python DSL and lowered through LLVM IR (`libamd_comgr`). Measured **like-for-
like** (fp16, same shape, same process, interleaved cycles so both see the same
clock, all `bad=0`):

| fp16 @ 8192³ | TF/s | ratio |
|---|---:|---:|
| rocBLAS | ~1900 | 1.00× |
| **ck_dsl (DTL + cache lever)** | ~1550 | **~0.82×** |
| ck_dsl (DTL, default cache) | ~1410 | ~0.73× |

Measured same-session, interleaved, fully warmed, all `bad=0`. (Only same-session
ratios are meaningful here — absolute TF drifts with the GPU's clock, so the
kernels are always compared back-to-back in the same run.)

### What the IR path matched

Independently of rocBLAS we reached its **exact geometry** (`DepthU=64`, 256-wide
macro-tile, double-buffered — confirmed by mining its Tensile solution), achieved
**0 LDS bank conflicts**, and emit the same direct-to-LDS + double-buffer
skeleton. The residual ~15% is **instruction scheduling**, and we proved that by
building the missing piece, not guessing.

### What the IR path cannot do — built and measured

The one rocBLAS lever that should close the gap is **PGR2** (depth-2 global
prefetch with register-staged operands). We implemented it fully: split the global
`buffer_load` (→ VGPR) from the `ds_write`, carried **two** staged tiles across
the single barrier as loop-carried registers (tile *i+3* loaded at iter *i*,
written at iter *i+2* — two MFMA phases of latency hiding), K-boundary clamped.

It is **numerically correct** (rel-error 0 across 1–13 tiles, rect, large) — but
**761 TF, 0.42×, half the DTL path's speed.** The reason is decisive: register
staging *requires* an explicit `ds_write` to LDS (the fused direct-to-LDS path
has **zero** `ds_write`), and `comgr` does not schedule that `ds_write` into the
MFMA shadow, so it costs full time and competes with the operand `ds_read`s. The
two capabilities are mutually exclusive in the IR world:

* **direct-to-LDS:** 0 `ds_write`, but depth-1 prefetch only (the fused load
  can't be register-staged).
* **register-staged:** depth-2 prefetch, but pays the `ds_write`.

rocBLAS gets **both** only because hand-written assembly places each `ds_write`
into a specific MFMA issue slot so it is hidden (`ScheduleIterAlg=3`). That
instruction-level placement is the **irreducible** missing capability — not the
register-staging primitive (we built that), but the scheduler that hides its cost.
A DSL→IR→comgr pipeline emits IR and lets the backend schedule; it cannot place
individual instructions at that granularity.

We also tried the *sanctioned* IR-level scheduling lever — `__builtin_amdgcn_iglp_opt`
(the intrinsic CK's own attention pipeline uses; added to the DSL as `b.iglp_opt`)
— which asks the backend to apply its canned GEMM MFMA/memory interleave. On the
fast DTL path it is **exactly neutral** (1559→1561 TF), because that path has
**no `ds_write`s to interleave** and is barrier-bound, not schedule-bound: there
is nothing for the scheduler to reorder. It would only bite on the `ds_write`-heavy
register-staged path, where the `ds_write` cost dominates anyway. So even the
sanctioned scheduling hint cannot move the DTL ceiling.

**Net:** a Python-authored kernel, lowered through IR, lands at **~0.81–0.85×
rocBLAS** on the same tile geometry with zero bank conflicts — within ~15–19% of
hand-tuned assembly, and that remaining gap is characterized down to a single
capability the IR path structurally lacks (assembly-grade `ds_write`/MFMA
interleaving), *proven by building the alternative and measuring it lose*. Every
number here is a same-session, `bad=0` measurement on the pinned hardware.
