# Case study — gfx942 bf16 MLA prefill, LDS and occupancy pass

**Kernel builder:** [`build_mla_prefill_fwd`](../../../kernels/mla/mla_prefill_gfx942.py)
**Arch:** gfx942 (MI300A, 228 CU)
**Scope:** resource characterisation and single-lever tuning. No functional change.

---

## Table of Contents

- [What this records](#what-this-records)
- [Reproducing the functional baseline](#reproducing-the-functional-baseline)
- [Reading the static resource report](#reading-the-static-resource-report)
- [The occupancy model for this kernel](#the-occupancy-model-for-this-kernel)
- [Lever ledger](#lever-ledger)
  - [Kept — wt_lds bank-conflict pad](#kept--wt_lds-bank-conflict-pad)
  - [Kept — kv_lds pad narrowed, staging width halved](#kept--kv_lds-pad-narrowed-staging-width-halved)
  - [Kept — register-resident k-loop prefetch](#kept--register-resident-k-loop-prefetch)
  - [Kept — score-GEMM read-ahead scheduler pin](#kept--score-gemm-read-ahead-scheduler-pin)
  - [Rejected — kv_lds XOR swizzle](#rejected--kv_lds-xor-swizzle)
  - [Rejected — num_warps 4 to 8](#rejected--num_warps-4-to-8)
  - [Rejected — r_kv_tile below 64](#rejected--r_kv_tile-below-64)
  - [Rejected — canned iglp_opt whole-loop interleave](#rejected--canned-iglp_opt-whole-loop-interleave)
- [Second pass — trace-guided levers](#second-pass--trace-guided-levers)
  - [Capturing a trace](#capturing-a-trace)
  - [Kept — score GEMM split across waves](#kept--score-gemm-split-across-waves)
  - [Kept — causal loop bound](#kept--causal-loop-bound)
  - [Kept — read-path transpose for the PV operand](#kept--read-path-transpose-for-the-pv-operand)
  - [Kept — two workgroups per CU](#kept--two-workgroups-per-cu)
  - [Rejected in the second pass](#rejected-in-the-second-pass)
- [Third pass — counter-guided levers](#third-pass--counter-guided-levers)
  - [Reading the counters](#reading-the-counters)
  - [Kept — transposed score and PV](#kept--transposed-score-and-pv)
  - [Kept — two query M tiles per workgroup](#kept--two-query-m-tiles-per-workgroup)
  - [Kept — the smaller VALU levers](#kept--the-smaller-valu-levers)
- [Fourth pass — parallel lanes](#fourth-pass--parallel-lanes)
  - [Diagnosis](#diagnosis)
  - [Kept — four heads per workgroup](#kept--four-heads-per-workgroup)
  - [Measured and not stacked](#measured-and-not-stacked)
- [Method](#method)
- [What remains unproven](#what-remains-unproven)

---

## What this records

Methodology and levers only. **Software-achieved performance figures — wall-clock,
throughput, TFLOP/s, runtime occupancy, speedup ratios — are deliberately absent
and must never be added here.** They are reported out of band. What is recorded
below are static properties of the compiled binary, which are not achieved
numbers, plus the qualitative keep/reject verdict for each lever.

## Reproducing the functional baseline

Tuning is only interpretable against a verified functional state, so reproduce it
first rather than inheriting it:

```bash
source .sandbox-env
export ROCKE_CPP_QUIET_FALLBACK=1
cd dnn-providers/hip-kernel-provider/rocke
"$PY" -m builders.mla.verify_score_probe_gfx942 --mode full --scenario all
```

Expect `SUMMARY mode=full 8/8 passed`. Every lever below was gated on re-running
this in full — not a subset. A lever that fails any case is reverted, not
investigated and kept.

CPU-side:

```bash
"$PY" -m pytest library/tests/test_mla_prefill_gfx942.py \
                library/tests/test_mla_prefill_gfx942_golden.py \
                library/tests/test_mla_reference.py \
                library/tests/test_mla_benchmark_shapes.py \
                library/tests/dispatch/mla
```

There is no byte-identity gate for this family — MLA has no C++ engine mirror, so
the Python lowering is the ground truth and the IR golden takes its place. The
golden is recorded per LLVM flavor; re-bless with

```bash
cd library && ROCKE_LLVM_FLAVOR=llvm20 "$PY" tests/test_mla_prefill_gfx942_golden.py --write
cd library && ROCKE_LLVM_FLAVOR=llvm22 "$PY" tests/test_mla_prefill_gfx942_golden.py --write
```

and never as a way to turn red green.

## Reading the static resource report

Two independent sources are cross-checked, and they must agree:

- the `@smem_pool.<kernel>` global in the lowered `.ll` (pre-codegen intent);
- `.amdhsa_group_segment_fixed_size` / `.amdhsa_next_free_vgpr` /
  `.amdhsa_private_segment_fixed_size` in the compiled ELF (what actually shipped),
  via [`rocke.analysis.isa.analyze_hsaco`](../../../../platform/python/rocke/analysis/isa.py).

`scratch_bytes > 0` is a spill and is a hard stop: revert the lever, do not tune
around it in the same step.

Note `analyze_hsaco` reports no occupancy — it must be derived. The VGPR-limited
estimate in
[`rocke.benchmark.perf.occupancy`](../../../../platform/python/rocke/benchmark/perf/occupancy.py)
is VGPR-only and knows nothing about LDS, so an LDS-limited kernel will be
over-reported by it. Derive both limits and take the minimum.

## The occupancy model for this kernel

gfx942: 65536 B LDS per workgroup, 512 VGPRs per SIMD, 4 SIMDs per CU, unified
VGPR/AGPR file (so an AGPR count of 0 alongside nonzero `accvgpr_*` instruction
counts is normal — the reported VGPR figure is the total).

```
workgroups/CU = min( LDS_per_CU // lds_bytes ,
                     (waves_per_simd_vgpr * SIMDS_per_CU) // waves_per_workgroup )
waves_per_simd_vgpr = min( 512 // round_up(vgpr, 8), 8 )
```

**Both terms bound in the first pass.** The kernel's LDS was well above the
32768 B that two workgroups would require, *and* its VGPR count was above the 256
that two waves/SIMD would require. The consequence was the central finding of
that pass:

> An LDS-only reduction cannot improve occupancy on this kernel. Reaching two
> workgroups per CU requires LDS ≤ 32768 **and** VGPR ≤ 256 simultaneously.

The second pass met both conditions: splitting the score GEMM across waves
(lever 10) brought VGPR under 256, and levers 20–22 brought the pool to 31488 B.
By the model above the kernel now fits two workgroups per CU; see
[Kept — two workgroups per CU](#kept--two-workgroups-per-cu).

This corrects the planning assumption that framed occupancy purely as an LDS
budget problem. It also makes "output-accumulator relocation" inapplicable: `O`,
`m` and `l` are already register-resident `scf_for` iter-args, not LDS buffers, so
there is nothing to move out of LDS — and moving them *into* LDS would spend the
resource with less slack to relieve the one with more.

## Lever ledger

One lever per step. Full 8-case parity after each. Keep or revert, never stack.

| # | Lever | Verdict | Basis |
|---|---|---|---|
| 1 | `wt_lds` bank-conflict pad (`WT_PAD`) | **Kept** | free in instructions and occupancy; measured improvement |
| 2 | `num_warps` 4 → 8 | Rejected | occupancy improved, measured latency regressed |
| 3 | `r_kv_tile` 64 → 32, 64 → 16 | Rejected | no occupancy change at any legal value; small measured regression |
| 4 | `kv_lds` pad `V8_PAD` → `WT_PAD`, `C_STAGE_W` 8 → 4 | **Kept** | pool total and occupancy unchanged; small static cost; measured improvement on every shape |
| 5 | `kv_lds` XOR swizzle | Rejected | structurally impossible without breaking a declared store alignment |
| 6 | Register-resident k-loop prefetch | **Kept** | LDS pool and barrier count unchanged, no spill; measured improvement on every shape |
| 7 | Canned `iglp_opt` whole-loop interleave | Rejected | both levels regressed measurably; level 1 undoes lever 6 by shortening live ranges |
| 8 | Score-GEMM read-ahead `sched_group_barrier` pin | **Kept** | instruction mix, LDS pool and barrier count unchanged, VGPR fell, no spill; measured improvement on every shape |
| 9 | P-publish barriers `sync` → `sync_lds_only` | **Kept** | the VMEM drain stalled on the run-ahead loads; measured improvement on every shape |
| 10 | Score GEMM K reduction split across waves | **Kept** | MFMA and LDS reads per wave cut, +1 barrier, VGPR fell, pool unchanged; measured improvement on every shape |
| 11 | Row reduce masks 1/2 on `quad_perm` DPP | **Kept** | half the `ds_swizzle`s; bit-identical; measured improvement on every shape |
| 12 | Causal k-loop bound | **Kept** | skips fully masked key tiles; bit-identical; large gain on full-prompt shapes, neutral on chunked |
| 13 | Prologue/epilogue read-ahead pin | Rejected | measured slightly slower |
| 14 | `q_lds` bank-conflict pad | **Kept** | absorb A read was single-bank; measured improvement on short shapes |
| 15 | `wq_lds` pad with 4-wide staging | Rejected | doubled staging ops cost more than the 2-way conflict saved |
| 16 | Heaviest query tiles launched first | **Kept** | reverse block order; measured improvement, none slower |
| 17 | `wt_lds` XOR swizzle on the epilogue store | **Kept** | store spread from ~4 banks; measured improvement on short shapes |
| 18 | Register prefetch of W slices | Rejected | neutral; the compiler already overlapped them |
| 19 | Drop the redundant WAR barrier before P | Rejected | provably redundant but neutral |
| 20 | Read-path transpose for the PV operand | **Kept** | `ct_lds` removed (−20 KB LDS, −32 `ds_write` per tile); measured improvement on every shape |
| 21 | Score A operand held in registers | **Kept** | `qa_lds` dead in the loop; measured improvement on every shape |
| 22 | `r_kv_tile` 64 → 32 with a gated absorb | **Kept** | pool 31488 B and VGPR 228: two workgroups per CU; large gain on every shape |
| 23 | Read-path transpose in the epilogue | Rejected | neutral at two workgroups per CU |
| 24 | No AGPRs (`agpr_alloc = (0, 0)`) | **Kept** | the rescale no longer copies accumulators out of and back into AGPRs; VGPR fell; measured improvement on every shape |
| 25 | `exp2_fast` in the online softmax | **Kept** | one `v_exp_f32` per exp, no range-reduction guard; measured improvement on every shape |
| 26 | Transposed score and PV (`Sᵀ`, `accᵀ`) | **Kept** | P stays in registers, one barrier per tile fewer, row reduce mostly in-lane; large gain on every shape |
| 27 | Buffer loads for the k-tile stage | **Kept** | page base in the SGPR offset; no per-tile address math; measured improvement on every shape |
| 28 | Unmasked main loop, masked tail loop | **Kept** | per-element masks only on the last tile or two; small gain, none clearly slower |
| 29 | `block_q` 16 → 32, prologue and epilogue staged by column window | **Kept** | each staged key tile feeds two query tiles; pool shrank, VGPR under 256, no spill: still two workgroups per CU; large gain, none slower |
| 30 | Block-table lookup one tile ahead | **Kept** | the page index is in a register before its tile's loads issue; small gain, none clearly slower |
| 31 | Four heads per workgroup, one wave per head (`heads_per_wg = 4`, `block_q = 16`), selected by dispatch | **Kept** | each staged key tile feeds four heads; no score split or partial combine; absorbed query never touches LDS; still two workgroups per CU; large gain on every shape |
| 32 | Permuted-row PV gather on the four-heads path | **Kept** | one `ds_read_b64` of a key row feeds four PV tiles instead of sixteen scalar reads; large gain on every shape |
| 33 | Occupancy guard tests | **Kept** | CPU tests fail if the shipped or the fallback layout leaves two workgroups per CU |

### Kept — wt_lds bank-conflict pad

`wt_lds` is `[d_nope, r_kv_tile]` bf16. At the natural shape its row stride is
`r_kv_tile * 2 == 128 B`, **exactly the 32-dword LDS bank width**, so every row of
the tile starts on bank 0. Both accesses walk the row index and therefore
serialise: the staging store steps by 8 rows, and the MFMA B-operand read — the
hot one, inside the k loop — steps by one row per lane, collapsing all 16
`lane_row` values onto a single bank.

The fix is an allocation-shape change only: `[d_nope, r_kv_tile + WT_PAD]`. Every
index expression keeps its logical range, so the kernel computes the same values;
`WT_PAD = 0` restores the natural layout. The pad must stay **even**, because
`smem_load_vN(..., n=4)` reads 8 B and an odd element pad misaligns every row.

Static effect: LDS grows by `d_nope * WT_PAD * 2` bytes; VGPR, SGPR and scratch
are unchanged; `ds_read`, `ds_write` and `mfma` counts are *identical* to the
unpadded build. The pad is pure addressing — it costs no instructions, and it
cannot perturb occupancy here because both limits already pin the kernel at one
workgroup per CU with LDS slack to spare.

**Choose the pad width empirically.** Bank-conflict degree is a runtime stall
property and appears in no static report, so the width cannot be derived from the
resource report. Widths 0, 2, 4 and 8 were swept; the ordering was consistent
across shapes and `WT_PAD = 4` won. Two different a-priori bank models disagreed
with each other *and* one of them disagreed with the measurement — which is the
reason the sweep exists. Do not re-derive the width; re-measure it if the access
pattern changes.

### Kept — kv_lds pad narrowed, staging width halved

Same class of defect as lever 1, one buffer over. `kv_lds` is
`[block_k, qa_cols + pad]` bf16 and was allocated with `V8_PAD = 8`. A pad of 8
bf16 elements displaces each row by 16 B — **4 dwords** — so the row stride is
`≡ 4 (mod 32)` and the bank sequence repeats with period 8. The hot score-GEMM
B read walks 16 `lane_row` values per execution group, so those 16 lanes land on
only 8 distinct banks: a clean 2-way conflict on the k-loop's critical read.

Getting off bank 0 is necessary but **not sufficient** — the period of the row
stride modulo 32 is what matters. `WT_PAD = 4` displaces by 8 B (2 dwords),
giving stride `≡ 2 (mod 32)`, period 16, and a conflict-free read for a 16-lane
group. The pad shrinks and the access gets *better*, which is why an "add more
padding" heuristic would have missed it.

The pad cannot be narrowed alone. [`smem_store_vN`](../../../../platform/python/rocke/core/ir.py#L1694)
*declares* `align = n * elem_bytes` unconditionally and verifies nothing, so an
`n = 8` store into a row that is only 8 B-aligned is a **silent miscompile**, not
an assert — the two constants are load-bearing for each other. Narrowing the pad
therefore requires narrowing the `c_kv` staging store to match, which is what
`C_STAGE_W = 4` does; the constant is named so the coupling is visible at the
definition site rather than implied by the allocation.

Static effect: the emitted LDS pool total is **unchanged**. The lowerer pools by
liveness with greedy first-fit, so shrinking one buffer only widens a stranded
hole — the arithmetic sum is not the pool. Occupancy is therefore untouched, and
there are no spills. The cost is `+4 ds_write`, `+4 global_load` and `+10` VGPR
(against a 256 ceiling), because the staging loop runs twice as many iterations
at half the width. The per-element `ct_lds` stores are count-invariant across the
change — 8 chunks × 4 elements and 4 chunks × 8 elements emit the same number —
so `ds_read`, `mfma` and `global_store` counts are identical.

The same empirical rule as lever 1 applies, and it is the reason this was
measured rather than reasoned: narrowing the staging width is a *real* static
cost, so a derived bank model claiming a win is not enough to keep the lever.
A/B it and keep it only if the wall clock agrees.

### Kept — register-resident k-loop prefetch

The k loop loaded tile `i`'s `c_kv` and `k_rope` from global **between its two
barriers**, so the full global latency sat exposed with no compute over it. At one
workgroup per CU and one wave per SIMD there is no other wave to switch to, so an
exposed load is a stall in the literal sense — which is what makes this the
highest-value shape of lever in this regime.

The staging block is split into a load half that touches no LDS and a store half
that touches no global, with one tile in flight carried across the iteration
boundary in the `scf_for` `iter_args`. A prologue primes tile 0; each iteration
drains the stage it was handed, then issues tile `i+1`'s loads, then runs the
score GEMM, so those loads retire under compute.

Body order is load-bearing: WAR barrier, store, RAW barrier, **then** the
run-ahead loads, then compute. Issuing the loads before the RAW barrier looks more
natural and is wrong — [`b.sync()`](../../../../platform/python/rocke/core/ir.py#L3177)
lowers to `s_waitcnt vmcnt(0) lgkmcnt(0)` ahead of `s_barrier`, so it drains the
VMEM stream and anything issued earlier is waited on immediately, collapsing the
hiding window back to zero.

The stage is carried in **registers**, not a second LDS buffer. The structural
donor, [`build_gfx942_4warp_gqa`](../../../kernels/gfx942/attention_tiled_2d.py#L6236)
([`fill_load`](../../../kernels/gfx942/attention_tiled_2d.py#L6260) /
[`fill_store`](../../../kernels/gfx942/attention_tiled_2d.py#L6276) /
[`body`](../../../kernels/gfx942/attention_tiled_2d.py#L6286)), uses a two-slot LDS
double buffer and gets down to one barrier per iteration as a result. That is not
available here: the pool is already near the 64 KiB ceiling and a second
`kv_lds`/`ct_lds` slot needs tens of KiB more. Registers were the one budget with
headroom, so the lever spends them instead — and the price is that the single LDS
buffer forces WAR and RAW to stay separate, leaving the barrier count at two.
Note that
[`_issue_k_load_runtime`](../../../kernels/gfx942/attention_tiled_2d.py#L2671) is
*not* the donor despite the suggestive name; it is an async global→LDS DMA path
that holds nothing in registers.

Two edge cases, both handled without adding control flow. `n_k_tiles` is zero when
the KV extent is zero and the early return guards on *q*, not *k*, while
`iter_args` initialisers must be defined unconditionally — so the prologue selects
the *page value* down to page 0 (always mapped) rather than predicating the load.
On the final iteration the next-tile index is **clamped rather than predicated**,
re-reading the current tile, whose page is already known mapped; the value is
yielded and never consumed.

Static effect: the LDS pool total, every slot offset, `ds_read`, `ds_write`,
`v_mfma` and `s_barrier` counts are all **unchanged**; the only delta is
`+10 global_load`, exactly the hoisted prologue. That the instruction mix is
otherwise identical is the evidence that this is a pure scheduling change — the
same work, issued earlier. It costs VGPRs (still well inside the 256 ceiling) and
produces no spill; a nonzero `scratch_bytes` here would have been a hard stop,
not something to tune around.

One correctness note worth keeping. Parity reported a worst-case residual slightly
above the figure recorded for the previous lever. A prefetch must not change
arithmetic, so rather than shrug at a difference well inside tolerance, the base
kernel was checked out and parity re-run: residuals are **bit-identical per
sequence**, and the earlier figure was a mis-transcription. One checkout turned
"probably fine" into "provably unchanged."

### Kept — score-GEMM read-ahead scheduler pin

Lever 6 issues the global loads a tile early; what the scheduler then interleaves
between them and the score GEMM decides how much of that latency is actually
covered. This lever is the natural follow-on, and it is a **call-site hint only** —
no structural change, no new buffer, no new barrier.

The shape of the score GEMM is what makes a hint worth placing. With
`block_q == block_k == MFMA_M == MFMA_N == 16` there is exactly one M tile and one
K tile, so each of its `SCORE_K_ITERS` steps is **two `ds_read`s feeding a single
MFMA**, and every MFMA of the loop accumulates into the *same* register. The MFMA
chain is therefore serial: the GEMM is LDS-read-bound with a dependent accumulator,
not MFMA-throughput-bound.

That rules out copying the donor's hint verbatim. The neighbouring gfx942 attention
kernel pins its k-loop in **lockstep**, one step at a time
([`_sched_group_pin_mfma_step`](../../../kernels/gfx942/attention_tiled_2d.py#L158),
called at [:4249](../../../kernels/gfx942/attention_tiled_2d.py#L4249) with
`mfma_count=1`). A 1:1 pin keeps each read adjacent to the MFMA that consumes it,
which is right when the MFMAs are independent and wrong here, where covering a
serial chain needs the reads to run *ahead* of it. The adopted form emits a
`(ds_read × 2G, MFMA × G)` pair every `G` steps — a block of reads hoisted above
the MFMAs they feed, `G` steps deep — via
[`b.sched_group_barrier()`](../../../../platform/python/rocke/core/ir.py#L3392).

`G` must divide the trip count. A group that leaves a partial trailing block
measured worse than placing no hint at all, so the emission is guarded on
`SCORE_K_ITERS % SCORE_SGB_GROUP == 0` and silently omits itself otherwise. Depth
was swept across the dividing values and the deepest one measured best, consistent
with the serial-chain reading above; the non-dividing candidate lost, as predicted.
`SCORE_SGB_GROUP` is a scheduling hint, so a wrong value costs speed and never
correctness — but it is still swept, not derived.

Static effect: `mfma`, `ds_read`, `ds_write`, `s_barrier` and the LDS pool are all
**unchanged** — the same instructions in a different order, which is exactly what a
scheduler hint should produce. VGPR *fell*, and `scratch_bytes` stayed zero.

Note this lever and `iglp_opt` are mutually exclusive: `iglp_opt` claims the whole
loop schedule, and the donor kernel
[raises if both flags are set](../../../kernels/gfx942/attention_tiled_2d.py#L772).
Rejecting `iglp_opt` (below) is what made this lever available.

### Rejected — kv_lds XOR swizzle

The usual alternative to a pad is an XOR swizzle, which costs no LDS at all.
It is structurally impossible on this buffer.

To break the period-8 pattern the swizzle must displace a row by an odd number
of dword *pairs* — an offset `≡ 2 (mod 4)` dwords. Working back through the
column addressing, the only column bit that produces that displacement is bit 2.
But bit 2 is precisely the bit an `n = 8` vector store must hold invariant to
keep its declared 16 B alignment true. Any swizzle strong enough to fix the
conflict breaks the alignment the store has already declared — and per the note
above, that failure is silent.

Narrowing the store to `n = 4` frees bit 2, but at that point `WT_PAD` alone
already makes the read conflict-free, so the swizzle buys nothing over lever 4.
Recorded here so the idea is not re-derived: the blocker is the alignment
contract, not the bank arithmetic.

### Rejected — num_warps 4 to 8

The most interesting negative result here, and the one worth not repeating.

Statically this lever looks like a clear win: doubling the warp count halves the
per-wave accumulator footprint, which drops VGPR **below the 256 threshold** and
so **doubles waves/SIMD** — the only single lever found that moves occupancy at
all, since it raises waves per workgroup rather than workgroups per CU. Scratch
stays at zero and LDS is untouched.

It measured *slower*, consistently, on every shape tried.

The plausible mechanism: at 8 warps each wave owns half as many n-tiles, so there
is less independent MFMA work per wave to hide LDS latency, while the per-slice
barriers now rendezvous twice as many waves. Occupancy bought less than ILP and
barrier cost gave up. Not investigated further, because the verdict does not
depend on the mechanism.

The lesson is general enough to state plainly: **a static occupancy improvement is
a hypothesis, not a result.** This is exactly why the loop requires a measurement
per lever rather than a resource report per lever.

Also recorded by the same sweep: `num_warps` of 1 and 2 **spill** — nonzero
scratch, badly so at 1. They are inadmissible under the stop condition and should
not be revisited.

### Rejected — r_kv_tile below 64

`r_kv_tile` is the only spec field that scales an LDS buffer (`wt_lds`); every
other buffer is sized by `block_q`, `block_k`, `r_kv`, `d_nope` or `d_v`.

Halving it does reduce LDS, but **not enough to matter**: even the smallest legal
value stays well above the 32768 B two-workgroup threshold, and VGPR barely moves
(it stays above 256 at every setting), so workgroups/CU and waves/SIMD are
unchanged across the whole sweep. Meanwhile the slice count `r_kv / r_kv_tile`
rises, adding barriers for the same total staging work, and the measurement
regressed monotonically as the tile shrank.

`r_kv_tile = 128` does not compile — the resulting pool exceeds the 65536 B
workgroup limit and comgr fails in `CODEGEN_BC_TO_RELOCATABLE`. That is the upper
bound, not a tuning candidate.

Verdict at the time: default retained at 64. The pairing partner it needed was a
VGPR reduction, not a second LDS reduction — and once levers 10 and 20–21
supplied that, lever 22 reversed this verdict. See
[Kept — two workgroups per CU](#kept--two-workgroups-per-cu).

### Rejected — canned iglp_opt whole-loop interleave

[`b.iglp_opt(level)`](../../../../platform/python/rocke/core/ir.py#L3371) asks the
post-RA scheduler to apply a canned interleaving to the enclosing loop — level 0 is
the GEMM MFMA-interleave pattern, level 1 the attention-style one. It is a one-line
lever, so it was tried before the hand-placed pin.

Both levels regressed on every shape, level 1 severely. The instruction mix is
identical to the baseline at both levels, so the regression is purely ordering —
and the VGPR count is the tell: at level 1 it drops **sharply** while nothing else
moves. Shorter live ranges here mean the canned pattern pulled the run-ahead global
loads back down toward their consumers, which is precisely the hiding window lever 6
exists to open. The canned attention pattern assumes a loop whose loads are issued
in place; this loop is no longer that loop.

This matches the codebase's own posture rather than contradicting it: on the
neighbouring gfx942 attention kernel both
[`use_iglp_opt`](../../../kernels/gfx942/attention_tiled_2d.py#L451) and
[`use_qk_pv_sched_group_barrier`](../../../kernels/gfx942/attention_tiled_2d.py#L457)
default **off**, and the two are mutually exclusive by construction.

Verdict: reverted, and the hint budget spent on the hand-placed pin above instead.
The general lesson is that a canned schedule is not a free win once a kernel has a
hand-built latency-hiding structure — it can silently dismantle it, and the static
resource report shows *that* as an improvement.

## Second pass — trace-guided levers

Levers 9–23 were chosen from ATT traces rather than from the static report, and
each was timed against the previous kept state before its parity run. All were
measured on a 304-CU MI300X across the DeepSeek-V3 shape set of
[`benchmark_mla_prefill_live`](../../../benchmarks/gfx942/attention/prefill/benchmark_mla_prefill_live.py).

### Capturing a trace

```bash
W=$ROCKE/dsl_docs/optimization/utilities/tools/wavescope
cd library
ROCKE_COMGR_LIB=/opt/rocm/lib/libamd_comgr.so.3 \
"$PY" $W/capture_wavescope_trace.py --kernel-regex mla_prefill_fwd_gfx942 \
  -- "$PY" -m benchmarks.gfx942.attention.prefill.benchmark_mla_prefill_live \
     --mode time --model DeepSeek --regime full_prompt --limit 1 --no-aiter
```

`ROCKE_COMGR_LIB` is required. `rocprofv3` preloads `/opt/rocm`'s comgr and
LLVM, and rocke otherwise loads torch's bundled comgr into the same process; the
two collide and the compile segfaults. Open the reported
`ui_output_*_dispatch_*` folder with **WaveScope: Open Trace Folder…** and switch
the Source tab to **+ inlined**.

Two cautions from this pass. A trace covers one CU, so at two workgroups per CU
one workgroup's stalls are hidden by the other's work: stall share stops
predicting wall time (levers 13, 18, 19 and 23 all looked worthwhile in the
trace and measured neutral). And `code.json` stall columns are totals over every
execution; divide by `Hit`.

### Kept — score GEMM split across waves

Every wave used to compute the whole score tile redundantly. Each wave now
reduces a quarter of the K = 576 axis, parks its f32 partial C fragment in LDS,
and after one LDS-only barrier every wave sums the four partials in the same
order — so the softmax state stays bit-identical across waves.

The partials live in tail rows of `kv_lds` rather than in their own buffer. The
LDS packer reuses a dead slot only from its base offset, so a separate buffer
could never land in the gap `kv_lds` leaves below the next live buffer and would
have opened a new slot above 64 KB. See
[`_s_part_rows`](../../../kernels/mla/mla_prefill_gfx942.py).

### Kept — causal loop bound

The k-loop ran over every key tile and masked the ones past the causal edge. It
now stops at the last tile any row of the query tile can see, and a tile whose
rows are all fully masked runs zero iterations — finishing with `l == 0`, which
is what the dead-row checks require. Launching the heaviest query tiles first
(lever 16) followed from the same change, since per-workgroup work now varies.

### Kept — read-path transpose for the PV operand

gfx942 has no transposing LDS read, so the PV GEMM's B operand used to come from
`ct_lds`, a second copy of the latent tile scattered in one bf16 at a time. It is
now gathered from `kv_lds` with four scalar reads per lane; at the `kv_lds`
stride those reads are conflict-free. That removed the scatter from the
critical section between barriers and freed 20 KB of LDS, which levers 21–22
spent on occupancy.

### Kept — two workgroups per CU

Three changes together: the score A operand is read into registers once (lever
21), so the prologue trio is dead during the loop; Q_rope is staged straight
into `qa_lds`; and `r_kv_tile` drops to 32, with the absorb's two n-tiles run on
two of the four waves inside a barrier-free `scf.if`. Allocation order matters —
each later phase allocates its small buffer first so the packer gives it the base
slot. The static result is a 31488 B pool and 228 VGPR: two workgroups per CU.

### Rejected in the second pass

- **Prologue/epilogue read-ahead pin (13).** The absorb's reads were already
  issued ahead; their cost was a single-bank conflict, which lever 14 fixed.
- **`wq_lds` 4-wide staging (15).** Doubling the staging ops cost more than the
  2-way read conflict saved, and the shrunk buffer stopped the k-loop buffer from
  folding into the prologue's slot.
- **W-slice register prefetch (18)** and **dropping the redundant P barrier
  (19).** Both neutral: the compiler already overlapped the loads, and the waves
  arrive at that barrier together.
- **Epilogue read-path transpose (23).** Neutral once a second workgroup hides
  the epilogue.

## Third pass — counter-guided levers

At two workgroups per CU a single-CU trace stops ranking levers reliably, so
levers 24–28 were chosen from PMC counters and the k-loop's instruction mix
instead, on the same part and shape set as the second pass.

### Reading the counters

```bash
rocprofv3 --pmc SQ_WAVE_CYCLES SQ_BUSY_CYCLES SQ_WAIT_ANY SQ_WAIT_INST_LDS \
                SQ_ACTIVE_INST_LDS SQ_ACTIVE_INST_VALU SQ_ACTIVE_INST_ANY SQ_INSTS_LDS \
  --kernel-include-regex mla_prefill_fwd -- <bench command>
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE SQ_VALU_MFMA_BUSY_CYCLES \
                SQ_BUSY_CU_CYCLES SQ_INSTS_MFMA SQ_INSTS_VALU SQ_INSTS_SALU \
  --kernel-include-regex mla_prefill_fwd -- <bench command>
```

`ROCKE_COMGR_LIB` is needed here too (see [Capturing a trace](#capturing-a-trace)).
They showed the loop issuing many VALU instructions per MFMA, with LDS bank
conflicts a small share of LDS-active cycles. The instruction mix of the k-loop
body then said where the VALU went: the accumulator rescale's AGPR round trip,
the `exp2` range-reduction guard, the 16-lane row reduce, 64-bit addresses for the
tile loads, and the per-element masks.

### Kept — transposed score and PV

The score GEMM's two operands are swapped, so it produces `Sᵀ` (key x query). In
that fragment a lane owns one query, so the softmax row reduce is in-lane over the
lane's four keys plus two cross-lane stages, instead of four stages per row for
four rows. `P` comes out already laid out as the B operand of the PV GEMM, which is
also swapped to produce `accᵀ`: `P` never goes through LDS, and the barrier pair
around it is gone. The rescale by `alpha` becomes lane-local. The epilogue's
`W_UV` GEMM is transposed the same way, so each lane normalizes its own query and
writes four consecutive output columns at once.

The removed `p_lds` had been the small buffer that kept `kv_lds` below the prologue
peak (see [Kept — two workgroups per CU](#kept--two-workgroups-per-cu)), so the score
partials move out of `kv_lds` into their own `s_part` buffer and take that role.

### Kept — two query M tiles per workgroup

At `block_q = 32` every key tile staged into `kv_lds`, and every PV operand
gathered from it, feeds two M tiles' MFMAs, so the loop's staging, barriers and
gathers are amortised over twice the queries. The k-loop body itself is the
same code with `M_TILES = 2`; what had to change is the prologue and epilogue,
because a full-height `qa_lds` or `accl_lds` would each pass 32 KB on its own.

Both now stage by column window rather than by M tile, so every weight slice is
staged once and feeds both M tiles:

- **Prologue.** `Q_nope` is read from global straight into the absorb's A
  fragments, so `q_lds` is gone. `qa_lds` holds both M tiles but only half the
  absorbed columns, and each half is exactly the score K-steps of two waves:
  after each half, every wave reads it and keeps the value only if the half is
  its own (`scf.if` carries no results, so the keep is a `select`).
- **Epilogue.** `accl_lds` holds both M tiles but one wave's latent columns.
  Each wave in turn writes its accumulators, and the `W_UV` projection walks
  only that wave's r-slices.

The pool is smaller than at `block_q = 16`, VGPR stays under 256 and nothing
spills, so two workgroups still fit per CU.

Two simpler layouts were measured first and lost. With the prologue and
epilogue simply run once per 16-row M tile, the weights are staged twice and
the gain is smaller; that version also spilled until each later M tile took its
weight base through an opaque `pin_sgpr` copy, because LLVM proved the second
M tile's weight loads identical to the first's and kept every slice live across
the boundary. With full-height buffers the pool rises past 32 KB, the kernel
drops to one workgroup per CU, and it is slower than `block_q = 16` on almost
every shape -- the occupancy, not the tile reuse alone, is what pays.

### Kept — the smaller VALU levers

- **No AGPRs (24).** Left free, the backend parks the PV accumulators in AGPRs and
  every per-tile rescale reads them out and writes them back. gfx942 MFMAs use
  arch VGPRs directly, and the register budget still fits two waves per SIMD.
- **`exp2_fast` (25).** Both softmax exponents are `<= 0`, so the guard against
  overflow is dead weight and an underflow to 0 is the right answer.
- **Buffer loads (27).** The tile stage uses `raw_ptr_buffer_load` with each thread's
  in-page byte offset computed once outside the loop and the page base, which is
  uniform across the workgroup, in the SGPR offset.
- **Loop split (28).** A key tile is fully visible when its last key is below `S_k`
  and at or below the causal bound of the tile's first query row. Those tiles run
  in a loop with no per-element masks; the rest run in a masked tail loop that
  continues from the first loop's results.

## Fourth pass — parallel lanes

Levers 29–33 came from a run of independent lanes, each timed back to back
against the same baseline on one 304-CU MI300X, with GPU access serialized so no
two timings overlapped. Kept lanes were then stacked one at a time and re-timed.

### Diagnosis

- **Occupancy is observed, not just derived.** Wave-count counters put the
  kernel near two workgroups per CU while busy on long shapes, a little lower on
  `s512` from tail imbalance; forcing one workgroup per CU made every shape
  slower.
- **Not cache-bandwidth bound.** L2 and L1-to-L2 traffic stay well under peak on
  long and short shapes. Long shapes are issue-bound with the LDS pipe the most
  loaded unit; `s512` is latency-bound in the per-workgroup weight staging.
- **LDS bank conflicts are a small share** of LDS-active cycles.
- **aiter `mla_prefill_fwd`** (absorb mode) runs the DeepSeek-V3 shapes and is the
  reference to beat; it does not run the 64-head shapes.

### Kept — four heads per workgroup

Every head of a workgroup reads the same `c_kv` and `k_rope`, so staging a key tile
once for four heads quadruples its reuse. Each wave owns one head outright: it
computes the head's full 36-step score (no split-K partials, no combine
barrier), runs its own softmax, and accumulates all 32 latent tiles.

- **Prologue.** The query absorb runs transposed, `Qaᵀ = W_UK · Q_nopeᵀ`, with
  both operands read from global. Its C fragment is exactly the score GEMM's B
  operand, so the absorbed query never touches LDS.
- **Epilogue.** `accᵀ` feeds `W_UVᵀ · accᵀ` directly as the B operand; only
  `W_UV` is staged, cooperatively for the four heads.
- **PV gather (32).** Each PV tile's latent rows are permuted so one 8-byte read
  of a key row covers four tiles; the epilogue stages `W_UV` through the same
  permutation. `Q_rope` is parked in LDS to keep VGPR at the two-workgroup line.

The four-heads layout needs `block_q = 16` (at 32 the accumulators spill) and no
AGPRs (with AGPRs it falls to one workgroup per CU and the rescale copies
return). Dispatch selects it whenever the head count divides by four and
admission passes; otherwise it falls back to the single-head spec defaults,
which the `fwd_*` golden cases pin. The verifier's full mode runs whatever
dispatch selects.

### Measured and not stacked

- **`block_q = 32` with eight waves at one workgroup per CU**, and **two heads per
  workgroup with eight waves**: both faster than the single-head base, both
  superseded by lever 31.
- **Unfused projections** (absorb and `W_UV` as batched GEMMs outside the kernel,
  timed end to end): faster, but it changes the kernel signature. Blocked on the
  provider API.
- **`iglp_opt(1)` on the main loop** and a **grouped PV gather**: both measured
  faster on the single-head path only, which dispatch now uses just as a fallback.
  Neither has been timed on the four-heads path.
- **`amdgpu-waves-per-eu = 2`**: compiles to identical ISA; nothing to keep.
- **Other knob re-sweeps** (`SCORE_SGB_GROUP`, `WT_PAD`, `r_kv_tile = 64`): neutral,
  slower or inadmissible.

## Method

Per lever, in order, with no deviation:

1. record the static resource report;
2. change **exactly one** lever;
3. re-run the **full** 8-case parity;
4. record the static resource report again;
5. check the stop condition — `scratch > 0` is a spill; revert and say so;
6. keep or revert, and write one qualitative line.

Levers are never stacked. The entire value of the pass is attribution, and two
levers moved together destroy it. Where a candidate could be evaluated through a
spec override without editing the kernel, it was evaluated that way and rejected
before adoption — so no parity budget was spent proving out a lever that lost on
measurement.

Timing instruments and their logs are session-local and git-excluded by design.
They are not deliverables and their output does not belong in this repository.

## What remains unproven

- **The first pass ran on a 228-CU MI300A and the second on a 304-CU MI300X.**
  First-pass verdicts were not re-measured on the 304-CU part.
- **`num_warps = 8` was never parity-verified.** It was rejected on measurement
  before adoption, so its functional correctness at 8 warps is untested.
- **Two-pass latent expansion was not attempted.** The in-loop path re-expands
  each KV tile once per q-block, so its cost grows with the q extent — a real
  redundancy. But removing it adds a workspace buffer, a second launch and new
  host-visible pointer arguments, which makes it an algorithmic change, not a
  tuning lever. It belongs to its own milestone.
- **The buffer-load bound is by design, not tested at scale.** The range check
  covers each load's in-page offset, not the page base in the SGPR offset, so the
  reachable cache size is the same 32-bit byte range as the global loads it
  replaced. No test reaches a cache that large.
- **Untried next levers.** The `s512` shapes, where per-workgroup fixed cost
  dominates, are furthest from aiter; the unfused projections address that
  directly but need an API change. The four-heads loop still spends a packed
  multiply per accumulator register per tile on the online-softmax rescale; a
  lazy rescale would need loop-carried state out of a branch, which `scf.if`
  cannot carry. `iglp_opt` and the grouped gather have only been timed on the
  single-head path.
