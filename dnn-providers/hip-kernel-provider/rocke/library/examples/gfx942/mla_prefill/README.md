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
  - [Rejected — num_warps 4 to 8](#rejected--num_warps-4-to-8)
  - [Rejected — r_kv_tile below 64](#rejected--r_kv_tile-below-64)
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

**Both terms bind here.** This kernel's LDS is well above the 32768 B that two
workgroups would require, *and* its VGPR count is above the 256 that two
waves/SIMD would require. The consequence is the central finding of this pass:

> An LDS-only reduction cannot improve occupancy on this kernel. Reaching two
> workgroups per CU requires LDS ≤ 32768 **and** VGPR ≤ 256 simultaneously.

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

Verdict: default retained at 64. This closes the open question of whether the
planned `r_kv_tile` reduction was worth pairing with a second lever — on the
evidence it is not, because the pairing partner it needs is a VGPR reduction, not
a second LDS reduction.

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

- **The 304-CU part was not run.** Only a 228-CU MI300A was reachable. Every
  workgroups/CU figure here is per-part and is *not* extrapolated to 304 CU.
- **Two workgroups per CU was never achieved,** and no combination reaching it was
  demonstrated. It needs a simultaneous LDS *and* VGPR reduction; this pass showed
  that no single available lever delivers both.
- **`num_warps = 8` was never parity-verified.** It was rejected on measurement
  before adoption, so its functional correctness at 8 warps is untested.
- **Two-pass latent expansion was not attempted.** The in-loop path re-expands
  each KV tile once per q-block, so its cost grows with the q extent — a real
  redundancy. But removing it adds a workspace buffer, a second launch and new
  host-visible pointer arguments, which makes it an algorithmic change, not a
  tuning lever. It belongs to its own milestone.
- **Bank-conflict degree was never measured directly,** only inferred from the
  stride arithmetic and confirmed end-to-end by the width sweep. An ATT trace
  would settle it; none was captured.
