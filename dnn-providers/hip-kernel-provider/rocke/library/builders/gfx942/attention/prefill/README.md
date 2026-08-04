# gfx942 dense flash-attention prefill (`attention_dense`) — AICK-1664

Port of the gfx950 dense prefill kernel (PR #9480 / AICK-1663) to **gfx942 (CDNA3)**.
This folder mirrors the gfx950 layout (`builders/gfx950/attention/prefill/`): the
kernel body lives in `kernels/gfx942/attention_dense.py`; this builder owns the host
path (spec → compile → launch → SDPA parity + benchmark).

## Status

**Shipped.** `build_attention_dense(arch="gfx942")` emits the tuned body: 32×32×8 atom
+ K-loop doubling, transposed-QK, conflict-free V at D128 fp16, `exp2_fast` +
fused softmax rescale, per-config `waves_per_eu`, the D64 K-bank-conflict pad, and
both the default and persistent (grid-stride) launch variants. Validated against an
fp32 SDPA reference across the in-scope cohort on **both** gfx942 parts (228-CU and
304-CU), 0 register spill within the VGPR/LDS budget.

The tuning is **per (head_size, dtype)**, not global — see the lever table below for
which config gets which lever and why. Levers default OFF at module scope
(`_P0_D64_KPAD`, `_P0_IGLP`), so a non-dispatch build is byte-identical and the
gfx950 golden is untouched by construction.

Still rejected with a structured reason by `supports_attention_dense`: varlen, ragged,
sliding-window. See the AICK-1664 plan for the deferred-findings backlog.

## Scope

gfx942 only · forward-inference prefill · dense causal (no paging / bias / SWA / sinks)
· bf16 + fp16 · head dims **D64 & D128** · MHA + GQA incl. non-power-of-2 (40/8, 28/4)
· default **and** persistent grids. D256 is out of scope (AICK-1495/1496).

`block_n` must divide the 256-row query tile, `block_n` over the 8 waves must give a
whole number of DMA row-groups, and `K_lds + V_lds` must fit the 64 KB gfx942 LDS —
all enforced by `supports_attention_dense`, so `supports(spec)[0] is True` implies
`build_attention_dense(spec)` succeeds. That equivalence is what stops dispatch from
selecting a spec it cannot build.

## Why a separate kernel (not an arch branch in the gfx950 file)

The gfx950 body bakes in CDNA4-only primitives; the algorithm genuinely diverges on
CDNA3, so the DSL convention (`dsl_docs/architecture/multi_arch_data_layout.md`) puts
it in a per-gfx module. This also keeps the gfx950 golden IR byte-identical by
construction. The CDNA3 deltas:

| Concern | gfx950 (CDNA4) | gfx942 (CDNA3) |
|---|---|---|
| MFMA atom | `mfma_f32_32x32x16` (K=16) | `mfma_f32_32x32x8` (K=8) → 2× per K=16 tile, A/B repack; **C-layout identical** |
| Conflict-free V | `ds_read_b64_tr_b16` (transpose read) | **no `ds_read_tr16`** → `perm_b32` store-path transpose |
| Cross-half exchange | `permlane32_swap` | **absent** → `perm_b32` / `ds_bpermute` |
| LDS / CUs | 160 KB, one CU count | 64 KB, two CU counts (228 / 304) → retune occupancy, `num_persistent` |
| Tile barrier | bare `s_barrier` is safe (NBUF=2 double buffer) | single buffer → the tile barrier **must** drain `lgkmcnt` (`sync_lds_only`) |

## Problem category (drives the optimization order)

**This kernel is occupancy-bound / MFMA-starved.** Not compute-bound, not
bandwidth-bound, and — importantly — **not** LDS-bank-conflict-bound, which is what the
inherited `attention_tiled_2d` framing assumed. rocprof PMC counters put mean occupancy
at a small fraction of the per-CU wave slots while the MFMA pipe sits far below the
compute-bound threshold, with L2 hit rate and memory-unit stall both ruling out the
memory path.

Two consequences worth stating explicitly, because both were mis-read at some point in
this port:

- The runbook's bottleneck decision tree short-circuits on low occupancy *first*, so
  the LDS-bound branch is structurally unreachable here regardless of conflict rate.
- `LDSBankConflict` counts conflicts **per LDS-active cycle**, so conflict-free V
  *raises* the rate while *lowering* the cost. Reading that rate as a bound is the trap
  that kept the LDS-bound framing alive.

The dominant remaining lever is therefore **occupancy** — getting a second workgroup
resident at D128, which needs an LDS cut and a register-floor cut *together* — plus
grid shape at small sequence lengths. Per `AGENTS.md`, measured counters, utilisation
and per-lever deltas are recorded outside the repo: see the AICK-1664 plan and the
protected results page.

## Lever record (gfx950 dense → gfx942, plus gfx942-only experiments)

Every lever evaluated for this port, with its verdict. Impact is qualitative;
magnitudes live outside the repo per `AGENTS.md`.

### Adopted

| Lever | Config | Mechanism | Verdict |
|---|---|---|---|
| 32×32×8 atom + K-loop doubling | all | CDNA3 has no 32×32×16 fp16/bf16 atom; C-layout is identical so softmax/epilogue port unchanged | **shipped** — enablement |
| Conflict-free V (`perm_b32` store transpose) | **D128 fp16** | V stored transposed → PV A-operand read is one contiguous `ds_read_b64` instead of 4 element-wise `ds_read_u16` | **shipped** — large; identity-preserving |
| `exp2_fast` | all except **bf16 D128** | softmax args are provably ≤ 0, so `llvm.exp2`'s guarded range reduction is dead work | **shipped** — dominant on the VALU-bound path; bit-identical |
| Fused softmax rescale | all | exp2 → accumulate → cast → pack in one pass instead of materializing a full f32 `p_vals` matrix | **shipped** — pure live-range relief, bit-identical |
| Per-config `waves_per_eu` | **bf16 D64** → 4 | forces the allocator low enough that a second workgroup co-resides (1 → 2 WG/CU) | **shipped** — large at long sequences |
| D64 K-bank-conflict pad | **D64 both dtypes** | 2-row-group boundary pad takes the `do_qk` K reads from 32-way to 4-way | **shipped** — large, cross-part confirmed |
| Persistent grid-stride | all | `num_persistent` CTAs grid-stride over decoded work items; qb-major and hkv-major decodes | **shipped** — large at long sequences; auto-on when work fills the grid |

### Evaluated and rejected

| Lever | Why it lost |
|---|---|
| Conflict-free V at **D64** | D64 is VGPR-bound; the register round-trip costs more than the LDS-instruction saving |
| Conflict-free V at **bf16 D128** | spills over the `waves_per_eu=2` cap on the `.1k` MFMA schedule |
| `exp2_fast` at **bf16 D128** | same — spills even after the fused rescale freed headroom |
| `waves_per_eu=3` at **fp16 D64** | reaches 2 WG/CU but loses more ILP than the second workgroup buys |
| Drop K/V LDS pads to reach 2 WG/CU at D128 | D128 stays 1 WG/CU even unpadded (register floor co-limits), so it only reintroduces bank conflicts — **catastrophic** |
| `block_n=32` (all configs) | halving the KV tile doubles the tile/grid count; the extra loop and barrier overhead outweighs the LDS relief. Marginally positive on **bf16 D128** but **part-dependent**, so not wired — it would need a CU-count-aware policy |
| `iglp_opt` (`_P0_IGLP`) | resource- and performance-neutral cross-part: the canned GEMM interleave does not match this loop, which is barrier-rendezvous-bound. Kept as a default-off knob |
| Smaller `BLOCK_M` | a *fully filled* grid at `BLOCK_M=64` measured **slower** than a one-third-filled one at 256 — 2 waves/CTA leaves half the CU's matrix cores unreachable at 1 CTA/CU. `BLOCK_M=128` is a small win on two configs only; not promoted |
| cfvst load/store chunking | does not bound register pressure: the later chunks' loads carry no dependency on the earlier ones, so LLVM hoists them above the intervening full `s_waitcnt(vmcnt=0)`, which then covers them anyway |
| PV-only `s_setprio` | **proven-negative** on gfx942 `attention_tiled_2d`. Note this is one lever in one placement — it is *not* a verdict on the scheduling-intrinsic family, and hand-written `sched_group_barrier` remains open |
| Diagonal two-phase causal peel | **proven-negative** on gfx942 `attention_tiled_2d`; gated on the bound shifting to compute, which it has not |
| partial-vmcnt software prefetch | N/A — a double-buffering lever, and NBUF=2 does not fit 64 KB LDS at D128 |

## Bench

```
python attention_dense_prefill.py                        # parity + bench, default shapes
python attention_dense_prefill.py --dtype fp16 --d 64
python attention_dense_prefill.py --persistent --np 304  # persistent grid
```

Full-cohort parity and perf are driven by the live harness at
`benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py` (`--mode all`),
which is the numeric gate for this kernel — the same role the bench plays on gfx950.

`_p0_occupancy.py` is the static resource guard (VGPR / AGPR / spill / LDS, comgr only,
no GPU): re-run it after any lever change to confirm 0 spill and to see which resource
is the occupancy limiter.

Per AGENTS.md, **measured throughput lives only in the protected results page, never in
the repo.**
