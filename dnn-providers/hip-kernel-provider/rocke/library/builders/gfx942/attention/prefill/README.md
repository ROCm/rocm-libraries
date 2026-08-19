# gfx942 dense flash-attention prefill (`attention_dense`)

Port of the gfx950 dense prefill kernel (PR #9480) to **gfx942 (CDNA3)**.
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
which config gets which lever and why. The levers do **not** all default off, and the
split matters when you build a spec by hand:

- The D64 K-bank-conflict pad is **ON by default**. It is the shared
  `AttentionDenseSpec.lds_k_group_pad`, whose default is 8, so a directly-built D64
  spec already carries it — the `kpad8` token in the kernel name is what says so. Set
  the field to `0` to get the unpadded layout; that is also how the lever is A/B'd.
- The gfx942-private knobs in `Gfx942DenseTuning` (`block_m`, the two LDS pads,
  the `use_cfvst` / `use_exp2_fast` overrides, `iglp`) all default to their shipped
  values, and the tri-state ones default to `None` = "ask the policy". A build that
  never mentions the struct therefore emits byte-identical IR under a byte-identical
  kernel name.

The gfx950 golden is untouched either way: this is a separate kernel module emitting
its own symbol against its own fixture, not an arch branch in the gfx950 file.

Still rejected with a structured reason by `supports_attention_dense`: varlen, ragged,
sliding-window. The deferred-findings backlog lives in the optimization plan for this
port, which is kept outside the repo.

### Gain over the unified-attention baseline

Relative gain of this kernel over the in-tree unified attention path
(`attention_tiled_2d`, the `unified_2d` / `dense_pipe` family). Same HIP-event timer,
same inputs, same fp32 paged reference for both arms, all cells `PASS` at that
reference; one run on a 304-CU gfx942 part. Every cell is a **ratio**, so per
`AGENTS.md` the absolute throughputs stay in the protected results page.

Both arms run the config that actually **ships**: dense builds its spec through the
production dispatch factory (`dispatch/attention/gfx942.py::_dense_spec`, so the
measured binary carries the shipped `waves_per_eu` / `d64_kpad` / persistent
decision), and the baseline takes `wide4`, `narrow` or `narrow_d64` as
`attention_tiled_2d` selects for that shape.

| config | vs unified attention | sequence lengths |
|---|---|---|
| bf16 D128 | **+63 % → +244 %** | 2048 → 8192 |
| fp16 D128 | **+65 % → +146 %** | 2048 → 8192, plus MHA16 at 4096 |
| fp16 D64 | **+79 % → +144 %** | 2048 → 8192 |
| bf16 D64 | **+48 % → +112 %** | 2048 → 8192 |

Every config wins at every measured length, and the margin grows with sequence length
— the persistent grid turns on at the long end, and the causal work per tile rises
faster than the fixed overhead. Two things worth knowing about how to read the table:

- **The D64 rows moved a long way, and the D64 K-bank-conflict pad is why.** An
  earlier comparison, run before the pad was adopted, measured D64 at −17 % (fp16,
  short) to +29 %. Re-running with the pad in place is what removed the regression: an
  explicit ON/OFF A/B in the same harness puts it at **1.88–2.13× (fp16)** and
  **1.72–1.86× (bf16)** on the D64 path alone, matching the independent cross-part
  figure recorded when the lever was adopted. A benchmark that hand-builds its spec
  will silently miss a lever like this; building it from the dispatch factory is what
  keeps the measured config and the shipped config the same thing.
- **fp16 D128 is now measured against a stronger baseline than before**, which is why
  its low end reads lower than an older number would suggest. `attention_tiled_2d`'s
  shipped `wide4` config used to overflow the 64 KB LDS and fail codegen at llvm22,
  forcing the comparison onto a fallback vehicle; it builds now, so the baseline got
  faster while dense did not move. The ratio fell; nothing regressed.

D64 remains the VGPR-bound regime described under *Problem category* — `waves_per_eu`
reaches 2 WG/CU for bf16-D64 but not fp16-D64 — so a `waves_per_eu` re-tune stays a
named follow-up, now as upside rather than as a fix for a deficit.

## Scope

gfx942 only · forward-inference prefill · dense causal (no paging / bias / SWA / sinks)
· bf16 + fp16 · head dims **D64 & D128** · MHA + GQA incl. non-power-of-2 (40/8, 28/4)
· default **and** persistent grids. D256 is out of scope — it is served by its own
wide-atom candidates.

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
and per-lever deltas are recorded outside the repo: see the optimization plan for this
port and the protected results page.

## Lever record (gfx950 dense → gfx942, plus gfx942-only experiments)

Every lever evaluated for this port, with its verdict. Impact is qualitative;
magnitudes live outside the repo per `AGENTS.md`.

### Adopted

| Lever | Config | Mechanism | Verdict |
|---|---|---|---|
| 32×32×8 atom + K-loop doubling | all | CDNA3 has no 32×32×16 fp16/bf16 atom; C-layout is identical so softmax/epilogue port unchanged | **shipped** — enablement |
| Conflict-free V (`perm_b32` store transpose) | **D128 fp16** | V stored transposed → PV A-operand read is one contiguous `ds_read_b64` instead of 4 element-wise `ds_read_u16` | **shipped** — large; identity-preserving |
| `exp2_fast` | all | softmax args are provably ≤ 0, so `llvm.exp2`'s guarded range reduction is dead work; bf16 D128 (the last holdout) re-enabled after re-measurement — 0 scratch and lower VGPR on both grids, numerically identical | **shipped** — dominant on the VALU-bound path |
| Fused softmax rescale | all | exp2 → accumulate → cast → pack in one pass instead of materializing a full f32 `p_vals` matrix | **shipped** — pure live-range relief, bit-identical |
| Per-config `waves_per_eu` | **bf16 D64** → 4 | forces the allocator low enough that a second workgroup co-resides (1 → 2 WG/CU) | **shipped** — large at long sequences |
| D64 K-bank-conflict pad | **D64 both dtypes** | 2-row-group boundary pad takes the `do_qk` K reads from 32-way to 4-way | **shipped** — large, cross-part confirmed |
| Persistent grid-stride | all | `num_persistent` CTAs grid-stride over decoded work items; qb-major and hkv-major decodes | **shipped** — large at long sequences; auto-on when work fills the grid |

### Evaluated and rejected

| Lever | Why it lost |
|---|---|
| Conflict-free V at **D64** | D64 is VGPR-bound; the register round-trip costs more than the LDS-instruction saving |
| Conflict-free V at **bf16 D128** | spills over the `waves_per_eu=2` cap on the `.1k` MFMA schedule |
| `exp2_fast` at **bf16 D128** (former holdout) | initially rejected on a measured spill over the `waves_per_eu=2` cap; **revisited and adopted** (see Adopted table) — re-measurement reads 0 scratch and lower VGPR on both grids, so the original reading no longer reproduces |
| `waves_per_eu=3` at **fp16 D64** | reaches 2 WG/CU but loses more ILP than the second workgroup buys |
| Drop K/V LDS pads to reach 2 WG/CU at D128 | D128 stays 1 WG/CU even unpadded (register floor co-limits), so it only reintroduces bank conflicts — **catastrophic** |
| `block_n=32` (all configs) | halving the KV tile doubles the tile/grid count; the extra loop and barrier overhead outweighs the LDS relief. Marginally positive on **bf16 D128** but **part-dependent**, so not wired — it would need a CU-count-aware policy |
| `iglp_opt` (`Gfx942DenseTuning.iglp`) | resource- and performance-neutral cross-part: the canned GEMM interleave does not match this loop, which is barrier-rendezvous-bound. Kept as a default-off knob |
| Smaller `BLOCK_M` | a *fully filled* grid at `BLOCK_M=64` measured **slower** than a one-third-filled one at 256 — 2 waves/CTA leaves half the CU's matrix cores unreachable at 1 CTA/CU. `BLOCK_M=128` is a small win on two configs only; not promoted |
| cfvst load/store chunking | does not bound register pressure: the later chunks' loads carry no dependency on the earlier ones, so LLVM hoists them above the intervening full `s_waitcnt(vmcnt=0)`, which then covers them anyway |
| PV-only `s_setprio` | **proven-negative** on gfx942 `attention_tiled_2d`. Note this is one lever in one placement — it is *not* a verdict on the scheduling-intrinsic family, and hand-written `sched_group_barrier` remains open |
| Diagonal two-phase causal peel | **proven-negative** on gfx942 `attention_tiled_2d`; gated on the bound shifting to compute, which it has not |
| partial-vmcnt software prefetch | N/A — it is a double-buffering lever and this kernel is single-buffered (NBUF=1), so there is no prefetch to partially overlap. The older "NBUF=2 does not fit 64 KB LDS at D128" reason is **retired**: it only holds at the shipped `block_n=64` (a K-only double buffer, or `block_n=32`, does fit), so what actually closes the door is the measured `block_n=32` verdict in the row above, not the LDS arithmetic |

## Bench

```
python attention_dense_prefill.py                        # parity + bench, default shapes
python attention_dense_prefill.py --dtype fp16 --d 64
python attention_dense_prefill.py --persistent --np 304  # persistent grid
```

Full-cohort parity and perf are driven by the live harness at
`benchmarks/gfx942/attention/prefill/benchmark_dense_prefill_live.py` (`--mode all`),
which is the numeric gate for this kernel — the same role the bench plays on gfx950.

`_occupancy_probe.py` is the static resource guard (VGPR / AGPR / spill / LDS, comgr only,
no GPU): re-run it after any lever change to confirm 0 spill and to see which resource
is the occupancy limiter.

Per AGENTS.md, **absolute measured throughput lives only in the protected results page,
never in the repo.** The relative gains under *Status* are ratios, which is the form that
can be stated here; every TFLOP/s, counter and per-lever magnitude behind them is on the
protected page.
