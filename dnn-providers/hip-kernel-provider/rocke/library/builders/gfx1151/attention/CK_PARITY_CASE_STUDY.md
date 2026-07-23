# CK-parity replay on gfx1151 — Lever 1-3 results

Companion results doc for [`plan.md`](plan.md) (the ASM/scheduler evidence and
replay plan for closing the CK-Tile FMHA gap: rocke ~9-11 TF vs CK ~17.5 TF on
gfx1151 D128, B4 Sq512 Sk512 H8, fp16, non-causal).

`plan.md` is the *why*; this file is the *what-was-done* + the *measured result*.
It records the code changes, the exact reproduce commands, and the result tables
to fill in on a gfx1151 board (the plan's node `ctr-halo-b47-02`, Radeon 8060S,
RDNA3.5, wave32, WMMA `wmma_f32_16x16x16_f16`).

> **Status:** code landed; **on-board numbers pending** — every "TBD" cell needs a
> gfx1151 device (not available in the authoring environment). Fill each row from
> the printed `tune`/`mw_tune` line and keep the correctness gate (`max_abs ≤ 2e-2`)
> on every run.

---

## What changed (code)

The plan's conclusion is that the ~40% gap is **structural + register-allocation,
not scheduling** — and that the load-bearing missing knob was **Lever 1**: an
explicit `waves_per_eu` occupancy target (CK's `__launch_bounds__(128, 2)`
equivalent) that lets the RDNA3 backend spend the top of the wave32 VGPR file
(toward 256) instead of spilling at its default ~192-VGPR heuristic ceiling.

All changes stay inside the `builders/gfx1151/attention/` vehicles (Python-only,
**not** byte-identity gated per `plan.md` §3), so no `core/` or `helpers/`
lowering changed and no C++ mirror / gate re-bless is required. The lowering
already consumed `attrs["waves_per_eu"]` (`platform/.../lower_llvm.py:4415`).

| lever | change | file(s) |
|---|---|---|
| **Lever 1** | `waves_per_eu: Optional[int]` on `SingleWaveCfg`; threaded into `b.kernel.attrs["waves_per_eu"]`; encoded in `kernel_name()` | `fmha_singlewave.py` |
| **Lever 1** | `waves_per_eu: Optional[int]` on `MultiWaveCfg`; same threading + name encoding | `fmha_multiwave.py` |
| **Lever 1** | `--wpe` sweep axis (`0` = backend default, `>0` = pinned target); `vpe=` in the result formatter | `tune.py`, `mw_tune.py` |
| **Lever 2** | *already implemented* — `fmha_multiwave.py` `kv_source="lds"` fans 4 waves over one 64×128 tile (16 M-rows/wave, full-N `n_dk` NIter blocking, cooperative K/V LDS staging). Lever 1 is what unblocks it from spilling. | `fmha_multiwave.py` |

`waves_per_eu=None` emits **no** occupancy attribute (backend heuristic preserved,
byte-identical to pre-change output); a set value emits
`"amdgpu-waves-per-eu"="N,N"`. Verified on the CPU lowering path.

---

## Reproduce

```bash
cd dnn-providers/hip-kernel-provider/rocke
export PYTHONPATH="platform/python:library"
```

### Lever 1 — single-wave, unpin VGPR (highest value / lowest effort)

```bash
# D128 (spilling regime): sweep the occupancy target against the default.
python3 -m builders.gfx1151.attention.tune \
    --head-size 128 --seqlen-q 512 --seqlen-k 512 --heads 8 --batch 4 \
    --bm 1 --fusek 0 1 --wpe 0 2 3 4
```

**Checkpoint (plan §Lever 1):** as `--wpe` drops the occupancy target, `vgpr`
should rise toward 256 and `spill` drop toward 0; record TF.

| shape | cfg | wpe | vgpr | spill | wmma | TF | note |
|---|---|---:|---:|---:|---:|---:|---|
| B4 S512 D128 | singlewave fuse_k | def | 192 | 29 | 16 | ~9.3 | baseline (from plan §1) |
| B4 S512 D128 | singlewave fuse_k | 2 | TBD | TBD | 16 | TBD | expect spill↓ |
| B4 S512 D128 | singlewave fuse_k | 3 | TBD | TBD | 16 | TBD | |
| B4 S512 D128 | singlewave fuse_k | 4 | TBD | TBD | 16 | TBD | |

### Lever 2 — 4-warp cooperative tile + Lever 1 (the structural core)

```bash
# CK's shape: 4 waves, shared-LDS K/V, launch_bounds(128,2)-equivalent.
python3 -m builders.gfx1151.attention.mw_tune \
    --head-size 128 --seqlen-q 512 --seqlen-k 512 --heads 8 --batch 4 \
    --waves 4 --kv-source lds --wpe 0 2 3 4
```

**Checkpoint (plan §Lever 2):** verify `wmma≈64`, `spill≈0`, `vgpr≈256`; target
**≥14 TF** non-causal. (The `w4 lds` config previously measured ~6.3 TF in the
README campaign — but that was at the default 192-VGPR pin *with spills*; the bet
in `plan.md` is that unpinning via `--wpe` is what makes the 64-WMMA density pay.)

| shape | waves | kv | wpe | vgpr | spill | wmma | TF | note |
|---|---:|---|---:|---:|---:|---:|---:|---|
| B4 S512 D128 | 4 | lds | def | TBD | TBD | ~64 | ~6.3 | README baseline (spilling) |
| B4 S512 D128 | 4 | lds | 2 | TBD | TBD | ~64 | TBD | target ≥14 TF |
| B4 S512 D128 | 4 | lds | 3 | TBD | TBD | ~64 | TBD | |
| B4 S512 D128 | 4 | lds | 4 | TBD | TBD | ~64 | TBD | |

### Lever 3 — LDS-vs-gather A/B under the new register budget

```bash
# Re-run the staging A/B at the winning wpe from Lever 2 (replace 2 as needed).
python3 -m builders.gfx1151.attention.mw_tune \
    --head-size 128 --seqlen-q 512 --seqlen-k 512 --heads 8 --batch 4 \
    --waves 4 --kv-source lds gather --wpe 2
```

**Checkpoint (plan §Lever 3):** direct A/B in one thermal window; keep whichever
verifies faster per shape. The README found gather beat LDS at the *spilling*
192-VGPR regime; the plan's hypothesis is the calculus may invert once Levers 1-2
remove the spill and 4 waves hide the barrier.

| shape | kv | wpe | TF | winner |
|---|---|---:|---:|---|
| B4 S512 D128 | lds | 2 | TBD | TBD |
| B4 S512 D128 | gather | 2 | TBD | TBD |

### Additional target shapes (plan §4)

Run the winning config across the plan's target shapes; gate `max_abs ≤ 2e-2`:

| shape | CK target | rocke best (this campaign) |
|---|---:|---:|
| D128 S512 | ~17.5 TF | TBD |
| D128 S1024 | ~16.6 TF | TBD |
| D64 S512 | ~16.5 TF | TBD |
| D128 causal | ~12.5 TF | TBD |

---

## Levers not needed / deferred

- **Lever 4 (explicit `sched_group_barrier`/`s_setprio` interleave)** — the plan
  marks this a *fallback*, expected unnecessary: CK emits no such hints on this
  path and the useful `s_delay_alu` interleave is backend-auto. Only pursue if the
  post-Lever-1/2 ISA shows poor WMMA/VALU/DS overlap vs CK. The IR builder already
  exposes `b.sched_group_barrier(...)` / `b.s_setprio(...)` if needed.
- **Lever 5 (fixed per-iteration costs)** — causal early-exit and `exp2` softmax
  are already in the kernels; `PermuteWarpGemmCToA` was investigated and rejected
  in the README campaign (`p_xpose="shuffle"` is a structural dead-end on the
  fixed WMMA a_map).
- **Optional `amdgpu-num-vgpr` hard cap** — deliberately **not** added: the plan
  marks it optional, CK doesn't use it, and it would require editing shared
  `core/` lowering + the C++ engine mirror + a byte-identity re-bless, violating
  the "keep changes in the vehicles" constraint.

---

## ASM-parity loop (measured, static — no gfx1151 device needed)

Driven by [`ck_parity_probe.py`](ck_parity_probe.py): compile each candidate to a
**gfx1151 HSACO via comgr** (codegen is independent of the running GPU, so this
works on the gfx950 box here), then decode the msgpack resource note + isolate the
rolled K-loop body from `llvm-objdump` to get **per-iteration** WMMA / vmem / LDS /
sync counts. Reproduce:

```bash
PYTHONPATH=platform/python:library ROCKE_LLVM_FLAVOR=llvm22 \
    python3 -m builders.gfx1151.attention.ck_parity_probe --head-size 128
```

**CK reference (plan.md appendix):** `vgpr=256  spill≈1  wmma/it=64  s_barrier≈12
lds=9216  s_setprio=0  sched=0`.

Measured convergence (D128, B4 H8; `vpe` = `waves_per_eu`):

| config | vgpr | spill | lds (B) | wmma/it | s_barrier/it | s_waitcnt/it | setprio | sched | targets hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `sw fuse_k` (baseline) | 190 | 8 | 512 | 16 | 1 | 83 | 0 | 0 | — |
| `mw w4 lds` (baseline) | 201 | 0 | 10240 | 16 | 2 | 81 | 0 | 0 | lds |
| **`rb w4 m1 n64 vpe2`** | **243** | **0** | 34816 | **64** | 2 | 347 | 0 | 0 | **vgpr+spill+wmma** |
| `rb w4 m1 n16 vpe2` | 249 | 0 | **10240** | 16 | 2 | 74 | 0 | 0 | vgpr+spill+lds |
| `rb w4 m4 n16 vpe2` | 256 | 844 | **10240** | **64** | 2 | 413 | 0 | 0 | vgpr+wmma+lds |

### Per-target verdict

- **(a) VGPR pressure — MET.** Lever 1 (`waves_per_eu`) does exactly what the plan
  predicted: it lifts the baseline **190 VGPR / 8-spill** (or the pinned 192/29 in
  the plan's run) to **243-256 VGPR** and, on the non-`m_repeat` configs, drives
  **spill → 0**. `rb w4 m1 n64 vpe2` = **vgpr 243, spill 0** vs CK's 256/≈1.
- **(b) WMMA utilization — MET (static density).** The register-blocked kernel
  reaches **64 WMMA / K-loop iteration**, matching CK's 64. (True *issue-rate*
  utilization needs rocprof on a gfx1151 board — see below.)
- **(c) LDS/VMEM — counts mapped, latencies hardware-gated.** LDS *bytes* hit
  **10240 ≈ CK's 9216** on the small-tile configs; VMEM is load-only in the loop
  (no in-loop global stores, as expected). Actual **load/store latencies are not
  measurable here** (no gfx1151 device for rocprof).
- **(d) sync primitives / iteration — MET (same kinds; we are leaner).** Same sync
  *kinds* as CK — `s_barrier` + `s_waitcnt`, and critically **`s_setprio=0`,
  `sched=0`** exactly matching CK's "no scheduler hints on this path" finding
  (plan §2). We issue **2 `s_barrier`/iteration** vs CK's ~12, i.e. *fewer* syncs,
  not more.

### Where the loop converged: a structural wall (as plan.md predicted)

No single config in the current kernel family hits **all** of
`{vgpr≈256, spill≈1, wmma≈64, lds≈9216}` at once — you get any **three**, and the
fourth breaks, because density is bought two mutually-exclusive ways:

- **density via `block_n`** (`m1 n64`): wmma 64, spill 0, but **LDS 34 KB** (stages
  64 keys × head_size for *both* K and transposed V);
- **density via `m_repeat`** (`m4 n16`): wmma 64, LDS 10 KB, but **spill 800+** (the
  per-wave accumulator = `m_repeat · n_dk · <8×f32>` overflows the 256-VGPR file).

CK escapes this because its 4-warp tile uses **`MWarp=4` / `MIter=1`** (the 64 M
rows are split *across warps*, each holding only ~96 VGPR of accumulator) and
stages a single ~9 KB K/V working set. Reproducing that needs a kernel rewrite
with **single-operand LDS staging + a co-designed accumulator distribution** — the
same structural change the README campaign found *loses on wall-clock* on this
issue-bound APU, and beyond what the `waves_per_eu` / `m_repeat` / `block_n`
tuning levers can express. The loop therefore converges to the plan's own
conclusion, now quantified per-metric: **the gap is structural + register-
distribution, not scheduling** (confirmed: `setprio=0 sched=0` on both sides).

### Not closable in this environment

- **rocprof latencies / occupancy on gfx1151** — this box is **gfx950 (MI355X)**;
  a gfx1151 HSACO won't run on it, and profiling the gfx950 device would measure a
  different kernel on different silicon. The load/store-latency half of target (c)
  and the true issue-rate half of (b) require the plan's `ctr-halo-b47-02` board.

## On-hardware perf results (gfx1151, MEASURED) — the loop, and its verdict

Ran on a **live gfx1151 board** acquired via the SLURM login node `ckdsl-login`,
co-scheduled into a GPU holder allocation on **`ctr-halo-b47-02`** (Ryzen AI Max
395, RDNA3.5, **ROCm 7.0.2 / llvm20** — the same vintage as the README campaign).
Every config correctness-gated (`max_abs 3.05e-05 ≤ 2e-2`) before timing;
HIP-event timed, torch-free (numpy reference + ctypes HIP).

**All numbers D128, B4 Sq512 Sk512 H8, non-causal. CK target = ~17.5 TF.**

| kernel / lever | best TF | vgpr | spill | vs CK |
|---|---:|---:|---:|---:|
| **single-wave `fuse_k`** (baseline) | **10.6** | 192 | 13 | **0.61×** |
| multiwave `w4 gather` | 9.16 | 213 | 0 | 0.52× |
| pipelined `sched=1` | 9.04 | 192 | 26 | 0.52× |
| multiwave `w4 lds` | 4.95 | 220 | 0 | 0.28× |
| regblocked `w4 m1 n32` | 4.52 | 235 | 0 | 0.26× |
| regblocked `w4 m1 n64` | 1.67–3.10 | 256 | 39–71 | 0.10–0.18× |

### Per-lever verdict (measured, not predicted)

- **Lever 1 (`waves_per_eu`) — NET NEGATIVE on hardware.** It does *not* reproduce
  the plan's "unpin 192→256 ⇒ speedup" hypothesis. On single-wave it is inert
  (VGPR stays 192). On multiwave it *lowers* occupancy (220→213 VGPR) and **drops
  TF** (gather 9.16→8.1, lds 4.95→4.14). Only in the catastrophic-spill
  `regblocked n64` case does it help relatively (1.67→3.10 TF by cutting spill
  71→39) — and that config is still 3× slower than single-wave. **The plan's
  headline lever is a regression here.**
- **Lever 2 (4-warp density) — REGRESSES.** Both the multiwave and register-blocked
  kernels lose decisively to single-wave (4.5–9.2 vs 10.6 TF), exactly as the
  README campaign found: on this cache-resident, issue-bound APU the extra
  instructions/barriers outweigh the WMMA density.
- **Lever 3 (lds vs gather) — gather wins** by ~1.85× (9.16 vs 4.95 TF).
  Cooperative LDS staging is a net loss; the cache already feeds the gather.
- **Lever 4 (sched hints) — noise-level** (±3%: D128 8.75→9.04, D64 7.97→7.76),
  as the plan predicted (no CK-authored hints on this path).
- **Lever 5 (causal early-exit)** — already in-tree; not the non-causal bottleneck.

### Verdict: NO perf parity, and not reachable with these levers

**Best rocke = 10.6 TF (single-wave) = 0.61× CK's 17.5 TF.** The loop did not
converge to parity — it *diverged*: every structural lever the plan proposed to
close the gap makes it **worse** on real gfx1151 silicon. This is the same
conclusion the README campaign reached, now re-confirmed on-hardware with the
`waves_per_eu` lever added: **the gap is structural (CK's co-designed 4-warp
`MIter=1` distribution + single-operand staging) and ultimately needs gfx12
intrinsics (larger-K WMMA, `ds_read_tr`, packed-f16 C)** — it is not closable by
occupancy hints, density blocking, LDS staging, or scheduler hints. Continuing to
loop these levers only produces more sub-11 TF points.

Reproduce (inside a gfx1151 GPU allocation):

```bash
export PYTHONPATH=$HOME/ckdsl_rocke/python:$HOME/ckdsl_rocke/library
export ROCKE_LLVM_FLAVOR=llvm20    # ROCm 7.0.2 node
python3 -m builders.gfx1151.attention.tune     --head-size 128 --bm 1 --fusek 1 --wpe 0 2 3 4
python3 -m builders.gfx1151.attention.mw_tune   --head-size 128 --waves 4 --kv-source lds gather --wpe 0 2 3 4
python3 -m builders.gfx1151.attention.prod_tune --head-size 128 --num-warps 4 --m-repeat 1 --block-n 32 64 --wpe 0 2 3
python3 -m builders.gfx1151.attention.sp_tune   --head-size 128 --sched 0 1
```

## Deep rocprof microarch probe (gfx1151, MEASURED — rocprofv3, ROCm 7.13)

Ran `rocprofv3` PMC counters on the single-wave D128 unified-attention kernel on a
real **gfx1151** board (`ctr-halo-b48-02`, Radeon 8060S / Strix Halo, 40 CU,
wave32, 2 SIMD/CU, 64 KB LDS/CU, ~2.9 GHz), via a minimal fixed-shape launch
runner (`rocprof_probe_runner.py`, 50 dispatches/pass, counters attributed to the
one attention dispatch). Grid = 1024 workgroups × 1 wave32; VGPR=192, SGPR=128,
LDS=512 B/wg, scratch=52 B.

Per-dispatch counter means (kernel rows, cold dispatch dropped):

| counter | value | per-wave |
|---|---:|---:|
| SQ_WAVES | 1024 | — |
| SQ_INSTS_VALU | 12,042,240 | **11,760** |
| SQ_INSTS_LDS | 2,424,832 | 2,368 |
| SQ_INSTS_SALU | 91,136 | 89 |
| LDSBankConflict | **9.09 %** | — |
| SQC_LDS_BANK_CONFLICT / IDX_ACTIVE | 262,144 / 2,883,584 | =9.09 % |
| MemUnitBusy | 47.9 % | — |
| WriteUnitStalled | 0.37 % | — |
| GPUBusy | 100 % | — |
| MeanOccupancyPerCU (rocprof raw) | 115.6 | — |

### The four asks, answered with real counters

- **(a) WMMA/matrix utilization — ~3.6 % of issued instructions.** RDNA3 has no
  WMMA-busy counter (WMMA counts within VALU), so combine counters with static
  ISA: WMMA/wave = 16 WMMA/K-tile × 32 K-tiles = **512 WMMA/wave** vs
  **14,217 issued/wave** (VALU 11,760 + LDS 2,368 + SALU 89). ⇒ the matrix engine
  is **~3.6 % of the instruction stream** (~1 WMMA per 28 issued ops). This is the
  hard, measured confirmation of the "issue-bound, not matrix-bound" thesis — you
  cannot approach the WMMA FLOP roofline when 96 % of issued instructions are not
  WMMA. The kernel is **VALU-issue-bound** (VALU:LDS:SALU = 11760 : 2368 : 89).
- **(b) Bubbles/stalls.** No VALU-issue-stall counter is exposed on this
  rocprofv3, but the memory side is clearly *not* the stall source:
  **WriteUnitStalled 0.37 %** (stores essentially never stall — few global
  stores) and **MemUnitBusy 47.9 %** (the vector-mem pipe is idle ~half the time).
  With **GPUBusy 100 %** (no host gaps) and the 96 %-non-WMMA issue mix, the
  bottleneck is **front-end/VALU instruction issue**, not memory or matrix — the
  "bubbles" are the matrix unit waiting behind the VALU issue stream, exactly the
  issue-bound wall.
- **(c) LDS bank conflicts — 9.1 %.** `LDSBankConflict = 9.09 %`
  (`262,144 / 2,883,584` conflict/idx-active cycles). The P-transpose LDS
  round-trip (2,368 LDS ops/wave) pays a **moderate ~9 % bank-conflict tax** — a
  real but secondary cost, not the dominant bottleneck.
- **(d) Wave occupancy.** 1 wave32 per workgroup, **192 VGPR/wave** (the occupancy
  limiter — LDS 512 B is negligible), 1024 waves scheduled over 40 CU. The kernel
  fills the machine (GPUBusy 100 %, MeanOccupancyPerCU rocprof-raw 115.6) but each
  CU runs relatively few concurrent waves due to the 192-VGPR footprint, so there
  is limited latency hiding for the VALU-issue stalls above.

### Microarch verdict

Real gfx1151 counters confirm the static/perf conclusions with hardware evidence:
the single-wave attention kernel is **instruction-issue-bound (VALU), with the
WMMA unit at ~3.6 % of issued ops**, memory ~half-idle and never store-stalled,
and LDS carrying a moderate ~9 % bank-conflict tax from the P-transpose. Raising
throughput requires raising the WMMA:issue ratio (fewer VALU/LDS ops per WMMA) —
which needs gfx12 larger-K WMMA / `ds_read_tr` (removes the P-transpose LDS
traffic) — not occupancy or scheduling levers. Probe reproducible via
`rocprof_probe_runner.py` + `rocprofv3 --pmc` (see `probe_out151/`).

## WMMA-utilization optimization campaign (gfx1151, MEASURED) - attempt to port attention_dense levers

Goal: raise WMMA utilization / TFLOPS of the single-wave WMMA MHA kernel using the
gfx950 `attention_dense.py` levers (wide-LDS staging, occupancy, lazy-O, persistent)
as reference. Board: holder `67538475` on `ctr-halo-b48-02` (ROCm 7.13/llvm22).
Baseline on this node = **12.6-12.8 TF** (D128 MHA B4 S512, non-causal; note the
newer node/ROCm is faster than the 10.6 TF earlier b47-02/ROCm 7.0.2 baseline).

| lever (from attention_dense) | gfx1151 result | verdict |
|---|---|---|
| **Lazy-O** (lazy rescale, skip O-rescale on wave_all max-stable vote) | 12.57 -> 12.65 TF, bit-identical correctness | **NEUTRAL** - the O-rescale is ~2% of the 11,760 VALU/wave, so eliding it can't move an issue-bound kernel. Kept as a default-off flag. |
| **Wide-LDS V staging** (`v_mode=lds_t`, transposed-V + ds_read_b128) | 12.83 -> 10.42 TF (-19%) | **REGRESSION** - gfx1151 has NO async-LDS DMA, so staging transposed-V needs an explicit scatter + barrier whose cost exceeds the cache-resident gather. The gather wins. |
| **CK-1 P-transpose elimination** (feed P to PV WMMA in native QK layout) | not attempted (structurally blocked) | **NOT EXPRESSIBLE** - QK output has P as `lane=K, slot=Qrow`; the PV WMMA A-operand needs `lane=Qrow, slot=K`. Converting needs each lane to gather P from 16 different lanes; the fixed wave32 WMMA `a_map` can only do that via LDS/`ds_bpermute` (same engine, more instructions - the prior `p_xpose=shuffle` experiment already measured this as a regression). The P-transpose LDS round-trip is irreducible on wave32. |
| **P-transpose barrier -> intra-wave waitcnt** (single-wave `b.sync()` -> `s_waitcnt(lgkmcnt=0)`; removes all `s_barrier`) | 12.67 TF, 0 s_barrier in ISA | **NEUTRAL** (kept: cleaner, no regression) - a 1-wave workgroup barrier is already ~free on hardware. |
| **Occupancy** (`waves_per_eu`) | inert on single-wave (VGPR stays 192); net-negative on multi-wave (earlier) | not a win here |
| **Persistent** (grid-stride) | not implemented (large refactor; amortizes dispatch, does NOT change per-CTA WMMA util) | deferred - wall-clock-only lever |

### Verdict: WMMA utilization is structurally capped on gfx1151

WMMA utilization (~3.6% of issued instructions) is bounded by the kernel's inherent
**VALU:WMMA ratio**, dominated by costs that these levers cannot remove: the softmax
cross-lane reductions (`wave_reduce_max/sum` butterflies ~64 ds-ops/tile - most of
the 2,368 LDS/wave), the `exp2` per P element, the mask/scale VALU, and the strided
V column-gather. None of the four attention_dense techniques raises it on wave32:
LDS staging regresses (no async DMA to hide the scatter), the P-transpose is
irreducible (fixed wave32 `a_map`), lazy-O and the barrier fix are neutral (they
target <2% of the issue stream), and occupancy is inert/negative. The gfx950
reference wins because CDNA has async-LDS DMA (overlap staging), a larger 32x32x16
MFMA atom (higher WMMA:issue ratio), and packed-f16 accumulators - none of which
exist on gfx1151. Raising WMMA utilization here requires **gfx12** (larger-K WMMA,
`ds_read_tr` transpose-loads to kill the P-transpose, async-LDS), not the
attention_dense levers. Net TFLOPS change from the campaign: ~0 (best kept config
= gather + eager + intra-wave-waitcnt = 12.6-12.8 TF, unchanged within noise).

## Pipelined kernel: register-trim + tile-size + Sq/Skv sweeps (gfx1151, MEASURED)

Software-pipelined single-wave kernel (`fmha_pipelined`, VGPR-prefetch of next-tile
QK) + a `waves_per_eu` register-trim knob, swept on the held board (b48-02).

**Register-trim / sched (D128 Sq=Sk=512, B2):** `sched=1` (sched_group_barrier
interleave) cuts spill 41 -> 16 and lifts 11.0 -> 12.7 TF; `waves_per_eu` is
inert (spill is capacity-bound, not heuristic-bound). Pipelined best = **12.7 TF
= on par with single-wave**, not beyond it - the issue-bound wall is not broken by
pipelining.

**Tile-size sweep (BLOCK_M = 16*bm, D128 Sq=Sk=512):**

| tile (bm) | Q rows | TF | spill | note |
|---:|---:|---:|---:|---|
| **1** | 16 | **13.08** | 12 | optimal (single wave32) |
| 2 | 32 | 5.3 | 248 | O-accumulator (bm*64 VGPR) overflows the file |
| 4 | 64 | 2.4 | 1188 | catastrophic spill |

The 16-row tile is register-optimal; bigger tiles need the accumulator distributed
across more waves (the CK 4-warp design), which regresses on gfx1151 for the
LDS-staging reason.

**Sq x Skv sweep 64..2048 (D128, B2, pipelined sched=1), TFLOPS:**

| Sq \ Skv | 64 | 128 | 256 | 512 | 1024 | 2048 |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 2.4 | 2.8 | 2.7 | 2.7 | 2.8 | 2.8 |
| 128 | 4.5 | 4.8 | 5.1 | 5.1 | 5.2 | 5.3 |
| 256 | 6.9 | 8.2 | 8.2 | 8.4 | 8.5 | 8.4 |
| 512 | 9.0 | 11.4 | 11.9 | 12.0 | 12.2 | 12.5 |
| 1024 | 7.4 | 9.9 | 11.3 | 12.3 | 12.7 | 12.6 |
| 2048 | 7.2 | 9.4 | 11.1 | 12.5 | **13.0** | 12.9 |

All spill=16, all correct (max_abs <= 1.2e-4). TFLOPS **plateaus ~13 TF (~22% of
the 59 TF roofline)** once both Sq,Skv >= 512; small shapes are launch/occupancy
bound (few CTAs), and skinny-Skv (Sk=64) drops to ~7 TF (short K-loop -> the
pipeline-fill/prologue dominates). The **~13 TF ceiling is shape-independent** for
large-enough shapes - confirming an issue-bound wall (WMMA:issue ratio capped by
the K=16 atom), not a tile/shape/occupancy problem.

### Combined verdict across all attempts
Every lever - lazy-O, wide-LDS, occupancy, pipelining, larger tiles, and the full
Sq/Skv space - lands at the same **~13 TF (~22% of the 59 TF f16-WMMA roofline)**.
The wall is the WMMA:issue ratio on the gfx1151 `16x16x16` atom (confirmed by
llvm-mc: no larger-K WMMA exists in the gfx11 ISA), which is inherent to attention
on this part. Raising it requires gfx1250-class hardware (K=32/64 WMMA, `ds_read_tr`,
async-LDS).

### Fine-grained synchronization levers (dpp_reduce + v_prefetch)

Two ASM-scan-motivated sync optimizations, each behind a `SingleWaveCfg` flag,
A/B'd on the held gfx1151 board (`ctr-halo-b48-02`, B4 Sq512 Sk512 D128 Hq8,
bm1 fusek1):

* **`dpp_reduce`** — softmax cross-lane max/sum butterfly via DPP (`v_max_f32_dpp`
  / `v_add_f32_dpp` `row_xmask`) in the VALU instead of `warp_shuffle_xor`
  (`ds_swizzle` on the LDS engine). Targets the per-stage `s_waitcnt(lgkmcnt(0))`
  drains the ASM scan flagged.
* **`v_prefetch`** — software-pipeline the PV `d`-loop so `d+1`'s 16-wide V gather
  is issued before `d`'s WMMA, letting the loads fly during the matmul (partial
  `vmcnt` instead of a full drain per column).

| config | TF | spill | s_waitcnt | ds_swizzle |
|---|---:|---:|---:|---:|
| baseline (shfl + plain gather) | **12.85** | 12 | 84 | 64 |
| v_prefetch only | 12.68 | 20 | — | — |
| dpp_reduce only | 12.18 | 19 | **24** | **0** |
| both | 12.01 | 20 | — | — |

All four **numerically identical** (max_abs 3.05e-05). The disassembly confirms
`dpp_reduce` did *exactly* what was intended: **all 64 `ds_swizzle` butterfly ops
eliminated** (→ 32 `v_max_f32_dpp` + 32 `v_add_f32_dpp`) and **`s_waitcnt`
collapsed 84 → 24** (−60 waits, the softmax `lgkmcnt(0)` drains gone). Yet perf
**regressed** — because removing the sync exposed no stall: the freed cycles were
never the gate. Instead both levers *add live VGPRs* (fused DPP result regs; the
prefetched `v_next` holds +8 VGPRs across the WMMA), pushing spill 12 → 19/20 on
the already-spilling single wave, and that extra spill traffic *is* on the
critical path. Net: −0.2 to −0.8 TF.

This is the **definitive confirmation** the ~13 TF wall is a register/VALU-issue
wall, not a synchronization wall: collapsing 60 waitcnts and all 64 LDS reduction
ops changed nothing but the register budget. Both flags kept default-off.

### Lever 1 revisited — vectorized-V (`v_mode="lds_c"`), measured dead-end

The ASM scan found the PV V-operand is fetched as **128 narrow strided global
loads/iter** (`global_load_d16_b16`), = **57% of all memory traffic** and the
source of most of the 348 SALU (buffer-address math). The confirmed gfx1151
transpose gap: `ds_read_b64_tr_b16` (the CDNA/gfx950 + gfx12 transpose-read that
would vectorize this) **errors "instruction not supported on this GPU"** on
gfx1151 (verified via llvm-mc) — so the V k×n transpose has no hardware support.

`lds_c` attempts it in software the *right* way (fixing why the older `lds_t`
regressed): 16 **wide** `b128` global loads + 16 **wide contiguous** LDS stores
(no scalar scatter), then a padded, bank-conflict-free strided LDS readback
(reads hoisted before the WMMAs to pipeline into one `lgkmcnt` drain).

Disasm confirms the memory-side transform fired perfectly:

| v_mode | TF | VALU | SALU | DS | MEM | narrow gather | spill |
|---|---:|---:|---:|---:|---:|---:|---:|
| gather (baseline) | **12.37** | 725 | 348 | 74 | 224 | **128** | 12 |
| lds_c | 7.9–8.0 | 695 | 394 | **218** | **112** | **0** | **0** |

`lds_c` **eliminates all 128 narrow global gathers** (MEM 224→112) and even
**kills the VGPR spill (12→0)** — yet regresses to ~8 TF. Root cause: without
`ds_read_tr` the transpose degenerates to **128 narrow `ds_read_b16`** on the
readback, and narrow strided *LDS* reads are *worse* than narrow *global* loads
(LDS port serializes them; global loads have far more memory-level parallelism).
Hoisting all reads before the WMMAs (single drain) didn't help (7.88→7.99 TF).

**Verdict:** three V-transpose variants (`lds_t` 10.4, `lds_c` 7.9, `lds_c`-hoisted
8.0) all lose to the direct `gather` (12.4). The transpose is irreducibly ~128
narrow ops on gfx1151, and the direct global gather is the local optimum. Kept
`lds_c` as a correct, default-off flag (its spill-elimination may compose with a
future register-hungry lever). **The path to 20 TF does not run through V
restructuring** — it needs the transpose-free levers (VOPD dual-issue of the
softmax VALU; `v_mov`/spill reduction; dense-path `cndmask`/`max` trim).

### Coordinate strength-reduction (`static_shape`) — address-SALU cut

The ISA scan attributed ~25% of the stream to SALU, almost all **address
arithmetic**: the kernel took the tensor strides as *runtime i32 params*, so
`TensorDescriptor.offset` emitted a runtime `mul(idx, stride)` for every address
term (`tensor_view.py:180`) that the compiler cannot strength-reduce or hoist.

`static_shape` bakes the strides as compile-time constants (packed BSHD:
`stride_head=D`, `stride_token=H*D`), flipping every term to a *constant*
multiply (→ shift/LEA) and unlocking LLVM LICM of the loop-invariant base. The
runtime stride params stay in the signature (launcher unchanged, DCE'd).

CPU disasm (llvm20 login-node build, D128 B?/bm1/fuse_k; counts are whole-kernel
so larger than the llvm22 loop-body numbers, but the A/B is valid):

| variant | VALU | SALU | int-mul | shift |
|---|---:|---:|---:|---:|
| dyn (runtime strides) | 1035 | 446 | **26** | 88 |
| **static_shape** | 1040 | **361** | **2** | 90 |
| static_shape + sseq512 | 1207 | 561 | 0 | 138 |

`static_shape` removes the runtime stride muls (**26→2**) and cuts **SALU −85
(−19%)** with VALU flat — exactly the address-arithmetic reduction targeted.
`static_seqlen` regresses (baking the loop bound triggers K-loop unrolling →
+SALU/+VALU), left default-off. **Board TFLOPS A/B pending** (the sole
accessible gfx1151 GPU was held by another user at implementation time).

### VOPD source-batching (`batch_softmax`) — ineffective (backend already schedules)

Attempted to feed the LLVM VOPD packer by emitting the online-softmax ops grouped
BY OPERATION across the 8 rows (all `fmax`, then all `mul`, then all `add`) so
independent same-class VALU sit adjacently. CPU disasm (llvm20 login build):

| variant | VALU | SALU | v_dual | vmul | vadd | vmax |
|---|---:|---:|---:|---:|---:|---:|
| base | 1042 | 450 | 63 | 57 | 52 | 27 |
| static_shape | 1034 | 365 | **70** | 74 | 39 | 22 |
| batch_softmax | 1039 | 418 | 66 | 59 | 49 | 28 |
| static + batch | 1051 | 365 | **53** | 79 | 55 | 31 |

On the **llvm20 login build** `batch_softmax` barely moved `v_dual` (63→66), so
it *looked* like a dead end (VOPD packing is post-RA, so source order can't feed
it). **That conclusion was WRONG on the board.** See the averaged board table
below: on llvm22 hardware `batch_softmax` alone is the **best single lever
(+4.1%)**. The gain is not VOPD packing — it is instruction **scheduling /
latency-hiding**: emitting the 8 rows' ops grouped by operation lets the post-RA
scheduler overlap the per-row `v_exp_f32` (TRANS pipe) and cross-lane reductions
across rows. Lesson: **llvm20 static `v_dual` counts do not predict llvm22 board
perf** — always confirm timing on the target.

### Averaged board sweep — static × dpp × batch_softmax (llvm22, b48-02, 5 trials)

D128 Sq=Sk=512 B4 bm1 fuse_k, mean of 5 runs each (baseline run-to-run ±0.15 TF):

| dpp | static | bsm | mean TF | Δ | spill |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 (baseline) | 12.54 | — | 12 |
| 0 | 0 | **1** | **13.05** | **+4.1%** | 12 |
| 0 | 1 | 0 | 12.63 | +0.8% | 9 |
| 0 | 1 | 1 | 12.76 | +1.8% | 9 |
| 1 | 0 | 0 | 11.91 | −5.0% | 19 |
| 1 | 1 | 0 | 12.38 | −1.3% | 9 |
| 1 | 0 | 1 | 12.47 | −0.6% | 19 |
| 1 | 1 | 1 | 12.78 | +1.9% | 9 |

**Findings (all correctness-gated, max_abs 3.05e-05):**
* **`batch_softmax` alone = 13.05 TF (+4.1%)** is the robust winner (every trial
  12.89–13.13, all above every baseline trial). New best single-wave config.
* **`static_shape`**: small robust win (+0.8%) and deterministically cuts spill
  **12→9** (SGPR 37→18). *But* stacking it on `batch_softmax` HURTS (12.76 <
  13.05) — static perturbs the scheduling the batching relies on.
* **DPP hypothesis (static absorbs DPP's spill → net win?): mechanism confirmed,
  verdict NO.** `static_shape` does erase DPP's spill (19→9) and recovers it
  11.91→12.38 — but 12.38 is still < the 12.54 baseline. DPP is
  neutral-to-negative in *every* pairing; its removed LDS/waitcnt ops are hidden
  at 100% GPUBusy, so eliminating them never buys wall-clock. Kept default-off.
* Net: the single-wave ceiling nudges from ~12.5 → **13.05 TF via
  `batch_softmax`**; the coordinate/DPP levers don't stack on top of it.

### `batch_softmax` shape sweep (llvm22, b48-02, B2, 3-trial mean)

Toggling only `batch_softmax` at the reference knobs (bm1/fusek1/wpe0/static0):

| shape | bsm0 | bsm1 | Δ |
|---|---:|---:|---:|
| D128 S128 | 5.11 | 5.42 | +6.1% |
| D128 S256 | 9.29 | 9.42 | +1.4% |
| D128 S512 | 12.58 | 13.12 | +4.3% |
| D128 S1024 | 13.19 | 13.91 | +5.5% |
| D128 S2048 | 13.88 | **14.15** | +1.9% |
| D128 S512 causal | 7.89 | 7.46 | **−5.5%** |
| D128 S512 GQA-2 | 13.39 | 13.98 | +4.4% |
| D64 S512 | 10.45 | 10.24 | **−2.0%** |

**`batch_softmax` is a robust win for DENSE `head_size=128`** across all sequence
lengths (+1.4%..+6.1%, larger for S≥512) and for GQA (+4.4%); S2048 reaches
**14.15 TF (~24% of the 59 TF roofline)**, a new single-wave high. **But it
REGRESSES for causal (−5.5%) and head_size=64 (−2.0%)** — the batched Pass-1
reorders the causal mask-select path unfavorably, and D64's smaller instruction
volume (n_dk=4) doesn't amortize the reordering. **Verdict: enable
`batch_softmax` only for dense D128 (+GQA); keep it off for causal / small head
dim.** It should be gated by `(mask_mode=="none" and head_size>=128)`, not
unconditional.

### Multi-wave + ping-pong validated on real gfx1151 (Windows Strix Halo board)

After all Linux gfx1151 boards became unusable (SLURM nodes drained; the standalone
mini + Austin gfx1150 have a broken ROCm 7.2.x `amdgpu` KMD that page-faults any GPU
dispatch), the levers were validated on a **standalone Windows 11 Strix Halo board**
via the cross-build → `--prebuilt` flow: HSACOs cross-built for **gfx11-generic** on
a Linux host (the board reports a garbled `gfx115E`; only the generic ELF loads),
tarball-deployed, launched through the Windows HIP runtime (`ROCKE_HIP_LIB=
amdhip64_7.dll`). Added `--emit`/`--prebuilt` to `tune.py`/`mw_tune.py` for this.
Correctness bit-matches Linux (`max_abs 3.05e-05`).

**Caveat:** the board exposes a small **~8-CU / 4-WGP virtualized slice** and
auto-clocks ±25-30%, so absolute TF (~1.6-2.7) is not comparable to the 40-CU chip
and only same-session interleaved A/B *ratios* are trustworthy. 15 interleaved
trials/config, medians:

| config | med TF | ratio |
|---|---:|---:|
| single-wave baseline | 1.615 | 1.00 |
| single-wave +bsm | 1.630 | 1.009 |
| single-wave +static | 1.590 | 0.985 |
| single-wave +dpp | 1.580 | 0.978 |
| **multi-wave w2 (no ping-pong)** | 2.110 | **1.31** |
| **multi-wave w2 + ping-pong** | **2.70-2.73** | **~1.68** |

**Findings:**
* **Multi-wave (2 waves/CTA) = +31%** over single-wave on this small partition.
* **Ping-pong (`s_setprio` WMMA/VALU staggering) = +28%** on top of multi-wave
  (2.11 -> 2.70), min/max bands cleanly separated -> robust, not clock noise. This
  is the intra-CTA WMMA/VALU overlap the ping-pong bookends were designed to force,
  and it lands big *because* the ~8-CU slice is small enough that a single-wave CTA
  underfills the WGP (the opposite regime from the 40-CU b48-02, where many
  single-wave CTAs already saturate and multi-wave was neutral/negative -- both
  results consistent once partition size is accounted for).
* **ILP and batch_softmax are neutral** in this multi-wave regime.
* Single-wave lever directions match Linux (dpp negative, bsm slightly positive)
  but magnitudes compress into the ±2% clock noise on this tiny slice.

Net: the multi-wave ping-pong design delivers **~1.68x over single-wave** on the
small gfx1151 partition; whether it helps the full 40-CU chip depends on partition
size and needs the b48-02-class board (a 7.13-KMD host) to confirm.

### FULL 40-CU chip result — ping-pong breaks the wall (+27%)

The Linux gfx1151 boards' "GPU page fault" turned out to be a **bad container image**
(the therock/nemanja 7.13 images fault on this host; torch/hipBLASLt/our kernel all
run correctly in `hyoon11/vllm-dev:...navi_upstream...` ROCm 7.2.1). Running in the
working image on the **full 40-CU Strix Halo mini** (Ryzen AI Max 395), 3-trial
medians, B4 Sq512 Sk512 D128 Hq8, all correct + spill=0:

| config | med TF | vs single-wave |
|---|---:|---:|
| single-wave (w1, baseline) | 13.45 | 1.00 |
| single-wave + ping-pong | 16.46 | **1.22** |
| w2 (no ping-pong) | 14.79 | 1.10 |
| w4 (no ping-pong) | 14.81 | 1.10 |
| **w2 + ping-pong + ilp2** | **17.07** | **1.27** |
| w4 + ping-pong | 16.79 | 1.25 |

**The `s_setprio` ping-pong lever is the big win: +22% on single-wave, +27% at the
best config (w2 ping-pong ilp2 = 17.07 TF ~= 29% of the 59 TF roofline)** — it
breaks clean past the ~13 TF single-wave issue-bound wall this whole study was stuck
at. Multi-wave alone adds ~10%; **ping-pong is the dominant factor and helps even a
single wave** (the setprio bookends win the SIMD dispatch arbitration for the WMMA
clusters against co-resident CTAs' waves). ILP is neutral. This confirms on real
40-CU silicon what the small Windows slice hinted at, and is the headline optimization
result of the campaign: **13 -> 17 TF via ping-pong scheduling.**

**batch_softmax STACKS on ping-pong** at full scale (w2, B4 S512, 3-trial medians):

| config | med TF |
|---|---:|
| w2 none bsm0 | 14.96 |
| w2 none bsm1 | 15.45 |
| w2 ping-pong bsm0 | 17.02 |
| **w2 ping-pong bsm1 ilp2** | **17.68** |

So the full winning stack is **~13.4 -> 17.68 TF (+31%)**: ping-pong ~+14%, batch_softmax
another ~+4% on top. Best config: `n_waves=2, sched_mode="pingpong", batch_softmax=True,
qk_ilp=2`.

**TFLOPS-vs-length vs the CK plot (B1 H24 D128): partial.** Winner config, H24 B1:
L512 = 12.84 TF, L2048 = 15.09 TF (climbing with length as expected). Long sequences
(L >= 4096) are **not measurable on this mini** -- it is a laptop-class Ryzen AI Max 395
that **thermally throttles under the sustained 115-iteration timing loop on large
launches** (observed a clock-crash stall: one L2048 trial ran 359 s -> 0.14 TF). The
plot's regime (L 1k-29k) is exactly where the throttle bites, so an in-regime comparison
to CK's ~26 TF needs a non-throttling **server board (b48-02, 40-CU, when un-drained)**.
At the short comparable lengths our ~13-15 TF (f16) sits ~half of the plot's CK-dense
~26 TF / triton ~20 TF -- but those are production library kernels (bf16) measured at
long sequences where throughput is higher; not an apples-to-apples point.

### Double-buffered cooperative LDS (`fmha_coop.py`) — measured dead-end

Implemented a separate double-buffered cooperative kernel: two rotating LDS slabs
for K/V, register-staged prefetch of tile kt+1 (non-blocking global loads that
overlap the current tile's WMMA), ping-pong `s_setprio`. Correct (max_abs
3.05e-05). Measured on the 40-CU mini (vLLM image, S512 B2 W2):

| config | TF | vgpr |
|---|---:|---:|
| coop LDS single-buffer | 6.21 | 216 |
| coop LDS double-buffer | 5.51 | 244 |
| gather + ping-pong winner | ~15-17 | 205 |

**Cooperative LDS staging is ~2.5-3x slower than cache-resident gather**, and
**double-buffering makes it worse** (5.51 < 6.21). Root cause: on this cache-rich
APU K/V are L1/L2-resident so the gather is nearly free; the LDS path's V-transpose
scatter + cross-wave barriers are pure overhead, and prefetching loads that were
never stalling just adds VGPR pressure (216->244). You cannot hide a latency that
isn't there. **The gather + ping-pong + batch_softmax winner (17.68 TF) stands.**
The remaining WMMA-density lever (bigger per-wave tiles, area #1) must be pursued
with the GATHER path (e.g. `fmha_regblocked` m_repeat/n_repeat blocking), not LDS
staging.

## Honest ceiling (plan §5)

Even CK sits at ~29% of the 59 TF f16-WMMA peak (WMMA is ~1% of issued
instructions). Levers 1-3 target the **rocke↔CK gap (10→~17 TF)**, not the
CK↔peak gap. Do **not** expect >~17 TF from this kernel family on gfx1151 — the
rest needs gfx12 (larger-K WMMA, `ds_read_tr`, packed-f16 C). If Lever 1 alone
under-delivers, it is expected: it unblocks the spill, but Lever 2's 4-warp
density is the load-bearing structural fix, and they must land together.

---

# Large-sequence (L≥8K) D128 campaign — failed experiments + what won

Follow-on campaign: push the compute-bound ~25 TF out to large sequence length
(L8K–32K) on gfx1151. Dense fp16, D128, H24 B1, wave32 WMMA, ~32 MB MALL, **no
`ds_read_tr`, no `buffer_load_lds`**. Numbers measured on the board; large-L is
timing-only (correctness at L≤2048, max_abs ~1e-4 for f16-O-carry paths).

## TL;DR
* **Won:** register **MQ2 + f16-O-carry** (25 TF to ~L4K), **host-side head-chunked
  launches** (bound concurrent KV to the MALL → 20.5/16/11 TF @ L8K/16K/32K), and
  **non-temporal O stores** (`o_nt`, +3–14% @ large L).
* **Blocked by hardware:** gfx1151 lacks **`buffer_load_lds`** (no DRAM→LDS DMA)
  and **`ds_read_tr`** (no LDS transpose read). Together they sink every
  LDS-staging / async-prefetch scheme. CK Tile hits 25 TF at large-Sq on **CDNA**
  because it has both.
* **Bottleneck @ L16K (rocprofv3):** NOT DRAM (DRAM ~13 GB/s ≈ 5% of peak; L2/GL2C
  hit 78.8%). **Memory-unit bound (MemUnitBusy ~80%, GPUBusy 100%, WriteUnitStalled
  ~0%)** on cache-resident KV; the 20% idle is *structural* (o_f16 f16↔f32 converts,
  register P-transpose `permlanex16`, WMMA-operand `vmcnt` drains) — not reducible
  scalar softmax VALU.

## Winning configs
| shape | TF | mechanism |
|---|---:|---|
| D128 L1024 | 25.45 | swapqk MQ2 + o_f16 |
| D128 L4096 | 24.07 | MQ2 + o_f16 (register KV reuse → knee at ~L4K) |
| D128 L8192 | 20.5 | head-chunked, C=8 (32 MB concurrent KV) |
| D128 L16384 | 16.0 | head-chunked, C=2 (16 MB) |
| D128 L32768 | 11.3 | head-chunked, C=1 (16 MB) |

Head-chunking = **serialized** per-`C`-head launches (offset Q/K/V/O ptrs, grid
`Y=C`), `C ≈ 16 MB / per-head-KV`, so only `C` heads' KV is MALL-resident at once.
Must stay single-stream/linear-graph — parallel chunks defeat the bound.

## Failed experiments (root cause)
**LDS / async (all blocked by the two missing instructions):**
1. `kv_lds` (coop K/V tile in LDS): ~3× slower, gap widens with L — VGPR round-trip
   spills (RDNA's only staging path), 16 uncoalesced `ds_load`s for V, 2 barriers/tile.
2. `kvstat` (KV-stationary, O+m/l in LDS): correct, KV-reuse works (flat with L) but
   pinned ~1.7 TF by LDS traffic (uncoalesced V column-gather + per-tile O RMW).
3. transposed-V-in-LDS: slightly *worse* — uncoalescing moves to a scattered store,
   the read still bank-conflicts, and V-read was never the binding cost.
4. `q_lds` (Q in LDS, incl. MQ2, per-wave `s_waitcnt`, no barrier): 3× dead-end —
   per-tile `ds_read` breaks the WMMA clause; Q is ~8 KB, L1-resident → not a
   MALL-pressure source, so staging it relieves nothing.
5. async `buffer_load_lds` (CK Tile's core lever): **HW-infeasible** —
   `raw.ptr.buffer.load.lds` fails to lower (`LLVM ERROR: Do not know how to expand
   this operator's operand!`). A/B: VGPR-staged control compiles+runs, async fails.
6. VGPR→LDS prefetch ring: same block — no DMA ⇒ in-flight loads must sit in VGPRs
   during WMMA ⇒ the D128 register wall (`pipeline` flag: 22→10 TF).

**Scheduling:** 7. `head_blocked` static cohorts — lose to the work-queue (which
already achieves residency); add imbalance. 8. persistent untransposed kernel —
superseded by head-chunked swapqk. 9. split-KV (studied) — decode-oriented; adds
O_acc DRAM traffic, no KV-traffic cut → net loss for large-Sq prefill.

**Compute micro-opts (softmax VALU is NOT the bottleneck):** 10. `prescale_k` (fold
scale into K): −38% (breaks load→WMMA fusion + register pressure). 11. lazy-rescale
in MQ: −53% (`scf.for`/`wave_all` control-flow > α-mul saved). 12. `q_nt` (stream
Q): regresses (Q reused/L1-resident). 13. shufflevector α-splat: no-op (spill
unchanged).

## Reproduce
Compile-here / run-on-board (host has no GPU; the gfx1151 board is `stx-mini`,
ROCm container `rocke_d16hi`; host `~/ckdsl_rocke` ⇄ container `/rocke`).

```bash
# --- 1. rsync artifacts to the board ---
B=stx-mini ; R=~/ckdsl_rocke
cd dnn-providers/hip-kernel-provider/rocke
# production kernel (+ o_nt) and its o_nt plumbing:
rsync -az library/kernels/gfx1151/wmma_fmha_swapqk.py \
    $B:$R/library/builders/gfx1151/attention/fmha_swapqk.py
for f in core/ir.py core/lower_llvm.py helpers/tensor_view.py helpers/distribution.py; do
  rsync -az platform/python/rocke/$f $B:$R/python/rocke/$f
done

# --- 2. perf sweeps on the board (inside the ROCm container) ---
ssh $B 'docker exec rocke_d16hi bash -lc "
  source /root/rocke-venv/bin/activate
  export PYTHONPATH=/rocke/python:/rocke/library && cd /rocke/library

  # (a) MQ2+o_f16 compute-bound sweep  [qb2 args: L D mq bn ilp of16 warmup iters [verify]]
  for L in 1024 2048 4096 8192 16384; do
    python -m builders.gfx1151.attention.qb2 \$L 128 2 32 2 1 10 30 ; sleep 4 ; done

  # (b) head-chunked large-L sweep  [cliff_chunk_probe args: L D H C [ont|nolazy|verify]]
  #     C chosen so C*per_head_KV ~ 16-32MB (fits MALL)
  python cliff_chunk_probe.py 8192  128 24 8            # 32 MB
  python cliff_chunk_probe.py 16384 128 24 2            # 16 MB
  python cliff_chunk_probe.py 32768 128 24 1            # 16 MB
  python cliff_chunk_probe.py 16384 128 24 2 ont        # + non-temporal O (o_nt)

  # (c) tracked-driver equivalents (no ad-hoc script)
  python -m builders.gfx1151.attention.benchmark --kernel swapqk \
    --seqlen-q 4096 --seqlen-k 4096 --head-size 128 --heads 24 --batch 1 \
    --set n_waves=2 --set q_block=2 --set o_f16=1 --set block_n=32 --set qk_ilp=2 \
    --set buffer_gather=1 --set dual_gather=1 --set fast_exp2=1 --no-verify
  python -m builders.gfx1151.attention.repro d128_mq2_of16
"'

# --- 3. rocprofv3 bottleneck probe (busy/stall + L2 + fetch bytes) ---
ssh $B 'docker exec rocke_d16hi bash -lc "
  source /root/rocke-venv/bin/activate
  export PYTHONPATH=/rocke/python:/rocke/library && cd /rocke/library
  rocprofv3 --pmc MemUnitBusy GPUBusy WriteUnitStalled Wavefronts -d /tmp/p -O csv -- \
    python3 -m builders.gfx1151.attention.rocprof_swapqk_runner \
      --head-size 128 --seqlen-q 16384 --seqlen-k 16384 --heads 2 \
      --q-block 2 --o-f16 1 --iters 3
  # GL2C hit-rate / FETCH_SIZE: rerun with --pmc GL2C_HIT_sum GL2C_MISS_sum  (then FETCH_SIZE)
"'
```

## Ceiling verdict (large-Sq)
The fp16 large-Sq ceiling on gfx1151 is **cache-read/gather throughput of the
memory unit** (≈641 GB/s MALL ⇒ ~20.5 TF at MQ2 AI=32), reached to ~80% by the
head-chunked MQ2 kernel. The last ~20% is structural compute intrinsic to the
transposed-QK + o_f16 + WMMA design. Closing it needs the two missing instructions
(→ CDNA/gfx12) or **higher AI per byte (fp8/quantized KV)** — every LDS, prefetch,
and softmax-VALU scheme above is exhausted and documented.
