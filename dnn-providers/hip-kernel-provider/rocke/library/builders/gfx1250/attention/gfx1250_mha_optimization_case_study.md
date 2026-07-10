# gfx1250 MHA / Decode-Attention Optimization Case Study

Checkpoint of the gfx1250 (gfx1250-class, wave32 WMMA) unified-attention work:
where the kernels stand, which optimization levers paid off, which regressed
and why, and what to try next. Companion to
`gfx1250_universal_attention_plan.md` (the original plan) — this doc is the
"what we learned" retrospective.

## 1. Context

- **Target model cohort:** Qwen3-30B-A3B attention — GQA-8 (32 q-heads / 4
  kv-heads), `head_dim=64`, `block_size=16`, bf16 Q/O, fp8e4m3 paged KV cache,
  sinks.
- **Decode (3D split-KV):** `q_len=1`, `kv_len ∈ {512,1024,2048,4096}`,
  `num_seqs ∈ {2..256}`. The hot path.
- **Prefill (2D tiled):** `q_len ∈ {64,128,2048}`.
- **Reference arch:** gfx950 (MI355X), wave64 MFMA, with async LDS DMA and
  `ds_read_tr` transpose reads.
- **gfx1250 constraints:** wave32 WMMA `16x16x32`; **no async global→LDS DMA**;
  **no `ds_read_tr`**; 160 KiB LDS/CU.

Kernels live in:
- `instances/gfx1250/attention_tiled_3d.py` (decode segment + reduce)
- `instances/gfx1250/attention_tiled_2d.py` (prefill)
- `instances/gfx1250/_wmma_attention_common.py` (shared WMMA building blocks)
- dispatch/wiring in `instances/common/attention_unified.py`
- harnesses: `examples/gfx1250/attention/decode_3d_verify.py`,
  `examples/gfx1250/attention/tiled_2d_verify.py`

## 2. Current state (measured 2026-06-17, DPP default-on)

Cross-arch decode latency (µs), fp8e4m3, sinks, total (seg+reduce), local gfx950
vs remote gfx1250, min-of-3. "best" = best gfx1250 config found per shape.

| shape       | gfx950 | gfx1250 old | **gfx1250 (best, DPP)** | config                  | **gap** (was) |
|-------------|--------|-------------|-------------------------|-------------------------|---------------|
| s2  kv1024  | 14.2   | 60.6        | **57.3**                | W1, nseg64, DPP on      | **4.04×** (4.3×) |
| s2  kv4096  | 23.5   | 68.1        | **60.2**                | W8, nseg16, DPP on†     | **2.56×** (2.9×) |
| s256 kv1024 | 69.5   | 95.7        | **83.3**                | W1, nseg16, DPP on      | **1.20×** (1.38×) |
| s256 kv2048 | 115.4  | 154.4       | **133.0**               | W1, nseg16, DPP on      | **1.15×** (1.34×) |
| s256 kv4096 | 220.6  | 259.1       | **235.6**               | W1, nseg16, DPP on      | **1.07×** (1.18×) |

† `s2 kv4096` best uses opt-in multi-wave (W8); the **production default** (W1,
nseg64, DPP) is ~87µs there. All other rows' "best" *is* the production default.

**The DPP `row_xmask` softmax lever (§4) is the driver of this checkpoint's gains**
— it closed the production large-batch cohort to **1.07–1.20×** and improved every
shape. The dispatcher already lands DPP on a winning `num_segments` for all shapes
(see §4 / §7), so no shape regresses.

**Key regime nuance (DPP is a latency-hiding lever, not a throughput lever):**
DPP moves the softmax row-reduction off the LDS port onto the VALU. It **wins when
the wave is LDS/latency-stalled** (small batch, or large batch at nseg16 where the
per-segment loops are short) and **loses when VALU/throughput-bound** (large batch
at *low* nseg=8, long per-segment loops). Measured on `s256 kv4096`: DPP at nseg16
**240.8→223.8µs (seg)**, but at nseg8 **242.6→269.2µs**. Production picks nseg16
for the large-batch cohort, so it lands on the winning side.

TFLOPS at `s256 kv4096`: gfx950 ≈ 38.9 TF/s, gfx1250 ≈ 36.4 TF/s (was 33.1).

**Headline:**
- Large-batch decode is now **~1.07–1.20× off gfx950** (was ~1.2–1.4× last
  checkpoint, ~1.5–1.6× at the start). The production-relevant regime is close.
- Small-batch decode: `s2 kv1024` **~4.0×**, `s2 kv4096` **~2.6×** (was ~2.9×).
  Overhead/occupancy + wave32 + no-async-DMA bound; the residual `s2 kv4096`
  multi-wave win (60 vs 87µs) needs `num_waves` auto-gating into production.
- Prefill-2D (gfx1250) default path: `prefill` ~52µs, `sw` ~55µs, `decode_big`
  ~362µs — all correctness-clean.

**Shipping default config (gfx1250):** single-wave `ldsP`, **DPP softmax on**
(auto `wmma_spacing=1`), no register-P, no wide-V, no hoist, `waves_per_eu=2`.
`num_segments` from AITER `select_3d` (64 for s2, 16 for s256). Multi-wave +
fancier levers gated off (opt-in).

## 3. Correctness journey (resolved)

**Symptom:** intermittent, seed-dependent FAIL on `num_seqs=256, kv_len=4096,
fp8, seg=16` — a `~1e30` sentinel leaked into ~128 output elements.

**Root-cause chain (resolved):**
1. Uninitialized fp32 partials workspace read by the reduce kernel — fixed by
   the dispatcher pre-filling `segm_max=-1e30`, `segm_expsum=0`, `segm_output=0`
   on gfx1250 before the segment launch.
2. Online-softmax sentinel handling: a fully-masked N-subtile produced
   `exp2(0)=1` mass on `-1e30` sentinels. Fixed in `softmax_row_update` with an
   `fcmp ogt(rowmax, -inf)` guard (mirrors gfx950) so empty subtiles contribute
   zero.
3. Reduce hardening: `m_finite = (m==m) ∧ (m>-inf)` before `exp2`, so any stray
   NaN/garbage partial contributes 0.
4. Binary search: `binary_search_iters = max(32, ceil(log2(num_seqs+1)))` to
   avoid under-iteration at large batch.

**Outcome:** seeds 3001–3020 PASS at the failing shape; `max_abs ≈ 7e-5`. A
regression case (256-seq build/lower) is in `tests/test_gfx1250_attention.py`.

## 4. Levers that WORKED

- **DPP `row_xmask` softmax reduction (`use_dpp_softmax`, default-on 2026-06-16):**
  the online-softmax 16-lane row butterfly (row-max + row-sum) was a 4-stage
  `ds_swizzle_b32` chain — an **LDS-port** op that serialises on `lgkmcnt`.
  Replaced it with the RDNA `row_xmask` DPP butterfly on the **VALU**, emitted as
  the **fused** `v_max_num_f32_dpp` / `v_add_f32_dpp` (one VALU op per stage; LLVM
  GCNDPPCombine refuses to fold `row_xmask` movs, so the fused op is forced via
  inline asm). ISA: **128 `ds_swizzle_b32` → 0**, 128 fused `v_*_dpp`, smaller
  binary. Perf: **1.4–1.56× on the segment kernel at single/low wave** (where
  softmax is on the critical path — `s2 kv4096` seg 80.5→52.3µs at W=1) and
  **~2–3% at the high-wave shipping configs** (softmax already hidden by
  occupancy; the softmax-ablation upper bound at W=8 is ~1µs). Numerically
  **exact** vs the ds_swizzle baseline (`max_abs ≈ 4e-5`), validated kv∈{512…4096}
  × {fp8,bf16} × W∈{1,4,8}. **Default on**, disable via
  `HIPDNN_GFX1250_3D_DPP=0` / harness `--no-dpp-softmax`. **Confirmed
  (2026-06-17) the largest production win is the single-wave large-batch (`s256`)
  cohort:** `s256 kv4096` 259→236µs, `s256 kv2048` 154→133µs, `s256 kv1024`
  96→83µs, `s2 kv1024` 81→57µs (dispatcher-selected nseg). **Regime caveat:** DPP
  is a *latency-hiding* lever — it loses when VALU/throughput-bound (large batch
  at *low* nseg8: `s256 kv4096` seg 243→269µs) but wins at the nseg16 the
  dispatcher picks for that cohort (243→224µs). The dispatcher lands DPP on a
  winning nseg for every shape (verified §7), so no shape regresses.
  *Hazard note:* DPP removes the `ds_swizzle` LDS
  serialization that **incidentally supplied the gfx1250 dependent-WMMA
  hazard gap**; without it, high-occupancy `bf16 + multi-wave` produced a garbage
  accumulator (NaN, deterministic). Fix: `__post_init__` auto-bumps
  `wmma_spacing≥1` (one `v_nop`) whenever `use_dpp_softmax` is set — re-supplies
  the gap, restores exact correctness, costs ~1µs.
- **Workspace pre-init + softmax/reduce sentinel guards (P0):** the correctness
  fix; also made the reduce robust. Kept.
- **`binary_search_iters` floor of 32:** removes a large-batch edge case. Kept.
- **Cooperative multi-wave32 CTA (`num_waves`) — small batch only:** W waves
  split a segment's KV tiles (strided, padded to a uniform trip count to keep
  CTA barriers safe), then merge partial `(m,l,acc)` via an LDS inter-wave
  online-softmax combine (wave 0 writes once). **Up to ~1.5x on small batch**
  (`s2 kv4096`: 104→68µs at W=8; `s2 kv1024`: 89→61µs at W=4). Correctness
  validated W=2/4/8. **Opt-in** (`HIPDNN_GFX1250_3D_WAVES`, harness
  `--num-waves`); not a production default yet because it regresses large batch.
- **`num_segments` (already in `select_3d_config`):** the AITER-style
  split-KV occupancy knob. More segments help small batch (`s2 kv4096`
  102→71µs at seg 32), clamp to min for large batch. Already occupancy-gated in
  production — same family of win as multi-wave.
- **`waves_per_eu=2` default for gfx1250 3D:** retained.

## 5. Levers that DID NOT work (and why)

The decode kernel is **memory-latency / bandwidth bound** (GEMV-like: each KV
element used once, arithmetic intensity ≈ 1). Levers that optimize compute or
LDS therefore don't touch the bottleneck, and several hurt by cutting occupancy
or adding cross-lane work.

- **Native fp8 PV-GEMM (`16x16x64` fp8 WMMA atom) — not built, ceiling too low
  (2026-06-16):** the plan was QK in bf16, PV in native fp8 (skip the V
  fp8→bf16 dequant + use the K=64 atom = half the PV WMMAs) for the fp8 cohort.
  Before building the from-scratch atom (operand layout undecoded, needs a K=64
  PV restructure + a per-row P→fp8 scale reduction), measured the **ceiling** via
  a new `ablate_pv` probe (skip V staging + PV-GEMM, keep QK+softmax live through
  the P_lds store). Result: PV+V-staging is only **0.6% of the segment kernel in
  the mw8 shipping config** (fully hidden by multi-wave; ablating it changes
  nothing) and **9.2% in the single-wave plain path**. Native-fp8 PV would
  recover only a fraction of that while **adding** the P→fp8 scale reduction +
  K=64 operand stitch — the same wave32 cross-lane tax that made register-P and
  ds_load_tr neutral. Net expected ≈ zero. **Not pursued.** `ablate_pv` kept as a
  debug-only ceiling probe (never ship).
- **HW transpose-LDS read for V (`use_ds_tr_reads`, `ds_load_tr16_b128`):** the
  gfx1250 `ds_read_tr` equivalent — stage V token-major, read transposed in HW
  (replaces 64 scalar `ds_load_u16`/tile). Fully decoded the WMMA B-operand layout
  + `ds_load_tr` semantics (GPU-validated) and built a GPU-**exact** `compute_pv_dstr`
  (incl. the K=32 `lane^8` `ds_bpermute` stitch). **Neutral/slightly-negative:**
  the lane-pair bpermute stitch needed to assemble the K=32 B operand eats the
  LDS-read-wall savings — the **same wave32 cross-lane tax** as register-P. Gated
  off (`--ds-tr` / `HIPDNN_GFX1250_3D_DSTR`).
- **Software-pipeline stack (`use_sw_pipeline`: DTLA async-V + `iglp_opt` /
  `sched_group_barrier` cadence):** bundled the async-V double-buffer prefetch
  with instruction-scheduling hints to overlap the per-tile serial ALU/LDS chains.
  **Neutral** on both ISA and GPU latency — the waitcnt was already tuned for
  full-tile overlap, and the softmax ablation showed the serial ALU chain
  (softmax wave-reductions, ~21% of seg at single-wave) was the real critical
  path, which scheduling hints don't shorten. Off by default. (Superseded by the
  DPP softmax lever, which attacks that chain directly.)
- **Register-resident P (drop the P→LDS round-trip):** correct version built
  (real WMMA C→A remap via `ds_bpermute`), but **~2x slower** on batched decode
  (`s256 kv4096` 259→671µs) because the wave32 cross-lane shuffles cost more
  than the LDS traffic they remove. An earlier same-lane-gather version was also
  numerically wrong (2D prefill NaN). **Gated off** (default `ldsP`); available
  via `--register-p` for further work.
- **Wide / double-buffered V (software prefetch):** doubles V_lds, drops
  occupancy; the latency it hides was already hidden. `s256 kv4096` ~259→335µs.
  Off by default.
- **Invariant hoisting (mask/index out of the KV loop):** the hoisted integer
  math is cheap and not the bottleneck — neutral to slightly negative. Off.
- **Multi-wave at large batch:** device already saturated (~24k waves); adding
  waves/CTA multiplies LDS + adds CTA barriers and serializes — **4–10x slower**
  (`s256 kv4096` 259→1061µs at W=8). Hence multi-wave must stay small-batch only.
- **Bigger KV tile for "reuse":** doesn't apply to decode — `q_len=1` means no
  Q reuse and K/V are each read once, so there is no reuse ratio to raise.
  (This lever is for prefill-2D, not decode-3D.)
- **3D graph replay (gfx1250):** wired but opt-in only
  (`HIPDNN_GFX1250_3D_GRAPH`); ROCm/CUDA-graph path not validated as a default.

## 6. Why the residual large-batch gap (~1.2x) persists

At `s256 kv4096` the time is almost entirely the segment kernel (gfx1250 248µs
vs gfx950 195µs ≈ 1.27x), both bandwidth-bound. The difference is the
latency-hiding hardware gfx1250 lacks and the levers can't synthesize:
- **No async global→LDS DMA** (gfx950 overlaps KV load with compute).
- **No `ds_read_tr`** (gfx950 reads V into the MFMA-B layout in 1–2 LDS ops).
- **wave32 vs wave64** (half the in-flight memory requests per workgroup).

These set a practical ceiling near ~1.2x without Phase-3 hardware features.

## 7. ISA-driven gap-closing program (current — 2026-06-16)

We cannot run a profiler on the remote gfx1250, so we use **static ISA
analysis** instead: compile the segment kernel to a gfx1250 code object
*locally* (`compile_kernel(arch="gfx1250")` works under ROCm 7.2/clang22) and
disassemble with `llvm-objdump -d`. This is a profiler substitute.

**Key correction:** gfx1250 **can** do async LDS DMA. `async_buffer_load_lds_addr`
lowers cleanly to `llvm.amdgcn.raw.ptr.buffer.load.lds` for gfx1250 (verified).
The `arch_specs.json` `has_async_global_lds:false` flag is **conservative/wrong**
— DTLA is usable. (Flag flip is lever 4 below.)

**Per-tile ISA hot-path (segment kernel, before lever 1):**
- `ds_load_u16` ×64 + `s_wait_dscnt` ×109 — the LDS-read wall (no `ds_read_tr`);
  the PV/P operand gather reads one 16-bit element per lane per slot.
- `s_delay_alu` ×209 — serial dependency chains, no prefetch overlap.
- `global_load_b128` ×8 — **KV loads are already wide/coalesced** (not the problem).
- `global_store_b32` (scattered) — the fp32 partials workspace writes.

**Four levers (implementing all; lever 1 default-on, levers 2–4 behind flags, then a
combined run). Dispatcher env knobs: `HIPDNN_GFX1250_3D_{WLDS,DTLA,FRED,WAVES}`.**

1. **DONE ✅ — Wide LDS reads (`use_wide_lds_reads`, default on).** Stage V
   dim-major `[HD, T]`; read each lane's WMMA-B fragment as wide `ds_load_b128`
   (2× vec8 + concat) instead of 16 scalar `ds_load_u16`. ISA: 64 → 10 LDS read
   ops/tile. GPU-correct across the cohort + flaky seeds; ~neutral standalone
   (transpose cost shifts to `ds_store_b16` ×80 on the store side), a building
   block for lever 2. Committed `026c8dab7f`.
2. **NEXT — DTLA + double-buffer + prefetch/scheduled loop (`use_dtla_prefetch`).**
   Mirror gfx950 `_issue_v_load`: byte paged-KV descriptor
   (`TensorDescriptor.naive(...).transform(indirect(block_table), unmerge(linear_half→token,dim))`),
   `async_buffer_load_lds_addr(value_rsrc, v_dst, voff, 0, dwords=4)` into a
   double-buffered token-major `V_lds[2,T,HD]` (8 calls/tile, THREADS·8=256
   halves/call, wave32), **per-wave LDS base offset** (multi-warp collision fix).
   Prologue issues tile 0; loop issues tile `i+1` before computing `i`; tune the
   `s_wait` so the load overlaps QK/PV. **fp8 cohort:** DTLA loads **raw fp8**
   into LDS (half the bytes) + dequant-on-read in `compute_pv` (the one extension
   beyond gfx950, which fell back to sync-dequant for fp8 — validate carefully).
   Note: DTLA yields **token-major** LDS, so it pairs with the scalar-gather
   `compute_pv` unless a transpose-on-read is added; the combined run measures both.
3. **Fuse-reduce (`use_fused_reduce`, requires `num_segments==1`).** Skip the
   ~268 MB fp32 partials workspace + second launch at large batch by writing the
   final normalized bf16 output directly from the segment kernel. Needs an output
   ptr in the segment ABI (a fused single-kernel variant / new signature). Extends
   the multi-wave LDS merge. Occupancy-gated (split for small batch, fuse for large).
4. **Flip `has_async_global_lds` → true (+ confirm fp8 KV).** Small, correct
   metadata fix now that DTLA is verified; check no gfx9 consumer regresses. fp8
   KV byte-reduction is already in decode + prefill.

Then: **combined cross-arch run** toggling the flags to find the best stack vs
gfx950 (primary goal). Accept per-stage regressions; the combined stack is what matters.

Deferred (still valid): faster reduce kernel (gfx1250 reduce 2–3x gfx950's,
serial over segments), small-batch startup/graph-replay, occupancy-gate
`num_waves` into production, prefill-2D parity pass.

## 8. Methodology notes

- **Always cross-arch benchmark** with `decode_3d_verify.py --bench
  --bench-split --csv` (local gfx950 = MI355X, remote gfx1250). Segment-vs-reduce
  split timing is essential to attribute wins.
- **Probe free GPUs first** (`rocm-smi --showpidgpus`) — the remote box shares
  GPUs with tensilelite jobs; pick an idle device.
- **Stop on regression.** Every lever here was kept only if it didn't regress
  the production (large-batch) cohort; small-batch-only wins are gated, not
  defaulted.
- **Correctness before perf:** run the seed sweep on `s256 kv4096 fp8` (the
  historically flaky shape) for any segment-kernel change.
- Local gfx950 harness runs under `/workspace/vllm-venv/bin/python` (has
  `ml_dtypes`); remote gfx1250 under `/tmp/rocke_gfx1250/venv/bin/python`.
- **Static ISA analysis (profiler substitute):** locally
  `compile_kernel(build_unified_attention_3d_tiled(spec), arch="gfx1250")`,
  write `art.hsaco`, then `/opt/rocm/llvm/bin/llvm-objdump -d file.hsaco`. Count
  `ds_load_*` / `s_wait_dscnt` / `s_delay_alu` / `global_load_*` to find the
  per-tile bottleneck without a GPU.
- **Remote GPU:** sync with `rsync python/ -> /tmp/rocke_gfx1250/python/` on
  `yraparti@ctheliosp-1b114-f06-1.mnb.dcgpu`; **use GPU 2** (`HIP_VISIBLE_DEVICES=2`)
  for gfx1250 today; probe `rocm-smi --showpidgpus` first (box is shared).
- **Push:** the working tree is `rocm-libraries-internal`; push with
  `GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_yraparti_amdeng -o IdentitiesOnly=yes" git push`
  (that key authenticates as `yraparti_amdeng`, which has repo access).

## 9. Checkpoint (2026-06-16, evening)

**Landed today:**
- **DPP `row_xmask` softmax reduction — default on** (§4). New IR primitives
  `dpp_xor` + fused `vop2_f32_dpp_xor` (`core/ir.py`), LLVM + HIP lowering
  (`update.dpp.i32`, `dpp_ctrl=0x160|mask`), `use_dpp` path in
  `helpers/attention.py::wave_reduce_max/sum`, spec flag `use_dpp_softmax=True` +
  `wmma_spacing≥1` auto-bump (`attention_tiled_3d.py`), dispatcher env
  `HIPDNN_GFX1250_3D_DPP` + cache key, harness `--dpp-softmax`/`--no-dpp-softmax`.
  23 gfx1250 attention unit tests pass; GPU-exact across kv∈{512…4096}×{fp8,bf16}
  ×W∈{1,4,8}.
- **`ablate_pv` debug probe** (skip V-staging + PV-GEMM) — used to kill the
  native-fp8-PV idea on a measured ceiling (§5). Harness `--ablate-pv` / `--plain`.

**Decisions captured:** native-fp8 PV, `ds_load_tr`, and the software-pipeline
stack are all **gated off / not pursued** with measured rationale in §5 (all hit
the wave32 cross-lane tax or a too-low ceiling).

**Update (2026-06-17): full cross-arch table re-measured (DPP on) — §2.**
DPP closed the production large-batch cohort to **1.07–1.20×** (`s256 kv4096`
259→236µs) and improved every shape (`s2 kv1024` 61→57µs, `s2 kv4096` 68→60µs).
**Dispatcher verified:** it already selects nseg64 (s2) / nseg16 (s256) + DPP —
the DPP-winning nseg for each cohort — so production lands on the right side of
the latency/throughput regime split with **no per-shape regression and no
dispatcher change needed**. (Tooling glitch aside: working tree was wiped by a
sparse-checkout hiccup; `git restore` recovered it — all work was
already committed in `b27ffa1c1b`.)

**Next steps (in priority order):**
1. **Occupancy-gate `num_waves` into production.** The only shape leaving a win
   on the table is `s2 kv4096`: production W1/nseg64 = ~87µs vs opt-in W8/nseg16 =
   **60µs**. Auto-select multi-wave for the small-batch cohort (currently
   `HIPDNN_GFX1250_3D_WAVES` env-only). Needs a batch-size gate in
   `_gfx1250_3d_num_waves` + confirm it composes with DPP (DPP wins at W8 too).
2. **Reduce kernel** (still ~2–3× gfx950's, serial over segments) — now a larger
   *relative* share. Candidates: DPP-ize its 32-lane reduction (needs a mask-16 +
   cross-row stage; the 16-lane `dpp_xor` already exists), `use_fused_reduce` for
   the `num_segments==1` cohorts.
3. **Remaining segment bottleneck** is now the QK/PV WMMA pipeline + fixed
   launch/occupancy overhead (softmax is no longer the lever at high occupancy —
   ablation saves ≤1µs at W=8). Re-profile the per-tile ISA hot-path after DPP.
4. prefill-2D parity pass.

Prior checkpoint: branch `users/yraparti/rocke/gfx1250_attention_prototype`;
lever 1 wide-LDS-reads + lever 2 DTLA (opt-in) landed earlier.
