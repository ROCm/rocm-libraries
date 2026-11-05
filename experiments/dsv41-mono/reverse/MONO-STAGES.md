# ATOM's DeepSeek-V4.1 mono kernels, reverse-engineered (W12 probe P2)

What ATOM's FlyDSL V4.1 mono decode kernels do in every stage, as compiled for the TP2 served widths. This covers K1
`attn_pre` and K2 `layer_post` (K2a `attn_post` + K2b `moe`, one launch), for both the target and the DSpark draft.
It also says where a rocKE port can take bitwise parity, and which Qwen3.8 levers apply. The section ends with the
first port, K1's `wqkv_a` GEMV as a rocKE stage function linked into ATOM's own K1: bitwise, and timed in situ against
ATOM's.

Ratios and static instruction counts only. Absolute times are in `hackathon-prep/private-perf/dsv41-mono/P2-ABS.md`.
Source: ATOM `origin/main` df86c482, an export in `dev/ATOM-dsv41` (read-only except for the default-off hook in
section 6). The image is `rocm/atom-dev:nightly_202610171506` (FlyDSL 0.4.0, torch 2.10 + ROCm 7.2.4).

Source-level structure (launches, contracts, plans) is in `../research/atom-v41-mono.md`. This file adds the compiled
code, the measurements and the port.

## 0. Summary

- **Where the time is.** Served TP2, mono + DSpark, CONC 1 (the B0 kineto trace, rank 0): K2 takes **0.69** of GPU
  busy time and K1 **0.15**. Everything else (native compressor, Engram, draft GEMMs, sampler, comms) is 0.16.
  So V4.1 is a K2 problem first.
- **K2 is register-bound before it is anything else.**
  - Every target K2 build spills **790-1209 SGPRs** to VGPR lanes (`v_writelane` / `v_readlane`: 790 / 1727 at the
    6-row verify width) and runs at 226-249 VGPRs.
  - The spills sit in the prologue and the indexer stages (`isel0/1`, `iblk0`). One K2 build serves all 40 layers,
    including the 30-odd that never select, so they all pay for the indexer's register pressure.
  - K1 builds do not spill (83-162 VGPRs).
- **Every scaled-MFMA GEMV rewrites its scale VGPR right before the MFMA.** This is the pattern P-L3 measured as much
  slower MFMAs on gfx950. Counts at the 6-row target verify:

  | stage | rewrites / MFMAs |
  |---|---|
  | wqkv_a | 4 / 5 |
  | wq_b | 7 / 8 |
  | wo_a | 7 / 8 |
  | wo_b | 6 / 8 |
  | shared | 9 / 10 |
  | ug | 12 / 20 |
  | down | 35 / 42 |
  | down at 48 rows | 103 / 126 |

- **What V4.1's mono already does that Qwen's did not.**
  - Weight streams are `nt`.
  - Polls pause with `s_sleep 1`.
  - TP reduction is in-kernel: rank-ordered sums over uncached HIP-IPC peer buffers.
  - Token-parallel reduce-scatter from 12 rows.

  What it lacks from our Qwen stack: NUMA dual copies, holds, XCD homing, prepoll, per-layer-kind builds, and epoch
  tags (it zeroes and fences the mailboxes every step).
- **The port: K1 `wqkv_a` in rocKE** (`rocke/library/kernels/gfx950/dsv41_mono/stage_wqkv.py`), linked into ATOM's own
  K1 through a default-off hook (`ATOM_DSV41_RK_EXTERN`).
  - It is **bitwise** with ATOM's K1: every output buffer plus the whole mailbox scratch, 3 seeds, rows 1 / 5 / 6.
  - Must-fail mutants are caught.
  - With the hook off, K1 / K2 `.ll` and `.text` are byte-identical to pristine ATOM.
  - In situ, the whole K1 runs at **0.984-0.989** of ATOM's K1 at the 5- and 6-row widths on **both nodes**. The
    A-vs-A twin and the recompiled control stay at 0.997-1.001. At 1 row it is a tie (0.993-0.995, inside the band).
  - The wqkv_a stage's own time is **0.95-0.97** of ATOM's.
  - The `sop` (packed scales) and `ra` (LDS read-ahead) levers stay bitwise but are neutral on this 5-MFMA stage.

## 1. Data and method

Each job is one 1-GPU job on the `rocke` partition (`hackathon-prep/dsv41/jobs/p2_job.sh` + `p2/p2_run.sh`).

| job | what |
|---|---|
| p2a1 (n01-09) | dumps, identity, splits; in-situ rig at rows 6 / 1 / 5 with 8 knob arms and 5 mutants |
| p2b7 (n07-17) | rig at rows 6 / 5 (second node) |
| p2c2 (n01-09) | rig with the `sop` / `ra` levers and the `opsel` mutant |

- **Dumps** (`p2/p2_dump.py`). Compile-only FlyDSL builds of the served TP2 widths, with each build's `.pre.ll`,
  `.hsaco` and resources.
  - Target K1 at 6 / 12 / 24 / 48 rows (fold, indexer, layer-0 variants).
  - Target K2 at 1 / 6 / 12 / 24 / 48 rows.
  - Draft K1 / K2 at 5 and 40 rows.
- **Stage markers.**
  - In the dump process only, FlyDSL's location capture (`meta.capture_user_location`, the statement-floor
    `_flydsl_loc`) and ATOM's own `enter_stage` calls are wrapped. Every captured location gets an outermost caller
    frame at a marker line for the stage being traced.
  - `rv_isa.py split --registry markers.json` (the Qwen reverse toolkit) then attributes 0.80-0.94 of each code
    object directly. The rest takes its neighbours' stage.
  - The marked builds' `.text` equals the plain builds' (2 of 2 checked).
- **Identity.** With the hook unset, K1 at 6 and 12 rows and K2 at 6 rows have the same `.pre.ll` and `.text` as
  pristine ATOM df86c482.
- **In-situ rig** (`p2/p2_rig.py`).
  - ATOM's real K1 runs with synthetic FP8 / E8M0 weights at TP2 widths, on a non-token-parallel build (no peers
    needed).
  - **Numerics:** each arm runs once on zeroed buffers. Every output and the whole mailbox scratch are compared byte
    for byte with the FlyDSL K1, over 3 input seeds, with a base-vs-base repeat as a determinism check.
  - **Timing:** each arm is one CUDA graph of a scratch zero plus 16 K1 launches, layer i on its own weight copy
    (beyond the MALL, so cold). There are 24 rounds of 5 replays, in shuffled arm order. Ratios are paired per round,
    then the median is taken. Controls are the A-vs-A twin graph (`base_b`) and a recompile of the FlyDSL K1 in the
    same process (`base_rc`).
  - **Stamped decomposition:** timeline builds of the same arms (K1's `s_memrealtime` after each stage), 40 eager
    launches each, interleaved.

## 2. The kernels and their stages (static code, TP2)

Shares are of a code object's instructions (static, not time). Column abbreviations:

| column | meaning |
|---|---|
| MFMA | MFMAs |
| ld | vector loads |
| nt | `nt` loads |
| sc1 | `sc1` (mailbox) accesses |
| vm0 | full `vmcnt(0)` waits |
| bar | CTA barriers |
| polls | poll loops |
| wl / rl | SGPR spill lane writes / reads |

### 2.1 K1 `attn_pre`, target, 6 rows (CONC-1 verify), fold, non-indexer layer

83 SGPRs and 124 VGPRs, no spills, 59,968 B of LDS.

| stage | share | MFMA | ld | nt | sc1 | vm0 | bar | polls | what the code does |
|---|---|---|---|---|---|---|---|---|---|
| slice (160 tasks) | 0.143 | 8 (f32 16x16x4) | 30 | 0 | 3 | 2 | 3 | 0 | Owed post into the residual. Mix projection partials as f32 MFMAs. |
| norm (S) | 0.182 | 0 | 42 | 0 | 24 | 2 | 2 | 1 | RMSNorm, then `quantize_fp8` (`code_ceil`). Puts x8 words and codes. |
| wqkv_a (112 x 16 rows) | 0.158 | 5 (scaled fp8) | 49 | 10 | 35 | 16 | 3 | 1 | FP8 32x32-block GEMV. Weights in flight before the x8 poll. |
| gate (S) | 0.192 | 0 | 19 | 0 | 16 | 5 | 2 | 1 | 160 partials, then Sinkhorn x 20 with DPP sums (285 `s_nop` cycles). |
| qkv (S) | 0.096 | 0 | 33 | 0 | 21 | 2 | 2 | 0 | q_norm to MXFP8, kv_norm + RoPE + FP8 QAT into the window ring. |
| wq_b (512 x 16 rows) | 0.214 | 8 (scaled fp8) | 43 | 16 | 10 | 3 | 9 | 1 | FP8 GEMV with a RoPE epilogue. Two tasks per CTA; weights before the q-latent poll. |

Notes on the code:

- The rows only grow `slice` and `wq_b`. At 48 rows `wq_b` is 0.312 of K1, `slice` 0.218, and VGPRs reach 162.
  Indexer layers add `index_w` and `index_quant` (0.22 of the code) and 4 SGPR spills.
- **wqkv_a waits:** 16 full `vmcnt(0)` waits for 49 loads. The first iteration of every poll is peeled
  (`sync._spin`: load, test, then loop). So each of the 2 poll batches is emitted twice, with its own wait.
  - The rocKE port's single-loop poll has 17 fewer loads and 14 fewer full waits in the whole K1.
  - The weight loads are issued first, but a poll's `vmcnt(0)` waits for them too: loads return in order.
- **wqkv_a MFMAs:** 4 of 5 come right after `s_waitcnt lgkmcnt(0)` (no LDS read-ahead), and 4 of 5 chain on the
  previous accumulator. This is the same pattern the Qwen catalogue found in A1 / M1 / B1 / Q1.

### 2.2 K2 `layer_post`, target, 6 rows

104 SGPRs **with 790 SGPR spills** (lane writes 790, lane reads 1727), 228 VGPRs, 149,520 B of LDS.

| stage | share | MFMA | ld | nt | sc1 | vm0 | bar | polls | wl / rl | what |
|---|---|---|---|---|---|---|---|---|---|---|
| entry | 0.034 | 0 | 9 | 0 | 0 | 2 | 0 | 0 | 151 / 101 | Argument and resource setup, spilled |
| iscore | 0.050 | 64 (fp8 16x16x32) | 40 | 0 | 17 | 9 | 1 | 0 | 18 / 30 | FP8 index logits, 61 of 64 MFMAs chained, to global at device scope |
| isel0 / isel1 | 0.141 / 0.141 | 0 | 39 / 65 | 0 | 58 / 67 | 6 / 8 | 10 / 9 | 1 / 1 | 119 / 468, 176 / 370 | `lds_topk` 4-pass radix, two levels |
| iblk0 | 0.100 | 0 | 6 | 0 | 26 | 4 | 10 | 1 | 252 / 396 | candidate-block top-k |
| attn_score | 0.021 | 16 (bf16) | 38 | 0 | 7 | 5 | 0 | 1 | 0 / 32 | bit-matched scores, `buffer_load ... lds` |
| irq | 0.058 | 16 (bf16) | 57 | 0 | 45 | 15 | 5 | 2 | 37 / 89 | PV, split combine, inverse RoPE, group-32 FP8 |
| wo_a | 0.039 | 8 (scaled fp8) | 52 | 16 | 30 | 3 | 3 | 1 | 0 / 1 | grouped FP8 GEMV |
| wo_b | 0.038 | 8 (scaled fp8) | 52 | 16 | 32 | 4 | 5 | 1 | 16 / 29 | FP8 GEMV; the bf16 partial is pushed to every rank (CM_SYS) |
| slice | 0.029 | 8 | 32 | 0 | 7 | 3 | 4 | 1 | 2 / 6 | rank-order sum of the TP partials (the all-reduce), FFN seam |
| norm | 0.018 | 0 | 42 | 0 | 43 | 2 | 3 | 0 | 0 / 5 | ffn_norm; rows published at device scope + per-token XRDY |
| xq | 0.015 | 0 | 6 | 0 | 13 | 4 | 1 | 1 | 0 / 13 | MXFP8 (`code_round`) for the experts, FP8 for the shared expert |
| router | 0.010 | 5 (bf16) | 22 | 5 | 18 | 3 | 2 | 1 | 0 / 17 | fixed-order bf16 GEMV, shared with the native router |
| route | 0.035 | 0 | 18 | 0 | 28 | 1 | 1 | 0 | 0 / 29 | sqrtsoftplus + bias, top-6, union U, LDS route table |
| shared | 0.060 | 10 (scaled fp8) | 68 | 20 | 44 | 19 | 8 | 3 | 5 / 40 | shared-expert FP8 GEMV + silu_and_mul |
| gate (FFN) | 0.034 | 0 | 19 | 0 | 16 | 5 | 2 | 1 | 0 / 3 | Sinkhorn for the next seam |
| ug | 0.064 | 20 (scaled fp4 x fp8) | 41 | 25 | 19 | 18 | 10 | 2 | 14 / 23 | gate/up interleaved MXFP4 x MXFP8, stage-2 MXFP8 quant |
| down | 0.098 | 42 (scaled fp4 x fp8) | 151 | 56 | 93 | 14 | 16 | 4 | 0 / 65 | down GEMV x routing weight, bf16 adds in top-k order, push to every rank |
| reduce | 0.014 | 0 | 4 | 0 | 4 | 2 | 0 | 3 | 0 / 10 | rank-order sum of the MoE partials |

Notes on the code:

- With rows, `irq` grows to 128 MFMAs at 48 rows, and `down` to 126 MFMAs with 54 full waits. SGPR spills grow to
  1209 (entry 455).
- Token-parallel builds (12 rows and up) add `gather` and push x through the peer XG region.
- The draft K2 (5 rows, 128 experts / top-3, no indexer) spills 220 SGPRs. There, `down` (0.18), `ug` (0.13),
  `shared` (0.12) and `irq` (0.10) lead.

### 2.3 Sync, mailboxes, TP reduction, MoE, FP8 / MXFP4: what the code confirms

- **Hand-offs.**
  - Every hand-off is an 8-byte `(payload, tag)` pair, written by one `buffer_store_dwordx2 sc1` (device scope) or
    `sc0 sc1` (peer).
  - Consumers poll batches of up to 12 pairs with `buffer_load_dwordx2 sc1`, `s_waitcnt vmcnt(0)`, tag compares and
    `s_sleep 1`. The first iteration is peeled.
  - Bulk data (indexer logits, `normed` rows) goes to global memory at device scope. It is then drained
    (`vmcnt(0)`), barriered, and announced by one flag (`publish`).
  - There is no grid barrier.
  - Tags are `layer + 1` (+64 for K2a, +128 for K2b). So `step_begin` zeroes every mailbox and fences the peers
    each step: one extra launch (0.003 of busy time) and a cross-GPU fence on the critical path.
- **TP reduction in-kernel.**
  - `wo_b` and `down` push bf16 partials to every rank's uncached IPC region.
  - The next stage (K2a `slice`, K2b `reduce`) sums them in rank order 0..TP-1 (f32 -> bf16), so the result is
    rank-identical.
  - From 12 rows (S % TP == 0), the pushes go only to each token's owner: a reduce-scatter. The per-token stages then
    run on owned rows, and the step's last hidden rows are gathered once by aiter's custom all-gather.
- **MoE.**
  - The router is a fixed-order bf16 GEMV.
  - A one-wave-per-token route builds an LDS route table: token bitmaps, the union U of experts with picks, slots.
  - Only the experts in U are streamed (`ug` |U| x parts, `down`).
  - The shared expert runs as an FP8 GEMV next to the router window.
- **FP8 / MXFP4.**
  - Attention projections and the shared expert are E4M3 with E8M0 scales per 32x32 block. They are consumed directly
    by `v_mfma_scale_f32_16x16x128_f8f6f4` (fp8 x fp8): the weight-scale byte per (row group, K step, lane / 16) comes
    from a dword load plus a VALU shift, and the token-scale byte from LDS.
  - Routed experts are MXFP4 (`float4_e2m1fn_x2`, scales in aiter `shuffle_scale` order) x MXFP8 activations
    (fp4 x fp8 scaled MFMA).
  - Two code rules are kept per stage: `code_ceil` for ATOM's `quantize_fp8` and the norms, and `code_round` for
    aiter's MXFP8 quant.

## 3. Versus the Qwen FlyDSL mono and our rocKE Qwen stages

| | Qwen3.8 FlyDSL mono (`reverse/FLYDSL-STAGES.md` of qwen38-true-mono) | V4.1 FlyDSL mono |
|---|---|---|
| launches per layer | 1 | 2 (K1, K2), plus native compressor / Engram between them |
| weight streams | plain `buffer_load_dwordx4` (no `nt`) | `nt` on every GEMV weight stream (wqkv 10, wq_b 16, wo 16, shared 20, ug 25, down 56 at 6 rows) |
| polls | hard spin (no pause) | `s_sleep 1` pause; first iteration peeled (twice the loads and waits per poll site) |
| MFMA chains | no LDS read-ahead, serial accumulator, scale rewrite 9/20 (A1) to 11/12 (C1) | the same three patterns in every GEMV (section 0) |
| registers | VGPR-bound, no spills | K2: 790-1209 **SGPR spills**, VGPR 226-249; K1: none |
| TP | TP1 only | in-kernel rank-ordered all-reduce or reduce-scatter |
| per-step reset | none (epoch tags) | `step_begin` zero + peer fence every step |
| schedule levers | ATOM-truemono `ATOM_Q38_SCHED` (numa, holds, spread, prio, dyn, prepoll) | none |

Our rocKE Qwen stages won by streaming `nt`, fusing stage boundaries, using wide loads, fused DPP reductions and fewer
drains (Qwen catalogue section 3). V4.1 already has `nt` and pauses. The remaining V4.1 headroom is these, in order of
evidence:

- K2's register pressure;
- the per-step reset;
- the schedule levers;
- the GEMV patterns (read-ahead, scale packing, drains);
- stage fusion across K1 -> K2 and the native ops between them.

## 4. Port candidates (bitwise parity first)

| candidate | share (CONC 1, TP2) | bitwise feasibility | cost | notes |
|---|---|---|---|---|
| **FP8 GEMV family**: wqkv_a (done), wq_b, wo_a, wo_b, shared expert, indexer wq_b | wq_b + wqkv_a about 0.37 of K1 code; wo_a + wo_b + shared about 0.14 of K2 code | **by construction** (integer MFMA operands, fixed wave-order sum, `bf16` RNE) | low: one emitter (`stage_wqkv`'s chain) plus per-stage epilogues (RoPE for wq_b, peer push for wo_b, silu for shared) | the same `_fp8_step_load` / `_fp8_step_mfma` / `row_sum` core everywhere |
| **K2b routed MoE** (xq, route, ug, down, reduce) | the largest K2 stage group at every width (ug + down 0.16-0.31 of K2 code) | yes: per-expert bf16 adds in top-k order, A8W4 MFMAs, ATOM's route table rules | high (1.6k-line `moe.py`) | template: our Qwen fused MoE (0.78-0.89 of the mono MoE) |
| **K2 indexer** (iscore, isel, iblk) | 0.43 of K2 code; runs on selecting layers only | yes (exact radix top-k, lowest-index ties) | medium-high | its register pressure is paid by every layer (section 5, lever K2-split) |
| attention (score, irq) | 0.08 of K2 code at 6 rows, 0.16 at 48 | yes (ATOM already bit-matches a fixed 6-row order) | medium | `buffer_load ... lds` keys |
| mHC slice / gate (Sinkhorn) | 0.33 of K1 code (+0.06 of K2) | yes (fixed DPP sum order) | medium | Qwen WP2a HC port pattern |

## 5. Which Qwen levers apply (all to be built as default-off knobs, with in-situ A/B)

| lever (Qwen evidence) | V4.1 site | why it should transfer |
|---|---|---|
| **NUMA dual copies** (round 0.93, layer 0.89-0.95) | every scratch mailbox (`x8`, `qkv`, `lin`, `pmix`, the K2a / K2b hand-offs) | same single-store tagged pairs, polled from every XCD; V4.1 has none |
| **holds** | K1 `wq_b` weights issued at stage entry while `wqkv_a` and `qkv` are critical; K2b shared-expert stream versus the router window; slice prefetch | the same early-stream contention; ATOM orders by "weights before the wait" only |
| **spread** | the serial per-token stages: `gate` (Sinkhorn, S tasks), `norm` (S), `qkv` (S) | few CTAs on the critical path while the rest wait |
| **XCD homing** | expert -> XCD affinity for `ug` -> `down` (an expert's gate/up and down in one L2); a token's `norm` -> `wqkv_a` -> `qkv` chain | intra-XCD hand-off protocol (no cross-XCD write-back) |
| **prepoll** | `wq_b` (q latent), `wo_a` (attention output), `down` (ug flags) | polls placed after weight issue already; prepoll moves the first poll earlier |
| **epoch tags / CCL** (our CCL: no per-launch reset) | `step_begin` zero + fence | the tag scheme forces the per-step reset |
| **per-kind builds** (new for V4.1) | split K2 into selecting and non-selecting layer builds | non-selecting layers would not carry the indexer's registers (SGPR spills, VGPR 228+) |
| **sop / read-ahead** (P-L3: rewritten scale VGPR, slow MFMA) | every GEMV (section 0); K2b `down` at 48 rows has 103 rewrites | measured neutral on wqkv_a (5 MFMAs a wave); try where MFMAs per wave are many (`down`, `ug`, `irq`) |
| **direct compiler** (Rust) | rocKE functions only; a hybrid build is compiled by FlyDSL's LLVM | applies once a whole K1 / K2 is rocKE |

## 6. The port: K1 `wqkv_a` as a rocKE stage function

- **rocKE** (`library/kernels/gfx950/dsv41_mono/stage_wqkv.py`). A device function
  `rocke_dsv41_wqkv_n<rows>` (rows 1-16) re-implements ATOM's `gemv_fp8_loads` + `load_x8` + `stage_wqkv`, in ATOM's
  order:
  - wave w takes K steps w, w + 8, .. with nt weights in flight before the poll;
  - the same poll batches into LDS at ATOM's padded row stride;
  - the 5-step scaled-MFMA chain on one accumulator;
  - the eight waves' partials summed from +0.0 in wave order, bf16 RNE, and put as `(f32 bits, tag)` pairs.

  Its ABI is the stage's own buffers: mailbox region addresses, and LDS word offsets from the `red` buffer.
  `check_atom()` compares its constants with the ATOM tree.
- **ATOM** (`ATOM-dsv41`, additive, default off). `kernels/rk_extern.py` reads
  `ATOM_DSV41_RK_EXTERN=wqkv=<dir>` and links `wqkv_n<own_n>.ll` with `flydsl.compiler.extern_link`.
  - Its digest goes into the build key and the kernel name.
  - It applies to one-token-tile builds (own rows <= 16: target verify widths up to 30 rows at TP2 with token
    parallel, draft widths 5 / 10 / 15).
  - `attn_pre.py` calls it in place of the FlyDSL stage under `const_expr`.
  - **Off: byte-identical.** The traced MLIR (locally), and `.pre.ll` and `.text` (cluster) of K1 at 6 / 12 rows and
    K2 at 6 rows, equal pristine df86c482.

### 6.1 Gates

- **Bitwise:** every arm, 3 seeds, rows 1 / 5 / 6, both nodes. All outputs (residual, post / comb / pre, q, q latent
  MXFP8, ring rows) and the whole mailbox scratch are equal to the FlyDSL K1's. The base-vs-base repeat is also
  equal (K1 is deterministic).
- **Must-fail mutants**, all caught at every width they apply to:
  - `put_bit`: the payload's low bit flipped;
  - `chain_drop`: a K step dropped;
  - `row`: tokens read token 0's sums;
  - `scale_byte`: the wrong weight-scale byte;
  - `opsel`: with `sop=2`, the wrong packed scale byte.

  `wave_order` (an fp32 reassociation of the wave sum) is report-only. It stayed bitwise in every run: the reordered
  f32 sums never crossed a bf16 rounding boundary.
- **Resources:** K1 with the function linked uses 114 VGPRs against FlyDSL's 124 (6 rows), the same LDS, and no
  spills.

### 6.2 In-situ timing (whole K1, ratio to ATOM's K1, median of 24 paired rounds [min, max])

| job (node) | rows | rocKE wqkv_a | A-vs-A twin | recompiled FlyDSL | wqkv_a stage time (stamps) |
|---|---|---|---|---|---|
| p2a1 (n01-09) | 6 | 0.986 [0.981, 0.991] | 1.000 [0.993, 1.007] | 1.000 | 0.967 |
| p2c2 (n01-09) | 6 | 0.986 [0.981, 0.991] | 0.998 [0.994, 1.004] | 0.998 | 0.962 |
| p2b7 (n07-17) | 6 | 0.984 [0.980, 0.989] | 1.000 [0.995, 1.008] | 0.998 | 0.948 |
| p2a1 (n01-09) | 5 | 0.989 [0.979, 0.991] | 1.000 [0.991, 1.007] | 1.001 | 0.965 |
| p2c2 (n01-09) | 5 | 0.988 [0.979, 0.993] | 1.001 [0.995, 1.006] | 1.001 | 0.958 |
| p2b7 (n07-17) | 5 | 0.983 [0.978, 0.991] | 0.997 [0.991, 1.005] | 1.000 | 0.966 |
| p2a1 (n01-09) | 1 | 0.995 [0.976, 1.009] | 1.000 [0.986, 1.006] | 1.001 | 0.989 |
| p2c2 (n01-09) | 1 | 0.993 [0.983, 1.000] | 0.991 [0.983, 1.001] | 0.991 | 0.960 |

- **Verdict:** a WIN at 5 and 6 rows on both nodes. The arm's maximum is at or below the controls' medians in every
  job. At 1 row it is a TIE: inside the controls' band.
- **Size of the win:** the stage runs at 0.95-0.97 of ATOM's. The whole K1 gains about 1.2-1.7 %. K1 is 0.15 of the
  served step, so this one stage is worth well under 0.5 % of a CONC-1 step.
- **Where it comes from** (whole-K1 ISA, 6 rows): 206 fewer instructions, 17 fewer vector loads and 14 fewer full
  `vmcnt(0)` waits (the peeled first poll iteration), and 10 fewer VGPRs.

### 6.3 Lever screen on the port (all bitwise)

Arms at 6 rows. Losers are kept as default-off knobs, "rejected on base rk".

| knob | value | ratio | verdict |
|---|---|---|---|
| poll pause | `s_nop 0` | 0.984 | tie with rk |
| poll pause | none | 0.985 | tie |
| `poll=asm` | rocKE asm poll | 0.986 | tie |
| `poll_max` | 16 | 0.991 | slight loss |
| `w_cpol` | 0 (no nt) | 0.986 | tie |
| `red_bar` | `lds` | 0.985 | tie |
| `sop` | 1 | 0.983 | tie |
| `sop` | 2 | 0.983 | tie |
| `ra` | 1 | 0.984 | tie |
| `sop=2` + `ra=1` | | 0.982 | tie, inside the rk band |
| `x_first` (must-lose order control: poll before issuing weights) | | 1.06 | LOSS, as designed |

The `x_first` control shows the rig resolves stage-order effects. The scale-rewrite fix (`sop`) does not move a
5-MFMA-per-wave GEMV; its place is `down` / `ug` / `irq` in K2.

### 6.4 Served smoke (integration, not a speed claim)

Job p2s1 (n07-17, 2 GPUs, `jobs/p2_srv_job.sh` + `p2/run_arms_p2.sh`): TP2, DSpark 5, mono on, the ATOM-dsv41 tree,
fresh server per arm. Three arms: FlyDSL, its A-vs-A twin, and `ATOM_DSV41_RK_EXTERN=wqkv=<the 16 rocKE
functions>`. CONC 1 / 2 / 4 x 16 prompts, then GSM8K 3-shot, 64 questions, CONC 4.

- **The hook serves.** Each rank logged the linked functions for target verify K1 builds at 6 rows (n6), the
  token-parallel 12 / 18 / 24-row builds (own rows n6 / n9 / n12), and the draft K1 at 5 / 10 / 15 rows. There were
  no tracebacks, and the mono stayed on for decode and draft.
- **Accuracy.** GSM8K-64 flexible-extract: FlyDSL 0.906, twin 0.922, rocKE 0.938.
  - The served path is not run-to-run deterministic: the A-vs-A twins return the same response text on only 0.28 of
    the questions. The rocKE arm matches FlyDSL on 0.30 and the twin on 0.31.
  - Per-question score disagreements, strict-match / flexible:
    - rocKE vs FlyDSL: 7 / 2;
    - rocKE vs twin: 4 / 1;
    - FlyDSL vs twin: 9 / 3.

  All are inside the A-vs-A floor, as expected for a stage that is bitwise in the kernel. This is a smoke run, not
  the accuracy gate.
- **Speed.** Per-request decode tok/s, rocKE / FlyDSL: CONC 1 0.97, CONC 2 1.00, CONC 4 1.00. The twin / FlyDSL
  spread is 0.96-1.06, which is larger than the expected effect (under 0.5 % of a step): no end-to-end claim.

## 7. Next steps (ranked)

1. **K2 per-layer-kind builds** (a selecting and a non-selecting `LayerPostBuild`). This is an ATOM-side knob with no
   numerics change. It attacks the 790-1209 SGPR spills on about 30 of 40 layers. A two-GPU K2 in-situ rig is needed
   to time it: K2 pushes to peers.
2. **Extend the GEMV port to `wq_b`, `wo_a`, `wo_b` and `shared`.** Reuse the same chain, with per-stage epilogues.
   Then search `sop` / `ra` / holds on the K2 GEMVs, where MFMAs per wave are many.
3. **NUMA dual copies + holds on K1 / K2** as ATOM-dsv41 schedule knobs, ported from ATOM-truemono `numa.py` /
   `sched.py`.
4. **Epoch tags** to drop `step_begin`'s zero and fence. Then a K2b fused MoE port from the Qwen template.
