I read every source you listed and wrote the report below. Two helper agents summarised the top-level rocKE docs and the hiptensor-rocke-rfc tuning docs; I checked their figures against the ledgers. All numbers are ratios: below 1 means faster for time, above 1 means faster for tok/s.

One point up front: the mono that Qwen3.8 was measured against launches **one kernel per layer**. FlyTape is a persistent megakernel, so it has already removed the per-layer launch boundary. Whatever beats FlyTape has to come from the in-kernel levers in §1.1–1.5, not from removing launches (see §1.6). I did not read the FlyTape source; that comparison is the next step.

---

# 1. Technique catalogue

Sources: `hackathon-prep/reports/qwen38-rocke/` (call it `R/`): `MONO_IMPROVEMENTS_SHEET.md`, `E2E_COMPOSED.md`, `BUBBLES.md`, `ledger/{schedule,fused-moe,gdn-b3,head,hybrid,moe,layerkernel}.md`, `MOE_FUSED_DESIGN.md`, `ROADMAP.md`.

**The end-to-end result** (`R/E2E_COMPOSED.md`, run "e2ecomp2"; MTP K=3, gsm8k, one request, 6 shuffled pairs). Every arm is bitwise-equal to mono at layer level, and acceptance length did not change (within 0.5%).

| Cumulative step | Verify-round time vs mono | Per-request tok/s vs mono | Increment |
|---|---|---|---|
| NUMA mailboxes + stage holds (`ATOM_Q38_SCHED=best`) | 0.930 | 1.079 | 0.930 |
| + fused expert-major MoE (rocKE) | 0.877 | 1.139 | 0.943 |
| + B3 gated delta rule (rocKE) | 0.866 | 1.159 | 0.988 |
| + lm_head MXFP4 GEMV + fused argmax (rocKE) | **0.856** | **1.170** | 0.987 |

At K=0 (no MTP) the stack reaches 0.868 per token (1.151x tok/s). Almost all of that is the NUMA lever (0.881). At rows 1 the fused MoE adds nothing on top of NUMA (0.999 median), and B3 adds 0.989.

**Where the time went before any change** (`R/GAP_TABLE.md`, `R/BUBBLES.md`). This tells you what kind of kernel you are fighting:
- The round was 0.79 target layer kernels, 0.11–0.14 non-layer work, 0.02 launch boundaries.
- The target pass ran at 2.31x its issued-bytes floor and 3.37x its unique-bytes floor. **It is bound by synchronisation and latency, not bandwidth.**
- CTAs spent 0.60–0.75 of their time polling.
- Mailbox hand-off hops were 0.26–0.55 of the critical path, and hop latency did not depend on where producer and consumer ran.
- Every stage sits on the critical path, because all hand-offs are all-to-all.

## 1.1 Scheduling and synchronisation (from the schedule track; applies to any persistent kernel)

| Technique | Mechanism | Result | Status |
|---|---|---|---|
| **NUMA dual-copy mailboxes** (`numa=1`) | Every mailbox is written to two copies, one homed on each IOD; consumers poll the local copy. Which IOD owns an address is address bit 13 XOR a per-2-MiB-page constant. It is calibrated at init by a ping-pong test (two CTAs on one XCD, 48 tagged round trips per 8 KiB block). | A remote-homed mailbox costs **1.48–1.50x** the round trip of a local one. Layer: delta r1 0.889, delta r4s 0.917–0.923, qsa r1 0.917, qsa r4s 0.949. Served round 0.930–0.934. Ablations: the layout alone gives 1.004–1.010 and always polling the remote copy gives 0.987–1.008, so **the whole gain is local polling**. | Largest single lever; applied everywhere. In the rocKE toy framework: 0.80 hop-only, 0.95 with weights. |
| **Stage-entry holds** (`hold.<stage>=N`, `s_sleep` at task entry) | Causal profiling found negative entry slopes: starting an off-critical stage early steals HBM/L2 from the critical stage. | `hold.b1=32` 0.982 (r1); `hold.m4=64` 0.991; `hold.m3=128` 0.989 (r4s); `head_sleep=128` 0.993–0.997 (`head_sleep=0` loses). Best sets on top of NUMA: delta r1 **0.858–0.860**, delta r4s 0.907–0.913, qsa r4s 0.939–0.942. Combinations are sub-additive. | Applied per variant (`sched.BEST`). |
| Kernarg preload (`kernarg_preload=8`) | Removes the kernarg `lgkmcnt(0)` wait, which was 0.44 of all stalls at stage entry. | Toy chain 0.94 hop-only / 0.98 with weights. Launch-to-first-stage ramp 0.47 / 0.41. | Applied in the rocKE framework. Python engine only. Mono doesn't use it. |
| DPP wave sums instead of `ds_swizzle` (`reduce=dpp`) | Removes LDS round trips. | Toy 0.991 / 0.938. | Applied. |
| Compiler-visible tagged poll (`mailbox.impl=ir`, `tagged_poll`) | The poll is an IR loop instead of an asm loop, so the compiler counts the waits. | Toy 0.983 hop-only (3/3 replicates), 1.00 with weights. B3 0.9967 / 0.9895. In the hybrid path, parity. | Applied in the framework and B3. |
| Prefetch while waiting + early weight issue (`prefetch=l2_touch`, `early=entry`) | Next level's weights are issued before the poll. | Single layer: depth 8 −5%, depths 2–4 +2–7%. 2-layer stack: depth 8 −12%. | Applied per shape (`best_spec`). |

**Schedule levers that lost** (kept as off knobs; don't spend time on them again):

| Lever | Result |
|---|---|
| `s_sleep` between polls (`pause`) | 1.002–1.010 |
| Two poll loads in flight (`pipe`) | 1.003–1.032; double the poll traffic costs more than earlier detection gains |
| `s_setprio` | no effect: one CTA per CU leaves nothing to arbitrate |
| Simulator-chosen placement (`place.m1`) | predicted 0.956, measured **1.041** |
| Dynamic ticket / ready-queue task assignment (`dyn`) | 1.012–1.079, and +10–17% on the best profile. The first CTAs to arrive start their weight streams earliest, which is exactly the contention the holds remove. |
| Last-arriver hop | 1.6–2.3x |
| `reduce_task` hop | 1.07–1.22; only wins without NUMA copies |
| `xcd_blocked` / sequential placement | 1.02–1.39 |
| B3 state prefetch by B1's idle waves | +0.2–1.0% |
| Scalar step-word load (`step_load=smem`) | 7–10x slower |

## 1.2 Fused expert-major MoE

Design in `R/MOE_FUSED_DESIGN.md`, results in `R/ledger/fused-moe.md`. Code: `copy-rocm-lib/.../qwen38_mono/moe_fused.py`, `stage_moe_fused.py`.

**Design:**
- **Slots:** slot 0 is the shared expert; slots 1..U are the union of the rows' top-10, sorted by id. Deduplicated rows go in as MFMA A rows.
- **Work units:** gate_up units are slot x 32-column block. The block is 32 so each MXFP8 amax/E8M0 block stays inside one CTA and no cross-CTA reduction is needed. Down units are slot x 16 hidden columns.
- **Placement:** a closed-form balanced byte line per IOD (`map=line`), so per-CTA bytes are equal to within one down unit, with no tables and one barrier.
- **Exchange:** the intermediate stays inside one IOD.
- **Combine:** rank-ordered, so results stay bitwise. There is no K-split and there are no atomics.

**Results:**
- MoE vs mono MoE: r4s **0.780** at served-like correlated rows (row-corr 0.7, U≈28), 0.836–0.844 at uncorrelated rows, r1 0.81–0.85.
- Layer: r4s 0.844, r1 0.875.
- Against a mono that also has NUMA: 0.83–0.89 (rows 4), 0.90–0.92 (rows 1).
- PMC FETCH_SIZE fused/mono is 0.875 at r4s (the dedupe). After routing the stream runs within about 10% of full-grid HBM bandwidth.

**What won:**
- Down weights issued **after** the gate_up epilogue. The first build with design defaults hit **1.18 because of VGPR spills**; this fixed it.
- Invalid chains issue no loads; ATT showed the issue phase was 75–85% load-issue stalls.
- `lg_copy=numa2`: 0.935 vs 0.948.
- Building the bodies against the NUMA mailbox layout (`numa=1`): 0.852 vs 0.948.
- `line_order=router_dn`: about 0.995.
- Rows 1 only: arithmetic slots, a 32-CTA shared group, `ys_home=numa2`.

**What lost:**
- Issuing down weights during the gate_up stream: the exchange waits on gate_up.
- `combine=last` (last arriver): 1.08–1.11.
- `combine=sh`: 1.05.
- `spec=prev`, i.e. prefetching last step's experts into MALL: **1.20–1.26 even with a perfect oracle**, because each CU's stream is already bandwidth-bound.
- `dn_issue=lds`, `slots=wave`, polling throttles, `dn_inflight=2`.

**What's left:** the router plus routing window. A null-bytes build (W0) showed fixed overhead at 0.625 (r4s) / 0.845 (r1) of mono's MoE, about 3x the design model, and routing ends at about 0.26–0.28 of mono's MoE. **For a 512-expert MoE, measure fixed overhead (F) first with a null-bytes build before designing for bytes.**

## 1.3 GDN B3 recurrence

`R/ledger/gdn-b3.md`; code in `stage_b3.py`.

| Lever | Result |
|---|---|
| `state_nt`: non-temporal stores of the f32 state snapshot | Largest B3 lever: r4s 0.9695. It stops the snapshot from evicting L2/MALL lines the following M4/M5 streams need. |
| `split=2`: fewer slices per head, so fewer exchange hops | r1 0.9875 |
| `pre=dma`: state and b/a weights DMA'd into LDS during B1 | r1 0.9975; r3/r4 don't fit in LDS |
| `mb_impl=ir` | 0.9967 / 0.9895 |
| `head_map=xcd` | r4s 0.995 |

Tuned sets, delta layer vs mono: r1 0.953–0.957, r2s 0.940–0.942, r4s 0.965–0.967. On top of NUMA: 0.963–0.970. Served alone at k=3: 0.987.

Lost: direct core put (1.025–1.035), `stores=end` (1.03–1.04), redundant norms, full barrier, swizzle exchange.

Mono B3 at r2s spent 39% of its time in `s_barrier` and 15% in `vmcnt` waits.

## 1.4 lm_head: MXFP4 GEMV with fused argmax

`R/ledger/head.md`; code in `mx_decode_gemv.py`.
- Mono fetched **1.23x** the head's bytes and spent 45–47% of its time waiting at `vmcnt(0)` plus barrier. rocKE fetches 1.00x and is streaming 0.87 of the time.
- Persistent CTAs, panels of 1 x 16 columns x K, x held in LDS, mono's MFMA order (bitwise logits): r1 0.827–0.840.
- **Tail balancing with `ctas=243`** (15520 panels over 972 wave slots): argmax builds 0.772–0.783.
- One-launch argmax fold using an epoch counter: applied.
- Serpentine column order for MALL reuse across draft heads: lost.

## 1.5 Hybrid path: linking rocKE stage functions into the FlyDSL kernel

`R/HYBRID.md`, `R/ledger/hybrid.md`.
- rocKE emits a stage as an LLVM device function (`.ll`/`.bc`) through `device_function.py` and `stage_fn.py`. FlyDSL links it in, and `ATOM_Q38_EXTERN_STAGES` turns it on per stage.
- On its own this is parity (Q2: 0.9997 / 1.0021). It is plumbing for in-situ A/B tests, not a speed lever.
- **Use `alwaysinline`.** A real call costs 9.2x the cycles of the inlined code, and LLVM's heuristic turned it into a real call at r4s.
- **Mark device functions `convergent`.** Without it, the stamped build hung (LLVM threaded the call between `tid==0` blocks).
- **MoE split port at mono's structure only reached parity** (`R/ledger/moe.md`: 0.98–1.01, some cells lose). **Porting the same structure doesn't pay; restructuring does** (fused expert-major).

## 1.6 Cross-layer stacking (most relevant to FlyTape)

- **Toy chain:** a FlyDSL 2-layer twin vs mono+NUMA gave 0.75 / 0.81. rocKE stacked vs FlyDSL stacked gave 0.96–0.99 / 0.95–0.96, so **rocKE's framework edge on equal structure is a few percent at most.**
- **Real mono layers stacked two per launch:** **1.024–1.040 (loss)**, and 1.03–1.09 on top of the best profile.
  - The fences cost about 1.4%.
  - The rest is the next layer's A1/A2 waiting on all 160 M5 tiles of the previous layer, plus the next layer's early weight streams contending with the previous layer's tail (the holds mechanism in reverse).
- **What this means for FlyTape:**
  - A persistent kernel has to handle write-after-read hazards across layers with layer-parity double-buffered scratch and mailboxes.
  - Next-layer prefetch should be issued after the CTA's last own load of the current layer. An earlier L2 touch gets evicted by the MoE stream (more than 100 MB against about 4 MB of L2 per XCD).
  - Check whether FlyTape uses NUMA-homed mailboxes and contention holds. If it does not, those are the first levers to bring over (0.86–0.95 per layer above).

## 1.7 Other facts worth knowing

- Bandwidth saturates at about 192 CTAs (`probes/bw_curve.py`). Mono's MoE ran at 110/160 CTAs at rows 1.
- Scaled MFMA on gfx950: 16x16x128 and 32x32x64 (fp8 x fp4, E8M0) have equal peak. 16x16x128 suits decode GEMVs with 16 or fewer rows. Matching mono bit-for-bit requires the 16x16x128 op with opsel bytes, which was a Python-only opcode (`ROADMAP.md` §3.7).
- MXFP8 requantization rule: amax·fl(1/448), ceil the exponent, clamp to [1, 254], RNE e4m3fn (`helpers/mx_quant.py`).
- One CTA per CU, which comes from the LDS size, is the co-residency argument that lets 256 persistent CTAs spin without a cooperative launch.
- Spec decode / MTP: GDN state rollback needs per-row snapshots or replay records (`R/SPEC_DECODE_ROCKE.md`). `snapshot=last` plus replay could save about 3.4e8 B per round, but needs an ATOM state-contract change.

# 2. Search and tuning heuristics, and timing methodology

**Finding where the bubbles are, before choosing levers** (`R/SCHEDULE_BRIEF.md`, `R/BUBBLES.md`, `hackathon-prep/qwen38/sched/`):
1. **Blame graph** (`blame.py`): producers stamp `s_memrealtime` plus their CTA id into a side copy of each mailbox, and consumers record the last arriver. This gives the critical path, per-task slack and stragglers. It slows the kernel 1.76–2.08x, so use it for structure only, never timing.
2. **Coz-style causal profile** (`coz.py`, `ATOM_Q38_DELAY=stage:entry|post=units`): measure d(layer time)/d(delay) per stage. A slope near 1 means the stage is critical. **A negative entry slope means contention and directly tells you which stage to hold.** This is what found the holds.
3. **Ping-pong home-IOD calibration**: this is what found the NUMA lever.
4. **Poll-spin heat maps** (CTA x time): `heatpng.py`.
5. **The discrete-event simulator was not a valid fitness proxy.** Don't let it rank levers.
6. **WaveScope ATT before and after every lever**:
   - Script: `rocke/platform/dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py`, with `ROCKE_DEBUG_LOC=1` and `llvm-dwarfdump` (from the container's `_rocm_sdk_devel`) on PATH so inlined frames resolve.
   - Split per stage at the stamps (`att_stage.py --auto`, `att_phases.py`).
   - PMC FETCH_SIZE, TCC_HIT and TCC_MISS must be collected in separate passes.
   - ATT sees one dispatch per CU and is noisier than an A/B run. **The A/B decides.**

**Search loop** (`R/ledger/schedule.md` method; rocKE-Search doc):
- Hill-climb one lever at a time on top of the current best.
- Lookback: exhaustively sweep every subset of the retained winners, then redo the sweep on top of any new structural base.
- **Signed lookback** (`climb_b3.py --lookback signed`):
  - The grid holds every applied lever plus the others ranked by mean signed gain, capped at about 1200 cells.
  - It excludes values that lost by more than 1.5% in every climb.
  - The top 6 are re-timed for 30 rounds.
- Losers stay registered as default-off knobs whose off state is byte-identical ISA; the knob shows up in the kernel_name suffix only when set. Losers can win in combination: the fused MoE went from 1.18 to about 0.94 once spills were fixed.
- rocKE tune engine (`hiptensor-rocke-rfc/tune-engine/{LIVE,OFFLINE}.md`):
  - A portfolio of hill climbing, GP-EI and SHADE DE, allocated by Thompson sampling.
  - Selector: LightGBM per (op, dtype, arch) predicting log(1+TFLOP/s). It picks within about 98–99% of the best config.
  - **Transfer pruning: warm-start from the two nearest shapes in log-M/N/K.** The first evaluation lands at 0.82–0.97 of the oracle instead of 0.29–0.33 cold. Cross-arch transfer is poor for GEMM/conv (about 0.5).
  - Warm search vs the dispatcher pick at 128 evaluations: 1.035 [1.019, 1.056].
- Per your memory rules: sweeps are model-guided, not brute-force grids; every knob value gets built; iglp1 candidates compile in a fresh process with a job-local comgr cache.

**Timing discipline. Each rule here exists because a false win got through without it:**
- In-situ HIP-graph A/B with 16 cold weight copies (at least 4x MALL), 96 launches, 12–30 rounds.
- **Arm order shuffled every round.** The ratio is taken per round against that round's mono.
- **Two controls:** mono vs mono (a second graph) and a **recompiled mono**. A separately compiled copy alone moves a layer 0.1–0.3%.
- **Win rule:** median below both controls' minima by 0.002, **and** at least 3/4 of rounds below 1. Then confirm on a fresh seed and allocation.
- Hybrid stage swaps must also show `calls == 0` in the ISA.
- **Stamps bias timing** (0.988 stamped vs 0.996–1.008 unstamped). Time only unstamped builds.
- **Fixed arm order plus best-of-N produced a false 0.993**, which was withdrawn.
- Numerics are bitwise before any timing:
  - mailboxes, layer output and every state pool;
  - with NUMA, copy 1 must equal copy 0;
  - a mutant must differ;
  - 12 consecutive launches without zeroing scratch;
  - forced U values (10, 30, 40).
- GEMM finals (hiptensor):
  - all arms in one process, 6–8 interleaved rounds of 1–1.5 s windows;
  - a 1 GiB read flush for cold reps;
  - sysfs SCLK/power sampled continuously; drop rounds with `sclk_min < 1000 MHz`;
  - Hodges-Lehmann estimate with an exact Wilcoxon interval;
  - kernels under 0.3 ms are timed one call per event pair behind a sleep spacer.
- End-to-end:
  - one server at a time;
  - a seeded permutation of arms rotated per pair, 6 pairs;
  - an A-vs-A control (spread about 0.3–1.5%);
  - **served text from unmodified mono is not run-to-run deterministic**, so prove numerics at layer level.

# 3. Reusable infrastructure

**Code:**
- **The rocKE monokernel sources live in `C:/Users/vanantha/Documents/dev/copy-rocm-lib`**, branch `users/vanantha/rocke-monokernels`, commit `89209ac7665` ("Q38 monokernel work"):
  - `.../rocke/library/kernels/gfx950/qwen38_mono/`: `moe_fused.py`, `stage_moe_fused.py`, `moe.py`, `stage_b3.py`, `stage_q2.py`, `framework.py`, `toy_chain.py`, `mailbox_ir.py`, `device_function.py`, `stage_fn.py`, `stage_adapt.py`, `abi.py`, `atom_layout.py`, `mono_layout.py`, `stamps.py`
  - `.../kernels/gfx950/mx_decode_gemv.py` (the lm_head)
  - helpers `platform/python/rocke/helpers/{mailbox,device_ops,task_schedule,lds_arena,mx_quant}.py`
- **`dev/rocm-libraries` is currently on `users/vanantha/rocke/hiptensor-support`.** Its `qwen38_mono/` has only `persistent.py` and `task_schedule.py`. Your memory note says Q38 work goes into `dev/rocm-libraries` on `rocke-monokernels`, so check which tree is canonical before editing.
- ATOM side: `dev/ATOM` branch `qwen38-megakernels`, baseline 352229d6. Patch: `R/atom-qwen38-rocke.patch`. Knobs:
  - `ATOM_Q38_SCHED=best|numa=1,...`
  - `ATOM_Q38_EXTERN_STAGES`, `ATOM_Q38_EXTERN_MISSING=mono`
  - `ATOM_Q38_ROCKE_HEAD=1`
  - `ATOM_Q38_STAMPS`, `BLAME`, `SPIN`, `DELAY`, `ROUND`, `DUMP`

**Harness** (`hackathon-prep/qwen38/`; see its `README.md`):
- `harness/`: `rig.py` (MonoRig), `bench_layer.py`, `bench_stack.py`, `timeline.py`, `bytes_model.py`, `gap_table.py`, `round_decomp.py`, `pmc_compare.py`, `golden.py`
- `tests/`: Tier A/B/C/E/F tests
- `probes/bw_curve.py`
- `sched/`: `ab_sched.py`, `sched_rig.py`, `blame.py`, `coz.py`, `att_compare.py`, `stack_rig.py`, plus `jobs/j_*.sh`
- `moe-fused/`: `fused_ab.py` (check/bench/stamps/run), `att_phases.py`, `line_probe.py`, plus `jobs/att.sh` (a ready-made ATT + PMC recipe)
- `gdn-b3/`, `head/`, `hybrid/`, `e2ecomp2/` (`e2e_comp2.sh`, `summarize2.py`), `common/server.sh`, `common/caches.sh`
- **`quick/quick_tps.sh`**: back-to-back mono / all / mono / all tok/s at MTP K=3 on gsm8k (`PROMPTS=32`). It generates the fused MoE functions with `python -m kernels.gfx950.qwen38_mono.stage_moe_fused --out /job/fns --knobs best --flavor llvm23`. This is the default quick progress metric.

**Cluster:**
- `hackathon-prep/tools/q38/q38_job.sh --tag T --gpus 1..7 --mode venv|container --stage rocke-x --stage atom-x --stage h-harness -- '<cmd>'`. Pack stages first with `q38_sync.sh`.
- rocke partition nodes: `smci355-ccs-aus-n01-09` and `smci355-ccs-aus-n07-17`. Container jobs only go to nodes seeded by `q38/seed.sh` (markers in `~/rocke-sdpa-claude/q38/seeded/`). GPU budget is `Q38_MAX_GPUS` (default 14).
- Jobs run in node-local `/tmp/vanantha-q38/jobs/<tag>-<id>`. Results go to `~/rocke-sdpa-claude/q38/results/<tag>/`, logs to `hackathon-prep/logs/q38/`.
- **Container mode:**
  - ROCm 10.0 image `/tmp/vanantha-q38/images/rocm-pytorch-rocm10.sqsh` (Austin copy in `~/rocke-sdpa-claude/q38/images/`)
  - ATOM repro venv `/work/venv`, model at `/model` (`Qwen3.8-Flash-Next-MXFP4`), ATOM stage at `/atom`
  - `ROCKE_BACKEND=python`, comgr flavor llvm23
- **Venv mode:** Austin venv `~/rocke-sdpa-claude/venv` (torch 2.12 + ROCm 7.2, llvm22 comgr).

**FlyDSL 0.3.4.1: yes, already installed, but only in the container.** `logs/q38/results/harness-t0s3/t0/toolchain.txt` shows `flydsl 0.3.4.1`, and `logs/q38/results/integ-w1-atom1/run/pip-freeze.txt` shows `flydsl==0.3.4.1`, both in `/work/venv` inside the ROCm 10 container on the seeded rocke nodes. The alola skill reference (`reference/gfx950-rocke-servers.md` §6) confirms Flash-Next builds with `flydsl==0.3.4.1` (aiter `c53e9b3f`), and a local worktree sits at `dev/FlyDSL-v0.3.4.1`. I found no evidence it is in the Austin venv used by venv-mode jobs. Qwen3.8-27B uses frontend `19418894` over wheel `0.4.0.dev963`. Check which FlyDSL FlyTape needs.

# 4. Pitfalls

1. **Byte-only work can't beat these kernels.** They are bound by hops and polling. Measure where the round goes (blame, Coz, ping-pong) before porting anything.
2. **Porting the existing structure gives parity.** The MoE split port and the Q2 swap both landed at parity. Wins came from restructuring (expert-major fusion) or from schedule levers.
3. **Contention is real.** Starting early, dynamic ticket assignment, extra prefetch streams and MALL speculation all lost. Holds won.
4. **Where a mailbox address is homed matters more than where producer and consumer run.**
5. **Toy results don't transfer.** The 2-layer stack gained 0.54–0.75 on the toy and lost 1.02–1.04 on real layers; the simulator's placement call was wrong by 8.5%.
6. **Correctness traps:**
   - A private tag `tag | 2^31` collides with launch tags from step 2^25 on; use `tag ^ 2^31`.
   - Rows-3 LDS overflow.
   - Liveness packing overlaid the argmax reduction buffer on x; mark it `exclusive=True`.
   - A toy LDS parity race under prefetch.
   - `fresh_ids=False` miscompiles hop-only chains at depth 3 or more.
   - Non-convergent device functions hang stamped builds.
   - A `pre=dma` B3 body linked without its prologue was accepted silently.
   - An inline-asm MFMA right after a VALU write to its operands needs `s_nop` on gfx950.
   - Never use `b.sync()`: its vmcnt drain kills in-flight weights. Use `s_barrier_bare(fence="compiler")`.
   - Scalar loads only for memory nothing writes during the launch.
   - Don't add attributes to mirrored ops; the C++ parser may drop them, e.g. a poll without sc1 spins forever.
7. **Toolchain:**
   - FlyDSL's cache key misses monkeypatches; use `FLYDSL_RUNTIME_ENABLE_CACHE=0`. FlyDSL also doesn't disk-cache extern-linked kernels.
   - iglp1 output depends on what else was compiled in the process, so compile fresh.
   - Kernarg preload, the IR mailbox and f64 sqrt are Python-engine only.
   - ATT decoder: an explicit `att_library_path` fails with "error 37".
   - In the ROCm 10 image, kineto records CPU events only and there is no `rocprof-attach`; use the offline driver (`t1/round_offline.sh`).
   - The extern-linked code object has no rocKE DWARF inline frames, so split ATT by phase stamps.
8. **Spills:** the mono layer already runs at 228–246 VGPRs. Check every new emitter for spills first; that is what caused the fused MoE's 1.18.
9. **Measurement hygiene:**
   - One gate per node.
   - Node-to-node differences exist: delta r4 was 0.984 on one node and 1.004 on the other.
   - The same set varies about 0.01 from job to job, so decide levers within one job.
   - Throttled nodes create false losses.
10. **Still open, and possible targets:**
    - Serial stages still on mono code: a1, m1, c1, b1, a2, the q-chain, PLE. About 0.4 of the round.
    - The fused MoE router window: about 0.07 of the round at rows 4.
    - Non-layer, out-of-graph work: 0.11–0.14 of the round. The draft step runs eagerly; meta and front could be folded into the kernel.
    - Grouped/wide builds (8–16 rows) still run mono's MoE and B3.
    - The step-counter wait is the top stall of the best toy build.