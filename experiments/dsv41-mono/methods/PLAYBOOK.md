# W15 playbook: rocKE methods applied to DeepSeek-V4.1-Flash decode (ATOM mono, gfx950, TP2)

W15 task PB, 2026-11-04. Ratios only in this tree; absolute numbers stay in
`hackathon-prep/private-perf/dsv41-mono/w15/`. Names: "ATOM mono" = ATOM's FlyDSL per-layer fused kernels (K1
`attn_pre`, K2 `layer_post` = K2a `attn_post` + K2b `moe`) served from FULL graphs; "ATOM native" = the regular
path (serves C >= 9 at TP2); "rocKE hybrid" = our work inside ATOM; "rocKE mono" = a pure persistent all-rocKE kernel.

What this file is:

- section 1 distils every method, search technique, lookback rule and WaveScope workflow from the rocKE docs and the
  Qwen3.8 work, with the tool, the entry point, how it was used and what it measured;
- section 2 maps the methods to the V4.1 stages and to the ATOM-dsv41 FlyDSL code;
- section 3 is the full knob surface: every existing W12-W14 knob plus every hard-coded performance choice in the
  kernels, each with its default, value set, numerics class and expected interactions;
- section 4 ranks the structural ideas the methods suggest that nobody has tried yet;
- section 5 is the climb protocol for W15.

Sources read (all paths from `dev/`):

| source | what it gave |
|---|---|
| `copy-rocm-lib/experiments/dsv41-mono/{PLAN,ANALYSIS,D0}.md`, `probes/P1.md`, `reverse/MONO-STAGES.md`, `wps/WP1-WP6.md`, `w14/T1,T2`, `wide/P4.md`, `tp1/TP1.md` | the V4.1 facts, every built knob and its verdict |
| `rocke/platform/dsl_docs/optimization/{optimization_runbook,runbook_mapping,measured_results,scheduler-policy}.md`, `utilities/tools/wavescope/README.md` | Step 0, the loop, the knob catalogue, the codegen levers, the probe tier, scheduler policy |
| `rocke/platform/dsl_docs/architecture/wavescope_integration.md`, `dsl_docs/autotune/overview.md` | the source-correlated ATT pipeline; the search engine (`rocke.dispatch.tuning.search`) |
| `rocKE-Search-Driven-Kernel-Optimization.md`, `rocKE-Qwen38-TrueMono.md` | the method as used on Qwen3.8, with its measured staircase and its corrected claims |
| `copy-rocm-lib/experiments/qwen38-true-mono/{search/GUIDE,search/INVENTORY,quick/README,FINAL-STACK,reverse/FLYDSL-STAGES,XCD-SCHEDULING,DESIGN}.md`, `research/{q38-learnings,rust-direct}.md`, `rust-track/{LEDGER,INTEGRATION}.md` | the tuner, the gates, the bubble tools, the Rust direct compiler and its per-kernel verdicts |
| rocKE code `library/kernels/gfx950/qwen38_mono/tune/` (`strategies.py`, `knobs.py`, `catalog.py`, `gate.py`, `tdb.py`), `dsv41_mono/` | the tuner API, the dsv41 rocKE stages and side kernels and their knob dataclasses |
| `copy-rocm-lib-rust/platform/dsl_docs/{runtime/direct_compiler.md,reference/env_flags.md}` | the direct compiler's entry points and limits |
| `ATOM-dsv41/atom/models/deepseek_v41/mono/` (`kernels/*.py`, `runner.py`, `draft_runner.py`, `config.py`, `*_plan.py`), `ATOM-dsv41/atom/mono/{plan/execution.py,device/sync.py,device/ops.py}`, `ATOM-dsv41/atom/models/deepseek_v41/rk/*.py` | every hard-coded performance choice of ATOM mono, every `ATOM_DSV41_RK_*` knob |
| `hackathon-prep/private-perf/dsv41-mono/raw/wp4/wp4g2.out` | graph-mode per-stage shares of K1 / K2 (shares only quoted here) |

---

## 0. Where V4.1 stands (the facts the climb starts from)

- **K2 is 0.64-0.80 of the target forward's GPU time** (graph mode, ANALYSIS section 6). Its critical path: routed MoE
  (ug + down) 0.44 (6 rows) to 0.61 (48 rows), at the byte floor for the experts touched; GEMVs (wo_a, wo_b, shared)
  0.13-0.19; attention (irq + score) 0.12-0.15; indexer 0.02-0.04; seams 0.11-0.19.
- **Per-stage time shares in graph mode** (stamped timeline builds, job wp4g2, both ranks agree within 0.01; stamps
  bias absolute time, so these shares rank stages, they never time a lever):

  | K2 stage | 6 rows | 12 | 24 | 48 |
  |---|---|---|---|---|
  | index (iscore + isel + iblk) | 0.04 | 0.03 | 0.03 | 0.02 |
  | attn_score | 0.05 | 0.04 | 0.04 | 0.03 |
  | irq | 0.10 | 0.09 | 0.09 | 0.09 |
  | wo_a | 0.07 | 0.08 | 0.05 | 0.07 |
  | wo_b | 0.06 | 0.06 | 0.07 | 0.05 |
  | slice | 0.03 | 0.02 | 0.02 | 0.02 |
  | norm | 0.03 | 0.05 | 0.04 | 0.03 |
  | xq | 0.03 | 0.03 | 0.02 | 0.02 |
  | router | 0.02 | 0.00 | 0.01 | 0.01 |
  | route | 0.03 | 0.03 | 0.02 | 0.01 |
  | shared | 0.06 | 0.07 | 0.04 | 0.03 |
  | gate | 0.04 | 0.03 | 0.00 | 0.01 |
  | ug | 0.25 | 0.27 | 0.33 | 0.34 |
  | down | 0.18 | 0.18 | 0.24 | 0.26 |
  | reduce | 0.01 | 0.01 | 0.01 | 0.00 |

  | K1 stage | 6 rows | 12 | 24 | 48 |
  |---|---|---|---|---|
  | slice | 0.20 | 0.19 | 0.18 | 0.20 |
  | norm | 0.14 | 0.17 | 0.14 | 0.12 |
  | wqkv_a | 0.20 | 0.17 | 0.22 | 0.18 |
  | gate | 0.01 | 0.00 | 0.00 | 0.07 |
  | qkv (q_norm, kv_norm, RoPE, FP8 QAT, window write) | 0.11 | 0.12 | 0.07 | 0.00 |
  | wq_b | 0.29 | 0.30 | 0.34 | 0.35 |
  | index (iw, iq, iquant) | 0.05 | 0.05 | 0.06 | 0.07 |

  K1 is about 0.21 of K2's time at 6 rows and 0.16 at 48. By layer kind, K2 time splits reuse (non-selecting) 0.74,
  full 0.10, reindex 0.10, window 0.06; so about 0.8 of K2 time is in layers that never run the indexer.
- **Statics** (P2): K2 spills 790 (6 rows) to 1209 (48 rows) SGPRs to VGPR lanes, 226-249 VGPRs, 149,520 B LDS; every
  scaled-MFMA GEMV rewrites its scale VGPR before nearly every MFMA (down at 48 rows: 103 of 126); every poll site
  is peeled (twice the loads and `vmcnt(0)` waits); MFMAs sit right after `lgkmcnt(0)` and chain on one accumulator.
  The draft K2 (5 rows, 128 experts / top-3, no indexer) spills 220 SGPRs.
- **Every solo lever so far ties** (K2SPLIT, NUMA, PAUSE, SCHED, TAIL, the CCL forms, WIDE / FOLD / SEAM, the byte
  cuts). The only consistent GPU-step win is the rocKE FP8 GEMV family `eA` at C1 (0.986-0.990 of the step), inside
  the served noise. Served wins: WP1 graph prefill x1.17-1.61 at C4-C32; batched prefill graphs +1-19 % more at C8-C64.
- **Measured bounds**: MALL-hit stream 1.11-1.14x the HBM floor rate (a prefetch re-reads the bytes, so it pays only
  in HBM-idle time); same-row tree verify +3.2-6.7 % accepted tokens; each extra verify row at C1 costs about
  4.7-5 % of the step; the TP2 exchange is latency-bound (`dup`: bytes on the link cost nothing; `lat`: added latency
  shows 1:1); `layer=1` (K1 + K2 in one launch with a grid hand-over) loses 1.054-1.064 at C1.
- **Nothing on V4.1 has been WaveScoped yet.** No ATT trace, no blame graph, no Coz slope exists for ATOM mono V4.1.
  Every W12-W14 lever was chosen from static ISA and Qwen analogy. This is the largest method gap (section 1.3).

---

## 1. The methods, distilled

Each entry: what it is, where it lives, how it was used, what it measured. "Q" = Qwen3.8 true-mono evidence,
"V" = V4.1 evidence.

### 1.1 Order of work

| method | entry point | rule | measured effect |
|---|---|---|---|
| **Step 0: exhaust the existing lever space first** | runbook "Step 0"; `utilities/tools/dsl_probes/probe_config_sweep.py`; the search engine | enumerate every implemented lever including the default-off ones, sweep the raw flags (not a preset menu), only then redesign | runbook: a +9-10 % default-off flag was invisible to the preset sweep for a whole loop; a "structural" gap was one mis-set `num_warps` |
| **The loop** | runbook "The Loop" | hypothesis, correctness, baseline, inspect IR/ISA, change one lever, re-verify, re-measure, explain by per-iteration ISA diff, keep or revert, record | - |
| **Reverse-engineer, then crush** (Q decision 16) | `hackathon-prep/qwen38/truemono/reverse/` (`rv_dump`, `rv_identity`, `rv_isa split`, `rv_ll`, `rv_diff`, `rv_catalogue`), V: `hackathon-prep/dsv41/p2/p2_dump.py`, `wp4r/h/{rv_isa,rv_common,wp4_dump}.py` | dump the competitor's final `.ll` and ISA per stage, reach parity by construction, then make every hard-coded choice a knob and let the search beat it. "1:1 is the floor of a port, never its goal" | Q: QSA port 21 / 21 `.ll` multisets equal, then search 0.947-0.965 per layer. V: `wqkv_a` port bitwise, stage 0.95-0.97, K1 0.984-0.989 |
| **Porting the same structure gives parity; restructuring wins** | q38-learnings 4.2 | wins came from structure (expert-major MoE) or schedule levers, not 1:1 ports | Q: MoE split port 0.98-1.01; fused expert-major MoE 0.874-0.887 of the mono MoE |
| **Learn from the loss, start from the competitor's winning structure** | TrueMono deck "MoE rebuild" | a losing kernel gets an ATT-guided restructure immediately | Q: LB MoE first design 1.05-1.29 of FlyTape (ATT: EXEC 10 %, WAIT 47 %, STALL 42 %; 618 / 700 down MFMAs chained, 298 SGPR spills) -> `deal=rr`, `dt=4`, `pcal`, `cmb=flow` -> 0.79-0.95 |
| **A static probe is faster than a sweep** | `dsl_probes/probe_{occupancy,intrinsic_counts,isa_inspect}.py`, `wp4r/h/isa_stats.py` | read resources and per-stage opcode histograms before timing anything; a change that does not move the per-iteration ISA did not change anything | runbook 17.4: a "register-P proxy" variant had identical hot-loop ISA and measured as noise |
| **Measure the bound with a prototype, never on paper** | V: `dsv41_mono/moe_floor.py` (bytes floor), `mall_probe.py`, T2's tree probe, WP5 `dup` / `lat` probes, `x_first` must-lose control | build a null-work or ideal-work variant of the stage and time it in the same job | V: MoE 1.09-1.31x its floor at wide rows; MALL hit 1.11-1.14x; exchange latency-bound; tree verify +3.2-6.7 % AL |

### 1.2 Finding the bubble (before choosing levers)

| tool | entry point | what it gives | Q / V evidence |
|---|---|---|---|
| **Stage stamps, graph mode** | ATOM `atom/mono/device/stamps.py` (`ATOM_MONO_TIMELINE` builds), `hackathon-prep/dsv41/wp4r/h/p1_htl.py` / `wp4_run.sh` phase `gtl` | per-CTA `s_memrealtime` after each stage; per-stage shares per width, rank, layer kind | V: the section-0 tables. **Stamps bias timing** (Q: 0.988 stamped vs 0.996-1.008 plain): shares only |
| **No-profiler host timeline** | `ATOM_DSV41_RK_HOSTTL=<dir>` (`rk/hosttl.py`), `KMARK=1` markers | host vs GPU per step without the API trace | V: inter-step GPU gap 0.0003-0.0011 of the step; host off the critical path. Q: the profiled trace overstated the boundary idle 3-8x |
| **HTL GPU decode step** | `wp4_run.sh` phase `htl` (`p1_htl.py`, `wp4_htl_cmp.py`) | fresh engine per arm-run, FULL graphs, closed-loop C1 / C2 / C4 / C8, GPU decode step per cell and rank | the W13 scoring metric; resolves about 0.5 % |
| **In-situ per-launch A/B** | ATOM `kernels/rk_bench.py` (`ATOM_DSV41_RK_BENCH*`) | one running engine, ATOM's build vs a variant per launch, interleaved | V: code-object lottery: a recompiled unchanged K1 moves about 1 % |
| **Blame graph** | Q: `hackathon-prep/qwen38/sched/blame.py` | producers stamp `s_memrealtime` + CTA id into a side copy of each mailbox, consumers record the last arriver: the critical path, per-task slack, stragglers | Q: hops 0.26-0.55 of the critical path. Slows the kernel 1.76-2.08x: structure only. **Not built for V4.1** |
| **Coz-style causal profile** | Q: `coz.py`, `ATOM_Q38_DELAY=stage:entry|post=units` | d(layer time) / d(delay) per stage; slope near 1 = critical; **a negative entry slope = contention: hold that stage** | Q: this found the holds (delta r1 with NUMA 0.858-0.860). V: `hold.<stage>=N` exists (an entry delay) but was only tried at fixed values, never as a slope sweep; post-delay points do not exist |
| **Poll-spin heat maps** | Q: `heatpng.py` (CTA x time) | where CTAs spin, per stage | Q: CTAs polled 0.60-0.75 of the time |
| **Ping-pong home-IOD calibration** | Q: P4 / P16; V: `kernels/rk_numa_cal.py` | which IOD homes an address (bit 13 XOR a per-2 MiB flag) | Q: remote-homed mailbox 1.48-1.50x the round trip; found the NUMA lever. V: NUMA=1 tie |
| **The simulator** | Q: DES of the layer | bounds only, never ranks | Q: predicted 0.956 for a placement, measured 1.041 |
| **Static ISA catalogue per stage** | `rv_isa.py split --registry markers.json` (P2 marker trampoline: 0.80-0.94 of instructions attributed directly; marked `.text` = plain `.text`) | per-stage MFMA / load / nt / sc1 / vmcnt(0) / barrier / poll / spill counts | V: MONO-STAGES section 2 |

### 1.3 WaveScope workflow (ATT + PMC, source-correlated) for ATOM mono V4.1

The rocKE pipeline (`wavescope_integration.md`): `ROCKE_DEBUG_LOC=1` on the building process -> `Op.loc` chains ->
`!DILocation` with `inlinedAt` -> DWARF in the HSACO -> `rocprofv3 --att` -> `emit_inline_frames.py` sidecar ->
WaveScope Source tab (`self` / `+ inlined`). One command: `capture_wavescope_trace.py -- python3 bench.py`
(default traces dispatches 2-3); PMC bundle: `capture_wavescope_pmc.py --trace-dir <dispatch dir>`.

What changes for V4.1, and how to do it:

1. **Attribution.** ATOM mono is FlyDSL, and the rocKE stages are extern-linked LLVM functions: "the extern-linked code
   object has no rocKE DWARF inline frames" (q38-learnings 4.7). Use P2's stage markers instead: the dump process
   wraps FlyDSL's location capture so every instruction's outermost frame is a marker line per stage
   (`p2_dump.py` / `wp4_dump.py --debug 1`); `rv_isa.py split` maps PCs to stages. The marked build's `.text` equals the
   plain build's (2 / 2 checked), so a trace of the **plain served build** is labelled by alignment
   (`rv_isa.py --labels-from marked.split.json`). For rocKE stage functions inside K1 / K2 the same labels apply
   (the stage marker encloses the call). Standalone rocKE kernels (`wide_gemm`, `router_topk`, `prefetch`, the CCL
   argmax, `mall_probe`) get the full `ROCKE_DEBUG_LOC` Source tab. Direct-compiled objects have no DWARF: align
   them to their COMGR twin's listing.
2. **One CU per trace, one CTA per CU.** ATT records one CU; K1 / K2 run 256 CTAs of 512 threads, one per CU, and a
   CTA's stage mix is fixed by `first_task` placement (section 3.2). CTA b lands on XCD b mod 8; which CU inside the
   XCD is not fixed. Capture several target CUs (at least one per XCD, plus the CUs that host the serial per-token
   stages) and identify the CTA from the stage sequence it ran (its task set is known per width). Build a per-CTA,
   per-stage ledger, not a per-kernel one.
3. **Pick the dispatch.** Filter the kernel by symbol (`atom::...` Itanium-mangled names: `kernel_symbol`), then by
   layer kind (reuse / window vs full / reindex: different code paths in one build) and width (6 / 24 / 48 rows =
   C1 / C4 / C8). Capture from the in-situ rigs (P2 `p2_rig.py` for K1; K2 needs a two-GPU rig because it pushes to
   peers, WP4 noted) or `rk_bench` inside an eager engine; FULL-graph served capture is the confirmation only.
4. **Read the state mix first** (WAIT = memory / dependency / poll; STALL = issue contention; IDLE = wave done or not
   started), then the hot instructions by stall (divide `code.json` columns 7-8 by `Hit`), then the stage. The bubble
   ledger per stage and width:

   | column | from |
   |---|---|
   | poll wait (s_sleep + poll-load `vmcnt`) | ATT, poll-loop PCs |
   | weight `vmcnt` wait before the first MFMA | ATT |
   | `lgkmcnt` waits before MFMAs (no read-ahead) | ATT |
   | barrier wait | ATT |
   | spill lane traffic (`v_writelane` / `v_readlane`) on the path | ATT, static counts |
   | MFMA busy, VALU busy (`MfmaUtil`, `VALUBusy`) | PMC |
   | HBM / L2 / MALL (`FETCH_SIZE`, `TCC_HIT`, `TCC_MISS`, `MemUnitStalled`) | PMC, **separate passes** |
   | LDS bank conflicts (`LDSBankConflict`) | PMC |
   | start skew of the CTA (first instruction vs kernel start) | ATT / stamps |

5. **Before and after every lever,** same CU set, same dispatch selection; regenerate the labels when the build
   changes (a stale sidecar attributes to the old layout). ATT is noisier than an A/B: **the A/B decides**, ATT
   explains.
6. **Tooling traps** (q38-learnings 4.7): an explicit `att_library_path` fails with "error 37"; in the ROCm 10 image
   kineto records CPU events only and `rocprof-attach` is missing (use an offline driver); `llvm-dwarfdump` from the
   container's `_rocm_sdk_devel` on PATH; PMC counter groups in separate passes.

Diagnostic signatures to look for (runbook 17.4 / 8.6):

- "compute-throttled by VALU / LDS plumbing": low `MfmaUtil`, dominant `VALUBusy`, `MemUnitStalled` near 0 ->
  byte levers are useless; fix the plumbing.
- "barrier-bound": no bank conflicts, no memory stall, MFMA idle at the rendezvous -> fewer barriers or more work
  per barrier; `iglp_opt` is exactly neutral there.
- "chain-bound": MFMAs back to back on one accumulator after `lgkmcnt(0)` -> read-ahead, independent accumulators
  (`dt`), scale packing (`sop`).
- "poll-bound": WAIT dominated by the poll loop on CTAs whose producer is late -> hop structure, placement, NUMA
  homing, holds on contending stages.

### 1.4 Levers become knobs (the knob contract)

| rule | where | detail |
|---|---|---|
| every idea ships as a default-off knob; knob off = byte-identical | Q GUIDE section 2; V: every `ATOM_DSV41_RK_*` | prove it: traced MLIR locally, `.pre.ll` + `.text` on the cluster (V: `wp4r/lt.py`, `wp5/local_trace_ccl2.py`) |
| build-time capture enters the build key and the kernel name | V: `rk_lev.py` (`_lv<hash>`), `rk_numa`, `rk_pause`, `rk_pfk1` | a cached build can never silently mean another config |
| an in-place gate per knob family + must-fail mutants | V: `ATOM_DSV41_RK_{LEV,K2SPLIT,CCL,NUMA,PREFETCH,WIDE,SEAM,FOLD,TAIL,TREE,VRAGGED,COLD,SHARD,DVOCAB}_CHECK / _MUTANT` | compare every byte a launch can write (26-75 MB a launch), scratch included |
| poison LDS between reference and candidate | V: `kernels/rk_poison.py`, `ATOM_DSV41_RK_LEV_POISON` | gfx950 keeps LDS across launches: a candidate that skips an LDS write can read the reference's data and pass (V: the `askip` no-op was caught only through its uncaught mutant) |
| catalogue registration | Q: `tune/catalog.py` `Knob(path, values, kind, gate=, numerics=, structural=)`; `Knob.sequence` for orders; `LeverStringKnobs` for env strings | values = every built value, losers included, the shipped default first; `numerics=TOLERANCE` sends a config to the accuracy ladder; `structural=True` re-opens the climb |
| a reject is "rejected on base X" | PLAN learning 3, Q decision 15 | nothing is ever permanently rejected; every reject is re-tested after each accepted move |

### 1.5 Search: hill climbing with lookback and the surrogate

| strategy | entry point | how it works | measured |
|---|---|---|---|
| **`phased` (Q default)** | `qwen38_mono/tune/strategies.py` `phased()` = `SignedLookback` then TPE | signed-lookback climb (one lever at a time, APPLY only beyond noise, solo gain recorded per base), then the combination lookback, then TPE on everything measured | best or within 0.01 of best gain in every replay row |
| **combination lookback** (Q decision 13) | `SignedLookback(lookback=True)` | pool = EVERY knob and level, losers included; pairwise covering screen (at most 1/4 of budget); GP on log time with constant + main + all pair + all triple terms, prior mean = solo gains + recorded pairs; EI over the product space densified around the best combinations; best EI + most uncertain points per round | live toy d8w8: picks beat the solo-winners-only combination by 5.4 % and 6.8 %, carrying 9 and 3 rejected levels; qsa r4s pick holds 9 solo losers, 0.6 % better |
| **re-test after every accepted move** (Q decision 15) | `SignedLookback.set_base`, `open_rejects`, revisit pass, `order_restarts` | model-promising rejects re-tested in front, others behind the untested levers or into the EI pool; a final revisit pass on the final base; restarts in other lever orders; **order itself is a knob** (`Knob.sequence`) | toy unit test: B loses alone and wins after A; the climb ends A-then-B |
| engine strategies | `rocke.dispatch.tuning.search`: `hill`, `de` (SHADE current-to-pbest/1), `jde`, `tpe`, `gp` (Matern-5/2, EI), `portfolio` (discounted Thompson bandit), `sh` / `hyperband` | hill: single moves, combination re-sweep of near-neutral moves, ordinal two-step escapes, lookback over centres, restarts from foreign finds | GEMM replay: search reaches the exhaustive answer for about 1 % of evaluations; Hyperband over a rep ladder did not pay (cheap rungs rank badly); warm starts matter more than strategy choice |
| budgets | GUIDE section 5 | exhaustive if the legal space fits; lookback climber at 16-32 evaluations; TPE / GP past about 64; a stage lever sweep is 16-48 evaluations per variant; a lookback round 2-4x the retained levers; reserve the confirmation (finalists + baseline + 2 controls, 12+ rounds) | Q: 128-evaluation searches never reached the hold levers on delta_ple: budget 256+ per cell |
| transfer | `tdb.Selector.top_k`, `TuningDB.warm_starts`, `GainLedger.prior / prior_pairs` | warm start from the same variant's record on a new base; across variants it did not help when optima differ in structural levers | toy: 32 warm evaluations matched the recorded optimum |
| W13's in-job tuner | `hackathon-prep/dsv41/wp4r/h/tune.py` | Bayesian linear regression on lever indicators + pairs, LCB, lookback re-tests, accept on two measurements | 23 lever values in 21 groups, 30 evaluations; result PENDING_R6D. **Simpler than `phased`**: W15 should drive the V4.1 levers through the `qwen38_mono/tune` engine with a V4.1 backend (section 5) |

Lessons that cost real time (Q):

- the first lookback pruned solo losers: review caught it, decision 13 forbids pruning;
- combinations are sub-additive and can change sign (delta r4s `spread.a1=2` 0.991 alone, worse with the pick);
- a climb that re-sweeps a whole lever after every accepted move starves the budget (r10 replay);
- in-situ wins over-predict serving (Q: schedule re-climb 0.972-0.992 per layer, 1.002 e2e); in-isolation wins can
  vanish end to end: only served numbers count for claims;
- toy results do not transfer (2-layer stack: toy 0.54-0.75, real 1.024-1.040).

### 1.6 Measurement discipline (each rule exists because a false win got through without it)

| level | rule | V / Q evidence |
|---|---|---|
| layer / kernel rig | seeded shuffled arm order every round; unstamped builds; A-vs-A second graph + recompiled control (V: `nop=N`); 16 cold weight copies (>= 4x MALL); 12-30 rounds; win = median below both controls' minima by 0.002 AND >= 3/4 rounds below 1 (`tune/gate.py win_rule`) | Q: fixed arm order + best-of-N produced a false 0.993 |
| code-object lottery | 4 separately loaded copies per arm; the worst copy's median clears the band | V: recompiled K1 moves about 1 %; Q: 0.1-0.3 % per layer |
| GPU step (HTL) | fresh engine per arm-run, rotated passes, nop control | V: twin 0.995-1.007 |
| served e2e | one server at a time, fresh server per arm-run, seeded rotated permutation, A-vs-A twins of hybrid AND comparator, medians over requests, 6+ pairs, WIN = median below the twins' minimum and >= 5/6 pairs below 1 (`gate.e2e_rule`); both nodes | V: served single-pass swings about ±9 % (WP6); Q: twins differ about 1 % |
| clocks / neighbours | sample SCLK / power by PCI BDF, drop throttled rounds; log every neighbour, at most one of ours | Q: throttled nodes create false losses |
| caches | READ flush before every replay; 1 GiB flush for cold reps | - |
| numerics before timing | bitwise in-place gate (or the T0-T5 tolerance ladder) in the same process; the first check of a session plants a must-fail mutant and stops if not seen | Q: a gate blind to first-launch faults at depth 8 (int32 factor 0 mod 2^32) |
| toolchain | container COMGR waves=8 can miscompile (alola 219): time prebuilt objects, verify bitwise after any toolchain change; iglp_opt(1) is compile-history-dependent: compile fresh with a job-local comgr cache; `FLYDSL_RUNTIME_ENABLE_CACHE=0` when monkeypatching | Q: P-L5 table withdrawn (miscompiled binaries) |
| accuracy | tolerance knobs: T2 teacher-forced KL budget, T4 GSM8K / HumanEval / RULER inside the A-vs-A spread, long-context caps >= 4096 | V: RULER "regression" was a 256-token cap artefact |

### 1.7 Code-generation levers (runbook + Qwen + P-L3), with V4.1 status

| lever | mechanism | Q / runbook evidence | V4.1 status |
|---|---|---|---|
| scale packing `sop` | the scaled MFMA picks its scale byte with `op_sel` from a packed dword; no VALU rewrite of the scale VGPR before each MFMA | P-L3: rewritten scale runs about 8x slower per MFMA; op_sel packing 8.5x on the same work at large batch; Q GEMV picks hold `asop` / `bsop` that lost alone 1.012-1.024 | `gsop`, `gsop2`, `usop`, `dsop` (ATOM stages), `sop` in `WqkvKnobs`: ties; `usop` loses at 24-48 rows |
| LDS read-ahead `ra` | a round's B operands read before its first MFMA | Q: MFMA chains 0.46-0.74 of FlyDSL's span with a tied accumulator | `gra`, `ura`, `ra`: ties; `uGs` (ug sop + ra) 0.989 at C1 |
| independent accumulators (`dt`) | break the serial accumulator chain | Q LB MoE: chains 69 -> 9 per kernel, 0.95-0.977 | ATOM `DOWN_CHAINS=2` in down only; not a knob (it changes fp32 association: tolerance class or its own reference) |
| tied accumulator / `mfma=asm` | stop the allocator splitting the accumulator | Q WP2b: 1:1 port lost 0.2-2.7 % from 91-114 extra wait states per chain | not needed so far (V ports are bitwise and faster) |
| partial waits | `vmcnt(N - k)` instead of `vmcnt(0)`; `SoftwarePipeline(overlap_vmcnt=True)` | runbook 8.7 | V: poll's `vmcnt(0)` waits for in-flight weights (loads return in order); `poll1` removes the peeled copy (tie) |
| compiler-visible poll | IR loop (`tagged_poll`), not asm | Q: toy 0.983 hop-only, B3 0.990-0.997 | the rocKE stages use `poll="ir"`; `poll=asm` tie |
| `nt` weight streams | non-temporal weights | Q: a main win of every rocKE stage | ATOM already streams `nt` (CM_NT): no headroom |
| fused DPP reductions | DPP instead of `ds_swizzle` / `s_nop`-padded chains | Q: Q4 had 338 of 547 `s_nop` in DPP chains; toy 0.938-0.991 | V gate Sinkhorn: 285 `s_nop` cycles (K1 gate) |
| fewer drains | no `b.sync()` (its vmcnt drain kills in-flight weights); `s_barrier_bare(fence="compiler")` | q38-learnings 4.6 | V: bulk publishes drain `vmcnt(0)` + barrier (indexer logits, normed rows) |
| scheduler strategy | `CodegenPolicy(scheduler_strategy=...)`: `max-ilp`, `max-memory-clause`, `iterative-ilp`, `iterative-minreg`, `iterative-maxocc` (LLVM `amdgpu-sched-strategy`) | runbook 12.1.M: bounded sweep; flags rarely close large gaps | **not tried on V4.1** (FlyDSL K1 / K2 builds and the linked stages) |
| `iglp_opt` / sched barriers | canned MFMA / ds interleave | helps only with `ds_write` traffic to interleave; neutral on direct-to-LDS and barrier-bound loops; compile-history dependent | not used by ATOM except `s_setprio` / `sched_barrier` in `index_score_fp4` |
| cache policy | `CACHE_ALL` for reuse, `CACHE_STREAM` (nt) for one-shot | runbook 8.6: +9 % for reuse-heavy GEMM | V: weights nt, mailboxes sc1, peers sc0 sc1 |
| chiplet / XCD swizzle | consecutive WGs on one XCD (`chiplet_wgm`, `chunk`) | runbook 17.7: the last 2.6 % on skinny decode GEMM | V: `xcd`, `xq` (ug XCD homing) ties; WIDE `order=xcd` |
| active-tile skip | CTAs of dead tiles exit early | runbook 17.6: 1.56-2.35x on MoE e2e | V: ATOM streams only the union U (already) |
| kernarg preload | removes the kernarg `lgkmcnt(0)` at entry | Q toy: 0.94 hop-only, ramp 0.41-0.47 | rocKE-framework only; not in FlyDSL |

### 1.8 Schedule levers (persistent kernels), with V4.1 status

| lever | Q evidence | V4.1 status (knob) |
|---|---|---|
| NUMA dual-copy mailboxes | largest Q lever: layer 0.889-0.950, served round 0.930-0.934 | `ATOM_DSV41_RK_NUMA=1` tie; `=4` / `=5` (homed single copy) lose 1.008-1.041 |
| stage-entry holds | found by Coz negative slopes; Q delta r1 0.858-0.860 with NUMA | `hold.<stage>=N`, `shold`, `qhold`: tie (0.995-1.001) at hand-picked values; never slope-guided |
| spread (more CTAs on a serial stage) | Q `spread.a1=10` 0.969-0.970 per r1 layer | not built for V4.1 (serial stages: K1 norm / gate / qkv, K2 norm / gate / route / xq, S tasks each) |
| prio (`s_setprio`) | Q: no effect with one CTA per CU (but it was in some picks: `prio.m1`, `prio.q2`) | not built |
| prepoll (first poll earlier) | Q QSA 0.95-0.97 | V: `poll1` only (tie) |
| prefetch while waiting / early weight issue | Q: depth-dependent, -5 % to +7 % | ATOM issues weights before the poll everywhere; `pfx` (wo_b before wo_a) loses 1.017-1.022 |
| dynamic ticket / ready queue | Q: LOSS 1.012-1.079 (early arrivers start streams: contention) | not built (do not repeat blindly) |
| poll pause | Q: `pause` 1.002-1.010 loss; `mailbox.pause=none` in one pick | `ATOM_DSV41_RK_PAUSE` tie |
| two polls in flight | Q: 1.003-1.032 loss | not built |
| epoch tags (no per-launch reset) | Q CCL: no per-launch reset | `CCL=epoch` tie (C1 0.993-0.998 with argmax) |
| per-kind builds | V-specific | `K2SPLIT=1` tie (resources of the noix build not yet recorded: measure) |
| one launch a layer (Path A element) | Q: persistent loop body 1.002-1.020 per layer | `LAYER=1` loss 1.054-1.064 |

### 1.9 The Rust direct compiler (rocKE IR -> HSACO, no LLVM)

- **What it is.** Crate `platform/rust/rocke-codeobj` (branch `users/vanantha/rocke/rust-compiler`, worktree
  `dev/copy-rocm-lib-rust`; the `rnr-v8` crate is in copy-rocm-lib). About 66k lines, byte-deterministic, median 27x
  faster to compile than COMGR (min 4.5x). gfx950 coverage: MFMA (plain, scaled f8f6f4, BF16x3), AGPR accumulators,
  LDS DMA, transposing LDS reads, fp8 / bf8 conversions, packed f32, cross-lane ops.
- **Entry points.** `ROCKE_CODEOBJ_BACKEND=comgr|direct` (or `code_object_backend=`); `ROCKE_CODEOBJ_DIRECT_LIB=<path>`
  (which crate build: the compiler version is itself a knob); `ROCKE_DIRECT_LEVERS=<a+b+..>` or
  `KernelDef.attrs["direct_levers"]` (per kernel, stripped before serialization); `ROCKE_DIRECT_STRICT=1` (a refusal
  is an error, for validation); `ROCKE_DIRECT_DEBUG` (stage switches for bisection); the serving guard
  `qwen38_mono/persist/rust_guard.py` + DB `persist/rust_backend_db.json` (`select(key)`: direct only where both nodes
  hold WIN, or TIE at ratio <= 1, for one lever set; `ROCKE_DIRECT_GUARD=1`).
- **Lever families** (all default-off): `dppcombine`, `dppcombinef`, `dppcombinei`, `dppmirror`, `ldspairs`,
  `ldspairsfar`, `ldsquads`, `ldsduos`, `ldsalias`, `litreuse`, `litshare`, `minmax`, `minmax3`, `nopause`, `selxor`,
  `subadd`, `bitop3`, `xorcmp`, `storewait`, `vmemfirst`, `loadfirst`, `loadorder`, `wgfence`, `nofence`, `scalesep`,
  `dsrebase`, `dsfold`, `addrfold`, `packfar`, `fmac`, `masksall`, `u16zext`, `i16vec`, `vmlat60`, `vmlat100`,
  `bufoffany`, `acqfence`. Named sets: B3, MoE r1, `Z`, `ZL`, `ZLv`, `ZLP`, `ZM`, `N6`, `N7`, `M1Nx`, `F`
  (INTEGRATION.md section 8).
- **Measured (Q, both nodes, direct / COMGR).** Wins: B3 all variants 0.940-0.990; fused MoE r1 0.968-0.976, r4
  0.993-0.996 (M1Nx); framework B3 0.915-0.949; 19 of 24 HC kernels 0.972-0.998; CCL `pair_os` 0.953-0.972; P4 `xl`
  hop probe 0.858-0.860; P5 scalar 0.939-0.944. Losses kept on COMGR: Q2 1.072-1.118 (poll shape and framework SALU,
  open), head 1.001-1.010, P5 kv 1.085-1.088, LB MoE parity 0.994-1.011. Root causes were all fixed by levers found
  in the ISA diff against COMGR (e.g. HC: LDS index constants not folded into DS offsets, per-load voffset chains,
  unpaired far reads). **Lever verdicts are base- and family-specific** (`nofence`: -1.6 % on MoE r4 at base plain,
  a loss at base M1N and +0.9 to +4.3 % on the head).
- **What it controls that COMGR does not:** polls as leaf ops expanded after isel (never hoisted; template
  `flag_wait_eq`); an exact per-register `s_waitcnt` scoreboard (`target/gfx9/waitcnt.rs`); `s_barrier_bare` without
  a drain; per-access cache bits (`cpol.rs`); fence-flagged instructions never crossed by the scheduler; spills that
  never cross an EXEC join (AGPR, then free LDS, then scratch); deterministic, history-free output. To build:
  `tagged_poll` (2-3 d), register pinning of tags / bases / NUMA offsets across stages (3-5 d), kernarg preload,
  branch relaxation (blocker above about ±128 KiB), a PC -> IR line map for WaveScope, counted `vmcnt` for "poll first,
  then weights", scalar-memory polls, waits per IR op as a knob.
- **Limits that shape W15.** One kernel per code object; `tile.inline_asm` only the empty pin and three `ds_read`
  forms; no DWARF; no branch relaxation; unknown op names raise (`ROCKE_DIRECT_PARSE`, no fallback). **It cannot
  compile a device function that is linked into a FlyDSL kernel**, so the WP4 / P2 rocKE stage functions are out of
  its reach (WP4 4.7). It applies to every standalone rocKE kernel now (wide GEMM, router fold, quant seam, prefetch /
  shadow, the CCL argmax merge, `mall_probe`, `moe_floor`) and to any future whole-K1 / whole-K2 rocKE build.

### 1.10 TP2 exchange and coherence rules (V + Q facts)

- Coherence: `buffer_wbl2 sc1` on the writer, `buffer_inv sc1` on the reader; never scalar-reread in-launch writes
  (98.4 % stale). Cost < 0.001 of a round: a gate, not a lever.
- Tagged pairs beat graph boundaries and fences at <= 4 rows (fenced 1.16-1.80); 8 stripes per XCD for counters;
  per-XCD residency ticket; abort on timeout.
- V: in-kernel push + rank-ordered fp32 sum is bit-identical across ranks; the exchange is latency-bound (`dup` tie,
  `lat` exposed 1:1); `lamport` (half bytes) tie; quantized wide all-reduces (`arfp8`) 3-10 % of the decode step at
  C32 / C64 (tolerance).
- V: a side kernel resident beside the persistent K1 / K2 delays 96-128 of their 256 CTAs by a large fraction of a
  layer; a full-grid side kernel hangs graph replay; HIP FULL-graph replay runs a forked branch serially (T1).

---

## 2. Method -> V4.1 stage map

Shares are the section-0 graph-mode shares (of K1 or K2). "B" = bitwise by construction, "B*" = bitwise but needs
the in-place gate, "T" = tolerance class. Code paths are under `ATOM-dsv41/atom/models/deepseek_v41/mono/`.

### 2.1 K1 `attn_pre` (about 0.17-0.21 of K1 + K2)

| stage | ATOM code | share (6 / 48 rows) | bound class (static evidence) | methods that apply | built so far |
|---|---|---|---|---|---|
| slice (160 tasks x 32 cols x 4 streams) | `kernels/attn_pre.py stage_slice`, `slice_fn_loads`, `SLICE_FN_EARLY_MAX=24` | 0.20 / 0.20 | f32 16x16x4 mix MFMAs, owed post into residual, partials to PMIX; 160 tasks on CTAs 0-159 | rocKE HC port (Q WP2a pattern, 0.97-0.99 in situ), spread, XCD-affine partials, Coz | none |
| norm (S tasks) | `stage_norm` (`NORM0=168`) | 0.14 / 0.12 | serial per-token: polls 160 partials (fan-in hop), RMSNorm + `quantize_fp8` | spread, hierarchical / XCD-local fan-in, fuse with slice tail, NUMA homing, holds on contenders | NUMA, holds (tie) |
| wqkv_a (112 tasks x 16 rows) | `stage_wqkv` + `gemv_fp8_loads` / `gemv_fp8_mfmas` (`WQKV0=176`) | 0.20 / 0.18 | FP8 GEMV, 5 MFMAs / wave, scale rewrite 4/5, 16 `vmcnt(0)` for 49 loads (peeled polls) | rocKE `stage_wqkv` (bitwise, stage 0.95-0.97), `sop` / `ra` / `poll_max` / `w_cpol` / `red_bar` | `EXTERN=wqkv` (win in K1), `gsop` / `gra` |
| gate (S, Sinkhorn x 20) | `stage_gate` (`GATE_BLOCK=8`, `COMB0=16`) | 0.01 / 0.07 | DPP sums with 285 `s_nop` cycles; next seam's pre | fused DPP (rocKE), spread at wide rows | none |
| qkv (S): q_norm -> MXFP8; kv_norm -> RoPE -> FP8 QAT -> window ring row | `stage_qkv` (`QKV0 = (176 + 112) mod 256 = 32`) | 0.11 / 0.00 | serial per-token; window write; token-parallel builds push to peers | spread, fuse into wqkv_a's epilogue (row-local), XCD-affine wqkv_a -> qkv | none |
| wq_b (TP2: 1024 tasks x 16 rows) + RoPE epilogue | `stage_wqb` (`WQB0=40`) | 0.29 / 0.35 | the largest K1 stage; FP8 GEMV, 8 MFMAs, scale rewrite 7/8; several tasks a CTA (`Dims.wqb_tasks` / 256 = 4 at TP2; P2's table lists 512 tasks, the TP4 count: re-check); weights before the q-latent poll | rocKE `stage_wqb` (in `eA`), `qhold`, `sop` / `ra`, MXFP4 weights (`QPROJ`, T), Rust direct only as a standalone kernel | `EXTERN=wqb`, `qhold` (tie), `QPROJ` (tie, T) |
| indexer q: iw (32 tasks), iq (256 tasks), iquant / iquant_fp4 | `kernels/index_query.py stage_iw`, `attn_pre.py stage_iq`, `stage_iquant(_fp4)` (`IW0=0`, `IQ0=32`, `IQUANT0=0`) | 0.05 / 0.07 (selecting layers only) | adds 0.22 of K1 code and 4 SGPR spills on indexer layers | K1 per-kind split (as K2SPLIT), GEMV family port of the indexer wq_b | `gsop` covers indexer wq_b |

### 2.2 K2 `layer_post` = K2a `attn_post` + K2b `moe` (0.64-0.80 of the target forward)

| stage | ATOM code | share (6 / 48 rows) | bound class | methods that apply | built so far |
|---|---|---|---|---|---|
| iscore | `kernels/index_score.py stage_iscore` / `run_scores` (`TASK_CHUNKS=8`); FP4 plane `index_score_fp4.run_scores4` | (index total) 0.04 / 0.02, selecting layers (about 0.2 of K2 time) | 64 fp8 MFMAs, 61 chained; logits through global memory + drain + flag | chain-break, read-ahead, fewer drains (bigger chunks per flag), K2SPLIT | K2SPLIT (tie) |
| isel (two levels) | `index_score.py stage_ilist` + `topk.py lds_topk` (`SEG=8192`, `PER_THREAD=64`) | in index | 4-pass LDS radix x 2 levels; 119-468 spill lane ops | single-pass / fewer-pass top-k, spill removal | - |
| iblk (candidate blocks, top-2048) | `stage_ilist` kind `blk` | in index | 252 / 396 spill lane ops | as isel | - |
| attn_score (S x tiles x 8 tasks, 64 splits x 16 keys) | `kernels/attention.py stage_attn_score` (`SCORE0=0`, `SPLITS=64`, `BK=16`, `SCORE_WAVES=8`, `ORDER_ROWS=6`) | 0.05 / 0.03 | bit-matched to Triton's split kernel; `buffer_load ... lds` keys; dead splits at short contexts | `askip` (B), `asplit` (T), score+PV fusion (section 4), Rust only standalone | `askip` (fixed, gate pending), `asplit` (T, pending) |
| irq (pv + split combine + inverse RoPE + group-32 FP8) | `attn_post.py stage_attn_irq`, `attention.py stage_attn_pv` (`IRQ0=64`) | 0.10 / 0.09 | 16 (6 rows) to 128 (48 rows) bf16 MFMAs, 15 `vmcnt(0)`, 2 polls | `pskip` (B), read-ahead, fewer polls, port | `pskip` (tie) |
| wo_a (TP2: 128 tasks x 32 rows, K split 4) | `stage_woa` (`WOA0=160`, `WOA_SPLIT_K`) | 0.07 / 0.07 | grouped FP8 GEMV, scale rewrite 7/8 | rocKE `stage_woa` (`eA`), `pfx`, `sop` / `ra`, MXFP4 (T) | `EXTERN=woa`, `pfx` (loss) |
| wo_b (320 tasks x 16 rows) + push to every rank | `stage_wob` (after wo_a's tasks) | 0.06 / 0.05 | FP8 GEMV; bf16 partial pushed CM_SYS; 16/29 spill ops | rocKE `stage_wob` (`eA`), CCL `epoch` / `lamport`, holds | `EXTERN=wob`, `hold.wob`, CCL (tie) |
| slice (160): rank-order sum of TP partials + FFN seam | `attn_post.py stage_reduce` + slice loop | 0.03 / 0.02 | the TP all-reduce point: latency of the xGMI hop | CCL forms, overlap with the other rank's skew, seam fusion | CCL (tie) |
| norm (ffn_norm, publish rows + XRDY) | `attn_post.py` norm loop (`NORM0=168`), `layer_post.py` publish | 0.03 / 0.03 | serial per-token + bulk publish (drain + barrier + flag) | spread, fuse norm -> xq, NUMA | NUMA (tie) |
| gate (FFN Sinkhorn) | `attn_post.py run_gates` (`GATE0=160`) | 0.04 / 0.01 | as K1 gate; may still poll while others are past norm | fused DPP, move off the critical path (it is for the next seam) | - |
| xq (S) | `moe.py stage_xq` (`XQ0=0`) | 0.03 / 0.02 | MXFP8 + FP8 of the row; plain words then X8RDY | fuse into norm (row-local, same CTA), Coz | `hold.xq` (tie) |
| router (E/16 x 4 parts) | `moe.py stage_router` (`router0 = max(8, S)`, `ROUTER_EARLY_FROM=12`) | 0.02 / 0.01 | fixed-order bf16 GEMV, 5 MFMAs, weights early from 12 rows | early-router reuse (T1 `ea`), fuse into norm CTAs, `FOLD` at wide rows | `FOLD` (native, tie) |
| route (S, one wave a token) | `moe.py stage_route` (`ROUTE_COPIES=8`, `SORT_NETS`) | 0.03 / 0.01 | sqrtsoftplus + bias, top-6, union U, LDS route table; polled by every MoE CTA | fewer copies / per-XCD copies, Coz, spread | `xq` homing (tie) |
| shared (TP2 72 tasks x token tiles) | `moe.py stage_shared` (`shared0 = route0 + max(8, S)`) | 0.06 / 0.03 | FP8 GEMV gate + up, silu, 19 `vmcnt(0)`, 3 polls | rocKE `stage_shared` (`eA`), `shold`, MXFP4 (`QSHARED`, T) | `EXTERN=shared`, `shold` (tie) |
| ug (\|U\| x parts) | `moe.py stage_ug` / `stage_ug_tile` (`ug0 = shared0 + shared units`; `ug_groups` 2, 1 below 7 rows; `ua = slices / 2`) | 0.25 / 0.34 | MXFP4 x MXFP8 scaled MFMA, 12/20 scale rewrites, MID + UGF flags | rocKE `stage_ug` (`uG`, `uGs` 0.989 C1), `usop` / `ura` / `ua`, XCD homing, expert-major fusion (section 4), MALL prefetch (T1) | `EXTERN=ug`, `xq`, `xcd`, `ua`, PREFETCH (loss) |
| down (320 tasks x 16 rows; a CTA's two together) | `moe.py stage_down` (`DOWN0=0`, `DOWN_R_MAX=2`, `DOWN_CHAINS=2`) | 0.18 / 0.26 | per expert of U: MXFP4 GEMV x weight, bf16 adds in top-k order, push to every rank; 35/42 (103/126) scale rewrites, 14-54 `vmcnt(0)` | `dsop`, read-ahead, slot-readiness pipelining (section 4), lamport push | `dsop` (tie) |
| reduce (160 x 32 cols) | `moe.py stage_reduce` (`RED_COLS=32`) | 0.01 / 0.00 | rank-order sum of MoE partials | CCL forms | CCL (tie) |
| (token parallel, >= 12 rows) gather | `layer_post.py` XG gather | - | extra hop; reduce-scatter instead of all-reduce | `SCHED tpf=` | `tpf` (knob) |

### 2.3 The draft (DSpark) K1 / K2

- Same code, `draft_runner.DraftMonoRunner`: block width 5, up to 8 requests (5 / 10 / ... / 40 rows), MoE
  128 experts / top-3, `ug_groups = 1`, no indexer (`index_bound_max = 0`). Draft K2 statics: down 0.18, ug 0.13,
  shared 0.12, irq 0.10 of its code; 220 SGPR spills. The draft is 0.05-0.10 of the step (C16-C1).
- Applies: the `eA` GEMV family already links at draft widths 5 / 10 / 15; `uGs` (ug sop + ra) for draft rows;
  the CCL `argmax` merge (Markov step, 0.958-0.983 of the draft segment); DVOCAB (acceptance-gated);
  Rust direct only for the standalone argmax / head kernels.

### 2.4 ATOM native path at C >= 9 (target rows > 48)

| bucket (C16 decode-only span, P4) | share | ATOM kernel | rocKE / W12-W14 lever | status |
|---|---|---|---|---|
| routed MoE (target + draft) | 0.54 | aiter `fused_moe` A8W4 (FlyDSL `mfma_moe1`, `gemm2_a4w4_port`) at 1.09-1.31x the rocKE bytes floor | none built; Q `moe_lb2` template (0.79-0.95 of FlyTape); VRAGGED cuts the union | VRAGGED: 1.8-7.3 % of the C16 step, served tie |
| dense GEMMs | 0.13 | `native_quant_linear` group32 FP8 | `WIDE` (rocKE `wide_gemm`, T) | tie; forced picks lose 4 % at C32 |
| all-reduce / gather | 0.11 | aiter custom all-reduce | `CCL=arfp8` (T) | +3-10 % at C32 / C64 only |
| mHC + norm, RoPE / quant | 0.08 + 0.07 | aiter `mhc_fused_post_pre_delayed_rmsnorm`, `quantize_fp8` | `SEAM=quant` (per call 0.90-0.94 at decode widths, 0.68 at prefill widths) | served tie |
| attention, indexer, elementwise | 0.05 + 0.01 + 0.02 | Triton paged decode, `score_topk_paged` | - | - |
| graph widths | - | capture sizes | `LADDER` (10 / 12 / 14) | inside the band |

About 1.5k kernels per step: a launch-count lever only pays if the graph replay leaves gaps (measure with KMARK /
kernel-only trace first).

### 2.5 Prefill graphs (WP1)

- `rk/prefill.py` + `rk/pfsched.py`: `ATOM_DSV41_RK_PREFILL_GRAPH=<lo>-<hi>[,cap=][,max=][,bk=]`, exact-length captures of
  ATOM's own eager prefill (40-1024 tokens); `PREFILL_BATCH=<n>,<ms>` admission hold (dominated by GRAPH).
- Methods that apply: the stall is the lever (served x1.17-1.61); inside a prefill graph the per-call kernel levers
  that won per call at prefill widths (SEAM quant 0.68, WIDE GEMM per-call winners) were never served inside the
  graphs; batched graphs (`bk=4`, `max=8`) as width knobs; long prompts need MIX (W14 T4).

---

## 3. The knob surface

Columns: default (today), value set to build (every value, losers included), class (B / B* / T, or "R" = runtime /
structure only), expected interactions. Existing knobs first, then every hard-coded choice that is not a knob yet
("proposed": name and grammar are suggestions; each must be default-off and knob-off byte-identical).

### 3.1 Existing W12-W14 knobs (`ATOM_DSV41_RK_*`, default off)

| knob | values | class | verdict so far (base) | expected interactions |
|---|---|---|---|---|
| `EXTERN=wqkv=;wqb=;woa=;wob=;shared=;ug=` | each stage on / off; per-stage `WqkvKnobs` (`sop` 0/1/2, `ra` 0/1, `poll` ir/asm, `poll_max` 1-16, `w_cpol`, `red_bar` wg/lds, `pause`) | B* | `eA` (5 GEMVs) C1 0.986-0.990 GPU step, served 1.016 (p 0.12); `uGs` 0.989 C1; `uG` tie | frees K1 VGPRs (114 vs 124): interacts with every register-sensitive lever; code-object lottery; `sop`/`ra` matter more on many-MFMA stages |
| `LEV=` codegen: `usop`, `dsop`, `ura`, `gsop`, `gsop2`, `gra`, `askip`, `pskip`, `poll1`, `asplit` | flags | B* (`asplit` T) | all tie alone; all-at-once loses 1.050-1.073; `usop` loses at 24-48 rows | `gsop`/`gra` add live registers (spill pressure in K2); `askip` x `asplit` exclusive above 12 rows; `poll1` x `PAUSE` x `NUMA` (same poll path) |
| `LEV=` schedule: `hold.<stage>=N` (20 stages), `shold`, `qhold`, `pfx`, `xcd`, `xq`, `ua=N`, `nop=N` | N in {8, 16, 32, 64, 128, 192, 256} s_sleep units; `ua` 1..slices | B (xq: own counters) | holds tie (0.995-1.001); `pfx` loss; `xcd` / `xq` tie | holds x contention levers (pfx, PREFETCH, ua); holds x NUMA (Q: holds won only on NUMA); `nop` is the recompiled control, never a lever |
| `K2SPLIT=1` | 0 / 1 | B* | tie | the enabler for every K2 register lever (codegen levers that add registers may only pay on the noix build); resources of the noix build are not recorded yet |
| `NUMA=` | 0, 1, 2, 3, 4, 5 | B (layout) | 1 tie; 4 / 5 lose 1.008-1.041 | homing x placement knobs (section 3.2); x holds (Q); x `PAUSE` / `poll1` |
| `PAUSE=` | 1 (ATOM), 2, 4, nop, none | B | tie | with NUMA and polls; Q `mailbox.pause=none` was in a pick |
| `SCHED=ug=,fug=,tpf=` | `ug` 1/2; `fug` rows 1-48; `tpf` 12..49 (49 = off) | B | not climbed | `tpf` x CCL forms x gather hop; `ug` x `xq` / `xcd` / `ua` |
| `CCL=` | `epoch`, `lamport`, `argmax`, `arfp8` (T), probes `dup`, `lat` | B (arfp8 T) | epoch+argmax C1 0.993-0.998; with TAIL 0.987-0.988; arfp8 3-10 % at C32/C64 | `epoch` x `TAIL=argmax`; `lamport` x `tpf`; `argmax` x DVOCAB |
| `LAYER=1` | 0 / 1 | B* | loss 1.054-1.064 (C1) | a Path A probe; re-test only on a new hand-over mechanism |
| `PREFETCH=` | `shadow`, `t`, `ea`, `ta`, `k1`, `k2`, `rec`, `worst`; options `la`, `kp`, `be`, `u`, `lo`, `b`, `bx`, `sk`, `rmax`, `maxit`; `RpfKnobs` / `ShadowKnobs` (`aux`, `threads`, `depth`, `unroll`, `pause`, `path` vgpr/lds) | B | every delivery loses 4-55 % of the step (side branch serializes; co-resident kernels delay CTAs; in-kernel prefetch costs its span); ceiling about 1 % | x grid size (a free-CU grid, section 4); x holds; x early-router |
| `TAIL=` | `argmax`, `dkv`, `nobcast`, `pad`, `1` | B* (pad fails) | GPU step 0.982-0.996, served tie | x `CCL=argmax` |
| `PREFILL_GRAPH=lo-hi,cap=,max=,bk=` | ranges up to 1024; `bk` 1-4; `max` 0-8 | B | x1.17-1.61 served (C4-C32) | x `PREFILL_BATCH`; x SEAM / WIDE at prefill widths (untried) |
| `PREFILL_BATCH=n,ms` | | R | dominated by GRAPH | - |
| `WIDE=<buckets>`, `WIDE_DIR`, `WIDE_SCOPE` | buckets 16-384; `WideKnobs` (`waves`, `mt`, `splits`, `depth`, `wcp`, `xb`, `order` sfast/rgfast/xcd, `fold` sc1/l2, `wfirst`, `gch`, `pipe`, `xsrc`, `sb`, `bra`, `occ1`) | T | tie / C32 forced loss | standalone rocKE: Rust direct applies; x LADDER |
| `FOLD=<buckets>` | `RouterKnobs` (`tiles`, `splits`, `fast`, `vbias`) | T | tie | standalone: Rust direct applies |
| `SEAM=quant` | | B | per call 0.90-0.94 (decode) / 0.68 (prefill); served tie | x PREFILL_GRAPH (untried) |
| `LADDER=1` | rungs 10 / 12 / 14 | R | inside the band | x WIDE, x VRAGGED |
| `QPROJ=mxfp4|mxfp4a`, `QSHARED=mxfp4`, `QHEAD=fp8`, `SHARD=`, `DVOCAB=k`, `RAGGED`, `DLEN` | | T | tie; QHEAD / QSHARED alone fail the literal gate; DLEN 1-3 loses | byte cuts pay only once latency buckets shrink: re-test after every structural win |
| `VRAGGED=surv:<th>|fixed:<L>|pad` | th in (0, 1) | B (exact) | C16 step -1.8 to -7.3 %, served tie | x LADDER, x native MoE |
| `TREE=<policy>` (prototype, eager) | `conf`, `best`, `gate`, `chain`; `TREE_BEAM`, `TREE_TOPK` | B (exact) | bound +3.2-6.7 % AL at same rows; not served | needs per-node storage (section 4) |
| `COLD=<plan>` (TP1) | | R | TP1 capacity route | - |

### 3.2 Hard-coded choices in ATOM mono that are not knobs yet (proposed)

Execution model (`atom/mono/plan/execution.py`, `atom/mono/device/sync.py`, `ops.py`):

| choice | default | value set | class | expected interaction |
|---|---|---|---|---|
| grid size `BLOCKS` (CTAs per launch) | 256 (one per CU; the runner refuses other CU counts) | 192, 224, 240, 248, 256 | B* (task -> CTA map changes; every task's arithmetic and every fixed-order sum stays) | Q: bandwidth saturates near 192 CTAs; frees CUs for co-resident side work (compressor, shadow prefetch, CCL comm kernel) that today delays the persistent CTAs (T1); x every placement base below |
| threads / waves per CTA | 512 / 8 | 256 / 4 (per stage where legal) | B* only where the wave split of a sum is preserved; else T | VGPR budget doubles at 4 waves; GEMV K splits are per wave (sum order): mostly T |
| poll batch `POLL_MAX` | 12 pairs | 4, 8, 12, 16 | B | live registers vs round trips; V rocKE `poll_max=16` slight loss on wqkv_a |
| poll pause | `s_sleep 1` | (= PAUSE) | B | - |
| poll shape | first iteration peeled | `poll1` | B | - |
| poll scope / home | device scope, one copy | (= NUMA) | B | - |
| step reset | `step_begin` zero + peer fence | (= CCL epoch) | B | - |
| `LOAD_BATCH` (strided-loop rounds in flight) | 4 | 2, 4, 8 | B | register pressure in slice / norm / gather |
| bulk publish protocol | `vmcnt(0)` + barrier + flag (`publish`) | per-wave flags, `sc1` bulk rule (Q P5: 0.62-0.82 of the bulk hop) | B* | x NUMA; x seam fusion |
| `kernel_symbol` / code layout | one code object | (lottery control) | - | always time with `nop` copies |

K1 (`kernels/attn_pre.py`):

| choice | default | value set | class | interaction |
|---|---|---|---|---|
| stage bases (CTA where a stage's first task lands) | slice 0, `GATE0=160`, `NORM0=168`, `WQKV0=176`, `QKV0=32`, `WQB0=40`, `IW0=0`, `IQ0=32`, `IQUANT0=0` | any base 0-255, plus a stride (place task t at base + t x stride, stride 1 or 8 = same XCD) | B | XCD-affine chains (norm -> wqkv_a -> qkv), NUMA homing, holds; the Q simulator mis-ranked placement: search it live |
| slice task width `COLS` | 32 hidden columns (160 tasks) | 32 only for bitwise; 16 / 64 | T (partial sum split changes) | - |
| `SLICE_FN_EARLY_MAX` | 24 rows (weights held through K2a's wait) | 12, 24, 36, 48 | B | spills above it (the comment says so): measure resources per value |
| `GATE_BLOCK` | 8 slices a gate thread sums | 8 (sum order) | B only at 8 | - |
| GEMV rows a task `ROWS` / token tile `TILE` | 16 / 16 | fixed by MFMA N | - | - |
| tasks per CTA for wq_b | consecutive tasks on one CTA (TP2: 4) | task -> CTA interleave order (`sequence` knob) | B | x `qhold`, x weight issue depth |
| weights-in-flight depth before a poll | every K step's weights at once | 1, 2, all (per GEMV) | B | x holds (contention) |
| FP4 index plane | `index_fp4` per deployment | - | - | - |

K2a (`kernels/attn_post.py`, `attention.py`, `index_score.py`, `topk.py`):

| choice | default | value set | class | interaction |
|---|---|---|---|---|
| stage bases | `SCORE0=0`, `IRQ0=64`, `WOA0=160`, wo_b after wo_a's tasks, slice 0, `GATE0=160`, `NORM0=168` | any base + stride | B | the score -> irq hop and the wo_a -> wo_b hop cross CTAs: XCD-affine placement |
| attention splits | 64 x 16 keys (pinned to Triton's 6-row order) | 64x16 (B), 32x32 (`asplit`, T), dead-split skipping (`askip`, `pskip`, B) | - | `asplit` T only |
| score waves / splits a wave | 8 / `SPLITS / 8 / per` | - | B | - |
| irq column groups | `pv_groups(s, tiles)` | per width | B | x irq task count vs CTAs |
| wo_a K split (`WOA_SPLIT_K`) | `WAVES x ROWS / WOA_ROWS` = 4 | fixed (sum order) | B only at 4 | - |
| indexer `TASK_CHUNKS` | 8 chunks of 128 columns a score task | 1, 2, 4, 8, 16 | B (flags per task: the logits are per column) | fewer drains and flags vs fewer CTAs |
| top-k structure | 4-pass LDS radix, 2 levels, `SEG=8192`, `PER_THREAD=64` | passes / radix width, `SEG` | B (exact top-k with lowest-index ties) | LDS budget (K2a's union) |
| indexer `s_setprio` / `sched_barrier` (FP4 scorer) | on | on / off | B | - |

K2b (`kernels/moe.py`, `moe_shape.py`, `layer_post.py`, `runner.py`):

| choice | default | value set | class | interaction |
|---|---|---|---|---|
| stage bases | `XQ0=0`, `router0 = max(8, S)`, `route0 = router0 + router tasks`, `shared0 = route0 + max(8, S)`, `ug0 = shared0 + shared units` ("shared units' CTAs last"), `DOWN0=0`, `RED0=0` | permutations of the order of these blocks; bases; strides | B | routing chain placement vs ug start skew; x `xq` / `xcd` |
| `ROUTER_EARLY_FROM` | 12 rows | 1, 6, 12, 24, 49 (off) | B | contention with xq / norm streams (comment: slower below 12) |
| router groups | one per token tile, as many as one round of CTAs holds | - | B | - |
| `ROUTE_COPIES` | 8 (CTA b polls copy b mod 8) | 1, 2, 4, 8, 16, 32 (or one per XCD) | B | x NUMA homing (`=4` homes the route copies per XCD) |
| `ug_groups` / `fine_ug_below` | 2 / 7 (draft: 1) | (= SCHED `ug`, `fug`) | B | x `ua`, x `xq` |
| ug loads in flight `ua` | slices / 2 | (= LEV `ua`) | B | x holds |
| `DOWN_R_MAX` (scale blocks a down wave holds) | 2 ("3 took no less time, 4 spills") | 1, 2, 3 | B | registers: x K2SPLIT |
| `DOWN_CHAINS` | 2 accumulators | 1, 2, 4 | T (fp32 association) | x `dsop` |
| down tasks per CTA | two together | 1, 2 | B | x down start skew |
| `RED_COLS` | 32 | 16, 32, 64, 128 | B (sum order per column fixed) | poll passes per thread (comment: 128 cost 6 pairs a thread) |
| shared expert placement | after route, before ug | before route / interleaved with ug | B | x `shold` |
| per-CTA stage visit order (K2a then K2b, barrier between) | ATOM's order | `Knob.sequence` over the stage list a CTA visits (only where data dependencies allow) | B | Q decision 15: order is a knob |
| `tok_par_from` | 12 rows | (= SCHED `tpf`) | B | x CCL |
| `TAGS` (mailbox tag space) | 256 | - | - | epoch form widens it |
| `CM_NT` on weight streams | on | nt / plain (per stage) | B | T1: nt loads do not allocate in MALL but hit what others placed: x PREFETCH |

Compile level (per build, FlyDSL K1 / K2 and every rocKE kernel):

| choice | default | value set | class | interaction |
|---|---|---|---|---|
| LLVM scheduler strategy (`amdgpu-sched-strategy` function attribute; rocKE `CodegenPolicy`) | none | `max-ilp`, `max-memory-clause`, `iterative-ilp`, `iterative-minreg`, `iterative-maxocc` | B* (gate catches miscompiles) | `iterative-minreg` is the cheap attack on the K2 SGPR spills; x K2SPLIT |
| `amdgpu-waves-per-eu` | none | 1, 2 | B* | one CTA of 8 waves per CU: shapes the VGPR budget only |
| code-object backend per rocKE standalone kernel | COMGR | `comgr`, `direct` (+ lever set, + crate build via `ROCKE_CODEOBJ_DIRECT_LIB`) | B* (bitwise vs COMGR, Q: every timed lever object bitwise) | per-family lever sets differ; the guard books per key and node |
| direct lever set | - | B3, MoE r1, Z, ZL, ZLv, ZLP, ZM, N6, N7, M1Nx, F, and single levers | B* | `nofence` flips sign by family |
| pass / lever order inside the direct compiler | crate default | `Knob.sequence` over the lever passes | B* | Q PLAN decision 18 / task rule: order is a knob |

rocKE stage functions (`dsv41_mono/stage_*.py`, `WqkvKnobs`): `sop` {0, 1, 2}, `ra` {0, 1}, `poll` {ir, asm},
`poll_max` 1-16, `w_cpol` {nt, 0}, `red_bar` {wg, lds}, `pause`, `x_first` (must-lose control only). The rocKE side
kernels: `RpfKnobs`, `ShadowKnobs`, `RouterKnobs`, `WideKnobs`, `FloorKnobs` (fields in section 3.1).

### 3.3 Interaction groups (what to screen together)

1. **Poll path:** NUMA x PAUSE x `poll1` x `POLL_MAX` x ROUTE_COPIES x CCL epoch: one code path, many small effects.
2. **Contention:** holds (20 stages) x `pfx` x `qhold` / `shold` x `ua` x `ROUTER_EARLY_FROM` x PREFETCH x weights-in-flight
   depth. Q found the holds by Coz slopes, not by guessing values.
3. **Registers:** K2SPLIT x `gsop` / `gsop2` / `gra` / `usop` / `ura` x `DOWN_R_MAX` x `SLICE_FN_EARLY_MAX` x scheduler
   strategy x EXTERN stages (the linked functions change the allocation of the whole kernel).
4. **Placement:** stage bases / strides x NUMA mode x `xq` / `xcd` x `BLOCKS`.
5. **Exchange:** CCL forms x `tpf` x TAIL x the slice / reduce placement.
6. **Width / runtime:** PREFILL_GRAPH x PREFILL_BATCH x LADDER x VRAGGED x WIDE / SEAM / FOLD.
7. **Bytes:** QPROJ / QSHARED / QHEAD / DVOCAB: re-test after every latency win (they tied because the step is not
   byte-bound at C1-C4; that changes as latency shrinks).

---

## 4. Creative structural ideas not tried yet (ranked)

Rank = expected size (from a measured share or bound) x confidence / cost. Every item starts with its prototype
bound, and a loss gets a WaveScope diagnosis plus one structurally different restructure before "rejected on base X".

**Prerequisite, not an idea: the first V4.1 bubble ledger.** WaveScope (section 1.3) on K1 / K2 at 6 / 24 / 48
rows and both layer kinds, plus a V4.1 blame graph (side-copy stamps in the mailbox pairs, as Q `blame.py`) and Coz
slopes. Everything below is ranked on today's evidence and must be re-ranked once the ledger exists.

| rank | idea | mechanism | where the methods point | bound / first prototype | numerics | owner risk |
|---|---|---|---|---|---|---|
| 1 | **Coz-guided holds, then a hold re-climb** | sweep `hold.<stage>=N` as an entry delay per stage (slope of K2 time vs delay), add post-delay points; hold every stage with a negative slope; then `phased` over holds + NUMA + PAUSE | Q: holds were found exactly this way and were the second largest Q lever; V holds tied at hand-picked values | slope sweep costs about 20 stages x 3 delays of HTL runs; no code beyond post points | B | low |
| 2 | **Seam hop fusion** (K2: slice -> norm -> xq -> router partials on the token's own CTAs; K1: slice tail -> norm, wqkv_a epilogue -> qkv) | row-local stages run on the CTA that produced the row, removing device-scope hops; the fan-in (160 partials) stays | seams are 0.11-0.19 of K2; Q: one hop is 0.007-0.022 of a round; ANALYSIS 6 | prototype: one fused pair (norm + xq) with a bitwise gate; measure the per-hop share in the blame graph first | B (row-local, same op order) | medium; planned as W14 T3, no result recorded |
| 3 | **Placement knobs + a free-CU grid** (`BLOCKS` 192-248) | stage bases / strides become knobs (XCD-affine chains, islands for the serial per-token stages); a smaller persistent grid leaves CUs for co-resident side work without delaying CTAs | Q: bandwidth saturates near 192 CTAs; XCD-SCHEDULING levers 1 / 4; T1: co-resident kernels delayed 96-128 CTAs | prototype: `BLOCKS=224` and the shadow prefetch at 32 CTAs; first measure K2 at 224 vs 256 alone | B* | medium (runner refuses other CU counts: knob must bypass it) |
| 4 | **Rust direct for every standalone rocKE kernel** | per-kernel auto-select with the guard: CCL argmax merge, shadow / side prefetch, router fold, quant seam, wide GEMM (all latency-bound small kernels) | Q: CCL `pair_os` 0.953-0.972, P4 `xl` 0.858-0.860; latency kernels gained most | no new code: compile each with `ROCKE_CODEOBJ_BACKEND=direct` + lever sets, bitwise vs COMGR, same-job A/B on both nodes | B* | low; a loss is a crate defect (decision 18) |
| 5 | **Slot-readiness MoE (down per expert as its ug finishes)** | today every down task waits on all of U's MID; let down start per expert slot when that slot's ug parts land, with the bf16 top-k-order adds done in a per-token combine (or by ordering down's slot loop by readiness while keeping the add order in registers) | Q fused expert-major MoE 0.874-0.887; ug + down are 0.44-0.61 of K2 | first measure the ug-tail -> down-head wait in the blame graph; prototype the ready-order loop | B (adds stay in top-k order) | medium-high (1.8k-line `moe.py`; rocKE `stage_ug` exists) |
| 6 | **SEAM / WIDE inside the WP1 prefill graphs** | the per-call wins at prefill widths (quant seam 0.68) were never served inside the captured prefill graphs, where prefill time is the stall the decode pays | WP1 (the stall is the lever) + WP6 per-call | served C4-C32 cells with GRAPH on both arms | B (SEAM) / T (WIDE) | low |
| 7 | **Scheduler strategy / attribute sweep on FlyDSL K1 / K2** | add `amdgpu-sched-strategy` (5 values) and `amdgpu-waves-per-eu` to the K2 (and K2SPLIT noix) builds | runbook 12.1.M + scheduler-policy.md; K2 SGPR spills 790-1209 | static first: spills and VGPRs per value; then HTL | B* | low (miscompiles caught by the gate) |
| 8 | **Score + PV in one CTA per (token, head tile) at small S** | removes the AM / AL / AP mailbox hop; one CTA runs all live splits (8 waves) then the combine in the same order | attention is 0.12-0.15 of K2; most splits are dead at GSM8K lengths (`askip`) | prototype at 6 rows: bitwise vs ATOM; bound = the hop share in the blame graph | B (same split arithmetic and combine order) | medium |
| 9 | **Wave-specialized polling** | one poll wave per CTA; loader / MFMA waves wait on LDS or `s_barrier` instead of on a poll whose `vmcnt(0)` also waits for their in-flight weights | XCD-SCHEDULING lever 10; V MONO-STAGES 2.1 note (loads return in order) | ATT: count poll retries that sit behind weight `vmcnt` | B | medium |
| 10 | **K1 mHC stages as rocKE functions** (slice, norm, gate) then `sop` / DPP / spread search | the Q WP2a HC port pattern on V4.1's slice (f32 MFMAs) and Sinkhorn gate (DPP with `s_nop` padding) | slice + norm are 0.34 of K1; Q HC 0.97-0.99 in situ | port slice first (bitwise gate exists: `EXTERN` + LEV_CHECK) | B* | medium |
| 11 | **Speculative early expert issue in ug** | ug CTAs homed by expert id (static expert -> CTA set) issue `buffer_load ... lds` of their predicted expert parts before the route poll; a hit costs no extra bytes, a miss is dropped | T1: early router recalls 0.72-0.77 of the union (EA 0.80-0.87 of bytes); MALL prefetch lost because it re-reads | bound: the route -> ug latency share x hit rate; static homing may imbalance ug (balls in bins): measure with placement knobs first | B | high |
| 12 | **Whole-K2 rocKE port, then Rust direct** | bitwise rocKE K2 (stages exist for wo_a, wo_b, shared, ug; attention, indexer, seams, down, route remain) compiled direct: register pinning, exact `vmcnt`, poll leaf ops, per-stage budgets, no SGPR spills | Rust track: largest control; LB MoE lesson: spills and chains cost 1.05-1.29 | only after the ledger shows spills / waits on the critical path; build stage by stage behind EXTERN, then lift to a standalone kernel | B* | very high |
| 13 | **Tree verify at the same rows** | per-node storage in every CSA2 store, ancestor masks, tree index builders in native and mono | T2: +3.2-6.7 % AL at the same 6 rows (measured, lossless) | already bounded; W14's top follow-up | B (exact) | very high (ATOM-wide) |
| 14 | **rocKE wide MoE for C >= 9** | Q `moe_lb2` structure (round-robin deal, broken chains, per-token combine counters) at V4.1 shapes | native MoE 0.54 of the C16 span at 1.09-1.31x the floor | bound 0.08-0.24 of MoE time; `moe_floor` exists | B or T | high |
| 15 | **Mono beyond 48 rows** (C9-C16) | a rocKE wide K2 (shared expert units, VGPRs) so the mono engages at C9-C16 | P4: mono's span 0.82 of native at C4, 0.95 at C8: the edge shrinks with rows | low expected value | B* | high |
| 16 | **Single-pass indexer top-k** | fewer radix passes / one level | indexer 0.02-0.04 of K2, selecting layers only | low | B | medium |

Not repeated without new information (Q / V losses): global dynamic tickets, simulator placement, two-layer stacking,
`s_setprio` alone, two polls in flight, MALL speculation of last step's experts as a side kernel, `layer=1` as built.

---

## 5. The W15 climb protocol

1. **Bubble ledger first** (section 1.3 + blame + Coz), at 6 / 24 / 48 rows, layer kinds reuse and full, both ranks.
2. **Objective hierarchy:** graph-mode K2 critical path (HTL GPU decode step, `nop` controls, rotated passes) ->
   GPU decode step per cell (C1 / C2 / C4 / C8) -> served per-request tok/s on both nodes (claims only here).
3. **One search space** over every knob of section 3 (built values only; losers included), driven by
   `qwen38_mono/tune` `phased` through a V4.1 backend: `compile` = build the arm's env (ATOM_DSV41_RK_* strings via
   `LeverStringKnobs`), `check` = the in-place gates with a planted mutant first, `measure_batch` = an HTL run with the
   baseline and the `nop` control interleaved. Structural knobs (`K2SPLIT`, `NUMA`, `BLOCKS`, `tpf`, EXTERN sets,
   backend) are `structural=True`; tolerance knobs are `numerics=TOLERANCE` and need T2 / T4.
4. **Order:** climb order is a knob (restarts in reversed and shuffled orders); every reject is re-tested after each
   accepted move; the final revisit pass re-tests all rejects; the confirmation includes the solo-winners-only arm.
5. **Rust direct per kernel:** every standalone rocKE kernel in the stack gets a `backend` knob and a lever-set knob;
   the guard books it only with WINs (or TIE <= 1) on both nodes; each loss goes to the crate as a defect with the
   direct-vs-COMGR ISA diff.
6. **After each structural win:** re-sweep the contention and register groups (3.3 groups 2 and 3) and the byte cuts.
7. **Record** every verdict with its base in `dsv41-mono/w15/` (ratios), absolutes in
   `hackathon-prep/private-perf/dsv41-mono/w15/`.

GPU budget: at most 8 gfx950 GPUs for W15 (check `squeue` for W13 / W14 neighbours); HTL and served runs are 2-GPU
jobs (TP2); static dumps, identity checks and replays go to CPU jobs.
