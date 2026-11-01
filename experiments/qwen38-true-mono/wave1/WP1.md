# WP1: the persistent-framework platform (W5, wave 1a, 2026-10-31)

D0.md section 5 (wave 1a, WP1), D0.2, D0.6, D0.8; DESIGN.md group 2. This file gives ratios and counts only.
The raw outputs with absolute device times are listed in
`hackathon-prep/private-perf/qwen38-true-mono/wave1/WP1.md`. No git state was changed. All new code is
Python-only, and every existing build is byte-identical with the new knobs off (section 6).

**Gate: met after the Fix round.** The first gate claim did not survive verification (the ptoy mutant verdict
was a race); the "Verification" and "Fix round" sections at the end give the details and the corrected results.
- CPU tests pass: 869 qwen38 and platform tests, 307 search-engine tests, and the 97 new platform tests (Fix round:
  882 and 307, with 110 platform tests).
- Byte identity holds for every existing build family: 2,949 parity-family files over 4 archs and 197 true-mono
  files, LLVM IR and serialized rocKE IR, compared byte for byte. Fix round: the ISA (`.text`) of the 87 true-mono
  builds is byte-identical too.
- On hardware, on both rocke nodes, the platform primitives reproduce P4's and P5's verdicts: the probes' own benches
  run on the platform builders, and every gate, negative control and mutant behaves as in the probes.
- Every negative control of the platform's own primitives fails as required.

Two platform bugs were found on hardware and fixed (section 8). The gaps are listed in section 9; none is hidden.

## 1. What was built

New package `dnn-providers/hip-kernel-provider/rocke/library/kernels/gfx950/qwen38_mono/persist/`:

| module | content | D0 item |
|---|---|---|
| `knobs.py` | `PersistKnobs`: P6's R1-R5 (`args`, `fresh`, `loop_carry`, `machine_licm`, `wgid`), `inline`, and the D0.8 sync defaults (`arrivals`, `stripes`, `ticket`, `deadline`, `fatal_word`, `kv_bulk`, `scalar_reread`, `xhold`, `unroll`). Presets: `D0_DEFAULTS` and `LEGACY`. `EdgeKnobs` / `EdgePlan`: per-edge `protocol`, `polarity`, `poll`, `pause`, `deadline`, `placement`, `queue`, `island`. `RustScope`: per-stage lever sets for the direct backend (the measured B3 and MoE r1 sets; `nofence` and `loadorder` are refused for the head) | D0.2, D0.6, D0.8, PLAN 10 |
| `sync.py` | P4's primitives as library emitters: `pad_lds`, `poll_counter`, `poll_pair_deadline`, `poll_rank_sum`, `ticket_xcd` (optionally with the fatal word). New: `GridSync` / `grid_sync` (8 stripes per XCD; the layout is computed in the kernel; P4's abort-on-timeout rule; launch-fatal word), `cta_arrive` (one arrival per CTA from its last wave, P9), `raise_fatal`, `fatal_reasons` | D0.8, P4 1.4-1.5, P9 |
| `kv.py` | P5's protocols: `bulk=sc1` (default), the bulk rule, the plain negative control, `acquire`, `kv_store`, `kv_load`, `publish`, `poll_ready`, and the C-B5 `scalar_reread` (`reg` default, `dinv_inv`, `dinv`, `plain`). Every op carries a `rocke.kv` marker for the static check; the marker does not reach the LLVM IR | D0.8, P5 |
| `lint.py` | Static checks: `check_kv_acquire` (P5); `spin_without_reload` (P2 finding 1); `exec_join_findings` (FlyTape / P6 lint, ported to objdump listings); `code_object_meta` / `resource_gate` (P6 CI gate); `check_staging_overlap` / `fused_moe_staging` (P8) | WP1 scope (P5, P8, P2, P6) |
| `isa_gate.py` + `isa_dist.py` (+ `isa_dist_selftest.py`) | WP0a's pooled ISA gate for rocKE builds. `isa_gate` writes `keys_isa`-format records (key = sha256 of the serialized IR, variant = sha256 of `.text`), with each build compiled in a fresh process or all builds in one. `isa_dist` and its self-test are vendored verbatim | WP0a |
| `driver.py` | `build_persistent_kernel`: the runtime layer loop (never unrolled). It applies R1 (`args=const` per-stage scalar reloads from a per-layer table; `args=lds` as a knob), R2, R3 (the loop carries only its index; the step is re-read per layer), R4 and R5. Runtime parity slot `li & 1`, `prev_tag`, steady-state barriers. `validate_round_plan` runs `validate_stack` over a whole layer sequence | D0.2, D0.7 test path, DESIGN-LB WP-L4 |
| `parity.py` | The P4 / P5 probe kernels rebuilt on the platform primitives. P4: the probe builders with their primitives replaced. P5: the KV builder rewritten on `kv`, and the scalar reads routed through `kv` | gate |
| `micro.py`, `ptoy.py` | Hardware-gate kernels: `grid_sync` with P4's counter-kernel ABI; the ticket with the fatal word; a 2-stage exact-int32 chain built as Path B (`layers=1` per launch) and as Path A (the loop driver), with a broken-tag mutant | gate |

Changes to tracked files. They are all additive or knob-gated, and all CRLF-preserving.
- `rocke/core/ir.py` and `core/lower_llvm.py`: the `tile.tagged_poll(reload="launder")` op variant. The original op
  carries no new attribute and lowers to the same bytes.
- `helpers/compile.py`: kernel attribute `machine_licm="off"` adds `-mllvm -disable-machine-licm` (R4).
- `qwen38_mono/mailbox_ir.py`: the mailbox implementation `irf` (suffix `mbirf`).
- `qwen38_mono/framework.py`: new `FrameworkKnobs` fields `persist`, `edges`, `err_arg` and `rust_scope`, all
  `None` or empty by default. Also: `LayerCtx.edge()`, `LayerCtx.persist_knob()`, runtime parity slots in
  `region_addr` / `_flip`, deadline polls with the error word, the R2 / R4 / R5 hooks, and the pinning of
  address-taken LDS words.
- `qwen38_mono/README.md`; `benchmarks/.../toy_chain_bench.py` (M10). Also `qwen38_mono/tune/evaluate.py` (M12
  fix), which is not tracked: it is the tuner's own working-tree file.
- M1-M4 and M6 / M7: `dispatch/core.py`, `dispatch/tuning/candidate.py`, `helpers/autotune.py`, `sweep.py`,
  `dispatch/tuning/__init__.py`, `dispatch/ARCHITECTURE.md`, `dsl_docs/autotune/overview.md`.
- Tests: `library/tests/test_qwen38_persist.py` (97).

Harness, outside the repos, in `hackathon-prep/qwen38/truemono/WP1/`:
- `wp1_p4.py`, `wp1_p5.py`: the probes' benches on the platform.
- `wp1_bench.py`: ptoy, gs and ticket.
- `wp1_p2fix.py`, `wp1_toy.py` (M10 / M11), `wp1_m12.py`.
- `wp1_job.sh`; `stage_cpu/` (identity census, tests, ISA gate, mutation testing; `make_overlay.sh`).

## 2. D0.2 / D0.8 as built

| rule | where | default |
|---|---|---|
| inline per layer body | rocKE stage emitters are inlined by construction. `PersistKnobs.inline="never"` is registered and refused explicitly: rocKE has no device-function call path in kernel lowering, and P6 measured `noinline` at 1.105-1.130 per layer with the scratch gate failing | always |
| R1 `args=const` (`args=lds` a knob) | loop driver: one `s_load_dwordx2` per argument per stage from `arg_tab[layer][k]` (test: 3 reloads per layer in ptoy). Static `layers>=1` builds keep the kernarg segment (`args="kernarg"`); other values are refused there | const |
| R2 fresh ids | `fresh="all"` gives each stage a fresh lane id (`fresh_v`) and `tid = wave*64 + lane`; the wave and workgroup ids come from the existing `fresh_ids` | all |
| R3 | the loop carries only `li`; the step counter is re-read per layer (`loop_carry=False` is a knob) | on |
| R4 | kernel attribute -> `-disable-machine-licm` (test checks the COMGR options) | off |
| R5 | workgroup id written to an LDS word at entry (CTA barrier) and read back per stage | lds |
| `xl` tagged pairs, never fences | `EdgeKnobs.protocol="pairs"` default; `fence` / `bulk` are knobs | pairs |
| local NUMA polarity per edge | `EdgeKnobs.polarity` (`local` / `remote` / `none`), applied in `LayerCtx.mb(local=True)` | local |
| counters: 8 stripes per XCD | `GridSync(stripes=8)`; 1 / 2 / 4 are knobs | 8 |
| per-XCD-start residency ticket | `sync.ticket_xcd`, optionally launch-fatal | xcd |
| deadline polls, abort-on-timeout, launch-fatal word | `GridSync(maxit)` (abort + `sync_timeout`); `EdgeKnobs(poll="deadline")` + `FrameworkKnobs.err_arg` (`poll_timeout`); ticket (`ticket_timeout`). The host decodes them with `sync.fatal_reasons` | off (knob) |
| one epoch arrival per CTA | `GridSync(arrivals="cta")`, `sync.cta_arrive`; `wave` is a knob | cta |
| in-launch KV rows `bulk=sc1`; words in registers / payload; `s_dcache_inv` re-reads | `kv` (defaults `sc1` / `reg`), with `lint.check_kv_acquire` | sc1 / reg |
| `xhold=xpre`, never unroll | `PersistKnobs.xhold` (consumed by emitters through `ctx.persist_knob`); `unroll` accepts only 1 | xpre |

**Nothing is silently ignored.** A non-default `PersistKnobs` field that the framework does not apply itself
(`arrivals`, `stripes`, `ticket`, `deadline`, `fatal_word`, `kv_bulk`, `scalar_reread`, `xhold`) must be read by a
stage emitter through `ctx.persist_knob(...)`. The same holds for a non-default edge `protocol`, `placement`,
`queue` or `island`, which an emitter must read through `ctx.edge(...)`. Otherwise the build is refused (tests).

## 3. The `tagged_poll` reload fix (P2 finding 1, re-diagnosed)

- **The P2 suggestion does not work.** P2 proposed a memory clobber on the pause asm, or volatile loads. Both
  variants were built and still compile to the hoisted spin:
  `s_and_b64 vcc, exec, 0; s_nop 0; s_mov_b64 vcc, vcc; s_cbranch_vccz -3`. This was checked with the local llvm23
  and the cluster's llvm22.
- **Root cause.** The polled buffer is a `noalias` kernel parameter, so alias analysis concludes that no asm in the
  loop can write it. The volatile `aux` bit is invisible to the IR passes.
- **The fix that works: `reload="launder"`.** Each retry passes every load's `voffset` through a side-effecting
  `"=v,0"` asm, which makes the address loop-variant. The pause also gets the `~{memory}` clobber.
- **`MailboxKnobs(impl="irf")` selects it.** The original op and every `ir` build are unchanged.

Evidence on hardware, both nodes:
- **Static lint.** On the P2 builds that hung (`pairs` and `bulk` with `pimpl="ir"`), the spin lint flags 2 and 1
  non-reloading loops in the original builds and 0 in the fixed builds.
- **Fixed builds run.** They complete 24/24 launches bit-equal to the served head, on n01-09 and on n07-17.
- **Negative control.** The original `pairs_ir` build hangs on n01-09: the process was killed after 90 s, as P2
  found.
- **Cost.** At the primitive level `irf` runs close to `ir`, measured in P4's own harness by comparing each arm's
  median ratio against the same boundary or counter reference (not paired against each other):
  - P4 section A (a2a, n2c): 1.00-1.02 of `ir`;
  - P4 section C (`xl` hop, n2c): 1.02-1.07 of `ir`.
  - `irf` still beats the graph boundary everywhere `ir` does.
  - That cost is one `v_mov` per retry, plus LLVM no longer peeling the first load.

## 4. Static checks and their validation

| check | validation |
|---|---|
| `check_kv_acquire` (P5) | Run on all 12 P5 KV writer/reader combinations and all 4 platform scalar modes. On the CPU it is clean exactly for P5's memory-model positives and flags every negative (plain writer, reader with `none` / `inv0`, `smem` without `s_dcache_inv`). Against P5's hardware result for every cell, on both nodes: **no false negative**. It also flags the combinations the numerics gate cannot see: `none` / `inv0` readers were wrong in only 2 of 4 tail x slice cells, as P5's verifier warned. `smem_dinv` is flagged as not memory-model safe under `strict`; it was clean on hardware. Acquire-before-poll and load-without-poll are flagged (tests) |
| `spin_without_reload` (P2) | Flags the P2 `ir` builds and not the fixed ones (section 3). Synthetic tests: a counted delay loop and a reloading poll are not flagged |
| `check_staging_overlap` (P8) | Flags the fused MoE's actual `ys` staging (`SC["ss"] + r * SCRATCH_BYTES`) over the polled `ss` region: P8's latent tag-wrap hazard. A relocated staging past the mailbox rows is clean. An unpolled region is not a hazard |
| `resource_gate` (P6) | Spill and private-segment counts from the code-object metadata, plus the spin lint. The exec-join lint is reported and gates only with `strict_exec_join`. Reason: FlyTape's heuristic also flags legitimate structurizer join blocks in rocKE builds (6 in the P2 `pairs` builds, all temporaries consumed under the same mask) |
| ISA gate (WP0a) | On a GPU node and on a CPUONLY node: two pools of the same ptoy builds PASS (`KEYS=same`); a kernel-side knob (the `irf` poll in one build) FAILs exactly that build (`--expect ...=fail:ptoy_static` holds). The vendored self-test passes 10/10 and kills 8/8 rule mutants |

## 5. Hardware parity and controls (both rocke nodes; 9.1 timing: shuffled interleaved rounds, two controls)

### P4 on the platform primitives (`wp1_p4.py` = `p4_bench.py` with `persist.parity.platform_p4`)

- **Gate PASS on both nodes.** Every positive passed: counters on every layout and page, a2a, ticket, `xl` and
  boundary.
- **Every control behaved as in P4.**
  - Dropped arrival at s8 / s1 / s0 and P 256 / 160: no release in the dropped epoch, every later epoch aborted,
    256/256 timeouts.
  - Oversize grids time out.
  - The deadline a2a with a dropped producer blocks exactly that epoch.
- **Every mutant failed**: early release, skip-last, `tagoff`, `noready`.
- **The verdicts match P4's section 1** (median ratios, lat_max):

| quantity (vs `s8L` unless said; lat_max and lat_med) | P4 | platform n01-09 | platform n07-17 |
|---|---|---|---|
| one counter per XCD, all arrive together (P256 idle) | 0.77-0.84 WIN | 0.761-0.766 WIN | 0.756 WIN |
| one counter per XCD, spread arrivals (P256 uni) | 1.72-1.80 LOSS | 1.748-1.888 LOSS | 1.753 LOSS |
| one device word | 1.74-4.93 LOSS | 1.716-5.162 LOSS | 1.740-4.653 LOSS |
| 2 / 4 stripes | 1.06-1.62 LOSS | 1.04-1.71 LOSS (one P160-late cell noise) | 1.03-1.72 LOSS |
| counter IOD home (`s8R`) | noise | noise (one P160-idle cell 1.04 LOSS) | noise |
| a2a pairs at the remote polarity, asm poll | 0.54-0.84 WIN | 0.548-0.865 WIN | 0.546-0.862 WIN |
| ticket per-XCD-start vs FlyTape's | 0.25-0.40 WIN | 0.252 WIN | 0.339 WIN |
| `xl` pairs hop (ir / irf poll) vs graph boundary, 1 / 4 / 16 rows | 0.33-0.94 (WIN; 1 row bunched noise) | 0.281-0.932 (WIN; 1 row bunched noise) | 0.283-0.942 (same) |
| `xl` pairs, asm poll at the calibrated-local polarity, 4 / 16 rows | 1.08-1.21 LOSS | 1.04-1.20 LOSS | 1.02-1.20 LOSS |
| fenced / bulk hop, 1 and 4 rows | 1.16-1.80 LOSS | 1.14-1.83 LOSS | 1.13-1.78 LOSS |
| bulk hop, 16 rows | 0.78-0.84 WIN | 0.760-0.837 WIN | 0.769-0.840 WIN |

The platform kernels' LLVM IR is byte-identical to the probe kernels' (CPU tests, every P4 spec). The new
information here is that the verdicts reproduce on fresh allocations and processes, plus the `irf` arms.

### P5 on the platform protocols (`wp1_p5.py` = `p5_bench.py` with `persist.parity.platform_p5`)

- **Gate PASS on both nodes.** Positives clean.
- **Every control failed as required**: plain writer, reader without acquire in the vL1D-resident cells, scalar
  re-read without `s_dcache_inv`. The rewrite-before-reads mutant failed.
- **Ratios match P5's section 1** (n01-09 / n07-17):
  - `bulk=sc1` hand-off vs the bulk rule: 0.66-0.81 / 0.67-0.81 on hop, 0.78-0.86 / 0.79-0.87 unstamped. P5:
    0.62-0.82 and 0.78-0.87.
  - Scalar re-read vs the payload (`reg`):
    - `s_dcache_inv` + `s_load`: 1.05 / 1.05 unstamped (P5: 1.03-1.09);
    - the memory-model-safe re-read: 1.80 / 1.78 (P5: 1.72-1.78);
    - `glc`: 3.65 / 3.53 (P5: 3.4-3.5).

### The platform's own primitives (`wp1_bench.py`)

**ptoy, Path A vs Path B.** 12 layers × 3 steps × 2 nodes, all on one scratch buffer whose stale pairs from the
previous step stay in place.
- Every arm's 12 layer outputs equal the host reference and are bit-identical across arms. The arms:
  - Path B (`layers=1`, launched per layer): `static`; `static_ps` (R2 / R4 / R5); `static_irf`; `static_ir`;
    `static_dl` (deadline edges, 0 timeouts, fatal word 0); `static_scope` (Rust scopes).
  - Path A (loop driver): `loop_const` (D0 defaults, R1-R5); `loop_lds`; `loop_lds_dl`; `loop_legacy`;
    `loop_nocarry`; `loop_irf`; `loop_dl`.
- The broken-tag mutant fails as required in both paths: 114-204 poll timeouts, `poll_timeout` in the fatal word,
  0-1 of 12 layers matching, and no hang. **Superseded (Fix round F1):** the numerics of this mutant are a race in
  both paths (up to 12/12 layers correct); the verdict is now the launch-fatal word, checked after every Path B
  launch, and a separate mutant (`wrongw`) fails numerics deterministically.

**`grid_sync` vs P4's `s8L` kernel** (P4's harness, P 256):
- **Positives** pass under idle, uniform and one-late arrivals.
- **Dropped-arrival control** passes for gs8, gs8w and gs1: earlier epochs released, the dropped epoch timed out
  on all 256, every later epoch aborted, `sync_timeout` set.
- **Early-release mutant** fails (about 49k mismatches and 100 order violations).
- **gs8 timing**: at idle and with one late producer it is within the controls (noise); under uniform arrivals it
  wins slightly (0.92-0.94, 83-100% of rounds below 1). Both nodes.
- **Per-wave arrivals (gs8w) lose 1.6-3.9x on both nodes.** This confirms P9's per-CTA rule at the primitive level.
- **gs1** reproduces P4's one-counter-per-XCD pattern: a WIN when arrivals are bunched, a LOSS of 1.88-2.00 when
  they spread.

**Ticket with the fatal word:**
- grid 256: clean.
- grid 257: 8-85 CTAs time out, with `ticket_timeout` set.
- grid 512: all 256 time out, with `ticket_timeout` set.

## 6. Byte identity of every existing build

The base is a pristine snapshot of the copy-rocm-lib rocKE tree, synced to the cluster before the first WP1 edit.
The new tree is that snapshot plus exactly the 28 WP1 files (`stage_cpu/make_overlay.sh`), so concurrent edits by
other agents cannot leak into the comparison. Lowering ran with `PYTHONHASHSEED=0` and flavor `llvm22`, on CPUONLY
jobs. `ir_diff` compares byte for byte, with no CRLF normalisation.

| set | files compared (LLVM `.ll` + serialized `.ir`) | result |
|---|---|---|
| every `library/tests/parity/*_emit.py` family + dense / problem / dispatch / gfx1250 extras, gfx950 | 821 (349 builds; 123 recorded build errors, identical in both trees) | BYTE-IDENTICAL |
| same, gfx942 | 768 | BYTE-IDENTICAL |
| same, gfx1151 | 860 | BYTE-IDENTICAL |
| same, gfx1201 | 500 | BYTE-IDENTICAL |
| true-mono / qwen38_mono builds: `rust-track/rt_builds.py` labels (B3 mono / tuned / `_ir`, fused MoE r1 / r4 / stamps, Q2, toy, P2 / P4 / P5, CCL, framework layer kernels incl. whole resident layers) + P9 tails + toy-chain best specs + plain heads rows 4 / 16 | 197 (87 builds; 22 recorded errors, identical in both trees: `irops` heads and `_ir` toys exist only in the rust tree, and heads have no rows 1 / 8) | BYTE-IDENTICAL |

- **Final run.** The last runs (68233569, 68233628) used the final overlay. CPU tests on the WP1 tree: 869 passed
  (qwen38 framework, stage adapter with the ATOM layout, mailbox `ir` / `irf`, P2 / P4 / P5 / P9, tune, stage_fn,
  MoE, platform `persistent_mem_ops` / `task_schedule` / `mailbox_helpers`, persist). Plus the 307 search-engine
  tests with M1-M4 applied.
- **C++ engine.** The C++ byte-identity test needs the `rocke_engine` binding and skips in both trees. The persist
  builds are Python-engine builds, because they carry kernel attributes and ops that the C++ engine does not lower.
  This matches the family README's cpp-parity exemption: names `q38pt_` / `q38wp1_`, and `mbirf` for the `irf`
  mailbox.

## 7. INVENTORY merge items M1-M12

| item | status |
|---|---|
| M1-M4 (engine hooks in `core.py`, `candidate.py`, `autotune.py`, `sweep.py`) | **applied** to the tracked files, CRLF kept. Byte identity holds (section 6); 307/307 search-engine tests pass |
| M5 (`ROCKE_CODEOBJ_BACKEND` / `ROCKE_DIRECT_LEVERS` in the cache fingerprint) | **deferred, by its own condition**: the direct backend is not in the copy-rocm-lib tree yet (WP-R3 ports it). The tuner's `environment()` already records both variables |
| M6 / M7 (docs) | **applied**: the `.search` docstring bullet, ARCHITECTURE section 14 "Searching instead of sweeping", and the `autotune/overview.md` search-engine section, copied from rocm-libraries (read only) |
| M8 (GEMM learned selector) | **not taken in WP1**: it belongs to the GEMM lane. The mono `tdb.Selector` runs without it (`lightgbm` optional) |
| M9 (mutation testing) | **done** (`stage_cpu/wp1_mutate.py`, the tune-engine `mutate_auto` method, 15 workers). 92 of 200 sampled mutants were killed in the mono tuner, and 142 of 240 in the new `persist` package, up from 108 / 240 before the mutation-driven tests were added. By file: `knobs` 19/21, `kv` 9/9, `parity` 17/20, `lint` 37/52, `sync` 24/39, `isa_gate` 15/30, `driver` 8/19. Most survivors are in GPU-only code (`micro` 3/25, `ptoy` 10/25, the tuner's `live_toy` 0/20, `evaluate` 2/15): constants and branches that only a GPU run exercises. **Superseded (Fix round F4):** the claim that the hardware gates cover these paths was false when tested (22 of
58 CPU survivors killed on hardware, 3 of 15 in `sync.py`). The gates and tests were extended; the final numbers
are in F4 |
| M10 (toy-chain blind spot) | **fixed and proven on hardware, both nodes**. `toy_chain_bench.check_launch` / `run_chain_checked` compare every launch's mailboxes. A fault planted in the earlier launch (depth 8, P = 160) is invisible to the old check and caught by the new one. The clean chain passes both |
| M11 (`best_spec(8, 8)` picks) | **Withdrawn (Fix round F3): under a 9.1 decision against `best_spec(8, 8)` itself, every pick is NOISE on both nodes.** The original claim follows: confirmed on fresh allocations on both nodes with the tuner's own confirmation (gate mutants first, A-vs-A + recompiled controls, 12 shuffled rounds). Five of the six picks of job 68225392 beat the recorded optimum on both nodes: 0.960-0.979 of it, every round below 1. The sixth won on n01-09 only (1.002 on n07-17). The picks share `mailbox impl=asm`, `numa_copies=1` and `step_first`. **`best_spec` is not changed**: that would change the bytes of existing toy builds. The owner can now adopt a pick (e.g. `pickA_3` / `pickA2_2`) as a versioned spec |
| M12 (`Q38Transport` on the cluster) | **ran end to end** (`wp1_m12.py`): one round of 4 candidates in one cluster job, the gate self-test caught both mutants, every item measured. The first attempt found a transport bug (the venv job lacks `ROCKE_HIP_LIB`, so the runner's GPU check fails). Fixed in `tune/evaluate.py`: the command exports torch's HIP library |

## 8. Bugs found on hardware (fixed)

1. **Address-taken LDS words were overlaid.**
   - What happened: the `args=lds` loop hung on both nodes.
   - Mechanism: the R5 workgroup-id word and the driver's argument rows are used only through inline asm on their
     integer address. The lowering's LDS liveness does not see those uses, so the pool packer overlaid one word on
     the other. The packed pool was 48 bytes for 52 bytes of live data.
   - Fix: the framework and the driver re-issue `smem_addr_of` for every such allocation before `ret`.
   - Test: the pool sizes must hold every live word, and the test fails when the pin is removed.
   - Latent elsewhere: any build that mixes an asm-addressed LDS word with a later allocation is exposed (recorded
     in alola rule 175).
2. **Argument-row overwrite.** The driver's `args=lds` copy for layer `l + 1` could overwrite the row a slow wave
   of layer `l` still reads. The rows are now double-buffered by layer parity, with one barrier per layer.
3. **Harness.** A hung kernel also hangs the Python process at exit; the bench now `os._exit`s and every section
   runs under `timeout -s KILL`. Also, `Q38Transport` lacked `ROCKE_HIP_LIB` (M12).

The ptoy results in section 5 are from the fixed code; the two jobs that hung were cancelled. The P4 / P5 / M10 /
ISA-gate runs on n01-09 (68233230) ran before fixes 1 and 2. They do not use the driver's `lds` path, and the
n07-17 job (68233512) re-ran them on the fixed code.

## 9. Not done in WP1 (explicit)

- **Emitter-level edge knobs are expressible, not implemented.** `protocol` `bulk` / `bulk_sc1` / `xcd_local` /
  `fence`, `placement` `xcd_affine` / `island` / `hier`, and `queue` `xcd_queue` / `global` are validated, carried
  in names, and enforced as consumed. The stage emitters and table builders of WP2 / WP3 implement them
  (XCD-SCHEDULING levers 1, 4, 5, 11). The framework applies only `poll`, `pause`, `polarity` and `deadline`.
- **Not built:** the per-variant run-ahead hold, A1 streams-first (WP2 / WP3 stage knobs), and a framework-level
  delay-injection stress harness. Delay injection exists in the microbench kernels (`dtab`) only.
- **R2 granularity is per stage, not per use site.** In the loop driver, stage stamps, `numa_flip_arg` and `entry`
  hooks are refused explicitly.
- **No timing of Path A vs Path B in WP1.** ptoy is a numerics rig; D0.7's switch test is WP3a. The loop driver
  makes that test, and DESIGN-LB WP-L4's `pass`, buildable on rocKE bodies.
- **Rust lever scoping is a hook only.** Ops carry `attrs["rocke.scope"]` and the kernel carries
  `attrs["direct_scopes"]`; COMGR objects are unchanged (test). WP-R3's direct backend consumes the hook.
- **The exec-join lint is report-only by default** (section 4).
- **Independent verification:** done, verdict PARTIAL (section "Verification"). The "Fix round" section closes
  every issue it raised.

## 10. Jobs

- CPU (CPUONLY):
  - identity: 68233002, 68233004, 68233005, 68233032, 68233033, 68233139, 68233138, 68233228, 68233569, 68233628;
  - tests: 68233006;
  - mutation testing: 68233144 (lost its summary: missing output dir), 68233227, 68233570.
- GPU:
  - 68233230 (n01-09: p4, p5, m10, isagate);
  - 68233512 (n07-17, bus 149: wp1, p2fix, m10, isagate, p4, p5);
  - 68233513 (n01-09, bus 21: wp1, p2fix, m11, p2neg);
  - 68233514 (n07-17, bus 245: m11);
  - 68233503 (M12).
- Superseded: 68233070 and 68233231 (the LDS-overlay hang), 68233490 (M12 without `ROCKE_HIP_LIB`).

## Verification (independent verifier, W5 vWP1, 2026-10-31)

**Verdict: PARTIAL.** Byte identity, the CPU tests, P4 on the platform, `grid_sync`, the ticket, the `tagged_poll`
reload fix with its hang control, and M10 all reproduce. **The ptoy hardware gate does not reproduce.** On a third
GPU the Path B broken-tag mutant gives correct numerics, so `[ptoy] gate FAIL`. WP1 is not fully gated until the
fix below is re-run on both nodes. Absolute data is in `private-perf/.../wave1/WP1-raw/w5-vWP1-*`.

**1. Inputs are what the report says.** All reads used read-only git commands.
- The cluster base `w5wp1-base` equals `git HEAD` for all 11 WP1-touched tracked code files. `moe_fused.py` and
  `stage_moe_fused.py` were changed by other agents; they are the same in the base and the current tree, so they
  cannot leak into the diff.
- The 28-file overlay equals the current copy-rocm-lib tree byte for byte.
- The builder's GPU stage (`w5-wp1-rocke`, synced 19:24 UTC) differs from the final tree only in three files, none
  used by a hardware kernel:
  - `persist/lint.py`: `resource_gate`'s exec-join became report-only at 19:35 UTC, after the GPU runs. Section 4
    discloses the policy. The `p2fix` lint prints its own spill check, so its output is unaffected.
  - `test_qwen38_persist.py`.
  - a W6 `large_batch` file.

**2. Byte identity, re-run with must-fail mutants** (CPUONLY job 68233690). The identity tool is the builder's; the
overlay was rebuilt fresh from the current tree.
- Clean, set q38: 197/197 files identical (87 builds; the same 22 errors in both trees).
- Mutant A (the `tagged_poll` pause gets `~{memory}` unconditionally): DIFFERS, 43 files. These are the `.ll` of
  every `ir`-mailbox / tagged-poll build (`b3_*_ir`, `b3_*_tuned`, `fmoe_*`, ...).
- Mutant B (the framework sets `machine_licm="off"` on every layer kernel): DIFFERS, 24 files, **only in the
  serialized `.ir`**. The LLVM IR is unchanged.
  - So a knob that acts only through COMGR options (R4) is caught by the serialized-IR half of the census alone.
  - The census does compare that half, so the claim holds. The ISA itself is not compared.
- Tests: 869 passed / 182 skipped (qwen38 and platform); 307/307 search-engine tests.
- The 2,949 parity-family files were not re-run. The 68233569 log shows both trees built 349 / 320 / 397 / 218
  kernels with zero differing.

**3. Hardware re-run** (job 68233733, n07-17, GPU bus 21, which neither builder job used). A fresh stage of the
current tree ran the sections `wp1 p2fix p4 m10 p2neg`.
- **P4 on the platform: gate PASS.** Every dropped-arrival, deadline and oversize-ticket control and every
  early-release / skip-last / `tagoff` / `noready` mutant fails as required. The ratio pattern matches section 5.
- **`grid_sync`**:
  - Positives pass. The dropped-arrival controls pass for gs8, gs8w and gs1. The early-release mutant fails.
  - gs8 vs `s8L`: noise at idle and with one late producer; a WIN under uniform arrivals at 0.945-0.955, below both
    controls' minima.
  - gs8w loses 1.68-4.03x. gs1 wins when bunched and loses 1.93-2.05x when spread.
  - All of this reproduces the report.
- **Ticket**: grid 256 is clean; 257 times out 33 CTAs with `ticket_timeout`; 512 times out all 256.
- **Reload fix**:
  - The lint finds 2 / 1 non-reloading spins in the original builds and 0 in the fixed ones.
  - The fixed `pairs_ir` and `bulk_ir` builds pass 24/24 launches.
  - The original `pairs_ir` hung and was killed at 90 s. This now reproduces on n07-17 as well (the builder showed
    it on n01-09 only).
- **M10** reproduces: the old check is blind to the planted fault, and the per-launch check catches it.
- **ptoy**:
  - All 13 positive arms (6 Path B, 7 Path A) equal the host reference on 12/12 layers and are bitwise equal across
    arms, in all 3 steps.
  - `loop_mut` fails as required (204 timeouts, 0/12).
  - **`static_mut` DID NOT FAIL in 3/3 steps**: 192 timeouts and `poll_timeout` set, but **12/12 layers equal to the
    reference**. The bench requires `not all(match)`, so the gate fails (section rc=1).

**4. Root cause** (per-layer data of all three runs).
- In Path B each layer is its own launch. When `g`'s deadline poll times out, it returns the stale `hx` payload.
  That payload is the correct output of the previous layer: the previous launch has completed, and only its tag is
  one behind.
- So numerics fail only if `r` (CTA 16, the same deadline) times out first. Then `r` reads stale `y` and publishes
  `hx` under the current tag. The `g` waves still polling accept it, and every layer goes wrong.
  - The builder's runs: 90 / 144-150 timeouts (fewer than 192) and 0/12 layers matching.
  - This run: exactly 16 x 12 = 192 timeouts. Every `g` wave timed out, `r` never did, and 12/12 layers matched.
- The outcome is a race between two equal deadlines, and it flips with the GPU.
- Consequences:
  - The claims "0-1 of 12 layers matching" and "ptoy gate PASS" are not reproducible. `static_mut` is not a
    reliable must-fail mutant for numerics.
  - **Substantive for D0 Path B.** A broken tag invariant on an edge whose producer ran in an earlier launch can be
    numerically invisible. Here only the launch-fatal word caught it.
  - The fatal word set `poll_timeout` in every mutant launch of every run, in both paths, and nothing hung.
    Checking the fatal word on every Path B launch must therefore be mandatory, not an option.
- **Required fix**, then re-run the ptoy gate on both nodes:
  1. Key the mutant verdict on `timeouts > 0` and `poll_timeout`.
  2. Add a mutant that numerics must catch deterministically, for example a wrong parity slot or a wrong layer's
     weights.

**5. Other observations**
- **M11 is weak evidence.** The `WIN` verdicts are against the shipped spec. The "vs recorded best" ratios
  (0.960-0.979) are not 9.1 verdicts. The A-vs-A control sat at 1.024-1.044 on n01-09, and the recompiled control
  at 1.006-1.032 on n07-17: the same size as the claimed 2-4% margin. Not adopting the picks was correct.
- **Mutation testing.** Survivors in `sync.py`'s `grid_sync` / `cta_arrive` (for example the abort-flag return at
  line 335) are said to be "covered by the hardware gates". No mutant was ever run on hardware, so that is unproven.
- **Timing discipline is OK for P4 and gs**: shuffled interleaved rounds, two controls, unstamped builds, node and
  bus recorded.
- **No new alola fact** beyond rule 175, except that n07-17 also has a GPU at decimal bus 21. That was not added to
  the skill: the skill was being edited by another agent at the time.

## Fix round (W5, 2026-10-31)

**Gate: met.** Every issue of the verification is fixed, and the fixes were re-run on both rocke nodes, on 6 GPU
allocations. Raw logs, with absolute values only in M11's JSON, are in `private-perf/.../wave1/WP1-raw/fix/`. No git
state was changed. All edits are additive or knob-gated, Python only, and LF.

| issue (verification) | fix | evidence |
|---|---|---|
| 1. ptoy mutant verdict not reproducible | the verdict is keyed on the launch-fatal word, checked after every Path B launch; new deterministic numerics mutant `wrongw` | F1: 63 steps on 6 GPUs, all PASS |
| 2. the root cause is a race | confirmed, and it also occurs in Path A | F1 |
| 3. the fatal-word check must be mandatory on Path B | `sync.check_launch`, `knobs.require_launch_fatal`, the D0.8 amendment in `D0.md`, the per-launch check in the bench | F2 |
| 4. M11 is weak evidence | re-run as a 9.1 decision with `best_spec(8, 8)` as the baseline: NOISE for every pick on both nodes; the claim is withdrawn | F3 |
| 5. GPU-path survivors "covered by hardware" unproven | hardware mutation testing (`wp1_hwmut.py`). The first run disproved the claim and found a latent `cta_arrive` bug. Gates and tests were extended | F4 |
| 6. exec-join lint changed after the GPU runs | every fix-round GPU run used the final tree; the `p2fix` lint verdicts are unchanged on both nodes | F5 |
| 7. the census does not compare ISA | ISA (`.text`) half added; a compiler-option-only mutant is caught only by it | F6 |

### F1. The ptoy gate (issues 1 and 2)

**The race is confirmed, and it occurs in both paths.** A timed-out deadline poll returns the stale `hx` payload,
which is the previous layer's correct output: in Path B it comes from the previous launch, in Path A from the previous
loop iteration. The layer is wrong only when `r` times out before the `g` waves. Over the 63 fix-round steps
(6 GPUs: n01-09 buses 245, 21, 5; n07-17 buses 117, 5, 21):
- `static_mut` (Path B) had all 12 layers correct in 45 of 63 steps and 0/12 in the other 18. Its timeouts were
  192 (every `g` wave) or 76-158.
- `loop_mut` (Path A) had 8-12 of 12 layers correct in 7 of 63 steps, all on n07-17. The original verdict would
  have failed Path A too.

**Fix** (`persist/ptoy.py`, `wp1_bench.py`):
- **`prevtag` verdict.** The mutant must report timeouts > 0 and `poll_timeout` in the fatal word. In Path B the host
  calls `sync.check_launch` after EVERY launch, and all 12 launches must raise `LaunchFatal`. Numerics are recorded,
  not required.
- **New mutant `wrongw`.** Task `t` uses weight row `(t + 1) % 16`: no protocol change and no timeout. Every layer must
  differ from the reference and equal the host twin of the mutant (`reference` with the rows rolled). It is built in
  both paths.
- **Positive deadline arm.** `static_dl` must also report every launch clean.

**Results.** All 6 jobs pass the gate: 68233999, 68234027, 68234224, 68234249, 68234591 and 68234832. The four
two-section jobs also ran `ptoy3`, which is three more fresh-process runs of the ptoy gate.
- All 13 positive arms equal the reference on 12/12 layers, bitwise equal across arms, in every step. `static_dl`
  launches are clean 12/12 in every step.
- `static_mut`: `check_launch` caught 12/12 launches in 63/63 steps. `loop_mut`: `poll_timeout` was set in 63/63
  steps.
- `wrongw`: 126/126 arm-steps fail as required (0/12 equal to the reference, 12/12 equal to the mutant twin).
- No hang.

The measured kernels are unchanged. The six ptoy builds of section 5 (`static`, `static_dl`, `static_mut`,
`loop_const`, `loop_lds`, `loop_mut`) are byte-identical in LLVM IR to the sources the builder and the verifier ran
(`fix/cmp_old_ptoy.py` against stage `w5-wp1-rocke`).

### F2. The Path B rule (issue 3)

- **`persist.sync.check_launch(err, what)`** raises `LaunchFatal` (with the reasons and the timeout count) when
  `err[ERR_FATAL]` is set. Timeouts without the fatal bit are counted, not raised.
- **`persist.knobs.require_launch_fatal(fk)`** refuses a Path B build without `err_arg`, or with any edge (the
  default or a named one) that is not a deadline poll.
- **`D0.md` D0.8 amendment** (additive, below the decision table): any build with a deadline edge must have its
  launch-fatal word checked by the host after every launch or every graph replay. The word is a sticky OR, so one
  check per replay covers every launch in it.
  - Validation and gate runs of Path B build every edge with a deadline and `err_arg`.
  - Without deadline polls, a broken tag hangs, which is visible.
  - Whether served builds keep deadline edges is a WP3a timing question.
- Tests: `test_check_launch_raises_on_the_fatal_word_only` and `test_require_launch_fatal_path_b_rule`.

### F3. M11 (issue 4): withdrawn

`wp1_toy.py m11b` runs the tuner's own 9.1 confirmation (`ToyChainBackend.confirm`) with `best_spec(8, 8)` as the
baseline. The A-vs-A and recompiled controls are therefore the recorded best's own. It ran 30 shuffled rounds, the
gate self-test passed, and the post-check was OK.

| node / bus (job) | A-vs-A median | recompiled median | picks (6) median vs best_spec | rounds below 1 | verdict |
|---|---|---|---|---|---|
| n07-17 / 117 (68234101) | 1.084 | 1.078 | 1.040-1.059 | 0.00 for every pick | NOISE x 6 |
| n01-09 / 21 (68234102) | 1.091 | 1.069 | 1.032-1.083 | 0.00 for every pick | NOISE x 6 |

- No pick beats `best_spec(8, 8)`, so the "0.960-0.979" claim is withdrawn and `best_spec` stays unchanged.
- **New observation, for the tuner owner.** In this rig the baseline arm runs about 7-9% faster than an identical
  copy of itself (A-vs-A 1.06-1.10 over 30 rounds). The win rule is protected, because it requires >= 0.75 of
  rounds below 1. A ratio of ratios between two non-baseline arms (the original M11 method) is not protected: it
  inherits that bias.

### F4. Mutation testing on hardware (issue 5)

**Tool.** `wp1_hwmut.py` regenerates exactly the CPU run's surviving mutants: the same sampler, seed and `n`,
matched on file, line and text. It writes each mutant into a private tree copy and runs the gate that exercises the
file, with gate only and no timing:
- `sync.py`: ptoy, ticket and gs;
- `micro.py`: ticket and gs;
- `ptoy.py` and `driver.py`: ptoy.

A mutant counts as killed when the gate fails, the bench raises, or the kernel hangs. The unmutated tree must pass
before and after the run.

**Round 1 (68233973, pre-fix sources) disproved the claim.** Only 22 of 58 CPU survivors were killed on hardware,
and only 3 of 15 in `sync.py`. The analysis found:
- **Untested API.** `cta_arrive` and `aborted` were used by no gated kernel. `cta_arrive` is the last-wave form of
  D0.8's per-CTA arrival (`PersistKnobs.arrivals="cta"`, P9 `ectr=cta`). `GridSync(arrivals="cta")` is the
  thread-0-after-barrier form.
- **A latent `cta_arrive` bug.** Its LDS counter (`LDS_ARRIVE`, word 2) was never zeroed: `pad_lds` zeroes words 0, 1
  and the last word only, and LDS is not cleared between launches. A non-multiple-of-8 start value makes a CTA
  arrive from a non-last wave, before the later waves stored.
- **Unchecked timing metadata.** The gate did not check the out stamps, column 3, the XCC ids or the info clocks.
- **Untested defaults and boundaries:** the `grid_sync` / `grid_sync_wait` defaults, `GridSync(maxit=1)`, the
  `TableParams` cache, `per_cta` with tasks == grid, the default `rows`, and the `launch_tag` i32 wrap at exactly
  2^31.

**Fixes:**
- **`sync.py`:**
  - `pad_lds(arrive_word=True)` zeroes the counter. The default keeps P4's bytes.
  - `GridSync(arrivals="last")` is `cta_arrive` from every wave, with no CTA barrier before the arrival (its real
    use). The previous sync's closing barrier orders the abort-word read.
- **`micro.py`:**
  - The gs kernel has a `last` mode in which every wave stores and checks its own data word, with an optional
    per-wave `stagger`.
  - New mutant `noinit` (the counter is not zeroed).
  - New `build_lds_poison_kernel`.
- **`wp1_bench.py` gs gate:**
  - New `gs8l` arm in the positives and the dropped-arrival control.
  - LDS-poisoned launches: `gs8l_st` must pass 3/3 and `gs8l_noinit` must fail 3/3.
  - A metadata check on every gs arm.
- **13 new CPU tests in the fix round in total**, 3 of them for F1 and F2. `test_qwen38_persist.py` now has 110.

The kernels measured in section 5 are unchanged: `gs8`, `gs8w`, `gs1`, `gs8_rc`, `early` and the ticket are
byte-identical in LLVM IR to the measured sources (`fix/cmp_old_micro.py`).

**Final hardware results** (68234591 on n01-09 bus 5, 68234832 on n07-17 bus 21): every new check passes, and
`gs8l_noinit` fails 3/3 on both nodes, with 0.69-1.08 M mismatches per launch. The earlier hardware mutation run
shows why this check is needed: without the poison, the `noinit` behaviour (mutant `micro.py:91`) survived on
hardware.

**Final numbers** (sources as shipped; CPU job 68234599, hardware job 68234720 on n01-09 bus 5):

| stage | killed | remaining |
|---|---|---|
| CPU tests (`test_qwen38_persist.py`, 110 tests), 240 sampled mutants | 156 (`driver` 12/17, `isa_gate` 12/25, `knobs` 18/22, `kv` 8/8, `lint` 39/52, `micro` 9/34, `parity` 15/19, `ptoy` 14/23, `sync` 29/40) | 84 |
| hardware gates, on the 50 CPU survivors in `sync` / `micro` / `ptoy` / `driver` | 27: hangs, illegal memory accesses, compile errors and gate failures, every one with its gate line | 23 |
| one test added after the run (`cta_arrive(waves=1)` must lower) | 2 (`sync.py:380` `<` to `<=` and `1` to `2`; checked locally: both fail the test, the clean source passes) | 21 |

The 21 remaining GPU-file survivors are all equivalent or benign; none is a protocol gap:
- **Same behaviour for every admitted input:**
  - `frozen=True` (x2);
  - the payload formula constants (x2: writer and checker share the formula);
  - the masked-lane selects (`sync.py:484`, `486`; `ptoy.py:102`);
  - the abort word value 1 to 2 (any non-zero aborts);
  - `word_of`'s `>> 3` to `>> 4` (another balanced stripe map);
  - `_barrier_points(specs, 2)` to `3` (the same steady state);
  - `layout(1)` to `layout(2)` and `_specs(True)` to `False` in ptoy (tags enforce the order the `Dep` declares);
  - the `ERR_MISMATCH` increment 1 to 2 and `4 * 0` to `5 * 0` (x2; the gate tests == 0 / > 0).
- **Still a must-fail or still a check:**
  - the `early` mutant's bias 1 to 2 (it still fails);
  - the stagger loop start and sleep length (x2; `noinit` still fails);
  - the poison loop bound (x2; it still covers word 2).
- **Benign:** the start / end clocks taken by CTA 1 instead of CTA 0.

The other 34 survivors are in `lint`, `isa_gate`, `knobs` and `parity`, which have no hardware path. They remain CPU
survivors, as in section 7. The tune package was not re-run (no tuner code changed).

### F5. The exec-join lint (issue 6)

Every fix-round GPU run staged the final `persist/lint.py`, in which exec-join is report-only. The `p2fix` lint
section ran on both nodes in rounds 1 and 2:
- the original `ir` builds spin without reloading (2 / 1 loops);
- the fixed builds have 0 such loops;
- the spill / scratch gate is OK;
- the fixed builds pass their launches.

The verdicts are unchanged.

### F6. ISA in the byte-identity census (issue 7)

- **Tool.** `wp1_identity.py --emit isa` compiles every build with the Python engine's `compile_kernel`, so kernel
  attributes become COMGR options. It writes each code object's `.text` and compares it byte for byte. It runs on
  CPUONLY nodes with `AMD_COMGR_CACHE=0`, and the build order is the same in both trees.
- **Clean.** Base vs the final WP1 tree, true-mono set (jobs 68233885, 68234131, 68234603; the last on the final
  overlay): **87/87 `.text` identical and 196/196 LLVM IR + serialized IR files identical** (the same 22 recorded
  errors in both trees).
- **Must-fail mutant B'.** `helpers/compile.py` adds `-disable-machine-licm` to EVERY kernel: a compiler-option-only
  change, with no kernel attribute.
  - The IR half is identical, so it is blind to this mutant, as expected.
  - The ISA half DIFFERS in 34 of 87 builds (CCL, fused MoE, heads, P2 / P4 / P5, P9, the toy chain), so it is caught.
- **Parity families not re-run.** The fix-round edits touch only `persist/{ptoy,sync,knobs,micro}.py` and the
  persist tests. No parity emitter imports them, and `framework.py` is unchanged since the census of section 6.

### F7. Other findings

- **rocKE: a repeated `scf_for` `iv_name` does not compile.** Two `persist.sync.delay` loops in one kernel lowered to
  duplicate SSA names (`%p4d`), and COMGR failed with `COMPILE_SOURCE_TO_BC` and no message (job 68234519). The
  stagger loop has its own name, and a test checks that SSA names are unique. This is recorded in the alola skill
  (rule 181).
- **LDS persists across launches on MI355X**, as the poison runs showed. This is also in rule 181.

### Files (fix round)

- **rocKE (copy-rocm-lib):**
  - `persist/sync.py`: `check_launch`, `LaunchFatal`, `pad_lds(arrive_word)`, `GridSync(arrivals="last")`;
  - `persist/knobs.py`: `require_launch_fatal`;
  - `persist/ptoy.py`: the `wrongw` mutant and the docstring;
  - `persist/micro.py`: the `last` mode, `stagger`, `noinit`, the LDS poison kernel;
  - `tests/test_qwen38_persist.py`: 97 tests became 110;
  - `experiments/qwen38-true-mono/D0.md`: the D0.8 amendment.
- **Harness (`hackathon-prep/qwen38/truemono/WP1/`):**
  - `wp1_bench.py` (`--no-timing`, the per-launch check, the new arms and checks);
  - `wp1_hwmut.py` (new), `wp1_toy.py` (`m11b`), `wp1_job.sh` (sections `gate`, `ptoy3`, `hwmut`, `m11b`);
  - `stage_cpu/wp1_identity.py` (`--emit isa`), `stage_cpu/wp1_cpu_job.sh` (`isaident [mutB]`);
  - `fix/` (the edit scripts, `cmp_old_micro.py`, `cmp_old_ptoy.py`).
- **Private:** `private-perf/qwen38-true-mono/wave1/WP1.md` (fix-round job index) and `WP1-raw/fix/`.
- **Alola skill:** rule 181.

### Jobs (fix round)

- **CPUONLY:**
  - tests and mutation testing: 68233879, 68234129, 68234599;
  - ISA / IR census: 68233880, 68233885, 68234131, 68234603.
- **rocke GPU:**
  - gate and ptoy: 68233999, 68234027, 68234224, 68234249, 68234591, 68234832;
  - hardware mutation testing: 68233973, 68234219, 68234720;
  - M11: 68234101, 68234102.
- **Superseded:** 68233869 and 68233875 (the `cluster_job.sh` argv mistake, rc 127), 68233880 (no mutant-B counts),
  68234519 (the duplicate `iv_name`).
