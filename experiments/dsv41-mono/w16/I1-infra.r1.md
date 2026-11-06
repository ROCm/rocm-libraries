# W16 I1: fast-iteration infrastructure for the DeepSeek-V4.1 rocKE path

W16, 2026-11-05. MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (`--level 0` + FULL graphs), atom-dev image
`nightly_202610171506` (FlyDSL 0.4.0).

- **Ratios only.** Absolute seconds and rates are in
  `hackathon-prep/private-perf/dsv41-mono/w16/I1-abs.md`.
- **Usage and code map.** See `hackathon-prep/dsv41/w16/HOWTO.md`.
- **Jobs.** a1 / a2 / a3 / x1 / x2 are GPU-less, b1 / b2 are servers and gates, c1-c6 are the rig. Raw outputs are in
  `hackathon-prep/dsv41/w16/results/<tag>/`.

The task had three parts, PLAN decision 12 plus alola rule 262. Each is built and measured, and each correctness
claim has a control that could have failed:

- **(a)** a persistent code-object cache whose hits are byte-identical to fresh compiles;
- **(b)** runtime arm switching inside one server;
- **(c)** the in-situ rig as the cheap rung of the search engine, with a served rung on top of (b).

## 0. Verdict

| what | before | after | ratio |
|---|---|---|---|
| Server start, rocKE hybrid arm `ext2` (K2b MoE + seam gate/norm externs) | status quo: FlyDSL never caches an extern-linked build | `ATOM_DSV41_RK_COCACHE` + `_FAST` | **0.228** of the status-quo start (4.4x faster) |
| Graph capture plus mono builds in that start | | | **0.025** |
| Same start, cache in IR mode only (the trace still runs) | | | 0.724 |
| Cached hybrid start vs ATOM mono's warm start (FlyDSL disk cache) | | | 0.957: a hybrid arm now starts as fast as ATOM mono |
| Next arm in the same server, kernels in the cache | a fresh server per arm-run | `POST /rk/switch` | **0.015-0.017** of the status-quo extern start; 0.065-0.073 of the cached fresh start |
| Next arm in the same server, never compiled | fresh server, cold | switch, cold | 0.42-0.43 |
| One A/B over 6 arms (2 cold, 4 warm) with probe and windows, measured (b1) | every arm-run a fresh status-quo start | one server, switched | **0.215** of the wall time (0.43 of fresh servers with the cache) |
| Candidates a minute | a fresh-server arm-run per candidate | in-situ rig, warm (functions and kernels cached) | about 500-700x |
| | | in-situ rig, cold (new rocKE functions and new kernels) | about 75x |
| Stage-function generation (795-function MoE set + 4 others) | regenerate | generation-cache hit | about 0.07 of the regeneration wall |

### Correctness

Every claim has a fresh control, a mutant or a twin.

**Cache hit equals fresh compile, 280 controls, 0 differ.** The fresh-compile control (`verify`) compiles every
artifact again in a new process and compares the compiled module text and the code object, byte for byte:

- 68 controls in a1 (2 arms x 17 builds x 2 runs, reversed and shuffled orders);
- 144 in b1: every artifact a server builds, both ranks;
- 34 in a2, across jobs on the same node;
- 34 in x1, on another machine type (a Markham CPU-only EPYC node) against entries compiled on n01-09.

Every comparison is `verify_SAME`. Five fresh servers of the hybrid arm (no cache, cold store, IR hit, fast hit,
verify) ran the same 144 code objects, as did the two ATOM mono servers.

**Knob-off identity.** With every W16 knob unset, the hooked trees build K1 / K2 byte-identical to the tree before
the hooks: 17 builds x 2 arms (ATOM mono and the hybrid), in both build orders. a1 checked tree i1a. a2 checked
tree i1b, including a process with `ATOM_DSV41_RK_SWITCH=1` set. The hooks did not change ATOM's own build either.

**Generation cache.** Regenerating in `verify` mode gave the same bytes in every job (6 of 6 generators,
4,022 files), on both machine types.

**rocKE HSACO cache.** 22 standalone stage kernels (11 COMGR, 11 Rust direct) were compiled in 5 processes: fresh,
two hit runs and two verify runs in two orders. Every run gave identical code objects, and every verify was SAME.
This held on the rocke node (a3) and, with COMGR's own on-disk cache off (`AMD_COMGR_CACHE=0`, so every
non-hit really compiles), on the CPU-only node (x2): 44 more fresh controls. A hit's compile step is under 0.001 of a
fresh compile's; Python lowering still runs.

**Switched arm equals the same arm in a fresh server.** Three checks:

- *Code objects.* The switched server's code objects equal a fresh server's, 32/32 (runner, width, rank) sets and
  144 objects, for ATOM mono and the hybrid. Repeated switches to the same arm are identical too.
- *Bitwise launch gate.* In an eager engine, after each switch, the in-place gate (`ATOM_DSV41_RK_LEV_CHECK`)
  compared every byte of 11.6-12.7k K1 / K2 launches (about 26 MB a launch, scratch included, LDS poisoned) and
  found 0 differing launches, three times. A mutant arm switched in (down `put_bit`) differed on 6,240 of 12,480
  launches: all of its K2 launches. The fresh-engine control of the hybrid arm differed on 0 of 12,664.
- *Tokens.* Greedy replies of the switched arm agree with a fresh server's on 2-8 of 16 prompts. Five fresh servers
  of the same arm, with identical code objects, agree with each other on only 2-7 of 16. The served V4.1 path is
  not run-to-run deterministic (alola rules 248 / 251 / 252), so token identity is not testable, and the switched
  arm sits inside the fresh-vs-fresh band.

**Rig.** Every parity variant wrote 0 differing bytes in every batch. Both down mutants (`put_bit`,
`scale_byte`) were caught in every batch of c1-c6.

## 1. (a) The code-object caches

### 1.1 What is cached, and the key

**ATOM / FlyDSL.** `ATOM_DSV41_RK_COCACHE=<dir>` (`atom/models/deepseek_v41/rk/cocache.py`, installed from
`mono/kernels/__init__.py`, off by default) wraps FlyDSL's `MlirCompiler.compile`. Its key is the sha256 of:

- the module text exactly as the compile parses it (the IR; every ATOM build knob reaches the kernel through it);
- the content of every linked rocKE `.ll`;
- FlyDSL's toolchain fingerprint (`_flydsl_key`: every FlyDSL source, the native libraries with the embedded LLVM,
  the version, the backend);
- the arch, the pass pipeline the backend builds for the current hints, the LLVM options and the hints;
- FlyDSL's codegen-changing environment.

The ATOM knob set is also recorded in each entry. A knob that changes a kernel changes its IR; keying on the raw
knob set as well would only split hits between arms whose K1 is the same. Store, verify and fresh modes are
described in the HOWTO.

**Fast mode skips the trace.** `ATOM_DSV41_RK_COCACHE_FAST=1` adds an index from the launcher's own FlyDSL key to the
IR key. That launcher key is FlyDSL's `_jit_function_cache_key`: the jit function's source, every dependency's source
and closure scalars (the kernel name carries the extern digests), the toolchain, plus the call's argument key.

- *Hit.* The stored artifact goes into the launcher's in-process cache before the call. The linked `.ll` files are
  re-hashed at their recorded paths first; a changed file is a miss.
- *Audit.* The index is written in every mode, so a `verify` run audits it (`FASTKEY_UNSOUND` is logged if one
  launcher key ever maps to two IR keys). No such event occurred.

**rocKE.** `dsv41_mono/w16_cocache.py` keeps two caches:

- **HSACO cache.** Every COMGR compile is keyed on the LLVM IR, ISA, options and the COMGR library file's sha256 plus
  its LLVM version and flavor. Every Rust-direct compile is keyed on the serialized rocKE IR, the direct levers and
  the direct library file's sha256. S1 / C1 whole kernels compile through it.
- **Generation cache.** It covers the stage-function generators (`stage_moe`, `stage_shared`, `stage_gate`,
  `stage_norm`, ...). The key is the command line, the `ROCKE_*` codegen environment and the sha256 of every rocKE
  source the generators can import. A generated `.ll` is exactly what the ATOM cache then hashes as a linked file,
  so the two caches chain.

**Storage.**

- *Node-local.* Entries live under `/tmp/vanantha-dsv41/w16/{cc,gc}` and are written atomically (temporary name,
  then rename), so ranks and processes share a directory.
- *Seed copy.* The seed is versioned: `/scratch/users/vanantha/w16/seed/<cache>-<job>.tar` plus a `<cache>.latest`
  pointer. A job seeds an empty node from it and, with `W16_SEED_SAVE=1`, writes a new version back.
- *Never deletes.* No file is deleted or replaced.

### 1.2 Why the trace matters: where the compile time goes

Compile-only, 17 representative builds (a1):

- The hybrid's cold cost splits into 0.61 trace (FlyDSL tracing ATOM's Python kernel bodies), 0.36 MLIR + LLVM
  compile, and the rest key and store.
- An IR-mode hit removes only the compile: 0.63 of the cold wall (0.68 for ATOM mono, whose trace is heavier).
- A fast hit removes both: 0.009 of the cold wall (0.007 for ATOM mono).

The same split shows in the server:

- *IR mode.* It leaves the trace of all 144 builds on the start's critical path (0.724 of the status quo).
- *Fast mode.* It removes it (0.228 of the status quo; graph capture 0.025).
- *What remains.* About 0.21 of the status-quo start is process start, weight load from node-local /tmp, and KV
  allocation and warmup. None of it is compile, and the switch avoids all of it.

### 1.3 The compile-order effect

alola rule 13 / 137: `iglp_opt(1)` code depends on earlier compiles in the process. So the cache is never trusted
from one process alone.

- *Fresh-compile controls.* Every proof above compares a hit against fresh compiles in other processes, in reversed
  and shuffled build orders (a1 C / E, b1 verify server, a2 / x1 `ccx`).
- *Knob-off identity.* That check also ran in both orders.

The controls were run on purpose and found no history dependence for these builds. The fresh-compile control stays
available: `ATOM_DSV41_RK_COCACHE_MODE=verify`, and `ROCKE_W16_COCACHE_MODE=verify` with `AMD_COMGR_CACHE=0`. Use
it once per new toolchain or tree.

### 1.4 Two observations along the way

- **FlyDSL's own disk cache also misses for ATOM mono.** It is keyed on Python sources, and any edit to the ATOM tree
  invalidates it. ATOM mono's first start on this tree was 0.909 of the status-quo hybrid start; the warm restart was
  0.238. The IR-mode cache survives tree edits that do not change a kernel's IR. Fast mode does not: it is keyed like
  FlyDSL's. Both re-validate.
- **The rocKE stage kernels are small.** A COMGR compile of one standalone stage body is quick, so the HSACO cache
  matters for whole-layer kernels (S1) and for compile-cliff kernels (alola rules 54 / 58), not for these.

## 2. (b) Runtime arm switching in one server

`ATOM_DSV41_RK_SWITCH=1` (`rk/switch.py`, off by default) opens two runner methods to ATOM's collective RPC:
`rk_switch_arm` and `rk_arm_digests`. The allowlist in `model_engine/collective_rpc.py` grows only with the knob
set. The OpenAI server gets `POST /rk/switch` and `GET /rk/digests`.

**What a switch does, on every TP rank in lockstep, between steps:**

1. Drop the captured graphs, using ATOM's own sleep-release `release_cudagraphs`.
2. Apply the new build-time environment.
3. Forget every mono width build of the target and draft runners, and re-apply `ATOM_DSV41_RK_SCHED` from the
   bind-time base.
4. Run `capture_cudagraph()`. Its warmup forwards rebuild every captured width from the code-object cache.

**Knobs that need a restart.** The switch refuses knobs read at bind time, because they change scratch layouts or
peer buffers: NUMA, CCL, `xq`, MXFP4 weights, prefetch, the layer kernels and timelines. Restart the server for those.

**Measured, b1 and b2.** Each warm switch costs 0.015-0.017 of a status-quo hybrid start. Nearly all of it is the
rebuild-plus-capture part; release and reset are under 0.3 s together. A cold switch to an arm never compiled costs
0.42-0.43 of a cold fresh start: the trace and compile of the new arm, shared across ranks through the cache.

**A/B wall clock.**

- *Measured.* The b1 sequence (extB, nat, ext2, ext, ext2, nat, each with probe and C1 / C4 windows) took 0.215 of
  the time of the same six arm-runs as status-quo fresh servers, and 0.43 of fresh servers with the cache.
- *Model for N arms x P passes* (one arm-run = start or switch plus W seconds of served work). With rule 262's
  W ≈ 130 s cell: 0.26 of status quo, 0.64 of cached fresh. With b1's short windows: 0.07 of status quo, 0.28 of
  cached fresh.

## 3. (c) The search ladder: in-situ rig, served rung, fresh-server confirmation

`dsv41_mono/w16_search.py` turns rungs into the search engine's backend protocol.

- **`LadderBackend`.** `compile` / `check` / `measure` take the fidelity and dispatch on its proxy: `rig`, `served`,
  or `""` for the fresh confirmation.
- **`PrefetchEvaluator`.** It hands each round's configs to one rung pass, so the rig checks and times a whole batch
  together. The measurement rules are unchanged: each artifact is checked before its samples are used, and results
  are cached.
- **`ServedRung`.** It drives the switch with interleaved base / arm windows.
- **`CallableRung`.** It wraps the fresh-server confirmation.
- **`default_ladder()`.** rig, then served, then fresh.
- **Self-test.** `hackathon-prep/dsv41/w16/h/test_w16_search.py`: `SuccessiveHalving` over three rungs, one rig pass
  a round, miscompiled configs refused as `numeric_fail`.

### 3.1 The rig rung

`ATOM_DSV41_RK_RIG` (`rk/rig.py`) runs in an eager engine, and each candidate batch is switched in at run time. On
every K1 / K2 launch of the rig layers it does this:

- start from the in-place gates' snapshot of the live decode state;
- run the engine's own build as the reference;
- check every variant and every loaded copy byte for byte, with LDS poisoned in between;
- time all of them in shuffled rounds.

`hackathon-prep/dsv41/w16/h/rig_driver.py` loads the engine once. For each batch it generates the function sets
(through the generation cache), switches the variants in (`reset_builds=False`) and runs a few decode steps. With
`--search N` it drives `run_search` + `SuccessiveHalving` over two rig fidelities on a K2b knob subspace.

| | candidates a minute, relative to one fresh-server arm-run a candidate |
|---|---|
| warm, 1 copy x 6 rounds (c3 / c4) | about 640-690x |
| warm, 3 interleaved copies x 8 rounds (c5 / c6) | about 510x |
| cold: new rocKE functions and new kernels (c2) | about 75x |

**Rig resolution, measured with A-vs-A twins.**

- *Built twice and loaded twice, 1 copy (c4).* Parity twins moved by up to 0.0062 per batch, and the batch-level duplicate by up to 0.0195.
- *The cause (c5).* Three copies of a variant loaded one after another showed that the whole twin group sat about
  1.2-1.6% from the original. It is a load-order (code-object placement) effect, consistent across copies. This is
  the compile / placement lottery of alola rules 147 / 215.
- *The fix (c6).* The rig now builds and first-loads copies copy-major, so each variant's copies are spread over the
  load order. A-vs-A twins then agree within 0.0016 (parity), 0.0063 (r_max=1) and 0.0033 (the batch-level
  duplicate).
- *Pooled over 4 batches* (512 launch-rounds per variant): parity 1.0012 vs its twin 0.9989; r_max=1 0.9963 vs
  0.9953.

The rig is a screen that resolves about ±0.5% per batch. Its winners go to the next rung.

### 3.2 The served rung

The served rung is one server: the switch, then a fixed steady-state decode window (C1, 16 requests x 256 tokens,
`ignore_eos`, warm-up excluded), base and arm interleaved. In b2 it measured:

- the A-vs-A twin `ext2b` / `ext2`: 0.966-1.000 (spread about 3.4%) over 4 pairs;
- `extB` / `ext2`: 0.973-1.042.

At this window size the served rung resolves only multi-percent effects. Use more prompts or concurrency per window,
and more pairs, before trusting it below about 3%. Acceptance (DSpark AL) varies run to run on the same prompts and
dominates the window noise.

### 3.3 The fresh-server rung

This is unchanged and remains the acceptance rung: same-job comparator, A-vs-A twins, rotated passes, medians, both
nodes for wins. Plug it in with `CallableRung`.

## 4. Open items and defects found

1. **W15's `k2moe_space` admits `ug.ug_sop=2`.** `stage_ug.build_ug_kernel` refuses it ("x_first / poll=asm / sop=2
   / wave_order do not apply"). That is 8 of 27 sampled configs in c2. The rig records them as compile errors, but
   `is_valid` should exclude them. This is W15's file; it is reported here, not edited.
2. **The rig's draft runner layers were not hit with `--rig-layers 1.2.3.10`.** Only the target's 6-row K2 launches
   were timed. The draft's K1 / K2 layer ids need their own selection (a follow-up for S1's rig).
3. **n07-17 rebooted at 05:46 UTC** (alola rules 223 / 261). Its /tmp seed is gone, and its BeeGFS mount was missing
   afterwards (a reseed job failed with exit 0:53). The second-machine proofs therefore ran on a Markham CPU-only node
   (x1 / x2). c3 had run on n07-17 before the reboot: it consumed fast hits compiled on n01-09, and its rig verdicts
   were correct (parity bitwise, mutants caught).
4. **COMGR-cache-off HSACO rerun (x2), done.** a3's HSACO run had COMGR's own on-disk cache on, so its later
   "fresh" compiles could have been COMGR hits. x2 reran it with `AMD_COMGR_CACHE=0` on the CPU-only node: 22 of 22
   kernels were identical in all 5 processes, and both verify passes were SAME (real compiles this time).
5. **Not done.** The switch re-captures every capture size. Capturing only the buckets a cell uses (decision 12,
   point 3) would cut it further; it is not done.

## 5. Files

- **ATOM-dsv41 (additive, default-off).**
  - New: `atom/models/deepseek_v41/rk/{cocache,switch,rig,w16}.py`.
  - Changed: `atom/models/deepseek_v41/mono/kernels/__init__.py` (one install call), `atom/model_engine/
    collective_rpc.py` (allowlist grows only with the knob), `atom/entrypoints/openai/api_server.py` (two routes,
    only with the knob).
- **rocKE.** `library/kernels/gfx950/dsv41_mono/{w16_cocache,w16_search}.py`.
- **Harness.** `hackathon-prep/dsv41/w16/`:
  - `mkstage.py`, `up.sh`, `submit.sh`, `release.sh` (W16 GPU cap 6), `w16_job.sh`, `xnode_job.sh`, `HOWTO.md`;
  - `h/`: `w16_run.sh` (phases), `cc_build.py`, `cmp_cc.py`, `start_summary.py`, `switch_drive.py`,
    `switch_summary.py`, `gate_driver.py`, `gate_summary.py`, `rig_driver.py`, `rig_variants*.json`, `hsaco_cc.py`,
    `textprobe.py`, `test_w16_search.py`, `local_cc_test.py`.
