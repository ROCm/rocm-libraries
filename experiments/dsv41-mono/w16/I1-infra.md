# W16 I1: fast-iteration infrastructure for the DeepSeek-V4.1 rocKE path (revision 2)

W16, 2026-11-05. MI355X (gfx950), TP2 no-EP, DSpark 5, the B0 recipe (`--level 0` + FULL graphs), atom-dev image
`nightly_202610171506` (FlyDSL 0.4.0).

- **Ratios only.** Absolute seconds and rates are in
  `hackathon-prep/private-perf/dsv41-mono/w16/I1-abs.md`.
- **Usage and code map.** See `hackathon-prep/dsv41/w16/HOWTO.md`.
- **Revision 1** (the verified report this one corrects) is kept as `I1-infra.r1.md`; section 6 lists every
  correction.
- **Jobs.** Revision 1: a1-a3, x1-x2 (GPU-less), b1-b2 (servers, gates), c1-c6 (rig), all on n01-09 (plus c3 on
  n07-17). Revision 2: x3 / x4 (Markham CPU-only), a4 (n07-17, GPU-less), a5 (m07-13, GPU-less), b3x / v1x / c7x / c8x /
  v3x / v4x (n14-05), b4n / v1n / c7n / c8n / v3n / v4n / sh1 / dg1-dg9 (n07-17), x5 / x6 (Markham CPU-only:
  `sched_scratch`, final-tree knob-off identity). v2n was cancelled (the `ug=1` hang of 5.2) and v2x never ran; v3 / v4
  replace them. Raw outputs: `hackathon-prep/dsv41/w16/results/<tag>/`.

**Nodes.** n01-09 went `draining` ("SlurmdSpoolDir is full") during this round, so the second and third gfx950 nodes
are n07-17 (the other `rocke` node) and n14-05 (an 8-GPU MI355X in the `miopen` partition). m07-13 was seeded but
gives us no GPU access (no `render` group: alola rule 265); only CPU-only work ran there.

## 0. Verdict

### 0.1 Measured speed-ups, per node

Every row comes from fresh servers started one after another in one job, with twins where marked. "Status quo" is
what each arm costs without W16: a hybrid (extern-linked) arm starts with no cache (FlyDSL never caches an
extern-linked build), and ATOM mono starts from FlyDSL's own warm disk cache.

| what | ratio | n01-09 (b1, 1 sample) | n07-17 (b4n, twins) | n14-05 (b3x, twins) |
|---|---|---|---|---|
| hybrid start, FAST cache | of the status-quo hybrid start | 0.228 | 0.224 (twins 0.215 / 0.232) | 0.333 (0.330 / 0.336) |
| hybrid start, FAST cache | of ATOM mono's warm start | 0.957 | 1.061 | 0.972 |
| warm switch (kernels in the cache) | of the status-quo hybrid start | 0.015-0.017 | 0.017 | 0.014 |
| warm switch | of the cached (FAST) fresh start | 0.065-0.073 | 0.074-0.077 | 0.042-0.043 |
| cold switch (arm never compiled) | of a cold fresh start that stores | 0.41-0.43 | 0.47 | 0.39 |
| A/B over 6 arm-runs (2 cold, 4 warm), one server | of the status quo, honest baseline | 0.286 | 0.289 | 0.265 |
| same | of the previous report's baseline (every arm at the hybrid cache-off start) | 0.215 | 0.215 | 0.208 |
| same | of fresh servers with the cache | 0.43 | 0.46 | 0.38 |

- **The A/B baseline is corrected** (problem 5). Revision 1 charged both ATOM-mono arm-runs at the hybrid's
  cache-off start. Its honest status quo (`h/ab_wall.py`) charges them at ATOM mono's warm FlyDSL start: 0.286 on
  n01-09, not 0.215, and 0.27-0.29 on the two new nodes.
- **n14-05 starts slower** (its `/tmp` is a busy, 88 %-full RAID). The cached start is 0.33 of the status quo
  there and 0.22-0.23 on the two `rocke` nodes. The warm switch is 0.014-0.017 of the status-quo start on all three.

### 0.2 Rig rate, timed candidates only

| rig run | node | timed candidates a minute, relative to one fresh-server arm-run a candidate on that node |
|---|---|---|
| c2, cold (new functions, new kernels), revision 1 | n01-09 | 53x (19 timed configs; revision 1 said 75x by counting 8 refused configs) |
| c3 / c4, warm, 1 copy, fixed order (biased: 4.1) | n07-17 / n01-09 | 473x / 498x (revision 1 said 640-690x) |
| c6, warm, 3 copies, copy-major, fixed order | n01-09 | 394x (revision 1 said about 510x) |
| c7 rig2, warm, copy-major, fixed order | n07-17 / n14-05 | 540x / 611x |
| c8, warm, copy-major, **shuffled order** (the new default) | n07-17 / n14-05 | 549x and 534x / 573x and 590x |
| c7 rig1, the job's first rig run (13 / 10 kernels compiled, 33 function sets generated) | n07-17 / n14-05 | 171x / 192x |

The headline is the copy-major, shuffled-order rig: **about 530-610x** on the two nodes. The baseline per node is its
own hybrid cache-off start plus rule 262's 130 s cell. Refused configs are no longer counted (problem 6): the rig now
prunes `ug.ug_sop=2` (the ug generator refuses it) before sampling, so 0 configs are refused, and the halving
completes (`stopped: strategy_done`, 36 evaluations) instead of ending on the evaluation budget.

### 0.3 Correctness: what is proven, and what is not

| claim | evidence | status |
|---|---|---|
| a cache hit equals a fresh compile | 365 genuine `verify_SAME` controls, 0 `verify_DIFF` (table in 1.3) | proven |
| the FAST path serves what the trace would build | the FAST audit (new) on every would-be fast hit: 0 unsound in 102 compile-only and 1014 in-server audited hits; the audit catches a deliberately unsound key; FAST servers equal audit servers for SCHED / LEV / PAUSE arms | proven on 2 nodes (1.4, 2.6) |
| knob-off: hooks change nothing | K1 / K2 identity (also with `ATOM_DSV41_RK_SWITCH=1`, now including the real hybrid) on 4 machines; RPC allowlist and server routes identical (new `kof` phase) | proven |
| a switched graph-mode server runs the same kernels as a fresh one | code-object digests 32/32 (runner, width, rank) sets on 3 nodes | proven |
| the bitwise launch gate on the SERVED (graph-mode) switched server | new: every K1 / K2 launch of the re-capture's warmup forwards, at every captured mono width (8 target + 8 draft), against ATOM's build: ext2 0 / 1376 launches differ twice per node, extB 0 / 1376, the `mut` arm 688 / 1376 (every K2), on n07-17 and n14-05 | proven |
| the switched server's captured graphs compute what a fresh server's do | new graph-replay probe: DRAFT mono widths bitwise across 9 fresh servers on 2 nodes and 2 trees, and every switched step (25/25); mutants `mut`, `keep_graphs`, `stale_in` caught on both nodes | proven for the draft graphs |
| the same, for the target graphs (logits / tokens) | the target step is not run-to-run reproducible on this stack even in one process (2.4: ATOM's compressor GEMMs and pooling are not deterministic); `stale_out` (a stale target pointer) is therefore not caught by any reproducible check | **not provable on this stack**; open (5.1) |
| the eager in-place gate on served decode states at every width, 3 prompt seeds, both nodes | 2.5: 416,960 ext2 launches over 3 seeds x 2 steps x 2 nodes plus 70,496 fresh-engine launches, 0 differ; every `mut` K2 differs (103,978 of 207,956 `mut` launches) | proven on 2 nodes |
| ext2 / extB / nat are bitwise equal to each other | the gate (every mono launch) and the draft probe agree | proven at every mono width |

## 1. (a) The code-object caches

### 1.1 What is cached, and the key

Unchanged from revision 1 (1.1 there), with one addition: the FAST audit.

- `ATOM_DSV41_RK_COCACHE=<dir>` wraps FlyDSL's `MlirCompiler.compile`; the key is the IR text, the content of every
  linked rocKE `.ll`, FlyDSL's toolchain fingerprint, the arch, the pass pipeline / hints / LLVM options and FlyDSL's
  codegen environment. `_FAST=1` maps the launcher's FlyDSL key to that IR key and skips the trace.
- **New: `_FAST=audit`** (problem 13). Revision 1's `FASTKEY_UNSOUND` audit only ran on calls that traced, so a fast
  hit was never checked. In audit mode a call whose launcher key is in the index is not served from it: it traces,
  and the traced IR key is compared with the one the index would have served (`fast_audit_SAME` /
  `FASTKEY_UNSOUND`). Run it with FlyDSL's disk cache off, or FlyDSL serves the call before it traces.
- **New: the audit's own mutant.** `_FAST_MUT=weak` gives every launcher one key. The audit must report it.

### 1.2 Where the compile time goes

Unchanged (revision 1, 1.2): the trace is about 0.6 of a cold hybrid build, which is why FAST matters.

### 1.3 Cache-hit equals fresh-compile: the controls, recounted

| job | machine | what | genuine `verify_SAME` |
|---|---|---|---|
| a1 | n01-09 | 2 arms x 17 builds x 2 verify runs (reversed, shuffled) | 68 |
| b1 | n01-09 | the verify server: every artifact both ranks build | 144 |
| a2 | n01-09 | cross-job, ATOM mono only | 17 (its 17 "hybrid" controls were ATOM mono: see below) |
| x1 | Markham CPU-only | cross-machine, both arms | 34 |
| x3 | Markham CPU-only | cross-machine, both arms, link guard on | 34 |
| a4 | n07-17 | cross-node (entries compiled on n01-09), link guard on | 34 |
| a5 | m07-13 | cross-node, link guard on | 34 |
| **total** | 5 machines | | **365, 0 DIFF** |

- **a2's hybrid controls did not test the hybrid** (problem 1). a2 ran `ident ccx` without `gen`, so the function-set
  directories were empty and every extern build fell back to ATOM mono (`EXTERN_MISSING=mono`): 89 fallback lines,
  `links=0`, IR keys identical to ATOM mono's. Revision 1 counted those 17 twice ("34 cross-job controls") and said
  nothing. The guards now make this impossible:
  - `need_fn`: an extern arm whose function sets are missing is refused (`NO_FN`) by every phase;
  - `cc_build.py` records each build's rocKE links, and `W16_CC_EXPECT_EXT=1` (set for every extern arm) fails a build
    with no link;
  - `ccx_summary.py` fails an extern run whose IR keys overlap ATOM mono's. On the old a2 data it reports both
    failures; on x1, x3, a4 and a5 the hybrid builds link 1-9 functions each and share no IR key with ATOM mono.
- **The rocKE HSACO cache, recounted** (problem 10). a3's 44 controls ran with COMGR's own disk cache on (its verify
  compile steps took about 1/14 of a fresh compile: COMGR cache hits), so they do not count. Genuine fresh controls with
  `AMD_COMGR_CACHE=0`: x2 (Markham CPU-only) 44, a4 (n07-17) 44, a5 (m07-13) 44, **132 in total, 0 differ**; 22 / 22
  kernels identical in all 5 processes of each job.

### 1.4 FAST soundness (compile-only, phase `fa`)

On x3, a4 and a5 (three machines), the same result each time:

- **The mutant is caught.** With one launcher key for every launcher, the audit run logs 16 `FASTKEY_UNSOUND` (the 16
  builds after the first) and 1 `fast_audit_SAME`.
- **Real keys are sound.** For ATOM mono and the hybrid, all 17 would-be fast hits audit `SAME` (34 a job, 102 in
  all), 0 unsound.
- **Knob arms.** For LEV (`usop,gsop`; `nop=1`) and PAUSE (`nop`) on ATOM mono and on the hybrid, the knob changes the
  launcher key itself (the kernel name), so no launcher of another knob value is ever served: the audit run stores 17
  new entries, and the FAST=1 run then serves all 17 with code objects equal to a cache-off compile, 17/17 per arm.
- SCHED changes runner attributes, not a compile-only build: it is tested in a server (2.6).

### 1.5 The compile-order effect

Unchanged: every proof compares a hit with fresh compiles in other processes and orders; no history dependence was
found for these builds.

## 2. (b) Runtime arm switching in one server

The mechanism is unchanged (revision 1, section 2). New in this round: the served-server gate, the graph-replay
probe, three verification mutants, and the determinism diagnosis.

### 2.1 Measured switch costs

See 0.1. A switch's release and reset parts stay under 1 s together; the rest is rebuild plus re-capture. A gated
switch (`arm@check`, 2.2) costs about 2x a plain warm switch (every mono launch of the warmup runs twice).

### 2.2 The bitwise launch gate on the served, graph-mode server (problem 2)

`ATOM_DSV41_RK_LEV_CHECK=<dir>` in a switch's env. The re-capture's warmup forward runs every captured width once
eagerly before recording its graph; with the gate on, every K1 / K2 launch there runs as ATOM's own build and as the
arm's build from the same snapshot (scratch, peer buffers and row buffers; LDS poisoned in between), and every byte it
can write is compared, about 46 MB a launch. The graph is recorded with the arm's kernels only. This is the gate of
revision 1's eager engine, now inside the graph-mode server's own switch, at every captured mono width.

| step (v1n on n07-17, v1x on n14-05) | launches gated | differing | widths covered |
|---|---|---|---|
| ext2@check (twice per node) | 1376 | 0 | target 6-48 rows (8 widths, 160 launches each), draft 5-40 rows (8 widths, 12 each) |
| mut@check (must differ) | 1376 | 688 = every K2 launch | same |
| extB@check (r_max=1) | 1376 | 0 | same |
| nat@check | 0 | (ATOM mono builds no reference: nothing to compare) | |

### 2.3 The graph-replay probe (problem 2, logits / tokens)

`POST /rk/probe` (ATOM `rk/gprobe.py`). On every TP rank, between steps, for 3 seeds and every captured graph:

1. snapshot the V4.1 cache arena (PAGEs, index planes, STATE slots) to host memory, and checksum it;
2. per replay: reset the arena to its as-allocated state and zero the mono row buffers, write seeded token ids,
   rebuild the capture-time metadata exactly as `capture_cudagraph` does, seed torch's generators, replay, and hash
   the hidden rows, the logits (`compute_logits`, the step's next op) and their argmax tokens; then each DSpark draft
   pass at each capture size, with seeded anchor ids;
3. restore the arena and check its checksum (`restored`), so the served state and the prefix cache stay valid.

Each replay runs twice (`repeat_same`). `h/gp_summary.py` compares probes across servers, jobs and nodes by width
class.

| width class | fresh twins (any job, node, tree) | switched vs fresh | must-fail |
|---|---|---|---|
| draft mono widths (bs 1-8) | **57/57 pairs equal** (16 fresh servers: 9 hybrid, 7 ATOM mono; 3 jobs, 2 nodes, trees i1e and i1l) | **25/25 equal** | `mut` 3/3, `keep_graphs` 3/3, `stale_in` 2/2 caught; `stale_out` 0/3 (it rebinds a target buffer only) |
| target mono widths (bs 1-8) | 0/57 | 0/25 | not usable: twins differ |
| native widths (bs 16-64) | 0/57 | 0/25 | not usable: twins differ |

- **ATOM mono, ext2, ext and extB give identical draft hashes** at every mono width, in graph mode.
- `stale_in` (new mutant): after the re-capture, the draft passes' staged input buffers are rebound, so the recordings
  read the old ones. The probe catches it at every seed and width.
- The target hashes of one arm differ between two replays in ONE process from the same reset state, so they cannot
  separate a switched server from a fresh one. 2.4 shows why.

### 2.4 Why the target step is not reproducible: the diagnosis

`POST /rk/diag` (dg1-dg9, n07-17) runs, per width: 3 graph replays from one reset state; and two eager forwards from
it, with a forward hook on every module (a checksum of its first input and its output) and checksums after every
mono K1, K2 and compressor call. The first module whose input repeats and whose output does not is the first
non-deterministic op.

| width | first non-deterministic op (input equal, output differs) | what it is |
|---|---|---|
| bs 1 (6 rows), default | `layers.2.attn.compressor.wkv_gate` | aiter bf16 GEMM, fp32 out, tuned to `bf16gemm_fp32bf16_tn_32x64_splitk_clean` with split-K 7: an atomic split-K sum |
| bs 1, split-K ASM rows forced to their no-split twins (`W16_DETGEMM=1`) | `layers.20.attn.compressor.wkv_gate` (layers 2, 8, 14 now repeat) | a compressor GEMM DETGEMM does not cover (its tuned kernel is not a split-K ASM one; the dsv41 table tunes the other small-M bf16 compressor shapes to FlyDSL `hgemm` with k-splits) |
| bs 2 / 8 (12 / 48 rows), DETGEMM | the compressor's arena writes (`L2` / `L8 compress.arena`) with every input equal (on rank 1 at bs 2 the first difference is K2, which all-reduces rank 0's partials) | the compressor's pooling / state-write kernels at more than 6 rows |
| bs 16 (96 rows, ATOM's native path) | `layers.0.ffn.gate` | the MoE router GEMM (FlyDSL `hgemm` `ks5`), then the routed MoE |

- Resetting the mono scratch too (mailboxes) changed nothing; every mono K1 / K2 before the first compressor
  difference repeats bit for bit, which agrees with the launch gate.
- **Consequence.** A logits / tokens comparison across servers cannot be bitwise on this stack: ATOM / aiter kernels
  outside the mono path (and outside W16's write scope) are not deterministic run to run. The verification therefore
  rests on the launch gate (2.2: every mono launch, every width), the code-object digests, and the draft graphs'
  bitwise probe (2.3); the target graphs' pointer wiring after a re-capture is covered only indirectly (the same
  re-capture code path as the draft, which `keep_graphs` / `stale_in` show the probe can see).

### 2.5 The eager in-place gate: every width, 3 prompt seeds, both nodes (problem 3)

The gate of revision 1's b2 (an eager engine; every K1 / K2 launch of the served decode steps checked in place
against ATOM's build, about 26 MB a launch), now with a concurrency plan C1-C8 so that every target width (6-48 rows)
and every draft width (5-40 rows) is served, on both ranks, for 3 prompt-shuffle seeds, plus a fresh-engine control.
Per seed: an engine starts on ATOM mono and switches to ext2 (gated), to the `mut` arm (gated, must differ), and back
to ext2 (gated).

| node | ext2 launches gated (6 steps) | differing | `mut` launches (3 steps) | differing | fresh-engine ext2 | widths in every step |
|---|---|---|---|---|---|---|
| n07-17 (v4n) | 206,668 | 0 | 103,216 | 51,608 (every K2) | 35,408, 0 differ | target 6-48, draft 5-40 (all 16, both ranks) |
| n14-05 (v4x) | 210,292 | 0 | 104,740 | 52,370 (every K2) | 35,088, 0 differ | target 6-48, draft 5-40 (all 16, both ranks) |

Every one of the 20 gated steps (both nodes) served all 16 mono widths on both ranks (`lev_r*.jsonl` rows), and every
`mut` difference is a K2 launch (K1 is not mutated). Code objects: after the last ext2 step of each seed, 32/32
(runner, width, rank) sets equal the fresh engine's; at the first ext2 step the eager engine had built 2 sets at
switch time, both equal (it builds the rest lazily while serving, and the gate checks those launches). The reply
comparisons in `gate_summary.txt` are not a check here: the fresh control served seed 0's prompts, and eager replies
are not reproducible on this stack anyway (2.4). Summaries: `results/v4n/gate_summary.txt`,
`results/v4x/gate_summary.local.txt`.

### 2.6 FAST in a server for SCHED / LEV / PAUSE (problem 13)

Phase `swk` (v3n on n07-17, v3x on n14-05): server A with `FAST=audit` and FlyDSL's disk cache off (every launcher
traces; every would-be fast hit is compared with the trace), then server B with `FAST=1`, over the same switch
sequence: `ext2 + SCHED ug=1` (must be refused, 5.2), `ext2 + SCHED tpf=49`, `ext2 + LEV usop,gsop`,
`ext2 + PAUSE nop`, `ext2 + LEV nop=1`, `ATOM mono + SCHED ug=1` (refused), `ATOM mono + LEV`, `ATOM mono + PAUSE`,
ext2.

| node | audit server | B against A, every step | refusals |
|---|---|---|---|
| n07-17 (v3n) | 1014 would-be fast hits `fast_audit_SAME`, 0 `FASTKEY_UNSOUND` | code objects equal (30-32 / 30-32 sets), draft probe hashes equal (48/48), B served from fast hits (72 a rank) | 2/2 refused on both servers |
| n14-05 (v3x) | 512 would-be fast hits `fast_audit_SAME` (a colder cache: 838 stores, against 140 on n07-17), 0 `FASTKEY_UNSOUND` | code objects equal (30-32 / 30-32 sets), draft probe hashes equal (48/48), B served from fast hits (72 a rank) | 2/2 refused on both servers |

Verdict 9/9 steps on each node (`results/v3x/swk_summary.regen.txt`; the in-job `swk_summary.txt` of v3x was written by the
earlier script that compared the non-reproducible target hashes, 2.4). Every one of the 32 probes of v3n and v3x (both
servers, every step) has draft hashes equal to v3n's first ext2 probe, 48/48: the LEV / PAUSE / `tpf` arms are bitwise
equal to ext2 in the served draft graphs, across nodes. The refusal names the scratch need (20,006,528 bytes against
19,965,056 sized at bind) on both ranks, and the server keeps serving its arm (the next switch and probe succeed).

## 3. (c) The search ladder

### 3.1 The rig rung

Changes in this round:

- **Load order shuffled per copy pass** (`ATOM_DSV41_RK_RIG_ORDER=shuffle`, the rig driver's default). See 4.1.
- **Must-fail mutants in every batch** (problem 11). `--sentinels` adds `mut_put_bit` and `mut_scale_byte` to every
  search batch. c7 / c8 on both nodes: 12/12 mutants caught in 6/6 batches per rig run (revision 1: `scale_byte` only
  in c1-c3, and no mutant at all in the search batches).
- **Refused configs pruned** before sampling (`ug.ug_sop=2`), and the halving budget covers the whole ladder.
- **Rates counted on timed candidates** (`h/rig_rates.py`; 0.2).

A-vs-A twins (ratio to ATOM's build per batch; |difference| between twins):

| run | fixed batches: `par` / `par_b`, `rmax1` / `rmax1_b` | 27-config batch: `c0` / `ctl_dup_c0` | 9-config batch |
|---|---|---|---|
| c7n, fixed order (n07-17) | up to 0.0065 / 0.0030 | 0.0091, 0.0074 | 0.0016, 0.0005 |
| c7x, fixed order (n14-05) | up to 0.0094 / 0.0090 | **0.0229**, 0.0047 | 0.0019, 0.0017 |
| c8n, shuffled (n07-17) | up to 0.0048 / 0.0053 | 0.0026, 0.0018 | 0.0027, 0.0006 |
| c8x, shuffled (n14-05) | up to 0.0088 / 0.0108 | 0.0019, **0.0209** | 0.0046, 0.0060 |

With shuffled order the rig resolves about ±0.5 % per batch on n07-17, but only about ±1-2 % on n14-05 (a busier,
shared node: up to 2.1 % between a candidate and its duplicate in a 27-config batch, 1.1 % between fixed-batch twins).
With fixed order neither node reached ±0.5 %. Revision 1's "±0.5 %" rested on one node and the fixed order
(problem 7). On n14-05 one rig run even promoted a spurious 3.7 % "winner" (c8x rig1's rung-2 best 0.963, against
0.999 for the same search in rig2): a rig verdict below about 2 % is a screen only, and its winners must be confirmed on
the served rung.

### 3.2 The served rung (problem 8)

- **Revision 1's outlier, diagnosed.** b1's ladder gave extB / ext2 = 1.209 and 1.117, b2's 0.97-1.04. The rung's
  sample was the ratio of two windows' median **seconds per output token**; with DSpark a request's per-token time
  follows its acceptance, which varies run to run (the same 8 prompts' per-request ratios between two windows of
  one arm ranged 0.91-1.55; their correlation across windows was 0.1-0.25). A bootstrap of two 8-request windows from
  b1's own requests gives a 95 % range of 0.87-1.14 and a 99 % range of 0.83-1.20: 1.209 is a request-noise draw at
  the edge of that range, not a kernel effect (`h/rung_noise.py`); b2's 16-request windows narrow it to 0.90-1.11.
- **The fix.** The served rung now times **seconds per DSpark decode step** (a request's decode time over its
  steps: what the kernels set, free of acceptance), on 16 requests a window. From the same requests, a two-window
  ratio's 95 % range is ±1.3-1.4 % per step against ±9-11 % per token.
- **Measured** (4 interleaved pairs each, both nodes): the A-vs-A twin ext2b / ext2 gave 1.0015-1.0035 (n14-05) and
  1.0019-1.0047 (n07-17): every twin sample above 1, a placement bias of the re-loaded code objects of about +0.3 %.
  extB / ext2 gave 0.998-1.004 and 1.000-1.002: a tie. The served rung now resolves about ±0.5 % with a twin beside it.

### 3.3 The fresh-server rung

Unchanged, and still the acceptance rung: same-job comparator, A-vs-A twins, rotated passes, medians, both nodes.

## 4. Noise and placement

### 4.1 The rig's load-order bias (problem 7)

- **c3's drift, diagnosed.** In c3's rung-2 batch (n07-17, 1 copy, fixed order) the 7 promoted configs' ratios fell
  from 1.0061 to 0.9555 in load order: rank correlation with the load index -0.86, and the rung-2 values ran
  opposite to the same configs' rung-1 values (which set their order). The spread was load order, not the configs.
- **Copy-major alone does not remove it.** With 3 copies, fixed order, the 27-config rung-1 batches still correlate
  with load order (-0.30 to -0.78 over c7n / c7x).
- **Shuffled order reduces it.** Each copy pass now builds and first-loads the variants in a fresh shuffled order:
  the 27-config batches correlate -0.11 / -0.17 (c8n) and -0.34 / -0.12 (c8x), against -0.48 to -0.78 in three of the
  four fixed-order runs. The residual noise is node-dependent (3.1).
- **The search winner is not reproducible at the rig's resolution.** c7 (fixed order) and c8 (shuffled) found
  different best configs, both about 0.99 of ATOM's build. The rig is a screen; its winners go to the served rung.

### 4.2 Served-path determinism

Unchanged in substance: greedy replies agree on 2-8 of 16 prompts between fresh servers of one arm; 2.4 now names the
kernels responsible.

## 5. Open items and defects found

1. **The target graphs cannot be compared bitwise across servers** (2.4): ATOM / aiter's compressor GEMMs (split-K
   ASM and FlyDSL `ks` kernels), the compressor's pooling at more than 6 rows, and the MoE router GEMM at native widths
   are not deterministic. `W16_DETGEMM=1` (verification mode) fixes the split-K ASM GEMMs; the FlyDSL split-K GEMMs and
   the pooling need kernel changes outside W16's scope. Until then a stale target pointer (`stale_out`) is caught by
   nothing reproducible.
2. **Found and fixed: switching to `ATOM_DSV41_RK_SCHED=ug=1` overran the runner's scratch.** After such a switch
   (b3x and v1n / v1x `swk`) the next graph replays hung and wedged the server, although the re-capture succeeded and
   the code objects matched a fresh build; in v2n the re-capture after a switch to ATOM mono with `ug=1` hung itself. A fresh server started with `ug=1` serves and probes normally (sh1). Cause
   (x5, `h/sched_scratch.py`): the runner sizes its scratch once, at bind, for the widest step under the bind-time
   schedule, and `ug=1`'s K2 needs about 0.2 % more (its gate/up task flags) than ATOM's `ug=2` sized; `fug` and
   `tpf` need no more. The switch now refuses a schedule whose kernels need more scratch than the runner sized at bind
   (`switch._sched_check`, before anything is released); v3n / v3x (2.6): refused on both nodes, both ranks, both servers.
3. **W15's `k2moe_space` admits `ug.ug_sop=2`** (the ug generator refuses it): reported, not edited; the rig prunes it.
4. **The rig's draft layers** are still not selected by `--rig-layers` (revision 1, 4.2).
5. **Capturing only a cell's buckets** (decision 12, point 3) is still not done.

## 6. Corrections to revision 1

| # | revision 1 said | corrected |
|---|---|---|
| 1 | 34 cross-job controls in a2; SWITCH=1 knob-off on tree i1b | a2's 17 hybrid controls were ATOM mono (no `gen` phase); now guarded, and redone with real hybrid builds on 4 machines (1.3); the SWITCH=1 identity now covers the hybrid (x3, a4, a5, x4) |
| 2 | the switched arm is backed by digests, an eager gate and reply overlap | the gate now runs inside the graph-mode switched server at every width (2.2); the graph-replay probe proves the draft graphs bitwise (2.3); target logits are shown to be non-reproducible on this stack, with the cause (2.4) |
| 3 | one eager gate, one node, one seed, 14-16 sets | 2.2 covers all 32 sets on 2 nodes; 2.5 the eager gate with 3 prompt seeds on 2 nodes |
| 4 | wins on one node, single samples | three nodes, twins (0.1) |
| 5 | A/B 0.215 of the status quo | 0.286 (n01-09); 0.27-0.29 on the new nodes |
| 6 | 640-690x warm, 75x cold | refused configs excluded: c3 / c4 473-498x, c2 53x; copy-major (+ shuffled): 530-610x |
| 7 | rig resolution ±0.5 % | with fixed order ±0.5-2.3 %; with shuffled order ±0.5 % on n07-17, ±1-2 % on n14-05 (3.1, 4.1); c3's n07-17 drift was load order |
| 8 | served rung 3.4 % spread | b1's 21 % outlier diagnosed (per-token acceptance noise); the rung now times seconds per step (3.2) |
| 9 | 88 HSACO verify controls | 44 genuine then; 132 now (1.3) |
| 10 | both mutants caught in every batch of c1-c6 | `scale_byte` was in c1-c3 only and the search batches had none; now both are in every batch (3.1) |
| 11 | `gate_summary.txt` 2/14, 14/16 | regenerated as `results/b2/gate_summary.regen-2026-11-05.txt` (2/2 and 14/14 common sets; 12 and 2 sets built on one side only, which the old count treated as differing) |
| 12 | FAST hits audited by `FASTKEY_UNSOUND` | that audit only saw traced calls; FAST=audit now checks every would-be hit (1.4, 2.6) |

## 7. Files

- **ATOM-dsv41 (additive, default-off).**
  - New: `atom/models/deepseek_v41/rk/{cocache,switch,rig,w16,gprobe}.py`.
  - Changed: `atom/models/deepseek_v41/mono/kernels/__init__.py` (one install call), `atom/model_engine/
    collective_rpc.py` (allowlist grows only with the knob), `atom/entrypoints/openai/api_server.py` (four routes,
    only with the knob).
- **rocKE.** `library/kernels/gfx950/dsv41_mono/{w16_cocache,w16_search}.py`.
- **Harness.** `hackathon-prep/dsv41/w16/` (`h/` adds `gp_summary.py`, `ccx_summary.py`, `fa_summary.py`,
  `swk_summary.py`, `ab_wall.py`, `rung_noise.py`, `rig_rates.py`, `knoboff.py`).
