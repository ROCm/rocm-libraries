# Case study: persistent layer-kernel framework overhead on the toy chain (gfx950)

Measured scope: the rocKE persistent layer-kernel framework
(`library/kernels/gfx950/qwen38_mono/framework.py`) running the toy chain of
`toy_chain.py`. It is compared with a FlyDSL twin of the same schedule
written with the ATOM `qwen4_exp_mono` layer-kernel idioms (mailbox
`launch_tag` / `put` / `poll_raw`, DPP `xred` wave sums, `gpu.barrier`, and
mono's `a1_preload` entry idiom). Diagnostic, non-production configuration:
a synthetic int32 GEMV chain that isolates hand-off latency x depth and the
launch-to-first-stage ramp. It is not a model layer, and the FlyDSL arm is a
twin built from mono's idioms, not mono's kernel.

## Setup

- Hardware: gfx950, one GPU per measurement process. FlyDSL 0.3.4.1; rocKE
  Python engine, llvm23 flavor (COMGR of the job's container).
- Schedule: `depth` levels (1, 2, 4, 8) of 160 tasks on CTAs 0..159, plus a
  1-task `out` stage on CTA 255. Grid 256 x 512 threads, one CTA per CU (LDS
  padded), the 41-argument layer ABI. `work = 0` (hop only) and `work = 8`
  (16 KiB of cold int32 weights per task per level).
- All arms share the buffers, mailbox addresses, LDS size and grid. Each runs
  one graph of 12 chained layers plus the step-counter kernel. Replays are
  interleaved across arms (forward / reverse order per round, 24-32 rounds),
  and L2 / MALL is flushed (512 MiB memset) before every replay.
- FlyDSL baseline arms:
  - mono as shipped, with and without its entry preload (step word load,
    first-level weight loads, `sched_barrier`, then the step wait);
  - the same plus calibrated dual-copy NUMA mailboxes (a lever mono can adopt);
  - a FlyDSL 2-layer twin of the stack.
  
  Each comparison baseline is the per-cell minimum over the relevant FlyDSL
  arms. This is the fairest mono and slightly favours it.
- Numeric gate (every arm, every replay): every mailbox pair (payload and tag),
  every layer's output words and running checksums, against an exact int32
  host model (`toy_chain.toy_reference`, `toy_chain_bench.check_state`).
  Unit tests add 10^4 graph replays straddling the tag wrap.
- Metric: per-layer time = replay time / 12 layers (including launch
  boundaries), median over rounds, as a ratio to the baseline at the same
  depth and work, averaged over the four depths. Stamped twins
  (`FrameworkKnobs(stamps=True)`) give the in-kernel ramp. Later layers of a
  multi-layer launch are decoded from the previous layer's end
  (`toy_chain_bench.decode_launch`).
- Method: ATT (WaveScope, `ROCKE_DEBUG_LOC=1`) before / after each lever, then
  hill climbing with lookback (full-factorial sweeps of the retained knobs).
  Winners are confirmed by three independent replicates. Every losing lever
  stays a default-off knob.
- Provenance: the runs use the `toy_chain_bench` helpers in this folder
  (allocation, graph launches, numeric checks, stamp decode) with an
  interleaving driver. The per-cell raw data is kept in the access-controlled
  record, not here.

## Results (ratios; < 1 = rocKE faster, at these measured configurations; two replicates)

Single-layer launches, against FlyDSL with mono's idioms plus the NUMA lever:

| rocKE build | hop only (`work=0`) | with weights (`work=8`) |
|---|---|---|
| framework defaults (vs mono as shipped) | 1.04-1.05 | 1.12 |
| `best_spec` (DPP sums, calibrated NUMA mailboxes, kernarg preload, `ir` mailbox; prefetch + early weight issue at depth 8) | 0.92-0.93 | 1.01 (depth 1: 1.05-1.07, depth 8: 0.93-0.96) |

Two layers per launch: the stack is a structural lever, and the FlyDSL twin
gains as much from it (0.75 | 0.81 of single-layer FlyDSL). The table
therefore compares stacked against stacked:

| rocKE build | hop only | with weights |
|---|---|---|
| `best_spec(layers=2)` (adds prefetch + early weight issue + `s_sleep` pause from depth 4) vs the FlyDSL 2-layer twin | 0.96-0.99 | 0.95-0.96 (depth 8: 0.89-0.90) |

- Against mono as shipped (no NUMA lever), `best_spec` is 0.73 | 0.98.
- Hop slope against the FlyDSL arm with the NUMA lever: 0.94-0.99 | 0.90-0.93.
- Stamped launch-to-first-stage ramp: 0.47 | 0.41, from kernarg preload.
- The remaining top entry stall is the step-counter load, which the FlyDSL arm also pays.

Losing levers (retained as knobs):
- last-arriver hop;
- reduce-task hop (wins only without NUMA copies);
- one wave per 64 producers;
- sequential and XCD-blocked placement;
- a scalar step-counter load (much slower);
- `fresh_ids=False`;
- prefetch / early issue / `s_sleep` outside the shapes above (they lose at depths 1-4).

## Correctness findings on the way

- The toy's LDS exchange words used a parity taken from the stage's table
  index. An a2a stack's layer boundary (and every level of `reduce_task`)
  reused a copy. Once the next level's weights were L2-hot (idle-CTA
  prefetch), other waves overwrote words thread 0 had not read yet: a wrong
  `y` about once per thousand launches. The parity now follows the per-CTA
  stage order (a barrier where a placement cannot alternate). A stress of
  10^4 replays after the fix found no mismatch.
- Stacked stamped launches: a layer's stamp flush raced the next layer's
  stamp reset (a barrier now separates them), and later layers' stamps need
  decoding from the previous layer's end.

## Caveats

- A toy: hop-only levels exaggerate the entry and hand-off terms relative to
  a real layer; the weight levels are a plain int32 dot product.
- The NUMA lever needs a host calibration of each 2 MiB scratch page's home
  IOD. Kernarg preload and the `ir` mailbox are Python-engine only.
- `fresh_ids=False` miscompiles the hop-only chain at depth >= 3 and was not
  measured there.
