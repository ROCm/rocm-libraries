# Tuning the TDM / ds_load window (gfx1250): a practical guide

How to use the TDM-anchored ds cap window and its neighbours, what was measured,
and how to tune a new kernel. Written for people and for agent sessions: every
section ends in something you can copy, and the measured numbers say what *not* to
repeat. Reference for every knob: [Scheduler tuning parameters](scheduler-tuning-parameters.md).

## TL;DR

```
1. Do nothing first. TDM kernels (>= 128 WMMA in the main loop, tensor_load present)
   already get the window cap from the policy:
     tEnd   = L + TdmStageBytes / 256         L = TdmWindowLatencyCycles, default 330
     window = [tEnd - 240, tEnd + 160)        ds_load cap applies while tensor_load age is inside
     cap    = DsReadPerCap-style 1 per span (2 if the kernel TDMs MX scale tensors)
   fp8 256x256x256 (128 KB/stage): window [602, 1002), +1.1..1.6% TFLOPS.
2. Other families (mxf8, bf16, nvf4, mxf4): the window is neutral (within +-0.3%). Leave it.
3. Already tuned the ds knobs (DsReadPerCap / DsReadThrottleLatency / DsReadDrainLatency)?
   Measure with the window off (DsCapTdmPerCap: 0) first; it can be neutral or negative there.
4. Ds pacing (planned release times) and a ds-cap burst inside the window were tried on hardware, lost
   3-5% / changed nothing, and were removed from the branch. Do not reintroduce them without a new hypothesis.
5. Keep the LDS padding at the default (see "LDS layout rule"): a bad pad costs up to 20x the idle.
```

## What the problem is

During the TDM write phase, back-to-back `ds_load_b128` pairs stall (~11 cycles on the second
load of a pair) when they compete with the TDM's LDS writes. On the fp8 henry kernel
(MT256x256x256, ClusterDim [2,4], TDMInst 3, HalfPLR, SIA4):

- stalls occur only at 400+ cycles after the `tensor_load`; before that ~0%;
- 99% of the ds->ds stalls fall in the write phase (age 400-1000);
- the LDS layout scales the stall rate (default 5.9% of pairs; bad pads 48-64%);
- it is not a ds backlog: stall% falls as recent ds issue density rises.

The window cap thins ds_loads in that phase. The throttle can remove the stalls too, but then
the gated post-barrier ds burst is delayed and idle gets worse (optimum throttle ~96, see below).

## Parameters

| Parameter | Default | Meaning | Measured guidance |
|---|---|---|---|
| `DsCapTdmBeginCycles`, `DsCapTdmEndCycles` | -1 (policy) | window in cycles after the latest `tensor_load` issue | start 400-602 is flat (+-0.2%); start 300: -0.9%; start 700: -0.5%; 800+: -1.3%. End 900-1000 flat |
| `DsCapTdmPerCap` | -1 (policy: 1, or 2 with scale tensors) | ds_loads per span inside the window; `0` = window off | 1 for plain kernels; 1 over-restricts nvf4 (-1.2%) |
| `TdmStageBytes` | set by TensileLite | LDS bytes TDM writes per stage: `(MT0*bytesA + MT1*bytesB)*DepthU` | drives the window width/position |
| `TdmWindowLatencyCycles` (L) | -1 -> 330 | tunable latency in `tEnd = L + bytes/256`; a window-placement knob, not the measured landing time | L 130-330 flat on fp8; L >= 450 loses (window starts too late) |
| `TdmScaleTensors` | set by TensileLite | kernel TDMs MX scale tensors besides data tiles | policy uses cap 2 instead of 1 |
| `DsReadThrottleLatency` | policy | saturated ds spacing = value/16 cycles | untuned henry: 96 is the best; <=88 leaves stalls, >=100 delays the post-barrier burst |
| `DsReadDrainLatency` | policy | barrier drain model | tuned perf-yaml optimum 64 (with throttle 96, DsReadPerCap 2) |
| `DsReadThrottleTransitionFactor/Entries` | 1.0 / 0 | cheaper spacing for the first N loads past the queue depth | removes ~90% of stalls but no TFLOPS gain over throttle 96; skip |
| `BarrierBeforeShiftWmmas` | 0 | shifts ds_load-feeding barriers later | +6..+8: +0.3..0.7% only with window OFF and throttle 96; hurts with the window on (-0.7..-1.7%) |
| `BarrierAfterShiftWmmas` | 0 | shifts tensor_load-feeding barriers | negative everywhere measured (-0.7% at +4); skip |
| `ClusterBarrierWaitHideWmmas` | 0 | places the cluster wait N WMMAs earlier | no gain measured |
| `WmmaQueueDepth/CoverCycles/LowCoverCycles` | see reference | WMMA issue queue | depth 4 / cover 40 neutral; 16/96 costs 2% |

Precedence for each field: your explicit value > the policy > off.

## How to set a value

TensileLite, per solution (fork parameter, can sweep several in one run):

```yaml
- StinkyTofuParameters: [{TdmWindowLatencyCycles: 228}]                       # one setting
- StinkyTofuParameters: [{DsCapTdmPerCap: 0}, {TdmWindowLatencyCycles: 330}]  # two solutions: off vs policy
- StinkyTofuParameters: [{DsCapTdmBeginCycles: 500, DsCapTdmEndCycles: 900}]  # explicit window
```

TensileLite, for every solution of a yaml: `GlobalParameters: StinkyTofuModuleOptions: {...}`.
`stinkytofu-opt`: the `--flag=N` listed in the reference page (not every option has a flag;
`TdmWindowLatencyCycles` is a module option / fork parameter).

After editing StinkyTofu C++: rebuild the binding TensileLite loads, or your results are stale:

```
cd projects/hipblaslt/tensilelite && python3 -m invoke rocisa      # NOT only `invoke build`
```

## Recipes

**A. Keep the default (most kernels).** Nothing to set.

**B. New fp8-class TDM kernel.** Run a two-solution sweep and compare:

```yaml
- StinkyTofuParameters: [{DsCapTdmPerCap: 0}, {}]       # window off vs policy
```

If the policy wins, you are done. Optionally sweep `TdmWindowLatencyCycles` in {130, 228, 330}
(expect a flat result) and `450` (expect a loss).

**C. Kernel with hand-tuned ds knobs.** Compare `{DsCapTdmPerCap: 0}` against the tuned setting
with the window on; on the tuned perf-yaml kernel the window cost -1.2%.

**D. Window position looks wrong for a new kernel.** Sweep an explicit window:

```yaml
- StinkyTofuParameters: [{DsCapTdmBeginCycles: 400, DsCapTdmEndCycles: 800},
                         {DsCapTdmBeginCycles: 500, DsCapTdmEndCycles: 900},
                         {DsCapTdmBeginCycles: 602, DsCapTdmEndCycles: 1002}]
```

Pick the middle of the flat region, then convert to L: `L = begin + 240 - bytes/256`.

## Formulas (what they predict and where they stop)

```
window            = [tEnd - 240, tEnd + 160),  tEnd = L + bytes/256,  L = 330
best window start ~ the TDM write onset (~500 cycles on fp8 128 KB), plateau 400..602
stall% vs throttle: ~13% for T <= 80 .. ~0 at T ~ 108-112 (spacing T/16 ~ 7 cycles/ds), convex
idle (post-barrier bubble): ~ 6.5-8.7 x mean barrier-wait; wait ~ max(0, t_land_all - (1112 + 8*shift))
shift_opt ~ ceil((t_land_all - 1112)/8) ~ 6..10 (limited by delaying the gated ds loads)
```

Limits: the 128 KB all-wave landing is ~1000 isolated and ~1130 in the loop, not 842, and it is
**not linear in stage bytes** (flat ~650 for 32-96 KB fp8 builds, then a jump at 128 KB; it tracks
the descriptor form, not bytes). So the window formula is a placement rule fitted to where the
stalls are, not a model of when data lands. Only the fp8 family showed any sensitivity.

## LDS layout rule (TensileLite)

Same schedule, only the LDS layout changed (henry fp8, idle cycles per iteration):

| Layout | idle/iter | ds pairs stalled |
|---|---|---|
| pad 16 B per 2048 B block (auto) | 255-315 | 5.9% |
| pad 48 / 80 / 112 | 370 / 470 / 510 | 8% and rising |
| LDSSegmentInterleave 0 | 600-680 | 16% |
| pad 8, 24, 32 | ~3000 | 48-53% |
| pad 64 | ~5900 | 64% |
| block 4096 (pad 16 or 32) | ~2700-2900 | n/a |

Rule: pad is 16 mod 32 bytes and the block is 2048 (8 rows of 256 B) for fp8 DepthU 256. Avoid a pad that
is a multiple of 32 or not a multiple of 16. The auto choice is already at the optimum; do not override it.

## Measuring (the part that was wrong in earlier attempts)

- TFLOPS: interleave the variants, >= 5 reps each, report mean and sd (typical sd 5-12). Differences
  under ~0.3% are noise. One run per solution is not enough.
- Stalls: SQTT, ds->ds pairs whose second issue gap is >= 8 cycles; look at stall rate versus age
  since the latest `tensor_load`. Capture noise on pipe-idle is about +-35 per iteration.
- Stall count is not the objective. Removing 90% of the in-window stalls (throttle 112 or the transition
  factor) did not raise TFLOPS; the remaining loss is the barrier wait. Judge by TFLOPS and pipe idle.
- Censoring: a `s_wait_tensorcnt` release time is max(landing, wait position). Measure landing with the
  wait directly after its own `tensor_load` (patched asm), not from the loop's own waits.
- Machine hygiene on a shared board: check that no one else is on the GPU (KFD process list) and run a
  canary kernel (a tuned fp8 kernel must reach >= 9800 TFLOPS) before and after; never run barrier-removal
  or un-timed probes; one Tensile run per problem size (rotating buffer); sensor utilization can read 100% always.
- Agent sessions: use `python3 -m invoke` (not bare `invoke`), run long jobs detached (`setsid nohup`)
  and poll, give every run a timeout.

## Measured results (fp8 henry, 4096x4096x1x65536, gfx1250)

| Config | TFLOPS | vs reference |
|---|---|---|
| throttle 96, window off | 9717 | reference |
| policy window (default throttle) | 9822 | +1.1% |
| window [500, 900) | 9822-9838 | same |
| window [300, 700) | 9741 | -0.9% |
| window [800, 1200) | 9695 | -1.3% |
| throttle 96 + barrier shift +6 | 9746 | +0.3% |
| window + barrier shift +8 | 9657 | -1.7% vs window alone |
| throttle 112 + transition 0.86 | 9674 | = throttle 96 |

Other families, window vs off: f8 +1.6% (L 228-330 best, 450 +1.05%), mxf8 -0.2..+0.1%, bf16 0.0%,
nvf4 +0.2%, mxf4 -0.1% (all noise). Tuned perf-yaml kernel: window off 9854, window on 9734 (-1.2%).

## Known limits and open items

- The window anchors to the `tensor_load` of the region it is issued in (the previous region's load is carried only through the
  after-barrier threshold, assuming identical regions); a region without its own `tensor_load` and an uneven region split are not modelled.
- `TdmStageBytes` is `(MT0*bytesA + MT1*bytesB)*DepthU`: it ignores sparse (compressed A), multicast shares and HalfPLR's extra stages.
- A user value for only some of `DsCapTdm{Begin,End,PerCap}` can leave an empty window (for example `PerCap` alone on a kernel without
  a policy window); the resolver still reports it as a user window. Set all three.
- `stinkytofu-opt` does not run the knob policy; the default-on window applies in the TensileLite/backend pipeline only.
- The 128 KB landing jump is unexplained (hypothesis: descriptor form / number of `AFEM8` operands).
- The best untuned config still leaves a one-per-region barrier-wait bubble after the `tensor_load`
  (WMMA index ~83); a split barrier (signal early, wait late, gated ds loads reordered) would be the real fix.
- Results are from one machine and the fp8 henry family; MX/fp4/bf16 were checked for sensitivity only.
