# Dense attention prefill with launch-time Q/K/V/O strides (gfx942 and gfx950): case study

Measured scope: the rocKE dense attention prefill kernels (`library/kernels/gfx942/attention_dense.py`
and `library/kernels/gfx950/attention_dense.py`) before and after they take the batch, head and
token element stride of Q, K, V and O as twelve trailing launch-time kernel arguments. The "before"
(the baseline) is the same kernels at the commits named per table (`74a573ca382` for gfx942,
`84c6e7ced46` for gfx950); the "after" is this change. The study has three parts: (1) the cost of
reading the strides, measured per kernel body on packed tensors against the baseline (bf16 and
fp16, head size 64 and 128, causal and not, MHA and GQA, sequence length 1K to 16K, LLVM 22, a
256-CU gfx950 and a 304-CU gfx942); (2) a strided-layout sweep of the hipDNN kernel catalog on both
architectures (BSHD, BHSD, packed QKV, packed KV, batch slice); (3) measurement lessons. Every
ratio below is a time ratio between two code paths measured in the same process or sweep on the
same device; absolute times are not recorded here (see
[`platform/AGENTS.md`](../../../../../platform/AGENTS.md), "Public software-performance evidence").
Every compared arm of every measured run passed an on-silicon numeric gate, stated per table.
Statements are bounded to the measured configurations; section 7 lists what was not measured,
and section 3.3 says which measured kernels have changed since. This file lives in the gfx950
prefill folder because that is where the scalar LDS-DMA destination change and most of the bodies
are; the gfx942 prefill README links here.

How to read a ratio: `time(new) / time(baseline)`, so a change of `-1.5%` means the new code is
1.5% faster. "[a, b]" is the 95% interval of a geomean over shapes (two-level bootstrap: shapes,
then rounds). A "shape" is one benchmark case (a model-sized problem); "above the band" is defined
in section 2.

## 1. What changed

Q, K, V and O are logical `[B, S, H, D]` views of any storage layout the host admits (head-dim
stride 1). Before this change a kernel addressed packed `[B, S, H, D]` only, so a head-major,
packed-QKV or batch-sliced tensor needed a copy before the launch. Now:

- **Twelve trailing `i32` kernel arguments**, `stride_{q,k,v,o}_{batch,head,token}`, on every dense
  kernel. They are not a spec field: the kernel name and the compile-cache key do not change, and one
  compiled binary serves every layout the admission rules accept
  (`library/kernels/common/_attention_dense_strides.py`: stride multiples, alignment, overlap and
  size limits).
- **Per-slab addressing** in the bodies that read them: a 64-bit base pointer per (batch, head) slab
  of each tensor, formed per workgroup (per work item in the persistent bodies), and one K and one V
  buffer resource per slab whose size comes from one slab-bound function that the host admission
  check also calls. A load past the slab returns zero and cannot reach another slab.
- **Lane / uniform split.** Everything that does not depend on the work item or the KV tile (the
  per-lane part of the K/V load offsets, slab bounds) is computed once per workgroup. The row term
  stays in the per-lane VGPR offset: a first form that put it into an SGPR multiply per load spilled
  until it went back to the VGPR side.
- **Scalar LDS-DMA destination ("the lever") on the gfx950 grid and KV-segment bodies.** The LDS
  destination of a `buffer_load ... lds` copy is wave-uniform. The wave's first group offset is now
  one SGPR value computed once; each copy adds a compile-time group constant to it, so the KV loop has
  no per-row VGPR and no per-row `v_readfirstlane` for the destination. Only the grid bodies that
  read the strides take it; the packed-only grid bodies (ragged, varlen, paged, additive bias) keep
  the baseline's per-row form. Of the gfx950 persistent bodies, the wide-DMA ones already formed
  this destination in a scalar register in the baseline, and the others keep the per-row form (not
  ported). The lever does not need the strides; it is part of this change because it pays for
  their cost on the gfx950 grid (section 3.1).
- **gfx942: two body changes besides the strides.** Both apply only to the bodies that read the
  strides, and both were chosen by timing the 30 gfx942 shapes of section 3.2, which lists the
  runs and the per-shape timings behind each choice (mechanism in section 4).
  - The LDS-DMA destination is chosen per body: a scalar base formed once per kernel on most
    strided bodies; the per-row VGPR form on the non-causal bf16 D128 grid and KV-segment bodies,
    where the scalar base timed slower (why was not found); and a scalar base formed once per KV
    tile with a sliding window, where the per-kernel form spills SGPRs.
  - On the strided D128 bodies with conflict-free V, O is rescaled first and each PV MFMA is
    emitted as soon as the P pack it reads exists, in the `PV_STEPS` order. With the default
    `pv_loop_order="d_major"` that holds for the first output tile's chain (8 of the 32 PV
    MFMAs); the other chains follow the last pack. The output is bit-identical. Setting
    `pv_priority` or a PV fence mask turns this order off.

Which bodies read the strides, and which keep packed addressing (they still declare the twelve
arguments, so the kernel ABI is one; the host admits only packed tensors for them):

| | read the strides | packed-only |
|---|---|---|
| gfx950 | grid, its KV-segment form, persistent in every work order that decodes through the shape arguments, wide-DMA persistent | `gqa_pair` work orders, ragged, varlen, paged, fp8 K/V-cache and additive-bias bodies |
| gfx942 | default grid, persistent grid, KV-segment grid | the opt-in K LDS double-buffer and V register-prefetch pipelines, compile-time-shape specs (paged), additive-bias bodies |

The additive-bias bodies (`use_additive_bias`, hipDNN's `attn_mask`) are packed-only on both
architectures. Their bias read indexes only by the bias's own three strides. They ignore the twelve
because they were added while this work was open and were not ported to it; on gfx950 the D128 bias
body with LSE already sits at the 256-VGPR budget (it keeps its bias loads scalar because the
4-wide ones spill), so the extra stride arithmetic may not fit. Serving strides there is tracked in
AICK-3355. The hipDNN dense engines decline a graph that has an `attn_mask` and operands that are
not packed BSHD, with the cause `attn_mask` in the INFO log.

A KV-segment launch with a strided O is refused (the combine step needs a packed O). The baseline
launches packed tensors only, so the strided layouts have no baseline arm: their effect is measured
against the BSHD twin in the same tree (section 5).

## 2. Method

**Speed harness** (`library/benchmarks/common/attention_dense_strided_ab.py`, statistics in
`attention_ab_stats.py`). Arms are rocKE trees that cannot be imported side by side, so each tree
exports its code objects, kernel arguments and launch geometry in its own process (one compiler, no
torch); a second process then times every arm:

- **One process, one set of buffers.** Inputs come from a fixed seed with decoy data around the views
  and NaN in O. Every arm's code objects are loaded into the same process with the pointer arguments
  patched to those buffers.
- **A/A' control.** The baseline's code objects are loaded a second time and timed as another arm.
  Its ratio to the baseline measures noise on shapes that did not change.
- **Rounds.** At least 24 rounds in a balanced mirrored order (each round reverses the previous one,
  the start rotates, so every arm sits in every position equally often). One measurement is one
  HIP-graph replay of N launches of one arm, with N sized to a fixed device time.
- **Per-shape ratio** = median over rounds of the per-round ratio, with a 95% interval over rounds;
  the geomean over shapes has a two-level bootstrap interval.
- **Band rule** (the gate of part (1)): band = max(1%, p95 over shapes of |A'/A - 1|). A body passes
  if the upper 95% bound of its geomean is below +1% and no shape has its whole interval above the
  band.
- **Specs** come from the production dispatch factory (`library/dispatch/attention/gfx950_dense.py`,
  `gfx942_dense.py`); a hand-built spec once hid a large real difference in an earlier study. A
  shape that the factory does not serve (below) is built from the factory's spec with one named
  field replaced.
- **Compiler and device.** Device runs here used LLVM 22 (the compiler the harness process loads,
  recorded in each result) on a 256-CU gfx950 and a 304-CU gfx942. Both arms of a comparison ran in
  the same job on the same device. Other compilers and the 228-CU gfx942 part were not run on a
  device for speed (section 7).
- **Numeric gate in the measured run, every arm.** The output of every arm is checked before and
  after timing. Two forms were used, stated per table: (a) the in-tree harness checks every arm
  against an fp64 reference on a row subset (up to three (batch, head) pairs and three windows of
  64 query rows) at max-abs error <= 2e-2 for fp16 and <= 4e-2 for bf16, with no NaN, and hashes
  the output; (b) the earlier scratch versions of the same in-process harness that measured the
  gate rows of part (1) checked the baseline arm against that fp64 reference and every other arm
  against the baseline's output; in every row of those runs the new output was bit-identical to
  the baseline's, so it has the same error against the reference. A case that fails the gate never
  enters a geomean.

The commands below run from the rocKE root with `platform/python` and `library` on `PYTHONPATH`
(`<baseline tree>` and `<this change>` are checkouts):

```
python library/benchmarks/common/attention_dense_strided_ab.py --list --arch gfx950 \
    --shapes library/benchmarks/common/attention_dense_strided_ab_shapes.yaml \
    --arm A=<baseline tree>/dnn-providers/hip-kernel-provider/rocke \
    --arm B=<this change>/dnn-providers/hip-kernel-provider/rocke       # CPU: factory choice per arm
python library/benchmarks/common/attention_dense_strided_ab.py --arch gfx950 \
    --shapes library/benchmarks/common/attention_dense_strided_ab_shapes.yaml \
    --arm A=<baseline tree>/dnn-providers/hip-kernel-provider/rocke \
    --arm B=<this change>/dnn-providers/hip-kernel-provider/rocke \
    --baseline A --control A2 --rounds 24 --seed 0 --out RESULT.yaml
python library/benchmarks/common/attention_dense_strided_ab.py bar RESULT.yaml
```

`run` and `bar` share one set of exit codes, so a script can gate on either: 0 only for a complete
run on a device whose bar says PASS; 1 FAIL or a failed case; 2 no verdict (no data,
`--no-stats`, or refused input such as an unreadable or stale export or arms built by different
compilers, with the reason on stderr); 3 too noisy to judge; 4 INCOMPLETE; 5 a dry run. A dry
run times nothing, so its bar reads NOT A MEASUREMENT and never PASS. The result lists the
expected cases at the start and is marked complete only when the case loop ends; a run that
stopped early, or a result with cases missing, reads INCOMPLETE. A case with fewer than 24
paired rounds fails the bar
(`bar --min-rounds` sets the number). Each export records the arch, the case ids, a shape
signature per case, the size of the largest segment-workspace buffer the runner asked for
(`workspace_bytes`), the CU count and the compiler (LLVM version and COMGR path); `run` refuses an
export made for another arch, case shape, CU count or compiler than the baseline's, so a stale
export cannot be timed by mistake. These checks were added after the scratch runs of section 3
(section 7).

The shipped shape file holds the gfx950 grid family (13 shapes). The other sets of this study are
listed in the table below in the same row schema; the two that are not shipped as files (the gfx950
persistent family and the gfx942 set) must be written from this table to repeat those rows.

| id | model / head slice | Hq/Hkv | D | S (Sq = Skv) | mask | dtype | gfx950 body | gfx942 body |
|---|---|---|---|---|---|---|---|---|
| g01 | Llama-3-8B | 32/8 | 128 | 1024 | none | fp16 | grid | persistent; forced grid as g01G |
| g02 | Llama-2-7B | 32/32 | 128 | 1024 | none | bf16 | grid | persistent; forced grid as g02G |
| g03 | - | 32/8 | 64 | 1024 | causal | bf16 | grid | grid |
| g04 | - | 32/32 | 64 | 1024 | none | fp16 | grid | grid |
| g05 | Llama-3-8B, 2-way TP | 16/4 | 128 | 2048 | none | bf16 | grid | persistent; forced grid as g05G |
| g06 | - | 16/2 | 64 | 2048 | causal | fp16 | grid | grid |
| k01 | Llama-3-70B, 8-way TP | 8/1 | 128 | 4096 | causal | fp16 | KV-segment | KV-segment (forced) |
| k02 | Llama-3-8B, 4-way TP | 8/2 | 128 | 2048 | none | bf16 | KV-segment | KV-segment (forced) |
| k03 | - | 8/1 | 64 | 4096 | causal | bf16 | KV-segment | KV-segment (forced) |
| h01 | Llama-3-8B | 32/8 | 128 | 4096 | causal | fp16 | grid, query-block-minor (pinned) | persistent; forced grid as h01G |
| h02 | Llama-3-8B | 32/8 | 128 | 16384 | none | bf16 | grid, query-block-minor (pinned) | persistent |
| h03 | - | 64/8 | 64 | 8192 | causal | fp16 | grid, query-block-minor (pinned) | grid |
| h04 | Llama-2-7B | 32/32 | 128 | 8192 | none | bf16 | grid, query-block-minor (pinned) | persistent |
| p01 | Llama-3-8B | 32/8 | 128 | 2048 | none | fp16 | persistent | persistent |
| p02 | Llama-3-8B | 32/8 | 128 | 8192 | causal | bf16 | wide-DMA persistent | persistent |
| p03 | Llama-3-8B | 32/8 | 128 | 16384 | none | bf16 | persistent | persistent |
| p04 | Llama-3-70B | 64/8 | 128 | 1024 | causal | fp16 | wide-DMA persistent | persistent |
| p05 | Llama-3-70B | 64/8 | 128 | 4096 | none | bf16 | persistent | persistent |
| p06 | Llama-2-7B | 32/32 | 128 | 2048 | causal | bf16 | wide-DMA persistent | persistent |
| p07 | Llama-2-7B | 32/32 | 128 | 4096 | none | fp16 | persistent | persistent |
| p08 | Llama-2-7B | 32/32 | 128 | 16384 | causal | fp16 | wide-DMA persistent | persistent |
| p09 | - | 64/8 | 64 | 2048 | causal | bf16 | persistent | grid |
| p10 | - | 64/8 | 64 | 8192 | none | fp16 | persistent | grid |
| p11 | - | 32/32 | 64 | 4096 | causal | fp16 | persistent | grid |
| p12 | - | 32/32 | 64 | 16384 | none | bf16 | persistent | grid |
| p13 | Llama-3-8B (batch 4) | 32/8 | 128 | 2048 | causal | fp16 | wide-DMA persistent | persistent |

All shapes have batch 1 except p13. "Pinned" and "forced" rows are not what the production factory
serves for that shape: the gfx950 query-block-minor grid at long sequence is not picked by the
dispatcher for these head counts, the KV-segment rows take the factory's grid spec with the factory's
own KV-split rule applied, and the gfx942 factory serves head size 128 by the persistent body, so the
"G" rows force the grid body. These rows are diagnostic, non-production configurations; they test
that a body that can be launched keeps its speed. Of the gfx950 grid family, 6 shapes (g01-g06) are
factory-served and 7 are forced or pinned; of the gfx942 grid group, 8 are factory-served and the
4 "G" rows are forced; all 3 gfx942 KV-segment shapes are forced.

## 3. Results against the baseline

All rows: packed contiguous BSHD tensors (the cost of reading the strides, not of a strided
layout), ratios `new / baseline`, geomean over the shapes of the row, one device and one compiler
per table. The numeric gate passed for every arm in every run of these tables.

### 3.1 gfx950 (256 CU, LLVM 22), baseline `84c6e7ced46`

These kernels were measured before a later change of the gfx950 bodies in the base (section 3.3).
The 26 shapes are the selection set of the gfx950 work. The re-time on the code after that change,
with the shipped harness, is in section 7.1; where the two differ, section 7.1 is the current
result (the grid family still passes, the persistent family does not).

Shapes: the 13 grid-family shapes (g01-g06 grid, k01-k03 KV-segment, h01-h04 grid pinned) and 13
persistent shapes (p01-p13; p02, p04, p06, p08, p13 take the wide-DMA body). A/A' p95 per table:
0.18% (grid family), 0.16% (persistent). Numeric gate: form (b): every arm bit-identical to the
baseline's output, the baseline within tolerance of the fp64 reference, before and after timing,
on all 26 shapes.

| Bodies | Arm | vs baseline | notes |
|---|---|---|---|
| grid, KV-segment, pinned grid (13 shapes) | this change (strides + lever) | **-1.48% [-1.79, -1.19]** | repeat with other inputs -1.56% [-1.81, -1.32]; per shape from -2.55% (g03) to -0.43% (h02); no shape slower |
| same | lever only (diagnostic: arguments declared and ignored, lever on) | -1.78% [-2.23, -1.35] | same run |
| same | reading the strides, relative to the lever-only arm | +0.30% [+0.04, +0.55] | per shape from -0.57% (h01) to +0.99% (g06) |
| persistent + wide-DMA (13 shapes) | this change (strides, no lever) | **+0.27% [+0.02, +0.53]** | repeat +0.28% [+0.04, +0.52]; persistent +0.20%, wide-DMA +0.39% |
| same | reading the strides, relative to the declared-but-ignored arm | +0.28% [+0.04, +0.52] | |
| same | declared-but-ignored arguments only (diagnostic) | +0.01% [-0.04, +0.06] | no measurable cost |
| control | harness with no change (a third load of the baseline's code) | -0.01% [-0.06, +0.06] (grid), -0.00% [-0.04, +0.04] (persistent) | the harness reads zero where there is no change |

So reading the strides costs about 0.3% on both families. On the grid family the lever more than
pays for it; the persistent family has the cost and no offset.

Pass rule: both bodies pass (upper bounds -1.19% and +0.53% against +1%, no shape wholly above the
1% band). Two things to know about the persistent row. First, it passes only through the 1% floor
of the band, not the measured noise: against the measured A/A' noise alone (0.16% p95), six of
the 13 shapes (p02, p06, p09, p10, p12, p13) have intervals wholly above it. The largest is p02
(Llama-3-8B, causal, S 8192, bf16, wide-DMA) at +1.02% [+0.90, +1.06]: its median is just above
the 1% floor and its interval straddles it; p10 is +0.96% [+0.87, +1.08]. Second, nothing offsets
the stride cost here as the scalar LDS-DMA destination does on the grid: the non-wide persistent
bodies keep the per-row destination (the scalar form is not ported to them), and the wide-DMA
ones, p02 among them, already had a scalar destination in the baseline, so for them the residual
is the stride addressing itself, whose cause within the body has not been isolated.

An independent re-measurement of the lever alone ran on an earlier version of the shipped
harness (`SCRIPT_VERSION` `dense-strided-ab-1`; the version in this change ran later, section 7.1).
Baseline `74a573ca382` against a build with the lever and the twelve
declared-but-ignored arguments, A/A' p95 0.15%: -1.86% [-2.30, -1.44], 13 of 13 shapes faster,
numeric gate of form (a) on every arm; two more seeds gave -1.76% [-2.18, -1.37] and -1.86%
[-2.30, -1.47]. (`74a573ca382` and `84c6e7ced46` differ in two gfx950 IR rows, both varlen,
which are not timed.) Run in a separate process per arm, the same pair gave A/A' p95 of 2.05% to
2.82% (section 6).

### 3.2 gfx942 (304 CU, LLVM 22), baseline `74a573ca382`

The selection set: the 30 shapes that chose the gfx942 body changes, so the intervals below are
post-selection (the held-out run on 22 other shapes is in section 7.1). 30 shapes in three groups
(table above): grid 12 (8 factory-served D64 shapes and the 4 forced "G" D128 rows), KV-segment 3
(forced), persistent 15. Numeric gate: form (b), 30 of 30 shapes, every
arm, before and after timing: the new output bit-identical to the baseline's, the baseline within
tolerance of the fp64 reference.

| Body | shapes | A/A' p95 | new vs baseline | repeat | shapes above the 1% band |
|---|---|---|---|---|---|
| grid | 12 | 0.16% | **-0.14% [-0.73, +0.42]** | -0.16% [-0.77, +0.41] | g02G +1.76% [+1.54, +1.95] |
| KV-segment | 3 | 0.22% | **+0.09% [-0.92, +1.20]** | +0.14% [-0.71, +0.97] | k03 +1.21% [+1.16, +1.23] (+0.97% in the repeat) |
| persistent | 15 | 0.15% | **-3.50% [-5.56, -1.49]** | -3.53% [-5.62, -1.53] | none |

Under the pass rule of section 2, the persistent row passes and the grid and KV-segment rows fail.
The grid geomean's upper bound is below +1%, but one shape is wholly above the 1% band: g02G
(forced grid, bf16 D128 non-causal MHA, S 1024, a non-production configuration: +1.76% in the run
above, +1.54% [+1.29, +1.73] in the repeat). The KV-segment row fails both conditions: its upper
bound is +1.20%, and k03 (bf16 D64 causal MQA 8/1, S 4096) is +1.21% [+1.16, +1.23], wholly
above the band. In the repeat, k03 is +0.97% [+0.92, +1.05], not wholly above the band; every
earlier arm that read the strides put it between +1.0% and +1.4%. With three shapes, all forced,
the interval's upper bound is essentially k03's value (+1.20% against +1.21%) and its lower bound
sits near k01 (-1.05%), so that row describes three shapes and gives no verdict on the body.
Both shapes are accepted as residuals of this change, not passed (section 7). The persistent
gain is concentrated at short sequence: the four 1K-2K shapes (g01, g02, g05, p01) are faster by
about 8% to 10%, the long-sequence shapes (S >= 4096) are within about +-1%, and the shapes with
a median above zero are p08 (+0.61%), p13 (+0.53%) and h01 (+0.19%), all inside the band.

**What the gfx942 persistent gain is made of.** The new body changes three things: it reads the
strides, it forms the LDS-DMA destination per body, and it emits the D128 PV MFMAs in issue
order, each once the P pack it reads exists, in `PV_STEPS` order (section 1; with the default
`d_major` order only the first output tile's chain follows its own pack, the rest follow the
last pack). Arms timed on the same 15 persistent shapes (geomean against the baseline): the
strides alone -2.31% [-4.41, -0.36]; the scalar LDS destination alone -1.95% [-3.99, -0.11];
strides with the scalar destination -2.84% [-5.18, -0.63] (the same code in another job: -2.83%
[-5.31, -0.65]); plus the PV order -3.52% [-5.62, -1.54]; the final code -3.50%. The PV order is
worth about 0.7 points, from p04 (-0.7% to -4.6%), p06 (-0.9% to -5.7%) and p08 (+2.1% to
+0.5%); it gives back 1.1 to 1.5 points on g02 and g05. Most of it is p04 and p06, block_m 128
causal kernels that run every softmax exp before the PV MFMAs in the baseline and in every arm
without the PV order, so there the order adds an overlap the baseline never had; on p08 it gives
back part of an overlap the address changes had removed (section 4). The 8-10% at 1K-2K (g01,
g02, g05, p01) is in every arm, the strides alone included: at LLVM 22 the baseline's block_m 128
non-causal persistent kernel runs all softmax exps before the PV MFMAs, and every changed kernel
of those shapes overlaps them. So that part of the gain is a compiler schedule effect, not the
effect of either body change. On the grid the same arms give: strides alone +0.74% [-0.21,
+1.83], the scalar destination alone -0.08% [-0.99, +0.86], both +0.40% [-0.54, +1.51]; the PV
order moves the grid by less than 0.1 point. Limits: LLVM 22 only, one 304-CU part, scratch
versions of the harness in two jobs, and the same 30 shapes that chose the destination form and
the PV order, so the intervals are post-selection. At LLVM 23 and 24 the baseline kernel's
schedule differs (offline ISA), so these sizes do not carry over.

**The timings that chose the gfx942 body changes.** Every run below used the 30 shapes of
section 2 (or a subset), LLVM 22, the same 304-CU part, scratch versions of the in-process
harness and the numeric gate of form (b) on every arm. They took three device allocations (the
first-form run, run S, and runs C to F'), so compare arms within one run.

| run | shapes, rounds | arms | used to |
|---|---|---|---|
| first form | 30, 24 | B | find the shapes the first form made slower |
| S (arm split) | 10 picked, a repeat, then 30; 24 | B, S2S, SLS, SA2 | locate that cost |
| C (candidate screen) | 30, 28 | B, LA, LAPS, LANC, LANCPS | choose |
| C' (repeat, other inputs) | 10 of the 30, 28 | same | choose |
| M (mini screen) | 7 of the 30, 24 | B, LANC, LANCPS, and LANCPS with a scheduling barrier before the tile's closing barrier | choose (the barrier was rejected) |
| F, F' (final, two input seeds) | 30, 24 | LANCPS = this change | report (the table above) |

Arms: SLS is the baseline's addressing (no strides read) with the scalar destination formed once
per kernel, the destination change alone; SA2 reads the strides with the baseline's per-row
destination, the strides alone; S2S reads the strides with the scalar destination formed once
per kernel; B reads the strides and forms it once per KV tile (the first form); LA is the S2S
form in the body of this change (once per KV tile only with a sliding window; on p01 and p08 it
compiles to the same `.text` as S2S); LANC adds the per-row form on the non-causal bf16 D128
grid and KV-segment bodies; LAPS and LANCPS add the PV issue order to LA and LANC. Per shape,
the median over rounds against the baseline with its 95% interval, on the four shapes behind the
two destination choices (all four are grid or KV-segment shapes; g02G, g05G and k02 are forced,
section 2):

| arm | run | g02G (bf16 D128, none) | g05G (bf16 D128, none) | k02 (bf16 D128, none, KV-segment) | p09 (bf16 D64, causal) |
|---|---|---|---|---|---|
| SLS: destination alone | S | +2.58% [+2.28, +2.90] | +3.31% [+3.06, +3.66] | +1.96% [+1.84, +2.11] | +0.26% [+0.10, +0.42] |
| SA2: strides alone | S | +1.23% [+1.04, +1.63] | +0.12% [-0.21, +0.22] | +0.38% [+0.01, +0.53] | +4.89% [+4.67, +5.10] |
| S2S: strides, base once per kernel | S | +4.02% [+3.64, +4.23] | +4.11% [+3.87, +4.20] | +1.22% [+1.05, +1.48] | +0.17% [+0.05, +0.31] |
| B: strides, base once per KV tile | S | +4.26% [+3.90, +4.72] | +4.51% [+4.31, +4.81] | +0.79% [+0.68, +0.94] | +2.25% [+1.97, +2.33] |
| B | C | +4.46% [+4.13, +4.57] | +4.54% [+4.45, +4.73] | +0.84% [+0.69, +1.03] | +2.11% [+1.93, +2.17] |
| LA: base once per kernel | C | +4.09% [+3.90, +4.18] | +4.05% [+3.79, +4.15] | +1.31% [+1.19, +1.51] | +0.12% [+0.04, +0.34] |
| LANC: LA, per-row form on the non-causal bf16 D128 grid and KV-segment bodies | C | +1.37% [+1.29, +1.45] | -0.13% [-0.28, +0.19] | -0.62% [-0.83, -0.39] | +0.23% [+0.14, +0.37] |
| LANCPS: this change | F | +1.76% [+1.54, +1.95] | +0.22% [+0.02, +0.63] | +0.13% [-0.18, +0.30] | +0.13% [+0.01, +0.23] |
| LANCPS: this change | F' | +1.54% [+1.29, +1.73] | +0.42% [+0.28, +0.53] | +0.28% [-0.12, +0.38] | +0.12% [+0.07, +0.27] |

What chose what:
- **Once per kernel, not once per KV tile.** p09 (B +2.25% against S2S +0.17% in run S; B +2.11%
  against LA +0.12% in run C) and g03 in run C (B +1.03%, LA +0.10%), both D64 bf16 causal grid
  shapes. The sliding-window bodies keep the per-tile form because the per-kernel one spilled
  SGPRs on 10 window grid kernels of the 432-kernel offline resource matrix; no shape here has a
  window, so neither form was timed on a window body.
- **The per-row exception (`LDS_ROW_VGPR`).** Chosen from g02G and g05G alone, both forced
  non-causal grid shapes (LA to LANC: +4.09% to +1.37%, +4.05% to -0.13%); the KV-segment half of
  the rule rests on k02 alone (+1.31% to -0.62%). Causal bodies are left out because the per-row
  form spills VGPRs to scratch on causal bf16 D128 grid and KV-segment kernels (offline, LLVM 22
  and 23), so that variant was never timed. Why the scalar base is slower on these shapes was
  not found: the static KV-loop counts do not rank the arms the way the times do, and no profiler
  ran.
- **The PV issue order.** LA to LAPS in run C only (the persistent paragraph above); on the grid
  it moves the geomean by less than 0.1 point.

Section 7.1 gives the held-out run on 22 other shapes that tested these choices: the PV order and
the scalar base held in part, the per-row exception showed no effect, and the grid and KV-segment
families fail the bar on one shape each.

### 3.3 The code that was timed and the code in this change

**gfx942: the timed code is the code in this change.** The representative IR rows that existed
when the gfx942 rows were measured are identical in the current tree at LLVM 20, 22, 23 and 24: 8
of 8 gfx942 rows in the shared shard
(`platform/tests/golden/rocke_representative_ir_sha256/attention_dense.json`) and 51 of 51 in the
gfx942 library golden. An offline LLVM 22 compile of the current tree gives the same `.text` as
the measured arms on the checked shapes (p01, p08, p09, g02G, g05G). The baseline side has not
moved either: `74a573ca382`, `c3448b0c5f5` and the current base have the same 8 shard rows and 51
library rows.

**gfx950: the timed code has changed since.** The gfx950 rows were measured on a tree built on
`84c6e7ced46`. A later change in the base, #15951 (each query row decides its own lazy rescale),
changed both bodies that were timed, the grid body and the persistent body, in the baseline as
much as in this change: it moved 45 of the 47 gfx950 representative rows that existed at
measurement. Section 3.1 and the gfx950 half of the catalog sweep (section 5) therefore describe
the gfx950 kernels before that change. The speed rows were re-timed on the code after it (section
7.1); the catalog sweep has not been re-run. (`84c6e7ced46` to
`c3448b0c5f5` moved none of the 47 rows.)

Rows added since the measurements were not timed: the fp8 K/V-cache and paged rows (packed-only),
the additive-bias rows (19 gfx942 and 5 gfx950 in the shared shard, 29 in the gfx942 library
golden; packed-only, section 1), and two rows that pin bodies the study did time (the gfx942
non-causal D128 grid and the gfx950 wide-DMA persistent body in the query-block-major order). Since
this study was first written no existing row has moved: 72 of 72 shard rows and 51 of 51 library
rows are unchanged at the four compilers. These are representative rows, not every shape timed
above. For arms taken from another tree, the harness runs of part (1) checked that the IR, the code
object and the `.text` section of each cross-loaded arm were identical to what its own tree builds,
on all 26 gfx950 shapes.

## 4. What the kernels showed

- **The lever pays for the strides on the gfx950 grid.** Moving the LDS-DMA destination to a
  scalar register is worth about 1.8%; the per-slab 64-bit base, per-slab K/V buffer resources and
  the stride arithmetic cost about 0.3% (the +0.30% row). Net, the grid family is faster than the
  baseline on the 13 measured shapes. The non-wide persistent bodies have the cost and not yet the
  gain (the scalar destination is not ported to them); the wide-DMA persistent bodies already had
  it in the baseline, so their residual is the stride addressing itself, with no isolated cause
  yet. On gfx942 the same lever alone is not a general win: it is -1.95% on the persistent family
  but +2.6% and +3.3% on the two forced non-causal bf16 D128 grid shapes g02G and g05G (the
  per-shape table in section 3.2), which is why gfx942 chooses the form per body.
- **Keep the lane / uniform split.** What does not depend on the work item is hoisted out of the
  persistent work loop; the row term belongs in the per-lane VGPR offset, not in a scalar multiply
  per load (that form spilled).
- **gfx942 D128 speed follows the compiler's schedule of the KV loop.** At LLVM 22 the
  address-code changes (the strides alone, the lever alone, and both) moved the softmax exps and
  the PV MFMAs in both directions, by kernel (offline ISA of the persistent D128 shapes):
  - block_m 256 causal (p02, p08, p13, h01): the baseline overlaps the exps with the PV MFMAs;
    every changed arm moved every exp ahead of the PV MFMAs, and those shapes got slower.
  - block_m 128 non-causal (g01, g02, g05, p01): the baseline runs every exp first; every
    changed arm overlaps them. This is the 8-10% gain at 1K-2K of section 3.2.
  - block_m 128 causal (p04, p06): the baseline and every arm without the PV order run every exp
    first.
  - block_m 256 non-causal (h02, h04, p03, p05, p07): overlapped in the baseline and in every arm.

  Emitting the PV MFMAs in issue order (section 1) gives the block_m 256 causal kernels part of
  their overlap back: in the LLVM 22 ISA of p08, 11 MFMAs sit between the first and the last exp,
  against 26 in the baseline. On the block_m 128 causal kernels it adds an overlap the baseline
  never had (13 MFMAs in the exp span of p04), which is where most of its gain comes from
  (section 3.2). With the default `d_major` order only the first output tile's chain (8 of the
  32 PV MFMAs) follows its own pack; all 32 do only with `pv_loop_order="k_major"`, which the
  held-out run timed (section 7.1): slower on the causal D128 persistent kernels and spilling on
  two D64 grid shapes, so it is not a default. With this order the persistent result moved from a smaller gain with
  several shapes above the band to the -3.5% above. The schedule is compiler-specific (offline
  ISA): at LLVM 23 and 24 the baseline's block_m 256 causal persistent kernel is itself
  serialized and the default PV order stays serialized, while the block_m 128 baseline kernels
  that LLVM 22 serializes are overlapped there. The destination form is a per-body choice on
  gfx942: the non-causal bf16 D128 grid and KV-segment bodies keep the per-row form.
- **Offline resource checks** at LLVM 20, 22, 23 and 24 (no scratch, no fewer waves per SIMD than
  the baseline, no extra SGPR spill) passed for every served body, with one exception: one gfx942
  grid kernel (g06) at LLVM 20 spills two SGPRs into VGPR lanes (no scratch). The form that avoids
  it costs speed on another measured shape, so it is kept. The per-body counts behind these checks
  are not reproduced here, and no test pins the ISA form of either lever; the IR hash goldens catch
  any change of the emitted IR.
- **LDS-DMA zero fill.** That an out-of-range lane of an LDS-DMA copy writes zeros was checked on
  gfx950 with a guard-page microtest; it was not checked on gfx942.

## 5. Strided-layout catalog sweep (hipDNN kernel catalog, both architectures)

Question: does a strided layout need its own kernels, or does the catalog's best kernel for a graph
change with the layout? And does the shipped catalog contain the best kernel at all?

Set-up. 120 graphs per architecture: 8 families (bf16 and fp16, D64 and D128, causal and no mask) x
3 LLM-prefill shapes (GQA 32/8, batch 2, S 2048; MHA 32/32, batch 1, S 4096; a 16/2 head slice of a
GQA model, batch 2, S 1024) x 5 layouts (BSHD, BHSD, packed QKV, packed KV, batch slice). A "cell" is
a family on one layout (40 cells; 32 strided). The catalog is the shipped dense attention catalog at
`c3448b0c5f5`, from before the additive-bias kernels were added (this change does not alter which
kernels ship), built by the tree measured so that its kernels take the strides; the enumeration arm
is the same manifest with the full knob space, cut to the head-count pairs the graphs use (gfx950:
3912 kernels, grid bodies only, as no persistent kernel ships in that catalog; gfx942: 256
kernels). Every graph is served by the hipDNN kernel-ingestor engine and timed through
`dnn-benchmark` by `sweep.py`. The tree that built the catalog and the enumeration predates
#15951 (section 3.3), so the gfx950 half of this sweep (sections 5.1 and 5.2) timed the grid
body as it was before that change, and it has not been re-run since; #15951 changes only gfx950
code, so the gfx942 half is not affected by it.

None of these graphs has an `attn_mask`, so the bias kernels shipped since would not serve them
(and they are packed-only, section 1). Counted as distinct binaries in the slice this recipe
measures (no LSE twin, no sliding window), the shipped catalogs now hold 60 on gfx950 (52 without
a bias, 8 with one) and 96 on gfx942 (72 + 24), and the full-space arm the recipe builds holds
1,920 on gfx950 (1,304 + 616) and 300 on gfx942 (256 + 44). The bias-free halves are the counts
this sweep was designed on; `projects/hipdnn/tools/IngestorGenerator/tests/test_catalog_sweep_report.py`
pins both halves.

Phases of one campaign per architecture, in one allocation. gfx950: a one-round screen of the whole
enumeration (timing only; used only to pick a shortlist, never for a verdict, since the minimum of
many noisy timings is biased fast); gfx942: no screen, the 256-kernel enumeration goes straight to
the verdict stage. Verdict stage: five rounds on the shortlist plus every shipped kernel, two copies
of one install tree as the A/A' control, with the `sweep.py` correctness phase enabled. Then
residual single-kernel sweeps for the graph and kernel pairs that no correctness phase had covered.
**Numeric gate:** the sweep times and checks in separate phases of the same campaign
(`dnn-benchmark` correctness against a PyTorch reference, rtol 0.03, atol 0.001); timing phases do
not check output. After the residual sweeps the best shipped kernel and the best enumerated kernel
of every graph were checked: 120 of 120 and 120 of 120 passed on each architecture (before the
residual sweeps: 115 and 116 of 120 on gfx950, 110 and 114 of 120 on gfx942). The numbers below are
cell geomeans over the graphs of the cell, against a pre-registered band of 3% (the floor; raised
per cell only when the round-to-round drift is larger). A/A' p95 was 0.99% on gfx950 and 0.82% on
gfx942.

Commands (from the repository root; see `projects/hipdnn/tools/IngestorGenerator/tools/README-sweeps.md`
for the sweep configuration schema):

```
python projects/hipdnn/tools/IngestorGenerator/tools/ground_truth.py layouts \
    --manifest projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_dense.engine.yaml \
    --shapes LAYOUT_SHAPES.yaml --out GRAPHS        # graphs + GRAPHS.layouts.yaml (the map)
python projects/hipdnn/tools/IngestorGenerator/tools/sweep.py --config SWEEP.yaml
python projects/hipdnn/tools/IngestorGenerator/tools/catalog_sweep_report.py screen \
    --sweep-output OUT --arm ARM --arm-kdp ARM_KDP --shipped-kdp SHIPPED_KDP \
    --map GRAPHS.layouts.yaml --manifest MANIFEST --out SCREEN.yaml --out-names SHORTLIST.txt
python projects/hipdnn/tools/IngestorGenerator/tools/catalog_sweep_report.py report \
    --sweep-output OUT --arm ver --control-arm ver_prime --check-arm ver --check-arm shipped \
    --arm-kdp ARM_KDP --shipped-kdp SHIPPED_KDP --map GRAPHS.layouts.yaml \
    --manifest MANIFEST --out REPORT.yaml
```

with `LAYOUT_SHAPES.yaml` (the schema is `read_layout_spec` in `ground_truth.py`):

```
dtypes: [bf16, fp16]
head_dims: [64, 128]
mask_types: [TOP_LEFT_CAUSAL, NO_MASK]
layouts: [bshd, bhsd, packed_qkv, packed_kv, batch_slice]
shapes:
  - {id: gqa_32_8, batch: 2, nhead_q: 32, nhead_k: 8, seqlen_q: 2048}
  - {id: mha_32_32, batch: 1, nhead_q: 32, nhead_k: 32, seqlen_q: 4096}
  - {id: tp_16_2, batch: 2, nhead_q: 16, nhead_k: 2, seqlen_q: 1024}
```

and `SWEEP.yaml` listing the install tree twice as the arms `ver` and `ver_prime`, 5 rounds,
`correctness: {enabled: true, reference: pytorch}`.

### 5.1 Layout effect: none measured

Per layout, the strided cell's geomean over its graphs divided by the geomean of its BSHD twin on
the same shapes, minus 1 (largest and median over the 8 families):

| layout | gfx950 max / median | gfx942 max / median |
|---|---|---|
| BHSD | +0.48% / -0.06% | +2.51% / +0.00% |
| packed QKV | +0.49% / -0.21% | +0.25% / +0.00% |
| packed KV | -0.14% / -0.31% | +2.30% / +0.00% |
| batch slice | +0.15% / -0.05% | +1.36% / +0.20% |

No strided cell is worse than its BSHD twin beyond the 3% band on either architecture (0 of 32 on
each; on gfx950, kernels from before #15951), so these graphs do not need strided-specific kernels
at these configurations. On gfx942 two
cells show a 2.3% to 2.5% strided-only loss inside the band (bf16 D128 causal, BHSD and packed KV);
for BHSD the best shipped kernel uses the diagonal split with one priority setting and the best
enumerated kernel another setting without the split. A literal reading of the pre-registered rule
flags one more gfx942 cell (fp16 D128 no-mask batch slice) only because its twin cell holds one
more shape; on the same shapes the excess is +0.45%.

### 5.2 The shipped catalog and the best enumerated kernel (a gap that was there before)

The comparison is best shipped kernel against best enumerated kernel per graph, in the same sweep
(time ratio, above 1 means the shipped catalog is slower).

- **gfx950:** 37 of 40 cells are above the band, on BSHD as much as on the strided layouts. BSHD
  cells: head size 64 at 1.10 to 1.16, head size 128 at 1.03 to 1.07. In all 90 of 120 graphs that
  are above their band, the best enumerated kernel differs from the best shipped one in a knob the
  shipped slice cannot express (the gfx950 manifest frees only `block_m` and `block_n`):
  `pv_sched_group_template` (66 graphs), `waves_per_eu` (55), `pv_sched_group_ds_read` (24); no
  graph differs in `block_m` / `block_n` alone. A typical GQA 32/8 D128 winner is
  `block_m 128, block_n 64, waves_per_eu 4, pv_sched_group_template 0` against the shipped
  `block_m 256, block_n 128, waves_per_eu 2`. The five-round verdict reproduced the
  one-round screen's gap on the one cell compared (bf16 D128 causal BSHD), so the gap is not the
  screen's bias.
- **gfx942:** 1 of 40 cells above the band (fp16 D128 no-mask batch slice, the cell of the previous
  section) and 11 graphs above their band; every knob is expressible in that catalog, the shipped
  set simply omits the faster kernels (for example a non-persistent kernel with a smaller LDS row
  pad for fp16 D128 no mask).

This is independent of the strides and is relevant to any regeneration of these catalogs.

## 6. Measurement lessons

- **One process, not one process per arm.** Timing each arm in its own process gave an A/A' p95 of
  2.05% to 2.82% on gfx950 (up to 3.2% on a shape), enough to hide a 1.5% gain; the same two code
  versions timed in one process, with a second load of the same code object as control, gave an A/A'
  p95 of 0.15% to 0.23% and recovered the gain. Every arm of both set-ups was numerically gated.
- **Specs from the dispatch factory only.** A hand-rolled spec once hid a large real difference.
- **The compiler and the CU count are part of the result.** Importing torch selects torch's LLVM
  for the whole process; export code objects in torch-free processes per compiler and check the
  recorded compiler label. gfx942 comes as 304-CU and 228-CU parts: compare like with like.
- **Engine-level timing is noisier than kernel-level timing.** Through the hipDNN engine and
  `dnn-benchmark`, the A/A' p95 was 0.8% to 1.0% (section 5) against 0.15% to 0.23% in the in-process
  kernel harness, so the sweep verdicts use a 3% band floor.
- **Winner's curse in sweeps.** The minimum over many noisy timings is biased fast. Screen first,
  re-measure the shortlist in independent rounds, and check numerically both the best enumerated and
  the best shipped kernel: an ordinary sweep checks only the served winner (in an earlier pilot,
  10 of 15 shipped best rows had no verdict).
- **Check the tool, not only the kernel.** `sweep.py` reads one result schema version of
  `dnn-benchmark`, so use a benchmark build that writes it, and it resolves its interpreter through
  symlinks (a venv python loses its venv; use a console script).

## 7. Caveats and open items

- **Compilers and parts.** Device speed numbers are LLVM 22 only, on a 256-CU gfx950 and a 304-CU
  gfx942. Offline checks (resources, IR rows) ran at LLVM 20, 22, 23 and 24; a device run on the
  final tree at the other three compilers and on the 228-CU gfx942 part has not been done. The
  gfx942 result follows the compiler's KV-loop schedule, and at LLVM 23 and 24 the baseline's
  schedule differs (section 4), so its sizes and signs there are unknown. This change's tree
  exports its code objects without torch at any COMGR (checked at LLVM 23). The baseline tree does
  not: its gfx942 runner needs torch at launch, so its torch-free export fails, and with torch
  imported the export is pinned to torch's LLVM 22. A cross-compiler A/B against the baseline is
  therefore not possible with the shipped harness as it is.
- **The gfx950 selection-set rows and the catalog sweep predate #15951** (section 3.3), which
  changed the timed gfx950 bodies in the base. The speed rows were re-timed on the code after it
  (section 7.1); the gfx950 half of the catalog sweep (sections 5.1 and 5.2) has not been re-run.
  #15951 changes only gfx950 code, so the gfx942 rows are not affected by it.
- **How the gate rows were measured.** The speed rows of sections 3.1 (apart from the independent
  lever re-measurement) and 3.2 come from earlier scratch versions of the in-process harness whose
  design the shipped harness inherits (they loaded each arm's kernel file as a separate module in
  one process, with a code-object identity proof where a file was cross-loaded, where the shipped
  harness exports and loads code objects). Their numeric gate has the form (b) described in
  section 2, and the completeness, export and compiler checks of section 2 did not exist in
  them. An earlier version of the shipped harness (`SCRIPT_VERSION` `dense-strided-ab-1`)
  reproduced the gfx950 lever result inside its interval. The version in this change
  (`dense-strided-ab-2`) changed in review: its export format (a shape signature and
  `workspace_bytes` per case), the segment-workspace buffers it sizes from `workspace_bytes`, the
  completeness, round-count and compiler checks, and its exit codes. That version ran on a
  device after the scratch runs, for the held-out gfx942 run and the gfx950 re-time (section
  7.1); the rows of sections 3.1 and 3.2 above it did not. The shape files of the gfx950
  persistent family and of the two gfx942 sets are not shipped (the table in section 2 and the
  table in section 7.1 are their source). The trees of the
  diagnostic arms (lever only, declared-but-ignored, and the gfx942 arms of section 3.2) are not
  in the repository.
- **gfx942 selection-set evidence is post-selection.** The destination form and the PV order were
  chosen by timing the same 30 shapes that section 3.2 reports. No sweep of the existing gfx942
  knobs (`pv_priority`, `pv_loop_order`, `pv_sched_fence_mask`, `waves_per_eu` and the others) on
  the strided build was recorded before the body was changed. The held-out run of section 7.1
  (22 shapes not used for the choice, arms, commands and pass rule written down before the run)
  tested these choices: LLVM 22 and one 304-CU part only, no sliding-window body, no non-BSHD
  layout.
- **Speed on strided layouts** is not a comparison against the baseline (the baseline cannot launch
  them); it rests on the catalog sweep against each graph's BSHD twin.
- **gfx950 persistent bodies:** the selection-set row (before #15951) was +0.27% overall, six
  shapes above the measured noise, and passed only through the band's 1% floor. The re-time after
  #15951 (section 7.1) fails the bar in three of three seeds: p09 (D64, 64/8 heads, S 2048,
  causal, bf16) is +1.5% to +1.6%, wholly above the band, with p10 near it and p12 unstable between
  runs. The failure is accepted and not explained. A speed pass like the gfx942 one would be a
  follow-up: a port of the scalar LDS destination to the non-wide persistent bodies, and, for the
  wide-DMA ones (which already have it), a look at what the stride addressing costs inside the
  work loop.
- **gfx942 residuals:** the grid and KV-segment rows fail the bar and are accepted exceptions:
  g02G (forced grid, not a configuration the dispatcher picks) and k03 (KV-segment D64) are above
  the 1% band, and the KV-segment row rests on three forced shapes. The held-out run adds two:
  q15 (a factory-chosen D64 bf16 causal grid shape, +1.2% to +1.4%) and q20K (a forced D128 bf16
  causal KV-segment shape, +1.7% to +2.1%); the 30-shape and 22-shape sets fail the same two
  families. Also accepted: the LLVM 20 two-SGPR spill of g06.
- **Bodies that ignore the strides** (section 1). A strided layout for one of them still needs a
  copy; serving them is tracked per body family:
  - AICK-3355: the additive-bias bodies, on both architectures.
  - AICK-3356: the gfx950 fp8 K/V-cache bodies.
  - AICK-3357: the ragged, varlen and paged bodies (on gfx942, paged).
  - AICK-3358: the gfx950 `gqa_pair` persistent work orders and the gfx942 K double-buffer and V
    register-prefetch pipelines.
- **Trained rankers and strided graphs.** The trained rankers (UHD) of both dense engines take the
  Q/K/V/O strides as input features, but they were trained on corpus graphs that are all packed
  BSHD (the corpus tool has no layout or stride facet), so a ranking for a graph that is not
  packed is extrapolated. The engine-level sweep of section 5 (0 of 32 strided cells per
  architecture slower than their BSHD twin) is the only check of it. The fingerprint gate of the
  rankers passes. This change adds no new instance to any catalog.
- **Catalog gap on gfx950** (section 5.2): a manifest that frees `waves_per_eu` and the PV
  scheduling knobs, or per-head-group variants, would remove it; it predates this change.
- **The sweep numbers are engine-level** (see section 6), the shortlist-then-verdict design bounds
  but does not remove the winner's curse, and the correctness gate of the sweep runs in its own phase
  of the same campaign, not inside every timing phase.

### 7.1 Held-out gfx942 validation and gfx950 re-time: results

Two sets of device runs of the shipped harness (`dense-strided-ab-2`, from this change, unedited),
packed BSHD tensors, LLVM 22 in every arm and in the timing process, 24 rounds, an A/A' control in
every run, one 304-CU gfx942 and one 256-CU gfx950. The code timed is this change at `2cb5a1aed24`. The
base merged since changed both dense kernel files only by naming the LDS-DMA widths (the capability
table, #16179); the dense IR goldens did not change and the current tree matches them at LLVM 20,
22, 23 and 24. The numeric gate passed for every arm, before and after timing, in every run, and
the output hash of every arm equals the baseline's in every case: the outputs are bit-identical to
the baseline's.

- **gfx942 held-out run.** The 22 shapes of the design below, none of them one of the 30 that chose
  the body changes. Baseline A: the merge base of this change, `05205668b34`. The pass rule
  below was fixed before the run. Arms B (this change) and C to G (one edit each).
- **gfx950 re-time.** Not held-out: the 26 selection shapes of section 3.1, re-timed after #15951.
  Baseline A: `e77263495e2` (contains #15951; its code objects equal those of the merge base on all
  26 shapes). Grid and KV-segment family: seeds 0 and 1. Persistent and wide-DMA family: seeds 0, 1
  and 2.

**gfx942 held-out, B against A.** The band is max(1%, A2/A p95) = 1.00% in every row.

| family (shapes) | seed | A2/A p95 | B/A geomean [95% interval] | shapes wholly above the band | speed bar |
|---|---|---|---|---|---|
| persistent (12) | 0 | 0.11% | -1.62% [-3.24, -0.40] | none | pass |
| persistent (12) | 1 | 0.10% | -1.61% [-3.21, -0.43] | none | pass |
| grid (8) | 0 | 0.13% | -1.55% [-4.03, +0.54] | q15 | **fail** |
| grid (8) | 1 | 0.08% | -1.61% [-4.12, +0.47] | q15 | **fail** |
| KV-segment (2) | 0 | 0.03% | +0.78% [-0.11, +1.76] | q20K | **fail** |
| KV-segment (2) | 1 | 0.18% | +0.94% [-0.19, +2.13] | q20K | **fail** |

Failing cases (95% interval wholly above the band, in both seeds):

- **q15** (D64 grid, bf16, causal, batch 2, 40/8 heads, S 3072, the factory's choice): +1.38%
  [+1.23, +1.56] and +1.23% [+1.13, +1.40]. No gfx942 lever touches the D64 code, so B, C, D and E
  run the same code object here, and the strides alone give +1.13% [+1.07, +1.26] against A: the
  cost is reading the strides in the D64 body. The selection set's k03 (a forced D64 bf16 causal
  KV-segment shape) failed the same way.
- **q20K** (KV-segment, D128, bf16, causal, 4/1 heads, S 4096, forced): +1.65% [+1.32, +1.85] and
  +2.06% [+1.78, +2.23]. The strides alone are neutral here (+0.13% [-0.12, +0.51]); the scalar base
  costs +1.40% [+1.20, +1.50] and the PV order +0.49% [+0.37, +0.61].

Near the band, not failing: q17 (+0.76%, +0.82%), q12G (+0.90%, +0.82%), q18 (+0.73%, +0.44%). The
persistent gain sits in two short-sequence block_m 128 shapes, q21 (-8.4%, -8.3%) and q08 (-5.7%,
-5.4%); the other ten persistent shapes are between -1.2% and +0.3%.

**gfx942 lever results** (each pair in its own run with its own A/A' control; "[a, b]" the 95%
interval):

| lever (pair) | rows | geomean | reading |
|---|---|---|---|
| PV issue order (B/C) | causal D128 persistent: q01, q02, q04, q06, q08, q09 | -1.77% [-3.02, -0.97] | helps |
| same | non-causal bf16 persistent: q21, q03, q07 | +1.95% [+1.67, +2.18], +0.26% [+0.19, +0.44], +0.27% [+0.03, +0.43] (per row) | costs |
| same | non-causal fp16 persistent: q05, q10, q22 | -0.16%, -0.16%, -0.04% (per row; every interval spans 0) | no effect |
| same | all persistent (12); grid (8) | -0.72% [-1.64, +0.07]; -0.04% [-0.17, +0.09] | no effect over the family |
| per-row exception (B/G) | q11G, q12G | -1.51% [-4.30, +1.39] | no measured effect; the two rows go opposite ways: q11G +1.31%, q12G -4.25% |
| scalar base (B/D) | the 13 rows where the two differ, without q14G and q20K (as registered) | -0.41% [-0.63, -0.12] | small gain |
| same | the same rows with q20K | -0.28% [-0.58, +0.08] | crosses 0 |
| same | q20K (causal KV-segment), q21 (non-causal persistent), per row | +1.40% [+1.20, +1.50]; +0.86% [+0.61, +1.43] | the opposite: a cost |
| strides alone (E/A) | causal D128 persistent (6) | +1.06% [+0.11, +1.75] | neutral to slower |
| k_major (F/B) | causal D128 persistent (6) | +3.17% [+2.39, +4.13] | slower on all six (+2.2% to +5.2%); not a default |

The short-sequence persistent gain is there with the strides alone on q21 (strides alone -8.60%,
this change -8.44%, the PV order +1.95%); on q08 the strides alone give -1.10% and the PV order
most of the rest (-4.66%). Resources, from the exported code objects: this change has no spill and
no scratch on any of the 22 kernels; the per-row form on every D128 body (arms D and E) spills on
q14G (5 VGPRs, 24 B scratch), as expected, but not on q20K (no spill, no scratch), where the
expectation of the rule below was wrong, so its D and E times are real measurements; `k_major`
spills on q15 and q18 (14 VGPRs, 60 B scratch) and is +8.2% and +8.8% slower there.

Pass rule, point by point (the rule is below):

| # | result |
|---|---|
| 1 numeric gate | pass in 7 of 7 runs; 22 of 22 cases |
| 2 shipped bar per family | persistent pass in both seeds; grid fail (q15); KV-segment fail (q20K) |
| 3 "persistent faster" | kept: -1.62% [-3.24, -0.40] and -1.61% [-3.21, -0.43], not the post-selection -3.50% |
| 4 a lever's comment may claim a gain only with an upper bound below 0 | PV order: yes on the causal D128 persistent rows (it costs on the non-causal ones); per-row exception: no, no measured effect; scalar base: yes as registered, not with q20K; `k_major`: no evidence for a default |
| 5 D and E diagnostic on q14G and q20K | q14G spills as expected; q20K does not |

**gfx950 re-time, B against A** (26 selection shapes, after #15951). The band is 1.00% in every
row.

| family (shapes) | seed | A2/A p95 | B/A geomean [95% interval] | shapes wholly above the band | speed bar |
|---|---|---|---|---|---|
| grid + KV-segment (13) | 0 | 0.05% | -1.84% [-2.70, -1.08] | none | pass |
| grid + KV-segment (13) | 1 | 0.04% | -1.80% [-2.66, -1.04] | none | pass |
| persistent + wide-DMA (13) | 0 | 0.16% | +0.17% [-0.07, +0.47] | p09 | **fail** |
| persistent + wide-DMA (13) | 1 | 0.25% | +0.20% [-0.04, +0.49] | p09 | **fail** |
| persistent + wide-DMA (13) | 2 | 0.14% | +0.25% [-0.02, +0.58] | p09 | **fail** |

The grid family is faster on every shape, from -5.18% (g06) to -0.38% (h04), a larger and more
spread-out gain than the -1.48% of section 3.1. The persistent family's geomean is about what it
was, but the bar fails on one shape:

- **p09** (D64, 64/8 heads, S 2048, causal, bf16, non-wide persistent body): +1.61% [+1.40, +1.68],
  +1.48% [+1.34, +1.62], +1.49% [+1.37, +1.55]. It is stable: +1.4% to +1.6% in 5 of 5 runs (three
  seeds, a 48-round run, and a reversed run with the head as the baseline, which gave -1.36%
  [-1.47, -1.26]). Its resources change from 61 SGPRs and 206 VGPRs (base) to 78 and 184 (this
  change), with no spill either way; p11, the same body at 32/32 heads and S 4096 in fp16, has the
  same resource change and costs nothing (-0.06%).
- **p10** (D64, 64/8 heads, S 8192, no mask, fp16): +0.86% to +0.97% in every run; its interval
  touches 1% but was never wholly above it.
- **p12** (D64, 32/32 heads, S 16384, bf16) is unstable between runs, and the seed is not the cause:
  over 9 runs it read about 0% in 4 and +1.06% to +1.74% in 5, with the same seed giving both
  outcomes and identical code objects and arguments; in three runs it was wholly above the band. The 95% interval of one run understates its run-to-run variation. Buffer placement is the
  likely cause and was not isolated.
- p02, the row that was +1.02% before #15951, is +0.16% to +0.21% now.

**These regressions are measured, not explained.** The gfx942 rows q15 and q20K and the gfx950 rows
p09, p10 and p12 were not profiled. The arm pairs say which change costs what on q15 (the strides
alone) and q20K (the scalar base and the PV order); why the stride addressing costs more on the D64
64/8-head persistent shapes of gfx950, and why p12 moves between runs, was not isolated. They are
accepted as residuals of this change, not passed.

**What the PR may say.** "Not slower than the baseline" holds on held-out or re-timed shapes for the
gfx942 persistent family and the gfx950 grid and KV-segment family only. It does not hold for the
gfx942 grid family (q15) and KV-segment family (q20K), or the gfx950 persistent family (p09). The
gfx942 "persistent faster" claim stays with the held-out interval (about -1.6%), not -3.50%. Not
covered: sliding-window bodies (the harness has no window field), compilers other than LLVM 22,
the 228-CU gfx942 part, and layouts other than BSHD (the shape file has `layouts: [bshd]` only).

**Design of the gfx942 held-out run** (fixed before the run, so the result is read against it and
not against a rule chosen after seeing it).

**Shapes.** None is one of the 30: each differs from every one of them in batch, head pair,
sequence length, or dtype with mask, and 11 of the 22 differ from each of the 30 in at least two
of these. They cover the persistent body at both block_m values with and without a mask, the
D64 grid, the per-row exception on the D128 grid (q11G, q12G), the scalar base on the D128 grid
(q13G fp16, q14G causal), and the KV-segment body at D64 and at causal bf16 D128. They do not
cover the KV-segment half of the per-row exception (a non-causal bf16 D128 KV-segment shape,
such as k02) or the sliding-window bodies. The body is the production factory's choice at 304
CUs, except the "G" rows (forced grid) and the "K" rows (forced KV-segment, the factory's grid
spec with its own KV-split rule), as in section 2.

| id | model / head slice | batch | Hq/Hkv | D | S (Sq = Skv) | mask | dtype | gfx942 body (block_m) |
|---|---|---|---|---|---|---|---|---|
| q01 | Llama-3-8B | 2 | 32/8 | 128 | 3072 | causal | bf16 | persistent (256) |
| q02 | Llama-3-70B, 2-way TP | 1 | 32/4 | 128 | 6144 | causal | fp16 | persistent (256) |
| q03 | - | 1 | 28/4 | 128 | 4096 | none | bf16 | persistent (256) |
| q04 | Llama-2-13B | 1 | 40/40 | 128 | 2048 | causal | fp16 | persistent (256) |
| q05 | Llama-3-8B | 4 | 32/8 | 128 | 1536 | none | fp16 | persistent (256) |
| q06 | Llama-3-8B, 2-way TP | 3 | 16/4 | 128 | 3072 | causal | bf16 | persistent (256) |
| q07 | - | 1 | 48/8 | 128 | 12288 | none | bf16 | persistent (256) |
| q08 | Llama-2-7B | 1 | 32/32 | 128 | 1536 | causal | bf16 | persistent (128) |
| q09 | Llama-3-70B | 1 | 64/8 | 128 | 6144 | causal | bf16 | persistent (256) |
| q10 | Llama-2-7B | 2 | 32/32 | 128 | 3072 | none | fp16 | persistent (256) |
| q21 | - | 1 | 24/8 | 128 | 1536 | none | bf16 | persistent (128) |
| q22 | - | 1 | 40/8 | 128 | 2560 | none | fp16 | persistent (256) |
| q11G | Llama-3-8B | 2 | 32/8 | 128 | 2048 | none | bf16 | forced grid (256), per-row destination |
| q12G | - | 1 | 16/2 | 128 | 4096 | none | bf16 | forced grid (128), per-row destination |
| q13G | Llama-3-8B | 2 | 32/8 | 128 | 2048 | none | fp16 | forced grid (256), scalar base |
| q14G | Llama-3-8B | 1 | 32/8 | 128 | 3072 | causal | bf16 | forced grid (256), scalar base |
| q15 | - | 2 | 40/8 | 64 | 3072 | causal | bf16 | grid (256) |
| q16 | - | 1 | 24/8 | 64 | 6144 | none | fp16 | grid (256) |
| q17 | - | 4 | 16/16 | 64 | 2048 | causal | fp16 | grid (256) |
| q18 | - | 1 | 64/4 | 64 | 4096 | causal | bf16 | grid (256) |
| q19K | - | 1 | 4/1 | 64 | 4096 | causal | fp16 | forced KV-segment (128, 4 segments) |
| q20K | - | 1 | 4/1 | 128 | 4096 | causal | bf16 | forced KV-segment (128, 4 segments) |

Write them in the schema of `library/benchmarks/common/attention_dense_strided_ab_shapes.yaml`
(`mask: causal_tl` or `none`, `select: auto`, `layouts: [bshd]`); the G rows add
`override: {persistent: false}` and the K rows `override: {persistent: false, kv_segments: rule}`.
`--list` on this tree, with the baseline and arms B to G below, gave the bodies above for every
arm and an equal spec in every arm (22 cases, no mismatch). The sliding-window bodies are left
out because neither the harness request nor its fp64 reference has a window field.

**Arms.** A is the base this change merges onto (its gfx942 IR rows equal those of
`74a573ca382`, section 3.3) and B is this change. C to G are copies of B's rocKE root, each with
one edit in `library/kernels/gfx942/attention_dense.py`:

| arm | edit | isolates |
|---|---|---|
| C | the `pv_in_issue_order` condition replaced by `False` (block order) | the PV issue order (B/C) |
| D | `LDS_ROW_VGPR = STRIDED and D == 128` (per-row destination on every strided D128 body) | the scalar base at D128 (B/D) |
| E | the edits of C and D together: at D128, the strides with neither body change | the strides alone (E/A) |
| F | `_DEFAULT_PV_LOOP_ORDER = "k_major"` | the k-major order (F/B) |
| G | `LDS_ROW_VGPR = False` (scalar base on every strided body) | the per-row exception (B/G) |

The kernel names do not change across arms, so each arm must be exported in its own process,
which the harness does.

**Commands** (from B's rocKE root, `HARN=library/benchmarks/common/attention_dense_strided_ab.py`,
`$S` the shape file, `$A`, `$B` and `$ARMS/<arm>` the rocKE roots, `$OUT` a result directory):

```
# 0. CPU, before the allocation: the factory's choice and the spec in every arm
python $HARN --list --arch gfx942 --shapes $S --arm A=$A --arm B=$B --arm C=$ARMS/C \
    --arm D=$ARMS/D --arm E=$ARMS/E --arm F=$ARMS/F --arm G=$ARMS/G
# 1. verdict, LLVM 22, two input seeds
python $HARN --arch gfx942 --shapes $S --arm A=$A --arm B=$B --baseline A --control A2 \
    --rounds 24 --seed 0 --label "gfx942 held-out seed 0" --out $OUT/ho_s0.yaml
python $HARN --arch gfx942 --shapes $S --arm A=$A --arm B=$B --baseline A --control A2 \
    --rounds 24 --seed 1 --label "gfx942 held-out seed 1" --out $OUT/ho_s1.yaml
python $HARN bar $OUT/ho_s0.yaml; python $HARN bar $OUT/ho_s1.yaml
# 2. one body change at a time, each pair in its own run with its own A/A' control
python $HARN --arch gfx942 --shapes $S --arm C=$ARMS/C --arm B=$B --baseline C --control C2 \
    --rounds 24 --out $OUT/pv_order.yaml
python $HARN --arch gfx942 --shapes $S --arm G=$ARMS/G --arm B=$B --baseline G --control G2 \
    --rounds 24 --only q11G,q12G,q13G,q14G,q20K --out $OUT/row_vgpr.yaml
python $HARN --arch gfx942 --shapes $S --arm D=$ARMS/D --arm B=$B --baseline D --control D2 \
    --rounds 24 --out $OUT/scalar_base.yaml
python $HARN --arch gfx942 --shapes $S --arm A=$A --arm E=$ARMS/E --baseline A --control A2 \
    --rounds 24 --out $OUT/strides_alone.yaml
python $HARN --arch gfx942 --shapes $S --arm B=$B --arm F=$ARMS/F --baseline B --control B2 \
    --rounds 24 --out $OUT/k_major.yaml
python $HARN bar <each result>
```

**Pass rule**, fixed before the run:

1. The numeric gate passes for every arm, before and after timing, in every run; otherwise stop.
2. Run 1, both seeds, per stratum (persistent, grid, KV-segment; the `bar` default
   `--group-by body-layout`): the shipped bar, that is a geomean upper 95% bound below +1% and no
   shape whose own 95% interval lies wholly above max(1%, A2/A p95). A pass lets this study and
   the PR say "not slower than the baseline on held-out shapes"; a fail lists the failing shapes,
   and the claims stay limited to the 30 shapes of section 3.2.
3. The headline "the persistent body is faster" stays only if the persistent held-out upper
   bound is below 0 in both seeds, and then the held-out interval is quoted, not the -3.50%.
4. A body change's comment may claim a gain only where its pair ratio from run 2 has an upper
   bound below 0: B/C for the PV issue order on the causal D128 persistent rows (q01, q02, q04,
   q06, q08, q09), B/G for the per-row exception on q11G and q12G, B/D for the scalar base. An
   interval across 0 means "no measured effect on held-out shapes", and the comment says so. An
   F/B upper bound below 0 on the causal D128 rows is evidence for a `k_major` default; switching
   it is a separate change (it keeps every kernel name while changing the code, so it needs a
   golden re-bless and a check of every name-keyed cache).
5. D and E are expected to spill to scratch on q14G and q20K (the per-row form on causal bf16
   D128), so their times there are diagnostic only. (Result: q14G spilled, q20K did not, so the
   q20K times were taken as measurements.)
