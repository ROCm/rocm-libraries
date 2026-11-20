# Dense attention prefill with launch-time Q/K/V/O strides (gfx942 and gfx950): case study

Measured scope: the rocKE dense attention prefill kernels (`library/kernels/gfx942/attention_dense.py`
and `library/kernels/gfx950/attention_dense.py`) before and after they take the batch, head and
token element stride of Q, K, V and O as twelve trailing launch-time kernel arguments. The "before"
is the same kernels on `develop` at the commits named per row (`74a573ca382` for gfx942,
`84c6e7ced46` for gfx950); the "after" is this change. The study has three parts: (1) the cost of
reading the strides, measured per kernel body on packed tensors against `develop` (bf16 and fp16,
head size 64 and 128, causal and not, MHA and GQA, sequence length 1K to 16K, LLVM 22, a 256-CU
gfx950 and a 304-CU gfx942); (2) a strided-layout sweep of the hipDNN kernel catalog on both
architectures (BSHD, BHSD, packed QKV, packed KV, batch slice); (3) measurement lessons. Every ratio below is a time ratio between two code paths measured in the same
process or sweep on the same device; absolute times are not recorded here (see
[`platform/AGENTS.md`](../../../../../platform/AGENTS.md), "Public software-performance evidence").
Every compared arm of every measured run passed an on-silicon numeric gate, stated per table.
Statements are bounded to the measured configurations. This file lives in the gfx950 prefill
folder because that is where the scalar LDS-DMA destination change and most of the bodies are; the
gfx942 prefill README links here.

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
  read the strides take it; the packed-only grid bodies (ragged, varlen, paged) keep `develop`'s
  per-row form. Of the gfx950 persistent bodies, the wide-DMA ones already formed this destination
  in a scalar register on `develop`, and the others keep the per-row form (not ported). On gfx942
  the destination form is chosen per body, and the strided PV MFMAs are issued in softmax order
  (section 4).

Which bodies read the strides, and which keep packed addressing (they still declare the twelve
arguments, so the kernel ABI is one; the host admits only packed tensors for them):

| | read the strides | packed-only |
|---|---|---|
| gfx950 | grid, its KV-segment form, persistent in every work order that decodes through the shape arguments, wide-DMA persistent | `gqa_pair` work orders, ragged, varlen, paged, fp8 K/V-cache bodies |
| gfx942 | default grid, persistent grid, KV-segment grid | the opt-in K LDS double-buffer and V register-prefetch pipelines, compile-time-shape specs (paged) |

A KV-segment launch with a strided O is refused (the combine step needs a packed O). `develop`
launches packed tensors only, so the strided layouts have no `develop` arm: their effect is measured
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
  against an fp64 reference on a row subset (up to three (batch, head) pairs
  and three windows of 64 query rows) at max-abs error <= 2e-2 for fp16 and <= 4e-2 for bf16, with no NaN, and hashes the
  output; (b) the earlier scratch versions of the same in-process harness that measured the gate
  rows of part (1) checked the `develop` arm against that fp64 reference and every other arm
  against the `develop` output; in every row of those runs the new output was bit-identical to
  `develop`'s, so it has the same error against the reference. A case that fails the gate never
  enters a geomean.

The commands below run from the rocKE root with `platform/python` and `library` on `PYTHONPATH`
(`<develop tree>` and `<this change>` are checkouts):

```
python library/benchmarks/common/attention_dense_strided_ab.py --list --arch gfx950 \
    --shapes library/benchmarks/common/attention_dense_strided_ab_shapes.yaml \
    --arm A=<develop tree>/dnn-providers/hip-kernel-provider/rocke \
    --arm B=<this change>/dnn-providers/hip-kernel-provider/rocke       # CPU: factory choice per arm
python library/benchmarks/common/attention_dense_strided_ab.py --arch gfx950 \
    --shapes library/benchmarks/common/attention_dense_strided_ab_shapes.yaml \
    --arm A=<develop tree>/dnn-providers/hip-kernel-provider/rocke \
    --arm B=<this change>/dnn-providers/hip-kernel-provider/rocke \
    --baseline A --control A2 --rounds 24 --seed 0 --out RESULT.yaml
python library/benchmarks/common/attention_dense_strided_ab.py bar RESULT.yaml
```

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
that a body that can be launched keeps its speed.

## 3. Results against `develop`

All rows: packed contiguous BSHD tensors (the cost of reading the strides, not of a strided
layout), ratios `new / develop`, geomean over the shapes of the row, one device and one compiler per
table. The numeric gate passed for every arm in every run of these tables.

### 3.1 gfx950 (256 CU, LLVM 22), baseline `develop` `84c6e7ced46`

Shapes: the 13 grid-family shapes (g01-g06 grid, k01-k03 KV-segment, h01-h04 grid pinned) and 13
persistent shapes (p01-p13; p02, p04, p06, p08, p13 take the wide-DMA body). A/A' p95 per table:
0.18% (grid family), 0.16% (persistent). Numeric gate: form (b): every arm bit-identical to
`develop`'s output, `develop` within tolerance of the fp64 reference, before and after timing, on
all 26 shapes.

| Bodies | Arm | vs develop | notes |
|---|---|---|---|
| grid, KV-segment, pinned grid (13 shapes) | this change (strides + lever) | **-1.48% [-1.79, -1.19]** | repeat with other inputs -1.56% [-1.81, -1.32]; per shape from -2.55% (g03) to -0.43% (h02); no shape slower |
| same | lever only (diagnostic: arguments declared and ignored, lever on) | -1.78% [-2.23, -1.35] | same run |
| same | reading the strides, relative to the lever-only arm | +0.30% [+0.04, +0.55] | per shape from -0.57% (h01) to +0.99% (g06) |
| persistent + wide-DMA (13 shapes) | this change (strides, no lever) | **+0.27% [+0.02, +0.53]** | repeat +0.28% [+0.04, +0.52]; persistent +0.20%, wide-DMA +0.39% |
| same | reading the strides, relative to the declared-but-ignored arm | +0.28% [+0.04, +0.52] | |
| same | declared-but-ignored arguments only (diagnostic) | +0.01% [-0.04, +0.06] | no measurable cost |
| control | harness with no change (a third load of develop's code) | -0.01% [-0.06, +0.06] (grid), -0.00% [-0.04, +0.04] (persistent) | the harness reads zero where there is no change |

Pass rule: both bodies pass (upper bounds -1.19% and +0.53% against +1%, no shape wholly above the
1% band). Two things to know about the persistent row. Its band is the 1% floor, not the measured
noise: against the measured A/A' noise alone (0.16% p95), six of the 13 shapes (p02, p06, p09, p10,
p12, p13) have intervals wholly above it. The largest is p02 (Llama-3-8B, causal, S 8192, bf16,
wide-DMA) at +1.02% [+0.90, +1.06], whose median sits just above the 1% band while its interval
straddles it; p10 is +0.96% [+0.87, +1.08]. Second, nothing offsets the stride cost here as the
scalar LDS-DMA destination does on the grid: the non-wide persistent bodies keep the per-row
destination (the scalar form is not ported to them), and the wide-DMA ones, p02 among them, already
had a scalar destination on `develop`, so for them the residual is the stride addressing itself,
whose cause within the body has not been isolated.

An independent re-measurement of the lever alone with the shipped harness (`develop` `74a573ca382`
against a build with the lever and the twelve declared-but-ignored arguments, A/A' p95 0.15%):
-1.86% [-2.30, -1.44], 13 of 13 shapes faster, numeric gate of form (a) on every arm; two more seeds
gave -1.76% [-2.18, -1.37] and -1.86% [-2.30, -1.47]. (`74a573ca382` and `84c6e7ced46` differ in two
gfx950 IR rows, both varlen, which are not timed.) Run in a separate process per arm, the same pair
gave A/A' p95 of 2.05% to 2.82% (section 6).

### 3.2 gfx942 (304 CU, LLVM 22), baseline `develop` `74a573ca382`

30 shapes in three groups (table above): grid 12 (8 factory-served D64 shapes and the 4 forced "G"
D128 rows), KV-segment 3 (forced), persistent 15. Numeric gate: form (b), 30 of 30 shapes, every
arm, before and after timing: the new output bit-identical to `develop`'s, `develop` within
tolerance of the fp64 reference.

| Body | shapes | A/A' p95 | new vs develop | repeat | shapes above the 1% band |
|---|---|---|---|---|---|
| grid | 12 | 0.16% | **-0.14% [-0.73, +0.42]** | -0.16% [-0.77, +0.41] | g02G +1.76% [+1.54, +1.95] |
| KV-segment | 3 | 0.22% | **+0.09% [-0.92, +1.20]** | +0.14% [-0.71, +0.97] | k03 +1.21% [+1.16, +1.23] (+0.97% in the repeat) |
| persistent | 15 | 0.15% | **-3.50% [-5.56, -1.49]** | -3.53% [-5.62, -1.53] | none |

Under the pass rule of section 2, the persistent row passes and the grid and KV-segment rows fail.
The grid geomean's upper bound is below +1%, but one shape is wholly above the 1% band: g02G
(forced grid, bf16 D128 non-causal MHA, S 1024, a non-production configuration: +1.76% in the run
above, +1.54% [+1.29, +1.73] in the repeat). The KV-segment row fails both conditions: its upper
bound is +1.20%, and k03 (bf16 D64 causal MQA 8/1, S 4096) is +1.21% [+1.16, +1.23], wholly
above the band; with three shapes, that bound is essentially k03's own. In the repeat, k03 is
+0.97% [+0.92, +1.05], not wholly above the band. Both shapes are accepted as residuals of this
change, not passed (section 7). The persistent gain is concentrated at short sequence: the four
1K-2K shapes (g01, g02, g05, p01) are faster by about 8% to 10%, the
long-sequence shapes (S >= 4096) are within about +-1%, and the shapes with a median above zero are
p08 (+0.61%), p13 (+0.53%) and h01 (+0.19%), all inside the band.

### 3.3 The code that was timed is the code on the branch

The gfx942 rows were measured on the final kernel bytes of the gfx942 body. Later `develop` merges
and the remaining work on this branch did not move the kernels measured here: the representative
IR rows of the dense attention goldens, hashed at LLVM 20, 22, 23 and 24, are identical between the
measured tree and the current one for 38 of the 47 gfx950 rows that existed at measurement and
every gfx942 row (8 of 8 in the shared shard, 51 of 51 in the library golden), the only new gfx950
rows being the 8 fp8 K/V-cache rows (packed-only). The other 9 gfx950 rows (ragged, varlen, paged)
are packed-only bodies that were never timed: the scalar LDS-DMA destination was later limited to
the bodies that read the strides, so these are back at `develop`'s form (their IR equals
`develop`'s once the twelve unused stride parameters are removed). The baseline side is unchanged too: `develop`
`74a573ca382` to `c3448b0c5f5` moved none of the 8 gfx942 shard rows nor the 51 gfx942 library rows,
and `84c6e7ced46` to `c3448b0c5f5` moved none of the 47 gfx950 rows (same 8 new fp8 rows). These are
representative rows, not every shape timed above. For arms taken from another tree, the harness
runs of part (1) checked that the IR, the code object and the `.text` section of each cross-loaded
arm were identical to what its own tree builds, on all 26 gfx950 shapes.

## 4. What the kernels showed

- **The lever pays for the strides on gfx950.** Moving the LDS-DMA destination to a scalar register
  is worth about 1.8%; the per-slab 64-bit base, per-slab K/V buffer resources and the stride
  arithmetic cost about 0.3% (the +0.30% row). Net, the grid family is faster than `develop` on the
  13 measured shapes. The non-wide persistent bodies have the cost and not yet the gain (the scalar
  destination is not ported to them); the wide-DMA persistent bodies already had it on `develop`,
  so their residual is the stride addressing itself, with no isolated cause yet.
- **Keep the lane / uniform split.** What does not depend on the work item is hoisted out of the
  persistent work loop; the row term belongs in the per-lane VGPR offset, not in a scalar multiply
  per load (that form spilled).
- **gfx942 D128 speed follows the compiler's schedule of the KV loop.** In `develop`'s causal
  persistent D128 kernel LLVM overlaps the softmax exps with the PV MFMAs. Any change of the address
  code (the strides alone, the lever alone, and both) lost that overlap and made those kernels
  slower, unless each PV MFMA is emitted right after its P pack, which restored it: with that, the
  persistent result moved from a smaller gain with several shapes above the band to the -3.5% above.
  The destination form is a per-body choice on gfx942: the non-causal bf16 D128 grid keeps
  `develop`'s per-row form.
- **Offline resource checks** at LLVM 20, 22, 23 and 24 (no scratch, no fewer waves per SIMD than
  `develop`, no extra SGPR spill) passed for every served body, with one exception: one gfx942 grid
  kernel at LLVM 20 spills two SGPRs into VGPR lanes (no scratch). The form that avoids it costs speed
  on another measured shape, so it is kept.
- **LDS-DMA zero fill.** That an out-of-range lane of an LDS-DMA copy writes zeros was checked on
  gfx950 with a guard-page microtest; it was not checked on gfx942.

## 5. Strided-layout catalog sweep (hipDNN kernel catalog, both architectures)

Question: does a strided layout need its own kernels, or does the catalog's best kernel for a graph
change with the layout? And does the shipped catalog contain the best kernel at all?

Set-up. 120 graphs per architecture: 8 families (bf16 and fp16, D64 and D128, causal and no mask) x
3 LLM-prefill shapes (GQA 32/8, batch 2, S 2048; MHA 32/32, batch 1, S 4096; a 16/2 head slice of a
GQA model, batch 2, S 1024) x 5 layouts (BSHD, BHSD, packed QKV, packed KV, batch slice). A "cell" is
a family on one layout (40 cells; 32 strided). The catalog is `develop`'s shipped dense attention
catalog at `c3448b0c5f5` (this change does not alter which kernels ship), built by the tree
measured so that its kernels take the strides; the enumeration arm is the same manifest with the
full knob space, cut to the head-count pairs the graphs use (gfx950: 3912 kernels, grid bodies only,
as no persistent kernel ships in that catalog; gfx942: 256 kernels). Every graph is served by the
hipDNN kernel-ingestor engine and timed through `dnn-benchmark` by `sweep.py`.

Phases of one campaign per architecture, in one allocation. gfx950: a one-round screen of the whole
enumeration (timing only; used only to pick a shortlist, never for a verdict, since the minimum of
many noisy timings is biased fast); gfx942: no screen, the 256-kernel enumeration goes straight to
the verdict stage. Verdict stage: five rounds on the shortlist plus every shipped kernel, two copies
of one install tree as the A/A' control, with the `sweep.py` correctness phase enabled. Then
residual single-kernel sweeps for the graph and kernel pairs that no correctness phase had covered. **Numeric gate:** the sweep times and checks in separate phases of the
same campaign (`dnn-benchmark` correctness against a PyTorch reference, rtol 0.03, atol 0.001);
timing phases do not check output. After the residual sweeps the best shipped kernel and the best
enumerated kernel of every graph were checked: 120 of 120 and 120 of 120 passed on each
architecture (before the residual sweeps: 115 and 116 of 120 on gfx950, 110 and 114 of 120 on
gfx942). The numbers below are cell geomeans over the graphs of the cell, against a pre-registered
band of 3% (the floor; raised per cell only when the round-to-round drift is larger). A/A' p95 was
0.99% on gfx950 and 0.82% on gfx942.

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
each), so these graphs do not need strided-specific kernels at these configurations. On gfx942 two
cells show a 2.3% to 2.5% strided-only loss inside the band (bf16 D128 causal, BHSD and packed KV);
for BHSD the best shipped kernel uses the diagonal split with one priority setting and the best
enumerated kernel another setting without the split. A literal reading of the pre-registered rule flags one more
gfx942 cell (fp16 D128 no-mask batch slice) only because its twin cell holds one more shape; on the
same shapes the excess is +0.45%.

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
  gfx942. Offline checks (resources, IR rows) ran at LLVM 20, 22, 23 and 24; a device run on the final
  tree at all four compilers and on the 228-CU gfx942 part has not been done.
- **How the gate rows were measured.** The speed rows of sections 3.1 (apart from the independent
  lever re-measurement) and 3.2 come from earlier scratch versions of the in-process harness whose
  design the shipped harness inherits (they loaded each arm's kernel file as a separate module in
  one process, with a code-object identity proof where a file was cross-loaded, where the shipped
  harness exports and loads code objects); the shipped harness reproduced the gfx950 lever result
  inside its interval. It has not yet been re-run on gfx942 or on the gfx950 persistent family, and the
  shape files of those two sets are not shipped (the table in section 2 is their source). Their
  numeric gate has the form (b) described in section 2.
- **Speed on strided layouts** is not a rocKE-versus-`develop` comparison (`develop` cannot launch
  them); it rests on the catalog sweep against each graph's BSHD twin.
- **gfx950 persistent bodies:** +0.27% overall, six shapes above the measured noise, p02 +1.02%;
  accepted (inside the 1% band). A speed pass like the gfx942 one would be a follow-up: a port of the scalar LDS destination to the non-wide
  persistent bodies, and, for the wide-DMA ones (which already have it), a look at what the stride
  addressing costs inside the work loop.
- **gfx942 residuals:** g02G (forced grid, not a configuration the dispatcher picks) and k03
  (KV-segment D64) above the 1% band, accepted; the LLVM 20 two-SGPR spill.
- **Catalog gap on gfx950** (section 5.2): a manifest that frees `waves_per_eu` and the PV
  scheduling knobs, or per-head-group variants, would remove it; it predates this change.
- **The sweep numbers are engine-level** (see section 6), the shortlist-then-verdict design bounds
  but does not remove the winner's curse, and the correctness gate of the sweep runs in its own phase
  of the same campaign, not inside every timing phase.
