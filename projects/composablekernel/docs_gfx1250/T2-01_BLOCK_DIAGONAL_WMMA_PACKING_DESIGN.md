# T2-01: Block-Diagonal WMMA Packing for Small-Channels-per-Group Grouped Conv — Design Spec

Status: **implemented** (2026-09-17). `GroupsPerWmma` template parameter added to
`TransformConvFwdToGemm` and `DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3`,
scoped exactly as below (`grouped_conv2d_fwd`, `NDimSpatial=2`, no multi-A/B/D, no
NGCHW transpose, no CTranspose, `ConvolutionForwardSpecialization::Default`).
Validated end-to-end on gfx1250 against the CPU reference at the roadmap's target
shape (`G=32,N=64,C=128,K=128,Y=X=3,Hi=Wi=28`, `GroupsPerWmma=4`); see
`example/70_grouped_conv2d_fwd_wmma_block_diagonal/`. Tuned instance parameters
(vector-transfer widths, `KPerBlock`) measured 0.047 ms / 9.86 TFlops / 548 GB/s at
that shape vs. 4.56 ms / 0.10 TFlops / 5.6 GB/s for the initial untuned parameters
— ~97x faster, same CPU-reference-verified correctness; see comments in that
example for the reasoning (A/E vector widths are capped by their im2col-merge
innermost sub-dim, `C_per_group`/`K_per_group`; B is uncapped since it reads the
GroupsPerWmma prepass's flat packed buffer). See
`GFX1250_CONV_OPTIMIZATION_ROADMAP.md` T2-01 for the original roadmap context.

Scope of this design: `grouped_conv2d_fwd` only, `C_per_group =
K_per_group = 4` only (`GPW = 16/4 = 4` groups packed per WMMA
instruction). Explicit non-goals, left for follow-up: bwd_data, bwd_weight,
`C_per_group ∈ {8, 16}`, and hipConv's second packing axis (pairing two
adjacent conv taps into one K=32 WMMA operand).

## 1. Mechanism (from hipConv, transposed to CK's A/B convention)

Read directly from `/home/sgundabo/hipconv/hipconv/src/arch/cdna5/grouped/grouped_multi_g/kernel.hpp`
(the `build_fprop_weights` path, lines 376-433; full citations in §5).

**Critical convention difference**: hipConv's WMMA A operand is the
**weight** (M = output channel, K = reduction), B is the **input**. CK's
grouped-conv-as-GEMM convention is the opposite: A = input/activation
(M = N·Ho·Wo, K = C·Y·X), B = weight (K = C·Y·X, N = output channel). Any
CK implementation must therefore make **B (weight)** block-diagonal, not A
— the mechanism transposes cleanly, but do not copy hipConv's "A operand"
framing verbatim into CK code/comments.

For `G_c = 4` (channels/filters per group), `GPW = 16/4 = 4`:

- One 16×16×32 WMMA instruction packs `GPW = 4` groups' worth of output
  channels into its 16 M-rows (rows `[0,4)` = group 0's 4 output channels,
  `[4,8)` = group 1's, etc.) and the same 4 groups' input channels into its
  32 K-columns, **block-diagonally**: K-columns `[0,4)` (tap `rs_lo`) and
  `[16,20)` (tap `rs_hi`) belong to group 0, `[4,8)`/`[20,24)` to group 1,
  etc. K=32 additionally packs **two conv taps** (rs_lo in the low 16
  K-columns, rs_hi in the high 16) — this is hipConv's *second* packing
  axis, explicitly out of scope here (see non-goals above); a first
  implementation should pack only groups, leaving K at the native
  single-tap width and taking the K-padding hit for now.
- The WMMA **N** dimension (16 output-spatial columns) is **shared/
  broadcast identically across all GPW packed groups** — i.e. one WMMA
  call computes GPW groups' outputs for the *same* batch/output-pixel
  neighborhood, not GPW different pixels.
- Off-diagonal (cross-group) products are zeroed by **never writing
  non-diagonal elements into a zero-initialized register tile** (a
  `nz` boolean predicate gates the load: `(row/G == (col%16)/G) &&
  (g_base+row/G < groups)`), not by an explicit mask/store-zero step.
  Only the **weight** (B in CK's convention) operand needs this treatment;
  the input (A in CK's convention) operand stays dense/unmasked — each
  lane already only ever carries its own group's channel data on that
  side.

## 2. CK's existing extension seam (proven in-tree precedent)

CK's WMMA v3 grouped-conv-fwd compute loop
(`gridwise_gemm_wmma_cshuffle_v3.hpp`'s `Run()`, the WMMA invocation code,
and the CShuffle/DirectStore epilogues in
`gridwise_gemm_wmma_cshuffle_v3_common.hpp`) is **entirely
descriptor-driven**: it only calls
`GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(a_grid_desc_m_k)` /
`MakeBGridDescriptor_BK0_N_BK1(b_grid_desc_n_k)` — both take a plain 2-D
`[M,K]`/`[N,K]` descriptor and never inspect `G`, `C_per_group`, or
`K_per_group` directly
(`gridwise_gemm_wmma_cshuffle_v3_common.hpp:445-456`, `:513-524`).

CK already has a **proven, in-production instance of exactly this kind of
descriptor-only group-packing**: the `NumGroupsToMerge > 1` depthwise path
(`device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp:232`,
`transform_conv_fwd_to_gemm.hpp:20`), valid only for `C_per_group == 1`.
It proves the seam end-to-end:

1. A device-op/`TransformConvFwdToGemm` template parameter
   (`NumGroupsToMerge`) selects an alternate `MakeADescriptor_M_K`/
   `MakeBDescriptor_N_K` overload.
2. The grid launch's Y-dimension divides by the merge factor:
   `gdy = arg.num_group_ / NumGroupsToMerge`
   (`device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp:1085`) so
   one workgroup covers multiple groups.
3. `ComputePtrOffset`'s per-group stride scales by the merge factor
   (`:729-734`).
4. `gridwise_gemm_wmma_cshuffle_v3.hpp`'s `Run()`, the WMMA calls, and the
   epilogue are **100% unmodified** — only the grid descriptors and
   grid-index arithmetic change.

`NumGroupsToMerge`'s packing lives in **M/N** (it concatenates whole
groups' output rows/columns) and needs an xor-permutation trick on the
**output** descriptor
(`transform_conv_fwd_to_gemm.hpp:1502-1534,1557-1594,1616-1653`) to keep
merged groups' outputs from cross-contaminating, because concatenating
along M/N with a shared K-reduction would otherwise let group A's rows dot
group B's K-columns.

T2-01's packing is structurally different: it lives in **K** (the
reduction axis), block-diagonally, which is *why* it needs the
zero-fill/masking mechanism at all (§1) — but for exactly the same reason,
it needs **no xor/pad trick on the output descriptor**: each workgroup
still produces one ordinary `[MPerBlock, NPerBlock]` output tile per
(group-cluster, N-block) grid cell, and that tile is already correctly
attributed to one packed-group-cluster via the existing per-group E grid
descriptor / `blockIdx.y`-driven pointer offset (unaffected, no change
needed there at all).

## 3. Concrete component list for a future implementation

1. **New template parameter** `GroupsPerWmma` (GPW), added next to
   `NumGroupsToMerge` in `TransformConvFwdToGemm<...>`
   (`transform_conv_fwd_to_gemm.hpp:15-22`) and in
   `DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<...>`'s own template
   list (`device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp:213-232`,
   next to `NumGroupsToMerge` at `:232`), defaulted to `1` so every
   existing instance is unaffected.
2. **New `MakeADescriptor_M_K` / `MakeBDescriptor_N_K` overloads** in
   `transform_conv_fwd_to_gemm.hpp` (alongside the existing SFINAE-dispatched
   overloads at `:510,723,964,1273`) for `GPW > 1`. The transform chain
   packs `{Y_,X_,C_}` per-group as today (`:906-909`/`:950-953` pattern)
   plus an extra `GPW` axis **folded into K** (not M/N, unlike
   `NumGroupsToMerge`) via `make_merge_transform({GPW, Y_*X_*C_})` — a
   genuinely new transform-chain shape, not reachable by tweaking
   `NumGroupsToMerge`. `matrix_padder.PadADescriptor_M_K`/
   `PadBDescriptor_N_K` are reused unchanged: after packing, effective K
   = `GPW*C_per_group*Y*X` (144 for GPW=4,C=4,Y=X=3) pads to a
   `KPerBlock` multiple exactly as today.
3. **New grid-index/pointer-stride arithmetic** in the device op, mirroring
   `NumGroupsToMerge`'s existing code line-for-line: `gdy = num_group_ /
   GPW`, `BatchStrideA_`/`BatchStrideB_` = `stride[0] * GPW`.
4. **A new zero-fill/packing prepass kernel** for the weight (B) operand
   only (the input/A operand needs no prepass — dense/unmasked per §1).
   The most CK-idiomatic home for this is the existing
   `gridwise_elementwise_2d.hpp`-based prepass mechanism already used in
   this same device-op file for NGCHW→NHWGC transposition (same file,
   include line ~26) — reuse that pattern rather than inventing a new
   prepass framework. The prepass writes a `[G/GPW, GPW*C_per_group*Y*X,
   K_total]`-shaped zero-initialized-then-block-diagonally-filled B
   operand to a scratch buffer that the (otherwise unmodified) gridwise
   gemm then reads as its ordinary B input.
5. **A new specialization-selection axis** (a boolean template parameter
   or a small dedicated enum living beside `GemmSpecialization` in
   `gemm_specialization.hpp`) so the device op knows to select the new
   descriptor overloads and prepass step only when `GPW > 1`.

**No changes required** to `gridwise_gemm_wmma_cshuffle_v3.hpp`'s `Run()`,
the WMMA invocation code, or the CShuffle/DirectStore epilogues — the
entire change is expressible as (new template param) + (new descriptor
overloads) + (new grid-index arithmetic) + (new B-operand prepass kernel).

## 4. Validation plan (for the future implementation)

- Target shape: `grouped_conv2d_fwd`, `G=32, N=64, C=128 (C_per_group=4),
  K=128 (K_per_group=4), Y=X=3, Hi=Wi=28` — the roadmap's own
  small-channels-per-group representative shape.
- Correctness: `ckProfiler grouped_conv_fwd -v 1` against the CPU
  reference, comparing the new `GPW=4` instance against CK's current
  padded-to-full-tile handling of the same shape.
- Perf: median-of-5 TFLOP/s per the roadmap's benchmarking methodology,
  compared against the current best instance for this shape (currently a
  padded, mostly-wasted-K instance per §2/`k_padding_mechanism` finding:
  today's only option for `K_per_group=4` is zero-padding raw K up to
  `KPerBlock`, wasting most of the WMMA's 32-wide K on zeros — this is the
  gap T2-01 closes).

## 5. Source citations

- hipConv mechanism: `/home/sgundabo/hipconv/hipconv/src/arch/cdna5/grouped/grouped_multi_g/kernel.hpp`
  — header overview lines 1-53; type defs 112-138; G/GPW consts 141-149;
  lane layout 260-288; `build_fprop_weights` (in-scope fprop A/weight-operand
  construction) 376-433, `nz` predicate 415-417, `offset_map` 406-414;
  B-operand (input, non-block-diagonal) comment 730-736; output store split
  799-853; WMMA instruction selection `bunnies_mi400.hpp:392-450`
  (`__builtin_amdgcn_wmma_f32_16x16x32_{f16,bf16}`); config `group_size`
  `grouped_multi_g/config.hpp:15-19`, `static_assert(G==4||G==8||G==16)`
  `kernel.hpp:142`. G=32 is a separate, non-block-diagonal kernel
  (`kernel.hpp:1171` onward) — not relevant to this design.
- CK seam: `device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp:327-345`
  (`MakeAGridDescriptor_M_K` entry point), `:213-232` (template param list,
  `NumGroupsToMerge` at `:232`), `:729-734` (`ComputePtrOffset` batch
  stride), `:1085-1086` (grid launch `gdy`/`gdz`), `:~1676-1687`
  (`NumGroupsToMerge>1` validity check, `C==1` requirement).
  `transform_conv_fwd_to_gemm.hpp:15-22` (template param list),
  `:721-864` (2D general `MakeADescriptor_M_K` SFINAE overload),
  `:879-909` (im2col transform chain: raw view → H/W pad → embed
  (im2col) → merge into `[M,K]`), `:950-953` (`NumGroupsToMerge` merge
  into M), `:1368-1394` (`NumGroupsToMerge` merge into B's N),
  `:1502-1534,1557-1594,1616-1653` (output xor/pad block-diagonal trick
  for `NumGroupsToMerge`, NOT needed for T2-01's K-axis packing).
  `gridwise_gemm_wmma_cshuffle_v3_common.hpp:445-456,513-524`
  (`MakeAGridDescriptor_AK0_M_AK1`/B counterpart, the descriptor-only
  compute-loop seam), `:429-436` (`GemmSpecialization`-driven `padK`),
  `:131-134` (`KPack` fixed by native WMMA K-tile, not by `C_per_group`).
  `gemm_specialization.hpp:11-27` (`GemmSpecialization` enum).
  `gridwise_ab_transfer_wave_tiles.hpp:91-115,124-199`
  (`MakeGridDescriptor<PadMN,PadK>`, where raw K actually gets rounded to
  a `KPack`/native-WMMA-K multiple).
