# Shape-generic AOT instances: hipkernel:Gfx950ConvFwdAot

rocKE's AOT convolution kernels take the problem as kernel arguments. One compiled
binary therefore serves every problem its build-time capabilities admit.
`hipkernel:Gfx950ConvFwdAot` packages such binaries as **instances** and matches them
to a graph by capability, not by exact shape. It ships next to the exact-shape engine
`hipkernel:Gfx950ConvFwd`, which it does not change.

## What an instance is

An instance is one rocKE `KernelIdentity`
(`rocke/library/kernels/conv/kernel_identity.py`): the compile-time tuple of a
rocKE sweep. It holds configuration and capability fields and no problem extents.
Two families are packaged, both gfx950, forward, fp16 or bf16:

- implicit GEMM, from the AOT sweep (`kernels/conv/aot_jobs.py`);
- direct, from the direct sweep (`kernels/conv/direct_aot_jobs.py`), every forward
  algorithm of it the packaged launch runs: see [Direct instances](#direct-instances).

The adapter builds each instance from the job rocKE's AOT build builds its identity
from (`job_for_identity`, `direct_job_for_identity`), serves the problems rocKE's
grouped-convolution dispatcher admits for the identity (`dispatch/grouped_convolution.py`),
and launches it with the dispatcher's launch plan (`launch_plan_for`).

An implicit-GEMM instance has:

| Field | Meaning |
|---|---|
| `tile_m`, `tile_n`, `tile_k`, `warp_m`, `warp_n`, `warp_tile_m/n/k`, `wave_size` | Tile and warp geometry. Wave64 only. |
| `pipeline`, `epilogue` | `mem`, `compv3`, `compv4`, `basic`; `default` or `cshuffle`. |
| `vector_size_a/b/c` | Baked load/store widths. A and B run along C/groups, D along K/groups. |
| `unroll_k`, `async_dma`, `async_chunk_a/b` | K-loop driver; an async binary's chunk widths are a capability. |
| `grouped` | Built with the grouped code path, which also serves groups == 1. The sweep builds only grouped binaries. |
| `group_merge` | Groups one workgroup owns: > 1 for a group-merged depthwise kernel (KMD default 1). |
| `instance_id` | `KernelIdentity.stable_hash()` of the source identity. This is the tracking key shared with rocKE's dispatcher. |

A group-merged instance (`group_merge` > 1, `aot_jobs.CACHE_FWD_GROUP_MERGES`) is
depthwise only. Its builder derives the vector widths from the merged problem, and the
identity records them as 0. It serves a depthwise graph whose group count it divides,
with the merged GEMM N, K/groups·`group_merge`, within one `tile_n`
(`identity_supports_problem`), and launches over the merged grid: (⌈K/groups·`group_merge`/`tile_n`⌉,
M-tiles, groups/`group_merge`), `implicit_gemm_conv_grid`, with the true problem's
kernel arguments (`conv::mergedInstanceLaunch`).

Every other `KernelIdentity` field must hold its default, because it selects code this
launch does not model. Examples are split-K, `chiplet_swizzle`, 3-D and the LDS layout
knobs. `gfx950_conv_fwd_aot_spec_for_identity` in the adapter names any such field and
refuses the identity.

Each instance is built the way its sweep builds it: `Gfx950ConvFwdAotSpec` in
`rocke/library/builders/common/convolution_forward.py` uses the sweep's probe problem.
Only the kernel symbol differs. `library/tests/test_convolution_forward_aot_packaging.py`
holds the adapter to that, and to the sweep's served-problem predicate.

## Direct instances

The direct sweep has seven forward algorithms. Each is a `direct_variant` of the AOT
engine; all share `kernel_family` 1 and the same forward ABI:

| Algorithm | `direct_variant` | Kernel spec | Identities per dtype |
|---|---|---|---:|
| `direct_depthwise` | `std` | `DirectDepthwiseSpec` | 57 |
| `direct_depthwise_spatial` | `spatial` | `DirectDepthwiseSpatialSpec` | 36 |
| `direct_depthwise_tiled` | `tiled` | `DirectDepthwiseTiledSpec` | 1,536 |
| `direct_depthwise_col` | `col` | `DirectDepthwiseColSpec` | 594 |
| `direct_grouped` | `grouped` | `DirectConvSpec` | 80 |
| `direct_nongrouped` | `nongrouped` | `DirectNongroupedConvSpec` | 4,874 (1,816 split-C) |
| `direct_grouped_halo` | `grouped_halo` | `DirectNongroupedConvSpec` with `grouped=True` | 2,850 |

That is 10,027 identities per dtype. `make_conv_aot_config.py` accepts every one of them
(`PACKAGED_DIRECT_ALGORITHMS`, checked by the direct sweep's own round trip
`direct_job_for_identity`). The one forward kernel of the direct sweep that is not an
instance is `direct_split_reduce`: it ships as the second launch of each split-C
instance (see [Split-C instances](#split-c-instances)), and the adapter refuses it as an
instance of its own. The exact-shape engine's direct launch runs only `std` and
`spatial`.

A direct binary bakes the filter, stride, padding and dilation, plus its variant's block
knobs, and for `grouped` the channels per group. Batch, image extents and the group count
are kernel arguments, and for `nongrouped` and `grouped_halo` the channel counts too: the
6 leading arguments plus the 14-argument direct block (`directKernargs`). LDS, where a
kernel uses it (`grouped`, `nongrouped`, `grouped_halo`), is static. A direct instance carries these trailing fields
of `Gfx950ConvFwdAotSpec`:

| Field | Meaning |
|---|---|
| `kernel_family` | 1 for every direct variant; 0 (the default) for implicit GEMM. |
| `direct_variant` | One of the seven above; `none` for implicit GEMM. |
| `filter_h/w`, `stride_h/w`, `pad_h/w`, `dilation_h/w` | The compiled values. The kernels compile one stride and one padding, and the sweep builds square filters only. Only `nongrouped` and `grouped_halo` have a dilation other than 1. |
| `block_w`, `block_waves` | `std`, `tiled`, `col`: output columns and waves per block. `spatial` derives its width from the runtime group count, so its `block_w` is 0. |
| `block_h`, `unroll_rows` | `tiled` and `col`: output rows per block; `tiled` can unroll its filter-row loop. |
| `block_q`, `block_groups`, `double_buffer`, `cpg`, `kpg`, `block_h` | `grouped`: output columns and groups per block, LDS ping-pong, the compiled channels per group, and the input-row tile (`DirectConvSpec`'s default, which the sweep does not vary). |
| `tile_h`, `tile_w`, `tile_k_out`, `ck`, `waves_m`, `waves_n`, `atom`, `swizzle_wgm`, `iglp`, `waves_per_eu` | `nongrouped` and `grouped_halo`: `DirectNongroupedConvSpec`'s tile, channel stage, waves, MFMA atom and compile hints. `tile_k_out` is its `tile_k` (output channels per block), renamed so it cannot be the implicit-GEMM `tile_k`; `iglp` -1 and `waves_per_eu` 0 stand for rocKE's `None`. |
| `split_c` | `nongrouped`: the channel split; > 1 is a split-C instance (KMD default 1, spelled in the identity's knobs only then). |
| `group_merge` | `grouped_halo`: groups per workgroup, run as one problem of `cpg·group_merge` input and `kpg·group_merge` output channels with block-diagonal weights. The identity also spells `grouped: true`, which no field carries. |

Its implicit-GEMM fields hold exactly what the direct sweep writes into a direct
identity: 0 for the tile, warp, vector, flag and chunk fields, and `direct` for the
pipeline and epilogue. A forced `tile_k` of 64 or 128 therefore never selects one. A
variant states only its own knobs; every other knob holds its KMD default, which the
native parser checks.

All the direct fields are `builder_default` in the KMD, with the dataclass defaults.
Implicit-GEMM descriptors omit them, so their specs, symbols and pack keys are as before.
Direct instances are generator variants that state them, one per direct kernel: `std`
and `spatial` share `gfx950_conv_fwd_aot_dw_<dtype>_<instance_id>` and state only the
fields they always had, so their descriptors did not change; `tiled`, `col`, `grouped`,
`nongrouped` and `grouped_halo` are
`gfx950_conv_fwd_aot_{dwtiled,dwcol,grouped,nongrouped,groupedhalo}_<dtype>_<instance_id>`
and also state their own knobs (`DIRECT_VARIANT_FIELDS`), and the split-C instances are
`gfx950_conv_fwd_aot_nongroupedsplit_<dtype>_<instance_id>`, stating `split_c` too.
Group-merged implicit-GEMM instances are `gfx950_conv_fwd_aot_merged_<dtype>_<instance_id>`
and state `group_merge`. Kernel symbols are prefixed
`hkp_conv_fwd_dw_aot_gfx950_`. The probe problem is `direct_aot_jobs.probe_problem`
(64x64, one image; 256 groups for the depthwise variants, 8 for spatial, 64 for grouped,
one group of 1024 channels for non-grouped).

`library/tests/test_convolution_forward_aot_packaging.py` checks the probe, the knobs'
round trip and packageability of every forward direct identity of both dtypes, and the
IR, byte for byte up to the symbol, for all std and spatial identities and an evenly
spread sample of the others.

### Which graphs a direct instance serves

`std` and `spatial` keep the exact-shape engine's direct rule
(`conv::directInstanceLaunch`, `_direct_serves` in the adapter): pure depthwise, equal
strides and paddings, no dilation, an odd filter with "same" padding, the input stream
covering every output row and column, the compiled filter, stride and padding, spatial
only below 64 groups, and every grid axis at most 65535.

The other variants serve what rocKE's grouped-convolution dispatcher admits for them
(and `grouped_halo` only for more than one group),
which is what its direct benchmark offers them (`_aot_direct_serves`;
`identity_supports_problem`, then `direct_spec_for_problem`, which rebuilds the kernel
spec on the real problem and runs its validator):
the graph has one stride and one padding, the compiled filter, stride, padding and
dilation are the graph's, and then:

| Variant | Problem | Launch (grid; block) |
|---|---|---|
| `tiled` | Pure depthwise; stride at most the filter. Outputs are indexed directly, so it is not tied to "same" padding. | (⌈Wo/block_w⌉, ⌈G/(64·block_waves)⌉, N·⌈Ho/block_h⌉); 64·block_waves |
| `col` | Pure depthwise; Ho ≤ Hi, padding below the filter, block_w ≤ Wo. Rows are bounded by the runtime Ho, so "valid" padding is served too. | as `tiled` |
| `grouped` | C/G and K/G are the compiled `cpg`/`kpg`; "same" padding, odd filter, stride 1; `block_groups` divides G. | (⌈Wo/block_q⌉, G/block_groups, N·⌈Hi/block_h⌉); 64·block_groups |
| `nongrouped` | G = 1; C a multiple of `ck` and of 8, K of the atom tile; every tensor below the kernel's masked-access offset; `tile_w` one of the widths a sweep of Wo tries (`tile_w_candidates`). Split-C: at least `split_c` whole `ck` chunks, and the workspace below that offset. | (⌈Wo/tile_w⌉·⌈Ho/tile_h⌉·N·⌈K/tile_k_out⌉·split_c, 1, 1); 64·waves_m·waves_n |
| `grouped_halo` | G > 1 (rocKE's `direct_plans` offers it no other), divisible by `group_merge`; merged C (`cpg·group_merge`) a multiple of `ck` and of 8, merged K of the atom tile and at least `tile_k_out`; `cpg` a multiple of the weight load (4 when merged, else the atom's fragment); tensors and `tile_w` as `nongrouped`. | (⌈Wo/tile_w⌉·⌈Ho/tile_h⌉·N·(G/group_merge)·⌈merged K/tile_k_out⌉, 1, 1); 64·waves_m·waves_n |

On top of rocKE's rules the packaged launch bounds every axis of a 3-D direct grid by
65535, as for std and spatial, and the flat grid of the non-grouped and grouped-halo
kernels by a 32-bit dispatch. A batch whose activation or output reaches the
masked-access offset runs in pieces, as rocKE's benchmark runs it (see "Batch-split
launches" below). The adapter's test compares the served verdict with rocKE's offer over
problems around each sampled instance (tails, images smaller than a block or the filter,
other paddings, strides, dilations and channel counts). The only differences
are where the packaged guard is knowingly stricter: spatial at exactly 64 groups (as
before), and graphs whose two axes differ in stride or padding, which rocKE's direct
problem cannot state.

Natively, `aotDirectInstance` in `Gfx950ConvFwdNative.cpp` parses the variant and its
knobs (`aotDirectVariantUses`), and `conv::directInstanceLaunch` in
`Gfx950ConvFwdGeometry.hpp` applies the variant's rule (`depthwiseTileLaunch`,
`groupedDirectLaunch`, `nongroupedDirectLaunch` for both DirectNongroupedConvSpec
variants). `TestGfx950ConvFwdGeometry.cpp`
holds it to a table of (instance, problem) cases that
`tests/test_make_conv_aot_config.py` recomputes from the adapter.

### Split-C instances

A split-C non-grouped instance (`split_c` > 1) runs two kernels: its conv pass writes
fp32 partial sums into a `[split_c, N, Ho, Wo, K]` workspace bound to D, over a grid
`split_c` times wider, and rocKE's `direct_split_reduce` (`DirectSplitReduceSpec`, 256
threads, 8 outputs each) sums the slabs into the output. Both take the forward
direct-conv arguments of the problem, as rocKE's direct benchmark runs them
(`direct_kernel_sweep.plan_for`).

Each split-C descriptor carries both as launches of its `sources` map (UKD 1.1,
RFC 0017 §15.1): `conv` is the instance's own builder and `reduce` is
`build_gfx950_conv_fwd_aot_split_reduce`, over the same spec. A dispatch handler declares
one set of launch names, so the split-C instances are packs of their own
(`<dtype>_direct_split`) that name their own dispatch: the generator's pack-level
`dispatch`, `split_c`, gives them the UDD `gfx950_conv_fwd_aot_split_c.udd.json` and the
native symbol `hipkernel.gfx950_conv_fwd_aot.split_c_dispatch`. Its handler, the same
admission as the engine's with `launchNames()` {conv, reduce}, sizes the workspace
(`conv::splitWorkspaceBytes`), loads both launches (`launchKernelDefinition`) and runs
them in order on the stream. Every other pack keeps the engine's single-launch handler.

### Batch-split launches

The non-grouped and grouped-halo kernels address each operand through one 32-bit buffer
resource and redirect masked accesses to the offset `0x7F000000`, so no operand may
reach it. Their images are independent, so rocKE's direct benchmark runs a batch whose
activation or output does as several launches over windows of the batch
(`direct_kernel_sweep._batch_chunks` and `_batch_split_plan`). As many images run per
launch as keep both windows below the offset, and the launches are balanced: a bf16
3x3 convolution of 4 images of 1024x1024 with 256 channels runs as 2 launches of 2
images.

The packaged engine runs it the same way. `conv::nongroupedBatchChunks` in
`Gfx950ConvFwdGeometry.hpp` (`gfx950_conv_fwd_aot_batch_chunks` in the adapter) gives
the images per launch. The graph matcher admits such a graph even when a whole tensor
passes the i32 launch ABI; only a non-grouped or grouped-halo instance serves it, and
only when it serves each chunk as a problem of its own. That is stricter than
`_batch_split_plan`, which re-validates a chunk's spec but not the rules of
`direct_spec_for_problem` (a grouped-halo `tile_k_out` past the merged K). A split-C
instance is never batch-split, since its workspace would span the whole batch. The
handler loads one code object per chunk, with that chunk's grid, and launches them in
order on the stream, each with its activation and output pointers offset to its
window and its own byte counts and batch; the filter is whole in every launch. The
overlap check covers the whole tensors, in 64-bit byte counts. Every other instance,
and the exact-shape engine, still needs the whole problem in one launch.

## Which graphs an instance serves

The graph contract is the exact-shape engine's, read by the same graph matcher: 2-D
cross-correlation, packed NHWC fp16/bf16, fp32 accumulation, symmetric padding. Grouped
pointwise graphs are served by grouped instances: an instance is built against a 3x3
probe, so it is never rocKE's pointwise GEMM.

An implicit-GEMM instance serves a graph when all of these hold (`aotInstanceLaunch` in
`Gfx950ConvFwdGeometry.hpp`, `gfx950_conv_fwd_aot_serves` in the adapter). Both apply
the exact-shape engine's capability check (`implicitGemmServes`,
`_implicit_gemm_serves`) to the widths the identity records, plus the async chunks:

- its dtype is the graph's;
- `vector_size_a` and `vector_size_b` divide C/groups, and `vector_size_c` divides
  K/groups;
- for an async binary, both chunk widths divide C/groups;
- the graph is ungrouped, or the binary is grouped;
- every grid axis fits 65535;
- Y·X·C/groups stays below 2^23 (the kernel's 24-bit address products).

A group-merged instance has no recorded widths to check; it serves by its own rule
(see [What an instance is](#what-an-instance-is)).

The launch is the exact-shape engine's: `warp_m * warp_n * 64` threads, static LDS, and
the 45-argument AOT ABI computed from the graph. Before any launch, the pack checks the
packaged symbol's recorded argument names against that ABI.

Without benchmarking or a recorded winner, `aotScore` orders instances. As in the
exact-shape engine, a depthwise direct instance (`std`, `spatial`, `tiled`, `col`) ranks
above every implicit-GEMM one, the default direct arm for the graph's group count first
(spatial with one wave below 64 groups, otherwise std with `block_w` 4 and one wave).
The `grouped`, `nongrouped` and `grouped_halo` kernels compete with implicit GEMM on ordinary graphs and
nothing measures yet which wins where, so they take a neutral placeholder below every
implicit-GEMM instance (and above a refusal): an unbenchmarked dense or grouped graph
stays on implicit GEMM, and these kernels are chosen by benchmarking, a recorded winner,
the `kernel_family` knob, or where nothing else serves. Among implicit-GEMM instances,
`aotInstanceScore` prefers less tile padding and a grid that fills the device, with
wider vectors breaking ties. It is a placeholder until rocKE's dispatcher ranks
instances. With benchmarking, hipDNN measures every instance selected for the graph (see
[Serves and selected](#serves-and-selected)) and records the fastest. The `kernel_family` knob forces one family, as on the exact-shape
engine.

## Serves and selected

"Serves" and "selected" are separate. An instance **serves** a graph when its binary
computes it: the capability check of the previous section, one instance at a time. Of
the instances that serve, the engine **selects** the ones it runs, and only those are
its candidates: ranked, benchmarked, or limited by `HIPDNN_INGESTOR_CANDIDATE_LIMIT`.

Narrower access does not win, so the policy keeps the problem's widest vector widths.
The problem's widest combination is the widest load width C/groups admits and the
widest store width K/groups admits, each the largest of 8, 4, 2 and 1 dividing it:

- When some serving implicit-GEMM instance bakes exactly that combination, only those
  run, of either epilogue. A configuration that has no copy at that combination does
  not run.
- When none does, every serving instance runs. The shipped catalog builds only some
  combinations ([The shipped catalog](#the-shipped-catalog)), so for a problem whose
  widest is one it lacks, the selection falls back to every combination the catalog
  does build for it rather than guessing which narrower one is best.

In a catalog that builds every combination of each configuration, as rocKE's AOT sweep
does, this is the default of rocKE's `--run-from-cache` benchmark (`_run_from_cache` in
`rocke/library/benchmarks/common/benchmark_implicit_gemm_conv.py`): `KernelCache.compatible`
gives the kernels that serve, then `_widest_vector_sizes` keeps, per kind (here the
epilogue), the widest combination present, and drops the scalar-store default epilogue
when anything stores wider.

Which combination runs depends on which instances serve, not on one instance, so the
policy is a property of the candidate set. The plugin SDK lets a pack refine a
kernel-scoped matcher with a set-level stage for this: a `CandidateFilterFn`
(`NativeRegistry.hpp`), registered under the symbol of a kernel-scoped matcher the pack
already declares. Once every pack is matched, it sees every kernel that matcher admitted
for the graph, across all packs listing it (every shard here), and keeps a subset.
No descriptor changes: a matcher with no filter registered keeps what it admitted.
`hipkernel:Gfx950ConvFwdAot` registers `aotCandidateFilter` under its kernel matcher's
symbol; the policy itself is `conv::widestVectorSelection` in
`Gfx950ConvFwdAotSelection.hpp`. Direct instances of every variant are all kept:
rocKE's direct benchmark (`direct_plans` in `benchmarks/common/direct_kernel_sweep.py`)
runs every direct kernel that serves, with no cross-candidate rule. So are group-merged
instances, whose widths are derived (recorded as 0) and never compared. A graph's direct
and merged candidates therefore add to its selected implicit-GEMM ones.

The selected instances are the engine's catalog for the graph, so everything after it
applies to them alone, in this order: the knob filter (a forced `tile_k` or
`kernel_family`), the ranking (`aotScore` or a benchmarked record), and
`HIPDNN_INGESTOR_CANDIDATE_LIMIT`. A forced knob therefore picks among the selected
instances; a knob value that only narrower copies have selects nothing.

The Python mirror is `gfx950_conv_fwd_aot_runtime_candidates` in
`rocke/library/builders/common/convolution_forward.py`. Three tests hold the three
together:

- `library/tests/test_convolution_forward_aot_packaging.py` calls rocKE's own
  `KernelCache.compatible` and `_widest_vector_sizes` on every identity of the AOT sweep,
  over a fixed grid of problems, and requires the mirror's selection to be exactly
  theirs, in both dtypes. It also drops the widest combination from the sweep and
  requires every serving instance to run.
- `TestGfx950ConvFwdAotSelection.cpp` runs the native kernel matcher and candidate
  filter over a sample of sweep identities and synthetic graphs, and the policy
  directly.
  `tests/test_make_conv_aot_config.py` recomputes its expected rows with the mirror.
- The plugin SDK's `TestKernelIngestorStateManager.cpp` covers the filter stage itself.

## The shipped catalog

`hipkernel:Gfx950ConvFwdAot` ships a curated list of rocKE kernels,
`projects/hipdnn/tools/IngestorGenerator/configs/gfx950_conv_fwd_aot.labels.txt`: one
`KernelIdentity.label()` per line, the same kernels in fp16 and bf16.
`tools/enumerate_conv_aot_instances.py` turns it into the per-dtype lists,
`configs/gfx950_conv_fwd_aot_<dtype>.instances.json`:

| Family | fp16 | bf16 |
|---|---:|---:|
| Implicit GEMM | 217 | 217 |
| Implicit GEMM, group-merged (`group_merge` 4, 8, 16) | 10 | 10 |
| Direct `nongrouped`, split-C (`split_c` 2, 4, 8, 16) | 91 | 91 |
| Direct `nongrouped` | 84 | 84 |
| Direct depthwise `tiled` | 79 | 79 |
| Direct `grouped_halo` (`group_merge` 1, 2, 4, 8) | 45 | 45 |
| Direct depthwise `std` | 5 | 5 |
| Direct `grouped` | 3 | 3 |
| Direct depthwise `spatial` | 3 | 3 |
| **Total** | **537** | **537** |

1,074 instances in all. Every entry must be an identity of rocKE's gfx950 forward
sweeps (`aot_jobs.py` for implicit GEMM, `direct_aot_jobs.py` for direct) and
must convert, that is, the packaged launch runs it. The tool fails on any entry that
is not, rather than dropping it.

The list is the union of the fastest kernel (top-1) per measured shape over rocKE's
forward sweeps on 2-D fp16 and bf16 forward convolutions. Unlike the previous catalog it
adds no coverage of its own: it has no `col` kernel, and only one implicit-GEMM
instance stores scalars.

The implicit-GEMM instances that record their widths bake these load/store widths (A
and B load widths are always equal):

| Epilogue | Load/store widths, fp16 instances |
|---|---|
| `cshuffle` | 8/8/8 (140), 1/1/8 (38), 4/4/8 (15), 8/8/4 (7), 8/8/1 (2), 1/1/1 (1) |
| `default` | 8/8/1 (14) |

So a problem whose widest combination is not one of these has no instance at it, and
runs every instance that serves it ([Serves and selected](#serves-and-selected)). Every
problem class still has an implicit-GEMM instance
(`test_the_catalog_serves_every_problem_class`), though odd or narrow channel counts
have few (one where only 1/1/1 serves). No instance uses the async DMA K loop.

Which instances serve a given problem, and which of those run, is decided at runtime,
not by the catalog. For the GPU suite's graphs the engine selects, per dtype, 140
implicit-GEMM candidates for a dense graph with C and K divisible by 8, 1 to 38 where
the widest combination is narrower, and 6 to 11 for depthwise
(`aotGraphs()` in `IntegrationGpuGfx950ConvFwd.cpp`).

The lists are compact: one identity per line, holding only the fields that have no
plain default or that differ from it. `KernelIdentity.from_dict` restores the rest, and
with it the `stable_hash()`. Each list holds its dtype's implicit-GEMM instances first,
then the direct ones, each ordered by hash. `stable_hash()` covers `llvm_flavor`, so the
lists are valid for the LLVM flavor the packer lowers to.

### Shards

Every file is far below the 50 MB GitHub warns about. The bundle has one
pack, and so one KDP, per dtype, implicit-GEMM pipeline and K loop, plus two per dtype
for the direct instances, the split-C ones apart because they launch through a dispatch
of their own (`make_conv_aot_config.py --sharding k-loop`, the default):

| KDP shard, per dtype | Instances |
|---|---:|
| `gfx950_conv_fwd_aot_<dtype>_mem_plain` | 76 |
| `gfx950_conv_fwd_aot_<dtype>_mem_unroll` | 18 |
| `gfx950_conv_fwd_aot_<dtype>_compv3_plain` | 60 |
| `gfx950_conv_fwd_aot_<dtype>_compv3_unroll` | 9 |
| `gfx950_conv_fwd_aot_<dtype>_compv4_plain` | 54 |
| `gfx950_conv_fwd_aot_<dtype>_compv4_unroll` | 10 |
| `gfx950_conv_fwd_aot_<dtype>_direct` | 219 |
| `gfx950_conv_fwd_aot_<dtype>_direct_split` | 91 |

The KDPs are in DVC, not git: they live in the bundle's `artifacts/` folder, tracked by
`artifacts.dvc`, and the trained model under `heuristics/` by `heuristics.dvc`. Fetch
both with `dvc pull -r ingestor artifacts.dvc heuristics.dvc` from the bundle folder;
see the descriptors README, "Large rocKE files live in DVC". The generator writes each
KDP with one kernel per line.

The packs declare no `discriminator`, so they are shards of one catalog: the engine's
graph matcher and the one shared kernel matcher admit all of them, and there is no
operation-scoped matcher (the IngestorGenerator README, "UMD policy"). The census
`TestGfx950ConvFwdAotPacks.cpp` lists every pack and kernel name exactly. The test-only
seed keeps one pack per dtype, and one more for its split-C instances
(`--sharding dtype`).

### Regenerating the bundle

From `projects/hipdnn/tools/IngestorGenerator`, with
`ROCKE=<repo>/dnn-providers/hip-kernel-provider/rocke`, `ROCM_PATH=/opt/rocm`,
`ROCKE_BACKEND=python`, and `PYTHONPATH` set to `$ROCKE/platform/python:$ROCKE/library`:

```bash
python tools/enumerate_conv_aot_instances.py --rocke-root "$ROCKE" --output-dir configs
python tools/make_conv_aot_config.py --rocke-root "$ROCKE" \
    --instances configs/gfx950_conv_fwd_aot_{fp16,bf16}.instances.json \
    --output "$WORK"/aot.yaml
python generate.py --config "$WORK"/aot.yaml --output-dir "$GEN" --force
DEST=../../../../dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_conv_fwd_aot
rm "$DEST"/*.json "$DEST"/artifacts/*.kdp.json
for f in "$GEN"/descriptors/rocKE/gfx950_conv_fwd_aot/*.json; do
  case "$f" in *.kdp.json) cp "$f" "$DEST"/artifacts/ ;; *) cp "$f" "$DEST"/ ;; esac
done
cp "$GEN"/tests/TestGfx950ConvFwdAotPacks.cpp \
   ../../../../dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/
python tools/make_conv_aot_config.py --stabilize-ids "$DEST"/artifacts/*.kdp.json
(cd "$DEST" && dvc add artifacts)
```

Then push the new `artifacts/` with `dvc push -r ingestor` and commit `artifacts.dvc`
with the rest (the descriptors README, "Adding or changing a large file"), and run
`clang-format` on the census. Keep the document ids of the KMD, UED, UDDs, UHD,
UMD and KDP files from the previous bundle, so that the diff shows only the kernels.
The generator mints new ones on every run. Keep the shipped UED itself too: it names the
trained `sort_kernel_catalog` model under `heuristics/`; take only its `knobs` from the
generated one. The whole flow takes a few minutes.

How the converter treats the lists:

- **Unsupported entries:** the tool lists each with its reason and writes nothing.
  `--skip-unsupported` drops them instead.
- **Duplicates:** entries with the same `instance_id` collapse to one.
- **Stable kernel IDs:** `--stabilize-ids` re-derives each kernel's UUID as
  `uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`. The generator mints random UUIDs, and
  a recorded autotune winner is keyed by kernel UUID, so this keeps records valid
  across regenerations. Another engine name (`--engine-name`) derives its ids under a
  namespace of its own name.

To change the catalog, edit `configs/gfx950_conv_fwd_aot.labels.txt` and rerun the
flow. Commit the labels and the instance lists. The generator config is not committed: the converter
regenerates it from the lists on every run. The tests hold the pieces together:

- `tests/test_enumerate_conv_aot_instances.py` fails when the lists are not what the
  tool builds from the committed labels and the rocKE tree beside them, when a label is
  not a sweep identity or does not convert, or when the catalog is not 537 kernels per
  dtype, the same in both.
- `tests/test_make_conv_aot_config.py` fails when a shipped shard is not exactly the
  lists' instances of its shard, with instance-derived ids, or when a committed file is
  50 MB or larger. It also fails when a problem class loses its last implicit-GEMM
  instance: odd C/groups or K/groups, C/groups or K/groups of 2 mod 4, grouped,
  pointwise and depthwise graphs.
- rocKE's `library/tests/test_convolution_forward_aot_packaging.py` builds one instance
  of every pipeline, epilogue, K loop, async chunk and width class in each dtype and
  checks that its IR is the sweep kernel's, byte for byte up to the symbol.

Then rebuild with the pack stamps removed, so every instance is compiled from the
current sources.

### Build and package cost

The product pack compiles every instance, 1,074 variants for this engine (and a second
launch for each of the 182 split-C ones).

The build figures below were measured for an earlier catalog, rocKE's whole forward
sweep with only the std and spatial direct kernels (about 159,000 variants with the
other product bundles), on one host with 96 pack workers
(`-DHIPKERNELPROVIDER_PRODUCT_PACK_JOBS=96`). They have not been re-measured for this
catalog, which is more than a hundred times smaller:

| Build output (earlier catalog) | Size |
|---|---:|
| gfx950 kpack archive (all product bundles) | about 1.3 GB |
| Packed implicit-GEMM KDP shard, each | about 40 to 44 MB |
| Provenance sidecar, each implicit-GEMM shard | about 4.2 to 4.6 MB, 110 to 120 MB inflated |

Packing all product bundles took about 7 minutes with a warm comgr cache and about
23 minutes with an empty one; the comgr cache grew to about 14 GB. The packer holds
every code object in memory before it writes the archive, so it peaked at about 16 GB
of resident memory. `HIPKERNELPROVIDER_PRODUCT_PACK_JOBS` sets the worker count of the
production pack alone (empty, the default, lets the packer size its pool to the
machine). The test roots keep their own small, fixed pools. The census compiles in
seconds, because it builds its name sets from constant arrays.

Size limits the catalog stays within:

- **kpack archive.** Offsets and the TOC offset are 64-bit, and each zstd frame's 32-bit
  size covers one code object. The C++ reader (`rocm-kpack` runtime) refuses a zstd
  blob above 4 GiB or more than 1,048,576 kernels in one archive, and reads the whole
  blob into memory when it opens the archive, so the archive's size is resident once a
  kernel is loaded.
- **Provenance sidecars.** A sidecar inflates past the reader's 64 MiB floor, so it is
  read under the per-UKD allowance (`MAX_DECOMPRESSED_BYTES_PER_UKD`, 64 KiB per UKD of
  its descriptor), which leaves a wide margin.
- **Plugin load.** Loading the product root parses every packed KDP once per process;
  for the earlier catalog that was about 570 MB of JSON.

### Selecting among many candidates

A dense graph is served by most implicit-GEMM instances of its dtype; the runtime
selection ([Serves and selected](#serves-and-selected)) keeps the problem's widest
combination, or every serving instance when the catalog lacks it. An unbenchmarked
plan takes the best by score.
Benchmarking loads and measures every selected candidate, so a cold plan build still
grows with the number of configurations. `HIPDNN_INGESTOR_CANDIDATE_LIMIT=<n>`
(`GlobalKnobDefines.hpp`) keeps only the `n` best-ranked candidates after knob
filtering, before the winner-record lookup, with benchmarking on or off. A record
benchmarked under a limit therefore covers exactly the candidates a later build under
the same limit walks. Unset, or any value that is not a positive integer, means no
limit.

In the GPU suite's synthetic graphs, a dense 3x3 graph selects 3,260 of its 78,940
serving implicit-GEMM instances, plus up to a few hundred direct ones (146 for the
default `Smoke` graph); a depthwise graph selects 7,440 implicit-GEMM instances (only
width 1 serves one channel per group) and about 110 direct ones.

### Plan-build cost

Each (graph, device) is matched once: every kernel of the engine goes through the
kernel matcher, then the candidate filter narrows what it admitted, and the result is
cached. These plugin SDK mechanisms keep that and every later call cheap:

- **Catalog views.** A cached catalog is a `CatalogView`: pointers into the engine's
  shared, immutable kernel definitions, handed out as `shared_ptr<const CatalogView>`.
  No plugin call copies the catalog, and a plan copies only the definition it launches.
  The cache keeps the 256 most recent (graph, device) catalogs at 8 bytes per selected
  kernel, so it is bounded at a few MB.
- **Prepared kernels.** `aotPrepare`, a `KernelPrepareFn` registered under the AOT
  kernel matcher's symbol, decodes each kernel's dtype, epilogue and instance once, when
  the engine loads. The kernel matcher, score and candidate filter read that instead of
  about fifty metadata fields per kernel, and read the matched problem from what the
  graph match bound instead of walking the graph per kernel.

- **A key for every graph.** The cache keys a graph by its v4 id when it carries one
  and otherwise by its exact serialized bytes (`CatalogGraphKey`). A corpus graph's
  content-derived id is not v4, and without this every plugin call for it rematched the
  whole engine.
- **Linear knob handling.** A knob pin on a non-integer field, such as the instance id,
  resolves its ordinal by binary search in the field's sorted domain, and the advertised
  knob values are deduplicated with a set, so an engine-details or plan call stays
  linear in the catalog. Plan calls copy only the kernels that survive the pins.

Without these, a `hipdnn_bench --sweep` over a graph, which pins every knob of each
candidate and makes several engine-details, workspace and plan calls per candidate, spent
almost all its time re-matching and re-filtering the catalog rather than timing kernels.
A benchmarked plan adds the candidates' code-object loads and timed runs, which
`HIPDNN_INGESTOR_CANDIDATE_LIMIT` bounds.

## The seed engine and the GPU tests

`hipkernel:Gfx950ConvFwdAotSeed` is a test-only engine over a 220-instance seed, taken
verbatim from the sweeps' valid forward grids
(`configs/gfx950_conv_fwd_aot_seed.instances.json`). It has 12 implicit-GEMM instances
per dtype, covering:

- every load width and every store width the catalog ships, in the pairs (1, 1),
  (2, 2), (4, 4), (8, 8), (1, 4), (4, 1), 8/8/4, 4/4/2 and a `cshuffle` 2/2/1;
- both epilogues, the default one with a store width of 1;
- the `mem`, `compv3` and `compv4` pipelines;
- `unroll_k` and `async_dma`.

1/1/4, 4/4/2, `cshuffle` 2/2/1 and the default epilogue's 4/4/1 are outside the shipped
catalog (see [The shipped catalog](#the-shipped-catalog)). The seed keeps them so that
the native launch's handling of them stays tested.

`tests/test_make_conv_aot_config.py` checks that the seed holds every pipeline,
epilogue, K loop, load width and store width of the shipped catalog, in each dtype. The
catalog's other width pairs run the same launch with other arguments; rocKE's packaging
test checks the IR of every width, K loop and async chunk class of the sweep.

It also has 96 direct instances per dtype, and 2 group-merged implicit-GEMM ones
(`group_merge` 4 and 8, served by its depthwise graphs). 4 are std and spatial: std 3x3 stride 1
(`block_w` 4, one wave), std 5x5 stride 2 (`block_w` 16, two waves), spatial 3x3
stride 1 (one wave) and spatial 7x7 stride 2 (four waves). The other 88 are 22 each of
`tiled`, `col`, `grouped` and `nongrouped`, chosen deterministically from the sweep so
that together they hold every value of every knob the sweep builds for the variant,
every grouped channel count (4, 8, 16, 32) and every non-grouped dilation, and so that
each serves at least two of the seed's graphs. The last 4 cover the direct kinds with a
launch of their own: `grouped_halo` at `group_merge` 1, 4 and 8, and a split-C non-grouped
kernel (`split_c` 2, two launches and a workspace), each served by some of the seed's
grouped or dense graphs. `tests/test_make_conv_aot_config.py` checks the knob coverage,
that every launch kind of the catalog has a seed instance, and the counts.

The seed is authored under
`src/engines/kernel_ingestor_engine/test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed`
and packed into the integration test root. Its descriptors name the native symbols of
`hipkernel:Gfx950ConvFwdAot` (the generator's `engine.native_symbols_from`), so it runs
the shipped code and registers nothing of its own. It needs a name of its own because
the integration binary loads the product root as well. The same tools generate it,
with the converter's `--engine-name hipkernel:Gfx950ConvFwdAotSeed`. Its config,
`configs/gfx950_conv_fwd_aot_seed.yaml`, is committed as a worked example. Its census,
`TestGfx950ConvFwdAotSeedPacks`, reads the integration root's gfx950 shard.

`IntegrationGpuGfx950ConvFwd.cpp` runs one table of 25 graphs per dtype against both
engines, 9 of them depthwise over several group counts, batches and images. The seed
also runs a second table, `aotSeedDirectGraphs()`, of 56 graphs for its tiled, col,
grouped and non-grouped instances: odd images, batches and group counts with tails,
images narrower than a block, "valid" and "same" padding for col, filters up to 31x31
for tiled, every grouped channel count, non-grouped output widths that each swept tile
width divides, stride 2 and every swept dilation.

- **`IntegrationGpuGfx950ConvFwdAotSeed`** benchmarks every selected instance, then pins
  each one in turn and checks it against the CPU reference. It also forces each family
  through the `kernel_family` knob, and checks that an unbenchmarked plan takes a
  depthwise direct instance on a depthwise graph and implicit GEMM elsewhere. Six graphs
  of the first table exist only so that every seed instance is selected for some graph:
  an instance never selected would never be pinned.
- **`IntegrationGpuGfx950ConvFwdAot`** runs the shipped catalog. An unbenchmarked plan
  is executed and checked. The selected count of each family is checked through the
  `kernel_family` knob. The 4 best instances by score are pinned and checked. So are 8
  more, drawn by a seeded shuffle of every kernel in the dtype's shards: a record that
  orders all of them covers every candidate, so it serves the first instance in its
  order that is selected for the graph. A benchmark under
  `HIPDNN_INGESTOR_CANDIDATE_LIMIT=64` is checked, and so is its reuse.

Each row of the table states how many instances of each family are selected for the
graph, in the seed and in the full catalog, with the full catalog's implicit-GEMM count
per dtype, since each dtype has a list of its own.
`tests/test_make_conv_aot_config.py` recomputes the counts with
`gfx950_conv_fwd_aot_runtime_candidates` over the committed lists, checks that they never
exceed what `gfx950_conv_fwd_aot_serves` counts, and checks that every seed instance is
selected for some graph. The integration
CMake compiles each engine's suite only where that engine ships: the seed's where the
integration root packs it for gfx950, the product's where the product root offers it.

## Current limits

- Forward only. The launch runs every forward kernel of rocKE's sweeps, but the shipped
  catalog packages only the 537 per dtype its label list names; the seed exercises
  knob values the catalog does not ship.
- The trained `sort_kernel_catalog` model predates the group-merged, grouped-halo and
  split-C instances. It ignores the `group_merge` and `split_c` knobs, and the
  `grouped_halo` value of `direct_variant` is unseen by it; benchmarking measures every
  selected instance regardless. Retrain it on this catalog before relying on the
  unbenchmarked ranking.
- One score heuristic, as described above. It is a placeholder until rocKE's
  dispatcher ranks instances. Benchmarking every selected instance grows with the
  number of configurations in the catalog; `HIPDNN_INGESTOR_CANDIDATE_LIMIT` bounds it
  to the best-ranked ones, which makes the quality of the score matter more.
