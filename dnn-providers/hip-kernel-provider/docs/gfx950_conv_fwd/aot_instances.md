# Shape-generic AOT instances: hipkernel:Gfx950ConvFwdAot

rocKE's AOT convolution kernels take the problem as kernel arguments. One compiled
binary therefore serves every problem its build-time capabilities admit.
`hipkernel:Gfx950ConvFwdAot` packages such binaries as **instances** and matches them
to a graph by capability, not by exact shape. It ships next to the exact-shape engine
`hipkernel:Gfx950ConvFwd`, which it does not change.

## What an instance is

An instance is one rocKE `KernelIdentity`
(`rocke/library/benchmarks/common/kernel_cache.py`): the compile-time tuple of a
rocKE sweep. It holds configuration and capability fields and no problem extents.
Two families are packaged, both gfx950, forward, fp16 or bf16:

- implicit GEMM, from the AOT sweep (`benchmarks/common/kernel_sweep.py`);
- direct, from the direct sweep (`benchmarks/common/direct_kernel_sweep.py`), every
  forward algorithm of it: see [Direct instances](#direct-instances).

An implicit-GEMM instance has:

| Field | Meaning |
|---|---|
| `tile_m`, `tile_n`, `tile_k`, `warp_m`, `warp_n`, `warp_tile_m/n/k`, `wave_size` | Tile and warp geometry. Wave64 only. |
| `pipeline`, `epilogue` | `mem`, `compv3`, `compv4`, `basic`; `default` or `cshuffle`. |
| `vector_size_a/b/c` | Baked load/store widths. A and B run along C/groups, D along K/groups. |
| `unroll_k`, `async_dma`, `async_chunk_a/b` | K-loop driver; an async binary's chunk widths are a capability. |
| `grouped` | Built with the grouped code path, which also serves groups == 1. The sweep builds only grouped binaries. |
| `instance_id` | `KernelIdentity.stable_hash()` of the source identity. This is the tracking key shared with rocKE's dispatcher. |

Every other `KernelIdentity` field must hold its default, because it selects code this
launch does not model. Examples are split-K, `chiplet_swizzle`, 3-D and the LDS layout
knobs. The tool names any such field and refuses the instance.

Each instance is built the way its sweep builds it: `Gfx950ConvFwdAotSpec` in
`rocke/library/builders/common/convolution_forward.py` uses the sweep's probe problem.
Only the kernel symbol differs. `library/tests/test_convolution_forward_aot_packaging.py`
holds the adapter to that, and to the sweep's served-problem predicate.

## Direct instances

The direct sweep has six forward algorithms. Each is a `direct_variant` of the AOT
engine; all share `kernel_family` 1 and the same forward ABI:

| Algorithm | `direct_variant` | Kernel spec | Identities per dtype |
|---|---|---|---:|
| `direct_depthwise` | `std` | `DirectDepthwiseSpec` | 54 |
| `direct_depthwise_spatial` | `spatial` | `DirectDepthwiseSpatialSpec` | 30 |
| `direct_depthwise_tiled` | `tiled` | `DirectDepthwiseTiledSpec` | 1,536 |
| `direct_depthwise_col` | `col` | `DirectDepthwiseColSpec` | 540 |
| `direct_grouped` | `grouped` | `DirectConvSpec` | 80 |
| `direct_nongrouped` | `nongrouped` | `DirectNongroupedConvSpec` | 3,058 |

That is 5,298 identities per dtype. `make_conv_aot_config.py` accepts every one of them
(`PACKAGED_DIRECT_ALGORITHMS`, checked by the direct sweep's own round trip
`direct_job_for_identity`). The exact-shape engine's direct launch runs only `std` and
`spatial`, and the shipped catalog holds only those two so far (see
[The shipped catalog](#the-shipped-catalog)).

A direct binary bakes the filter, stride, padding and dilation, plus its variant's block
knobs, and for `grouped` the channels per group. Batch, image extents and the group count
are kernel arguments, and for `nongrouped` the channel counts too: the 6 leading
arguments plus the 14-argument direct block (`directKernargs`). LDS, where a kernel uses
it (`grouped`, `nongrouped`), is static. A direct instance carries these trailing fields
of `Gfx950ConvFwdAotSpec`:

| Field | Meaning |
|---|---|
| `kernel_family` | 1 for every direct variant; 0 (the default) for implicit GEMM. |
| `direct_variant` | One of the six above; `none` for implicit GEMM. |
| `filter_h/w`, `stride_h/w`, `pad_h/w`, `dilation_h/w` | The compiled values. The kernels compile one stride and one padding, and the sweep builds square filters only. Only `nongrouped` has a dilation other than 1. |
| `block_w`, `block_waves` | `std`, `tiled`, `col`: output columns and waves per block. `spatial` derives its width from the runtime group count, so its `block_w` is 0. |
| `block_h`, `unroll_rows` | `tiled` and `col`: output rows per block; `tiled` can unroll its filter-row loop. |
| `block_q`, `block_groups`, `double_buffer`, `cpg`, `kpg`, `block_h` | `grouped`: output columns and groups per block, LDS ping-pong, the compiled channels per group, and the input-row tile (`DirectConvSpec`'s default, which the sweep does not vary). |
| `tile_h`, `tile_w`, `tile_k_out`, `ck`, `waves_m`, `waves_n`, `atom`, `swizzle_wgm`, `iglp`, `waves_per_eu` | `nongrouped`: `DirectNongroupedConvSpec`'s tile, channel stage, waves, MFMA atom and compile hints. `tile_k_out` is its `tile_k` (output channels per block), renamed so it cannot be the implicit-GEMM `tile_k`; `iglp` -1 and `waves_per_eu` 0 stand for rocKE's `None`. |

Its implicit-GEMM fields hold exactly what the direct sweep writes into a direct
identity: 0 for the tile, warp, vector, flag and chunk fields, and `direct` for the
pipeline and epilogue. A forced `tile_k` of 64 or 128 therefore never selects one. A
variant states only its own knobs; every other knob holds its KMD default, which the
native parser checks.

All the direct fields are `builder_default` in the KMD, with the dataclass defaults.
Implicit-GEMM descriptors omit them, so their specs, symbols and pack keys are as before.
Direct instances are generator variants that state them, one per direct kernel: `std`
and `spatial` share `gfx950_conv_fwd_aot_dw_<dtype>_<instance_id>` and state only the
fields they always had, so their descriptors did not change; `tiled`, `col`, `grouped`
and `nongrouped` are `gfx950_conv_fwd_aot_{dwtiled,dwcol,grouped,nongrouped}_<dtype>_<instance_id>`
and also state their own knobs (`DIRECT_VARIANT_FIELDS`). Kernel symbols are prefixed
`hkp_conv_fwd_dw_aot_gfx950_`. The probe problem mirrors `direct_kernel_sweep.probe_problem`
(64x64, one image; 256 groups for the depthwise variants, 8 for spatial, 64 for grouped,
one group of 256 channels for non-grouped).

`library/tests/test_convolution_forward_aot_packaging.py` checks the probe, the knobs'
round trip and packageability of every forward direct identity of both dtypes, and the
IR, byte for byte up to the symbol, for all std and spatial identities and an evenly
spread sample of the others. The IR identity of all 10,596 identities was also checked
once, out of tree.

### Which graphs a direct instance serves

`std` and `spatial` keep the exact-shape engine's direct rule
(`conv::directInstanceLaunch`, `_direct_serves` in the adapter): pure depthwise, equal
strides and paddings, no dilation, an odd filter with "same" padding, the input stream
covering every output row and column, the compiled filter, stride and padding, spatial
only below 64 groups, and every grid axis at most 65535.

The other variants serve what rocKE's own direct benchmark offers them
(`_aot_direct_serves`; `KernelCache.supports_problem`, then `direct_plans` and
`plan_for`, which rebuild the kernel spec on the real problem and run its validator):
the graph has one stride and one padding, the compiled filter, stride, padding and
dilation are the graph's, and then:

| Variant | Problem | Launch (grid; block) |
|---|---|---|
| `tiled` | Pure depthwise; stride at most the filter. Outputs are indexed directly, so it is not tied to "same" padding. | (⌈Wo/block_w⌉, ⌈G/(64·block_waves)⌉, N·⌈Ho/block_h⌉); 64·block_waves |
| `col` | Pure depthwise; Ho ≤ Hi, padding below the filter, block_w ≤ Wo. Rows are bounded by the runtime Ho, so "valid" padding is served too. | as `tiled` |
| `grouped` | C/G and K/G are the compiled `cpg`/`kpg`; "same" padding, odd filter, stride 1; `block_groups` divides G. | (⌈Wo/block_q⌉, G/block_groups, N·⌈Hi/block_h⌉); 64·block_groups |
| `nongrouped` | G = 1; C a multiple of `ck` and of 8, K of the atom tile; every tensor below the kernel's masked-access offset; `tile_w` one of the widths a sweep of Wo tries (`tile_w_candidates`). | (⌈Wo/tile_w⌉·⌈Ho/tile_h⌉·N·⌈K/tile_k_out⌉, 1, 1); 64·waves_m·waves_n |

On top of rocKE's rules the packaged launch bounds every axis of a 3-D direct grid by
65535, as for std and spatial, and the non-grouped kernel's flat grid by a 32-bit
dispatch. The adapter's test compares the served verdict with rocKE's offer over
problems around each sampled instance (tails, images smaller than a block or the filter,
other paddings, strides, dilations and channel counts); out of tree, every one of the
10,596 identities was compared the same way with no disagreement. The only differences
are where the packaged guard is knowingly stricter: spatial at exactly 64 groups (as
before), and graphs whose two axes differ in stride or padding, which rocKE's direct
problem cannot state.

Natively, `aotDirectInstance` in `Gfx950ConvFwdNative.cpp` parses the variant and its
knobs (`aotDirectVariantUses`), and `conv::directInstanceLaunch` in
`Gfx950ConvFwdGeometry.hpp` applies the variant's rule (`depthwiseTileLaunch`,
`groupedDirectLaunch`, `nongroupedDirectLaunch`). `TestGfx950ConvFwdGeometry.cpp`
holds it to a table of (instance, problem) cases that
`tests/test_make_conv_aot_config.py` recomputes from the adapter.

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

The launch is the exact-shape engine's: `warp_m * warp_n * 64` threads, static LDS, and
the 45-argument AOT ABI computed from the graph. Before any launch, the pack checks the
packaged symbol's recorded argument names against that ABI.

Without benchmarking or a recorded winner, `aotScore` orders instances. As in the
exact-shape engine, a depthwise direct instance (`std`, `spatial`, `tiled`, `col`) ranks
above every implicit-GEMM one, the default direct arm for the graph's group count first
(spatial with one wave below 64 groups, otherwise std with `block_w` 4 and one wave).
The `grouped` and `nongrouped` kernels compete with implicit GEMM on ordinary graphs and
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
the instances that serve, the engine **selects** the ones rocKE's own runtime selection
would run, and only those are its candidates: ranked, benchmarked, or limited by
`HIPDNN_INGESTOR_CANDIDATE_LIMIT`.

The policy is the default of rocKE's `--run-from-cache` benchmark
(`_run_from_cache` in `rocke/library/benchmarks/common/benchmark_implicit_gemm_conv.py`):
`KernelCache.compatible` gives the kernels that serve, then `_widest_vector_sizes` keeps
the widest vector widths the problem admits. The catalog holds a binary per width
combination of each configuration, so a problem whose channel runs admit 8-wide access
is served by the 8-, 4-, 2- and 1-wide copies of most configurations. Narrower access
does not win, so only one combination runs:

- Kernels are grouped by kind: direction, epilogue, split-K mode, two-stage and group
  merge. Every packaged instance is forward with the default split-K, two-stage and group
  merge, so here the kind is the epilogue.
- The default epilogue stores scalars. When any serving instance stores wider than 1,
  every default-epilogue instance is dropped, since a `cshuffle` copy stores wider.
- Per kind, only the widest combination present among the serving instances is kept,
  widest being the greatest (`a + b + c`, `a`, `b`, `c`). A configuration that cannot
  reach it is dropped: its tile admits only narrower loads, or its grid does not fit.

The widest combination depends on which instances serve, not on one instance, so the
policy is a property of the candidate set. The plugin SDK lets a pack refine a
kernel-scoped matcher with a set-level stage for this: a `CandidateFilterFn`
(`NativeRegistry.hpp`), registered under the symbol of a kernel-scoped matcher the pack
already declares. Once every pack is matched, it sees every kernel that matcher admitted
for the graph, across all packs listing it (both dtype shards here), and keeps a subset.
No descriptor changes: a matcher with no filter registered keeps what it admitted.
`hipkernel:Gfx950ConvFwdAot` registers `aotCandidateFilter` under its kernel matcher's
symbol; the policy itself is `conv::widestVectorSelection` in
`Gfx950ConvFwdAotSelection.hpp`. Direct depthwise instances are all kept: rocKE's direct
benchmark (`direct_plans` in `benchmarks/common/direct_kernel_sweep.py`) runs every
direct kernel that serves, with no cross-candidate rule.

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
  theirs, in both dtypes.
- `TestGfx950ConvFwdAotSelection.cpp` runs the native kernel matcher and candidate
  filter over a sample of sweep identities and synthetic graphs.
  `tests/test_make_conv_aot_config.py` recomputes its expected rows with the mirror.
- The plugin SDK's `TestKernelIngestorStateManager.cpp` covers the filter stage itself.

## The shipped catalog

`hipkernel:Gfx950ConvFwdAot` ships rocKE's full forward sweep for gfx950 with no
selection of ours. The only limits are the sweeps' own validity checks and, for
direct kernels, which algorithms the packaged launch runs. The catalog is listed
per dtype in `projects/hipdnn/tools/IngestorGenerator/configs/gfx950_conv_fwd_aot_<dtype>.instances.json`:

| Family | fp16 | bf16 |
|---|---:|---:|
| Implicit GEMM | 78,940 | 78,940 |
| Direct depthwise, `std` | 54 | 54 |
| Direct depthwise, `spatial` | 30 | 30 |
| **Total** | **79,024** | **79,024** |

158,048 instances in all. `tools/enumerate_conv_aot_instances.py` writes the lists.

**Implicit GEMM** is every valid identity of the AOT sweep's gfx950 forward grid
(`kernel_sweep.py`): its 620 tile and warp geometries, the `mem`, `compv3` and
`compv4` pipelines, the `default` and `cshuffle` epilogues, every load and store width
it builds, and the plain, unrolled and async DMA K loops with each async chunk pair.
Every one of them converts; an identity the converter refused would fail the
enumeration rather than be dropped. Identities whose kernels compile to the same IR are
still separate instances, each with its own `instance_id`: 69,402 of the 78,940 per
dtype have distinct IR.

**Direct** is every forward identity of the direct sweep (`direct_kernel_sweep.py`)
that `make_conv_aot_config.py` accepts. Today that is the 54 `std` and 30 `spatial`
depthwise identities per dtype. The sweep's other forward direct algorithms (tiled and
column depthwise, grouped and non-grouped; 5,214 identities per dtype) have no packaged
launch yet, and the tool reports them by algorithm with the converter's reason. Once
the converter accepts them, rerunning the tool adds them with no change to the tool.

Which instances serve a given problem is decided at runtime, not by the catalog.

The lists are compact: one identity per line, holding only the fields that have no
plain default or that differ from it. `KernelIdentity.from_dict` restores the rest, and
with it the `stable_hash()`. Each list holds its dtype's implicit-GEMM instances first,
then the direct ones, each ordered by hash. `stable_hash()` covers `llvm_flavor`, so the
lists are valid for the LLVM flavor the packer lowers to.

### Shards

No committed file may reach 50 MB, the size GitHub warns about. The bundle has one
pack, and so one KDP, per dtype, implicit-GEMM pipeline and K loop, plus one per dtype
for the direct instances (`make_conv_aot_config.py --sharding k-loop`, the default):

| KDP shard, per dtype | Instances |
|---|---:|
| `gfx950_conv_fwd_aot_<dtype>_mem_plain`, `_mem_unroll` | 11,130 each |
| `gfx950_conv_fwd_aot_<dtype>_mem_async` | 12,160 |
| `gfx950_conv_fwd_aot_<dtype>_compv3_plain`, `_compv3_unroll` | 11,130 each |
| `gfx950_conv_fwd_aot_<dtype>_compv4_plain`, `_compv4_unroll` | 11,130 each |
| `gfx950_conv_fwd_aot_<dtype>_direct` | 84 |

The packs declare no `discriminator`, so they are shards of one catalog: the engine's
graph matcher and the one shared kernel matcher admit all of them, and there is no
operation-scoped matcher (the IngestorGenerator README, "UMD policy"). Each authored
implicit-GEMM shard is about 19 to 20 MB, and each instance list about 31 MB. The census
`TestGfx950ConvFwdAotPacks.cpp` lists every pack and kernel name exactly, at about
12 MB. The test-only seed keeps one pack per dtype (`--sharding dtype`).

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
rm "$DEST"/*.json && cp "$GEN"/descriptors/rocKE/gfx950_conv_fwd_aot/*.json "$DEST"/
cp "$GEN"/tests/TestGfx950ConvFwdAotPacks.cpp \
   ../../../../dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/
python tools/make_conv_aot_config.py --stabilize-ids "$DEST"/*.kdp.json
```

Then run `clang-format` on the census. Keep the document ids of the KMD, UED, UDD, UHD,
UMD and KDP files from the previous bundle, so that the diff shows only the kernels.
The generator mints new ones on every run. The whole flow takes a few minutes.

How the converter treats the lists:

- **Unsupported entries:** the tool lists each with its reason and writes nothing.
  `--skip-unsupported` drops them instead.
- **Duplicates:** entries with the same `instance_id` collapse to one.
- **Stable kernel IDs:** `--stabilize-ids` re-derives each kernel's UUID as
  `uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`. The generator mints random UUIDs, and
  a recorded autotune winner is keyed by kernel UUID, so this keeps records valid
  across regenerations. Another engine name (`--engine-name`) derives its ids under a
  namespace of its own name.

Commit the instance lists. The generator config is not committed: the converter
regenerates it from the lists on every run. The tests hold the pieces together:

- `tests/test_enumerate_conv_aot_instances.py` fails when the lists are not what the
  tool enumerates from the rocKE tree beside it, or when the implicit-GEMM catalog is
  not exactly what `kernel_sweep` enumerates.
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

The product pack compiles every instance, about 159,000 variants with the other
product bundles. On one host with 96 pack workers
(`-DHIPKERNELPROVIDER_PRODUCT_PACK_JOBS=96`):

| Artifact | Size |
|---|---:|
| Instance list, each dtype (committed) | about 31 MB |
| Authored implicit-GEMM KDP shard, each (committed) | about 19 to 20 MB |
| Authored direct KDP shard, each (committed) | about 0.2 MB |
| Census `TestGfx950ConvFwdAotPacks.cpp` (committed) | about 12 MB |
| gfx950 kpack archive (build output, all product bundles) | about 1.3 GB |
| Packed implicit-GEMM KDP shard, each (build output) | about 40 to 44 MB |
| Provenance sidecar, each implicit-GEMM shard (build output) | about 4.2 to 4.6 MB, 110 to 120 MB inflated |

Packing all product bundles takes about 7 minutes with a warm comgr cache and about
23 minutes with an empty one; the comgr cache grows to about 14 GB. The packer holds
every code object in memory before it writes the archive, so it peaks at about 16 GB
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
- **Plugin load.** Loading the product root parses every packed KDP, about 570 MB of
  JSON for this engine, once per process.

### Selecting among thousands of candidates

At full size, a dense graph is served by every implicit-GEMM instance of its dtype; the
runtime selection ([Serves and selected](#serves-and-selected)) keeps one width
combination per configuration and epilogue. An unbenchmarked plan takes the best by score.
Benchmarking loads and measures every selected candidate, so a cold plan build still
grows with the number of configurations. `HIPDNN_INGESTOR_CANDIDATE_LIMIT=<n>`
(`GlobalKnobDefines.hpp`) keeps only the `n` best-ranked candidates after knob
filtering, before the winner-record lookup, with benchmarking on or off. A record
benchmarked under a limit therefore covers exactly the candidates a later build under
the same limit walks. Unset, or any value that is not a positive integer, means no
limit.

## The seed engine and the GPU tests

`hipkernel:Gfx950ConvFwdAotSeed` is a test-only engine over a 208-instance seed, taken
verbatim from the sweeps' valid forward grids
(`configs/gfx950_conv_fwd_aot_seed.instances.json`). It has 12 implicit-GEMM instances
per dtype, covering:

- every load width and every store width the catalog ships, in the pairs (1, 1),
  (2, 2), (4, 4), (8, 8), (1, 4), (4, 1), 8/8/4, 4/4/2 and a `cshuffle` 2/2/1;
- both epilogues, the default one with a store width of 1;
- the `mem`, `compv3` and `compv4` pipelines;
- `unroll_k` and `async_dma`.

`tests/test_make_conv_aot_config.py` checks that the seed holds every pipeline,
epilogue, K loop, load width and store width of the shipped catalog, in each dtype. The
catalog's other width pairs and async chunk pairs run the same launch with other
arguments; rocKE's packaging test checks the IR of every one of those classes.

It also has 92 direct instances per dtype. 4 are std and spatial: std 3x3 stride 1
(`block_w` 4, one wave), std 5x5 stride 2 (`block_w` 16, two waves), spatial 3x3
stride 1 (one wave) and spatial 7x7 stride 2 (four waves). The other 88 are 22 each of
`tiled`, `col`, `grouped` and `nongrouped`, chosen deterministically from the sweep so
that together they hold every value of every knob the sweep builds for the variant,
every grouped channel count (4, 8, 16, 32) and every non-grouped dilation, and so that
each serves at least two of the seed's graphs. `tests/test_make_conv_aot_config.py`
checks the knob coverage and the counts.

The seed is authored under
`src/engines/kernel_ingestor_engine/test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed`
and packed into the integration test root. Its descriptors name the native symbols of
`hipkernel:Gfx950ConvFwdAot` (the generator's `engine.native_symbols_from`), so it runs
the shipped code and registers nothing of its own. It needs a name of its own because
the integration binary loads the product root as well. The same tools generate it,
with the converter's `--engine-name hipkernel:Gfx950ConvFwdAotSeed`. Its config,
`configs/gfx950_conv_fwd_aot_seed.yaml`, is committed as a worked example. Its census,
`TestGfx950ConvFwdAotSeedPacks`, reads the integration root's gfx950 shard.

`IntegrationGpuGfx950ConvFwd.cpp` runs one table of 24 graphs per dtype against both
engines, 9 of them depthwise over several group counts, batches and images. The seed
also runs a second table, `aotSeedDirectGraphs()`, of 56 graphs for its tiled, col,
grouped and non-grouped instances: odd images, batches and group counts with tails,
images narrower than a block, "valid" and "same" padding for col, filters up to 31x31
for tiled, every grouped channel count, non-grouped output widths that each swept tile
width divides, stride 2 and every swept dilation.

- **`IntegrationGpuGfx950ConvFwdAotSeed`** benchmarks every selected instance, then pins
  each one in turn and checks it against the CPU reference. It also forces each family
  through the `kernel_family` knob, and checks that an unbenchmarked plan takes a
  depthwise direct instance on a depthwise graph and implicit GEMM elsewhere. Five graphs
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
per dtype, since each dtype's list is rocKE's sweep for that dtype.
`tests/test_make_conv_aot_config.py` recomputes the counts with
`gfx950_conv_fwd_aot_runtime_candidates` over the committed lists, checks that they never
exceed what `gfx950_conv_fwd_aot_serves` counts, and checks that every seed instance is
selected for some graph. The integration
CMake compiles each engine's suite only where that engine ships: the seed's where the
integration root packs it for gfx950, the product's where the product root offers it.

## Current limits

- Forward only. The launch runs every forward direct kernel of rocKE's sweep, but the
  shipped catalog packages the std and spatial ones only; the others are exercised by
  the seed.
- One score heuristic, as described above. It is a placeholder until rocKE's
  dispatcher ranks instances. Benchmarking every selected instance grows with the
  number of configurations in the catalog; `HIPDNN_INGESTOR_CANDIDATE_LIMIT` bounds it
  to the best-ranked ones, which makes the quality of the score matter more.
