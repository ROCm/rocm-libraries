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
- direct depthwise, from the direct sweep (`benchmarks/common/direct_kernel_sweep.py`):
  see [Direct depthwise instances](#direct-depthwise-instances).

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

## Direct depthwise instances

The exact-shape engine's direct launch runs exactly two of the direct sweep's forward
algorithms unchanged: `direct_depthwise` (the std variant, `DirectDepthwiseSpec`) and
`direct_depthwise_spatial` (`DirectDepthwiseSpatialSpec`). The sweep holds 54 std and
30 spatial identities per dtype; the tool packages all of them and refuses every other
direct algorithm with its reason.

A direct binary bakes the filter, stride, padding and dilation, plus the variant and
its block knobs. Batch, image extents and the group count are kernel arguments: the
6 leading arguments plus the 14-argument direct block (`directKernargs`). A direct
instance carries these trailing fields of `Gfx950ConvFwdAotSpec`:

| Field | Meaning |
|---|---|
| `kernel_family` | 1 for direct depthwise; 0 (the default) for implicit GEMM. |
| `direct_variant` | `std` or `spatial`; `none` for implicit GEMM. |
| `filter_h/w`, `stride_h/w`, `pad_h/w`, `dilation_h/w` | The compiled values. The kernels compile one stride and one padding and no dilation, and the sweep builds square, "same"-padded, odd filters only. |
| `block_w`, `block_waves` | Output columns and waves per block. Spatial derives its width from the runtime group count, so its `block_w` is 0. |

Its implicit-GEMM fields hold exactly what the direct sweep writes into a direct
identity: 0 for the tile, warp, vector, flag and chunk fields, and `direct` for the
pipeline and epilogue. A forced `tile_k` of 64 or 128 therefore never selects one.

All the direct fields are `builder_default` in the KMD, with the dataclass defaults
(0, or `none`). Implicit-GEMM descriptors omit them, so their specs, symbols and
pack keys are as before. Direct instances are a second generator variant that
states them, named `gfx950_conv_fwd_aot_dw_<dtype>_<instance_id>`, with kernel
symbols prefixed `hkp_conv_fwd_dw_aot_gfx950_`. The probe problem mirrors
`direct_kernel_sweep.probe_problem` (64x64, one image, 256 groups, or 8 for spatial).
`library/tests/test_convolution_forward_aot_packaging.py` checks the probe and the IR,
byte for byte up to the symbol, for every one of the 168 identities.

A direct instance serves a graph by the exact-shape engine's direct rule
(`conv::directInstanceLaunch`, `_direct_serves` in the adapter):

- the direct guard `directProblemSupported`: pure depthwise (groups == C == K), equal
  strides and paddings, no dilation, an odd filter with "same" padding, and the input
  stream covers every output row and column;
- its compiled filter, stride, padding and dilation are the graph's (`directServes`
  over `DirectCapabilities`);
- `directLaunchGeometry`: spatial only below 64 groups, and every grid axis fits
  65535 (z carries N).

The adapter's test compares this with what rocKE's direct benchmark offers
(`KernelCache.supports_problem`, then the per-shape validator `plan_for` runs). The
only difference is spatial at exactly 64 groups, which rocKE's validator admits, its
benchmark never runs, and the guard declines.

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
exact-shape engine, a direct instance ranks above every implicit-GEMM one, the default
direct arm for the graph's group count first (spatial with one wave below 64 groups,
otherwise std with `block_w` 4 and one wave). Among implicit-GEMM instances,
`aotInstanceScore` prefers less tile padding and a grid that fills the device, with
wider vectors breaking ties. It is a placeholder until rocKE's dispatcher ranks
instances. With benchmarking, hipDNN measures every instance that serves the graph and
records the fastest. The `kernel_family` knob forces one family, as on the exact-shape
engine.

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

The catalog is not narrowed for any problem: at full size, a dense graph is served by
every implicit-GEMM instance of its dtype. Which of them a problem considers is decided
at runtime. An unbenchmarked plan takes the best by score. Benchmarking loads and
measures every candidate, so a cold plan build grows with the catalog.
`HIPDNN_INGESTOR_CANDIDATE_LIMIT=<n>` (`GlobalKnobDefines.hpp`) keeps only the `n`
best-ranked candidates after knob filtering, before the winner-record lookup, with
benchmarking on or off. A record benchmarked under a limit therefore covers exactly the
candidates a later build under the same limit walks. Unset, or any value that is not a
positive integer, means no limit.

## The seed engine and the GPU tests

`hipkernel:Gfx950ConvFwdAotSeed` is a test-only engine over a 32-instance seed, taken
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

It also has 4 direct instances per dtype: std 3x3 stride 1 (`block_w` 4, one wave),
std 5x5 stride 2 (`block_w` 16, two waves), spatial 3x3 stride 1 (one wave) and
spatial 7x7 stride 2 (four waves).

The seed is authored under
`src/engines/kernel_ingestor_engine/test_descriptors/integration/rocKE/gfx950_conv_fwd_aot_seed`
and packed into the integration test root. Its descriptors name the native symbols of
`hipkernel:Gfx950ConvFwdAot` (the generator's `engine.native_symbols_from`), so it runs
the shipped code and registers nothing of its own. It needs a name of its own because
the integration binary loads the product root as well. The same tools generate it,
with the converter's `--engine-name hipkernel:Gfx950ConvFwdAotSeed`. Its config,
`configs/gfx950_conv_fwd_aot_seed.yaml`, is committed as a worked example. Its census,
`TestGfx950ConvFwdAotSeedPacks`, reads the integration root's gfx950 shard.

`IntegrationGpuGfx950ConvFwd.cpp` runs one table of 19 graphs per dtype against both
engines, 9 of them depthwise over several group counts, batches and images:

- **`IntegrationGpuGfx950ConvFwdAotSeed`** benchmarks every serving instance, then pins
  each one in turn and checks it against the CPU reference. It also forces each family
  through the `kernel_family` knob.
- **`IntegrationGpuGfx950ConvFwdAot`** runs the shipped catalog. An unbenchmarked plan
  is executed and checked. The serving count of each family is checked through the
  `kernel_family` knob. The 4 best instances by score are pinned and checked. So are 8
  more, drawn by a seeded shuffle of every kernel in the dtype's shards: a record that
  orders all of them covers every candidate, so it serves the first instance in its
  order that serves the graph. A benchmark under `HIPDNN_INGESTOR_CANDIDATE_LIMIT=64`
  is checked, and so is its reuse.

Each row of the table states how many instances of each family serve the graph, in the
seed and in the full catalog, with the full catalog's implicit-GEMM count per dtype,
since each dtype's list is rocKE's sweep for that dtype.
`tests/test_make_conv_aot_config.py` recomputes the counts with
`gfx950_conv_fwd_aot_serves` over the committed lists. The integration
CMake compiles each engine's suite only where that engine ships: the seed's where the
integration root packs it for gfx950, the product's where the product root offers it.

## Current limits

- Forward implicit GEMM, and direct depthwise for the std and spatial variants only.
  rocKE's other direct kernels (tiled and column depthwise, grouped, non-grouped) need
  launches this pack does not have, so the converter refuses them and the enumeration
  leaves them out, reporting each algorithm with its reason.
- One score heuristic, as described above. It is a placeholder until rocKE's
  dispatcher ranks instances. Benchmarking every serving instance grows with the
  catalog; `HIPDNN_INGESTOR_CANDIDATE_LIMIT` bounds it to the best-ranked ones, which
  makes the quality of the score matter more.
