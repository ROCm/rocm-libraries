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

## Packaging a list of instances (step 4)

The input is a JSON list of `KernelIdentity.to_dict()` entries, or objects carrying one
under `"identity"`. `stable_hash()` covers `llvm_flavor`, so generate the list with the
flavor the packer lowers to. From `projects/hipdnn/tools/IngestorGenerator`, with
`ROCKE=<repo>/dnn-providers/hip-kernel-provider/rocke`:

```bash
python tools/make_conv_aot_config.py --rocke-root "$ROCKE" \
    --instances configs/gfx950_conv_fwd_aot.instances.json \
    --output configs/gfx950_conv_fwd_aot.yaml
python generate.py --config configs/gfx950_conv_fwd_aot.yaml --output-dir "$GEN" --force
cp "$GEN"/descriptors/rocKE/gfx950_conv_fwd_aot/*.json \
   ../../../../dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_conv_fwd_aot/
cp "$GEN"/tests/TestGfx950ConvFwdAotPacks.cpp \
   ../../../../dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/
python tools/make_conv_aot_config.py --stabilize-ids \
   ../../../../dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx950_conv_fwd_aot/gfx950_conv_fwd_aot.kdp.json
```

How the tool treats the list:

- **Unsupported entries:** the tool lists each with its reason and writes nothing.
  `--skip-unsupported` drops them instead.
- **Duplicates:** entries with the same `instance_id` collapse to one.
- **Stable kernel IDs:** `--stabilize-ids` re-derives each kernel's UUID as
  `uuid5(INSTANCE_UUID_NAMESPACE, instance_id)`. The generator mints random UUIDs, and
  a recorded autotune winner is keyed by kernel UUID, so this keeps records valid
  across regenerations.

Commit the instance list next to the config. `tests/test_make_conv_aot_config.py`
fails when the shipped bundle is not what the tool generates from it.

Then rebuild with the pack stamps removed, so every instance is compiled from the
current sources.

## Seed catalog and its tests

The shipped `configs/gfx950_conv_fwd_aot.instances.json` is a 24-instance seed, taken
verbatim from the sweeps' valid forward grids. It has 8 implicit-GEMM instances per
dtype, covering:

- vector widths 8/8/8, 8/8/4, 4/4/2, 2/2/1 and 1/1/1;
- both epilogues;
- the `mem`, `compv3` and `compv4` pipelines;
- `unroll_k` and `async_dma`.

It also has 4 direct instances per dtype: std 3x3 stride 1 (`block_w` 4, one wave),
std 5x5 stride 2 (`block_w` 16, two waves), spatial 3x3 stride 1 (one wave) and
spatial 7x7 stride 2 (four waves).

`IntegrationGpuGfx950ConvFwdAot` runs 19 graphs per dtype on the GPU, 9 of them
depthwise over several group counts, batches and images. For each graph it benchmarks
every serving instance, then pins each one in turn and checks it against the CPU
reference. It also forces each family through the `kernel_family` knob. The expected
number of serving instances of each family per graph is part of each test variant,
taken from `gfx950_conv_fwd_aot_serves`.

**When the seed is replaced by a production list:** those counts and the census test
are tied to the seed.

- The census file is regenerated with the bundle, as shown above.
- For the GPU test, either move the seed to a test-only descriptor root, or recompute
  the counts.
- Pinning every serving instance of a ~1000-instance catalog on every graph is also
  slow, so the production suite should sample instances per graph instead.

## Current limits

- Forward implicit GEMM, and direct depthwise for the std and spatial variants only.
  rocKE's other direct kernels (tiled and column depthwise, grouped, non-grouped) need
  launches this pack does not have, so the tool refuses them.
- One score heuristic, as described above. Plan build benchmarks every serving
  instance, which grows with the catalog, so a per-graph shortlist (for example the
  top N by rocKE's dispatcher) is the expected next step.
