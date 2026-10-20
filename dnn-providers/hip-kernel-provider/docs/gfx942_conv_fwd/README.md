# Shape-generic AOT instances for gfx942: hipkernel:Gfx942ConvFwdAot

`hipkernel:Gfx942ConvFwdAot` is the gfx942 counterpart of `hipkernel:Gfx950ConvFwdAot`:
rocKE's forward-convolution AOT kernels, packaged as **instances** that are matched to a
graph by capability rather than by exact shape. Read
[the gfx950 AOT instances document](../gfx950_conv_fwd/aot_instances.md) first. This
page describes only what differs on gfx942. The graph contract, the instance fields, the
direct variants and the launch ABI are the same.

## What is shared and what is not

rocKE's forward kernels take their MFMA atoms, LDS capacity and async load width from the
target, so a gfx942 instance is the same `KernelIdentity` built for gfx942. The adapter
(`rocke/library/builders/common/convolution_forward.py`) has these gfx942 entry points:

| Role | gfx942 | gfx950 |
|---|---|---|
| Spec class | `Gfx942ConvFwdAotSpec` (a `Gfx950ConvFwdAotSpec` with `ARCH = "gfx942"`) | `Gfx950ConvFwdAotSpec` |
| Builder | `build_gfx942_conv_fwd_aot` | `build_gfx950_conv_fwd_aot` |
| Instance predicate | `supports_gfx942_conv_fwd_aot` | `supports_gfx950_conv_fwd_aot` |
| Graph contract | `gfx942_conv_fwd_aot_problem_for_request`, `supports_gfx942_conv_fwd_aot_problem` | the `gfx950_` ones |
| Served problems | `gfx950_conv_fwd_aot_serves` (shared) | the same |
| Runtime selection | `gfx942_conv_fwd_aot_runtime_candidates` | `gfx950_conv_fwd_aot_runtime_candidates` |

Kernel symbols name the architecture (`hkp_conv_fwd_aot_gfx942_…`,
`hkp_conv_fwd_dw_aot_gfx942_…`).

The native code is the gfx950 AOT engine's (`Gfx950ConvFwdNative.cpp`,
`Gfx950ConvFwdGeometry.hpp`, `Gfx950ConvFwdAotSelection.hpp`). It runs over a
`ConvFwdTarget` (`GFX942_TARGET`) that sets these values:

- the device arch, `gfx942`;
- the graph match's bound tokens, `gfx942_conv_fwd.{x,w,y}.uid` and
  `_gfx942_conv_fwd.problem`. Public tokens are UHD query features, so each engine names
  its own;
- the selection policy, described below;
- the device size the unbenchmarked fallback score assumes, 304 CUs (MI300X).

The engine registers its own symbols, `hipkernel.gfx942_conv_fwd_aot.{graph_match,
kernel_match,score,dispatch}`. The tool profile is
`projects/hipdnn/tools/IngestorGenerator/configs/gfx942_conv_fwd_aot.profile.yaml`.

## The catalog

The catalog is a curated list of rocKE sweep kernels:
`projects/hipdnn/tools/IngestorGenerator/configs/gfx942_conv_fwd_aot.labels.txt`, one
`KernelIdentity.label()` per line, with the same kernels in fp16 and bf16.
`tools/enumerate_conv_aot_instances.py --arch gfx942` checks every label against rocKE's
gfx942 forward sweeps (`kernel_sweep.py` for implicit GEMM, `direct_kernel_sweep.py` for
direct) and writes `configs/gfx942_conv_fwd_aot_<dtype>.instances.json`:

| Family | Per dtype |
|---|---:|
| Implicit GEMM | 372 |
| of which group-merged (`group_merge` 4, 16, 32: 7, 11, 12 instances) | 30 |
| Direct `nongrouped` | 98 |
| Direct depthwise `tiled` | 64 |
| Direct `grouped` | 11 |
| Direct depthwise `std` / `spatial` | 4 / 3 |
| Direct depthwise `col` | 1 |
| **Total** | **553** |

Both dtypes together hold 1,106 instances, 60 of them group-merged. The implicit-GEMM
instances use the gfx942 MFMA atoms 16x16x16 (301 per dtype) and 32x32x8 (71). None of
them uses the async DMA K loop. The lists are for LLVM flavor `llvm22`, which every
identity's `stable_hash()` covers.

The bundle is `src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx942_conv_fwd_aot`.
As on gfx950, it has one KDP per dtype, implicit-GEMM pipeline and K loop, and one per
dtype for the direct instances (`--sharding k-loop`). Merged instances sit in the shard
of their pipeline and K loop:

| KDP shard, per dtype | Instances | of which merged |
|---|---:|---:|
| `gfx942_conv_fwd_aot_<dtype>_mem_plain` | 141 | 12 |
| `gfx942_conv_fwd_aot_<dtype>_mem_unroll` | 32 | 2 |
| `gfx942_conv_fwd_aot_<dtype>_compv3_plain` | 80 | 8 |
| `gfx942_conv_fwd_aot_<dtype>_compv3_unroll` | 14 | 0 |
| `gfx942_conv_fwd_aot_<dtype>_compv4_plain` | 83 | 7 |
| `gfx942_conv_fwd_aot_<dtype>_compv4_unroll` | 22 | 1 |
| `gfx942_conv_fwd_aot_<dtype>_direct` | 181 | 0 |

The census `TestGfx942ConvFwdAotPacks.cpp` lists every pack and kernel name. It is
registered where the product ships the engine for gfx942.

## Group merge

An implicit-GEMM identity whose `group_merge` (Gm) is greater than 1 is a depthwise kernel
in which one workgroup computes Gm groups. `make_conv_aot_config.py` packages such
identities for gfx942 only (`TARGETS["gfx942"]["group_merge"]`). For gfx950, `group_merge`
must hold its default of 1, like every other identity field the launch does not model.

- **Build.** The probe is `kernel_sweep`'s, with 64 depthwise groups. The builder derives
  the vector widths from the merged problem, so the identity records them as 0, and the
  adapter passes them to rocKE as `None`. A merged instance must be built on the grouped
  code path. The native launch also refuses a merged instance with async DMA; the catalog
  holds none.
- **Serves** (`_aot_merged_serves`; natively `conv::mergedInstanceLaunch`). These are
  rocKE's `KernelCache._group_merge_fits` rules for forward. The graph must be depthwise
  (C/groups = K/groups = 1), Gm must divide the group count, and the merged GEMM N,
  K/groups · Gm, must fit in `tile_n`. Every grid axis must fit 65535, and the reduction
  Y·X·C/groups must stay below 2^23.
- **Launch.** The grid is the merged one: x tiles the merged GEMM N (one tile, under the
  bound above), y tiles GEMM M, and z = groups / Gm. The kernel arguments are the true
  problem's, as in rocKE's benchmark.
- **Descriptors.** The KMD has `group_merge` as an int field with `default_value` 1 and
  `builder_default`, so an unmerged descriptor omits it. The engine exposes it as a knob,
  and it is metadata. Merged instances form their own generator variant,
  `gfx942_conv_fwd_aot_gm_<dtype>_<instance_id>`, whose spec order is the implicit-GEMM
  one followed by `group_merge`. `group_merge` is part of the identity's `stable_hash()`,
  so a merged instance and the unmerged instance of the same tile have different
  `instance_id`s, kernel ids and symbols.

## Selection policy

As on gfx950, an instance **serves** a graph when its binary computes it. The engine
**selects** some of the serving instances, and only those are ranked, benchmarked or
limited by `HIPDNN_INGESTOR_CANDIDATE_LIMIT`. The two engines select differently:

- **gfx950** (`conv::widestVectorSelection`): the problem's widest combination is the
  widest load width C/groups admits and the widest store width K/groups admits. If some
  serving implicit-GEMM instance bakes exactly that combination, only those run, of
  either epilogue. Otherwise every serving instance runs.
- **gfx942** (`conv::widestPerKindSelection`, `gfx942_conv_fwd_aot_runtime_candidates`):
  rocKE's current `--run-from-cache` policy, `_widest_vector_sizes` in
  `benchmarks/common/benchmark_implicit_gemm_conv.py`. The gfx942 catalog was curated
  against this policy. Serving implicit-GEMM instances are grouped by kind, that is by
  epilogue and merge degree. Default-epilogue instances are dropped where a serving
  instance of the same merge degree stores wider than 1. Of each kind, only the
  instances with the greatest (a + b + c, a, b, c) run. Merged instances record their
  widths as 0, so every serving merged instance of a kind runs.

On both engines every serving direct instance runs. `aotCandidateFilter`, registered
under each engine's kernel matcher, applies the target's policy.

## Regenerating the bundle

From `projects/hipdnn/tools/IngestorGenerator`, set `ROCKE` to
`<repo>/dnn-providers/hip-kernel-provider/rocke`, `ROCM_PATH=/opt/rocm`,
`ROCKE_BACKEND=python` and `PYTHONPATH=$ROCKE/platform/python:$ROCKE/library`. The
enumeration hashes identities under the process's LLVM flavor, so pin it to the lists'
flavor:

```bash
ROCKE_LLVM_FLAVOR=llvm22 python tools/enumerate_conv_aot_instances.py \
    --rocke-root "$ROCKE" --arch gfx942 --output-dir configs
python tools/make_conv_aot_config.py --rocke-root "$ROCKE" \
    --instances configs/gfx942_conv_fwd_aot_{fp16,bf16}.instances.json \
    --output "$WORK"/aot.yaml
python generate.py --config "$WORK"/aot.yaml --output-dir "$GEN" --force
DEST=../../../../dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/gfx942_conv_fwd_aot
rm "$DEST"/*.json && cp "$GEN"/descriptors/rocKE/gfx942_conv_fwd_aot/*.json "$DEST"/
cp "$GEN"/tests/TestGfx942ConvFwdAotPacks.cpp \
   ../../../../dnn-providers/hip-kernel-provider/src/tests/engines/kernel_ingestor_engine/packs/
python tools/make_conv_aot_config.py --stabilize-ids "$DEST"/*.kdp.json \
    --engine-name hipkernel:Gfx942ConvFwdAot
```

`make_conv_aot_config.py` takes the engine, builder and variant prefix from the
identities' architecture. `--stabilize-ids` needs
`--engine-name hipkernel:Gfx942ConvFwdAot`. Without it the tool derives the ids of
`hipkernel:Gfx950ConvFwdAot`. The gfx942 engine's kernel ids are uuid5 of the
`instance_id` under a namespace derived from its own name, so they never repeat a gfx950
kernel id in one process.

Then run `clang-format` on the census. As on gfx950, keep the previous bundle's document
ids (KMD, UED, UDD, UHD, UMD, KDP). Also keep the UED's `sort_kernel_catalog` default
once a trained model has been promoted, along with the `heuristics/` directory beside the
descriptors.

To change the catalog, edit the labels file and rerun the flow. Commit the labels, the
instance lists, the descriptors and the census. These tests hold the pieces together:

- `tests/test_enumerate_conv_aot_instances.py` checks the following: the gfx942 lists are
  what the tool writes from the labels; every label is a gfx942 forward sweep identity
  that converts to a `Gfx942ConvFwdAotSpec`; the catalog has the counts above, the same in
  both dtypes. The first two checks run only under the lists' LLVM flavor and are skipped
  otherwise.
- `tests/test_make_conv_aot_config.py` checks these points:
  - The shipped bundle is what the flow above regenerates, byte for byte up to document
    ids.
  - Every shard holds exactly its instances with instance-derived ids.
  - The config carries `group_merge` as a KMD field, knob and metadata, and merged
    instances form the `_gm_` variant.
  - Mixing gfx950 and gfx942 identities in one catalog is refused, and so are gfx942
    identities that set stream-K, `prefetch_addr` or `uniform_k_decode`.
  - The profile passes `launch_surface.py --check`.
  - The GPU suite's counts are what `gfx942_conv_fwd_aot_runtime_candidates` selects.

## GPU tests

`IntegrationGpuGfx942ConvFwdAot`, in
`src/integration_tests/kernel_ingestor_engine/IntegrationGpuGfx950ConvFwd.cpp`, is
compiled under `HKP_TEST_GFX942_CONV_FWD_AOT` where the product offers the engine. It
skips on a device that is not gfx942. `gfx942AotGraphs()` holds 24 graphs, each run in
fp16 and bf16:

- dense graphs whose channel counts pick each width combination;
- grouped and depthwise graphs, including those that select the merged instances;
- graphs that cover the direct kernels' filters, strides, paddings, dilations and output
  widths.

Each row states how many direct and implicit-GEMM instances the engine selects. The
counts are the same in both dtypes. For each graph the suite benchmarks the selected
instances and checks their count. It then serves every selected instance pinned and
compares it with the CPU reference. It also executes and checks an unbenchmarked plan.
Together the graphs select every instance of the catalog in both dtypes, so every
shipped instance is checked on the GPU. `tests/test_make_conv_aot_config.py` recomputes
the counts and that coverage from the committed lists.

## Heuristic

The UED's `sort_kernel_catalog` default is the generated native selector
(`gfx942_conv_fwd_aot.uhd.json`, `hipkernel.gfx942_conv_fwd_aot.score`), which orders
instances by the fallback score. The UHD pipeline trains the learned
`sort_kernel_catalog` ranker for this engine on the `gfx942-mi300x` SKU. A promoted model
is applied to this bundle as on gfx950: model files under `heuristics/`, and the model
listed after the native selector in the UED's `sort_kernel_catalog` default.
