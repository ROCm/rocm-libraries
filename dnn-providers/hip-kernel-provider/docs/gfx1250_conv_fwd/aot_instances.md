# Shape-generic AOT instances: hipkernel:Gfx1250ConvFwdAot

Like [gfx950's](../gfx950_conv_fwd/aot_instances.md), every packaged kernel is a
shape-generic AOT **instance**: one rocKE `KernelIdentity`
(`rocke/library/kernels/conv/kernel_identity.py`), built once from the job rocKE's own
AOT build builds it from, and matched to a graph by capability rather than by exact
shape. This page covers what differs on gfx1250.

## The catalog

The catalog is rocKE's gfx1250 forward dispatcher list,
`rocke/library/dispatch/grouped_convolution_instances/gfx1250_fwd.json`. Its order is
the dispatcher's priority, fitted to measured gfx1250 timings. The engine ships all of
it, 263 instances per dtype, the same in fp16 and bf16, all wave32 and built with the
gfx1250 default LLVM flavor:

| `kernel_family` / algorithm | `direct_variant` | Per dtype |
|---|---|---:|
| `direct_fwd_wmma` | `wmma` | 112 |
| `direct_dw_fwd_wmma` | `dw_wmma` | 37 |
| `direct_fwd_smallc_wmma` | `smallc_wmma` | 30 |
| `direct_fwd_gstream_wmma` | `gstream_wmma` | 26 |
| `direct_depthwise_tiled` | `tiled` | 22 |
| `direct_fwd_tapouter_wmma` | `tapouter_wmma` | 14 |
| `direct_depthwise` | `std` | 4 |
| `direct_fwd_smallk_wmma` | `smallk_wmma` | 4 |
| `direct_depthwise_col` | `col` | 1 |
| `direct_depthwise_spatial` | `spatial` | 1 |
| `implicit_gemm` (`mem` and `wavelet` pipelines) | | 12 |

Winograd instances are not packaged.

## Instance fields

`Gfx1250ConvFwdAotSpec` is gfx950's `ConvFwdAotSpec` plus the knobs of the WMMA direct
pipelines (`GFX1250_DIRECT_KNOBS`): `ck`, `stages`, `tile_w`, `tiles_k/p/c`,
`waves_k/p/c`, `prefetch`, `pack_groups`, `pp_group`, `sched`, `walk`, `pf_depth`,
`tdm_split`, `out_h/w`, `tiles_per_block`, the ring, tap and view switches, and so on.
rocKE's grid writes a knob only where it is off its default, so every gfx1250 knob is
optional and the KMD encodes its absence:

- an int knob is 0 when absent (every rocKE int knob of these pipelines is positive);
- a bool knob is 0 when absent, 1 for true and 2 for false;
- `sched` is `none` when absent.

The WMMA pipelines' `tile_n` (output-channel tiles per wave) is carried in
`tile_n_out`, apart from the implicit-GEMM `tile_n`. `direct_variant` and
`kernel_family` are engine knobs, as on gfx950.

## Which graphs an instance serves

An instance serves a graph when rocKE's grouped-convolution dispatcher admits the
graph for it (`_admits`) and its launch plan (`launch_plan_for`) fits the packaged
contract: one dispatch, grid y and z at most 65535, threads per block within the
device limit. `conv_fwd_aot_serves` in the adapter states this, and
`gfx1250_conv_fwd::servedPlan` in `packs/Gfx1250ConvFwdGeometry.hpp` is its C++ port:
each catalog kernel is parsed once from its KMD fields (`CatalogKernel::parse`), and
the port answers admission and launch from them. The launch goes through the shared
`ConvLaunch` and kernel-argument code of `ConvFwdAotNative.cpp`.

Unlike gfx950, there is no exact-shape direct rule: the WMMA kernels compile per-axis
geometry (rectangular filters, their own column padding, per-axis dilation), and every
direct instance serves what the dispatcher admits. The implicit-GEMM port folds width
like rocKE's dispatcher (`fold_width`). gfx1250 ships no split-C kernel.

`TestGfx1250ConvFwdAotGeometry` holds the port to golden vectors from the adapter
(`Gfx1250ConvFwdAotTestVectors.hpp`, written by `make_conv_aot_test_vectors.py`): for
every instance, its admission count and a digest of every launch plan over the
recorded problems.

## Selecting among candidates

The engine shares gfx950's candidate filter (`conv::widestVectorSelection` among
implicit-GEMM instances; direct instances are all kept). Its score is the catalog
priority, falling with list position, so without a model an unbenchmarked plan takes
the first serving instance in the dispatcher's order. The UED's default
`sort_kernel_catalog` model (`heuristics/`, in DVC) ranks the candidates when it is
bound; `TestGfx1250ConvFwdAotModelBinding` checks that it loads and scores every
shipped kernel.

## Regenerating the bundle

From `projects/hipdnn/tools/IngestorGenerator`, with the environment of gfx950's
[Regenerating the bundle](../gfx950_conv_fwd/aot_instances.md#regenerating-the-bundle)
and `ROCKE_LLVM_FLAVOR` set to the gfx1250 lists' flavor if comgr resolves to another:

```bash
python tools/enumerate_conv_aot_instances.py --arch gfx1250 --rocke-root "$ROCKE" --output-dir configs
python tools/make_conv_aot_config.py --arch gfx1250 --rocke-root "$ROCKE" \
    --instances configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json \
    --output "$WORK"/aot.yaml
python generate.py --config "$WORK"/aot.yaml --output-dir "$GEN" --force
# copy descriptors and KDPs into descriptors/rocKE/gfx1250_conv_fwd_aot/ and the
# census TestGfx1250ConvFwdAotPacks.cpp, as for gfx950, then:
python tools/make_conv_aot_config.py --stabilize-ids "$DEST"/artifacts/*.kdp.json
(cd "$DEST" && dvc add artifacts)
python tools/make_conv_aot_test_vectors.py --arch gfx1250 --rocke-root "$ROCKE" \
    --instances configs/gfx1250_conv_fwd_aot_{fp16,bf16}.instances.json \
    --header <provider>/src/tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestVectors.hpp \
    --problems-from-header <the same header>
```

The catalog is split by K loop (`--sharding k-loop`, the default) into 6 KDPs:
`direct`, `mem_plain` and `wavelet_plain`, per dtype. The instance lists are in DVC
(`configs/*.instances.json.dvc`). Keep the bundle's document ids and its UED (it names
the trained model) as for gfx950, and retrain the model when the knobs change. To
change the catalog, edit `gfx1250_fwd.json` and rerun the flow.

## Tests

- `tests/test_enumerate_conv_aot_instances_gfx1250.py` and
  `tests/test_make_conv_aot_config_gfx1250.py` (IngestorGenerator) check the lists
  against the dispatcher list and the shards against the lists.
- rocKE's `library/tests/test_gfx1250_conv_fwd_aot_packaging.py` checks the adapter's
  gfx1250 target: specs, knob encoding, serves and launch plans against the dispatcher.
- `TestGfx1250ConvFwdAotGeometry`, `TestGfx1250ConvFwdAotMatchers`, the census
  `TestGfx1250ConvFwdAotPacks` and `TestGfx1250ConvFwdAotModelBinding` run on the host.
- `IntegrationGpuGfx1250ConvFwd.cpp` runs every instance against the CPU reference, an
  unbenchmarked plan and a capped benchmark over a table of graphs in both dtypes, and
  checks each graph's per-family candidate counts against the adapter's.
- The engine lane, `hip-kernel-provider-gfx1250-conv-fwd-aot-external-integration-check`,
  runs `integration-test-bundles/standard/Gfx1250ConvFwd/{Smoke,Spatial,Grouped}` against
  the GPU reference with the bundles' support claims enforced.

## Current limits

- Graphs with a tensor of 2 GiB or more are declined (i32 byte sizes and offsets).
- No instance admits depthwise convolution with channel multiplier 2, or dense
  convolution with 1, 2 or 4 output channels per group; those graphs fall to other
  engines.
- TheRock CI does not run this provider's tests on gfx125X.
