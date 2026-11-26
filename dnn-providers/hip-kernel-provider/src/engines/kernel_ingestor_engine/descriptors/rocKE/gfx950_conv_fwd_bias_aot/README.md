# gfx950_conv_fwd_bias_aot

Forward convolution plus a per-output-channel bias for gfx950, fused: rocKE's shape-generic
forward AOT instances (the `gfx950_conv_fwd_aot` catalog) built with the bias epilogue.
Engine: `hipkernel:Gfx950ConvFwdBiasAot`; native code `packs/Gfx950ConvFwdBiasNative.cpp` on
`packs/Gfx950ConvFwdAotCommon.cpp`, which it shares with `hipkernel:Gfx950ConvFwdAot`.

## Graph

Two nodes: a `ConvolutionFwd` (fp32 compute, the unfused engine's rules: 2-D cross-correlation,
symmetric padding, packed NHWC/KYXC fp16 or bf16 tensors, groups encoded by the filter) whose
output is virtual (fp32 or the io dtype), and a pointwise `ADD` (fp32 compute) of that output
and a bias `B`. `B` is a non-virtual device tensor of the io dtype, dims `(1, K, 1, 1)` or
`(K, 1, 1)`, contiguous along K. The ADD's output is the stored, packed NHWK `y`.

The kernel computes `y = round(acc_fp32 + float(B[k]))` with one rounding; `k` is the absolute
output channel (`g * K/groups + k_in_group`).

## Kernels and ABI

Every KDP descriptor carries `bias` = 1 (KMD field, no default). The kernel matchers of both
engines read it: a bias instance serves only this engine's graph, and an instance without it only
the unfused engine's. A bias kernel takes rocKE's forward AOT ABI (`conv_abi.py`) followed by
`(Bias: pointer, Bias_bytes: i32)`, `Bias_bytes` = K x 2. A split-C kernel passes the pair to both
its conv pass and its fold (one kernarg block, as for the other arguments); the fold adds the
bias. A batch-split kernel passes the whole bias to every chunk.

## Artifact

The KDPs are not in this tree yet: they come from rocKE's bias catalog. A KDP is a list of
rocKE specs that the build compiles at pack time from this tree's `rocke/`, so the build makes
the kernels itself once the KDPs exist; like the unfused engine's, they belong in a DVC folder
`artifacts/` (layout in `../../README.md`, "Large rocKE files live in
DVC"). Until it is added, the bundle carries no kernel, gfx950 ships no kernel of the engine, and
its integration check (`src/CMakeLists.txt`) is not registered; the build is unaffected. To add
it, from `projects/hipdnn/tools/IngestorGenerator`:

1. `python tools/enumerate_conv_aot_instances.py --bias --rocke-root "$ROCKE"` writes the bias
   catalog lists `configs/gfx950_conv_fwd_bias_aot_{fp16,bf16}.instances.json`: the forward
   catalog's instances with rocKE's `bias` set, those whose bias twin the packaging adapter
   (`gfx950_conv_fwd_aot_spec_for_identity`) packages. It needs a rocKE whose AOT forward spec
   carries `bias`; until then every twin is refused and nothing is written.
2. `python tools/make_conv_aot_config.py --bias --rocke-root "$ROCKE" --instances
   configs/gfx950_conv_fwd_bias_aot_{fp16,bf16}.instances.json --output "$WORK"/bias.yaml`
3. `python generate.py --config "$WORK"/bias.yaml --output-dir "$GENERATED" --force`, then
   `python tools/make_conv_aot_config.py --bias --stabilize-ids "$GENERATED"/<...>/*.kdp.json`.
4. Keep this folder's document ids (UED, KMD, UDDs, UMD, UHD), so the KDPs name them: replace the
   generated ones by these or pass them as prior ids. Move the KDPs into `artifacts/`, add
   `/artifacts` to a `.gitignore` here, `dvc add artifacts`, append `remote: ingestor` to
   `artifacts.dvc`, and `dvc push -r ingestor`.

The UED ranks with the native fallback score only (`hipkernel.gfx950_conv_fwd_bias_aot.score`,
the unfused engine's order); no trained ranker exists for it.
