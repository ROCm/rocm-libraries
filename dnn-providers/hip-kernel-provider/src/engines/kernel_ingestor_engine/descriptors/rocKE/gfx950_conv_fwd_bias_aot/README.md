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
output channel (`g * K/groups + k_in_group`). A graph that declares the convolution's output in the
io dtype is served the same way: the kernel skips that intermediate rounding, so its result is
closer to the FP32 sum than the graph's two-rounding reference.

## Kernels and ABI

Every KDP descriptor carries `bias` = 1 (KMD field, no default). The kernel matchers of both
engines read it: a bias instance serves only this engine's graph, and an instance without it only
the unfused engine's. A bias kernel takes rocKE's forward AOT ABI (`conv_abi.py`) followed by
`(Bias: pointer, Bias_bytes: i32)`, `Bias_bytes` = K x 2. A split-C kernel passes the pair to both
its conv pass and its fold (one kernarg block, as for the other arguments); the fold adds the
bias. A batch-split kernel passes the whole bias to every chunk.

## Artifact

The KDPs are rocKE's bias catalog: the bias twin of every instance of the unfused engine's
catalog, 233 per dtype in 16 packs. Per dtype: 89 implicit-GEMM instances in
`<dtype>_{mem,compv3,compv4}_{plain,unroll}`, 111 direct ones in `<dtype>_direct` and 33
split-C ones in `<dtype>_direct_split`, whose fold is built with the bias too. The packaging
adapter (`gfx950_conv_fwd_aot_spec_for_identity`) packages a bias twin by family
(`AOT_BIAS_FAMILIES` in `rocke/library/builders/common/convolution_forward.py`: implicit GEMM,
direct and split-C), and `enumerate_conv_aot_instances.py --bias` reports and leaves out any twin
it refuses. A KDP is a list of rocKE specs (`bias` = 1) that the build compiles at pack time
from this tree's `rocke/`, so the build makes the kernels itself.
Like the unfused engine's, the KDPs are in DVC: `artifacts/`, tracked by `artifacts.dvc`
(layout in `../../README.md`, "Large rocKE files live in DVC"); fetch them with
`dvc pull -r ingestor artifacts.dvc` from this folder. Without them the bundle carries no
kernel, gfx950 ships no kernel of the engine, and its integration check
(`src/CMakeLists.txt`) and census (`TestGfx950ConvFwdBiasAotPacks`) are not registered; the
build is unaffected.

To regenerate it, from `projects/hipdnn/tools/IngestorGenerator` (environment as in
`docs/gfx950_conv_fwd/aot_instances.md`, "Regenerating the bundle"):

1. `python tools/enumerate_conv_aot_instances.py --bias --rocke-root "$ROCKE" --output-dir configs`
   writes the bias catalog lists `configs/gfx950_conv_fwd_bias_aot_{fp16,bf16}.instances.json`:
   the forward catalog's instances with rocKE's `bias` set (rocKE's AOT bias axis,
   `aot_jobs.fwd_bias_twin`), those whose bias twin the packaging adapter packages.
2. `python tools/make_conv_aot_config.py --bias --rocke-root "$ROCKE" --instances
   configs/gfx950_conv_fwd_bias_aot_{fp16,bf16}.instances.json --output "$WORK"/bias.yaml`
3. `python generate.py --config "$WORK"/bias.yaml --output-dir "$GENERATED" --force`
   (`--new-ids` while this folder holds no KDP: the generator reuses kernel ids only from an
   installed bundle that has them).
4. Keep this folder's document ids (UED, KMD, UDDs, UMD, UHD) and the UED and UMD
   themselves: rewrite the generated ids in the KDPs to these, move the KDPs into
   `artifacts/`, then `python tools/make_conv_aot_config.py --bias --stabilize-ids
   artifacts/*.kdp.json`. Copy the generated census
   `tests/TestGfx950ConvFwdBiasAotPacks.cpp` to
   `src/tests/engines/kernel_ingestor_engine/packs/` and `clang-format` it.
5. `dvc add artifacts`, keep `remote: ingestor` in `artifacts.dvc`, and `dvc push -r ingestor`.

The UED ranks with the native fallback score only (`hipkernel.gfx950_conv_fwd_bias_aot.score`,
the unfused engine's order); no trained ranker exists for it.
