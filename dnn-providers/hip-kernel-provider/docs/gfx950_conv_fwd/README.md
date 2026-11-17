# Packaged gfx950 forward convolution

`hipkernel:Gfx950ConvFwdAot` runs rocKE's implicit-GEMM forward convolution,
and for pure depthwise convolution rocKE's direct depthwise kernels, through
the HIP Kernel Provider. It accepts plain 2D cross-correlation with dense
channels-last storage, FP16 or BF16 storage, FP32 accumulation, and symmetric
padding. Grouped convolution, including depthwise and channel-multiplier
depthwise, is supported: the group count is inferred as X channels divided by
W dimension 1, and each group runs on grid z. Grouped pointwise convolution
(1x1 filter, stride 1, no padding, groups > 1) is served by grouped kernels
built against a non-pointwise problem; no kernel is built for it, because
rocKE's flat pointwise path does not select the group's channel slabs.

The engine holds two kernel families, told apart by the integer metadata field
and engine knob `kernel_family`:

- `0`, implicit GEMM: every accepted graph.
- `1`, direct depthwise: rocKE's `DirectDepthwiseSpec` (`direct_variant`
  `std`, tuned by `block_w` and `block_waves`) and `DirectDepthwiseSpatialSpec`
  (`spatial`, tuned by `block_waves`, for fewer than 64 channels), plus the
  tiled and windowed direct families. They serve only pure depthwise graphs
  (`groups == C == K`) with equal strides, symmetric equal padding and no
  dilation, and only when the last input row and column feed the last output
  row and column: rocKE's direct kernels compute wrong answers otherwise, so
  those graphs are left to implicit GEMM.

The graph matcher validates the output tensor's packed-NHWK layout and
inferred dimensions directly, rather than deferring that check to dispatch
preparation: the frontend infers a sole, non-virtual output's dims and
strides during `Graph::build()`'s `validate()` pass, strictly before engine
matching, so its layout is already final by the time the matcher runs.

rocKE's AOT kernels take the problem as kernel arguments, so a packaged kernel
serves every graph its build-time capabilities admit, not only the shape it
was compiled for. Each kernel ships as a shape-generic AOT instance, one
rocKE `KernelIdentity` built once against the AOT sweep's probe problem
rather than a catalog shape: see [AOT instances](aot_instances.md) for what an
instance is, which graphs it serves, the shipped catalog, build and packaging,
selection among many candidates, and the GPU test suites (including the
test-only seed engine, `hipkernel:Gfx950ConvFwdAotSeed`, that pins a small
catalog slice instance by instance).
