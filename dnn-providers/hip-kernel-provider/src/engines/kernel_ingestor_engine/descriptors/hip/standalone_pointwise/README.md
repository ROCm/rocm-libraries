# Standalone pointwise

`hipkernel:StandalonePointwise` (engine ID `4178892485795623278`,
`0x39fe68933a65b16e`) is a production HIP ingestor pack for exactly one
`PointwiseAttributes` node. It does not change legacy normalization builders or
the one-element pointwise packaging fixtures.

## Supported contract

- Unary: `ABS`, `NEG`, `IDENTITY`, `RELU_FWD`, `SIGMOID_FWD`, `TANH_FWD`,
  `GELU_FWD`, `GELU_APPROX_TANH_FWD`, `SWISH_FWD`.
- Binary: `ADD`, `SUB`, `MUL`, `MIN_OP`, `MAX_OP`, `CMP_GT`, `RELU_BWD`,
  `SIGMOID_BWD`, `TANH_BWD`, `ELU_BWD`, `SOFTPLUS_BWD`, `SWISH_BWD`.
  Unary modes additionally include `ELU_FWD`, `SOFTPLUS_FWD`, and `GEN_INDEX`.
  `BINARY_SELECT` takes two same-storage floating values and a BOOLEAN third
  operand; all three inputs may broadcast. Backward operands are the upstream gradient and
  forward input, not the forward output.
- Compute type is `FLOAT`; floating storage is `FLOAT`, `HALF`, or `BFLOAT16`.
  Unary input/output storage must match, or FLOAT input may produce HALF or
  BFLOAT16. Binary equal input storage may produce that storage or FLOAT;
  FLOAT paired with HALF or BFLOAT16 produces FLOAT. HALF/BFLOAT16 pairs and
  cross-storage low-precision outputs decline. `CMP_GT` requires matching
  floating input types and BOOLEAN output. `GEN_INDEX` accepts floating input
  storage and produces FLOAT or INT32 output.
- Each participating tensor has rank 1-5, positive extents and strides, and no
  virtual or ragged metadata. Inputs right-align broadcast to the output;
  singleton axes do not advance. Padded and permuted strides are supported.
  Output axes, sorted by stride, must step beyond the occupied span of
  smaller-stride axes. Overlapping outputs, overflowing sizes/offsets and output
  UIDs shared with an input decline. Stride-zero views are unsupported;
  broadcasting uses singleton dimensions.
- Inputs may be device buffers, baked scalar values, or runtime pass-by-value
  host scalars. Value operands must contain exactly one element, match their
  declared storage type, and cannot be both baked and runtime. Outputs must be
  device buffers. Scalar operands broadcast like singleton tensors.
- ReLU reads `relu_lower_clip`, `relu_upper_clip`, and `relu_lower_clip_slope`;
  swish reads `swish_beta`; ELU reads `elu_alpha`; softplus reads
  `softplus_beta` (strictly positive). Scalar activation parameters must be finite,
  except that a positive-infinity ReLU upper clip is allowed. The lower clip
  cannot exceed the upper clip. Defaults follow the CPU reference.
  Parameters supplied to modes that do not read them decline, even when the
  supplied value equals the default.
- `GEN_INDEX` interprets `axis_tensor_uid` as an axis number, not a tensor UID,
  requires `0 <= axis < output rank`, and bounds the axis extent to INT32 range.
- Other modes and dtype/compute signatures decline. This contract
  follows the executable CPU-reference unary/binary subset; it does not claim
  completion of a separate per-operation test plan.

## Compilation and persistence

The pack emits one code object per configured architecture, with a bounded
grid-stride loop and no workspace. Mode, storage types, dimensions and strides
are fixed-size launch metadata rather than shape-baked compile variants.
Compilation does not classify tensor layout as NCHW or NHWC.

The metadata uses fixed-size `std::array` fields. ABI tests pin field offsets
and storage codes so saved plans retain their original layout.

The current dispatch checks four pointers and a 272-byte metadata struct.
Saveable plans use `hipkernel.standalone_pointwise.dispatch.v2`; restore validates
modes, arity, types, parameters, sizes, strides and aliases before loading device
code. The original `.v1` handler and its three-pointer/184-byte ABI remain
registered unchanged for previously saved unary/binary plans. Production
descriptors select only `.v2`, so retaining `.v1` adds no production code variant.

Plans with runtime pass-by-value inputs **cannot be saved**. Each execution reads
the current host scalar into its own launch metadata; no host pointer is retained
in a saved plan. Baked scalars are preserved across save/restore.

## Testing

Native applicability, formula and plan-contract tests:

```sh
build/bin/hip_kernel_provider_tests \
  --gtest_filter='TestStandalonePointwise*.*:TestGpuStandalonePointwiseV2Plan.*'
```

Graph correctness uses 83 frontend-captured cases under
`quick/StandalonePointwise`, covering ranks 1-5, right-aligned broadcasting,
padded strides, BOOLEAN comparison, mixed FLOAT/HALF/BFLOAT16 storage and all
25 supported modes, all seven parameter fields, and baked/runtime scalar operands.
Their support sidecars claim `hipkernel:StandalonePointwise` on `gfx90a/linux`,
verified against the CPU reference. Existing `Pointwise/Default`,
`Pointwise/Activation` and `Pointwise/Binary` bundles also carry verified claims
for this engine on that target. Claims do not extend to other architectures or
to legacy normalization fusion.
The engine-pinned external CTest lanes run these bundles when the configured
production descriptor root actually ships this pack.

```sh
build/bin/hipdnn_integration_tests \
  --test-article build/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
  --test-engine hipkernel:StandalonePointwise \
  --test-config dnn-providers/hip-kernel-provider/config/hipkernel_StandalonePointwise.toml \
  --verification-mode cpu --validator cpu \
  --golden-data-dir dnn-providers/integration-tests/integration-test-bundles \
  --gtest_filter='quick_StandalonePointwise*.*'
```
