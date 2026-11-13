# Standalone unary and binary pointwise

`hipkernel:StandalonePointwise` (engine ID `4178892485795623278`,
`0x39fe68933a65b16e`) is a production HIP ingestor pack for exactly one
`PointwiseAttributes` node. It does not change legacy normalization builders or
the one-element pointwise packaging fixtures.

## Supported contract

- Unary: `ABS`, `NEG`, `IDENTITY`, `RELU_FWD`, `SIGMOID_FWD`, `TANH_FWD`,
  `GELU_FWD`, `GELU_APPROX_TANH_FWD`, `SWISH_FWD`.
- Binary: `ADD`, `SUB`, `MUL`, `MIN_OP`, `MAX_OP`, `CMP_GT`, `RELU_BWD`,
  `SIGMOID_BWD`, `TANH_BWD`. Backward operands are the upstream gradient and
  forward input, not the forward output.
- Compute type is `FLOAT`; floating storage is `FLOAT`, `HALF`, or `BFLOAT16`.
  Unary input/output storage must match, or FLOAT input may produce HALF or
  BFLOAT16. Binary equal input storage may produce that storage or FLOAT;
  FLOAT paired with HALF or BFLOAT16 produces FLOAT. HALF/BFLOAT16 pairs and
  cross-storage low-precision outputs decline. `CMP_GT` requires matching
  floating input types and BOOLEAN output.
- Each participating tensor has rank 1-5, positive extents and strides, and no
  virtual or ragged metadata. Inputs right-align broadcast to the output;
  singleton axes do not advance. Padded and permuted strides are supported.
  Output axes, sorted by stride, must step beyond the occupied span of
  smaller-stride axes. Overlapping outputs, overflowing sizes/offsets and output
  UIDs shared with an input decline. Stride-zero views are unsupported;
  broadcasting uses singleton dimensions.
- All operands are device buffers. Baked values, runtime pass-by-value operands,
  ternary modes, GEN_INDEX, and explicitly supplied activation parameters
  decline with a logged reason, including explicitly supplied default values.
  ReLU and swish use their unparameterized CPU-reference semantics.
- Other modes and dtype/compute signatures decline. This initial contract
  follows the executable CPU-reference unary/binary subset; it does not claim
  completion of a separate per-operation test plan.

## Compilation and persistence

The pack emits one code object per configured architecture, with a bounded
grid-stride loop and no workspace. Mode, storage types, dimensions and strides
are fixed-size launch metadata rather than shape-baked compile variants.
Compilation does not classify tensor layout as NCHW or NHWC.

The metadata uses fixed-size `std::array` fields. ABI tests pin field offsets
and storage codes so saved plans retain their original layout.

The dispatch checks the kernel argument ABI (three pointers and a 184-byte
metadata struct). Plans save and restore their launch metadata through
`hipkernel.standalone_pointwise.dispatch.v1`; restore validates modes, arity,
types, sizes, strides and aliases before loading device code.

## Testing

Native applicability, formula and plan-contract tests:

```sh
build/bin/hip_kernel_provider_tests --gtest_filter='TestStandalonePointwise.*'
```

Graph correctness uses 27 frontend-captured cases under
`quick/StandalonePointwise`, covering ranks 1-5, right-aligned broadcasting,
padded strides, BOOLEAN comparison, mixed FLOAT/HALF/BFLOAT16 storage and all
18 supported modes.
Their support sidecars are intentionally empty until engine claims are added.
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
