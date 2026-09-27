# gfx950 convolution kernel mining

Builder: `rocke.instances.common.conv_implicit_gemm.build_implicit_gemm_conv`.
Packaging adapter: `builders.common.convolution_forward.build_gfx950_conv_fwd`.
Base: `9285929e1de2ada46cc01e9054ff3ed5fa6c0456`.

The primary source is `platform/python/rocke/instances/common/conv_implicit_gemm.py`.
Supporting sources are its `_conv_implicit_gemm_common.py`,
`library/dispatch/grouped_convolution.py`, the graph schema and frontend header
named in `graph_contract.md`, and the existing convolution reference tests.

| Constraint | Verdict | Enforcement |
|---|---|---|
| Architecture | gfx950 only for this integration | Pack architecture, adapter, native matcher. |
| Direction | Plain forward | One `ConvolutionFwdAttributes` node; adapter requires forward request. |
| Dtype | Uniform FP16 or BF16 with FP32 accumulation | Graph match and adapter; per-kernel dtype equality. |
| Layout | Dense channels-last | Graph strides and prepare-time output check. |
| Spatial rank | 2D only in this engine | Rank/attribute vector checks. |
| Group count | 1 in this engine | X and W channels equal; adapter group check. |
| Padding | Nonnegative and symmetric | Both graph padding vectors checked. |
| Stride and dilation | Positive, both dimensions baked | Exact per-kernel metadata match. |
| Shape | Every dimension is baked | Exact per-kernel metadata match. |
| Fusion callbacks | None | Adapter delegates without callbacks; graph declines fusion. |
| Buffer ABI | Three 16-byte-aligned, nonaliasing pointers | Launch validation. |
| Buffer byte counts | Signed i32 | Checked multiplication before narrowing. |
| Grid bounds | All axes at most 65535 | Real `is_valid_spec_for_problem` and native geometry. |
| Pipeline | `mem` | Spec and native matcher. |
| Epilogue | Dispatcher-derived `cshuffle` or `default` | Resolved in factory, serialized, validated. |
| Tune validity | Real rocKE rules for atoms, waves, divisibility and LDS | Delegate to `is_valid_spec_for_problem`; no duplicated permissive oracle. |
| Workspace | Zero | Forward kernel has no scratch argument. |

Grouped and 3D convolution exist in the upstream family. Their rejection here
is an integration-contract gap, not evidence that rocKE cannot implement them.
Likewise, a supported shape absent from the package is a catalog gap. The
coverage report keeps these distinct from actual family predicate failures.

## Default tuning and initial candidate pair

The gfx950 forward dispatcher resolves `tile_m=64`, `tile_n=64`, `tile_k=64`,
`warp_m=2`, `warp_n=2`, atoms `32x32x16`, `wave_size=64`, and `pipeline=mem`.
The smoke shape resolves the `cshuffle` epilogue. Its tuning twin changes only
`tile_k` to 128. Both are real integer engine-knob values. Fallback ranking
prefers 64; runtime benchmarking can select either valid candidate.

The adapter serializes resolved values. The original builder's name omits dtype,
padding, stride, and dilation; a deterministic digest over the complete adapter
spec prevents distinct compiled kernels from sharing a symbol.

## Launch contract

Arguments, in declaration order:

1. Input A pointer.
2. Filter B pointer.
3. Output D pointer.
4. `A_bytes` (`i32`): `2*N*Hi*Wi*C`.
5. `B_bytes` (`i32`): `2*K*Y*X*C`.
6. `D_bytes` (`i32`): `2*N*Ho*Wo*K`.

Grid is `(ceil(K/tile_n), ceil(N*Ho*Wo/tile_m), 1)` and block is
`(warp_m*warp_n*wave_size,1,1)`. Dynamic shared memory is zero; static LDS is
part of the compiled code object. The prepared dispatch must retain both the
compiled program and its runnable kernel and copy graph UIDs/scalars, because
match contexts do not outlive plan preparation.

## Verification obligations

Compare adapter IR with direct original-builder IR for each initial dtype/tile
pair. Validate descriptors before and after packing. Force both knob values
through the exact engine, check numerical results against an independent
reference, and test all rejection rows. Benchmark using a fresh cache and
explicit benchmarking, then verify plan recreation and process-restart reuse.
Keep the initial search out of steady-state timing and compare against direct
rocKE execution using the same spec, compiler, and GPU.
