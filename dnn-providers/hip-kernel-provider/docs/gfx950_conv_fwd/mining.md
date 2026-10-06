# gfx950 convolution kernel mining

Builder: `kernels.common.conv_implicit_gemm.build_implicit_gemm_conv`; for the
direct depthwise family see [below](#direct-depthwise-family).
Packaging adapter: `builders.common.convolution_forward.build_gfx950_conv_fwd`.
Base: `9285929e1de2ada46cc01e9054ff3ed5fa6c0456`.

The primary source is `library/kernels/common/conv_implicit_gemm.py`.
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
| Group count | Any divisor of both C and K, at most 65535; grouped pointwise (1x1, stride 1, no padding) declined | Graph match infers X.C / W.dims[1]; adapter `_problem_error`; per-kernel `groups` equality. |
| Padding | Nonnegative and symmetric | Both graph padding vectors checked. |
| Stride and dilation | Positive, both dimensions baked | Exact per-kernel metadata match. |
| Shape | Every dimension is baked | Exact per-kernel metadata match. |
| Fusion callbacks | None | Adapter delegates without callbacks; graph declines fusion. |
| Buffer ABI | Three 16-byte-aligned, nonaliasing pointers | Launch validation. |
| Buffer byte counts | Signed i32 | Checked multiplication before narrowing. |
| Grid bounds | All axes at most 65535; grid z is the group count | Real `is_valid_spec_for_problem` and native geometry. |
| Pipeline | Catalog ships `mem`; `compv3`, `compv4` and `basic` share its launch and are admitted | Adapter `_PACKAGED_PIPELINES` and native matcher; `wavelet` refused. |
| Epilogue | Dispatcher-derived `cshuffle` or `default` | Resolved in factory, serialized, validated; native matcher takes `default` only when `K/groups` is odd. |
| Tune validity | Real rocKE rules for atoms, waves, divisibility and LDS | Delegate to `is_valid_spec_for_problem`; no duplicated permissive oracle. |
| Workspace | Zero | Forward kernel has no scratch argument. |

Grouped 2D forward convolution, depthwise included, is in scope. Grouped
pointwise is declined because the kernel's flat pointwise shortcut ignores the
group slab offsets and computes wrong results; the fix belongs in the rocKE
kernel, not here. 3D convolution exists in the upstream family; its rejection
here is an integration-contract gap, not evidence that rocKE cannot implement
it.
Likewise, a supported shape absent from the package is a catalog gap. The
coverage report keeps these distinct from actual family predicate failures.

## Direct depthwise family

Builders: `kernels.common.conv_direct_grouped.build_direct_depthwise`
(`DirectDepthwiseSpec`) and `build_direct_depthwise_spatial`
(`DirectDepthwiseSpatialSpec`), reached through the same adapter entry point
when `kernel_family` is 1. rocKE has no dispatcher for them;
`gfx950_conv_fwd_direct_spec_for_request` applies the packaged default
(spatial with `block_waves=1` below 64 channels, otherwise standard with
`block_w=4` and `block_waves=1`) or an explicit arm.

| Constraint | Verdict | Enforcement |
|---|---|---|
| Group structure | Pure depthwise, `groups == C == K` | Adapter `_direct_error`; native matcher. |
| Stride, padding, dilation | One stride and one padding for both axes; no dilation | Adapter and native matcher; the builder takes a single `stride` and `PAD`. |
| Filter and padding | Odd filter extents and "same" padding `PAD == (KH - 1) / 2` | rocKE's own `forward_padding_reason`, run by its spec validators; native `directPaddingSupported`. |
| Row and column coverage | `floor((in - 1) / stride) == out - 1` on both axes | Adapter `_direct_rows_covered`; native `directRowsCovered`. Still needed for the width of a rectangular filter, which rocKE's padding rule does not check. |
| Tuning | `block_waves` 1 to 16; standard `block_w > 0`; spatial only below 64 channels with `block_w` 0 | Adapter, then rocKE's own spec validators; native `directLaunchGeometry`. |
| Grid bounds | All axes at most 65535; grid z is N | Adapter and native geometry. |
| Placeholders | Implicit-GEMM tuning fields fixed (`tile_k=0`, `pipeline`/`epilogue` `none`) | Adapter and native matcher require them exactly. |
| Workspace and LDS | Zero | The kernels take no scratch argument and use no LDS. |

The ABI is rocKE's direct-conv one: the same six leading arguments as implicit
GEMM, with the same byte counts, followed by the direct problem block (see
[Launch contract](#launch-contract)). The standard grid is
`(ceil(Wo/block_w), ceil(groups/(64*block_waves)), N)` and the spatial grid
`(ceil(Wo/(block_waves*(64/groups))), 1, N)`; both use a block of
`64*block_waves` threads. These match rocKE's
`benchmarks/common/benchmark_direct_conv.py`.

The direct kernels' own names omit the filter size, padding and stride, so
distinct kernels would share a symbol; the adapter names them
`hkp_conv_fwd_dw_gfx950_` plus a digest over every spec field. The four family
fields are left out of an implicit-GEMM spec's digest while they hold their
defaults, so implicit-GEMM symbols and code objects are unchanged by them.
Direct kernels are lowered with rocKE's Python backend, which `hkp_pack` pins;
the C++ lowering slows down sharply as the unrolled kernel grows with
`block_w`. Both variants stream input rows through a runtime loop over `Hi`;
the filter taps and output columns of a block stay unrolled.

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

The kernels are rocKE's AOT builds, which take the problem as kernel arguments.
The order is defined once in `rocke/library/kernels/common/conv_abi.py` and the
values by `ConvArgs.to_launch_values` in `conv_args.py`; the adapter exposes
both as `Gfx950ConvFwdSpec.launch_signature` / `launch_values`. Each packaged
kernel is still built for one exact problem, because the builder makes
code-shape decisions (pointwise, grouping, load widths) from it, so the matcher
still requires every geometry field.

Both families open with, in declaration order:

1. Input A pointer.
2. Filter B pointer.
3. Output D pointer.
4. `A_bytes` (`i32`): `2*N*Hi*Wi*C`.
5. `B_bytes` (`i32`): `2*K*Y*X*(C/groups)`.
6. `D_bytes` (`i32`): `2*N*Ho*Wo*K`.

Implicit GEMM follows with 39 `i32`s: the extents and attributes `N` through
`dW`, `groups`, `Ho`, `Wo`, `C/groups`, `K/groups`; the GEMM extents
`K_gemm = Y*X*C/groups` and `M = N*Ho*Wo`; the packed NHWC, KYXC and NHWK
strides; magic-division pairs for `Ho`, `Wo`, `X` and `C/groups`; and the tile
counts `num_pid_m`, `num_pid_n`, which must be the grid's. `K_gemm` must stay
below `2^23` for the kernel's 24-bit address products.

The direct kernels follow with 14 `i32`s: `N`, `Hi`, `Wi`, `Ho`, `Wo`,
`groups`, total `C` and `K`, then the NHWC and NHWK strides. Filter size,
stride and padding are compiled in.

The native pack computes the same values (`implicitGemmKernargs`,
`directKernargs`, `magicDivision` in `Gfx950ConvFwdGeometry.hpp`) and checks
the loaded symbol's recorded argument names against them before any launch,
because the packing is positional and every problem argument has the same kind
and size.

Grid is `(ceil((K/groups)/tile_n), ceil(N*Ho*Wo/tile_m), groups)` and block is
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
