# gfx950 forward convolution graph contract

Integration base: `9285929e1de2ada46cc01e9054ff3ed5fa6c0456`.
Engine: `hipkernel:Gfx950ConvFwd`. Descriptor dialect: `packaged`.

## 1. Operation match

Exactly one `ConvolutionFwdAttributes` node implements plain 2D forward
cross-correlation. The node binds three distinct tensor UIDs: input X, filter W,
and output Y. Bias, activation, backward, and other fused graphs decline.
The operation uses FP32 accumulation and uniform FP16 or BF16 input, filter,
and output. There is no workspace.

## 2. Field audit

The authority is `flatbuffers_sdk/schemas/convolution_fwd_attributes.fbs`.

| Field | Treatment |
|---|---|
| `x_tensor_uid` | Bind a nonvirtual, dense, rank-4 input tensor. |
| `w_tensor_uid` | Bind a nonvirtual, dense, rank-4 filter `[K, C/groups, Y, X]`; W dimension 1 must divide X's channels, and the quotient is the group count, which must also divide K. |
| `y_tensor_uid` | Bind output; validate its inferred dimensions and layout during prepare. |
| `pre_padding` | Require two nonnegative integers; match both compiled values. |
| `post_padding` | Require equality with pre-padding on both axes. |
| `stride` | Require two positive integers; match both compiled values. |
| `dilation` | Require two positive integers; match both compiled values. |
| `conv_mode` | Require `CROSS_CORRELATION`; decline `CONVOLUTION` and `UNSET`. |

The enclosing node's compute type must be `FLOAT`. Tensors must have positive
extents and valid physical strides. Reject virtual tensors, constant values,
runtime pass-by-value, ragged offsets, incompatible dtypes, and overlapping
runtime buffers. Actual addresses must satisfy the builder's 16-byte alignment
contract. Dense byte counts and kernel index arithmetic must fit signed 32-bit
values. Tensor names and graph names do not affect computation.

The enclosing graph's `is_override_shape_enabled` field (from `graph.fbs`)
must be false. Each packaged kernel has fixed dimensions and strides, so the
matcher rejects graphs that enable execution-time shape overrides.

This schema has no alpha/beta, bias, or group-count scalar. Plain convolution
overwrites Y. Groups are inferred as X channels divided by W dimension 1; X
channels and K must both be divisible by the group count, which is at most
65535 because the group rides on grid z. Depthwise (`C/groups == 1`) and
channel-multiplier depthwise (`K/groups > 1`) are ordinary grouped cases.
Grouped pointwise convolution -- groups > 1 with a 1x1 filter, stride 1 and no
padding -- is declined: rocKE's flat pointwise path indexes the input and
output as one GEMM and does not select the group's channel slabs. Fusions are
graph compositions and are rejected through the single-node requirement.

## 3. Frontend and reference reading

`ConvFpropAttributes` defaults to cross-correlation. `set_padding` sets both
padding vectors; separate pre/post setters can express asymmetric padding,
which this engine declines. Logical dimensions remain NCHW even when storage
is channels-last. The GPU reference `GpuRefConvFwd.cpp` indexes logical NCHW
using supplied strides and computes cross-correlation without reversing W.
The equivalent framework operation is `torch.nn.functional.conv2d` with no
bias, `groups` equal to the inferred group count, and explicit stride, padding,
and dilation. The CPU and GPU references infer the group count the same way.

No deprecated convolution attribute spelling appears in the pinned schema.
The output is inferred by the frontend and need not be complete during graph
matching, so output layout validation belongs to dispatch preparation.

## 4. Real graph sources

The in-tree convolution bundles and published hipDNN graph JSON carry logical
NCHW tensor dimensions and explicit strides. Both sources include
layouts and operations outside this engine's contract; layout must be derived
from strides, never from a tensor name or the dimensions alone. Published
workloads are substantially broader than the smoke shape and include backward,
3D, and fused graphs, which this engine declines, and grouped graphs, which it
serves when a matching catalog entry exists.

The convolution miner retains each source graph, its attributes, and its
provenance. Coverage is assessed against that inventory, independently of the
small correctness matrix. A legal rocKE request without a compiled catalog
entry is reported as a catalog gap.

## 5. Mapping to the compiled kernel

| rocKE field | hipDNN source / rule |
|---|---|
| `N` | X dimension 0; Y dimension 0 must agree. |
| `C` | X dimension 1; W dimension 1 is `C/groups`. |
| `K` | W dimension 0; Y dimension 1 must agree. |
| `Hi`, `Wi` | X dimensions 2 and 3. |
| `Y`, `X` | W dimensions 2 and 3 (filter height and width). |
| `sH`, `sW` | `stride[0]`, `stride[1]`. |
| `pH`, `pW` | `pre_padding[0]`, `pre_padding[1]`, equal to post-padding. |
| `dH`, `dW` | `dilation[0]`, `dilation[1]`. |
| `Ho`, `Wo` | `(input + 2*padding - dilation*(filter-1) - 1)/stride + 1`, with nonnegative numerator. |
| `dtype` | `HALF` maps to `fp16`; `BFLOAT16` maps to `bf16`. |
| `layout` | Dense channels-last. |
| `groups` | X dimension 1 divided by W dimension 1; must divide both C and K; at most 65535; must be 1 for a pointwise 1x1/stride-1/unpadded problem (grouped pointwise is declined). |
| `tile_m`, `tile_n`, `tile_k` | Compiled KMD fields; `tile_k` is an exposed engine knob (as is `kernel_family`, below). Direct kernels carry `tile_k=0`, so on a graph the direct family serves the knob also advertises 0, which forces the direct family (a known quirk; prefer `kernel_family`). |
| `warp_m`, `warp_n`, `warp_tile_m`, `warp_tile_n`, `warp_tile_k`, `wave_size` | Resolved compile-time tuning, checked by the builder and native geometry. |
| `pipeline`, `epilogue` | Dispatcher-resolved and carried in metadata. The catalog ships `mem`; the native matcher also admits `compv3`, `compv4` and `basic`, which share its launch. The `default` epilogue is admitted only when `K/groups` is odd, `cshuffle` always. |
| `kernel_family` | `0` implicit GEMM (KMD default), `1` direct depthwise; also an integer engine knob. Any other value is refused. |
| `direct_variant`, `block_w`, `block_waves` | `none`, 0, 0 (KMD defaults) for implicit GEMM; for direct kernels see section 6. |

Physical layouts (strides in elements):

| Tensor | Logical dimensions | Strides | rocKE physical order |
|---|---|---|---|
| X | `[N,C,Hi,Wi]` | `[Hi*Wi*C,1,Wi*C,C]` | NHWC |
| W | `[K,C/G,Y,X]` | `[Y*X*C/G,1,X*C/G,C/G]` | KYXC per group |
| Y | `[N,K,Ho,Wo]` | `[Ho*Wo*K,1,Wo*K,K]` | NHWK |

`G` is the group count. Group `g` reads input channels `[g*C/G, (g+1)*C/G)`
and writes output channels `[g*K/G, (g+1)*K/G)`.

A unit-extent axis does not participate in address arithmetic, so its declared
stride need not equal the canonical value. All nondegenerate axes must match.

## 6. Direct depthwise family

Kernels with `kernel_family=1` are rocKE's direct depthwise kernels from
`kernels/common/conv_direct_grouped.py`. Their implicit-GEMM fields hold
placeholders that the matcher requires exactly: `tile_m`, `tile_n`, `tile_k`,
`warp_m`, `warp_n` and the three `warp_tile_*` fields are 0, `wave_size` is 64,
and `pipeline` and `epilogue` are `none`. An implicit-GEMM kernel must carry
`direct_variant=none` and `block_w=block_waves=0`. Both the Python adapter and
the native matcher enforce these rules and the guard below.

| `direct_variant` | rocKE spec | `block_w` | `block_waves` | Channels |
|---|---|---|---|---|
| `std` | `DirectDepthwiseSpec` | Output columns per workgroup, positive | 1 to 16 | Any |
| `spatial` | `DirectDepthwiseSpatialSpec` | 0; the kernel uses `block_waves * (64 / groups)` | 1 to 16 | Fewer than 64 |

A graph is served by the direct family only when all of these hold; otherwise
only implicit-GEMM kernels match it:

- pure depthwise: `groups == C == K` (W dimension 1 is 1 and there is no
  channel multiplier);
- `stride[0] == stride[1]`, `pre_padding[0] == pre_padding[1]` (padding is
  already symmetric), and dilation 1 on both axes;
- row and column coverage: `floor((Hi - 1) / stride) == Ho - 1`, and the same
  for `Wi` and `Wo`. rocKE's direct kernels emit output row `p / stride` as
  they stream input row `p` (for `p` divisible by the stride), so other
  paddings leave output rows unproduced or, on the runtime row loop, write past
  the last output row into the next image;
- grid axes at most 65535, including grid z, which is N.

Launch geometry, block `(64 * block_waves, 1, 1)`, no LDS and no workspace:

| Variant | Grid |
|---|---|
| `std` | `(ceil(Wo / block_w), ceil(groups / (64 * block_waves)), N)` |
| `spatial` | `(ceil(Wo / (block_waves * (64 / groups))), 1, N)` |

The tensor storage, argument list and byte counts are the same as for implicit
GEMM (section 5).
