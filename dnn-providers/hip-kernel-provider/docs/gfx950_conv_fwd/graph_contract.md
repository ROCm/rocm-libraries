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
| `w_tensor_uid` | Bind a nonvirtual, dense, rank-4 filter; require input-channel equality, hence groups=1. |
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
overwrites Y. Groups are inferred from X channels divided by filter channels;
requiring those channel dimensions equal enforces groups=1. Fusions are graph
compositions and are rejected through the single-node requirement.

## 3. Frontend and reference reading

`ConvFpropAttributes` defaults to cross-correlation. `set_padding` sets both
padding vectors; separate pre/post setters can express asymmetric padding,
which this engine declines. Logical dimensions remain NCHW even when storage
is channels-last. The GPU reference `GpuRefConvFwd.cpp` indexes logical NCHW
using supplied strides and computes cross-correlation without reversing W.
The equivalent framework operation is `torch.nn.functional.conv2d` with no
bias, `groups=1`, and explicit stride, padding, and dilation.

No deprecated convolution attribute spelling appears in the pinned schema.
The output is inferred by the frontend and need not be complete during graph
matching, so output layout validation belongs to dispatch preparation.

## 4. Real graph sources

The in-tree convolution bundles and the public `ROCm/dnn-benchmarking`
`Workloads/headline/conv` and `Workloads/microbench/conv_sweep` archives carry
logical NCHW tensor dimensions and explicit strides. Both sources include
layouts and operations outside this engine's contract; layout must be derived
from strides, never from a tensor name or the dimensions alone. Published
workloads are substantially broader than the smoke shape and include backward,
grouped, 3D, and fused graphs.

The convolution miner retains each source graph, its attributes, and its
provenance. Coverage is assessed against that inventory, independently of the
small correctness matrix. A legal rocKE request without a compiled catalog
entry is reported as a catalog gap.

## 5. Mapping to the compiled kernel

| rocKE field | hipDNN source / rule |
|---|---|
| `N` | X dimension 0; Y dimension 0 must agree. |
| `C` | X dimension 1; W dimension 1 must agree. |
| `K` | W dimension 0; Y dimension 1 must agree. |
| `Hi`, `Wi` | X dimensions 2 and 3. |
| `Y`, `X` | W dimensions 2 and 3 (filter height and width). |
| `sH`, `sW` | `stride[0]`, `stride[1]`. |
| `pH`, `pW` | `pre_padding[0]`, `pre_padding[1]`, equal to post-padding. |
| `dH`, `dW` | `dilation[0]`, `dilation[1]`. |
| `Ho`, `Wo` | `(input + 2*padding - dilation*(filter-1) - 1)/stride + 1`, with nonnegative numerator. |
| `dtype` | `HALF` maps to `fp16`; `BFLOAT16` maps to `bf16`. |
| `layout`, `groups` | Dense channels-last, groups=1. |
| `tile_m`, `tile_n`, `tile_k` | Compiled KMD fields; only `tile_k` is an exposed engine knob. |
| `warp_m`, `warp_n`, `warp_tile_m`, `warp_tile_n`, `warp_tile_k`, `wave_size` | Resolved compile-time tuning, checked by the builder and native geometry. |
| `pipeline`, `epilogue` | `mem` pipeline; dispatcher-resolved epilogue, carried in metadata. |

Physical layouts (strides in elements):

| Tensor | Logical dimensions | Strides | rocKE physical order |
|---|---|---|---|
| X | `[N,C,Hi,Wi]` | `[Hi*Wi*C,1,Wi*C,C]` | NHWC |
| W | `[K,C,Y,X]` | `[Y*X*C,1,X*C,C]` | KYXC |
| Y | `[N,K,Ho,Wo]` | `[Ho*Wo*K,1,Wo*K,K]` | NHWK |

A unit-extent axis does not participate in address arithmetic, so its declared
stride need not equal the canonical value. All nondegenerate axes must match.
