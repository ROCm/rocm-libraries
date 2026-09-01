# Fused Epilogue Extensions: RMSNorm and Composable Epilogue Chains

This document specifies the hipBLASLt API extensions for fused epilogues, starting with
RMSNorm, and the composable epilogue-chain mechanism that lets RMSNorm compose with
optional residual add, AMax capture, and output requantization (e.g. FP8). It also reserves
the design points required so that Gated Linear Units (SwiGLU/GeGLU/ReGLU) can be added later
without reworking the API.

## 1. Motivation and scope

The public hipBLASLt matmul computes:

```text
D = Activation(alpha * op(A) * op(B) + beta * op(C) + bias)
```

Today the epilogue is selected by a single combinatorial enum `hipblasLtEpilogue_t`
(e.g. `HIPBLASLT_EPILOGUE_RELU_AUX_BIAS`). That model does not scale to ordered chains of
post-GEMM operations such as `GEMM -> residual -> RMSNorm -> requant`, because every
combination would require a distinct enum value.

This design defines:

1. RMSNorm as the first fused, composable epilogue stage.
2. Two RMSNorm flows built on a shared partial-reduction producer: a full RMSNorm flow that
   applies the per-row scale immediately, and a decomposed `GEMM -> residual -> RMSNorm ->
   GEMM` flow that defers the scale into the consuming GEMM's epilogue.
3. An explicit, ordered epilogue-chain descriptor so that allowed and disallowed stage
   sequences can be expressed and validated at API-call time, before any kernel selection
   or launch.
4. Reserved extension points for adjacent epilogue components and additional epilogue families
   without requiring new combinatorial enum values.

## 2. RMSNorm definition

For each row `x` of the post-GEMM tensor (optionally after a residual add), RMSNorm is:

```text
y = x * rsqrt(mean(x^2) + eps) * gamma
```

- Normalization axis: per row of `D`, over the `N` (feature) dimension.
- Accumulation precision: FP32 for the sum of squares and the reciprocal sqrt, regardless
  of the storage type of `D`.
- `eps`: user-specified small constant added inside the rsqrt for numerical stability.
- `gamma`: required per-feature scale vector of length `N` (the column count of `D`). This
  corresponds to the RMSNorm affine scale.
- Unlike LayerNorm, RMSNorm does not subtract the mean and has no `beta`. It also does not
  emit `mean`/`invvar` side outputs (the standalone `hipblasltExtLayerNorm` does; the fused
  RMSNorm epilogue writes only the normalized result into `D`).

When a residual-add stage precedes RMSNorm, the chain has two relevant values:

```text
z = gemm_out + residual
y = RMSNorm(z, gamma, eps)
```

The residual-add stage produces the intermediate sum `z`; RMSNorm then consumes `z` and
produces the normalized value `y`. The normalized value `y` remains the main output written
to `D`, while `z` is the updated residual stream that transformer blocks carry forward. The
fused epilogue therefore needs an explicit write-back rule for `z`, independent of the main
`D` output.

## 3. Cross-tile RMSNorm via partial reduction

Because RMSNorm reduces over the full `N` feature dimension of each row, computing it inside a
single GEMM epilogue only works when a whole output row lives in one workgroup, i.e. when the
N-direction macro tile (`MacroTile1`) covers all of `N`. In the transformer shapes this fusion
targets, `N` is much larger than `MacroTile1`, so each row is split across
`ceil(N / MacroTile1)` workgroups and no single workgroup holds the whole row needed to
compute `mean(x^2)`. RMSNorm therefore cannot be a single tile-local epilogue in general; it
is realized as a producer epilogue that emits tile-local partial statistics, plus a
lightweight cross-tile reduction.

This section adopts the GEMM-epilogue reparameterization idea from CODA
([arXiv:2605.19269](https://arxiv.org/abs/2605.19269)): emit tile-local partial statistics in
the first GEMM's epilogue, combine them with a lightweight
reduction, and either apply the per-row scale immediately (full RMSNorm) or defer it into the
epilogue of a *consuming* GEMM (`GEMM -> residual -> RMSNorm -> GEMM`, for example attention
out-projection -> residual -> RMSNorm -> MLP gate/up projection).

### 3.1 Algebraic basis

The RMSNorm reciprocal scale, also commonly called `rstd`, is `r = rsqrt(mean(x^2) + eps)`,
a single scalar per row. Because the following projection is linear and `r` is shared across
the row, the scale commutes with the second GEMM:

```text
y = (r ⊙ ((x @ W0 + z) ⊙ gamma)) @ W1
  = r ⊙ (((x @ W0 + z) ⊙ gamma) @ W1)
```

Here `@` denotes matrix multiplication and `⊙` denotes elementwise multiplication with
broadcasting: `gamma` is broadcast across rows, and `r` is broadcast across columns. Since
`r` has one value per row, it can be applied before or after the second GEMM. This lets the
normalization reduction be split out of the critical dependency between the two GEMMs.

### 3.2 Kernel organization for the two flows

Two flows are built on the partial-reduction structure above:

- **Full RMSNorm** materializes the normalized result. The cross-tile reduction applies the
  per-row scale, and the library hides the whole thing behind one `hipblasLtMatmul` call.
- **Decomposed RMSNorm** never materializes the normalized tensor. The reduction returns the
  per-row scale, and a consuming GEMM applies it in its own epilogue, so the flow spans two
  `hipblasLtMatmul` calls.

Both run the same producer (Kernel 1) followed by the same cross-tile reduction (Kernel 2); they
differ only in whether that reduction applies the per-row scale or returns it.

#### 3.2.1 Kernel 1: the shared producer

The GEMM plus its epilogue:

```text
h0   = x @ W0
h1   = h0 + z              # residual add (tile-local)
h2   = h1 ⊙ gamma         # RMSNorm weight (tile-local)
r_hat = partialRMS(h1)     # tile-local partial sum-of-squares
```

The producer emits two independent results from `h1`:

- `h2 = h1 ⊙ gamma`, the downstream value. The `gamma` multiply is tile-local.
- `r_hat = partialRMS(h1)`, a tile-local partial sum-of-squares over `h1`. This is a raw
  statistic, not a partially normalized output, and it is taken before the `gamma` multiply.

The mean (division by `d`) and the `rsqrt` are deliberately left out of the producer: they can
only be computed once the partials from all tiles in a row are combined, so they happen after
the cross-tile reduction.

Kernel 1 writes one FP32 partial per (token, free-0 macro tile), with ordinary non-atomic stores,
into a partials buffer carved from the tail of the solution workspace.

#### 3.2.2 Full RMSNorm flow (2 kernels)

Kernel 2 is a small separate kernel, hand-written for gfx950 rather than generated with the GEMM,
launched on the same stream immediately after Kernel 1 with one workgroup per token row. It
reduces the partials and applies the row-wise scale, optionally fusing quantization
(section 3.3):

```text
C = h2 ⊙ rsqrt(reduce(r_hat) / d + eps)   # reduce + apply (+ optional quant)
```

The result `C` is `RMSNorm(GEMM)`, and the flow is exposed to the caller as a single `RMSNorm`
stage.

#### 3.2.3 Decomposed flow feeding a GEMM (3 kernels)

Kernel 2 reduces the partials and returns the per-row scale (it does not apply it); Kernel 3 is
the second GEMM whose epilogue applies the deferred scale:

```text
Kernel 2:  r = rsqrt(reduce(r_hat) / d + eps)   # reduce + return r (no apply)
Kernel 3:  h3 = h2 @ W1;   y = r ⊙ h3           # GEMM2 + CODA scale-apply epilogue
```

Only the finalized per-row scale crosses the call boundary, in the opaque handoff descriptor.
Tiling-dependent metadata such as the partial-buffer column count remains internal to the
producer call.

### 3.3 Quantization methods

Both RMSNorm flows can support `GEMM -> residual -> RMSNorm -> fp8/MX quant`, but the
quantization is placed differently: the full flow quantizes the normalized result in Kernel 2,
while the decomposed flow quantizes the producer output in Kernel 1 and still defers `rstd` to
GEMM2.

#### 3.3.1 Full RMSNorm quantization

Kernel 2 already materializes the logical RMSNorm output, so it can requantize that value
directly:

```text
y_i,j = r_i * h2_i,j
q_i,j = round_fp8(y_i,j / s_i(y))
D      = q(y)
```

The scale `s_i(y)` is the dequant scale for the logical RMSNorm output. It may be static or
dynamic, with granularity defined by the requant policy.

#### 3.3.2 MX block requant in the decomposed producer

`HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX` on a partial-RMS producer chain selects K1-local E4M3/UE8M0
requantization, so the producer output can feed an MX-input GEMM2 directly. Relative to the
ordinary decomposed flow of section 3.2.3, only Kernel 1 changes: it block-quantizes `h2` instead
of storing it in BF16/FP16.

```text
Kernel 1:  h2, r_hat as above, then block-quantize h2 as:
             q_e4m3(h2), s = mx_quant(h2)
           write this tile's codes and scales, plus r_hat

Kernel 2:  r = rsqrt(reduce(r_hat) / d + eps)   # reduce + return r (no h2 read)
Kernel 3:  u = MXGEMM(q_e4m3(h2), s, W1)        # K1 output is GEMM2 A + A-scale input
           y = r ⊙ u                           # CODA scale-apply epilogue
```

Kernel 2 still produces the FP32 `rstd` handoff exactly as it does for the ordinary decomposed flow.

**Block scales.** The scale is derived from the tile-local output values while K1 produces `D`; it
is neither a static input scale nor an `AMAX_POINTER` side output. For each block `b` of the
producer output:

```text
s_b = encode_UE8M0(amax(|D_b|) / mxfp8_max)
q   = quant_e4m3(D / s_b)   # every element of block b

D   ≈ s_b * q
```

The block size is configured by `REQUANT_MX_BLOCK_SIZE` (default 32).
`REQUANT_SCALE_COMPUTE_MODE` is not used to choose MX behavior: the K1 MX path always produces its
scale tensor from the output values. The current implementation also does not wire
`REQUANT_AMAX_POINTER` for MX requant.

**Direct handoff to GEMM2.** The producer blocks its output along the axis that becomes GEMM2's
reduction dimension, so each block is exactly one A-side block of the consumer GEMM, and Kernel 1
writes the scale bytes in the swizzled layout that the consumer A-scale mode expects. Kernel 3 can
therefore take the same device `D` buffer as its FP8 A input and the same UE8M0 scale buffer as its
A-side matrix scale, with no host download, dequantization, requantization, or device conversion
pass in between.

### 3.4 Choosing a flow

The caller selects the flow explicitly through the epilogue stages it adds, based on its own
use case; the library never substitutes one flow for the other based on a shape heuristic.

- Add a single `RMSNorm` stage for the full flow when the normalized result is the final
  output of interest. The library realizes it as producer + reduce-and-apply internally.
- Add `partial RMSNorm stats` on the first GEMM and `RMSNorm scale-apply` on the second GEMM
  for the decomposed flow when the RMSNorm output directly feeds another matmul. This removes
  the standalone RMSNorm pass and defers the scale into the consuming GEMM's epilogue.

Both flows run the same producer plus a cross-tile reduction, so launch counts are comparable.
The ordinary decomposed variant writes only the per-row scale `[M]` and folds the scale-apply
into the consuming GEMM's epilogue, avoiding the write and re-read of the full `[M, N]`
normalized tensor that the full flow materializes. The MX producer mode writes E4M3 `q(h2)` plus
UE8M0 scale metadata instead of BF16/FP16 `h2`; GEMM2 consumes that representation directly,
avoiding both the BF16/FP16 activation materialization and a format-conversion pass.

### 3.5 Numerics

Applying `r` after the projection (decomposed flow) changes rounding relative to applying it
before. Partial sums of squares, the reciprocal square root, and the row-wise scale `r`/`rstd`
are kept in FP32 regardless of storage type: the epilogue multiplies by the FP32 `rstd` and only
then casts the result to the requested output type. CODA paper reports this delayed-scale
reparameterization on BF16 Llama-style layers against an FP32 reference and does not observe a
numerical regression. hipBLASLt should still validate both flows and the decomposed variants
against an FP32 reference for the target model shapes and dtypes before enabling them by default.

The MX producer variant quantizes the pre-`rstd` `h2` representation. Its FP8 codes therefore do
not match a per-row dynamic quantization of `rstd * h2`. The remaining numerical differences are
the MX block-scale quantization error and CODA's delayed application of `rstd` after GEMM2
accumulation. This direct producer-to-consumer path should be validated against the target MX
format, scale encoding, saturation, and rounding rules.

## 4. Composable-stage model

RMSNorm is modeled as a discrete epilogue *stage*, configured by its own descriptor
attributes, rather than as another value baked into the combinatorial `hipblasLtEpilogue_t`
enum. Its stage-specific inputs are `gamma` and `eps`. Other epilogue components, such as
residual add, AMax, and requant, can use the same descriptor mechanism with their own
stage-specific attributes instead of new enum combinations. The decomposed cross-tile flow of
section 3 is expressed with the same mechanism, using two additional stages plus an opaque
handoff descriptor.

### 4.1 Stage taxonomy

Epilogue stages can be classified by how they affect the main output tensor and whether they
also produce side outputs. Compatibility and ordering rules are defined per supported chain
family:

| Category              | Main output effect                                           | Stages                                  |
|-----------------------|--------------------------------------------------------------|-----------------------------------------|
| Shape-preserving      | `[M,N] -> [M,N]`                                             | bias, activation, residual add, RMSNorm |
| Side-output reduction | `[M,N] -> [M,N]` plus per-row side output                    | AMax capture, partial RMSNorm stats     |
| Deferred scale        | `[M,N] -> [M,N]` scaled by a per-row vector produced earlier | RMSNorm scale-apply                     |
| Type-changing         | `[M,N] -> [M,N]` with different storage type                 | requant (e.g. FP8)                      |
| Shape-changing        | `[M,2N] -> [M,N]`                                            | GLU / SwiGLU (split-activate-gate)      |

RMSNorm (single-stage) performs an internal row reduction to compute the reciprocal RMS scale,
but its main output remains `[M,N]`. AMax capture records a reduction result as a side output
while leaving the main tensor available to later stages.

The decomposed flow of section 3 adds two stages in this taxonomy:

- **Partial RMSNorm stats** is a side-output reduction producer: its main output is the
  tile-local value `h1 ⊙ gamma` written by GEMM1, and its side output is the per-row RMSNorm
  consumer scale, produced by the internal cross-tile reduction and stored in an opaque handoff
  descriptor (section 5.2). In the MX/block-quantized decomposed variant this RMSNorm handoff
  remains `rstd`; the producer's activation block scales are separate output metadata and are
  not stored in the handoff descriptor.
- **RMSNorm scale-apply** is a deferred-scale consumer: it multiplies the second GEMM's
  accumulator by the per-row scale carried in the handoff descriptor. It performs no reduction
  of its own.

The supported chain families in this design are the full RMSNorm path and the decomposed
RMSNorm path. Shape-changing stages such as GLU are reserved as an orthogonal extension point
and are not part of either ordering rule.

### 4.2 RMSNorm chain order and the legality rule

The library defines a supported order for the single-call RMSNorm chain:

```text
bias -> residual add -> RMSNorm -> AMax -> requant
```

A user-specified RMSNorm chain is **legal** if and only if it is an order-preserving
subsequence of this supported order, with each stage appearing at most once. This one rule
produces the allowed/disallowed examples:

- Allowed: `GEMM + residual + RMSNorm + requant`
  (subsequence of the supported RMSNorm order)
- Disallowed: `GEMM + residual + requant + RMSNorm`
  (requant precedes RMSNorm, violating the supported RMSNorm order)

### 4.3 Decomposed cross-call chains

The decomposed flow is not a single chain on one matmul call; it is two chains on two matmul
calls, linked by an opaque RMSNorm handoff descriptor (section 5.2):

- **Producer chain (GEMM1):** `bias -> residual add -> partial RMSNorm stats`. The supported
  order places the partial-stats stage after residual add, since it reduces the post-residual
  value before `gamma` scaling.
- **Consumer chain (GEMM2):** `RMSNorm scale-apply`. The GEMM2 epilogue only applies the
  deferred per-row scale to complete the normalization on GEMM2's output; it adds no further
  stages.

The MX/block-quantized decomposed variant extends the producer chain to
`bias -> residual add -> partial RMSNorm stats -> requant`. In this chain, `requant` writes
E4M3 codes for `h2` and records the K1-local UE8M0 block scales, while the handoff descriptor
carries `rstd`. The caller passes the producer `D` buffer to GEMM2 as A and the same UE8M0 scale
buffer as GEMM2's A-side matrix scale. The consumer chain remains `RMSNorm scale-apply`: GEMM2
consumes the block scales in its mainloop and applies `rstd` from the handoff descriptor in its
epilogue.

The section 4.2 legality rule applies within each chain independently; the
producer-before-consumer ordering is a data dependency through the handoff descriptor, not a
within-chain rank. A single chain uses either `HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM` or the
decomposed producer/consumer stages, never both.

## 5. API surface

The composable chain is built through an opaque, handle-based builder rather than baking
each combination into the flat `hipblasLtEpilogue_t` enum or carrying an `int32_t[]` token
array on the matmul descriptor. The builder makes the order explicit, keeps stage parameters
attached to the stage that owns them, and gives a single attachment point on the matmul
descriptor.

### 5.1 Fuseable-epilogue enum

```c
typedef enum {
  HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD          = 0, // residual add component
  HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM               = 1, // full RMSNorm (library realizes internally)
  HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS = 2, // GEMM1: producer, emits per-row RMSNorm stats
  HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY   = 3, // GEMM2: apply deferred per-row RMSNorm scale
  HIPBLASLT_FUSEABLE_EPILOGUE_AMAX                  = 4, // capture result AMax as a side output
  HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT               = 5, // requantize result to D's narrow type (e.g. FP8)
  HIPBLASLT_FUSEABLE_EPILOGUE_SWIGLU                = 6, // reserved epilogue family
} hipblasLtFuseableEpilogue_t;
```

These enum values are independent of pipeline order; chain ordering (sections 4.2 and 4.3) is
enforced by an internal rank map, not by the numeric value.
`HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS` and
`HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY` are the decomposed-flow stages of section 3.
When `HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT` follows
`HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS`, it enters the CODA MX/block-quantized
producer mode of section 3.3.2.

Bias is intentionally absent from this enum. It continues to be configured through the
existing matmul descriptor attributes (`HIPBLASLT_MATMUL_DESC_BIAS_POINTER` and
`HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`). When a fused-epilogue chain is attached, the
library applies bias first (if set on the matmul descriptor), then enters the fused chain.
The ordering in sections 4.2 and 4.3 reflects this: bias precedes residual add logically but
is not a member of the builder-managed chain.

### 5.2 Builder handle and functions

```c
typedef struct hipblasLtFusedEpilogueDescriptor* hipblasLtFusedEpilogueDescriptor_t;

hipblasStatus_t hipblasLtFusedEpilogueCreate(hipblasLtFusedEpilogueDescriptor_t* desc);
hipblasStatus_t hipblasLtFusedEpilogueAdd(hipblasLtFusedEpilogueDescriptor_t desc,
                                          hipblasLtFuseableEpilogue_t        epilogue);
hipblasStatus_t hipblasLtFusedEpilogueSetAttribute(hipblasLtFusedEpilogueDescriptor_t desc,
                                                   hipblasLtFusedEpilogueAttribute_t  attr,
                                                   const void*                        value,
                                                   size_t                             sizeInBytes);
hipblasStatus_t hipblasLtFusedEpilogueDestroy(hipblasLtFusedEpilogueDescriptor_t desc);
```

The decomposed flow additionally uses an opaque **RMSNorm handoff descriptor**. Like
`hipblasLtFusedEpilogueDescriptor_t`, it is an opaque handle. The caller supplies its device
buffer, passes the descriptor into the producer and consumer calls, and destroys the descriptor
after both calls.

```c
typedef struct hipblasLtFusedEpilogueRMSNormDescriptor* hipblasLtFusedEpilogueRMSNormDescriptor_t;

hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorCreate(
    hipblasLtFusedEpilogueRMSNormDescriptor_t* desc);
hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(
    hipblasLtFusedEpilogueRMSNormDescriptor_t desc, void* buffer, size_t sizeInBytes);
hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorDestroy(
    hipblasLtFusedEpilogueRMSNormDescriptor_t desc);
```

The producer matmul call runs Kernel 1 plus the cross-tile reduction, and the reduction folds
`1/d` and `eps` into the finalized per-row scale before the call returns. The **only** state that
must cross the API boundary into the consumer call (`RMSNorm scale-apply`) is therefore small:

- the consumer row scale buffer, FP32, tightly packed `[M * batch]` for the initial per-row
  modes (one value per valid output row, per batch).
- a populated/validity token, so a `RMSNorm scale-apply` chain attached to a descriptor that no
  producer has written is rejected at `hipblasLtMatmul` time.

`Add` accumulates stages in call order. The chain is attached to a matmul descriptor with a
single new attribute:

| Attribute                              | Type                                 | Meaning                             |
|----------------------------------------|--------------------------------------|-------------------------------------|
| `HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE` | `hipblasLtFusedEpilogueDescriptor_t` | Attach a built fused-epilogue chain |

The matmul descriptor stores a non-owning pointer to the fused-epilogue handle; the caller
owns the handle. The handle must remain valid for as long as any matmul descriptor references
it, until that descriptor is destroyed or the fused-epilogue attribute is replaced or set to
`NULL`. For each `hipblasLtMatmul` call, a fused-kernel implementation must copy any needed
scalar values and device pointers from the host handle into the launch parameters before
returning; it must not retain the host handle for asynchronous device execution. Once no
descriptor references the handle and all `hipblasLtMatmul` calls that used it have returned,
the handle may be destroyed without waiting for the launched kernels to complete on the
device. A single handle may be shared across multiple matmul descriptors, but must not be
mutated while any referencing matmul call is in-flight on another thread.

For the decomposed flow, the producer chain (GEMM1) and consumer chain (GEMM2) are separate
fused-epilogue handles attached to separate matmul descriptors. The caller is responsible for
issuing the producer matmul before the consumer matmul, and for keeping the RMSNorm handoff
descriptor alive across both.

### 5.3 Stage-specific attributes

Stage parameters are set on the handle (not on the matmul descriptor), so they travel with
the stage that consumes them. They are grouped by stage below.

#### 5.3.1 Residual-add attributes

| Attribute                                          | Type    | Meaning                                                                                        |
|----------------------------------------------------|---------|------------------------------------------------------------------------------------------------|
| `HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER`        | `void*` | Non-null device pointer to the residual input tensor                                           |
| `HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER` | `void*` | Optional device pointer that receives the updated residual stream (`NULL` or unset = in-place) |

The residual input tensor has the same logical shape `[M,N]`, layout, data type, batch
count, and batch stride as `D` unless a future extension adds an explicit residual layout
descriptor. This keeps the initial API narrow and matches the transformer residual-stream
case targeted by the first RMSNorm fusion.

`HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER` is optional. If the attribute is never
set on the handle, or if it is explicitly set to `NULL`, the residual input pointer is also
the write-back target and the updated residual stream is written in place. This lets callers
clear a previously configured separate output pointer without destroying the descriptor. If
it is set to a valid non-null device pointer, the updated residual stream is written there
instead. The output pointer may alias the residual input pointer for explicit in-place
operation. It must not alias `D` when a later stage such as RMSNorm writes a different main
output value to `D`; that invalid alias is rejected by kernel-specific validation once fused
kernels are wired in. The same restriction applies when a later single-call stage such as
requant changes the main output storage value written to `D`.

#### 5.3.2 RMSNorm attributes

These attributes apply to full RMSNorm and the decomposed producer/consumer stages.

| Attribute                                | Type                                        | Stage                       | Meaning                                                                                    |
|------------------------------------------|---------------------------------------------|-----------------------------|--------------------------------------------------------------------------------------------|
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA` | `void*`                                     | RMSNorm / partial stats     | Non-null device pointer to gamma, length `N`                                               |
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS`   | `float`                                     | RMSNorm / partial stats     | Epsilon inside the rsqrt                                                                    |
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS` | `hipblasLtFusedEpilogueRMSNormDescriptor_t` | partial stats / scale apply | Opaque handoff object with a caller-owned FP32 buffer; producer writes it and consumer reads it |

For the full flow, `gamma` and `eps` are set on the single `RMSNorm` handle and no handoff
descriptor is created: the library derives `1/d` and the tiling metadata from the selected
solution and keeps the `partialBuf` scratch in the matmul preference workspace for the duration
of the single call. The reduction applies the per-row scale to `D` in place, so no per-row-scale
buffer is materialized.

For the decomposed flow, `gamma` and `eps` are set on the producer handle: `gamma` is applied
tile-locally in Kernel 1, and `eps` (together with `1/d`, which the library derives from the
problem shape) is consumed by the internal reduction that finalizes the per-row scale. The
handoff descriptor is set on both the producer and consumer handles and must refer to the same
object. The descriptor carries the finalized RMSNorm row scale `rstd`. When the producer also
has an MX/block requant stage, that stage's scale pointer carries the activation block-scale
tensor separately. The caller retains that pointer and supplies it to GEMM2 through the A-side
matrix-scale attributes; it is intentionally not normalization state stored in the handoff
descriptor. The partial sum-of-squares scratch stays in the matmul preference workspace and is
consumed by the reduction inside the producer call.

#### 5.3.3 Requant attributes

| Attribute                                             | Type                                 | Meaning                                                                          |
|-------------------------------------------------------|--------------------------------------|----------------------------------------------------------------------------------|
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER`      | `void*` (f32)                        | Ordinary FP8 dequant scale: input in static mode or output in dynamic mode       |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_AMAX_POINTER`       | `void*` (f32)                        | Ordinary FP8 optional amax side output                                           |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE` | `hipblasLtRequantScaleComputeMode_t` | Ordinary FP8 static vs dynamic policy (default: static)                          |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY`  | `hipblasLtRequantScaleGranularity_t` | Ordinary scale shape or `PER_BLOCK_MX`                                           |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER`   | `void*` (UE8M0 bytes)                | Required output buffer for MX block scales                                       |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE`      | `int32_t`                            | Elements per MX block along producer `D`'s hidden/free-0 dimension (default: 32) |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE`     | `hipDataType`                        | MX output type; currently only `HIP_R_8F_E4M3` is accepted                       |

##### Requant policy

The motivating model path is:

```text
RMSNorm output (BF16/FP16) -> requantize to FP8 activation -> next GEMM
```

Observed model paths motivate scalar scale support for different compute modes, for example,
Mixtral FP8 uses a checkpoint-provided scalar f32 scale (`STATIC` + `PER_TENSOR`), while
Llama3 FP8 uses a scalar scale derived from the result amax (`DYNAMIC_FROM_AMAX` + `PER_TENSOR`).

The ordinary FP8 public scale convention is a **dequant scale**, matching the scale carried with
observed FP8 activations and consumed by the next GEMM:

```text
x_approx = quantized_value * dequant_scale
```

The producer derives the reciprocal quant multiplier internally, so no public scale-direction
attribute is needed.

The requant attributes define the policy that is not captured by the stage enum:

- `SCALE_POINTER`, `SCALE_COMPUTE_MODE`, and `AMAX_POINTER` describe the ordinary FP8 API.
  The currently wired full-RMSNorm FP8 requant path supports a static, per-tensor scale only;
  dynamic-scale and requant-amax outputs are not wired in this path.
- `SCALE_GRANULARITY = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX` selects the MX path.
- `REQUANT_MX_SCALE_POINTER` is required for that path and receives the K1-produced UE8M0 scale
  bytes. MX requant does not consume `SCALE_POINTER`, does not honor `SCALE_COMPUTE_MODE`, and
  does not currently write `AMAX_POINTER`.
- `REQUANT_MX_BLOCK_SIZE` must be positive. The initial gfx950 logic and tests use block size 32.

##### FP8/MX variant and architecture compatibility

For MX requant, the fused-epilogue attribute `REQUANT_MX_OUTPUT_TYPE` must be
`HIP_R_8F_E4M3`; this is the only supported MX output element type. The output matrix layout must
be compatible with the selected E4M3 solution.

ROCm's MI300/MI350 workload guidance calls out an architecture-visible format difference:
MI300 Series (`gfx942`) uses the FNUZ FP8 variants, while MI350 Series (`gfx950`) uses the OCP
FP8 variants. The initial MI350 fused-requant and CODA paths should therefore target the OCP
HIP data types:

```text
HIP_R_8F_E4M3
HIP_R_8F_E5M2
```

and should not silently reinterpret FNUZ tensors:

```text
HIP_R_8F_E4M3_FNUZ
HIP_R_8F_E5M2_FNUZ
```

as OCP tensors, or vice versa. The encodings have different value sets and special-value
semantics, so the scale alone is not sufficient to make a FNUZ checkpoint activation compatible
with an OCP GEMM. A model or inference engine that stores FNUZ FP8 activations must either use a
matching MI300/FNUZ kernel path or explicitly convert the checkpoint/runtime tensors and scales
to the OCP representation expected by MI350. Conversely, vLLM quantized models that already
target OCP E4M3/E5M2 should use the OCP HIP data types end to end.

The `fp8_max`/`mxfp8_max` constants in section 3.3 are maximum finite magnitudes of the selected
element format, not global FP8 constants. Requant kernels derive their quant multiplier,
saturation threshold, rounding behavior, and scale encoding from the selected format.

##### CODA MX/block-quantized producer mode

A producer chain that contains both `HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS` and
`HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT` with `PER_BLOCK_MX` selects this mode. It splits the
producer's results across three separate outputs:

```text
producer D      = q_e4m3(h2)               # E4M3 codes, 32x1 hidden/K blocks
MX scale tensor = UE8M0 producer output    # pre-swizzled for the consumer A-scale mode
handoff scale   = rstd                     # FP32 per row, from Kernel 2
```

On the consumer call the caller attaches `RMSNORM_SCALE_APPLY`, passes the producer `D` pointer as
GEMM2's A, and passes the same UE8M0 scale tensor through
`HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER` with the matching A-scale mode, initially
`HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT`.

##### Relationship to the existing matmul-descriptor scale/amax attributes

The stage-level attributes intentionally mirror existing matmul-descriptor concepts:
`HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER` and `HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER`. They are kept
on the fused-epilogue descriptor because requant is part of the epilogue chain and needs its own
policy (`SCALE_COMPUTE_MODE`, `SCALE_GRANULARITY`). The implementation should reuse the existing
D-scale / AMax-D plumbing where the policy matches, rather than introduce a separate low-level
path.

### 5.4 Usage sketch

#### 5.4.1 Full RMSNorm

Full RMSNorm, exposed as a single `hipblasLtMatmul` call (the library runs the producer and
the reduce-and-apply internally). Note there is no handoff descriptor: the internal producer ->
reduce-and-apply state lives in the preference workspace for the duration of the call
(section 5.2):

```c
hipblasLtFusedEpilogueDescriptor_t fused;
hipblasLtFusedEpilogueCreate(&fused);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
                                   &residual, sizeof(residual));
// Optional: omit this attribute to update `residual` in place.
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER,
                                   &residual_out, sizeof(residual_out));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                                   &gamma, sizeof(gamma));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                                   &eps, sizeof(eps));
hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &fused, sizeof(fused));
// ... hipblasLtMatmul(...) ...
// Internally launches GEMM producer + auxiliary reduce-and-apply kernel.
hipblasLtFusedEpilogueDestroy(fused);
```

#### 5.4.2 Full RMSNorm + requant

Full RMSNorm followed by static/dynamic per-tensor FP8 requant, capturing the derived dequant scale
(which can feed a consuming GEMM as its `A_SCALE_POINTER`) and, optionally, the amax:

```c
hipblasLtFusedEpilogueDescriptor_t fused;
hipblasLtFusedEpilogueCreate(&fused);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM);
hipblasLtFusedEpilogueAdd(fused, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT);
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                                   &gamma, sizeof(gamma));
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                                   &eps, sizeof(eps));
// The default is STATIC + PER_TENSOR; opt into dynamic scale derivation explicitly.
hipblasLtRequantScaleComputeMode_t mode = HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX;
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE,
                                   &mode, sizeof(mode));
// Dynamic mode writes the derived scale here (f32[1] for per-tensor).
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER,
                                   &d_scale, sizeof(d_scale));
// Optional amax side output (f32[1] for per-tensor).
hipblasLtFusedEpilogueSetAttribute(fused, HIPBLASLT_FUSED_EPILOGUE_REQUANT_AMAX_POINTER,
                                   &d_amax, sizeof(d_amax));
hipblasLtMatmulDescSetAttribute(matmulDesc, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &fused, sizeof(fused));
// ... hipblasLtMatmul(...) writes FP8 D ...
hipblasLtFusedEpilogueDestroy(fused);
```

#### 5.4.3 Decomposed RMSNorm handoff

Decomposed flow for `GEMM -> residual -> RMSNorm -> GEMM`, using the opaque handoff descriptor.
The optional MX producer mode adds `REQUANT` to GEMM1. Its 32x1 scale layout is a direct GEMM2
A-side representation; see section 5.3.3.

```c
// Handoff object shared by the producer and consumer matmul calls.
hipblasLtFusedEpilogueRMSNormDescriptor_t stats;
hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats);
// Caller-owned device storage: one FP32 scale for each producer output row and batch.
// Allocate it before stream capture when the matmul pair is captured in a HIP graph.
size_t stats_bytes = M * batch_count * sizeof(float);
void* d_stats;
hipMalloc(&d_stats, stats_bytes);
hipblasLtFusedEpilogueRMSNormDescriptorSetBuffer(stats, d_stats, stats_bytes);

// GEMM1 producer: residual add + gamma + partial RMSNorm stats.
hipblasLtFusedEpilogueDescriptor_t prod;
hipblasLtFusedEpilogueCreate(&prod);
hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_RESIDUAL_ADD);
hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS);
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_POINTER,
                                   &residual, sizeof(residual));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_GAMMA,
                                   &gamma, sizeof(gamma));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_EPS,
                                   &eps, sizeof(eps));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS,
                                   &stats, sizeof(stats));

// Optional MX/block-quantized decomposed variant:
//   - matD1 has HIP_R_8F_E4M3 element type and receives q_e4m3(h2), not q_e4m3(rstd * h2)
//   - stats carries rstd_i
//   - d_h2_mx_scale receives K1's UE8M0 producer scale tensor
// Omit this block for the ordinary BF16/FP16 decomposed flow, where matD1 receives h2
// and stats carries only rstd.
hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT);
hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_BLOCK_MX;
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                   &gran, sizeof(gran));
// UE8M0 scale tensor written by the producer.
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_SCALE_POINTER,
                                   &d_h2_mx_scale, sizeof(d_h2_mx_scale));
int32_t mx_block_size = 32;
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_BLOCK_SIZE,
                                   &mx_block_size, sizeof(mx_block_size));
hipDataType mx_output_type = HIP_R_8F_E4M3;
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_MX_OUTPUT_TYPE,
                                   &mx_output_type, sizeof(mx_output_type));
hipblasLtMatmulDescSetAttribute(matmulDesc1, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &prod, sizeof(prod));
// ... hipblasLtMatmul(GEMM1) ...
// Internally launches GEMM1 producer + auxiliary reduce-and-return kernel.
// After the queued work completes, `stats` carries rstd.

// GEMM2 consumer: apply the deferred per-row scale from `stats` in the epilogue.
// For ordinary decomposed flow, A points directly at matD1's BF16/FP16 h2 output.
// For the MX variant, A points directly at matD1's E4M3 q_e4m3(h2) output and
// d_h2_mx_scale is supplied as GEMM2's A-side matrix scale tensor with the matching
// 32x1 hidden/K-block scale mode.
hipblasLtFusedEpilogueDescriptor_t cons;
hipblasLtFusedEpilogueCreate(&cons);
hipblasLtFusedEpilogueAdd(cons, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY);
hipblasLtFusedEpilogueSetAttribute(cons, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS,
                                   &stats, sizeof(stats)); // same object as the producer
hipblasLtMatmulDescSetAttribute(matmulDesc2, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &cons, sizeof(cons));
// MX direct handoff: matA2 points at matD1, and the same K1-produced scale tensor is used
// as GEMM2's A scale. The K1 producer emits 32x1 hidden/K blocks in this exact layout.
hipblasLtMatmulMatrixScale_t a_scale_mode =
    HIPBLASLT_MATMUL_MATRIX_SCALE_BLK32_UE8M0_32_8_EXT;
hipblasLtMatmulDescSetAttribute(matmulDesc2, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                &a_scale_mode, sizeof(a_scale_mode));
hipblasLtMatmulDescSetAttribute(matmulDesc2, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                &d_h2_mx_scale, sizeof(d_h2_mx_scale));
// ... hipblasLtMatmul(GEMM2) ...
// launches GEMM2; its epilogue reads `stats`

hipblasLtFusedEpilogueDestroy(prod);
hipblasLtFusedEpilogueDestroy(cons);
hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
hipFree(d_stats);
```

### 5.5 Datatype requirements

The initial release targets BF16/FP16 storage for the RMSNorm input and normalized output. The
residual input and residual-output write-back use the same storage type as the residual-add
sum. The `gamma` vector must use the same storage type as the RMSNorm input/pre-requant value,
not necessarily the final `D` storage type when a later single-call requant stage changes
the main output type. Regardless of the BF16/FP16 storage type, RMSNorm accumulation uses FP32
for the sum of squares and the reciprocal square root.

For the decomposed flow, the internal partial-stats scratch and the per-row scale carried in
the handoff descriptor are FP32, independent of the GEMM storage types, so the cross-tile
combine and the deferred scale match the full-flow accumulation precision.

For the MX/block-quantized producer variant, `D` uses `HIP_R_8F_E4M3`; no other MX output type is
accepted. The source value for quantization is still `h2`, so `gamma` has the same storage type as
the pre-requant producer output. The RMSNorm handoff scale `rstd` remains FP32, while producer
block scales are UE8M0 bytes in the producer's gfx950 A-compatible layout. GEMM2 uses the same
E4M3 buffer and UE8M0 scale tensor as its A input and A-side scale metadata, then applies `rstd`
through `RMSNorm scale-apply`.

### 5.6 Strided-batched semantics

When `batch_count > 1` in a strided-batched GEMM:

- `gamma` is broadcast across batches (a single vector of length `N` shared by all batches).
- `eps` is a scalar and is always shared across batches.
- The residual input tensor follows the same batch stride as `D`. If
  `HIPBLASLT_FUSED_EPILOGUE_RESIDUAL_OUTPUT_POINTER` is set, the residual output also uses
  the same batch stride as `D`.
- For the decomposed flow, the per-row scale is indexed per `(row, batch)` inside the handoff
  descriptor. The per-batch row count `M` and the batch count are derived from the matmul
  descriptor, so the producer and consumer stay consistent without any caller-supplied layout.

## 6. Error conditions and return codes

| Condition                                                                            | Return code                    | Detected at                                 |
|--------------------------------------------------------------------------------------|--------------------------------|---------------------------------------------|
| Unrecognized or unsupported `hipblasLtFuseableEpilogue_t` passed to `Add`            | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| `Add` breaks the supported RMSNorm order (out-of-order stage)                        | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Duplicate stage added                                                                | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Full and decomposed RMSNorm stages mixed in one chain                                | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Unknown attribute passed to `SetAttribute`                                           | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time                         |
| Required pointer attribute explicitly set to `NULL`                                  | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time                         |
| Residual-add stage present but residual pointer unset                                | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| RMSNorm stage present but `gamma` or `eps` unset                                     | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| Partial-stats or scale-apply stage present but stats descriptor unset                | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| Ordinary requant stage present but f32 scale pointer unset                           | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| MX requant missing its UE8M0 scale pointer, positive block size, or E4M3 output type | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| Requant scale compute mode or granularity is outside the enum range                  | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time / attach                |
| CODA requant producer is missing the RMSNorm handoff descriptor                      | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| Requested narrow/MX variant is not supported natively by the selected GPU            | `HIPBLAS_STATUS_NOT_SUPPORTED` | heuristic / `hipblasLtMatmul`               |
| Scale-apply consumes a stats descriptor not populated by a producer                  | `HIPBLAS_STATUS_INVALID_VALUE` | `hipblasLtMatmul`                           |
| Attached fused epilogue without a matching kernel implementation                     | `HIPBLAS_STATUS_NOT_SUPPORTED` | `hipblasLtMatmul`                           |

The key requirement: an illegal *ordering* for a supported RMSNorm chain is rejected at the
API-call level, incrementally as stages are added, before any codegen or kernel launch.
Residual completeness (`residual`), RMSNorm completeness (`gamma` and `eps`), requant
completeness (an ordinary f32 scale, or the MX scale pointer/block size/E4M3 attributes), and
decomposed-flow completeness (a set stats descriptor) are validated when the handle is attached to
the matmul descriptor via
`HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE`. This means
required stage-specific attributes must be set on the handle before attachment and must not be
cleared or mutated after attachment. The matmul descriptor stores a non-owning pointer to the
handle, so later optional-attribute updates are visible to `hipblasLtMatmul`, but callers
should not rely on late updates to satisfy attach-time required-attribute validation. For the
decomposed flow, the producer-before-consumer dependency on the handoff descriptor can only be
fully validated at `hipblasLtMatmul` time.

For MX requant, attachment validates only the MX-specific producer attributes:
`PER_BLOCK_MX`, a non-null MX scale output pointer, a positive block size, and E4M3 output type.
It does not require `DYNAMIC_FROM_AMAX` or inspect the ordinary f32 scale/amax attributes. The
consumer validates that the handoff has been populated before using `rstd`. The producer and
consumer must use matching E4M3 and A-scale layouts; this cross-call compatibility is not carried
by the handoff descriptor and is validated by the selected GEMM2 solution.

## 7. Interaction with existing descriptors and preferences

- A new opaque builder object (`hipblasLtFusedEpilogueDescriptor_t`) is introduced; it is
  attached to the existing `hipblasLtMatmulDesc_t` via a single attribute. The matmul
  descriptor stores only a non-owning pointer. The decomposed flow adds one more opaque handle
  type (`hipblasLtFusedEpilogueRMSNormDescriptor_t`) that carries a caller-owned device buffer
  across calls without exposing its fields.
- `hipblasLtMatmulPreference_t` is unchanged. Workspace and SM-count hints apply as usual.
  In the full flow, `partialBuf` and any synchronizer/flag buffer are transient and drawn from the
  matmul preference workspace for the single call; Kernel 2 applies the row scale to `D` in place,
  so no per-row-scale buffer is materialized. In the decomposed flow, the caller supplies the
  persistent FP32 consumer-scale buffer through the handoff descriptor. The optional
  MX/block-quantized flow also writes a producer scale tensor through
  `REQUANT_MX_SCALE_POINTER`. That tensor is activation quantization metadata; the caller
  supplies the same tensor to GEMM2 as its A-side matrix scale. The producer's `partialBuf`
  scratch and any synchronizer/flag buffer remain internal, transient workspace sized through
  the workspace-size query.
- On the codegen side, the ordinary GEMM kernels for both flows come from a single
  TensileLite option (`FusedEpilogues = 1`) that bundles the optional-residual, RMSNorm-producer, and
  RMSNorm-scale-apply epilogue options and selects among them at runtime. The MX/block-quantized
  producer uses a K1 epilogue that computes local block amax, encodes the UE8M0 scale tensor, and
  stores E4M3 `q(h2)` in 32x1 hidden/K blocks. GEMM2 consumes this output directly as its MX A
  operand and uses the same scale tensor with `RMSNorm scale-apply`. The cross-tile reduction
  remains a separate custom kernel.
- The C++ extension API (`hipblaslt_ext::GemmEpilogue` / `GemmInputs`) is outside this
  document; the C handle API is the defined surface.

## 8. SwiGLU/GLU extensibility

GLU/SwiGLU is a separate epilogue family, not part of the RMSNorm chain order in section 4.2.
It can reuse the same fused-epilogue descriptor handle, `Add`/`SetAttribute` builder flow,
matmul-descriptor attachment point, and stage-specific attribute model. It still requires
its own validation because it is shape-changing: the GEMM produces `[M, 2N]` for the packed
gate/up projection, while the GLU stage emits the final `[M, N]` tensor.
