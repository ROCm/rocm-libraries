# Fused Epilogue Extensions: RMSNorm and Composable Epilogue Chains

This document specifies the hipBLASLt API extensions for fused epilogues, starting with
RMSNorm, and the composable epilogue-chain mechanism that lets RMSNorm compose with
optional residual add, AMax capture, and output requantization (e.g. FP8). It also reserves
the design points required so that Gated Linear Units (SwiGLU/GeGLU/ReGLU) can be added later
without reworking the API.

## 1. Motivation and scope

The public hipBLASLt matmul computes:

```
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

```
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

```
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

This section adopts the GEMM-epilogue reparameterization idea from CODA ([arXiv:2605.19269](https://arxiv.org/abs/2605.19269)):
emit tile-local partial statistics in the first GEMM's epilogue, combine them with a lightweight
reduction, and either apply the per-row scale immediately (full RMSNorm) or defer it into the
epilogue of a *consuming* GEMM (`GEMM -> residual -> RMSNorm -> GEMM`, for example attention
out-projection -> residual -> RMSNorm -> MLP gate/up projection).

### 3.1 Algebraic basis

The RMSNorm reciprocal scale, also commonly called `rstd`, is `r = rsqrt(mean(x^2) + eps)`,
a single scalar per row. Because the following projection is linear and `r` is shared across
the row, the scale commutes with the second GEMM:

```
y = (r ⊙ ((x @ W0 + z) ⊙ gamma)) @ W1
  = r ⊙ (((x @ W0 + z) ⊙ gamma) @ W1)
```

Here `@` denotes matrix multiplication and `⊙` denotes elementwise multiplication with
broadcasting: `gamma` is broadcast across rows, and `r` is broadcast across columns. Since
`r` has one value per row, it can be applied before or after the second GEMM. This lets the
normalization reduction be split out of the critical dependency between the two GEMMs.

### 3.2 Kernel organization for the two flows

Both flows share the same GEMM producer. They differ in what happens after the cross-tile
reduction: the full flow applies the scale immediately, while the decomposed flow returns the
consumer scale for a subsequent GEMM to apply. In the ordinary decomposed variant that consumer
scale is `rstd`; in the dynamic-quantized variant it is the composed scale needed by the
subsequent GEMM.

**Kernel 1 (producer, shared by both flows).** The GEMM plus its epilogue:

```
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

**Full RMSNorm flow (2 kernels).** Kernel 2 reduces the partials and applies the row-wise
scale, optionally fusing quantization:

```
C = h2 ⊙ rsqrt(reduce(r_hat) / d + eps)   # reduce + apply (+ optional quant)
```

The result `C` is `RMSNorm(GEMM)`. This flow is exposed as a single `RMSNorm` stage and hides
both kernels behind one `hipblasLtMatmul` call.

**Decomposed flow feeding a GEMM (3 kernels).** Kernel 2 reduces the partials and returns the
per-row scale (it does not apply it); Kernel 3 is the second GEMM whose epilogue applies the
deferred scale:

```
Kernel 2:  r = rsqrt(reduce(r_hat) / d + eps)   # reduce + return r (no apply)
Kernel 3:  h3 = h2 @ W1;   y = r ⊙ h3           # GEMM2 + CODA scale-apply epilogue
```

Kernel 1 and the GEMM in Kernel 3 are both GEMM kernels produced by the single codegen option,
selecting different epilogue options. The reduction (Kernel 2) is a small custom kernel; it does
slightly different work in the two flows (apply vs return). In the ordinary decomposed flow, only
the finalized per-row scale crosses the call boundary in an opaque descriptor; tiling-dependent
metadata such as the partial-buffer column count remains internal to the producer call.

### 3.3 Quantization methods

Both RMSNorm flows can support `GEMM -> residual -> RMSNorm -> fp8 quant`, but the quantization
is placed differently.

**Full RMSNorm quantization.** Kernel 2 already materializes the logical RMSNorm output, so it
can requantize that value directly:

```
y_i,j = r_i * h2_i,j
q_i,j = round_fp8(y_i,j / s_i(y))
D      = q(y)
```

The scale `s_i(y)` is the dequant scale for the logical RMSNorm output. It may be static or
dynamic, with granularity defined by the requant policy.

**Decomposed dynamic quantization.** The decomposed flow does not materialize `y = r * h2`
before GEMM2. It can still represent dynamic per-row quantization because the RMSNorm scalar
commutes into the dynamic scale. Here `r_i` is the per-row RMSNorm reciprocal scale (`rstd`) and
`fp8_max` is the maximum finite magnitude of the selected FP8 format:

```
s_i(h2) = amax_j(|h2_i,j|) / fp8_max
s_i(y)  = amax_j(|r_i * h2_i,j|) / fp8_max
        = r_i * amax_j(|h2_i,j|) / fp8_max
        = r_i * s_i(h2)

q_i,j   = quant(y_i,j, s_i(y))
        = round((r_i * h2_i,j) / (r_i * s_i(h2)))
        = round(h2_i,j / s_i(h2))
rho_i   = r_i * s_i(h2)                    # handoff scale, equal to s_i(y)
```

The producer writes FP8 `q(h2)` and the handoff carries `rho_i`. Numerically `rho_i` is the same
logical RMSNorm-output dequant scale as `s_i(y)`, but the separate name marks its API role: it is
the scale GEMM2 consumes to finish the delayed-scale computation.

```
GEMM2 raw:  u_i = q_i @ W1
epilogue:  out_i = rho_i * u_i
```

The initial decomposed quantized target is `DYNAMIC_FROM_AMAX + PER_ROW`. Dynamic block/MX
scales can use the same idea when each block is contained within one RMSNorm row. Static
final-output scales do not have this cancellation: for fixed `S`,
`round_fp8((r_i * h2_i,j) / S)` cannot generally be recovered from
`round_fp8(h2_i,j / S)` by a later epilogue multiply.

### 3.4 Choosing a flow

The caller selects the flow explicitly through the epilogue stages it adds, based on its own
use case; the library never substitutes one flow for the other based on a shape heuristic.

- Add a single `RMSNorm` stage for the full flow when the normalized result is the final
  output of interest. The library realizes it as producer + reduce-and-apply internally.
- Add `partial RMSNorm stats` on the first GEMM and `RMSNorm scale-apply` on the second GEMM
  for the decomposed flow when the RMSNorm output directly feeds another matmul. This removes
  the standalone RMSNorm pass and defers the scale into the consuming GEMM's epilogue.

Within the decomposed flow, adding `requant` after `partial RMSNorm stats` selects the
dynamic-quantized variant described in section 3.3. The producer writes FP8 codes for `h2` and
carries the composed row scale `rho = rstd * dequant_scale(h2)` to the consuming GEMM.

Both flows run the same producer plus a cross-tile reduction, so launch counts are comparable.
The ordinary decomposed variant writes only the per-row scale `[M]` and folds the scale-apply
into the consuming GEMM's epilogue, avoiding the write and re-read of the full `[M, N]`
normalized tensor that the full flow materializes. The dynamic-quantized decomposed variant
materializes FP8 `q(h2)` but avoids materializing the BF16/FP16 RMSNorm output `rstd * h2`.
Those bandwidth savings grow with `M` and `N`.

### 3.5 Reduction backend (internal)

The "combine partials -> rstd" step is the same class of cross-workgroup reduction that
Stream-K already performs to combine partial output tiles, so its workspace, flag/synchronizer
handshake, and device-scope fences can be reused. The combine is realized as a custom kernel
and its backend (for example atomic accumulation, an in-kernel tree fixup, or a separate
reduction kernel) is an internal implementation choice, not a user-selected option.

### 3.6 Numerics

Applying `r` after the projection (decomposed flow) changes rounding relative to applying it
before. Partial sums of squares, the reciprocal square root, and the row-wise scale `r`/`rstd`
are kept in FP32 regardless of storage type: the epilogue multiplies by the FP32 `rstd` and only
then casts the result to the requested output type. CODA paper reports this delayed-scale
reparameterization on BF16 Llama-style layers against an FP32 reference and does not observe a
numerical regression. hipBLASLt should still validate both flows and the decomposed variants
against an FP32 reference for the target model shapes and dtypes before enabling them by default.

The decomposed dynamic quantization variant changes the rounding point from
`round_fp8(r * h2 / s(y))` to the algebraically identical `round_fp8(h2 / s(h2))` for per-row
dynamic scale. The FP8 codes match the materialized `RMSNorm -> quant` path under the same amax,
scale, saturation, and rounding rules. The remaining numerical differences are the same CODA
differences as above: the composed row scale `rho` is applied to GEMM2's accumulated result
instead of dequantizing the input before the multiply. This should be validated for target models,
especially when block/MX scales, scale-format quantization, or non-per-row scale domains are
enabled.

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
  descriptor (section 5.2). In the dynamic-quantized decomposed variant this scale includes the
  activation dequant scale as well as `rstd`.
- **RMSNorm scale-apply** is a deferred-scale consumer: it multiplies the second GEMM's
  accumulator by the per-row scale carried in the handoff descriptor. It performs no reduction
  of its own.

The supported chain families in this design are the full RMSNorm path and the decomposed
RMSNorm path. Shape-changing stages such as GLU are reserved as an orthogonal extension point
and are not part of either ordering rule.

### 4.2 RMSNorm chain order and the legality rule

The library defines a supported order for the single-call RMSNorm chain:

```
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

The dynamic-quantized decomposed variant extends the producer chain to
`bias -> residual add -> partial RMSNorm stats -> requant`. In this chain, `requant` writes
the FP8 codes for `h2`, while the handoff descriptor carries the composed row scale
`rho = rstd * dequant_scale(h2)` needed by GEMM2. The consumer chain remains
`RMSNorm scale-apply`: it reads the same handoff descriptor and applies `rho` to GEMM2's
accumulator. This is not a normal output requant on GEMM2; it is the first half of
`GEMM -> residual -> RMSNorm -> fp8 quant` plus a delayed scale that lets GEMM2 finish the
logical computation.

Static activation quantization for the consuming GEMM, or dynamic quantization with a scale
domain that does not share one RMSNorm factor, cannot use this exact chain transformation.
Those cases either use the full-RMSNorm path that materializes the quantized activation, or a
future kernel that explicitly applies `rstd` before quantizing.

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
They are attached to two different matmul descriptors and are linked by the opaque RMSNorm
handoff descriptor. When `HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT` follows
`HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS`, it enters the CODA dynamic-quantized
producer mode of section 3.3: the main producer output is the FP8 code for `h2`, and the
handoff descriptor carries the composed row scale consumed by the second GEMM.

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
`hipblasLtFusedEpilogueDescriptor_t`, it is an opaque handle, but it is library-populated rather
than user-configured: the caller only creates it, passes it into the producer and consumer calls,
and destroys it, never setting or reading its fields directly.

```c
typedef struct hipblasLtFusedEpilogueRMSNormDescriptor* hipblasLtFusedEpilogueRMSNormDescriptor_t;

hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorCreate(
    hipblasLtFusedEpilogueRMSNormDescriptor_t* desc);
hipblasStatus_t hipblasLtFusedEpilogueRMSNormDescriptorDestroy(
    hipblasLtFusedEpilogueRMSNormDescriptor_t desc);
```

The producer matmul call runs Kernel 1 plus the cross-tile reduction, and the reduction folds
`1/d` and `eps` into the finalized per-row scale before the call returns. The **only** state that
must cross the API boundary into the consumer call (`RMSNorm scale-apply`) is therefore small:

- the consumer row scale buffer, FP32, tightly packed `[M * batch]` for the initial per-row
  modes (one value per valid output row, per batch). In the unquantized decomposed flow this
  value is `rstd`; in the dynamic-quantized decomposed variant it is
  `rho = rstd * dequant_scale(h2)`.
- a populated/validity token, so a `RMSNorm scale-apply` chain attached to a descriptor that no
  producer has written is rejected at `hipblasLtMatmul` time.
- optional quantization metadata describing whether the producer output is an unquantized `h2`
  tensor or an FP8 `q(h2)` tensor, and which scale granularity the consumer row scale uses.

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
| `HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS` | `hipblasLtFusedEpilogueRMSNormDescriptor_t` | partial stats / scale apply | Opaque handoff object; producer writes it, consumer reads it (same object on both handles) |

For the full flow, `gamma` and `eps` are set on the single `RMSNorm` handle and no handoff
descriptor is created: the library derives `1/d` and the tiling metadata from the selected
solution and keeps the `partialBuf` scratch in the matmul preference workspace for the duration
of the single call. The reduction applies the per-row scale to `D` in place, so no per-row-scale
buffer is materialized.

For the decomposed flow, `gamma` and `eps` are set on the producer handle: `gamma` is applied
tile-locally in Kernel 1, and `eps` (together with `1/d`, which the library derives from the
problem shape) is consumed by the internal reduction that finalizes the per-row scale. The
handoff descriptor is set on both the producer and consumer handles and must refer to the same
object. The descriptor carries the finalized consumer row scale: `rstd` for the unquantized
decomposed flow, or the composed CODA scale `rho = rstd * dequant_scale(h2)` when the producer
also has a dynamic requant stage. The partial sum-of-squares scratch stays in the matmul
preference workspace and is consumed by the reduction inside the producer call.

#### 5.3.3 Requant attributes

| Attribute                                             | Type                                    | Meaning                                                                                                                            |
|-------------------------------------------------------|-----------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER`      | `void*` (f32)                           | Device pointer to the dequant scale; read-only input in static mode, written in dynamic mode. Element count follows the granularity |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_AMAX_POINTER`       | `void*` (f32)                           | Optional device pointer receiving the result amax side output, same granularity as the scale (`NULL`/unset = no amax written)      |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE` | `hipblasLtRequantScaleComputeMode_t`    | Static caller-provided scale vs dynamic-from-amax (default: static)                                                                |
| `HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY`  | `hipblasLtRequantScaleGranularity_t`    | Scale/amax shape: per-tensor, per-row, or block (default: per-tensor)                                                              |

##### Requant policy

The motivating model path is:

```text
RMSNorm output (BF16/FP16) -> requantize to FP8 activation -> next GEMM
```

Observed model paths motivate scalar scale support for different compute modes, for example,
Mixtral FP8 uses a checkpoint-provided scalar f32 scale (`STATIC` + `PER_TENSOR`), while
Llama3 FP8 uses a scalar scale derived from the result amax (`DYNAMIC_FROM_AMAX` + `PER_TENSOR`).

The public scale convention is always a **dequant scale**, matching the scale carried with the
observed FP8 activations and consumed by the next GEMM:

```text
x_approx = quantized_value * dequant_scale
```

This scale is passed forward to `A_SCALE_POINTER` / `B_SCALE_POINTER`; the producer derives the
reciprocal quant multiplier internally, so no public scale-direction attribute is needed.

The requant attributes define the policy that is not captured by the stage enum:

- `SCALE_POINTER` supplies the dequant scale in `STATIC` mode and receives the derived scale in
  `DYNAMIC_FROM_AMAX` mode.
- `SCALE_COMPUTE_MODE` separates checkpoint-provided static scales from dynamic scales derived
  from the result amax.
- `SCALE_GRANULARITY` records the scale/amax shape: `PER_TENSOR` (`[1]`) for the observed scalar
  policies, `PER_ROW` (`[M]`) for per-token policies, and block modes for finer-grained schemes.
- `AMAX_POINTER` is an optional f32 side output with the same granularity as the scale. It is
  not needed for a static-scale path, but is useful when the caller wants the raw result amax
  for dynamic scaling or delayed-scaling bookkeeping.

##### FP8 variant and architecture compatibility

The `REQUANT` stage does not by itself choose the FP8 number representation. The narrow output
format is selected by the producer's `D` matrix layout data type, and the consuming GEMM must use
the same representation for its `A` matrix layout when it reads that output. The scale policy
(`STATIC` vs `DYNAMIC_FROM_AMAX`, per-tensor vs per-row) is therefore orthogonal to, but not
independent of, the FP8 encoding.

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

The `fp8_max` constant in section 3.3 is the maximum finite magnitude of the selected FP8 format,
not a global FP8 constant. Requant kernels must derive their quant multiplier, saturation
threshold, rounding behavior, and optional amax reporting from the actual `hipDataType` selected
for `D`.
The CODA handoff must record enough metadata to validate that the producer's FP8 `D` type and
the consumer's FP8 `A` type match, and that the selected device has a native kernel for that
variant.

MI350 also introduces OCP MX formats (`MXFP8`, `MXFP6`, `MXFP4`) with a shared exponent/scale
per 32-element block. These are not just plain FP8 plus a different per-row scale; they carry
block-scale metadata with layout requirements that must match the GEMM mainloop. A future CODA
MX mode can reuse the same algebra when each block is contained within one RMSNorm row, replacing
`rho_i` with a block scale `rho_{i,b} = rstd_i * scale_{i,b}(h2)`. That extension should be gated
on the quantization format expected by the vLLM model and on MI350 block-scale kernel support,
rather than enabled by the generic `PER_ROW` FP8 path.

##### CODA dynamic-quantized producer mode

When a producer chain contains both `HIPBLASLT_FUSEABLE_EPILOGUE_PARTIAL_RMSNORM_STATS` and
`HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT`, the `REQUANT` stage has specialized CODA semantics:

```text
producer D  = q(h2)                    # FP8 codes, using dynamic scale of h2
handoff rho = rstd * dequant_scale(h2) # logical RMSNorm-output dequant scale
```

The initial supported policy for this mode is:

- `SCALE_COMPUTE_MODE = HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX`.
- `SCALE_GRANULARITY = HIPBLASLT_REQUANT_SCALE_PER_ROW`.
- `SCALE_POINTER` receives the logical RMSNorm-output dequant scale `rho` (`f32[M * batch]`)
  and is also recorded in the handoff descriptor for the consumer. If a future implementation
  lets the handoff descriptor allocate this buffer privately, `SCALE_POINTER` can become an
  optional side-output attribute for this mode; the initial API keeps it required so the scale
  lifetime is explicit.
- `AMAX_POINTER`, when set, receives the logical RMSNorm-output amax
  `amax(y)_i = rstd_i * amax(h2)_i`, not the pre-RMSNorm `amax(h2)_i`.

The consumer matmul attaches `HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY` with the same
handoff descriptor. Its left input `A` points at the producer's FP8 `D` output. The consumer
kernel computes `q(h2) @ W1` and applies the handoff scale `rho` as a per-row epilogue factor.
Equivalently, a kernel with native per-row input scaling may consume `rho` before or during the
mainloop, but it must expose the same numerical contract.

Static scales are intentionally excluded from this CODA mode. With a caller-provided final
scale `S`, the required FP8 code is `round_fp8((rstd * h2) / S)`, so the producer must either
apply `rstd` before quantization or use a row-dependent quant multiplier `1 / (S / rstd)`. That
is not the cancellation in section 3.3 and should be modeled as a full materializing path or a
separate future mode.

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

Full RMSNorm followed by dynamic per-tensor FP8 requant, capturing the derived dequant scale
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
// ... hipblasLtMatmul(...) writes FP8 D plus the scale (and amax) side outputs ...
hipblasLtFusedEpilogueDestroy(fused);
```

#### 5.4.3 Decomposed RMSNorm handoff

Decomposed flow for `GEMM -> residual -> RMSNorm -> GEMM`, using the opaque handoff descriptor.
The same sketch covers the dynamic per-row FP8 variant of section 3.3: add `REQUANT` on the
producer, set the dynamic per-row policy, and use an FP8 producer `D` / consumer `A`.

```c
// Handoff object shared by the producer and consumer matmul calls.
hipblasLtFusedEpilogueRMSNormDescriptor_t stats;
hipblasLtFusedEpilogueRMSNormDescriptorCreate(&stats);

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

// Optional dynamic-quantized decomposed variant:
//   - matD1 has FP8 type and receives q(h2), not q(rstd * h2)
//   - stats carries rho_i = rstd_i * dequant_scale_i(h2)
// Omit this block for the ordinary BF16/FP16 decomposed flow, where matD1 receives h2
// and stats carries only rstd.
hipblasLtFusedEpilogueAdd(prod, HIPBLASLT_FUSEABLE_EPILOGUE_REQUANT);
hipblasLtRequantScaleComputeMode_t mode = HIPBLASLT_REQUANT_SCALE_DYNAMIC_FROM_AMAX;
hipblasLtRequantScaleGranularity_t gran = HIPBLASLT_REQUANT_SCALE_PER_ROW;
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_COMPUTE_MODE,
                                   &mode, sizeof(mode));
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_GRANULARITY,
                                   &gran, sizeof(gran));
// f32[M * batch], receives rho_i for the dynamic-quantized variant.
hipblasLtFusedEpilogueSetAttribute(prod, HIPBLASLT_FUSED_EPILOGUE_REQUANT_SCALE_POINTER,
                                   &d_coda_row_scale, sizeof(d_coda_row_scale));
hipblasLtMatmulDescSetAttribute(matmulDesc1, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &prod, sizeof(prod));
// ... hipblasLtMatmul(GEMM1) ...
// Internally launches GEMM1 producer + auxiliary reduce-and-return kernel.
// After the queued work completes, `stats` carries rstd (ordinary) or rho (quantized).

// GEMM2 consumer: apply the deferred per-row scale from `stats` in the epilogue.
// For the dynamic-quantized variant, A points at matD1's FP8 q(h2) output.
hipblasLtFusedEpilogueDescriptor_t cons;
hipblasLtFusedEpilogueCreate(&cons);
hipblasLtFusedEpilogueAdd(cons, HIPBLASLT_FUSEABLE_EPILOGUE_RMSNORM_SCALE_APPLY);
hipblasLtFusedEpilogueSetAttribute(cons, HIPBLASLT_FUSED_EPILOGUE_RMSNORM_STATS,
                                   &stats, sizeof(stats)); // same object as the producer
hipblasLtMatmulDescSetAttribute(matmulDesc2, HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE,
                                &cons, sizeof(cons));
// ... hipblasLtMatmul(GEMM2) ...
// launches GEMM2; its epilogue reads `stats`

hipblasLtFusedEpilogueDestroy(prod);
hipblasLtFusedEpilogueDestroy(cons);
hipblasLtFusedEpilogueRMSNormDescriptorDestroy(stats);
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

For the dynamic-quantized decomposed variant, the producer's `D` storage type is the requested
narrow requant type (initially FP8). The source value for quantization is still `h2`, so `gamma`
has the same storage type as the pre-requant producer output. The handoff scale `rho` remains
FP32. On MI350 (`gfx950`), the supported native FP8 producer/consumer type should be an OCP FP8
type (`HIP_R_8F_E4M3` or `HIP_R_8F_E5M2`). FNUZ types remain distinct HIP data types and require
a matching FNUZ-capable kernel path or explicit conversion before they can feed an OCP MI350 GEMM.

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

| Condition                                                                 | Return code                    | Detected at                                 |
|---------------------------------------------------------------------------|--------------------------------|---------------------------------------------|
| Unrecognized or unsupported `hipblasLtFuseableEpilogue_t` passed to `Add` | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| `Add` breaks the supported RMSNorm order (out-of-order stage)             | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Duplicate stage added                                                     | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Full and decomposed RMSNorm stages mixed in one chain                     | `HIPBLAS_STATUS_INVALID_VALUE` | `Add` time                                  |
| Unknown attribute passed to `SetAttribute`                                | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time                         |
| Required pointer attribute explicitly set to `NULL`                       | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time                         |
| Residual-add stage present but residual pointer unset                     | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| RMSNorm stage present but `gamma` or `eps` unset                          | `HIPBLAS_STATUS_INVALID_VALUE` | attach (`SetAttribute` of `FUSED_EPILOGUE`) |
| Partial-stats or scale-apply stage present but stats descriptor unset     | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| Requant stage present but scale pointer unset                             | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| Requant scale compute mode or granularity is outside the enum range       | `HIPBLAS_STATUS_INVALID_VALUE` | `SetAttribute` time / attach                |
| CODA requant producer uses a non-dynamic or non-per-row scale policy      | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| CODA requant producer is missing the RMSNorm handoff descriptor           | `HIPBLAS_STATUS_INVALID_VALUE` | attach                                      |
| CODA consumer A type/layout does not match the producer's FP8 output      | `HIPBLAS_STATUS_INVALID_VALUE` | `hipblasLtMatmul` / kernel validation       |
| Requested FP8 variant is not supported natively by the selected GPU       | `HIPBLAS_STATUS_NOT_SUPPORTED` | heuristic / `hipblasLtMatmul`               |
| Producer and consumer use different FP8 variants (for example OCP vs FNUZ) | `HIPBLAS_STATUS_INVALID_VALUE` | `hipblasLtMatmul` / kernel validation       |
| Scale-apply consumes a stats descriptor not populated by a producer       | `HIPBLAS_STATUS_INVALID_VALUE` | `hipblasLtMatmul`                           |
| Attached fused epilogue without a matching kernel implementation          | `HIPBLAS_STATUS_NOT_SUPPORTED` | `hipblasLtMatmul`                           |

The key requirement: an illegal *ordering* for a supported RMSNorm chain is rejected at the
API-call level, incrementally as stages are added, before any codegen or kernel launch.
Residual completeness (`residual`), RMSNorm completeness (`gamma` and `eps`), requant
completeness (`scale` and valid policy enums), and decomposed-flow completeness (a set stats
descriptor) are validated when the handle is attached to the matmul descriptor via
`HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE`. This means
required stage-specific attributes must be set on the handle before attachment and must not be
cleared or mutated after attachment. The matmul descriptor stores a non-owning pointer to the
handle, so later optional-attribute updates are visible to `hipblasLtMatmul`, but callers
should not rely on late updates to satisfy attach-time required-attribute validation. For the
decomposed flow, the producer-before-consumer dependency on the handoff descriptor can only be
fully validated at `hipblasLtMatmul` time.

For the dynamic-quantized decomposed variant, attaching a producer chain with
`PARTIAL_RMSNORM_STATS -> REQUANT` additionally validates that the chain has a handoff
descriptor, uses `DYNAMIC_FROM_AMAX + PER_ROW`, and writes a narrow producer `D` type supported
by the matching consumer kernel. The consumer validates that the attached handoff was populated
by a compatible CODA producer before using its row scale. FP8 validation includes the selected
HIP data type: the producer `D` type, consumer `A` type, quantization scale interpretation, and
available kernel must agree on OCP vs FNUZ encoding.

## 7. Interaction with existing descriptors and preferences

- A new opaque builder object (`hipblasLtFusedEpilogueDescriptor_t`) is introduced; it is
  attached to the existing `hipblasLtMatmulDesc_t` via a single attribute. The matmul
  descriptor stores only a non-owning pointer. The decomposed flow adds one more opaque handle
  type (`hipblasLtFusedEpilogueRMSNormDescriptor_t`) that is only passed across calls, never
  inspected by the caller.
- `hipblasLtMatmulPreference_t` is unchanged. Workspace and SM-count hints apply as usual.
  Neither flow has caller-allocated normalization buffers. In the full flow, `partialBuf` and any
  synchronizer/flag buffer are transient and drawn from the matmul preference workspace for the
  single call; Kernel 2 applies the row scale to `D` in place, so no per-row-scale buffer is
  materialized. In the decomposed flow, the handoff descriptor owns or records only the finalized
  consumer row scale used by GEMM2 (`rstd` for the unquantized decomposed flow, `rho` for the
  dynamic-quantized decomposed variant). The producer's `partialBuf` scratch and any
  synchronizer/flag buffer remain internal, transient workspace sized through the workspace-size
  query.
- On the codegen side, the GEMM kernels for both flows come from a single TensileLite option
  (`FusedEpilogues = 1`) that bundles the optional-residual, RMSNorm-producer, and
  RMSNorm-scale-apply epilogue options and selects among them at runtime. The cross-tile
  reduction is a separate custom kernel. This keeps the number of generated GEMM kernels
  bounded (no combinatorial explosion) at the cost of larger kernels.
- The C++ extension API (`hipblaslt_ext::GemmEpilogue` / `GemmInputs`) is outside this
  document; the C handle API is the defined surface.

## 8. SwiGLU/GLU extensibility

GLU/SwiGLU is a separate epilogue family, not part of the RMSNorm chain order in section 4.2.
It can reuse the same fused-epilogue descriptor handle, `Add`/`SetAttribute` builder flow,
matmul-descriptor attachment point, and stage-specific attribute model. It still requires
its own validation because it is shape-changing: the GEMM produces `[M, 2N]` for the packed
gate/up projection, while the GLU stage emits the final `[M, N]` tensor.
