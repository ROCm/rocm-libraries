# Kernel Signatures

A kernel **signature** is the mathematical contract: what a kernel computes. It describes
typed tensors, scalar parameters, and the operation that relates them.

The signature says nothing about *how* the kernel computes — tile geometry, pipeline
strategy, and scheduling belong to the **algorithm**. Together, a signature and an
algorithm fully specify a compiled kernel variant.

## Tensors

A tensor has compile-time properties (fixed when the kernel is compiled) and runtime
properties (provided at each launch).

**Compile-time:**

| Property | Description |
|----------|-------------|
| **name** | Semantic role: `"A"`, `"bias"`, `"query"`, `"output"` |
| **dtype** | Element type: FP32, FP16, BF16, FP8, INT8, ... |
| **rank** | Number of dimensions |
| **layout** | Memory layout: `Row`, `Col`, `Contiguous`, or strides |

**Runtime:**

| Property | Description |
|----------|-------------|
| **pointer** | Device memory address |
| **sizes** | Extent of each dimension |
| **strides** | Memory stride of each dimension |

### Layout

Layout describes how logical dimensions map to memory. The canonical representation is
**strides** — one integer per dimension giving the element offset between consecutive
indices.

For rank-2 tensors, layout enums provide convenient shorthand:

```
Row-major [M, K]:  strides = [K, 1]    (rows are contiguous)
Col-major [M, K]:  strides = [1, M]    (columns are contiguous)
```

For higher-rank tensors, layout tags name common stride patterns:

```
NHWC [N, H, W, C]:  strides = [H*W*C, W*C, C, 1]
NCHW [N, C, H, W]:  strides = [C*H*W, H*W, W, 1]
```

Layout enums are **input sugar** — they resolve to strides. Strides are the internal
representation because they handle every case: transposes, padding, non-contiguous views,
and layouts we haven't named yet.

### Type resolution

Signatures use an optional dtype cascade for ergonomic type specification:

```cpp
// Set everything to FP16 (GemmOp accumulator defaults to FP32):
Signature{.dtype = FP16, .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// Mixed precision — FP16 compute, FP32 output tensor:
Signature{.dtype = FP16,
          .tensors = {Tensor{.name = "C", .dtype = FP32}},
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}

// Fully explicit per-tensor:
Signature{.tensors = {Tensor{.name = "A", .dtype = FP16},
                      Tensor{.name = "B", .dtype = BF16},
                      Tensor{.name = "C", .dtype = FP32}},
          .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"}}}
```

The cascade rules:
- `Signature::dtype` sets the default for all tensors
- Per-tensor `Tensor::dtype` overrides take precedence
- `GemmOp::acc_dtype` defaults to FP32 independently

Resolution is `consteval` — unresolvable types are compile errors, not runtime surprises.

## Scalars

Scalar parameters have a name, type, optional default, and a runtime value.

```
alpha     : float, default 1.0
scale     : float                   (required, no default)
groups    : int32                   (convolution group count)
causal    : bool, default false     (FMHA causal masking)
```

**Problem-size scalars** (M, N, K, sequence lengths) describe the extent of tensor
dimensions. They are always runtime values but may have compile-time constraints
(e.g., "K must be a multiple of 32 for this variant").

## Operations

Operations are **typed structs** held in a `std::variant` array. Each struct defines
named tensor slots. A signature's `ops` array forms a directed compute graph — tensors
are nodes, operators are edges. Adding a new operation = new struct + add to variant.

### Operator Structs

```cpp
struct GemmOp    { string_view lhs="A", rhs="B", out="C"; DataType acc_dtype=FP32; };
struct AddOp     { string_view lhs, rhs, out; };
struct MulOp     { string_view lhs, rhs, out; };
struct ReluOp    { string_view in, out; };
struct FastGeluOp { string_view in, out; };
struct SoftmaxOp { string_view in, out; };  // reduces last dimension
struct ScaleOp   { string_view in, out, scale; };  // scale names a Scalar
// ... plus GeluOp, SiluOp, SigmoidOp
```

### Compute Graph Examples

**GEMM:**
```
A, B → [GemmOp] → C
```

**GEMM + bias + ReLU (fused epilogue):**
```
A, B → [GemmOp] → C → [AddOp] ← bias → D → [ReluOp] → E
```

**FMHA (decomposes into existing ops):**
```
Q, K → [GemmOp] → S → [SoftmaxOp] → P
                                       P, V → [GemmOp] → O
```

**Elementwise:**
```
A, B → [AddOp] → C
```

### SSA Naming

Each operator output gets a unique name. Graph edges = shared names between
operator outputs and inputs. SSA uniqueness is enforced by `resolve()` at
compile time — duplicate output names produce a compile error.

### Operator-Implied Defaults

`GemmOp` implies rank-2 tensors: lhs=Row, rhs=Col, out=Row. Binary and unary
ops propagate rank/layout from connected tensors. Explicit `Tensor` entries
in the signature override any propagated values.

## Fusion

Fusion is expressed as **operator composition** in the signature's ops array.
Rather than enumerating combinations with enums, each fusion step is a typed
operator in the compute graph:

```cpp
// GEMM + bias + ReLU
Signature{
    .dtype = FP16,
    .ops = {GemmOp{.lhs = "A", .rhs = "B", .out = "C"},
            AddOp{.lhs = "C", .rhs = "bias", .out = "D"},
            ReluOp{.in = "D", .out = "E"}}}
```

`make_kernel()` pattern-matches the ops sequence to select the CK Tile kernel
and epilogue configuration. The mapping is:

| Ops Pattern | Kernel |
|-------------|--------|
| `[GemmOp]` | Plain GEMM |
| `[GemmOp, AddOp]` | GEMM + CShuffleEpilogue (bias) |
| `[GemmOp, AddOp, ReluOp]` | GEMM + CShuffleEpilogue (bias + activation) |
| `[GemmOp, SoftmaxOp, GemmOp]` | FMHA kernel |
| `[AddOp]` | Elementwise kernel |

This replaces the earlier enum model (`CombineOp`, `Activation`) with a
composable graph that naturally extends to arbitrary epilogue chains.

## Compile-Time / Runtime Boundary

Every kernel has two distinct phases:

**Compile time** — the signature and algorithm are validated, and a kernel binary is
produced. This determines:
- Tensor dtypes, ranks, layouts
- Layout patterns
- Operation type and epilogue structure
- Accumulator type
- Tile geometry and pipeline (from the algorithm)

**Runtime** — the kernel is launched with concrete arguments:
- Tensor pointers, sizes, strides
- Scalar values (problem sizes, alpha/beta, etc.)

The same compiled binary handles different problem sizes. Validation at compile time
(`consteval make_kernel`) catches configuration errors as compiler diagnostics, not
runtime crashes.

The runtime arguments are a **flat POD struct** with stable ABI — no constructors,
no virtual functions, no heap allocation. Field order and alignment are documented
and asserted with `static_assert`.

## Design Decisions

**Batch dimensions are explicit.** Batch, head count, and other semantic dimensions are
first-class in the signature, not implicit extra tensor dimensions. Keep semantic
information in the signature — it enables better validation and clearer code.

**Epilogue composition is general.** Epilogue fusion is expressed as a composable graph
of typed operators in the signature's `ops` array. Each fusion step is a typed operator
(`AddOp`, `ReluOp`, etc.) with named tensor slots. `make_kernel()` pattern-matches the
operator sequence to select the CK Tile epilogue configuration. This avoids accumulating
special cases — adding a new epilogue combination requires no new enum values.

**Split-K is algorithmic.** Parallelism strategies like split-K, workspace allocation,
and tuning parameters belong in the algorithm, not the signature. The signature describes
*what* is computed; the caller should not need to know *how* partial results are managed.

**Problem-size constraints live in the algorithm.** Constraints like "K must be a multiple
of 32" arise from tile geometry choices, not from the mathematical operation. The algorithm
declares its constraints; `make_kernel` cross-validates them against the signature. Runtime
argument validation checks actual values against the compiled variant's requirements.
