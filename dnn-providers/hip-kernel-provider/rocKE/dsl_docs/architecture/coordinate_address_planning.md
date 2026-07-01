# Coordinate Transform To Physical Address Planning

This document proposes a planning layer between `rocke.helpers.transforms` and
`IRBuilder` emission. The goal is to keep CK-style coordinate transforms as the
semantic source of truth while preventing complex convolution and attention
descriptors from lowering into repeated hot-loop VALU address arithmetic.

## Problem

CK DSL inherits CK Tile's coordinate-transform model: a tensor descriptor is a
logical program that maps operator coordinates to storage coordinates plus an
optional validity predicate. This is the right vocabulary for non-bijective
operator mappings such as padding, dilation, sliding windows, implicit-GEMM
convolution, ragged attention, page-table indirection, and fused attention
layouts.

The current lowering path is too direct:

```text
TensorDescriptor.offset(...)
  -> walk transform DAG
  -> emit IRBuilder scalar SSA immediately
  -> rely on generic canonicalization later
```

That direct expansion is easy to reason about for correctness, but it can emit
the same coordinate math at every load site. In hot loops this can become extra
VALU instructions: div/mod from `unmerge`, mul/add from `merge` and `embed`,
and compare/select chains from padding predicates.

The transform methodology is not inherently inefficient. A hand-written kernel
must compute the same logical mapping. The issue is that the framework does not
yet have a transform-aware lowering stage that knows which terms are uniform,
which terms form address recurrences, and which validity predicates can be
removed or hoisted.

## Goal

Separate the two questions currently answered by `.offset()`:

```text
Logical transform:
  What tensor element does this operator coordinate refer to?

Physical address plan:
  How should this CTA/wave/lane/vector compute that address cheaply?
```

The logical transform remains the source of truth. The physical address plan is
the performance contract used by production kernels.

The proposed flow is:

```text
TensorDescriptor transform DAG
  -> Coordinate Expression Graph
  -> Physical Address Plan
  -> optimized IRBuilder emission
```

## Non-Goals

- Do not replace CK Tile-style non-bijective transforms with a pure
  shape/stride model. Shape/stride layouts are useful for physical tiling, but
  they do not naturally express all operator semantics.
- Do not change `transforms.py` into an architecture-specific layer. Logical
  coordinate algebra remains target-agnostic.
- Do not require every descriptor user to opt into the planner immediately.
  Literal `.offset()` lowering remains the correctness fallback.

## Coordinate Expression Graph

Add a non-emitting representation of descriptor math. The expression graph
mirrors the operations that `.offset()` emits today, but it is analyzable before
SSA is generated.

Suggested module layout:

```text
rocke/coord/
  expr.py       # expression nodes and simplification primitives
  analysis.py   # uniformity, bounds, recurrence, cost
  plan.py       # address and predicate plan data structures
  lower.py      # staged IRBuilder emission
```

Initial expression nodes:

```python
Const(value)
Coord(name)
Add(a, b)
Sub(a, b)
Mul(a, b)
Div(a, b)
Mod(a, b)
Cmp(pred, a, b)
And(a, b)
Or(a, b)
Select(pred, true_value, false_value)
Indirect(table, index)
```

`TensorDescriptor` gains a non-emitting API:

```python
result = desc.offset_expr(coords={"m": Coord("m"), "k": Coord("k")})
```

The result carries:

```python
CoordExprResult(
    offset=Expr,
    valid=Optional[Expr],
    named_coords=dict[str, Expr],
)
```

The implementation should share transform semantics with `.offset()` so the
literal and planned paths cannot drift.

## Uniformity Analysis

Every expression should be classified by where it varies:

```text
CONST < BLOCK < LOOP < WAVE < LANE < VECTOR
```

Examples:

```text
Hi, Wi, C, stride_h, dilation_h   -> CONST
block_id_x * tile_m               -> BLOCK
k_loop * block_k                  -> LOOP
wave_id-derived tile offset       -> WAVE
thread_id-derived lane offset     -> LANE
vector element i                  -> VECTOR
```

The propagation rule is conservative:

```text
uniformity(op(a, b)) = max(uniformity(a), uniformity(b))
```

The lowerer uses this classification to emit each expression at the earliest
legal stage. Workgroup-uniform terms should not be recomputed as lane-varying
VALU. Loop-invariant terms should not be emitted inside an unrolled vector
element path.

## Physical Address Plan

The address planner consumes a coordinate expression result plus context from
the kernel builder:

```python
plan = desc.address_plan(
    coords={"m": m_expr, "k": k_expr},
    vector_axis="k",
    vector_width=4,
    loop_axis="k",
    assumptions=AddressAssumptions(...),
)
```

The plan answers:

- Which expressions are static, block-uniform, loop-uniform, wave-uniform,
  lane-varying, and vector-varying?
- Can the vector elements be expressed as `base + i * delta`?
- Can the loop update be expressed as `base += loop_delta`?
- Are validity predicates scalar, vector, or unnecessary?
- Which div/mod operations can be strength-reduced or avoided?
- Which expressions are duplicated and should be CSE'd before emission?

Suggested structure:

```python
AddressPlan(
    base_offset=Expr,
    vector_offsets=list[Expr] | DeltaExpr,
    loop_delta=Optional[Expr],
    validity=PredicatePlan,
    hoisted=HoistPlan,
    cost=AddressCost,
    notes=list[str],
)
```

`PredicatePlan` should distinguish:

```text
always true
scalar predicate
per-vector predicate
interior/tail split predicate
```

`AddressCost` should estimate hot-loop integer work:

```text
adds, muls, divs, mods, compares, logical ops, indirect loads
```

The estimate does not need to be exact ISA accounting. Its purpose is to catch
regressions and explain why one plan is expected to be cheaper than literal
lowering.

## Core Optimizations

### Expression CSE And Constant Folding

Run CSE before SSA emission, not only after IR has been emitted. Descriptor
lowering often recomputes the same `unmerge`, `embed`, offset, or validity
subexpression at multiple vector elements and load sites.

Static shape math should fold inside the expression graph. For example,
products such as `Ho * Wo` and `S * C` should be constants before uniformity and
cost analysis.

### Merge/Unmerge Cancellation

Preserve decomposed coordinates when they are already available. If a plan sees:

```text
merge(n, ho, wo) -> m
unmerge(m) -> n, ho, wo
```

it should cancel the round trip when legality is obvious. This is especially
important when the kernel builder starts from tile coordinates that are already
decomposed but the descriptor API accepts packed implicit-GEMM coordinates.

### Div/Mod Strength Reduction

`unmerge` introduces div/mod by static dimensions. The planner should mark:

```text
x / power_of_two -> shift
x % power_of_two -> and
x / constant     -> magic multiply/shift candidate
```

The bigger win is avoiding div/mod entirely when decomposed coordinates or
recurrences are available.

### Vector Recurrence Detection

Detect when vector elements are contiguous or regularly strided:

```text
offset(i) = base + i * delta
```

For common channel-vector loads, the planned lowering should compute one full
address and then use small deltas:

```text
base, base + 1, base + 2, base + 3
```

instead of emitting the full descriptor transform four times.

### Loop Recurrence Detection

For K-loop loads, detect updates such as:

```text
next_k_tile = current_k_tile + block_k
next_base   = current_base + loop_delta
```

When legal, loop-carried address updates can replace repeated full
reconstruction of the same base coordinates.

### Bounds And Validity Simplification

Validity should be planned, not merely emitted. The planner should support
assumptions such as:

```python
AddressAssumptions(
    known_in_bounds={"hi", "wi"},
    tail_axes={"c"},
    interior_tile=True,
)
```

Interior tiles should remove padding predicates. Tail tiles should keep only the
necessary masks. Spatial predicates that are common to all vector elements
should be emitted once and combined with vector-tail predicates only when
needed.

## Staged Emission

`plan.emit(b)` should emit SSA according to the plan's stages:

```text
CONST   -> constants and folded terms
BLOCK   -> near block-id setup
LOOP    -> loop prologue or per-unrolled iteration
WAVE    -> once per wave slice
LANE    -> once per lane
VECTOR  -> only when base+delta recurrence is unavailable
```

The first implementation can emit into the current `IRBuilder` insertion point,
but the plan should preserve stage metadata so later integration can place
expressions more precisely.

Return type:

```python
AddressEmission(
    offsets=list[Value] | VectorAddress,
    valid=Optional[Value] | list[Value],
)
```

`VectorAddress` can represent `base + delta * i` without forcing the planner to
materialize every scalar offset immediately.

## Debuggability

Every physical plan must be explainable. A descriptor author should be able to
ask why a plan emits a given amount of address math.

Example output:

```text
A_desc address plan, vector_width=4

Semantic transform:
  m -> (n, ho, wo)
  k -> (y, x, c)
  hi = ho * sH - pH + y * dH
  wi = wo * sW - pW + x * dW
  offset = ((n * Hi + hi) * Wi + wi) * C + c
  valid = 0 <= hi < Hi && 0 <= wi < Wi && c < C

Uniformity:
  n, ho, wo: BLOCK
  r, s: LOOP
  c: LANE + VECTOR
  spatial_valid: LANE
  c_tail_valid: VECTOR

Physical lowering:
  base_offset = full transform at vector element 0
  vector recurrence = base_offset + i
  validity = spatial_valid && c_tail_valid[i]

Estimated hot-loop integer work:
  literal: 4 div/mod, 8 mul/add, 4 cmp per element
  planned: 4 div/mod, 8 mul/add, 4 cmp per vector
```

This makes CK-style transforms easier to debug without giving up their
expressive power.

## Integration Strategy

Keep `.offset()` as the simple semantic lowering path. Add planned APIs in
parallel:

```python
desc.offset_expr(...)
desc.address_plan(...)
desc.offset_planned(b, ...)
```

Initial integration should target one narrow and measurable path, such as
implicit-GEMM convolution A-tile loads. That path stresses `unmerge`, `embed`,
padding validity, and vectorized loads, so it is a good proof point for both
correctness and VALU reduction.

After the convolution path is stable, extend the planner to:

- attention Q/K/V tiled loads;
- page-table or block-table indirection;
- block-scaled and MX GEMM scale addressing;
- LDS descriptor plans where physical swizzles and vectorization dominate.

## Validation

Correctness tests should compare planned offsets against literal `.offset()`
over a broad coordinate sample:

- merge/unmerge descriptors;
- padded and unpadded embeds;
- dynamic and static bounds;
- indirect descriptors;
- vector-width variants;
- interior and tail tiles.

IR and ISA tests should track the intended performance signal:

- fewer duplicate expression nodes before lowering;
- fewer `arith.div`, `arith.mod`, and repeated `arith.mul` operations in IR;
- lower VALU counts in selected address-heavy kernels;
- no increase in VMEM/LDS hazards or predicate misuse;
- identical numeric results versus literal lowering.

## Rollout Milestones

1. Add expression-only descriptor lowering with parity tests against `.offset()`.
2. Add expression CSE, constant folding, and uniformity analysis.
3. Add `AddressPlan`, `PredicatePlan`, `AddressCost`, and `plan.explain()`.
4. Add vector recurrence and bounds simplification.
5. Integrate planned lowering into one convolution A-load path behind a flag.
6. Add attention and indirect-address support.
7. Make planned lowering the default for hot descriptor loads once parity and
   ISA-count tests are stable.

## Design Principle

Semantic transforms describe what is correct. Physical address plans describe
what is cheap.

The planner should let CK DSL keep CK Tile's expressive coordinate-transform
methodology while giving kernel authors FlyDSL-like visibility into shape,
stride, recurrence, validity, and address-generation cost.
