# `tensor_transforms` -- how do I ...

Worked examples for `rocke.helpers.tiling.tensor_transforms`. Every signature, rule and error is defined once,
in [`tensor_transforms_api.md`](tensor_transforms_api.md) (cited below as "API" with its section); this guide
shows how to use them. Every example is real code: the blocks run top to bottom as one script, starting
with the setup block.

```python
from rocke.core.ir import F16, IRBuilder
from rocke.helpers.tiling import make_tensor_desc
from rocke.helpers.tiling.tensor_transforms import (
    embed, emit_accesses, inputs, make_transform_graph, merge, offset, outputs,
    required_checks, transform_address_of, unmerge, value_ranges, xor,
)
```

---

## How do I declare a graph?

List the inputs (with their lengths), then the steps in read -> write order, then the outputs in tensor-axis
order. The graph holds no tensor: output `i` is bound to tensor axis `i` later. The implicit-GEMM convolution
input read, `m -> (n, ho, wo)`, `k -> (y, x, c)`, `h = ho * stride + y * dilation - pad`:

```python
N, Hi, Wi, C = 2, 8, 8, 16          # NHWC input
Y, X = 3, 3                         # filter
sH, sW, dH, dW, pH, pW = 1, 1, 1, 1, 1, 1
Ho = (Hi + 2 * pH - dH * (Y - 1) - 1) // sH + 1
Wo = (Wi + 2 * pW - dW * (X - 1) - 1) // sW + 1

g = make_transform_graph(
    inputs(m=N * Ho * Wo, k=Y * X * C),
    unmerge("m", into=("n", "ho", "wo"), lengths=(N, Ho, Wo)),
    unmerge("k", into=("y", "x", "c"), lengths=(Y, X, C)),
    embed(("ho", "y"), into="h_dil", coeffs=(sH, dH)),
    offset("h_dil", into="hi", shift=-pH),
    embed(("wo", "x"), into="w_dil", coeffs=(sW, dW)),
    offset("w_dil", into="wi", shift=-pW),
    outputs("n", "hi", "wi", "c"),
)
x_td = make_tensor_desc((N, Hi, Wi, C), (Hi * Wi * C, Wi * C, C, 1), F16)
```

`make_transform_graph` checks the whole graph when it is built (API 13, "Building a graph"): each coordinate is
written once, every read is declared or written, there is no cycle, restated domains agree, every input is read
and every output written.

Anti-examples:

```python
# offset("h", into="h", shift=-1)        # a step never rewrites its own read: name the result anew
# unmerge("k", into=("y", "x", "c"), lengths=(Y, X, C + 1))   # pieces must multiply to k's domain
```

`merge` combines N -> 1 and `unmerge` splits 1 -> N; CK and the V4 transform graph use the opposite names
(API 3).

## How do I check a graph with `evaluate`?

`evaluate` and `resolve` compute in pure Python what every lane computes. Coordinates are a tuple in
`inputs(...)` order (API 4):

```python
assert g.evaluate((9, 20)) == (0, 0, 1, 4)                  # (n, hi, wi, c)
assert g.resolve((9, 20))["y"] == 0                         # every named coordinate, intermediates too
assert g.edge("y").domain == Y                              # the domain the unmerge declared
assert transform_address_of(g, x_td, (9, 20)) == 0 * Hi * Wi * C + 0 * Wi * C + 1 * C + 4
```

Use them as the golden for an emitted address (`transform_address_of` is `address_of` applied to `evaluate`).
Anti-example: `g.evaluate({"m": 9, "k": 20})` -- coordinates are positional, never a mapping by name.

## How do I add a halo?

A halo is a negative `offset` on the coordinate that leaves the tensor (above: `hi = h_dil - pH`). The tensor's
own length masks BOTH sides; nothing restates it:

```python
lo, hi = value_ranges(g, ((0, N * Ho * Wo), (0, Y * X * C)))["hi"]
assert (lo, hi) == (-pH, Ho - 1 + (Y - 1) + 1 - pH)            # [-1, 9): one row past each edge
```

Shift the halo once: in the graph (`offset`) or in the window origin, never both.

## How do I split K?

`unmerge` one input into pieces, the last piece innermost; the pieces must multiply to the input's domain:

```python
ks = make_transform_graph(
    inputs(k=256),
    unmerge("k", into=("k_outer", "k_inner"), lengths=(8, 32)),
    outputs("k_outer", "k_inner"),
)
assert ks.evaluate((70,)) == (2, 6)
```

A read past the product is not wrapped: it drives the outer piece past its length, and that is exactly the read
domain a check is derived for (the K tail of the convolution graph).

## How do I map query heads onto KV heads (grouped-query attention)?

Query head `h` reads KV head `h // (Hq / Hkv)`. `unmerge` the query head into `(kv_head, group)` and leave the
group unread; only the KV head reaches the K / V tensor:

```python
Hq, Hkv, S, D = 40, 8, 256, 128
gqa = make_transform_graph(
    inputs(q_head=Hq, token=S, d=D),
    unmerge("q_head", into=("kv_head", "group"), lengths=(Hkv, Hq // Hkv)),
    outputs("token", "kv_head", "d"),
)
k_td = make_tensor_desc((S, Hkv, D), (Hkv * D, D, 1), F16)
assert gqa.evaluate((13, 7, 5)) == (7, 2, 5)          # query head 13 reads KV head 13 // 5 = 2
assert transform_address_of(gqa, k_td, (13, 7, 5)) == 7 * Hkv * D + 2 * D + 5
```

When the query head comes from a block id, its division is emitted once per wave, never per lane.

## How do I build a swizzle?

`unmerge -> xor -> merge`: split the column into groups, xor the group with the row, combine back. `xor` needs
its second read to have a power-of-two domain (that is what makes it a bijection); `graph_injective` confirms no
two inputs collide:

```python
from rocke.helpers.tiling.tensor_transforms import graph_injective

swz = make_transform_graph(
    inputs(row=64, col=64),
    unmerge("col", into=("group", "within"), lengths=(8, 8)),
    xor(("row", "group"), into=("row_out", "group_sw")),
    merge(("group_sw", "within"), into="col_sw", lengths=(8, 8)),
    outputs("row_out", "col_sw"),
)
assert swz.evaluate((3, 10)) == (3, 18)              # group 1 xor 3 = 2: column 2 * 8 + 2
assert graph_injective(swz, ((0, 64), (0, 64)))
```

An 8-wide vector along `col` that starts at a multiple of 8 stays contiguous through it (API 11).

## How do I get an address?

Call `emit_accesses` with one tuple of PLAIN values per access, in `inputs(...)` order: `(m, k)` here. Write
the coordinates as the IR computes them; the layer splits each into its per-wave and per-lane parts and proves
their ranges from the IR. Set the workgroup size first: thread-id ranges come from it (API 8).

```python
TM, TK = 64, 32
b = IRBuilder("conv_input_tile")
b.kernel.attrs["max_workgroup_size"] = 256          # right after creating the builder
tid = b.thread_id_x()
m_lane = b.div(tid, b.const_i32(4))                                # per-lane row, 0..63
k_lane = b.mul(b.mod(tid, b.const_i32(4)), b.const_i32(8))         # per-lane column, 0, 8, 16, 24
with b.scf_for(b.const_i32(0), b.const_i32(N * Ho * Wo), b.const_i32(TM), iv_name="m0") as m0:
    with b.scf_for(b.const_i32(0), b.const_i32(Y * X * C), b.const_i32(TK), iv_name="k0") as k0:
        m = b.add(m0, m_lane)                     # per-wave origin + per-lane row
        k = b.add(k0, k_lane)                     # per-wave counter + per-lane column
        (acc,) = emit_accesses(b, g, x_td, [(m, k)], vector_axis=1, vector_width=8)
address = acc.address
assert [term.inputs for term in address.uniform] == [frozenset({0}), frozenset({1})]
assert address.const == -(pH * Wi * C + pW * C)     # the halo lands in the constant
assert acc.valid is not None                        # the halo and the tail tiles can fail
```

What came back: one uniform term per graph input (`m0`'s and `k0`'s, kept apart so the memory side can treat a
loop-invariant one differently), one per-lane part with its range, a constant, and `valid`, the AND of every
check that can fail for this 8-wide vector along `k` (`vector_axis=1` is the graph-input index of `k`).

Anti-examples:

```python
# emit_accesses(b, g, x_td, [{"m": m, "k": k}])         # an access is a tuple in inputs(...) order
# emit_accesses(b, g, x_td, [[m, k]])                   # ... a tuple, not a list
# m = b.add(b.mul(b.block_id_x(), b.const_i32(TM)), m_lane)
#     # a raw block origin is [0, +inf): the unmerge of m cannot divide it and raises the dividend
#     # error -- a per-wave origin that feeds an unmerge needs a stated range first
```

## How do I see which checks a graph needs?

`required_checks` answers it in pure Python, from one `(lo, hi)` range per input:

```python
exact = required_checks(g, x_td, ((0, N * Ho * Wo), (0, Y * X * C)))
assert [c.coordinate for c in exact] == ["hi", "wi"]                  # only the halo, both sides
rounded = required_checks(g, x_td, ((0, 3 * TM), (0, 5 * TK)))        # tile-rounded m and k
assert [c.coordinate for c in rounded] == ["m", "k", "hi", "wi"]      # + the ragged M tile, the K tail
```

`y`, `x`, `c` and `n` never need a compare: the unmerges prove their ranges. Each `BoundCheck` reports which
side can fail and why (`INPUT_DOMAIN`, `READ_DOMAIN`, `TENSOR_BOUND`); a proven side is `None`. Checks are
derived, never hand-added. Anti-example: giving an input `(None, None)` "because it is in range anyway" -- that
is "not known", and it is checked; state `(0, n)`, or pass the plain IR value to `emit_accesses` and let the
walk derive its range.

## Why do bounds live on the tensor?

A length has one home: `TensorDesc.lengths`. The graph never holds one, and no entry takes a `bounds=`
argument, so there is no second copy to drift from the first and no way to pass a wider bound than the tensor's.
A runtime length (a `RuntimeValue`) proves a tile inside it by identity -- an end computed from the same IR value
-- which works only because every user reads the same object (API 7). A narrower region is a different tensor
description; one value against arbitrary ends is `emit_bound_check` (API 10).

---

## How to write coordinates the layer can split

The IR range walk (API 8) reads how each coordinate was computed. Any spelling gives a correct value; this
spelling also keeps the per-wave part on the scalar unit and the ranges proven:

- **Add per-lane terms together before adding the per-wave origin**, and add the origin last:
  `origin + (row_lane + col_lane)`.
- **Scale a sum term by term**: `origin * 2 + lane * 2`, not `(origin + lane) * 2`. The walk does not split
  through a multiply of a mixed value, so the second spelling makes the whole value per-lane.
- **Divide with `div`, not a right shift**: `x // 8` of a non-negative `x` is `b.div(x, 8)`; `lshr` gives an
  unknown range. Mask with `and (2^c - 1)` (exact for every sign). Write `x | c` as `add` when the bits are
  known disjoint.
- **Pass a per-wave value loaded from memory through `b.readfirstlane`** (a varlen start from a sequence-offset
  buffer): a load is per-lane to the walk whatever its index.
- **Bound a per-wave origin that feeds an `unmerge`**: its division needs a proven `[0, 2^31)`.

What an unknown range costs: a check that stays, or a part that lives in vector registers -- never a wrong
answer.

```python
from rocke.helpers.tiling.tensor_transforms.walk import split_coordinate   # internal: for illustration

b = IRBuilder("spelling")
b.kernel.attrs["max_workgroup_size"] = 256
origin = b.mul(b.block_id_x(), b.const_i32(64))                  # per-wave
lane = b.div(b.thread_id_x(), b.const_i32(4))                    # per-lane, [0, 64)
two = b.const_i32(2)
good = split_coordinate(b, b.add(b.mul(origin, two), b.mul(lane, two)))
bad = split_coordinate(b, b.mul(b.add(origin, lane), two))
assert len(good.uniform) == 1 and good.lane is not None          # origin * 2 stays per-wave
assert bad.uniform == () and bad.lane is not None                # all of it is per-lane
```

---

## CK `pad` in rocke

CK's `pad(len, left, right)` is `offset(shift=-left)` in a graph (or a negative origin on a plain window) plus
the tensor's own length. The right pad is the tile overhang past the length, masked by that length with no
parameter.

| CK pad transform | rocke |
|---|---|
| the naive view's length `len` | the tensor's length (`TensorDesc.lengths`; a `RuntimeValue` for a runtime one) |
| left pad | `offset(read, into, shift=-left)`, or a negative window origin |
| right pad | nothing |

Anti-examples: restating `len` in the graph (there is no PAD kind and no bound in a graph); shifting the same
halo twice (a negative origin AND an `offset`).

### Why rocKE differs from CK tile here

The coordinate-side departures, each with CK's behaviour, the failure it allows, rocKE's rule and its cost:

- **One length home.** CK's `pad` restates the tensor's length inside the descriptor. A restated length is a
  second home: a derived form of it is easy to get wrong (a strided convolution's padded extent is not the
  input's length), and the two then disagree silently. rocKE: the length lives only on the tensor; the halo is a
  shift. Cost: none.
- **Every escaping domain is checked, not only pad kinds.** CK derives validity only from its pad transforms:
  the top index is unchecked, and a `merge` read past its length wraps into the next row silently. rocKE: every
  input domain, every read domain a kind requires (an `unmerge` read past its pieces' product) and every tensor
  bound gets a check whenever its range can escape, and only then (API 7). Cost: one compare per coordinate that
  can actually fail; none where the ranges prove it.
- **`right` has no counterpart**, because the tensor's length already masks the tile overhang.
- **Porting a CK pad transform, line by line:** drop `pad`; keep the length on the tensor description; write the
  left pad as `offset(..., shift=-left)` on the coordinate CK padded (or fold it into the window origin); write
  CK's `merge` as `unmerge` and its `unmerge` as `merge`; delete every hand-added validity pad -- the checks are
  derived.

The memory-side departures (how a failing check is enforced, vector width at an edge, the fill value) belong to
the memory side that consumes an `Access`, not to this package.
