# `tensor_transforms` -- API surface

The package `rocke.helpers.tiling.tensor_transforms` gives a kernel author one way to say "these loop
coordinates become these tensor coordinates", for any workload (copy, prefetch, reduction, attention,
convolution, GEMM). From one coordinate graph and one tensor description it produces:

- each access's **element address**, split into wave-uniform terms, one per-lane part and a constant, so
  the memory side can put each part where it is cheapest;
- each access's **validity**: exactly the bound checks that can fail, decided from coordinates (never from
  addresses), as one i1 per access (or per vector).

This doc is the one home of the package's facts: its units, kinds, rules, signatures and errors. The
how-to guide with worked examples is [`tensor_transforms_guide.md`](tensor_transforms_guide.md). Memory
delivery (which load or store instruction, buffer or global addressing, which register each address part
rides in) is not this package's job: it consumes an `Access` and decides that.

"Fragment transform" in [`label_flow_and_transforms.md`](label_flow_and_transforms.md) is a different
thing: it moves a datum between register / LDS positions. The transforms here are **coordinate transforms**:
they map one index space onto another.

---

## 1. The model in one picture

```
 author / tiling layer                       tensor_transforms                          memory side
 ---------------------                       -----------------                          -----------
 per access: one plain i32 IR value   IR range walk (reads IR, emits nothing):
 (or int) per graph input,     --->     each value -> constant + per-wave parts + per-lane part,
 in inputs(...) order                   each part with its proven range and multiple-of fact
                                               |
                                       TransformGraph (holds no tensor, no bound)
                                         inputs -> steps (8 kinds) -> outputs
                                               |  value ranges
                                               v
             TensorDesc  --->          which SIDES of which coordinates can fail
        (lengths, strides, dtype:      (input domain, read domain, tensor bound)
         the only home of lengths)             |
                                               v
                                       emit_accesses(b, graph, tensor, accesses,
                                                     vector_axis=, vector_width=)    one per-call table
                                         - ElementAddress: uniform terms / lane / const
                                         - valid (the AND of the failing checks)   ---> load / store
                                         - valid_span (vectors that straddle an edge)
```

The seam in one rule: this package owns coordinates -> coordinates (the graph), turning a plain IR value
into a split coordinate with a proven range (the IR range walk), value ranges, which sides can fail, the one
compare (`emit_bound_check`), validity per access or per vector, and coordinates -> element address (the
stride fold). It knows nothing of pointers, buffers, windows or loops.

---

## 2. The units, one job each

Tiers: **door** = exported by `tensor_transforms/__init__.py` for kernel authors; **toolbox** = exported by
the door too, for tests and custom memory code; **internal** = in a module's `__all__` for other tiling-layer
modules, not re-exported by the door, so no author imports it.

| Unit | Its one job | Tier |
|---|---|---|
| `TensorDesc` (in `helpers/tiling/descriptors.py`) | Holds a tensor's lengths, strides and dtype; the sole owner of strides and of every validity length. | (tiling layer) |
| `TransformGraph` | A frozen value that maps named input coordinates to named output coordinates through typed steps, holding no tensor. | door |
| `make_transform_graph` + `inputs` / `outputs` + one builder per kind | Assemble and check a `TransformGraph` from declarations written in read -> write order. | door |
| `TransformGraph.evaluate` / `.resolve` / `.edge` | Compute the outputs (or every named coordinate) for integer inputs, in pure Python; inspect one coordinate. | door |
| `Node`, `Edge`, `TransformType`, `InputsDecl`, `OutputsDecl`, `EmbedParams` ... `BroadcastParams` | The values the builders return and a graph is read back with. | door |
| `emit_accesses` | The entry: emit, over one shared table, each access's split address and its validity (per element, or for a whole vector with its valid span), every failing side included. | door |
| `Access`, `ElementAddress`, `UniformTerm` | The frozen results `emit_accesses` returns (read, never built by authors). | door |
| `RuntimeValue` | A runtime integer with its stated range and multiple-of fact, passed on as a length; read-only. | door |
| `value_ranges` | The range every coordinate can take for given input ranges, in pure Python. | toolbox |
| `required_checks` -> `BoundCheck`, `CheckReason` | Which side of which coordinate can leave its domain or its tensor bound and so needs a run-time check. The one decision home; it does not decide HOW a side is enforced. | toolbox |
| `graph_injective` | Whether distinct in-domain inputs always reach distinct outputs (store safety). | toolbox |
| `address_of` / `transform_address_of` | A tensor's strides applied to a position (or to a graph's outputs) as one Python int: the golden every emitted address is tested against. | toolbox |
| `emit_coords` | Every named coordinate of a graph as one emitted value. | toolbox |
| `emit_addresses` | One `ElementAddress` per access, all accesses over one shared table. | toolbox |
| `emit_access_validity` | Each access's validity alone. | toolbox |
| `emit_bound_check` | The one predicate: `lo <= value < hi` for one plain value, its per-wave side folded on the scalar unit. | toolbox |
| `emit_accesses_except_buffer_checked` | `emit_accesses` leaving out of `valid` the sides the buffer hardware checks. | internal (sibling surface, see 6) |
| IR range walk (`walk.py`: `split_coordinate`, `value_facts`, `runtime_value_of`, `leaf_varying_within_workgroup`) | Turn a plain IR value into a split coordinate and derive each part's range and multiple-of fact; rebuild the `RuntimeValue` a range marker states; find a leaf that can differ between the waves of a workgroup. Reads IR, emits nothing. | internal |
| `Interval`, `Congruence`, `AffineCoord`, `UniformPart`, `AxisBound`, `CheckSide`, `Side`, `VectorRun`, `identity_graph`, the run queries | The machinery (section 12). | internal |

Package files: `__init__.py` (the door), `graph.py`, `ranges.py` (a leaf module: it also holds
`RuntimeValue`, `value_ranges`, `required_checks`, `graph_injective`), `walk.py`, `address.py`, `emit.py`
(builder first, IR side effects), `runs.py`.

**A public value carries only public types.** Every field of a door or toolbox value is an int, a coordinate
name the graph was built with, an IR `Value`, or a door / toolbox type. A range on a public value
(`UniformTerm.range`, `ElementAddress.lane_range` / `.uniform_delta_range`, `BoundCheck`'s ends) is a plain
half-open `(lo, hi)` pair; `Interval`, `Congruence` and the other internal values never appear on one.

---

## 3. Kinds (eight)

Names are flow-correct (read -> write): `merge` combines N -> 1, `unmerge` splits 1 -> N. **CK and the V4
transform graph use the opposite names** (their MERGE is this `unmerge`).

| Kind | Builder | Computes | Domain it REQUIRES of its reads | Range of what it writes |
|---|---|---|---|---|
| EMBED | `embed(reads, into, coeffs)` | `into = sum(read_i * coeff_i)`, coefficients `>= 0` | none | interval sum |
| MERGE | `merge(reads, into, lengths)` | mixed-radix combine, the last read innermost (read `i`'s weight = product of the later `lengths`) | `read_i in [0, lengths_i)` | `[0, prod(lengths))` when reads are in domain |
| UNMERGE | `unmerge(read, into, lengths)` | split by divide / remainder, the last piece innermost; the outer piece is not wrapped | `read in [0, prod(lengths))` | piece `i` in `[0, lengths_i)` |
| OFFSET | `offset(read, into, shift)` | `into = read + shift` (a negative shift is a halo) | none | shifted interval |
| SLICE | `slice_window(read, into, begin, end)` | `into = read + begin` | `read in [0, end - begin)` | `[begin, end)` |
| FREEZE | `freeze(into, value)` | `into = value` (reads nothing) | -- | `[value, value + 1)` |
| XOR | `xor(reads, into)` | `into[0] = reads[0]`; `into[1] = reads[1] xor (reads[0] & (len - 1))`, `len` = `reads[1]`'s declared domain, a power of two | `reads[1] in [0, len)`; `reads[0] >= 0` | `into[1]` in `[0, len)` |
| BROADCAST | `broadcast(reads)` | reads coordinates that address nothing (a bias broadcast over rows) | none | -- |

**Domains.** A domain is declared in exactly one place: an input (`inputs(k=192)`; `inputs(t=None)` declares
an input over a runtime-length axis, with no domain, bounded only by its tensor), an `unmerge` piece, a `merge`
output (the product, a checked restatement of its reads' domains) or an `xor` output (it carries `reads[1]`'s
domain). OFFSET, EMBED, SLICE and FREEZE outputs have a range, not a domain. Output coordinates get their bound
from the tensor, never from the graph.

**Two range kinds, never mixed.** The "Range of what it writes" column is the *conditioned* range, valid only
when the reads are in their domain; it is used only to decide which checks can be dropped (checks are AND-ed, so
each may assume the ones before it passed). Every emit decision (integer width, contiguity) and every range
reported to the memory side uses the *unconditioned* range: what every lane computes, valid or not. An invalid
lane still computes an address, and its arithmetic must never overflow.

**PAD = offset + the tensor's own length; there is no PAD kind.** CK's `pad(len, left, right)` is two facts in
one object: a coordinate shift (`left`) and a restated validity length (`len`); `right` never affects validity.
Here each half has one home: the shift is `offset(..., shift=-left)` in a graph (or a negative window origin),
and the length is the tensor's `TensorDesc.lengths`, which the graph never holds. Every output is checked against
its tensor bound unless the ranges prove it, so "reads as zero / write dropped" needs no opt-in. A PAD kind would
restate the length (a second home that can drift) or duplicate `offset`. Shift a halo once: the window origin or
a graph `offset`, never both. Negative `offset` shifts are allowed; negative `embed` coefficients are refused
(they would make the per-lane address part negative).

**V4 mapping.** V4 `PAD(left, right)` = `offset(-left)` (or a negative window origin) + the tensor length;
`right` has no counterpart.

---

## 4. The index spaces and the positional rule

- **Graph input `i` = window coordinate `i`.** A plain strided view of a tensor is the identity graph of its
  lengths, so its `R` inputs are the tensor's `R` axes. Graph output `i` is tensor axis `i`.
- **A rank-`r` tile covers the LAST `r` of the window's `R` coordinates**: tile axis `j` = window coordinate
  `R - r + j`. The leading `R - r` coordinates are positioned by the window origin only.
- **Every index this package takes or reports is a graph-input index**: an access tuple's positions and
  `vector_axis`. A caller holding a tile-axis index adds `R - r` first.

Example: a batched tensor `(G, M, K)` with a rank-2 `(M, K)` tile. Graph inputs 0, 1, 2 are G, M, K; tile
axis 1 (K) is graph input 2; a vector along K is `vector_axis=2`. Anti-example: `vector_axis=1` there (the
tile-axis index) puts the vector on M.

**Positional calls.** Coordinate names appear only while BUILDING a graph (`inputs(...)`, the kind builders,
`outputs(...)`) and in what reads it back (`resolve`'s keys, `edge`, `value_ranges`'s keys,
`BoundCheck.coordinate`). Every call that SUPPLIES coordinates takes a tuple in `inputs(...)` order:
`evaluate`, `resolve`, `transform_address_of`, `value_ranges` / `required_checks` / `graph_injective` (one
`(lo, hi)` pair per input) and the access entries. `inputs(m=.., n=..)` and `inputs(n=.., m=..)` are different
graphs, as two functions with swapped parameters are.

**Plain values.** A coordinate is a plain i32 IR value or a Python int. Splitting it into its per-wave and
per-lane parts, and proving each part's range, is the package's job (the IR range walk, section 8); the author
states nothing the IR already shows.

---

## 5. Author-facing data

```python
RuntimeValue           # door, read-only
    .value             # the i32 IR value, for arithmetic: b.add(K.value, ...)
    str(rv)            # its stated facts in plain words, by its IR value's name:
                       #   "%range1: 0 <= value <= 65536, a multiple of 8"
```

A runtime integer with its stated range (its *envelope*) and multiple-of fact. It is received from the tiling
layer and passed on as a length (or a range end); authors never build one. Its `value` is always the result of a
range marker (`arith.assume_range`, which emits no instruction) whose stated attributes are these facts, so any
arithmetic an author writes on `.value` walks back to the same facts (section 8, "Runtime ends"). Two
`RuntimeValue`s over the same IR value that state different facts are an error wherever they meet.

```python
Access(address: ElementAddress, valid: Value | None, valid_span: tuple[Value, Value] | None)
ElementAddress(uniform: tuple[UniformTerm, ...], lane: Value | None, lane_range: tuple[int, int] | None,
               const: int, elem_bits: int,
               uniform_delta: Value | None, uniform_delta_range: tuple[End, End] | None)
UniformTerm(value: Value, range: tuple[End, End], inputs: frozenset[int])
End = int | RuntimeValue | None      # a range end; None = unbounded on that side
```

- `Access.valid`: the i1 "every element of the access is inside the tensor and every domain"; `None` =
  statically always valid, nothing emitted. `valid_span`: set only for a vector whose validity can differ
  between its elements -- `(first, end)`, i32 element indices inside the vector; elements `first <= e < end`
  are valid (none when `end <= first`).
- `ElementAddress` is `sum(uniform) + lane + const`, **in elements**. `uniform` holds one term per distinct set
  of graph inputs (`()` when there is none); `lane` is the i32 per-lane part with its unconditioned
  `lane_range`, which starts at 0 or above; `const` is a Python int that may be negative (a halo). `elem_bits` is
  the element size in bits, so byte proofs also hold for packed sub-byte types. `uniform_delta` (with its range)
  is the access's own wave-uniform term on runtime-stride axes; it is `None` while every stride is an int.
- `UniformTerm.value` is i32 or i64 by its range; `inputs` names the graph inputs (by index) it derives from.
  Terms of different input sets are never summed here, so the memory side can decide per term where it goes (a
  term that does not change inside a loop, for example). Every op emitted for a uniform term is side-effect free,
  so a term's roots are exactly the per-wave values the caller passed in.

```python
BoundCheck(coordinate: str, lo: int | RuntimeValue | None, hi: int | RuntimeValue | None,
           reasons: tuple[CheckReason, ...])                     # toolbox
CheckReason = Enum(INPUT_DOMAIN, READ_DOMAIN, TENSOR_BOUND)      # toolbox
```

One coordinate's run-time check, `lo <= coordinate < hi` on the sides that can fail. A side is `None` when it is
proven (nothing to check there), so the two sides can be enforced separately. `reasons` lists every reason a side
was found failing.

---

## 6. Signatures, by tier

```python
# door -- graph.py
def inputs(**name_to_length: int | None) -> InputsDecl     # None = runtime-length axis (no domain)
def outputs(*names: str) -> OutputsDecl                    # tensor-axis order
def embed(reads: tuple[str, ...], into: str, coeffs: tuple[int, ...]) -> Node
def merge(reads: tuple[str, ...], into: str, lengths: tuple[int, ...]) -> Node
def unmerge(read: str, into: tuple[str, ...], lengths: tuple[int, ...]) -> Node
def offset(read: str, into: str, shift: int) -> Node
def slice_window(read: str, into: str, begin: int, end: int) -> Node   # `slice` is a Python builtin
def freeze(into: str, value: int) -> Node
def xor(reads: tuple[str, str], into: tuple[str, str]) -> Node
def broadcast(reads: tuple[str, ...]) -> Node
def make_transform_graph(inputs: InputsDecl, *steps_then_outputs: Node | OutputsDecl) -> TransformGraph
TransformGraph.evaluate(coords: tuple[int, ...]) -> tuple[int, ...]   # outputs, in outputs(...) order
TransformGraph.resolve(coords: tuple[int, ...]) -> dict[str, int]     # every named coordinate
TransformGraph.edge(name: str) -> Edge                                # Edge(name, domain, source)

# door -- emit.py (builder first, IR side effects)
def emit_accesses(b, graph, tensor, accesses: Sequence[tuple[Value | int, ...]], *,
                  vector_axis: int | None = None, vector_width: int = 1) -> tuple[Access, ...]

# toolbox -- ranges.py, address.py (pure Python, no IR)
def value_ranges(graph, input_ranges: tuple[tuple[End, End], ...]) -> dict[str, tuple[End, End]]
def required_checks(graph, tensor, input_ranges: tuple[tuple[End, End], ...]) -> tuple[BoundCheck, ...]
def graph_injective(graph, input_ranges: tuple[tuple[End, End], ...]) -> bool
def address_of(tensor, position: tuple[int, ...]) -> int             # one int per tensor axis
def transform_address_of(graph, tensor, coords: tuple[int, ...]) -> int   # = address_of(tensor, graph.evaluate(coords))

# toolbox -- emit.py (builder first, IR side effects)
def emit_coords(b, graph, coords: tuple[Value | int, ...]) -> dict[str, Value]
def emit_addresses(b, graph, tensor, accesses) -> tuple[ElementAddress, ...]
def emit_access_validity(b, graph, tensor, accesses) -> tuple[Value | None, ...]
def emit_bound_check(b, value: Value | int, lo: End, hi: End) -> Value | None
```

- `tensor` is a `TensorDesc`; the entries read only its `lengths`, `strides` and `dtype`. A length is an int, or
  a `RuntimeValue` for a runtime-length axis. Strides are non-negative ints.
- `input_ranges` gives every input a range, positionally: `(None, None)` means "not known" and is always
  checked. There is no "trusted" default: being in domain is stated (`(0, n)`) or derived by the walk.
- `emit_coords` returns each coordinate as its parts summed, `(uniform + lane) + const`: i32, or i64 when its
  range needs it, keyed by name (inputs and intermediates included).
- `emit_bound_check` returns `None` when the value's walked range already proves both sides.

**Sibling surface, not for authors.** `emit_accesses_except_buffer_checked(b, graph, tensor, accesses, *,
buffer_checked_sides: frozenset[CheckSide], vector_axis=None, vector_width=1)` is `emit_accesses` with the
named sides left out of `valid`. It is in `emit.py`'s `__all__` for the tiling layer's buffer delivery, its one
importer, and is not re-exported by the door: its caller MUST have the buffer hardware check every side it names,
and the returned `valid` alone is NOT safe. Each named side must be one `required_checks` reports as failing for
some access of the call (a set is per call; per access, a side that is already proven has nothing to drop), and an
empty set is an error. Both entries are one internal body: the first passes no dropped sides.

---

## 7. The validity rule

A coordinate needs a run-time check exactly when its value range can leave one of:

1. its **input domain** -- an input whose supplied range is not inside `[0, declared length)`;
2. a **read domain** -- a range a step's read is not inside (an `unmerge` read past the product of its pieces:
   a K tail, derived from the graph, never written by hand);
3. its **tensor bound** -- an output whose range is not inside the tensor's `[0, length)`.

- Checks on one coordinate are intersected into one `BoundCheck` with a per-side result.
- The access is valid when every check passes. Validity never reads the address.
- Whether an axis is checked follows from data (ranges against domains and bounds), never from a per-axis
  switch.
- `value_ranges` reports unconditioned ranges; the decision uses the conditioned ones (section 3): each
  coordinate's range is narrowed by its own domains, in order, before the steps that read it compute theirs.
- **The bounds come from the tensor.** The toolbox and the entries build `[0, length)` per tensor axis from
  `tensor.lengths`; there is no `bounds` parameter, so no wider bound can be given. A narrower region is a
  different tensor description. To test one value against arbitrary ends (ints or `RuntimeValue`s, either side,
  an empty or inverted range included), call `emit_bound_check`, the one predicate.
- **`RuntimeValue` ends prove relations to a runtime bound by IDENTITY.** An upper side bounded by a runtime
  length `L` is proven only by a computed upper end over the same IR value object with a shift `<= 0` (`L`
  itself, or `L` shifted down); the lower side mirrors it. An int bound is proven numerically (a runtime end
  counts with its envelope's extreme value). Identity works because a length has one home (the tensor's): no
  restated copy exists to break the proof.
- When two failing ends meet on one side of one coordinate, the tighter is kept, so one compare enforces both;
  when which is tighter cannot be known at build time, it is an error (section 13).
- `graph_injective` is conservative: `True` only when every input can be recovered from the outputs through
  steps that lose no information on their in-domain ranges. `False` is a real collision (an `embed` whose
  coefficients let two inputs meet -- a convolution's `h = ho + y` is one; a `broadcast` that drops a
  coordinate) or one the ranges cannot rule out.

---

## 8. The IR range walk (internal)

`walk.py` turns one plain i32 value into a split coordinate -- a constant, per-wave (uniform) parts and one
per-lane part -- and derives each part's range and residue from how the value was computed. It reads each
`Value`'s defining op and emits nothing. It owns no arithmetic: every rule maps an op onto a range operation of
`ranges.py` (section 12).

**The split rules** (top-down; the first match wins):

| The value is | It becomes |
|---|---|
| a Python int, or `arith.constant` | the constant part |
| `x + c`, `c + x` or `x - c`, `c` exact (not an `add nuw`) | `x` split by these rules, `c` moved into the constant |
| uniform (below) | one uniform part (the value itself) |
| `arith.add` (not `nuw`) that is not wholly uniform | both operands split and added: constants summed, uniform parts concatenated left first; when BOTH operands have a per-lane part the add itself is the per-lane part |
| anything else: `gpu.thread_id`, a load, an intrinsic, an `iter_args` value, any op with a per-lane operand | the per-lane part (the value itself) |

**Uniform** (the values LLVM's uniformity analysis is known to prove uniform): `gpu.block_id`, a kernel
parameter, a `tile.readfirstlane` result, the induction variable of an `scf.for` whose start, stop and step are
uniform, a range marker on a uniform value, and a *pure arith op* with at least one operand, all uniform. Pure
arith ops: `add`, `sub`, `mul`, `div`, `mod`, `shl`, `and`, `smin`, `smax`, `zext`, `sext`, `trunc`, `bitcast`,
`cmp`, `select`, `or`, `xor`, `lshr`. A load, an intrinsic or a thread id is never uniform, whatever its
operands; a per-wave value loaded from memory becomes uniform by passing it through `b.readfirstlane`. A value
the walk cannot see is uniform is per-lane: correct, at worst slower. (A divergent value in a scalar operand
would make LLVM wrap every memory instruction in a waterfall loop, so the walk never guesses.)

**The range rules** (`value_facts`). The rule set is closed, and **unknown is a fact, never an abort**: a leaf or
op outside the set, or a rule whose precondition fails, contributes an unbounded range with no multiple-of fact,
and the rows combine as usual, so `x * 128` is a multiple of 128 whatever `x` was computed from. Interval
arithmetic assumes no i32 wrap, exactly as the `nsw` flags the lowering emits let LLVM assume.

| Op | Range | Residue |
|---|---|---|
| `arith.constant` `c` | exact `[c, c + 1)` | `0 mod abs(c)` (`0 mod 1` for 0) |
| `gpu.thread_id` | `[0, max_workgroup_size)` (below) | none |
| `gpu.block_id` | `[0, +inf)` | none |
| `tile.readfirstlane` of `x` | `x`'s | `x`'s |
| `scf.for` induction variable, exact start `l`, exact step `st > 0` | (a) exact stop `u`: `[l, last + 1)`, `last` the last trip's value (`[0, 256)` step 32 gives `[0, 225)`); a loop that never runs (`u <= l`) gives `[l, l + 1)`. (b) a runtime stop (below): `[l, L + c)`, proven by identity. (c) any other stop: its own upper end (`[l, h)`, or `[l, l + 1)` when it never runs, or `[l, +inf)`) | `l mod st` |
| range marker on `x` | `x`'s range intersected with the stated range, then snapped to the residue | the stated one when the marker states one, else `x`'s |
| `x + c`, `c + x`, `x - c` (`c` exact) | shifted by `c` | shifted by `c` |
| `c - x` (`c` exact) | negated, then shifted | likewise |
| `add` / `sub`, neither exact | interval sum (`sub`: plus the negated range); an `add nuw` is unknown | combined (gcd rule) |
| `mul` with an exact `c` | scaled by `c` (`c = 0`: the exact 0) | scaled by `c` |
| `div` / `mod` by an exact `c > 0`, dividend proven `>= 0` | floor division / remainder | likewise |
| `shl` by an exact `c` in `[0, 31]` | scaled by `2^c` when that is proven inside i32; otherwise unknown | `0 mod 2^c` at least (the low bits are zero through any wrap) |
| `and` with an exact low mask `2^c - 1` (either operand) | `x mod 2^c` (exact for every sign) | `mod 2^c` |
| `smin` / `smax` | min / max of the ranges (symmetric) | the residue both share |
| anything else: a kernel parameter, a load, an `iter_args` value, `mul` of two non-exact values, `div` / `mod` by a non-exact divisor or of a dividend not proven `>= 0`, `or`, `xor`, `lshr` | unknown | none |

"Exact" = a width-one int range. IR `div` / `mod` truncate, so they equal the floor operations only for a
non-negative dividend, hence that precondition. rocKE's `shl` carries no `nsw`, so a result outside i32 wraps
instead of being assumed away, hence its precondition; `block_id << 7` still gives `0 mod 128`.

**The loop counter, its three cases.** An exact stop gives the tight end of the last trip, so a constant-bound
loop proves `k + TK <= K` exactly as a runtime one does. A loop that never runs gives `[l, l + 1)`, never an
empty range (every range operation needs a non-empty one). A stop `L + c` over a runtime length `L` gives the
counter the end `L` shifted by `c`, so `for k in [0, K - TK + 1) step TK` proves `k + TK <= K` by identity.

**Runtime ends: how the walk rebuilds a `RuntimeValue` from a bare value.** An *envelope marker* is a range
marker with two int ends. `runtime_value_of(v)`, for `v` the result of one, returns the `RuntimeValue` with the
marker's STATED facts -- exactly what its producer returned -- and `None` for any other value. It is never built
from what the walk derives (the stated envelope contains the derived range, so the rebuilt end is looser, never
wrong, and shares the IR value an identity proof keys on). A runtime stop is a stop that, once exact constants
added or subtracted are removed, is an envelope-marker result `v`: its end is `runtime_value_of(v)` shifted by
the removed constants. A marker's runtime upper end (a second operand) is rebuilt the same way.

**The workgroup size is read once and never silently changed.** The first walk that reaches `gpu.thread_id`
requires `b.kernel.attrs["max_workgroup_size"]` to be set explicitly (the core's launch-bounds fallback is not a
fact the author stated) and records the value it used in `b.kernel.attrs["thread_ranges_workgroup_size"]`. Every
later walk refuses a kernel whose size has changed since: a larger size set after the body was built would launch
more threads than the ranges were proven for.

**Values the same for the whole workgroup** (`leaf_varying_within_workgroup`, for a guard whose early exit ends
whole workgroups). Wave-uniform is not enough there: `readfirstlane(thread_id_x / 64)` differs between waves. The
same for the whole workgroup: ints and constants, block ids, kernel parameters, a range marker, a
`readfirstlane` or a pure arith op whose operands all are, the counter of a loop whose bounds and step are, and a
read-only load -- a scalar global load at such an index through a kernel parameter declared `readonly=True` and
`noalias=True` (or with `addr_space="constant"`). Anything else may differ (a refusal, never a wrong answer).

---

## 9. The structured address (address side)

**A per-lane part never starts below 0.** A per-lane part whose range starts at `lo < 0` (`c - lane`,
`smin(lane - 3, 10)`) is rebased once per call to `lane - lo`, and `lo` moves into the constant, so the value and
its residue are unchanged. This runs at the two places that receive split coordinates: each input coordinate of an
access, and `emit_bound_check` after its split. The rebased range must fit i32.

**How the parts move through the graph** (one shared table per call: a sub-expression that two accesses share is
emitted once, in first-use order):

- `embed` / `merge` / `offset` / `slice_window` / `freeze` keep the three parts apart: constants folded in
  Python, uniform parts scaled and kept per input set, per-lane parts scaled and added.
- `unmerge` splits each part separately and recombines them with exact 0 / 1 carries, innermost piece first (no
  divisibility precondition). The read is `s + L + c` (uniform side, per-lane part, constant). Per level of
  length `k`: the uniform side is divided once per call (on the scalar unit); the per-lane part at most once per
  call, by the method its range allows (nothing when it is below `k`, one compare + select when below `2k`,
  else a division); the constant in Python; the scalar remainder plus the scalar carry gets one compare +
  select per distinct constant; and each access's piece is `t - select(t >= k, k, 0)`, one compare + select,
  never a per-lane division per access. Only the outermost piece keeps a uniform part (the quotient plus the
  scalar carry). A uniform side known to be a multiple of `k` emits no scalar remainder at that level.
- `xor` first adds its reads' parts, `(uniform + lane) + const`; the result is per-lane when a per-lane part
  reached it, else one uniform term keyed by both reads' inputs.
- **The stride fold** multiplies each output's parts by its stride: per distinct set of graph inputs one
  `UniformTerm` (i64 unless its range fits i32), one per-lane sum in i32, and the constant in Python.

**Integer widths** follow the unconditioned ranges, so no emitted `nsw` operation can overflow even in a lane
whose coordinates are out of bounds. A part is i32 when its range is proven inside i32, a uniform part otherwise
i64 (an unknown uniform range is i64); a per-lane part outside i32 is an error. The division of an `unmerge`
needs its dividend in `[0, 2^31)`: when the uniform side is not proven there (it may be negative, a halo
origin), the whole coordinate is split as one per-lane value instead, and when that is not proven there either
it is an error -- a raw `block_id * TM` is `[0, +inf)`, so a block origin that feeds an `unmerge` needs a stated
range first.

**Integer emit rules** (the emitted IR text is pinned by tests):
- every multi-operand sum is a left fold in a fixed order (tensor-axis order for the stride fold, read order for
  `embed` / `merge`); the first term starts the fold; a coefficient or stride of 1 emits no multiply; a zero part
  emits nothing;
- a power-of-two `unmerge` length uses a shift and a mask; any other uses magic division (with the no-unsigned-wrap
  add); every inner piece is the compare + select above, never a remainder by multiply-subtract;
- `xor`'s `reads[0] mod len` is `and (len - 1)`;
- the accesses of one call share every identical sub-expression, built once in ONE per-call table that both the
  address and the validity read.

---

## 10. The bound check (the one predicate)

`emit_bound_check(b, value, lo, hi)` splits `value` with the walk and folds its per-wave side on per-wave values;
one unsigned compare decides. The access entries use the same function body on their table's coordinates.

**The fold.** The coordinate is `s + lane`: `s` its per-wave side (uniform parts plus constant), `lane` its
per-lane part in `[0, L)` (the constant 0 and `L = 1` when it has none). The ends fold and saturate:

```
a = clamp(lo - s, 0, L)       # the lowest valid lane value     (no lower end: not emitted)
c = clamp(hi - s, 0, L)       # one past the highest one        (no upper end: not emitted)
no lower end:  lane u< c
no upper end:  (lane - a) u< (L - a)
both ends:     (lane - a) u< smax(c - a, 0)
```

For every lane in `[0, L)`, `lo <= lane + s < hi` exactly when `a <= lane < c`; a negative `lane - a` reads as a
large unsigned value, so it fails. An empty or inverted runtime window (`hi <= lo`) accepts nothing. The per-lane
compare is always i32. Anti-example: the unclamped `(lane + (s - lo)) u< (hi - lo)` is wrong when `hi - lo`
exceeds `2^31`; the clamp of both ends is what makes the fold exact.

**Fold width.** i32 when every range it touches is proven inside i32 (the common case). Otherwise each end is
pre-clamped once in i64 per distinct (per-wave side, end) over the call, measured from the call's least
constant, and finished per access in i32, so no subtraction can wrap whatever the per-wave values hold
(an unchecked kernel argument included). That fold needs the lane range plus the spread of the constants
below `2^31` (its own error).

**The single compare.** A lone i32 per-wave value `u` with no constant and no per-lane part, checked against
`[0, hi)` (or with no lower end when `u` is proven `>= 0`), `hi` an int in `[1, 2^31)` or an unshifted
`RuntimeValue` whose stated envelope lies in `[0, 2^31)`, is the one
compare `u u< hi`, whatever `u`'s range: a negative `u` reads as `>= 2^31` unsigned. It answers the same predicate
as the fold (tests pin that the two agree) at the cost of one i32 compare.

**Check the raw value, not the marked one.** A caller that checks a value and then states the checked fact with
a range marker passes the value BEFORE its marker: on the marked value the walk reads the stated fact back and
proves the check away.

**Sharing.** Compares are shared per distinct (per-lane value, folded ends), so accesses that differ only in their
constant share a compare when their folded ends coincide.

---

## 11. Vectors

With `vector_width = w > 1`, each access stands for `w` consecutive values of graph input `vector_axis` (a
graph-input index, section 4). Three build-time questions decide it, answered from the access's first-element
coordinates (each carries its step along the vector and its start residue, combined from the constant, every
per-wave part and the per-lane part):

- **contiguous**: the elements are consecutive in memory (element step exactly 1) -- the address is the first
  element's, else an error naming the coordinate and why;
- **uniform validity**: every check has one answer for the whole vector -- a coordinate that does not move, or
  moves by 1 from a start whose distance to every checked end is a multiple of `w` (an int end, or a
  `RuntimeValue` end whose stated multiple-of fact is a multiple of `w`);
- **monotone**: every checked coordinate moves by a constant step -- else an error naming the coordinate.

How steps pass through the kinds: `embed` / `merge` weight and add the read steps; `offset` / `slice_window`
keep the step and shift the residue; `freeze` has step 0. An `unmerge` read of step 1 keeps the vector inside the
inner piece only when it starts at a multiple of `w` and `w` divides the inner length. An `xor` keeps step 1 only
when its first read does not move and is a multiple of `w`, its second read moves by 1 from a multiple of `w`, and
`w` divides its length (`reads[0] = 1` with `w = 2` maps 0, 1 to 1, 0: same block, not ascending).

`valid` means "every element is valid": nothing when statically valid; the first element's checks when validity
is uniform; otherwise the first AND the last element's checks (exact, because every check is an interval and every
checked coordinate is monotone), plus `valid_span`, the valid elements `[first, end)`, from the same folded ends
(a step `st > 1` ceil-divides them by `st`). Every coordinate steps by `>= 0` along the vector (coefficients and
strides are non-negative), so the valid elements form one contiguous span. A runtime length stated as a multiple
of `w` keeps uniform validity (the first element's checks); one with no multiple-of fact takes the
first-and-last path.

---

## 12. How it works inside (internal, for tiling-layer code)

None of these is on the door; they are listed so the glass box stays open.

- `Interval(lo, hi)` -- the half-open range of one value; each end an int, a `RuntimeValue` or `None`
  (unbounded); int ends need `lo < hi`. `UNBOUNDED = Interval(None, None)`. Its pure methods are the one home of
  range arithmetic: `plus`, `shifted`, `scaled`, `floor_div`, `floor_mod` (Python floor semantics), `minimum` /
  `maximum` (symmetric in their operands), `overlaps`, `intersect` (a value known to lie in both; on the upper
  side a `RuntimeValue` end is kept over an int so it can still prove a runtime bound; raises when the two cannot
  both hold), `snapped(congruence)`, `within(lo, hi)`, and `from_pair` / `as_pair`, the only conversions to and
  from the public pair form.
- `Congruence(modulus, residue)` -- `value == residue (mod modulus)`, a multiple-of fact; `plus`, `shifted`,
  `scaled`, `floor_div`, `floor_mod`, `join` (the residue two values share, for min / max). A constant is an int,
  never a `Congruence`.
- `RuntimeValue._shifted(k)` -- the only way to write an end relative to a length (the length's envelope and
  residue are stated once, by its producer).
- `AxisBound.of(lo, hi)` -- one tensor axis's valid extent; a `RuntimeValue` end must be unshifted (a bound is a
  length).
- `AffineCoord(const, uniform, lane, lane_range, lane_congruence)` with `UniformPart(value, range, congruence)`
  -- one coordinate split into a constant, wave-uniform parts and one per-lane part, each with an explicit range
  (`UNBOUNDED` when unknown). Built by the walk (`split_coordinate`), by `of_int`, or by tiling-layer code that
  already holds the parts; `+` emits nothing (two per-lane parts are an error). Its `range` and `congruence` are
  combined from the parts by the operations above. The access entries accept an `AffineCoord` in place of a plain
  value (nothing is split twice).
- `CheckSide(coordinate, Side.LOWER | Side.UPPER)` -- one side of one checked coordinate, taken only by the
  sibling entry's `buffer_checked_sides`.
- `VectorRun(axis, width)` -- the internal form of `vector_axis=` / `vector_width=`.
- `identity_graph(lengths)` -- the graph of a plain window (inputs and outputs `axis0`, `axis1`, ...): a plain
  window and a graph window have their checks decided by the same rule. `_value_ranges` / `_required_checks` are
  the toolbox rules on `Interval`s with explicit `AxisBound`s; `_emit_bound_check` is `emit_bound_check` on an
  already-split coordinate.
- Run queries (`runs.py`): `address_contiguous_run`, `validity_uniform_over_run`, `checks_monotone_over_run`
  (section 11).

---

## 13. Every error, verbatim

`ValueError` unless marked; `{...}` are filled values.

**Building a graph** (`graph.py`)

| Where | Message |
|---|---|
| builder | `embed reads {n} coordinates but has {m} coeffs -- give one coefficient per read` |
| builder | `embed coefficient {c} on read '{r}' is negative -- reversed axes are not supported; reorder the tensor's strides instead` |
| builder | `merge reads {n} coordinates but has {m} lengths -- give one length per read` |
| builder | `unmerge writes {n} coordinates but has {m} lengths -- give one length per written coordinate` |
| builder | `{kind} lengths must be >= 1, got {lengths}` (`inputs`, `merge`, `unmerge`) |
| builder | `{kind} into='{name}' rewrites its own read -- give the step a new output name (each coordinate is written once)` |
| builder | `slice_window needs begin < end, got begin={b}, end={e}` |
| builder | `xor needs exactly two reads and two outputs, got reads={r}, into={i}` |
| assembly | `make_transform_graph needs inputs(...) first and outputs(...) last, got {first} ... {last}` |
| assembly (`TypeError`) | `make_transform_graph got {arg} between inputs(...) and outputs(...) -- every step there is what a kind builder returns (embed, merge, unmerge, ...)` |
| assembly | `duplicate step {node} -- remove the repeat` |
| assembly | `coordinate '{name}' is written by two steps ({a}, {b}) -- give one of them a new output name` |
| assembly | `coordinate '{name}' is read by {node} but never declared or written -- declare it in inputs(...) or write it in an earlier step` |
| assembly | `the steps form a cycle through {names} -- a coordinate cannot depend on itself` |
| assembly | `unmerge of '{read}' declares lengths {lengths} (product {p}) but '{read}' has domain {d} -- the pieces must multiply to the read's domain` |
| assembly | `merge into '{into}' declares lengths {lengths} but its reads have domains {domains} -- restated lengths must match` |
| assembly | `xor reads '{r1}', which has no declared domain -- xor needs a power-of-two domain; split it with unmerge first` |
| assembly | `xor reads '{r1}' whose domain is {d}; xor needs a power-of-two domain to be a bijection` |
| assembly | `output '{name}' is never written or declared` |
| assembly | `input '{name}' is never read -- remove it or use broadcast to consume it` |
| call | `'{name}' is not a coordinate of this graph -- its coordinates are {names}` (`edge`) |
| call | `{caller} got {n} coordinates, the graph has {m} inputs {names} -- give one int per input, in inputs(...) order` (`evaluate`, `resolve`, `transform_address_of`) |
| call (`TypeError`) | `{caller} takes the coordinates as a tuple in inputs(...) order, got {type} -- coordinates are positional` (the same three) |
| bind | `graph has {n} outputs {names} but the tensor has rank {r}` (every function that takes a graph and a tensor) |

**Ranges and checks** (`ranges.py`)

| Where | Message |
|---|---|
| toolbox | `input_ranges has {n} entries but the graph has {m} inputs {names} -- give one (lo, hi) pair per input, in inputs(...) order; use (None, None) when a range is not known` |
| toolbox | `bound of axis {axis} is empty: [{lo}, {hi}) -- an always-empty window cannot be read` |
| toolbox | `coordinate '{name}' must be checked against both {a} and {b} on its {lower or upper} side, and which of the two is tighter is not known at build time -- one check holds one end per side; bound the coordinate by one of them (an input over a runtime-length axis is declared inputs(name=None))` |
| any | `two RuntimeValues over the same value {value} state different facts (envelope {e1} vs {e2}, congruence {c1} vs {c2}) -- write ends relative to a length as length.shifted(k), never restate its envelope` |
| range | `Interval [{lo}, {hi}) is empty -- int ends need lo < hi; an empty range states no fact (to ask whether two ranges meet, use Interval.overlaps)` |
| range | `Interval.intersect: {a} and {b} have no value in common -- the two ranges state contradictory facts; check where each was derived` (also when a range marker's stated range contradicts what the IR proves) |
| range | `Interval [{lo}, {hi}) has no value congruent to {r} mod {m} -- the stated range and congruence contradict each other` |
| internal | `{Interval or Congruence}.{floor_div or floor_mod} needs a divisor >= 1, got {k}` |
| internal | `Congruence modulus must be >= 1, got {m}` |
| internal | `Congruence.scaled(0): the product is the constant 0 -- fold it as the int 0 instead of a congruence` |
| internal | `AxisBound end {end} is RuntimeValue(..., plus={p}) -- a bound is a length (plus 0); an end relative to a length belongs in an Interval: length.shifted(k)` |
| internal | `bounds has {n} entries but the tensor has rank {r} -- give one AxisBound per tensor axis` |
| internal | `AffineCoord {uniform or lane} part has no range -- every part needs an explicit Interval (UNBOUNDED if unknown)` |
| internal | `AffineCoord + AffineCoord: both operands have a lane part ({a}, {b}) -- a coordinate has one lane part; add the per-lane values first and split the sum with the walk` |

**Addresses** (`address.py`)

| Where | Message |
|---|---|
| toolbox (`TypeError`) | `address_of takes the position as a tuple in tensor-axis order, got {type} -- coordinates are positional` |
| toolbox | `address_of got {n} coordinates but the tensor has rank {r} -- give one int per tensor axis, in axis order` |
| bind | `structured address needs non-negative strides; tensor axis {axis} has stride {s}` (also `emit_addresses` and the access entries) |

**The IR range walk** (`walk.py`)

| Where | Message |
|---|---|
| walk (`TypeError`) | `IR range walk: {v} has type {t} -- a coordinate is an i32 value` |
| walk | `IR range walk: {v} depends on gpu.thread_id but the kernel states no max_workgroup_size -- set b.kernel.attrs["max_workgroup_size"] to the launch block size right after creating the builder` |
| walk | `IR range walk: max_workgroup_size is now {new} but thread-id ranges were already derived with {old} -- set it once, before positioning windows from thread ids` |
| walk (internal) | `IR range walk: the marker on {v} states an end over {h}, which is not a runtime value's marker result -- a runtime end is the .value of a RuntimeValue (from runtime_value)` |

**Emitting** (`emit.py`)

| Where | Message |
|---|---|
| entry (`TypeError`) | `access {i} is a {type}, not a tuple -- give each access as a tuple of coordinates in inputs(...) order` |
| entry (`TypeError`) | `coordinate {j} ('{name}') of access {i} is a {type} -- a coordinate is a plain i32 IR value or an int` |
| entry | `access {i} gives {n} coordinates, the graph has {m} inputs {names} -- give one coordinate per input, in inputs(...) order` |
| entry | `vector_width must be >= 1, got {w}` |
| entry | `vector_width {w} needs vector_axis -- give the index, in inputs(...) order, of the input the vector runs along` |
| entry | `vector_axis {a} is not an input index -- the graph has {m} inputs {names}` |
| entry | `vector along input {axis} ('{name}') of width {w} is not contiguous at coordinate '{coord}' ({reason}) -- use a narrower vector, or start it at a multiple of {w} (the walk reads that fact from how the coordinate was computed; a runtime value the IR cannot show it for is stated with runtime_value(..., multiple_of=...))` |
| entry | `vector along input {axis} ('{name}') is not monotone at checked coordinate '{coord}' -- its validity cannot be decided from the first and last element; emit per element (vector_width=1)` |
| emit | `unmerge of '{read}' needs its dividend in [0, 2^31) for magic division, but its range is {range} -- bound the origin with exit_if_tile_outside(b, origin, length), or state a trusted range with runtime_value(b, v, at_most=...)` |
| emit | `lane part of '{name}' has range {range}, outside 32 bits -- per-lane offsets must fit i32; move the large part into the uniform coordinate` |
| check (`TypeError`) | `emit_bound_check takes a plain i32 IR value or an int, got {type}` |
| check (`TypeError`) | `bound check of '{name}': an end is an int, a RuntimeValue or None, got {type}` |
| check | `emit_bound_check of '{name}' got lo=None and hi=None -- that checks nothing; pass the end(s) to check, or emit no check` |
| check | `bound of '{name}' is empty: [{lo}, {hi}) -- an always-empty window cannot be read` |
| check | `bound check of '{name}' needs lane range + constant spread below 2^31, got {lane_hi} + {span} -- narrow the lane range or split the accesses into separate calls` |
| sibling | `buffer_checked_sides is empty -- call emit_accesses` |
| sibling | `buffer_checked_sides names the {lower or upper} side of '{coord}', which required_checks did not report as failing for any access of this call -- only a side that can fail can be left to the buffer` |
| internal | `_emit_bound_check: lane part {v} has range {range}, starting below 0 -- pass the coordinate through _emit_coord_table or emit_bound_check, which rebase it to start at 0` |

The `{reason}` of the contiguity error is one of: `its elements are {n} apart in memory`; `it moves by {s} per
element, and the unmerge into {names} splits it`; `it may start at {r} mod {m}, so the unmerge into {names} can
split the vector`; `the unmerge into {names} has an inner length {inner} that {w} does not divide`; `the xor into
'{into}' can reorder the vector's elements`; `runtime stride`. `{name}` in the check errors is the graph
coordinate's name inside the entries, and the value's IR name (or the int) for a direct call.
