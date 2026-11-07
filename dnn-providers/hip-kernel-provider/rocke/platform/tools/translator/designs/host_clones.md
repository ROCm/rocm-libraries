# Host clones: one generated version per value kind

Code that runs on plain host
ints and on builder values is generated once per value kind, two versions
named by a rule. The hand port did this by hand (`lane_source` /
`lane_source_host` in the ring port, `rocke_attn_bwd_host_band_t` in the
band port, `int_eval` over a scratch builder in the frag port).

## Value kinds

- **IR.** The builder is the `IRBuilder` (`rocke_ir_builder_t*`); values are
  `rocke_value_t*`. Every function has this version (unchanged).
- **Host on H.** The builder is a value of a host builder class H; values
  are Python ints (`int64_t`).

## Host builder classes

A class is a host builder when it has no state (no fields, no constructor)
and every method it defines is named like an `IRBuilder` method (a name in
the IR sigmap). Today: `HostInt` (band; ring imports it) and
`IntLaneBuilder` (frag). Python passes them where it passes the builder;
the recordings show them next to `IRBuilder` (`dma_lane_source`, the
`FragCoords.base` closures).

- **Methods.** Every parameter but `self` is a host value. A parameter the
  recordings never saw (`HostInt.sub`, `select`, `lnot`, ...) is typed
  `int` like the recorded ones; Python compares and selects on ints and
  bools alike, and C++ converts bool to int.
- **Instances.** A host builder has no state, so `H()` and a module global
  `_INT = H()` lower to a value-initialized struct (`H{}`).

## Which functions get a host clone

Demand-driven. A call of a module function or method whose IR builder
parameter receives a host builder value of class H asks for the callee's
host clone on H. Inside a host clone the builder is the H value, so its own
calls ask for further clones, transitively. A function whose recordings show
only `IRBuilder` still gets a clone when asked (`static_kv_tile_plan` asks
for `q_tile_range` and `is_edge_tile`). A clone that cannot be translated is
a hole of its caller ("call to f on H, which is not translated"), never of
the IR version.

## Functions on ints called with IR values

A function with no builder whose own version takes ints (annotated
`tuple[int, int]`, like `_mn_k`) but whose IR-kind callers pass IR values
(the recordings show both) gets an IR version, `py_<name>__IRBuilder`, typed
from the recorded alternatives that hold IR values. It is asked for the same
way: a call that passes `rocke_value_t*` where the int version takes
`int64_t`.

## Recorded types of dual code

The recorder mixes the calls of both kinds (`lane: Value | int`, a return
`tuple[Value, Value] | tuple[int, int]`). The IR version of a lifted closure
and of an `__IRBuilder` version keeps the alternatives that hold IR values;
the host clone swaps the IR version's values for ints. Functions translated
before host clones keep their types (`dma_lane_source`'s lane stays int-or-value).

## Naming

- Internal functions: `py_<name>` (IR, unchanged) and `py_<name>__<H>` (the
  host clone on H); methods `py_<Class>_<m>__<H>`; the IR version of a
  function on ints `py_<name>__IRBuilder`.
- A lifted closure: `py_<enclosing>__<closure>` and
  `py_<enclosing>__<closure>__<H>`.
- Host clones are internal. The C ABI keeps one entry per Python function
  (the IR version). The hand ABI's single host entry
  (`rocke_attn_bwd_dma_lane_source_host`) has no caller outside the ring
  port; if one is needed later, its C name is the IR entry's plus `_host`.

## Types inside a host clone

The clone's signature is the IR version's with the value kind swapped:

- the IR builder parameter becomes the host builder (`const <H struct>*`);
- `rocke_value_t*` becomes `int64_t`; the int-or-value union
  (`rocke_attn_bwd_iv_t`) becomes `int64_t`; inside optionals, arrays,
  tuples and vectors the same, element by element;
- everything else is unchanged.

Locals are typed by the dry pass as usual: H's methods return ints.
`isinstance(x, int)` of an int-or-value is statically true in a host clone.

## Platform emitters called with a host builder

A platform method or helper that has only an IR C twin (`LayoutMap.coord`,
`as_i32`) gets its host twin from the IR one: run it on a scratch builder
with constant inputs, then fold the values it returns with H's own methods
(`py_adapt::host_fold`, given a per-H op table the translator emits from H's
methods). The fold keeps H's semantics: `IntLaneBuilder.div` floors,
`HostInt.div` truncates, as Python does. Only the integer ops a host builder
defines can be folded; any other op is a `ValueError` at run time (Python
would fail with an `AttributeError` on H at the same call).

## Closures called with both kinds

A local closure called with an IR builder and with a host builder
(`FragCoords.base`: `emit_base(b, lane)` and `int_coords(lane)` on `_INT`)
is lowered once per kind it is called with.

A `Callable` field of a dataclass whose values are closures of this module
(`FragCoords.base`) is a tagged member, the tagged-value shape applied to closures: a
kind per closure definition site (in source order, 0 the first) plus each
site's captured values as fields of the struct (named
`<field>_<site>_<captured name>`). Calling the field dispatches on the kind
to the site's lifted function, one per value kind (Naming). The kind enum
is `<struct>_<field>_kind_t`, its constants
`<STRUCT>_<FIELD>_<ENCLOSING FUNCTION>`; the captured values are, in order,
the closure's parameters with defaults (`operand_coords`' `_layout=layout`:
evaluated where the closure is defined) and its free variables in source
order (types from the recorder's `cells`, the closure's captured values).
Python reads a free variable when the closure runs, so a captured name must be
bound once in the enclosing function; otherwise the construction is a hole.
A lifted closure, like a host clone, is an internal function, not a unit of
the report.

## Imported classes built on a host builder

A platform class constructed with a host builder (`BandEmitter(hb,
bounds)` in `static_kv_tile_plan`) has a host image: a private struct with
the IR struct's fields, values swapped as above, filled by the IR
constructor on a scratch builder and folded field by field.

## Not covered

- A host builder with state (none today).
- A function called with two different host builders gets two clones; none
  today.
- Generators and `scf` loops inside a host clone (none of the targets).
