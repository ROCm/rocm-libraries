# Tagged values: class families and duck-typed peers

A value that can be one of
several classes becomes one tagged struct in the generated C ABI, a kind tag
plus the members, with the dispatch generated from it. This is the shape the
hand port already invented by hand (`rocke_attn_bwd_lds_view_t` with
`kind`, `rocke_attn_bwd_mask_policy_t`, `rocke_attn_bwd_dq_sink_t`, the
dense shell's `BufferRowSlab` as a `rocke_attn_bwd_row_slab_t`).

## Which classes form a family

- **Polymorphic subclasses.** A class of a backward module and its
  subclasses in the same module form a family when a subclass overrides a
  method or property other than `__init__` / `__post_init__`, or a class
  attribute. Subclasses that only add fields and a constructor
  (`StridedBatchedAddr`, `ThdOffsetAddr`) keep the prefix-layout rule of
  decision `d0a010a` (no kind).
- **Duck-typed peers.** A class of a hand-ported module (no Python facts for
  it in the translated module) that callers pass where Python passes the
  family's class. Python cannot show it from the translated module, so it
  is declared: `data/abi/peers.json`, `{"<module>": {"<Class>":
  ["<peer module>:<Peer>"]}}`. The peer's fields and methods come from its
  Python source.

## Tag layout

    typedef enum rocke_attn_bwd_lds_view_kind {
        ROCKE_ATTN_BWD_LDS_VIEW_KIND_LDS_VIEW = 0,      /* the base: a zeroed struct */
        ROCKE_ATTN_BWD_LDS_VIEW_KIND_DMA_SLAB_VIEW = 1,
        ROCKE_ATTN_BWD_LDS_VIEW_KIND_XOR_LDS_VIEW = 2
    } rocke_attn_bwd_lds_view_kind_t;

    typedef struct rocke_attn_bwd_lds_view {
        rocke_attn_bwd_lds_view_kind_t kind;
        /* the base's fields, then each member's own fields in class order
         * (subclasses, then peers); a field several members have is one
         * field (same name and C type; an int field joins an
         * IntOrValue field of the same name) */
    } rocke_attn_bwd_lds_view_t;

- One struct per family, named after the base. Every member class maps to
  it (`struct_name(DmaSlabView)` is `rocke_attn_bwd_lds_view_t`); members'
  functions keep their own names (`rocke_attn_bwd_dma_slab_view_index`).
- Kind constants in base, subclass (source order), peer order. The base is
  0, so a zeroed struct (memset, `{}`) is the base class.

## How callers construct values

- Dataclass literal of a member: `.kind = <member constant>` first (left out
  for the base: zero).
- `_init` / `_default` of a member set `kind` after the memset.
- Hand callers that build a peer set `kind` themselves (the dense shell's
  `rocke_attn_bwd_buffer_row_slab`, in the addr caller patch).

## How dispatch is emitted

- **Exported entries.** The C entry of a family method `m` that more than
  one member implements (resolved through the bases) starts with a
  `switch (self->kind)`: each member whose implementation differs calls the
  implementing class's C entry (`rocke_attn_bwd_<impl>_<m>`) with the same
  arguments; the default case is the generated body. A peer's
  implementation is its hand C entry, declared `extern "C"` in the
  generated .cpp with the base entry's parameters. The C signatures must be
  equal, else the entry is a hole.
- **Inside the module.** `x.m(...)` on a family value calls an internal
  dispatcher `py_<Static>__<m>` (one per receiver class and name, emitted
  once) that switches on `kind` to the internal `py_<Impl>_<m>`. The
  receiver's static class is the method's own class for `self`, else the
  family base; only members derived from it are cases. One implementation
  is a direct call. Signatures must agree (else hole).
- **Class attributes** a subclass overrides read through the same switch.
- **`isinstance(x, C)`** on a family value is a kind test (C and its
  subclasses). `Base.m(self, ...)` / `Base.prop.fget(self)` is a direct call
  of the base's implementation.

## Typing

- Recorded tag sets whose classes all belong to one family are the family
  type (`LdsOperand.smem` records `DmaSlabView | LdsView`).
- A `Value` annotation whose recorded types are only classes of one family
  is the family (the recorder shows the annotation is loose).

## Unions of unrelated alternatives (exported parameters)

A parameter whose record shows several unrelated alternatives (`Value` and
a class or family, `load_frag_lds(smem)`) is a tagged union struct, guarded
so several headers can define it:

    typedef enum rocke_attn_bwd_value_or_lds_view_kind {
        ROCKE_ATTN_BWD_VALUE_OR_LDS_VIEW_KIND_VALUE = 0,
        ROCKE_ATTN_BWD_VALUE_OR_LDS_VIEW_KIND_LDS_VIEW = 1
    } rocke_attn_bwd_value_or_lds_view_kind_t;
    typedef struct rocke_attn_bwd_value_or_lds_view {
        rocke_attn_bwd_value_or_lds_view_kind_t kind;
        rocke_value_t* value;
        rocke_attn_bwd_lds_view_t lds_view;
    } rocke_attn_bwd_value_or_lds_view_t;

Passed by value. Callers (generated code) build it with a designated
initializer of the alternative they hold; `isinstance(x, C)` is a kind test
and narrows `x` to the member (the other one in the else branch when there
are two). An annotation that names one of the recorded alternatives (`smem:
Value`) does not narrow the union.

## Locals and returns (internal only)

- A local bound to values of different classes in if-blocks that each end in
  a return or raise (`_normal_layout`'s `lay`) is one C++ variable per block
  (the binding cannot flow out). Applied only after the dry pass refuses the
  union, so no existing output changes.
- A return annotated with `Any` whose statements return several struct types
  (and `None`) widens that item to `std::variant<A, B, std::monostate>`.
  Such functions are private; the exported ABI never sees a std::variant.

## Not covered

- A peer that wraps a family value (the dense shell's `DenseFixedAddr`
  holds a `StridedTensorAddr` and overrides `slab` by delegation): its state
  cannot be members of the struct it would contain. Its C-only field and
  dispatch fill stay.
- A family one of whose members has a field with no type (the sinks'
  `RegisterAccumulateSink._layout: tuple | None`) does not derive; that module
  keeps one struct per class (the old rule) and says so in its refusals.
- Internal dispatch to a duck-typed peer (a call inside the module on a value
  that may be a peer) is a hole: only the exported entry dispatches to it.
