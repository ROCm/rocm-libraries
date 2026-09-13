# The ABI and versioning contract

Status: proposed design, prototype-backed. This chapter is normative: it states the rules a
provider, a loader, and a symbol map must obey. Where a rule has a proof, the proof's ctest
is named. For the prose-level provider protocol, see the reference spec
[provider-protocols.md](provider-protocols.md).

There are two independent versioning mechanisms in this tree, and they answer two different
questions. Keep them separate in your head:

1. **The table ABI header** governs *dispatch-table compatibility inside one process* - can
   this consumer call this provider's table safely. It is a struct convention, checked at
   runtime.
2. **ELF version nodes** govern *symbol resolution across libraries* - which
   `rocblas_sgemm` a caller binds to when more than one exists. It is a linker construct:
   the `.dynsym` entry keeps the bare name, while `.gnu.version` and `.gnu.version_d` carry
   the node association.

## Mechanism 1: the table ABI header

Every dispatch table and every extensible record begins with the same three fields
(`protocols/include/rocm/interfaces/common.h`, `rocm_interfaces_abi_header`):

| Field | Type | Meaning |
| --- | --- | --- |
| `struct_size` | `uint32_t` | how many bytes this table actually is |
| `abi_major` | `uint16_t` | incompatible-change counter |
| `abi_minor` | `uint16_t` | compatible-addition counter |

The rule that falls out of this:

> A consumer accepts a table whose `abi_major` matches what it was built for and whose
> `struct_size` is at least the size it needs. It uses the prefix it understands and
> ignores any tail. A required entry that sits past the reported `struct_size` means the
> provider is incompatible - it is rejected, never called into garbage.

So you grow a table by **appending** function pointers to the end and bumping `abi_minor`
(the provider response's minor is enforced at selection, but the dispatch table's own
embedded header is not yet read; see the implementation-status note). You never reorder a field,
never insert in the middle, never repurpose a field's meaning.
Old callers keep working because the prefix they read is byte-for-byte the same. This is the
same discipline the response struct's `dispatch_table_size` enforces at selection time: a
provider that reports a table too small for the required entries is skipped.

### Implementation status (prototype)

The prototype enforces the selection-time prefix and minor floors through the provider
response, rather than by reading the dispatch-table header. At selection
(`runtime/src/provider_registry.cpp`, `ProviderRegistry::query_entry`), the runtime requires
an exact `abi_major`, a provider `abi_minor` at least as new as
`ROCM_INTERFACES_ABI_MINOR`, and a `dispatch_table_size` at least as large as the
`required_table_size` requested by the loader. Domain loaders currently request `sizeof` the
whole table (for example, `blas_loader.cpp` requests `sizeof(rocm_blas_provider_v1)`), cast
the returned table to that type, and null-check required entry points.

`rocm_interfaces.table_abi_negotiation` proves the registry rule in four discriminating
cases: exact minor/exact size and newer minor/larger table are accepted; an older minor or a
table shorter than the required prefix is rejected. What remains unimplemented is
per-domain optional-tail use: no consumer reads the dispatch table's own embedded
`rocm_interfaces_abi_header`, and current domain loaders require their whole current table.
An optional appended entry therefore needs a loader that requests only the stable prefix,
checks the reported size before reading the tail, and supplies a fallback.

The base ABI is stamped in the header itself: `ROCM_INTERFACES_ABI_MAJOR` is `1`,
`ROCM_INTERFACES_ABI_MINOR` is `1`.

## Mechanism 2: ELF version nodes

A version node is a label the linker attaches to an exported symbol. `rocblas_sgemm` tagged
with node `ROCBLAS_ABI_5` is rendered `rocblas_sgemm@@ROCBLAS_ABI_5` by `nm -D`/`readelf`; the
`.dynsym` name stays `rocblas_sgemm` and the node link lives in `.gnu.version_d`. To
the C source it is still `rocblas_sgemm`; to the dynamic loader it is a distinct symbol. Two
majors that both define `rocblas_sgemm` under different nodes coexist without interposing
each other - each caller binds to the node it was linked against.

- `@@NODE` marks the **default** definition (the one a plain link picks up).
- `@NODE` marks a **non-default**, older definition. A binary that recorded a requirement on
  `NODE` at link time binds to it through ordinary relocation resolution; code that did not link
  against it can still request it explicitly at runtime with `dlvsym`.

You assign nodes with a version script (a `.map` file) passed as
`--version-script=<file>`.

### The version-node registry

These are the nodes this tree defines. Vertical-slice nodes are load-bearing within this
prototype loader/provider; test-fixture nodes exist to prove the mechanism across majors and shapes.

| Node | Where | Role |
| --- | --- | --- |
| `ROCBLAS_ABI_5` | `loader/rocblas_loader.map` | Slice (load-bearing): tags the 11 real rocBLAS loader entry points (create/destroy handle, stream and pointer-mode accessors, `saxpy`/`sdot`, the `sgemm` family). |
| `ROCM_INTERFACES_PROVIDER_1` | `providers/provider.map` | Slice (load-bearing): tags the single provider bootstrap symbol of every recording and system-backed provider; hides everything else. |
| `ROCBLAS_ABI_5` / `ROCBLAS_ABI_6` / `ROCBLAS_ABI_7` | `tests/abi04_rb5.map`, `tests/abi03_provA.map`, `tests/abi03_provB.map` | Test: three distinct majors used to prove co-residency, ordering, and interposition defeat. |
| `ROCBLAS_ABI_6` | `tests/abi05_rocrand_cpp.map`, `tests/abi06_data.map` | Test: proves nodes attach to C++ mangled/RTTI symbols and to data objects. |
| (anonymous, no node) | `tests/abi03_anon.map` | Test: the negative control - same symbols, no node, so interposition reproduces. |

### Symbol-map idioms

Two shapes appear in this tree. Use the right one for the job.

**Named node (co-residency).** Use when the symbols must survive next to another major.
The node name is the ABI contract:

```
ROCBLAS_ABI_5 {
  global:
    rocblas_create_handle;
    rocblas_sgemm;
    ...
  local:
    *;
};
```

**Named node for leak containment.** Use when you only need to hide everything but a known
allowlist and co-residency is not the goal - for example a provider that exports one bootstrap
symbol. The node name still versions the allowlisted symbols (here
rocm_interfaces_provider_query_v1@@ROCM_INTERFACES_PROVIDER_1); leak containment is the reason
for the map, not the naming:

```
ROCM_INTERFACES_PROVIDER_1 {
  global:
    rocm_interfaces_provider_query_v1;
  local:
    *;
};
```

A truly anonymous version script omits the node name and starts directly with `{` (see
tests/abi03_anon.map); it controls symbol visibility only and assigns no version node - which is
exactly why it is the negative control.

`global:` is the allowlist. `local: *` hides everything else - your helpers, your C++
runtime, the leaked `std::filesystem` symbols. For C++ symbols the `global:` entries are
**mangled-name globs**, not source names: `_ZN11rocrand_cpp5error*` matches the methods of
`rocrand_cpp::error`, `_ZTVN..`/`_ZTIN..`/`_ZTSN..` match its vtable, typeinfo, and
typeinfo-name. An `extern "C++" { "rocrand_cpp::error::*"; }` clause is a trap here - it
catches the RTTI but misses the methods.

## When a new SONAME major is required

Bump the public SONAME major when, and only when, the public call ABI cannot remain
identical. Everything that can be done compatibly must be:

- Adding a public function never changes an existing provider-table prefix. Optional
  capabilities are appended and guarded by `struct_size` (Mechanism 1).
- Existing enum names, values, and underlying types are never changed. New enum values get
  explicit, previously-unused numbers at the end.
- Existing record fields are never reordered. A caller-sized record may consume documented
  reserved storage only after size/alignment tests prove every supported old caller stays
  valid; otherwise the new major indirects through edge allocation.

When a break is unavoidable, do not edit the old declaration in place. Add the new spelling,
retain the old major's metadata and generated loader, adapt the old call forward at the
loader edge, and give the new public ABI its own SONAME major and its own version node. Both
majors then coexist by Mechanism 2. (This absorbs the former `api-change-process.md`; the
step-by-step recipe lives in [05-extending.md](05-extending.md).)

## The non-vacuity proof recipe

A test that asserts "the symbol carries version node X" is worthless if it would also pass
when the versioning is broken. Every ABI proof in this tree is designed to fail when the thing
it checks is absent - that is the non-vacuity discipline below. The core proofs carry
explicit negative controls. For the three-line ordering proof,
`abi04_three_line_order` is paired with `abi04_same_node_negative`: the control puts all
three DSOs on the same `ROCBLAS_ABI_6` node and requires the `ABI_5` and `ABI_7` lookups to
be nil everywhere. When
you add a proof (see [05-extending.md](05-extending.md)), make it non-vacuous the same way:

1. **Positive.** Build the DSO the correct way. Assert the exact node is present on the
   exact symbols - `nm -D` shows `sym@@NODE`, and the node name is the one you expect.
2. **Negative control.** Build a second DSO with the node removed or wrong (the anonymous
   map, or all symbols collapsed to one node). Assert the check now **fails**. If it still
   passes, your assertion is not discriminating and the proof is vacuous.
3. **Genuineness.** Assert the DSO is what it claims: an lld build carries an lld
   `.comment` stamp (a bfd build has none), an ASan build carries `__asan_` symbols, a TSan
   build carries `__tsan_` symbols. This catches a silent fallback that would make the test
   pass for the wrong reason.

The worked examples are in [04-hardening.md](04-hardening.md).
`abi05_cpp_mangled_version_node` (positive plus node-dropped, RTTI-removed, and no-ODR-use
negatives) is one complete template. The other is `abi04_three_line_order` paired with
`abi04_same_node_negative`; the `_lld` mirrors run the same positive/control pair under lld.
