# The authoring → packaging handover

This page owns the seam. rocKE's own documentation ends at "the numbers match torch";
hipDNN's descriptor documentation begins at "given a builder that already exists". Nobody
states what an author must produce so that a descriptor can be generated, and nobody
states what changes about the kernel's guarantees once it is packed. That is here.

## 1. The spec dataclass IS the descriptor's spec block

The builder's single positional parameter must be annotated with a dataclass.
`hkp_pack._resolve_spec_class` reads exactly that annotation and hard-fails when it is not
a dataclass
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:159-176`).
The generator's introspection adapter reads the same annotation, so the two agree by
construction rather than by convention
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:4-27`).

Three facts follow, and all three are the author's responsibility:

- **Every field of that dataclass is a descriptor field.** There is no second place to put
  a knob. A parameter that is not a field of the spec is a parameter no descriptor can
  set, which is why `hkp_pack` refuses a builder that has one
  (`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:209-220`).
- **Every field without a default is mandatory in every descriptor.** The packer hydrates
  with `Spec(**fields)`, so a missing required field is a `TypeError` at pack time, not a
  diagnostic. Keep the no-default set small and semantic; anything with a sane default
  should carry one.
- **Field names are the vocabulary the descriptor inherits.** rocKE's spelling and
  hipDNN's differ — rocKE's `bf16` against hipDNN's `BF16`, for one — and the translation
  is declared per kernel in the generator config, not fixed up later
  (`projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_dense.yaml:1-27`).

**`source` is a dotted Python module path through the importable `kernels` package, not a
file path.** `kernels/gfx942/attention_tiled_2d.py` is resolved through the installed
rocKE wheel, which is why a rocKE descriptor folder ships no sources at all. The packaging
README calls it the single biggest clarity trap in the format
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/README.md:48-52`),
and that is accurate. The generator accepts the slash spelling and normalizes it
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:46-58`); the packer
imports the dotted form.

A `kind: rocke` descriptor carries exactly `source`, `builder` and `spec`
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/descriptors.py:296-298`).
RFC 0017 introduces the kind with `entry`/`build` vocabulary
(`projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md:1600-1606`); where the RFC
and the shipped validator disagree on spelling, the validator is authoritative.

### The observation that discharges this

`introspect()` reports the spec class, every field with its type and whether it is
required, the signature verdict, and the arches its `supports_*` predicate accepted
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:213-221`). Pass the
config's own `spec` values, because arch support is a property of a specific spec and a
synthesized one usually just trips the spec's own validation.

An **empty** `supported_arches` means *could not be determined*, never *none*: rocKE
declares arch support nowhere and it is only ever derived by asking
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:84-87`). Report a probe
result as probed.

## 2. Baked by default, and what that obliges

Everything in the spec is compiled into the binary unless the kernel deliberately emits it
as a kernel parameter. The consequence is a matcher obligation:

> **Every baked quantity must be a KMD field that a `$kernel.*` criterion pins.**

The engine cannot branch on a field the descriptor does not carry, and a graph whose value
differs from the baked one is not refused — it is served by a kernel compiled for a
different problem. That failure is silent at the hipDNN layer: no exception, no status
code, a wrong number or a buffer nobody wrote. It has already happened in this tree, when
a `batch` value baked into buffer-resource extents was absent from the KMD and a batch-4
graph matched a batch-1 kernel. `rocke-mining.md` states the general form —
wrong geometry can leave output unwritten without raising an error
(`projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine/rocke-mining.md:87-88`).

So, for every field, decide and record one of three dispositions:

| Disposition | Meaning | KMD obligation |
|---|---|---|
| **Baked** | The builder branches on it, or it enters an address formula or a trip count that is compiled in | **Pin it** with a criterion. Omitting it is the `batch` defect |
| **Runtime parameter** | The body emits it as a kernel argument; the binary is indifferent to its value | **Do not pin it.** Pinning costs one packed instance per value of something the binary does not care about |
| **Matcher-only** | Neither baked nor a kernel argument — it constrains which graphs are legal | Pin it, but do not expect it in the binary |

The partition must be exhaustive and disjoint, and the declaration format for it belongs
to the generator, not to this skill —
`projects/hipdnn/tools/IngestorGenerator/README.md` and
`projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine/rocke-mining.md:44-64`.

### The decision rule for a new field

When you add a field to a spec, the default is *baked*, and the default is expensive. Make
it a runtime parameter when **both** hold:

1. The field changes address-arithmetic bounds, loop trip counts that are already dynamic,
   or buffer extents — but **not** the instruction sequence the builder emits.
2. Nothing in the builder branches on it structurally. A field the builder reads inside an
   `if` that selects a different body, a different tile shape or a different epilogue is
   baked by definition, and declaring it a runtime parameter would be a lie the compiler
   cannot catch.

The cost of getting this wrong in the baked direction is measured, not theoretical: a
shipped gfx950 dense pack held 847 distinct descriptor keys, 454 once `batch` was ignored
— `batch` alone was a 1.86x multiplier before the sequence-length axes. The cost of
getting it wrong in the runtime direction is a wrong answer.

### `runtime_param_fields`, and a branch check you must run

A spec class may declare a `runtime_param_fields` property naming the fields it emits as
kernel parameters. Where it exists, the baked set is
`dataclasses.fields(spec) - spec.runtime_param_fields`, and the obligation for those fields
**inverts**: they must not be pinned.

**This mechanism is not present on every branch.** Check before relying on it:

```bash
grep -rn "runtime_param_fields" dnn-providers/hip-kernel-provider/rocke/library/
```

No hits means every field of every spec is baked, the first row of the table above is the
only row, and any collapse you were planning is not available yet. Do not assume from a
kernel's documentation that a field is a runtime parameter; the property is the only
evidence.

## 3. Which checks stop running when the kernel is packed

`hkp_pack` calls the builder's own support predicate before building
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:294-343`)
— **once, at pack time, against the descriptor's spec values.** Nothing calls it again.

rocKE's own docstrings are actively misleading here. A preflight check that justifies
itself on the grounds that "it runs per launch, since the dispatcher calls `supports()`
before the cache lookup" is describing rocKE's in-process JIT path. On the hipDNN path
there is no cache lookup and no `supports()` call. **Read every rocKE `supports_*` and
preflight docstring with that substitution in mind.**

The discriminator for what must be replicated is narrow and precise:

> For any check whose inputs include a field that is *not* baked, the check must be
> re-expressed as a **matcher criterion** evaluated per graph, because the packed binary
> now accepts values the pack-time check never saw.

A check over baked geometry alone is genuinely discharged at pack time and needs nothing.
A 32-bit addressing bound over shape fields is the case that matters: past the bound, the
offsets lower to `nsw` i32 arithmetic, so signed overflow is undefined behaviour — the
address chain may be poisoned outright rather than merely reading the wrong place. That is
a silent-wrong-numbers failure, not a clean abort.

Matcher criteria are a declarative `{"op": [args]}` expression tree over the symbols the
engine's pattern bound
(`projects/hipdnn/docs/rfcs/0017_UniversalKernelDescriptor.md:698-724`), and arithmetic
over normalized-problem fields is already expressible in it. Emitting them is
descriptor-generation work, not format work — but naming *which* ones are owed is yours,
because you are the only one who knows what the builder stopped checking.

## 4. A packed `.kpack` is not a finished integration

What a successful pack proves: the builder imported, the spec hydrated, comgr lowered it
for the named arch, the code object went into an archive, and the shipped descriptor was
rewritten to `kind: kpack` with a matching `sha256`.

What it does not prove: that hipDNN ever dispatches the kernel.

**No rocKE-specific native pack exists in this tree.** Both real in-tree examples borrow
the pointwise pack's native symbols, so they prove the packaging path — authored
descriptor, comgr-lowered kernel, kpack archive, install layout — and explicitly not a
rocKE-specific runtime dispatch
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/README.md:81-88`).
Writing that pack is named as the next step in both sources, it is its own planned work
with its own design, and **this skill does not authorize it.**

**The `hiprtc_file` drop-in branch does not exist for you, but its condition does.** There
is no runtime rocKE adapter — `ROCKE_BUILDER` falls through to the "no kernel source
adapter" diagnostic (`IngestorKernelCode.hpp:354-359`) — so a rocKE kernel reaches the
loader as `kind: kpack` and can never be dropped in as source. What carries over is the
reason that branch is dangerous: borrowing another pack's symbols hands that pack's
handler the launch geometry, the kernarg list and the number of launches, and every
mismatch is silent, exactly as
[hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md) states it for hipRTC. For
rocKE that is not the exceptional branch: today it is the only one.

Still owed, and owned by
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md), not by you:

1. The native pack and its hooks — the symbols a descriptor actually resolves against.
2. The descriptors and matchers, including every criterion §3 above says is owed.
3. Registration with the testing system.
4. Graphs that verify the kernel and that assert **which** engine served them, because a
   suite exercises the winning engine for a graph, not every engine that could serve it.

Hand it these facts, stated explicitly, because nothing downstream can recover them from
the artifact:

- The spec dataclass, its field dispositions, and the no-default set.
- The probed arch list, labelled probed.
- Every baked constant, especially strides and extents that the kernarg signature does
  **not** take — a kernel without stride arguments cannot honour arbitrary strides.
- The launch geometry: the grid and block formulas, every branch in them, and the fields
  that decide each branch.
- The kernarg signature in order, and whether the ABI is fixed or conditional.
- Which pack-time checks are now unenforced, and the criteria that must replace them.

The engine restates every one of these in C++, and **no build step, packer check,
validator or test compares the two statements.** On a mismatch the kernel does not fail;
it computes something else. [rocke-mining.md](../hipdnn-ingestor-engine/rocke-mining.md)
owns extracting these facts and is written for a reader who did not author the kernel —
you did, so state them rather than making it mine them.
