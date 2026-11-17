# LLVM flavor inventory and COMGR capability migration

## Contents

- [Purpose](#purpose)
- [Production compiler boundary](#production-compiler-boundary)
- [Current flavor model](#current-flavor-model)
- [Behavior axes hidden by flavors](#behavior-axes-hidden-by-flavors)
- [Current source consumers](#current-source-consumers)
- [COMGR probe corpus](#comgr-probe-corpus)
- [Facts that are not compiler capabilities](#facts-that-are-not-compiler-capabilities)
- [Completeness rules](#completeness-rules)
- [Migration plan](#migration-plan)
- [Deferred qualification](#deferred-qualification)

## Purpose

This inventory identifies every concern currently compressed into rocKE's LLVM
flavor labels and separates production compiler behavior from offline/artifact
identity. It is the source-derived plan for replacing major-version inference
with focused observations from the COMGR library that actually compiles rocKE
kernels.

## Production compiler boundary

The production path is:

```text
KernelDef
  -> Python or C++ LLVM lowering
  -> textual AMDGPU LLVM IR
  -> selected libamd_comgr
  -> bitcode
  -> relocatable
  -> executable HSACO
```

`clang` and `opt` executables are not on that path. They can help investigate a
compiler, but their acceptance is not evidence that the loaded COMGR accepts the
same module. This capability work therefore interrogates COMGR directly.

The relevant identity is the loaded COMGR binary and the compiler evidence
queried through that handle. A reported LLVM major remains provenance only.

## Current flavor model

Automatic lowering currently asks loaded COMGR for its embedded LLVM version
and maps the major to one lane:

| Reported LLVM major | Lane |
| --- | --- |
| 20 or older | `llvm20` |
| 21–22 | `llvm22` |
| 23 | `llvm23` |
| 24 | `llvm24` |
| newer | refused as unqualified |

The selected lane chooses one structural profile containing:

- a complete target data layout;
- a declaration ABI generation;
- a broad `llvm23_target_ops` bit; and
- a named-barrier pointer-address-space bit.

The same model is mirrored in:

- `platform/python/rocke/core/llvm_profile.py`;
- `platform/cpp/core/llvm_profile.cpp`; and
- `platform/cpp/include/rocke/llvm_profile_internal.h`.

The model is safer than silently feeding one fixed module to every compiler,
but its classifier is not evidence. AMD LLVM development builds and patch sets
can change independent contracts without changing, or in lockstep with, the
reported major.

## Behavior axes hidden by flavors

### Complete module layout

rocKE emits the complete target data layout. Historical forms differ in at
least:

- address-space-8 pointer/index fields;
- the ELF mangling component; and
- address spaces 10 through 15.

The exact layout is one atomic emitted string. COMGR must accept it through
final code-object creation. Parsed differences are diagnostics, not substitutes
for the complete candidate.

### Intrinsic declaration and call ABI

Known independently varying families include:

- buffer-resource spelling and record-count width;
- fp8/bf8 MFMA packed operand type for each homogeneous and mixed family;
- named-barrier pointer address space for each operation; and
- scaled-MFMA historical/current argument counts for each shape.

A single declaration-generation enum assumes these families migrate together.
The capability model does not.

### Processor names

Concrete and generic processor names are compiler inputs. Generic names cannot
be derived by slicing digits from a concrete gfx identifier; only explicit
catalogued mappings are candidates. COMGR's ISA setup and full compilation
answer whether a name works.

### Operation admission

`llvm23_target_ops`, exact flavor checks, and `minimum_llvm_flavor` are umbrella
or ranked gates. Each operation must eventually name the exact COMGR behavior it
requires. The current source contains gates for target-specific ISA helpers,
scaled operations, named barriers, and inline assembly.

### Options and attributes

Compiler-option acceptance already has a production COMGR measurement path in
`gen_compiler_options.py`. Function-attribute parser acceptance does not show
that an attribute changes generated code or metadata. Neither belongs in this
flavor-replacement corpus; effect checks remain qualification work.

### Generated code and semantics

Successful HSACO creation establishes compiler compatibility, not kernel
qualification. Instruction selection, resources, metadata, and numeric/state
correctness remain separate evidence.

## Current source consumers

### Python lowering

`platform/python/rocke/core/lower_llvm.py` currently uses flavor-derived state to:

- resolve the declaration table in `_decls_for_flavor`;
- choose fp8/bf8 packed operand forms;
- choose the buffer-resource intrinsic and count conversion;
- choose named-barrier pointer address space;
- admit broad target operations;
- enforce exact or minimum flavor on inline assembly; and
- emit the profile's complete module layout.

`platform/python/rocke/core/isa/backend.py` also gates target operations through
the broad profile capability.

### C++ lowering

The native engine mirrors these decisions through
`rocke_ll_emission_profile`, `rocke_ll_profile_has`,
`rocke_ll_flavor_at_least`, declaration-table selection, and per-operation
lowering branches. Any production migration must update Python and C++ in the
same change and preserve byte identity.

### Runtime and target selection

`platform/python/rocke/runtime/comgr.py` maps loaded compiler major back to a
flavor for its data-layout-generation guard. Target selection reads committed
flavor columns to decide whether a concrete or generic processor name is
accepted.

Those are useful guards built on the best evidence previously available. They
must not run inside the capability worker, because they would prejudge the
candidate before COMGR sees it.

### Non-capability flavor consumers

The following use flavor as emitted-byte or historical identity rather than as
a compiler fact:

- explicit offline lowering;
- byte-identity matrices;
- golden IR shards;
- cache and artifact partitioning;
- portable-IR replay matrices; and
- C ABI values naming legacy emission modes.

These need a migration or compatibility adapter, not a compiler probe.

`test_flavor_consumer_inventory_is_mechanically_closed` scans both source
engines and pins every current use of emission-profile lookup, broad capability,
ranked/exact flavor admission, data-layout selection, major-to-flavor mapping,
and `minimum_llvm_flavor`/`required_llvm_flavor`. A new consumer fails until it
is classified here and in the migration backlog rather than silently extending
the flavor model.

## COMGR probe corpus

Every candidate is a complete LLVM module compiled in an isolated process
through COMGR's source-to-bitcode, codegen, and executable-link actions.

| Capability | Mechanical corpus rule |
| --- | --- |
| `amdgpu.comgr.explicit_data_layout` | every complete layout rocKE currently or historically emitted |
| `amdgpu.comgr.intrinsic.make_buffer_resource.abis` | every spelling × count width |
| `amdgpu.comgr.intrinsic.mfma_fp8_bf8.packed_operand_type` | one all-family module per packed type |
| `amdgpu.comgr.intrinsic.named_barrier.pointer_address_space` | one all-operation module per address space |
| `amdgpu.comgr.intrinsic.mfma_f8f6f4.call_abi` | one all-shape module per historical/current call form |
| `amdgpu.comgr.accepts_processor_name` | requested processor plus its explicit catalogued generic candidate |

The checkout validator pins exact ordered coverage. Adding a production form
without adding its candidate must fail a closure test in the integration slice.

## Facts that are not compiler capabilities

### Identity and provenance

Library path, binary digest, embedded LLVM version, query source, and processor
identify the context. They do not imply behavior.

### Selection policy

When several candidates are accepted, rocKE still needs an explicit preference
and fallback rule. That rule is reviewed and qualified separately from raw
evidence.

### Artifact identity

Two policies that emit different bytes need distinct golden/cache/artifact
keys. This is a consequence of selection, not a compiler capability.

### Qualification

Byte identity, complete production kernels, generated ISA/resources, and
numeric execution establish properties that microprobes cannot.

## Completeness rules

1. **Declaration closure.** Every flavor-varying declaration and every
   `MATRIX_CALLS` form maps to a bounded candidate family or an explicit reason
   it does not vary.
2. **Operation-gate closure.** Every use of broad profile capabilities,
   ranked/exact flavor checks, and `minimum_llvm_flavor` maps to a named
   capability before migration.
3. **Policy grouping.** A global lowering choice is one candidate containing
   every operation or shape controlled by that choice.
4. **Existing-owner reuse.** Compiler options stay in the existing production
   COMGR option measurement; attributes require effect qualification, not a
   parser-only duplicate.
5. **Full-route fidelity.** Parser success alone is insufficient; accepted means
   an executable HSACO was returned.
6. **Context fidelity.** COMGR library, target ISA, options, probe source, and
   schema participate in cache identity.
7. **Unknown is first-class.** Rejection, crash, timeout, and probe error are
   distinct.
8. **Archaeology retention.** Removing a production workaround adds its emitted
   form to the candidate corpus when it represented a real compiler difference.
9. **Dual-engine policy.** Python and C++ consume the same normalized policy and
   remain byte-identical.
10. **Qualification remains mandatory.** Capability-driven selection does not
    waive complete-module or on-device gates.

## Migration plan

The intended layering is:

```text
exact COMGR identity
        ↓
raw capability evidence
        ↓
explicit rocKE selection policy
        ↓
normalized emission policy
       ↙                     ↘
Python lowerer            C++ lowerer
```

Migrate one independent decision at a time:

1. choose a capability whose candidate corpus is complete;
2. prove the report belongs to the COMGR library that will compile;
3. define preference, fallback, and unknown behavior;
4. serialize/generate one normalized policy for both engines;
5. retain flavor inference as a visible temporary fallback;
6. update Python and C++ together;
7. run byte identity and the relevant compiler/numeric qualification; and
8. remove the corresponding flavor branch only after fallback coverage is no
   longer required.

The first suitable vertical slice is a structural ABI with a small candidate
set, such as buffer-resource count/spelling. The complete data layout needs an
atomic policy because its fields cannot be selected independently in emitted
text.

## Deferred qualification

This research change produces evidence but does not alter lowering, wire a CI
gate, or claim that an accepted compiler is qualified. Production integration
must also establish how a report is generated or retrieved for the exact loaded
library in Python-free provider paths.

The strong conclusion supported by this design is narrow: direct COMGR probing
is a sounder basis for production compatibility decisions than reported-major
inference. Whether each observed distinction is sufficient is tested during the
vertical migrations and their qualification gates.
