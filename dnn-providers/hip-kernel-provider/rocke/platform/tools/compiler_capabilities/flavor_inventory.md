# LLVM flavor inventory and compiler capability taxonomy

## Purpose

This document traces rocKE's LLVM flavor-dependent source and tests to the
concrete compiler behaviors those labels currently represent. It is source-code
research for replacing broad compiler-label inference with focused behavior
probes. It does **not** change the current lowering contract.

The labels `llvm20`, `llvm22`, `llvm23`, and `llvm24` are rocKE names for
current emission lanes. They are not AMD LLVM release versions and must not be
read as an AMD LLVM compatibility taxonomy.

### Evidence boundary

A flavor branch proves that rocKE currently assumes a behavior can vary. A test
of that branch proves the expected rocKE emission or admission decision. Neither
proves that every AMD LLVM build assigned to the flavor has that behavior.
Source and test inspection can establish:

- which module fields, declarations, operand types, and operations rocKE varies;
- where Python and C++ make the corresponding decisions;
- which tests preserve those decisions and which tools compile complete output.

It cannot establish which exact compiler revisions exhibit a behavior, whether
two currently coupled behaviors always vary together, or whether an assumption
is still necessary. A focused probe supplies that compiler evidence.
Qualification tests then establish whether rocKE responds correctly.

The intended reader is an engineer who needs to answer four questions quickly:

1. What concrete decisions are currently selected by an LLVM flavor?
2. What rocKE source and test evidence supports each candidate behavior?
3. Which decisions can be replaced by a focused probe?
4. Which flavor references are not compiler capabilities and must be migrated
   differently?

## Contents

- [The engineering problem](#the-engineering-problem)
- [Executive summary](#executive-summary)
- [Scope and method](#scope-and-method)
- [Current resolution model](#current-resolution-model)
- [Capability taxonomy](#capability-taxonomy)
- [Flavor uses that are not compiler capabilities](#flavor-uses-that-are-not-compiler-capabilities)
- [Test and tool inventory](#test-and-tool-inventory)
- [Demonstration report](#demonstration-report)
- [Demonstration sequence](#demonstration-sequence)
- [Conclusion](#conclusion)

## The engineering problem

AMD LLVM is maintained in the public `ROCm/llvm-project` repository as a fork
that continuously incorporates LLVM development. It does not publish a series
of AMD LLVM releases whose semantics are defined by upstream-style major-version
labels. A compiler banner such as `23.0.0git` or `24.0.0git` therefore identifies
pre-release upstream-development lineage; it does not establish a stable AMD
compiler contract.

The source identity of a compiler build is instead a pinned commit on the
`amd-staging` branch, together with any release patch set applied to that
revision. The complete compiler banner can expose the source revision and an
optional `+PATCHED:<revision>` suffix. Those revisions are the appropriate
provenance for comparing local compiler builds, but they still do not prove a
particular behavior: an AMD patch, backport, or change in compiler context can
alter a capability independently of an upstream-development label or commit
ordering.

rocKE currently maps the reported LLVM major into labels such as `llvm23` and
uses those labels to select emitted IR, admit operations, choose compiler
targets, and organize test artifacts. That is the mismatch this project is
addressing:

```text
reported development label -> rocKE flavor lane -> assumed compiler behavior
```

The defensible model is:

```text
pinned compiler revision + release patches -> provenance
focused probe in the real compiler context  -> observed behavior
observed behavior                           -> future rocKE decision
```

Commit identity explains **which** compiler was measured. A focused probe is
the evidence for **what** it accepts or emits. The demonstration in this
folder records both, while intentionally using only probe outcomes as
capability results.

## Executive summary

`llvm_flavor` currently stands in for at least five distinct concerns:

1. **Compiler-lane classifier** — a bucket selected from a reported LLVM major.
2. **Emission behavior** — the LLVM IR data layout, declarations, call shapes,
   and operations rocKE emits.
3. **Target-support behavior** — whether a compiler accepts a processor name
   and whether an intrinsic lowers for that processor.
4. **Qualification evidence** — the correctness, validity, and resource tests
   run against a compiler build.
5. **Artifact partitioning** — the flavor component in golden files, cache keys,
   AOT labels, and differential-test matrices.

Only items 2 and 3 are direct compiler-behavior observations: focused probes
establish what the selected compiler emits and accepts. Compiler identity
records which source revision and patch state produced those observations; it
does not predict them. rocKE should adapt its lowering and expectations to the
observed capabilities, then run its qualification tests. A qualification failure
may identify a rocKE defect or reveal a compiler behavior not yet captured by a
focused probe. In the latter case, add the smallest probe that makes the
variability explicit and re-qualify the affected compiler builds.

The current code is already partway toward a capability model:

- `core/llvm_profile.py` separates compiler lanes from structural emission
  profiles.
- `tools/gen_arch_domain.py` performs direct compile-and-link probes and records
  explicit negative and no-data states per intrinsic and processor.
- `core/arch/domain.py` preserves the important distinction between a measured
  negative and no answer.
- `tools/check_ir_validity.py` verifies complete emitted modules with the
  compiler rather than trusting a version label.

The main remaining problem is that broad profile switches such as
`llvm23_target_ops`, and version-ranked checks such as `minimum_llvm_flavor`,
still bundle independent observations together.

The migration model is:

```text
Compiler revision       provenance for one measured build
Focused probe           a narrow observed compiler behavior
Probe-derived decision  rocKE lowering or expectation selected from that behavior
Qualification tests     validate the resulting rocKE behavior and expose gaps
New qualification gap   rocKE defect, or a new behavior that needs a focused probe
Artifact migration      later follow-on work once probe-derived decisions are stable
```

The first four rows are the feedback loop. Artifact partitioning matters, but it
is not evidence of compiler behavior and is not part of the proof-of-concept
decision path.

### Evidence map

The inventory starts from decisions already present in rocKE. The table below
summarizes what kind of evidence exists and whether that evidence directly asks
a compiler the relevant question.

| Candidate behavior | Existing rocKE evidence | Evidence type | Direct probe today? | In this demonstration? |
| --- | --- | --- | --- | --- |
| AMDGPU data layout | emission profiles, module preambles, layout tests | emitted structure and expected output | no | yes |
| Buffer-resource intrinsic ABI | declarations, call lowering, focused emission tests | emitted structure and compiler-validity assumption | no | yes |
| fp8/bf8 MFMA packed operand type | Python/C++ declarations and call branches | emitted structure | no | no |
| Named-barrier pointer address space | Python/C++ declarations, casts, and tests | emitted structure | no | no |
| Target operation availability | broad admission bit and operation tests | admission and emitted structure | partially | no |
| Processor-name support | arch-domain generation and target-selection tests | direct compiler observation | yes | existing mechanism |
| Intrinsic availability by processor | arch-domain artifacts and audit tools | direct compiler observation | yes | existing mechanism |
| Golden and cache partitioning | flavor-keyed hashes, paths, and identities | artifact identity, not compiler behavior | not applicable | no |

A row backed only by emitted structure is a question to ask the compiler, not a
claim about a particular compiler revision.

## Scope and method

The inventory covered:

- Python and native flavor resolution;
- LLVM emission profiles and every direct flavor branch in the lowerers;
- compiler-target and intrinsic-target support tables;
- library-side admission, packaging, and cache uses;
- tests and tools that detect, validate, or merely enumerate flavors;
- the Python/C++ byte-identity boundary.

Generated build trees, emitted golden payloads, examples, and performance
narratives were not treated as independent capability definitions. They were
reviewed as consumers of the current contracts and qualification evidence.

For each migration candidate, this inventory asks three separate questions:

1. What exact compiler behavior does the current flavor label stand in for?
2. What is the smallest focused probe that can observe that behavior?
3. Which existing qualification tests must pass after rocKE selects its behavior
   from that probe result?

A focused probe answers the first two questions. The third is the feedback loop:
a failure either identifies a rocKE defect or tells us the probe vocabulary is
missing a compiler-variable behavior.

The primary sources are:

| Concern | Authoritative source |
| --- | --- |
| Qualified lanes and Python emission profiles | `platform/python/rocke/core/llvm_profile.py` |
| Python lowering decisions | `platform/python/rocke/core/lower_llvm.py` |
| Native emission profiles | `platform/cpp/core/llvm_profile.cpp` |
| Native lowering decisions | `platform/cpp/core/lower_llvm/` |
| COMGR compiler identity | `platform/python/rocke/runtime/comgr.py` and `platform/cpp/core/lower_llvm/compiler_version.cpp` |
| Processor and intrinsic support measurements | `platform/tools/gen_arch_domain.py` |
| Measurement read side | `platform/python/rocke/core/arch/domain.py` |
| Compiler target selection | `platform/python/rocke/core/arch/target.py` |
| Full-module validity | `platform/tools/check_ir_validity.py` |
| Cross-engine contract | `platform/tools/check_byte_identity.py` |

## Current resolution model

### Compiler identity query

Automatic lowering asks the compiler loaded through COMGR for its LLVM version.
The Python and native engines use the same two-stage strategy:

1. call `LLVMGetVersion` when that symbol is visible;
2. otherwise preprocess a version probe through the same loaded COMGR.

The query records where the version evidence came from. It deliberately does
not infer the compiler from ROCm package metadata or an unrelated executable on
`PATH`.

An explicit `ROCKE_LLVM_FLAVOR` or lowering API argument bypasses automatic
selection for offline emission. Compilation still queries the compiler that
will consume the IR; an emission override does not rewrite that evidence.

### Major-to-lane ladder

The current automatic mapping is:

| Reported LLVM major | Selected lane |
| --- | --- |
| 20 or older | `llvm20` |
| 21–22 | `llvm22` |
| 23 | `llvm23` |
| 24 | `llvm24` |
| Newer than 24 | rejected as unqualified |

This ladder is a rocKE compiler-lane classifier, not a direct measurement and
not an AMD LLVM release map. In particular, it assumes all builds reporting one
upstream-development major share the profile assigned to that lane, despite AMD
LLVM builds being identified by pinned `amd-staging` revisions and possible
release patches.

### Current structural profiles

| Lane | Data-layout shape | Declaration ABI | Broad operation capability |
| --- | --- | --- | --- |
| `llvm20` | plain address-space-8 layout; no ELF mangling field | base declarations | none |
| `llvm22` | indexed address-space-8 layout; no ELF mangling field | updated buffer-resource and fp8/bf8 MFMA signatures | none |
| `llvm23` | indexed address-space-8 layout, ELF mangling, address spaces 10–15 | inherits `llvm22` declarations | `llvm23_target_ops` |
| `llvm24` | same data layout as `llvm23` | named-barrier pointers move to address space 15 | `llvm23_target_ops`, `named_barrier_as15` |

This table is useful history, but it should not become the new capability
schema or be read as a chronology of AMD LLVM releases. Each column combines
independent observations that can vary separately across compiler revisions and
release patches.

## Capability taxonomy

Capability IDs below describe facts rather than generations. They are candidate
names for focused probe results, not a frozen public API or a promise that the
proof-of-concept report must contain every item.

### 1. Module IR contract

#### `amdgpu.ir.data_layout`

**Current decision.** The emission profile chooses one complete AMDGPU target
data-layout string.

The currently significant fields are:

- whether address space 8 has an explicit index width;
- whether the ELF symbol-mangling component is present;
- whether address spaces 10–15 are declared.

**Consumers.** Module preamble emission, the COMGR mismatch guard, Python/native
profile consistency tests, and golden IR.

**Probe.** Ask the selected compiler to emit LLVM IR for a minimal AMDGPU module
under the real target triple and flags, then parse its `target datalayout`.
Preserve the complete string as the primary value and parsed fields as readable
diagnostics.

**Important boundary.** The existing COMGR guard recognizes only the address-
space-8 generation. It intentionally cannot distinguish all currently emitted
layouts. A future probe set must not mistake that guard for complete profile
identification.

### 2. Intrinsic declaration and call ABI

#### `amdgpu.intrinsic.make_buffer_resource.abi`

**Current decision.** rocKE selects both the intrinsic spelling and the byte-
count operand width:

| Current profile | Intrinsic spelling | Count type |
| --- | --- | --- |
| base | `llvm.amdgcn.make.buffer.rsrc.p1` | `i32` |
| indexed address-space-8 | `llvm.amdgcn.make.buffer.rsrc.p8.p1` | `i64` |

The lowering also changes operand preparation: an `i32` count is extended on
the `i64` path, while an `i64` count is refused on the base path.

**Consumers.** Declaration table, buffer-resource lowering, async buffer
operations, and tests in `platform/tests/test_rocke.py`.

**Probe.** Compile and link minimal calls for the candidate signatures. A
successful declaration parse alone is too weak because an unknown `llvm.*`
name can survive as an external symbol. Preserve both the submitted candidate
and the compiler's canonical declaration: a newer compiler can accept a legacy
spelling while resolving it to a distinct canonical ABI.

**Selection is a separate decision.** A compiler can accept more than one
canonical ABI. That observation does not itself select rocKE emission. For this
family, a future rocKE resolver should explicitly prefer the canonical
`p8_p1_i64` ABI when it is accepted, preserving wide byte counts and rocKE's
existing modern emission path; it should use legacy `p1_i32` only when the
preferred ABI is unavailable. This is rocKE policy to qualify, not a compiler
capability to infer from a version label.

**Recommended first vertical slice.** This is a strong demonstration candidate:
it is contained, has visibly different emitted IR, already has focused tests,
and exercises both declaration and call-site selection.

#### `amdgpu.intrinsic.mfma_fp8_bf8.packed_operand_type`

**Current decision.** Eight packed fp8/bf8 values use `<2 x i32>` in the base
profile and `i64` in the indexed address-space-8 profile. The decision appears
in:

- four hand-written homogeneous fp8/bf8 MFMA declarations and handlers;
- four mixed fp8/bf8 matrix-call declarations represented by `FlavorType`;
- both Python and native engines.

**Probe.** Compile and link representative homogeneous and mixed MFMA calls for
the relevant processor. The result value should be the exact accepted LLVM
register type, not `modern: true`.

**Naming note.** `modern` is not a capability name. It loses meaning as soon as
a later ABI exists and currently couples this fact to the address-space-8 data
layout.

#### `amdgpu.intrinsic.named_barrier.pointer_address_space`

**Current decision.** Named-barrier declarations and calls use LDS address
space 3 in earlier profiles and address space 15 in the newest profile. rocKE
inserts an `addrspacecast` on the address-space-15 path.

The affected operations are:

- `s.barrier.init`;
- `s.barrier.signal.var`;
- `s.barrier.join`;
- `s.wakeup.barrier`.

**Probe.** Compile and link representative calls for all four intrinsics using
each candidate pointer address space. Record the accepted integer address space.
One successful intrinsic should not be assumed to prove the other three.

### 3. Operation availability

The current `llvm23_target_ops` profile bit is an umbrella. It admits several
independent operation families after also checking target facts:

| Candidate capability family | Current operations |
| --- | --- |
| `amdgpu.operation.workgroup_cluster.available` | cluster IDs, workgroup IDs, cluster barrier, cluster dimensions |
| `amdgpu.operation.scalar_data_prefetch.available` | scalar and buffer data prefetch |
| `amdgpu.operation.global_prefetch.available` | global and flat prefetch |
| `amdgpu.operation.tensor_data_movement.available` | tensor load/store through LDS and tensor wait |
| `amdgpu.operation.extended_scalar_barrier.available` | signal, wait, init, signal-var, join, wakeup, leave |
| `amdgpu.operation.async_store_from_lds.available` | global async store from LDS |
| `amdgpu.operation.global_transpose_load.available` | 128-bit transpose global load |
| `amdgpu.operation.cluster_multicast_load.available` | cluster load and cluster async load to LDS |
| `amdgpu.operation.scaled_wmma.available` | scaled WMMA forms currently gated by the broad profile |
| `amdgpu.inline_asm.scalar_control.accepted` | gfx1250 scalar-control inline assembly currently guarded by `minimum_llvm_flavor` |

These names are intentionally operation-oriented. The AMD execution model is
kept separate from compiler support: a target fact can say the processor has a
facility while a compiler capability says the selected toolchain can express
and lower it.

**Probe.** Reuse the proven arch-domain method:

1. establish whether the exact intrinsic spelling exists;
2. compile and link a call for the processor;
3. where needed, inspect or assemble the generated instructions so a backend
   cannot silently emit an encoding unavailable to that processor;
4. retain `unknown` when the processor cannot be targeted or the tool fails.

**Do not reproduce the umbrella in the manifest.** If one toolchain backports
only one family, `llvm23_target_ops: false` loses useful support while
`llvm23_target_ops: true` overclaims the others.

### 4. Processor-name support

#### `amdgpu.compiler.accepts_processor_name`

**Current decision.** Each committed arch-domain column records whether its
compiler accepts every known processor name. Compile helpers use this answer to:

- pass the concrete processor when accepted;
- fall back to its catalogued generic processor when that is accepted;
- reject when neither can be used;
- leave an unmeasured processor to the compiler rather than inventing a
  negative.

**Scope.** This is one boolean observation per processor, under the AMDGPU
triple and effective compiler flags.

**Probe.** Compile and link an empty AMDGPU kernel for the exact processor name.
This is already implemented as target preflight in `gen_arch_domain.py`.

### 5. Intrinsic availability by processor

#### `amdgpu.intrinsic.<stable-name>.available`

The committed intrinsic arch-domain artifacts are already a compiler capability
database. For every declaration key and processor, they distinguish:

- `ok` — measured available;
- `name_absent` — the exact intrinsic spelling is not recognized;
- `arch_absent` — the name exists but cannot lower for the processor;
- `target_unsupported` — the compiler did not accept the processor, so the
  intrinsic question was not answered;
- tool crash, timeout, or probe error — no capability answer.

Any future expansion of the probe results should preserve these semantics. It
may normalize the result to `known: true`, `known: false`, `unknown`, or
`error`, but it must retain the more specific diagnostic code and evidence.

**Migration caution.** The current runtime gate treats only `arch_absent` as an
enforceable negative. `name_absent` is handled by the full-module validity lane.
Changing that policy is separate from renaming the data.

### 6. Compiler behavior and quirks

No production flavor branch currently names a focused semantic miscompilation
probe. If one is added, use a positive capability or an explicit quirk name:

```text
amdgpu.behavior.<required_semantics>
amdgpu.quirk.<specific_symptom>_present
```

A quirk result must come from a focused reproducer. Compiler identity, release
metadata, or a commit hash is not a quirk probe.

### 7. Toolchain identity and provenance

The following data belongs in the manifest but is **not** a capability:

- compiler interface used (`comgr` or `clang`);
- requested and resolved tools/libraries;
- full version text;
- target triple, processor, and flags;
- generator version;
- a fingerprint of every cache-relevant input;
- commands, diagnostics, and observation times.

Keeping identity outside `capabilities` prevents consumers from silently
reintroducing logic such as `if compiler major >= 23`.

## Flavor uses that are not compiler capabilities

### Golden and differential matrices

Many tests iterate every `LLVM_FLAVORS` entry and store one hash per flavor.
They prove deterministic Python/native emission and preserve the current
expected output matrix. They do not show that the host compiler accepts or
correctly compiles that output.

Near-term direction: retain these artifacts while a focused probe replaces one
emission decision at a time. Once the probe-derived decision is qualified,
inspect whether a flavor-keyed artifact needs a more precise key. Selecting that
future key is follow-on work, not a result this inventory can assume.

### Cache keys, labels, and AOT identities

Convolution identities, benchmark caches, and packaged descriptors include the
flavor because it can change emitted bytes and generated code. Removing that
field before a probe-derived replacement is designed and qualified could cause
different binaries to collide.

Near-term direction: do not change these identities in the proof of concept.
When a probe-derived lowering decision is integrated, audit each affected cache
and artifact key together with the qualification tests that cover it.

### Packaging and qualification gates

The packaged convolution catalog refuses a lower-ranked flavor than the one
under which it was established. Attention resource gates similarly require a
specific flavor and compiler cohort.

These are qualification tests and evidence, not feature probes. Passing the
focused structural probes does not automatically reproduce numeric or resource
results from another compiler build. Conversely, a failure in one of these
gates is valuable feedback: determine whether rocKE is wrong or whether the
failure exposes a compiler behavior that the probe set does not yet describe.

Near-term direction: let focused probes drive the relevant emission decisions,
then rerun the existing qualification gates. Add a probe only when the failure
reveals a real compiler-variable behavior; otherwise fix rocKE or the test.

### Kernel admission lists

The Winograd path currently lists the flavors under which a kernel lowers, and
one inline-assembly helper carries `minimum_llvm_flavor`. These are indirect
proxies for operation support.

Migration direction: declare the exact required operation capability. Admission
then reports the missing capability rather than a minimum release label.

### Historical documentation and examples

Flavor names in examples and historical results identify how an artifact was
produced. They should not be mechanically rewritten as capabilities. Historical
records remain provenance; new executable decisions should use measured facts.

## Test and tool inventory

### Detection and selection tests

| Area | What it proves |
| --- | --- |
| `platform/tests/test_compiler_detection.py` | loaded-COMGR provenance, override precedence, lane boundaries, future-major refusal, and unknown evidence |
| `platform/tests/test_compiler_detection_native.py` | Python/native detector agreement and fallback behavior against loadable fixtures |
| `platform/tests/test_binding_compiler_selection.py` | Python-driven native bindings use Python's selected compiler evidence |
| `platform/tests/core/test_compile_target_selection.py` | compiler target selection uses the compiler that actually runs, not the emission override |
| `platform/tests/core/test_arch_target_identity.py` | processor acceptance, generic fallback, and unmeasured-target behavior |

As focused probes replace individual decisions, these tests should use
controlled probe-result fixtures while retaining provenance and unknown-state
coverage. They should not infer behavior from a reported major.

### Structural emission tests

| Area | What it proves |
| --- | --- |
| `platform/tests/test_rocke.py` | data-layout partitions, buffer-resource ABI, fp8/bf8 MFMA packing, mismatch guard, and flavor enumeration |
| `platform/tests/core/flavor_datalayout_ssot.cpp` | native profile registry and data-layout consistency |
| `platform/tests/core/test_gfx1250_prefetch.py` and related target-op tests | current broad target-op admission and focused emission |
| `platform/tests/tools/test_arch_domain_artifact.py` | measured-column completeness, provenance, processor support, and no-data semantics |

These are the strongest candidates for focused probe-result and
probe-derived-emission tests.

### Whole-module and cross-engine tests

| Tool or suite | Role |
| --- | --- |
| `tools/check_ir_validity.py` | validates emitted modules by compiling and linking with the selected toolchain |
| `tools/gen_arch_domain.py` | directly measures intrinsic and processor support |
| `tools/check_arch_domain.py` | audits corpus demands against measured target support |
| `tools/check_byte_identity.py` | proves Python/native byte identity; it does not prove compiler validity |
| differential golden suites | protect each qualified emission contract offline |

Focused probes configure or explain a compiler-dependent decision; the
qualification gates test the resulting rocKE behavior. Neither replaces byte
identity, full-module validity, or numeric verification.

## Demonstration report

### Why these two behaviors were selected

The report schema does not attempt to encode the complete inventory. It selects
one module-level contract and one intrinsic-call contract for the first vertical
slice:

1. `amdgpu.ir.data_layout` has a visible flavor-dependent module representation;
2. `amdgpu.intrinsic.make_buffer_resource.abis` has visibly different
   declarations, call shapes, and operand preparation.

Each decision is already represented in rocKE source and focused tests, can be
measured with a small standalone compiler invocation, and does not require the
initial demonstration to solve target-operation or numeric qualification. The
schema is therefore a proposed vocabulary for this slice, not a complete AMD
LLVM compatibility model.

The proof-of-concept artifacts are:

- [`probe.py`](probe.py) — standalone probe runner;
- [`capabilities-v1.schema.json`](capabilities-v1.schema.json) — strict schema
  for its report;
- [`example.capabilities.json`](example.capabilities.json) — an illustrative
  report.

`capabilities.json` is deliberately small. It exists to show that the focused
probes produce understandable, repeatable answers across local compiler builds.
It is not a production configuration contract and rocKE lowering does not read
it.

The report records:

- complete compiler identity text;
- the primary source revision and optional patched revision reported by the
  compiler;
- exact compiler and `opt` executable digests;
- the target triple, processor, probe version, and cache fingerprint;
- the observed data layout;
- which current rocKE buffer-resource ABI candidates the compiler accepts and
  how `opt` canonicalizes them.

The reported LLVM major remains only part of the unparsed identity text. Source
revisions distinguish compiler source states; executable digests distinguish
the exact local tools; neither substitutes for a behavior probe. A future
measurement may show that a legacy candidate resolves to a distinct canonical
`p8` ABI, or that data-layout details vary independently. The report preserves
those observations as separate facts rather than reconstructing a
version-defined profile.

A capability is either `known`, with its observed value, or `error`, with a
concise diagnostic. A rejected ABI candidate is still a known answer and is
recorded inside the capability value. A timeout or malformed probe output is an
error and is not cached.

Cache entries are keyed by the probe version, target, processor, selected tool
paths, complete tool identities, and executable digests. The report itself
contains no timestamp or cache-hit marker, so a fresh
run and a cache hit produce byte-identical JSON. Cache behavior and elapsed time
are printed by the CLI rather than mixed into the evidence.

## Demonstration sequence

1. Select a compiler with `--llvm-bin`, then `ROCKE_LLVM_BIN`, then `PATH`.
2. Hash `clang` and `opt`, read the full compiler banner, and retain its source
   revisions as provenance only.
3. Ask `clang` to emit a minimal AMDGPU module and record its exact data layout.
4. Ask `opt` to resolve each buffer-resource declaration and call shape rocKE
   currently emits; record accepted, rejected, and canonical forms.
5. Write `capabilities.json`, print a short status summary, and cache only a
   complete report.
6. Repeat against the local compiler set and compare behavior by source
   revision rather than by the reported development-version label.
7. For the first integrated decision, run the relevant byte-identity,
   emitted-IR-validity, and numeric qualification gates. Use a failure to decide
   whether to fix rocKE or add one new focused probe.

Integration with lowering, COMGR probing, and the full intrinsic arch-domain
is intentionally outside this demonstration. The next integration should be one
small emission decision followed by its existing qualification tests, not a
permanent configuration API designed in advance.

## Conclusion

The current flavor model is not one version check; it is a compact label for
several distinct concerns. The migration should not replace that label with one
large `capabilities` boolean or a renamed profile number. It should expose the
small observed facts rocKE actually consumes, retain explicit unknown states,
and use qualification failures to test whether those facts are sufficient.

The existing arch-domain generator demonstrates that direct, evidence-bearing
probes are practical in this repository. The safest next implementation is to
reuse that approach for one structural ABI decision, compare it with current
flavor-derived behavior, and run the affected qualification tests. Expand the
probe vocabulary only when those results expose a real compiler-variable
behavior that the existing probes do not describe.
