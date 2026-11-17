# LLVM flavor inventory and compiler capability taxonomy

## Purpose

This document traces rocKE's LLVM flavor-dependent source and tests to the
concrete compiler behaviors those labels currently represent. It is source-code
research for replacing broad compiler-label inference with focused behavior
probes. It does **not** change the current lowering contract.

The labels `llvm20`, `llvm22`, `llvm23`, and `llvm24` are rocKE names for
current emission lanes. They are not AMD LLVM release versions and must not be
read as an AMD LLVM compatibility taxonomy.

The intended reader is an engineer or stakeholder who needs to answer four
questions quickly:

1. What can vary between AMD LLVM compiler builds?
2. What happens to rocKE when that variation is handled incorrectly?
3. What evidence tells us how a selected compiler actually behaves?
4. What should rocKE do with that evidence?

## Contents

- [Why this matters](#why-this-matters)
- [The kernel compatibility journey](#the-kernel-compatibility-journey)
- [Three meanings of supported](#three-meanings-of-supported)
- [The eight compatibility variations](#the-eight-compatibility-variations)
- [The engineering problem](#the-engineering-problem)
- [Executive summary](#executive-summary)
- [Scope and method](#scope-and-method)
- [Current resolution model](#current-resolution-model)
- [Capability taxonomy](#capability-taxonomy)
- [Exhaustive probe backlog](#exhaustive-probe-backlog)
- [Completeness and future-growth rules](#completeness-and-future-growth-rules)
- [Flavor uses that are not compiler capabilities](#flavor-uses-that-are-not-compiler-capabilities)
- [Test and tool inventory](#test-and-tool-inventory)
- [Demonstration report](#demonstration-report)
- [Demonstration sequence](#demonstration-sequence)
- [Conclusion](#conclusion)

## Why this matters

rocKE does more than generate syntactically valid LLVM IR. It must generate a
module that the selected compiler understands, compile that module for the
requested GPU, obtain the instructions and resource behavior the kernel design
requires, and then execute with correct results.

Compatibility can fail at any of those steps. The visible symptom depends on
where it fails:

- the module can be rejected before code generation;
- an intrinsic can verify but remain an unresolved external symbol;
- a backend option can terminate the compiler process;
- an accepted attribute can be ignored;
- code generation can select a different instruction or spill registers; or
- the kernel can compile and launch but compute the wrong answer.

A single label such as `llvm23` cannot explain which of those contracts holds.
The practical goal of this work is:

> **Probe what the compiler accepts, inspect what it produces, and qualify what
> the GPU executes.**

That separation reduces risk in two ways. First, rocKE can decline an unsupported
operation or choose the right emission before a large kernel reaches the
compiler. Second, when qualification fails, developers know which layer to
investigate instead of treating every failure as an undifferentiated “LLVM
version problem.”

### Stakeholder view

| Variation | Question rocKE needs answered | Impact when the answer is wrong | Evidence needed |
| --- | --- | --- | --- |
| Module format | What kind of LLVM module does this compiler require? | Every kernel can fail before code generation | compiler-emitted module plus candidate-module codegen |
| Intrinsic ABI | How must a compiler operation be named and typed? | calls are rejected, truncated, or interpreted incorrectly | parser canonicalization plus compile/link |
| Target and operation availability | Does this compiler support this processor and operation together? | clean refusal, compiler crash, or undefined symbol | processor preflight plus per-operation compile/link/assembly |
| Compiler options | Does the compiler recognize each backend switch rocKE passes? | wrong kernel, rejected compile, or host-process termination | isolated option probe and effect check |
| Function attributes and metadata | Does the compiler accept and honor rocKE's scheduling, occupancy, and cluster requests? | compilation succeeds but produces a materially different kernel | parse/codegen plus code-object inspection |
| Pipeline behavior | At what stage does LLVM recognize, upgrade, reject, or tolerate the input? | a weak probe reports success and a later stage fails | stage-by-stage parser, verifier, codegen, link, and assembly results |
| Generated-code behavior | What instructions, resources, and metadata were actually produced? | wrong instruction, spills, lost occupancy, or unloadable code object | disassembly, resource report, and code-object inspection |
| Runtime semantics | Does the compiled kernel compute and update state correctly? | everything builds and launches, but results are wrong | independent on-device numeric/state oracle |

The first six variations can often be isolated with small host-side probes. The
seventh needs target code generation and artifact inspection. The eighth needs a
GPU and an independent reference. None of these evidence classes substitutes
for another.

## The kernel compatibility journey

The eight variations follow the path a kernel takes rather than an arbitrary
compiler taxonomy:

```text
rocKE KernelDef and lowering policy
        │
        ├── 1. module format
        ├── 2. intrinsic ABI
        ├── 3. target and operation availability
        ├── 4. compiler options
        └── 5. function attributes and metadata
        ▼
LLVM parser → verifier → optimizer → target codegen → link → assembly
        │                 6. pipeline behavior
        ▼
HSACO / code object
        │
        └── 7. instructions, resources, and code-object metadata
        ▼
GPU execution
        │
        └── 8. numeric and state semantics
```

This model is also a triage guide:

- **Compiler rejects or crashes:** start with variations 1–6 and identify the
  first stage without a reliable answer.
- **Compiler succeeds but the artifact is unexpected:** inspect variation 7;
  do not infer “honored” from “accepted.”
- **Artifact looks plausible but results are wrong:** variation 8 is the ruling
  evidence, while the earlier observations explain what changed.

## Three meanings of supported

The word “supported” is too ambiguous for compiler compatibility. This document
uses three progressively stronger claims:

| Claim | Meaning | Typical evidence |
| --- | --- | --- |
| **Accepted** | The compiler parses or compiles the construct without a probe failure | parser, verifier, or compile result |
| **Honored** | Generated code reflects the requested operation, option, attribute, or metadata | canonical IR, disassembly, resource report, or code-object note |
| **Qualified** | A complete rocKE kernel passes structural, resource, and correctness gates | byte identity, full-module validity, artifact checks, and numeric/state verification |

For example, accepting `amdgpu-waves-per-eu` does not prove that the resulting
resource allocation meets the kernel's occupancy plan. Accepting an intrinsic
declaration does not prove that it links for a processor. Compiling and linking
a kernel does not prove that its result is numerically correct.

Probe reports should also label the strength of each fact:

- **Observed** — a focused probe directly obtained the answer;
- **Derived** — rocKE selected a policy from one or more observations;
- **Qualified** — a complete-kernel gate validated the selected policy;
- **Assumed** — current source relies on the behavior without direct evidence;
- **Unknown** — no reliable answer was obtained.

## The eight compatibility variations

Each variation below answers the same practical questions: what varies, what
rocKE example demonstrates it, what failure developers see, how to measure it,
and what decision the measurement enables.

### 1. Module format

**Question.** What target triple, data layout, calling convention, and module
flags does this compiler require for AMDGPU code generation?

**Concrete rocKE example.** rocKE carries different address-space-8 layouts,
the `m:e` ELF mangling field, and the address-space-10-through-15 block. Compiler
builds reporting the same development major have emitted different complete
layouts. Source history also shows that one compiler overwrote a supplied stale
layout while a later compiler validated and rejected it.

**Failure developers see.** Code generation rejects every kernel with an
incompatible-data-layout error. A permissive source-input path can hide the
problem until bitcode or a stricter backend is used.

**Why a version check is insufficient.** Canonical layout and supplied-layout
policy can change independently, including between builds carrying the same
reported major.

**Probe and decision.** Ask the compiler to emit a minimal AMDGPU module, retain
the complete preamble, and separately compile candidate supplied layouts. rocKE
can then emit the observed module contract and reject an unknown policy.

**Does not prove.** A matching preamble does not prove any intrinsic, target,
resource, or numeric behavior.

### 2. Intrinsic ABI

**Question.** What name, overload suffix, return type, argument types, and call
shape does this compiler require for an LLVM intrinsic?

**Concrete rocKE example.** `make.buffer.rsrc` varies between a legacy spelling
with an `i32` count and an indexed spelling with an `i64` count. Eight fp8/bf8
MFMA families vary between `<2 x i32>` and `i64` packed operands. Named barriers
move their pointers from address space 3 to address space 15. Source history also
records 11- versus 9-argument scaled-MFMA forms.

**Failure developers see.** LLVM rejects a declaration or call, a wide count
cannot be represented, or a seemingly accepted legacy form is canonicalized to
a different ABI.

**Why a version check is insufficient.** Each intrinsic family can be
backported or changed independently, and a compiler can accept more than one
form without telling rocKE which one to prefer.

**Probe and decision.** Submit every candidate declaration and call, record the
canonical declaration separately from the submitted form, then compile and link
it. rocKE applies an explicit preference policy among accepted forms.

**Does not prove.** ABI acceptance does not prove processor availability,
selected ISA, or correct execution.

### 3. Target and operation availability

**Question.** Does this compiler recognize the requested processor, recognize
the operation's intrinsic, and lower that intrinsic for that processor?

**Concrete rocKE example.** LLVM had no `ds.read.tr16.b128` intrinsic, although
a verifier accepted it as an external function; only a later link exposed the
undefined symbol. Other transpose operations exist but lower only for specific
processors. Older compilers can reject a concrete processor while accepting a
catalogued generic target.

**Failure developers see.** Unknown processor, undefined symbol, “Cannot
select,” fatal backend error, invalid selected instruction, or a crash.

**Why a version check is insufficient.** Processor-name support, intrinsic-name
support, and processor-specific lowering are separate axes. A compiler may know
one without the others.

**Probe and decision.** Preflight every exact and generic processor name. For
every resolved declaration key, preserve parser recognition, compile/link, and
text-assembly outcomes per processor. rocKE can select a safe compiler target,
admit the operation, or decline with the exact missing capability.

**Does not prove.** Successful lowering does not prove an option or attribute
was honored, nor that execution is correct.

### 4. Compiler options

**Question.** Does the selected compiler recognize each backend option rocKE
passes, and does the option have the requested effect?

**Concrete rocKE example.** The production path uses
`-amdgpu-mfma-vgpr-form`; an in-tree example uses
`-amdgpu-early-inline-all=true`. An older compiler can terminate the process on
an unknown `-mllvm` option instead of returning a normal COMGR error.

**Failure developers see.** Pytest or the serving process exits, compilation is
refused, or omitting the option silently compiles a different kernel than the
resource plan requested.

**Why a version check is insufficient.** Options can be added, removed,
backported, or have their defaults changed independently of other compiler
features.

**Probe and decision.** Run each constructible option in an isolated child
process using the same compiler interface as the real command. Then inspect the
representative generated code to distinguish “accepted” from “honored.” rocKE
must refuse an unknown required option rather than guess.

**Does not prove.** Option acceptance on one target does not prove its effect on
all relevant operations or targets.

### 5. Function attributes and metadata

**Question.** Does LLVM accept and honor the kernel attributes and metadata that
carry rocKE's launch, scheduling, occupancy, aliasing, and memory policies?

**Concrete rocKE example.** rocKE emits `amdgpu-flat-work-group-size`,
`amdgpu-sched-strategy`, `amdgpu-waves-per-eu`, `amdgpu-agpr-alloc`, and
`amdgpu-cluster-dims`. Cluster dimensions must reach the code object;
`waves-per-eu` and AGPR allocation must affect resources rather than merely
survive parsing.

**Failure developers see.** The module compiles, but the launch contract,
cluster shape, register form, scheduler policy, occupancy, or memory behavior
differs from the kernel design.

**Why a version check is insufficient.** Attribute syntax and semantic effect
can change separately, and compilers may accept an attribute they ignore for a
particular target.

**Probe and decision.** Probe each emitted value class, then inspect assembly,
resource reports, and code-object notes for the promised effect. rocKE can emit
the attribute only when its required semantics are observed and qualified.

**Does not prove.** Artifact metadata does not establish numeric correctness.

### 6. Toolchain pipeline behavior

**Question.** Which compiler stage recognizes, upgrades, rejects, ignores, or
silently transforms the input?

**Concrete rocKE example.** Unknown `llvm.*` names can pass verification as
external functions and fail only at link. Legacy intrinsics can canonicalize
during parsing. Declaration verification can happen before auto-upgrade. An
optimizer can delete a probe call, creating a false availability result. A
backend can overwrite or validate the module data layout.

**Failure developers see.** A focused probe says “supported,” but the complete
kernel fails later—or the probe passes only because the operation under test was
removed.

**Why a version check is insufficient.** This behavior depends on pipeline
stage, flags, optimization level, and compiler entry point, not just source
lineage.

**Probe and decision.** Preserve separate parser, verifier, optimizer, codegen,
link, assembler, and code-object results. Keep the operation observable and run
fatal stages in subprocesses. Consumers can require the exact evidence stage
their decision needs.

**Does not prove.** One stage's success must never be promoted to whole-pipeline
or runtime qualification.

### 7. Generated-code behavior

**Question.** What instructions, register classes, resource counts, spills,
target metadata, and occupancy limits did the compiler actually produce?

**Concrete rocKE example.** A compiler can accept an MFMA request but choose a
different scaled/unscaled or VGPR/AGPR form. `amdgpu-waves-per-eu` can alter the
register budget. Resource gates depend on VGPR, AGPR, LDS, scratch, spill, and
code-object target fields.

**Failure developers see.** The kernel spills, loses expected occupancy, uses a
wrong or unavailable instruction, crashes during codegen, or produces a code
object that cannot be loaded for the requested target.

**Why a version check is insufficient.** Instruction selection, optimization,
and register allocation depend on the compiler revision, processor, wave size,
optimization level, backend options, function attributes, and complete IR shape.
Two compiler candidates can accept byte-identical IR yet allocate different
register classes, spill differently, schedule instructions differently, or
optimize the same expression into different instruction sequences. Small source
changes can also move a workload across a resource boundary.

**Probe and decision.** Use two evidence levels. Focused microprobes establish
stable control contracts—for example, whether zero AGPR allocation plus the
VGPR-form option actually selects VGPR-form MFMA, or whether an optimization
preserves a required operation. Complete-kernel qualification records the actual
VGPR/AGPR/SGPR counts, allocation granularity, scratch and spill state,
occupancy-relevant metadata, selected instructions, and compiler failure mode.
Disassemble representative kernels and inspect compiler resource reports and
code-object metadata.

The routing depends on what is being observed:

- variation 4 or 5 records whether the controlling option or attribute is
  **accepted**;
- variation 6 records which optimization or compiler stage transformed the IR;
- variation 7 records whether the control was **honored** in ISA and resources;
- variation 8 rules when the transformation changes correctness.

Per-kernel resource counts and optimization outcomes generally remain
qualification evidence keyed by compiler and kernel identity; they are not
promoted into global capability booleans.

**Does not prove.** Plausible ISA and resources do not prove the computation is
correct, and one kernel's register allocation does not predict another's.

### 8. Runtime semantics

**Question.** Does the compiled kernel produce the correct output and correctly
update every observable state value?

**Concrete rocKE example.** A historical scaled-MFMA form could compile while
scale operands were effectively lost. Other compiler defects can preserve
module validity, instruction presence, and successful launch while changing the
answer.

**Failure developers see.** The kernel builds, links, loads, and launches, but
returns incorrect values or mutates state incorrectly.

**Why a version check is insufficient.** Semantic correctness is specific to
the operation, compiler context, target, and inputs. No compiler identity or
structural probe proves it.

**Probe and decision.** Run the smallest focused reproducer and the relevant
complete-kernel numeric gate against an independent reference at declared dtype
and tolerance. A behavior is qualified only after this layer passes.

**Does not prove.** One workload shape does not establish general correctness;
qualification must state its covered inputs and axes.

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

`llvm_flavor` currently compresses the eight compatibility variations above:
module format, intrinsic ABI, target/operation availability, backend options,
function attributes, pipeline behavior, generated-code behavior, and runtime
semantics. These are independent observations even where the current source
selects several of them with one profile bit.

Three cross-cutting concerns are related but are not themselves compiler
capabilities:

1. **Compiler-lane classification and provenance** identify which compiler was
   measured; they do not prove how it behaves.
2. **Qualification evidence** records which complete kernels, targets, resource
   gates, and numeric cases passed after rocKE selected a policy.
3. **Artifact partitioning** keeps golden files, caches, AOT labels, and
   differential matrices from colliding while different policies emit different
   bytes or generated code.

Structural facts can usually be measured by tiny probes. Instruction selection
and resource behavior require target-specific codegen. Semantic correctness
requires an on-device oracle. Compiler identity records which source revision
and patch state produced those observations; it does not predict them.

The up-front design requirement is therefore broader than the first two
implemented probes: enumerate every known behavior axis now, give every axis an
independent machine ID and observation method, and stage implementation without
collapsing unimplemented facts into one generation flag. Qualification failures
remain a feedback mechanism for discovering genuinely new axes, not the primary
way to inventory already-known variation.

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
| Complete module preamble/data layout | emission profiles, module preambles, layout tests | emitted structure and expected output | partial | yes |
| Data-layout enforcement vs overwrite | source history and complete-module validity failures | compiler-pipeline behavior | yes | yes |
| Buffer-resource intrinsic ABI | declarations, call lowering, focused emission tests | emitted structure and compiler-validity assumption | yes | yes |
| fp8/bf8 MFMA packed operand type | 8 homogeneous/mixed declarations and call branches | emitted structure | yes | yes |
| Named-barrier pointer address space | 4 declarations, casts, and tests | emitted structure | yes | yes |
| Every declaration's existence and canonical spelling | 247 resolved declaration keys in the current source snapshot | direct parser observation | yes, arch-domain stage A | existing mechanism |
| Every declaration's target lowerability | arch-domain artifacts and audit tools | compile/link/assembly observation | yes, arch-domain stage B | existing mechanism |
| LLVM-23 umbrella operations | broad admission bit and operation tests | admission and emitted structure | partially | no |
| Processor-name and generic-target support | arch-domain generation and target-selection tests | direct compiler observation | yes | selected context plus existing mechanism |
| Frontend builtin source ABI | HIP/C++ builtin call sites and compiler migration history | direct Clang prototype observation | yes, representative family | yes |
| Optimization-level codegen | emitted-IR validity gates and reduced reproducers | target codegen outcome | yes, representative reproducer | yes |
| Backend option acceptance | `-mllvm -amdgpu-mfma-vgpr-form` compile path | process-safety and option-parser behavior | yes, selected controls | yes |
| Kernel attribute acceptance | emitted function attributes and focused tests | emitted structure | yes, selected controls | yes |
| Kernel attribute semantic effect | resource plans and code-object/resource gates | backend-result behavior | only through qualification | no |
| Instruction selection and code-object metadata | ISA/resource tests and target probes | backend-result behavior | partial | no |
| Compiler crash/refusal behavior | subprocess probes and workload refusal tables | backend-result behavior | partial, workload-specific | no |
| Numeric semantics | GPU verification suites | on-device behavior | qualification only | no |
| Golden and cache partitioning | flavor-keyed hashes, paths, and identities | artifact identity, not compiler behavior | not applicable | no |

A row backed only by emitted structure is a question to ask the compiler, not a
claim about a particular compiler revision.

## Scope and method

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

### Research coverage

The inventory covered:

- Python and native flavor resolution;
- LLVM emission profiles and every direct flavor branch in both lowerers;
- the resolved declaration tables, including table-derived matrix calls;
- every call site of the broad `llvm23_target_ops` gate and every ranked or
  exact flavor admission check;
- compiler-target and intrinsic-target support tables;
- every LLVM backend option and every AMDGPU function attribute rocKE emits;
- library-side admission, resource refusal, packaging, and cache uses;
- tests and tools that detect, validate, or merely enumerate flavors;
- source history for compiler-compatibility fixes where the current source has
  normalized away the old branch; and
- the Python/C++ byte-identity boundary.

At this source snapshot, `_decls_for_flavor` resolves 247 keys for each of the
four lanes. Comparing adjacent lanes finds exactly nine declaration-text
changes from `llvm20` to `llvm22`, none from `llvm22` to `llvm23`, and four from
`llvm23` to `llvm24`. That census is a reproducible source observation, not a
schema limit: the completeness rule below derives the probe set from the table,
so adding a key cannot silently leave the research inventory stale.

Generated build trees, emitted golden payloads, examples, and performance
narratives were not treated as independent capability definitions. They were
reviewed as consumers of current contracts and qualification evidence.
Workload-specific spill/crash/refusal lists are evidence that backend-result
behavior varies, but each workload hash is not promoted into a global compiler
feature. The capability is the observable result class and probe protocol;
qualification retains the workload-specific evidence.

For each migration candidate, this inventory asks five separate questions:

1. What exact compiler behavior does the current flavor label or workaround
   stand in for?
2. Is the answer global, processor-specific, intrinsic-specific,
   option/attribute-specific, or workload-specific?
3. What is the smallest probe that obtains a reliable answer, including the
   parser, verifier, link, assembly, code-object, or on-device stage required?
4. Which independent facts must remain separate even when all sampled compilers
   currently change them together?
5. Which existing qualification tests must pass after rocKE selects its behavior
   from the probe result?

The source inventory answers the first, second, and fourth questions up front.
Focused probes answer the third. The fifth is the feedback loop: a failure either
identifies a rocKE defect or tells us the probe vocabulary is missing a genuinely
new compiler-variable behavior.

### Regression archaeology and probe gaps

Project issue and fix history was scanned for compiler-version-related build
breakages, validation failures, resource regressions, and generated-code
regressions in rocKE and adjacent compiler-facing CK code. Confirmed compiler
boundaries and strongly suspected compiler-dependent failures were included;
release comparisons without compiler isolation and harness-only defects were
excluded. The scan deliberately treats each report as evidence of a behavior
class, not as permission to encode a ticket, workload, or compiler hash as a
global capability.

| Observed regression class | Earliest reliable evidence stage | Current structural probe | Gap exposed by the history |
| --- | --- | --- | --- |
| An explicitly supplied data layout was once overwritten, then became a hard codegen error after compiler policy changed | canonical module plus explicit-layout replay | covered | Retain both canonical layout and accept/overwrite/reject policy; neither alone is sufficient. |
| An intrinsic changed operand count or canonical declaration and failed verification | parser/verifier canonicalization | parser recognition and canonicalization covered for selected ABI families | Generate the candidate corpus from the complete declaration table and add compile/link evidence so another historical ABI or unresolved external cannot be omitted. |
| A compiler accepted declarations for operations absent on a processor, sometimes failing only during instruction selection or a later fatal pass | per-operation codegen/link/assembly in a subprocess | not covered | Integrate the arch-domain operation-by-processor matrix and preserve `name_absent`, `arch_absent`, clean refusal, crash, and timeout separately. |
| Valid LLVM IR compiled at normal optimization but failed during unoptimized instruction selection | codegen at each optimization level rocKE uses | covered for one representative reproducer | Close over every bounded historical reproducer and compiler entry point while retaining stage-specific outcomes instead of one `accepted` result. |
| A backend option or function attribute parsed successfully but was ignored, produced the same binary, or was not forwarded by one compile route | ISA and code-object effect check | acceptance covered for selected controls; effect not covered | Probe acceptance and effect separately for every emitted option/attribute and every compile route. Binary inequality is not by itself proof that a request was honored. |
| SLP, SLSR, or LSR changed dependency graphs, vectorization, or address recurrences and moved kernels across register-pressure cliffs | pass-local optimized IR, remarks, then final ISA/resources | not covered | Add pass-stage transformation fingerprints for focused reproducers and qualification sentinels for optimization behavior; do not model this as a global `modern` boolean. |
| Machine scheduling changed load/use distance or wait placement while structural IR remained valid | final ISA schedule and wait-state model | not covered | Add focused instruction-order and wait-semantics checks. Successful compilation and resource metadata do not prove scheduling intent. |
| Byte-identical input IR gained VGPR pressure, scratch, or spills in a newer compiler snapshot | code-object metadata, resource report, and disassembly | not covered | Record VGPR/AGPR/SGPR, scratch, spill, and occupancy outcomes for representative kernel identities keyed by exact compiler context. |
| Narrow data types retained equivalent memory operations but acquired substantially different register-level movement | selected instruction sequence plus resources | not covered | Add dtype/layout-sensitive generated-code sentinels; memory-operation parity is too weak. |
| Source-level AMDGPU builtin parameter types changed and downstream C++ call sites stopped compiling | Clang source parsing/type checking | covered for one representative builtin family | Close over every builtin family used by HIP/C++ routes; LLVM-IR probes cannot detect source builtin migrations. |
| Compiler builds reporting the same development major produced different intrinsic-domain cells | exact tool identity plus full domain artifact | provenance only | Compare and preserve build-specific measurements; a single baseline per reported major silently loses coverage. |
| Tests or bindings selected a compiler profile different from the compiler that consumed the IR | loaded-compiler identity and route provenance | out of scope for behavior probes | Keep compiler ownership, route selection, and state isolation as configuration correctness. Do not invent a compiler capability for a harness defect. |

The history confirms that the current structural report is only the first
layer. It directly addresses canonical module policy and selected declaration
ABIs, but most observed regressions arise after parsing. The next useful host-
side additions are therefore:

1. complete declaration, frontend-builtin, and operation-by-processor closure,
   with normalized fatal-stage outcomes;
2. expand the current optimization-level reproducer into a pipeline-stage
   matrix for every bounded historical reproducer;
3. backend-option and function-attribute effect checks, not acceptance alone;
4. focused pass-transformation, ISA schedule, wait, and resource sentinels; and
5. compile-route parity so Clang, hipcc, and loaded COMGR evidence cannot be
   silently substituted for one another.

Complete-kernel resource and performance behavior remains qualification evidence
keyed by compiler and workload identity. It must not be collapsed into the
portable structural manifest, but it is the feedback that determines whether
the focused vocabulary is sufficient.

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
| Backend options emitted by rocKE | `platform/python/rocke/helpers/compile.py` |
| Function attributes emitted by rocKE | `platform/python/rocke/core/lower_llvm.py` (`finalize`) |
| ISA/resource/code-object observations | `platform/python/rocke/benchmark/perf/`, focused compiler-backed tests, and library resource gates |
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

The technical sections split some reader-facing variations into multiple probe
families:

| Reader-facing variation | Technical sections |
| --- | --- |
| 1. Module format | §1 |
| 2. Intrinsic ABI | §2 |
| 3. Target and operation availability | §3–§5 |
| 4. Compiler options | §6 |
| 5. Function attributes and metadata | §7 |
| 6. Toolchain pipeline behavior | §9 |
| 7. Generated-code behavior | §8, plus stage results from §9 |
| 8. Runtime semantics | semantic behavior in §8 and qualification gates |

Section 10 is provenance shared by every observation, not a ninth compatibility
variation.

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

The complete layout must remain one atomic observation, but its parsed deltas
are independent facts useful to a resolver:

- address-space-8 pointer/index shape;
- ELF mangling component;
- address spaces 10 through 15; and
- any future field not in the current constants.

Source history adds a second, separate contract:

#### `amdgpu.ir.explicit_data_layout_policy`

A compiler can overwrite a supplied module layout, validate it, or reject it.
The addition of address spaces 10–15 became fatal only when a later compiler
stopped overwriting the input layout. Probe both the compiler-emitted canonical
layout and code generation of a module carrying each candidate layout. Recording
only the canonical string loses whether a compatibility superset is accepted.

#### `amdgpu.ir.module_preamble`

rocKE also owns the target triple and kernel calling convention. They do not
currently branch by flavor, but they are compiler-facing module syntax and must
be observed rather than permanently assumed. Record the canonical target triple,
accepted kernel calling convention, and any mandatory module flags emitted by a
minimal AMDGPU kernel. This is one of the exhaustive baseline probes even when
all sampled compilers agree.

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
in exactly eight intrinsic families at this snapshot:

- four hand-written homogeneous declarations: fp8×fp8 and bf8×bf8 at
  16×16×32 and 32×32×16; and
- four mixed fp8×bf8 / bf8×fp8 matrix-call declarations at the same two
  shapes, represented by `FlavorType`.

Both Python and native engines change the declaration, operand preparation, and
call type. These eight keys account for eight of the nine declaration deltas
between the `llvm20` and `llvm22` tables; buffer-resource is the ninth.

**Probe.** Compile and link every affected key with both candidate register
types for each processor on which its instruction is expected to lower. Record
the exact accepted and canonical register type per key. A single representative
is useful during implementation but is not exhaustive: homogeneous and mixed
intrinsics use different source tables and can drift independently. The result
must not be `modern: true`.

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

#### `amdgpu.intrinsic.mfma_f8f6f4.call_abi`

Source history records a compiler-dependent 11-argument versus 9-argument
scaled-MFMA call contract. Current rocKE has deliberately converged on one
mangled 9-argument form that all four supported lanes accept, so no direct
flavor branch remains. That normalization does not erase the known variation:
probe the accepted spelling, overload suffix, operand count, selector positions,
and scale positions for both 16×16×128 and 32×32×64 forms. Preserve canonical
output separately from submitted legacy candidates.

This is the model for archaeology findings: a compatibility fix that removes a
branch still contributes a regression probe when an older compiler was known to
accept, ignore, auto-upgrade, or miscompile another form.

#### `amdgpu.intrinsic.table_contracts`

The complete declaration table is itself the exhaustive set. For every resolved
key, observe:

1. whether LLVM recognizes the name rather than round-tripping it as an external
   function;
2. its canonical intrinsic name and declaration;
3. whether each rocKE candidate call shape verifies;
4. whether it compiles and links for each supported processor; and
5. when target support is claimed, whether generated assembly is accepted for
   that processor.

`gen_arch_domain.py` already implements most of this protocol and preserves the
important `name_absent`, `arch_absent`, `target_unsupported`, crash, timeout, and
probe-error distinctions. The capability report should reference or embed those
measured answers rather than create a second weaker availability database.

### 3. Operation availability

The current `llvm23_target_ops` profile bit is an umbrella. Source inspection
of every call site expands it into the following independent families:

| Candidate capability family | Operations that must be probed independently |
| --- | --- |
| `amdgpu.operation.workgroup_cluster.available` | each cluster/workgroup ID read, cluster size, cluster barrier, and cluster-dimensions function attribute/code-object field |
| `amdgpu.operation.scalar_data_prefetch.available` | `s.setreg`, scalar pointer prefetch by each accepted address space, and buffer-resource data prefetch |
| `amdgpu.operation.global_prefetch.available` | global and flat prefetch, each cache-policy form rocKE exposes |
| `amdgpu.operation.tensor_data_movement.available` | every tensor load/store-through-LDS form and tensor wait |
| `amdgpu.operation.extended_scalar_barrier.available` | signal, wait, init, signal-var, join, wakeup, and leave; the four pointer-taking forms additionally have the independent address-space ABI |
| `amdgpu.operation.async_store_from_lds.available` | every width of global async store from LDS |
| `amdgpu.operation.global_transpose_load.available` | each type/width of 128-bit transpose global load |
| `amdgpu.operation.cluster_multicast_load.available` | each width of cluster load and cluster async load to LDS |
| `amdgpu.operation.scaled_wmma.available` | every scaled-WMMA op ID in the target MMA catalog, not one representative spelling |
| `amdgpu.inline_asm.scalar_control.accepted` | every inline-assembly template carrying `minimum_llvm_flavor`, including its register constraints |

The declaration table also contains operations introduced during LLVM-23 bring
up that are not all guarded by this umbrella: async markers and waits, async
buffer-to-LDS, byte global-to-LDS, DPP/swizzle, wave reductions, lane
read/write, permlane/alignbyte/WQM, AV b128 memory, scheduler controls, and
instruction prefetch. They are already included by the declaration-table and
arch-domain completeness rule. They must not disappear from the backlog merely
because their current admission path relies on measured intrinsic availability
instead of `llvm23_target_ops`.

These names are intentionally operation-oriented. The AMD execution model is
kept separate from compiler support: a target fact can say the processor has a
facility while a compiler capability says the selected toolchain can parse,
lower, and encode it.

**Probe.** Reuse the proven arch-domain method, but keep each stage's answer:

1. establish whether the exact intrinsic spelling exists and how it
   canonicalizes;
2. compile and link a call for the processor;
3. inspect the selected instruction and assemble it for that processor so a
   backend cannot silently emit an unavailable encoding;
4. for attributes or operations whose result is metadata, inspect the code
   object rather than treating successful compilation as success;
5. for operations with observable semantics, run the smallest on-device numeric
   or state-transition check; and
6. retain `unknown` when the processor cannot be targeted or a stage fails to
   answer.

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

### 6. Compiler backend-option contracts

#### `amdgpu.compiler_option.<name>.accepted`

The production compile path currently passes
`-mllvm -amdgpu-mfma-vgpr-form` when a kernel requests VGPR-form MFMA. A shipped
example compiler command also passes
`-mllvm -amdgpu-early-inline-all=true`. Source and repository history show why
option acceptance is a first-class capability rather than ordinary error
handling: an older option parser can print `Unknown command line argument` and
terminate the host process instead of returning a recoverable COMGR error.

Probe every `-mllvm` option any in-tree compiler command can construct, in a
child process using the same compiler interface that command will use. Record
accepted, rejected, process exit, timeout, and diagnostic states. Unknown is not
accepted. Production COMGR options should be derived from the compile-option
builder; example/tool commands need a source guard so a newly added option
cannot bypass the inventory.

Acceptance is distinct from effect. For `-amdgpu-mfma-vgpr-form`, separately
probe whether representative MFMAs use the requested register form and whether
the code object honors the requested AGPR allocation. A compiler may accept an
option, ignore it, change its default, or apply it only to some targets.

### 7. Kernel attribute and metadata contracts

rocKE emits five AMDGPU-specific function attributes in addition to the generic
kernel attributes:

| Candidate capability | Emitted attribute | Required observation |
| --- | --- | --- |
| `amdgpu.attribute.flat_work_group_size` | `amdgpu-flat-work-group-size` | accepted syntax and code-object workgroup limit |
| `amdgpu.attribute.scheduler_strategy` | `amdgpu-sched-strategy` | accepted values and selected scheduling policy/effect |
| `amdgpu.attribute.waves_per_eu` | `amdgpu-waves-per-eu` | accepted range plus resulting resource cap/occupancy metadata |
| `amdgpu.attribute.agpr_alloc` | `amdgpu-agpr-alloc` | accepted range plus actual AGPR/VGPR form and allocation |
| `amdgpu.attribute.cluster_dims` | `amdgpu-cluster-dims` | accepted dimensions plus emitted cluster metadata |

`uniform-work-group-size`, `norecurse`, and `nounwind` are also part of the
module contract even though they are not AMDGPU-specific. Probe parser
acceptance separately from semantic effect. A successful compile does not prove
that an attribute was honored; inspect the emitted assembly/code-object notes or
the compiler's resource report.

The same rule applies to rocKE's compiler-facing metadata and synchronization
annotations: native floating-point atomic metadata, agent scope metadata,
`nontemporal`, alias-scope metadata, and debug metadata. They need parse/codegen
coverage, while only metadata that affects a rocKE decision needs to become a
runtime selection capability.

### 8. Backend result contracts

#### `amdgpu.codegen.instruction_selection.<operation>`

For operations where LLVM can accept the IR but select the wrong instruction,
emit a call that cannot be optimized away and inspect disassembly. Known source
examples include target-specific conversion instructions, transpose loads,
scaled versus unscaled MMA forms, VGPR versus AGPR MFMA forms, scheduler
intrinsics, and instructions whose availability the arch-domain generator must
confirm with an assembler after linking.

#### `amdgpu.codegen.code_object_metadata.<field>`

Record fields rocKE consumes or promises: resolved target ID, wavefront size,
workgroup and cluster dimensions, VGPR/AGPR/SGPR counts, LDS, scratch/spill
state, and occupancy-relevant limits. Compiler-emitted target strings have
already appeared in more than one valid spelling, and resource allocation steps
are compiler observations used by occupancy checks. These are not implied by IR
acceptance.

#### Register allocation and optimization outcomes

Candidate machine IDs include:

```text
amdgpu.codegen.register_allocation.resources
amdgpu.codegen.register_allocation.register_classes
amdgpu.codegen.register_allocation.allocation_granularity
amdgpu.codegen.register_allocation.spill_state
amdgpu.codegen.optimization.selected_sequence.<operation>
amdgpu.codegen.optimization.required_operation_preserved
```

Register allocation observations must include enough context to reproduce the
answer: compiler identity, processor, wave size, optimization level, backend
options, function attributes, input-IR fingerprint, register counts, scratch and
spill state, and reported occupancy. The same IR under two compiler candidates
can produce different resource results without either compiler rejecting it.

Optimization observations similarly separate cause from result. Section 9
records the stage and transformation—for example, a call removed at `-O3`, an
operation combined, or an expression hoisted. This section records the resulting
instruction sequence and resources. If the transformation changes observable
results, the ruling evidence moves to runtime semantics and numeric
qualification.

Use focused probes only for stable, reusable contracts such as register-form
selection, allocation granularity, preservation of an operation required by a
probe, or selection of a specific instruction family. Full-kernel register
counts, schedules, spill outcomes, and compiler crashes are shape-dependent
qualification records keyed by compiler and kernel identity. They must not be
represented as global claims such as `good_register_allocator: true` or
`optimizer_supported: true`.

#### `amdgpu.codegen.failure_mode`

Compiler subprocesses can succeed, reject cleanly, crash, abort, or time out.
The arch-domain machinery already preserves these distinctions. Workload
refusal tables show that full-kernel codegen failures also depend on IR shape
and resource pressure. Record the generic result taxonomy in the capability
model; keep individual workload hashes in qualification artifacts rather than
pretending they are portable compiler features.

#### Semantic behavior and focused quirks

Machine IDs: `amdgpu.behavior.<required_semantics>` and
`amdgpu.quirk.<specific_symptom>_present`.

No broad flavor label may stand in for semantic correctness. A positive
behavior or explicit quirk needs a focused reproducer and, when execution is
required, an independent on-device reference. Source history contains cases
where an invalid intrinsic form was accepted while scale operands were dropped,
and cases where unknown `llvm.*` names survived verification as external calls.
Those pipeline behaviors need negative controls even after rocKE normalizes its
emission.

### 9. Toolchain pipeline behavior

The probe design must preserve which stage answered:

| Stage | Distinct fact it can establish |
| --- | --- |
| parser/canonicalizer (`opt -S`) | recognized intrinsic, canonical name and declaration |
| verifier/assembler (`llvm-as` or verifier pass) | structural IR validity, not symbol existence |
| optimizer | whether the test call survives; never use an optimizable-away call as availability evidence |
| target codegen | instruction selection or target-specific refusal |
| link | unknown intrinsic left as an external symbol |
| text assembler | selected instruction is valid for the target |
| code-object inspection | metadata and resource result |
| device execution | numeric and state-transition semantics |

Known policy variations include verifying declarations before auto-upgrade,
auto-upgrading legacy intrinsic spellings, treating unknown `llvm.*` names as
externals, overwriting versus validating an explicit data layout, and fatal
rather than recoverable failures. A report that reduces these to `compile:
true/false` is not sufficient for a generalized resolver.

### 10. Toolchain identity and provenance

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

### 11. Frontend source ABI

Machine ID pattern: `amdgpu.frontend.builtin.<stable-name>.source_abi`.

rocKE's HIP/C++ compile routes and downstream source consumers can call Clang
AMDGPU builtins rather than submit LLVM IR directly. Compiler migrations can
change a builtin's accepted C/C++ parameter or result type while the underlying
LLVM intrinsic remains unchanged. A structural IR probe cannot observe that
breakage.

Probe the smallest source call for every builtin family rocKE emits or invokes,
using each historical language-level type form, and preserve Clang parse/type-
check diagnostics separately from LLVM-IR lowering. This is a source-interface
contract, not evidence that the lowered instruction is available, honored, or
numerically correct.

## Exhaustive probe backlog

The following is the complete known backlog derived from current source and
compatibility history. “Existing” means rocKE already has a direct measurement
mechanism that should be reused; it does not mean the current structural report
contains the result.

| Layer | Probe unit | Cardinality rule | Status |
| --- | --- | --- | --- |
| Identity | compiler, COMGR interface, `clang`, `opt`, linker, assembler | every executable/library participating in a result | partial in current report |
| Frontend source ABI | AMDGPU builtin names and parameter/result language types used by HIP/C++ routes | every builtin family emitted or called by rocKE | representative family in current report; closure backlog |
| Module | canonical triple, calling convention, complete data layout, mandatory flags | each target context; preserve exact text | current report |
| Module policy | supplied-layout overwrite/accept/reject | each distinct candidate layout against each compiler | current report |
| Declaration ABI | buffer-resource candidates | every spelling × count width rocKE has emitted | current report |
| Declaration ABI | fp8/bf8 MFMA packing | all 8 affected homogeneous/mixed keys × candidate packed type | current report |
| Declaration ABI | named barriers | all 4 pointer-taking keys × candidate address space | current report |
| Declaration ABI | scaled MFMA historical/current forms | both known shapes × spelling × operand-count form | current report |
| Declaration universe | recognition, canonical declaration, call verification | every key returned by `_decls_for_flavor` | existing arch-domain stage A; integrate |
| Target names | exact and generic processor acceptance | every target in `arch_specs.json` and every fallback | selected context in current report; closure backlog |
| Target lowerability | compile/link/assembly outcome | every declaration key × supported processor | existing arch-domain stage B; integrate |
| Broad operations | parser, lowerability, selected ISA, metadata/semantics where relevant | every operation listed in §3, not one per umbrella family | partial/existing plus backlog |
| Backend options | acceptance and semantic effect | every option constructible by compile helpers × relevant target | known controls accepted in current report; effect and closure backlog |
| Function attributes | acceptance and semantic effect | every attribute × every emitted value class × relevant target | known controls codegen in current report; effect and closure backlog |
| Register allocation controls | requested register form, allocation granularity, and option/attribute effect | focused control probes × relevant target/compiler context | partial plus backlog |
| Register allocation outcomes | VGPR/AGPR/SGPR, scratch, spills, occupancy metadata | every qualified kernel identity × compiler context | workload qualification |
| Optimization behavior | stage transformation and selected instruction sequence | focused stable contracts plus every affected qualification workload | optimization-level reproducer in current report; pass effects backlog |
| Code-object results | target and resource metadata | representative minimal kernels plus qualification workloads | partial |
| Failure taxonomy | reject/crash/abort/timeout | every subprocess probe and qualification compile | partial; normalize |
| Semantics | numeric/state correctness | every selected behavior that cannot be proved statically | qualification, on device |
| Artifacts | golden/cache/AOT partitioning | every behavior that changes emitted bytes or generated code | migration after resolver |

This table deliberately mixes small structural probes and larger qualification
lanes while keeping their evidence classes distinct. “Exhaustive” means no
known behavior category or source-derived member is omitted; it does not claim
that a finite set of microprobes can replace whole-kernel correctness and
resource qualification.

## Completeness and future-growth rules

A generalized design should make omissions mechanically difficult:

1. **Declaration closure.** The report's declaration/availability section is
   generated from `_decls_for_flavor`, including `MATRIX_CALLS`. Tests compare
   the measured key set with every engine's resolved table.
2. **Operation-gate closure.** An AST/source guard enumerates every use of
   `llvm23_target_ops`, ranked/exact flavor checks, and `minimum_llvm_flavor`.
   Each must map to a named capability or an explicit non-capability rationale.
3. **Option closure.** Every `-mllvm` option constructible by production
   helpers, examples, or tools is present in the option-probe registry before
   the corresponding compiler interface can receive it.
4. **Attribute closure.** Every `amdgpu-*` attribute emitted by finalization is
   registered with an acceptance probe and, where meaningful, an effect probe.
5. **Stage fidelity.** Results retain parser, codegen, link, assembly,
   code-object, and execution stages. Later consumers may aggregate them; the
   evidence must not.
6. **Context fidelity.** Target, wave size, flags, optimization level, input
   module, and tool digests are cache inputs. A result from `opt` does not
   silently describe the loaded COMGR, and an emission override does not describe
   the compiler that performs codegen.
7. **Unknown is first-class.** Unsupported target, absent name, clean refusal,
   crash, timeout, and unmeasured are distinct. No-data never becomes false or
   true by default.
8. **Archaeology retention.** A compatibility fix that removes a flavor branch
   adds a regression candidate to the probe corpus when the removed form exposed
   a real compiler difference.
9. **Qualification remains mandatory.** Byte identity, complete-module
   validity, ISA/resource checks, and numeric verification continue after
   capability-driven selection.

With these rules, implementation can still land in small vertical slices, but
the design is based on the whole known problem rather than whatever the first
slice happened to measure.

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
then rerun the existing qualification gates. Implement the source-derived
backlog systematically; when a qualification failure reveals an axis absent
from that backlog, add the newly discovered probe. Otherwise fix rocKE or the
test rather than turning every failure into a compiler capability.

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

### Why the probe started with two behaviors

The report design began with one module-level contract and one intrinsic-call
contract:

1. `amdgpu.ir.data_layout` has a visible flavor-dependent module representation;
2. `amdgpu.intrinsic.make_buffer_resource.abis` has visibly different
   declarations, call shapes, and operand preparation.

Each decision is already represented in rocKE source and focused tests, can be
measured with a small standalone compiler invocation, and does not require the
probe to solve target-operation or numeric qualification. Before fixing the
initial schema, the structural corpus expanded to the other module and
intrinsic ABI contracts listed below. It remains a focused evidence format, not
a complete AMD LLVM compatibility model.

The current artifacts are:

- [`probe.py`](probe.py) — the standalone probe runner;
- [`capabilities.schema.json`](capabilities.schema.json) — the strict report
  schema;
- [`example.capabilities.json`](example.capabilities.json) — an illustrative
  report; and
- [`SCHEMA_DECISION.md`](SCHEMA_DECISION.md) — the normative naming, status,
  evolution, cache, addition, and retirement rules.

`capabilities.json` is deliberately small. It exists to show that focused
probes produce understandable, repeatable answers across compiler builds. It is
not a production configuration contract and rocKE lowering does not read it.

The current report records:

- complete compiler and supporting-tool provenance;
- target triple, processor, probe/schema source digests, and cache fingerprint;
- canonical module layout and preamble;
- explicit-layout acceptance or overwrite policy;
- the complete buffer-resource candidate matrix;
- fp8/bf8 packed-operand, named-barrier pointer, and scaled-MFMA call ABIs;
- a representative frontend FP16 builtin source ABI;
- requested and generic processor-name acceptance;
- focused codegen outcomes at each optimization level rocKE uses;
- selected backend-option and function-attribute acceptance; and
- submitted candidates separately from compiler-canonical output.

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

Cache entries are keyed by probe/schema source digests, target, processor,
selected tool paths, complete tool identities, and executable digests. The report itself
contains no timestamp or cache-hit marker, so a fresh
run and a cache hit produce byte-identical JSON. Cache behavior and elapsed time
are printed by the CLI rather than mixed into the evidence.

## Demonstration sequence

1. Select a compiler with `--python-env` or `--llvm-bin`; environment variables
   and then `PATH` provide fallbacks.
2. Hash `clang` and `opt`, read their complete identities, and retain source and
   patch revisions as provenance only.
3. Ask `clang` to emit a minimal AMDGPU kernel module and replay every known
   explicit layout.
4. Ask `opt` to resolve every candidate in each bounded structural ABI corpus;
   record accepted, rejected, and canonical forms.
5. Write checkout-matched `capabilities.json`, optionally validate it, print a compact
   summary, and cache only a complete report.
6. Repeat with `--refresh` across one selected public wheel per weekly window and
   compare behavior columns rather than inferred compiler versions.

Integration with lowering, COMGR probing, CI, and the full intrinsic arch-domain
is intentionally outside this schema-and-evidence change. A later vertical
slice should consume one capability in the centralized profile, retain flavor
inference as a fallback, update Python and C++ together, and run the relevant
qualification gates.

## Conclusion

The current flavor model is not one version check; it is a compact label for
several distinct concerns. The migration should not replace that label with one
large `capabilities` boolean or a renamed profile number. It should expose the
small observed facts rocKE actually consumes, retain explicit unknown states,
and use qualification failures to test whether those facts are sufficient.

The existing arch-domain generator demonstrates that direct, evidence-bearing
probes are practical in this repository. The safest implementation order is to
reuse that approach for one structural ABI decision, compare it with current
flavor-derived behavior, and run the affected qualification tests, then proceed
through the source-derived backlog. The schema and resolver must nonetheless be
designed for independent module, declaration, target, option, attribute,
codegen, metadata, failure-mode, and semantic observations from the start.
Qualification can discover additional behaviors, but it is no longer the plan
for discovering the already-visible scope.
