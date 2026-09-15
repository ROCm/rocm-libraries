---
name: hipdnn-rocke-kernel-authoring
description: "Author a rocKE kernel that satisfies a given hipDNN graph: stand up the rocKE Python environment, pick or write the builder and its spec dataclass, prove the numbers in rocKE's own harness, and lower and pack it into a kpack the ingestor can ship."
argument-hint: "<graph-json-or-binary-or-sample-path> --arch <gfx950|gfx942|gfx1151|gfx1201>"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN rocKE Kernel Authoring

Input: a hipDNN graph. Output: a rocKE builder plus the spec dataclass a descriptor can
hydrate, numeric evidence from rocKE's own harness, and a packed `.kpack` whose shipped
descriptor reads `kind: kpack`. Execute [RUNBOOK.md](RUNBOOK.md), the **only ordered
workflow**. These pages supply contracts, not alternate procedures:

| Reference | Owns |
|---|---|
| [rocke-docs-map.md](rocke-docs-map.md) | Where rocKE's own documentation answers a question, which pages to ignore, and which two read as current and are not |
| [handover-contract.md](handover-contract.md) | The spec-dataclass ↔ KMD contract, the baked/runtime-param decision, and what a packed kpack still owes |
| [build-and-environment.md](build-and-environment.md) | The author-scoped CMake options, the two environment variables that change results, and the Python environment |
| [hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md) | The same arrow for the hipRTC dialect: HIP source compiled from a string at `prepare()` |
| [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) | Everything downstream: native symbols, descriptors, registration and graph coverage |
| [rocke-mining.md](../hipdnn-ingestor-engine/rocke-mining.md) | Mining an *existing* builder for applicability, specialization agreement, geometry and ABI |

This skill does **not** restate rocKE's authoring interface. rocKE ships its own
documentation tree at `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/`; that
tree owns the Python API, the IR vocabulary, the primitives, the learning program and the
per-family instance maps.
[rocke-docs-map.md](rocke-docs-map.md) is the router into it. What this skill owns is the
seam rocKE's docs never mention: outside `dsl_docs/hipdnn_provider/plan.md`, which is
superseded, no page under `dsl_docs/` mentions a UKD, a KMD, a kpack or `hkp_pack` at all.

## The fact that reorganizes everything

**rocKE never reaches the hipDNN runtime as rocKE.** `hkp_pack` imports the named builder
module at *build* time, calls the builder, lowers it through comgr, packs the code object
into a per-arch `.kpack`, and rewrites the shipped descriptor from `kind: rocke` to
`kind: kpack` before the runtime loader ever sees it
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/pipeline.py:999-1011`).
The authored `spec` block survives only as `provenance` on the shipped document
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/pipeline.py:982-988`).
There is no runtime rocKE adapter, and this is by design.

Three consequences, each of which bites an author who assumes otherwise:

- **No rocKE Python runs at launch.** Not the dispatcher, not `supports_*`, not any
  preflight check. Every guard those functions perform runs once, at pack time, against
  the *descriptor's* spec values. Any per-launch check the builder would have made must
  be replicated as a matcher criterion — [handover-contract.md](handover-contract.md).
- **A packed `.kpack` is not a finished integration.** It is a compiled artifact. Read
  [handover-contract.md](handover-contract.md) for what is still owed, and to whom.
- **`arch` is a Python keyword argument through comgr, not a compiler flag.** There is no
  `--genco` and no `--offload-arch` on this path. `gfx950` lowers fine on a gfx942 or
  CPU-only host, because arch is a codegen target rather than a host requirement. Only
  *running* the kernel needs the device.

## Position in the flow

`graph → kernel → integrate → optimize`. This skill owns the first arrow for the rocKE
dialect only. Its sibling [hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md)
owns the same arrow for hipRTC — HIP source compiled from a string at `prepare()`. Same
destination, different machinery: a rocKE kernel is a Python builder that emits typed SSA
IR, lowered ahead of time through comgr into a code object and packed at build time, so
choosing between them is choosing a toolchain, not a quality level.

The second arrow is [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md).
The fourth is measure-driven tuning and is **not** in scope here: a kernel that is correct
and slow is this skill's success, and reporting a speed claim it did not measure is its
failure.

[rocke-mining.md](../hipdnn-ingestor-engine/rocke-mining.md) sits downstream of this
skill, not beside it. It extracts applicability, the specialization contract, geometry and
ABI *from a builder that already exists*, and it explicitly refuses to invent one
(`projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine/rocke-mining.md:3-5`). This skill
produces the builder it mines.

## Entry contract

Record before authoring, and treat a missing item as a blocked gate, never a default:

| Fact | Why it blocks |
|---|---|
| Graph and its form (live `Graph`, JSON, binary blob, sample source) | Fixes the operation, the dtypes, the layouts and the shape envelope the spec must express |
| Target architecture(s), and whether one is locally present | `arch` is a required builder keyword and a required KDP field; lowering needs no device but the numeric proof does |
| Kernel family, and whether you extend an existing builder or write a new one | Decides whether this is an authoring job at all — an existing builder routes straight to [rocke-mining.md](../hipdnn-ingestor-engine/rocke-mining.md) |
| An independent numeric reference for this operation | rocKE's harness proves against torch; without a reference there is nothing to prove |
| Which spec fields are baked and which are runtime parameters | Decides the descriptor count and, when wrong, produces silently wrong numbers — [handover-contract.md](handover-contract.md) |

**Extending an existing builder is the common case and is not this skill.** Roughly thirty
builders already ship under `dnn-providers/hip-kernel-provider/rocke/library/kernels/`,
mapped per family by `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/instances/index.md`.
Check for one before writing a new one; if a shipped builder serves the graph, the work is
mining and descriptor generation, and this skill's only remaining job is
[handover-contract.md](handover-contract.md).

When the request names no target architecture, the architecture is whichever device the
numeric proof will run on. Name it explicitly and scope every claim to it. rocKE's
supported set is `gfx942`, `gfx950` (CDNA3/4, Linux only, wave64) and `gfx1151`, `gfx1201`
(RDNA3.5/4, Linux and Windows, wave32) —
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/setup_guide.md:27-28`.
An unnamed architecture is the blocked case; a deliberately chosen one is not.

## The builder contract, stated once

`hkp_pack` refuses to pack a builder whose signature is not exactly `(spec, *, arch)`
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:179-232`).
One positional parameter, annotated with a dataclass; `arch` keyword-only; nothing else,
and no `*args`/`**kwargs`.

**A parameter with a default is still rejected** — having a default is exactly what makes
it dangerous, because it freezes a performance knob that no descriptor can reach and
nothing records
(`dnn-providers/hip-kernel-provider/descriptor-packaging/python/hkp_pack/rocke_compile.py:194-195`).
That rule is about *extra* parameters. `arch` itself may carry a default and does in every
shipped builder, because the descriptor always supplies it.

Write the signature in exactly that form. The packer and the generator's preview check
implement the rule independently
(`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:100-133`), they differ
at the edges, and only the exact form passes both.

## Completion and handoff

Completion requires all of:

- A spec dataclass whose every field is classified — baked, runtime parameter, or
  matcher-only — with the no-default fields listed, because a descriptor must supply each
  one or the pack dies with a `TypeError`.
- A builder whose signature `introspect()` accepts with an empty `signature_error`, and
  the arch list it probed, stated as *probed*, not as *declared* — rocKE declares arch
  support nowhere (`projects/hipdnn/tools/IngestorGenerator/codegen/sources/rocke.py:84-87`).
- Numeric evidence from a run that actually executed on a device of the named
  architecture, against an independent reference, with the tolerance and its provenance.
- A `.kpack` produced by a real `hkp_pack` run, with the shipped descriptor's
  `kernel_source.kind` observed to be `kpack` and its `sha256` matching the archive.
- The handover statement of [handover-contract.md](handover-contract.md): what is baked,
  what the matcher must therefore pin, which pack-time checks must be replicated as
  matcher criteria, and the launch geometry and kernarg order the engine will have to
  restate in C++.
- An explicit does-not-prove list: untested shapes, untested dtypes, untested
  architectures, and the fact that a packaging run proves packaging, not dispatch.

A lowering is not a correctness result. A pack that validated is not a kernel that
dispatched — **no rocKE-specific native pack exists in this tree**, so a descriptor
authored today resolves against symbols some other pack registered
(`dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors/README.md:81-88`).
Say that rather than implying a finished path. Generated source, proposed commands and
queued jobs cannot be reported as an authored kernel.
