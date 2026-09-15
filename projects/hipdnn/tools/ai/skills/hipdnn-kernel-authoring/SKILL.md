---
name: hipdnn-kernel-authoring
description: "Author a hipRTC-compilable HIP kernel that satisfies a given hipDNN graph: read the graph, specify the operation, mine prior art and device facts, write the kernel, and prove it against the hipDNN reference executor."
argument-hint: "<graph-json-or-binary-or-sample-path> [--arch gfx942|gfx950|...]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Kernel Authoring

Input: a hipDNN graph. Output: HIP source that computes it, plus evidence that the
numbers match a hipDNN reference on the shapes actually run. Execute
[RUNBOOK.md](RUNBOOK.md), the **only ordered workflow**. These pages supply
contracts, not alternate procedures:

| Reference | Owns |
|---|---|
| [graph-analysis.md](graph-analysis.md) | Reading a graph, operation specification, fusion vs multiple launches, generalization |
| [prior-art.md](prior-art.md) | Where to mine an algorithm: in-tree, external repositories, HIP/ISA documentation |
| [device-envelope.md](device-envelope.md) | hipRTC language/header/flag envelope and target-architecture facts, local or remote |
| [harness.md](harness.md) | Reference oracle, input parity, tolerance, launch, comparison and reporting |
| [hipdnn-rocke-kernel-authoring](../hipdnn-rocke-kernel-authoring/SKILL.md) | The same arrow for the rocKE dialect: a Python builder lowered through comgr and packed at build time |
| [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) | Everything downstream: native symbols, descriptors, registration and graph coverage |
| [hipdnn-ingestor-engine](../hipdnn-ingestor-engine/SKILL.md) | Production mining and lowering once an integration exists: corpus, sweeps, tuning, packaging, rocKE |

## Position in the flow

`graph → kernel → integrate → optimize`. This skill owns the first arrow only. The
second is [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md), which
consumes this skill's four handover facts — the entry-point signature, the bundle's file
set, the macros the source requires bound with their legal values, and the launch
geometry and workspace the kernel assumes — and owns everything after them: the native
symbols, the descriptors, registration with the testing system, and the graphs that
verify it. The fourth is measure-driven tuning and is **not** in scope here: a kernel
that is correct and slow is this skill's success, and reporting a speed claim it did not
measure is its failure.

Correctness is the deliverable. Performance observations are optional context and
must be labelled as unmeasured unless timed.

## Entry contract

Record before authoring, and treat a missing item as a blocked gate, never a default:

| Fact | Why it blocks |
|---|---|
| Graph and its form (live `Graph`, JSON, binary blob, sample source) | Determines how nodes/UIDs/strides are enumerated — [graph-analysis.md](graph-analysis.md) |
| Target architecture(s), and whether one is locally present | Compile flag, ISA availability, and whether device proof needs a scheduler — [device-envelope.md](device-envelope.md) |
| A capable hipDNN reference for this graph | Without an independent oracle there is nothing to prove against — [harness.md](harness.md) |
| Integration target's launch ABI, or **"unbound" — the normal answer** | A kernel authored against the wrong *bound* argument list is diagnosed nowhere downstream |
| Dtypes, layouts (derived from strides) and the shape envelope to support | Decides what is a runtime parameter and what is a `-D` specialization |

**"Unbound" is the normal answer, not a fallback.** The integration writes the dispatch
handler, so the kernel's argument list is an **output** of this skill and the handler's
`launch()` is written from it. Only say "bound" when you are genuinely adding a kernel to
a pack that already exists and already ships a handler — in which case name that pack and
take the argument list from its `launch()` body, in order, because wrong arity is
diagnosed nowhere. Do not invent a handler signature either way.

When the request names no target architecture — a generic in-tree sample, say — the
architecture is whichever device the correctness proof will run on. Name that device
explicitly and scope every claim to it. An unnamed architecture is the blocked case;
a deliberately chosen one is not.

## Generalization

Solving exactly the given graph is acceptable. A kernel that is general over the
dimensions in that graph's own schema is preferred, and is the default whenever
generality is trivial — dimensions, strides and counts passed as kernel arguments
cost nothing but registers.

| Property | Default treatment |
|---|---|
| Dimensions, counts, strides, scalars | Runtime kernel arguments |
| Element type, layout family, algorithmic structure (tile shape, unroll, shared-memory budget) | Compile-time `-D` specialization |
| An enumerated list of supported shapes | **Anti-pattern.** Existing engines' shape tables are a coverage model to avoid, not to imitate |

Generality claims are bounded by what ran. A kernel parameterized over `seq_len`
that was validated at one length is "parameterized, validated at one point" — say
exactly that.

## Completion and handoff

Completion requires all of:

- A written operation specification derived from frontend attributes and the
  matched `*_attributes.fbs`, with every schema field consumed, explicitly rejected,
  or proven inert; and every tensor UID classified input, output, or virtual.
- HIP source that compiles through hipRTC for each target architecture, with the
  exact compile options used, and the entry-point signature stated verbatim.
- A named decomposition: one kernel, a legal fusion, or an ordered sequence of
  launches with each launch's inputs, outputs and any scratch it requires.
- Reference-vs-kernel numerics from a run that actually executed both sides on
  byte-identical inputs, with per-output tolerance and its provenance, on named
  shapes, on a named device.
- An explicit does-not-prove list: untested shapes, untested dtypes, untested
  architectures, reference features that declined, and any performance silence.

A compile is not a correctness result. A reference that skipped or declined is an
unmeasured bucket, not a pass. A green harness that never executed the kernel is the
failure mode this skill exists to prevent — assert the kernel ran and its output
buffer changed. Generated source, proposed commands and queued jobs cannot be
reported as an authored kernel.
