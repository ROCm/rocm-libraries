# Runbook: author a kernel for a hipDNN graph

The **only ordered workflow**. [SKILL.md](SKILL.md) owns entry and completion; the
linked pages own contracts. Each gate requires a current observation; a previous
session's log, a sibling graph's result or a proposed command is not evidence.

## Paths and interpreters

Resolve absolute paths first. These are explicit arguments, not implicit tool inputs:

```bash
REPO=/absolute/path/to/rocm-libraries
HIPDNN="$REPO/projects/hipdnn"
INSTALL=/absolute/path/to/hipdnn-install      # find_package root for the harness
WORK=/absolute/path/to/scratch                # kernel source + harness, outside product source
GRAPH=/absolute/path/to/graph.json-or-.bin    # or a samples/ source path
ARCH=gfx942
```

Keep authored experiment material — kernel drafts, harness, logs — under the
workspace's WIP/evidence roots, not in product source, until the kernel is handed to
integration. Do not run GPU work on a login host; see [device-envelope.md](device-envelope.md).

## 1. Entry

Record [SKILL.md](SKILL.md)'s entry contract: the graph and its form, target
architecture(s) and local availability, intended launch ABI or "unbound", dtypes,
layouts, and the shape envelope requested. State whether the request is "serve this
graph" or "serve this operation family"; the second is a scope decision, not an
assumption you may make.

**Gate:** a graph in hand, a named architecture, and a stated ABI position.

## 2. Read the graph and specify the operation

Follow [graph-analysis.md](graph-analysis.md). Enumerate nodes, node types, tensor
UIDs, dims, strides, dtypes and virtual status; reconstruct edges from the
`*_tensor_uid` fields; then write the operation specification — the mathematics, the
conventions it depends on, and the disposition of every schema field.

Do not begin from the operation's popular name. `SDPA_FWD` names a family whose
masking, scaling, GQA, layout and deprecated-field precedence differ per graph, and
at least one precedence rule in tree is a live trap (`Knowledge/hipdnn/sdpa-mask-attribute-precedence.md`).

**Gate:** every node classified, every tensor classified input/output/virtual with
its UID resolved as [graph-analysis.md](graph-analysis.md) describes, every matched
schema field consumed, rejected or proven inert, and the specification written down.
An unresolved semantic question blocks the affected path.

## 3. Establish the oracle

Confirm a hipDNN reference can execute *this* graph before authoring anything —
[harness.md](harness.md), and the decline list in
`$REPO/dnn-providers/integration-tests/README.md` ("What the reference
executors cannot verify"). Paged KV, varlen, ragged offsets, block-sparse masks,
sink tokens, dropout, FP8 descale and softmax statistics are declined by both CPU
and GPU references; **CPU is not a fallback for any of them.**

If the graph carries a declined feature, the choice is: narrow the graph to a
representable form and state the narrowing, supply an independently trusted
reference and state its provenance, or record BLOCKED. Fabricated golden data and a
reference derived from the kernel under test are both disqualifying.

**Gate:** a named, capable, independent oracle, or an explicit BLOCKED.

## 4. Mine prior art and pin the device envelope

Use [prior-art.md](prior-art.md) to find how the operation is actually implemented —
in-tree first (`projects/composablekernel`, `projects/miopen`, `projects/hipblaslt`,
`projects/rocprim`, `projects/rocwmma`), then external semantic references. Record
what you took from where, and what you deliberately did not take.

Use [device-envelope.md](device-envelope.md) to pin the compile envelope and the
target device's facts. When the target architecture is not the local one, obtain its
facts from that device, not from memory.

**Gate:** a named algorithm with a source, and a recorded compile envelope and
device-fact set for each target architecture.

## 5. Decompose

Decide, per [graph-analysis.md](graph-analysis.md): one kernel, a legal fusion, or an
ordered sequence of launches. Fusion is preferred where the producer/consumer edge is
elementwise-local or already tiled together; it is not preferred where it forces a
global synchronization inside a kernel.

A multi-launch decomposition is a legitimate answer, not a fallback to apologize for.
Record for each launch: its inputs and outputs by UID, its grid/block shape, and any
scratch buffer it needs. Scratch is the decomposition's cost: an intermediate that
the graph marks virtual has no caller-supplied buffer, so a multi-launch plan needs
workspace, and whether the downstream launch ABI can supply it is an integration
constraint that must be raised now, not discovered later.

Record the generalization decision from [SKILL.md](SKILL.md): which quantities are
runtime arguments and which are `-D` specializations, with a reason for each
specialization.

**Gate:** a written decomposition with per-launch contracts, scratch requirements and
the generalization decision.

## 6. Author the kernel

Write HIP source inside the hipRTC envelope ([device-envelope.md](device-envelope.md)).
Non-negotiables:

- The entry point is `extern "C" __global__`. If the ABI is bound, its argument list
  is the downstream handler's `launch` argument list verbatim, in order — **wrong
  arity or order is diagnosed nowhere**: hipRTC compiles it, symbol lookup resolves
  it, and the launch passes whatever it has into whatever you declared.
- Index through the graph's **strides**, not an assumed contiguous layout. Layout is
  not an enum in the schema; it exists only as the stride pattern.
- Accumulate in `float` regardless of storage dtype, and cast on store — the in-tree
  precedent is `ConvFwd.cpp`.
- Use `int64_t` index arithmetic wherever an offset can exceed 2^31.
- Guard every compile-time-bound macro with `#ifndef <NAME>` / `#error`. An unbound
  token otherwise compiles against whatever it happens to mean and fails only in the
  numbers.
- One source, many variants: resolve a dtype tag to a device type with a two-level
  macro paste, never a one-level one.

**Gate:** hipRTC compiles the source for every target architecture, with the compile
log captured. A compile is not a correctness result and must not be reported as one.

## 7. Prove it

Build and run the harness of [harness.md](harness.md): identical seeded inputs to
both sides, hipDNN reference on one, your kernel on the other, per-output comparison
at a tolerance whose provenance you state.

Three assertions the harness must make, because each corresponding failure is
otherwise indistinguishable from success:

1. The reference actually executed — not declined, not skipped.
2. Your kernel actually launched and wrote its output — sentinel-fill outputs before
   launch and confirm they changed.
3. The comparison covered every graph output, including ones you did not expect to
   be interesting.

Run the shape set you claim, not one point of it, plus the boundary cases the
specification implies: a non-contiguous stride, a dimension of 1, a
non-tile-multiple size, and the smallest and largest shapes in the claimed envelope.

**Gate:** per-output pass at a stated tolerance on named shapes on a named device,
with the three assertions above satisfied. A skipped reference, an unlaunched kernel
or an all-zero comparison is a failed gate.

## 8. Report and hand off

Report: the operation specification, the decomposition, the kernel source and its
entry-point signature, the compile options per architecture, the harness invocation,
the numeric results per output and shape, and the does-not-prove list from
[SKILL.md](SKILL.md).

For integration, state the four facts
[hipdnn-ingestor-engine](../hipdnn-ingestor-engine/SKILL.md) needs at its RUNBOOK
stage 3: the entry-point signature, the bundle's file set (sources plus every header
they include), the compile-time macros the source requires bound and their legal
values, and the launch geometry and workspace the kernel assumes. If those conflict
with an existing pack's registered handler, say so — that conflict is a rebuild
decision for the integration skill, not something to paper over by changing the
kernel's mathematics.

**Gate:** the report is complete and every claim in it traces to an observation in
stages 2-7.
