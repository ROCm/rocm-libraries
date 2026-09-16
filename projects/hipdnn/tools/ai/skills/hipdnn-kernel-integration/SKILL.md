---
name: hipdnn-kernel-integration
description: "Land a kernel you hold as a complete hipDNN integration: decide and write its native symbols, emit its descriptors, register it with the shared testing system, and prove it with graphs."
argument-hint: "<kernel-source-path-or-dir> [--arch gfx942|gfx950|...]"
allowed-tools: Bash, Read, Write, Edit, Grep, Glob
---

# hipDNN Kernel Integration

Input: a kernel whose numerics are already proved. Output: a hipDNN integration that a
graph reaches without you in the room. Execute [RUNBOOK.md](RUNBOOK.md), the **only
ordered integration workflow**. These pages supply contracts, not alternate procedures:

| Reference | Owns |
|---|---|
| [native-pack.md](../hipdnn-ingestor-engine/native-pack.md) | Every native hook: signatures, obligations, registration and inventory proof |
| [graph-contract.md](../hipdnn-ingestor-engine/graph-contract.md) | Graph semantics, UID edges, field dispositions and reference capability |
| [hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md) | Upstream: authoring the kernel source and proving its numerics |
| [hipdnn-ingestor-engine](../hipdnn-ingestor-engine/SKILL.md) | Production mining and lowering: corpus and workloads, sweeps and tuning, packaging, rocKE, and the `hiprtc_file` reuse case |
| [hipdnn-superbuild](../hipdnn-superbuild/SKILL.md) | Configuring, building and installing the provider your new symbols live in |
| [Integration test suite](../../../../../../dnn-providers/integration-tests/README.md) | Bundles, sweeps, `--test-engine`, per-provider TOML and external registration |

## The four deliverables

An integration is four things, and **a change that produces three of them is not
done**:

1. **Descriptors** — the UED/UHD/UDD/KMD/UMD/KDP set that names your kernel, its
   metadata and its matchers.
2. **The native escape hatches this kernel needs** — its `graph_match`, any kernel- or
   graph-scoped matcher, `workspaceBytes`, an `IKernelDispatchHandler`, and optionally a
   score, written and registered. [native-pack.md](../hipdnn-ingestor-engine/native-pack.md)
   specifies all of them; step 3 of the RUNBOOK writes them.
3. **Registration with the testing system** — an `add_external_integration_test_target`
   entry naming your engine and its TOML, so the shared suite can be pointed at you by
   name rather than by hand.
4. **Tests and graphs that verify it** — bundle cases under
   `dnn-providers/integration-tests/integration-test-bundles/`, covering what your pack
   actually admits.

Deliverable 4 without 3 is a private harness. Deliverable 2 without 4 is a pack nobody
has run. Report all four or report the integration as incomplete.

## New symbols are the default

An integration almost always adds its own symbols. **Reuse is the narrow exception**:
another kernel into a pack that is already yours, with the pack named and the reason
stated. Three things follow, and all are load-bearing:

- **Two of the three shipped ingestor packs are reference scaffolds, not integration
  targets.** `PointwiseAdd` computes one element under
  `if(blockIdx.x == 0 && threadIdx.x == 0)` (`kernels/PointwiseAdd.cpp:11-12`) at grid
  1×1×1 (`PointwiseNative.cpp:436`), and `ConvFwd` is a naive direct convolution that
  serves 6 of 1218 `ConvolutionFwd` bundle cases. They exist to exercise the ingestor
  path end to end. Hanging a real kernel off one is never the answer to "is there an
  existing pack".
- **The third, `hipkernel:BatchnormInference`, is a real pack — and extending it still
  turns on whether it is yours.** It is not a scaffold by any of the tests above: a
  full-tensor kernel computing one element per thread across N·C·H·W with its own
  bounds guard (`kernels/BatchnormInference.cpp:64-94`), a computed grid rather than a
  fixed one (`BatchnormInferenceNative.cpp:642-647`), three io dtypes
  (`BatchnormInferenceNative.cpp:97-101`), nine shipped variants
  (`TestBatchnormInferencePacks.cpp:44-63`) and a matcher with a real refusal surface —
  15 parameterized refusals against 9 acceptances
  (`TestBatchnormInferenceMatchers.cpp:165-314`,
  `TestBatchnormInferenceMatchers.cpp:65-129`). Its limits are equally concrete and are
  the axes an extension would move: one proved architecture
  (`IngestorGenerator/configs/batchnorm_inference.yaml:49`), no device-level
  integration test wiring it, and 10 of 82 `BatchnormInference/Default` bundle cases
  that mirror its own unit-test shapes. A reader who did not ship it is still on the
  create path.
- **A fourth, `hipkernel:ConvPointwiseRtc`, is the only pack serving a two-node graph,
  and it is likewise available only to whoever shipped it.** A rank-4 `ConvolutionFwd`
  whose output tensor is virtual, consumed by a unary `Pointwise`, computed in one
  launch with no workspace (`packs/ConvPointwiseRtcNative.cpp`,
  `kernels/ConvFwdPointwiseFused.cpp`). Nine shipped variants over three block sizes and
  three activations; its handler routes `kernel_source.kind` through
  `buildIngestorKernelCode`, so it can serve a `hiprtc_file` descriptor. Its live axes
  are its stated limits: gfx90a alone, FLOAT alone of the three `HKP_IO_DTYPE` tags that
  compile, and `{RELU_FWD, ABS, NEG}` of the four `HKP_ACTIVATION` tags.
- **The operations people ask for mostly have no pack at all.** Only `ConvNative.cpp`,
  `PointwiseNative.cpp`, `BatchnormInferenceNative.cpp` and
  `ConvPointwiseRtcNative.cpp` exist under
  `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/`, and
  the registration table names exactly those four (the `s_packs` table in
  `IngestorPacks.cpp`). Layernorm, RMSnorm and resample live on `hip_mlops_engine` as
  hand-written plan builders with no ingestor pack; batchnorm now has both, because
  `hip_mlops_engine`'s builder still claims the identical single-node graph
  (`BatchnormPlanBuilder.cpp:371-374`, `BatchnormPlanBuilder.cpp:550-554`). For the
  three with no pack, "write the symbols" is not a fallback, it is the job.

Adding a pack to this engine is ordinary work with a rebuild in it. It is not a reason
to look for something to attach to.

## The hipRTC drop-in path, in its place

`kind: hiprtc_file` — descriptors plus a source bundle copied into an installed tree and
compiled at `prepare()` — is real and proved on device. It is **the reuse branch with a
rebuild avoided**: it adds no native symbol, so it can only ever serve a pack whose
symbols are already installed *and* whose handler routes `kernel_source.kind` —
Pointwise (`PointwiseNative.cpp:432-433`) and BatchnormInference
(`BatchnormInferenceNative.cpp:632-633`) do, ConvFwd (`ConvNative.cpp:501-502`) does
not. Use it to iterate on variants of a pack you already shipped. Do not choose it
because it looks cheaper than writing symbols; it cannot produce deliverable 2, and a
kernel that needs deliverable 2 cannot be dropped in at all. The page that owns that
case is [hiprtc-mining.md](../hipdnn-ingestor-engine/hiprtc-mining.md).

## Position in the flow

`graph → kernel → integrate → optimize`. This skill owns the **third** arrow. The
seam with the second is one handover: this skill consumes
[hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md)'s four handover facts —
the entry-point signature, the bundle's file set, the macros the source requires bound
with their legal values, and the launch geometry and workspace the kernel assumes — and
owns everything after them. The fourth arrow is measure-driven tuning and is not in
scope here.

Because the handler is **yours**, an "unbound" ABI from the upstream skill is the normal
answer, not a problem to be resolved before integration: you write the `launch()` that
the kernel's argument list dictates. [RUNBOOK.md](RUNBOOK.md) §The seam states which
side of that boundary each decision falls on.

## Entry contract

Record before writing anything, and treat a missing item as a blocked gate rather than a
default:

| Fact | Why it blocks |
|---|---|
| The kernel source and its proved numerics — device, shapes, tolerance and provenance | Integrating an unproved kernel debugs two things at once, and the shared suite cannot tell a wrong kernel from a wrong descriptor |
| The graph it serves, in the form [graph-contract.md](../hipdnn-ingestor-engine/graph-contract.md) describes | Your `graph_match` is written against node topology and UID edges, not against a node-type name |
| Target architecture(s), and whether one is reachable through a scheduler | Decides what the device gate can actually observe; host loading proves nothing about dispatch |
| The engine name and its UED identity | The generator derives the native symbol namespace from it, and the shared suite pins runs by it |
| **Whether any existing pack is genuinely yours to extend — default no** | Reuse is the narrow case; answering yes without naming the pack and the reason is how a real kernel ends up on a scaffold |

Missing source, an unrepresentable graph or no capable independent reference blocks the
corresponding gate. Scope changes need explicit approval. Keep scheduling and experiment
evidence outside product source per workspace policy.

## Completion and handoff

Completion requires **all four** deliverables, each with its own evidence:

- **Descriptors:** the emitted set, validated against the installed tree by
  `hipdnn_validate_descriptors`, with the engine name and uuids stated.
- **Native hooks:** each of `graph_match`, the matchers you declared, `workspaceBytes`,
  `IKernelDispatchHandler` and any score implemented with no reachable placeholder,
  registered in the pack's `SymbolScope<Handle>` and present in the `IngestorPacks.cpp`
  table — both, per [native-pack.md](../hipdnn-ingestor-engine/native-pack.md).
- **Registration:** the `add_external_integration_test_target` entry and its TOML, shown
  as a real `ctest -N` listing rather than as a CMake diff.
- **Tests and graphs:** the bundle case ids you added and the shared suite run that
  executed them, with **passed counts and case names**.

State what each observation proves and does not prove: device, architecture, dtype,
shape envelope, and every case the pack declined. A green run under `--test-engine` is
compatible with near-zero coverage, because an unservable case skips — the conv scaffold
passes 6 of 1218 and exits 0. Generated files, proposed commands and queued jobs are
none of the four deliverables.
