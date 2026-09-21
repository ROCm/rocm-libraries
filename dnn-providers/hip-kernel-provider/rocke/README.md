# rocKE

rocKE is a kernel authoring stack for AMDGPU. You describe a kernel as a Python
spec, a builder turns that spec into a typed SSA `KernelDef`, and a lowering
engine emits AMDGPU LLVM IR that `libamd_comgr` compiles to HSACO in-process and
HIP launches.

```
Spec dataclass -> build_*() -> KernelDef -> lower -> .ll -> comgr -> HSACO -> launch
```

The mental-model shift is that Python does not generate a HIP string: it builds a
typed SSA object, and a lowering engine turns that object into IR
([mental_model.md](platform/dsl_docs/architecture/mental_model.md)).

## Objectives

**Agentic kernel authoring with a very short turnaround.** The whole path from
spec to launched kernel runs in one process — no external compiler driver, no
template instantiation step — so authoring, lowering, inspecting the IR and ISA,
verifying numerics, and re-measuring form one tight loop. That makes it cheap to
sweep many spec variants and cheap for an agent to drive: the IR is small and
inspectable, so a bad variant is diagnosed from artifacts instead of guessed at.
The process is written down for machines as much as for people —
[AGENTS.md](AGENTS.md) owns the invariants, [KERNEL_AUTHORING.md](KERNEL_AUTHORING.md)
owns the definition of done and the one command that proves it, and
[platform/dsl_docs/](platform/dsl_docs/README.md) is the field manual — so an
agent follows the same checklist an engineer does.

**CK-Tile-style tile abstraction without C++ template metaprogramming.** rocKE
keeps the CK Tile concepts — tensor descriptors and views, coordinate-transform
DAGs, tile windows, tile distributions, MFMA/WMMA atoms, cshuffle epilogues,
software pipelines — but expresses them as concrete Python data objects and
first-class SSA ops instead of C++ templates. Transform algebra reads as
algebra rather than as nested template types; non-bijective mappings such as
convolution padding and paged-attention page tables are expressible directly
instead of being bent into layout permutations; and one more autotuning variant
is one more spec, not one more template instantiation. The abstraction does not
hide the GPU programming model: explicit LDS allocation and layout, raw buffer
resource descriptors, async DRAM-to-LDS copies, `s_waitcnt` / barrier / priority
scheduling, and pointer aliasing and alignment metadata all stay under the
author's control.

**No Python at runtime.** The same lowering exists as a C++20 engine
(`librocke_core.a`), so kernels authored in Python can be built, lowered, and
served with no Python present — which is what the hipDNN provider ships. Both
engines must emit **byte-identical** LLVM IR for every kernel family; that is
rocKE's #1 invariant, and mirroring an emission change in both engines is part
of every change that touches emitted IR
([engines_and_switching.md](platform/dsl_docs/architecture/engines_and_switching.md)).

## Layout

```
rocke/
├── platform/   # authoring SDK + engine (installable `rocke`) — NON-attention kernels
│   ├── python/rocke/   core, helpers, instances, runtime, dispatch, analysis, benchmark
│   ├── cpp/            C++20 engine (librocke_core.a) + pybind bindings
│   └── dsl_docs/       the field manual
└── library/    # the SDPA/MHA product — build-time-only Python (NOT installed, no wheel)
    └── kernels/ builders/ dispatch/ benchmarks/
```

One-way dependency: **`library → platform`** only, so platform stays
standalone-installable. `library/` is a build/verify harness for attention
kernels and is never packaged into the `rocke` wheel; the provider plugin is
emitted by the provider's own `src/`.

## Start here

| I want to… | Go to |
|---|---|
| Understand the rules / invariants (agents + contributors) | [AGENTS.md](AGENTS.md) — the canonical entry point |
| Build & run | [BUILDING.md](BUILDING.md) |
| Test | [TESTING.md](TESTING.md) |
| Know when a change is "done" (DoD, process, commit/branch conventions) | [KERNEL_AUTHORING.md](KERNEL_AUTHORING.md) |
| Learn the engine deeply (IR, lowering, primitives, instances) | [platform/dsl_docs/README.md](platform/dsl_docs/README.md) |
| Author a new kernel | [platform/dsl_docs/architecture/authoring_model.md](platform/dsl_docs/architecture/authoring_model.md) |
| Optimize a kernel | [platform/dsl_docs/optimization/optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md) |
