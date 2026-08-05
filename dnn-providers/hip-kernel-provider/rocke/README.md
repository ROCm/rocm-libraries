# rocke

A dual-engine GPU kernel stack that emits **byte-identical AMDGPU LLVM IR** from
two implementations: a **Python authoring frontend** (`import rocke`) and a
**C++20 engine** (`librocke_core.a`). Author kernels in Python (a typed SSA
`KernelDef`), lower to LLVM IR, compile to HSACO in-process via `libamd_comgr`,
and launch through HIP — or serve the same kernels with no Python at runtime.

```
Spec dataclass -> build_*() -> KernelDef -> lower -> .ll -> comgr -> HSACO -> launch
```

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

One-way dependency: **`library → platform`** only; platform stays
standalone-installable. `library/` is a build/verify harness for attention
kernels — never packaged into the `rocke` wheel; the provider plugin is emitted by
the provider's own `src/`.

## Start here

| I want to… | Go to |
|---|---|
| Understand the rules / invariants (agents + contributors) | [AGENTS.md](AGENTS.md) — the canonical entry point |
| Build & run | [BUILDING.md](BUILDING.md) |
| Test | [TESTING.md](TESTING.md) |
| Know when a change is "done" | [DEFINITION_OF_DONE.md](DEFINITION_OF_DONE.md) |
| Contribute (flow, branch, commits) | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Learn the engine deeply (IR, lowering, primitives, instances) | [platform/dsl_docs/README.md](platform/dsl_docs/README.md) |
| Author a new kernel | [platform/dsl_docs/architecture/authoring_model.md](platform/dsl_docs/architecture/authoring_model.md) |
| Optimize a kernel | [platform/dsl_docs/optimization/optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md) |

## The #1 invariant

The Python and C++ engines **must emit the same LLVM-IR bytes** for every kernel
family. Mirror every emission change in both engines and keep the byte-identity
gate GREEN — details in [AGENTS.md](AGENTS.md).
