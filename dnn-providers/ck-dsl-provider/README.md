# ck-dsl-provider

hipDNN engine plugin that exposes kernels produced by the Composable
Kernel Python DSL (`ck_dsl`).

## What it does

The provider embeds a **MicroPython** interpreter inside the plugin
`.so` — it has **no dependency on a system Python installation or on
pybind11**. The `ck_dsl` / `ck_dsl_provider` Python ships with the
plugin: frozen into the `.so` by default, or as `.py` / `.mpy` files
beside it (see [Distribution modes](#distribution-modes)). A thin Python
compile service (`ck_dsl_provider/compile_service.py`) is invoked from
C++ through a `CompileServiceBridge`, which dispatches on `op_kind`,
builds a `ck_dsl` dataclass from a typed payload, calls
`ck_dsl.helpers.compile.compile_kernel`, and returns HSACO bytes plus
the launch metadata the C++ side needs. The LLVM-IR → HSACO step runs
through `comgr` (libamd_comgr), exposed to the interpreter as a native
`comgr` module so `ck_dsl` keeps its existing compile flow without any
ctypes / FFI.

A graph-key JIT cache (`JitCache`) memoises the compile result per
process. Subsequent calls with the same logical shape return the
cached `HipModule` rather than re-running the DSL.

The provider currently ships one engine: `CkDslConvImplicitGemmEngine`,
which serves forward 2D implicit-GEMM convolution at FP16 / NHWC. The
adapter accepts the hipDNN `ConvolutionFwdAttributes` graph shape and
rejects asymmetric padding, true-convolution mode, non-FP16 dtypes,
and 3D conv.

## Architecture

### At a glance

The plugin embeds a MicroPython interpreter and turns a hipDNN conv graph
into a launchable GPU kernel by calling the `ck_dsl` Python DSL. The
expensive compile happens **once per (shape, arch)**; a process-wide
`JitCache` short-circuits every repeat request to a ready `HipModule`.

```mermaid
flowchart LR
    SDK["hipDNN SDK"]

    subgraph plugin["ck_dsl_provider plugin (.so)"]
        direction TB
        Front["Engine + Adapter<br/>validate graph,<br/>build spec, detect arch"]
        Cache["JitCache<br/>key = shape + arch"]
        Py["Embedded MicroPython<br/>ck_dsl → HSACO<br/>(native comgr)"]
        Mod["HipModule"]
    end

    SDK <-->|isApplicable / build / execute| Front
    Front -->|look up| Cache
    Cache -.->|miss: compile once| Py
    Py -.->|HSACO + launch meta| Mod
    Cache --> Mod
    Mod -->|launch| GPU["GPU"]
```

The request lifecycle, abstracted to the one decision that matters —
cache hit vs. miss:

```mermaid
flowchart TB
    Start["conv request<br/>(graph + HIP stream)"]
    Arch["detect arch from stream<br/>→ gfx token"]
    Spec["build spec +<br/>arch-validate (DSL)"]
    Key["key = shape + arch"]
    Q{"in JitCache?"}
    Compile["embedded MicroPython:<br/>ck_dsl compiles HSACO<br/>(once per shape + arch)"]
    Mod["HipModule"]
    Exec["execute → launch on GPU"]

    Start --> Arch --> Spec --> Key --> Q
    Q -->|miss| Compile --> Mod
    Q -->|hit| Mod
    Mod --> Exec
```

The two diagrams below expand these into the full class-level component
graph and the precise call sequence.

<details>
<summary><b>Detailed component view</b></summary>

The plugin is organised into layers. The container is created once per
handle generation and owns the long-lived collaborators (the embedded
interpreter, the Python compile bridge, and the process-wide JIT
cache); the engine, adapter, and plan are per-request. The target
`arch` is detected from the request's HIP stream and threaded — as an
orthogonal compile target, not a spec field — into codegen-config
selection, the cache key, and the compile/applicability calls.

```mermaid
flowchart TB
    subgraph host["hipDNN host"]
        SDK["hipDNN SDK<br/>(EnginePluginImpl)"]
    end

    subgraph plugin["ck_dsl_provider_plugin.so"]
        direction TB

        subgraph entry["Entry / lifetime"]
            Container["CkDslContainer<br/>(EngineManager + bridge + cache)"]
            Handle["CkDslHandle<br/>(HIP stream)"]
            Context["CkDslContext<br/>(holds Plan)"]
        end

        subgraph engine["Engine layer (per request)"]
            Engine["CkDslConvImplicitGemmEngine"]
            Builder["ConvImplicitGemmPlanBuilder"]
            Plan["ConvImplicitGemmPlan<br/>execute() → launch"]
        end

        subgraph adapt["Adapter layer"]
            Adapter["ConvImplicitGemmAdapter<br/>validate + buildSpec()"]
            Spec["ConvImplicitGemmSpec"]
            ArchCfg["applyArchCodegenConfig<br/>(spec, arch)"]
            Payload["convImplicitGemmSpecToPayload()"]
            Sig["GraphSignature<br/>computeForSpec(opKind, spec, arch)<br/>→ SignatureHash key"]
        end

        subgraph runtime["Runtime layer"]
            Arch["DeviceArch<br/>detectDeviceArch(stream)<br/>→ gfx token (memoized)"]
            Cache["JitCache<br/>key → shared_ptr&lt;HipModule&gt;"]
            Module["HipModule<br/>(hipModule_t + hipFunction_t)"]
            Artifact["KernelArtifact<br/>(HSACO + launch metadata)"]
            Abi["LaunchAbi<br/>pack() arg buffer"]
        end

        subgraph pybound["Python boundary"]
            Interp["EmbeddedInterpreter<br/>(embedded MicroPython,<br/>interpreter lock — no GIL)"]
            Bridge["CompileServiceBridge<br/>isApplicable / compile<br/>(opKind, payload, arch)"]
            Comgr["native comgr module<br/>(libamd_comgr): IR → HSACO"]
        end
    end

    subgraph pysrc["ck_dsl modules<br/>(frozen in the .so, or on-disk bundle)"]
        Service["ck_dsl_provider.compile_service"]
        DSL["ck_dsl<br/>(build + compile_kernel)"]
    end

    SDK -->|create| Container
    SDK -->|isApplicable / init| Engine
    SDK -->|execute| Plan

    Container --> Engine
    Container --> Bridge
    Container --> Cache
    Engine --> Builder
    Builder -->|detect| Arch
    Builder --> Adapter
    Adapter --> Spec --> ArchCfg
    Arch -.arch.-> ArchCfg
    ArchCfg --> Payload
    Builder --> Sig
    Arch -.arch.-> Sig
    Builder -->|getOrLoad key, loader| Cache
    Cache -->|miss| Bridge
    Arch -.arch.-> Bridge
    Payload -.payload dict.-> Bridge
    Bridge --> Interp
    Bridge -->|import + call| Service
    Service --> DSL
    DSL -.lower → IR.-> Comgr
    Comgr -.HSACO.-> DSL
    DSL -.HSACO + metadata.-> Bridge
    Bridge --> Artifact --> Module
    Cache --> Module
    Builder --> Plan
    Plan --> Module
    Plan -.uses.-> Abi
    Plan --> Context
```

</details>

<details>
<summary><b>Detailed end-to-end sequence</b></summary>

The compile step (heavy, Python/DSL) runs once per logical shape *and*
target arch; the `JitCache` short-circuits every subsequent request
with the same `GraphSignature` (which folds in the arch) to a cached
`HipModule`.

```mermaid
sequenceDiagram
    autonumber
    participant SDK as hipDNN SDK
    participant Eng as Engine
    participant Bld as PlanBuilder
    participant Adp as Adapter
    participant Arch as DeviceArch
    participant Cache as JitCache
    participant Br as CompileServiceBridge
    participant Py as compile_service.py / ck_dsl
    participant Mod as HipModule
    participant Plan as ConvImplicitGemmPlan

    SDK->>Eng: isApplicable(handle, graph)
    Eng->>Bld: isApplicable(handle, graph)
    Bld->>Adp: buildSpec(convAttr, tensors)
    Adp-->>Bld: Spec (or reject)
    Bld->>Arch: detectDeviceArch(stream)
    Arch-->>Bld: gfx token (nullopt → decline)
    Bld->>Adp: applyArchCodegenConfig(spec, arch)
    Bld->>Br: isApplicable(opKind, payload, arch)  [lock]
    Br-->>Bld: applicable? (DSL is_valid_spec)

    SDK->>Eng: initializeExecutionContext(graph)
    Eng->>Bld: buildPlan(handle, graph, context)
    Bld->>Adp: buildSpec(...)
    Bld->>Arch: detectDeviceArch(stream)
    Arch-->>Bld: gfx token
    Bld->>Adp: applyArchCodegenConfig(spec, arch)
    Bld->>Bld: GraphSignature::computeForSpec(opKind, spec, arch) → key
    Bld->>Cache: getOrLoad(key, loader)

    alt cache miss
        Cache->>Br: compile(opKind, payload, arch)  [lock]
        Br->>Py: compile(op_kind, payload, arch)
        Py-->>Br: dict{hsaco, kernel_name, grid, block, arg_schema, ...}
        Br-->>Cache: KernelArtifact
        Cache->>Mod: HipModule(artifact)<br/>hipModuleLoadData + GetFunction
    else cache hit
        Cache-->>Bld: cached HipModule
    end

    Cache-->>Bld: shared_ptr of HipModule
    Bld->>Plan: new Plan(module, uids, byte sizes)
    Bld-->>SDK: plan stored in context

    SDK->>Plan: execute(handle, deviceBuffers)
    Plan->>Plan: resolve x/w/y pointers, pack 36-byte args
    Plan->>Mod: launch(args, grid, block, stream)
    Mod-->>SDK: hipModuleLaunchKernel
```

</details>

## Trust boundary

The Python that runs inside `compile()` is part of the plugin's trust
boundary — whoever controls it controls the HSACO bytes that reach
`hipModuleLoadData`.

In the default **frozen** mode the `ck_dsl` / `ck_dsl_provider` modules
are compiled into the `.so` as frozen bytecode. There is no external
Python source, no `sys.path`, and no filesystem import, so the trust
boundary is exactly the `.so` itself — the same surface as any other
shared library.

In the on-disk **py** / **mpy** modes the modules ship as files in a
bundle directory beside the plugin (`CKDSL_BUNDLE_DIR`, added to
`sys.path` at startup). That directory must have the same permissions as
the `.so`: world-readable, not user-writable. Anyone able to write to it
can substitute the code that runs inside `compile()`.

Because the interpreter is MicroPython, the CPython environment-injection
surface does not exist: there is no `PYTHONPATH` / `PYTHONHOME` /
`PYTHONSTARTUP` / `PYTHONUSERBASE` influence on import resolution, and
the interpreter has its own process-owned GC heap (no shared CPython
runtime to inherit from a sibling embedder).

## Distribution modes

How `ck_dsl` ships into the plugin is selected at configure time with
`-DCKDSL_MICROPYTHON_MODE=<mode>`:

| Mode | What ships | Trade-off |
|------|------------|-----------|
| `frozen` (default) | `ck_dsl` + shims frozen into the `.so` as bytecode | Self-contained, smallest surface; editing a `.py` means rebuilding + relinking the static lib. |
| `py` | `.py` files in a bundle dir beside the plugin | Fast iteration — edit the on-disk `.py` and rerun, no relink. |
| `mpy` | pre-compiled `.mpy` bytecode beside the plugin | Like `py`, but smaller / faster to load. |

All three modes are built from the same source by the embed build
(`micropython/build_embed.py`, wired via `cmake/CkDslMicroPython.cmake`).
That driver is a **cross-platform Python script** — it needs only Python
and a C compiler (no `bash`, no GNU `make`), so it builds on Windows as
well as Linux/macOS. It drives MicroPython's own build tools directly to
generate the qstr/genhdr headers and (frozen mode) `frozen_content.c`.
`mpy-cross` (needed by `frozen`/`mpy`, not `py`) is **built from the same
pinned MicroPython source, make-free, the same way on every platform** —
reusing the driver's genhdr machinery. This is deliberate: no released
`mpy-cross` pip wheel matches the pin (the `ck_dsl` codegen emits braces
via f-strings whose parsing only exists in MicroPython newer than any
release — which is why the pin is a dev commit), so a source build is the
only portable, format- and compiler-compatible option. The on-disk modes
load through the embed port's filesystem importer and are installed beside
the plugin (resolved at runtime via `dladdr`). `CKDSL_MICROPYTHON_DIR` can
point the build at an existing MicroPython checkout instead of cloning the
pinned commit; `MPY_CROSS` / `CKDSL_MPY_CROSS_BIN` can supply a prebuilt
`mpy-cross` to skip the source build.

The `ck_dsl` source is kept directly MicroPython-compatible (it is no
longer rewritten at build time): every `@dataclass` field is explicit
`= field(...)`, displays avoid PEP-448 star-unpacking, and `os.getenv` /
`object.__setattr__` are used in place of the constructs MicroPython
lacks. The compatibility lint below guards against regressions.

## Tests

- `ninja ck-dsl-provider-unit-check` — host-only + GPU-gated unit
  suite covering the interpreter, bridge, adapter, payload
  round-trip, signature, cache, plan-builder, launch ABI, and
  perf-measurement helpers. Also runs the **MicroPython-compatibility
  lint** (`micropython/check_compat.py`): it fails if a construct the
  embed can't handle (a bare `@dataclass` field, PEP-448 display,
  `os.environ`, `match`/`async`) creeps into the `ck_dsl` sources, and —
  when `mpy-cross` is available — additionally compiles every module
  with the real MicroPython compiler. Runnable standalone too:
  `python3 micropython/check_compat.py [--mpy-cross PATH] <dir>...`.
- `ninja ck-dsl-provider-integration-check` — end-to-end conv-fwd
  across a set of shapes on whatever DSL-supported device is present
  (gfx942 / gfx950 / gfx1151), comparing against
  `CpuFpReferenceConvolution::fprop` (via the `computeTensorDiff` test
  helper) and logging kernel time + TFLOPS via the `PerfMeasurement`
  helper. The production plan builder detects the device arch and the
  adapter's `applyArchCodegenConfig` selects a valid per-arch codegen
  config, so the same graph runs on each supported arch.

GPU-gated tests skip cleanly on hosts without a HIP-visible device or
on an arch outside the DSL-supported set.

## Design plan

The provider's architecture, milestone scope, and resolved design
questions are recorded in the implementation plan:

- [CK DSL hipDNN Provider — Plan](../../projects/composablekernel/python/ck_dsl/dsl_docs/hipdnn_provider/plan.md)

This document is the source of truth for the Milestone 1 goal and
non-goals, the runtime embedded-Python architecture, and the rationale
behind decisions such as the embedded interpreter and provider-local
compile service that are summarised above. (Note: that plan predates the
move to MicroPython and still describes the original embedded-CPython /
pybind11 design; this README reflects the current implementation.)
