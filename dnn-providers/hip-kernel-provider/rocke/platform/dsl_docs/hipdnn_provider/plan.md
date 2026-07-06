# rocke hipDNN Provider — Plan v0.9

**Status:** All open questions resolved; plan is implementation-ready
pending a final review pass. v0.9 renames the single M1 engine to
`RockeConvImplicitGemmEngine` to make the eventual per-op engine
structure obvious from M1. v0 amended 2026-05-21 to make **runtime
Python execution** the primary architecture. v0.2 split the embedding
mechanism (interpreter location) from the binding library (pybind11 vs
raw C API) and clarified the plugin C-export pattern. v0.3 moved per-op
adapter logic from Python into C++ so it survives an eventual migration
to a C++ DSL/codegen pipeline. v0.4 records the user's confirmation of
Q1a (embedded interpreter), Q1b (pybind11), and Q2 (provider-local
Python service). v0.5 promoted the first kernel from elementwise to
**implicit-GEMM convolution**, with hipEvent-based perf measurement and
CPU-reference verification both inside M1. v0.6 removed the NHWC↔NCHW
host-side transpose after verifying the CPU reference is layout-
agnostic via strides. v0.7 records resolution of Q4 (provider location),
Q5 (no constraints; branch off `users/<user>/rocke-prototype`), Q6
(install `rocke` for runtime discovery), Q7 (sequential Python critical
path with the C++ adapter and PerfMeasurement as parallel sub-streams),
Q8 (cold-cache perf acceptable), Q9 (log absolute TFLOPS in M1).
**Author:** orchestrator session 2026-05-21.
**Scope:** Stand up a hipDNN engine plugin that exposes kernels produced by
the rocke, proven by one end-to-end integration test that builds a hipDNN
graph in C++, has the provider **generate the DSL spec from that graph at
runtime**, JIT-compile it, and launch it.

Each section's open questions are tagged `[Q]` so the user can hand back
answers without re-explaining the whole plan.

---

## 1. Goal and non-goals

### Goal (Milestone 1)

A new provider `dnn-providers/rocke-provider/` produces a `.so` that
hipDNN's backend can load. Its `IntegrationGpuRockeConvFp16` test:

1. Constructs a single-op hipDNN `Graph` (forward 2D convolution) in C++
   using the frontend API.
2. The rocke engine reports applicable on that graph.
3. The engine's plan builder calls a C++ adapter that walks the `IGraph`,
   builds a typed `ImplicitGemmConvSpec` payload, derives a cache
   signature, and on a miss invokes the embedded-Python compile service
   to lower the spec to IR and produce HSACO. The HSACO is cached for
   the lifetime of the plugin handle.
4. Plan `execute()` launches the kernel via `hipModuleLaunchKernel`
   against DevPtrs supplied by the runtime, **timed with hipEvents**
   over a warmup-and-iterate loop.
5. Output is **validated against `CpuFpReferenceConvolution::fprop`**
   from `test_sdk` within tolerance.
6. The test logs both **achieved kernel time and TFLOPS** for the
   exercised shape. No perf-target assertion in M1 — measurement only.

The integration test exercises the JIT path, not a precompiled blob, so
the runtime pipeline is proven end-to-end before any fast-path shortcuts
land.

### Branch model

Work happens on a new feature branch (proposed name:
`users/<user>/rocke-provider`) cut from
**`users/<user>/rocke-prototype`** — *not* from `develop`. Develop
does not yet carry the rocke prototype itself, so a branch off develop
would be missing the dependency. Per-step Implementor WIP branches (for
the parallel sub-streams in §6.5) come off the feature branch in the
usual workspace pattern.

### Non-goals (for Milestone 1)

- Multiple engines or multiple operations. One op (conv fwd), one
  engine, one plan.
- Autotuning, kernel selection heuristics, **perf-target assertions** —
  measurement yes, gating no. Settling on a perf-comparison baseline
  (vs MIOpen, vs native CK_Tile, both) is Q9 and deferred to M2.
- On-disk HSACO cache (in-memory only for M1).
- Build-time pre-baked kernels (a useful M2+ optimisation, but skipping
  for M1 forces us to prove the JIT path on the critical path).
- ROCm-version matrix coverage. Build and prove on the CI baseline.
- Touching any other provider. MIOpen, hip-kernel, fusilli are references
  only.
- Surfacing all 36 conv spec fields through the adapter. Only ~15 are
  graph-derived; the rest stay as defaults inside the C++ `*Spec`
  struct (copied from the existing bake-off example) until M2+ when
  autotuning becomes relevant.

---

## 2. Key findings from research (load-bearing context)

These were established by parallel research agents. If any are wrong, the
plan below shifts.

### 2.1 rocke surface

- The DSL compile entry is `rocke.helpers.compile.
  compile_kernel(kernel) → KernelArtifact` which contains **portable
  HSACO bytes** (loadable via `hipModuleLoadData`, no Python state
  required at launch time). (This finding reflects the M1 state, when
  the DSL was Python-only. There is now also a peer C++ engine
  (`Cpp/`) that lowers byte-identical LLVM-IR; the provider has
  since shipped with three execution modes — prebuilt HSACO, JIT from
  shipped `.ll` via comgr, and a Python-free C-JIT path — so the
  "runtime JIT requires a Python interpreter" implication below is
  superseded for the C-JIT mode. See the provider README.)
- Per-op spec builders live in `rocke/instances/common/` (e.g.
  `elementwise.py`, `gemm_universal.py`, `conv_implicit_gemm.py`), with
  arch-specialized overrides under `rocke/instances/<arch>/`. Each
  takes a Python dataclass `Spec` and returns a `KernelDef` IR object.
- An existing C++ launcher at `projects/composablekernel/example/ck_tile/
  dsl/common/launcher.cpp` already loads `gen.py` output and runs it —
  not a provider, but the load-and-launch code is proven.
- Compile latency for one kernel is "typically <150 ms wall" per
  `examples/bake_off_implicit_gemm.py:16–17`, which is the floor for
  JIT overhead on a cold cache.

> **Implication for the new design.** Since `compile_kernel` is Python,
> **runtime JIT requires the provider to invoke a Python interpreter**.
> Section 3 picks the embedding mechanism.

### 2.2 hipDNN Plugin SDK

- Provider exports C symbols (`hipdnnPluginGet*`, `hipdnnEnginePlugin*`)
  from `PluginApi.h` / `EnginePluginApi.h`, and internally implements
  `IEngine`, `IPlanBuilder`, `IPlan` from
  `plugin_sdk/include/hipdnn_plugin_sdk/interfaces/`.
- Graphs arrive as FlatBuffers (`schemas/graph.fbs`); inspect via
  `IGraph` / `GraphWrapper` / `NodeWrapper`.
- Device buffers reach `execute()` as a `(uid, void*)` array.
- Loader uses `dlopen` with `RTLD_NOW | RTLD_LOCAL` — each plugin's
  symbols are isolated.
- The SDK ships `PluginGraphTestUtils.hpp` and the reusable
  `IntegrationGraphVerificationHarness` pattern used by both
  miopen-provider and hip-kernel-provider. We will reuse it.
- Documentation: `projects/hipdnn/docs/PluginDevelopment.md` and the
  reference plugin at `projects/hipdnn/samples/example_engine_plugin/`.

### 2.3 Existing providers as templates

| Provider                | Kernel sourcing                                | Fit as our template                                                                                  |
|-------------------------|------------------------------------------------|------------------------------------------------------------------------------------------------------|
| `miopen-provider`       | Runtime library calls into MIOpen              | **Best for plumbing**: CMake, plugin C exports, engine/plan-builder/plan split, integration harness. |
| `hip-kernel-provider`   | C++ source strings → HIPRTC → `hipModuleLoadData` | **Best for the kernel-launch path**: already does `hipModuleLoadData` + `hipModuleLaunchKernel`, just from HIPRTC instead of from a DSL artifact. |
| `fusilli-provider`      | IREE compiler runtime (a different JIT pipeline) | **Most architecturally similar**: also wraps a "transform a graph at runtime into GPU code" pipeline. Worth re-reading for lessons on JIT-cache layout and graph-keyed compilation, even though we will not adopt its tech stack. |

The plan therefore borrows skeleton from `miopen-provider`, the
HipProgram/HipKernel launch path from `hip-kernel-provider`, and looks to
`fusilli-provider` for prior art on graph-keyed JIT caching inside a
hipDNN engine plugin.

---

## 3. Architecture: runtime DSL pipeline and Python integration

This is the largest design choice. Two axes: **what the runtime pipeline
looks like** (§3.1) and **how C++ calls into Python** (§3.2).

### 3.1 Runtime pipeline

```
hipDNN graph (FlatBuffer)
        │
        ▼
  GraphAdapter (C++)          ← walks IGraph, extracts op kind, dtypes,
        │                       shapes, strides, layout
        ▼
  cache lookup ───────► hit ──► HipModule (already loaded)
        │ miss                    │
        ▼                         │
  SpecBuilder (C++, per-op)   ← picks block_size, vec, knobs from the
        │                       graph; emits a typed Payload struct
        ▼                         │
  Payload → dict (pybind11)       │
        │                         │
        ▼                         │
  rocke_provider.compile_       │
     payload(payload_dict)        │   ─── thin Python glue (~30 LoC):
        │                         │       unpacks dict, instantiates
        ▼                         │       the matching rocke Spec
  rocke.instances.<op>(spec)     │       dataclass, calls compile
        │                         │
        ▼                         │
  rocke.helpers.compile.         │
     compile_kernel() → HSACO ───►│
                                  ▼
                          hipModuleLoadData
                                  │
                                  ▼
                          cache insert
                                  │
                                  ▼
                         plan ready → execute()
                                  │
                                  ▼
                         hipModuleLaunchKernel
```

The interesting per-op logic — graph attribute reading, spec-knob
selection, validation — lives in C++. Python is reduced to a thin
"compile this payload and give me bytes" service. When/if the DSL itself
moves to C++, the Python service is deleted; the C++ adapter is unchanged.

- The provider owns a Python sub-environment scoped to the plugin handle
  (or process, depending on §3.2). All in-process JIT calls funnel through
  one `rocke_provider.spec_builder` module we author inside the provider.
- The cache lives in the plugin handle, keyed by a deterministic
  signature derived from `(op_kind, dtypes, shape tuple, stride tuple,
  layout tag, DSL version)`.
- M1 caches in memory only. M2+ adds a disk cache (default location
  `$XDG_CACHE_HOME/rocke-provider/<hash>.hsaco`) so warm starts skip the
  compile altogether.
- Eventually a *build-time pre-bake* fast path can pre-populate the cache
  with kernels for known-hot shapes, but that is M2+. M1 always JITs.

### 3.2 Python integration mechanism

Three viable choices for letting C++ invoke the DSL:

#### Option E — embedded CPython (recommended)

- Link the provider .so against `libpython3.x.so`. Initialise the
  interpreter once per process on first `hipdnnEnginePluginCreate`.
- **Pros:** lowest latency (no fork, no IPC); easy to debug.
- **Cons:** the provider .so depends on libpython; need to discover the
  `rocke` module at runtime (set `PYTHONPATH` or install to
  site-packages); one shared interpreter per process means *all*
  in-proc Python users (other plugins, host app) share state.

Sub-decision: **how we call into Python from C++** — raw CPython C API
vs. pybind11. The interpreter init itself is trivial either way
(`Py_Initialize` is ~one line; `pybind11::scoped_interpreter` is the
same plus RAII). The real cost lives in the per-call boundary code:
acquiring the GIL, importing a module, packing a typed payload, calling
the function, catching Python exceptions, extracting `bytes` results,
and managing refcounts on every `PyObject*` that crosses our hands.

| Boundary op                       | Raw C API                                                                  | pybind11                                          |
|-----------------------------------|----------------------------------------------------------------------------|---------------------------------------------------|
| GIL acquire/release               | `PyGILState_Ensure` / `Release` (manual pairing)                           | `py::gil_scoped_acquire` (RAII)                   |
| Import a module                   | `PyImport_ImportModule` + DECREF                                           | `py::module_::import(...)`                        |
| Build a dict/dataclass payload    | `PyDict_New` + `PyLong_FromLong` + `PyDict_SetItemString` + DECREFs        | `py::dict("k"_a=v, ...)`                          |
| Call a function                   | `PyObject_GetAttrString` + tuple-pack + `PyObject_CallObject` + DECREFs    | `m.attr("f")(payload)`                            |
| Catch a Python exception          | `PyErr_Occurred` + `PyErr_Fetch` + `PyErr_NormalizeException`              | `catch (py::error_already_set&)`                  |
| Extract `bytes` (HSACO) result    | `PyBytes_AsStringAndSize` + lifetime care                                  | `py::bytes` → `std::string_view`                  |
| Every `PyObject*` we touch        | Manual `Py_INCREF` / `Py_DECREF`                                           | `py::object` is RAII                              |

Net: raw C API for the M1 boundary is ~50–100 lines of careful
refcount/exception bookkeeping per call site; pybind11 is ~10–20. The
risk of the raw version isn't lines, it's that refcount and exception-
state bugs typically don't fail loudly — they corrupt interpreter state
silently and surface as crashes much later.

**Plan of record: pybind11** for the boundary. Header-only, one CMake
`find_package`, no extra runtime dependency beyond libpython (which we
already have). For M1 alone the raw C API is defensible; the case for
pybind11 strengthens as the boundary grows in M2+.

If pybind11 is unavailable in our build environment or otherwise
objectionable, raw C API is a viable substitute — the rest of the
architecture in this section does not change.

#### Option S — subprocess per JIT compile (fallback)

- For each cache miss, spawn `python -m rocke_provider.compile_for_graph
  --signature=…` reading stdin / writing HSACO bytes to stdout (or to a
  named temp file).
- **Pros:** zero linkage against libpython; full isolation from other
  Python code in the process; trivial to swap out the Python toolchain.
- **Cons:** per-call overhead of ~100–300 ms (interpreter startup,
  module import) on top of the DSL compile itself. Caching hides this
  for steady-state workloads but the first call to each shape pays the
  bill, often visibly.

#### Option D — long-running Python daemon (deferred)

- One `python -m rocke_provider.daemon` subprocess per plugin handle,
  RPC via Unix socket / shared memory.
- Best steady-state perf with full isolation. Operationally
  heaviest — lifecycle, crash recovery, socket discovery, supervisor
  policy. Not for M1.

#### Decision

**Plan of record: Option E (pybind11 embedded interpreter).** Option S
as a same-day fallback if libpython linkage proves incompatible with
hipDNN's plugin loader (e.g. RTLD_LOCAL plus libpython's symbol
visibility quirks, or conflicts with another plugin that also embeds
Python). Option D revisited only if scaling needs demand it.

`[Q1]` Confirm Option E as the default, Option S as the fallback?

### 3.3 Where the spec adapter lives — and why it is C++

The graph-to-spec translation answers "what does this hipDNN graph node
mean in terms of a rocke spec dataclass". The instinctive answer is "do
it in Python so we can use the dataclasses directly," but on closer
inspection that argument is weak: the inputs to the translation (op kind,
dtypes, shapes, strides, attrs) all live in C++ on the `IGraph` side and
have to be marshaled across the boundary either way.

The plan instead puts the adapter logic in C++ and keeps Python as a thin
compile service. Three reasons:

1. **It survives an eventual C++-DSL migration.** If the DSL itself
   later moves to C++, today's Python "compile this payload" glue is
   deleted and the C++ adapter feeds a C++ compile pipeline directly.
   The per-op adapter logic — which is where the real complexity lives
   over time — never gets rewritten.
2. **Less marshaling across the Python boundary.** A typed C++ `Payload`
   crosses once per JIT compile, not once per graph attribute lookup.
3. **It forces us to start thinking now about what a C++ spec API
   should look like**, which is useful intellectual capital for the
   future DSL pivot.

The runtime shape:

```
src/adapters/                                    (C++, the durable surface)
    GraphAdapter.{hpp,cpp}                       # IGraph → AdapterContext
    elementwise/
        ElementwiseAdapter.{hpp,cpp}             # node → ElementwiseSpec
        ElementwiseSpec.hpp                      # mirrors the Python dataclass
    # gemm/, conv/ … added in later milestones

Python/rocke_provider/                          (Python, the throwaway glue)
    __init__.py
    compile_service.py                           # ~30 LoC entry point
```

Each C++ `*Spec` struct mirrors the corresponding `rocke.instances.<op>`
Python dataclass. The struct knows how to serialise itself to a plain
dict (`Spec::to_payload() → std::unordered_map<...>`). The Python
`compile_service.compile(op_kind, payload)` instantiates the matching
dataclass via `**payload` kwargs, calls `build_<op>(spec)` + `compile_
kernel()`, returns HSACO bytes + a small launch-ABI dict.

**Cost of mirroring.** The structs and dataclasses must stay in sync. For
M1 (elementwise) that is four fields. A unit test that does `Spec()
.to_payload()` → builds the Python dataclass → asserts field-set
equality is a cheap divergence canary; defer to M2 when we add a second
op.

`[Q2]` Comfortable with the adapter logic living in C++
(`src/adapters/<op>/`) and the Python side reduced to a ~30-line compile
service shipped from `dnn-providers/rocke-provider/python/
rocke_provider/`? Alternative: fold the Python service into `rocke`
under `rocke/integrations/hipdnn/`. My recommendation is provider-local
— keeps the DSL surface itself unaware of hipDNN.

### 3.4 Caching

- **In-memory cache (M1):** `std::unordered_map<SignatureHash,
  std::unique_ptr<HipModule>>` per plugin handle. Mutex-guarded.
- **Signature:** a deterministic hash over `(op_kind_string, dtype_tuple,
  shape_tuple, stride_tuple, layout_string, dsl_version_string)`.
  `dsl_version_string` comes from `rocke.__version__` (or a git SHA if
  the package doesn't expose one — to be confirmed during prep).
- **HipModule lifetime:** owned by the cache, never reloaded for the
  same signature. Destroyed on plugin handle destroy.
- **Negative caching:** if `compile_kernel` raises for a signature,
  record the failure and return "not applicable" fast next time rather
  than re-trying every call.

---

## 4. First kernel choice — implicit-GEMM convolution

Chosen because the milestone now scopes in performance measurement and
real CPU-reference verification, both of which are meaningful only on a
non-trivial kernel.

**Op:** forward 2D convolution via `build_implicit_gemm_conv`
(`dnn-providers/hip-kernel-provider/rocKE/Python/rocke/instances/common/conv_implicit_gemm.py`).

**Shape:** the bake-off shape from
`dnn-providers/hip-kernel-provider/rocKE/Python/rocke/examples/common/bake_off_implicit_gemm.py`
— `N=8, H=W=56, C=64, K=64, Y=X=3, stride=1, pad=1, dilation=1`, FP16,
NHWC. This is the smallest shape we know already compiles cleanly; the
example documents `248 TFLOPS per-launch / 280 TFLOPS graph 5×200` on
MI300X (`bake_off_implicit_gemm.py:70–71`).

**DSL compile cost:** documented as "typically <150 ms wall" for this
kernel (`bake_off_implicit_gemm.py:16–17`). Tolerable for an
integration test; the second invocation hits the JIT cache and is
sub-millisecond.

**Launch ABI:** 6 kernel args — `A_ptr, B_ptr, D_ptr` (FP16 globals)
plus `A_bytes, B_bytes, D_bytes` (i32). Grid is 2D over `(num_pid_n,
num_pid_m, 1)` where `num_pid_m = ceil((N·Ho·Wo)/tile_m)` and
`num_pid_n = ceil(K/tile_n)`. Block size: `warp_m · warp_n ·
wave_size`. All of this is derivable from the spec at JIT time and
emitted by the DSL into the kernel artifact's launch metadata — the C++
side does not hard-code it.

### Spec mirroring scope

`ImplicitGemmConvSpec` has 23 fields plus a nested `ConvProblem` (13
more) = 36 total. The C++ mirror struct `ConvImplicitGemmSpec` carries
all 36 so the on-wire payload to Python stays exhaustive, but only ~15
are touched by the adapter from the graph:

- From `ConvProblem`: `N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW`
- From the top-level spec: `dtype` (derived from graph tensor dtype),
  `name` (derived from a hash for cache-key stability)

The remaining 21 fields (`tile_m`, `tile_n`, `tile_k`, `warp_m`,
`warp_n`, `pipeline`, `epilogue`, `async_dma`, `chiplet_*`, etc.) are
hard-coded `constexpr` defaults in the C++ struct, copied verbatim from
the bake-off example. They become tunable in M2+ when autotuning is in
scope.

### CPU reference (no transpose needed)

`projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/
CpuFpReferenceConvolution.hpp` — template `fprop<XDataType, WDataType,
YDataType, ComputeDataType=float>` taking `TensorBase<T>` wrappers and
plain `std::vector<int64_t>` for strides/dilations/padding. Already used
by miopen-provider's integration tests via
`CpuReferenceGraphExecutor::execute()`.

The reference is **layout-agnostic via strides**, not NCHW-only. Its
loops walk *logical* dimensions `xDims[0]=N, xDims[1]=C, …` and call
`getHostValue({n, c, h, w})`. `TensorBase::getHostValue` resolves
indices with `std::inner_product(indices, strides, 0)` — pure stride
math (`data_sdk/utilities/Tensor.hpp:352–360`). The Tensor header
documents NHWC as a first-class layout (`Tensor.hpp:33–34`): NHWC is
expressed by giving NCHW-ordered logical `dims` with channel-last
`strides` like `{H·W·C, 1, W·C, C}`.

The integration test therefore wraps the DSL's NHWC input/output device
buffers in `Tensor<T>` with logical dims `{N, C, H, W}` and physical
NHWC strides, wraps the KYXC weight buffer with logical dims
`{K, C, Y, X}` and KYXC strides, and hands them directly to
`fprop(...)`. No data movement, no transpose.

### Performance measurement

No existing pattern in any integration test. `shared/primbench/
primbench.hpp` provides a hipEvent-based timing primitive but is not
wired in. M1 introduces the wiring: warmup N iters, time M iters,
report min/median kernel time and TFLOPS. **Logged only, not asserted**
— Q9 covers what we eventually compare against.

---

## 5. Provider skeleton

Target tree:

```
dnn-providers/rocke-provider/
├── CMakeLists.txt
├── README.md
├── version.json
├── cmake/
│   ├── rocke_python_embedding.cmake     # libpython + pybind11 discovery
│   └── rocke_install_python_package.cmake
├── include/                              # public-ish headers if any
├── python/
│   └── rocke_provider/
│       ├── __init__.py
│       └── compile_service.py            # ~30 LoC: dict → rocke Spec → compile → bytes
├── src/
│   ├── RockePluginPublic.cpp             # ~12 lines: defines 5 macros + #include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
│   ├── RockeContainer.{hpp,cpp}          # engine factory; registers one engine per op
│   ├── RockeHandle.{hpp,cpp}             # owns interpreter token, JIT cache, stream
│   ├── RockeSettings.hpp
│   ├── adapters/                         # per-op C++ adapter logic — the durable surface
│   │   ├── GraphAdapter.{hpp,cpp}        # IGraph walk → AdapterContext
│   │   └── conv_implicit_gemm/
│   │       ├── ConvImplicitGemmAdapter.{hpp,cpp}  # node → ConvImplicitGemmSpec
│   │       └── ConvImplicitGemmSpec.hpp           # mirrors ImplicitGemmConvSpec + ConvProblem
│   ├── python/
│   │   ├── EmbeddedInterpreter.{hpp,cpp} # init/teardown, GIL helpers
│   │   ├── PythonError.{hpp,cpp}         # py::error → hipdnn error translation
│   │   └── CompileServiceBridge.{hpp,cpp} # Payload → rocke_provider.compile_service
│   ├── runtime/
│   │   ├── KernelArtifact.{hpp,cpp}      # HSACO bytes + launch ABI metadata
│   │   ├── HipModule.{hpp,cpp}           # hipModule_t RAII
│   │   ├── LaunchAbi.{hpp,cpp}           # arg packing per artifact
│   │   └── JitCache.{hpp,cpp}            # signature → HipModule
│   ├── graph/
│   │   └── GraphSignature.{hpp,cpp}      # cache-key derivation from IGraph
│   └── engines/                              # one IEngine per op (M1 has just conv-igemm;
│       │                                     # M2+ adds sibling engines for gemm, attention, …)
│       └── conv_implicit_gemm/
│           ├── RockeConvImplicitGemmEngine.{hpp,cpp}   # IEngine for implicit-GEMM conv
│           ├── ConvImplicitGemmPlanBuilder.{hpp,cpp}
│           └── ConvImplicitGemmPlan.{hpp,cpp}
├── integration_tests/
│   ├── CMakeLists.txt
│   ├── IntegrationGraphVerificationHarness.hpp
│   ├── PerfMeasurement.{hpp,cpp}                # hipEvent warmup-and-iterate, TFLOPS calc
│   └── IntegrationGpuRockeConvFp16.cpp
└── tests/
    └── (unit tests deferred to M2)
```

### What we copy verbatim from miopen-provider

- `CMakeLists.txt` shape (`find_package(hipdnn_plugin_sdk …)`, object-
  library + shared lib pattern, install layout under
  `lib/hipdnn_plugins/engines/`).
- The `IntegrationGraphVerificationHarness` pattern.

### Plugin C-export wiring (no hand-written C)

Both miopen-provider and hip-kernel-provider confirm that **we never
hand-write the `hipdnnPluginGet*` / `hipdnnEnginePlugin*` C functions**.
The SDK provides `hipdnn_plugin_sdk/EnginePluginImpl.inl`, which
synthesises every required C export from five macros plus three C++
types the provider supplies. The whole `RockePluginPublic.cpp` will be
roughly:

```cpp
// SPDX-License-Identifier: MIT
#include "RockeContainer.hpp"
#include "RockeHandle.hpp"
#include "version.h"

#define HIPDNN_PLUGIN_NAME            "rocke_provider_plugin"
#define HIPDNN_PLUGIN_VERSION         ROCKE_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_API_VERSION     "1.0.0"
#define HIPDNN_PLUGIN_CONTAINER_TYPE  rocke_provider::RockeContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE     rocke_provider::RockeHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE    rocke_provider::RockeContext

#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
```

That is the only file in `src/` that touches the plugin C ABI. All real
work happens in the C++ `Container` / `Handle` / `Context` /
`IEngine` / `IPlanBuilder` / `IPlan` classes.

hip-kernel-provider proves we can put the macros directly in this .cpp
(no separate `Defines.hpp` needed). miopen-provider hoists them into a
header because it shares them across translation units. The rocke
provider can follow either style; M1 starts with the inline form for
simplicity.

### What we copy from hip-kernel-provider

- The `hipModuleLoadData` / `hipModuleLaunchKernel` execution path,
  stripped of HIPRTC.
- The `(uid → device buffer)` lookup helper for `execute()`.

### What is fresh

- `src/python/` — embedded interpreter, GIL handling, error translation.
- `runtime/JitCache.{hpp,cpp}` and `graph/GraphSignature.{hpp,cpp}`.
- The Python `rocke_provider` package and the spec builders for each op.

`[Q4]` Confirm provider location at `dnn-providers/rocke-provider/`
(parallel to the other four)?

---

## 6. Milestone 1 — end-to-end integration test (JIT path)

### 6.1 Prep (orchestrator-driven research before any code)

- **P-1.** Read `common/launcher.cpp` (~780 lines) end to end to nail
  down: manifest JSON shape, arg-packing convention, dynamic-LDS
  handling. Decide what `KernelArtifact` should look like in C++.
- **P-2.** Read `samples/example_engine_plugin/` end to end — smallest
  reference for the plugin C-export and engine plumbing.
- **P-3.** Spike: confirm pybind11 embedded interpreter works inside a
  hipDNN plugin .so (dlopen RTLD_LOCAL + libpython). One paragraph
  report. If it fails, we pivot to Option S before committing to
  skeleton layout. **This is the single biggest unknown of M1.**
- **P-4.** Confirm the `rocke` package version surface. Does
  `rocke.__version__` exist? If not, plan a small upstream patch to
  add it (we need it for the cache key).
- **P-5.** Read `instances/common/conv_implicit_gemm.py` and
  `examples/common/bake_off_implicit_gemm.py` end to end. Inventory the 36
  spec fields, mark which are graph-derived vs constexpr defaults, and
  capture the bake-off knob values verbatim. This becomes the
  initialiser for `ConvImplicitGemmSpec.hpp`.
- **P-6.** Stride conventions for the CPU reference: write a tiny unit
  test that constructs a `Tensor<T>` over a hand-filled NHWC buffer
  with `dims = {N, C, H, W}` and NHWC physical strides, then asserts
  `getHostValue({n, c, h, w})` returns the expected value for a few
  hand-picked positions. Repeat for KYXC weights. This is cheap
  (~30 lines) but locks down the stride math before the integration
  test depends on it.
- **P-7.** Decide perf measurement protocol: how many warmup iters,
  how many timed iters, which summary stat (min vs median), what to
  log. Capture in `PerfMeasurement.hpp` doc-comment. No magic numbers
  in the test file itself.

### 6.2 Implementation (each step is a buildable state)

1. **I-1. Provider skeleton compiles.** Copy `miopen-provider/CMake
   Lists.txt` + plugin C exports + empty `RockeConvImplicitGemmEngine`
   (reports no applicable plans). New superbuild preset
   `rocke-provider`. The engine is named per-op because M2+ adds
   sibling engines (`RockeGemmEngine`, `RockeAttentionEngine`, …) — one
   per rocke spec.
   *Test:* `cmake --preset rocke-provider && cmake --build build`
   produces `lib/hipdnn_plugins/engines/rocke_plugin.so`.

2. **I-2. Embedded interpreter inside the handle.** Link libpython,
   pull in pybind11. `RockeHandle` constructs an `EmbeddedInterpreter`
   on first instantiation per process (refcounted singleton). Hook
   teardown into plugin destroy.
   *Test:* tiny C++ smoke run loads the plugin, creates a handle,
   destroys it, no leaks under ASAN; logging confirms `Py_Initialize`
   ran once.

3. **I-3. C++ → Python → rocke round-trip.** `CompileServiceBridge`
   imports `rocke_provider.compile_service` and calls a no-op function
   that returns a constant — proves the import path and PYTHONPATH
   setup work in a plugin context.
   *Test:* a temporary unit test asserts the returned constant.

4. **I-4. KernelArtifact / HipModule round-trip from a known blob.**
   Call a `rocke_provider.compile_service.compile_smoke()` helper that
   builds and returns a trivial precompiled HSACO via
   `rocke.helpers.compile`. C++ loads it via `hipModuleLoadData`,
   fetches the function, launches over a 1-element buffer.
   *Test:* a temporary unit test verifies the launch returns without
   hipError.

5. **I-5. JitCache.** In-memory `signature → HipModule` map with mutex.
   Cache lookup before invoking the adapter+bridge path, insert after
   compile, eviction policy = none for M1 (handle-lifetime cache).
   *Test:* a temporary unit test asserts the second JIT call for the
   same signature is sub-millisecond.

6. **I-6. ConvImplicitGemmAdapter + Spec mirror.** Write
   `adapters/conv_implicit_gemm/` (C++): `ConvImplicitGemmSpec.hpp`
   (the 36-field mirror with constexpr defaults from P-5) +
   `ConvImplicitGemmAdapter::buildSpec(node, tensorMap)` populating the
   ~15 graph-derived fields + `Spec::to_payload()`.
   *Test:* a temporary unit test asserts that an adapter run on a
   hand-built graph node produces the expected payload dict.

7. **I-7. ConvImplicitGemmPlanBuilder.** `isApplicable()` returns true
   for a single-node conv-fwd graph (matching dtype/layout).
   `buildPlan()` runs the adapter, derives the cache signature, hits
   the JitCache, on miss calls `CompileServiceBridge::compile("conv_
   implicit_gemm", payload)` which returns HSACO bytes + launch ABI.

8. **I-8. ConvImplicitGemmPlan::execute().** Maps uids to DevPtrs, packs
   the 6 kernel args per the artifact's manifest, launches over the
   2D grid the manifest specifies.

9. **I-9. PerfMeasurement helper.** hipEvent-based warmup-and-iterate
   over `plan.execute()`. Computes TFLOPS from the spec's
   `2·N·Ho·Wo·K·C·Y·X` arithmetic-intensity. Logs results in a
   consistent format. **No perf assertions.**

10. **I-10. Integration test.** `IntegrationGpuRockeConvFp16` builds
    the conv-fwd graph via the hipDNN frontend (model the file on
    `miopen-provider/integration_tests/IntegrationGpuConvForward.cpp`),
    runs through the harness, wraps the DSL's NHWC/KYXC device buffers
    in `Tensor<T>` with NCHW-ordered logical dims + NHWC/KYXC physical
    strides, calls `CpuFpReferenceConvolution::fprop` directly,
    compares within tolerance, calls `PerfMeasurement` to log timing
    and TFLOPS.

11. **I-11. CI / pre-commit clean.** `pre-commit run --all-files`,
    `ninja rocke-provider-integration-check` green.

The throwaway tests in I-2, I-3, I-4, I-5, I-6 are explicit milestones
so each layer is verified before the next is stacked on top. They can
be deleted (or promoted to unit tests in M2) once I-10 is passing.

### 6.3 Review gates

- After I-1 and I-2: orchestrator inspects diff, no Reviewer needed.
- After I-3 and I-4: Reviewer agent specifically checks pybind11 usage
  (GIL discipline, error translation, lifetime).
- After I-5 and I-7: Reviewer agent checks `execute()` and cache for
  correctness, lifetime, and SDK conformance.
- After I-9: Reviewer agent reads the integrated change set end-to-end.

### 6.4 Implementation streams (parallel where the gating allows)

Q7's discipline — "temporary tests gate every step; only parallelise if
that survives" — defines the shape:

```
                    I-1 (skeleton compiles, sequential foundation)
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  Stream P (Python)    Stream A (Adapter)  Stream M (PerfMeasure)
  I-2 interpreter      I-6 ConvImplicit-   I-9 hipEvent warmup-
  I-3 import → rocke     GemmAdapter +       and-iterate +
  I-4 HSACO load+launch   Spec mirror +       TFLOPS calc
  I-5 JitCache            unit test           (standalone util)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                  Join: I-7 (PlanBuilder needs
                  cache + adapter + bridge)
                            ▼
                  I-8 Plan::execute()
                            ▼
                  I-10 Integration test
                            ▼
                  I-11 CI / pre-commit clean
```

**Stream P (Python critical path, ~5 Implementor-days).** Sequential
because each step's temporary test depends on the previous step's
output. Highest-risk stream (the libpython × RTLD_LOCAL spike from P-3
sits here). One Implementor.

**Stream A (Adapter, ~2 Implementor-days).** Pure C++. The
`ConvImplicitGemmSpec` mirror is mechanical from P-5's field
inventory. `ConvImplicitGemmAdapter::buildSpec` is small per-field
extraction logic against a hand-built test graph node. One Implementor.

**Stream M (PerfMeasurement, ~1 Implementor-day).** Pure C++ utility
on top of hipEvents. Self-contained; can be exercised with a no-op
kernel. One Implementor, or folded into Stream A if budget is tight.

**Shared contract.** Before any stream starts, the orchestrator
fixes the `Payload` shape (the C++ struct emitted by Stream A's
`Spec::to_payload()` and consumed by Stream P's
`CompileServiceBridge::compile(op_kind, payload)`). This is the only
interface the streams need to agree on; everything else is internal.

**Join point.** Stream P's I-5 and Stream A's I-6 must both be on the
feature branch before I-7 starts. Builder merges as usual; orchestrator
spawns the I-7 Implementor after the merge.

**Max concurrent Implementors:** 3 (one per stream). Well under the
7-agent ceiling, so no scheduling pressure.

### 6.5 Risk register

| Risk                                                                               | Likelihood | Severity | Mitigation                                                                                              |
|------------------------------------------------------------------------------------|------------|----------|---------------------------------------------------------------------------------------------------------|
| `libpython` + `RTLD_LOCAL` plugin loading interact badly (symbol visibility, multi-plugin conflicts) | Medium | High | P-3 spike before committing skeleton. Fall back to Option S if proven unworkable.               |
| Multiple in-proc Python consumers (host app, other plugins) initialise the interpreter differently | Low–Med | High | Use `Py_IsInitialized()` check; never `Py_Finalize` from the plugin. Document the assumption.        |
| `rocke` module discovery in plugin context (no virtualenv, custom PYTHONPATH)     | High       | Medium   | Provider's CMake installs a tiny `.pth` or sets `PYTHONPATH` via the embedded interpreter's `sys.path`. |
| `rocke` lacks a stable version string for cache keys                              | Medium     | Low      | P-4 confirms; upstream a small patch if missing.                                                        |
| `comgr` availability at runtime (not build time, since we JIT)                     | Low        | High     | `comgr` is part of ROCm — should be present whenever HIP is. Document the runtime dependency.           |
| GIL contention under concurrent plugin calls                                       | Low (M1)   | Low      | M1 cache is per-handle; one thread per handle is the de facto pattern.                                  |
| Compile failure on an unexpected graph variant turns into per-call latency        | Medium     | Medium   | Negative cache (§3.4) — failed signatures recorded and short-circuited.                                 |
| hipDNN frontend pointwise op surface doesn't quite match elementwise              | Low        | Low      | If `PointwiseOperation` is ambiguous, use a custom op id in the harness and revisit in M2.              |
| `IntegrationGraphVerificationHarness` not reusable as-is                          | Low        | Low      | Two providers already use it; copying is the cheapest path.                                             |
| Stride math for NHWC `Tensor<T>` wrong (CPU ref silently reads wrong elements)     | Low–Med    | High     | P-6 unit-tests stride construction against a hand-filled buffer at hand-picked positions before any integration use. |
| Constexpr defaults from bake-off example don't actually compile cleanly at our chosen shape | Low–Med | Medium | P-5 captures the exact bake-off knob values; first JIT compile in I-7 will surface any divergence early. |
| hipDNN graph's conv-fwd surface differs from the DSL's `ConvProblem` (stride/pad/dilation conventions) | Medium | Medium | P-6 reads the hipDNN conv-fwd schema; the adapter normalises conventions explicitly.                |
| Perf numbers look bad and we don't know if it's the DSL, the wrapper, or the shape | Medium  | Low (M1) | M1 only logs perf; Q9 decides what to compare against in M2.                                            |
| 36-field spec mirror drifts from Python dataclass over time                       | Medium     | Low (M1) | Cheap divergence test deferred to M2: build the Python dataclass via `**payload` and assert field-set parity. |

---

## 7. Beyond Milestone 1 (gestural, for context)

- **M2.** Second kernel (transpose or GEMM). Forces generalisation of the
  spec-builder Python package and of the graph signature.
- **M3.** On-disk HSACO cache so warm starts skip the JIT entirely.
  Cache invalidation on DSL version change.
- **M4.** Build-time pre-bake fast path: a list of known-hot shapes
  compiled at provider build time and seeded into the disk cache (or
  embedded into the .so). Runtime JIT still serves the long tail.
- **M5.** Additional per-op engines (`RockeGemmEngine`,
  `RockeAttentionEngine`, …) registered alongside the M1
  `RockeConvImplicitGemmEngine`. Applicability ranking against MIOpen.
  The per-op engine split was baked into the M1 file layout (§5) so
  this milestone is additive, not a refactor.
- **M6.** Autotuning hooks: let the JIT path try several specs for the
  same graph and persist the winner.

---

## 8. Open questions for the user

Consolidated; each is also tagged inline above. Resolved questions are
listed below for traceability; full rationale is in the change log
(§10).

### Resolved

- ~~`[Q1a]`~~ — Option E (embedded interpreter) confirmed; Option S
  (subprocess) is the fallback; Option D (daemon) deferred. *(v0.4)*
- ~~`[Q1b]`~~ — pybind11 confirmed as the binding library; raw CPython
  C API is the fallback. *(v0.4)*
- ~~`[Q2]`~~ — Provider-local Python service confirmed
  (`dnn-providers/rocke-provider/Python/rocke_provider/compile_service.py`).
  C++ adapter logic lives under `src/adapters/<op>/`. *(v0.4)*
- ~~`[Q3]`~~ — First kernel is **implicit-GEMM convolution** with the
  bake-off shape (N=8, 56×56×64→64, 3×3, stride 1, pad 1, FP16, NHWC).
  Perf measurement and CPU-reference verification now in M1 scope.
  *(v0.5)*
- ~~`[Q4]`~~ — Provider location at `dnn-providers/rocke-provider/`
  confirmed. *(v0.7)*
- ~~`[Q5]`~~ — No release-branch or Python-version constraints.
  **Branch model:** work happens on a new feature branch off
  `users/<user>/rocke-prototype`, not off develop (which lacks the
  rocke prototype). *(v0.7)*
- ~~`[Q6]`~~ — `rocke` discovered at runtime by **installing** it via
  the provider's build (`pip install` into the embedded interpreter's
  site-packages, or equivalent). Most straightforward path for POC.
  *(v0.7)*
- ~~`[Q7]`~~ — M1 runs as **one sequential Python critical path
  (Stream P)** plus **two parallel C++ sub-streams** (adapter, perf
  helper). Gating preserved: every step in every stream still ships
  with its temporary test. See §6.4 for the stream diagram. *(v0.7)*
- ~~`[Q8]`~~ — Cold-cache JIT cost (~150 ms for the bake-off conv)
  accepted as M1 baseline; measurement only, no perf assertion. *(v0.7)*
- ~~`[Q9]`~~ — M1 logs absolute TFLOPS only; apples-to-apples
  comparison vs MIOpen / native CK_Tile deferred to M2 once a second
  op exists to amortise the comparison harness against. *(v0.7)*

### Open

(none — all resolved as of v0.7)

---

## 9. Out-of-band references

All entries are committed code or in-tree documentation.

- DSL compile entry: `dnn-providers/hip-kernel-provider/rocKE/Python/rocke/helpers/compile.py`
- DSL instances: `dnn-providers/hip-kernel-provider/rocKE/Python/rocke/instances/`
- DSL conv builder used for M1: `dnn-providers/hip-kernel-provider/rocKE/Python/rocke/instances/common/conv_implicit_gemm.py`
- DSL conv example (shape + perf numbers cited in §4): `dnn-providers/hip-kernel-provider/rocKE/Python/rocke/examples/common/bake_off_implicit_gemm.py`
- Existing DSL C++ launcher (HSACO load + launch): `projects/composablekernel/example/ck_tile/dsl/common/launcher.cpp`
- Plugin SDK developer guide: `projects/hipdnn/docs/PluginDevelopment.md`
- RFC 0002 (Plugin SDK design): `projects/hipdnn/docs/rfcs/0002_PluginSdkDesign.md`
- RFC 0006 (plugin-agnostic integration tests): `projects/hipdnn/docs/rfcs/0006_PluginAgnosticIntegrationTests.md`
- Reference plugin: `projects/hipdnn/samples/example_engine_plugin/`
- Plugin C-export `.inl`: `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/EnginePluginImpl.inl`
- CPU conv reference: `projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp`
- Tensor (strides + layout): `projects/hipdnn/data_sdk/include/hipdnn_data_sdk/utilities/Tensor.hpp`
- hipEvent timing primitive: `shared/primbench/primbench.hpp`
- Closest provider template (plumbing): `dnn-providers/miopen-provider/`
- Closest provider template (kernel launch): `dnn-providers/hip-kernel-provider/`
- Most architecturally similar provider (runtime JIT, different stack): `dnn-providers/fusilli-provider/`

---

## 10. Change log

- **v0** (2026-05-21) — initial draft. Build-time HSACO embedding (Option
  A1) as primary, runtime Python (Option C) ruled out.
- **v0.1** (2026-05-21) — runtime Python execution made the primary
  architecture; embedded interpreter (Option E) as plan of record;
  subprocess (Option S) as fallback. Build-time pre-bake demoted to an
  M4 optimisation. Milestone 1 integration test now exercises the JIT
  path, not a precompiled blob.
- **v0.2** (2026-05-21) — clarified that the plugin C-export wiring
  uses the SDK's `EnginePluginImpl.inl` (no hand-written C); dropped
  `RockePluginDefines.hpp` from the skeleton tree. Split Option E into
  two orthogonal decisions: where the interpreter lives (embedded vs
  subprocess vs daemon) and how C++ calls into it (pybind11 vs raw C
  API). Recommendation remains embedded + pybind11.
- **v0.3** (2026-05-21) — moved per-op adapter logic from Python to
  C++ (new `src/adapters/<op>/` tree with `<Op>Spec` structs that
  mirror the Python dataclasses + `<Op>Adapter` that walks `IGraph` to
  build them). Python side reduced to a ~30 LoC `compile_service.py`
  that unpacks the payload dict into the matching `rocke` dataclass
  and calls `compile_kernel`. Motivation: when/if the DSL itself
  becomes C++, the throwaway code is the Python glue, not the
  adapter. Updated §3.1 pipeline diagram, §5 skeleton, §6 step I-6,
  and Q2 accordingly.
- **v0.4** (2026-05-21) — user confirmed three open questions:
  **Q1a** Option E (embedded interpreter) as default, Option S
  (subprocess) as the fallback if libpython linkage proves
  unworkable. **Q1b** pybind11 as the binding library, raw CPython C
  API as fallback. **Q2** the Python compile service lives provider-
  local at `dnn-providers/rocke-provider/Python/rocke_provider/`,
  not under `rocke/`. Five open questions remain (Q3–Q8); the most
  load-bearing is Q3 (first kernel — default still elementwise).
- **v0.5** (2026-05-21) — **Q3 resolved: implicit-GEMM convolution**
  is the first kernel, bake-off shape (N=8, 56×56×64→64, 3×3, s=1,
  p=1, FP16, NHWC). M1 scope expands to include CPU-reference
  verification (`CpuFpReferenceConvolution::fprop` from `test_sdk`,
  with a host-side NHWC↔NCHW transpose bridge) and hipEvent-based
  perf measurement (warmup-and-iterate, TFLOPS logged, no perf
  assertion). New prep steps P-5/P-6/P-7. M1 implementation grows
  from 9 to 12 steps. Risk register adds layout-transpose bug,
  spec-knob drift, and graph/DSL conv-convention mismatch. New Q9
  asks what M2's perf comparison baseline should be (MIOpen,
  native CK_Tile, or absolute TFLOPS only).
- **v0.6** (2026-05-21) — direct read of `CpuFpReferenceConvolution.
  hpp` and `data_sdk/utilities/Tensor.hpp` revealed the CPU reference
  is **layout-agnostic via strides**, not NCHW-only: index resolution
  is `std::inner_product(indices, strides, 0)` and the Tensor header
  documents NHWC as a first-class layout (`dims={N,C,H,W}` +
  channel-last `strides={H·W·C, 1, W·C, C}`). The earlier research
  summary was wrong on this point. Removed `NhwcNchwLayoutBridge.{hpp,
  cpp}` from §5, deleted I-9 transpose step (renumbered the rest, 12→
  11 total), reframed P-6 from "harden the transpose" to "unit-test
  stride construction." Downgraded the corresponding risk-register
  entry: "transpose bug" → "stride math wrong"; Medium/High → Low-Med/
  High.
- **v0.7** (2026-05-21) — six remaining open questions resolved.
  **Q4** provider location at `dnn-providers/rocke-provider/`
  confirmed. **Q5** no constraints, but feature branch is cut from
  `users/<user>/rocke-prototype`, not from develop, because develop
  lacks the rocke prototype. Added a "Branch model" subsection to §1.
  **Q6** `rocke` is installed into the embedded interpreter's
  site-packages by the provider's build. **Q7** M1 runs as one
  sequential Python critical path (I-2→I-3→I-4→I-5) plus two parallel
  C++ sub-streams (adapter at I-6, perf helper at I-9), joining at
  I-7; temporary-test gating preserved within each stream. New §6.4
  "Implementation streams" subsection captures this; old §6.4 risk
  register renumbered to §6.5. **Q8** cold-cache JIT (~150 ms)
  accepted as M1 baseline. **Q9** M1 logs absolute TFLOPS only; perf
  comparison deferred to M2. Plan is now implementation-ready.
- **v0.9** (2026-05-21) — renamed the single M1 engine
  `RockeEngine` → `RockeConvImplicitGemmEngine` and moved it under
  `src/engines/conv_implicit_gemm/` alongside its plan builder + plan.
  Motivation: the eventual provider has one engine per rocke spec
  (M5); baking that into M1's naming means M2+ adds sibling engine
  directories rather than refactoring the existing one. §5 file tree
  updated, §6.2 I-1 reworded, §7 M5 reworded.
- **v0.8** (2026-05-21) — user removed `dsl_docs/BRANCH_NOTES.md` and
  emptied `dsl_docs/00_for_newcomers/`. Dropped both from §9
  references and re-anchored every assertion that previously leaned
  on them to committed code: compile-latency claim now cites
  `bake_off_implicit_gemm.py:16–17` ("typically <150 ms wall"); the
  280 TFLOPS perf number now cites `bake_off_implicit_gemm.py:70–71`
  ("248 TFLOPS per-launch / 280 TFLOPS graph 5×200"); dropped the
  unsourced "~50 ms elementwise" comparison. Cleaned up §8 (the
  earlier resolution edits had left the original open-question text
  in place underneath the "all resolved" marker). Added several
  in-tree references to §9 that were already in use in the plan body
  but not previously listed (`EnginePluginImpl.inl`,
  `CpuFpReferenceConvolution.hpp`, `Tensor.hpp`,
  `bake_off_implicit_gemm.py`, RFC 0006, `primbench.hpp`).
