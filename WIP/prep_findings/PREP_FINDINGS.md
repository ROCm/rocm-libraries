# CK DSL hipDNN Provider — Prep Phase Findings

Synthesis of plan §6.1 prep work (P-1 … P-7). All findings are referenced
back to specific files. Findings flagged **[BLOCKING]** must be resolved
before the next step; **[ADAPTATION]** is a deviation from the plan that
the orchestrator chose and the user can override.

---

## P-1. `launcher.cpp` design memo → `KernelArtifact`

Source: `projects/composablekernel/example/ck_tile/dsl/common/launcher.cpp` (~780 lines, self-contained — only HIP + STL includes).

Key facts (cite file:line):
- HSACO load path: `hipModuleLoadData` then `hipModuleGetFunction`; blob may be released after the load call (`launcher.cpp:571-575`). Module unloaded at end (`:777`).
- Args: serialized into a contiguous `std::vector<char>` and handed to `hipModuleLaunchKernel` via the `HIP_LAUNCH_PARAM_BUFFER_POINTER/SIZE/END` "extra" path; `kernelParams` is always nullptr (`:651-655`, `:680-690`).
- Natural alignment only — no explicit padding. GEMM args = 3 ptrs + 3 i32 (`:184-201`); conv args = 3 ptrs + 3 i32 (`:369-394`) when `sig_has_bytes=1`.
- `sharedMemBytes` hardcoded to 0 (`:687`, `:708`, `:738`) — **launcher cannot launch kernels with dynamic LDS**; our `KernelArtifact` adds an `lds_bytes` field to close that gap.
- Default stream only (`:688`, `:709`); we must take the stream from `CkDslHandle::setStream`.
- Manifest fields used: `kind`, `kernel_name`, `grid_order` ("MN"/"NM"), `block_m`, `block_n`, `threads_per_block`, `default_shape`, `conv` (≥7 ints), `groups/cpg/kpg`, `sig_has_bytes`, `grid_explicit[3]`, `grid_z`, `timed_iters`, `warmup_iters`. No LDS field. No dtype enum.

**Recommended `KernelArtifact` shape** (for `src/runtime/KernelArtifact.hpp` in I-1):

```cpp
struct KernelArtifact {
    std::vector<std::byte> hsaco;
    std::string            kernelName;
    struct GridSpec  { uint32_t gx{1}, gy{1}, gz{1}; } grid;
    struct BlockSpec { uint32_t x{1},  y{1},  z{1};  } block;
    uint32_t ldsBytes{0};   // fills launcher.cpp's gap
    enum class ArgKind : uint8_t { Pointer, I32, I64, F32, F16 };
    struct ArgSlot { ArgKind kind; uint16_t size; uint16_t align; };
    std::vector<ArgSlot> argSchema;
    std::string kind;
};
```

Push grid/block computation out of the artifact and into the per-op
`SpecBuilder` (the engine knows the problem shape).

`LaunchAbi::pack(const std::vector<ArgSlot>&, const std::vector<ArgValue>&)`
→ contiguous `std::vector<std::byte>`. Schema-driven (generalizes the
launcher's hardcoded per-`kind` packing).

---

## P-2. `example_engine_plugin` skeleton map

Source: `projects/hipdnn/samples/example_engine_plugin/` and `projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/`.

**Required macros for `CkDslPluginPublic.cpp`** (verbatim, per
`EnginePluginImpl.inl:59-77` + `ExampleProviderPluginPublic.cpp:16-22`):

```cpp
#define HIPDNN_PLUGIN_NAME            "ck_dsl_provider_plugin"
#define HIPDNN_PLUGIN_VERSION         CK_DSL_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_CONTAINER_TYPE  CkDslContainer
#define HIPDNN_PLUGIN_HANDLE_TYPE     CkDslHandle
#define HIPDNN_PLUGIN_CONTEXT_TYPE    CkDslContext
#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>
```

**Interfaces** (all under `plugin_sdk/include/hipdnn_plugin_sdk/interfaces/`):

- `IEngine` — 5 pure virtuals: `id()`, `isApplicable()`, `getDetails()`, `getMaxWorkspaceSize()`, `initializeExecutionContext()`. Engine picks the first applicable plan builder internally (no plan-list surface). `getDetails()` MUST stash its FlatBuffer in the handle's detached-buffer map.
- `IPlanBuilder` — 5 pure virtuals; `buildPlan(...)` returns void and stores the plan via `executionContext.setPlan(std::move(plan))`.
- `IPlan` — `getWorkspaceSize()` + `execute(handle, deviceBuffers, num, workspace)`. `deviceBuffers` is a flat `{uid, ptr}[]`, NOT a map — O(n) lookup via `findDeviceBuffer` (in `ExampleProviderUtils.hpp:20-36`).

**CMake pattern** (mirror `samples/example_engine_plugin/src/CMakeLists.txt`):
- Three-target split: OBJECT (impl) → STATIC (private, for unit tests) → SHARED (loadable .so, hidden visibility, `--exclude-libs=ALL`, `INSTALL_RPATH "$ORIGIN;$ORIGIN/../.."`).
- `find_package(hipdnn_plugin_sdk CONFIG REQUIRED)` transitively pulls FlatBuffers, HIP, nlohmann_json.
- Install to `${HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR}`.

**Gotcha** — `HIPDNN_REGISTER_ENGINE(NAME)` produces a hash-derived
`int64_t` engine ID from the string name. Choose stable names; renames
break engine selection.

---

## P-3. pybind11 + RTLD_LOCAL spike → **SUCCESS** (architecture-clear)

Built `WIP/pybind11_rtld_local_spike/`: tiny `libpyplugin.so` linked to
libpython3.12 + pybind11, loaded twice in succession via `dlopen(...,
RTLD_NOW | RTLD_LOCAL)` from a host harness. Both rounds: interpreter
initialized, imported `sys` and `json`, executed `json.dumps({"ok": 1})`,
parsed result, returned 0.

**Architectural implication:** Option E (embedded interpreter) is viable
for the plugin loader. No pivot to Option S (subprocess) needed.

**[BLOCKING]** Python pinning is non-obvious: CMake's
`find_package(Python3)` greedily picks up a uv-managed Python 3.10 in
`~/.local` and crashes at runtime ("failed to get the Python codec of
the filesystem encoding") because uv's stdlib lives at an internal path.
The provider's CMake **must** pin the system Python explicitly:

```cmake
find_package(Python3
    COMPONENTS Interpreter Development REQUIRED
    PATHS /usr   # FIND_STRATEGY=LOCATION
)
# or at configure time:
#   -DPython3_EXECUTABLE=/usr/bin/python3
#   -DPython3_ROOT_DIR=/usr
#   -DPython3_FIND_STRATEGY=LOCATION
```

`pybind11` discovered via `python3 -m pybind11 --cmakedir` (3.0.4, installed with `pip install --user`).

Also: **never call `Py_Finalize`** from the plugin (per plan risk
register). Hold `py::scoped_interpreter` on a static pointer leaked at
plugin-unload.

---

## P-4. `ck_dsl` version surface and install metadata

**Negative findings:**
- `ck_dsl/__init__.py` has **no `__version__`** attribute.
- `projects/composablekernel/python/` has **no `pyproject.toml` /
  `setup.py` / `setup.cfg`**. The package is in-tree only.

**Consequences for M1:**
1. **Cache key** (§3.4) cannot use `ck_dsl.__version__`. M1 fallback:
   embed a git SHA at provider build time (CMake reads `git rev-parse
   --short HEAD` from the `projects/composablekernel/python/ck_dsl/`
   subtree) and use that as the `dslVersionString`. Tracked as M2
   cleanup: upstream a `__version__` constant to `ck_dsl/__init__.py`.
2. **`ck_dsl` import discovery** (Q6 resolution assumed `pip install`).
   Without packaging metadata, the M1 path is to **manipulate `sys.path`**
   in the embedded interpreter's init before importing
   `ck_dsl_provider.compile_service`:
   ```python
   import sys; sys.path.insert(0, "<CK_DSL_PATH_INJECTED_BY_CMAKE>")
   ```
   The path is supplied via a generated header
   (`ckdsl_provider_paths.h`) emitted by the provider's CMake. M2
   cleanup: package `ck_dsl` with a `pyproject.toml`, drop the sys.path
   hack.

**[ADAPTATION]** Deviating from Q6 ("install into site-packages"):
M1 uses sys.path injection instead, because no install metadata exists
upstream. Same end-effect (interpreter can `import ck_dsl`), simpler to
land in M1, reversible when packaging arrives.

---

## P-5. `ImplicitGemmConvSpec` field inventory (36 fields) + bake-off values

Source: `projects/composablekernel/python/ck_dsl/instances/conv_implicit_gemm.py` (`ConvProblem`: lines 78-105; `ImplicitGemmConvSpec`: lines 135-220) and `examples/bake_off_implicit_gemm.py:47-86`.

**13 `ConvProblem` fields** — all graph-derived from a hipDNN conv-fwd node:
`N, Hi, Wi, C, K, R, S, sH, sW, pH, pW, dH, dW`

**23 `ImplicitGemmConvSpec` fields** — 1 is graph-derived (the nested
`problem`), 22 are constexpr defaults.

**Verbatim bake-off invocation** (the C++ struct initializer copies this):

```python
ConvProblem(N=8, Hi=56, Wi=56, C=64, K=64, R=3, S=3,
            sH=1, sW=1, pH=1, pW=1, dH=1, dW=1)
ImplicitGemmConvSpec(
    problem=problem,
    name="ck_dsl_ex08_bake_off_implicit_gemm",
    tile_m=64, tile_n=64, tile_k=64,
    warp_m=2,  warp_n=2,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="mem", epilogue="cshuffle",
)
```

**[BLOCKING]** **Dataclass defaults are NOT the bake-off values**. The
bake-off overrides:
- `tile_k`: dataclass 128 → bake-off 64
- `warp_tile_m / warp_tile_n`: dataclass 16 → bake-off 32  (32×32×16 MFMA atom, not 16×16×32)
- `warp_tile_k`: dataclass 32 → bake-off 16
- `epilogue`: dataclass `"default"` → bake-off `"cshuffle"`  (largest single perf lever)

The C++ `ConvImplicitGemmSpec.hpp` constexpr defaults MUST take the
bake-off values, not the dataclass defaults. Annotate with a comment.

**Launch ABI** (matches plan §4 exactly): 6 args `(A_ptr, B_ptr, D_ptr,
A_bytes, B_bytes, D_bytes)`; block size = `warp_m * warp_n * wave_size`
= 2·2·64 = 256; grid = `(num_pid_n, num_pid_m, 1)` with `grid_order="NM"`
which swaps to map `block_id_x` ↔ N-tile, `block_id_y` ↔ M-tile. For the
bake-off shape: M = 8·56·56 = 25088 → num_pid_m = 392; num_pid_n = 1.
Grid = `(1, 392, 1)`.

**Verified claims** (compile cost and perf, used in plan §4):
- `bake_off_implicit_gemm.py:21-22` — "Codegen time for one kernel is typically <150 ms wall."
- `bake_off_implicit_gemm.py:69-72` — "per-launch: 248 TFLOPS; graph 5x200: 280 TFLOPS".

---

## P-6. NHWC/KRSC stride math — verified by inspection

The plan called for a standalone 30-line CMake test. **[ADAPTATION]**
verified by inspection of `Tensor.hpp` instead — saves redundant CMake
plumbing that I-1 will set up anyway. The empirical test will land as a
unit test inside the provider tree in I-1 or I-6.

`TensorLayout::NHWC` has `strideOrder = {3, 0, 2, 1}` (Tensor.hpp:63),
which means dim 1 (C) is innermost. `generateStrides` then produces
`{N=H·W·C, C=1, H=W·C, W=C}` for logical dims `{N,C,H,W}`. The CPU
reference walks logical indices `{n,c,h,w}` and resolves them via
`std::inner_product(indices, strides, 0)` (Tensor.hpp:360).

Hand-verified for `dims={1,64,28,28}, (n=0,c=5,h=2,w=3)`:
- inner_product: 0·50176 + 5·1 + 2·1792 + 3·64 = 3781
- direct NHWC formula: 0 + 2·(28·64) + 3·64 + 5 = 3781 ✓

KRSC weights use the same pattern with logical dims `{K,C,R,S}`:
`strideOrder = strideOrderNhwc(4) = {3,0,2,1}` → strides `{R·S·C, 1, S·C, C}`. Verified analogously.

Integration test plan (I-10): wrap the DSL's NHWC input/output device
buffers with logical dims `{N,C,H,W}` + `TensorLayout::NHWC`; wrap KRSC
weights with logical dims `{K,C,R,S}` + `strideOrderNhwc(4)`. No data
movement.

---

## P-7. PerfMeasurement protocol design

Captured here so I-9 (`PerfMeasurement.hpp`) has zero magic numbers in
the doc-comment:

| Knob              | Value | Rationale                                                                                         |
|-------------------|-------|---------------------------------------------------------------------------------------------------|
| Warmup iters      | 5     | Matches `launcher.cpp:559` default. Sufficient for caches/JIT to warm.                            |
| Timed iters       | 50    | Half of launcher.cpp's 100 — keeps the integration test ≤1s of kernel time while still stable.    |
| Summary stats     | min + median | Min = best-case; median = robust typical. **Report both; use median for TFLOPS.**         |
| TFLOPS formula    | `2 · N · Ho · Wo · K · C · R · S / median_seconds / 1e12`                                                 |
| Sync between iters| `hipEventRecord` only (no `hipDeviceSynchronize` between launches)                                |
| Log format        | `[CkDslPerf] op=<name> shape=<...> dtype=<...> warmup=5 iters=50 min_us=X.X median_us=Y.Y tflops=Z.Z` |
| Assertions        | **None.** Logging-only per plan Q9.                                                               |

API sketch for `PerfMeasurement.hpp` (I-9):

```cpp
struct PerfResult {
    double minUs;
    double medianUs;
    double tflops;
};
class PerfMeasurement {
public:
    PerfMeasurement(uint32_t warmupIters = 5, uint32_t timedIters = 50);
    template <typename LaunchFn>
    PerfResult measure(LaunchFn&& launch, double flops, hipStream_t stream);
    void log(std::string_view tag, const PerfResult& r) const;
};
```

---

## Branch model (confirmed)

Working on `users/dahawkin/ck-dsl-provider`, cut from
`users/vanantha/ck-dsl-prototype` (merge-base 519632731d3). Working tree
clean. Sparse checkout at 51% — some paths may need lazy fetching as we
hit them.

---

## Adaptations summary (deviations from plan v0.8)

1. **P-4**: M1 uses sys.path injection + git SHA cache key, not pip-install + `ck_dsl.__version__`. Upstream packaging deferred to M2.
2. **P-6**: stride test verified by inspection; the empirical unit test will land inside the provider tree (I-1 or I-6) instead of as a standalone WIP spike.

Both are reversible and reduce M1 scope without changing architecture.

---

## Recommended next step

Proceed to **I-1** (provider skeleton compiles): copy CMake + plugin
C-exports from `samples/example_engine_plugin` + `miopen-provider`,
empty `CkDslEngine` that reports nothing applicable, new superbuild
preset, target: `cmake --preset ck-dsl-provider && cmake --build build`
produces `lib/hipdnn_plugins/engines/ck_dsl_provider_plugin.so`.
