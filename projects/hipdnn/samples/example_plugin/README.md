# hipDNN Example Plugin

A self-contained example project that demonstrates how to build a hipDNN engine
plugin from scratch.  It is designed for external developers (including non-AMD
developers) who want to extend hipDNN with custom GPU-accelerated engine
implementations.

The plugin implements two GPU operations compiled at runtime via HIPRTC (HIP
Runtime Compilation):

- **ReLU forward** (pointwise): element-wise `max(0, x)` with a custom
  `example.relu.negative_slope` knob for leaky ReLU support
- **Convolution forward** (naive): 2D cross-correlation, NCHW layout, single
  thread per output element

A third engine (`EXAMPLE_PLUGIN_ADVANCED_ENGINE`) is provided as a scaffold
demonstrating multi-node graph matching patterns, with a placeholder
`isApplicable()` that always returns `false`.

## Prerequisites

| Dependency | Purpose | Notes |
|---|---|---|
| CMake >= 3.20 | Build system | |
| C++17 compiler | GCC/G++ (Linux) or MSVC (Windows) | No GPU compiler needed at build time |
| ROCm (HIP SDK + HIPRTC) | GPU kernel compilation and execution | `hipStream_t`, `hipMalloc`, HIPRTC APIs |
| hipDNN (installed) | Plugin SDK, data SDK, frontend library | Installed at `/opt/rocm` |
| GPU hardware | Runtime execution of HIPRTC-compiled kernels | Any ROCm-supported GPU |
| Internet access | GTest is downloaded via CMake `FetchContent` | Only needed for the first build |

The plugin C++ source code compiles with standard compilers (GCC, MSVC).  GPU
kernels are plain `.cpp` files that are embedded as string literals at CMake
configure time and compiled at runtime by HIPRTC -- no GPU compiler (`hipcc`,
`amdclang++`) is needed during the build.

## Directory Structure

```
example_plugin/
├── CMakeLists.txt                       # Root CMake: project options, dependencies
├── README.md                            # This file
├── kernels/                             # GPU kernel source files (embedded at configure time)
│   ├── CMakeLists.txt                   # embed_kernel_sources() function
│   ├── templates/                       # .in templates for kernel embedding
│   │   ├── kernel_sources.cpp.in
│   │   ├── kernel_sources.hpp.in
│   │   ├── kernel_includes.cpp.in
│   │   └── kernel_includes.hpp.in
│   ├── relu/
│   │   └── ReluForward.cpp              # ReLU GPU kernel (~10 lines)
│   └── conv/
│       └── ConvForwardNaive.cpp         # Naive ConvFwd GPU kernel (~35 lines)
├── src/
│   ├── CMakeLists.txt                   # OBJECT, static, and shared library targets
│   ├── ExamplePluginPublic.cpp          # C entry points (5 macros + EnginePluginImpl.inl)
│   ├── ExamplePluginContainer.hpp/cpp   # Engine registration and EngineManager
│   ├── ExamplePluginHandle.hpp/cpp      # Plugin handle (stream, container reference)
│   ├── ExamplePluginContext.hpp         # Execution context
│   ├── ExamplePluginSettings.hpp        # Execution settings (reluNegativeSlope)
│   ├── CurrentDevicePropertyProvider.hpp  # IDevicePropertyProvider implementation
│   ├── hip/                             # HIPRTC infrastructure (DI interfaces + impls)
│   │   ├── IKernelCompiler.hpp          # Interface: compile(filename, options)
│   │   ├── ICompiledProgram.hpp         # Interface: getRunnableKernel(name)
│   │   ├── IRunnableKernel.hpp          # Interface: launch(stream, args...)
│   │   ├── IDevicePropertyProvider.hpp  # Interface: getDeviceProperties()
│   │   ├── HipUtils.hpp                # HIP_CHECK and HIPRTC_CHECK error macros
│   │   ├── HipKernelCompiler.hpp        # Concrete IKernelCompiler (HIPRTC)
│   │   ├── HipCompiledProgram.hpp/cpp   # Concrete ICompiledProgram (HIPRTC compilation + module)
│   │   └── HipRunnableKernel.hpp/cpp    # Concrete IRunnableKernel (hipFunction_t)
│   └── engines/
│       ├── ExamplePluginEngine.hpp/cpp  # Engine: owns PlanBuilders, delegates isApplicable
│       ├── ExamplePluginUtils.hpp       # Utility: UID-to-buffer lookup
│       ├── AdvancedEngineScaffold.hpp/cpp  # Placeholder engine (isApplicable=false)
│       └── plans/
│           ├── ReluPlanBuilder.hpp/cpp  # PlanBuilder: graph matching for ReLU_FWD
│           ├── ReluPlan.hpp/cpp         # Plan: GPU ReLU execution via HIPRTC
│           ├── ConvFwdPlanBuilder.hpp/cpp  # PlanBuilder: graph matching for ConvFwd
│           └── ConvFwdPlan.hpp/cpp      # Plan: GPU ConvFwd execution via HIPRTC
├── tests/                               # Unit tests (GTest, no GPU required)
│   ├── CMakeLists.txt
│   ├── TestHelpers.hpp                  # FlatBuffer graph construction helpers
│   ├── mocks/                           # Mock objects for GPU-free unit testing
│   │   ├── MockKernelCompiler.hpp
│   │   ├── MockCompiledProgram.hpp
│   │   ├── MockRunnableKernel.hpp
│   │   ├── MockDevicePropertyProvider.hpp
│   │   └── MockPlanBuilder.hpp
│   ├── TestExamplePluginContainer.cpp
│   ├── TestReluPlanBuilder.cpp
│   ├── TestReluPlan.cpp
│   ├── TestConvFwdPlanBuilder.cpp
│   ├── TestConvFwdPlan.cpp
│   └── TestAdvancedEngineScaffold.cpp
├── integration_tests/                   # Integration tests (full hipDNN stack, GPU required)
│   ├── CMakeLists.txt
│   └── TestPluginIntegration.cpp
└── sample/                              # Sample application (GPU required)
    ├── CMakeLists.txt
    └── ExamplePluginSample.cpp
```

## Build Instructions

### Linux (GCC)

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/hip"
cmake --build .
```

Run the unit tests (no GPU required):

```bash
ctest --test-dir build
```

Run the integration tests (requires GPU):

```bash
cmake .. -DHIPDNN_EXAMPLE_PLUGIN_BUILD_INTEGRATION_TESTS=ON
cmake --build .
ctest --test-dir build
```

Run the sample application (requires GPU):

```bash
cmake .. -DHIPDNN_EXAMPLE_PLUGIN_BUILD_SAMPLE=ON
cmake --build .
HIPDNN_PLUGIN_DIR=build/src ./build/sample/example_plugin_sample
```

Install the plugin:

```bash
cmake --install build --prefix /opt/rocm
# Plugin .so is installed to <prefix>/lib/hipdnn_plugins/engines/
```

### Windows (MSVC) -- Untested

```powershell
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="C:\rocm;C:\rocm\hip" -G "Visual Studio 17 2022"
cmake --build . --config Release
ctest --test-dir . --build-config Release
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `HIPDNN_EXAMPLE_PLUGIN_BUILD_UNIT_TESTS` | `ON` | Build unit tests (no GPU required) |
| `HIPDNN_EXAMPLE_PLUGIN_BUILD_INTEGRATION_TESTS` | `OFF` | Build integration tests (requires `hipdnn_frontend` + GPU) |
| `HIPDNN_EXAMPLE_PLUGIN_BUILD_SAMPLE` | `OFF` | Build sample application (requires `hipdnn_frontend` + GPU) |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation path (for RPATH and library discovery) |

To build everything:

```bash
cmake .. -DHIPDNN_EXAMPLE_PLUGIN_BUILD_INTEGRATION_TESTS=ON \
         -DHIPDNN_EXAMPLE_PLUGIN_BUILD_SAMPLE=ON
```

To build only the plugin library (no tests):

```bash
cmake .. -DHIPDNN_EXAMPLE_PLUGIN_BUILD_UNIT_TESTS=OFF
```

## Architecture

A hipDNN plugin is a shared library that implements a C API defined by the
plugin SDK.  The SDK provides `EnginePluginImpl.inl`, which generates all
required C entry points when five macros are defined in
`ExamplePluginPublic.cpp`:

- `HIPDNN_PLUGIN_NAME` -- display name string
- `HIPDNN_PLUGIN_VERSION` -- version string
- `HIPDNN_PLUGIN_CONTAINER_TYPE` -- fully qualified Container class name
- `HIPDNN_PLUGIN_HANDLE_TYPE` -- fully qualified Handle struct name
- `HIPDNN_PLUGIN_CONTEXT_TYPE` -- fully qualified Context struct name

### Type Hierarchy

```
Container
├── Owns EngineManager<Handle, Settings, Context>
├── Owns IKernelCompiler (HipKernelCompiler) and IDevicePropertyProvider
├── Registers engines via getEngineDefinitions()
│   ├── Engine (EXAMPLE_PLUGIN_RELU_ENGINE)
│   │   └── PlanBuilder (ReluPlanBuilder)
│   │       └── Plan (ReluPlan) -- HIPRTC-compiled GPU kernel
│   ├── Engine (EXAMPLE_PLUGIN_CONV_FWD_ENGINE)
│   │   └── PlanBuilder (ConvFwdPlanBuilder)
│   │       └── Plan (ConvFwdPlan) -- HIPRTC-compiled GPU kernel
│   └── Engine (EXAMPLE_PLUGIN_ADVANCED_ENGINE -- scaffold)
└── copyEngineIds() -- returns registered engine IDs to hipDNN

Handle
├── Holds shared_ptr<Container>
├── setStream(hipStream_t) -- stream management
└── getEngineManager() -- provides access to engines

Context
└── Inherits HipdnnEnginePluginExecutionContext + ExecutionContextBase

Settings
└── Plugin-specific execution settings (reluNegativeSlope)
```

### Engine Execution Flow

1. **Container** registers engines with unique string IDs via
   `HIPDNN_REGISTER_ENGINE()`.  The macro creates a compile-time FNV-1a hash
   used as the numeric engine ID.

2. hipDNN calls `isApplicable()` on each engine to check whether it supports a
   given operation graph.

3. The engine delegates to its **PlanBuilders**.  Each PlanBuilder inspects the
   graph's node attributes (e.g., `PointwiseAttributes` with
   `PointwiseMode::RELU_FWD`, or `ConvolutionFwdAttributes` with
   `ConvMode::CROSS_CORRELATION`).

4. `buildPlan()` extracts tensor metadata (UIDs, dimensions) from the graph,
   creates a **Plan** object, and calls `plan->compile(deviceProps)` to compile
   the GPU kernel via HIPRTC.

5. `Plan::execute()` reads device pointers from the variant pack buffers
   (matched by tensor UID) and launches the pre-compiled GPU kernel on the
   specified HIP stream.

### HIPRTC Compilation Flow

```
Kernel Source File (e.g., kernels/relu/ReluForward.cpp)
        │
        ▼  CMake configure time
Embedded as C++ string literal (kernel_sources.cpp.in template)
        │
        ▼  Plan::compile() at runtime
HipKernelCompiler::compile(filename, options)
  → hiprtcCreateProgram() with embedded source
  → hiprtcCompileProgram() with --offload-arch=gfxNNN
  → hiprtcGetCode() extracts compiled binary
  → hipModuleLoadData() loads binary as HIP module
        │
        ▼
HipCompiledProgram::getRunnableKernel(kernelFunctionName)
  → hipModuleGetFunction() extracts kernel function
        │
        ▼
IRunnableKernel::launch(stream, args...)
  → hipModuleLaunchKernel() executes on GPU
```

### DI Interfaces for Testability

The HIPRTC infrastructure is abstracted behind dependency-injection interfaces,
enabling unit tests to run without GPU hardware:

| Interface | Production Implementation | Test Mock |
|---|---|---|
| `IKernelCompiler` | `HipKernelCompiler` | `MockKernelCompiler` |
| `ICompiledProgram` | `HipCompiledProgram` | `MockCompiledProgram` |
| `IRunnableKernel` | `HipRunnableKernel` | `MockRunnableKernel` |
| `IDevicePropertyProvider` | `CurrentDevicePropertyProvider` | `MockDevicePropertyProvider` |

The Container creates the production implementations at construction time and
passes them to engine factory lambdas.  Unit tests substitute the mocks.

## Custom Knobs

The ReLU engine demonstrates the full custom knob lifecycle with
`example.relu.negative_slope`:

1. **`getCustomKnobs()`** (PlanBuilder) defines the knob: `FLOAT64`, default
   `0.0`, range `[0.0, 1.0]`.  At `0.0`, standard ReLU; at `>0`, leaky ReLU
   (`output = x >= 0 ? x : slope * x`).

2. **Frontend exposes** the knob via `graph->get_knobs_for_engine()` after
   building execution plans.

3. **User sets** the value via `KnobSetting` on the engine config.

4. **`initializeExecutionSettings()`** reads the value from `IEngineConfig`
   into the `Settings` struct.

5. **`buildPlan()`** passes the setting to the Plan constructor.

6. **`execute()`** passes `negativeSlope` as a kernel argument.

The ConvFwd engine has no custom knobs (`getCustomKnobs()` returns empty).

## Plugin Loading

hipDNN supports three ways to load plugins:

### 1. HIPDNN_PLUGIN_DIR Environment Variable

Set before creating a hipDNN handle.  hipDNN scans this as the default directory for loading plugin
shared libraries (`.so` on Linux, `.dll` on Windows).

```bash
export HIPDNN_PLUGIN_DIR=/path/to/plugin/directory
```

### 2. Programmatic: ADDITIVE Mode

Load additional plugin directories alongside the system-installed plugins.
This is the default mode.

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;

std::vector<std::string> paths = {"/path/to/my/plugins"};
auto err = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ADDITIVE);

hipdnnHandle_t handle;
hipdnnCreate(&handle);
```

### 3. Programmatic: ABSOLUTE Mode

Replace all plugin search paths.  Only the specified directories are searched;
system-installed plugins are ignored.

```cpp
std::vector<std::string> paths = {"/path/to/my/plugins"};
auto err = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ABSOLUTE);

hipdnnHandle_t handle;
hipdnnCreate(&handle);
```

### Path Resolution

hipDNN resolves plugin paths as follows:

**Relative paths** are resolved against the directory containing
`libhipdnn_backend.so` (NOT the current working directory).  For example, if
the backend library is loaded from `/opt/rocm/lib/libhipdnn_backend.so`, then
`HIPDNN_PLUGIN_DIR=my_plugins` resolves to `/opt/rocm/lib/my_plugins/`.

**Absolute paths** are used as-is after canonicalization.

When a **plugin file** (not a directory) is specified:

- If the file has a `.so` (Linux) or `.dll` (Windows) extension, it is loaded
  directly.
- If the file has no extension, hipDNN adds the platform-appropriate prefix and
  extension: `lib` prefix + `.so` suffix on Linux, `.dll` suffix on Windows.
- If the file has an incorrect extension (e.g., `.so` on Windows or `.dll` on
  Linux), it is rejected with an error.

## Engine Selection

By default, hipDNN selects the best engine using heuristic ranking.  To force
a specific engine, use `set_preferred_engine_id_ext()` on the graph before
building:

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend::graph;

auto graph = std::make_shared<Graph>();
// ... configure graph ...

// Select engine by name (string is hashed to the engine ID at runtime)
graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_RELU_ENGINE");
// or: graph->set_preferred_engine_id_ext("EXAMPLE_PLUGIN_CONV_FWD_ENGINE");

graph->build(handle);
```

You can also query available engines after building the operation graph:

```cpp
graph->validate();
graph->build_operation_graph(handle);

std::vector<int64_t> engineIds;
graph->get_ranked_engine_ids(engineIds);

// engineIds contains all applicable engine IDs ranked by heuristic score
```

## How to Add a New Operation

The ConvFwd engine serves as a concrete example of adding a second operation
alongside ReLU.  The general pattern:

1. **Create a GPU kernel** in `kernels/<operation>/<KernelName>.cpp`:
   - Use `extern "C" __global__` for HIPRTC symbol lookup
   - Keep the kernel focused; preprocessing goes in the Plan

2. **Register the kernel** in `kernels/CMakeLists.txt` via the
   `embed_kernel_sources()` function so it is embedded at configure time.

3. **Create a PlanBuilder** in `src/engines/plans/`:
   - Accept `IKernelCompiler&` and `IDevicePropertyProvider&` via constructor
   - Implement `isApplicable()` to match the graph's node attributes
   - Implement `buildPlan()` to extract tensor metadata, create a Plan, and
     call `plan->compile(deviceProps.getDeviceProperties())`

4. **Create a Plan** in `src/engines/plans/`:
   - Inherit `ICompilablePlan<ExamplePluginHandle>` from the plugin SDK
   - Implement `compile()` to build HIPRTC options, compile the kernel, and
     set block/grid sizes
   - Implement `execute()` to extract device pointers by tensor UID and
     launch the kernel

5. **Register the engine** in `ExamplePluginContainer.cpp`:
   ```cpp
   HIPDNN_REGISTER_ENGINE(YOUR_ENGINE, "YOUR_ENGINE")

   // In getEngineDefinitions():
   {YOUR_ENGINE_ID,
    [](const IKernelCompiler& compiler, const IDevicePropertyProvider& deviceProps) {
        auto engine = std::make_unique<ExamplePluginEngine>(YOUR_ENGINE_ID);
        engine->addPlanBuilder(std::make_unique<YourPlanBuilder>(compiler, deviceProps));
        return engine;
    }},
   ```

6. **Add unit tests** in `tests/` for `isApplicable()`, `compile()`, and
   `execute()`, using mock objects instead of real GPU hardware.

7. **Update `src/CMakeLists.txt`** to include the new source files.

## Technical Details

### Why `-fvisibility=hidden` Is Critical

The plugin shared library uses `CXX_VISIBILITY_PRESET hidden`.  Without this,
all internal symbols (including C++ standard library and SDK symbols) are
exported from the plugin `.so`.  When multiple plugins are loaded into the same
process, their exported symbols can collide and cause unpredictable behavior
(wrong function called, ABI mismatches, crashes).  With hidden visibility, only
the explicitly exported C API entry points are visible.

### Why Position-Independent Code (PIC) Is Required

Shared libraries loaded via `dlopen()` / `LoadLibrary()` must be compiled with
position-independent code (`-fPIC` on GCC, default on MSVC).  CMake's
`CMAKE_POSITION_INDEPENDENT_CODE ON` ensures this.  Without PIC, the dynamic
linker cannot relocate the code to an arbitrary address, and `dlopen()` will
fail.  Additionally, thread-local storage (TLS) models differ between PIC and
non-PIC code; mixing them causes linker errors.

### `RTLD_NOW | RTLD_LOCAL` Loading Behavior

hipDNN loads plugins with `dlopen(path, RTLD_NOW | RTLD_LOCAL)` on Linux:

- **`RTLD_NOW`** forces immediate resolution of ALL symbols.  If any dependency
  (including `libhiprtc.so`) cannot be found, the plugin fails to load
  entirely.  This is a deliberate design choice: a plugin either loads
  completely or not at all.  hipDNN logs the error and continues without the
  plugin.

- **`RTLD_LOCAL`** prevents the plugin's symbols from being visible to other
  shared libraries in the process.  This isolates plugins from each other,
  preventing symbol pollution.

On Windows, `LoadLibraryW()` provides similar behavior with its default
DLL search order.

### Runtime Dependency Resolution

The plugin links against `hiprtc::hiprtc`, making `libhiprtc.so` a transitive
dependency of the plugin `.so`.  **The user's application does NOT need to link
against hiprtc** -- when hipDNN loads the plugin via `dlopen()`, the dynamic
linker resolves `libhiprtc.so` independently from the user's application binary.

The plugin achieves this via RPATH embedded in the `.so`:

```cmake
set_target_properties(example_plugin PROPERTIES
    INSTALL_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)
```

- `INSTALL_RPATH "${ROCM_PATH}/lib"` -- tells the dynamic linker where to find
  `libhiprtc.so` at runtime
- `INSTALL_RPATH_USE_LINK_PATH TRUE` -- automatically adds directories of
  linked libraries to RPATH
- `BUILD_WITH_INSTALL_RPATH TRUE` -- the plugin works from the build tree
  without needing `LD_LIBRARY_PATH`

To customize the ROCm path:

```bash
cmake .. -DROCM_PATH=/custom/rocm/path
```

On Windows, `hiprtc.dll` must be findable via the system `PATH` or placed
alongside the plugin DLL.

#### Troubleshooting Plugin Loading

If the plugin fails to load silently (no engines from this plugin appear):

1. Check library dependencies:
   ```bash
   ldd build/src/libexample_plugin.so
   ```
   All dependencies should resolve.  Look for `not found` entries.

2. Trace the dynamic linker's search:
   ```bash
   LD_DEBUG=libs your_application 2>&1 | grep example_plugin
   ```

3. Verify RPATH is embedded:
   ```bash
   readelf -d build/src/libexample_plugin.so | grep RPATH
   ```

### RPATH Configuration

ROCm libraries (including `libhiprtc.so`) are typically installed in
`/opt/rocm/lib`, which is NOT registered with `ldconfig` and is not in the
default library search path.  Without RPATH, the dynamic linker cannot find
`libhiprtc.so` when `dlopen()` loads the plugin, causing a silent load failure.

The RPATH solution matches the pattern used by hipDNN's production plugins
(miopen-provider and hip-kernel-provider).

### Windows spdlog Caution

On Windows, if the plugin uses spdlog (brought in transitively by the plugin
SDK), `spdlog::shutdown()` must be called before static destructors run.  On
Linux this is handled automatically, but on Windows the DLL unload order can
cause the spdlog registry to be destroyed before the plugin's static loggers,
leading to crashes during process exit.

## Extending for Real-World Use

This example uses a naive convolution kernel and single-precision floats for
simplicity.  To build a production plugin:

- **Support multiple data types**: Check `TensorAttributes::data_type()` in
  `isApplicable()` and `buildPlan()` to handle FLOAT, HALF, BFLOAT16, etc.
  The naive kernels only support FLOAT.

- **Optimize GPU kernels**: The naive convolution kernel (one thread per output
  element, no shared memory, no tiling) is deliberately simple for educational
  purposes.  Production convolutions should use shared memory tiling, register
  blocking, and vectorized loads.  See MIOpen for optimized implementations.

- **Add workspace management**: Return non-zero from `getMaxWorkspaceSize()`
  if your engine needs temporary scratch memory.  hipDNN allocates the
  workspace and passes it to `execute()`.

- **Implement custom knobs**: Override `getCustomKnobs()` in your PlanBuilder
  to expose tuning parameters (e.g., tile sizes, algorithm variants).

- **Support multi-node graphs**: Extend `isApplicable()` to match fused
  operation patterns (e.g., Conv + BiasAdd + ReLU).  The `AdvancedEngineScaffold`
  demonstrates the multi-node matching pattern.

- **Add Windows support**: The CMake project uses generator expressions for
  cross-platform compiler flags.  Verify with MSVC and adjust as needed.

## Further Reading

- `docs/PluginDevelopment.md` -- detailed plugin development guide
- `docs/Knobs.md` -- custom knob system documentation
- `docs/HowTo.md` -- hipDNN how-to guides
