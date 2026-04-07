# hipDNN Example Plugin

A self-contained example project that demonstrates how to build a hipDNN engine
plugin to extend hipDNN with custom GPU-accelerated engines.

The example implements two GPU operations compiled at runtime via HIPRTC (HIP
Runtime Compilation):

- **ReLU forward** (pointwise): element-wise `max(0, x)` with a custom
  `example.relu.negative_slope` knob for leaky ReLU support
- **Convolution forward** (naive): 2D cross-correlation, NCHW layout, single
  thread per output element

## Prerequisites

| Dependency | Purpose | Notes |
|---|---|---|
| CMake >= 3.20 | Build system | |
| C++17 compiler | GCC/G++ or MSVC | No GPU compiler needed at build time |
| ROCm (HIP SDK + HIPRTC) | GPU kernel compilation and execution | `hipStream_t`, `hipMalloc`, HIPRTC APIs |
| hipDNN (installed) | Plugin SDK, data SDK, frontend library | Typically installed at `/opt/rocm` (Linux) |
| GPU hardware | Runtime execution of HIPRTC-compiled kernels | Any ROCm-supported GPU |
| Internet access | GTest is downloaded via CMake `FetchContent` | Only needed for the first build |

The plugin C++ source code compiles with standard compilers (GCC, MSVC). GPU
kernels are plain `.cpp` files that are embedded as string literals at CMake
configure time and compiled at runtime by HIPRTC. No GPU compiler (`hipcc`,
`amdclang++`) is needed during the build.

## Directory Structure

```
example_engine_plugin/
├── CMakeLists.txt                       # Root CMake: project options, dependencies
├── README.md                            # This file
├── kernels/                             # GPU kernel source files (embedded at configure time)
│   ├── CMakeLists.txt                   # embed_kernel_sources() function
│   ├── cmake/
│   │   └── EmbedKernelSources.cmake     # CMake kernel embedding function
│   ├── templates/                       # .in templates for kernel embedding
│   │   ├── kernel_sources.cpp.in
│   │   ├── kernel_sources.hpp.in
│   │   ├── kernel_includes.cpp.in
│   │   └── kernel_includes.hpp.in
│   ├── common/
│   │   └── IndexType.hpp                # Shared kernel header (embedded for HIPRTC #include)
│   ├── relu/
│   │   └── ReluForward.cpp              # ReLU GPU kernel (~10 lines)
│   └── conv/
│       └── ConvForwardNaive.cpp         # Naive ConvFwd GPU kernel (~35 lines)
├── src/
│   ├── CMakeLists.txt                   # OBJECT, static, and shared library targets
│   ├── ExampleProviderPluginPublic.cpp     # C entry points (5 macros + EnginePluginImpl.inl)
│   ├── ExampleProviderContainer.hpp/cpp   # Engine registration and EngineManager
│   ├── ExampleProviderHandle.hpp/cpp      # Plugin handle (stream, container reference)
│   ├── ExampleProviderContext.hpp         # Execution context
│   ├── ExampleProviderSettings.hpp        # Execution settings (reluNegativeSlope)
│   ├── hip/                             # HIPRTC infrastructure (DI interfaces + impls)
│   │   ├── IKernelCompiler.hpp          # Interface: compile(filename, options)
│   │   ├── ICompiledProgram.hpp         # Interface: getRunnableKernel(name)
│   │   ├── IRunnableKernel.hpp          # Interface: launch(stream, args...)
│   │   ├── HipUtils.hpp                # HIP_CHECK and HIPRTC_CHECK error macros
│   │   ├── HipKernelCompiler.hpp        # Concrete IKernelCompiler (HIPRTC, handles --offload-arch)
│   │   ├── HipCompiledProgram.hpp/cpp   # Concrete ICompiledProgram (HIPRTC compilation + module)
│   │   └── HipRunnableKernel.hpp/cpp    # Concrete IRunnableKernel (hipFunction_t)
│   └── engines/
│       ├── ExampleProviderEngine.hpp/cpp  # Engine: owns PlanBuilders, delegates isApplicable
│       ├── ExampleProviderUtils.hpp       # Utility: UID-to-buffer lookup
│       └── plans/
│           ├── ReluParams.hpp           # ReLU plan parameter struct
│           ├── ReluPlanBuilder.hpp/cpp  # PlanBuilder: graph matching for ReLU_FWD
│           ├── ReluPlan.hpp/cpp         # Plan: GPU ReLU execution via HIPRTC
│           ├── ConvFwdParams.hpp        # ConvFwd plan parameter struct
│           ├── ConvFwdPlanBuilder.hpp/cpp  # PlanBuilder: graph matching for ConvFwd
│           └── ConvFwdPlan.hpp/cpp      # Plan: GPU ConvFwd execution via HIPRTC
├── tests/                               # Unit tests (GTest, no GPU required)
│   ├── CMakeLists.txt
│   ├── TestHelpers.hpp                  # FlatBuffer graph construction helpers
│   ├── mocks/                           # Mock objects for GPU-free unit testing
│   │   ├── MockKernelCompiler.hpp
│   │   ├── MockCompiledProgram.hpp
│   │   └── MockRunnableKernel.hpp
│   ├── TestExampleProviderContainer.cpp
│   ├── TestReluPlanBuilder.cpp
│   ├── TestReluPlan.cpp
│   ├── TestConvFwdPlanBuilder.cpp
│   └── TestConvFwdPlan.cpp
└── sample/                              # Demo app + acceptance test
    ├── CMakeLists.txt
    └── ExampleProviderSample.cpp
```

## Build Instructions

Run these commands from the example_engine_plugin folder.

### Linux (GCC)

```bash
cmake -B build -DCMAKE_PREFIX_PATH="/opt/rocm"
cmake --build build
```

Run all tests, including the example sample app:

```bash
ctest --test-dir build
```

Run the sample application (requires GPU for full execution):

```bash
ctest --test-dir build -R example_provider_sample
```

The tests and sample can also be run directly:

```bash
./build/bin/example_provider_tests
```
```bash
./build/bin/example_provider_sample
```

Install the plugin:

```bash
cmake --install build --prefix /opt/rocm
# Plugin .so is installed to <prefix>/lib/hipdnn_plugins/engines/
```

### Windows

Ensure that the ROCm install bin folder is in your system PATH. E.g.

```powershell
set PATH=C:\ROCm\bin;%PATH%
```

### Windows (MSVC)

With the ROCm bin folder in your system PATH:

```powershell
cmake -B build -G "Visual Studio 17 2022"
cmake --build build --config Release
ctest --test-dir build --build-config Release
```

The tests and sample can also be run directly:

```powershell
.\build\bin\Release\example_provider_tests.exe
```
```powershell
.\build\bin\Release\example_provider_sample.exe
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `HIPDNN_EXAMPLE_PROVIDER_BUILD_UNIT_TESTS` | `ON` | Build unit tests (no GPU required) |
| `HIPDNN_EXAMPLE_PROVIDER_BUILD_SAMPLE` | `ON` | Build sample application (serves as acceptance test via `ctest`) |
| `ROCM_PATH` | `/opt/rocm` | ROCm installation path (for RPATH and library discovery) |

To build only the plugin library (no tests or sample):

```bash
cmake .. -DHIPDNN_EXAMPLE_PROVIDER_BUILD_UNIT_TESTS=OFF \
         -DHIPDNN_EXAMPLE_PROVIDER_BUILD_SAMPLE=OFF
```

## Architecture

A hipDNN plugin is a shared library that implements a C API defined by the
plugin SDK. The SDK provides `EnginePluginImpl.inl`, which generates all
required C entry points when five macros are defined in
`ExampleProviderPluginPublic.cpp`:

- `HIPDNN_PLUGIN_NAME` -- display name string
- `HIPDNN_PLUGIN_VERSION` -- version string
- `HIPDNN_PLUGIN_CONTAINER_TYPE` -- fully qualified Container class name
- `HIPDNN_PLUGIN_HANDLE_TYPE` -- fully qualified Handle struct name
- `HIPDNN_PLUGIN_CONTEXT_TYPE` -- fully qualified Context struct name

### Type Hierarchy

```
Container
├── Owns EngineManager<Handle, Settings, Context>
├── Owns IKernelCompiler (HipKernelCompiler)
├── Creates engines defined via getEngineDefinitions()
│   ├── Engine (EXAMPLE_PROVIDER_RELU_ENGINE)
│   │   └── PlanBuilder (ReluPlanBuilder)
│   │       └── Plan (ReluPlan)
│   └── Engine (EXAMPLE_PROVIDER_CONV_FWD_ENGINE)
│       └── PlanBuilder (ConvFwdPlanBuilder)
│           └── Plan (ConvFwdPlan)
└── copyEngineIds() -- returns registered engine IDs to hipDNN

Handle
├── Holds shared_ptr<Container>
├── setStream(hipStream_t)
└── getEngineManager()

```

The `hip/` directory contains the HIPRTC abstraction layer (`IKernelCompiler`,
`ICompiledProgram`, `IRunnableKernel`, and their concrete implementations).
This layer is independent of any specific operation. Developers copy it as-is
and update only the namespace.

### Engine Execution Flow

1. **Container** creates engines and returns available engines IDs.

2. hipDNN calls `isApplicable()` on each engine to check whether it supports a
   given operation graph.

3. The engine delegates to its **PlanBuilders**. Each PlanBuilder inspects the
   graph's node attributes (e.g., `PointwiseAttributes` with
   `PointwiseMode::RELU_FWD`, or `ConvolutionFwdAttributes` with
   `ConvMode::CROSS_CORRELATION`).

4. `buildPlan()` extracts tensor metadata (UIDs, dimensions) from the graph,
   creates a **Plan** object, and calls `plan->compile(kernelCompiler)` to
   compile the GPU kernel via HIPRTC.

5. `Plan::execute()` reads device pointers from the variant pack buffers
   (matched by tensor UID) and launches the compiled GPU kernel on the
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

### GPU Kernel Compilation: Source, Module, Kernel

GPU kernel compilation follows a three-stage pipeline. A GPU source file can
define multiple kernel functions (each marked `__global__`), but the entire file
is compiled as a single unit. The compiled binary is loaded as a module, and
individual kernels are then extracted from it by name. The three DI interfaces
(`IKernelCompiler`, `ICompiledProgram`, `IRunnableKernel`) model each stage
directly.

Note that compilation is distinct from the source embedding described in the
HIPRTC Compilation Flow above. At CMake configure time, all kernel `.cpp` files
are embedded as C++ string literals into a generated source registry. This is
just text storage, not GPU compilation. At runtime, each Plan compiles only its
own kernel source file via HIPRTC, and only when its engine is selected for a
graph.

**Stage 1: Source compilation.** A GPU source file (e.g., `ReluForward.cpp`) is
compiled into a binary blob at runtime via HIPRTC. The result is loaded as a HIP
module (`hipModule_t`), analogous to a `.so` or `.dll` containing compiled code
for all `__global__` functions in that source file.
`IKernelCompiler::compile()` performs this step and returns an
`ICompiledProgram`.

**Stage 2: Kernel extraction.** Individual kernel functions are extracted from
the loaded module by name via `hipModuleGetFunction`. A single module can contain multiple kernels:

```cpp
auto module = compiler.compile("MyKernels.cpp", options);
auto addKernel = module->getRunnableKernel("add_vectors");
auto mulKernel = module->getRunnableKernel("multiply_vectors");
```

**Stage 3: Kernel launch.** The extracted kernel is configured (block size, grid
size, shared memory) and launched on a HIP stream via
`IRunnableKernel::launch()`.

**Module lifetime matters.** The kernel function pointer (`hipFunction_t`) is
only valid while its module remains loaded. Each Plan holds both
`_compiledProgram` (keeps the module loaded) and `_kernel` (function pointer
into the module). The `_compiledProgram` is never accessed after `compile()`
completes; it exists solely to prevent the module from being unloaded.

In this example plugin, each source file contains exactly one kernel. The
three-stage structure is preserved because it models the HIP runtime API and
prepares developers for the general case.

### Dependency Injection Interfaces for Testability

The HIPRTC infrastructure is abstracted behind dependency-injection interfaces,
enabling unit tests to run without GPU hardware:

| Interface | Production Implementation | Test Mock |
|---|---|---|
| `IKernelCompiler` | `HipKernelCompiler` | `MockKernelCompiler` |
| `ICompiledProgram` | `HipCompiledProgram` | `MockCompiledProgram` |
| `IRunnableKernel` | `HipRunnableKernel` | `MockRunnableKernel` |

## Using This Example as a Template

### Step-by-Step Adaptation Workflow

1. **Choose a name for your plugin**: Pick a short, descriptive name that
   identifies the technology or backend your plugin provides (e.g.,
   `rocblas_conv`, `custom_gemm`, `your_name`). This name will be
   used throughout the plugin as:
   - **Class prefix**: `ExampleProvider*` becomes `YourName*` (e.g.,
     `YourNameContainer`, `YourNameHandle`, `YourNameEngine`)
   - **Namespace**: `example_provider` becomes `your_name`
   - **Engine identifiers**: `EXAMPLE_PROVIDER_RELU_ENGINE` becomes
     `YOUR_NAME_xxx_ENGINE` (e.g., `YOUR_NAME_CONV_ENGINE`).
     These names are visible to applications that select engines, so choose
     something meaningful.
   - **Plugin display name**: The `HIPDNN_PLUGIN_NAME` macro value (e.g.,
     `"Your Name xxx engine"`)

2. **Copy and rename the directory**: Copy `example_engine_plugin/` to your new
   plugin directory (e.g., `your_name-provider/`).

3. **Verify the build on your system**: Before making any code changes, build
   the plugin and run the tests from your new plugin directory to confirm
   the example plugin builds correctly in your environment and that tests
   pass successfully. Resolve any build issues such as missing dependencies,
   incorrect paths, or toolchain incompatibilities, before continuing. This
   ensures that any issues encountered later are caused by changes made to the
   code or project files and not by the build & test environment.

4. **Rename classes**: Replace all `ExampleProvider*` class names with your
   plugin prefix (e.g., `YourNameKernel*`). This affects `Container`,
   `Handle`, `Context`, `Settings`, `Engine`, and `Public`.

5. **Update the namespace**: Change the `example_provider` namespace to your
   plugin's namespace throughout all source files.

6. **Update the 5 macros** in `ExampleProviderPluginPublic.cpp`: Set
   `HIPDNN_PLUGIN_NAME` to your plugin's display name and generate new unique
   values for the four type macros.

7. **Replace example PlanBuilders and Plans**: Remove `ReluPlanBuilder`,
   `ReluPlan`, `ReluParams`, `ConvFwdPlanBuilder`, `ConvFwdPlan`, and
   `ConvFwdParams`. Create your own PlanBuilder, Plan, and Params for each
   operation your plugin supports. Plans inherit
   `ICompilablePlan<YourPluginHandle>` from the plugin SDK. Key methods:
   `isApplicable()`, `getCustomKnobs()`, `buildPlan()`, `compile()`,
   `execute()`. To add further operations later, repeat this step and
   steps 8-11.

8. **Write your GPU kernels**: Replace the kernel source files in `kernels/`
   with your own. Each kernel must use `extern "C" __global__` and include
   only HIPRTC-compatible headers.

9. **Update CMake targets and kernel file list**: Update `KERNEL_FILES` in
   `kernels/CMakeLists.txt` with your kernel filenames. Update the source
   file list in `src/CMakeLists.txt` with your `.cpp` files.

10. **Register your engines**: Update `ExampleProviderContainer.cpp` to register
    your engines via `HIPDNN_REGISTER_ENGINE` with unique engine names and add
    lambdas to create the new engines:

    ```cpp
    HIPDNN_REGISTER_ENGINE(YOUR_ENGINE, "YOUR_ENGINE")

    // In getEngineDefinitions():
    {YOUR_ENGINE_ID,
     [](const IKernelCompiler& compiler) {
         auto engine = std::make_unique<ExampleProviderEngine>(YOUR_ENGINE_ID);
         engine->addPlanBuilder(std::make_unique<YourPlanBuilder>(compiler));
         return engine;
     }},
    ```

11. **Build and run unit tests**: Follow the [Build Instructions](#build-instructions)
    to verify successful compilation and tests.

### File Classification

Comment markers are used to identify files that will be modified when using this
example as a template for creating a new plugin. Look for `TEMPLATE ADAPTATION`
and `TEMPLATE REFERENCE` comment markers in the source files for per-file guidance.

| Files | Marker | What to Do |
|-------|--------|------------|
| `ExampleProviderPluginPublic.cpp`, `ExampleProviderContainer.hpp/cpp`, `ExampleProviderHandle.hpp/cpp`, `ExampleProviderContext.hpp`, `ExampleProviderSettings.hpp`, `ExampleProviderEngine.hpp/cpp`, `ExampleProviderUtils.hpp` | `TEMPLATE ADAPTATION` | Rename `ExampleProvider` to `YourPlugin`. Adjust class names, namespace, and includes. These files are framework plumbing; the structure stays the same. |
| `hip/IKernelCompiler.hpp`, `hip/ICompiledProgram.hpp`, `hip/IRunnableKernel.hpp`, `hip/HipKernelCompiler.hpp`, `hip/HipCompiledProgram.hpp/cpp`, `hip/HipRunnableKernel.hpp/cpp`, `hip/HipUtils.hpp` | *(none)* | Update namespace only. These implement the HIPRTC compilation pipeline and do not contain operation-specific logic. |
| `engines/plans/ReluPlanBuilder.hpp/cpp`, `engines/plans/ReluPlan.hpp/cpp`, `engines/plans/ReluParams.hpp` | `TEMPLATE REFERENCE` | Study to learn the PlanBuilder/Plan pattern, then replace with your own operation's PlanBuilder, Plan, and Params. Key methods: `isApplicable()`, `getCustomKnobs()`, `initializeExecutionSettings()`, `buildPlan()`, `compile()`, `execute()`. |
| `engines/plans/ConvFwdPlanBuilder.hpp/cpp`, `engines/plans/ConvFwdPlan.hpp/cpp`, `engines/plans/ConvFwdParams.hpp` | `TEMPLATE REFERENCE` | Second example of the same pattern. Compare with ReLU to see how different operations handle graph matching, parameters, and kernel launch. |
| `kernels/relu/ReluForward.cpp`, `kernels/conv/ConvForwardNaive.cpp` | *(none)* | Replace with your GPU kernel source files. Each kernel must use `extern "C" __global__` and include only HIPRTC-compatible headers. |
| `kernels/CMakeLists.txt` | `TEMPLATE ADAPTATION` | Update `KERNEL_FILES` list with your kernel filenames. |
| `tests/TestReluPlanBuilder.cpp`, `tests/TestReluPlan.cpp`, `tests/TestConvFwdPlanBuilder.cpp`, `tests/TestConvFwdPlan.cpp` | `TEMPLATE REFERENCE` | Study the testing patterns, then write equivalent tests for your operations. |
| `tests/TestHelpers.hpp` | `TEMPLATE ADAPTATION` | As preferred, replace `createReluFwdGraph()` / `createConvFwdGraph()` with helpers that build your operation's FlatBuffer graphs. Keep `createEngineConfig()`. |
| `tests/mocks/MockKernelCompiler.hpp`, `tests/mocks/MockCompiledProgram.hpp`, `tests/mocks/MockRunnableKernel.hpp` | *(none)* | As preferred, copy into your test directory. Update namespace only. These mocks implement the interfaces for GPU-free unit testing. |
| `sample/ExampleProviderSample.cpp` | `TEMPLATE ADAPTATION` | As preferred, adapt scenarios to exercise your operations. Keep the plugin loading and engine selection patterns; replace the graph construction and verification logic. This file can alternatively be replaced with a suite of integration tests or a custom application. |

## Testing Your Plugin

### Unit Testing Architecture

Unit tests run without GPU hardware using the DI interfaces and mocks described
in [Dependency Injection Interfaces for Testability](#dependency-injection-interfaces-for-testability).

### What to Test in a PlanBuilder

See `tests/TestReluPlanBuilder.cpp` for the complete pattern.

- **`isApplicable()`** -- returns `true` for matching operation graphs, `false`
  for non-matching (wrong operation type, wrong node count, wrong data type)
- **`getCustomKnobs()`** -- returns the correct knob definitions (IDs, types,
  ranges, defaults)
- **`getMaxWorkspaceSize()`** -- returns expected workspace bytes
- **`buildPlan()`** -- sets a valid plan on the execution context (mock
  expectations verify the correct kernel filename and function name are used)

### What to Test in a Plan

See `tests/TestReluPlan.cpp` for the complete pattern.

- **`compile()`** -- calls the compiler with the correct kernel filename and
  extracts the correct kernel function name
- **`execute()`** -- sets correct grid/block dimensions and launches the kernel
  (verified via mock expectations)
- **`getWorkspaceSize()`** -- returns expected bytes
- **Error handling** -- missing device buffers throw `HipdnnPluginException`

### Acceptance Testing

The sample application (`sample/ExampleProviderSample.cpp`) serves as the
acceptance test. It is registered as a `ctest` and verifies end-to-end
correctness on GPU hardware. When writing a new plugin, you can adapt the
sample scenarios to exercise your operations with correctness verification.
This file can alternatively be replaced with a suite of integration tests or
a custom application.

### Running Tests

See [Build Instructions](#build-instructions) for build and test commands.

## Integrating the Plugin into Your Application

### Plugin Loading

Ensuring the plugin is loaded is the **application's** responsibility, not the
plugin's. The plugin developer builds a shared library (`.so` / `.dll`) and
ensures it is placed in a discoverable location.

By default, hipDNN loads all plugins in the ROCm install
`/lib/hipdnn_plugins/engines` folder. There are three ways to override this:

**Environment variable** (`HIPDNN_PLUGIN_DIR`): Set before creating a hipDNN
handle. This becomes the new default plugin directory that hipDNN scans for
plugin shared libraries (`.so` on Linux, `.dll` on Windows).

```bash
export HIPDNN_PLUGIN_DIR=/path/to/plugin/directory
```

**ADDITIVE mode**: Load additional plugin directories alongside any paths
already specified, including the hipDNN default plugin directory.

```cpp
#include <hipdnn_frontend.hpp>
using namespace hipdnn_frontend;
std::vector<std::string> paths = {"/path/to/my/plugins"};
auto err = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ADDITIVE);
```

**ABSOLUTE mode**: Replace all plugin search paths. Only the specified
directories are searched; system-installed plugins are ignored.

```cpp
std::vector<std::string> paths = {"/path/to/my/plugins"};
auto err = setEnginePluginPaths(paths, PluginLoadingMode::MODE_ABSOLUTE);
```

### Path Resolution

hipDNN resolves plugin paths as follows:

**Relative paths** are resolved against the directory containing
`libhipdnn_backend.so` (NOT the current working directory). For example, if
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

### Verifying the Plugin Is Loaded

After creating a hipDNN handle, query loaded plugins to confirm yours is
present:

```cpp
auto paths = getLoadedEnginePluginPaths();
for (const auto& path : paths) {
    std::cout << "Loaded: " << path << std::endl;
}
```

### Engine Selection

By default, hipDNN selects the best engine using heuristic ranking. To force
a specific engine, use `set_preferred_engine_id_ext()` on the graph before
building:

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend::graph;

auto graph = std::make_shared<Graph>();
// ... configure graph ...

// Select engine by name (string is hashed to the engine ID at runtime)
graph->set_preferred_engine_id_ext("EXAMPLE_PROVIDER_RELU_ENGINE");
// or: graph->set_preferred_engine_id_ext("EXAMPLE_PROVIDER_CONV_FWD_ENGINE");

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

### Install Location

The plugin `.so` is installed to
`${CMAKE_INSTALL_PREFIX}/lib/hipdnn_plugins/engines/` by default (configurable
via `HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR`).

See `sample/ExampleProviderSample.cpp` for a complete example showing plugin
loading, engine selection, knob modification, and correctness verification.

## Quick Checklist

- [ ] Copy and rename `example_engine_plugin/` directory
- [ ] Perform a preliminary build and test runs to verify environment.
- [ ] Rename all `ExampleProvider*` classes to `YourPlugin*`
- [ ] Update namespace from `example_provider`
- [ ] Update the 5 macros in `Public.cpp`
- [ ] Write your GPU kernel(s) in `kernels/`
- [ ] Update `KERNEL_FILES` in `kernels/CMakeLists.txt`
- [ ] Implement your PlanBuilder (`isApplicable`, `getCustomKnobs`, `buildPlan`)
- [ ] Implement your Plan (`compile`, `execute`, `getWorkspaceSize`)
- [ ] Create your Params struct
- [ ] Register your engine (`HIPDNN_REGISTER_ENGINE`) and create in Container
- [ ] Update `ExampleProviderSettings` with your settings fields
- [ ] Write unit tests for PlanBuilder and Plan
- [ ] Create graph construction helpers in `TestHelpers.hpp`
- [ ] Build and verify: `cmake --workflow --preset release`
- [ ] Adapt sample app scenarios and verify on GPU

## Custom Knobs

The ReLU engine demonstrates the full custom knob lifecycle with
`example.relu.negative_slope`:

1. **`getCustomKnobs()`** (PlanBuilder) defines the knob: `FLOAT64`, default
   `0.0`, range `[0.0, 1.0]`. At `0.0`, standard ReLU; at `>0`, leaky ReLU
   (`output = x >= 0 ? x : slope * x`).

2. **Frontend exposes** the knob via `graph->get_knobs_for_engine()` after
   building execution plans.

3. **User sets** the value via `KnobSetting` on the engine config.

4. **`initializeExecutionSettings()`** reads the value from `IEngineConfig`
   into the `Settings` struct.

5. **`buildPlan()`** passes the setting to the Plan constructor.

6. **`execute()`** passes `negativeSlope` as a kernel argument.

The ConvFwd engine has no custom knobs (`getCustomKnobs()` returns empty).

## Technical Details

### Why Position-Independent Code (PIC) Is Required

Shared libraries loaded via `dlopen()` / `LoadLibrary()` must be compiled with
position-independent code (`-fPIC` on GCC, default on MSVC). CMake's
`CMAKE_POSITION_INDEPENDENT_CODE ON` ensures this. Without PIC, the dynamic
linker cannot relocate the code to an arbitrary address, and `dlopen()` will
fail. Additionally, thread-local storage (TLS) models differ between PIC and
non-PIC code; mixing them causes linker errors.

### `RTLD_NOW | RTLD_LOCAL` Loading Behavior

hipDNN loads plugins with `dlopen(path, RTLD_NOW | RTLD_LOCAL)` on Linux:

- **`RTLD_NOW`** forces immediate resolution of ALL symbols. If any dependency
  (including `libhiprtc.so`) cannot be found, the plugin fails to load
  entirely. This is a deliberate design choice: a plugin either loads
  completely or not at all. hipDNN logs the error and continues without the
  plugin.

- **`RTLD_LOCAL`** prevents the plugin's symbols from being visible to other
  shared libraries in the process. This isolates plugins from each other,
  preventing symbol pollution.

On Windows, `LoadLibraryW()` provides similar behavior with its default
DLL search order.

### Runtime Dependency Resolution and RPATH

ROCm libraries (including `libhiprtc.so`) are typically installed in
`/opt/rocm/lib` (Linux), which is NOT registered with `ldconfig` and is not in the
default library search path. The plugin links against `hiprtc::hiprtc`, making
`libhiprtc.so` a transitive dependency of the plugin `.so`. **The user's
application does NOT need to link against hiprtc**. When hipDNN loads the
plugin via `dlopen()`, the dynamic linker resolves `libhiprtc.so` independently
from the user's application binary.

The plugin project embeds RPATH in the `.so`:

```cmake
set_target_properties(example_provider_plugin PROPERTIES
    INSTALL_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)
```

- `INSTALL_RPATH "${ROCM_PATH}/lib"` -- tells the dynamic linker where to find
  `libhiprtc.so` **at runtime**
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
   ldd build/src/libexample_provider_plugin.so
   ```
   All dependencies should resolve. Look for `not found` entries.

2. Trace the dynamic linker's search:
   ```bash
   LD_DEBUG=libs your_application 2>&1 | grep example_provider_plugin
   ```

3. Verify RPATH is embedded:
   ```bash
   readelf -d build/src/libexample_provider_plugin.so | grep 'RPATH|RUNPATH'
   ```

## Extending for Real-World Use

This example uses a naive convolution kernel and single-precision floats for
simplicity. To build a production plugin:

- **Support multiple data types**: Check `TensorAttributes::data_type()` in
  `isApplicable()` and `buildPlan()` to handle FLOAT, HALF, BFLOAT16, etc.
  The naive kernels only support FLOAT.

- **Optimize GPU kernels**: The naive convolution kernel (one thread per output
  element, no shared memory, no tiling) is deliberately simple for educational
  purposes.

- **Add workspace management**: Return non-zero from `getMaxWorkspaceSize()`
  if your engine needs temporary scratch memory. hipDNN allocates the
  workspace and passes it to `execute()`.

- **Implement custom knobs**: Override `getCustomKnobs()` in your PlanBuilder
  to expose tuning parameters (e.g., tile sizes, algorithm variants).

- **Support multi-node graphs**: Extend `isApplicable()` to match fused
  operation patterns (e.g., Conv + BiasAdd + ReLU).

## Further Reading

- `docs/PluginDevelopment.md` -- detailed plugin development guide
- `docs/Knobs.md` -- custom knob system documentation
- `docs/HowTo.md` -- hipDNN how-to guides
