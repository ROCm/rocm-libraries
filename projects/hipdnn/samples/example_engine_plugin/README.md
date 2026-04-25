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
| CMake >= 3.25 | Build system | |
| C++17 compiler | GCC/G++ or MSVC | No GPU compiler needed at build time |
| ROCm (HIP SDK + HIPRTC) | GPU kernel compilation and execution | `hipStream_t`, `hipMalloc`, HIPRTC APIs |
| hipDNN (installed) | Plugin SDK, data SDK, frontend library | Typically installed at `/opt/rocm` (Linux) |
| GPU hardware | Runtime execution of HIPRTC-compiled kernels | Any ROCm-supported GPU |
| Internet access | GTest is downloaded via CMake `FetchContent` | Only needed for the first build |

## Directory Structure

```
example_engine_plugin/
├── CMakeLists.txt                       # Root CMake: project options, dependencies
├── README.md                            # This file
├── README_TEMPLATE.md                   # Template README for new plugins (copy as README.md)
├── ai_plugin_rename_prompt.md            # AI agent prompt for copy-and-rename plugin creation
├── cmake/                              # CMake utility modules
│   └── VersionUtils.cmake              # Version setup and header generation functions
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
│   ├── TestExampleProviderEngine.cpp
│   ├── TestReluPlanBuilder.cpp
│   ├── TestReluPlan.cpp
│   ├── TestConvFwdPlanBuilder.cpp
│   └── TestConvFwdPlan.cpp
├── sample/                              # Demo app + acceptance test
│   ├── CMakeLists.txt
│   └── ExampleProviderSample.cpp
├── templates/                          # Template files for code generation
│   └── version.h.in                    # Version header template
└── version.json                        # Authoritative plugin version
```

## Build Instructions

Run these commands from the example_engine_plugin folder.

### Linux (GCC)

```bash
cmake -B build -DCMAKE_PREFIX_PATH="/opt/rocm"
cmake --build build
```

Run all tests, including the sample app:

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

### Windows (MSVC)

Ensure that the ROCm `bin` folder is in your system PATH before building:

```powershell
set PATH=C:\AMD\ROCm\bin;%PATH%
```

With MSVC installed:

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

### Windows (GNU/Clang with Ninja)

With Clang and Ninja installed, and with the ROCm `bin` folder in your system PATH:

```powershell
cmake -B build -G "Ninja"
cmake --build build
ctest --test-dir build
```

The tests and sample can also be run directly:

```powershell
.\build\bin\example_provider_tests.exe
```
```powershell
.\build\bin\example_provider_sample.exe
```

### CMake Options

| Option | Default | Description |
|---|---|---|
| `EXAMPLEPROVIDER_BUILD_UNIT_TESTS` | `ON` | Build unit tests (no GPU required) |
| `EXAMPLEPROVIDER_BUILD_SAMPLE` | `ON` | Build sample application (serves as acceptance test via `ctest`) |

To build only the plugin library (no tests or sample):

```bash
cmake -B build -DCMAKE_PREFIX_PATH="/opt/rocm" \
    -DEXAMPLEPROVIDER_BUILD_UNIT_TESTS=OFF \
    -DEXAMPLEPROVIDER_BUILD_SAMPLE=OFF
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

1. **Container** creates engines and returns available engine IDs.

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

### Working with Operation Graphs

hipDNN represents operation graphs as serialized FlatBuffers data. When hipDNN
calls plugin methods like `isApplicable()` or `buildPlan()`, it passes an
`IGraph` reference that wraps a FlatBuffers `Graph` object. The graph
contains nodes (operations) with typed attributes and tensor metadata.

#### GraphWrapper and NodeWrapper

The data SDK provides wrapper classes in `hipdnn_data_sdk/flatbuffer_utilities/`
that simplify working with the serialized graph data:

- **`GraphWrapper`** -- wraps the serialized graph buffer. It validates the buffer on
  construction. Key methods: `isValid()`, `nodeCount()`, `hasOnlySupportedAttributes()`,
  `getNode(index)`, `getNodeWrapper(index)`, `getTensorMap()`.

- **`NodeWrapper`** -- wraps individual graph nodes. Key methods: `isValid()`,
  `attributesType()` (returns the union type), `attributesAs<T>()` (safely casts
  to the specified type with type validation), `name()`, `computeDataType()`.

#### Null-Checking FlatBuffers Accessors

FlatBuffers accessors can return `nullptr`. Always check before use:

- **Strings**: `node.name()` can be `nullptr`. Use the pattern:
  `name != nullptr ? name->str() : ""`
- **Vectors/containers**: `attrs->dilation()`, `tensor->dims()`,
  `graph->nodes()` can all be `nullptr`. Always check before calling `size()`
  or iterating.
- **Attributes**: After `attributesAs<T>()`, check the result is not `nullptr`
  before accessing fields.
- **Tensor map lookups**: When looking up tensors by UID via `getTensorMap()`,
  always verify the iterator before dereferencing.

```cpp
// Check vector before iterating
const auto* dilation = attrs->dilation();
if(dilation != nullptr)
{
    for(size_t i = 0; i < dilation->size(); ++i)
    {
        // safe to access dilation->Get(i)
    }
}
```

See `ReluPlanBuilder.cpp` and `ConvFwdPlanBuilder.cpp` for concrete examples of
`isApplicable()` implementations that demonstrate graph inspection and
null-checking. `ConvFwdPlanBuilder` is a particularly good reference because it
checks `dilation()` for `nullptr` before iterating.

### HIPRTC Compilation Flow

```
Kernel Source File (e.g., kernels/relu/ReluForward.cpp)
        │
        ▼  Using the CMake configure function (at build time)
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

Source embedding is distinct from GPU compilation. At CMake configure time,
kernel `.cpp` files are embedded as C++ string literals into a generated source
registry. This is text storage, not GPU compilation. At runtime, each Plan
compiles only its own kernel source file via HIPRTC, and only when its engine is
selected for a graph.

The three DI interfaces (`IKernelCompiler`, `ICompiledProgram`,
`IRunnableKernel`) model each stage of the GPU compilation pipeline. A source
file can define multiple kernel functions (each marked `__global__`), but the
entire file is compiled as a single unit. A single module can contain multiple
kernels:

```cpp
auto module = compiler.compile("MyKernels.cpp", options);
auto addKernel = module->getRunnableKernel("add_vectors");
auto mulKernel = module->getRunnableKernel("multiply_vectors");
```

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

1. **Choose a brand name for your plugin**: Pick a short, descriptive name
   that identifies the technology or backend your plugin provides (e.g.,
   `CustomGemm`, `YourName`). C++ identifiers use the brand name alone
   (e.g., `YourNameContainer`, `YourNameEngine`); build/deployment
   identifiers append "provider" (e.g., `your_name_provider` namespace,
   `your_name_provider_impl` CMake target). See
   [Adaptation Reference](#adaptation-reference) for the complete naming
   tables, case conversion rules, file rename lists, and SDK identifiers.

2. **Copy and rename the directory**: Copy `example_engine_plugin/` to your new
   plugin directory (e.g., `your-name-provider/`). After copying, replace the
   copied `README.md` with `README_TEMPLATE.md` (rename `README_TEMPLATE.md`
   to `README.md`) and fill in the placeholders for your plugin. Also remove
   `ai_plugin_rename_prompt.md` from the new plugin directory.

3. **Verify the build on your system**: Before making any code changes, build
   and run the tests from your copied directory to verify the example works in
   your environment. Resolve any build issues before continuing.

4. **Rename classes, source files, and log strings**: Replace all `ExampleProvider*` class
   names with your brand prefix (e.g., `YourNameContainer`,
   `YourNameHandle`, `YourNameEngine`). This affects `Container`, `Handle`,
   `Context`, `Settings`, `Engine`, and `PluginPublic`. Rename the source
   files to match the new class names, and update the source file lists in
   `src/CMakeLists.txt`, `tests/CMakeLists.txt`, and `sample/CMakeLists.txt`.
   Also update `ExampleProvider` references in log message strings (e.g.,
   `HIPDNN_PLUGIN_LOG_INFO("Creating ExampleProviderContainer")`).
   See [Files Requiring Rename](#files-requiring-rename) for the complete list.

5. **Update the namespace**: Change the `example_provider` namespace to your
   brand + "provider" namespace (e.g., `your_name_provider`) throughout all
   source files, including nested namespaces (e.g.,
   `example_provider::test_helpers` in test files) and the kernel embedding
   templates in `kernels/templates/`. Note that `Handle`, `Context`, and
   `Settings` are declared at global scope, not inside the namespace; keep
   them at global scope and only update the forward declaration of
   `Container` in the Handle header. See
   [Naming Convention Reference](#naming-convention-reference) for the
   convention.

6. **Update the 5 macros** in your renamed PluginPublic file (was
   `ExampleProviderPluginPublic.cpp`): Update all five macros to reflect
   your plugin's name, version, and class names. `HIPDNN_PLUGIN_NAME`
   should use the convention snake_case brand + `_provider_plugin` (e.g.,
   `"your_name_provider_plugin"`). See
   [Naming Convention Reference](#naming-convention-reference).

7. **Replace example PlanBuilders and Plans**: Remove `ReluPlanBuilder`,
   `ReluPlan`, `ReluParams`, `ConvFwdPlanBuilder`, `ConvFwdPlan`, and
   `ConvFwdParams`. Create your own PlanBuilder, Plan, and Params for each
   operation your plugin supports. Plans inherit
   `IPlan<YourPluginHandle>` from the plugin SDK. See the
   [File Classification](#file-classification) table for the key methods to
   implement. To add further operations later, repeat this step and
   steps 8-11.

8. **Write your GPU kernels**: Replace the kernel source files in `kernels/`
   with your own. Each kernel must use `extern "C" __global__` and include
   only HIPRTC-compatible headers.

9. **Update CMake targets and kernel file list**: Update `KERNEL_FILES` in
   `kernels/CMakeLists.txt` with your kernel filenames. Update the source
   file list in `src/CMakeLists.txt` with your `.cpp` files. See
   [CMake Targets and Options Requiring Rename](#cmake-targets-and-options-requiring-rename)
   for the complete list of targets and options.

10. **Register your engines**: In your renamed Container file (was
    `ExampleProviderContainer.cpp`), register your engines via
    `HIPDNN_REGISTER_ENGINE` with unique engine names and add lambdas to
    create the new engines:

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

    **Engine-to-PlanBuilder mapping:** The typical pattern is one engine per
    PlanBuilder, as shown above. However, `addPlanBuilder()` can be called
    multiple times on a single engine if grouping multiple PlanBuilders under one
    engine ID is more appropriate for your use case. The `ExampleProviderEngine`
    class supports both patterns without modification.

11. **Build and run unit tests**: Follow the [Build Instructions](#build-instructions)
    to verify successful compilation and tests.

12. **Update your Settings struct**: Replace the fields in your renamed
    Settings struct (was `ExampleProviderSettings`) with settings relevant
    to your operations (e.g., replace `reluNegativeSlope` with your knob
    values).

13. **Update Engine and Container tests**: In your renamed test files (were
    `TestExampleProviderContainer.cpp` and
    `TestExampleProviderEngine.cpp`), update the expected engine count to
    match the number of engines you registered.

14. **Create graph construction helpers in `TestHelpers.hpp`**: Replace
    `createReluFwdGraph()` and `createConvFwdGraph()` with helpers that
    build FlatBuffer graphs for your operations. Keep `createEngineConfig()`
    and `createEngineConfigWithFloatKnob()` as-is.

15. **Write unit tests for PlanBuilder and Plan**: Following the patterns in
    `TestReluPlanBuilder.cpp` and `TestReluPlan.cpp`, write tests for your
    PlanBuilder's `isApplicable()`, `getCustomKnobs()`, `buildPlan()`, and
    your Plan's `compile()` and `execute()`. The tests included in this
    example are not exhaustive; additional test coverage may be added.

16. **Incorporate the new operations into your application**: Use the
    patterns from `sample/ExampleProviderSample.cpp` to load the plugin,
    select engines, and execute operations in your application.

### File Classification

Comment markers are used to identify files that will be modified when using this
example as a template for creating a new plugin. Look for `TEMPLATE ADAPTATION`
and `TEMPLATE REFERENCE` comment markers in the source files for per-file guidance.

**`TEMPLATE ADAPTATION`** -- Rename and adjust. The structure stays the same.

| File | What to Do |
|------|------------|
| `ExampleProviderPluginPublic.cpp` | Rename file, update 5 macros and `using namespace` |
| `ExampleProviderContainer.hpp/cpp` | Rename file, update class name and engine registrations |
| `ExampleProviderHandle.hpp/cpp` | Rename file, update class name |
| `ExampleProviderContext.hpp` | Rename file, update class name |
| `ExampleProviderSettings.hpp` | Rename file, update class name and fields |
| `ExampleProviderEngine.hpp/cpp` | Rename file, update class name |
| `ExampleProviderUtils.hpp` | Rename file, update namespace |
| `kernels/CMakeLists.txt` | Update `KERNEL_FILES` list with your kernel filenames |
| `tests/TestExampleProviderContainer.cpp` | Rename file, update class references and engine count |
| `tests/TestExampleProviderEngine.cpp` | Rename file, update class references |
| `tests/TestHelpers.hpp` | Replace graph construction helpers for your operations |
| `sample/ExampleProviderSample.cpp` | Rename file, adapt scenarios to your operations |
| `version.json` | Update version key name from `example_provider_version` to `your_name_provider_version` |
| `templates/version.h.in` | Rename header guard and macro prefix from `EXAMPLE_PROVIDER_VERSION_*` to `YOUR_NAME_PROVIDER_VERSION_*` |
| `cmake/VersionUtils.cmake` | Rename three function names from `example_provider_*` to `your_name_provider_*` |

**`TEMPLATE REFERENCE`** -- Study, then replace with your own implementations.

| File | What to Do |
|------|------------|
| `engines/plans/ReluPlanBuilder.hpp/cpp` | Study `isApplicable()`, `getCustomKnobs()`, `initializeExecutionSettings()`, `buildPlan()`, then write your own |
| `engines/plans/ReluPlan.hpp/cpp` | Study `compile()`, `execute()`, then write your own |
| `engines/plans/ReluParams.hpp` | Study parameter extraction pattern, then write your own |
| `engines/plans/ConvFwdPlanBuilder.hpp/cpp` | Second example; compare with ReLU for graph matching differences |
| `engines/plans/ConvFwdPlan.hpp/cpp` | Second example; compare with ReLU for kernel launch differences |
| `engines/plans/ConvFwdParams.hpp` | Second example of parameter extraction |
| `tests/TestReluPlanBuilder.cpp`, `tests/TestReluPlan.cpp` | Study testing patterns, then write equivalent tests |
| `tests/TestConvFwdPlanBuilder.cpp`, `tests/TestConvFwdPlan.cpp` | Second example of testing patterns |

**No marker** -- Update namespace only or replace contents.

| File | What to Do |
|------|------------|
| `hip/` directory (all files) | Update namespace only. HIPRTC infrastructure has no operation-specific logic. |
| `tests/mocks/` (all files) | Update namespace only. Mock implementations for GPU-free testing. |
| `kernels/relu/ReluForward.cpp` | Replace with your GPU kernel source file |
| `kernels/conv/ConvForwardNaive.cpp` | Replace with your GPU kernel source file |

### Adaptation Reference

Use the tables below when performing the rename steps in the workflow above.

#### Naming Convention Reference

| Context | Convention | Example Plugin | Your Plugin |
|---|---|---|---|
| C++ classes | Brand only | `ExampleProviderContainer` | `YourNameContainer` |
| C++ source files | Brand only | `ExampleProviderContainer.hpp` | `YourNameContainer.hpp` |
| Engine names | Brand only | `EXAMPLE_PROVIDER_RELU_ENGINE` | Keep as-is until step 10 |
| Namespace | Brand + provider | `example_provider` | `your_name_provider` |
| CMake targets | Brand + provider | `example_provider_impl` | `your_name_provider_impl` |
| CMake options | Concatenated | `EXAMPLEPROVIDER_*` | `YOURNAMEPROVIDER_*` |
| CMake project name | Hyphenated | `hipdnn-example-provider` | `your-name-provider` |
| `HIPDNN_PLUGIN_NAME` | Brand + provider + plugin | `"example_provider_plugin"` | `"your_name_provider_plugin"` |

#### Case Conversion Rules

The brand name must be expressed in three case forms. For a brand name like
`YourName`, split at PascalCase word boundaries:

| Form | Rule | Example |
|---|---|---|
| PascalCase | As chosen | `YourName` |
| snake_case | Lowercase words joined by `_` | `your_name` |
| UPPER_SNAKE_CASE | Uppercase words joined by `_` | `YOUR_NAME` |

Apply each form as follows:
- **C++ classes/files**: PascalCase brand (`YourNameContainer`)
- **Namespace**: snake_case brand + `_provider` (`your_name_provider`)
- **CMake targets**: snake_case brand + `_provider_` + suffix (`your_name_provider_impl`)
- **CMake options**: UPPER brand + `PROVIDER_` + suffix (`YOURNAMEPROVIDER_BUILD_UNIT_TESTS`)
- **Engine names** (step 10): UPPER_SNAKE brand + `_` + operation (`YOUR_NAME_GEMM_ENGINE`)
- **CMake project**: Hyphenated lowercase brand + `-provider` (`your-name-provider`)
- **`HIPDNN_PLUGIN_NAME`**: snake_case brand + `_provider_plugin` (`"your_name_provider_plugin"`)

#### Files Requiring Rename

Rename files with the `ExampleProvider` prefix to your brand
prefix. These are the `ExampleProvider`-prefixed files from the
[File Classification](#file-classification) `TEMPLATE ADAPTATION` table:

| Original File | Renamed To |
|---|---|
| `ExampleProviderPluginPublic.cpp` | `YourNamePluginPublic.cpp` |
| `ExampleProviderContainer.hpp/cpp` | `YourNameContainer.hpp/cpp` |
| `ExampleProviderHandle.hpp/cpp` | `YourNameHandle.hpp/cpp` |
| `ExampleProviderContext.hpp` | `YourNameContext.hpp` |
| `ExampleProviderSettings.hpp` | `YourNameSettings.hpp` |
| `ExampleProviderEngine.hpp/cpp` | `YourNameEngine.hpp/cpp` |
| `ExampleProviderUtils.hpp` | `YourNameUtils.hpp` |
| `TestExampleProviderContainer.cpp` | `TestYourNameContainer.cpp` |
| `TestExampleProviderEngine.cpp` | `TestYourNameEngine.cpp` |
| `ExampleProviderSample.cpp` | `YourNameSample.cpp` |
| `version.json` (key inside) | Update key to `your_name_provider_version` |
| `templates/version.h.in` | Rename guard/macros to `YOUR_NAME_PROVIDER_VERSION_*` |

#### CMake Targets and Options Requiring Rename

| Original Target/Option | Renamed To |
|---|---|
| `hipdnn-example-provider` (project name) | `your-name-provider` |
| `example_provider_impl` | `your_name_provider_impl` |
| `example_provider_private` | `your_name_provider_private` |
| `example_provider_plugin` | `your_name_provider_plugin` |
| `example_provider_compile_options` | `your_name_provider_compile_options` |
| `example_provider_kernel_embed` | `your_name_provider_kernel_embed` |
| `example_provider_tests` | `your_name_provider_tests` |
| `example_provider_sample` | `your_name_provider_sample` |
| `EXAMPLEPROVIDER_BUILD_UNIT_TESTS` | `YOURNAMEPROVIDER_BUILD_UNIT_TESTS` |
| `EXAMPLEPROVIDER_BUILD_SAMPLE` | `YOURNAMEPROVIDER_BUILD_SAMPLE` |
| `EXAMPLEPROVIDER_ENABLE_CLANG_TIDY` | `YOURNAMEPROVIDER_ENABLE_CLANG_TIDY` |
| `example_provider_version_file_dir` (function) | `your_name_provider_version_file_dir` |
| `example_provider_setup_version` (function) | `your_name_provider_setup_version` |
| `example_provider_generate_version_header` (function) | `your_name_provider_generate_version_header` |

#### SDK Identifiers (Do Not Rename)

These identifiers are defined by the hipDNN SDK packages and must not be
changed. They are the same for every plugin:

| Identifier | Source |
|---|---|
| `hipdnn_plugin_sdk` | CMake package (plugin SDK) |
| `hipdnn_data_sdk` | CMake package (data SDK) |
| `hipdnn_frontend` | CMake package (frontend library) |
| `HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR` | CMake variable exported by `hipdnn_data_sdk` |
| `HIPDNN_PLUGIN_NAME`, `HIPDNN_PLUGIN_VERSION`, `HIPDNN_PLUGIN_CONTAINER_TYPE`, `HIPDNN_PLUGIN_HANDLE_TYPE`, `HIPDNN_PLUGIN_CONTEXT_TYPE` | Macro names expected by `EnginePluginImpl.inl` (values are plugin-specific) |
| `EnginePluginImpl.inl` | Plugin SDK header that generates C entry points |
| `IPlan`, `EngineManager` | Plugin SDK types |
| `IGraph`, `IEngineConfig` | Data SDK types |

## Testing Your Plugin

### Unit Testing Architecture

Unit tests run without GPU hardware using the DI interfaces and mocks described
in [Dependency Injection Interfaces for Testability](#dependency-injection-interfaces-for-testability).

### What to Test at the Engine and Container Level

See `tests/TestExampleProviderEngine.cpp` and `tests/TestExampleProviderContainer.cpp`.

- **Engine `isApplicable()` delegation** -- verifies that the engine correctly
  delegates to its PlanBuilders and returns `true` when a matching builder
  exists, `false` otherwise. Uses mock PlanBuilders with controllable return
  values.
- **Container engine registration** -- verifies that `copyEngineIds()` returns
  the expected number of engines with distinct IDs

These tests validate framework wiring that is independent of specific operations.
Rename `ExampleProvider` references to your plugin prefix and update the expected
engine count to match your registered engines. No operation-specific changes are
needed.

### Graph Construction Helpers

`tests/TestHelpers.hpp` provides helper functions such as `createReluFwdGraph()`
and `createConvFwdGraph()` that build in-memory FlatBuffer graphs mimicking what
the hipDNN frontend produces. These helpers let unit tests construct realistic
operation graphs without a running hipDNN instance. When writing a plugin for a
new operation type, replace these helpers with functions that construct your
operation's graph attributes (e.g., `createMyOpGraph()` building the appropriate
`NodeAttributes` variant). Keep `createEngineConfig()` and
`createEngineConfigWithFloatKnob()` as-is; they create generic engine
configurations usable by any PlanBuilder test.

### What to Test in a PlanBuilder

See `tests/TestReluPlanBuilder.cpp` for the complete pattern.

- **`isApplicable()`** -- returns `true` for matching operation graphs, `false`
  for non-matching (wrong operation type, wrong node count, wrong data type)
- **`getCustomKnobs()`** -- returns the correct knob definitions (IDs, types,
  ranges, defaults)
- **`getMaxWorkspaceSize()`** -- returns expected workspace bytes
- **`initializeExecutionSettings()`** -- reads knob values from the engine
  config into the settings struct
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
acceptance test in this project. It is registered as a `ctest` and verifies
end-to-end correctness on GPU hardware.

### Numerical Accuracy Testing

To verify the numerical accuracy of your plugin's engines against CPU reference
implementations across a wide range of tensor configurations, data types, and
convolution parameters, use the hipDNN
[integration test harness](https://github.com/ROCm/rocm-libraries/tree/develop/dnn-providers/integration-tests).
The integration test harness exercises engines through the hipDNN frontend API,
covering cases that are not practical to validate with hardcoded expected values
alone.

For custom accuracy tests, the `hipdnn_test_sdk` package provides CPU reference
implementations (e.g., `CpuFpReferenceConvolution`) that can be used to compute
expected results for comparison against your plugin's GPU output. Link your test
executable against `hipdnn_test_sdk` and use these references to validate
correctness across arbitrary tensor shapes and convolution parameters.

## Integrating the Plugin into Your Application

Plugins are automatically loaded by the hipDNN library when a hipDNN handle is created.

Integration involves two concerns:
[Plugin Loading](#plugin-loading) (hipDNN finding the plugin) and
[Runtime Dependency Resolution and RPATH](#runtime-dependency-resolution-and-rpath)
(the plugin finding its own dependencies).

### Plugin Loading

Ensuring the plugin is loaded is the **application's** responsibility, not the
plugin's.

By default, hipDNN loads all plugins in the `lib/hipdnn_plugins/engines` subfolder
of the ROCm install folder. The plugin CMake project uses the
`HIPDNN_RELATIVE_INSTALL_PLUGIN_ENGINE_DIR` CMake variable (exported by the
`hipdnn_data_sdk` package) to install the plugin into this subfolder. hipDNN
automatically loads plugins from this location.

```bash
cmake --install build --prefix /opt/rocm
```

There are three ways to override the default hipDNN plugin loading behavior so that
plugins can be loaded from folders outside the ROCm install folder:

**Environment variable** (`HIPDNN_PLUGIN_DIR`): Set before creating a hipDNN
handle. This becomes the new default plugin directory that hipDNN scans for
plugin shared libraries (`.so` on Linux, `.dll` on Windows).

```bash
export HIPDNN_PLUGIN_DIR=/path/to/plugin/directory
```

**ADDITIVE mode**: Load additional plugin directories alongside all existing
search paths (the hipDNN default plugin directory or any paths set by
`HIPDNN_PLUGIN_DIR`).

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

#### Path Resolution

hipDNN resolves plugin paths as follows:

**Relative paths** are resolved against the directory containing
`libhipdnn_backend.so` (**not** the current working directory). For example, if
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

#### Verifying the Plugin Is Loaded

After creating a hipDNN handle, query loaded plugins to confirm yours is
present:

```cpp
std::vector<std::filesystem::path> paths;
auto err = getLoadedEnginePluginPaths(handle, paths);
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

See `sample/ExampleProviderSample.cpp` for a complete example demonstrating
plugin loading, engine selection, knob modification, and correctness verification.

## Quick Checklist

- [ ] Step 1: Choose a brand name for your plugin
- [ ] Step 2: Copy and rename the directory
- [ ] Step 3: Verify the build on your system
- [ ] Step 4: Rename classes, source files, and log strings
- [ ] Step 5: Update the namespace (including nested and template namespaces)
- [ ] Step 6: Update the 5 macros in your renamed PluginPublic file
- [ ] Step 7: Replace example PlanBuilders and Plans
- [ ] Step 8: Write your GPU kernels
- [ ] Step 9: Update CMake targets and kernel file list
- [ ] Step 10: Register your engines
- [ ] Step 11: Build and run unit tests
- [ ] Step 12: Update your renamed Settings struct
- [ ] Step 13: Update Engine and Container tests (adjust engine count)
- [ ] Step 14: Create graph construction helpers in `TestHelpers.hpp`
- [ ] Step 15: Write unit tests for PlanBuilder and Plan
- [ ] Step 16: Incorporate the new operations into your application

## Creating a Plugin with AI Assistance

An AI agent can automate the file copying, renaming, and text replacement
steps of the adaptation workflow. See
[`ai_plugin_rename_prompt.md`](ai_plugin_rename_prompt.md) for a prompt
template and usage instructions.

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

The ConvFwd engine follows the same pattern with a `BLOCK_SIZE` knob that
controls the GPU thread block size for kernel launches.

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

ROCm libraries (including `libhiprtc.so`) are typically installed in the `/lib`
folder of the ROCm install path (e.g., `/opt/rocm/lib` on Linux), which is **not**
registered with `ldconfig` and is not in the default library search path. The example
plugin links against `hiprtc::hiprtc`, making `libhiprtc.so` a transitive dependency
of the plugin `.so`. **The user's application does *not* need to link against hiprtc**.
When hipDNN loads the plugin via `dlopen()`, the dynamic linker resolves `libhiprtc.so`
independently from the user's application binary.

**Note:** this dependency resolution is distinct from ensuring that hipDNN is
able to locate the plugin for loading. See [Plugin Loading](#plugin-loading)
for further details on this topic.

The approach for managing the plugin's library dependencies changes depending on
the environment and how the plugin is deployed.

#### Windows

On Windows, dynamic libraries are loaded from the hipDNN application's folder
or from folders listed in the system PATH. Ensure the ROCm `bin` folder is in
the system PATH so that the `hiprtc` library used by the plugin can be found
when the plugin is loaded by hipDNN. For example:

```powershell
set PATH=C:\AMD\ROCm\bin;%PATH%
```

#### Linux

On Linux, `RPATH`/`RUNPATH` and `LD_LIBRARY_PATH` are used to control which folders
will be searched for dependent libraries when loading the plugin. How you set these depends on your development and deployment environment and how
the hipDNN application is written. The configuration will likely need to be
tailored to your specific deployment.

To facilitate plugin development, the plugin CMake project embeds `RPATH` (`RUNPATH`)
in the `.so` as follows:

```cmake
set_target_properties(example_provider_plugin PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../.."
    INSTALL_RPATH_USE_LINK_PATH TRUE
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
```
- `BUILD_WITH_INSTALL_RPATH TRUE` -- the RPATH is the same for the library in the
  build folder and when the library is installed (otherwise CMake uses a different
  RPATH for the library in the build folder and changes this when the library is
  installed using `cmake --install`).
- `INSTALL_RPATH "\$ORIGIN;\$ORIGIN/../.."` -- search both the folder that the
  plugin is located in and the plugin's grandparent folder.
- `INSTALL_RPATH_USE_LINK_PATH TRUE` -- automatically adds directories of
  linked libraries to the plugin's RPATH (e.g., the `hiprtc` library's ROCm
  library folder).

It is important to ensure the `RPATH` set on the plugin matches the deployment environment.
Hard-coded paths like `/opt/rocm/lib` will fail on machines with different
layouts. The default `$ORIGIN` entries mitigate this by resolving relative to
the plugin's installed location. See
[Plugin Loading](#plugin-loading) for further context on deploying the plugin.

**Note:** `INSTALL_RPATH_USE_LINK_PATH` is a development convenience that
hard-codes linked library paths (e.g., the ROCm folder) into the `RPATH`. Consider
removing it for production builds because it can mask deployment failures when
folder layouts differ from the development environment. See
[Troubleshooting Plugin Loading](#troubleshooting-plugin-loading).

To add additional paths to the plugin's `RPATH`, modify the `INSTALL_RPATH`
target property directly in the CMake project, or pass `CMAKE_INSTALL_RPATH`
on the command line:

```bash
cmake -B build -DCMAKE_INSTALL_RPATH=/custom/rocm/path/lib
```

Here, `/custom/rocm/path/lib` is the path to the needed libraries as it exists on
the deployment machine.

#### Troubleshooting Plugin Loading

If the plugin silently fails to load (no engines from this plugin appear):

1. Check library dependencies:
   ```bash
   ldd build/bin/libexample_provider_plugin.so
   ```
   All dependencies should resolve. Look for `not found` entries.

2. Trace the dynamic linker's search:
   ```bash
   LD_DEBUG=libs your_application 2>&1 | grep example_provider_plugin
   ```

3. Verify RPATH is embedded:
   ```bash
   readelf -d build/bin/libexample_provider_plugin.so | grep -E 'RPATH|RUNPATH'
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
