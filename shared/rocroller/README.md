
# AMD's rocRoller Assembly Kernel Generator

rocRoller is a software library for generating AMDGPU kernels.

[![Develop Branch Code Coverage Report for gfx942](https://img.shields.io/badge/Code%20Coverage-gfx942-informational)](http://math-ci.amd.com/job/rocm-libraries/job/codecov/job/rocroller/job/develop/Code_20coverage_20gfx942_20report)
[![Develop Branch Code Coverage Report for python](https://img.shields.io/badge/Code%20Coverage-Python-informational)](http://math-ci.amd.com/job/rocm-libraries/job/codecov/job/rocroller/job/develop/Python_20Code_20coverage_20gfx942_20report)
[![Develop Branch Performance Report for gfx942](https://img.shields.io/badge/Performance-gfx942-critical)](http://math-ci.amd.com/job/rocm-libraries/job/performance/job/rocroller/job/develop/Performance_20Report_20for_20gfx90a)
[![Develop Branch Generated Documentation](https://img.shields.io/badge/Documentation-Generated-informational)](http://math-ci.amd.com/job/rocm-libraries/job/documentation/job/rocroller/job/develop/Generated_20Docs)

## Jump to

- [AMD's rocRoller Assembly Kernel Generator](#amds-rocroller-assembly-kernel-generator)
  - [Building the Library](#building-the-library)
    - [Quick Start](#quick-start)
    - [Building](#building)
    - [with docker](#with-docker)
    - [natively (e.g. Ubuntu 22.04)](#natively-eg-ubuntu-2204)
    - [CMake and Make Commands](#cmake-and-make-commands)
    - [Common CMake Options](#common-cmake-options)
  - [Running the Tests (from a build directory)](#running-the-tests-from-a-build-directory)
    - [With CTest](#with-ctest)
    - [With GTest](#with-gtest)
    - [With Catch2](#with-catch2)
  - [Updating Pregenerated GPUArchitecture YAML Files](#updating-pregenerated-gpuarchitecture-yaml-files)
  - [GEMM Client](#gemm-client)
  - [Development](#development)
    - [File Structure](#file-structure)
    - [Coding Practices](#coding-practices)
      - [Style](#style)
      - [ISO C++ Standard](#iso-c-standard)
      - [Documentation](#documentation)
      - [PR submissions](#pr-submissions)
      - [Testing](#testing)
  - [Logging and Debugging](#logging-and-debugging)
    - [Logging](#logging)
    - [Debugging](#debugging)
  - [Analysis](#analysis)
    - [Performance Analysis](#performance-analysis)
    - [Kernel Analysis](#kernel-analysis)
    - [KernelGraph Visualization](#kernelgraph-visualization)
    - [Memory Access Visualization](#memory-access-visualization)

## Building the Library

### Quick Start

**Clone and Sparse Checkout:**
```bash
git clone --no-checkout --filter=blob:none git@github.com:ROCm/rocm-libraries.git
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set shared/rocroller shared/mxdatagenerator
git checkout develop   # or your desired branch
```

### Building

rocRoller uses CMake for configuration and building. If all dependencies are installed in standard locations, CMake should work out of the box. If not, set `CMAKE_PREFIX_PATH` to help CMake find required packages (e.g., `/opt/rocm;/opt/rocm/llvm`). A `CMakePresets.json` file in the project root provides several build presets:

**Common Presets:**

- **default:release**: Emulates the current dev workflow.
  - `CMAKE_CXX_COMPILER`: `/opt/rocm/bin/amdclang++`
  - `ROCROLLER_ENABLE_FETCH`: `ON`
  - `CMAKE_PREFIX_PATH`: `/opt/rocm;/opt/rocm/llvm`
- **precheckin**: Used for CI pipelines.
  - `ROCROLLER_ENABLE_CPPCHECK`: `ON`
  - `ROCROLLER_ENABLE_YAML_CPP`: `OFF`
  - `CMAKE_CXX_COMPILER`: `/opt/rocm/bin/amdclang++`
  - `CMAKE_BUILD_TYPE`: `Release`
  - `ROCROLLER_ENABLE_FETCH`: `ON`
  - `ROCROLLER_TESTS_SKIP_SLOW`: `OFF`
  - `CMAKE_PREFIX_PATH`: `/opt/rocm;/opt/rocm/llvm`
- **asan**, **amd-mrisa**, **coverage**, **docs**: See `CMakePresets.json` for details.

**Usage Example:**
```bash
cmake --preset default:release -B build -S . [additional cmake options]
```

If dependencies are missing, enable FetchContent with `ROCROLLER_ENABLE_FETCH=ON` to automatically download and build them.

#### with docker

```bash
./docker/user-image/start_user_container
docker exec -ti -u ${USER} ${USER}_dev_clang bash
cd /data/shared/rocroller
cmake --preset default:release -B build -S .
cmake --build build -j
```

#### natively (e.g. Ubuntu 22.04)

```bash
# As root:
apt update
apt install -y libopenblas-dev ninja-build

# As regular user:
cd <path-to>/rocm-libraries/shared/rocroller
cmake --preset default:release -B build -S .
cmake --build build -j
```

### CMake and Make Commands

CMake uses the compiler specified by the `CXX` environment variable. To set compilers manually:
```bash
CXX=<g++ or clang++ path> CC=<gcc or clang path> cmake .. [cmake options]
```

#### Common CMake Options

| Option                                   | Default | Description                                               |
|-------------------------------------------|---------|-----------------------------------------------------------|
| ROCROLLER_ENABLE_CLIENT                   | ON      | Build the rocRoller client                                |
| ROCROLLER_ENABLE_YAML_CPP                 | ON      | Enable yaml-cpp backend                                   |
| ROCROLLER_ENABLE_LLVM                     | OFF     | Enable LLVM yaml backend                                  |
| ROCROLLER_BUILD_TESTING                   | ON      | Build rocRoller testing                                   |
| ROCROLLER_ENABLE_CATCH                    | ON      | Build Catch2 unit tests                                   |
| ROCROLLER_ENABLE_ARCH_GEN_TEST            | ON      | Build architecture generator test                         |
| ROCROLLER_ENABLE_TEST_DISCOVERY           | ON      | Use gtest/catch2 test discovery                           |
| ROCROLLER_ENABLE_COVERAGE                 | OFF     | Build code coverage                                       |
| ROCROLLER_TESTS_SKIP_SLOW                 | OFF     | Disable slow tests                                        |
| ROCROLLER_EMBED_ARCH_DEF                  | ON      | Embed msgpack architecture data in library                |
| ROCROLLER_BUILD_SHARED_LIBS               | ON      | Build as shared library                                   |
| ROCROLLER_ENABLE_FETCH                    | OFF     | Fetch dependencies if not found                           |
| ROCROLLER_ENABLE_LLD                      | OFF     | Build LLD-dependent functionality                         |
| ROCROLLER_ENABLE_TIMERS                   | OFF     | Enable timer code                                         |
| ROCROLLER_ENABLE_CPPCHECK                 | OFF     | Enable cppcheck                                           |
| ROCROLLER_MRISAS_DIR                      | `<build>/GPUArchitectureGenerator/amd-mrisa` | MRISA XML directory |
| ROCROLLER_ENABLE_PREGENERATED_ARCH_DEF    | ON      | Use pregenerated GPU architecture YAML files              |

## Running the Tests (from a build directory)

### With CTest
**Run All Tests:**
  ```bash
  ctest
  # or
  make test # or ninja, if using
  ```
  Runs all tests, one process per test.

**Exclude GPU Tests (for CPU-only machines):**
  ```bash
  ctest -LE GPU
  ```

### With GTest
**Run All GTest Tests:**
  ```bash
  ./test/rocroller-tests
  ```

**List All Tests:**
  ```bash
  ./test/rocroller-tests --gtest_list_tests
  ```

**Run a Specific Test:**
  ```bash
  ./test/rocroller-tests --gtest_filter="<test-name-or-regex>"
  ```

**List a Specific Test:**
  ```bash
  ./test/rocroller-tests --gtest_list_tests --gtest_filter="<test-name-or-regex>"
  ```

**Exclude GPU Tests (for CPU-only machines):**
  ```bash
  ./test/rocroller-tests --gtest_filter="-*GPU_*"
  ```

**Get Help:**
  ```bash
  ./test/rocroller-tests --help
  ```

**Debugging Options:**
  - Prevent exceptions from being caught by GoogleTest:
    ```
    --gtest_catch_exceptions=0
    ```
  - Break on test failure:
    ```
    --gtest_break_on_failure
    ```

### With [Catch2](https://github.com/catchorg/Catch2)

**Run All Catch2 Tests:**
  ```bash
  ./test/rocroller-tests-catch
  ```

**List All Tests:**
  ```bash
  ./test/rocroller-tests-catch --list-tests
  ```

**List All Test Tags:**
  ```bash
  ./test/rocroller-tests-catch --list-tags
  ```

**Run a Specific Test:**
  ```bash
  ./test/rocroller-tests-catch "<test-name-or-regex>"
  ```

**List Specific Tests:**
  ```bash
  ./test/rocroller-tests-catch --list-tests "<test-name-or-regex>"
  ```
**List Specific Tests via Tags:**
  ```bash
  ./test/rocroller-tests-catch --list-tags "<tag-name-or-regex>"
  ```

**Get Help:**
  ```bash
  ./test/rocroller-tests-catch --help
  ```


### Updating Pregenerated GPUArchitecture YAML Files

Update the [GPUArchitecture YAML files](GPUArchitectureGenerator/pregenerated) whenever MRISA XML files or [GPUArchitecture_def](GPUArchitectureGenerator/include/GPUArchitectureGenerator/GPUArchitectureGenerator_defs.hpp) change.

**Steps:**
```bash
cmake --preset amd-mrisa -B build -S .
cmake --build build --target GPUArchitecture_def
cp build/GPUArchitectureGenerator/split_yamls/*.yaml GPUArchitectureGenerator/pregenerated/
cd GPUArchitectureGenerator/pregenerated/
../scripts/format_yaml.py -I *.yaml
```

## GEMM Client

rocRoller includes a GEMM client for exploring GEMM workloads and conducting performance tests. The GEMM client is built automatically with the library.

To launch the GEMM client from your build directory, run:

```bash
./bin/client/rocroller-gemm --help
```

## Development
### File Structure

 - `Foo.hpp`: Contains definitions for classes, concepts, etc.  Functions should be declaration-only.
 This is meant to allow for easy reading of the interface.
 - `Foo_impl.hpp`: Contains definitions for short (inlinable) functions.
 Should be included at the bottom of `Foo.hpp`.  This makes it easy to move definitions between here and `Foo.cpp`, since they are not within the class definition.
 - `Foo_fwd.hpp`: Contains forward declarations for all types defined in `Foo.hpp`, type aliases such as `FooPtr` for `std::shared_ptr<Foo>` and `std::variant`.  Should have zero or very few includes.
 - `Foo.cpp`: Contains definitions for longer functions.

Generally, we prioritize inlining.

### Coding Style

 - Static and free functions should start with an uppercase character
 - Instance functions should start with lowercase
 - Private member variables should start with `m_`
 - Public member variables should not start with `m_`
 - If there is no invariant, public member variables are ok
 - Naming convention should follow camelCase
 - Macro names and CMake options should be `UPPER CASE` using `snake_case` and have a prefix `ROCROLLER_`
 - Before submitting code as a PR for review all sources need to be formatted with `clang-format`, version 13.


Code should be writing in a clearly descriptive fashion. [Good resource for expressing ideas in code.](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#p1-express-ideas-directly-in-code)

### ISO C++ Standard

While the `rocRoller` library does have C++20 features, new C++20 specific features should not be introduced into the code without explicit design review.

In general, C++17 modern coding practices should be followed. This means for example type aliases should be done with `using` instead of `typedef`.

Memory should not be allocated with `new` keyword syntax, instead use managed pointers via either `std::make_unique` or `std::make_shared`. If an array of elements need to be allocated then a `std::vector` can be used.

### Documentation

Documentation of code is as important for maintainability as writing clear and concise code. Documentation is built for `rocRoller` using [Doxygen](https://www.doxygen.nl/index.html), which means contributors need to follow [Doxygen style](https://www.doxygen.nl/manual/docblocks.html) comments.
- Developers must add documentation to describe new data structures such as `Class` declarations and member functions.
- Per line comments are important when the semantics of the code section isn't obvious.
- Incomplete features, or sections that need to implemented in future PRs should have a comment with a `TODO` heading.

Note: If using VSCode consider installing the [Doxygen Documentation Generator](https://marketplace.visualstudio.com/items?itemName=cschlosser.doxdocgen) extension. This extension helps with quickly implementing documentation sections in a Doxygen format.

[Graphviz](https://graphviz.org/) and [Doxygen](https://www.doxygen.nl/index.html) both need to be installed to build the html documentation.

The Doxygen documentation can be built via the make command from the main build directory:
```bash
make -j docs # or ninja, if using
```

This will generate an HTML website from the markdown readme files and Doxygen comments and source/object relationships. The `index.html` file can be found in `doc/html` from the root of your local repository.


### PR submissions

- PRs submitting should have fewer than `500` lines of code change, unless of course those changes are merely lines moved or deleted.
- If a new feature cannot be implemented in fewer than `500` lines of code then consider either refactoring, or splitting the feature into two or more PRs.
- When opening a PR, details of what the feature does are essential.
- PR authors should provide a description on how to review the PR, such as, outlining key changes. This is especially important if the PR is complex, novel, or on the higher side of lines of code change.
- PR code needs to be formatted with `clang-format` version 13 (rocRoller's docker images already have this version of `clang-format`). This can be done manually, with `scripts/fix-format`, or via githooks (with `.githooks/install`).
- If a PR alters the compilation process in a way that causes the performance job to fail when building the master branch, adding the label `ci:no-build-master` will instead compare the PR against the latest build of the master branch.


### Testing

Each new feature is required to have a test.
- Test sources are placed in the `test` folder.
- CPP Files for Unit Tests should be included in the `rocroller-tests` executable in [CMakeLists.txt](https://github.com/ROCm/rocm-libraries/shared/rocroller/blob/develop/test/CMakeLists.txt).

Some tests require multiple threads for properly testing a desired or undesired behaviour (e.g. thread-safety) or to benefit from faster execution. Therefore, it is recommended to set `OMP_NUM_THREADS` appropriately. A value between `[NUM_PHYSICAL_CORES/2, NUM_PHYSICAL_CORES)` is recommended. Setting `OMP_NUM_THREADS` to the number of available cores or higher can cause test to run slower due to oversubscription (e.g. increased contention).

Note a few conditions:
- If your test requires a context but does not actually need to run on a GPU, inherit from `GenericContextFixture`.
- If your test needs to run on an actual GPU, inherit from `CurrentGPUContextFixture`. This:
  - provides functionality to assemble a kernel
  - sets up `targetArchitecture()` to reflect a real GPU.
  - will include functionality to run a kernel in the future.
  - Name your test starting with `GPU_`.  This ensures that it is can be filtered out when running on a CPU-only node.
- If your test needs to run against different option values, use `Settings::set()` to run the test with those values (overwriting previous values in memory). Once a test is completed, the context fixture calls `Settings::reset()`, which resets the settings to the env vars (if set) or default settings.

[Catch2](https://github.com/catchorg/Catch2) is the preferred unit testing framework for new unit tests.  GTest information for working with older unit tests can be found in the [GoogleTest User's Guide](https://google.github.io/googletest/).


## Logging and Debugging
For a full list of `ROCROLLER_*` environment variables see: [Settings.hpp](lib/include/rocRoller/Utilities/Settings.hpp)
### Logging

- By default, logging messages of level `info` and above are sent to the console.
- Set `ROCROLLER_LOG_LEVEL` (e.g., `Debug`) to change the logging level and emit debug messages.
- Set `ROCROLLER_LOG_CONSOLE=0` to disable console output.
- Set `ROCROLLER_LOG_CONSOLE_LEVEL` (e.g., `Debug`) to change the logging level and emit debug messages to the console.
- Set `ROCROLLER_LOG_FILE` to a file name to log output to.
- Set `ROCROLLER_LOG_FILE_LEVEL` (e.g., `Debug`) to change the logging level and emit debug messages to a file.

### Debugging

- Set `ROCROLLER_SAVE_ASSEMBLY=1` to write generated assembly code to a `.s` text file in the current directory. The file name is based on the kernel name. To specify a custom file name, set `ROCROLLER_ASSEMBLY_FILE` to your desired name.
- Set `ROCROLLER_RANDOM_SEED` to an integer to control the seed for the `RandomGenerator` used in unit tests.
- Set `ROCROLLER_BREAK_ON_THROW=1` to trigger a segfault when library code throws an exception. This allows GDB to break at the original failure point, not where the exception is rethrown. For more precise debugging, set a breakpoint at the exception constructor (e.g., `b std::bad_variant_access::bad_variant_access()`). Note: STL exceptions are not affected.
- Set `ROCROLLER_IGNORE_OUT_OF_REGISTERS=1` to allow kernel generation even when running out of registers. This lets you analyze peak register usage.
- Set `ROCROLLER_ENFORCE_GRAPH_CONSTRAINTS` and `ROCROLLER_AUDIT_CONTROL_TRACERS` to enable additional checks during graph creation and code generation. These checks are enabled by default in gtest/catch tests and in `rrperf run`.
- Set `AMD_COMGR_SAVE_TEMPS=1`, `AMD_COMGR_EMIT_VERBOSE_LOGS=1`, and `AMD_COMGR_REDIRECT_LOGS=stderr` to get detailed assembler debug output.
- Set `ROCROLLER_ASSEMBLER=Subprocess` and specify `ROCROLLER_DEBUG_ASSEMBLER_PATH=<assembler like amdclang>` to use an external assembler.

You can explicitly enable or disable several environment variables by setting the `ROCROLLER_DEBUG` bit field. This variable aggregates multiple debug options into a single value. The following table lists the options and their corresponding bits:

| Option                      | Bit (Hex Value) |
|-----------------------------|-----------------|
| `ROCROLLER_LOG_CONSOLE`     | 0 (0x0001)      |
| `ROCROLLER_SAVE_ASSEMBLY`   | 1 (0x0002)      |
| `ROCROLLER_BREAK_ON_THROW`  | 2 (0x0004)      |

For example, to enable `ROCROLLER_LOG_CONSOLE` and `ROCROLLER_BREAK_ON_THROW` while disabling `ROCROLLER_SAVE_ASSEMBLY`, set `ROCROLLER_DEBUG=0x0005`. To enable all options, use `ROCROLLER_DEBUG=0xFFFF`. To disable all, use `ROCROLLER_DEBUG=0x0000`.

Favour `AssertFatal` instead of `assert`. `AssertFatal` verifies assumptions and invariants via a conditional check. If it evaluates to false it prints detailed information and exits the program. You should also use the `ShowValue()` macro (defined in [Error_impl.hpp](lib/include/rocRoller/Utilities/Error_impl.hpp)) to print variable names and values that were used in the condition.

Example usage:

```cpp
auto vr = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto ar = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int32, 1);

// This call does NOT exit the program
AssertFatal(vr->registerCount() == ar->registerCount(), "Register counts are not equivalent");

// This call WILL exit the program and print relevant info
AssertFatal(
  vr->regType() == ar->regType(),
  ShowValue(Register::toString(vr->regType())),
  ShowValue(Register::toString(ar->regType())),
  "Register types are not equivalent"
);
```

The first argument is the condition to check. Additional arguments (such as `ShowValue()` or a custom message) are optional and can appear in any order after the condition.

You can also use `Throw<FatalError>("message")` to catch and report incorrect code.

## Analysis

### Performance Analysis

To run performance tests, use the `rrperf` tool. The `autoperf` command benchmarks multiple commits as well as your current workspace. For usage details, refer to the help documentation:

```bash
./scripts/rrperf autoperf --help
```

### Kernel Analysis

Setting the environment variable `ROCROLLER_KERNEL_ANALYSIS=1` will enable the following kernel analysis features built into rocRoller.

- Register Liveness
  - The file `${ROCROLLER_ASSEMBLY_FILE}.live` is created which reports register liveness.
  - Legend:
    | Symbol | Meaning |
    | ------ | ------- |
    | *space* | "Dead" register |
    | `:` | "Live" register |
    | `^` | This register is written to this instruction. |
    | `v` | This register is read from this instruction. |
    | `x` | This register is written to and read from this instruction. |
    | `_` | This register is allocated but dead. |

To view a summary plot of the generated file, run:

```bash
  python3 ./scripts/plot_liveness.py --help && \
    python3 ./scripts/plot_liveness.py ${ROCROLLER_ASSEMBLY_FILE}.live
```

and visit http://127.0.0.1:8050/.

### KernelGraph Visualization

The kgraph script can be run on an assembly file or on log output to generate a .dot file or rendered .pdf of the internal graph.  Run it with the `--help` option to see invocation.

When using the diff functionality, the coloring is as follows:
* Red means the node will be removed in the next step
* Blue means the node was added since the last step
* Yellow means both, and the node is temporary for this step

Run the following on RocRoller output in a .log file to extract dots printed with logger->debug("CommandKernel::generateKernelGraph: {}", m_kernelGraph.toDOT());:

```bash
./kgraph.py gemm.log -o dots
```
Log files can be found in the output of RRPerf runs. Use --omit_diff to omit colouring difference when extracting log output.

Run the following to compare multiple dot files:

```bash
./dot_diff.py dots_0000.dot dots_0001.dot dots_0002.dot -o dots
```

### Memory Access Visualization

The [GEMM client](client/src/gemm.cpp) can produce memory trace files, using `--visualize=True`, which can be rendered using the [show_matrix script](scripts/show_matrix).

The following commands can be used to visualize memory access patterns to png files:

```console
$ cd ${build_dir}

$ ./bin/client/rocRoller_gemm --M=512 --N=768 --K=512 --mac_m=128 --mac_n=256 --mac_k=16 --alpha=2.0 --beta=0.5 --workgroup_size_x=256 --workgroup_size_y=1 --type_A=half --type_B=half --type_C=half --type_D=half --type_acc=float --num_warmup=2 --num_outer=10 --num_inner=1 --trans_A=N --trans_B=T --loadLDS_A=True --loadLDS_B=True --storeLDS_D=False --scheduler=Priority --visualize=True --match_memory_access=False

Visualizing to gemm.vis
Wrote workitem_A.dat
Wrote workitem_B.dat
Wrote LDS_A.dat
Wrote LDS_B.dat
Wrote workgroups_A.dat
Wrote workgroups_B.dat
Wrote loop_idx_A.dat
Wrote loop_idx_B.dat
Result: Correct
RNorm: 1.4083e-05

$ ../scripts/show_matrix *.dat

Wrote LDS_A.png
Wrote LDS_B.png
Wrote loop_idx_A.png
Wrote loop_idx_B.png
Wrote workgroups_A.png
Wrote workgroups_B.png
Wrote workitem_A.png
Wrote workitem_B.png
```
