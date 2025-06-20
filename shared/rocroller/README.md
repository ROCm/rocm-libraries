
# AMD's rocRoller Assembly Kernel Generator

rocRoller is a software library for generating AMDGPU kernels.

[![Master Branch Code Coverage Report for gfx90a](https://img.shields.io/badge/Code%20Coverage-gfx90a-informational)](http://math-ci.amd.com/job/enterprise/job/code-coverage/job/rocRoller/job/master/Code_20coverage_20gfx90a_20report)
[![Master Branch Code Coverage Report for python](https://img.shields.io/badge/Code%20Coverage-Python-informational)](http://math-ci.amd.com/job/enterprise/job/code-coverage/job/rocRoller/job/master/Python_20Code_20coverage_20gfx90a_20report)
[![Master Branch Performance Report for gfx90a](https://img.shields.io/badge/Performance-gfx90a-critical)](http://math-ci.amd.com/job/enterprise/job/performance/job/rocRoller/job/master/Performance_20Report_20for_20gfx90a)
[![Master Branch Performance Report for gfx942](https://img.shields.io/badge/Performance-gfx942-critical)](http://math-ci.amd.com/job/enterprise/job/performance/job/rocRoller/job/master/Performance_20Report_20for_20gfx942)
[![Master Branch Generated Documentation](https://img.shields.io/badge/Documentation-Generated-informational)](http://math-ci.amd.com/job/enterprise/job/documentation/job/rocRoller/job/master/Generated_20Docs)

## Jump to

- [AMD's rocRoller Assembly Kernel Generator](#amds-rocroller-assembly-kernel-generator)
  - [Building the library](#building-the-library)
    - [Quick start instructions](#quick-start-instructions)
    - [Detailed commandline instructions](#detailed-commandline-instructions)
    - [Cmake and make commands](#cmake-and-make-commands)
      - [CMake Options](#cmake-options)
    - [Running the tests](#running-the-tests)
    - [Updating pregenerated GPUArchitecture yaml files](#updating-pregenerated-gpuarchitecture-yaml-files)
  - [GEMM client](#gemm-client)
  - [File Structure](#file-structure)
  - [Coding Practices](#coding-practices)
    - [Style](#style)
    - [ISO C++ Standard](#iso-c-standard)
    - [Documentation](#documentation)
      - [Building Documentation](#building-documentation)
  - [PR submissions](#pr-submissions)
  - [Testing](#testing)
  - [Logging](#logging)
  - [Debugging](#debugging)
  - [Profiling](#profiling)
  - [Performance](#performance)
  - [Kernel Analysis](#kernel-analysis)
  - [Graph Visualization](#graph-visualization)
  - [Memory Access Visualization](#memory-access-visualization)

## Building the library

### Quick start instructions

To build rocRoller using Docker:
```
git clone --recurse-submodules git@github.com:ROCm/rocRoller.git rocRoller
cd rocRoller
./docker/user-image/start_user_container
docker exec -ti -u ${USER} ${USER}_dev_clang bash
cd /data
mkdir -p build
cd build
cmake -DROCROLLER_ENABLE_TIMERS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j
```

To build rocRoller natively on Ubuntu 22 (jammy):
```
# As root, once:
apt update
apt install -y libboost-container1.74-dev libopenblas-dev ninja-build

# As regular user:
git clone --recurse-submodules git@github.com:ROCm/rocRoller.git rocRoller
cd rocRoller
mkdir -p build
cd build
cmake -DROCROLLER_ENABLE_TIMERS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j
```

To run the unit tests:
```
cd rocRoller/build
make test
```

### Detailed commandline instructions

The rocRoller repository includes several Docker files.  We recommend
using these for development work.

[Instructions for building and launching docker](docker/README.md) are available.

The rocRoller repo can be cloned from the internal GitHub repo. The
tip of the `master` branch contains the latest commits:
https://github.com/ROCm/rocRoller

```
git clone --recurse-submodules git@github.com:ROCm/rocRoller.git rocRoller
```

From inside the docker container launched previously, the library can
be built using these steps.  The cloned directory should be available
from the `/data` directory in the docker container.  First create a
`build` directory in the root of the cloned repo and enter that
directory:

```
mkdir -p build
cd build
```

### CMake and make commands

CMake will use the default compiler specified in the environment variable `CXX`. If you want to set it manually, configure using the cmake command:
```
CXX=<g++ or clang++ path> CC=<gcc or clang path> cmake .. <cmake options>
```

Then build:
```
make -j$(nproc)
```

#### CMake Options

- Ninja is an alternative to `make` and is generally faster at
  building the rocRoller library.  Include `-G Ninja` in your `cmake`
  command and then use `ninja` to build.
- `-DSKIP_CPPCHECK=ON` Skips running every source file through
  `cppcheck` (skip is on by default).  This speeds up builds.
- `-DCMAKE_BUILD_TYPE=Debug` will build a debug version with debugging
  information for gdb.  Specify `Release` instead for release builds.
- `-DBUILD_SHARED_LIBS=OFF` will build the project as a static
  library, `ON` (default) will build it as a shared library.
- `-DYAML_BACKEND=YAML_CPP` (default) or `-DYAML_BACKEND=LLVM`
  determines the backend used for YAML serialization.
- `-DINTERNAL_MRISAS=<CUSTOM_MRISA_XML_FILES>` where `<CUSTOM_MRISA_XML_FILES>`
  is a semi-colon delimitted list of paths to MRISA XML files. Allows the user
  to specify additional MRISA XML files to use during compilation.
- To see what options your build directory was configured with, run `cmake -LAH`
- It is highly recommended _not_ to reconfigure a build directory with cmake.
  Doing so can lead to caching problems with the configuration.  When in doubt,
  remove your build directory and start fresh.
- `-DROCROLLER_USE_PREGENERATED_ARCH_DEF=OFF` will not use the
  [GPUArchitecture yaml file(s)](GPUArchitectureGenerator/pregenerated) checked
  into the repo, and will instead generate them from scratch using the MRISA XML
  files and [GPUArchitecture_def](GPUArchitectureGenerator/include/GPUArchitectureGenerator/GPUArchitectureGenerator_defs.hpp) file.

### Running the tests

There are three ways to launch the test applications:

```
make test
```

This will run our unit tests using `ctest` (one process per test).

Alternatively, from the build directory simply run:
```
./rocRollerTests
```

This command will run all of the tests all together. The tests will
run faster than using `make test`, however, if there is an
segmentation fault in any of the tests it will stop.

Individual tests can be run at the command line from the build
directory. For example:
```
./rocRollerTests --gtest_filter="*MemoryInstructionsExecuter.GPU_ExecuteFlatTest1Byte*"
```

Tests that require a GPU are prefixed with `GPU_`.  If you are running
on a machine without a supported GPU, you can use:
```
ctest -LE GPU
```
or
```
./bin/rocRollerTests --gtest_filter="-*GPU_*"
```

A full list of gtests available can be listed using the command:
```
./bin/rocRollerTests --gtest_list_tests
```

To prevent exceptions from being eaten by GoogleTest:
```
--gtest_catch_exceptions=0
```
This allows catch/throw to work in GDB.

To turn a test failure directly into a GDB breakpoint:
```
--gtest_break_on_failure
```

### Updating pregenerated GPUArchitecture yaml files

The [GPUArchitecture yaml file(s)](GPUArchitectureGenerator/pregenerated) checked
into the repo should be updated anytime there are changes to the underlying MRISA XML files or the
[GPUArchitecture_def](GPUArchitectureGenerator/include/GPUArchitectureGenerator/GPUArchitectureGenerator_defs.hpp) file.

Updated yaml files can be copied from `./build/share/rocRoller/split_yamls/` after
building with `-DROCROLLER_USE_PREGENERATED_ARCH_DEF=OFF`.

```bash
mkdir ./build
cd ./build
cmake -DROCROLLER_USE_PREGENERATED_ARCH_DEF=OFF ../
make -j GPUArchitecture_def
cp ./share/rocRoller/split_yamls/*.yaml ../GPUArchitectureGenerator/pregenerated/
cd ../GPUArchitectureGenerator/pregenerated/
../../scripts/format_yaml.py -I *.yaml
```

## GEMM client

To explore GEMM workloads and do performance testing, rocRoller
provides a GEMM client.  The GEMM client is built alongside the
library by default.

To run the GEMM client from your build directory:

```
./bin/client/rocRoller_gemm --help
```

## File Structure

 - `Foo.hpp`: Contains definitions for classes, concepts, etc.  Functions should be declaration-only.
 This is meant to allow for easy reading of the interface.
 - `Foo_impl.hpp`: Contains definitions for short (inlinable) functions.
 Should be included at the bottom of `Foo.hpp`.  This makes it easy to move definitions between here and `Foo.cpp`, since they are not within the class definition.
 - `Foo_fwd.hpp`: Contains forward declarations for all types defined in `Foo.hpp`, type aliases such as `FooPtr` for `std::shared_ptr<Foo>` and `std::variant`.  Should have zero or very few includes.
 - `Foo.cpp`: Contains definitions for longer functions.

Generally, we want as much as possible to be inlined since performance is important.

## Coding Practices

### Style

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

#### Building Documentation

[Graphviz](https://graphviz.org/) and [Doxygen](https://www.doxygen.nl/index.html) both need to be installed to build the html documentation.

The Doxygen documentation can be built via the make command from the main build directory:
```
make -j$(nproc) docs
```

This will generate an HTML website from the markdown readme files and Doxygen comments and source/object relationships. The `index.html` file can be found in `doc/html` from the root of your local repository.


## PR submissions

- PRs submitting should have fewer than `500` lines of code change, unless of course those changes are merely lines moved or deleted.
- If a new feature cannot be implemented in fewer than `500` lines of code then consider either refactoring, or splitting the feature into two or more PRs.
- When opening a PR, details of what the feature does are essential.
- PR authors should provide a description on how to review the PR, such as, outlining key changes. This is especially important if the PR is complex, novel, or on the higher side of lines of code change.
- As mentioned in the coding style section, PR code needs to be formatted with `clang-format` version 13. This can be done manually.
- rocRoller's dockers already have this version of `clang-format` and formatting can be automated using githooks. The command for installing the hook to the local repo: `.githooks/install`
- If a PR alters the compilation process in a way that causes the performance job to fail when building the master branch, adding the label `ci:no-build-master` will instead compare the PR against the latest build of the master branch.


## Testing

Each new feature is required to have a test.
- Test sources are placed in the `test` folder.
- CPP Files for Unit Tests should be included in the `rocRollerTests` executable in [CMakeLists.txt](https://github.com/ROCm/rocRoller/blob/master/test/CMakeLists.txt).

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

Additionally, there are tests that run against guidepost kernels that have been generated by Tensile. See the [README](https://github.com/ROCm/rocRoller/blob/master/test/unit/GemmGuidePost/README.md) for more information.

## Logging

- By default, logging messages of level `info` and above are sent to the console.
- To disable console output, set the environment variable `ROCROLLER_LOG_CONSOLE=0`.
- To change the logging level, set the environment variable `ROCROLLER_LOG_LEVEL`.  For example, setting `ROCROLLER_LOG_LEVEL` to `Debug` will cause debug messages to be emitted.
- To log to a file, set the environment variable `ROCROLLER_LOG_FILE` to a file name.

## Debugging

- Setting the environment variable `ROCROLLER_SAVE_ASSEMBLY=1` will cause assembly code to be written to a text file in the current working directory as it is generated. The file name is based on the kernel name and has a `.s` extension. To manually set the assembly file name to something else, set `ROCROLLER_ASSEMBLY_FILE` to the desired name.
- Setting the environment variable `ROCROLLER_RANDOM_SEED` to an unsigned integer value will set the seed of the `RandomGenerator` used by the unit tests.
- Setting the environment variable `ROCROLLER_BREAK_ON_THROW=1` will cause exceptions thrown directly by library code to cause a segfault. This causes GDB to break at the original point of failure, instead of at the point where it is rethrown by the Generator class.
   - You can also set a breakpoint at the constructor of an exception class (e.g. `b std::bad_variant_access::bad_variant_access()` ), and GDB will more reliably break there than on `catch throw`.
- Setting `ROCROLLER_ARCHITECTURE_FILE` will overwrite the default GPU architecture file generated at `source/rocRoller/GPUArchitecture_def.msgpack`. Currently supported file formats are YAML and Msgpack.
- STL exceptions will not be affected by this.  Sometimes a better stack trace can be obtained by placing a breakpoint in the constructor of the particular exception that is being thrown.
- Setting `AMD_COMGR_SAVE_TEMPS=1` `AMD_COMGR_EMIT_VERBOSE_LOGS=1` `AMD_COMGR_REDIRECT_LOGS=stdout` can help provide more assembler debug output.
- Set `ROCROLLER_ASSEMBLER=Subprocess` and `ROCROLLER_DEBUG_ASSEMBLER_PATH=<assembler like amdclang>` to use an external assembler.


To explicitly enable/disable some of the mentioned environment variables, `ROCROLLER_DEBUG` can be set accordingly. `ROCROLLER_DEBUG` is a bit field that aggregates options together. The following options are covered by `ROCROLLER_DEBUG`:
| Option | Respective Bit |
| ------ | -------------- |
| `ROCROLLER_LOG_CONSOLE` | 0 (0x0001)
| `ROCROLLER_SAVE_ASSEMBLY` | 1 (0x0002)
| `ROCROLLER_BREAK_ON_THROW` | 2 (0x0004)

Therefore, enabling `ROCROLLER_LOG_CONSOLE` and `ROCROLLER_BREAK_ON_THROW` and disabling `ROCROLLER_SAVE_ASSEMBLY` would require `ROCROLLER_DEBUG=0x0005`. Enabling/disabling all corresponding options requires `ROCROLLER_DEBUG=0xFFFF` / `ROCROLLER_DEBUG=0X0000`, respectively.

`AssertFatals` can be used in place of `assert`. `AssertFatal` is similar to `assert` but offers more debugging functionality.

An example usage:

```
auto vr = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto ar = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int32, 1);

// This call will NOT exit the program
AssertFatal(vr->registerCount() == ar->registerCount(), "Register counts are not equivalent");

// This call WILL exit the program and print relevant info
AssertFatal(vr->regType() == ar->regType(), ShowValue(Register::toString(vr->regType()),
            ShowValue(Register::toString(ar->regType()), "Register types are not equivalent");
```

The first argument is a condition that is checked, `ShowValue()` is a macro defined in `rocRoller/Error_impl.hpp` where it prints out the variable name and value. The last argument is a custom message the developer can print to screen. If the condition is false then the program exits and the file name, line number, and corresponding arguments passed to AssertFatal are then printed.

Note: The condition check must be the first argument, but the other arguments can be in any order and are optional.

The use of `Throw<FatalError>("message")` can also be used to catch incorrect code.

## Profiling

On Linux, we can use the "perf" tool to sample the callgraph and
generate a flame graph.

Using FlameGraph: https://github.com/brendangregg/FlameGraph

CMake configure with `-DROCROLLER_ENABLE_TIMERS=ON` and compile.

Then run with the steps outlined in [scripts/flamegraph.py](scripts/flamegraph.py):
```
  perf record -F 99 -g ./bin/rocRollerTests --gtest_filter="KernelGraph*03"
  perf script > out.perf
  ~/FlameGraph/stackcollapse-perf.pl out.perf > out.folded
  ~/FlameGraph/flamegraph.pl out.folded > kernel.svg
```

Additionally, when using the trace Dockerfile, you can invoke rrperf to profile RocRoller or Tensile guideposts with Omniperf.
To see how this works, check rrperf's help documentation:
```
  ./scripts/rrperf profile --help
```

## Performance

Invoke the rrperf tool to run performance tests.
The autoperf command will test the performance of multiple commits and your current workspace.
To see how this works, check rrperf's help documentation:
```
  ./scripts/rrperf autoperf --help
```

## Kernel Analysis

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

## Graph Visualization

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

## Memory Access Visualization

The [GEMM client](client/gemm.cpp) can produce memory trace files, using `--visualize=True`, which can be rendered using the [show_matrix script](scripts/show_matrix).

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
