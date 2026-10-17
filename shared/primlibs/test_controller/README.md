# Primitives libraries test controller

The test controller lets a library skip whole GoogleTest tests, or individual test input sizes, from a plain text
*control file* instead of with `#ifdef`s and hard-coded checks in the test sources. Rules can be limited by
architecture (eg. `gfx942`, `<apus>`), build type (eg. `asan`, `windows`), and input size.

It is header-only and depends only on the C++17 standard library, GoogleTest, and the HIP runtime, so it can be used
by any of the primitives libraries (hipCUB, rocPRIM, rocThrust, rocRAND, hipRAND).

## Contents

| Path | Description |
| --- | --- |
| `include/test_utils_controller.hpp` | `TestController`, the `ControlledTest` fixtures, and the `CHECK_*` macros. |
| `include/test_controller_detail.hpp` | Internal helpers (environment variables, ctest device selection). |
| `TestController.cmake` | CMake helpers for adding the controller to a project. |
| `test/test_controller_tests.cpp` | Unit tests for the controller itself. |

## Adding the test controller to a project

1. Include the CMake helpers from the project's test `CMakeLists.txt`:

   ```cmake
   include("${ROCM_LIBRARIES_ROOT}/shared/primlibs/test_controller/TestController.cmake")
   ```

2. Configure each test target. `PROJECT_DIR` is the name of the directory the control file is installed into,
   relative to the installed test binaries:

   ```cmake
   test_controller_configure_target(${TEST_TARGET} PROJECT_DIR myproject)
   ```

3. Add the project's control file. It is copied into `BUILD_DIR` whenever it changes (custom target
   `<dependent target>_copy_control_file`) and installed into `INSTALL_DESTINATION`:

   ```cmake
   test_controller_add_control_file(myproject
     SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/control.txt"
     BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}"
     INSTALL_DESTINATION "${CMAKE_INSTALL_BINDIR}/myproject"
   )
   ```

   At runtime, the control file is searched for relative to the running test binary, in
   `<binary dir>/../control.txt` (build tree) and `<binary dir>/<PROJECT_DIR>/control.txt` (install tree).
   If both exist, the most recently modified one is used.

4. Optionally, build the controller's own unit tests as one of the project's test targets, using the source
   path in `${TEST_CONTROLLER_UNIT_TEST_SOURCE}`. Give the target a project-specific name so that it does not
   clash with other projects in a superbuild.

See `projects/hipccl/hipccl3/hipcub/test/CMakeLists.txt` and `projects/hipccl/hipccl3/hipcub/test/hipcub/CMakeLists.txt`
for an example.

## Writing controlled tests

```cpp
#include "test_utils_controller.hpp"

class MyTest : public test_controller::ControlledTest<>
{};

TEST_F(MyTest, Sizes)
{
    std::vector<size_t> sizes = {1, 1000, 1 << 30};
    // Removes any sizes that are disabled by the control file.
    for(const auto size : CHECK_SIZE_FILTERS(sizes))
    {
        // ...
    }
}
```

- Test fixtures that inherit from `ControlledTest` (or `ControlledTestWithParam` for value-parameterized tests)
  are skipped automatically when the control file disables them for all sizes.
- `CHECK_SIZE_ENABLEMENT(size)` skips the test if a single, fixed size is disabled.
  `CHECK_SIZE_ENABLEMENT_WITH_CONTINUE(size)` does the same inside a loop.
- `CHECK_SIZE_FILTERS(sizes)` filters a vector of sizes in place and returns the result.
- Tests that use non-scalar size types (eg. `std::tuple<size_t, size_t>`) can pass a *size transformer* as a template
  argument to `ControlledTest`. See the documentation of `TestController::set_size_transformer` in
  `test_utils_controller.hpp`.

## Control file format

Each non-empty, non-comment (`#`) line has the format:

```
/test_name_regex/ : /arch_regex/ : size_limit[units] : build_type[, build_type...] : "skip_msg"
```

- `test_name_regex` is matched against the fully qualified test name (`TestSuiteName.TestName`). The `Id<integer>/`
  prefix used by tests that are compiled in parallel is stripped before matching.
- `arch_regex` is matched against the gfx id of the current device (eg. `gfx942`), or `nvidia` on the CUDA backend.
  It can contain keywords in angle brackets (eg. `/<mi300-family>|gfx1100/`), which expand to the architectures they
  represent. Available keywords: `<all>`, `<amd>`, `<nvidia>`, `<apus>`, `<navi2x-family>`, `<navi3x-family>`,
  `<navi4x-family>`, `<mi100-family>`, `<mi200-family>`, `<mi300-family>`. See the `keywords` map in
  `test_utils_controller.hpp` for their definitions.
- `size_limit` is an integer: all sizes `>=` the limit are skipped. It can have a unit (`K`, `Ki`, `M`, `Mi`, `G`,
  `Gi`). Use `*` to skip the whole test, or `0` to run the test but skip all of its sizes.
- `build_type` is a comma- or space-separated list of `asan`, `valgrind`, `windows`, `linux`, or `*` (any). The rule
  only applies when one of the listed build types matches.
- `skip_msg` is printed when tests or sizes are skipped.

Rules are examined from top to bottom. Example:

```
/MyDeviceSort\/.*/ : /<apus>/ : 4Gi : *    : "Skipping sizes over 2^32 on APUs"
/MyAllocatorTests\.Timing/ : /<all>/ : * : valgrind, asan : "Skipping timing-sensitive test under Valgrind and ASAN."
```

## Environment variables

| Variable | Description |
| --- | --- |
| `PRIMLIBS_TEST_CONTROL_FILE` | Path to the control file to use, instead of searching for it relative to the test binary. |
| `PRIMLIBS_EXTRA_TC_INFO` | When set to `1`, prints the control file path that's used, the individual sizes that were skipped, and the control file lines that caused them to be skipped. |
