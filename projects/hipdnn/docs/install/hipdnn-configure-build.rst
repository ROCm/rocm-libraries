.. meta::
  :description: hipDNN install 
  :keywords: Component, ROCm, install

***************************
Configure hipDNN builds
***************************


Release build (default)
=======================

```bash
cmake -GNinja ..
```

Debug build
===========

```bash
cmake -GNinja -DCMAKE_BUILD_TYPE=Debug ..
```

Code coverage build
===================

```bash
cmake -GNinja -DHIPDNN_ENABLE_COVERAGE=ON ..
ninja coverage
# Unit tests will be run and coverage reports will be generated in build/coverage_report/
```

Address sanitizer build
=======================

```bash
cmake -GNinja -DBUILD_ADDRESS_SANITIZER=ON ..
ninja check
# Note: Some HIP-related tests may be skipped due to AddressSanitizer incompatibility
```

Build specific components
=========================

```bash
# Build without plugins
cmake -GNinja -DHIP_DNN_BUILD_PLUGINS=OFF ..

# Build without frontend
cmake -GNinja -DHIP_DNN_BUILD_FRONTEND=OFF ..

# Build without backend
cmake -GNinja -DHIP_DNN_BUILD_BACKEND=OFF ..
```

``ROCM_PATH``, ``ROCM_CMAKE_PATH``, and ``CMAKE_INSTALL_PREFIX``
================================================================

If the ROCm ``bin`` folder is included in your system path then the AMD toolchain should be detected automatically. 
If not, these CMake variables can be used to assist CMake in the tool discovery.

- ``ROCM_PATH``: Specifies the root ROCm folder location and the toolchain folders are hard-coded using that path, skipping auto-detection of the toolchain (does not have a default value). **DO NOT SET ROCM_PATH IN YOUR ENVIRONMENT.** Setting ROCM_PATH in the environment will cause the compiler check to fail. Instead, use the -D option to cmake. E.g.: `-DROCM_PATH=/path/to/rocm`.
- ``ROCM_CMAKE_PATH`` (preferred): Similar to ``ROCM_PATH`` but relies on CMake's built-in detection to locate toolchain. (Default: `/opt/rocm` (Linux) / `C:/dist/therock` (Windows)). Can be set in your system environment. Will be set automatically if the ROCm bin folder is in your system path.

If ``ROCM_PATH`` is set using the -D option to cmake then it will take precedence over ``ROCM_CMAKE_PATH``.

The HIP compiler is required to build some integration tests but is not required for the hipDNN library itself.

Use the following CMake variable to control where the hipDNN library files will be installed when the ``install`` target is run:

``CMAKE_INSTALL_PREFIX``: Specifies where hipDNN will be installed (defaults to ``ROCM_PATH`` if ``ROCM_PAth`` is set, then ``ROCM_CMAKE_PATH`` if set, otherwise uses the CMake system default).

These variables can all be set independently:

```bash
# Default: Use system path to locate ROCm folder, install path is unset.
cmake -GNinja ..

# Install hipDNN to custom location, find ROCm dependencies in the default location
cmake -GNinja -DCMAKE_INSTALL_PREFIX=/custom/install/path ..

# Both custom
cmake -GNinja -DROCM_CMAKE_PATH=/custom/rocm -DCMAKE_INSTALL_PREFIX=/another/path ..
```

Clang tools
===========

Different versions of Clang tools are required. For example, clang-format version 18 and clang-tidy version 20. The hipDNN project tool discovery provides two mechanism to assist with finding the needed version of each tool.

Version Suffix
--------------

Before searching for the tool using it's standard name, a search will be made for a tool that has the version appended as a suffix. E.g. before looking for `clang-format` a search for a file named `clang-format-18` will be run first, and if that fails then a search will be made for `clang-format`. Similarly, `clang-tidy-20` will be searched-for first, and then `clang-tidy`. This approach can be used if it is possible to modify the Clang toolchain folder(s) on your system to give the tools the corresponding names.

``LLVM_TOOLS_SEARCH_PREFIX``
----------------------------

As an alternative to the above, ``LLVM_TOOLS_SEARCH_PREFIX`` can be set as a prefix for the folder path where the Clang tools are installed, such that `${LLVM_TOOLS_SEARCH_PREFIX}18/bin` is where the Clang version 18 tools are located, and `${LLVM_TOOLS_SEARCH_PREFIX}20/bin` is where the Clang version 20 tools are located. The CMake configuration step will automatically select the required version for each tool from these folders. For example with `-DLLVM_TOOLS_SEARCH_PREFIX=c:\tools\clang` the the following folders will be searched for Clang tools (depending on the version of each tool that is needed):

- ``c:\tools\clang18\bin``
- ``c:\tools\clang20\bin``
- ``c:\tools\clang\bin``

Build targets
=============

.. note::

   Make is supported for all targets. Configure with ``cmake -G "Unix Makefiles" ..`` if it is not the default generator in your environment. For parallel builds, use `make -j$(nproc)` on Linux. Unlike `ninja`, `make` does not build in parallel by default.

All targets support parallel builds with ninja.

| Target | Description |
|--------|-------------|
| \<no target\> | Build all components |
| `check` or `check-verbose` | Build and run all tests (see [Testing](./Testing.md)) |
| `unit-check` or `unit-check-verbose` | Build and run exclusively the unit tests and API tests (minimal version of `check`) |
| `integration-check` or `integration-check-verbose` | Build and run exclusively the E2E integration tests (this is the bulk of the testing time) |
| `install` | Install libraries and headers |
| `format` | Auto-format all C++ source files |
| `check_format` | Check code formatting compliance |
| `coverage` | Run `check` and generate test coverage reports (requires `-DHIPDNN_ENABLE_COVERAGE=ON`) |
| `unit-coverage` or `integration-coverage` | Run `unit-check` or `integration-check` (respectively) and generate test coverage reports (requires `-DHIPDNN_ENABLE_COVERAGE=ON`) |
| `current-coverage` | Generate test coverage reports using coverage data already on disk (does not automatically run `check`; requires `-DHIPDNN_ENABLE_COVERAGE=ON`) |
| `clean` | Clean build artifacts |
| `validate_test_names` | Validates test names conform to naming rules |
| `generate_hipdnn_data_sdk_headers` | Generate C++ headers from schema (`.fbs`) files |

The following example build commands are equivalent (depending on which generator was used) and will build the `check` target, to build and run all tests.

Using ``cmake`` to invoke build (regardless of which generator was used):

```bash
projects/hipdnn/build> cmake --build . --target check
```

If ``Ninja`` was used as the generator:

```bash
projects/hipdnn/build> ninja check
```

If a Makefile-type generator was used (not recommended):

```bash
projects/hipdnn/build> make check
```

