# Shared packaged generator

hipBLASLt prepares `tensilelite-generator` for its dense libraries and ExtOps.
Other embedded consumers can request it with
`HIPBLASLT_PREPARE_TENSILELITE_GENERATOR=ON`, even when hipBLASLt device generation,
clients and test artifacts are disabled. The target is visible outside the
`block()` containing the hipBLASLt subdirectory.

By default, preparation builds the canonical TensileLite wheel from the current
source and refreshed build metadata. Set the `TENSILELITE_GENERATOR_WHEEL` CMake
file path to consume an explicit artifact instead. This does not install into
the selected Python environment: each prepared generator extracts the wheel into
its own build directory. The selected Python still needs TensileLite's declared
Python dependencies. A command-scoped Python startup hook keeps editable installs
from overriding the selected packages, including in spawned workers. Native rocisa, StinkyTofu and their resources are provided
by the native build/install, independently of the pure-Python wheel.

## Consumer interface

Use an already prepared target:

```cmake
include(hipblaslt_codegen)
create_device_library(
    TARGET sparse-device-libraries
    GENERATOR tensilelite-generator
    LOGIC_PATH "${CMAKE_CURRENT_SOURCE_DIR}/logic"
    OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/Tensile"
    ARCHES gfx942
)
```

For a separately configured consumer, stage the producer's `tensilelite-devel` component,
including its native dependencies, then select a wheel explicitly:

```cmake
find_package(hipblaslt CONFIG REQUIRED)
include(TensileLiteGenerator)
tensilelite_prepare_generator(
    TARGET tensilelite-generator
    WHEEL "${TENSILELITE_GENERATOR_WHEEL}"
    PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
    TOOL_BIN_DIRS "${toolchain_bin_dir}"
    ENVIRONMENT "ROCM_VERSION=${hip_VERSION}" "ROCM_PATH=${ROCM_PATH}"
)
```

TheRock consumers pass `TENSILELITE_ROCM_ADAPTER=therock-ci` and, when available,
`THEROCK_PACKAGE_VERSION` instead of `ROCM_PATH`. `ASAN` and `TSAN` preparation
options select the host sanitizer environment; product sanitizer flags remain
generation options. Consumers needing another module, such as ExtOps, use
`tensilelite_generator_command(target command dependencies)` and use **both**
outputs in their custom command. They do not read target properties or construct
the extraction/native paths themselves.

The installed helper uses relocatable native-support metadata. It does not need
the producer's source checkout or build directory. The supplied wheel must match
the producer's distribution version and its source revision when one is known.
An archive with no Git identity keeps the existing policy: package/version checks
apply, but source pairing is disabled. Supplied artifacts are checked for readable
identity metadata and a matching package/version/revision during configuration
and again before extraction when their inputs change. Failures do not select a
different generator. Release-content checks run when producing the canonical wheel;
package/build tests exercise the packaged commands. Consumers do not maintain a
required-module inventory.

The generator's dependencies cover wheel extraction and native inputs.
`create_device_library` additionally tracks logic edits and membership changes.
Explicit `GENERATOR` cannot be combined with explicit `CODEGEN_ROOT` or
`PYTHON_EXECUTABLE`. Existing source calls retain their behavior during rollout.

## hipSPARSELt rollout

`HIPSPARSELT_USE_PACKAGED_TENSILELITE` defaults to OFF. When enabled, standalone
hipSPARSELt requests the shared generator from its embedded hipBLASLt build.
TheRock's staged mode requires `TENSILELITE_GENERATOR_WHEEL` and the development
support from the matching hipBLASLt producer. Neither mode needs test artifacts.

This capability is step 2 of the AIHPBLAS-3989 migration, following PR #11022.
TheRock's canonical artifact producer/consumer integration is step 3. Default
activation waits for both repositories' dependency references to include that
integration (step 4). Removing `CODEGEN_ROOT` follows adoption (step 5).
`rocisa_stinkytofu_adaptor` was removed in #15452; it is not a packaging
prerequisite. Conversion now lives in native rocisa/StinkyTofu.
