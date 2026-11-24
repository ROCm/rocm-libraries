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
from overriding the selected packages, including in spawned workers. Native
rocisa, with StinkyTofu linked into it, comes from the native build or install,
independently of the pure-Python wheel.

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
`create_device_library` requires `GENERATOR`; generating directly from a source
tree (`CODEGEN_ROOT`, `HIPBLASLT_CODEGEN_ROOT`) is no longer supported. The
generator also owns the Python interpreter and the host sanitizer runtime.

## hipSPARSELt

hipSPARSELt always generates with the packaged generator. Standalone hipSPARSELt
requests the shared generator from its embedded hipBLASLt build. Neither mode
needs test artifacts.

TheRock's staged mode uses the development support from the staged hipBLASLt.
With `TENSILELITE_GENERATOR_WHEEL` it uses that wheel. Otherwise it builds the
canonical wheel itself from the shared source and hipBLASLt's installed identity
metadata, so both products generate from the same package version and source
revision. Configuration fails if this source tree or `THEROCK_PACKAGE_VERSION`
does not match that producer. TheRock does not yet pass `THEROCK_PACKAGE_VERSION`
to hipSPARSELt; until it does, the ROCm identity comes from hipBLASLt's metadata.

A hipBLASLt build with `TENSILELITE_GENERATOR_WHEEL` installs that wheel into
its test artifact instead of building another one.

A TheRock stage that builds the wheel once and hands it to both products remains
future work, as does consuming a prebuilt rocisa wheel.
`rocisa_stinkytofu_adaptor` was removed in #15452; it is not a packaging
prerequisite. Conversion now lives in native rocisa/StinkyTofu.
