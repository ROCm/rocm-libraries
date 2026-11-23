# Origami Python bindings

`rocm-origami` provides the `origami` Python package for analytical GEMM kernel
selection. It requires Python 3.12 or later and builds a `cp312-abi3` wheel.
The extension uses the installed shared Origami library. It never builds or
bundles native Origami.

## Build against system ROCm

Install a ROCm development environment containing Origami's headers, shared
library, and CMake package. From the repository root:

```sh
CMAKE_PREFIX_PATH=/opt/rocm python -m pip install ./bindings/python/rocm-origami
```

This build has no dependency on ROCm Python packages. Its native libraries must
be discoverable through the system loader or the extension's relative paths.
A relocated tarball installation may require its normal loader configuration.

## Build against ROCm SDK wheels

Install the intended SDK, including development files. Pass its version and
supported targets to the isolated build:

```sh
python -m pip install 'rocm[libraries,devel]==<SDK version>'
ROCM_SDK_VERSION="$(rocm-sdk version)" \
ROCM_SDK_TARGETS=gfx942,gfx950 \
CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)" \
  python -m pip install ./bindings/python/rocm-origami
```

Use targets provided by that SDK. TheRock's helper can read the installed `rocm`
package's device extras automatically. SDK builds add the SDK identity to the
binding version, pin `rocm[libraries]` to that exact SDK, and expose matching
`device-<target>` extras. Before extension import, the generated initializer
uses `rocm_sdk.initialize_process` to load HIP and Origami and enforce the SDK
version. SDK failures propagate; a different system library is not a fallback.

Published wheels must be built and validated against SDK wheels. Relaxing the
exact pin for stable promotion belongs in TheRock's release tooling; this
package does not perform promotion.

## Native and Python development

Build and install native Origami separately:

```sh
cmake -S shared/origami -B build/origami -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/origami-prefix" -DORIGAMI_BUILD_TESTING=ON
cmake --build build/origami
ctest --test-dir build/origami --output-on-failure
cmake --install build/origami
CMAKE_PREFIX_PATH="$PWD/build/origami-prefix;/opt/rocm" \
  python -m pip install -e './bindings/python/rocm-origami[test]'
python -m pytest bindings/python/rocm-origami/tests
```

The native library is not rebuilt by an editable binding install. Rebuild and
reinstall it after changing native code. ABI compatibility is declared by the
native SONAME; an ABI break must update native versioning. Successful matched
build tests do not establish compatibility with every library retaining `.so.1`.

## Use

```python
import origami

hardware = origami.get_hardware_for_arch(origami.architecture_t.gfx942)
print(origami.datatype_to_bits(origami.data_type_t.Float))
```

The optional `[torch]` extra enables the Python selectors. The `origami.origami`
alias remains available with a deprecation warning.

TheRock builds binding wheels in a separate downstream step. The native SDK
build installs no copy of this Python package. Release validation must check
wheel contents, SDK ownership and loaded paths, all supported Python minors,
and applicable GPU tests before stable promotion.
