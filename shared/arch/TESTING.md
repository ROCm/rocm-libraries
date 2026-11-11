# Architecture catalog testing

Catalog tests run on the host with Python 3.9+, CMake 3.25.2+, and a C++14 compiler.
They require no HIP, GPU, or third-party Python test framework.

```sh
cmake -S shared/arch -B build/arch \
  -DROCM_ARCH_BUILD_TESTING=ON -DROCM_ARCH_BUILD_INTEGRATION_TESTING=ON
cmake --build build/arch
ctest --test-dir build/arch --output-on-failure
```

`ROCM_ARCH_BUILD_INTEGRATION_TESTING` requires a monorepo checkout. Disable it
when testing the standalone source distribution.

The tests cover:

- Schema validation, exact names, processor/generic separation, and invalid inputs.
- C++ constexpr identity round trips and enumeration.
- Identical lists and digests across CMake, Python, and C++.
- Reproducible generation and discovery of added or removed definitions.
- FFT inline default policies, exclusions, Python enum names, and serialization.
- Shared dependency resolution from source and installed packages, including
  repeated inclusion of the dependency module and separate consumer directories.

## Origami consumers

Origami's architecture tests require HIP headers and a C++17 compiler. They do
not execute GPU kernels or require its Catch2 suite.

```sh
cmake -S shared/origami -B build/origami \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DORIGAMI_BUILD_TESTING=OFF \
  -DORIGAMI_BUILD_ARCH_TESTING=ON
cmake --build build/origami
ctest --test-dir build/origami --output-on-failure
```

These tests check the shared type, supported-model enumeration, canonical name
round trips, excluded or invalid identities, heuristic-key hashing, and Formocast
identity assignment. The public header checks that
every non-excluded processor has architecture constants at compile time.

## Packaging

Build a wheel and source distribution with `python3 -m build shared/arch`. Install
each into an isolated environment and check enumeration and exact lookup outside
the checkout. Install the native package into a separate prefix, then configure
and build an external consumer using `find_package(rocm-arch 0.4 CONFIG REQUIRED)`.
For Origami, also build an external consumer of its installed public headers to
verify the exported dependency.

The `Architecture Catalog Consumers` job in
[Shared CTest CI](../../.github/workflows/shared-ctest-ci.yml) runs the catalog and
FFT consumer tests. Catalog tests establish identity and build integration;
architecture-specific kernel support requires the consumer's device tests.
