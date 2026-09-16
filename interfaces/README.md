# ROCm library interfaces

Status: proposed design with a working noncanonical rocBLAS vertical slice. The tree
implements a stable public-loader boundary and narrow implementation-provider protocols for
ROCm math libraries. It builds standalone or through the root opt-in, and it does not alter
existing math-library targets or install canonical ROCm library names.

**New here? Start with the documentation set: [docs/README.md](docs/README.md).** It explains
why the boundary exists, how the three layers fit together, and links every capability to the
test that proves it.

The working vertical slice contains:

- A versioned common provider bootstrap and a cross-platform module-load primitive (dlopen/LoadLibrary); the ABI-versioning hardening it wraps is Linux/ELF only.
- Narrow BLAS, solver, and RAND protocol headers.
- A generated `librocblas-loader.so.5` shadow DSO implementing all 1,219 callables in the
  current rocBLAS C header through a typed brute-force compatibility table.
- A single-export `librocblas-provider-system.so` that binds the canonical
  `librocblas.so.5` by DSO handle and fills that complete table. Unexpected missing symbols
  fail provider negotiation; six grouped-GEMM APIs are compatibility-adapted over ordinary
  GEMM when the selected rocBLAS 5 backend predates those spellings. The two public variadic
  device-memory helpers are normalized through an explicit array adapter with a current limit
  of 32 size arguments.
- A checked 1,219-row categorization ledger and an experimental ten-call narrow BLAS provider
  with a generated all-symbol facade mapping 1,156 of 1,162 compute spellings to typed
  requests. The six grouped-GEMM callables remain typed bridge-only entries because the
  narrow request has no audited per-group descriptor.
- A combined replacement BLAS recording provider plus separate legacy-shaped rocBLAS and
  hipBLASLt provider DSOs in one compatibility cohort.
- Loadable solver and RAND recording providers.
- Installed strict manifests for the exhaustive real provider and the first real narrow-v2
  provider. The narrow provider executes the single-batch FP32 vector-transform cluster
  (AXPY, SCAL, COPY, and SWAP, with 32- and 64-bit indices) against canonical rocBLAS.
- Parser-based API extraction using Clang LibTooling.
- API-policy and append-only enum validation tooling.
- Host-only unit and DSO integration tests, plus the ABI hardening proof suite.
- An isolated shadow package-config/install consumer test.

## Build

The `hip` package is produced by CLR. Point `CMAKE_PREFIX_PATH` at a TheRock ROCm
distribution containing that package; do not add CLR source include directories directly.

```shell
cmake -S rocm-libraries/interfaces -B build/interfaces -GNinja \
  -DCMAKE_PREFIX_PATH="$ROCM_PREFIX" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DClang_DIR="$LLVM_PREFIX/lib/cmake/clang"
cmake --build build/interfaces
ctest --test-dir build/interfaces --output-on-failure
cmake --build build/interfaces --target rocm-interfaces-api-snapshots
cmake --build build/interfaces --target rocm-interfaces-check-api-snapshots
cmake --build build/interfaces --target rocm-interfaces-check-rocblas-categorization
```

The same targets can be enabled from the repository root without changing the default root
build:

```shell
cmake -S rocm-libraries -B build/root -GNinja \
  -DROCM_LIBS_ENABLE_INTERFACES=ON \
  -DCMAKE_PREFIX_PATH="$ROCM_PREFIX" \
  -DLLVM_DIR="$LLVM_PREFIX/lib/cmake/llvm" \
  -DClang_DIR="$LLVM_PREFIX/lib/cmake/clang"
cmake --build build/root --target rocblas_loader_shadow \
  rocm_rocblas_bridge_provider_system rocm_blas_narrow_v2_provider_system
```

Real-GPU differential coverage is opt-in locally and mandatory in the dedicated GPU
workflow:

```shell
cmake -S rocm-libraries/interfaces -B build/interfaces-gpu -GNinja \
  -DCMAKE_PREFIX_PATH="$ROCM_PREFIX" \
  -DROCM_INTERFACES_BUILD_TOOLS=OFF \
  -DROCM_INTERFACES_CHECK_API_DRIFT=OFF \
  -DROCM_INTERFACES_ENABLE_GPU_TESTS=ON
cmake --build build/interfaces-gpu --target rocblas_gpu_differential_test
ctest --test-dir build/interfaces-gpu -R rocblas_gpu_differential --output-on-failure
```

The executable returns CTest skip code 77 when no GPU is visible. The
`interfaces-gpu-ci.yml` workflow invokes the executable directly, so lack of an assigned GPU
is a hard failure there rather than a false-green skip.

The snapshot target parses ten C and C++ entry profiles. Function metadata includes the
target linkage name, C linkage, visibility, inline marker, and templated kind; record metadata
includes layout where Clang can form one. `hipblaslt-ext` and the rocRAND/hipRAND C++ headers
are deliberately separate profiles so C++ ABI does not disappear inside a C-only inventory.
The check target regenerates all profiles, byte-compares them with `api/snapshots`, and also
checks the rocBLAS categorization ledger. With `BUILD_TESTING=ON`, it is registered in CTest as
`rocm_interfaces.api_snapshot_drift` by default because `ROCM_INTERFACES_CHECK_API_DRIFT`
defaults to `ON`; set that option to `OFF` to opt out explicitly. `BUILD_TESTING=OFF`
registers no CTest. The check is not part of the default build (`ALL`) or an automatically
wired presubmit, and it remains directly runnable with
`cmake --build build/interfaces --target rocm-interfaces-check-api-snapshots`.

This tree temporarily stages the public rocBLAS and rocRAND headers directly from their
projects. This is deliberate migration scaffolding: those headers are in scope to move
under `interfaces/`. HIP is different; the real implementation must retain a normal
package dependency on CLR's exported `hip::host` target.

## Noncanonical operation

The complete shadow rocBLAS loader is named `librocblas-loader`, not `librocblas`. The
installed `rocblas-system.json` manifest selects `librocblas-provider-system`; set
`ROCM_INTERFACES_ROCBLAS_PROVIDER_MANIFEST` to select an alternate installed manifest.
`ROCM_INTERFACES_REAL_ROCBLAS_LIBRARY` overrides the provider's default
`librocblas.so.5` backend for testing and controlled deployments. The direct
`ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER` variable remains a test corner.

The experimental semantic loader is `librocblas-loader-narrow-v2`; its installed
`rocblas-narrow-v2-system.json` manifest is selected with
`ROCM_INTERFACES_BLAS_V2_PROVIDER_MANIFEST`. The older hand-written narrow slice remains as
`librocblas-loader-narrow` and uses `ROCM_INTERFACES_BLAS_PROVIDER` only in tests.

Canonical mode is intentionally absent until every exported declaration is classified,
all required adapters exist, package-config parity is demonstrated, and coexistence tests
cover the ABI majors being published.

## Documentation

- [docs/README.md](docs/README.md) - the documentation index and read order.
- [docs/01-architecture.md](docs/01-architecture.md) - how the layers fit together.
- [docs/02-why-a-stable-boundary.md](docs/02-why-a-stable-boundary.md) - the threat model.
- [docs/03-abi-and-versioning-contract.md](docs/03-abi-and-versioning-contract.md) - the normative contract.
- [docs/04-hardening.md](docs/04-hardening.md) - each proof and what it stops.
- [docs/05-extending.md](docs/05-extending.md) - how-to recipes.
- [docs/07-status-and-roadmap.md](docs/07-status-and-roadmap.md) - done vs planned.

Reference layer: [provider-protocols.md](docs/provider-protocols.md),
[rocblas-provider-clusters.md](docs/rocblas-provider-clusters.md),
[audit-findings.md](docs/audit-findings.md),
[api-change-process.md](docs/api-change-process.md).
