# ROCm library interfaces

Status: proposed design, prototype-backed. This tree prototypes a stable public-loader
boundary and narrow implementation-provider protocols for ROCm math libraries. It is
intentionally standalone: it does not alter the existing math-library builds or install
canonical ROCm library names.

**New here? Start with the documentation set: [docs/README.md](docs/README.md).** It explains
why the boundary exists, how the three layers fit together, and links every capability to the
test that proves it.

The working vertical slice contains:

- A versioned common provider bootstrap and cross-platform module loader.
- Narrow BLAS, solver, and RAND protocol headers.
- A generated `librocblas-loader.so.5` shadow DSO implementing all 1,213 declarations in the
  current rocBLAS C header through a typed brute-force compatibility table.
- A checked 1,213-row categorization ledger and an experimental ten-call narrow BLAS provider
  with a generated all-symbol facade mapping all 1,156 compute spellings to typed requests.
- A combined replacement BLAS recording provider plus separate legacy-shaped rocBLAS and
  hipBLASLt provider DSOs in one compatibility cohort.
- Loadable solver and RAND recording providers.
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

The snapshot target parses ten C and C++ entry profiles. Function metadata includes the
target linkage name, C linkage, visibility, inline marker, and templated kind; record metadata
includes layout where Clang can form one. `hipblaslt-ext` and the rocRAND/hipRAND C++ headers
are deliberately separate profiles so C++ ABI does not disappear inside a C-only inventory.
The check target regenerates all profiles and byte-compares them with `api/snapshots`; it is
the presubmit hook for draft-to-launch header drift.

This tree temporarily stages the public rocBLAS and rocRAND headers directly from their
projects. This is deliberate migration scaffolding: those headers are in scope to move
under `interfaces/`. HIP is different; the real implementation must retain a normal
package dependency on CLR's exported `hip::host` target.

## Noncanonical operation

The complete shadow rocBLAS loader is named `librocblas-loader`, not `librocblas`. Its
brute-force test provider is selected with `ROCM_INTERFACES_ROCBLAS_BRIDGE_PROVIDER`. The
earlier narrow vertical slice remains as `librocblas-loader-narrow` and uses
`ROCM_INTERFACES_BLAS_PROVIDER`. Both variables are test corners, not production selection
mechanisms; production selection uses validated manifests supplied by the distribution.

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
