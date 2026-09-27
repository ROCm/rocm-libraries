# Initial ABI audit findings

## hipSOLVER is a public facade ABI

`hipsolver.h` is installed as the master header and includes the dense, dense64, sparse,
refactor, types, and functions headers even though their paths contain `internal`.
`libhipsolver` has a SOVERSION and exported CMake target. The current tree contains more than
five hundred `HIPSOLVER_EXPORT` declarations. All are therefore in the compatibility
inventory.

On AMD, `hipsolverCreate` passes the public result pointer directly to
`rocblas_create_handle`; destroy, stream, deterministic-mode, workspace, and dense routines
repeatedly cast `hipsolverHandle_t` to `rocblas_handle`. This is representation coupling, not
opaque use. A correct migration must move both sides to a shared loader-owned identity cohort
or initially preserve the identical handle object.

hipSOLVER is not only a thin symbol rename. It also owns dynamically allocated Jacobi/SVD
info objects, workspace policy, sparse and refactor objects, logging behavior, and a host
LAPACK fallback for functionality absent from rocSOLVER. It remains a facade with no new
provider protocol, but every one of those behaviors needs an explicit edge classification.

## Public enum coupling

hipSOLVER re-declares or aliases hipBLAS operation, fill, diagonal, and side values. These
numeric relationships are already public compatibility constraints. Provider protocols now
use canonical rocBLAS/rocRAND public enum and status types directly, and facade builds must
static-assert every promised alias.

## rocBLAS and hipBLASLt behavior

rocBLAS may consult hipBLASLt/TensileLite and then fall back to legacy rocBLAS solutions.
Direct hipBLASLt calls do not receive the legacy fallback. A unified provider cohort must
preserve these separate policy entry points until an independently tested rationalization.

## Caller storage and C++ surface

All complete public records require generated size, alignment, field-offset, and reserved
capacity reports. Opaque pointers may be redirected through dynamic allocation. hipblaslt-ext
requires a separate C++ ABI inventory including mangled names, layouts, inline/template weak
definitions, and ownership of LT algorithm tokens.

## RAND visibility

RAND's provider build must use hidden C++ visibility and an explicit export allowlist. Weak
template definitions in headers may resolve in callers, but that does not justify exporting
the provider's template instantiations. The recording provider demonstrates the intended
single-symbol export boundary.

The parser audit also found that `rocrand/rocrand.hpp` is not self-contained: an included
MTGP32 header calls `printf` without including its declaration. The snapshot profile currently
forces `<cstdio>` into that translation unit and records this as compatibility scaffolding, not
as an interface dependency. The header should be fixed before it moves into this tree.

## Fortran

Legacy in-library Fortran bindings are excluded. They are not deployment-portable and hipfort
is the supported direction. Snapshot profiles must exclude Fortran modules and tests must
ensure no Fortran target enters a loader package export set.
