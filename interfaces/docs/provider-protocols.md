# Provider protocol specification

Status: proposed target contract; partially implemented. This spec states the destination
provider protocol, not current prototype behavior. Where the two differ, the
implementation-status note in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md#implementation-status-prototype)
is authoritative, and [07-status-and-roadmap.md](07-status-and-roadmap.md) records what is
DONE, COMMITTED-NEXT, and ASPIRATIONAL.

## Contract shared by all domains

Provider protocols are implementation contracts, not public client APIs. They are C ABI
tables with a single exported bootstrap symbol. Every table and extensible record starts with `struct_size`, `abi_major`, and `abi_minor`.
Provider selection enforces an exact response major, a response minor floor, and a `dispatch_table_size` floor.
`rocm_interfaces.table_abi_negotiation` proves that an exact table and a newer/larger table
are accepted while an older minor or a short required prefix is rejected. Current domain
loaders still request their full table, and no consumer reads the dispatch-table
header, so per-domain optional-tail consumption is not implemented; see
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md#implementation-status-prototype).

Modules are opened with local symbol scope and remain pinned while any dispatch table,
context, token, or loader object refers to them. Provider selection happens once at the
library-specific context boundary. An operation cannot select a different provider.

Manifests are strict JSON documents. Module paths are relative to the manifest and cannot
escape its directory. Entries name a provider, compatibility cohort, domain, query symbol,
priority, and numeric gfx list; zero is the wildcard. An exact gfx match outranks a wildcard
before priority is considered. A provider response is validated in two stages, not one. At
parse time unknown keys, missing/non-string identities, absolute or escaping module paths,
nonexistent modules, invalid priorities, negative/out-of-range gfx values, duplicate
domain/id/gfx tuples, and empty provider arrays are rejected. Parsing is atomic: one bad
entry leaves the registry unchanged. At
selection the runtime checks the response's `abi_major` against
`ROCM_INTERFACES_ABI_MAJOR`, requires its `abi_minor` to meet the runtime floor, validates the
requested domain and cohort, and requires `dispatch_table_size` to cover the requested table
size; a provider failing any of these is skipped. The individual required
entry points are null-checked by the domain loader after selection returns (for example
`BlasContext::create`), which throws if a required pointer is null -- they are not inspected
during selection. See the implementation-status note in
[03-abi-and-versioning-contract.md](03-abi-and-versioning-contract.md#implementation-status-prototype)
and [01-architecture.md](01-architecture.md#how-a-provider-gets-picked).

Host services provide allocation, deallocation, and structured tracing. Providers must not
retain pointers to request records after a call. Device pointers and asynchronous workspace
remain valid according to the public operation's stream semantics, not merely until the C
call returns.

### Public enums and status values

Protocol operations use canonical public family enum and status types wherever one exists:

- BLAS and solver use rocBLAS public types and `rocblas_status`.
- RAND uses rocRAND public types and `rocrand_status`.
- hipBLAS, hipBLASLt, hipSOLVER, and hipRAND are facades whose aliases and translations are
  checked against the canonical provider family.

There is no duplicate normalized status namespace at the operation boundary. Protocol-only
enums are allowed only for concepts with no public counterpart, such as batch encoding or
provider bootstrap errors.

Existing public enum names, numeric values, and underlying types are immutable. New values
must have explicit numbers and must not reuse holes that a released header could already
interpret differently. Reordering source declarations is prohibited even when explicit
values would make it ABI-neutral, because ordering leaks into generated bindings and user
code. Aliases must retain compile-time equality assertions. The API snapshot check rejects
removal, renumbering, or underlying-type changes.

## BLAS protocol

### Context and selection

The loader owns the public rocBLAS handle and retains stream, pointer mode, math mode,
atomics mode, numerics mode, workspace policy, and device identity. Provider context
creation receives the selected gfx and effective policy. Providers never receive or inspect
the public handle.

rocBLAS and hipBLASLt form one compatibility cohort but use separate query domains. This
supports both required deployment shapes:

- A replacement implementation ships one DSO whose query function returns the classic table
  for the BLAS domain and the LT table for the BLASLt domain.
- The legacy migration keeps rocBLAS/Tensile and hipBLASLt/TensileLite in separate provider
  DSOs--concretely `librocblas-provider-tensile.so` and
  `libhipblaslt-provider-tensilelite.so`--and assigns their manifest entries the same cohort
  identity.

An integrated replacement may instead use a name such as
`librocm-blas-provider-<implementation>.so` and answer both domains from its one exported
query symbol. The filenames are deployment policy; the two independently versioned table
contracts and domain queries are the compatibility mechanism.

Physical module identity is therefore not required; compatible cohort identity is. rocBLAS
fallback may only consult the LT entry selected from its own cohort. Provider-owned algorithm
tokens cannot cross cohort boundaries. hipBLAS is a public facade over the same cohort and
has no provider protocol of its own.

Each domain owns its provider-private context lifecycle. In particular, a hipBLASLt provider
never receives a rocBLAS provider's context pointer: equal cohort IDs assert behavioral and
selection compatibility, not shared address-space types. An integrated replacement may share
state internally because it owns both tables. The loader coordinates separate legacy providers
through loader-owned device, stream, policy, and fallback state expressed in protocol values.

### Operand model

The narrow protocol separates orthogonal dimensions that the public ABI expresses through
thousands of symbol spellings:

- Public `rocblas_datatype` for input, output, scale, and compute types.
- Public `rocblas_operation`, fill, side, and diagonal enums.
- Explicit 32- or 64-bit index width.
- Single, pointer-array, and strided batching; grouped batching remains bridge-only until the protocol carries per-group descriptors.
- Host- or device-resident scalar values.
- Vector length/increment/stride.
- Matrix rows, columns, leading dimension, and batch stride.
- Effective stream and workspace policy from the edge context.

There are now two executable protocol levels. The generated compatibility bridge has one
typed slot for every one of the 1,219 current rocBLAS callables and is suitable for the NFC
implementation migration. The earlier vector/matmul table remains a small vertical test.
The proposed destination is the experimental v2 table with ten semantic calls described in
`rocblas-provider-clusters.md`. The public inventory contains 1,162 compute callables; a
generated shadow facade translates 1,156 into those typed requests. A recording provider
covers the complete table structurally, and the system-backed provider implements the first
migrated slice: single-batch FP32 AXPY, SCAL, COPY, and SWAP through `vector_transform`.
The six grouped-GEMM callables remain `bridge_only`
because the narrow matmul request cannot represent per-group shapes, operations, leading
dimensions, and scalars; the narrow edge returns `rocblas_status_not_implemented` for a
valid handle. This proves structural closure, not numerical or behavioral equivalence for
the remaining clusters, so the table is intentionally not yet the adopted provider ABI.

### Operation clusters

The complete public inventory is assigned to context/policy, vector transform, vector
reduction, matrix-vector, rank update, matmul, structured matrix, triangular, transform,
solution query, solution execution, LT plan, LT execute, extension, or bridge-only clusters.
The clustering report records every public spelling collapsed by type, width, or batching,
and calls out operations that cannot yet share a semantic request.

### LT solution semantics

Layouts, preferences, matmul descriptors, transform descriptors, and their attributes are
loader-owned records. Heuristic calls submit a complete normalized problem. Algorithm
results contain opaque provider tokens tied to the provider lease that produced them.
Tokens cannot cross providers, devices, or incompatible protocol majors.

The initial migration preserves existing behavior:

- A rocBLAS call may try the coordinated LT provider and fall back to the legacy rocBLAS
  solution set.
- A hipBLASLt call sees only LT solutions and does not gain a legacy fallback.

Rationalizing this behavior is a later, separately measured change.

### Transitional bridge

The exhaustive typed bridge has one entry per current rocBLAS header callable. It exists to make
legacy implementation conversion mechanical while narrow clusters are proven. It may not
use untyped argument arrays or expose public handle storage. Each bridge entry carries a
documented target cluster or a concrete blocker. The bridge is not the long-term provider
SDK and must not become the only maintained contract.

## Solver protocol

The solver provider receives a loader-owned BLAS client-services table, not a public handle
or a BLAS provider-private context. The callbacks expose normalized BLAS operations and carry
an opaque loader token. This permits the solver and BLAS providers to remain separate DSOs,
prevents either from inspecting the other's private types, and preserves provider-cohort
selection. The initial table contains vector and matmul calls and grows by appending narrow
operations actually needed by solver implementations.

Legacy rocSOLVER code that currently inspects rocBLAS internals cannot be migrated merely by
relabeling that pointer. Its provider adapter must either replace those accesses with the
client services and explicit context options, or use a temporary, versioned cohort-private
bridge between the matched legacy rocBLAS and rocSOLVER providers. Such a bridge is not part
of the public provider SDK and may not accept a loader/public handle.

Requests describe public rocBLAS data and policy enums, index width, dense operands, pivots,
tau, info, workspace, and operation-specific dimensions. Workspace query and execution are
separate operations so a facade can preserve both rocSOLVER and hipSOLVER conventions.

The full protocol is divided into:

- LU, Cholesky, QR/LQ/RQ/QL, bidiagonal, and tridiagonal factorization.
- Solve, inverse, condition estimation, and refinement.
- Orthogonal or unitary generation and application.
- Symmetric, Hermitian, nonsymmetric, and generalized eigenproblems.
- Singular-value decomposition.
- Reduction and auxiliary routines.
- Sparse direct and iterative solve.
- Refactorization.

Dense hipSOLVER is a grandfathered public facade. Its loader handle participates in the same
BLAS/solver cohort and initially reproduces the existing fact that `hipsolverCreate` creates
a rocBLAS-compatible context. Loader-owned Jacobi, SVD, parameter, sparse, and refactor
objects use dynamic allocation during migration. Sparse and refactor handles are explicit
child contexts rather than casts of a dense handle.

Host LAPACK fallbacks, logging activation, device-memory-size queries, and workspace mutation
are facade policy and are inventoried individually. They must not leak into the narrow solver
provider contract merely because the current hipSOLVER implementation performs them.

## RAND protocol

RAND has an independent provider cohort. A generator retains public rocRAND algorithm,
ordering, seed, offset, and quasirandom dimension values. Generator kind distinguishes
device, host, and host-blocking behavior. Provider selection is deferred until a generation
request supplies a usable device/gfx key, then remains fixed for the generator lifetime.

Generation requests normalize raw integer, uniform, normal, log-normal, Poisson, and
discrete operations. Output type and distribution parameters are explicit because the public
API often encodes them in the function name. The provider also covers discrete-distribution
lifecycle and direction-vector/scramble-constant queries.

hipRAND is a public facade over this protocol and has no separate provider contract. RAND
provider DSOs must hide all C++ template and weak implementation artifacts; only the query
symbol is exported.

## Threading and failure behavior

Loader-owned mutable state is synchronized. Independent contexts can execute concurrently.
Providers state their stronger restrictions through capabilities; absence of a restriction
means calls on distinct contexts may overlap.

Invalid public objects and arguments are rejected at the edge using the public family's
status values. Provider bootstrap failures use `rocm_interfaces_status` because no public
operation is active. Exceptions never cross a C ABI. A provider returning a malformed table
or unknown identity is rejected before context creation.

## Alternatives considered

- Keeping independent public and provider enum sets was rejected: perpetual numeric mapping
  is error-prone and adds no value where the public enums are already defensible.
- Making the provider table identical to every public function is retained only as a
  transitional bridge; it would freeze decades of accidental API structure into new code.
- Per-operation provider selection was rejected because it complicates token, workspace,
  handle, fallback, and pointer-identity semantics without a product requirement.
- Reusing public opaque handle layouts was rejected because hipSOLVER already demonstrates
  how representation coupling spreads across libraries.
