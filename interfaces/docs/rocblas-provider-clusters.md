# rocBLAS provider narrowing map

This document defines the required classification for the exhaustive AST inventory and records
what the executable v2 spike proved. It is directional input, not an adopted provider ABI.

| Cluster | Provider primitive | Public variation absorbed |
| --- | --- | --- |
| `edge.context` | Context create/destroy | Handle spelling and version queries |
| `edge.policy` | No execution primitive | Stream, pointer, math, atomics, numerics, logging |
| `vector.transform` | Typed vector transform | Precision, 32/64-bit width, batching |
| `vector.reduction` | Typed reduction | Result type/location, conjugation, index result |
| `matrix.vector` | Matrix-vector execute | Transpose, structured matrix kind, batching |
| `matrix.rank_update` | Rank update execute | Rank-1/rank-k, symmetric/Hermitian, batching |
| `matrix.matmul` | General contraction | Precision, compute type, batching, GEMM_EX |
| `matrix.structured` | Structured multiply/update | Symmetric/Hermitian families |
| `matrix.triangular` | Triangular multiply/solve | Side, fill, diagonal, transpose, batching |
| `matrix.transform` | Copy/transpose/scale | GEAM, DGMM, matrix transform variants |
| `solution.query` | Solution enumeration/heuristic | Public query spellings and workspace probes |
| `solution.execute` | Execute opaque solution token | Index width and public algorithm wrappers |
| `lt.plan` | LT descriptor normalization | Layout/preference/algorithm attribute objects |
| `lt.execute` | LT matmul/transform | Heuristic token and workspace handling |
| `extension` | Named optional capability | Operations not yet defensibly generalized |
| `bridge_only` | Typed compatibility slot | Semantics requiring further audit |

Every public function row records its cluster, canonical operand model, edge-owned state,
fallback policy, and narrowing blocker. A type, batching, or integer-width suffix is not by
itself a valid blocker.

## Closed inventory

`api/rocblas-categorization.json` classifies all 1,219 parser-visible C callables. Generation
fails on an unknown spelling; there is no catch-all category. The current closure is:

This closes the current header surface, not the adoption binary audit. Release Linux and
Windows rocBLAS artifacts may still contain accidental C++ or unheadered C exports; those
must be added to the relevant public ABI-line facade or deliberately removed before its
immutable launch snapshot.

| Cluster | Public callables | Proposed provider primitive |
| --- | ---: | --- |
| Edge diagnostic/lifecycle/version | 9 | None |
| Edge policy | 18 | Context options and request execution state |
| Edge transfer | 16 | None; HIP copy service at the loader edge |
| Edge memory | 14 | None long-term; seven accidental helpers retain bridge slots during migration |
| Vector transform | 126 | `vector_transform` |
| Vector reduction | 162 | `vector_reduce` |
| Vector rotation | 90 | `vector_rotate` |
| Matrix-vector | 148 | `matrix_vector` |
| Rank update | 168 | `rank_update` |
| General matmul | 66 | `matmul_query` plus `matmul`; six grouped-GEMM callables remain bridge-only |
| Structured matrix | 146 | `structured_matrix` |
| Matrix transform | 49 | `matrix_transform` |
| Triangular matrix | 63 | `triangular_matrix` |
| Triangular matrix-vector | 144 | `matrix_vector` with triangular semantics |

The 1,162 compute spellings split into 1,156 semantic adapters and six grouped-GEMM
bridge-only callables. The 1,156 adapters narrow to nine execution callbacks plus matmul
enumeration. Precision prefixes, explicit-datatype variants, 32/64-bit indices, ordinary,
pointer-array, and strided batching become request fields. This is a 99.1% reduction in the
adapted operation entry points without pretending semantically different algorithms are
identical.

The concrete proposal is
`protocols/include/rocm/interfaces/experimental/blas_narrow_v2.h`. The generated
`librocblas-loader-narrow-v2` facade constructs one of these typed requests for every one of
the 1,156 semantically adapted compute callables and dispatches through the reduced table.
Generation asserts the exact adapter count and fails on an unknown primitive. The six
grouped-GEMM callables retain typed slots in the separate brute-force compatibility bridge;
the narrow facade validates the handle and returns `rocblas_status_not_implemented` rather
than fabricating a lossy request. The brute-force bridge remains the NFC reference, not an
escape hatch in the narrow table.

## Recommended boundary

The common descriptors express memory representation separately from mathematical shape:

- `rocm_blas_v2_execution` carries stream, index width, batch kind/count, and behavior flags.
- `rocm_blas_v2_memory` distinguishes a base allocation from a pointer array and carries
  element offset and batch stride.
- Vector and matrix descriptors carry public `rocblas_datatype`. Matrix descriptors also
  carry dense/banded/packed storage, general/symmetric/Hermitian/triangular kind, fill,
  diagonal, leading dimension, and bandwidth.
- Scalars retain public datatype plus host/device location. Public enums such as operation,
  fill, diagonal, side, datatype, GEMM algorithm, and status are shared directly.

Protocol-owned discriminants are fixed-width `uint32_t` typedefs with append-only named
constants; they do not rely on a C compiler's enum layout. Public rocBLAS enums remain their
canonical public types so the loader cannot introduce a second numeric mapping.

Separate request types are retained where validation, outputs, or provider optimization are
materially different. A single `execute(opcode, void*)` entry was considered and rejected:
it would erase type checking, make struct evolution global, and turn every provider into a
large switch over unrelated operand invariants.

The proposed table has these calls:

1. `create_context` / `destroy_context` establish a gfx-pinned implementation context.
2. `vector_transform` covers SCAL, COPY, SWAP, and AXPY.
3. `vector_reduce` covers DOT/DOTC/DOTU, NRM2, ASUM, IAMAX, and IAMIN, including result
   location and type.
4. `vector_rotate` covers ROT, ROTG, ROTM, and ROTMG. ROT/ROTG scalar in/out operands remain
   explicit; ROTM/ROTMG also retain the public five-element parameter block with its batch
   stride because it may reside on the device.
5. `matrix_vector` covers general, banded, symmetric, Hermitian, packed, and triangular
   multiply/solve forms.
6. `rank_update` covers GER/GERU/GERC and symmetric/Hermitian packed or dense rank-1/rank-2.
7. `matmul_query` and `matmul` cover GEMM/GEMMT and EX forms. GEMMT fill is an explicit field.
   Fixed-size provider tokens do
   not expose native solution objects. Query results retain the public solution index where
   that behavior is observable.
8. `structured_matrix` covers SYMM/HEMM and SYRK/HERK/SYR2K/HER2K/SYRKX/HERKX.
9. `triangular_matrix` covers TRMM, TRSM, and TRTRI. TRSM_EX's caller-provided inverse is
   distinct from provider workspace and is represented separately.
10. `matrix_transform` covers GEAM and DGMM. GEAM_EX retains C input versus D output, compute
    type, auxiliary dimension, and the canonical public extended-operation enum.

## Executable closure and remaining gaps

The spike has three deliberately coexisting artifacts:

- `librocblas-loader.so` and `librocblas-provider-bruteforce-recording.so` prove the complete
  1,219-symbol compatibility-table baseline.
- `librocblas-loader-narrow-v2.so` exports the same 1,219 symbols but translates 1,156 of the
  1,162 compute spellings to the semantic protocol. Its six grouped-GEMM exports return
  `rocblas_status_not_implemented` for a valid handle. There is no generic `void**`, vararg
  argument packet, public-function ordinal, or fallback provider slot in this path.
- `librocblas-provider-narrow-v2-recording.so` exports only the bootstrap query and implements
  the 12-pointer table: lifecycle, nine execution callbacks, and matmul enumeration.

The all-symbol link test runs against both facades. A second test executes representative
AXPY, DOT, ROT, GEMV, GER, GEMM, SYMM, TRSM, and GEAM calls through the narrow provider and
checks that grouped GEMM with a valid handle returns `rocblas_status_not_implemented`. This
establishes structural closure, not numerical equivalence.

Before adopting the request structs as ABI, peers must settle these named gaps:

1. The 57 edge callables are classified but the v2 spike implements only handle lifecycle;
   most currently return `not_implemented`. Stream/pointer/math/atomics/workspace state,
   transfers, version reporting, and the seven allocator-object bridges need their production
   edge implementation and tests. None requires a provider-table entry.
2. Per-operation validation, quick-return, aliasing, negative-increment, scalar-location, and
   asynchronous lifetime behavior is not reproduced by the recording provider. Differential
   tests against legacy rocBLAS are required before converting a cluster. The first conversion
   is executable for single-batch FP32 AXPY, SCAL, COPY, and SWAP through the system-backed
   narrow-v2 provider; other datatypes and batch forms remain to be migrated.
3. Logical matrix dimensions in the requests are normalized, but transposed storage extent
   conventions and banded/packed offsets need a written invariant and adversarial tests.
4. Classic GEMM-to-LT eligibility, no-solution fallback, solution-index observability, memory
   query sizing, and numerics/logging hooks remain policy work. The request has fields for the
   data; the spike does not claim behavioral equivalence.
5. The six grouped-GEMM callables carry arrays of shapes, operations, leading dimensions, and
   scalars. The current homogeneous matmul request cannot preserve those per-group semantics,
   so they remain bridge-only until an audited grouped descriptor exists.
6. `matmul_query` is exercised as a provider contract, but current classic public spellings in
   this inventory execute matmul directly. The hipBLASLt facade will be the primary client of
   enumeration and must validate token lifetime and cohort rules.
7. The generated translation is source-derived. Accidental release-binary exports still need
   the Linux/Windows audit already called out above.

These are implementation and validation gaps, not unclassified public operations. If review
finds a request cannot carry a required semantic value, the protocol record should grow before
adoption; adding the original public function as a provider slot is not the default remedy.

## rocBLAS versus hipBLASLt routing

The narrow BLAS table is the classic rocBLAS policy domain; the BLASLt table remains a
separate domain. For a rocBLAS call, the loader applies the frozen eligibility gates, asks
the same-cohort BLASLt provider first when required, and then invokes classic `matmul` on a
no-solution outcome. A direct hipBLASLt call never invokes classic fallback.

This works with both physical layouts:

- legacy `librocblas-provider-tensile.so` and
  `libhipblaslt-provider-tensilelite.so` own separate contexts and tables but share a cohort;
- a replacement DSO answers both table queries and may share private state internally.

No context pointer or provider token crosses between the two legacy DSOs. Cohort identity is
selection compatibility, not representation compatibility.

## Edge-owned and transitional functions

The 57 non-compute functions do not justify provider entries:

- handle lifecycle, stream/pointer/math/atomics/performance modes, logging/event policy,
  version strings, status strings, and solution-fitness policy are loader state;
- matrix/vector copy helpers use HIP runtime services and do not select a math provider;
- ordinary workspace and device-memory query state belongs to the loader context.

Seven declared allocation helpers expose rocBLAS implementation objects or allocator policy.
They are categorized `bridge_only`. The two variadic functions are normalized at the edge to
`(count, const size_t*)`; forwarding C varargs through a dispatch table is neither portable
nor defensible. The accidental `rocblas_device_malloc_base` family remains typed in the
brute-force bridge until solver/hipBLAS consumers are migrated, then should be hidden before
the adoption snapshot if release policy permits.

Six compute callables are also explicitly `bridge_only`:
`rocblas_{s,d}gemm_grouped_batched{,_64}` and
`rocblas_gemm_grouped_batched_ex{,_64}`. Grouped GEMM is not ordinary pointer-array batching:
each group has its own shape, operations, leading dimensions, scalars, and group size. The
homogeneous narrow matmul request cannot encode those arrays, so the compatibility bridge
retains the typed calls while the narrow facade reports `rocblas_status_not_implemented`.

## Migration recommendation

Use the complete generated bridge as the NFC starting point, and the executable narrow v2
facade as the proposed endpoint, but do not publish the 1,219-slot table as the provider SDK.
Convert and differentially validate clusters in this order:

1. context/policy and edge-local helpers;
2. vector transform/reduction and matrix-vector/rank-update;
3. classic GEMM/GEMM_EX plus explicit LT eligibility/fallback tests;
4. TRSM/TRMM and structured rank-k, which exercise workspace and aliasing rules;
5. remaining homogeneous spellings and the seven accidental allocator bridges.

For each converted row, change its ledger disposition from the brute compatibility slot to
the named narrow primitive and run differential tests against the legacy provider. Delete the
brute slot only after every supported old public loader maps through an adapter. This permits
the current provider generation to stay narrow while old public ABI loaders remain permanent.
