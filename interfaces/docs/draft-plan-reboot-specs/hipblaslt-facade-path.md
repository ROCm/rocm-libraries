# hipBLASLt facade path specification

Status: integrated specification for the narrow hipBLASLt GEMM facade path. It defines the APIs
and ownership boundaries needed to qualify one public GEMM path before horizontal expansion.

## Scope

This specification covers the selected hipBLASLt public API and ABI subset, one representative
GEMM, the private provider path, shadow deployment, one implementation-private backend boundary,
and coexistence of public ABI lines.

This specification does not define the complete hipBLASLt surface. Remaining public C-linkage
functions require explicit disposition through the hipBLASLt horizontal specification.

## Representative GEMM

| Property | Value |
| --- | --- |
| M, N, K | 8192, 2560, 320 |
| A, B, C, D storage | FP16 |
| Accumulation and scalars | FP32 |
| Operations | None, none |
| Layout | Packed column-major |
| C and D | Alias |
| Workspace | At most 32 MiB, 16-byte aligned |
| Device order | gfx90a, gfx942, gfx950 |

The mathematical result is `D = A × B` because alpha is one and beta is zero.

## Terms

| Term | Meaning |
| --- | --- |
| Public context | The client-visible hipBLASLt handle and its facade-owned state. |
| Provider lease | The immutable broker result that keeps a selected provider module resident. |
| Provider binding | Facade-private state containing a provider lease, validated operations, and provider context. |
| Provider context | Opaque provider-owned state created for one public context. |
| Provider query | The provider module entry point that returns identity, metadata, and operations. |
| Provider operation table | Fixed-layout C function-pointer record used after a provider query succeeds. |
| Provider adapter | Private code translating boundary records into one implementation's calls and results. |
| Implementation backend | The current hipBLASLt implementation reached privately by the adapter. |

The ownership chain is:

```text
client → public context → provider binding → provider lease + provider context
```

User code receives only the public context. It does not receive a provider lease, binding, or
provider context.

## Provider boundary API

**Components:** broker, provider module, provider adapter, facade.

The provider boundary is a distribution-private C ABI. It uses fixed-layout records and opaque
state; no C++ type, implementation handle, or exception crosses it.

```cpp
extern "C" {

rocm_interfaces_status rocm_interfaces_provider_query_v1(
    const rocm_interfaces_provider_request* request,
    rocm_interfaces_provider_response* response);

}
```

The request names the domain, required operation-table prefix, required ABI minor, and host
services. The response provides provider identity, build identity, operation-table pointer,
operation-table size, and capability mask.

The boundary defines records for the representative GEMM inputs, results, workspace, stream,
algorithm token, provider context, and statuses. Every reader validates a record's ABI header and
advertised size before reading it.

The broker validates the query response and operation-table header. The facade validates required
callbacks and capabilities before creating a provider context. The provider validates foreign,
stale, or invalid algorithm tokens before using them.

## Broker API

**Component:** hipBLASLt broker.

The broker runs once when a public context is established. It discovers candidates, validates
their query response, orders compatible candidates, and returns a provider lease.

```cpp
namespace rocm::interfaces {

class ProviderLease;

class ProviderRegistry {
public:
  std::shared_ptr<const ProviderLease> select(
      rocm_interfaces_domain domain,
      std::string_view architecture,
      uint32_t required_table_size,
      const std::string& required_cohort = {},
      uint16_t required_abi_minor = ROCM_INTERFACES_ABI_MINOR);
};

class ProviderLease {
public:
  const std::string& provider_id() const noexcept;
  const std::string& cohort_id() const noexcept;
  const void* table() const noexcept;
  uint32_t table_size() const noexcept;
  uint64_t capability_mask() const noexcept;
};

} // namespace rocm::interfaces
```

The broker reads approved discovery inputs, applies deterministic ordering, returns one immutable
lease, and records one diagnostic for every rejected candidate. It does not create a provider
context or dispatch ordinary GEMM operations.

## Provider module and adapter API

**Components:** hipBLASLt provider module and provider adapter.

The provider module is a loadable private artifact. It exports only the provider query entry point.
Its query returns a fully initialized operation table for the selected GEMM profile.

The provider adapter is the only component that uses current hipBLASLt implementation types,
headers, statuses, and loading rules. It translates each boundary record into an implementation
call and translates every result or failure back into the provider status taxonomy.

The adapter opens its implementation backend locally, validates the private backend record before
use, retries a later request after transient acquisition failure, and never exposes implementation
symbols through the provider module.

## Facade API

**Component:** hipBLASLt compatibility facade.

The facade owns the public library identity, public object state, public error behavior, and the
following public C API subset:

```text
hipblasLtCreate
hipblasLtDestroy
hipblasLtMatrixLayoutCreate
hipblasLtMatrixLayoutDestroy
hipblasLtMatrixLayoutSetAttribute
hipblasLtMatmulDescCreate
hipblasLtMatmulDescDestroy
hipblasLtMatmulDescSetAttribute
hipblasLtMatmulPreferenceCreate
hipblasLtMatmulPreferenceDestroy
hipblasLtMatmulPreferenceSetAttribute
hipblasLtMatmulAlgoGetHeuristic
hipblasLtMatmul
```

At handle creation, the facade obtains one provider lease, verifies required operations and
capabilities, creates one provider context, and commits a complete provider binding. Later public
operations reuse that binding and never rediscover or reselect a provider.

The facade translates public descriptors, layouts, scalars, algorithms, workspace, and stream into
the provider boundary. It maps provider failures to public hipBLASLt statuses.

## Shadow deployment API

**Components:** shadow package, facade, broker, provider module.

The shadow deployment installs a distinct facade identity, a provider manifest, and provider
modules beside the unchanged legacy hipBLASLt package. A default consumer resolves legacy
hipBLASLt; an explicit shadow consumer resolves the facade path.

The installed package preserves unique paths, package identities, exports, dependency rules, and
rollback to the legacy package.

## Implementation backend API

**Components:** provider adapter and current hipBLASLt implementation.

The adapter reaches the implementation through a private backend query rather than public
hipBLASLt operation names. The backend query returns only the operations needed by this GEMM path.

The backend query name, record layout, and versioning rules remain TBD until the implementation
boundary is finalized. The adapter must not depend on a public facade symbol for backend work.

## Coexisting public ABI lines

**Components:** compatibility facade and ABI adapters.

Each supported public ABI line has its own immutable library identity and exact symbol-version
definitions. An unchanged client binds to the line it was linked against.

Each public ABI line adapts directly to the current provider boundary. No ABI line forwards through
another public ABI line of the same library family.

## Qualification

Qualification uses the representative GEMM, the retained legacy baseline, host-side boundary
tests, binary inspection, package coexistence, failure injection, coverage, and available devices.

The completed path must show all of the following:

- unchanged clients retain their selected public ABI behavior;
- provider selection, provider translation, and facade dispatch preserve required GEMM behavior;
- shadow and legacy installations coexist and roll back cleanly;
- private implementation choices do not create public-facade recursion; and
- the applicable subset of this specification has retained test and inspection evidence.
