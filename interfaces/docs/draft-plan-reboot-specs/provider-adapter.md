# Provider adapter specification

Status: normative specification for provider adapter source code, not a description of the tree.
Implementation status, source citations, test inventories, and prototype evidence are in
[ledger/provider-adapter.md](ledger/provider-adapter.md); nothing there changes an obligation here.
Unsettled points are marked TBD-n and listed in [Open decisions](#open-decisions-tbd).

## Scope

The source code inside a provider that translates the provider protocol to one concrete
implementation. This document owns: how the implementation is reached; translation duties in both
directions; mapping the implementation's status space onto the protocol taxonomy; adapter-local
state and its teardown; what must be validated before translating; host-service use from inside an
adapter.

## Non-goals

Each item is specified elsewhere and no rule about it is stated here. Protocol identity and
versioning, the query entry point, dispatch table shape and layout, the status taxonomy, argument
ownership, object lifetime, and concurrency rules for protocol calls:
[provider-protocol.md](provider-protocol.md). Shared-object identity, exported symbol surface,
visibility, load and unload, linkage hazards: [provider-module.md](provider-module.md). The binding
object, its token representation, validity, invalidation: [provider-binding.md](provider-binding.md).
Selection and negotiation: [broker.md](broker.md). Declaration and discovery:
[manifest.md](manifest.md). Co-selection: [cohort.md](cohort.md). Public API and ABI:
[facade.md](facade.md). Numerical tolerance policy: this document requires differential tests but
does not set their tolerances.

## Terminology

Parent terms are used as defined in [../draft-plan-reboot.md](../draft-plan-reboot.md). Two are
local: a **backend** is the canonical implementation library the adapter translates to, reached at
run time and never linked; a **dispatch slot** is one function-pointer member of a dispatch table
plus the adapter function installed in it.

## 1. Adapter responsibilities

1.1 An adapter MUST be the only place in a provider where implementation types, implementation
headers, and implementation status values appear.

1.2 An adapter MUST NOT link the implementation it translates to; the dependency MUST be resolved at
run time (section 2).

1.3 An adapter SHOULD keep every internal type, helper, and dispatch function in an anonymous
namespace, so only the protocol entry point has external linkage independent of the version script.

1.4 An adapter MUST implement the query and table contract of
[provider-protocol.md](provider-protocol.md). This file adds only the duties that arise because a
real implementation sits behind that contract.

1.5 The adapter MUST advertise only capability bits whose guarantees it implements for every context
it will create, MUST advertise zero when it has none, and MUST NOT advertise a bit a later runtime
condition can withdraw. Bit values are protocol-owned; honesty about them is adapter-owned. See
TBD-15.

1.6 Identity fields the adapter stamps into the query response MUST be honest and MUST agree with
what the manifest declares for this module and domain (TBD-3); the query MUST NOT mutate the
response on any failure path.

1.7 Every slot in the mandatory prefix MUST be non-null; a null one surviving negotiation is a
configuration error, never a fall-back request. A slot the adapter does not implement MUST be left
unfilled where the protocol permits, or MUST report the domain's not-implemented status. An adapter
MUST NOT install a success-returning stub, nor publish a table larger than it fully initializes.
## 2. Reaching the implementation

Threat: a provider that resolves its backend through the global symbol namespace binds to the facade
exporting the same public names and calls back into itself, or silently takes a different
implementation than the one it was qualified against.

2.1 An adapter MUST open its backend with `RTLD_NOW | RTLD_LOCAL`, under the process-wide
dynamic-loader guard so the open serializes against every other loader operation in the process.
`RTLD_LOCAL` is what keeps the backend out of the global namespace.

2.2 The backend MUST be named by a compile-time SONAME default overridable by exactly one documented
environment variable per backend. An adapter MUST NOT depend on an install RPATH to find it.

2.3 Where the backend offers a private query symbol, the adapter MUST resolve that symbol and route
every later lookup through it rather than `dlsym`-ing public names. The private query is what
guarantees references originating inside the backend bind to the backend's own implementation even
when a facade exporting the same names was loaded first.

2.4 The adapter MUST validate the backend API record before use - non-null, `struct_size` at least
the prefix it reads, matching major version, non-null mandatory entries - and MUST close the library
and fail otherwise. Optional newer entries MUST be gated on both a minor-version floor and a
non-null pointer.

2.5 Acquisition MUST be lazy: inside the query, at most once per process, under a once-guard, never
at module load time. A load-time constructor has no channel to report a status, and a module that
fails to load is retried on a later selection.

2.6 Acquisition failure MUST be reported as allocation failure for `std::bad_alloc` and as provider
failure otherwise, and SHOULD be traced first (section 7). It MUST NOT be reported as not-supported;
not-supported means "not this domain".

2.7 An adapter MUST NOT cache a failed acquisition as a permanent negative result that blocks a
later successful retry, unless it documents that restriction at its query.

2.8 A resolved-symbol cache MUST be mutex-guarded and MUST be keyed so two different requests cannot
collide on one key. Negative results MAY be cached subject to 2.7.

2.9 An adapter that synthesizes a backend symbol name from request fields MUST derive it only from
already-validated fields, and MUST treat an unresolved name as the domain's not-implemented status,
never as an internal error.

2.10 A locally declared function-pointer type used to call a resolved symbol MUST come from the
backend's own headers or be pinned by a translation test. A signature mismatch here is silent stack
corruption with no diagnostic.
## 3. Translation duties

### 3.1 Validate before translating

3.1.1 A dispatch function MUST validate in this order, returning on the first failure: opaque
context null; request pointer null; ABI header of the request and of every nested record it will
read; validity of every discriminator it will switch on; shape consistency between discriminators
and counts; rejection of unsupported features; range checks required by a narrower backend
signature; backend symbol resolution; then implementation state programming and the backend call.
Section 4 gives the status each failure maps to.

3.1.2 A record's ABI header MUST be validated before any semantic field of that record is read. The
header is what proves the field exists in the caller's build.

3.1.3 A dispatch function MUST NOT read beyond the caller's advertised `struct_size`. Each group
appended across ABI minors MUST be gated on the offset of its first member, and a smaller, older
record MUST leave the adapter's prior settings in place rather than resetting them to defaults.

3.1.4 An adapter MUST reject an unknown value of a discriminating enum, and MUST NOT fall through to
a default branch that guesses.

3.1.5 An adapter MAY early-return success for a degenerate size before validating data pointers, but
MUST document at that slot that it does so; otherwise pointer validation MUST precede the early
return.

### 3.2 Translating requests

3.2.1 The adapter MUST NOT retain a pointer into a request record, or into anything it points at,
after the dispatch function returns. Boundary argument ownership is protocol-owned; this is the
adapter's side of it.

3.2.2 Request data that must outlive the call MUST be copied by value into adapter-owned storage,
with the fields that must not outlive the call - data pointers, scalars, tokens, workspace - nulled
in the copy.

3.2.3 A dispatch function MUST program the implementation's mutable execution state from the request
before the operation and MUST NOT rely on state left by a previous call.

3.2.4 An adapter that caches implementation state to skip redundant programming MUST verify by
read-back that each setter took effect, at least for state whose divergence would silently mis-target
work, and MUST fail the call when the observed value differs.

3.2.5 An adapter MUST NOT re-read caller-visible state from anywhere but the request record mid-call.

### 3.3 Translating results

3.3.1 Output slots are caller-allocated. Where an operation writes an array of extensible result
records, the adapter MUST validate every slot's ABI header before writing any of them, then
overwrite each slot wholesale and re-stamp its header.

3.3.2 A count-out parameter MUST be written on every path the operation defines as producing a
count, including the zero case, before returning success.

3.3.3 An adapter MUST write zero to every reserved field of a record it produces. Whether it MUST
reject a nonzero reserved field in a record it consumes is TBD-7; until settled it SHOULD reject
nonzero reserved input.

3.3.4 Where the protocol carries an execution-effect field, the adapter MUST report conservatively:
"not started" only when it can prove no work was enqueued and no caller or workspace byte was
modified; "submitted" only when host submission completed successfully; "may have effects"
otherwise, including whenever it cannot prove "not started".

3.3.5 An adapter consuming an effect record MUST validate it: success implies a returned-status
origin and "submitted", failure implies not-"submitted"; a violation is a provider failure, not a
semantic result. "Not started" alone MUST NOT be read as permission to fall through (TBD-11).

3.3.6 Descriptors built for one submission MAY be torn down immediately after submission rather than
after completion, only where the implementation documents that submission copies what it needs.
## 4. Mapping implementation errors onto the protocol taxonomy

The status taxonomy and the domain vocabularies are defined once, in
[provider-protocol.md](provider-protocol.md). This section specifies only the mapping duty.

4.1 An adapter MUST NOT conflate transport status with semantic result. Where a callback returns a
transport status and also carries a semantic result record, the semantic outcome MUST go in the
record and the transport status MUST be reserved for boundary problems. See TBD-10.

4.2 A translation between two status spaces MUST be an explicit total mapping, written once, with a
default arm. It MUST NOT be a numeric cast.

4.3 A mapping MAY be lossy. Where lossiness would destroy information a second consumer needs, the
adapter MUST preserve the untranslated status alongside the translated one rather than choose.

4.4 Where a result record carries an origin field, the adapter MUST distinguish a status the
implementation returned from one it manufactured after catching an exception, and MUST
pre-initialize the record to the boundary-failure outcome so an unwritten record is distinguishable
from a real one.

4.5 A mapping MAY differ deliberately between two surfaces of the same adapter when the consumers'
fallback contracts differ; the divergence MUST be stated at the mapping in source.

4.6 An adapter MUST NOT invent a status the domain does not define, and MUST NOT return success when
it did not perform the operation.

4.7 Each condition MUST map to the stated category. A domain with a different status vocabulary MUST
publish the equivalent table for that vocabulary.

| Condition | Required category |
| --- | --- |
| null opaque context | invalid-handle |
| null request pointer | invalid-pointer |
| record header too small, or wrong major | not-implemented |
| unknown or unsupported enum value | not-implemented |
| supported enum, inconsistent shape | invalid-size |
| value out of range for the narrower backend signature | invalid-size |
| backend symbol not resolvable | not-implemented |
| allocation failure inside the adapter | memory-error |
| caught exception with no better mapping | internal-error |

4.8 Exceptions MUST NOT cross the protocol boundary; the register of that rule is protocol-owned
([provider-protocol.md](provider-protocol.md) TBD-1). The adapter's duty is that every escape point maps through 4.7 instead of propagating. One
guard template plus a thin wrapper per slot keeps translation logic exception-based and the boundary
exception-free.

```cpp
template <typename Action>
rocblas_status guard_callback(Action&& action) noexcept {
    try {
        return action();
    } catch (const std::bad_alloc&) {
        return rocblas_status_memory_error;
    } catch (...) {
        return rocblas_status_internal_error;
    }
}
```

4.9 A void-returning slot MUST still swallow every exception.

4.10 A slot with an epilogue that must run after the backend call MUST place a catch around the
backend call itself, so the epilogue still runs when the backend throws.

4.11 An adapter MUST NOT call `std::abort()` or `std::terminate()` from a dispatch function for a
condition the protocol can express as a status. See TBD-13.
## 5. Adapter-local state

5.1 What may cross the boundary is fixed by [provider-protocol.md](provider-protocol.md); casting an
untyped boundary pointer to the implementation's own type happens inside the adapter, nowhere else.

5.2 An adapter MUST NOT redefine or renumber a public enum or status value it reuses in a protocol
record.

5.3 Internal use of the C++ standard library and of the full implementation API is unconstrained,
subject only to 4.8 and 6.1.

5.4 Context creation MUST be failure-atomic: build under an owning smart pointer, transfer ownership
to the caller only on the final success path, release every implementation object already created on
any intermediate failure.

5.5 Creation MUST set the out-parameter to null before doing anything that can fail, so a caller
ignoring the status cannot read an uninitialized pointer.

5.6 Destruction MUST adopt the pointer unconditionally and MUST release the adapter's own allocation
regardless of what the backend reports. A destroy that frees the adapter object only when the
backend succeeds leaks it otherwise. See TBD-8.

5.7 Destruction with a null pointer MUST be a no-op, or a defined invalid-argument status for a
status-returning destroy. It MUST NOT be undefined behavior.

5.8 Threat: a token minted for one context and redeemed after that context is destroyed reaches a
recycled address and runs against unrelated state. Every opaque token an adapter hands out MUST be
bound to a generation drawn from a process-wide monotonic atomic counter and held by the issuing
context, and a token whose generation does not match MUST be rejected with a defined status.
Generation and token counters MUST be atomic and MUST reject exhaustion explicitly rather than wrap.

5.9 A token MUST additionally be validated against the problem it was issued for whenever the
redeeming operation depends on that problem.

5.10 Where tokens name retained implementation objects that cannot be evicted while the context
lives, the adapter MUST bound admission rather than evict: pre-count the slots a call requires before
mutating its map, and return an allocation-failure status plus a trace when the bound would be
exceeded.

5.11 An adapter that receives a foreign or stale opaque object MUST reject it with a defined status
rather than dereference it. See TBD-9.

5.12 An adapter MUST NOT rebind an existing context to a different provider, table, or backend after
creation.
## 6. Ownership at the adapter's edge

Ownership of protocol arguments is defined in [provider-protocol.md](provider-protocol.md); the
duties below are the adapter's.

6.1 An adapter MUST NOT allocate buffers on behalf of the caller. Device buffers, workspace, and
result locations are caller-supplied and used in place; a workspace pointer in a request is caller
input storage, not adapter scratch.

6.2 Where the protocol defines a provider-produced object returned through an out-parameter, the
adapter MUST also provide the matching destroy slot, and MUST destroy any partially constructed
object itself when the producing call fails.

6.3 Where the adapter returns a pointer into implementation-static data, no ownership transfers and
the caller MUST NOT free it. An adapter MUST document, per slot, whether 6.2 or 6.3 applies.

6.4 A pointer supplied only for the duration of a callback MUST NOT be retained past that callback.

6.5 A broker-issued service reference MUST be treated as valid only for the lifetime of the
receiving context, MUST be passed complete as the first argument to that reference's callbacks, and
MUST NOT be used to infer provider state from the cohort key.

6.6 An adapter MUST NOT free, close, or unload anything the broker owns: not the module, not the
binding, not the host-services record.
## 7. Host services and tracing

7.1 The host-services record is optional input. An adapter MUST re-validate its header before every
use, MUST null-check the individual function pointer, and MUST NOT assume the record outlives the
call that supplied it unless it was captured from retained context options. An adapter that
genuinely requires host services MUST document that requirement at its query (TBD-2).

7.2 A host callback is foreign code that may throw. Every call into one MUST be wrapped so no
exception escapes; an escaping trace exception would destroy a call that otherwise succeeded.

7.3 A trace MUST NOT change the status the adapter returns and MUST NOT be required for correctness.
An adapter SHOULD trace at minimum backend acquisition failure and any admission-limit rejection,
because those are the failures whose cause is invisible in the returned status.
## 8. Concurrency

Concurrency rules for protocol calls are defined in
[provider-protocol.md](provider-protocol.md); whether a context must be internally thread-safe is
TBD-12. Until that resolves, this document requires the safe side.

8.1 An adapter MUST be safe against concurrent dispatch on one context, either by serializing
internally or by holding no mutable per-context state that a dispatch function writes. An adapter
relying on caller serialization instead MUST state that reliance at its context type.

8.2 Process-wide adapter state - symbol caches, backend singletons, token maps, retained
implementation maps - MUST be mutex-guarded or immutable after once-initialization.

8.3 An adapter MUST NOT hold a lock across a call into foreign code that can re-enter it. Where
re-entry is structurally possible, the edge MUST be broken by a thread-local recursion guard that
rejects the reentrant call with a defined status rather than deadlocking or recursing.
## 9. Required tests

An adapter is not complete until it carries all four classes. The ctest inventory per class is in
[ledger/provider-adapter.md](ledger/provider-adapter.md).

9.1 Translation tests: call each implemented slot against a controlled backend and assert both that
the request became the expected backend call and that the backend status became the expected domain
status. Where the adapter synthesizes symbol names, cover each batch form and index width it claims.

9.2 Boundary tests: malformed query input (null pointers, short header, mismatched major, newer
minor, wrong domain, out-of-range table size), the domain's mandatory-prefix rule, acquisition
failure, an incomplete backend, and symbol isolation where a private backend query is used. Record
layouts the adapter depends on MUST be pinned by a test.

9.3 Differential tests: the same operations through the adapter and through the canonical
implementation, compared, including the degenerate cases the adapter special-cases - zero extents,
zero increments, single-element batches, every early return admitted by 3.1.5. Where registration is
conditional on an optional dependency, a green run does not prove coverage; see TBD-14.

9.4 Concurrency and no-throw checks: an adapter claiming internal thread safety MUST be exercised
under a race detector. Compile-time assertions that every dispatch slot is non-throwing SHOULD be
present and become mandatory if TBD-6 resolves to MUST.

9.5 Two obligations are review-gated with no mechanical check: sections 5 and 6 together (TBD-8),
and the validation order of 3.1.1. They are named rather than omitted so the check list does not
overstate what the suite proves.
## Open decisions (TBD)

Numbering is stable; other specs cite these identifiers. What each would take to close is in the
ledger.

| # | Open question | What it blocks |
| --- | --- | --- |
| TBD-1 | Is a non-default query symbol supported, given the module's export set? | Whether an adapter may name one |
| TBD-2 | Must the host-services record be non-null, or may an adapter require it? | 1.4, 7.1 |
| TBD-3 | Does build identity ever become a selection or compatibility constraint? | Whether 1.6 stays "be honest" |
| TBD-4 | Is the private backend API brought under the common ABI header or permanently separate? | Whether 2.4 is one rule or two |
| TBD-5 | Which table members are the frozen mandatory prefix per domain, and is the marker mechanical? | Whether 1.7 is compile-checkable |
| TBD-6 | Is a non-throwing dispatch slot a requirement or an accepted asymmetry? | 4.8, 9.4 |
| TBD-7 | Is "reserved fields must be zero" protocol-wide or specific to one record family? | 3.3.3 |
| TBD-8 | Must destroy release adapter-owned state regardless of backend status? | 5.6, 9.5 |
| TBD-9 | Is handle-identity validation beyond a null check required at the public edge? | Reach of 5.11 |
| TBD-10 | Is the transport-versus-semantic status split protocol-wide or one seam's convention? | 4.1 |
| TBD-11 | May a consumer fall through after a failure reported as "may have effects"? | 3.3.5 |
| TBD-12 | Must a provider context be internally thread-safe, or may caller serialization be required? | Whether 8.1 relaxes |
| TBD-13 | Is abort adopted behavior for an unrecoverable teardown failure anywhere in the stack? | Reach of 4.11 |
| TBD-14 | Must a configuration register a minimum differential test set and fail otherwise? | 9.3 |
| TBD-15 | Are capability-bit namespaces disjoint across domains? | Whether 1.5 can require a global bit meaning |
| TBD-16 | Are the in-tree recording providers permanently test-only fixtures? | Whether 7 and 9 apply to them |
| TBD-17 | Which transport status values are reachable from a provider query? | Whether section 4 needs query-side rules |

## Cross-links

- Parent concepts document: [../draft-plan-reboot.md](../draft-plan-reboot.md);
  [architecture-component-model.md](architecture-component-model.md) - lanes and call directions;
  [provider.md](provider.md) - the facility this adapter belongs to
- [provider-protocol.md](provider-protocol.md) - entry point, tables, status taxonomy, ownership,
  lifetime, concurrency; [provider-module.md](provider-module.md) - the artifact containing this
  adapter; [provider-binding.md](provider-binding.md) - what the broker retains after the query
- [broker.md](broker.md), [manifest.md](manifest.md), [cohort.md](cohort.md), [facade.md](facade.md)
- [ledger/provider-adapter.md](ledger/provider-adapter.md) - relocated evidence, citations, ctest
  inventory
