# Architecture component model specification

Status: specification, not a description of a finished system. Umbrella document for the
facade/provider architecture defined in [../draft-plan-reboot.md](../draft-plan-reboot.md). It is
an overview: the nine siblings here are normative for each component's detail, and each component
gets one line plus a link. Where a sibling disagrees, the sibling wins for its own component.
Implementation status, test names, and evidence are non-normative and live in
[ledger/architecture-component-model.md](ledger/architecture-component-model.md).

## Scope

The component inventory and each component's lane; the complete allowed-call-direction matrix,
with everything not listed prohibited; the context-establishment flow, the ordinary-operation
dispatch flow, and the rule separating them; legacy, shadow, and facade as three overlays over the
same three lanes; the architecture-level invariants.

## Non-goals

Not specified here: the public contract of any math library (a facade inherits that contract, it
does not define it), nor any sibling-owned topic - the manifest schema
([manifest.md](manifest.md)), the protocol wire format ([provider-protocol.md](provider-
protocol.md)), discovery and candidate ordering ([broker.md](broker.md)), what a provider,
adapter, or module implements ([provider.md](provider.md), [provider-adapter.md](provider-
adapter.md), [provider-module.md](provider-module.md)), binding state and lifetime ([provider-
binding.md](provider-binding.md)), cohort identity ([cohort.md](cohort.md)), or the delivery plan.
Nor does it claim the broker is a separately packaged component (TBD-2).

## Terminology

Terms are as defined in [../draft-plan-reboot.md](../draft-plan-reboot.md). Three are introduced
here because the component model needs a name for them.

- **Lane** - one of three boundary bands; every component sits in exactly one. The **application
  lane** is code the distribution does not own; the **public-contract lane** is the API, ABI, and
  library identity promised to it; the **private-implementation lane** is everything behind that
  promise, and nothing in it is a supported third-party interface.
- **Lease** - the broker-issued object naming the selected provider and keeping its module
  resident. The broker's half of a provider binding, not a synonym for it.
- **Underlying implementation** - the pre-existing math-library implementation a provider adapter
  reaches through its own private path. Not part of any public contract, even when in the legacy
  overlay that binary still owns one.

## 1. The three lanes

- The application lane MUST reach the distribution only through the public-contract lane.
- The public-contract lane MUST be owned by exactly one component per contract. Two components
  MUST NOT both claim one public library identity in a shipped distribution.
- The private-implementation lane MUST NOT be reachable from the application lane at source, link,
  or load time. No symbol, header, or artifact in it is a supported application dependency.

Documentation is not a boundary. A rule here that no check holds is a review-time rule.

## 2. Component inventory

Normative for lane assignment and allowed dependency direction. The sibling specification is
normative for what each component owns and forbids.

| Component | Lane | Role | Allowed outbound | Allowed inbound |
| --- | --- | --- | --- | --- |
| Application | application | Client code; owns only its own source and its choice of contract | Public-contract lane only | none |
| Compatibility facade | public contract on its exported surface, private implementation inside | Owns a public API, ABI, and library identity and delegates through the protocol; [facade.md](facade.md) | Broker at establishment only; its retained binding; the protocol through the retained table; the loader-lock DSO | Application |
| Broker | private implementation | Turns a domain plus device key into a validated lease; [broker.md](broker.md) | Manifest; module load and query symbol; the protocol query; the loader-lock DSO | Facade only |
| Manifest | private implementation (data) | Declares the candidate set to the broker; [manifest.md](manifest.md) | none, it is data | Broker only |
| Provider module | private implementation (binary artifact) | The loading and packaging unit carrying a protocol entry point; [provider-module.md](provider-module.md) | Underlying implementation by `dlopen`; the loader-lock DSO; protocol headers | Broker only, by load then query symbol lookup |
| Provider | private implementation (logical facility) | Supplies capabilities without owning any public contract; [provider.md](provider.md) | Underlying implementation; the host services record; delegated bindings handed in at context creation | Broker at query time; facade through the retained dispatch table |
| Provider protocol | private implementation (private ABI) | The versioned query envelope, tables, records, and status vocabulary; [provider-protocol.md](provider-protocol.md) | none, it is a contract not code | Facade, broker, provider adapter |
| Provider adapter | private implementation (source inside a provider) | Translates protocol calls to one implementation; [provider-adapter.md](provider-adapter.md) | Underlying implementation; the host services record | The provider's own query symbol and dispatch table only |
| Provider binding | private implementation (held by the facade) | The retained result of selection; [provider-binding.md](provider-binding.md) | The protocol through the retained table | Facade only |
| Cohort | private implementation (policy over data) | Asserts that a set of providers was qualified together; [cohort.md](cohort.md) | none | Broker during selection; facade for multi-domain contracts |
| Underlying implementation | private implementation | The pre-existing implementation reached as a backend; [provider.md](provider.md) | none within this architecture | Provider adapter only, at run time |
| Dynamic-loader lock DSO | private implementation (supporting) | Process-shared serialization of `dlopen`, `dlsym`, `dlclose` | none | Facade, broker, provider module |
| Host services record | private implementation (part of the protocol) | Trace and service callbacks supplied to a provider | none | Provider adapter, as callbacks |

Three clarifications are normative. The facade straddles two lanes deliberately: its exported
surface is the public-contract lane, everything inside it is private-implementation lane and MUST
be invisible to the application. The cohort is not an artifact: no cohort binary, object, or API
exists. The provider is not the module: one module MAY serve several domains and one provider MAY
span several modules, so a rule about "the provider binary" is a category error. Two obligations
fall to no sibling: the dynamic-loader lock DSO MUST export nothing but its lock interface and
MUST NOT be optional for a facade; the host services record MUST NOT be load-bearing for
correctness, and a provider MUST NOT assume it is non-null absent a stated rule (TBD-3).


## 3. Allowed call directions

Rows are callers, columns are callees; a cell holds the edge identifier when allowed and `-` when
prohibited. **Any direction not listed is prohibited**, and a prohibited direction is a
conformance failure even when nothing currently breaks.

| caller \ callee | App | Facade | Broker | Manifest | Module | Provider | Binding | Impl | Lock | Host svcs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| App | - | D1 | - | - | - | - | - | - | - | - |
| Facade | - | - | D2 | - | - | D8 | D7 | - | D12 | - |
| Broker | - | D6 | - | D3 | D4 | D5 | - | - | D13 | - |
| Manifest | - | - | - | - | - | - | - | - | - | - |
| Module | - | - | - | - | - | - | - | D9 | D14 | - |
| Provider | - | - | - | - | - | D11 | - | D9 | D14 | D10 |
| Binding | - | - | - | - | - | - | - | - | - | - |
| Impl | - | - | - | - | - | - | - | - | - | - |
| Lock | - | - | - | - | - | - | - | - | - | - |
| Host svcs | - | - | - | - | - | - | - | - | - | - |

| # | Caller -> callee | Mechanism, and when |
| --- | --- | --- |
| D1 | Application -> facade | Public API and ABI, export-allowlisted and version-noded; any time |
| D2 | Facade -> broker | In-process call into the registry; context establishment only |
| D3 | Broker -> manifest | File read plus strict parse, inside one atomic load; during discovery |
| D4 | Broker -> provider module | Trusted-path load then query symbol lookup, serialized through the loader lock; during selection |
| D5 | Broker -> protocol | The versioned query call into a loaded module; once per candidate evaluation |
| D6 | Broker -> facade | Return of the lease, plus rejection diagnostics through a non-load-bearing trace callback; return path of D2 only |
| D7 | Facade -> binding | Local retention and read of binding state; establishment and every operation |
| D8 | Facade -> protocol | Call through the retained table with the retained provider context; ordinary operation, context create and destroy |
| D9 | Provider adapter -> underlying implementation | `dlopen` plus the implementation's private backend entry point, serialized through the loader lock; lazily, at most once per process for acquisition, then per call |
| D10 | Provider adapter -> host services record | Callback supplied in the request or context-creation record, called defensively; during a query or dispatch call |
| D11 | Provider adapter -> delegated binding in another provider | Facade-constructed services record authenticated by a service token; only when the facade handed that delegated binding to this provider context at creation |
| D12, D13, D14 | Facade, broker, provider module -> loader-lock DSO | Link dependency and lock acquisition around `dlopen`, `dlsym`, `dlclose`; any loader operation |

Three readings are normative. The `Manifest`, `Binding`, `Impl`, `Lock`, and `Host svcs` rows are
entirely `-`: they are leaves, called and never calling. The `App` row has exactly one allowed
cell, the boundary the architecture exists to create. `Provider -> Provider` is D11 and only D11.

These prohibited edges are named because each has been reached for in a real design conversation:

- Application -> broker, provider, module, manifest, protocol, or underlying implementation: makes a private-lane item a de facto public contract, versioned forever.
- Facade -> underlying implementation: linking or loading the canonical library adds a layer instead of moving the implementation behind a boundary.
- Facade -> provider module: a facade that knows a module path or name has hard-coded a selection that belongs to the broker.
- Facade -> broker on the operation path: destroys context pinning and reintroduces per-call selection cost and risk.
- Broker -> application, and broker -> facade other than the D6 return and the trace callback: the broker has no public surface, and calling facade logic inverts contract ownership.
- Provider -> facade public API: re-enters the contract the provider implements, and under overlay C would recurse.
- Provider -> broker: a provider that selects, orders, or rejects is a second broker with no diagnostics.
- Provider -> another provider directly: bypasses cohort, lifetime, and token control; only D11 is legal.
- Anything -> a provider adapter by name: requires exporting a symbol the module MUST NOT export.
- Underlying implementation -> anything here: it is a leaf; a callback into a facade makes the private boundary circular.

Two consequences are requirements. The graph over D1-D14 MUST be acyclic once D6, D10, and D11 are
read as return paths and callbacks; a genuine cycle between facade and provider, or between two
providers, MUST NOT exist. Every edge crossing into a separately compiled artifact - D4, D5, D8,
D9, D11 - MUST be a C ABI with an explicit size and version header, or, for D9, the private
contract the implementation itself declares (TBD-26).

## 4. Context establishment

Context establishment is the only moment at which discovery, selection, and validation may run. A
conforming facade MUST perform it at most once per public context and MUST NOT perform any part of
it again. The flow runs in three stages. The facade first validates its public arguments and
resolves the device key. It then calls the broker with domain, device key, required prefix, and
required abi_minor, and the broker runs its selection sequence - manifest parse, candidate filter,
candidate ordering, and, per candidate, module load, query-symbol resolution, query, and response
validation - and hands back a lease. That sequence and every rule governing it are
[broker.md](broker.md)'s; the steps are named here only so the flow reads end to end. The facade
then checks required callbacks and capabilities, creates the provider context through the table,
retains the binding, and returns a public handle.

- The facade MUST validate its public arguments before it reaches the broker.
- What the broker validates before it hands back a lease, and its no-caching of failures, are [broker.md](broker.md)'s; establishment relies on both and states no rule of its own about them.
- Required-callback and capability checks MUST happen before a provider context is created; whether the facade or the broker performs them is open (TBD-10).
- When the contract requires several domains, every step from the manifest read to the capability check MUST complete for every domain under one cohort identity before the public context is returned. A partially bound context MUST NOT be returned.
- The facade MUST NOT return a public context whose binding is incomplete, and MUST release everything already acquired when any step fails.
- Whether a facade MAY defer establishment past construction to first use is open (TBD-24); selection is once-only either way, and pinning holds from the moment it happens.

## 5. Ordinary operation dispatch

The broker is absent from this path, and that absence is the point of the architecture: selection
cost and risk are paid once and never again for that context. The facade validates arguments,
reads the retained table pointer and provider context from the binding, calls the table slot,
translates the protocol status into its public status vocabulary, and returns.

- The facade MUST NOT re-derive the table pointer or provider context per call; re-derivation is a
  second selection and is prohibited.
- A failure at operation time MUST NOT trigger reselection or rebinding. The established context
  stays bound to the provider it was bound to and reports the failure.
- No exception may escape the provider into the facade, or the facade into the application.
- Argument ownership across the protocol boundary is [provider-protocol.md](provider-protocol.md)'s; nothing on this path relaxes it.
- A delegated binding is reached only through the facade-constructed services record and its
  token, never by a direct provider-to-provider call.
- Whether one public context may be dispatched concurrently from several threads is open (TBD-20).

## 6. Legacy, shadow, and facade as overlays

Legacy, shadow, and facade are not three architectures but three overlays over the same three
lanes; what moves between them is which component owns the public-contract lane.

| Lane | Overlay A: legacy | Overlay B: shadow | Overlay C: facade |
| --- | --- | --- | --- |
| Application | Links the canonical SONAME | Links or preloads the shadow SONAME deliberately, and MUST NOT get it by accident | Links the same canonical SONAME as in A, unchanged and not relinked |
| Public contract | Owned by the canonical implementation library | Owned by the canonical library; the shadow carries a parallel, distinct identity | Owned by the facade |
| Private implementation | The canonical library's own internals; no broker, manifest, or provider | The full private lane exists and is exercised, with the canonical library reached as a backend | Same as B; the legacy implementation is one provider backend among possibly several |
| What is proved | Nothing new | That the private lane is correct and complete while rollback costs nothing | That the identity transfer preserved the contract |

- A distribution MUST be in exactly one overlay for a given contract at a given time (INV-1).
- Overlay B MUST NOT interpose on overlay A: the shadow MUST be installable with no file and no SONAME conflict and MUST NOT be selected by default. An application reaching the shadow without asking for it is overlay C without the qualification.
- Overlay B MUST subject the shadow to every check the overlay C facade will face: export allowlist equality, a version node on every export, absence of a canonical DSO in `DT_NEEDED`, presence of the dynamic-loader lock, and no RPATH.
- Overlay B MUST be qualified from an installed tree, not only from a build tree.
- The transition into overlay C MUST be reversible: a distribution MUST be able to reinstall the legacy library and have existing compiled clients keep working without relinking.
- In overlay C the legacy implementation MUST be reachable only as a provider backend and MUST NOT retain the public identity. The private path to it MUST guarantee that references originating inside the legacy DSO bind to that DSO's own implementation even when a facade exporting the same public names loaded first.
- Upgrade and rollback MUST be tested as a sequence, not as two independent installs.
- Exit criteria for leaving overlay B MUST be written before the transfer is attempted (TBD-1).

## 7. Architecture-level invariants

These belong to no single component and MUST be checked at the system level.

- **INV-1 One owner per public identity.** Exactly one shipped component owns a given public library identity at a time.
- **INV-2 No private symbol in a public export set.** A facade's export set MUST equal its allowlist exactly; a provider module's export set MUST be its single query symbol.
- **INV-3 No canonical implementation on a facade's link or load line.**
- **INV-4 Selection happens once per context.** No component may perform discovery, ordering, or selection on the ordinary-operation path.
- **INV-5 Fail closed at every private-ABI boundary.** Any size, version, identity, prefix, or capability disagreement MUST reject the candidate or the call; "proceed and hope" is prohibited.
- **INV-6 Atomic completeness.** A multi-domain contract MUST NOT observe a partially acquired provider set.
- **INV-7 Residency follows the binding.** A provider module MUST stay resident at least as long as any binding holds a pointer into it; the lease is the mechanism.
- **INV-8 Loader serialization.** Every `dlopen`, `dlsym`, and `dlclose` in the private lane MUST be serialized through the loader lock (for modules, see TBD-29).
- **INV-9 No exception crosses a C ABI.** Not the public ABI, not the protocol; translation is mandatory at both boundaries.
- **INV-10 Nothing in the private lane is a supported dependency.** No manifest, module, protocol header, or provider symbol may be documented, packaged, or promised as third-party consumable.
- **INV-11 Concurrent facades in one process.** Two facades in one process MUST NOT corrupt each other's discovery or loading.
- **INV-12 A public object carries no private identity.** No protocol type, provider identity, cohort identity, manifest path, or module name may appear in a public header, a public struct layout, or a public status value.

## 8. Open decisions

Only decisions that block a contract stated above. Full statements, closing conditions, and the
cross-cutting decisions owned by siblings are in [ledger/architecture-component-
model.md](ledger/architecture-component-model.md).

| Id | Blocks | Question |
| --- | --- | --- |
| TBD-1 | Section 6 overlay C; INV-1 | How is the public identity transferred from the legacy library to the facade, and what are the exit criteria for leaving shadow qualification? |
| TBD-2 | Section 2; the D2 edge | Who owns the broker in production: one shared instance per process, or a private registry per facade? |
| TBD-3 | Section 2 host services; D10 | Which component injects and owns the host services record, and may a provider require it to be non-null? |
| TBD-4 | Section 3 | Is the allowed-call-direction matrix enforceable, or does it remain a review-time rule? |
| TBD-10 | Section 4 | Does required-callback and capability gating belong in the facade or in the broker, so a deficient candidate falls through? |
| TBD-11 | INV-6 | Which domains require a cohort? |
| TBD-15 | Every evidence claim | Is there a required minimum registered test set, so an unregistered test family is visible as a skip rather than as an absence? |
| TBD-20 | Section 5 | Must a provider context be internally thread-safe, or may an adapter require caller serialization? |
| TBD-24 | Section 4 | Is deferred first-use binding an approved binding shape or an inconsistency to reconcile? |
| TBD-26 | Section 3, D9 | Does the private backend contract come under the protocol's ABI header, or stay a separate private contract? |
| TBD-29 | INV-8 | Must a provider module route its own loader calls through the dynamic-loader lock, and is that checked? |

## 9. Cross-links

Siblings, each normative for its own component: [facade.md](facade.md), [broker.md](broker.md),
[provider.md](provider.md), [provider-protocol.md](provider-protocol.md), [provider-
binding.md](provider-binding.md), [provider-adapter.md](provider-adapter.md), [provider-
module.md](provider-module.md), [manifest.md](manifest.md), [cohort.md](cohort.md). Terminology:
[../draft-plan-reboot.md](../draft-plan-reboot.md). Non-normative evidence: [ledger/architecture-
component-model.md](ledger/architecture-component-model.md). Style guide:
[../STYLE.md](../STYLE.md).
