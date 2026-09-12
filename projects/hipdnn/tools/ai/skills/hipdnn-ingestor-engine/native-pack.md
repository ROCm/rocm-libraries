# Native pack contract

[RUNBOOK.md](RUNBOOK.md) owns implementation/build/test ordering. Derive native
semantics from the current graph and kernel contracts; reuse existing ownership and
registration patterns, not another engine's applicability rules.

## Roles

| Role | Signature / interface | Obligation |
|---|---|---|
| Engine graph match | `std::optional<BoundTokens>(const MatchContext&)` | Supported topology/semantics and operand UID bindings |
| Pack graph criterion | `bool(const MatchContext&, const BoundTokens&)` | Genuine pack narrowing; otherwise omit |
| Candidate matcher | `bool(const MatchContext&, const BoundTokens&, const KernelDefinition&)` | Graph versus candidate's baked metadata |
| Native score | `double(const MatchContext&, const BoundTokens&, const KernelDefinition&)` | Rank surviving candidates for `heuristic: native` |
| Dispatch | `IKernelDispatchHandler<Handle>` | Workspace, owned preparation and exact launch |

Complete reachable placeholders. Zero workspace is valid only when none is needed.
A native scorer ranks on justified free axes; higher wins. `heuristic: none` means
no UHD or score declaration/body/constant/registration. Loading alone cannot prove
absence: inspect the existing score registry for disabled/enabled controls.

## Matching

`graph_match` runs before a candidate exists; `nullopt` empties this engine's
catalog. Check the graph contract's node topology and UID edges, not only node types.
Resolve operands by UID and validate presence, rank, dims/strides lengths, extents
and applicable virtual/pass-by-value restrictions before indexing. Keep independent
operand constraints; do not assume identical dimensions or copy a single-node gate
into a fusion.

Use [graph-contract.md](graph-contract.md)'s field dispositions and existing
canonical semantic helpers. `field_audit.py` is only a lexical accessor inventory,
not proof of consumed/rejected semantics. Candidate matching uses KMD-typed values
and defaults, correct vocabulary, exact baked extents or proven capacity bounds.
Tile-dependent divisibility belongs here so another candidate can serve the graph.

Baked layout constrains per-operand strides; extent-one axes may have arbitrary
strides because their index is zero. Defer output checks to `prepare()` only when
inference makes them unavailable at matching. Fields consumed by scoring, geometry
or workspace still need metadata even when no graph comparison uses them.

## Workspace, preparation and launch

`workspaceBytes` reflects candidate-specific scratch. `prepare()` owns copied
UIDs/scalars, workspace and required module/program lifetimes; nothing may retain
transient `MatchContext` or `BoundTokens` references. A runnable kernel is a view,
so keep its compiled program alive too.

Use `buildIngestorKernelCode` for source loading and path bounds. Embedded source
uses real compile options. KPACK loads library/toc-key/symbol relative to descriptor
origin within `treeRoot`. A layout-neutral stand-in is valid only on a path proven
not to consume those compile options, never as an embedded-source workaround.

Grid/block and workspace formulas must match current source and every deciding
metadata field. A launch-surface declaration links Python source, C++ mirror,
inputs, guard and test; its checker proves structure, not semantic equivalence.

`launch()` resolves copied UIDs and supplies exact argument types/order. Replay
presence guards for a conditional ABI; retain every slot for a fixed ABI, including
unused pointers. Record each synthesized buffer's source, lifetime and enforced
preconditions. Prepared state must remain immutable across concurrent launches.

## Registration and inventory proof

`SymbolScope<Handle>` supplies typed symbols for UED graph match, graph/kernel UMDs,
optional UHD score and UDD dispatch. The non-owning dispatch registry requires
handler/module-cache lifetime beyond a per-handle `Container`. Extensions preserve
existing symbols. Both the registration declaration and `IngestorPacks.cpp` table
row are necessary for static-archive consumers, as are actual source/test targets.

Native proof executes real registrations and
`discoverDescriptorSets()` / `loadValidatedDescriptorSets<Handle>()`. A standalone
structural validator substitutes no-op native stubs and cannot establish this.
Fresh processes are required because registration/discovery is memoized.

The **packaged census covers packaged engines only** and is a direct native
obligation: CMake registers one independent test
`hip-kernel-provider-hkp-census-<arch>-<suite>` per declared suite and configured
packaging arch, running `hip_kernel_provider_tests --gtest_filter=<suite>.*` with
`HIPDNN_TEST_CENSUS_SUITE`, an explicit `HIPDNN_TEST_EXPECTED_ARCH` and the arch's
own `HIPDNN_DESCRIPTOR_DIR` shard. There is no Python launcher and no XML census
guard. Expected names/counts, runtime `KPACK` source kind and SDK version come from
the finalized emitted inventory; the arch comes from packaging configuration, never
from loaded descriptors or the host GPU — a bundle cannot be its own expectation.

The guard is active only for a nonempty census-suite variable, and it fails closed:
an empty arch, a missing/empty/nonexistent explicit root, an absent or empty named
suite, any case that skips or fails, list-only or zero-iteration invocations, and
partial runs that never complete one full iteration. Ordinary invocations without
the variable keep normal filtering and skip behavior, and the runtime's production
descriptor-root fallback is unchanged. An empty `HKP_CENSUS_TEST_SUITES` registers
nothing, and tests built OFF yields no census evidence at all — absence, not a pass.
Direct-load engines use their ordinary unit/inventory suites against the
arch-independent tree; a generated inventory suite invoked directly still requires
an explicit expected arch and descriptor root.

Placeholder/audit, native inventory and numerical dispatch are distinct evidence.
Neither a structural pass nor a loaded registry proves that a graph was served.
