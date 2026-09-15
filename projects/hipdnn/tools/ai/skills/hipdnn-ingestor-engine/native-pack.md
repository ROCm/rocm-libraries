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

A KPACK load is **digest-checked before the driver sees the bytes**. The descriptor's
`sha256` is the digest of the decompressed code object, 64 lowercase hex, and the
loader rehashes and compares it, raising `DIGEST_MISMATCH` on disagreement. The
archive reader cannot catch this itself: a TOC entry pointing at the wrong offset
decompresses cleanly and returns another entry's code object rather than an error.
Every stage — archive missing, archive unreadable, arch mismatch, `toc_key` absent,
decompress, digest mismatch, module load — raises its own distinct message, and a
missing symbol is raised later by `KpackProgram::getKernel`, the only site that can
see it.

**The module cache is keyed by device ordinal, and the device is made current across
the load.** The key is archive path, `toc_key`, the feature-stripped device arch, the
ordinal, and the expected digest; `symbol` is deliberately excluded so one module is
shared by kernels differing only by entry point. A device that cannot be made current
fails the load rather than yielding a foreign module, because an entry cached under
one ordinal and resident on another is a wrong answer every later dispatch reuses.

**A malformed descriptor is a hard failure, not one fewer candidate.**
`GenericPlanBuilder` rethrows `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` instead of
absorbing it and trying the next candidate, in plan build, in the filtered path and
in benchmarking alike. Falling past it would hide the fault and silently serve a
different kernel than the one authored. Read the consequence when triaging: a
descriptor bug presents as **no plan at all**, not as a thinner candidate list, so
"the engine declined" and "the descriptor is broken" look nothing alike.

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

The census is a direct native obligation, registered one
`hkp_register_census_tests(TARGET … PACK_NAME … SUITES …)` call per packed target in
`src/tests/CMakeLists.txt`, beside `hkp_verify_embedded_sources()` and after the test
target exists. Per declared suite and per arch in that pack target's own recorded
list, CMake registers an independent test
`hip-kernel-provider-hkp-census-<arch>-<suite>` running
`hip_kernel_provider_tests --gtest_filter=<suite>.*` with
`HIPDNN_TEST_CENSUS_SUITE`, an explicit `HIPDNN_TEST_EXPECTED_ARCH` and a
`HIPDNN_DESCRIPTOR_DIR` of that pack target's **own** `OUT_ROOT` shard for the arch —
not a shared stage tree. There is no Python launcher and no XML census guard.
Expected names/counts, runtime source kind and SDK version come from the finalized
emitted inventory; the arch comes from the wired arch list, never from loaded
descriptors or the host GPU — a bundle cannot be its own expectation.

**What may be censused is decided by shard count, not by dialect.** An entry hands
the binary exactly one directory, so only a suite confined to one pack target's shard
qualifies. `TestPointwisePacks` is censused at the `unit` target even though
`unit/pointwise/` is authored in the `embedded_source` dialect, because every case in
it reads that one shard; `TestConvFwdPack` reaches across the `unit` and
`unit_shared` shards and is censused nowhere. Declaring one suite at two pack targets
is fatal: the entry name carries arch and suite alone, so the second registration
would silently take the first one's shard.

The guard is active only for a nonempty census-suite variable, and it fails closed:
an empty arch, a missing/empty/nonexistent explicit root, an absent or empty named
suite, any case that skips or fails, list-only or zero-iteration invocations, and
partial runs that never complete one full iteration. Ordinary invocations without
the variable keep normal filtering and skip behavior, and the runtime's production
descriptor-root fallback is unchanged. An empty `SUITES` registers nothing, and tests
built OFF yields no census evidence at all — absence, not a pass. Every other missing
prerequisite is fatal at configure instead: an unwired `PACK_NAME`, a missing or
absent `TARGET`, an empty recorded arch list. An uncensused suite states its
inventory through its ordinary host run; invoked directly it still requires an
explicit expected arch and descriptor root.

Placeholder/audit, native inventory and numerical dispatch are distinct evidence.
Neither a structural pass nor a loaded registry proves that a graph was served.
