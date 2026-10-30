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

A multi-launch pack (descriptors carrying `sources`) overrides `launchNames()` to return
the launch names of its kernels' `sources` maps, the same set for every kernel; the state
manager drops a kernel whose set differs. The handler owns launch order and the workspace
between launches. See [Multi-launch handler](#multi-launch-handler).

Complete reachable placeholders. Zero workspace is valid only when none is needed. A
native scorer ranks on justified free axes; higher wins. `heuristic: none` means no UHD
and no score declaration, body, constant or registration. Loading alone cannot prove
absence: inspect the existing score registry for disabled/enabled controls.

## Matching

`graph_match` runs before a candidate exists; `nullopt` empties this engine's catalog.
Check the graph contract's node topology and UID edges, not only node types. Resolve
operands by UID and validate presence, rank, dims/strides lengths, extents and
applicable virtual/pass-by-value restrictions before indexing. Keep independent operand
constraints; do not assume identical dimensions or copy a single-node gate into a
fusion.

Use [graph-contract.md](graph-contract.md)'s field dispositions and existing canonical
semantic helpers. `field_audit.py` is a lexical accessor inventory, not proof of
consumed/rejected semantics. Candidate matching uses KMD-typed values and defaults,
correct vocabulary, exact baked extents or proven capacity bounds. Tile-dependent
divisibility belongs here so another candidate can serve the graph.

Baked layout constrains per-operand strides; extent-one axes may have arbitrary strides
because their index is zero. Defer output checks to `prepare()` only when inference
makes them unavailable at matching. Fields consumed by scoring, geometry or workspace
still need metadata even when no graph comparison uses them.

## Workspace, preparation and launch

`workspaceBytes` reflects candidate-specific scratch. `prepare()` owns copied
UIDs/scalars, workspace and required module/program lifetimes; nothing may retain
transient `MatchContext` or `BoundTokens` references. A runnable kernel is a view, so
keep its compiled program alive too.

Use `buildIngestorKernelCode` for source loading and path bounds. A pack that may compile
from source passes the compiler and real compile options. A pack whose kernels are all
KPACK calls the overload that takes neither, which refuses an `embedded_source` kernel;
never fabricate a tensor to construct compile options. KPACK loads
library/toc-key/symbol relative to descriptor origin within `treeRoot`.

A KPACK load is **digest-checked before the driver sees the bytes**: the descriptor's
`sha256` is the digest of the decompressed code object, 64 lowercase hex, and the loader
rehashes and compares it, raising `DIGEST_MISMATCH` on disagreement. The archive reader
cannot catch this — a TOC entry at the wrong offset decompresses cleanly and returns
another entry's code object. Every stage raises its own message (archive missing,
unreadable, arch mismatch, `toc_key` absent, decompress, digest mismatch, module load);
a missing symbol is raised later by `KpackProgram::getKernel`.

**The module cache is keyed by device ordinal, and the device is made current across the
load.** The key is archive path, `toc_key`, the feature-stripped device arch, the
ordinal and the expected digest; `symbol` is excluded so one module is shared by kernels
differing only by entry point. A device that cannot be made current fails the load
rather than yielding a foreign module, which every later dispatch would reuse.

A broken descriptor presents in one of two ways, depending on whether the fault is
caught at load time or at plan time:

- **Loader-time drop — silent, and indistinguishable from a decline.**
  `loadValidatedDescriptorSets` pre-flights each set and drops it, logging at ERROR and
  continuing, when any graph or kernel match symbol, the engine's `graph_match` symbol,
  a dispatch symbol or a `native` heuristic's score symbol is unregistered; when the
  engine name collides on engine id with an already-registered engine; or when the probe
  `makeStateManager` throws, logged as `does not validate: … dropping it`. A dropped set
  never reaches `GenericPlanBuilder`: the engine is absent from the registry and the
  graph gets a plain "no engine" outcome that nothing in the plan result distinguishes
  from a legitimate decline.
- **Plan-time rethrow — loud.** Once a set is loaded, `GenericPlanBuilder` rethrows
  `HIPDNN_PLUGIN_STATUS_INVALID_VALUE` instead of absorbing it and trying the next
  candidate — in plan build, in the filtered path and in benchmarking alike — because
  falling past it would silently serve a different kernel than the one authored. A
  malformed descriptor that did load presents as no plan at all, not as a thinner
  candidate list.

Triage consequence: **never classify a missing engine as a supported decline from the
plan outcome alone.** Check the loader diagnostics for `dropping it` at ERROR and
confirm against the loaded-descriptor inventory that the engine is present. An
unregistered native symbol is the common cause and produces no plan-time error.

Grid/block and workspace formulas must match current source and every deciding metadata
field. A launch-surface declaration links Python source, C++ mirror, inputs, guard and
test; its checker proves structure, not semantic equivalence.

`launch()` resolves copied UIDs and supplies exact argument types/order. Replay presence
guards for a conditional ABI; retain every slot for a fixed ABI, including unused
pointers. Record each synthesized buffer's source, lifetime and enforced preconditions.
Prepared state must remain immutable across concurrent launches.

### Multi-launch handler

A kernel whose UKD carries a `sources` map launches several kernels in order. The sketch
is a pack whose kernels carry `decode` and `merge` launches; the decode launch writes
partial results to a workspace slice and the merge launch reads them. `Binding`,
`intermediateBytes`, `decodeSignature`, `mergeSignature`, `bindOperands`,
`configureDecode` and `configureMerge` stand for pack-local types and helpers
(matching, signatures and geometry follow `ConvNative.cpp`).
`Gfx1151WmmaAttentionNative.cpp` is a shipped decode and merge handler.

```cpp
class PreparedDecodeMerge : public PreparedDispatch
{
public:
    PreparedDecodeMerge(IngestorKernelCode decode, IngestorKernelCode merge, Binding binding)
        : _decode(std::move(decode))
        , _merge(std::move(merge))
        , _binding(binding)
    {
    }

    const IngestorKernelCode& decode() const { return _decode; }
    const IngestorKernelCode& merge() const { return _merge; }
    const Binding& binding() const { return _binding; }

private:
    IngestorKernelCode _decode;
    IngestorKernelCode _merge;
    Binding _binding;
};

class DecodeMergeDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit DecodeMergeDispatchHandler(const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    // The same set for every kernel of the pack; a kernel with another set is dropped.
    std::set<std::string> launchNames() const override
    {
        return {"decode", "merge"};
    }

    // The slice between the launches. Sized from the graph, not from the kernel.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& /*kernel*/) const override
    {
        return intermediateBytes(context, bound);
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        // Each launch resolves through the ordinary kpack path, one definition at a time.
        auto decode = buildIngestorKernelCode(_kpackLoader,
                                              context,
                                              launchKernelDefinition(kernel, "decode"),
                                              decodeSignature());
        auto merge = buildIngestorKernelCode(
            _kpackLoader, context, launchKernelDefinition(kernel, "merge"), mergeSignature());

        const auto binding = bindOperands(context, bound);
        configureDecode(decode, context, binding); // setBlockSize / setGridSize
        configureMerge(merge, context, binding);

        return std::make_unique<PreparedDecodeMerge>(
            std::move(decode), std::move(merge), binding);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& dispatch = dynamic_cast<const PreparedDecodeMerge&>(prepared);
        const auto& binding = dispatch.binding();

        const auto in
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.in, deviceBuffers, numDeviceBuffers);
        const auto out
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.out, deviceBuffers, numDeviceBuffers);

        const hipStream_t stream = handle.getStream();

        // Order is the handler's: decode writes the workspace slice, then merge reads it,
        // on the one stream of the handle.
        dispatch.decode().kernelForStream(stream).launch(stream, in.ptr, workspace);
        dispatch.merge().kernelForStream(stream).launch(stream, workspace, out.ptr);
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};
```

What the sketch relies on beyond the contract above and `LaunchKernelDefinition.hpp`:

- The generator refuses a config whose launch names disagree, so alternatives with
  another set (for example a single-launch variant) belong to a separate engine.
- The hipDNN workspace is not zeroed. A launch that accumulates into the slice must
  initialise it, or an earlier launch must write all of it.
- `launchNames()` is provisional until the `launches[]` dispatch program exists
  ([RFC 0017 §15.1](../../../../docs/rfcs/0017_UniversalKernelDescriptor.md)).

## Registration and inventory proof

`SymbolScope<Handle>` supplies typed symbols for UED graph match, graph/kernel UMDs,
optional UHD score and UDD dispatch. The non-owning dispatch registry requires
handler/module-cache lifetime beyond a per-handle `Container`. Extensions preserve
existing symbols. Both the registration declaration and the `IngestorPacks.cpp` table
row are necessary for static-archive consumers, as are actual source/test targets.

Native proof executes real registrations and `discoverDescriptorSets()` /
`loadValidatedDescriptorSets<Handle>()`. A standalone structural validator substitutes
no-op native stubs and cannot establish this. Fresh processes are required because
registration/discovery is memoized.

**Packaged census: direct native CTest entries** in [RUNBOOK.md](RUNBOOK.md) is the
only statement of the census procedure — the per-suite, per-arch test family, entry
naming, invocation and environment, control semantics, the guard's fail-closed
conditions, what an empty `SUITES`, tests built OFF or a dormant `PACK_NAME` register,
where dormancy becomes a configure-time fatal, and what an unpinned registration
forfeits. Do not re-derive any of it here.

This page adds only the **registration site**: one
`hkp_register_census_tests(TARGET … PACK_NAME … SUITES … EXPECTED_CASES …)` call per
packed target in `src/tests/CMakeLists.txt`, beside `hkp_verify_embedded_sources()` and
after the test target exists. There is no Python launcher and no XML census guard.

Expected names/counts, runtime source kind and SDK version come from the finalized
emitted inventory; the arch comes from the wired arch list, never from loaded
descriptors or the host GPU — a bundle cannot be its own expectation.

**Shard count, not dialect, decides what may be censused**: an entry hands the binary
one directory, so only a suite confined to one pack target's shard qualifies. An
uncensused suite states its inventory through its ordinary host run.

Placeholder/audit, native inventory and numerical dispatch are distinct evidence.
Neither a structural pass nor a loaded registry proves that a graph was served.
