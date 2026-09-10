# Native pack contract

Use this reference from [RUNBOOK.md](RUNBOOK.md)'s implementation, splice and host
stages. It owns hook semantics, not another create/extend procedure. Read existing
packs for their structure and ownership patterns; derive applicability from the
current graph and kernel contracts rather than copying another engine's rules.

## Applicable roles

| Role | Signature / interface | Obligation |
|---|---|---|
| Engine graph match | `std::optional<BoundTokens>(const MatchContext&)` | Accept the supported topology/semantics and bind all operand UIDs |
| Pack graph criterion | `bool(const MatchContext&, const BoundTokens&)` | Genuine per-pack narrowing; omit when unnecessary |
| Candidate matcher | `bool(const MatchContext&, const BoundTokens&, const KernelDefinition&)` | Compare this candidate's baked metadata with the graph |
| Native score | `double(const MatchContext&, const BoundTokens&, const KernelDefinition&)` | Rank surviving candidates when `heuristic: native` is configured |
| Dispatch | `IKernelDispatchHandler<Handle>` | Workspace, owned preparation and exact kernel launch |

The generator emits placeholders where kernel knowledge is required. Fill every
reachable placeholder. A zero workspace size is valid when the kernel genuinely
needs none; an empty implementation disguised as a successful result is not.

A scorer is **conditional**, not a mandatory fifth hook for every configuration.
With `heuristic: none`, omit the UHD and all corresponding score declarations,
bodies, constants and registrations. With a native heuristic, rank on justified
free axes; higher wins. Explain the ranking's limits and check it against measured
selection rather than assuming a larger knob is faster. Successful loading alone
cannot prove absence of a leftover score registration: host checks must inspect
the existing score registry for disabled/enabled controls.

## Engine-wide graph match

`graph_match` runs before any candidate exists. `nullopt` empties the whole engine
catalog for that graph; candidate-specific restrictions do not belong here.
Validate the actual supported node topology and tensor-UID edges. A single-node
gate is correct for a single-node engine, **not** for a fused-subgraph engine.

Resolve operands by UID; validate presence, rank, dims/strides lengths, positive
extents and applicable virtual/pass-by-value restrictions before indexing. Check
cross-operand constraints from the kernel contract rather than assuming all
operands have identical dimensions. For attention, Q/K contraction dimensions,
V/output dimension and GQA divisibility are separate obligations.

Account for every field of every matched schema. A deprecated spelling, modern
bound field, optional UID and scalar can express the same feature differently.
Use existing canonical graph-semantic helpers where available; do not derive
causality from only the deprecated booleans or reject a nondefault enum before
reading what actual frontend graphs set. Missing required launch scalars without
a graph-defined default must be declined, not invented.

Field auditing is an inventory aid:

```bash
"$PY" "$GEN/tools/field_audit.py" \
  "$REPO/projects/hipdnn/flatbuffers_sdk/schemas/sdpa_attributes.fbs" \
  "$PROVIDER/src/engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseNative.cpp"
```

Use exactly one schema and all applicable native sources; repeat per schema for a
fusion. Exit 0 means parsed fields have identifier-boundary accessor references,
1 identifies missing references, and 2 rejects unusable input. A reference in a
source file is not evidence that the field is handled correctly on every path.
Review consumed/rejected/inert dispositions and actual behavior separately.

A baked layout is an applicability constraint. Derive strides from source arithmetic
per operand and check all address-relevant axes. Extent-one axes may have arbitrary
strides because their index is always zero. Do not over-reject byte-identical
single-head layouts. Where inferred output information is unavailable at matching,
check it in `prepare()` rather than declining a valid input early. Where the graph
contract already guarantees output information, validate what the kernel needs.

## Candidate matching

Use the candidate's KMD-typed metadata, including defaults resolved through the
referenced schema. Match graph-derived fields, vocabulary, exact baked extents or
capacity bounds as appropriate. A divisibility requirement involving a tunable
tile belongs here so another shipped candidate can still serve the graph.

Metadata required by dispatch/workspace/scoring also belongs in the schema even
if no graph comparison uses it. Conversely, a tuning-only field is not a reason
to manufacture a graph constraint. Compiler agreement checks metadata against
actual effective builder decisions; native matching still needs its own review
and behavioral/device evidence.

## Workspace, preparation and launch

`workspaceBytes` returns actual scratch needs, depending on candidate metadata
where necessary. `prepare()` resolves graph information into an owned prepared
object: copy UIDs/scalars, retain required workspace and module/program ownership,
and never reference the transient `MatchContext` or `BoundTokens` afterward.
A runnable kernel is a view into a compiled program; keep both alive for the plan.

Use `buildIngestorKernelCode` rather than reimplementing source loading or path
boundary checks. Embedded source compilation uses real compile options. The KPACK
path loads the prebuilt code object by library/toc-key/symbol relative to descriptor
origin and bounded by `treeRoot`. If a layout classifier cannot represent the
actual tensor, a layout-neutral stand-in is justified **only** for a path proven
not to consume those compile options; it is not an embedded-source workaround.

Restate grid/block and workspace formulas from the current kernel source, retaining
source references and every deciding KMD field. The existing `launch_surface`
profile declaration records Python source, C++ mirror, inputs, guard and test;
its structural checker cannot establish semantic equivalence.

In `launch()`, resolve device buffers by the copied UIDs and pass arguments in the
exact declared order and types. For a **conditional ABI**, replay the same presence
guards. For a **fixed ABI**, pass all slots even when a disabled feature does not
read one. See [rocke-mining.md](rocke-mining.md#fixed-and-conditional-abi-are-different-contracts).
Name the source and lifetime of every synthesized buffer and enforce the input
assumptions making its synthesis valid. Launch may run concurrently; do not mutate
the prepared object.

## Registration and proof

Use the existing `SymbolScope<Handle>` registration path. Each descriptor names
the appropriate typed symbol: UED graph match, graph/kernel UMD criterion/matcher,
optional UHD score, and UDD dispatch. The dispatch handler/module cache must have
the lifetime required by the non-owning registry, not a per-handle `Container`
lifetime. Preserve existing engine symbols during extension.

The engine's registration declaration and `IngestorPacks.cpp` table row both
matter: a static archive can drop an unreferenced translation unit even while the
plugin shared library appears to work. Apply both emitted fragments, and include
the native/test sources in their actual CMake targets.

Native proof is **compiled execution of the real typed registrations followed by
`discoverDescriptorSets()` / `loadValidatedDescriptorSets<Handle>()`**, plus an
expected census from finalized emitted artifacts. Source-text matching of symbol
constants cannot certify existence, kind, duplicate handling or callable bodies.
A standalone structural validator does not substitute for this host gate.

Run host census checks in fresh processes because registration/discovery is
memoized. Select `HIPDNN_TEST_EXPECTED_ARCH` explicitly from the configured packaging
architecture list, with its descriptor shard and a nonempty exact test filter;
expectations must not be inferred from loaded descriptors or the host GPU. Census
covers finalized names/counts, runtime source kind (`KPACK` for packaged kernels),
SDK version and architecture, including unchanged extension inventory.

Placeholder and field audits, compiled registration/census and numerical dispatch
are separate evidence. The first two do not prove that any graph is served. Return
to RUNBOOK for build/install, exact-engine device checks and final corpus acceptance.
