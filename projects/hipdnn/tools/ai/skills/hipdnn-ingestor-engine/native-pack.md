# Implementing the native pack — the five hooks

The generator emits `packs/<Name>Native.cpp` with every body `// TODO - FILL THIS OUT`.
Filling them is **the** integration work; descriptors are scaffolding around it. This file
is how.

Reference implementation to read before writing anything:
`src/engines/kernel_ingestor_engine/packs/AttentionDenseNative.cpp` (~700 lines, a rocKE
SDPA pack). `PointwiseNative.cpp` in the working tree shows the same shapes with a
non-trivial `workspaceBytes` and multiple `GraphCriterionFn`s.

---

## The five hooks

| Hook | Signature | Job |
|---|---|---|
| `GraphMatchFn` | `std::optional<BoundTokens>(const MatchContext&)` | Engine-wide gate. Runs once per graph. Returns the operand UIDs, or `nullopt`. |
| `GraphCriterionFn` | `bool(const MatchContext&, const BoundTokens&)` | Pack-scoped refinement (which operation). Omit for a single-kernel-family engine. |
| `KernelMatcherFn` | `bool(const MatchContext&, const BoundTokens&, const KernelDefinition&)` | Does this candidate's *baked* metadata fit the graph? Runs per candidate. |
| `ScoreFn` | `double(const MatchContext&, const BoundTokens&, const KernelDefinition&)` | Ranks survivors. Higher wins. |
| `IKernelDispatchHandler<THandle>` | `workspaceBytes` / `prepare` / `launch` | Turn a chosen candidate into a launched kernel. |

**`GraphMatchFn` returning `nullopt` empties the whole engine's catalog** — not just one
pack. Every remaining pack of that engine is skipped for that graph, including ones this
match never inspected. It is the widest blast radius in the system, so an overly narrow
`graph_match` is far more expensive than an overly narrow `kernel_match`.

**Division of labour, and why:** `graph_match` runs *before any candidate exists*, so it
can only judge the graph. Anything comparing the graph against a specific kernel's baked
constants belongs in `kernel_match`. Putting geometry equality in `graph_match` is a
category error that silently disables the engine.

---

## Hook 1 — `graph_match`: applicability

Structure, in order:

1. **Node shape.** `context.graph.nodeCount() != 1` → `nullopt`. Then confirm the node's
   attribute type is the one you serve.
2. **Fetch operands** by UID from `context.graph.getTensorMap()`.
3. **Total predicates on every operand.** This runs on an *unvalidated* graph, so a caller
   can present anything: check rank, positive extents, not `virtual_()`, not
   pass-by-value. A helper that indexes `dims->Get(3)` before checking rank is a crash.
4. **Layout check** — see below. This is usually the most important line in the file.
5. **Cross-tensor consistency.** K/V agree with Q on head size, batch, dtype; GQA
   divisibility (`numQueryHeads % numKvHeads == 0`) — the kernel derives its group size by
   integer division, so a non-divisible pair silently drops the remainder heads.
6. **Reject every unsupported optional attribute, explicitly.** *An unchecked mode is
   accepted and then silently not performed* — the graph asked for dropout, you ignored
   it, the numbers are wrong and nothing errored.

   **This is an exhaustive audit of the graph schema, not a check of the modes you
   happen to remember.** The attributes table for an op carries roughly two dozen
   optional fields, several of which are the *same* feature under different spellings —
   dropout alone appears as `dropout_probability`, `dropout_mask_tensor_uid`,
   `dropout_scale_tensor_uid`, `seed_tensor_uid` and `offset_tensor_uid`, and a check
   covering only the first admits graphs carrying the rest. Aux outputs and quantization
   fan out the same way.

   So work from the schema, not from memory: open the op's `.fbs` table (e.g.
   `SdpaAttributes` in `projects/hipdnn/**/*.fbs`), list **every** optional field, and
   account for each one — implemented, or explicitly rejected. Two of them are easy to
   miss because they are hipDNN-graph concepts with no rocKE counterpart, so
   `rocke-mining.md`'s Python-scoped checklist will never surface them:

   - `scale_tensor_uid` — a *device-resident* scale. A kernel taking `scale` as a launch
     scalar cannot accept it, and nothing about the Python spec says so.
   - `mma_core_mode` and the quantization scale UIDs — likewise absent from the spec.

   The mining checklist tells you what the *kernel* cannot do. This audit tells you what
   the *graph* can ask for. Both are required; neither substitutes for the other.
7. **Bind the tokens** and return them.

### The layout check

Most rocKE kernels bake a memory layout into their address arithmetic and accept **no
stride parameters**. A graph in another layout is indexed as if it were in this one:
in-bounds reads of the wrong elements — **silently wrong values, no fault**.

Derive the expected strides from the kernel's own arithmetic (see `rocke-mining.md`), then
assert them. The reference, for a kernel whose dims are `[B,H,S,D]` but whose memory is
token-major BSHD:

```cpp
bool hasBshdStrides(const data_objects::TensorAttributes* tensor)
{
    const auto* dims = tensor->dims();
    const auto* strides = tensor->strides();
    const int64_t heads    = dims->Get(HEAD_COUNT_AXIS);
    const int64_t sequence = dims->Get(SEQUENCE_AXIS);
    const int64_t headSize = dims->Get(HEAD_SIZE_AXIS);
    return strides->Get(BATCH_AXIS) == sequence * heads * headSize
        && strides->Get(HEAD_COUNT_AXIS) == headSize
        && strides->Get(SEQUENCE_AXIS) == heads * headSize
        && strides->Get(HEAD_SIZE_AXIS) == 1;
}
```

Only call it after rank and extents are validated.

**Do not layout-check the OUTPUT tensor here.** Its shape is inferred by the frontend and
is not reliably populated during matching; requiring it at match time makes the provider
decline graphs it can serve. Check the output in `prepare()` instead — the reference does
exactly this, and says so.

---

## Hook 2 — `kernel_match`: does this candidate fit?

Compare the graph-derived problem against the candidate's baked metadata, field by field,
via `kernel.getIntMetadata(...)` / `getStringMetadata(...)`. For a shape-specialized rocKE
kernel this is usually strict equality on every KMD field: head size, head counts,
sequence lengths, causal flag, dtype spelling.

Keep the descriptor's dtype spelling and the graph's dtype enum in **one** vocabulary —
convert the graph enum to the descriptor's spelling in a helper, and return `nullopt`/false
for a dtype the kernel cannot be built for.

---

## Hook 3 — `score`: ranking

Every candidate reaching `score` already fits, so score only the axes that remain free.
Higher wins.

A **legitimate placeholder**: return the one knob that trades off (e.g. a larger KV tile
amortises the per-tile loop, so `return kernel.getIntMetadata(BLOCK_N_FIELD);`). Say in
the report that it is heuristic and what would replace it (measurement, a UHD model).
Returning a constant is *not* a placeholder — it makes ranking arbitrary and hides
mis-specialization.

---

## Hook 4 — dispatch

Three members, with a strict ownership rule.

**`workspaceBytes`** — bytes of scratch this kernel needs. Return `0` when it needs none
(LDS + registers only); that is a real answer, not a stub. Where it depends on the
candidate, read the metadata.

**`prepare`** — resolve everything graph-derived into an owned object.

```cpp
auto code = buildIngestorKernelCode(_kernelCompiler, _kpackLoader, context, kernel, options);
```

**The `options` argument throws on a BSHD tensor — do not pass the real one.**
`KernelCompileOptions` classifies whatever tensor it is handed as NCHW or NHWC, and a
BSHD attention tensor is neither, so constructing it from your actual query tensor
throws at `prepare()` time. `PointwiseNative.cpp` passes its real tensor and is the
natural thing to copy — it works there only because a pointwise tensor *is* classifiable.

For a KPACK attention kernel, pass a layout-neutral stand-in (a minimal tensor whose
classification is irrelevant), and say in a comment why it is safe: the KPACK path
loads a prebuilt code object and never reads these options. An `EMBEDDED_SOURCE` kernel
in a BSHD pack does *not* get that escape — the options reach a real compile, so it
needs a genuine answer.

`buildIngestorKernelCode` (`IngestorKernelCode.hpp`) switches on `kernel.source.kind`:
`EMBEDDED_SOURCE` compiles the named source; `KPACK` resolves `source.library` relative to
`kernel.originDirectory`, **bounded by `kernel.treeRoot`**, and loads by `tocKey` +
`symbol`. Use the helper. Do not reimplement the boundary check — anchoring it on
`originDirectory` instead of `treeRoot` made every production-packaged kernel unloadable
while flat fixture trees stayed green.

Then set geometry, restating the kernel's own grid/block formula from its Python source:

```cpp
const int64_t queryBlocks = (problem->seqLenQ + DENSE_BLOCK_M - 1) / DENSE_BLOCK_M;
code.kernel->setBlockSize(DENSE_BLOCK_THREADS, 1, 1);
code.kernel->setGridSize(static_cast<unsigned>(queryBlocks),
                         static_cast<unsigned>(problem->numQueryHeads),
                         static_cast<unsigned>(problem->batch));
```

The constants are hand-carried from the builder's `<op>_grid` / `<op>_block`. Cite the
Python line in a comment — nothing checks this correspondence.

**The ownership rule:** the returned `PreparedDispatch` **MUST NOT reference the
`MatchContext` or `BoundTokens` it was built from.** Copy out UIDs and scalars. Storing a
`DeviceProperties` reference compiles and dangles on the next `launch()`.

`prepare()` is also where checks that were unreliable at match time go — the output
tensor's layout above all. Re-deriving the problem here and throwing on failure is
deliberate defensive re-validation, not redundancy.

**`launch`** — `dynamic_cast` to your concrete prepared type, resolve device pointers per
UID, call the kernel in the builder's exact ABI order. That order is a hand-maintained
contract with the Python `<op>_signature`, unchecked by the type system, and **conditional
arguments are appended in a fixed order** (see `rocke-mining.md` §ABI). A hardcoded arg
list is a memory-corruption bug for any spec with optional inputs enabled.

`launch` may run concurrently and must not mutate the prepared object.

---

## Registration

```cpp
constexpr std::string_view GRAPH_MATCHER_SYMBOL  = "<ns>.<engine>.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL = "<ns>.<engine>.kernel_match";
constexpr std::string_view SCORE_SYMBOL          = "<ns>.<engine>.score";
constexpr std::string_view DISPATCH_SYMBOL       = "<ns>.<engine>.dispatch";

void register<Name>Symbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL),  &graphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL),          &score);
    scope.add(std::string(DISPATCH_SYMBOL),       &dispatchHandler());
}
```

Which descriptor field names which hook:

| Descriptor | Field | Symbol |
|---|---|---|
| UED | `graph_match.native` | `GRAPH_MATCHER_SYMBOL` |
| UMD (`scope: kernel`) | `match_symbol` | `KERNEL_MATCHER_SYMBOL` |
| UMD (`scope: graph`) | `match_symbol` | a `GraphCriterionFn`, per pack |
| UHD (`kind: native`) | `payload` | `SCORE_SYMBOL` |
| UDD | `dispatch_symbol` | `DISPATCH_SYMBOL` |

**Nothing in the type system ties the JSON strings to the C++ constants.** A mismatch is
caught only by the loader's symbol pre-flight, which drops the set. The generator emits
both sides consistently — keep them that way, and use
`hipdnn_validate_descriptors --native-source <pack.cpp>` to cross-check.

The dispatch handler and its module cache must be **process-lifetime statics**: the
registry holds a non-owning pointer, while a provider's `Container` is created and
destroyed per handle.

---

## Traps that keep green CI meaningless

1. **`graph_match` → `nullopt` disables the entire engine** for that graph.
2. **The output tensor is not reliable at match time.** Check it in `prepare()`.
3. **`KernelCompileOptions` classifies its tensor as NCHW or NHWC and throws on anything
   else.** A BSHD tensor is neither. The reference passes a layout-neutral stand-in, which
   is safe *only* because the KPACK path never reads those options. An `EMBEDDED_SOURCE`
   kernel in a BSHD pack needs a real answer here.
4. **An unchecked optional attribute is silently not performed.** Reject every mode you do
   not implement.
5. **`DeviceProperties` in arch-gated tests must be constructed by value**, never queried
   from the host, or the test is vacuous on every arch but the one CI runs.
6. **The `ingestorPacks()` table row is load-bearing.** Declaration without the row: the
   static-archive linker drops the translation unit, so the pack vanishes from unit tests
   while the `.so` keeps working — no build error either way.
