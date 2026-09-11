# Implementing the native pack — the five hooks

**You were sent here from RUNBOOK step 6.** You owe `packs/<Name>Native.cpp` with all
five hooks filled in — `generate.py --check-placeholders` exiting 0 — before you go to step 7. This
file tells you what each hook's body must do and the traps that make a green build mean
nothing.

**Variables.** `$REPO`, `$PROVIDER`, `$GEN`, `$BUILD` are set in `RUNBOOK.md`'s "Set
these once" block. Set them before running any command below — nothing enforces that you
arrived through RUNBOOK, and the first command here uses `$PROVIDER`.

**What you arrive with.** `graph_contract.md` told you what the graph can ask for and
where kernel and hipDNN disagree; `mining.md` told you what the kernel can actually
answer, including its rejection checklist. This step is where those two documents become
C++: your step-2 rejection checklist is the `graph_match` body, in severity order.

The generator emits `packs/<Name>Native.cpp` with every body `// TODO - FILL THIS OUT`.
Filling them is **the** integration work; descriptors are scaffolding around it. This file
is how.

Reference implementations to read before writing anything — **find them, do not assume a
filename.** What exists varies by branch:

```bash
ls $PROVIDER/src/engines/kernel_ingestor_engine/packs/*Native.cpp
```

`PointwiseNative.cpp` and `ConvNative.cpp` ship on every branch and are the ones to read
for *structure*: the match/prepare/launch split, the ownership rules, a non-trivial
`workspaceBytes`, multiple `GraphCriterionFn`s. A rocKE SDPA pack (`AttentionDenseNative.cpp`)
exists on some branches and not others — `git log --all --diff-filter=A -- '**/packs/*Native.cpp'`
finds every one ever added if your tree has none.

**Copy their SHAPES, not their RULES.** Every rocKE reference pack in this codebase's
history has shipped at least one *applicability* defect, and a copier inherits it
silently. Two were found by checking a pack's rejection list against real bundle data —
both present in the first authored SDPA pack, worth knowing as a worked illustration of
the failure shape:

- Its `causal` derivation read only the deprecated `causal_mask` /
  `causal_mask_bottom_right` booleans. Every shipped `quick/SdpaFwd` bundle leaves those
  `false` and expresses causality through `left_bound`/`right_bound`, so it computed
  `causal = 0` for a causal graph and matched a **non-causal kernel**. Wrong numbers, no
  fault, matcher green.
- Its `mma_core_mode != UNSET` rejection declined every shipped SdpaFwd bundle, because
  they all set `"float"` (`FLOAT = 1`, `data_types.fbs`). The engine then served nothing
  and the suite stayed green because another engine won.

**The general rules behind those two, which outlive any particular file:**

- **A graph concept with no matching spec field is a *derivation*, and derivations are
  where reference packs go wrong.** `graph_contract.md` §5 already named which of your
  fields are derivations and how the graph side spells them; `mining.md` has the kernel's
  own mapping. If an incumbent engine already serves this op, its plan builder —
  `src/engines/<incumbent-engine>/plans/` — has almost certainly solved the same
  derivation; find the canonical implementation there and call it rather than
  reimplementing it. *(For SDPA that derivation is `getMaskType` —
  `git grep -n 'getMaskType' -- '*/plans/*.hpp'`; the reference pack above
  shipped without calling it.)*
- **Never write an optional-enum rejection as `field() != DEFAULT` without checking what
  shipped bundles actually set.** A frontend default that is a non-zero enum value makes
  the naive form an over-rejection that fails *silently*: the engine registers, builds and
  passes CI by never being selected. There is no error to notice.

So: derive your rules from the kernel source and the op's `.fbs`, then use an existing pack
to check your structure — not the other way round. And validate every rejection you write
against real bundle data (`integration-tests/integration-test-bundles/`): a check that
declines every shipped case is as broken as one that admits a bad graph.

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

   So work from the schema, not from memory: open the op's `.fbs` table under
   `flatbuffers_sdk/schemas/<op>_attributes.fbs`, list **every** optional field, and
   account for each one — implemented, or explicitly rejected. Some are easy to miss
   because they are hipDNN-graph concepts with no rocKE counterpart, so
   `rocke-mining.md`'s Python-scoped checklist will never surface them. *(SDPA's table
   has three: a device-resident `scale_tensor_uid` a launch-scalar kernel cannot accept,
   an `mma_core_mode` field and quantization scale UIDs with no rocKE analogue, and an
   `implementation` execution-strategy hint (`AttentionImplementation`:
   AUTO/COMPOSITE/UNIFIED) that a pack silently ignores if it never reads it.)*

   **Do not rely on a curated list, including this one — it has already gone stale once**
   (`implementation` was missing from it, and the one shipped pack audited against this
   section had exactly that hole). After writing your rejection block, run the
   completeness check and treat anything it prints as a hole:

   ```bash
   # Absolute paths: this same check, transcribed into two documents, once used a
   # relative pack path that matched nothing when run from another directory and
   # reported every field accounted for. One script, invoked, not copied.
   $GEN/tools/field_audit.sh \
       $REPO/projects/hipdnn/flatbuffers_sdk/schemas/<op>_attributes.fbs \
       $PROVIDER/src/engines/kernel_ingestor_engine/packs/<Name>Native.cpp
   ```

   Pass ONE schema, not a glob (`*sdpa*attributes.fbs` matches both the forward and
   backward tables, and the script rejects a glob rather than silently auditing the
   wrong set), and pass **every** pack source of a multi-pack engine.

   **GATE:** every name it prints is either consumed or explicitly rejected — not merely
   absent because no shipped bundle varies it yet. A field is safe to skip only once you
   have written down, in a comment, why its only legal value is inert for this kernel.
   Run it against a branch's rocKE SDPA pack (see the `ls` above — it does not ship on
   every branch) and it prints exactly `UNCHECKED: implementation` out of SDPA's field
   set — one real, still-open hole, which is what a working completeness check looks like.

   **Do not invent a default for a scalar the graph did not supply**, even a
   mathematically obvious one like `attn_scale_value = 1/sqrt(head_size)`. If the schema
   marks a scalar optional and your kernel's ABI takes it as a required launch argument,
   require presence in `graph_match` and decline its absence. Guessing a value the graph
   never asked for silently overrides whatever the frontend's omission meant.

   The mining checklist tells you what the *kernel* cannot do. This audit tells you what
   the *graph* can ask for. Both are required; neither substitutes for the other.
7. **Bind the tokens** and return them.

### The layout check

**The general rule: a kernel that bakes a layout must assert it, or a graph in another
layout is indexed as if it were in this one** — in-bounds reads of the wrong elements,
**silently wrong values, no fault**. Most rocKE kernels bake a memory layout into their
address arithmetic and accept **no stride parameters**, so this check has nowhere else to
live but `graph_match`.

Derive the expected strides from the kernel's own arithmetic (see `rocke-mining.md`), then
assert every one the kernel depends on. Worked example — a kernel whose dims are
`[B,H,S,D]` but whose memory is token-major BSHD (the shipped attention instance):

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

**Exempt unit-extent axes, or you over-reject.** A stride multiplies an index; when an
axis has extent 1 that index is always 0, so no address depends on its stride and a
producer may declare anything there. A one-head tensor is byte-identically BSHD and BHSD
while the two spellings disagree on `strides[H]` — a strict comparison declines a graph
the kernel serves perfectly, and `graph_match` returning `nullopt` empties the **whole
engine catalog**. Guard each axis as `dims->Get(i) == 1 || strides->Get(i) == expected`.

Write the degenerate case as a test. It is invisible by inspection and it is the reason
to have a layout test at all: assert both that multi-head BHSD is declined *and* that
single-head is accepted under either spelling.

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

**The general trap: `options` is constructed by classifying a real tensor, and throws if
that tensor's layout does not fit the classifier's vocabulary — do not pass the real one
when your layout doesn't.** `KernelCompileOptions` classifies whatever tensor it is
handed as NCHW or NHWC; a layout the classifier has no category for (BSHD attention
memory is the instance today) throws at `prepare()` time if you construct it from your
actual query tensor. `PointwiseNative.cpp` passes its real tensor and is the natural
thing to copy — it works there only because a pointwise tensor *is* classifiable.

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

**The kernel is a VIEW into its program's module,** so a `PreparedDispatch` holding only
the `IRunnableKernel` dangles. Hold the `ICompiledProgram` alongside it for the plan's
lifetime — `buildIngestorKernelCode` hands you both for this reason.

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
3. **`KernelCompileOptions` throws if the tensor you construct it from doesn't fit its
   layout vocabulary** (BSHD attention memory is neither NCHW nor NHWC). Pass a
   layout-neutral stand-in when the real tensor doesn't classify — safe *only* when the
   kernel path never reads those options (KPACK). An `EMBEDDED_SOURCE` kernel needs a
   real answer here.
4. **An unchecked optional attribute is silently not performed.** Reject every mode you do
   not implement.
5. **`DeviceProperties` in arch-gated tests must be constructed by value**, never queried
   from the host, or the test is vacuous on every arch but the one CI runs.
6. **The `ingestorPacks()` table row is load-bearing.** Declaration without the row: the
   static-archive linker drops the translation unit, so the pack vanishes from unit tests
   while the `.so` keeps working — no build error either way.

---

## GATE

```bash
$GEN/.venv/bin/python $GEN/generate.py --config <your-config> \
    --output-dir $PROVIDER/src --check-placeholders
```

Must exit `0`. **Not** a `packs/*Native.cpp` glob: the generator also emits test stubs
that carry placeholders, and those splice into a different tree, so a glob over one
directory reports clean while they sit unfilled. That alone is necessary, not sufficient:
every rejection in your step-2
checklist must actually appear in `graph_match`, every schema field from the completeness
check must be consumed or explicitly rejected, and the `ingestorPacks()` row must exist
for both the declaration and the `s_packs` entry.

Then go to step 7 and splice, build, pack and confirm. You now have five real hooks;
step 7 makes the build see them.
