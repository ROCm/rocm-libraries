# Adding a variant to a pack that is already yours: `kind: hiprtc_file`

**This page applies when you are adding a kernel variant to a pack you already shipped
and whose native symbols are already installed.** Descriptors plus a directory of HIP
sources, copied into an installed tree and compiled at `prepare()`: a new variant with no
rebuild. **If you are integrating a new kernel, you want
[hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md)**, whose RUNBOOK owns
the create path end to end — new symbols, descriptors, registration and graphs. That is
the normal case; this one is the exception with a rebuild avoided.

For the reuse case this page covers, [RUNBOOK.md](RUNBOOK.md) §4 owns execution — generate,
copy the descriptor directory whole, restart — and [rocke-mining.md](rocke-mining.md) owns
the kernel contracts assumed here.

## Scope: variants over installed symbols only

`loadValidatedDescriptorSets` (`DescriptorLoader.hpp:2052-2106`) pre-flights every
`match_symbol`, `graph_match`, `dispatch_symbol` and score symbol against the native
registries and drops the **whole engine** on any miss, with one `LOG_ERROR` at a severity
the default log level never shows. At the API that is indistinguishable from a healthy
decline, so **read the loader's log before believing anything else.**

A dropped-in set therefore reuses an installed pack's registered symbol strings
(`PointwiseNative.cpp:57-63`, `PointwiseNative.cpp:498-507`). A genuinely new symbol
needs a provider rebuild; no format change fixes that, and it is ordinary create-path
work under [hipdnn-kernel-integration](../hipdnn-kernel-integration/SKILL.md) rather than
anything this page can help with.

**Reusing the symbols is necessary and not sufficient.** The pack's dispatch handler must
also dispatch on `kernel_source.kind`, i.e. call `buildIngestorKernelCode`. Grep the
handler's `prepare()` for that call before authoring anything: a handler that calls
`_kernelCompiler.compile(kernel.source.sourceFile, …)` directly serves `embedded_source`
only, and a `hiprtc_file` descriptor under it throws at `prepare()` no matter how correct
the descriptor is. This is a per-pack property, not a property of the format: as of this
writing Pointwise (`PointwiseNative.cpp:432-433`), BatchnormInference
(`BatchnormInferenceNative.cpp:632-633`) and ConvPointwiseRtc
(`ConvPointwiseRtcNative.cpp`, in `prepare()`) route; ConvFwd (`ConvNative.cpp:501-502`)
does not. Routing a pack is a two-line handler change and a rebuild.

**Four packs have installed symbols today, and this page still serves only the one you
shipped.** It adds no native symbol, so it can never widen the set of packs, and of the
four only the three named above can serve a `hiprtc_file` descriptor at all. Two of the
four are reference scaffolds in any case — `PointwiseAdd` computes one element under
`if(blockIdx.x == 0 && threadIdx.x == 0)` (`kernels/PointwiseAdd.cpp:11-12`) and
`ConvFwd` is a naive direct convolution serving 6 of 1218 `ConvolutionFwd` bundle cases —
so the real extension targets here are `BatchnormInference` and `ConvPointwiseRtc`, each
only for the person who shipped it (see the `s_packs` table in `IngestorPacks.cpp`). If
none of the four is a pack *you* shipped, this page is not the one you want.

**Within those bounds it is the cheapest vehicle for an exhaustive sweep.** A variant here
costs a descriptor entry and a file in the bundle rather than a rebuild, so the whole
descriptor-cost tier of a tuning space — block sizes and any metadata value the
substituter can render into a `-D` — can be enumerated and measured against one installed
provider. That tier is what an exhaustive sweep is scoped to; axes that cost a rebuild or
new authored source stay on `knob_sweep.py`'s staged isolate-then-pair order in
[RUNBOOK.md](RUNBOOK.md) §6. The two constraints just argued are what bound the cheapness
and they do not relax for a sweep: this page adds no native symbol, so it serves only a
pack already installed — and only the one you shipped — and only a pack whose handler
routes `kernel_source.kind`. An exhaustive sweep over a pack failing either is a rebuild
wearing a descriptor's clothes, and it is priced accordingly.

### Two drop-in shapes, and only one of them is observable

A KDP under `HIPDNN_DESCRIPTOR_RUNTIME_DIR` whose `engine` is an installed UED's uuid
attaches to that engine (`DescriptorLoader.hpp:1798-1803`), and its inline kernels are
stamped with the runtime root as the `treeRoot` their bundle is contained against
(`:1830-1837`). Redefining a uuid the installed tree already defines is refused and logged,
never honoured (`:1119-1132`) — a drop-in is additive only.

That **KDP-only** shape — one `.kdp.json` plus a bundle — is the smallest thing that loads,
and it has **no API-visible identity**. One ingestor engine is constructed per discovered
descriptor set, and its engine id is `engineNameToId(set.engine.name)`
(`Container.cpp:106-144`), so a KDP that joins the installed set is served under the
*installed* engine's id. The engine is listed whether or not your drop-in is present, and
`get_execution_plan_engine_id()` — the only identity the frontend exposes — returns that
same id either way. You can observe that *something* produced the right numbers; you cannot
observe that **your** kernel did.

**Ship your own UED over the installed pack's symbols instead.** A full descriptor set
(UED, UHD, UDD, KMD, the UMDs, the KDP) whose `graph_match`, heuristic payload,
`dispatch_symbol` and `match_symbol` are the installed pack's registered symbol strings
*verbatim* satisfies the scope constraint above — it adds no native symbol — while giving
the drop-in its own engine name, therefore its own id, therefore an appearance and a
disappearance you can assert on. This is the shape that has actually been run end to end on
device; the worked example is `Results/hiprtc-dropin-kernels/phase5/` in the
claude-workspace (`pointwise_dropin.yaml`, `pointwise_dropin_sources/`, written up in
`PHASE5-endtoend.md`). The KDP-only shape remains **unrun**.

Choose the KDP-only shape only when you are adding a variant to an engine you already trust
and never need to tell apart from the shipped kernels. Choose your own UED whenever anyone —
you, a test, a bug report — has to confirm which kernel served.

## Entry point: the handler's argument list, verbatim

Launch geometry and operand binding belong to the registered `IKernelDispatchHandler`, not
to your descriptor. Pointwise `prepare` (`PointwiseNative.cpp:417-440`) fixes grid 1×1×1
and block `block_size`×1×1 (`:435-436`); `workspaceBytes` returns 1024 only at
`block_size == 256` (`:408-415`); `launch` (`:442-459`) passes exactly three pointers in
`(inputA, inputB, output)` order. So the entry point is

```cpp
extern "C" __global__ void <entry_point>(const T* a, const T* b, T* c);
```

— the signature shipped `kernels/PointwiseAdd.cpp:7-9` already has. **Wrong arity or order
is diagnosed nowhere**: hipRTC compiles it, `getKernel(entry_point)` resolves it, and the
launch passes three arguments into whatever you declared. Derive the list from the
handler's `launch` body, never from a sibling descriptor's source.

The handler supplies defines to every kernel it prepares, `hiprtc_file` included:
`HIP_PLUGIN_POINTWISE_TYPE` (`elementTypeFor(kernel)` → `float` or `_Float16`, `:355-370`)
and `HIP_PLUGIN_POINTWISE_BLOCK_SIZE` (`:429-430`), over `KernelCompileOptions`'
arch/dtype/layout base set (`KernelCompileOptions.hpp:113-145`). Use them; do not declare
them.

## What belongs where

| Fact | Home | Cost |
|---|---|---|
| Which graph a kernel may serve | `metadata` + the pack's matcher symbols | none |
| A metadata field's value, verbatim, as `-D` | `kernel_source.defines` | none |
| Anything derived, conditional or computed | the pack's dispatch handler | **rebuild** |
| Which candidate wins | the pack's score symbol | none |

`pointwiseKernelMatches` (`:277-288`) requires `metadata.dtype` to equal the graph dtype's
flatbuffers enum spelling — `FLOAT`, `HALF`, `BFLOAT16`. `pointwiseScore` (`:290-295`)
returns `block_size`, highest first. **That ranking is within one engine and decides nothing
between engines**: `rank()` orders the entries of a single catalog
(`IKernelHeuristic.hpp:52-82`), and a catalog belongs to one descriptor set's state manager
(`KernelIngestorStateManager.hpp:231`). So `block_size` decides which of *your* variants
wins against *your* other variants — a drop-in with its own UED is chosen against the
shipped engine by engine preference, not by score, and a KDP-only drop-in that joins the
installed set is the only case where outscoring an installed kernel is even the question.
Ties are broken by `priority`, then by ascending `kernelId` (`IKernelHeuristic.hpp:63-73`) —
not by load order. Three traps:

- `elementTypeFor` knows only `FLOAT` and `HALF`. A `BFLOAT16` variant **matches**, then
  throws at `prepare()`.
- Bound defines are added after the handler's and `add` overwrites
  (`KernelCompileOptions.hpp:80-83`, `IngestorKernelCode.hpp:323-336`), so a `defines` key
  naming `HIP_PLUGIN_POINTWISE_TYPE` wins over the handler's. Deliberate — the more
  specific statement wins — and a loaded gun.
- `$kernel.dtype` renders `FLOAT`, not `float`. Mapping a tag to a device type is
  `elementTypeFor`'s job in native code, or your bundle's preprocessor.

## Rendering, pinned

`renderMetadataValueForDefine` (`KernelDefineSubstitution.hpp:170-212`):

| KMD type | Renders as |
|---|---|
| `bool` | `1` / `0` |
| `int` | decimal |
| `string` | verbatim |
| `float` | **rejected** |
| `int_list` | **rejected** |

`float` has no single spelling: `std::to_chars(1.0)` gives `1`, Python's `repr(1.0)` gives
`1.0`, and `-DALPHA=1` and `-DALPHA=1.0` are different types in device code. That is not
tidiness. The compile cache is keyed `(resolved source path, options)`
(`IngestorKernelCode.hpp:346-350`), so **two variants differing only in a bound field whose
rendering is unpinned compile once and silently become one kernel.** A float or a list a
kernel genuinely needs goes through the dispatch handler.

## Bundle layout

A bundle is a **directory**, not an archive. `bundle` resolves against the descriptor's own
directory and must stay inside the walked `treeRoot`; `source_file` is containment-checked
separately, being authored too (`IngestorKernelCode.hpp:287-316`).

```
$HIPDNN_DESCRIPTOR_RUNTIME_DIR/pointwise_add_dropin/
    pointwise_add_dropin.kdp.json     # "bundle": "sources"
    sources/PointwiseDropin.hip       # "source_file"
    sources/PointwiseDropinTypes.h    # visible to #include
```

Headers are the provider's embedded list first (`getKernelIncList`), then bundle siblings —
`.h`, `.hpp`, `.cuh` only, one level deep, sorted by name (`IngestorKernelCode.hpp:115-212`)
— handed to `hiprtcCreateProgram` as virtual headers (`Program.cpp:40-59`, `:68-73`). A
bundle header whose name collides with an embedded one is a **load error, not a shadow**
(`:160-172`): hipRTC resolves the first match, so either outcome would be invisible. A
second `.hip` is not a header and cannot be included. One bundle serves many kernels; ship
every file the sources need, not only the ones a descriptor names.

## What the substituter refuses

Literal `$kernel.<field>` replacement, single pass, nothing else
(`KernelDefineSubstitution.hpp:94-166`).

| Authored | Outcome | Instead |
|---|---|---|
| `$kernel.<undeclared>` | pack dropped at set resolution, `LOG_ERROR` (`DescriptorLoader.hpp:1298-1301`, `:1912-1929`) | declare the field in the KMD |
| `$kernel.block_size * 2` | refused: `+ - * / % = < > ! & \| ^ ~ ? : ( )` in a value that binds a token (`:62-92`) | compute it in the handler, or in the bundle's preprocessor |
| `$graph.x`, any other `$` | refused (`:131-141`) | no other binding source exists |
| `$kernel.` with no field | refused (`:149-154`) | name the field |
| a `float` or `int_list` field | refused | dispatch handler |
| a `$` inside a rendered value | not rescanned (`:161-163`) | nesting is not a feature |
| `$kernel.dtype_t`, with a field named `dtype` | refused: the identifier scan is greedy (`KernelDefineSubstitution.hpp:142-147`), so this binds a field called `dtype_t` and is rejected as undeclared | rename the macro, or move the suffix into the bundle's preprocessor |

A value containing **no** `$` is opaque and passes through byte-identical, so `-DLIMIT=-1`
stays authorable. Validation is schema-level and runs once at set resolution, so a field
the descriptor omits and the KMD defaults still validates (`:243-287`).

**A bound token cannot be followed by an identifier character.** The scan runs from
`$kernel.` to the first character that is not a letter, digit or underscore, so
`"$kernel.dtype_t"` asks for a field named `dtype_t` rather than for `dtype` followed by
the text `_t`. **There is no `${kernel.dtype}` brace form**, and adding one is out of
scope — the substituter does not grow. The workaround is to put the fixed text on the
other side of the boundary: bind `HIPDNN_MY_DTYPE: "$kernel.dtype"` and let the bundle's
own `#define`/`##` paste build `MyType_t` from the tag, exactly as the tag-to-type idiom
below already does. It fails loudly at set resolution, so this is an authoring
ergonomics gap rather than a wrong answer.

**The operator refusal is a lint over *authored* text only, and does not constrain the
rendered result.** A `string` metadata value is inserted verbatim and never inspected, so
binding `"$kernel.expr"` against a metadata value of `2 + 1` really does put `2 + 1` into
a `-D` flag. That is correct per the rendering table above — `string` → verbatim — and
the *mechanism* is pinned by two tests, `StringRendersVerbatim`
(`TestKernelDefineSubstitution.cpp:109-112`) and `ReplacementTextIsNotRescanned`
(`TestKernelDefineSubstitution.cpp:211-222`). Do not read the refusal as a guarantee that
no operator reaches the compiler: it guarantees only that you did not *write* one into a
value that binds a token.

## The escape hatch and its price

`PointwiseNative.cpp:427-431` already is the hook — it builds `KernelCompileOptions` by
hand and adds a dtype conditional (`elementTypeFor(kernel)`) beside a metadata int.
Anything derived or conditional goes there, with the `KernelDefinition` and `MatchContext`
in hand. It is native code, so it costs a rebuild and a reinstall. That is the trade: the
substituter buys no-rebuild variants of a **fixed** compile command and nothing more. Do
not grow it.

## Known limitations

1. Drop-in serves new variants of an existing pack only; new native symbols need a rebuild.
2. Compile-arg binding reads one metadata field and nothing else; conditional or computed
   args mean a dispatch-handler change, i.e. a rebuild.
3. `float` and `int_list` cannot be bound into defines.

The phase 3 generator cannot express four things a drop-in needs. Each is a hand edit after
`generate.py`:

4. **No "installed symbols, new engine name".** The native symbol namespace is *derived
   from* the engine name in `IngestorGenerator/codegen/models.py`:
   `hipkernel:Pointwise` → `hipkernel.pointwise` →
   `hipkernel.pointwise.{graph_match,score,dispatch,kernel_match}`. Those are the
   installed symbols, so the config must say `engine.name: hipkernel:Pointwise` — which is
   also the installed engine's name, and therefore its id. Author the config with the
   installed name, then rewrite the emitted UED's `name` field. Nothing else in the set
   carries the engine name.
5. **Discriminate the operation in your matcher, or emit a graph-scope discriminator
   UMD.** `build_operation_umd` in `IngestorGenerator/codegen/generator.py` returns `None`
   unless the engine is multi-pack, so a single-pack engine's KDP lists only its
   kernel-scoped matchers. Whether that is complete is a property of **your**
   `graph_match`, not of this path, and these two shipped matchers fall on opposite sides:

   - `pointwiseGraphMatches` (`PointwiseNative.cpp:191`) checks shape and arity and says
     nothing about the operation — the operation lives in separate graph-scoped matchers
     (`PointwiseNative.cpp:251-270`). A single-pack pointwise drop-in therefore claims MUL
     and SUB graphs too **and adds them**: a silent wrong answer, not a load error.
   - `convFwdGraphMatches` (`ConvNative.cpp:192`) admits the node type *and* validates it
     in one pass, so the generator's output is already complete. The shipped
     `conv_fwd.kdp.json` carries exactly one matcher, and `config_loader` actively
     **rejects** a discriminator declared for a single-pack engine.

   So: if your `graph_match` is itself the discriminator, check the shipped KDP's
   `matchers` list and match it — there is no discriminator UMD to hunt for, and inventing
   one fails generation. If it is not, add the installed `operation_is_<op>` UMD uuid to
   the KDP's `matchers`, read out of the installed tree.

   The generator cannot tell which case you are in, because the answer lives in native
   code it never sees. It therefore states the condition and makes you answer:

   > warn whenever a single-pack engine emits no graph-scope discriminator, unless the config carries `engine.pack_discriminates: true`

   Setting that key is you asserting the conv case; leaving it unset and ignoring the
   warning is how the pointwise case ships broken.
6. **One KDP per pack**, so both variants land in one file and a drop-in cannot stage one
   variant at a time. Splitting is a file split plus a fresh KDP uuid. The obvious
   alternative — one pack per variant — collides with limitation 5 for a pack whose
   `graph_match` does not discriminate, because two such packs sharing a discriminator
   emit the same `operation_is_<disc>.umd.json` filename. For a self-discriminating pack
   there is no such collision, and one pack per variant may work — untested.
7. **The emitted `provenance` block is dead weight in a drop-in**, and costs one
   `descriptor loader: extension key 'provenance' … ignoring it` WARN per KDP load. Harmless,
   but it is noise in exactly the log you are told to read.

## The authoring loop

**Author** from `IngestorGenerator/configs/hiprtc_dropin.yaml` — two kernels, one source
file, differing only in `metadata`. Set `kernel_source_kind: hiprtc_file` and put the
bundle directory beside the config under the name `bundle` names. A kernel's
`kernel_source` keys **replace** `kernel_defaults` key for key, so restate `defines` in
full per kernel.

```yaml
- name: pointwise_add_dropin.f32_block512
  kernel_source:
    kind: hiprtc_file
    bundle: sources
    source_file: PointwiseDropin.hip
    entry_point: PointwiseDropin
    defines:
      HIPDNN_DROPIN_DTYPE: "$kernel.dtype"        # renders FLOAT / HALF
      HIPDNN_DROPIN_BLOCK: "$kernel.block_size"   # renders 512
  metadata: { block_size: 512, dtype: FLOAT, operation: ADD }
  priority: 0
```

`operation: ADD` in that `metadata` is documentation and nothing else: no pointwise symbol
reads it (`PointwiseNative.cpp:277-295` read only `dtype` and `block_size`), and the
operation is decided by the graph-scoped discriminator UMD — which the generator will not
emit for you (limitation 5).

**The second variant is that block copied with `dtype: HALF`** and a new `name`;
`kernel_source` stays identical byte for byte. The descriptor ships the template and the
target resolves it per kernel — that is the whole feature. Ids are minted fresh per run
(`uuid4`, in `IngestorGenerator/codegen/generator.py`), so uniqueness is automatic, and a
**replacement** for a dropped one, never an addition beside it. The bundle turns the tag
into a type, because the substituter evaluates nothing:

```cpp
#define HIPDNN_DROPIN_T_FLOAT float
#define HIPDNN_DROPIN_T_HALF  _Float16

// Two levels, deliberately. `##` suppresses expansion of its operands, so the one-level
// form `HIPDNN_DROPIN_T_##tag` pastes the macro NAME and yields the non-existent
// HIPDNN_DROPIN_T_HIPDNN_DROPIN_DTYPE — "does not name a type", reproduced on device.
// PASTE does the paste; CAT expands its argument first.
#define HIPDNN_DROPIN_PASTE(tag) HIPDNN_DROPIN_T_##tag
#define HIPDNN_DROPIN_CAT(tag) HIPDNN_DROPIN_PASTE(tag)
using DropinElement = HIPDNN_DROPIN_CAT(HIPDNN_DROPIN_DTYPE);

extern "C" __global__ void PointwiseDropin(const DropinElement* a,
                                           const DropinElement* b,
                                           DropinElement* c)
```

Guard each bound macro with `#ifndef <NAME> / #error` before using it: an unbound token
otherwise compiles against whatever the tag happens to mean and fails only in the numbers.
This block is the one compiled on device; the verified copy is
`pointwise_dropin_sources/PointwiseDropinTypes.h` under the phase 5 evidence directory in
the claude-workspace, outside this repository.

**Generate**, from the worktree root:

```bash
cd projects/hipdnn/tools/IngestorGenerator
./.venv/bin/python generate.py --config configs/<your>.yaml --output-dir "$OUT"
```

It stages every regular file in the bundle and refuses an escaping bundle, a non-string
define, or a token its KMD cannot render — the loader's rules, applied before shipping.

**Drop in**: copy `"$OUT"/descriptors/<pack>/` whole — descriptor JSONs plus the bundle —
into `$HIPDNN_DESCRIPTOR_RUNTIME_DIR`. `packs/`, `tests/` and `fragments/` are the
rebuild-requiring half and are not part of a drop-in. For the **own-UED** shape, rewrite the
emitted UED's `name` to a name nothing installed uses, leaving every symbol string it
carries untouched (limitation 4 above), and add the installed graph-scope discriminator UMD
to the KDP's `matchers` (limitation 5). For the **KDP-only** shape, ship only the
`.kdp.json` and its bundle, with `engine`, `dispatch` and `matchers` rewritten to the
installed set's uuids and a fresh uuid for the KDP and each kernel — and accept that you
cannot confirm it served. Read those uuids out of the installed descriptor tree, not out of
this worktree.

**Restart the process.** Discovery is memoized in a function-local static
(`KernelIngestorEngine.cpp:141-154`) over the shipped tree then
`HIPDNN_DESCRIPTOR_RUNTIME_DIR` (`:75-96`); a restart is required and sufficient. The
compile cache is process-lifetime, so an edited bundle source needs one too.

**Confirm it served**: with your own UED, the drop-in's engine id appears in
`get_ranked_engine_ids` only while the tree is present, and
`get_execution_plan_engine_id()` names it for the graphs it wins — that pair, plus correct
numbers, is the proof. With the KDP-only shape neither of those changes, so the only
available signals are the loader's log (the set loaded from the runtime root) and the
numbers; an engine id proves nothing there. Either way, an absent engine is a dropped set —
read the loader's `LOG_ERROR`. A kernel that loaded but lost is a score question, not a
binding one.
