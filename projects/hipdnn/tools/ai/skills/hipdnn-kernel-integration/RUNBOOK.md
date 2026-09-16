# Runbook: land a kernel as a hipDNN integration

The **only ordered integration workflow**. [SKILL.md](SKILL.md) owns entry and
completion; the linked pages own contracts. Six steps, each with a named artifact. Each
gate needs a current observation; a previous session's log or a proposed command is not
evidence.

| Step | Artifact it must produce |
|---|---|
| 1. Read the kernel | A written **kernel contract**: ABI, geometry needs, specialization axes, admitted shapes |
| 2. Decide the symbols | A written **symbol decision**: which hooks, their names, new or reused, with the reason |
| 3. Write the native hooks | `<Pack>Native.cpp` with every declared hook implemented and registered |
| 4. Emit the descriptors | A generated descriptor set for those symbols, validated |
| 5. Register with the testing system | An `add_external_integration_test_target` entry plus its engine TOML, visible in `ctest -N` |
| 6. Add tests and graphs | Bundle cases under `integration-test-bundles/`, and a shared-suite run that names them |

**Nobody discovers step 5 at step 5.** Registration and graph authoring are the second
half of the job, not a follow-up ticket; read this table before you start step 1 and
budget for all six.

## Paths and interpreters

Resolve absolute paths before executing commands. These are explicit arguments, not
implicit tool inputs:

```bash
REPO=/absolute/path/to/rocm-libraries
PROVIDER="$REPO/dnn-providers/hip-kernel-provider"
GEN="$REPO/projects/hipdnn/tools/IngestorGenerator"
PY="$GEN/.venv/bin/python"
BUILD=/absolute/path/to/build
INSTALL=/absolute/path/to/install
GENERATED=/absolute/path/to/empty-generation-destination
CONFIG=/absolute/path/to/generator-config.yaml
ARCH=gfx942
ENGINE=hipkernel:YourPack
VALIDATOR="$BUILD/.../hipdnn_validate_descriptors"
FINAL_DESCRIPTOR_ROOT=/absolute/path/to/installed/descriptor-root
```

Use [generator setup](../../../IngestorGenerator/README.md#setup) for `$PY`. Build and
install through [hipdnn-superbuild](../hipdnn-superbuild/SKILL.md) with
`HIPDNN_ENABLE_KERNEL_INGESTOR=ON` and `HIPKERNELPROVIDER_ENABLE_TESTS=ON`: a new native
symbol is a rebuild and a reinstall, and steps 5 and 6 run against the installed tree.

## Environment

None of this is rediscoverable from an error message. Read the half that applies to you.

### Any host — these gate the build itself

- **`HIPDNN_ENABLE_KERNEL_INGESTOR=ON` pulls in Python build-time dependencies.** The
  `rocm_kpack` package must be importable, and it needs `msgpack` and `zstandard`. A
  configure that cannot import them fails or silently drops the ingestor, depending on
  where it trips.
- **The generator needs `PyYAML` and `Jinja2`.** Where a `.venv` is absent — it is not
  checked in, and no bootstrap step creates one for you — the interpreter you point `PY`
  at must already have both. Targets guarded on "a Python that can import PyYAML and
  Jinja2" **skip silently** when it cannot, leaving a `message(STATUS)` as the only
  signal.
- `PY="$GEN/.venv/bin/python"` and the generator README's setup block are **POSIX-only**.
  On Windows use the interpreter directly (`.venv\Scripts\python.exe`, or any Python with
  the two packages); there is no Windows variant of that block anywhere in the repo.

### Scheduled / multi-node hosts only

Skip this entire subsection on a local box — none of it applies.

- **A worktree under `claude-workspace/worktrees/` may be a symlink into the login
  node's `/var/tmp`**, which is login-node-local storage no compute node can see —
  `--constraint MARKHAM` or not. A bare `ls -d` on the symlink succeeds even when the
  target is invisible, which is exactly what hides it. Check what the link resolves to,
  not that the link exists.
- **`rsync` the checkout to the shared MARKHAM home before any scheduled job**, minus
  the build directory and any `.venv`, and build from the copy.
- **Build inside the container, in its own build directory.** The container's ROCm and
  the login node's differ, so a login-node build tree is not reusable inside the image
  and must not be overwritten by one.
- **Do not run GPU work on a login host**; go through the scheduler skill.

## Read these before you write the kernel

| Read | For | Needed by |
|---|---|---|
| [native-pack.md](../hipdnn-ingestor-engine/native-pack.md) | Every native hook's signature and obligation, plus registration and inventory rules | step 2 |
| [graph-contract.md](../hipdnn-ingestor-engine/graph-contract.md) | UID edges, node topology and field dispositions your `graph_match` must honour | steps 1-2 |
| Your own pack's `*Native.cpp` | The launch ABI and the geometry — **once you have written it.** Until then you are *choosing* them, not reading them | steps 3-4 |
| `IngestorKernelCode.hpp` | Bundle layout, containment and the compile-cache key — only if you use `kind: hiprtc_file` | step 4 |
| Your engine's TOML under `dnn-providers/hip-kernel-provider/config/` | Tolerance overrides and legitimate case skips for the shared suite | step 5 |
| [Integration test suite README](../../../../../../dnn-providers/integration-tests/README.md) | Bundle and sweep formats, `--test-engine`, external registration | steps 5-6 |

**Symbol names and dtype vocabulary are per pack, and the pack is now yours.** What your
kernel matcher compares, what your score returns and which dtype spellings your
`elementTypeFor` maps are three facts about *your* pack, not rules of the path. Where an
existing pack is quoted below it is quoted as a worked example, never as a constraint on
yours.

## The seam

Everything in this section is a **choice** when you own the handler, and a constraint
only when you are adding a variant to a pack that is already installed. That inversion
is the whole reason this skill exists separately from the drop-in page.

**Entry-point signature.** Derive it from your own `launch()` body, in order. Wrong
arity or order is diagnosed nowhere: hipRTC compiles the kernel, `getKernel(entry_point)`
resolves it, and the launch pushes whatever it has into whatever you declared. Two
shipped shapes, to calibrate: pointwise passes three pointers
(`PointwiseNative.cpp:458`), conv passes three pointers then seven ints
(`ConvNative.cpp:537-547`). Write the `launch()` first, then the kernel's parameter list
from it — not the other way round, and never from a sibling descriptor's source.

**Handler-supplied vs descriptor-bound defines.** Your handler adds the defines every
kernel it prepares needs, built by hand next to the compile call — conv adds
`HIP_PLUGIN_CONV_TYPE` and `HIP_PLUGIN_CONV_BLOCK_SIZE` (`ConvNative.cpp:498-499`) on top
of `KernelCompileOptions`' arch/dtype/layout base set. Descriptor-bound defines are
appended **after** the handler's and overwrite them by key
(`IngestorKernelCode.hpp:323-336`). Decide deliberately which macro belongs to which
side: anything derived, conditional or computed is the handler's, because the
descriptor's substituter does literal replacement and nothing else. Do not grow the
substituter to avoid a handler change.

**Geometry is the handler's, so the guard is the kernel's.** `prepare()` sets grid and
block. If the grid is a constant the kernel needs no bounds guard — that is the
pointwise scaffold's 1×1×1 (`PointwiseNative.cpp:436`) and it is the exception. If the
grid is computed from the shape (`ConvNative.cpp:509-515`) the final block is partially
populated and **your kernel must guard `index >= total` itself**; the shipped kernel
shows the idiom, including the `int64_t` arithmetic that keeps the guard honest
(`kernels/ConvFwd.cpp:25-35`). A missing guard writes past the output with nothing
raising an error.

**Header rules, if you ship a bundle.** A bundle is a directory, not an archive. Headers
are the provider's embedded list first, then bundle siblings: `.h`, `.hpp` and `.cuh`
only, one level deep, sorted by name (`IngestorKernelCode.hpp:115-212`). A second `.hip`
is not a header and cannot be included. A bundle header whose name collides with an
embedded one is a **load error, not a shadow**, deliberately. The bundle and the
`source_file` are each canonicalized and refused if they resolve outside the descriptor
tree (`IngestorKernelCode.hpp:287-316`).

**The compile-cache key is `(resolved source path, options)`**
(`IngestorKernelCode.hpp:346-350`) — the resolved path, not the bare `source_file`, so
two bundles each holding `attention.hip` do not share one entry. The source text is not
in the key, which is correct and has one consequence: an edited bundle source needs a
process restart, because the cache is process-lifetime.

## 1. Read the kernel

From the kernel, and from its author's handoff where one exists — never from a sibling
descriptor and never from the operation's popular name. An authored handoff is a statement
about *this* kernel; a sibling descriptor and a popular name are guesses about a different
one, and that prohibition stands whether or not a handoff arrived.
[hipdnn-kernel-authoring](../hipdnn-kernel-authoring/SKILL.md) hands over four facts — the
entry-point signature, the bundle's file set, the macros the source requires bound with
their legal values, and the launch geometry and workspace the kernel assumes. Take each as
an input and check it against the source; with no handoff, read all four out of the kernel
yourself. Write down:

- **Its ABI** — every parameter, its type and its position, and which are pointers to
  graph tensors versus scalars you must supply.
- **Its launch-geometry needs** — what grid and block shape make it correct, and whether
  it guards its own bounds. This decides your `prepare()`, and the seam above decides
  which side owes the guard.
- **Its specialization axes** — which quantities must be compile-time `-D` values and
  which can be runtime arguments. Compile-time axes become metadata fields; runtime ones
  become `launch()` arguments and cost nothing but registers. A macro the handoff names
  without its legal values is an unfinished handoff, not a default.
- **Its bundle file set**, if it ships one — every file you will place beside the source,
  by name. The seam above owns the mechanism (one directory level, `.h`/`.hpp`/`.cuh`
  only, and a name that collides with the provider's embedded list is a load error, not a
  shadow); step 1's job is that the set is enumerated before a descriptor names it,
  because a file nobody wrote down is a file nobody checked for a collision.
- **What it admits** — dtypes, layouts as stride patterns, alignment, divisibility,
  minimum and maximum extents. This is the raw material of `graph_match`, and anything
  you cannot state here will be claimed by accident. **The handoff does not supply this**:
  the author states what was *proven*, so read their does-not-prove list as the part of
  the envelope nobody validated, and derive the envelope itself from the source.
- **What scratch it needs**, if any — the handoff's workspace fact. Zero is a legitimate
  answer and the common one.

**Artifact:** a written kernel contract with those six sections.

**Gate:** every kernel parameter classified, every compile-time macro named with its
legal values, every bundle file named, and the admitted-shape envelope written down. An
unstated precondition is a shape you will silently claim.

## 2. Decide the symbols

Decide which of these this kernel needs, and name each one. **New by default.**

| Hook | Add it when | Specified by |
|---|---|---|
| Engine `graph_match` | Always. It is the gate on the whole engine's catalog | [native-pack.md](../hipdnn-ingestor-engine/native-pack.md) §Roles and §Matching |
| A graph-scoped pack criterion | Your engine has more than one pack and `graph_match` cannot tell them apart | same |
| A kernel-scoped candidate matcher | Two candidate kernels differ in something baked into metadata — dtype, tile, block size | same |
| `workspaceBytes` | The kernel needs scratch. Return 0 otherwise; do not invent a workspace | same |
| `IKernelDispatchHandler` | Always. It owns `prepare()`, geometry, defines and `launch()` | same |
| A native score | More than one candidate can serve the same graph and you can justify the ranking axis | same |

**Reuse only with a named existing pack that is yours and a stated reason.** "An
existing pack exists" is not the reason; "this is another block size for the pack I
shipped last month" is.

**Two of the three shipped packs are reference scaffolds and are never that reason.**
`PointwiseAdd` is one thread writing `c[0]` under
`if(blockIdx.x == 0 && threadIdx.x == 0)` (`kernels/PointwiseAdd.cpp:11-12`) with grid
1×1×1 (`PointwiseNative.cpp:436`); `ConvFwd` is a naive direct convolution admitting only
packed NCHW/KCRS, stride 1, dilation 1, zero padding and FLOAT/HALF, which is 6 of 1218
`ConvolutionFwd` bundle cases. They are there to exercise the ingestor path end to end
and to be read as worked examples. Attaching a real kernel to either inherits its
matcher, its geometry and its ABI — all three chosen for a toy — and it is not what
"extend an existing pack" means.

**The third, `hipkernel:BatchnormInference`, is the one pack for which "yes" is
available — and it is available only to whoever shipped it.** Nothing about it is a
toy: a full-tensor kernel with its own bounds guard
(`kernels/BatchnormInference.cpp:64-94`), a grid computed from the element count
(`BatchnormInferenceNative.cpp:642-647`), three io dtypes
(`BatchnormInferenceNative.cpp:97-101`), nine shipped variants
(`TestBatchnormInferencePacks.cpp:44-63`), and a matcher that refuses 15 parameterized
cases and admits 9 (`TestBatchnormInferenceMatchers.cpp:165-314`,
`TestBatchnormInferenceMatchers.cpp:65-129`). Extending it means another block size,
another io dtype, another architecture or another variant *of batchnorm inference* —
and its stated gaps are exactly the live axes: one proved architecture
(`IngestorGenerator/configs/batchnorm_inference.yaml:49`), no device-level integration
test wiring it, 10 of 82 `BatchnormInference/Default` bundle cases mirroring its own
unit-test shapes, and a per-channel parameter dtype pinned to FLOAT
(`BatchnormInferenceNative.cpp:103-107`). Hanging an unrelated kernel off it is not an
extension, and if you did not ship it you are on the create path.

**A fourth, `hipkernel:ConvPointwiseRtc`, is the second conditional yes and the only
pack serving a two-node graph.** A rank-4 `ConvolutionFwd` whose output tensor is
virtual, consumed by a unary `Pointwise`, in one launch with no workspace
(`packs/ConvPointwiseRtcNative.cpp`, `kernels/ConvFwdPointwiseFused.cpp`): nine variants
over three block sizes and three activations, a matcher that answers every field of both
attribute tables, and a handler that routes `kernel_source.kind` through
`buildIngestorKernelCode`. Its open axes are its stated limits — gfx90a alone, FLOAT
alone of the three `HKP_IO_DTYPE` tags that compile, and `{RELU_FWD, ABS, NEG}` of the
four `HKP_ACTIVATION` tags — and extending it along one of them is for whoever shipped
it, on the same ownership test batchnorm gets.

Most requests have no pack at all: only `ConvNative.cpp`, `PointwiseNative.cpp`,
`BatchnormInferenceNative.cpp` and `ConvPointwiseRtcNative.cpp` exist under
`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/`, and the
`s_packs` table in `IngestorPacks.cpp` registers exactly those four. Layernorm, RMSnorm
and resample are `hip_mlops_engine` plan builders with no ingestor pack; batchnorm has both, the
hand-written builder still serving the same single-node graph
(`BatchnormPlanBuilder.cpp:371-374`, `BatchnormPlanBuilder.cpp:550-554`). For the three
without a pack the answer to "which existing pack" is **none**, and step 3 is where the
work is.

**Artifact:** a written symbol decision — one row per hook, with its symbol name, new or
reused, and the reason.

**Gate:** every hook either named or explicitly declined with a reason, and the
create-vs-reuse answer stated. A reuse answer that does not name a pack you own is a
create answer that has not been written down yet.

## 3. Write the native hooks

Implement exactly what step 2 named, against
[native-pack.md](../hipdnn-ingestor-engine/native-pack.md), which already specifies all
of them: the roles and their signatures
(`../hipdnn-ingestor-engine/native-pack.md:7-21`), matching
(`../hipdnn-ingestor-engine/native-pack.md:22-40`), workspace, `prepare()` and
`launch()` (`../hipdnn-ingestor-engine/native-pack.md:42-61`), and registration and
inventory proof (`../hipdnn-ingestor-engine/native-pack.md:63-74`). It needs no
restating here — read it there.

**Ordering stays this RUNBOOK's.** That page opens with
`[RUNBOOK.md](RUNBOOK.md) owns implementation/build/test ordering`
(`../hipdnn-ingestor-engine/native-pack.md:3`), and that link points at the *ingestor*
skill's runbook — a production-mining workflow with corpus approval, packaging, tuning
and a final corpus proof. Take the contracts from that page and the order from this one;
do not follow that link for sequencing.

Two splices are both required and neither is optional: the pack's `SymbolScope<Handle>`
registration, and the row in `IngestorPacks.cpp`. `PointwiseNative.cpp:498-507` is the
worked example of the first, with its graph matcher, its three graph-scoped operation
matchers, its kernel matcher, its score and its dispatch handler all added to one scope.
Add the pack's source and test files to the engine's `target_sources` in the same pass.

**Landing a pack obliges you to refresh the named pack examples.** The
create-versus-extend decision is argued from which packs exist, which are scaffolds and
which is genuinely extendable, so a new file under
`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/` or a new
row in `IngestorPacks.cpp:16-26` makes those examples stale in the commit that adds it.
Four places carry them and must be updated in that same commit: step 2 above,
SKILL.md §New symbols are the default, `hipdnn-ingestor-engine/RUNBOOK.md` §1 entry
question 1, and `hipdnn-ingestor-engine/hiprtc-mining.md` §Scope — say in each which
your pack is, and whether it routes `kernel_source.kind`.

**The `IngestorPacks.cpp` row's cache fields follow your handler, not your dialect.** The
row is `(label, registerSymbols, ownsModuleCache, resetModuleCache)`, and
`TestIngestorPacksModuleCacheOwnership` asserts only that the last two agree — it passes
for both shapes, so it cannot tell you which one you needed. The rule that holds in
shipped code: **if your `prepare()` routes through `buildIngestorKernelCode`, your handler
holds a `KpackKernelLoader` and owns a module cache, so the row is
`true, &reset<Name>ModuleCache`** — define the reset outside the pack's anonymous
namespace and declare it in `IngestorPacks.hpp`. `hipkernel:Pointwise` is exactly this
case and ships `true, &resetPointwiseModuleCache` (`IngestorPacks.cpp:16`) despite its
kernels being `embedded_source`. Only a handler that does **not** route —
`hipkernel:ConvFwd`, `IngestorPacks.cpp:19` — takes `false, nullptr`. The generator's
`ingestor_packs_cpp.j2` fragment keys this off the *packaged* dialect instead, which
emits `false, nullptr` for a routed non-packaged engine; prefer this rule over the
fragment's comment when they disagree.

**The embedded header set is global and shared across the whole binary.** Adding a header
to your pack's embedded list changes what every other pack sees. `TestHiprtcFileKernelSource`
pins the virtual-header vector on the premise that the test binary embeds no headers of
its own, so a new embedded header breaks
`CollectsOnlyTopLevelHeadersByExtensionInNameOrder` — a real signal about a shared
resource, not incidental breakage. Expect to update that expectation in the same commit,
and say in the message that the set is shared.

Use `buildIngestorKernelCode` in `prepare()` rather than calling the compiler directly:
it is the one place source loading and path containment are handled for every
`kernel_source.kind` (`IngestorKernelCode.hpp:235-248`). A handler that calls
`_kernelCompiler.compile(kernel.source.sourceFile, options)` itself serves
`embedded_source` only — `ConvNative.cpp:501-502` is exactly that, and it is why a
`hiprtc_file` descriptor under the conv pack throws at plan-build time no matter how
correct the descriptor is.

**Artifact:** `<Pack>Native.cpp` with no reachable placeholder, its registration
function, its `IngestorPacks.cpp` row, and behavioural tests for the matcher and the
handler.

**Gate:** the provider builds and installs; every declared symbol is registered and
resolves; the pack's unit suite passes in a fresh process, because registration and
discovery are memoized. Host loading proves the symbols exist and nothing about
dispatch.

## 4. Emit the descriptors

Generate into an **empty** destination, never over the live engine, and never hand-write
a descriptor the generator can emit:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" --force
```

`--force` is not optional after the first run: generating a second time into a non-empty
output directory is refused, and every iteration on a bundle or a metadata value hits
that. Note also that the generator derives the native symbol namespace from the engine
name, so the engine name in the config is not a free label.

Check placeholders and, for a direct-load engine, the emitted destinations:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" \
  --check-placeholders \
  --emitted-root "$PROVIDER/src/engines/kernel_ingestor_engine" \
  --emitted-root "$PROVIDER/src/tests/engines/kernel_ingestor_engine"
```

Then validate the installed tree with a validator built from a provider that knows your
`kernel_source.kind`:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" --expect-engine "$ENGINE" --json
```

If — and only if — you are shipping variants of a pack that is already installed, the
`hiprtc_file` bundle, substituter and drop-in rules are owned by
[hiprtc-mining.md](../hipdnn-ingestor-engine/hiprtc-mining.md). A genuinely new native
symbol is a rebuild and cannot be dropped in.

**Artifact:** the emitted descriptor set, its engine name and uuids, and a validator exit
0 naming your engine.

**Gate:** descriptors emitted with zero hand edits, placeholders clean, validator green
against the installed tree.

## 5. Register with the testing system

The shared suite binary already takes `--test-engine` on the command line, so you can run
your engine by hand before this step exists — but **registration is what makes anyone
else run it.** Add an `add_external_integration_test_target` entry for your engine
alongside the provider's existing ones. The two long-standing examples are
`HIP_MLOPS_ENGINE` and `ASM_SDPA_ENGINE` (`dnn-providers/hip-kernel-provider/src/CMakeLists.txt:408-413`,
`:472-477`); the four `hipkernel:*` entries at `:313`, `:322`, `:339` and `:376` are closer
models for a new ingestor engine, and `:376` is the one that also stages a descriptor tree.
**Re-derive these line numbers before citing them** — this file gains entries regularly and
the numbers drift; `grep -n add_external_integration_test_target` is the reliable form. Supply:

- `ENGINE_NAME` — your UED engine name, passed through as `--test-engine`.
- `TEST_CONFIG` — a TOML your engine owns under
  `dnn-providers/hip-kernel-provider/config/`, for per-case tolerance overrides and
  legitimate skips. `HIP_MLOPS_ENGINE.toml` and `ASM_SDPA_ENGINE.toml` are the two
  shipped examples.
- `INSTALL_TEST_FILE` / `TEST_CATEGORIES_YAML` as the sibling entries use them, so the
  entry is registered at install time and tier labels apply. An engine registered only
  in the build tree is not exercisable against an install, which is where a customer
  meets it.

The macro's invocation contract and every argument it accepts are documented in the
[integration test suite README](../../../../../../dnn-providers/integration-tests/README.md).

**Artifact:** the CMake entry, the engine TOML, and a `ctest -N` listing from the
installed tree showing the registered test.

**Gate:** `ctest -N` lists your entry from the install prefix, not only from the build
tree, and its command line names your engine and the installed plugin.

## 6. Add tests and graphs

Graph coverage is added as **bundles**, not C++. The C++ graph tests are gated behind
`BUILD_CPP_GRAPH_TESTS`, which is `OFF` by default, and CMake enforces bundles for new
graph-verification coverage. Use `import_graph.py`, which is dedup-aware and assigns the
case id:

```bash
"$PY" "$REPO/dnn-providers/integration-tests/migration-scripts/import_graph.py" \
    --graph your_graph.json \
    --bundle-dir "$REPO/dnn-providers/integration-tests/integration-test-bundles/"
```

**Do this before the step 3/4 build, not after it.** The step numbering here is a
dependency order for *authoring*, not a wall-clock order for *building*: bundles are
`file(COPY)`-ed at CMake **configure** time, so a case imported after your last configure
is absent from the build and install trees no matter how correct it is. The symptom is
silent — "my case doesn't run" with no error — so import your graphs, then configure and
build, then run. If you have already built, re-configure after importing.

**`import_graph.py` is not turnkey when a skeleton hash is shared.** Sweeps are matched by
topology skeleton, and several can collide on one hash — `BatchnormInference` shares its
skeleton with `BatchnormFwdInference`. The tool then picks the alphabetically first within
the tier, whose template may use different tensor names, and dies with
`ERROR: round-trip verify failed after extraction`, naming neither the sweep it chose nor
the mismatched field. Two fixes, usually both: **narrow `--bundle-dir`** to the exact
target directory so no other sweep is a candidate, and **match your graph's tensor names to
the target template's** (for `BatchnormInference`: `X`, `BatchnormInference_0::Y`, and so
on). Read the template you are appending to before generating the graph.

It reports `DUPLICATE` and skips an identical case, appends to an existing topology's
`sweep.json`, or creates a new template+sweep directory, and prints the generated case id
to stderr — that id is the gtest name you will cite.

Cover **what your pack admits**, from step 1's envelope: each dtype, each layout, the
boundary shapes, and at least one case per specialization axis so more than one of your
candidates is actually selected. Add cases the pack must *decline* only where a wrong
answer rather than a skip is the risk.

A C++ integration test is the right tool only for something other than "a graph runs on
an engine" — error paths, API contracts, serialization, determinism, pass-by-value
semantics.

**Artifact:** the bundle case ids added, and the sweep or template files they live in.

**Gate:** the closing section below, run against the installed tree.

## Prove it in the shared suite, not in a private harness

**An integration is done when the shared suite runs graphs against it by name.** Not
when a bespoke harness says it served. A harness you wrote proves your kernel; the shared
suite proves your *integration*, against the same reference executor and the same bundle
corpus every other provider is held to.

**First, check the reference executor can actually run your op.** This gate comes before
the command, not after a failed run. The GPU reference executor implements a plan builder
per op family, and today that is exactly six — `ConvolutionFwd`, `LayernormFwd`,
`LayernormBwd`, `RMSNorm`, `Pointwise`, `SdpaFwd`
(`dnn-providers/integration-tests/src/harness/gpu-graph-executor/detail/`, one
`Gpu<Op>Plan.hpp` each). **If your op is not on that list, `--verification-mode gpu`
produces the exact silent skip it is prescribed to prevent**: every case reports *"GPU
reference cannot run this op"* and skips, the run exits 0 having passed zero, and the
target goes green. Measured in a dry run: `BatchnormInference` with `gpu`, **0 of 731
passed**, exit 0.

Pick the mode from that check and name it explicitly:

| Your op | Mode | Why |
|---|---|---|
| has a `Gpu<Op>Plan.hpp` | `gpu` | live GPU reference; strongest oracle |
| does not | `cpu` | live CPU reference; still an independent oracle |
| either | **never `auto`** | its golden → GPU → CPU → **skip** chain ends in a silent skip |

`graph-contract.md`'s rule still governs: missing capable independent numerics blocks the
feature. If neither reference executor can run your op, you do not have an oracle and that
is a `STOP`, not a mode to work around.

```bash
hipdnn_integration_tests --test-article <prefix>/lib/hipdnn_plugins/engines/<your>.so \
                         --test-engine "$ENGINE" \
                         --verification-mode <gpu|cpu, per the check above> \
                         --gtest_filter='*<YourOp>*'
```

- `--test-engine` pins the run to your engine, so an op your engine cannot serve **SKIPs**
  instead of falling through to another loaded engine. That is what makes the result
  attributable.
- An **explicit** mode is a demand for a specific oracle rather than `auto`'s fallback
  chain, which can silently land on golden tensors that were never pulled — or on nothing
  at all. Explicit does not mean `gpu`; it means the one you checked for above.
- **Naming an absent engine is a hard failure**, not a skip:
  `Error: Engine '<name>' is not loaded. Check the plugin path.`, exit 1, zero tests run.
  Use it as the negative control — it proves the positive run's passes were conditional
  on your engine existing at all.

**A green run is not coverage.** Because unservable cases skip, a run can exit 0 having
passed almost nothing. Both the shipped conv pack and its drop-in variant pass **6 of
1218** `ConvolutionFwd` cases; the other 1212 skip because the pack's matcher refuses
them, and that is correct behaviour rather than a failure. So:

- **Cite the passed count and the case names**, never the exit code. "Green" is not a
  result; `quick_ConvolutionFwd_Default.1_16_16_16_fp32_nchw_dil1x1_postpad0x0_prepad0x0
  passed` is.
- **Cite which of your candidates were selected.** If every pass came from one kernel,
  your second variant is unexercised no matter how many cases ran.
- A new pack's admitted shapes are a thin slice of an existing sweep, and **widening that
  slice is part of the work** — report the slice honestly rather than reporting the exit
  code.

Two flags that must not be assumed into this loop: `--enforce-support-claims` requires a
claims config no ingestor engine owns today, so it is not part of the loop; and
`--fail-on-unsupported`, measured against 1218 cases, left 1212 skips as skips — it does
**not** turn skips into failures and must not be described as doing so.

**Gate:** a named passed count with case names, from the installed tree, with the
absent-engine control run and its exit 1 recorded. Zero selected, all-skipped, or another
engine's work is not correctness evidence.

## Handoff

Report [SKILL.md](SKILL.md)'s four deliverables, each with its own evidence, and the
does-not-prove list: architectures not run, dtypes not covered, shapes declined, and any
performance silence. Name the last completed step and the missing prerequisite for
anything blocked. A generated descriptor, a proposed command or a queued job is not a
completed step.
