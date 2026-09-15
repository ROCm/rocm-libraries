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

These cost two failed jobs the first time and are not rediscoverable from an error
message:

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
(`IngestorKernelCode.hpp:283-295`). Decide deliberately which macro belongs to which
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
only, one level deep, sorted by name (`IngestorKernelCode.hpp:118-155`). A second `.hip`
is not a header and cannot be included. A bundle header whose name collides with an
embedded one is a **load error, not a shadow**, deliberately. The bundle and the
`source_file` are each canonicalized and refused if they resolve outside the descriptor
tree (`IngestorKernelCode.hpp:251-277`).

**The compile-cache key is `(resolved source path, options)`**
(`IngestorKernelCode.hpp:301-305`) — the resolved path, not the bare `source_file`, so
two bundles each holding `attention.hip` do not share one entry. The source text is not
in the key, which is correct and has one consequence: an edited bundle source needs a
process restart, because the cache is process-lifetime.

## 1. Read the kernel

From the kernel, not from a sibling descriptor and not from the operation's popular
name. Write down:

- **Its ABI** — every parameter, its type and its position, and which are pointers to
  graph tensors versus scalars you must supply.
- **Its launch-geometry needs** — what grid and block shape make it correct, and whether
  it guards its own bounds. This decides your `prepare()`, and the seam above decides
  which side owes the guard.
- **Its specialization axes** — which quantities must be compile-time `-D` values and
  which can be runtime arguments. Compile-time axes become metadata fields; runtime ones
  become `launch()` arguments and cost nothing but registers.
- **What it admits** — dtypes, layouts as stride patterns, alignment, divisibility,
  minimum and maximum extents. This is the raw material of `graph_match`, and anything
  you cannot state here will be claimed by accident.
- **What scratch it needs**, if any. Zero is a legitimate answer and the common one.

**Artifact:** a written kernel contract with those five sections.

**Gate:** every kernel parameter classified, every compile-time macro named with its
legal values, and the admitted-shape envelope written down. An unstated precondition is
a shape you will silently claim.

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

**The two shipped packs are reference scaffolds and are never that reason.**
`PointwiseAdd` is one thread writing `c[0]` under
`if(blockIdx.x == 0 && threadIdx.x == 0)` (`kernels/PointwiseAdd.cpp:11-12`) with grid
1×1×1 (`PointwiseNative.cpp:436`); `ConvFwd` is a naive direct convolution admitting only
packed NCHW/KCRS, stride 1, dilation 1, zero padding and FLOAT/HALF, which is 6 of 1218
`ConvolutionFwd` bundle cases. They are there to exercise the ingestor path end to end
and to be read as worked examples. Attaching a real kernel to either inherits its
matcher, its geometry and its ABI — all three chosen for a toy — and it is not what
"extend an existing pack" means.

Most requests have no pack at all: only `ConvNative.cpp` and `PointwiseNative.cpp` exist
under `dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/packs/`.
Batchnorm, layernorm, RMSnorm and resample are `hip_mlops_engine` plan builders with no
ingestor pack. For those the answer to "which existing pack" is **none**, and step 3 is
where the work is.

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

Use `buildIngestorKernelCode` in `prepare()` rather than calling the compiler directly:
it is the one place source loading and path containment are handled for every
`kernel_source.kind` (`IngestorKernelCode.hpp:195-208`). A handler that calls
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
alongside the provider's existing ones
(`dnn-providers/hip-kernel-provider/src/CMakeLists.txt:213`,
`dnn-providers/hip-kernel-provider/src/CMakeLists.txt:277`), supplying:

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

```bash
hipdnn_integration_tests --test-article <prefix>/lib/hipdnn_plugins/engines/<your>.so \
                         --test-engine "$ENGINE" \
                         --verification-mode gpu \
                         --gtest_filter='*<YourOp>*'
```

- `--test-engine` pins the run to your engine, so an op your engine cannot serve **SKIPs**
  instead of falling through to another loaded engine. That is what makes the result
  attributable.
- `--verification-mode gpu` demands a live GPU reference rather than `auto`'s fallback
  chain, which can silently land on golden tensors that were never pulled.
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
