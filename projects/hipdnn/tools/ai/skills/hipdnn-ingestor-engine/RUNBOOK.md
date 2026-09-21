# Runbook: create or extend an ingestor engine

This is the **only ordered create/extend workflow**. Use [SKILL.md](SKILL.md) for
entry/completion requirements and the linked domain pages for contracts. Each gate
requires current evidence; an old install or copied command is not an observation.

## Paths and interpreters

Resolve absolute paths before executing commands. These are explicit arguments,
not implicit tool inputs:

```bash
REPO=/absolute/path/to/rocm-libraries
PROVIDER="$REPO/dnn-providers/hip-kernel-provider"
GEN="$REPO/projects/hipdnn/tools/IngestorGenerator"
PY="$GEN/.venv/bin/python"
BUILD=/absolute/path/to/build
INSTALL=/absolute/path/to/future-install
SWEEP_ROOT=/existing/device-visible/writable/experiment-root
GENERATED=/absolute/path/to/empty-generation-destination
CONFIG=/absolute/path/to/generator-config.yaml
PROFILE=/absolute/path/to/authoring.profile.yaml
SHAPES=/absolute/path/to/request-shapes.json
CORPUS_DIR=/absolute/path/to/graph-corpora
ARCH=gfx942
ENGINE=<your-bundle-engine-id>
```

`ENGINE` carries the `<your-bundle-engine-id>` placeholder and is consumed verbatim as
`--expect-engine` later, so it must be replaced with your own bundle's engine ID
before any command below will resolve. A gfx942 dense
attention bundle would spell it `hipkernel:Gfx942AttentionDense`; no such engine is
shipped from this tree.

Follow the **Setup** section of the generator's own README at `$GEN/README.md`.
Authoring/mining
imports need the profile's rocKE library environment; production packaging uses
its selected compiler/wheel interpreter instead. Full artifact checking also needs
`rocm_kpack` and its dependencies in `$PY` (including `zstandard` and `msgpack`).
`--kpack-python-dir` supplies an import root, not missing Python dependencies.

Verify source, build, install, corpus and output paths on the execution host. A
workspace symlink backed by login-local **`/var/tmp` is invisible from compute
nodes**, even when its link name is under shared storage. Use shared source or stage
the exact checkout/artifacts to compute-local scratch through the active scheduler
skill. Do not run GPU work on a login host. Retain command logs and source/artifact,
machine, device and allocation identities under the workspace's evidence policy.

## 1. Entry and early feasibility

Record the entry contract; for extend, inventory the installed baseline according
to [extend.md](extend.md). Select `direct_load` or `packaged` before staging files.
Establish that the graph is representable and has a capable independent numerical
reference using [graph-contract.md](graph-contract.md). A reference's skip is not
verification; unavailable semantics require an explicit scope/reference decision.

On the actual allocated execution host:

```bash
"$PY" "$GEN/tools/device_probe.py" \
  --mode early --arch "$ARCH" --sweep-root "$SWEEP_ROOT"
```

Early mode requires the requested device and an existing writable root. It ignores
inherited `INSTALL` and rejects `--install`; `$INSTALL` may name a future directory.
Exit 0 proves feasibility only, exit 1 a device/path/write failure, exit 2 an
invalid invocation, and exit 3 that neither `rocminfo` nor `hipInfo` could be run, so
the device was never observed. Exit 3 is not a device-absent verdict and the gate it
leaves unmet is discharged by obtaining an inspection utility on this host, not by
moving to another one. For rocKE, confirm the actual builder/spec and
`(spec, *, arch)` interface; an unknown architecture inventory needs source
investigation.

**Gate:** feasible target/workspace, representable scope and capable reference. A
missing dependency blocks its gate; host-only research may continue while a device
allocation is pending, but cannot discharge device proof.

## 2. Contracts, corpus and baseline approval

Record [graph-contract.md](graph-contract.md)'s topology/UID edges and field
dispositions, then [rocke-mining.md](rocke-mining.md)'s applicability, specialization,
layout, geometry/workspace and ABI evidence. Direct-load engines obtain these facts
from HIP source without inventing a profile. Reuse unchanged extension contracts;
reopen any topology or feature the addition changes.

Inventory owner-published results, owner benchmark shapes and external graphs per
[workloads.md](workloads.md). `$SHAPES` is a JSON list of semantic requests; the
benchmark consumes actual graph JSON under `$CORPUS_DIR/<corpus>`. For the attention
miner, using the sources in scope:

```bash
"$PY" "$GEN/tools/mine_shapes.py" \
  --published /absolute/path/to/owner-results.csv \
  --graphs "$CORPUS_DIR/servable" --arch "$ARCH" \
  --include-windowed --out "$SHAPES"
```

Add `--rocke-bench <actual-benchmark-tree>` when applicable. Reconcile each source's
total, parsed, servable, covered and excluded counts; do not discard window/sink or
independent operand dimensions to fit the request schema.

For rocKE, resolve the provisional baseline through its actual dispatcher:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" \
  --shapes "$SHAPES" --out "$CONFIG" --report-knobs --report-gaps
"$PY" "$GEN/tools/reconcile_applicability.py" \
  --profile "$PROFILE" --shapes "$SHAPES"
```

The second command is **offline applicability**, not runtime coverage or numerics.
Scope reference candidates to the kernel family/algorithm and required opt-in
selector. API failures are operational errors, never unsupported-shape evidence.

Present the feature/shape boundary, per-source coverage and exclusions, architecture,
engine identity, knobs and provisional baseline for approval. A legal cross-product
is not a measured shipping set; a genuinely single-candidate engine needs no extra
variants.

**Gate:** approved baseline and scope, with no unresolved semantic loss or reference
assumption.

## 3. Generate, implement and splice

Generate into an empty scratch directory, never over the live engine:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED"
```

**The verification root follows the configured dialect**, by the same rule stated
under **Descriptor placement** below: `direct_load` emits under `test_descriptors/`,
`packaged` under `descriptors/`. Point the verifier at the root your dialect actually
wrote — a root holding no `*.kdp.json` is a hard failure, so the wrong one fails every
time rather than reporting nothing to check. Set `EMITTED_ROOT` accordingly:

```bash
EMITTED_ROOT="$GENERATED/descriptors"        # packaged
# EMITTED_ROOT="$GENERATED/test_descriptors" # direct_load -- swap the two lines
"$PY" "$GEN/tools/verify_variant_sets.py" --mode structural \
  baseline "$EMITTED_ROOT"
```

Exactly one of those assignments must be live. Leaving both commented passes an empty
root, which resolves to the current directory instead of failing — the one way to reach
this gate without the wrong-root protection the paragraph above relies on.

Review the finalized inventory after deduplication. Resolve KDP → UED → KMD by UUID;
tuple identity includes schema types/defaults and effective architecture overlap.
Structural mode reports compiled agreement as `NOT CHECKED` and can exit 0 with
unrun checks. `--profile` optionally selects the bundle and supplies vocabulary;
it cannot supply compiler evidence. Full artifact checking belongs after packing.

Implement [native-pack.md](native-pack.md)'s referenced hooks and behavioral tests.
Extensions remap scratch references and copy only additions per [extend.md](extend.md),
including consumer IDs in shared KDP or per-UKD specialization declarations. Never
edit packed evidence to match a changed descriptor; rebuild from authored inputs.

Apply fragments to their actual consumers, preserving unrelated entries:

| Splice | Consumer | Required when |
|---|---|---|
| Engine `target_sources` | `$PROVIDER/src/engines/kernel_ingestor_engine/CMakeLists.txt` | Always — the native implementation |
| `IngestorPacks.hpp` declaration **and** `IngestorPacks.cpp`'s `s_packs` row | `$PROVIDER/src/engines/kernel_ingestor_engine/` | Always — both, or the pack vanishes from the static-archive binary |
| Engine test `target_sources` | `$PROVIDER/src/tests/engines/kernel_ingestor_engine/CMakeLists.txt` | Always — the applicable tests and any census suite |
| `add_kernels_for_embedding(TARGET … FILES … KEYS …)` | `$PROVIDER/src/tests/CMakeLists.txt` | Only `kernel_source.kind == "embedded_source"` — see [extend.md](extend.md) |
| `hkp_register_census_tests(TARGET … PACK_NAME … SUITES … EXPECTED_CASES …)` | `$PROVIDER/src/tests/CMakeLists.txt` | A census suite that reads exactly one pack target's shard |
| Descriptors themselves | — | **Never.** There is no descriptor splice |

**Descriptors need no CMake edit at all.** The packer walks a source root recursively
and no descriptor is ever named in CMake, so installing one is dropping files in the
right folder. Which folder is the whole mechanism — see **Descriptor placement**
below. A `cmake_descriptor_files.txt` fragment is a statement of that fact, not a
list to paste.

Census registration is one `hkp_register_census_tests()` call per packed target, made
in `$PROVIDER/src/tests/CMakeLists.txt` beside `hkp_verify_embedded_sources()`, after
the test target exists:

```cmake
hkp_register_census_tests(
    TARGET hip_kernel_provider_tests
    PACK_NAME unit
    SUITES TestPointwisePacks
    EXPECTED_CASES
        EachPackShipsThreeKernelsCoveringTwoBlockSizesAndTwoDataTypes
        EveryKernelNamesItsPacksEmbeddedSource
        EveryEmbeddedSourceKeyResolvesInTheCompiledInTable
        EveryPackNamesTheArchitectureItWasPackedFor
        EveryPackSharesTheEngineDispatchAndAllButOneMatcher
        ExposesBlockSizeAsAKnobAndDtypeAsInternal
        MatchersCoverBothScopes
        SubtractsInTheRightDirection
)
```

`PACK_NAME` selects the wired pack target whose own `OUT_ROOT` and recorded arch list
the entries address. A suite is declarable **only where it reads exactly one pack's
shard**, because an entry hands the binary a single directory and the native guard
requires every case to pass without skipping. `TestPointwisePacks` qualifies at the
`unit` target; `TestConvFwdPack` reads both the `unit` and `unit_shared` shards and is
censused nowhere. Declaring one suite at two pack targets is fatal — the entry name
carries only arch and suite, so the two would collide.

`EXPECTED_CASES` pins the suite's case-name set, and every registration carries one. For
a hand-written suite the list is maintained by hand: adding or removing a `TEST()`
without editing it is a red census, which is the point. CMake accepts a call without the
pin and registers the entries anyway — what that costs is under **Packaged census:
direct native CTest entries**.

**Descriptor placement.** The authored subpath decides everything; there is no list
to join.

| Bundle | Authored under | Reached through |
|---|---|---|
| Shipped | `$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/<producer>/<bundle>/` | `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, a `CACHE PATH` defaulting to that in-tree root |
| Test | `$PROVIDER/src/engines/kernel_ingestor_engine/test_descriptors/<set>/<slug>/` | `HIPKERNELPROVIDER_TEST_DESCRIPTOR_SOURCE_ROOT`, with `<set>` one of `shared`, `unit`, `integration`, `archive_fixture` |

`descriptors/` **ships**; `test_descriptors/` does **not** — it is staged into the
build tree and installed only under `HIPKERNELPROVIDER_ENABLE_TESTS`. Those two names
are the entire convention. Overriding the production cache variable is how a consumer
repoints the shipped root; neither root is ever repointed by adding CMake.

**Through the generator the root is a consequence of the dialect, not a free choice.**
`direct_load` emits under `test_descriptors/`, `packaged` under `descriptors/`, and a
packaged `authored_subpath` resolving outside `descriptors/` is refused at config load,
so no config can cross a bundle from one root to the other. Hand-authored bundles are
the exception: `integration/pointwise`, `archive_fixture/pointwise` and
`shared/conv_fwd` are packaged-dialect `hip` sets living under `test_descriptors/`, and
no generator config produces them. Author a generated bundle under the root its dialect
names; read an existing one's root as evidence of nothing until its dialect is checked.

Three packaging-time constraints, not conventions:

- **One level of nesting in every set.** Each set is packed by its top-level folder,
  so every descriptor lands in a *child* of its shard root while the archive is
  written at the shard root itself. That climb out of a child folder is what the
  runtime containment guard checks, and a set packed from its own leaf folder never
  produces it.
- **`archive_fixture` is a sibling of `integration`, not a child.** One source root
  cannot contain another: packing the parent sweeps the child's descriptors into the
  parent's archive. A set that must be able to fail on its own needs a top-level
  folder and an `OUT_ROOT` of its own.
- **No engine id in two dialects within one discovery root.** The two spellings
  collide on the completed metadata tuple and the collision removes that engine from
  the whole suite. `hipkernel:Pointwise` is authored twice for exactly this reason —
  `unit/pointwise/` in the `embedded_source` dialect, `integration/pointwise/` in the
  `hip` dialect — feeding two roots that never merge. The two sets are not a matched
  pair; edit the one whose binary reads it.

**There is no shared stage tree.** Each root packs straight to its own `OUT_ROOT`, so
"stage the descriptors" means "author them under the right root". Re-emit after every
regeneration and compare content and identities, not only counts. A fragment file is
not evidence that its splice was applied.

Set `SCHEMA` to the actual operation `.fbs` and `NATIVE_SOURCE` to its implementation;
pass all relevant source files and repeat for each schema in a fusion:

```bash
"$PY" "$GEN/tools/field_audit.py" "$SCHEMA" "$NATIVE_SOURCE"
```

Exit 0 covers lexical accessor references only; review semantic dispositions
separately. For an unspliced tree, check placeholders with:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --check-placeholders
```

Two roots cover **both** dialects, because `descriptors/` and `test_descriptors/` are
siblings under the engine directory and each root is searched at the engine-specific
relative path the bundle was emitted to. No separate staging root is needed:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" \
  --check-placeholders \
  --emitted-root "$PROVIDER/src/engines/kernel_ingestor_engine" \
  --emitted-root "$PROVIDER/src/tests/engines/kernel_ingestor_engine"
```

Add a third `--emitted-root` only when the production root was overridden away from
its in-tree default. With `--emitted-root`, `--output-dir` is required but not
searched. The check covers all emitted shippable files at engine-specific paths:
nonexistent roots, missing files and ambiguous matches fail, and a file found at the
same relative path under two roots is an error rather than a pick. An unrelated
same-basename file cannot satisfy it. It reports how many of the engine's shippable
files it located — read that count, because an unfilled-placeholder exit and a
could-not-locate exit are both `1`.
If the profile declares a launch-surface audit, also run:

```bash
"$PY" "$GEN/tools/launch_surface.py" "$PROFILE" --check
```

**Gate:** final authored inventory, completed hooks and source/test splices, no
selected-path placeholders, and reviewed structural/field/ABI results. None proves
native loading or numerical dispatch.

## 4. Build, pack, install and prove the host boundary

Configure from `$REPO` using `hipdnn-superbuild`, with
`CMAKE_INSTALL_PREFIX="$INSTALL"`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`,
`HIPKERNELPROVIDER_ENABLE_ROCKE=ON` and `HIPKERNELPROVIDER_ENABLE_TESTS=ON`. The
rocKE flag is mandatory, not conditional — see the coupling below. SDPA needs
`HIPDNN_ENABLE_SDPA=ON` consistently in SDK and provider.

The component selection must also actually include the provider. The
`hipdnn-providers` preset does **not** build hip-kernel-provider; the presets that do
are `hipdnn-providers-all`, `hip-kernel-provider`, `hipdnn-dev-all` and
`miopen-hipdnn-dev-all`. Configuring the wrong one leaves every step below with
nothing to observe.

There is **no per-producer production switch**: producer
selection is per-UKD on `kernel_source.kind`, so one root feeds every producer.

Production packaging is wired on exactly one condition — the root named by
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` holds at least one **non-hidden
`*.kdp.json`**. Standalone UKDs, kernel sources and READMEs do not by themselves make
a pack, because a KDP is what arch pruning consumes.

That root is `$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/`, and it holds
no bundle, so a default configure leaves production packaging dormant. **Supply your
own bundle under it** — or point the cache variable at a root that has one — before
expecting any of the production-pack steps below to produce output, and substitute your
bundle's own name wherever a bundle path appears below. `descriptors/README.md` carries
the authoring rules that root enforces, including the native pack whose symbols a
bundle's UKDs must name before it serves.

With no KDP under the root, production packaging is **dormant**, any stale product
tree from an earlier configure is removed, and neither is an error. A KDP that
*is* present but is pruned on every arch remains a **hard failure** for a root this
build NAMED, since naming a root asserts it ships here: the gate separates "nothing to
ship" from "something to ship that did not". The built-in default root goes **dormant**
in that case instead, so configuring for an arch your bundle does not declare is not a
build error. A root that is set but is not a directory is fatal at configure.

The packaging dependencies, documented from the repository root in
`dnn-providers/hip-kernel-provider/descriptor-packaging/README.md`, are still
required, and rocKE is resolved once for **every** root, test roots included, so an
unresolvable comgr is fatal at configure even in a hip-only build.

**`HIPKERNELPROVIDER_ENABLE_ROCKE=ON` is not a separate question — the coupling is
unconditional.** The provider's top-level `CMakeLists.txt` raises a `FATAL_ERROR`
whenever `HIPDNN_ENABLE_KERNEL_INGESTOR` is ON and `HIPKERNELPROVIDER_ENABLE_ROCKE`
is OFF. That condition inspects nothing else: not any UKD's `kernel_source.kind`, not
the production source root, not whether a single rocKE KDP exists anywhere. It
therefore fires for a HIP-only bundle, for an `embedded_source` bundle, and for a
default configure whose production root is dormant. Turning the ingestor on obliges
you to turn rocKE on, whatever you intend to pack.

Build the provider, validator and required test targets through the configured
superbuild. For packaged engines, run `hkp_packaging_product` after the full build
and after any reconfigure. Require the final staged descriptors, not merely an
up-to-date packaging stamp or an intermediate archive. Then install:

```bash
cmake --install "$BUILD" --prefix "$INSTALL"
```

**Install before provider-wide integration runs.** ASM SDPA loads loose `.co`
kernels from the configured install prefix by default; building its executable or
copying kernels into a build tree does not satisfy that path. Its explicit runtime
override is a separate test setup, not evidence of the final installation. Keep the
configured prefix aligned with `$INSTALL`, rather than relocating only this command.

Set `FINAL_DESCRIPTOR_ROOT` to the actual installed per-arch shard. Every root is
staged per architecture, `embedded_source` included: the packer stamps the shard
architecture onto a passthrough descriptor and records the authored values in its
provenance block, so there is no arch-independent installed tree to point at.
Resolve `VALIDATOR` to the built
`hipdnn_validate_descriptors` executable and validate the runtime dialect:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" --expect-engine "$ENGINE" --json
```

**The embedded-source invariant, and why its pass is not reachability.** A staged
tree holds descriptor JSON only — the packer copies no kernel source into it — so an
`embedded_source` descriptor resolves its `source_file` against a key table the build
compiles into the binary. `descriptor-packaging/tools/hkp_verify_embedded_sources.py`, wired by
`hkp_verify_embedded_sources()` beside the census registration, runs at build time
over emitted JSON alone and checks two things: every named `source_file` is a key of that
table (**presence**), and the file registered under that key is the file at the
authored location the descriptor's provenance records (**location**, joining the
`provenance.source_label` root with `rel_dir` and `source_file`). A separate
stamp-keyed rule requires a pack root whose stamp file is present to hold at least one
descriptor.

Its walk runs **one way only**, staged descriptor → key table, and neither reverse
direction is checked. Per its own docstring, *"a descriptor that never reaches a
staged root is not an error either. Authored under a folder no pack is wired to, it is
never staged, so this walk never sees it and passes while the runtime never receives
it."* A key the table holds that no descriptor names is likewise not an error, because
most embedded kernels have no descriptor at all.

**A green verification step is therefore NOT evidence the bundle is reachable.** The
check to state is *does a shard appear under that pack target's `OUT_ROOT`*, not *did
the verifier pass*. An absent root, an empty root, a root holding no `embedded_source`
descriptor and an absent key table each pass; a pass reports the two counts it
compared, so read those counts rather than the exit status. The dormant production
root contributes no stamp and is not checked at all.

For packaged output, check each selected architecture using the producing-build
record, not today's imported producer:

```bash
"$PY" "$GEN/tools/verify_variant_sets.py" --mode full --arch "$ARCH" \
  --profile "$PROFILE" final "$FINAL_DESCRIPTOR_ROOT"
```

Omit `--profile` when neither bundle selection nor extra vocabulary needs it. Add
`--kpack-python-dir <dir>` if required by the reader environment. Interpret outcomes
separately:

- Use full checking only for **packed `kind: kpack` descriptors**; confirm the input
  dialect instead of inferring it from exit 0. Missing declarations, mismatches
  and required checks `NOT RUN` block acceptance, including missing vocabulary.
- A packed kernel declaring no specialized `metadata_fields` and carrying no
  `effective_spec` reports **`NOT VERIFIED HERE`** — but **only when its
  `provenance.origin_kind` is absent or `hip`**. It then neither fails the gate nor
  gains compiled-specialization proof. Full-mode exit 0 does not certify those
  binaries; AOT HIP specialization remains outside this check.
- **The exemption does not extend to rocKE.** When that same
  no-`metadata_fields`-and-no-`effective_spec` condition holds and
  `provenance.origin_kind` is `rocke`, full verification records a **hard failure**:
  the packer publishes a rocKE kernel's compiler-owned `effective_spec` when it ships
  it, so the pair means the record was lost and the archive bytes were never read.
  Relabelling a rocKE kernel's specialized fields as matcher-only does not convert it
  into an unspecialized source; it fails the gate.
- For declared specialization, the per-kernel record binds the current descriptor,
  schema, metadata, architecture and named payload bytes. Generation supplies the
  declaration, not that compiler-owned evidence; see [rocke-mining.md](rocke-mining.md).
- **An `embedded_source` root legitimately produces descriptors and no archive.**
  `embedded_source` is a packaging *passthrough* kind: it is emitted exactly as
  authored, no producer runs for it, and it contributes no code object and no archive
  entry. A shard with no compiled variant therefore holds no `kpack/` directory, and
  that is not a packing failure. Compiled-specialization obligations are scoped to the
  compiling kinds they are defined for and stay mandatory for every one of those.
  "Descriptors but no archive" is legal; "no descriptors" never is.
- **Two independent artifact checks bind a packed kernel to its binary, not one.**
  `sha256` is byte identity of the *decompressed* code object, 64 lowercase hex, and
  `kernel_signature.py` records the argument list read back out of the object the
  packer just compiled. They fail on different drift: a TOC entry pointing at the
  wrong offset decompresses cleanly and hands back another entry's code object, which
  only the digest catches; a kernel whose parameters changed still hashes to whatever
  it now is, which only the signature catches. Neither is hand-authored. Argument
  *names* are producer-dependent — clang omits them for HIP `extern "C" __global__`
  kernels and the ASM producers carry them — so a missing `name` is not drift.

### Packaged census: direct native CTest entries

Run real provider registration/loading and inventory checks in fresh processes.
The census is a direct native obligation with no Python launcher and no XML guard.
It covers a suite that reads exactly **one pack target's shard**; the authored
dialect does not decide eligibility, the shard count does. `TestPointwisePacks` is
censused although `unit/pointwise/` is `embedded_source`, because every one of its
cases reads that single shard.

Registration is one
`hkp_register_census_tests(TARGET … PACK_NAME … SUITES … EXPECTED_CASES …)` call
per packed target. For each declared suite and each arch in that pack target's own
recorded list, CMake registers **four** independent tests, not one. The census entry
is `hip-kernel-provider-hkp-census-<arch>-<suite>`, which invokes

```text
hip_kernel_provider_tests --gtest_filter=<suite>.*
```

with `HIPDNN_TEST_CENSUS_SUITE=<suite>`, `HIPDNN_TEST_EXPECTED_ARCH=<arch>` and
`HIPDNN_DESCRIPTOR_DIR=<that pack target's OUT_ROOT>/<arch>` — **that target's own
output-root shard, not a shared stage tree** — labelled
`unit_test;hip-kernel-provider;host`. The other three carry that same name with
`-control-unvisited`, `-control-absent-root` and `-control-unregistered-case`
appended; the last is registered only where a pin exists. Each control breaks exactly
one precondition on purpose — no case is visited, the explicit descriptor root is a
shard name nothing can create, the pin names a case the suite never registers — and
each passes on the census's own refusal wording rather than on exit status, so a red
control means the refusal it watches for has stopped happening. The entry alone is one
test of four and says nothing about whether the gate is still live; run the family.
The architecture comes from the arch list the
pack target was wired with, never from a detected device or from the descriptors
themselves. Set `CENSUS_SUITE` to the generated suite name and `PROVIDER_BUILD` to
the provider's own binary directory (`$BUILD/dnn-providers/hip-kernel-provider` in
the superbuild layout), then run every requested arch's entry and its controls:

```bash
ctest --test-dir "$PROVIDER_BUILD" --no-tests=error -V \
  -R "^hip-kernel-provider-hkp-census-${ARCH}-${CENSUS_SUITE}(-control-.*)?$"
```

Those entries bind the build-tree shard. For **final installed packaged evidence**,
run the same suite against the installed shard by supplying the same explicit
environment to the installed binary — including the pin, which the CTest entries get
from `EXPECTED_CASES` and a hand-run invocation does not:

```bash
HIPDNN_TEST_CENSUS_SUITE="$CENSUS_SUITE" \
HIPDNN_TEST_EXPECTED_ARCH="$ARCH" \
HIPDNN_TEST_CENSUS_EXPECTED_CASES="$EXPECTED_CASES" \
HIPDNN_DESCRIPTOR_DIR="$FINAL_DESCRIPTOR_ROOT" \
"$INSTALL/bin/hip_kernel_provider_tests" --gtest_filter="${CENSUS_SUITE}.*"
```

Set `EXPECTED_CASES` to the same reviewed comma-separated case-name list the build-tree
registration pins — the one `hkp_register_census_tests()` carries into
`HIPDNN_TEST_CENSUS_EXPECTED_CASES`. **Do not derive it from the installed binary under
test**: a list read back out of that binary agrees with it by construction and pins
nothing. Omitting it is silent, not an error — the listener returns early when the
expected list is empty, so the expected-vs-registered comparison never runs and the
census still reports complete. Adjust the binary path for a nondefault install bindir.
A nonempty
`HIPDNN_TEST_CENSUS_SUITE` activates the native strict guard: before default-root
setup it rejects an empty expected arch and a missing, empty or nonexistent explicit
descriptor root, and it rejects an absent or empty named suite. Every registered
case in that suite must execute and pass **without skipping in every iteration**,
with at least one completed iteration. Disabled, filtered-out, sharded-out, failed
or skipped cases, list-only invocations and zero iterations cannot satisfy it, and
repeated partial runs do not accumulate coverage. Normal invocations without the
variable keep ordinary GoogleTest filtering and skip behavior, and the production
runtime's descriptor-root fallback is unchanged.

`EXPECTED_CASES` reaches the binary as `HIPDNN_TEST_CENSUS_EXPECTED_CASES`,
comma-separated, and pins the suite's case-name set. CMake accepts a call without it and
registers the entries anyway: the pin is optional to configure and required for the
census to mean what it claims. The execution half draws its obligations from the cases
the suite itself registered, so a case that stops being compiled — commented out, or in
a source dropped from `target_sources` — takes its own obligation with it and the census
still reports complete. The pin is the half that notices, compared by name in both
directions, because a lost case and a new one cancel in a count and call for opposite
remedies. An unpinned call also drops `-control-unregistered-case`, the entry that
proves the comparison is live, so the missing pin is invisible exactly as the lost case
is.

A missing prerequisite the configuration could not have chosen deliberately is
**fatal at configure**, because a census that registers nothing is indistinguishable
from one that passed: a `PACK_NAME` no `hkp_wire_pack_target()` call wired and no
dormancy accounts for (the message names the wired roots and the dormant ones
separately), a `TARGET` that does not exist or was not given, and a recorded arch list
that is empty. A `PACK_NAME` the registry records as **dormant** is the deliberate
case and the one exception: the call registers nothing and reports at `STATUS`, naming
the suites it left unregistered, so a generated integration's census call stays valid
in a configuration that packs no product root. That is a drop, never a silent one.
Declaring one suite at two pack targets is fatal for a different reason — the entry
name carries arch and suite alone, so the second registration would silently take the
first one's shard.

Tests built OFF, an empty `SUITES`, and a dormant `PACK_NAME` all register nothing at
all: that is **absence of census evidence**, not a pass. A suite that reads more than one shard
cannot be censused and must state its inventory through its ordinary host suite
instead; that suite invoked directly still requires an explicit expected arch and
descriptor root. Check retained extension inventory and heuristic-disabled score
absence per [native-pack.md](native-pack.md).

**Gate:** current installation, artifact checks at their stated strength, and real
registration/loading plus applicable inventory/census checks. Report `NOT VERIFIED
HERE` separately. Neither the structural validator's stubs nor host loading proves
dispatch.

## 5. Baseline device proof from the installation

On the allocated target host:

```bash
"$PY" "$GEN/tools/device_probe.py" --mode installed --arch "$ARCH" \
  --sweep-root "$SWEEP_ROOT" --install "$INSTALL"
```

A missing/invisible installation fails even if early feasibility passed. Use
`hipdnn-superbuild-test` discovery with component **`hip-kernel`**:

```bash
"$PY" "$REPO/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts/discover_test_targets.py" \
  --build-dir "$BUILD" --component hip-kernel --scope external-integration
```

Do not accept the helper's first provider-prefixed command as exact-engine proof.
The provider's default installed CTest root is **`$INSTALL/bin/hip_kernel_provider`**,
not `$INSTALL`; substitute the configured bindir if customized.

The provider does register `hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests`,
which is the ASM SDPA engine and **not** ingestor evidence. A passing ASM SDPA run
proves nothing about whether your ingested engine dispatches; it is a different
engine reached by a different path. Never substitute it for your bundle's own
registration.

The production descriptor root ships no bundle, so no dense-attention target is
registered here; the `<your-bundle-ctest-target>` placeholder below must be
replaced with your own bundle's CTest target before the block will run at all (a
gfx942 dense bundle would name something shaped like
`hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests`, which
exists nowhere in this tree):

```bash
CTEST_ROOT="$INSTALL/bin/hip_kernel_provider"
DEVICE_TEST=<your-bundle-ctest-target>
ctest --test-dir "$CTEST_ROOT" -N -V -R "^${DEVICE_TEST}$"
```

Require exactly your bundle's registration. Inspect its command/config for that
bundle's engine ID — `hipkernel:Gfx942AttentionDense` would be the illustration's —
installed executable/plugin/config paths and the intended quick/standard
selection. Then execute:

```bash
ctest --test-dir "$CTEST_ROOT" --no-tests=error -V -R "^${DEVICE_TEST}$"
```

Keep verbose output in the retained log. **`--output-on-failure` hides passing
suites' case counts; an all-skip suite can still report CTest PASS.** Record selected,
served, skipped/declined and failed counts and observed reasons. Missing registration,
zero selected, all-skipped support, wrong engine/path or numerical failure blocks
this gate. Resolve the exact UED name → engine ID from the installation; a prefix
or registry listing is not dispatch attribution.

Use nontrivial inputs for quick feature breadth and bounded standard numerical
depth. Exercise required declines separately; another winning engine must not hide
them. NaN/unwritten output is a failure, not a tolerance adjustment.

Extensions must select the addition explicitly. The disposable pointwise example
adds HALF/block_size=256 to ADD, preserves MUL/SUB and changes ADD's expected census
from three to four. Select HALF/256 on logical dims `{1,1,1,1}`, check the actual
`hipkernel:Pointwise` plan and arithmetic, and retain old ADD/MUL/SUB and required
multi-element/two-node declines. Its source computes one element; a default-FLOAT
pass or this one-element smoke proves no arbitrary-size coverage.

**Gate:** intended-engine dispatch, capable-reference numerics and complete case
accounting on `$ARCH`.

## 6. Tune the runnable baseline and rebuild the final selection

For rocKE, propose bounded candidates with the actual profile:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" --plan
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --isolate --out-dir "$ARM_CONFIG_ROOT"
```

Set `ARM_CONFIG_ROOT` to an experiment-owned directory. Generate, implement/splice,
build, pack, install and check every supported arm before measurement. Keep separate
install/output trees; never mutate the baseline and call it a comparison.

Populate `configs/sweep-isolation.sweep.yaml.example` using actual corpus counts,
installed KDP-entry counts and proven engine identity. YAML paths resolve relative
to the YAML file; there is no shell/environment interpolation. Set `SWEEP_CONFIG`
to its absolute path, then run:

```bash
"$PY" "$GEN/tools/sweep.py" --config "$SWEEP_CONFIG"
```

[workloads.md](workloads.md)'s *Installed measurement contract* owns every rule this
run must satisfy — session and arm ordering, warmup, rounds, cache isolation,
separate correctness, and the sweep's ownership of the benchmark's `--engine` and its
other phase-owned arguments. Satisfy it there; this page owns only the commands and
their order.

For rocKE, investigate measured survivors. Set `PAIRWISE_KNOBS` to comma-separated
surviving knob names and `PAIRWISE_CONFIG_ROOT` to their config output directory,
then generate, build and install the supported pairwise arms:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --pairwise "$PAIRWISE_KNOBS" --out-dir "$PAIRWISE_CONFIG_ROOT"
```

Obtain selection approval from coverage, correctness and per-corpus measurements.
For the rocKE shipping cross, `APPROVED_KNOBS_JSON` contains the approved JSON object
of knob value lists, **not a filename**:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --knobs "$APPROVED_KNOBS_JSON" --out "$FINAL_CONFIG"
```

Repeat **stages 3–5** with the final config and a new empty generation destination.
Neither an isolation arm nor an old install certifies the regenerated shipping set.
An explicitly untuned extension may retain its approved baseline selection, but
still requires final installed artifact and corpus proof.

**Gate:** justified selection and revalidated final installation.

## 7. Final corpus proof and runtime reconciliation

Run the final installed artifact through a fresh-output YAML sweep with
`correctness.enabled: true`, and require the exact phase key set.
[workloads.md](workloads.md)'s *Installed measurement contract* owns what each
terminal status means — `SWEEP_DONE`, `SWEEP_TIMING_ONLY`, `SWEEP_INCOMPLETE` — and
what resume does and does not make a single cohort; read the verdict there.

Harvest final phase results and available engine logs into the per-input outcome
ledger, then join within each corpus/phase before making its graph-name-to-reason
JSON. [workloads.md](workloads.md)'s *Complete final runtime join* owns the ledger
fields, the join scope and every rejection rule; satisfy it there rather than from a
restatement here. This is an explicit evidence review, not a promised automatic
matcher-reason extractor.

For rocKE, set `CORPUS_SHAPES` and `RUNTIME_DECLINES` to that corpus's requests and
complete runtime-derived mapping, then reconcile:

```bash
"$PY" "$GEN/tools/reconcile_applicability.py" --profile "$PROFILE" \
  --shapes "$CORPUS_SHAPES" --declines "$RUNTIME_DECLINES"
```

Without the complete join, label reconciliation **offline only**. Investigate
reference-only supported rows and obtain explicit scope decisions for exclusions;
escape flags cannot excuse broken reference APIs or unexplained gaps. Direct-load
pointwise instead uses its explicit one-element corpus, installed engine results
and independent arithmetic/reference correctness, with `exclude_tensors: none`.

Report exactly the populations and statistics [workloads.md](workloads.md)'s
*Required reporting statistics* requires; that page owns the reporting list, and
timing is reported separately from the outcome accounting.

**Gate:** zero wrong answers, complete final-runtime accounting, and no missing,
ambiguous, erroneous or unexplained in-scope outcomes. Changed installed artifacts
invalidate old evidence and return to stages 3–5.

## 8. Handoff

Report [SKILL.md](SKILL.md)'s completion evidence and exact limitations. Keep
experiment copies/probes disposable and retain their inputs/results under the
workspace evidence policy. For blocked work, name the last completed stage and
missing prerequisite; do not substitute a proposed command or queued job for proof.
