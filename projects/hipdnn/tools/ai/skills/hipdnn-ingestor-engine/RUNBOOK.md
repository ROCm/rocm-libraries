# Runbook: create or extend an ingestor engine

The **only ordered create/extend workflow**. [SKILL.md](SKILL.md) holds entry and
completion requirements; the linked domain pages hold the contracts. Every gate needs
current evidence — an old install or a copied command is not an observation.

## Paths and interpreters

Resolve absolute paths before running anything. These are explicit arguments, not
implicit tool inputs:

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

Replace `<your-bundle-engine-id>` with your bundle's engine ID; it is consumed verbatim
as `--expect-engine` below. A gfx942 dense attention bundle would spell it
`hipkernel:Gfx942AttentionDense`; this tree ships no such engine.

Follow the **Setup** section of `$GEN/README.md`. Authoring and mining imports need the
profile's rocKE library environment; production packaging uses its own selected
compiler/wheel interpreter. Full artifact checking also needs `rocm_kpack` and its
dependencies (`zstandard`, `msgpack`) importable from `$PY`; `--kpack-python-dir`
supplies an import root, not missing dependencies.

Verify every path on the actual execution host. Host-local backing paths and symlink
targets are not automatically visible on a different host, even when the link itself
is on shared storage. On a cluster with shared login and compute nodes, do not run GPU
work on a shared login node: use shared source, or stage the exact checkout and
artifacts to the execution host through your scheduler or remote-execution tooling.
Record commands and source/artifact, machine and device identities, plus allocation
identity when applicable. Follow a configured local evidence policy when one exists;
otherwise retain logs, manifests and outcomes in a user-selected per-run evidence
directory outside product source.

## 1. Entry and early feasibility

Record the entry contract; for extend, inventory the installed baseline per
[extend.md](extend.md). Select `direct_load` or `packaged` before staging files.
Establish representability and a capable independent numerical reference using
[graph-contract.md](graph-contract.md); a reference's skip is not verification, and
unavailable semantics require an explicit scope/reference decision.

On the actual execution host:

```bash
"$PY" "$GEN/tools/device_probe.py" \
  --mode early --arch "$ARCH" --sweep-root "$SWEEP_ROOT"
```

Early mode requires the requested device and an existing writable root. It ignores
inherited `INSTALL` and rejects `--install`, so `$INSTALL` may name a future directory.
Exits: 0 feasible, 1 device/path/write failure, 2 invalid invocation, 3 neither
`rocminfo` nor `hipInfo` could run. Exit 3 means the device was never observed, not that
it is absent: obtain an inspection utility on this host. For rocKE, confirm the actual
builder/spec and its `(spec, *, arch)` interface; an unknown architecture inventory
needs source investigation.

**Gate:** feasible target/workspace, representable scope and capable reference. A
missing dependency blocks its gate; host-only research may continue while a device
allocation is pending but cannot discharge device proof.

## 2. Contracts, corpus and baseline approval

Record [graph-contract.md](graph-contract.md)'s topology/UID edges and field
dispositions, then [rocke-mining.md](rocke-mining.md)'s applicability, specialization,
layout, geometry/workspace and ABI evidence. Direct-load engines take these facts from
HIP source without inventing a profile. Reuse unchanged extension contracts; reopen any
topology or feature the addition changes.

Inventory owner-published results, owner benchmark shapes and external graphs per
[workloads.md](workloads.md). `$SHAPES` is a JSON list of semantic requests; the
benchmark consumes actual graph JSON under `$CORPUS_DIR/<corpus>`. For the attention
miner:

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

The second command is **offline applicability**, not runtime coverage or numerics. Scope
reference candidates to the kernel family/algorithm and required opt-in selector. API
failures are operational errors, never unsupported-shape evidence.

Present the feature/shape boundary, per-source coverage and exclusions, architecture,
engine identity, knobs and provisional baseline for approval. A legal cross-product is
not a measured shipping set; a single-candidate engine needs no extra variants.

**Gate:** approved baseline and scope, with no unresolved semantic loss or reference
assumption.

## 3. Generate, implement and splice

Generate into an empty scratch directory, never over the live engine:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED"
```

Run both against the revision you are about to splice, after the last edit to `$CONFIG`
and anything it draws on; a render from an earlier revision is not evidence for this one.
The config loader rejects a YAML mapping that repeats a key — including a repeated `<<` —
and names the key with its source position. Anchors, aliases and explicit overrides of
merged keys remain valid. Fix the authored file; there is no later-wins fallback.

When the engine already ships a catalog — an extension, or a regeneration of a shipped
config — compare the scratch render with that catalog **before splicing**. The target is
the live authored root (see **Descriptor placement**): for a packaged engine, its
authored source root `descriptors/<producer>/<bundle>/`, never the packed tree, whose
`kernel_source` is lowered to kpack; for `direct_load`, its live
`test_descriptors/<set>/<slug>/` root. Compare semantics, not bytes: resolve
KDP → UED → KMD by UUID within each tree separately, then
compare descriptor names, completed metadata (KMD defaults and types filled in),
`kernel_source` and the families each tree carries. Every difference is either an
intended, reviewed change or a defect in the authored input; an unexplained difference
blocks the splice. Minted UUIDs differ on every run and are the only values normalized
away. They stay in scratch: never copy them over live descriptors, native sources or
tests. [extend.md](extend.md)'s *Addition-only splicing* owns how scratch references map
onto retained IDs.

`verify_variant_sets.py` (below, and in full mode in stage 4) checks its own documented
rungs on one tree. It supplements this comparison and does not replace it: it never
compares the authored render with what ships.

The verification root follows the configured dialect (see **Descriptor placement**):
`direct_load` emits under `test_descriptors/`, `packaged` under `descriptors/`. A root
holding no `*.kdp.json` is a hard failure, so the wrong root fails immediately:

```bash
EMITTED_ROOT="$GENERATED/descriptors"        # packaged
# EMITTED_ROOT="$GENERATED/test_descriptors" # direct_load -- swap the two lines
"$PY" "$GEN/tools/verify_variant_sets.py" --mode structural \
  baseline "$EMITTED_ROOT"
```

Exactly one assignment must be live. Both commented passes an empty root, which the
descriptor index expands as the recursive glob `/**/*.json` — a walk of the whole
filesystem.

Review the finalized inventory after deduplication. Resolve KDP → UED → KMD by UUID;
tuple identity includes schema types/defaults and effective architecture overlap.
Structural mode reports compiled agreement as `NOT CHECKED` and can exit 0 with unrun
checks. `--profile` optionally selects the bundle and supplies vocabulary, never
compiler evidence. Full artifact checking belongs after packing.

Implement [native-pack.md](native-pack.md)'s referenced hooks and behavioral tests.
Extensions remap scratch references and copy only additions per [extend.md](extend.md),
including consumer IDs in shared KDP or per-UKD specialization declarations. Never edit
packed evidence to match a changed descriptor; rebuild from authored inputs.

Apply fragments to their actual consumers, preserving unrelated entries:

| Splice | Consumer | Required when |
|---|---|---|
| Engine `target_sources` | `$PROVIDER/src/engines/kernel_ingestor_engine/CMakeLists.txt` | Always — the native implementation |
| `IngestorPacks.hpp` declaration **and** `IngestorPacks.cpp`'s `s_packs` row | `$PROVIDER/src/engines/kernel_ingestor_engine/` | Always — both, or the pack vanishes from the static-archive binary |
| Engine test `target_sources` | `$PROVIDER/src/tests/engines/kernel_ingestor_engine/CMakeLists.txt` | Always — the applicable tests and any census suite |
| `add_kernels_for_embedding(TARGET … FILES … KEYS …)` | `$PROVIDER/src/tests/CMakeLists.txt` | Only `kernel_source.kind == "embedded_source"` — see [extend.md](extend.md) |
| `hkp_register_census_tests(TARGET … PACK_NAME … SUITES … EXPECTED_CASES …)` | `$PROVIDER/src/tests/CMakeLists.txt` | A census suite that reads exactly one pack target's shard |
| Descriptors themselves | — | **Never.** There is no descriptor splice |

**Descriptors need no CMake edit.** The packer walks a source root recursively and no
descriptor is named in CMake, so installing one is dropping files under the right root.
A `cmake_descriptor_files.txt` fragment states that fact; it is not a list to paste.

Census registration is one `hkp_register_census_tests()` call per packed target, in
`$PROVIDER/src/tests/CMakeLists.txt` beside `hkp_verify_embedded_sources()`, after the
test target exists:

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

`PACK_NAME` selects the wired pack target whose `OUT_ROOT` and recorded arch list the
entries address. A suite is declarable **only where it reads exactly one pack's shard**:
`TestPointwisePacks` qualifies at the `unit` target, while `TestConvFwdPack` reads both
the `unit` and `unit_shared` shards and is censused nowhere. `EXPECTED_CASES` pins the
suite's case-name set and is hand-maintained for a hand-written suite: adding or
removing a `TEST()` without editing it is a red census. See **Packaged census: direct
native CTest entries** for what an unpinned call forfeits.

**Descriptor placement.** The authored subpath decides everything; there is no list to
join.

| Bundle | Authored under | Reached through |
|---|---|---|
| Shipped | `$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/<producer>/<bundle>/` | `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, a `CACHE PATH` defaulting to that in-tree root |
| Test | `$PROVIDER/src/engines/kernel_ingestor_engine/test_descriptors/<set>/<slug>/` | `HIPKERNELPROVIDER_TEST_DESCRIPTOR_SOURCE_ROOT`, with `<set>` one of `shared`, `unit`, `integration`, `archive_fixture` |

`descriptors/` ships; `test_descriptors/` is staged into the build tree and installed
only under `HIPKERNELPROVIDER_ENABLE_TESTS`. Overriding the production cache variable is
how a consumer repoints the shipped root; neither root is repointed by adding CMake.

Through the generator the root follows the dialect, and a packaged `authored_subpath`
resolving outside `descriptors/` is refused at config load. Hand-authored bundles are
the exception: `integration/pointwise`, `archive_fixture/pointwise` and
`shared/conv_fwd` are packaged-dialect `hip` sets under `test_descriptors/` that no
generator config produces. Author a generated bundle under the root its dialect names;
an existing bundle's root proves nothing until its dialect is checked.

Three packaging-time constraints:

- **One level of nesting in every set.** A set is packed by its top-level folder, so
  every descriptor lands in a *child* of its shard root while the archive is written at
  the shard root. The runtime containment guard checks that climb out of a child folder.
- **`archive_fixture` is a sibling of `integration`, not a child.** Packing a parent
  root sweeps a nested child's descriptors into the parent's archive; a set that must be
  able to fail on its own needs a top-level folder and its own `OUT_ROOT`.
- **No engine id in two dialects within one discovery root.** The spellings collide on
  the completed metadata tuple and remove that engine from the whole suite.
  `hipkernel:Pointwise` is authored twice for this reason — `unit/pointwise/` in the
  `embedded_source` dialect, `integration/pointwise/` in the `hip` dialect — feeding two
  roots that never merge. They are not a matched pair; edit the one whose binary reads
  it.

**There is no shared stage tree.** Each root packs straight to its own `OUT_ROOT`, so
"stage the descriptors" means "author them under the right root". Re-emit after every
regeneration and compare content and identities, not only counts. A fragment file is not
evidence that its splice was applied.

Set `SCHEMA` to the operation `.fbs` and `NATIVE_SOURCE` to its implementation; pass all
relevant source files and repeat for each schema in a fusion:

```bash
"$PY" "$GEN/tools/field_audit.py" "$SCHEMA" "$NATIVE_SOURCE"
```

Exit 0 covers lexical accessor references only; review semantic dispositions separately.
For an unspliced tree, check placeholders with:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --check-placeholders
```

Two roots cover **both** dialects, each searched at the engine-specific relative path
the bundle was emitted to:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" \
  --check-placeholders \
  --emitted-root "$PROVIDER/src/engines/kernel_ingestor_engine" \
  --emitted-root "$PROVIDER/src/tests/engines/kernel_ingestor_engine"
```

Add a third `--emitted-root` only when the production root was overridden away from its
in-tree default. With `--emitted-root`, `--output-dir` is required but not searched. The
check covers all emitted shippable files at engine-specific paths: nonexistent roots,
missing files, ambiguous matches and a file found at the same relative path under two
roots all fail, and an unrelated same-basename file cannot satisfy it. Read the reported
count of located shippable files — an unfilled-placeholder exit and a could-not-locate
exit are both `1`.

If the profile declares a launch-surface audit, also run:

```bash
"$PY" "$GEN/tools/launch_surface.py" "$PROFILE" --check
```

**Gate:** a current-revision dry-run and render, every difference from any shipped
catalog explained, final authored inventory, completed hooks and source/test splices, no
selected-path placeholders, and reviewed structural/field/ABI results. None proves
native loading or numerical dispatch.

## 4. Build, pack, install and prove the host boundary

Configure from `$REPO` using `hipdnn-superbuild`, with
`CMAKE_INSTALL_PREFIX="$INSTALL"`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON`,
`HIPKERNELPROVIDER_ENABLE_ROCKE=ON` and `HIPKERNELPROVIDER_ENABLE_TESTS=ON`. SDPA needs
`HIPDNN_ENABLE_SDPA=ON` consistently in SDK and provider.

The component selection must include the provider. The `hipdnn-providers` preset does
**not** build hip-kernel-provider; the presets that do are `hipdnn-providers-all`,
`hip-kernel-provider`, `hipdnn-dev-all` and `miopen-hipdnn-dev-all`.

**`HIPKERNELPROVIDER_ENABLE_ROCKE=ON` is unconditional.** The provider's top-level
`CMakeLists.txt` raises `FATAL_ERROR` whenever `HIPDNN_ENABLE_KERNEL_INGESTOR` is ON and
it is OFF, inspecting no `kernel_source.kind`, no production root and no descriptor, so
it fires for HIP-only and `embedded_source` bundles and for a dormant default configure.
rocKE is likewise resolved once for **every** root, test roots included, so an
unresolvable comgr is fatal at configure even in a hip-only build.

There is **no per-producer production switch**: producer selection is per-UKD on
`kernel_source.kind`, so one root feeds every producer.

Production packaging is wired on exactly one condition — the root named by
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` holds at least one **non-hidden
`*.kdp.json`**. Standalone UKDs, kernel sources and READMEs do not make a pack, because
a KDP is what arch pruning consumes. Outcomes:

| Condition | Outcome |
|---|---|
| No KDP under the root | Dormant; any stale product tree is removed. Not an error |
| KDP present, pruned on every arch, root explicitly named by this build | **Hard failure** — naming a root asserts it ships here |
| KDP present, pruned on every arch, built-in default root | Dormant, so configuring for an undeclared arch is not a build error |
| Root set but not a directory | Fatal at configure |

That default root is `$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/` and
holds no bundle, so a default configure leaves production packaging dormant. Supply your
own bundle under it — or repoint the cache variable — before expecting output, and
substitute your bundle's name wherever a bundle path appears below.
`descriptors/README.md` carries the authoring rules that root enforces, including the
native pack whose symbols a bundle's UKDs must name before it serves. The packaging
dependencies are documented from the repository root in
`dnn-providers/hip-kernel-provider/descriptor-packaging/README.md`.

Build the provider, validator and required test targets through the configured
superbuild. For packaged engines, run `hkp_packaging_product` after the full build and
after any reconfigure; require the final staged descriptors, not a packaging stamp or an
intermediate archive. Then install:

```bash
cmake --install "$BUILD" --prefix "$INSTALL"
```

**Install before provider-wide integration runs.** ASM SDPA loads loose `.co` kernels
from the configured install prefix by default; building its executable, copying kernels
into a build tree, or using its runtime override does not satisfy that path. Keep the
configured prefix aligned with `$INSTALL`.

Set `FINAL_DESCRIPTOR_ROOT` to the installed per-arch shard. Every root is staged per
architecture, `embedded_source` included: the packer stamps the shard architecture onto
a passthrough descriptor and records the authored values in its provenance block, so
there is no arch-independent installed tree. Resolve `VALIDATOR` to the built
`hipdnn_validate_descriptors` executable and validate the runtime dialect:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" --expect-engine "$ENGINE" --json
```

**The embedded-source invariant.** A staged tree holds descriptor JSON only, so an
`embedded_source` descriptor resolves its `source_file` against a key table compiled
into the binary. `descriptor-packaging/tools/hkp_verify_embedded_sources.py`, wired by
`hkp_verify_embedded_sources()` beside the census registration, runs at build time over
emitted JSON alone and checks **presence** (every named `source_file` is a key of that
table) and **location** (the file under that key is the file at the authored location,
joining `provenance.source_label` with `rel_dir` and `source_file`). A separate
stamp-keyed rule requires a pack root whose stamp file is present to hold at least one
descriptor.

The walk runs **one way only**, staged descriptor → key table: a descriptor authored
under a folder no pack is wired to is never staged and passes unseen, and a key no
descriptor names is not an error either. **A green verification is therefore not
reachability evidence.** State whether a shard appeared under that pack target's
`OUT_ROOT`; an absent root, an empty root, a root with no `embedded_source` descriptor
and an absent key table each pass. Read the two counts a pass reports. The dormant
production root contributes no stamp and is not checked.

For packaged output, check each selected architecture using the producing-build record,
not today's imported producer:

```bash
"$PY" "$GEN/tools/verify_variant_sets.py" --mode full --arch "$ARCH" \
  --profile "$PROFILE" final "$FINAL_DESCRIPTOR_ROOT"
```

Omit `--profile` when neither bundle selection nor extra vocabulary needs it. Add
`--kpack-python-dir <dir>` if the reader environment requires it. Interpret outcomes
separately:

- Use full checking only for **packed `kind: kpack` descriptors**; confirm the input
  dialect rather than inferring it from exit 0. Missing declarations, mismatches and
  required checks `NOT RUN` block acceptance, including missing vocabulary.
- A packed kernel declaring no specialized `metadata_fields` and carrying no
  `effective_spec` reports **`NOT VERIFIED HERE`** when `provenance.origin_kind` is
  absent or `hip`: no gate failure and no compiled-specialization proof. AOT HIP
  specialization stays outside this check.
- **The exemption does not extend to rocKE.** The same condition with
  `provenance.origin_kind` of `rocke` is a **hard failure**: the packer publishes a
  rocKE kernel's `effective_spec` when it ships it, so the pair means the archive bytes
  were never read. Relabelling its specialized fields as matcher-only does not make it
  an unspecialized source.
- For declared specialization, the per-kernel record binds the current descriptor,
  schema, metadata, architecture and named payload bytes. Generation supplies the
  declaration, not that compiler-owned evidence; see [rocke-mining.md](rocke-mining.md).
- **An `embedded_source` root legitimately produces descriptors and no archive.** It is
  a passthrough kind: emitted as authored, no producer, no code object, no archive
  entry, so its shard holds no `kpack/` directory. Compiled-specialization obligations
  stay mandatory for every compiling kind. "Descriptors but no archive" is legal; "no
  descriptors" never is.
- **Two independent artifact checks bind a packed kernel to its binary.** `sha256` is
  byte identity of the *decompressed* code object, 64 lowercase hex;
  `kernel_signature.py` records the argument list read back out of the compiled object.
  They catch different drift: a TOC entry at the wrong offset decompresses cleanly and
  returns another entry's code object (digest only); changed parameters hash to whatever
  they now are (signature only). Neither is hand-authored. Argument *names* are
  producer-dependent — clang omits them for HIP `extern "C" __global__` kernels, the ASM
  producers carry them — so a missing `name` is not drift.

### Packaged census: direct native CTest entries

Run real provider registration/loading and inventory checks in fresh processes. The
census is a direct native obligation with no Python launcher and no XML guard. Shard
count decides eligibility, not the authored dialect: `TestPointwisePacks` is censused
although `unit/pointwise/` is `embedded_source`.

For each declared suite and each arch in the pack target's recorded list, CMake
registers **four** tests. The census entry is
`hip-kernel-provider-hkp-census-<arch>-<suite>`, which invokes

```text
hip_kernel_provider_tests --gtest_filter=<suite>.*
```

with `HIPDNN_TEST_CENSUS_SUITE` set to that suite, `HIPDNN_TEST_EXPECTED_ARCH` to that
arch — taken from the wired arch list, never from a detected device or the descriptors —
and `HIPDNN_DESCRIPTOR_DIR` to that pack target's own `OUT_ROOT` shard for the arch,
never a shared stage tree; labelled `unit_test;hip-kernel-provider;host`. The other
three append `-control-unvisited`, `-control-absent-root` and
`-control-unregistered-case`, the last only where a pin exists. Each control breaks one
precondition deliberately and passes on the census's own refusal wording rather than on
exit status, so a red control means that refusal stopped happening. Run the whole
family.

Set `CENSUS_SUITE` to the generated suite name and `PROVIDER_BUILD` to the provider's
binary directory (`$BUILD/dnn-providers/hip-kernel-provider` in the superbuild layout),
then run every requested arch's entry and its controls:

```bash
ctest --test-dir "$PROVIDER_BUILD" --no-tests=error -V \
  -R "^hip-kernel-provider-hkp-census-${ARCH}-${CENSUS_SUITE}(-control-.*)?$"
```

Those entries bind the build-tree shard. For **final installed packaged evidence**, run
the same suite against the installed shard with the same explicit environment, including
the pin that the CTest entries get from `EXPECTED_CASES`:

```bash
HIPDNN_TEST_CENSUS_SUITE="$CENSUS_SUITE" \
HIPDNN_TEST_EXPECTED_ARCH="$ARCH" \
HIPDNN_TEST_CENSUS_EXPECTED_CASES="$EXPECTED_CASES" \
HIPDNN_DESCRIPTOR_DIR="$FINAL_DESCRIPTOR_ROOT" \
"$INSTALL/bin/hip_kernel_provider_tests" --gtest_filter="${CENSUS_SUITE}.*"
```

Set `EXPECTED_CASES` to the same reviewed comma-separated case-name list the build-tree
registration pins. **Do not derive it from the installed binary under test**: such a
list agrees by construction and pins nothing. Omitting it is silent — the listener
returns early on an empty expected list and the census still reports complete. Adjust
the binary path for a nondefault install bindir.

A nonempty `HIPDNN_TEST_CENSUS_SUITE` activates the native strict guard: before
default-root setup it rejects an empty expected arch, a missing/empty/nonexistent
explicit descriptor root, and an absent or empty named suite. Every registered case in
that suite must execute and pass **without skipping in every iteration**, with at least
one completed iteration. Disabled, filtered-out, sharded-out, failed or skipped cases,
list-only invocations and zero iterations cannot satisfy it, and repeated partial runs
do not accumulate coverage. Invocations without the variable keep ordinary GoogleTest
filtering and skip behavior, and the production runtime's descriptor-root fallback is
unchanged.

The pin is optional to configure and required for the census to mean what it claims: the
execution half draws its obligations from the cases the suite registered, so a case that
stops being compiled takes its obligation with it. The pin is compared by name in both
directions, and an unpinned call also drops `-control-unregistered-case`.

Fatal at configure, because a census that registers nothing looks like one that passed:
a `PACK_NAME` no `hkp_wire_pack_target()` call wired and no dormancy accounts for (the
message names wired and dormant roots separately); a `TARGET` missing or not given; an
empty recorded arch list; and one suite declared at two pack targets, whose entry names
carry arch and suite alone, so the second registration would silently take the first
one's shard. A **dormant** `PACK_NAME` is the deliberate exception: the call registers
nothing and reports at `STATUS`, naming the suites it left unregistered.

Tests built OFF, an empty `SUITES` and a dormant `PACK_NAME` all register nothing: that
is **absence of census evidence**, not a pass. A suite reading more than one shard states
its inventory through its ordinary host suite, which invoked directly still requires an
explicit expected arch and descriptor root. Check retained extension inventory and
heuristic-disabled score absence per [native-pack.md](native-pack.md).

**Gate:** current installation, artifact checks at their stated strength, and real
registration/loading plus applicable inventory/census checks. Report `NOT VERIFIED HERE`
separately. Neither the structural validator's stubs nor host loading proves dispatch.

## 5. Baseline device proof from the installation

On the allocated target host:

```bash
"$PY" "$GEN/tools/device_probe.py" --mode installed --arch "$ARCH" \
  --sweep-root "$SWEEP_ROOT" --install "$INSTALL"
```

A missing or invisible installation fails even if early feasibility passed. Use
`hipdnn-superbuild-test` discovery with component **`hip-kernel`**, running the helper
from the checkout being built so its revision matches the source:

```bash
"$PY" "$REPO/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts/discover_test_targets.py" \
  --build-dir "$BUILD" --component hip-kernel --scope external-integration
```

Do not accept the helper's first provider-prefixed command as exact-engine proof. The
provider's default installed CTest root is **`$INSTALL/bin/hip_kernel_provider`**, not
`$INSTALL`; substitute the configured bindir if customized. The provider does register
`hip_kernel_provider_asm_sdpa_gpu_ref_integration_tests`, which is the ASM SDPA engine
reached by a different path and never ingestor evidence.

The production descriptor root ships no bundle, so no dense-attention target is
registered. Replace `<your-bundle-ctest-target>` with the name your own registration
creates — a gfx942 dense bundle would be shaped like
`hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests`, which exists
nowhere in this tree:

```bash
CTEST_ROOT="$INSTALL/bin/hip_kernel_provider"
DEVICE_TEST=<your-bundle-ctest-target>
ctest --test-dir "$CTEST_ROOT" -N -V -R "^${DEVICE_TEST}$"
```

Require exactly your bundle's registration. Inspect its command/config for that bundle's
engine ID — `hipkernel:Gfx942AttentionDense` in the illustration — installed
executable/plugin/config paths and the intended quick/standard selection. Then execute:

```bash
ctest --test-dir "$CTEST_ROOT" --no-tests=error -V -R "^${DEVICE_TEST}$"
```

Keep verbose output in the retained log. **`--output-on-failure` hides passing suites'
case counts; an all-skip suite can still report CTest PASS.** Record selected, served,
skipped/declined and failed counts with observed reasons. Missing registration, zero
selected, all-skipped support, wrong engine/path or numerical failure blocks this gate.
Resolve the exact UED name → engine ID from the installation; a prefix or registry
listing is not dispatch attribution.

Use nontrivial inputs for quick feature breadth and bounded standard numerical depth.
Exercise required declines separately; another winning engine must not hide them. NaN or
unwritten output is a failure, not a tolerance adjustment.

Extensions must select the addition explicitly. The disposable pointwise example adds
HALF/block_size=256 to ADD, preserves MUL/SUB and changes ADD's expected census from
three to four. Select HALF/256 on logical dims `{1,1,1,1}`, check the actual
`hipkernel:Pointwise` plan and arithmetic, and retain old ADD/MUL/SUB and required
multi-element/two-node declines. Its source computes one element, so neither a
default-FLOAT pass nor this smoke proves arbitrary-size coverage.

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
installed KDP-entry counts and proven engine identity. YAML paths resolve relative to
the YAML file; there is no shell or environment interpolation. Set `SWEEP_CONFIG` to its
absolute path, then run:

```bash
"$PY" "$GEN/tools/sweep.py" --config "$SWEEP_CONFIG"
```

[workloads.md](workloads.md)'s *Installed measurement contract* owns every rule this run
must satisfy — session and arm ordering, warmup, rounds, cache isolation, separate
correctness, and the sweep's ownership of the benchmark's `--engine` and other
phase-owned arguments. This page owns only the commands and their order.

For rocKE, investigate measured survivors. Set `PAIRWISE_KNOBS` to comma-separated
surviving knob names and `PAIRWISE_CONFIG_ROOT` to their config output directory, then
generate, build and install the supported pairwise arms:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --pairwise "$PAIRWISE_KNOBS" --out-dir "$PAIRWISE_CONFIG_ROOT"
```

Obtain selection approval from coverage, correctness and per-corpus measurements. For
the rocKE shipping cross, `APPROVED_KNOBS_JSON` contains the approved JSON object of
knob value lists, **not a filename**:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --knobs "$APPROVED_KNOBS_JSON" --out "$FINAL_CONFIG"
```

Repeat **stages 3–5** with the final config and a new empty generation destination.
Neither an isolation arm nor an old install certifies the regenerated shipping set. An
explicitly untuned extension may retain its approved baseline selection but still needs
final installed artifact and corpus proof.

**Gate:** justified selection and revalidated final installation.

## 7. Final corpus proof and runtime reconciliation

Run the final installed artifact through a fresh-output YAML sweep with
`correctness.enabled: true`, and require the exact phase key set.
[workloads.md](workloads.md)'s *Installed measurement contract* owns what each terminal
status means — `SWEEP_DONE`, `SWEEP_TIMING_ONLY`, `SWEEP_INCOMPLETE` — and what resume
does and does not make a single cohort.

Harvest final phase results and available engine logs into the per-input outcome ledger,
then join within each corpus/phase before making its graph-name-to-reason JSON.
[workloads.md](workloads.md)'s *Complete final runtime join* owns the ledger fields, the
join scope and every rejection rule. This is an explicit evidence review, not an
automatic matcher-reason extractor.

For rocKE, set `CORPUS_SHAPES` and `RUNTIME_DECLINES` to that corpus's requests and
complete runtime-derived mapping, then reconcile:

```bash
"$PY" "$GEN/tools/reconcile_applicability.py" --profile "$PROFILE" \
  --shapes "$CORPUS_SHAPES" --declines "$RUNTIME_DECLINES"
```

Without the complete join, label reconciliation **offline only**. Investigate
reference-only supported rows and obtain explicit scope decisions for exclusions; escape
flags cannot excuse broken reference APIs or unexplained gaps. Direct-load pointwise
instead uses its explicit one-element corpus, installed engine results and independent
arithmetic/reference correctness, with `exclude_tensors: none`.

Report exactly the populations and statistics [workloads.md](workloads.md)'s *Required
reporting statistics* requires; timing is reported separately from the outcome
accounting.

**Gate:** zero wrong answers, complete final-runtime accounting, and no missing,
ambiguous, erroneous or unexplained in-scope outcomes. Changed installed artifacts
invalidate old evidence and return to stages 3–5.

## 8. Handoff

Report [SKILL.md](SKILL.md)'s completion evidence and exact limitations. Keep experiment
copies and probes disposable and retain their inputs/results in the evidence directory
defined under **Paths and interpreters**. For blocked work, name the last completed
stage and missing prerequisite; do not substitute a proposed command or queued job for proof.

Open the handoff with a scope/proof cover sheet. It binds each claim to the reports that
prove it; it is not another report, and it has no schema or tool of its own:

- **Source.** The exact commit SHA. For an uncommitted candidate, add a fingerprint of
  its dirty state or source content — for example digests of the diff against `HEAD`
  and of every untracked input. `HEAD` alone does not identify the tested source.
- **Engine and scope.** The installed engine ID (`$ENGINE`) and the families and
  architectures the claims cover.
- **Catalog and artifact.** A digest of the descriptor catalog the claims are about, and,
  recorded separately, the digest of each tested artifact (packed archive, installed
  plugin). The authored catalog and the artifact that ran are distinct identities. The
  catalog digest is the sha256 over the path-sorted `sha256sum` lines
  (`<sha256>  ./<relative path>`) of every `*.json` under the compared root; the artifact
  digest is the `sha256sum` of each tested archive or plugin file:

  ```bash
  (cd "$ROOT" && find . -name '*.json' -type f -print0 | sort -z | xargs -0 sha256sum) | sha256sum
  ```
- **Evidence.** Links to the authoritative reports and logs, unchanged — the tools'
  original outputs, never edited or summarized copies. Every count is derived from those
  catalogs and reports, never retyped into a hand-maintained table.
- **Proof rung per claim.** Constructed/static, compiled, loaded/censused or GPU-served:
  the rung the linked evidence actually observed for that claim. A rung not run is
  written as not run; a missing report is missing, not implied by a lower or older one.
- **Outcome accounting.** Served, skipped and declined counts taken from the reports,
  with each intentional decline named as intentional.

A report's identities are those recorded beside it in the handoff/evidence directory
when it was produced; the existing tools do not record them. A report is current proof
only when those identities — source, engine, catalog digest, families/arches and
artifact digest — match the current candidate. A report whose identities do not match
the current catalog is historical: cite it as such, never as proof of the current
catalog.

When the change alters generator output or its spelling — a template, fragment, helper
or emitted CMake — the handoff also inventories every cross-format caller of that
output, each with its owner: helper references, emitted fragments, CMake callers,
examples (including runbook and README snippets) and consumer tests. Targeted searches
at change time build the inventory; it is evidence for this change, not a permanent
source-text test. Consumption is proved by rendering and running the real consumer —
configure the CMake, then list and run its CTest entries — never by assertions on
source strings. A test pinning an obsolete spelling is removed, not repinned to the new
one.
