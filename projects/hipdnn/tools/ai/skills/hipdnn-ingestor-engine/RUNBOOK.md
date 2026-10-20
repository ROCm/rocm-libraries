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
CONFIG=/absolute/path/to/generator-config.yaml  # a registry-backed engine's authored selection, configs/<slug>.yaml
MANIFEST=/absolute/path/to/<slug>.engine.yaml   # rocKE registry-backed engines
SHAPES=/absolute/path/to/request-shapes.json
CORPUS_DIR=/absolute/path/to/graph-corpora
SCRATCH=/absolute/path/to/scratch-dir          # disposable tool output
EVIDENCE=/absolute/path/to/per-run-evidence-dir
ARCH=gfx950
ENGINE=<your-bundle-engine-id>
```

Replace `<your-bundle-engine-id>` with your bundle's engine ID; it is consumed verbatim
as `--expect-engine` below. The worked example's is `hipkernel:Gfx950AttentionDense`.

Follow the **Setup** section of `$GEN/README.md`. Authoring and mining imports need the
rocKE library environment (`$PROVIDER/rocke/library`); production packaging uses its own selected
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
otherwise retain logs, manifests, reports and outcomes in `$EVIDENCE`, a user-selected
per-run evidence directory outside product source. `$SCRATCH` holds regenerable tool
output; create both before the first command that writes there:

```bash
mkdir -p "$SCRATCH" "$EVIDENCE"
```

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
needs source investigation. A packaged engine whose kernel is a prebuilt per-arch code
object plus its symbol is authored as `hsaco`, not mined for a rocKE builder; see the
authored-source table in [SKILL.md](SKILL.md).

**rocKE: find the registry candidate.** In the family's dispatch module (attention:
`$PROVIDER/rocke/library/dispatch/attention/`), find the candidate that builds the user's
spec. Its `KernelCandidate` tuple function, the family's `OperatorRequest` class and the
attributes selecting exactly that candidate become the manifest's `enumerate.candidates`,
`enumerate.request` and `enumerate.pin`. Author `source`/`builder` by hand: no candidate
exposes them.

**Gate:** feasible target/workspace, representable scope and capable reference. A
missing dependency blocks its gate; host-only research may continue while a device
allocation is pending but cannot discharge device proof.

## 2. Contracts, corpus and the variant space

Record [graph-contract.md](graph-contract.md)'s topology/UID edges and field
dispositions, then [rocke-mining.md](rocke-mining.md)'s applicability, specialization,
layout, geometry/workspace and ABI evidence. Direct-load engines take these facts from
HIP source without inventing an engine manifest. Reuse unchanged extension contracts; reopen any
topology or feature the addition changes.

This stage mines a rocKE registry family; gfx950 dense attention
(`configs/gfx950_attention_dense.engine.yaml`) is the worked example. A hand-authored
source (`hip`, `hsaco`, a hand-written rocKE selection) writes its selection, `reduction:`
block and each cell's `default: true` arm directly and continues at stage 3.

### Corpus

Inventory owner-published results, owner benchmark shapes and external graphs per
[workloads.md](workloads.md). `$SHAPES` is a JSON list of semantic requests; the
benchmark consumes actual graph JSON under `$CORPUS_DIR/<corpus>`. `mine_shapes.py`
mines attention only; another family hand-authors `$SHAPES` and its graph corpus. For
the attention miner:

```bash
"$PY" "$GEN/tools/mine_shapes.py" \
  --published /absolute/path/to/owner-results.csv \
  --graphs "$CORPUS_DIR/servable" --arch "$ARCH" \
  --include-windowed --out "$SHAPES"
```

Add `--rocke-bench <actual-benchmark-tree>` when applicable. A dense prefill
benchmark (`benchmark_dense_prefill_live.py`) keeps its shapes in Python, so the tree
alone misses them. `--rocke-bench` takes one tree, and a repeated flag keeps only the
last one, so copy the benchmark tree, emit the dense shapes into the copy and mine the
copy in place of the original tree. Emitting needs no GPU, torch or numpy, only an
interpreter that imports the rocKE library:

```bash
OWNER_BENCH=/absolute/path/to/owner-bench-tree   # a new directory; cp creates it
cp -r "$PROVIDER/rocke/library/benchmarks/$ARCH/attention" "$OWNER_BENCH"
"$PY" "$PROVIDER/rocke/library/benchmarks/$ARCH/attention/prefill/benchmark_dense_prefill_live.py" \
  --emit-shapes "$OWNER_BENCH/dense_prefill_live_shapes.json" \
  --dtype bf16 --hq 128 --hkv 8 --d 128
```

Repeat the emit with another file name per dtype or head configuration the owner
measures, then pass `--rocke-bench "$OWNER_BENCH"`. Packed varlen rows are skipped and
counted. Reconcile each source's
total, parsed, servable, covered and excluded counts; do not discard window/sink or
independent operand dimensions to fit the request schema.

### Engine manifest and shapes report

The engine manifest (`$MANIFEST`) states what the dispatcher registry cannot: engine
identity, KMD schema, metadata mapping, native symbols, the builder, the request space
(`enumerate:`) and how it is reduced (`reduction:`). The README's *Config surface*
documents every key. Choose `enumerate.cells` from `$SHAPES`, then report how the corpus
falls on them:

```bash
"$PY" "$GEN/tools/registry_enumerator.py" --manifest "$MANIFEST" \
  --out "$SCRATCH/registry.yaml" --shapes "$SHAPES"
```

Each shape lands in the first category that applies: `outside cell axes` (it sets a field
neither `fixed` nor in `cells`), `declined by registry` (with the candidate's reason),
`covered`, or `GAP` (admitted, but its cell is not listed). This is offline applicability
against the registry, not runtime coverage
([workloads.md](workloads.md#applicability-and-reference-contract)).

### User checkpoints

Ask each question and record the answer in `$EVIDENCE`; accepting what was presented is a
valid answer.

- **Corpus:** present the sources, distinct shapes and graphs per source, and the
  `--shapes` categories; ask for shapes or graphs to add or drop. Rerun the report if the
  corpus changed.
- **Ground truth:** always ask, never block on the answer: does the user have the kernel
  owner's performance or accuracy data? If not, record "no ground truth supplied".
- **Knobs:** after enumerating the variant space below, ask whether to exclude or
  prioritise knobs or narrow cells, using `--plan-space`'s exact instance counts; above
  `reduction.aot_budget_per_arch` the user narrows them before any pack. Estimate compile
  time as instances × seconds per kernel compile, measured on the build machine.

**Owner data.** Set `OWNER_DATA` to the owner's file (CSV with a header, or a JSON list
of rows) and convert each row into a graph of a separate `ground_truth` corpus. Pass one
`--map <request-field>=<owner-column>` for each `graph_template` placeholder and
`value_map` key, and no other field (gfx950: `batch, seqlen_q, seqlen_k, nhead_q, nhead_k,
hdim_q, hdim_v, dtype, mask_type`); an unconvertible value exits 2, so rewrite other
spellings first. Set `<owner-time-column>` and drop other architectures' rows with
`--filter <column>=<value>`:

```bash
GT="$EVIDENCE/ground_truth.json"
"$PY" "$GEN/tools/ground_truth.py" ingest --input "$OWNER_DATA" --manifest "$MANIFEST" \
  --map <request-field>=<owner-column> --map <request-field>=<owner-column> \
  --time-col <owner-time-column> --time-unit ms --device "$ARCH" --out "$GT"
"$PY" "$GEN/tools/ground_truth.py" graphs --gt "$GT" --manifest "$MANIFEST" \
  --out "$CORPUS_DIR/ground_truth"
```

Set `GT_MARGIN` to the comparison margin the user confirms, as a fraction (`0.05` accepts
hipDNN up to 5% slower than the owner). Keep unpublished owner data, `$GT`, its graphs and
reports local unless the owner allows publication.

### The registry variant space

Populate from the whole registry: `space: full` keeps, per cell, every production-sweep
spec sharing the cell's `select_spec` feature fields, projecting each varying performance
knob into the spec, KMD, metadata, bindings and kernel name. A varying field in no
`reduction.fields` list is held at its registry default with a `WARN`. The shipped gfx950
manifest stays `space: slice` until a full-derived selection ships with the flip; set
`space: full` in a copy beside it in `configs/` (`graph_template` is manifest-relative).

Print the space by knob first, for the knob checkpoint. Nothing is written:

```bash
"$PY" "$GEN/tools/registry_enumerator.py" --manifest "$MANIFEST" --plan-space
```

Set `KNOBS` to the comma-separated names it lists under `varying spec fields` as
`(performance)`, in `reduction.fields.performance` order. A performance name the registry
does not vary in this space is not a knob, and `factorise_config.py` refuses it. Set
`VOCABULARY` to the manifest's `specialization.vocabulary` as `field:from=to,from=to`,
fields separated by `;` (gfx950 dense: `dtype:bf16=BF16,fp16=FP16`). Then enumerate and
compact into the selection:

```bash
"$PY" "$GEN/tools/registry_enumerator.py" --manifest "$MANIFEST" --out "$SCRATCH/registry.yaml"
"$PY" "$GEN/tools/factorise_config.py" --config "$SCRATCH/registry.yaml" \
  --out "$CONFIG" --knobs "$KNOBS" --vocabulary "$VOCABULARY"
```

`registry_enumerator.py` marks each cell's `select_spec` arm `default: true`;
`factorise_config.py` keeps it and refuses output that does not re-expand. Nothing
compiles here. Record the knob answer in `$MANIFEST`
(`reduction.fields.performance` in priority order, `reduction.fields.ignored`,
`enumerate.cells`) and rerun all three commands if it changed.

**Gate:** every GAP cell added to `enumerate.cells` or excluded with its reason; recorded
checkpoint answers; a round-tripped `$CONFIG`. A shipped engine's selection is also gated
per the README's *Updating the captured catalog*.

## 3. Generate, implement and splice

Generate from the selection `$CONFIG` into an empty scratch directory, never over the live
engine:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED"
```

Run both against the revision you are about to splice; a render from an earlier revision
is not evidence for this one. The config loader rejects a YAML mapping that repeats a key,
including a repeated `<<`, and reports both source positions.

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
tuple identity includes schema types/defaults and the arch tier at which kernels compete.
Structural mode reports compiled agreement as `NOT CHECKED` and can exit 0 with unrun
checks. `--manifest "$MANIFEST"` optionally names the bundles to gate (the KDPs whose
UED name equals the manifest's `engine.name`; every pack of that engine is gated) and
supplies the matcher vocabulary (`specialization.vocabulary`), never compiler evidence. Full artifact checking belongs
after packing.

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
| `hkp_register_census_tests(TARGET hip_kernel_provider_census_tests PACK_NAME … [ARCHES …] SUITES … EXPECTED_CASES …)` | `$PROVIDER/src/tests/CMakeLists.txt` | A census suite that reads exactly one pack target's shard |
| Descriptors themselves | — | **Never.** There is no descriptor splice |

**Descriptors need no CMake edit.** The packer walks a source root recursively and no
descriptor is named in CMake, so installing one is dropping files under the right root.
A `cmake_descriptor_files.txt` fragment states that fact; it is not a list to paste.

Census registration is one `hkp_register_census_tests()` call per packed target, in
`$PROVIDER/src/tests/CMakeLists.txt` beside `hkp_verify_embedded_sources()`, after the
test target exists:

```cmake
hkp_register_census_tests(
    TARGET hip_kernel_provider_census_tests
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

`TARGET` is the census binary, `hip_kernel_provider_census_tests`: census suites
(`Test<Name>Packs.cpp`) are compiled into it, never into `hip_kernel_provider_tests`,
because every census case needs a shard and the census environment that an ordinary
unit run must not require. `PACK_NAME` selects the wired pack target whose `OUT_ROOT`
and recorded arch list the entries address. `ARCHES` optionally narrows that list:
omitted, the suite registers at every arch the pack target was wired for (right for a
suite whose fixtures cover the whole root, as here); given, at the intersection with
the wired list, so a bundle that emits only for gfx950 passes `ARCHES gfx950` and a
build packing other arches registers nothing for it; the keyword with no arch is fatal.
A suite is declarable **only where it reads exactly one pack's shard**:
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

If the manifest declares a `launch_surface:` block, also run:

```bash
"$PY" "$GEN/tools/launch_surface.py" "$MANIFEST" --check
```

The audit never compares what the C++ and Python compute: record every rule the native
pack restates by hand as an unchecked agreement ([rocke-mining.md](rocke-mining.md)).

**Gate:** a current-revision dry-run and render, final authored inventory, completed hooks
and source/test splices, no selected-path placeholders, and reviewed structural/field/ABI
results. None proves native loading or numerical dispatch.

## 4. Build, pack, install and prove the host boundary

**Build and sweep plan approval comes first.** Before packing, get the user's approval of
the instances per architecture, corpus sizes, sweep rounds and estimated compile and sweep
time, and of the job envelope for any remote or cluster run; record both in `$EVIDENCE`.

Configure from `$REPO` using `hipdnn-superbuild`, with
`CMAKE_INSTALL_PREFIX="$INSTALL"`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` and
`HIPKERNELPROVIDER_ENABLE_TESTS=ON`, plus `HIPKERNELPROVIDER_ENABLE_ROCKE=ON` when the
build packs a rocKE bundle. SDPA needs `HIPDNN_ENABLE_SDPA=ON` consistently in SDK and
provider.

The component selection must include the provider. The `hipdnn-providers` preset does
**not** build hip-kernel-provider; the presets that do are `hipdnn-providers-all`,
`hip-kernel-provider`, `hipdnn-dev-all` and `miopen-hipdnn-dev-all`.

**`HIPKERNELPROVIDER_ENABLE_ROCKE=ON` is required to pack a rocKE bundle.** With it ON,
rocKE is resolved once for **every** root, test roots included, so an unresolvable comgr
is fatal at configure even in a hip-only build. With it OFF the ingestor still
configures, builds and packs: the hip producer packs alone with no rocKE wheel, pip or
comgr, the `rocKE/` family folder is excluded from every root, and any `rocke` UKD
elsewhere is pruned like an arch-pruned one. HIP-only and `embedded_source` bundles pack
in either mode.

There is **no per-producer production switch**: producer selection is per-UKD on
`kernel_source.kind`, so one root feeds every producer. Two filters remove content: a
family folder (`rocKE/`) is excluded when its option is OFF, matched by exact name as a
top-level folder of a root, and only producers gated by a build option (rocKE) are
disabled by kind. Whoever adds a new
`kernel_source.kind` to the packer also adds its entry to
`descriptor-packaging/tools/hkp_probe_kinds.py`; a probe keeping a UKD of an unregistered
kind fails configure.

Every root the provider wires, production and test alike, is probed at configure with
the same filters the pack step gets. A root that is empty, or that would ship nothing
for any architecture this build packs for, is **dormant**: it is skipped at pack, any
stale output tree is removed, and one STATUS line says why. That is never an error,
whether the root was named or inherited. A root set but not a directory is fatal at configure. A KDP is
what arch pruning consumes, so standalone UKDs, kernel sources, READMEs and hidden-path
KDPs alone do not make a pack.

The large rocKE bundle files under that root (KDPs, kernel objects, heuristic model files) are DVC outputs (a `<file>.dvc` or `<folder>.dvc` pointer beside the output, remote `ingestor`), not git files: run `dvc pull -r ingestor <pointer>` before a production configure with `HIPKERNELPROVIDER_ENABLE_ROCKE=ON`. Without the pull, configure warns and the shipped bundle is not packed; `HIPKERNELPROVIDER_REQUIRE_DVC_DESCRIPTORS=ON` makes it an error. See `descriptors/README.md`.

The production root is `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, a `CACHE PATH`
defaulting to `$PROVIDER/src/engines/kernel_ingestor_engine/descriptors/`. It carries the
rocKE `gfx950_attention_dense` bundle, so production packaging is dormant for a build
that excludes `rocKE/` or packs no arch that bundle's KDP declares. Substitute your
bundle's name wherever a bundle path appears below.
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
a passthrough descriptor and records the authored values in the descriptor's
provenance sidecar, so there is no arch-independent installed tree. Resolve `VALIDATOR`
to the built `hipdnn_validate_descriptors` executable and validate the runtime dialect:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" --expect-engine "$ENGINE" --json
```

**Where provenance lives.** The packer writes each packed descriptor's UKD provenance
to a sidecar named for the descriptor file, `foo.kdp.provenance.json.gz` beside
`foo.kdp.json` (`foo.ukd.provenance.json.gz` beside `foo.ukd.json`), so a staged
build-tree root holds descriptor JSON plus sidecars. The runtime loader
never reads provenance, so the installed runtime tree
(`<engines>/arch_content/hip-kernel-provider/`) holds **no** sidecars. With
`HIPKERNELPROVIDER_ENABLE_TESTS=ON` they install with the test content at the same
relative paths, under
`<engines>/test_arch_content/hip-kernel-provider/provenance/`, where `<engines>` is the
installed plugin engine directory (`$INSTALL/lib/hipdnn_plugins/engines` by default on
Linux, under the bindir on Windows). Set `FINAL_PROVENANCE_ROOT` to the shard's mirror
there, for example `<engines>/test_arch_content/hip-kernel-provider/provenance/$ARCH`
for `FINAL_DESCRIPTOR_ROOT=<engines>/arch_content/hip-kernel-provider/$ARCH`.

**The tree says whether it is packed.** The packer writes an empty `hkp-packed.marker`
into every directory it writes a packed descriptor into; it holds no provenance and
installs with the runtime tree. `hkp_desk_check.py`, `verify_variant_sets.py` and
`coverage_gate.py` read a descriptor as packed exactly when
its own directory holds the marker. A packed descriptor must have its sidecar, so an
installed shard read without `--provenance-root` fails, naming the first descriptor
whose sidecar is missing. A descriptor without the marker is authored: its sidecar is not
read, and a `kpack` UKD in it is an error. A tree that lost its marker (a copy that
globbed `*.json`, a tree packed before the marker existed, a hand-staged tree) reads as
authored, so a shard of only `embedded_source` UKDs then gets no provenance check.
`--provenance-root` only relocates sidecars: given for a descriptor whose directory holds
no marker, it is a usage error.

**The embedded-source invariant.** A staged tree holds descriptor JSON, provenance
sidecars and packed markers only, so an `embedded_source` descriptor resolves its
`source_file` against a key table compiled into the binary.
`descriptor-packaging/tools/hkp_verify_embedded_sources.py`, wired by
`hkp_verify_embedded_sources()` beside the census registration, runs at build time over
emitted JSON and sidecars alone and checks **presence** (every named `source_file` is a
key of that table) and **location** (the file under that key is the file at the
authored location, joining the sidecar entry's `provenance.source_label` with `rel_dir`
and `source_file`). A separate stamp-keyed rule requires a pack root whose stamp file is
present to hold at least one descriptor. Pointed at installed roots instead, give it one
`--provenance-root` per `--staged-descriptor-root`, in the same order.

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
  --manifest "$MANIFEST" final "$FINAL_DESCRIPTOR_ROOT" \
  --provenance-root final="$FINAL_PROVENANCE_ROOT"
```

`--provenance-root LABEL=DIR` pairs with the `LABEL ROOT` of the same name and is
repeatable. For a staged build-tree root, whose sidecars sit beside the descriptors,
drop it.
`hkp_desk_check.py` takes `--provenance-root DIR` with `--descriptor-root` (the root
the provenance root mirrors, holding the KDP), and `coverage_gate.py` takes
`--provenance-root DIR` mirroring its `--tree`.

Omit `--manifest` when neither bundle selection nor extra vocabulary needs it. Add
`--kpack-python-dir <dir>` if the reader environment requires it. Interpret outcomes
separately:

- Use full checking only for **packed `kind: kpack` descriptors**; confirm the input
  dialect rather than inferring it from exit 0. Missing declarations, mismatches and
  required checks `NOT RUN` block acceptance, including missing vocabulary.
- A packed kernel declaring no specialized `metadata_fields` and carrying no
  `effective_spec` reports **`NOT VERIFIED HERE`** when `provenance.origin_kind` is
  absent, `hip` or `hsaco`: no gate failure and no compiled-specialization proof. AOT
  HIP specialization stays outside this check.
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
  stay mandatory for every compiling kind. "Descriptors but no archive" is legal, and
  so is a root with nothing to pack for the build: it is skipped, and configure leaves
  the production root dormant.
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

For each declared suite and each eligible arch — the pack target's recorded list,
narrowed by `ARCHES` when given — CMake registers **four** tests. The census entry is
`hip-kernel-provider-hkp-census-<arch>-<suite>`, which invokes

```text
hip_kernel_provider_census_tests --gtest_filter=<suite>.*
```

with `HIPDNN_TEST_CENSUS_SUITE` set to that suite, `HIPDNN_TEST_EXPECTED_ARCH` to that
arch — taken from the wired arch list, never from a detected device or the descriptors —
and `HIPDNN_DESCRIPTOR_DIR` to that pack target's own `OUT_ROOT` shard for the arch,
never a shared stage tree; labelled `unit_test;hip-kernel-provider;host` plus the tier
labels `HKP_PACK_CTEST_CATEGORIES_YAML` assigns, which the installed twin carries too. The other
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
the pin that the CTest entries get from `EXPECTED_CASES`. Set `EXPECTED_CASES` to the
same reviewed comma-separated case-name list the build-tree registration pins. **Do not
derive it from the installed binary under test**: such a list agrees by construction and
pins nothing. Omitting it is silent — the listener returns early on an empty expected
list and the census still reports complete. Adjust the binary path for a nondefault
install bindir:

```bash
HIPDNN_TEST_CENSUS_SUITE="$CENSUS_SUITE" \
HIPDNN_TEST_EXPECTED_ARCH="$ARCH" \
HIPDNN_TEST_CENSUS_EXPECTED_CASES="$EXPECTED_CASES" \
HIPDNN_DESCRIPTOR_DIR="$FINAL_DESCRIPTOR_ROOT" \
"$INSTALL/bin/hip_kernel_provider_census_tests" --gtest_filter="${CENSUS_SUITE}.*"
```

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
`ARCHES` keyword naming no arch; an empty recorded arch list; and one suite declared at
two pack targets, whose entry names
carry arch and suite alone, so the second registration would silently take the first
one's shard. A **dormant** `PACK_NAME` is the deliberate exception: the call registers
nothing and reports at `STATUS`, naming the suites it left unregistered.

Tests built OFF, an empty `SUITES` and a dormant `PACK_NAME` all register nothing: that
is **absence of census evidence**, not a pass. A suite reading more than one shard states
its inventory through its ordinary host suite, which invoked directly still requires an
explicit expected arch and descriptor root. Check retained extension inventory and
heuristic-disabled score absence per [native-pack.md](native-pack.md).

**Gate:** an approved build and sweep plan, current installation, artifact checks at
their stated strength, and real registration/loading plus applicable inventory/census
checks. Report `NOT VERIFIED HERE` separately. Neither the structural validator's stubs
nor host loading proves dispatch.

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

Other bundles in the production root may register their own targets; they are not
evidence for yours. Replace `<your-bundle-ctest-target>` with the name your own
registration creates. A gfx942 dense bundle would be shaped like
`hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests`; treat it as an
illustration, not a name to copy:

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

## 6. Reduce the variant selection

Every packaged engine whose selection carries a `reduction:` block runs this loop. An
explicitly untuned engine (a single candidate, or an extension retaining its approved
baseline) has none, skips it, and still needs stage 7's proof. Every feature cell keeps
its floor: its `default: true` variant plus the smallest `divides` values (e.g. tiles)
among variants sharing its other knobs, which divide every shape a larger tile divides.

1. **Build, pack and install the selection** through stages 3–5. Set `INSTALLED_KDP` to
   the installed `<slug>.kdp.json` under `$FINAL_DESCRIPTOR_ROOT`.
2. **Sweep it.** Populate `configs/reduction.sweep.yaml.example`: one arm (the selection's
   install; recount `expected_descriptors` after every reinstall), every approved corpus
   plus `ground_truth` when supplied, at least 2 `rounds`, `correctness.enabled: true`.
   Set `SWEEP_CONFIG` to its absolute path and `SWEEP_OUTPUT` to its `output_dir`, then
   run it to `SWEEP_DONE` under [workloads.md](workloads.md)'s *Installed measurement
   contract*:

   ```bash
   "$PY" "$GEN/tools/sweep.py" --config "$SWEEP_CONFIG"
   ```

3. **Check correctness first.** Apply no reduction while a correctness failure is
   uninvestigated. Only each graph's autotune winner is numerically checked.
4. **Reduce and present the proposal.** The report holds the ε-versus-instances curve,
   each variant's keep/drop reason and numerics tag, the cells with no corpus evidence,
   and proposed rules (`drop knob X`, `drop X=v`, `keep X=v only for <cells>`) with their
   saving and cost. Exit 2 is unusable or stale input. Exit 3: the sweep cannot support a
   reduction of this arm, because `summary.json` or its `expected_phases` is missing or unusable (the
   sweep did not finish), an expected timing or correctness phase of the arm has no
   record, or a timing phase ranked nothing (nothing is written), or timing phases are
   missing for some (round, corpus), or a correctness attempt failed (the report is
   written). No selection is written, so investigate or rerun the
   sweep. A single round warns
   that the noise floor is assumed (`reduction.single_round_epsilon`, default 0.05), not
   measured; with two or more rounds the floor is `reduction.drift_multiplier` (default
   2) times the measured drift. Exit 1: floor plus coverage alone exceed the budget; with
   the user's approval raise it (`reduction.aot_budget_per_arch` or `--budget`) or narrow
   cells, corpus or knobs and rerun from stage 2:

   ```bash
   "$PY" "$GEN/tools/variant_reduce.py" --selection "$CONFIG" --kdp "$INSTALLED_KDP" \
     --sweep-output "$SWEEP_OUTPUT" --out "$SCRATCH/reduced.yaml" \
     --report "$EVIDENCE/variant_reduce.md"
   ```

5. **Apply what the user picks.** Set `RULES` to the comma-separated rule ids the user
   chose from the report's rule table (`R` and six hex digits, derived from the rule
   text, so an id is stable while the rule's text is; a `keep X=v only for <cells>` rule
   names cells chosen with the budget's ε, so its text and id can change with
   `--budget`; a rule the sweep no longer proposes exits 2), then write and adopt that
   selection (or copy `$SCRATCH/reduced.yaml` if the user accepts the ε keep-set). The
   report gives the applied rules' combined worst gain lost and the graphs that lose
   their best kernel; above the rules' ε it exits 1 and writes no selection, so choose
   fewer rules:

   ```bash
   "$PY" "$GEN/tools/variant_reduce.py" --selection "$CONFIG" --kdp "$INSTALLED_KDP" \
     --sweep-output "$SWEEP_OUTPUT" --out "$SCRATCH/selection.rules.yaml" \
     --report "$EVIDENCE/variant_reduce.rules.md" --apply-rules "$RULES" &&
     cp "$SCRATCH/selection.rules.yaml" "$CONFIG"
   ```

6. **Repeat stages 3–6** with a new empty generation destination and a fresh sweep
   `output_dir`. With ground truth, compare it every iteration; its report lists every row
   outside the margin, failing correctness, with no usable reference, or missing:

   ```bash
   "$PY" "$GEN/tools/ground_truth.py" compare --gt "$GT" --sweep-output "$SWEEP_OUTPUT" \
     --margin "$GT_MARGIN" --report "$EVIDENCE/ground_truth.md"
   ```

   This iteration's `compare` exits 0 whatever the comparison finds: its report is for
   review, and an outside-the-margin row is expected while the selection is still being
   reduced. It exits 2 when a ground-truth row, or the graph a sweep ran for it, no longer
   matches what `graphs` recorded; rerun `graphs` and the sweep. Stage 7 reruns it with
   `--strict` as the gate.

7. **Stop when the user accepts the instance count.** No served graph may lose coverage
   or regress beyond the chosen ε or the applied rules' reported worst gain lost (`inf`:
   a graph lost every ranked kernel), and the last iteration's stages 3–5 ran on exactly the
   selection being shipped. A `space: full` selection ships with the manifest's flip, per
   the README's *Updating the captured catalog*.

**Gate:** a user-accepted selection with its applied rules recorded in `$EVIDENCE`, no
open correctness failure, and a revalidated installation of exactly that selection.

## 7. Final corpus proof and runtime reconciliation

Run the final installed selection through a fresh-output YAML sweep across the whole
corpus, plus `ground_truth` when supplied, with `correctness.enabled: true`, and require
the exact phase key set; stage 6 checked only each graph's winner.
[workloads.md](workloads.md)'s *Installed measurement contract* owns what each terminal
status means — `SWEEP_DONE`, `SWEEP_TIMING_ONLY`, `SWEEP_INCOMPLETE` — and what resume
does and does not make a single cohort. With ground truth, rerun stage 6's `compare` on
this output with `--strict`: exit 1 means a row is outside the margin, missing, declined
in any timing round, failing correctness or without a passing correctness verdict, so fix
it or record the user's acceptance of that row in `$EVIDENCE`. Check the budget the same way; a dry run writes
nothing, and exit 1 means a pack arch is over `reduction.aot_budget_per_arch`, so reduce
further or justify the count in `$EVIDENCE`:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" --dry-run --strict
```

Harvest final phase results and available engine logs into the per-input outcome ledger,
then join within each corpus/phase.
[workloads.md](workloads.md)'s *Complete final runtime join* owns the ledger fields, the
join scope and every rejection rule. This is an explicit evidence review, not an
automatic matcher-reason extractor.

No tool reconciles declines against a reference library: the runtime ledger is the
evidence, and without the complete join the accounting is **incomplete**. Investigate
every unexplained decline and obtain explicit scope decisions for exclusions. Direct-load
pointwise instead uses its explicit one-element corpus, installed engine results and
independent arithmetic/reference correctness, with `exclude_tensors: none`.

No tool checks offline that every shipped variant is reachable from some corpus shape
(the retired `variant_reachability` check): `variant_reduce` reports a variant as "never
ranked" only after a GPU sweep, and `registry_enumerator.py --shapes` classifies shapes,
not shipped variants.

Report exactly the populations and statistics [workloads.md](workloads.md)'s *Required
reporting statistics* requires; timing is reported separately from the outcome
accounting.

**Gate:** zero wrong answers, complete final-runtime accounting, no missing, ambiguous,
erroneous or unexplained in-scope outcomes, every ground-truth row (when supplied) within
the margin or accepted, and no unjustified budget WARN. Changed installed artifacts
invalidate old evidence and return to stages 3–5.

**Final step: packaging probe.** The architecture probe is automatic and needs no edit:
every explicit `arch` of the bundle's KDPs/UKDs that the build does not target gets probe
`<arch>`, packing the default sample (for rocKE, each value of each varying `spec` field
once; combinations are not guaranteed). Optionally, when specific UKDs must be packed on
every PR, add a named probe,
`hkp_add_packaging_probe(ARCH <gfx> NAME <integration> UKDS <representative UKD names>)`,
to `dnn-providers/hip-kernel-provider/descriptor-packaging/probes/probes.cmake`. Configure
with `-DHIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES=ON` and a `GPU_TARGETS` that excludes the
bundle's architecture, check the configure STATUS lines `hkp: automatic probe arches:
[...]` (the architecture is listed) and `hkp: probe <arch>: ... (<n> UKD(s) for <arch>)`,
build (`cmake --build <build> --target hkp_packaging_probes`), and run
`ctest --test-dir <build> -R '^hkp-probe-' --no-tests=error`; every `hkp-probe-*` entry must
pass. See "Packaging probes" in the descriptor-packaging README.

## 8. Handoff

Report [SKILL.md](SKILL.md)'s completion evidence and exact limitations. Keep experiment
copies and probes disposable and retain their inputs/results in the evidence directory
defined under **Paths and interpreters**. For blocked work, name the last completed
stage and missing prerequisite; do not substitute a proposed command or queued job for proof.

Record every checkpoint answer, the build and sweep plan approval and the applied
reduction rules; cite the local ground-truth report instead of quoting owner numbers.

Open the handoff with a scope/proof cover sheet that binds each claim to the reports
proving it:

- **Source.** The exact commit SHA. For an uncommitted candidate, add a fingerprint of
  its dirty state, for example digests of the diff against `HEAD` and of every untracked
  input.
- **Engine and scope.** The installed engine ID (`$ENGINE`) and the families and
  architectures the claims cover.
- **Catalog and artifact.** The digest of the descriptor catalog the claims are about,
  and separately the `sha256sum` of each tested artifact (packed archive, installed
  plugin). The catalog digest is the sha256 of the `sha256sum` lines
  (`<sha256>  ./<relative path>`) of every `*.json` under the compared root, in byte
  order of path. The recipe exits non-zero and prints no digest when the root is missing
  or holds no `*.kdp.json`:

  ```bash
  CATALOG_ROOT=/absolute/path/to/compared-descriptor-root
  (
    set -euo pipefail
    cd "$CATALOG_ROOT"
    [ -n "$(find . -name '*.kdp.json' -type f -print -quit)" ] ||
      { echo "no *.kdp.json under $CATALOG_ROOT" >&2; exit 1; }
    lines=$(find . -name '*.json' -type f -print0 | LC_ALL=C sort -z | xargs -0r sha256sum)
    printf '%s\n' "$lines" | sha256sum
  )
  ```
- **Evidence.** Links to the tools' unedited reports and logs; every count is derived
  from them, never retyped into a hand-maintained table.
- **Proof rung per claim.** Constructed/static, compiled, loaded/censused or GPU-served:
  the rung the linked evidence observed for that claim. A rung not run is written as not
  run; a missing report is missing, not implied by a lower or older one.
- **Outcome accounting.** Served, skipped and declined counts taken from the reports,
  with each intentional decline named as intentional.

No tool records a report's source revision or dirty-state fingerprint, or its catalog
digest in this format; record them beside the report in the evidence directory when it is
produced. A report is current proof only when its source, engine, catalog digest,
families/arches and artifact digest match the current candidate; otherwise it is
historical, cited as such and never as proof of the current catalog.
