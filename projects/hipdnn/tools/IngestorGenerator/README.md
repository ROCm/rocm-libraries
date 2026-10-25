# hipDNN IngestorGenerator

Generates a hipDNN generic-kernel-ingestor descriptor bundle from a YAML config:
KMD/UED/UMD/UHD/UDD/KDP JSON, a native-symbol stub, a complete pack-shape census test, a
matcher-test stub, and five CMake/registration text fragments. It follows
`projects/hipdnn/tools/DescriptorGenerator`'s conventions with two deviations:
`undefined=StrictUndefined`, so an unset UUID cross-reference fails at generation time,
and a required `--force` to overwrite a non-empty output directory. Generate into scratch
space, including when extending an engine, and splice addition-only with
`tools/splice_additions.py` (see [Extending a live engine](#extending-a-live-engine)).

Generation is toolchain-free and issues **no compiler evidence**: a complete integration
also needs artifact agreement, real native loading and engine-attributed numerical device
proof. The [ingestor RUNBOOK](../ai/skills/hipdnn-ingestor-engine/RUNBOOK.md) owns the
only ordered create/extend procedure.

## Prerequisites

- Python 3.10+
- PyYAML >= 6.0
- Jinja2 >= 3.1

## Setup

```bash
cd projects/hipdnn/tools/IngestorGenerator
/absolute/path/to/python3.10+ -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Name the interpreter by absolute path; `python3` resolves through `PATH`. On a Windows
checkout every `.venv/bin/<tool>` below is `.venv/Scripts/<tool>.exe`.

Introspecting a rocKE builder (`codegen.sources.introspect`, and every config or tool that
resolves a `kernel_source.kind: rocke` builder) imports the kernel module, so the rocKE
library must be on `PYTHONPATH`. From this directory, for the in-tree provider:

```bash
PROVIDER=../../../../dnn-providers/hip-kernel-provider
export PYTHONPATH="$PROVIDER/rocke/library:$PROVIDER/rocke/platform/python"
```

Without it the import fails with `RockeIntrospectionError: module not importable ...
No module named 'kernels'`. On Windows the separator is `;`.

### The optional archive dependency

`tests/test_verify_variant_sets.py`'s `TestRealArchiveSelectedConsumer` writes a real kpack
archive and needs `rocm_kpack`, which ships in the rocm-kpack checkout rather than on PyPI.
Point `HIPKERNELPROVIDER_ROCM_KPACK_DIR` at that checkout's `python` directory, the same one
the packaging suite's `PYTHONPATH` entry and `--kpack-python-dir` name:

```bash
export HIPKERNELPROVIDER_ROCM_KPACK_DIR=/opt/rocm-kpack/python
```

Unset -- or exported empty, which counts as unset -- that class runs if `rocm_kpack` is
importable anyway and **skips** otherwise; set but wrong fails loudly, because naming one is
a request to run the class. See the
[packaging reference](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).

## Usage

```bash
# Preview what would be generated, without writing anything or creating the
# output directory.
.venv/bin/python generate.py \
    --config configs/scale_add.yaml \
    --output-dir /tmp/scale-add-bundle \
    --dry-run

# Generate for real.
.venv/bin/python generate.py \
    --config configs/scale_add.yaml \
    --output-dir /tmp/scale-add-bundle

# Regenerate a disposable output directory, never a live engine directory.
# --force is REQUIRED for a non-empty output directory.
.venv/bin/python generate.py \
    --config configs/scale_add.yaml \
    --output-dir /tmp/scale-add-bundle \
    --force

# Do not generate. Report unfilled stub markers in the files this config's
# engine emits. --emitted-root is repeatable and defaults to --output-dir.
.venv/bin/python generate.py \
    --config configs/scale_add.yaml \
    --output-dir /tmp/scale-add-bundle \
    --check-placeholders \
    --emitted-root <provider-engine-dir> \
    --emitted-root <provider-tests-dir>
```

Exit codes: `0` success; `1` a gate refusal or internal failure: with `--strict`, a pack
arch over `reduction.aot_budget_per_arch` (checked before anything is written),
`--check-placeholders` findings, an emitted file that cannot be read back, or a template
error; `2` unusable input: a missing or invalid config, a non-existent `--emitted-root`, an
`--output-dir` that is not a directory, a non-empty output directory without `--force`,
a config the renderer rejects, or a bad flag (argparse).

### Extending a live engine

Generation mints fresh UUIDs on every run, so never copy a scratch descriptor directory
over a live one: that replaces every retained identity, which
[extend.md](../ai/skills/hipdnn-ingestor-engine/extend.md) forbids. Render the edited
config into an empty scratch directory, then splice only the additions:

```bash
.venv/bin/python tools/splice_additions.py \
    --scratch <scratch>/descriptors/<producer>/<bundle> \
    --live <provider>/src/engines/kernel_ingestor_engine/descriptors/<producer>/<bundle> \
    --report splice_report.json      # add --check to verify without writing
```

The tool pairs documents by file name and kernel entries by `name`, maps every scratch UUID
to its live twin, and refuses (exit 1, nothing written) if a live file or kernel is missing
from scratch, or if any retained document or kernel differs beyond that UUID map. New
kernel entries are appended after the live ones with their new ids; a scratch-only file
(for example a new pack's KDP) is written whole with its references remapped. Retained
bytes and line endings are kept, so the KDP change is a pure insertion.

The generated census test (`tests/Test<Name>Packs.cpp`) carries no UUIDs and is emitted in
the provider's clang-format form, so copy it over the live one. The native and matcher
stubs are never copied over hand-maintained files.

For a provider that exports an engine-name constant in
`IngestorPacks.hpp`, set `engine.census_name_symbol` to that C++ identifier.
The generated census then uses the provider's name for its lookup and names
a descriptor/pack mismatch explicitly. Omit the field for engines without
an exported constant; their census uses the literal `engine.name`.

Review descriptor diffs with `--diff-algorithm=histogram` (or `patience`). Kernel entries
are near-identical blocks, and Git's default (Myers) algorithm can pair the wrong ones:
inserting 52 entries between existing ones in the 840-entry gfx950 KDP reads as 5249
insertions and 2493 deletions by default, and as 2756 insertions and 0 deletions under
histogram or patience. To make that the default for KDPs in your clone only:

```bash
git config diff.kdp.algorithm histogram
echo '*.kdp.json diff=kdp' >> "$(git rev-parse --git-path info/attributes)"
```

The splice tool appends rather than interleaves, so its output reads as pure insertion
under every algorithm.

## Output

The descriptor directory mirrors the authored root, so the emitted tree copies across
verbatim. `direct_load` emits under `test_descriptors/<set>/<slug>/`, `<set>` being the
`authored_subpath` the config must name; `packaged` emits under
`descriptors/<producer>/<bundle>/`, defaulting the subpath to `<kernel_source_kind>/<slug>`.

```
<output-dir>/
  <descriptor-dir>/                 # test_descriptors/<set>/<slug>/ (direct_load)
                                    # descriptors/<producer>/<bundle>/ (packaged)
    <slug>.kmd.json                 # KMD -- the engine's per-kernel metadata schema
    <slug>.ued.json                 # UED -- the engine descriptor
    <slug>.udd.json                 # UDD -- dispatch symbol
    <slug>.uhd.json                 # UHD -- only when engine.heuristic != "none"
    kernel_dtype_matches_graph.umd.json   # the one shared kernel-scoped matcher
    <slug>.kdp.json                 # single-pack engine: one KDP named after the slug
    <slug>_<pack>.kdp.json          # multi-pack engine: one KDP per pack
    operation_is_<disc>.umd.json    # multi-pack engine ONLY -- one operation-scoped
                                     # UMD per pack. A single-pack engine emits ZERO of
                                     # these; see "UMD policy" below.
  packs/<Name>Native.cpp            # native-symbol stub (graph_match/kernel_match/
                                    # score/dispatch bodies are all `// TODO`)
  tests/Test<Name>Packs.cpp         # COMPLETE pack-shape census -- not a stub
  tests/Test<Name>Matchers.cpp      # matcher-test stub (fixture shape only)
  fragments/*.txt                   # 5 CMake/registration fragments, see below
```

### UMD policy

A UMD is emitted **only** for genuine per-pack narrowing; topology/shape/dtype
applicability belongs in the UED's `graph_match`. A **single-pack** engine gets **zero**
graph-scoped UMDs, its one pack's `matchers[]` naming only the shared kernel-scoped dtype
matcher; a **multi-pack** engine gets **one** graph-scoped operation-matcher UMD per pack
(each pack setting a unique `discriminator`), plus that same shared matcher.

### Native-symbol stub shape

`packs/<Name>Native.cpp` declares symbol names and registers hooks through typed
`SymbolScope<Handle>` calls in `register<Name>Symbols()`. Fill every applicable
placeholder. Returning `std::nullopt` from `graph_match` empties the **whole** engine
catalog; it is not a per-candidate decline. A heuristic-disabled engine has no scorer
declaration, implementation or registration, and no UHD.

### Matcher-test stub

`tests/Test<Name>Matchers.cpp` constructs its `DeviceProperties` fixture **by value**,
never by querying the host (`hipGetDeviceProperties`/`getDeviceProperties()`), which
would make an arch-gated matcher test vacuous everywhere except CI's own arch
(`TestSdpaFwdPlanBuilder.cpp`).

### Pack-shape census test

`tests/Test<Name>Packs.cpp` exercises `discoverDescriptorSets()` and the provider's typed
registration/loading path against the finalized emitted inventory: pack/kernel identities,
counts, SDK version and runtime source kind come from the actual output, after
normalization and deduplication. Packaged runtime source kind is KPACK, not the authored
builder kind. The file is emitted in the form the provider's `.clang-format` (pre-commit
pins clang-format 18.1.4) produces, so a copy is not reformatted later;
`tests/test_packs_formatting.py` checks the shipped gfx950 config against the live file and,
with `HIPDNN_CLANG_FORMAT` naming a binary, every shipped config against clang-format.

**The census lives in its own binary.** `Test<Name>Packs.cpp` is spliced into
`hip_kernel_provider_census_tests`; the generated matcher test stays in the ordinary
`hip_kernel_provider_tests`. Every census case needs a descriptor shard and the census
environment, so a census case inside the ordinary unit binary makes an ordinary unit run
require package state it should never need. The `cmake_test_sources.txt` fragment emits
the two `target_sources()` lines separately for exactly this reason; splice them into the
two different blocks rather than merging them.

Declare the literal `Test<Name>Packs` suite in one `hkp_register_census_tests()` call in
`.../src/tests/CMakeLists.txt`, beside `hkp_verify_embedded_sources()` and after the test
target exists:

```cmake
hkp_register_census_tests(
    TARGET hip_kernel_provider_census_tests
    PACK_NAME <the pack target holding this bundle's shard>
    ARCHES <the architectures this bundle actually emitted>   # optional; see below
    SUITES Test<Name>Packs
    EXPECTED_CASES
        <one line per case name the suite registers>
)
```

`EXPECTED_CASES` pins the suite's case-name set: optional to CMake, required for the census
to mean anything, since the execution half takes its obligations from the registered cases.
The `cmake_test_sources.txt` fragment emits it pre-filled; re-splice when the bundle's shape
changes.

`PACK_NAME` selects the wired pack target whose `OUT_ROOT` and arch list the entries
address.

**Shard eligibility is restricted, not universal.** `ARCHES` narrows registration to the
architectures the bundle emitted, intersected with the pack target's wired list, and the
fragment fills it in from this run's own inventory. The three spellings are distinct:

| `ARCHES` | Meaning |
|---|---|
| omitted | every arch the pack target was wired for -- the right answer for a bundle whose descriptors name no architecture |
| given | the intersection with the wired list; an arch the bundle ships nothing for is not registered at all |
| present with no values | a configure error, not "no restriction" |

`ARCHES *` is never correct: `*` is how a descriptor says it ships everywhere, and no
shard is named after it. Registering a bundle's suite for every product arch is the
failure this prevents -- a gfx950-only bundle otherwise gets a gfx942 entry that asserts
a gfx950 inventory against a gfx942 shard.

CMake registers one `hip-kernel-provider-hkp-census-<arch>-Test<Name>Packs` per eligible
arch, each running `hip_kernel_provider_census_tests --gtest_filter=Test<Name>Packs.*`
directly, with:

- `HIPDNN_TEST_CENSUS_SUITE=Test<Name>Packs`
- `HIPDNN_TEST_EXPECTED_ARCH=<arch>` (from the eligible list, not detected or read from descriptors)
- `HIPDNN_DESCRIPTOR_DIR=<that pack target's OUT_ROOT>/<arch>` -- its own shard, not a shared stage tree
- `HIPDNN_TEST_CENSUS_EXPECTED_CASES=<the pinned case names, comma-separated>`

Run it from the build-tree provider CTest directory:

```bash
ctest --test-dir <build>/dnn-providers/hip-kernel-provider \
  --no-tests=error -V -R '^hip-kernel-provider-hkp-census-<arch>-Test<Name>Packs$'
```

Nonempty `HIPDNN_TEST_CENSUS_SUITE` enables the native strict guard: an existing explicit
descriptor root and a nonempty expected arch are required before default-root setup, and
every registered case must execute and pass without skipping in **every** iteration,
including cases excluded by filters, disable flags or sharding. Listing only, zero
iterations, partial repeated runs, missing suites, wrong-arch data and missing/extra
identities fail. Without that variable, ordinary GoogleTest behavior applies.

**Eligibility follows the shard count, not the dialect.** An entry hands the binary one
directory, so declare a suite only where every case reads one pack target's shard;
`direct_load` qualifies on the same terms as packaged (the shipped `TestPointwisePacks` is
censused although `unit/pointwise/` is `embedded_source`). A suite reading two shards is
censused nowhere and states its inventory through its ordinary host run. Declaring one
suite at two pack targets is a configure error: the entry name carries arch and suite
alone. Host registration/loading proves nothing about dispatch or numerical correctness.

**The generator writes the call for a kpack bundle only.** A packaged bundle lowers to
one production pack target with a known arch list, so the fragment can emit a complete,
arch-restricted, case-pinned call. A direct-load bundle's authored set is chosen by the
author, so its call is hand-added -- and it **omits `ARCHES` entirely**, because its
descriptors declare no architecture and ship on every arch its pack target carries.
Writing `ARCHES *` or a bare `ARCHES` keyword there turns "unrestricted" into a shard
name nothing materializes or into a configure error.

## The five CMake/registration splice points

`fragments/*.txt` are text for a human (or the driving skill's extend flow) to hand-apply;
**nothing is auto-applied**. Each fragment names its splice point in a leading comment:

| Fragment | Splices into |
|---|---|
| `cmake_descriptor_files.txt` | Nothing -- a **statement** that no CMake edit installs descriptors; see below |
| `cmake_target_sources.txt` | `.../kernel_ingestor_engine/CMakeLists.txt`'s `target_sources(hip_kernel_provider_impl ...)` block |
| `ingestor_packs.hpp.txt` | `.../kernel_ingestor_engine/IngestorPacks.hpp` -- the `register<Name>Symbols` declaration |
| `ingestor_packs.cpp.txt` | `.../kernel_ingestor_engine/IngestorPacks.cpp` -- the `s_packs` table row |
| `cmake_test_sources.txt` | `.../src/tests/engines/kernel_ingestor_engine/CMakeLists.txt` -- **two** blocks: the census source into `target_sources(hip_kernel_provider_census_tests ...)`, the matcher source into `target_sources(hip_kernel_provider_tests ...)`. For a kpack bundle it also carries the `hkp_register_census_tests()` call for `.../src/tests/CMakeLists.txt` |

**`cmake_descriptor_files.txt` names no splice target because there is none.** The packer
walks a source root recursively and no descriptor is named in CMake, so installing a bundle
means placing its files under the right root: `test_descriptors/<set>/<slug>/` for
direct-load, `descriptors/<producer>/<bundle>/` for packaged. Each set is a separate pack
target walked by directory, so a bundle in the wrong set installs cleanly and is invisible
to the binary meant to read it. Authors repoint only
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT` and
`HIPKERNELPROVIDER_TEST_DESCRIPTOR_SOURCE_ROOT`.

An `embedded_source` engine additionally needs an
`add_kernels_for_embedding(TARGET … FILES … KEYS …)` entry in
`.../src/tests/CMakeLists.txt`, each `KEYS` value equal to the `source_file` string its
descriptor authors; that belongs to the kernel source, so no fragment carries it. Authored
`hip`, `rocke` and `hsaco` are packaged into the shipped `kpack`.

**Both `IngestorPacks.hpp` and `IngestorPacks.cpp` edits are required.** A pack registered
in the header but missing from the `.cpp` `s_packs` table silently vanishes from the
unit-test binary (the static-archive linker drops an unreferenced object) while still
working in the plugin `.so`.

**The `s_packs` row's `resetModuleCache` follows cache ownership, not the dialect.** A
packaged row names `reset<Name>ModuleCache`, whose generated stub the implementation fills
in to clear its module cache; a packaged row set to `nullptr` still links and passes, and
the cache is never dropped. A `direct_load` row is `nullptr` unless the implementation
caches modules, as `hipkernel:Pointwise` does although its `unit/pointwise/` kernels are
`embedded_source`; that implementation defines its reset outside the pack's anonymous
namespace, declares it in `IngestorPacks.hpp` and names it in the spliced row.

## The generate -> validate round trip

`hipdnn_validate_descriptors` requires a build configured with
`-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` (default **OFF**) and the validator target built; a
missing binary means an unbuilt target, disabled capability or wrong build/install path.
Feed it runtime descriptors -- authored direct-load output or packed per-architecture
output, never unlowered rocKE authoring input.

```bash
# 1. Generate a bundle.
.venv/bin/python generate.py --config configs/scale_add.yaml --output-dir /tmp/scale-add

# 2. Validate it structurally, with no GPU and no linked provider. The path is the
#    bundle's own descriptor directory, which mirrors the authored root: scale_add
#    is direct_load with authored_subpath `unit`.
<build-dir>/bin/hipdnn_validate_descriptors \
    /tmp/scale-add/test_descriptors/unit/scale_add \
    --expect-engine hipkernel:ScaleAdd \
    --json
```

Exit 0 establishes descriptor parsing, cross-references and the expected engine's presence
within the validator's structural scope, and certifies nothing about native registrations,
matcher semantics, compiled specialization or device correctness.

`tests/test_round_trip.py` keeps this as a regression, opt-in under `-m round_trip` because
it needs a validator binary this repo does not build by default: point
`HIPDNN_VALIDATE_DESCRIPTORS` at your build's binary and run
`.venv/bin/python -m pytest -m round_trip`.

## Updating the captured catalog

The descriptors under `src/engines/kernel_ingestor_engine/descriptors/rocKE/<slug>/` are
generated (never at build time) from the selection `configs/<slug>.yaml`, drawn from the
registry space its manifest `configs/<slug>.engine.yaml` enumerates (RUNBOOK stages 2 and
6). `tests/test_registry_regeneration.py` gates the three against each other. The gates
compare specs, not code: `hkp_pack` builds unshipped spec fields at their current default
with the builder's current code, so review rocKE default and builder changes as changes to
the shipped engine.

| Gate | Fails when |
|---|---|
| g1 In registry | a selection kernel is not in the manifest's enumeration (name, shipped spec fields, metadata, `kernel_source` kind/source/builder, `default` marker), or ships an unshipped spec field off its dataclass default |
| g2 Selection regenerates shipped | the shipped descriptors differ from `generate.py`'s output for the selection (`catalog_diff --strict-spec`) |
| g3 Registry regenerates shipped | the shipped descriptors differ from `generate.py`'s output for the manifest's enumeration, compared on the shipped spec fields |
| g4 Header matches manifest | the selection's `engine`, `dialect`, `kernel_source_kind`, `workspace_policy`, `authored_subpath`, `kmd_fields`, `specialization` or `reduction` differs from the manifest's |
| g5 Vocabulary matches the matcher | a `specialization.vocabulary.dtype` spelling is not one the C++ matcher compares, or the matcher's source or dtype literals can no longer be read |

To update a catalog, from this directory: run the gates, diff the edited selection against
the shipped bundle (the report is the change under review), and check with
`catalog_diff.py --subset` that the shipped catalog is a subset of the regenerated registry,
which also lists the registry kernels the selection does not ship (none while the manifest
is `space: slice`):

```bash
PY="$PWD/.venv/bin/python"
SLUG=gfx950_attention_dense
SHIPPED="$PWD/../../../../dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/rocKE/$SLUG"
SCRATCH=$(mktemp -d)
D="descriptors/rocKE/$SLUG/$SLUG"
"$PY" -m pytest tests/test_registry_regeneration.py -q
"$PY" generate.py --config "configs/$SLUG.yaml" --output-dir "$SCRATCH/selection"
# --strict-spec compares the FULL spec dict of every entry (not just the
# intersection of keys in the shipped catalog), catching new spec fields that
# hkp_pack would hydrate differently.  Use it whenever the shipped catalog and
# the regenerated one come from the same enumerate.spec_fields declaration.
"$PY" tools/catalog_diff.py --ignore-ids --strict-spec \
    --shipped "$SHIPPED/$SLUG.kdp.json" --shipped-kmd "$SHIPPED/$SLUG.kmd.json" \
    --regenerated "$SCRATCH/selection/$D.kdp.json" --regenerated-kmd "$SCRATCH/selection/$D.kmd.json"
"$PY" tools/registry_enumerator.py --manifest "configs/$SLUG.engine.yaml" --out "$SCRATCH/registry.yaml"
"$PY" generate.py --config "$SCRATCH/registry.yaml" --output-dir "$SCRATCH/registry"
# --subset only: shipped ⊆ regenerated is the right check for the full registry
# (the registry may contain more tiles than the shipped selection). Its header
# names the regenerated-only spec fields it did not compare.
"$PY" tools/catalog_diff.py --subset --ignore-ids --shipped "$SHIPPED/$SLUG.kdp.json" --shipped-kmd "$SHIPPED/$SLUG.kmd.json" \
    --regenerated "$SCRATCH/registry/$D.kdp.json" --regenerated-kmd "$SCRATCH/registry/$D.kmd.json"
```

Splice the reviewed change into `$SHIPPED` by hand, keeping the UUID of every kernel that
stays, and check it against the pre-splice KDP with `catalog_diff.py --expect-ids`. Ship
any manifest change with it (raise `max_kernels` only deliberately), then rerun the gates.
Adding, removing or renaming kernels is also a heuristic-retraining decision.

## The pipeline tools

`generate.py` emits a bundle; the host tools below inspect distinct contracts. The engine
manifest (`configs/<slug>.engine.yaml`) is an authoring input, not compiler evidence, and
full compiled agreement reads the self-contained producing-build record, needing no rocKE
on the verifying machine.

| Tool | Answers | Invocation |
|---|---|---|
| `tools/verify_variant_sets.py` | Structural nesting/runtime tuple identity, sentinels and vocabulary; artifact-bound compiler agreement is the distinct, stronger mode. Checks every launch of a multi-launch (`sources`) descriptor: a binary's identity is its launches' `(name, builder, spec, role)` tuple, a failure names its launch, and a descriptor with `sources` never nests in a single-launch one | `verify_variant_sets.py --mode {full,structural} [--provenance-root LABEL=DIR]... [--arch A] [--manifest MANIFEST] [--kpack-python-dir D] LABEL ROOT...`. `--mode` is required and has no default: `structural` reports compiled specialization agreement as NOT CHECKED by name and still exits 0 on the rest; `full` fails on a missing, unsupported or mismatched producing-build record **and** on any check that could not run (`GATE FAILED (N check(s) NOT RUN: ...)`). `--manifest` takes the engine manifest: the KDPs whose UED name equals its `engine.name` are the bundles to gate, and every pack of that engine is gated, and its `specialization.vocabulary` is the matcher vocabulary, which full mode requires over string fields no declaration spells out. `--provenance-root` names the sidecar tree mirroring that label's ROOT, needed for an installed tree, which ships no sidecars; an unknown LABEL or a malformed `LABEL=DIR` exits 2. Without `--kpack-python-dir`, full mode imports the installed `rocm_kpack`, then tries `/opt/rocm-kpack/python` (KpackPython.cmake's default); if neither imports it exits **3** with `ENVIRONMENT ERROR`, never a per-kernel gate failure |
| `tools/launch_surface.py` | Is every surface the C++ restates from the kernel's Python declared, guarded and tested? It never compares what the two halves compute | `launch_surface.py MANIFEST --check [--allow-unguarded]` |
| `tools/coverage_gate.py` | Structural, loading and serving obligations, reported separately; an unmet required obligation cannot pass | `coverage_gate.py --tree T --mode {full,structural} [--provenance-root DIR] [--arch A] [--manifest MANIFEST] [--validator V] [--expect-engine E] [--min-served N]`; `--mode` is required and governs what rung 1 may claim. `--manifest` is forwarded to rung 1 as above. A missing `--validator` makes rung 2 `loads-not-run`, a failure rather than a skip; an offline result is not serving evidence. This tool cannot run rung 3, so `--min-served N` with N > 0 fails the gate (`serving-not-run`); `tools/sweep.py`'s `min_served` is the device gate. `--provenance-root` mirrors `--tree`, for an installed tree, and is forwarded to rung 1 |
| `tools/registry_enumerator.py` | Which kernels does the rocKE registry offer this engine? Expands an engine manifest into a generator config; reports how a shape corpus falls on its cells (by cell axes; it does not check that a sequence length can be served); prints the space by knob with exact instance counts | `registry_enumerator.py --manifest M --out C [--shapes S] [--strict]`, or `--manifest M --plan-space [--strict]` (`--plan-space` with `--out` or `--shapes` is a usage error). Exit 2 is unusable input, naming the field: a malformed manifest, a duplicate key, a `ground_truth.value_map` whose keys differ from a cell field's values, or a builder value with no `specialization.vocabulary` mapping; `--strict` makes exit 1 mean an instance count over `aot_budget_per_arch` or a `--shapes` GAP; an over-budget build writes nothing |
| `tools/factorise_config.py` | Compacts an enumerated config into the `variants` form, refusing output that does not re-expand to the input | `factorise_config.py --config C --out S --knobs K1,K2,... [--vocabulary 'field:from=to,...']` |
| `tools/catalog_diff.py` | Is a regenerated catalog the shipped one? Order-blind; UUID-blind with `--ignore-ids`; `--subset` requires only that every shipped kernel is present. An entry is its name, whole `kernel_source` (every key, the spec as below), `arch`, metadata, priority and version. Without `--strict-spec` (and with `--subset`) the spec comparison is keyed on the shipped fields: the header's `spec fields` line lists the regenerated-only fields it did not compare. A malformed entry or KMD exits 2, as does an `--expect-ids` name present in both catalogs used a different number of times | `catalog_diff.py --shipped K --regenerated K2 [--ignore-ids] [--expect-ids] [--strict-spec \| --subset] [--shipped-kmd M --regenerated-kmd M2]` |
| `tools/variant_reduce.py` | Which variants of a selection do autotune rankings justify? Reports the ε curve and per-variant reasons and proposes knob-level rules. Rule ids (`R` and six hex digits) are derived from the rule text, so an id is stable while the text is; a `keep X=v only for <cells>` rule's cells, and so its id, can change with `--budget`; a rule no longer proposed exits 2. Exit 1: over budget, or the applied rules together lose more of some graph's best time than the rules' ε (the report names those graphs); 2: unusable input; 3: a failed correctness verdict, timing phases missing for some (round, corpus), or the reduced arm's sweep is incomplete (no `summary.json` or no usable `expected_phases`, no expected timing phase for the arm, an expected phase with no record, a timing phase with no winners). `--out` is never written on 1 or 3; the report is, except for an incomplete sweep, which writes nothing. A sweep of fewer than 2 rounds warns that its noise floor is assumed (`reduction.single_round_epsilon`) | `variant_reduce.py --selection S --kdp INSTALLED_KDP --sweep-output DIR [--arm A] --out S2 --report R [--budget N] [--apply-rules ID,ID]` |
| `tools/ground_truth.py` | How does hipDNN compare with the kernel owner's numbers? | `ground_truth.py {ingest,graphs,compare}`; see RUNBOOK stages 2 and 6. `graphs` records each graph's semantic identity and its row's digest; `compare` exits 2 when a row or a swept graph no longer matches that record (rerun `graphs`), and otherwise 0 whatever it finds (the report is for review; an accepted exception is a normal outcome); `compare --strict` exits 1 unless every row is within `--margin`, not declined in any timing round, and has a passing correctness verdict (a timing-only sweep, or a row with no usable reference, refuses) |
| `tools/mine_shapes.py` | Build the attention shape corpus, refusing categoricals it does not recognise. | see `--help` |
| `tools/field_audit.py` | Which schema fields are never named as an accessor call in the native sources? A lexical inventory bounding the unchecked set | `field_audit.py SCHEMA.fbs SOURCE...`; prints `UNCHECKED: <field>` per unreferenced field and exits 1. Exits 2 on a schema that parses to zero fields, so an unparsed schema cannot read as a clean audit |
| `tools/splice_additions.py` | Is a scratch render an addition-only change of the live engine, and what does it add? Applies only the additions | `splice_additions.py --scratch S --live L [--report R] [--check]`; see [Extending a live engine](#extending-a-live-engine). Exit 1 on any conflict, with nothing written |
| `tools/inventory.py` | What does an installed (or authored) tree hold for one engine: kernels per KDP, source kinds, distinct metadata tuples, catalog digest, validator verdict? | `inventory.py ROOT --engine E [--group-by F1,F2] [--validator V] [--json J]`. Counts come from `kernelDescriptors`; a KDP without that key fails. The digest is RUNBOOK stage 8's recipe over the engine's bundle directory. Validator diagnostics are folded per message, so a warning repeated once per kernel is one line |

A green tool proves only the properties it checked: missing, unsupported or mismatched
required evidence fails full agreement, and structural-only results satisfy no
compiled-agreement or device gate.

`tools/sweep.py --config <absolute-YAML>` drives measurement with declarative input. Use
`configs/reduction.sweep.yaml.example` and the
[sweep reference](tools/README-sweeps.md) for exact input keys, hazard exclusions, engine
attribution, correctness gates and current-input-bound resume semantics.

### Retired tools

| Retired tool | What it checked | Now |
|---|---|---|
| `dispatch_parity` | a config generated from rocKE's per-shape dispatcher resolution for a hand-written profile | `registry_enumerator.py` plus `catalog_diff.py` (gates g1–g3). Per-shape dispatcher parity is no longer checked; the profile it read was already broken |
| `knob_sweep` | isolate and pairwise sweep configs, planned without measuring | `variant_reduce.py` over a measured sweep (`configs/reduction.sweep.yaml.example`). The planning convenience is gone |
| `reconcile_applicability` | candidate declines against a reference library | none: the runtime ledger and explicit scope decisions (RUNBOOK stage 7) |
| `variant_reachability` | that every shipped variant is selected by some corpus shape, offline | none offline: `variant_reduce.py` reports "never ranked" after a GPU sweep; `registry_enumerator.py --shapes` classifies shapes, not variants |

## Specialization agreement

The producing compiler -- not the generator or a later verifier's installed library -- is
authoritative for effective specialization. Generation carries declarations as data in each
UKD's ignored provenance extension:

```yaml
provenance:
  specialization_contract:
    schema_version: 1
    consumers:
      - engine_id: <UED UUID>
        kmd_id: <KMD UUID>
        metadata_fields: [dtype, use_v_swizzle]
        matcher_only_fields: [layout]
        bindings:
          dtype: {field: dtype}
          use_v_swizzle: {method: resolved_use_v_swizzle}
        vocabulary:
          dtype: {bf16: BF16}
```

That consumer is illustrative; real names and accessors come from the builder's
source-use-site audit. Per consumer:

- `metadata_fields` and `matcher_only_fields` exhaustively and disjointly partition the
  referenced KMD fields.
- `bindings` keys are exactly `metadata_fields`, each value exactly `{field: "<attr>"}` or
  `{method: "<accessor>"}`, read off the hydrated spec object passed to the builder.
- IDs reference existing descriptors; KMD types and defaults are not duplicated. A UKD
  shared by engines carries each consumer's entry; duplicate or conflicting entries fail.
- A direct field is legal only where the builder consumes it without further resolution.
  Where it consumes an effective accessor, read that zero-argument bound method **even if
  the raw field is non-null**: coupling can override explicit values. Missing or
  noncallable readouts, exceptions, unsupported values, unresolved `None` and
  non-repeatable resolution block full agreement.
- Metadata completion and comparison use the referenced KMD's defaults and types: BOOL
  stays boolean, a builder boolean may deliberately project to 0/1 for INT, FLOAT values
  are canonicalized numerically.

`None` is authored intent, **never a compiled-artifact wildcard**. Authored
`provenance.spec` stays distinct and preserved; only the producing compiler writes
`provenance.effective_spec`, and authored inputs supplying that reserved record are
rejected. Matcher-only classification requires a source-use-site audit and independent
review; a consumed field with no authoritative binding stays unsupported.

Packaging observes the actual builder object and compares **every consumer** independently
before publication, recording effective values and declaration digests, authored inputs,
producer identities and origins, descriptors, KMD content, completed metadata, architecture
and library/toc-key/symbol/payload hashes; authored passthrough never overwrites fresh
observations. That record, like all of a packed UKD's provenance, ships in the
descriptor's type-named sidecar (`foo.kdp.provenance.json.gz` beside `foo.kdp.json`,
format v1), bound to its UKD by `ukd_sha256`, not inline. Every checker reads a
descriptor as packed exactly when its directory holds the packer's empty
`hkp-packed.marker`, which ships with the runtime tree; a copied tree without it reads
as authored. The runtime package carries no sidecars; with tests enabled they install
under `test_arch_content/hip-kernel-provider/provenance/`, mirroring the descriptor
tree, and a checker pointed at an installed tree takes that folder through
`--provenance-root`, without which a marked descriptor fails on its missing sidecar.
Full checking verifies that record against the current descriptors and named payload
bytes without importing rocKE on the verifier,
structural-only checking cannot supply missing compiler agreement, and neither proves
machine-code equivalence, native semantics or numerical correctness -- see the
[packaging reference](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).
There is no packaging `--profile`, CMake `PROFILES` or external root manifest.

## Configs

| Config | Shape |
|---|---|
| `configs/scale_add.yaml` | Single-pack engine, like the shipped `conv_fwd`: one pack, one operation, its `graph_match` both admits the node type and validates shape |
| `configs/binary_ops.yaml` | Multi-pack engine, like the shipped `pointwise`: one pack per operation sharing one KMD/UED/UHD/UDD, each naming its own operation-scoped UMD via `discriminator` |
| `configs/gfx950_attention_dense.yaml` | The authored selection for the shipped `hipkernel:Gfx950AttentionDense` descriptors: 840 rocKE kernels in the `variants` form, a builder lowered by `hkp_pack` at build time |
| `configs/gfx950_attention_dense.engine.yaml` | That engine's manifest: the registry space the selection is drawn from, its `reduction:` classification, ground-truth graph template and `launch_surface:` audit |
| `configs/axes_example.yaml` | Pack-level `axes`: one `kernel_template` crossed with value lists, expanded at load time |
| `configs/variants_example.yaml` | Pack-level `variants`: a shape list crossed per-shape with a named knob set (below) |
| `configs/gfx1151_wmma_attention.yaml` | The shipped `hipkernel:Gfx1151WmmaAttention` catalog: packaged rocKE runtime-shape kernels under `rocKE/gfx1151_wmma_attention` |
| `configs/gfx950_attention_dense.profile.yaml` | Not a generator config: the gfx950 dense attention profile that the `tools/` scripts read through `--profile` |

## Generated variant sets: `variants`

An enumerated gfx942 attention_dense set runs to **89,265 lines for 2,710 kernels**;
`variants` states it in **about 1,150 lines**, generating byte-identical descriptors:

```yaml
packs:
  - name: attention_dense
    kernel_defaults:                        # constant across every kernel
      kind: rocke
      source: kernels/gfx942/attention_dense.py
      builder: build_attention_dense
    variants:
      - name: dense.{dtype}_sq{seqlen_q}_bm{block_m}_{tag}
        metadata: [dtype, seqlen_q, block_m, use_exp2_fast]
        vocabulary: {dtype: {bf16: BF16}}     # the spelling the MATCHER compares
        policy_knobs: [use_exp2_fast]         # the kernel's policy decides these
        spec_order: [dtype, seqlen_q, block_m]  # key order reaches the descriptor
        spec_defaults: {block_n: 64}          # constant across THIS group
        knob_sets:
          pair:
            - {block_m: 128, tag: 'e{md_use_exp2_fast}'}
            - {block_m: 256, use_exp2_fast: false, tag: ed}
        shapes:
          - {dtype: bf16, seqlen_q: 512, knobs: pair, resolved: {use_exp2_fast: 1}}
```

`configs/variants_example.yaml` is the runnable version, exercising every key.

**Why not `axes`.** `axes` crosses ONE `kernel_template`, and a dispatcher-derived set has
none: `registry_enumerator.py` asks the registry for specs per cell, so every entry
carries its own resolved values for the fields the dispatcher derives.

**It is not a grid.** Each shape names its own knob set; on the shipped sets most shapes
carry four arms and 63 carry six.

**Shared shape lists.** Where every shape of a group repeats the same values, the group's
`shape_axes` states them once: a mapping of spec field to a non-empty value list, crossed
with each shape first axis outermost (`shape_axes: {dtype: [fp16, bf16], head_size: [64,
128]}` with one shape `{knobs: rt}` expands to four shapes). An axis must name a spec field
of the group, and a shape may not restate an axis field.
`configs/gfx1151_wmma_attention.yaml` keeps its head-size bucket list there.

**The tri-state.** An omitted or null policy knob retains the builder's policy intent, not
an explicit false. `resolved` supplies an authored metadata projection, not compiler
evidence; final composition includes pack defaults before projection, and the producing
compiler compares metadata against the declared effective readout. A metadata override is
legal only under its reviewed binding or matcher-only classification.

**Names.** The name must encode everything that varies; the loader rejects an expansion
producing two kernels with the same name. A slot is a spec field, an `md_<field>` metadata
mirror, the arm's `{tag}`, or `{ordinal}`. A shape sets its serial with `ordinal:`;
each arm shifts it with `ordinal_offset:`.

**Arch restriction.** A group may carry its own `arch`, emitted as each expanded kernel's
UKD `arch` exactly like a hand-authored kernel's: it must be a subset of the pack's, and an
absent one inherits the pack's. Use it for kernels only some of the pack's architectures
can run -- `configs/gfx1151_wmma_attention.yaml` restricts its transposed-QK group to the
gfx11 shards of a gfx11+gfx12 pack.

Expansion runs at load time (`codegen/config_loader.py`), so `generate.py`, the emitters and
the dedup pass see ordinary kernel dicts. `tools/registry_enumerator.py` emits the enumerated form;
`tools/factorise_config.py` converts an enumerated config, re-expanding its own
output and refusing to write anything that does not reproduce the input kernel-for-kernel.

## Config surface

```yaml
engine:
  name: hipkernel:MyEngine        # required, scoped namespace:local. The LOCAL half
                                  # must derive a valid C++ identifier and a valid
                                  # single path stem: PascalCase, snake_case and
                                  # kebab-case work; a '.' or a leading digit is
                                  # rejected here though the scoped-name regex takes it.
  sdk_version: "1.0.0"            # optional, three components, default "1.0.0"
  behavior_notes: [runtime_compilation]   # optional, closed vocabulary
  knobs: [block_size]             # optional; must all be int-typed kmd_fields
  heuristic: native | none        # optional, default "native"; "none" omits the UHD

kmd_fields:                       # the KMD's fields[] -- one per axis this engine's
  - name: block_size              # kernels vary along
    type: int                     # bool | int | float | string | int_list
    default_value: 64             # omit entirely for a MANDATORY field
  - name: dtype
    type: string

graph_match:                      # documentation of shape, not consumed by templates
  shape: shared_shape | disjoint_attributes
  discriminator: none | field_value | disjoint_topology

dialect: direct_load | packaged   # optional, default "direct_load"
kernel_source_kind: embedded_source   # direct-load example; packaged sources use
                                        # their build-time source kind: hip | rocke |
                                        # hsaco
authored_subpath: unit            # REQUIRED for direct_load, naming one of the four
                                    # authored sets: shared | unit | integration |
                                    # archive_fixture. Each is a separate pack target
                                    # reaching a different binary. Optional for
                                    # packaged, defaulting to <kernel_source_kind>/<slug>;
                                    # must be RELATIVE and stay under descriptors/.
workspace_policy: none | fixed | derived

packs:
  - name: add
    arch: [gfx942]                # optional; empty means arch-independent
    discriminator: add             # REQUIRED iff this engine has >1 pack; forbidden
                                    # for a single-pack engine
    kernels:
      - name: my_engine.f32_block64
        kernel_source:
          kind: embedded_source
          source_file: MyEngine.cpp
          entry_point: MyEngine
        metadata: { block_size: 64, dtype: FLOAT }
        priority: 0
        arch: []                   # optional; must be a subset of the pack's arch
```

### The engine manifest and `reduction:`

A rocKE registry-backed engine's `configs/<slug>.engine.yaml`, expanded by
`tools/registry_enumerator.py`, carries the generator config's engine header plus:

```yaml
pack: gfx950_attention_dense      # the one pack's name and slug
arch: gfx950                      # must equal enumerate.fixed.arch
provider_root: dnn-providers/hip-kernel-provider   # rocke/library is imported from here
source: kernels/gfx950/attention_dense.py   # the builder; no candidate exposes it
builder: build_attention_dense
kernel_name_template: attention_dense.{dtype}_d{head_size}_bm{block_m}_bn{block_n}.gfx950

enumerate:
  candidates: dispatch.attention:attention_candidates  # module:function -> KernelCandidate tuple
  request: dispatch.attention.common:AttentionRequest  # module:Class, the OperatorRequest
  pin: {algorithm: attention_dense_grid, spec_id: gfx950_dense_grid}  # selects ONE candidate
  fixed: {arch: gfx950, batch: 1, seqlen_q: 512, seqlen_k: 512}  # request fields, every cell
  cells:                          # crossed, first group outermost, values in order
    - fields: [hdim_q, hdim_v, nhead_q, nhead_k]
      values: [[64, 64, 8, 1], [128, 128, 8, 8]]
    - fields: [dtype]             # a one-field group may list scalars
      values: [bf16, fp16]
    - fields: [mask_type]         # enum request fields take member names
      values: [TOP_LEFT_CAUSAL, NO_MASK]
  sweep_level: production         # the registry sweep level
  space: slice | full             # full: every sweep spec sharing the cell's feature fields
  free_fields: [block_m, block_n] # slice only: fields that may differ from select_spec
  spec_fields: [dtype, head_size] # slice only: spec fields the KDP carries
  max_kernels: 840                # slice only: more is an error

reduction:                        # copied into the selection; read by tools/variant_reduce.py
  fields:
    feature: [dtype, head_size]   # spec/metadata fields that define a cell
    performance:                  # tuning knobs, in priority order
      - {name: block_m, divides: seqlen_q}   # applicable when seqlen_q % block_m == 0; the target must be a kmd_fields name
      - {name: exp_per_pv_step, unset_resolves_by: resolved_exp_per_pv_step}  # resolves a None knob
    ignored: []                   # held at the registry default, no warning
  aot_budget_per_arch: 1000       # optional, default 1000; generate.py and registry_enumerator.py warn above it, --strict fails
  single_round_epsilon: 0.05      # optional; variant_reduce's assumed noise floor for a one-round sweep
  drift_multiplier: 2             # optional; floor = multiplier x measured round-to-round drift
                                  # both: finite numbers >= 0

ground_truth:                     # optional; read by tools/ground_truth.py graphs
  graph_template: templates/attention_sdpa_fwd.graph.json   # manifest-relative; "stride_order" orders packed strides
  value_map:                      # request value -> template scalar, or patches
    dtype: {bf16: bfloat16, fp16: half}
    # A patch pointer's list segments are element names (preferred) or indices.
    mask_type: {NO_MASK: {/nodes/sdpa_node/attributes/causal_mask: false}}
```

Each cell's request is `Request(**fixed, **pin, **cell)`; a declined cell exits 2. Metadata
fields and name slots the spec lacks are read from the request. A field sits in at most
one `reduction.fields` list; in `space: full`, a varying field in none is held at its
registry default with a WARN. `reduction:` never reaches a descriptor; a selection's arm
or flat kernel may carry `default: true`, the variant its cell always keeps. An optional
`launch_surface:` block feeds `tools/launch_surface.py`.

A packaged `rocke` `kernel_source` (or the pack's `kernel_defaults`, which every kernel
inherits) may set `target: native | generic`, the processor `hkp_pack` compiles each shard
for: `native` (the default when absent) is the shard arch, `generic` its LLVM generic
processor (`gfx11-generic`, `gfx12-generic`), so one kernel set serves the whole family.
Any other value is a `ConfigError`; the key is rejected for every other kind. See the
[packaging reference](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).

A packaged `rocke` `kernel_source` (or `kernel_defaults`) may also set
`launches: {<name>: {<overrides>}}` to declare the launches of a multi-stage kernel
([RFC 0017 §15.1](../../docs/rfcs/0017_UniversalKernelDescriptor.md)). Each value is merged
onto the base `kernel_source`: `{role: merge}` selects a companion kernel of the same
builder's `KernelModule`, and a different `builder` and `spec` name another builder
entirely. The generator then emits the authored descriptor with a `sources` map keyed by
launch name, in place of `kernel_source`, and the descriptor declares UKD 1.1. Launch names
are lowercase identifiers (`[a-z][a-z0-9_]*`), and at least two are required. Anything else
is a `ConfigError`, and so is the key on any kind but `rocke`. Two launch names may name
one kernel (a stage that runs one kernel twice with different arguments), so equal
launches are valid config. A kernel's launches are part
of its candidate identity, so two entries that differ only in their launches are not
de-duplicated. A config with no `launches` key generates exactly what it did before.

A launch's `spec` override **replaces** the spec for that launch and is never varied by
`variants`, `axes` or `shape_axes`: it is the same static spec in every expanded kernel. It
may not state a metadata field at a value that differs from the variant being generated,
which is a `ConfigError` naming the variant and the field, so a launch whose spec depends
on an expanded axis cannot be spelled this way. A launch with no `spec` override takes the
variant's own expanded spec. The packer checks each declared consumer field against the
launches that report it (see the packaging reference).

The descriptor only lists the launches: the pack's native dispatch handler declares the
names it launches and owns their order, and the provider drops a kernel whose launch names
differ from the handler's when it builds its catalog. One engine has one handler, so every
kernel of an engine must name the same set of launches: a config mixing single- and
multi-launch kernels, or two different sets, is a `ConfigError` (split the kernels into
engines by launch set). See the
[packaging reference](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md#multi-launch-kernels-sources).

## The five pre-mint config-loader checks

Run, in this order, **before any UUID is minted**:

1. `engine.name` matches the scoped `namespace:local` regex.
2. Every `engine.knobs` entry names a declared **and int-typed** `kmd_fields` entry: the
   real loader takes a non-int knob and silently yields no usable knob, visible only at
   plan-build time against a real device.
3. Every kernel's `metadata` type-checks against the KMD, with no mandatory field (one with
   no `default_value`) omitted; the real loader drops the whole pack instead.
4. Every kernel's `arch` is a subset of its pack's `arch`.
5. Every `arch` entry is a plausible `gfx`-prefixed base id (lowercase, no feature suffix):
   an error if malformed, a **warning** if well-formed but unrecognized (e.g. `gfx94` for
   `gfx942`), since either looks like an ordinary INFO decline at match time and this tool
   keeps no exhaustive arch list.

### Generic targets

Configs accept generic `arch` targets under the
[packaging rules](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md#generic-gpu-targets).
`build_kdp` refuses a tuple shared by two kernels only at the same arch tier
([RFC 0017](../../docs/rfcs/0017_UniversalKernelDescriptor.md#81-applicability-once-per-engine)). The generated census gives every member of
a generic row the generic's descriptors.

## Source adapters (`codegen/sources/`)

One protocol (`SourceAdapter.infer(*sources) -> SourceAdapterResult`), three
implementations:

- `InteractiveAdapter` -- no inference; a human or the driving skill fills every field.
- `HiprtcAdapter` -- scans `.cpp`/`.hip` files for `extern "C" __global__` entry points and
  candidate KMD fields (externally-supplied `HIP_PLUGIN_*` defines, template parameters).
- `RockeAdapter` -- introspects a rocKE builder (`source`/`builder` pair) and reports its
  spec dataclass fields as candidate KMD fields. rocKE is imported lazily; an
  unreadable builder raises `RockeIntrospectionError`.

`rocke` authoring uses the packaged path, its effective policy observations belonging to the
producing compiler. `hsaco` authoring also uses the packaged path: `kernel_source` is
`{kind: hsaco, file, symbol}`, `file` naming a prebuilt code object relative to the
descriptor that names it and `symbol` its kernel. `file` must stay inside the source root,
with no root-relative fallback. `hkp_pack` packs that object as-is, without compiling, so
like `hip` the specialization declares `metadata_fields: []`. The packer does not check the
object's format or target processor; `arch` follows
[hsaco kernels and `arch`](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md#hsaco-kernels-and-arch).
`hsaco_file` is rejected explicitly, naming `supportsSourceKind()` as the missing
prerequisite on `IKernelDispatchHandler`.

## Tests

This suite is **developer-run**, not registered with CTest or run by superbuild CI. Cases
needing `HIPDNN_VALIDATE_DESCRIPTORS` (`-m round_trip`) skip when it is unset; the
real-archive class above skips only when `HIPKERNELPROVIDER_ROCM_KPACK_DIR` is unset and
`rocm_kpack` is not importable.

```bash
.venv/bin/python -m pytest
```

That form collects no coverage, so the `fail_under = 80` floor `pyproject.toml` configures
is never applied. To enforce it:

```bash
.venv/bin/python -m pip install pytest-cov   # not in requirements.txt
.venv/bin/python -m pytest --cov --cov-report=term-missing
```

The floor covers `codegen/` alone. `tools/` and `generate.py` sit outside the configured
`source`: the tools are gated by the suites that drive each script as a subprocess
(`tests/test_sweep_tools.py`, `tests/test_launch_surface.py`, `tests/test_coverage_gate.py`,
`tests/test_verify_variant_sets.py`, `tests/test_device_probe.py`,
`tests/test_registry_enumerator.py`, `tests/test_splice_additions.py`,
`tests/test_inventory.py`), and the spawned CLI is traced only under
`COVERAGE_PROCESS_START` plus a `coverage.process_startup()` `.pth`.

The suite exercises descriptor identities, declaration carriage, semantic deduplication,
emitted inventory and CLI outcomes. Native registration, loading and runtime correctness
require the compiled provider and create/extend execution gates.

### Native-stub compilation (`tests/test_native_stub.py`)

```bash
.venv/bin/python -m pytest tests/test_native_stub.py
```

`TestRealCompile` host-compiles emitted single-pack, multi-pack, packaged-dialect and
matcher-test stubs with `g++`/`clang++`, plus an intentionally broken source control.
`-fsyntax-only` proves parsing and type-checking only. The fixture uses the
plugin/data/flatbuffers SDK and provider sources beside this checkout plus
`flatbuffers/array.h` under `/opt/rocm/include`, and generates stand-ins for the
CMake-configured `version.h`/`CacheRootDefaults.h` headers. Missing prerequisites skip
explicitly, so a checkout with no compiler or ROCm headers still runs green. Setting
`HIPDNN_INGESTOR_STUB_INCLUDE_FILE` to a file of include directories (one per line,
containing `flatbuffers/array.h` and `hip/hip_runtime.h`) replaces the
`/opt/rocm/include` probe, and a bad value then fails rather than skips.

Under `HIPDNN_ENABLE_TOOLING_TESTS` the hipDNN build runs this suite with the
`tools/pytest_plugins/hipdnn_strict_skips.py` plugin (`-p hipdnn_strict_skips`), which turns
every skip into a failure, so a missing compiler, validator binary, `hkp_pack` or
`rocm_kpack` fails CI instead of passing by not running. A skip that is correct by design
(an author-supplied input such as `HIPDNN_INGESTOR_SHAPES` or an authoring profile, or the
POSIX-only sweep tests on Windows) carries `@pytest.mark.allow_skip(reason="...")` beside the
skip. Run by hand, the plugin is not loaded: `python -m pytest -p hipdnn_strict_skips` (with
`tools/pytest_plugins` on `PYTHONPATH`) reproduces the CI behavior.

### Fragment contracts (`tests/test_fragment_contracts.py`)

```bash
.venv/bin/python -m pytest tests/test_fragment_contracts.py
```

Contracts between the emitted fragments, checked against each other:

- `IngestorPacks.hpp` and the `IngestorPacks.cpp` `s_packs` row name the same register
  function.
- A packaged engine carries a reset pointer whose symbol is defined somewhere; a
  direct-load engine defines none and leaves it null.
- `cmake_test_sources.txt` names files this run wrote.
- The census case pin is exactly the case set the emitted suite renders, follows the suite's
  conditional arms, and survives the wire to the binary.
- The emitted `ARCHES` restriction is exactly the concrete architectures of the emitted
  inventory: never the `*` wildcard, and never the keyword with nothing after it. A
  direct-load bundle's documented call carries no `ARCHES` at all.
- The placeholder scan sees every emitted file across the two provider trees, reports an
  unlocatable one as missing, and treats two files at one spliced path as ambiguous.

Agreement with the provider's own headers is deliberately not checked here: the compiler
catches that at the splice.
