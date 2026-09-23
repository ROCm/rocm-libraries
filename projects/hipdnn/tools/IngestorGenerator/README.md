# hipDNN IngestorGenerator

Generates a hipDNN generic-kernel-ingestor descriptor bundle from a YAML config:
KMD/UED/UMD/UHD/UDD/KDP JSON, a native-symbol stub, a complete pack-shape census test, a
matcher-test stub, and five CMake/registration text fragments. It follows
`projects/hipdnn/tools/DescriptorGenerator`'s conventions with two deviations:
`undefined=StrictUndefined`, so an unset UUID cross-reference fails at generation time,
and a required `--force` to overwrite a non-empty output directory. Generate into scratch
space, including when extending an engine, and splice addition-only.

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

### The optional archive dependency

`tests/test_verify_variant_sets.py`'s `TestRealArchiveSelectedConsumer` writes a real kpack
archive and needs `rocm_kpack`, which ships in the rocm-kpack checkout rather than on PyPI.
Point `HIPKERNELPROVIDER_ROCM_KPACK_DIR` at that checkout's `python` directory, the same one
the packaging suite's `PYTHONPATH` entry and `--kpack-python-dir` name:

```bash
export HIPKERNELPROVIDER_ROCM_KPACK_DIR=/opt/rocm-kpack/python
```

Unset -- or exported empty, which counts as unset -- that class **skips**; set but wrong
fails loudly, because naming one is a request to run the class. See the
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
```

Exit codes: `0` success; `1` on a `ConfigError` or a template-rendering failure;
`2` from argparse itself on a bad flag.

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
builder kind.

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
descriptor authors; that belongs to the kernel source, so no fragment carries it. The
`hip`, `rocke`, `hsaco` and `kpack` kinds lower at pack time.

**Both `IngestorPacks.hpp` and `IngestorPacks.cpp` edits are required.** A pack registered
in the header but missing from the `.cpp` `s_packs` table silently vanishes from the
unit-test binary (the static-archive linker drops an unreferenced object) while still
working in the plugin `.so`.

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

## The pipeline tools

`generate.py` emits a bundle; the host tools below inspect distinct contracts. Profiles are
optional analysis or authoring inputs, not compiler evidence, and full compiled agreement
reads the self-contained producing-build record, needing no rocKE on the verifying
machine.

| Tool | Answers | Invocation |
|---|---|---|
| `tools/verify_variant_sets.py` | Structural nesting/runtime tuple identity, sentinels and vocabulary; artifact-bound compiler agreement is the distinct, stronger mode | `verify_variant_sets.py --mode {full,structural} [--arch A] [--profile P] [--kpack-python-dir D] LABEL ROOT...`. `--mode` is required and has no default: `structural` reports compiled specialization agreement as NOT CHECKED by name and still exits 0 on the rest; `full` fails on a missing, unsupported or mismatched producing-build record **and** on any check that could not run (`GATE FAILED (N check(s) NOT RUN: ...)`). `--profile` supplies the bundle to gate and the matcher vocabulary, which full mode requires over string fields no declaration spells out |
| `tools/variant_reachability.py` | Can any shape in the corpus actually select each variant, or is one dead weight? | `variant_reachability.py --kdp K --shapes S [--profile P]` |
| `tools/launch_surface.py` | Is every surface the C++ restates from the kernel's Python declared, guarded and tested? | `launch_surface.py PROFILE --check [--allow-unguarded]` |
| `tools/coverage_gate.py` | Structural, loading and serving obligations, reported separately; an unmet required obligation cannot pass | `coverage_gate.py --tree T --mode {full,structural} [--arch A] [--validator V] [--expect-engine E] [--min-served N]`; `--mode` is required and governs what rung 1 may claim. A missing `--validator` makes rung 2 `loads-not-run`, a failure rather than a skip; an offline result is not serving evidence |
| `tools/knob_sweep.py` | Which knob arms are worth measuring, isolation first then pairwise. | `knob_sweep.py --profile P --shapes S [--plan]` |
| `tools/dispatch_parity.py` | Do the emitted descriptors match what the kernel's real dispatcher resolves? | see `--help` |
| `tools/reconcile_applicability.py` | Does this engine decline anything the reference library serves? | `reconcile_applicability.py --profile P --shapes S [--declines D]` |
| `tools/mine_shapes.py` | Build the shape corpus, refusing categoricals it does not recognise. | see `--help` |
| `tools/field_audit.py` | Which schema fields are never named as an accessor call in the native sources? A lexical inventory bounding the unchecked set | `field_audit.py SCHEMA.fbs SOURCE...`; prints `UNCHECKED: <field>` per unreferenced field and exits 1. Exits 2 on a schema that parses to zero fields, so an unparsed schema cannot read as a clean audit |

A green tool proves only the properties it checked: missing, unsupported or mismatched
required evidence fails full agreement, and structural-only results satisfy no
compiled-agreement or device gate.

`tools/sweep.py --config <absolute-YAML>` drives measurement with declarative input. Use
`configs/sweep-isolation.sweep.yaml.example` and the
[sweep reference](tools/README-sweeps.md) for exact input keys, hazard exclusions, engine
attribution, correctness gates and current-input-bound resume semantics.

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
observations. Full checking verifies that record against the current descriptors and named
payload bytes without importing rocKE on the verifier, structural-only checking cannot
supply missing compiler agreement, and neither proves machine-code equivalence, native
semantics or numerical correctness -- see the
[packaging reference](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).
There is no packaging `--profile`, CMake `PROFILES` or external root manifest.

## Configs

| Config | Shape |
|---|---|
| `configs/scale_add.yaml` | Single-pack engine, like the shipped `conv_fwd`: one pack, one operation, its `graph_match` both admits the node type and validates shape |
| `configs/binary_ops.yaml` | Multi-pack engine, like the shipped `pointwise`: one pack per operation sharing one KMD/UED/UHD/UDD, each naming its own operation-scoped UMD via `discriminator` |
| `configs/gfx950_attention_dense.yaml` | Packaged rocKE kernel: a builder lowered by `hkp_pack` at build time |
| `configs/axes_example.yaml` | Pack-level `axes`: one `kernel_template` crossed with value lists, expanded at load time |
| `configs/variants_example.yaml` | Pack-level `variants`: a shape list crossed per-shape with a named knob set (below) |

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
none: `dispatch_parity.py` asks the library for a spec per shape, so every shape carries its
own resolved values for the fields the dispatcher derives.

**It is not a grid.** Each shape names its own knob set; on the shipped sets most shapes
carry four arms and 63 carry six.

**The tri-state.** An omitted or null policy knob retains the builder's policy intent, not
an explicit false. `resolved` supplies an authored metadata projection, not compiler
evidence; final composition includes pack defaults before projection, and the producing
compiler compares metadata against the declared effective readout. A metadata override is
legal only under its reviewed binding or matcher-only classification.

**Names.** The name must encode everything that varies; the loader rejects an expansion
producing two kernels with the same name. A slot is a spec field, an `md_<field>` metadata
mirror, the arm's `{tag}`, or `{ordinal}`. A shape sets its serial with `ordinal:`;
each arm shifts it with `ordinal_offset:`.

Expansion runs at load time (`codegen/config_loader.py`), so `generate.py`, the emitters and
the dedup pass see ordinary kernel dicts. `tools/dispatch_parity.py` emits this form
directly; `tools/factorise_config.py` converts an enumerated config, re-expanding its own
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
                                        # their build-time source kind
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

## Source adapters (`codegen/sources/`)

One protocol (`SourceAdapter.infer(*sources) -> SourceAdapterResult`), two v1
implementations:

- `InteractiveAdapter` -- no inference; a human or the driving skill fills every field.
- `HiprtcAdapter` -- scans `.cpp`/`.hip` files for `extern "C" __global__` entry points and
  candidate KMD fields (externally-supplied `HIP_PLUGIN_*` defines, template parameters).

`rocke` authoring uses the packaged path, its effective policy observations belonging to the
producing compiler. `hsaco_file` is rejected explicitly, naming `supportsSourceKind()` as the
missing prerequisite on `IKernelDispatchHandler`.

## Tests

This suite is **developer-run**, not registered with CTest or run by superbuild CI. Cases
needing `HIPDNN_VALIDATE_DESCRIPTORS` (`-m round_trip`) or `HIPKERNELPROVIDER_ROCM_KPACK_DIR`
(the real-archive class above) skip when those are unset.

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
`tests/test_dispatch_parity.py`), and the spawned CLI is traced only under
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
explicitly.

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
