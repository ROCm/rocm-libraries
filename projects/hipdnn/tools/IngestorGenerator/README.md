# hipDNN IngestorGenerator

Generates a hipDNN generic-kernel-ingestor descriptor bundle -- KMD/UED/UMD/UHD/UDD/KDP
JSON, a native-symbol stub, complete pack-shape census tests, a matcher-test stub, and
six CMake/registration text fragments -- from a YAML config. Modeled on
`projects/hipdnn/tools/DescriptorGenerator`'s conventions, with two deliberate
deviations: `undefined=StrictUndefined` on the Jinja2 environment (an unset UUID
cross-reference fails loudly at generation time, not as a confusing empty-string
rejection at load time), and a required `--force` flag to overwrite a non-empty output
directory. Generate into scratch space, including when extending an engine; preserve
existing IDs and hand-written bodies through addition-only splicing.

Generic generation is toolchain-free and issues **no compiler evidence**. The test
suite checks generation behavior; a complete integration also needs artifact
agreement, real native loading and engine-attributed numerical device proof. The
[ingestor RUNBOOK](../ai/skills/hipdnn-ingestor-engine/RUNBOOK.md) owns the only
ordered create/extend procedure.

## Prerequisites

- Python 3.10+
- PyYAML >= 6.0
- Jinja2 >= 3.1

## Setup

```bash
cd projects/hipdnn/tools/IngestorGenerator
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

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

```
<output-dir>/
  descriptors/<engine_slug>/
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
  fragments/*.txt                   # 6 CMake/registration fragments, see below
```

### UMD policy

A UMD is emitted **only** for genuine per-pack narrowing. Topology/shape/dtype
applicability belongs in the UED's `graph_match`, not a UMD -- PR #10839 deleted eight
UMDs that existed only to carry a topology gate. Concretely:

- A **single-pack** engine gets **zero** graph-scoped UMDs. Its one pack's
  `matchers[]` names only the shared kernel-scoped dtype matcher.
- A **multi-pack** engine gets **one** graph-scoped operation-matcher UMD per pack
  (each pack's config entry must set a unique `discriminator`), plus the same one
  shared kernel-scoped matcher every pack references.

### Native-symbol stub shape

`packs/<Name>Native.cpp` declares symbol names and registers hooks through typed
`SymbolScope<Handle>` calls in `register<Name>Symbols()`. Fill every applicable
placeholder. Returning `std::nullopt` from `graph_match` empties the **whole**
engine catalog; it is not a per-candidate decline. A heuristic-disabled engine has
no scorer declaration, implementation or registration and no UHD.

Native proof executes the provider's actual registration and descriptor-loading
path. Matching strings in C++ source cannot establish a hook's type, presence or
uniqueness and is not certification.

### Matcher-test stub

`tests/Test<Name>Matchers.cpp` constructs its `DeviceProperties` fixture **by value**,
never by querying the host (`hipGetDeviceProperties`/`getDeviceProperties()`), with a
comment explaining why: a test that queries the host is vacuous on every arch except
whatever happens to be running CI (`TestAsmSdpaForwardMatchers.cpp:27-33`).

### Pack-shape census test

`tests/Test<Name>Packs.cpp` exercises `discoverDescriptorSets()` and the provider's
typed registration/loading path against the finalized emitted inventory. Expected
pack/kernel identities, counts, SDK version and runtime source kind derive from the
actual output, after normalization and deduplication. Packaged runtime source kind
is KPACK, not its authored builder kind.

For packaged engines, append the literal `Test<Name>Packs` suite from
`cmake_test_sources.txt` to `HKP_CENSUS_TEST_SUITES`. CMake registers a separate
`hip-kernel-provider-hkp-census-<arch>-Test<Name>Packs` for every configured packaging
architecture. Each runs `hip_kernel_provider_tests --gtest_filter=Test<Name>Packs.*`
directly, without a Python launcher, with:

- `HIPDNN_TEST_CENSUS_SUITE=Test<Name>Packs`
- `HIPDNN_TEST_EXPECTED_ARCH=<arch>` (configured, not detected or read from descriptors)
- `HIPDNN_DESCRIPTOR_DIR=<descriptor-build-dir>/<arch>`

Run the registered obligation from the build-tree provider CTest directory:

```bash
ctest --test-dir <build>/dnn-providers/hip-kernel-provider \
  --no-tests=error -V -R '^hip-kernel-provider-hkp-census-<arch>-Test<Name>Packs$'
```

Nonempty `HIPDNN_TEST_CENSUS_SUITE` enables the native strict guard. It requires an
existing explicit descriptor root and a nonempty expected arch before default-root
setup. Every registered case in the exact, nonempty suite must execute and pass
without skipping in **every** iteration, including cases excluded by filters,
disable flags or sharding. Listing only, zero iterations, partial repeated runs and
missing suites fail; complete repeated iterations pass. Wrong-arch data and
missing/extra identities fail in the generated inventory checks.

Direct-load engines retain ordinary host suites; do not add them to the packaged
census list. Supply their expected arch and direct-load descriptor root explicitly.
Normal invocations without the census-suite variable keep ordinary GoogleTest
filtering and skip behavior. Host registration/loading does not prove graph dispatch
or numerical device correctness.

## The five CMake/registration splice points

`fragments/*.txt` are text for a human (or the driving skill's extend flow) to
hand-apply; **nothing here is auto-applied**. Each fragment names its own splice point
in a leading comment:

| Fragment | Splices into |
|---|---|
| `cmake_descriptor_files.txt` | `dnn-providers/hip-kernel-provider/CMakeLists.txt`'s `HIPDNN_DESCRIPTOR_FILES` list |
| `cmake_ingestor_kernels.txt` | `.../kernel_ingestor_engine/IngestorKernels.cmake`'s `HIPDNN_INGESTOR_PACK_KERNELS` list |
| `cmake_target_sources.txt` | `.../kernel_ingestor_engine/CMakeLists.txt`'s `target_sources(hip_kernel_provider_impl ...)` block |
| `ingestor_packs.hpp.txt` | `.../kernel_ingestor_engine/IngestorPacks.hpp` -- the `register<Name>Symbols` declaration |
| `ingestor_packs.cpp.txt` | `.../kernel_ingestor_engine/IngestorPacks.cpp` -- the `s_packs` table row |
| `cmake_test_sources.txt` | `.../src/tests/engines/kernel_ingestor_engine/CMakeLists.txt`'s `target_sources(hip_kernel_provider_tests ...)` block |

**Both `IngestorPacks.hpp` and `IngestorPacks.cpp` edits are required.** A pack
registered in the header but missing from the `.cpp` table's `s_packs` vector silently
vanishes from the unit-test binary (a static-archive linker drops an object nothing
references) while still working in the plugin `.so` -- no error either way.

## The generate -> validate round trip

`hipdnn_validate_descriptors` requires a build configured with
`-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` (default **OFF**) and the validator target
built. A missing binary can mean an unbuilt target, disabled capability or a wrong
build/install path. Use runtime descriptors: authored direct-load output or packed
per-architecture output, never unlowered rocKE authoring input.

```bash
# 1. Generate a bundle.
.venv/bin/python generate.py --config configs/scale_add.yaml --output-dir /tmp/scale-add

# 2. Validate it structurally, with no GPU and no linked provider.
<build-dir>/bin/hipdnn_validate_descriptors \
    /tmp/scale-add/descriptors \
    --expect-engine hipkernel:ScaleAdd \
    --json
```

Exit 0 from this structural invocation establishes descriptor parsing,
cross-references and the expected engine's presence within the validator's
structural scope. It does **not** certify real native registrations, matcher
semantics, compiled specialization or device correctness. Execute actual provider
registration/loading and the emitted-bundle census separately, then prove dispatch
and numerics against the exact intended engine.

`tests/test_round_trip.py` checks this in as a permanent (though `-m round_trip`
opt-in, since it depends on a validator binary this repo does not build by default)
regression: point `HIPDNN_VALIDATE_DESCRIPTORS` at your build's binary and run
`.venv/bin/python -m pytest -m round_trip`.

## The pipeline tools

`generate.py` emits a bundle. The host tools below inspect distinct contracts;
profiles are optional analysis or authoring inputs where supported, not compiler
evidence. Full compiled agreement reads the self-contained producing-build record
and requires no rocKE installation on the verifying machine.

| Tool | Answers | Invocation |
|---|---|---|
| `tools/verify_variant_sets.py` | Structural nesting/runtime tuple identity, sentinels and vocabulary. Artifact-bound compiler agreement is a distinct, stronger mode checked against the packed producing-build record | `verify_variant_sets.py --mode {full,structural} [--arch A] [--profile P] [--kpack-python-dir D] LABEL ROOT...`; `--mode` is required and has no default, because the two make different claims. `--mode structural` reports compiled specialization agreement as NOT CHECKED by name and still exits 0 on the rest; `--mode full` fails on a missing, unsupported or mismatched producing-build record **and** on any check that could not run (`GATE FAILED (N check(s) NOT RUN: ...)`). `--profile` is optional and supplies only the bundle to gate and the matcher vocabulary — but full mode needs that vocabulary to actually run, so a full invocation over string fields no declaration spells out requires one |
| `tools/variant_reachability.py` | Can any shape in the corpus actually select each variant, or is one dead weight? | `variant_reachability.py --kdp K --shapes S [--profile P]` |
| `tools/launch_surface.py` | Is every surface the C++ restates from the kernel's Python declared, guarded and tested? | `launch_surface.py PROFILE --check [--allow-unguarded]` |
| `tools/coverage_gate.py` | Structural, loading and serving obligations, reported separately; an unmet required obligation cannot pass | `coverage_gate.py --tree T --mode {full,structural} [--arch A] [--validator V] [--expect-engine E] [--min-served N]`; `--mode` is required and governs what rung 1 may claim. A missing `--validator` makes rung 2 `loads-not-run`, a failure and never a silent skip; an offline result is not serving evidence |
| `tools/knob_sweep.py` | Which knob arms are worth measuring, isolation first then pairwise. | `knob_sweep.py --profile P --shapes S [--plan]` |
| `tools/dispatch_parity.py` | Do the emitted descriptors match what the kernel's real dispatcher resolves? | see `--help` |
| `tools/reconcile_applicability.py` | Does this engine decline anything the reference library serves? | `reconcile_applicability.py --profile P --shapes S [--declines D]` |
| `tools/mine_shapes.py` | Build the shape corpus, refusing categoricals it does not recognise. | see `--help` |

A green tool proves only the properties it checked. Missing, unsupported or
mismatched required evidence fails full agreement. Structural-only results must be
labeled as such and cannot satisfy a compiled-agreement or device gate.

`tools/sweep.py --config <absolute-YAML>` drives measurement with declarative input.
Use `configs/sweep-isolation.sweep.yaml.example` and the
[sweep reference](tools/README-sweeps.md) for exact input keys, hazard exclusions,
engine attribution, correctness gates and current-input-bound resume semantics.

## Specialization agreement

The producing compiler, not the generator or a later verifier's installed library,
is authoritative for effective specialization. Generation carries declarations as
data in each UKD's ignored provenance extension:

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

This illustrative consumer assumes its KMD has exactly these three fields; names
and accessor choices must come from the actual builder's source-use-site audit.
For each consumer, `metadata_fields` and `matcher_only_fields` must exhaustively
and disjointly partition its referenced KMD fields. `bindings` keys are exactly
`metadata_fields`, and each value is exactly `{field: "<attr>"}` or
`{method: "<accessor>"}`. Bindings refer to the actual hydrated spec object passed
to the builder, never a reconstructed policy object. IDs reference existing
descriptors; KMD types/defaults are not duplicated in the declaration. A UKD shared
by engines carries each consumer's entry; duplicate/conflicting entries fail.

A direct field is legal only when the builder consumes it without further
resolution. When the builder consumes an effective accessor, that zero-argument
bound method must be read **even if the raw field is non-null**: coupling can
override explicit values. For example, raw swizzle true can resolve false when
conflict-free V is disabled. Do not guess accessor names or copy policy formulas.
Missing/noncallable readouts, exceptions, unsupported values, unresolved `None`
and non-repeatable resolution block full agreement.

`None` is authored intent, **never a compiled-artifact wildcard**. Authored
`provenance.spec` remains distinct and preserved; only the producing compiler
writes `provenance.effective_spec`. Authored inputs supplying that reserved
evidence record are rejected. The generator neither imports a compiler to resolve
policy nor issues observations on its behalf.

Metadata completion and comparison use the referenced KMD's defaults and types.
BOOL remains boolean; a builder boolean may deliberately project to 0/1 for INT;
FLOAT values are canonicalized numerically. Matcher-only classification requires a
source-use-site audit and independent review, not merely a mechanically complete
partition. A consumed specialization field with no authoritative binding remains
unsupported; reclassifying it to make a check pass is invalid.

Packaging observes the actual builder object, retains serializable observations
with serial/prewarm/shared compilation results, and compares **every consumer**
independently before publication. The record binds effective values and declaration
digests, authored inputs, actual producer identities/origins, descriptors, KMD
content, completed metadata, architecture and library/toc-key/symbol/payload hashes.
Authored passthrough cannot overwrite fresh observations.

Full checking verifies that self-contained record against the current descriptors
and named payload bytes without importing rocKE on the verifier. Structural-only
checking cannot supply missing compiler agreement. Neither strength proves arbitrary
machine-code equivalence, native semantics or numerical correctness. See the
[packaging reference](../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).
There is no packaging `--profile`, CMake `PROFILES` or external root manifest;
existing profiles remain authoring/mining inputs whose declarations travel in UKDs.

## Configs

`configs/scale_add.yaml` -- a single-pack engine (mirrors the shipped `conv_fwd`
engine's shape: one pack, one operation, its `graph_match` both admits the node type
and validates shape).

`configs/binary_ops.yaml` -- a multi-pack engine (mirrors the shipped `pointwise`
engine's shape: one pack per operation, sharing one KMD/UED/UHD/UDD, each pack naming
its own operation-scoped UMD via `discriminator`).

`configs/axes_example.yaml` -- pack-level `axes`: one `kernel_template` crossed with
a few value lists, expanded at load time.

`configs/variants_example.yaml` -- pack-level `variants`: a shape list crossed
per-shape with a named knob set. See below.

## Generated variant sets: `variants`

A generated set is written one YAML block per kernel. The largest shipped gfx942
attention_dense config was **89,265 lines for 2,710 kernels**, committed compressed
because that was the only way it fit. Compression is not the fix: the file is
unreadable either way, and it is the ONE file worth reviewing in a descriptor PR,
because the descriptors are its deterministic output.

`variants` states what the enumeration stands for -- **about 1,150 lines for the same
2,710 kernels**, generating byte-identical descriptors:

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

`configs/variants_example.yaml` is the runnable version, with every key exercised.

**Why not `axes`.** `axes` crosses ONE `kernel_template`. A dispatcher-derived set has
no single template: `dispatch_parity.py` asks the library for a spec per shape, so
every shape carries its own resolved values for the fields the dispatcher derives.

**It is not a grid.** Each shape names its own knob set. On the shipped sets most
shapes carry four arms and 63 carry six; one global cross-product would invent
variants for some shapes and drop them for others.

**The tri-state.** An omitted or null policy knob retains the builder's policy
intent; it is not an explicit false. `resolved` supplies an authored metadata
projection, not compiler evidence. Final composition includes pack defaults before
projection, and the producing compiler must compare metadata against the declared
effective readout. A metadata override is legal only under its reviewed binding or
matcher-only classification; it cannot disguise contradictory specialization.

**Names.** The template must encode everything that varies, and the loader rejects a
pack whose expansion produces two kernels with the same name. A slot is a spec field,
an `md_<field>` metadata mirror, the arm's `{tag}`, or `{ordinal}` -- a per-shape
serial the shape sets and each arm shifts with `ordinal_offset`, for grammars that
number their kernels instead of naming every field.

Expansion runs at load time (`codegen/config_loader.py`), so `generate.py`, the
emitters and the dedup pass see ordinary kernel dicts. `tools/dispatch_parity.py`
emits this form directly; `tools/factorise_config.py` converts an already-enumerated
config, re-expanding its own output and refusing to write anything that does not
reproduce the input kernel-for-kernel.

## Config surface

```yaml
engine:
  name: hipkernel:MyEngine        # required, scoped namespace:local
  sdk_version: "1.0.0"            # optional, three components, default "1.0.0"
  behavior_notes: [runtime_compilation]   # optional, closed vocabulary
  knobs: [block_size]             # optional; must all be int-typed kmd_fields
  heuristic: native | none        # optional, default "native"; "none" omits the UHD

kmd_fields:                       # the KMD's fields[] -- declared, one per human-
  - name: block_size              # meaningful axis this engine's kernels vary along
    type: int                     # bool | int | float | string | int_list
    default_value: 64             # omit entirely for a MANDATORY field
  - name: dtype
    type: string

graph_match:                      # documentation of shape, not consumed by templates
  shape: shared_shape | disjoint_attributes
  discriminator: none | field_value | disjoint_topology

kernel_source_kind: embedded_source   # direct-load example; packaged sources use
                                        # their build-time source kind
workspace_policy: none | fixed | derived
delegates_to_existing_plan: false

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
2. Every `engine.knobs` entry names a declared **and int-typed** `kmd_fields` entry --
   a non-int knob is accepted by the real loader and produces no usable knob at all,
   silently, discovered only at plan-build time against a real device.
3. Every kernel's `metadata` type-checks against the KMD, with no mandatory field
   (one with no `default_value`) omitted -- otherwise the real loader drops the whole
   pack.
4. Every kernel's `arch` is a subset of its pack's `arch`.
5. Every `arch` entry is a plausible `gfx`-prefixed base id (lowercase, no feature
   suffix) -- an error if malformed; a **warning** (not an error) if well-formed but
   not a recognized target id (e.g. `gfx94` for `gfx942`), since match-time evidence
   for either case looks identical (an ordinary INFO decline) and this tool does not
   claim to maintain an exhaustive, always-current arch list.

## Source adapters (`codegen/sources/`)

One protocol (`SourceAdapter.infer(*sources) -> SourceAdapterResult`), two v1
implementations:

- `InteractiveAdapter` -- no inference; a human or the driving skill fills every field.
- `HiprtcAdapter` -- scans one or more `.cpp`/`.hip` files for
  `extern "C" __global__` entry points and candidate KMD fields (externally-supplied
  `HIP_PLUGIN_*` defines, template parameters).

`rocke` authoring uses the packaged path; its actual builder/spec and effective
policy observations belong to the producing compiler, not these source adapters.

`hsaco_file` is rejected explicitly (naming `supportsSourceKind()` as the missing
prerequisite on `IKernelDispatchHandler`), not silently accepted and left to fail later
with a generic "no implementation yet".

## Tests

```bash
.venv/bin/python -m pytest
```

`pyproject.toml` sets `fail_under = 80` for `coverage`. Content/substring assertions on
rendered output plus CLI subprocess exit-code tests -- not golden-file diffing, per
`DescriptorGenerator`'s own test shape. Two assertions are load-bearing and
non-negotiable (`tests/test_generator.py::TestRequiredTrapAssertions`): that the
emitted `graph_match` stub's doc comment literally contains the whole-catalog
blast-radius warning, and that the emitted `Test<Name>Matchers.cpp` constructs
`DeviceProperties` by value.

### The native stub's own shape (`tests/test_native_stub.py`)

```bash
.venv/bin/python -m pytest tests/test_native_stub.py
```

Covers what the two required trap assertions above do not: that every hook body is
genuinely a `TODO` placeholder (none silently emitted as working logic), that the
symbol constants the stub declares match what the SAME run's descriptor JSON names,
that the registration block wires every declared symbol and none more, and basic
structural soundness (balanced braces, every hook present). `TestRealCompile` also
host-compiles the emitted stub with `g++`/`clang++` when one is on `PATH` and the
plugin/data/flatbuffers SDK sources are found beside this checkout (walking up from
`tools/IngestorGenerator`) plus a vendored `flatbuffers/array.h` (checked at
`/opt/rocm/include`) -- it generates minimal stand-ins for the CMake-configured
`version.h`/`CacheRootDefaults.h` headers from their real `.h.in` templates rather
than skipping outright. Skips (never fails) when any prerequisite is absent, so a
box without those trees still runs the rest of the suite.

### Fragment/struct arity (`tests/test_fragment_struct_arity.py`)

```bash
.venv/bin/python -m pytest tests/test_fragment_struct_arity.py
```

`fragments/ingestor_packs_cpp.j2` once emitted a two-field `s_packs` row against a
three-field `IngestorPack` struct (the mismatch did not compile as spliced and was
fixed by hand during a real integration run). This parses the REAL field count out
of the provider's `IngestorPacks.hpp` and asserts the emitted row's arity matches it
-- not a hardcoded `3`, which would just re-freeze today's coincidental agreement.
Skips if the provider source is not found beside this checkout. Also checks that the
`.hpp`/`.cpp` fragment pair name the same register-function symbol, and that
`cmake_test_sources.txt` names files this generator's own run actually wrote.
