# hipDNN IngestorGenerator

Generates a hipDNN generic-kernel-ingestor descriptor bundle -- KMD/UED/UMD/UHD/UDD/KDP
JSON, a native-symbol stub, complete pack-shape census tests, a matcher-test stub, and
six CMake/registration text fragments -- from a YAML config. Modeled on
`projects/hipdnn/tools/DescriptorGenerator`'s conventions, with two deliberate
deviations: `undefined=StrictUndefined` on the Jinja2 environment (an unset UUID
cross-reference fails loudly at generation time, not as a confusing empty-string
rejection at load time), and a required `--force` flag to overwrite a non-empty output
directory (the extend flow points this tool at a *live* descriptor directory that may
hold hand-filled `graph_match`/matcher bodies).

This tool has **no CMake/CI hookup**, matching `DescriptorGenerator`'s own convention:
its correctness gate is running `pytest` (below), not a build.

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

# Regenerate over an existing, non-empty output directory (the extend flow) --
# --force is REQUIRED, or generate.py exits 1 without touching anything.
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

`packs/<Name>Native.cpp` uses `constexpr std::string_view` symbol constants at the top
of an anonymous namespace and `scope.add(std::string(CONST), &fn)` calls in
`register<Name>Symbols()` -- the exact form `hipdnn_validate_descriptors`'s
`--native-source` check resolves (it looks for named constants referenced from that
function, not inline string literals). The `graph_match` stub's doc comment carries,
verbatim, the warning that returning `std::nullopt` empties the **whole** engine's
catalog and skips every remaining pack (`KernelIngestorStateManager.hpp:450-455`) --
the widest blast radius in the system.

### Matcher-test stub

`tests/Test<Name>Matchers.cpp` constructs its `DeviceProperties` fixture **by value**,
never by querying the host (`hipGetDeviceProperties`/`getDeviceProperties()`), with a
comment explaining why: a test that queries the host is vacuous on every arch except
whatever happens to be running CI (`TestAsmSdpaForwardMatchers.cpp:27-33`).

### Pack-shape census test

`tests/Test<Name>Packs.cpp` is **complete**, not a stub. It reads the descriptors this
engine actually ships via `discoverDescriptorSets()` (the same helper
`PointwiseTestGraphs.hpp`'s `loadedSet()` wraps) rather than a hand-built twin, so a
broken *shipped* descriptor -- a missing pack, a dropped knob, a wrong kernel count --
fails this fast suite instead of surfacing only in the slow GPU integration suite.

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

`hipdnn_validate_descriptors` only exists in a build configured with
`-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON` (default **OFF**). If it is not present in your
build's `bin/` directory, that build was not configured with the flag -- reconfigure
and rebuild the `hip-kernel-provider`/`tools` targets, or check with whoever owns the
build.

```bash
# 1. Generate a bundle.
.venv/bin/python generate.py --config configs/scale_add.yaml --output-dir /tmp/scale-add

# 2. Validate it structurally, with no GPU and no linked provider.
<build-dir>/bin/hipdnn_validate_descriptors \
    /tmp/scale-add/descriptors \
    --expect-engine hipkernel:ScaleAdd \
    --json

# 3. (Optional, once you've filled in packs/<Name>Native.cpp) cross-check that the
#    native file's constexpr symbol constants agree with what the descriptors name.
<build-dir>/bin/hipdnn_validate_descriptors \
    /tmp/scale-add/descriptors \
    --expect-engine hipkernel:ScaleAdd \
    --native-source /tmp/scale-add/packs/ScaleAddNative.cpp
```

Exit 0 means: every root loaded with zero ERROR diagnostics, every `--expect-engine`
name is present, and (if given) every `--native-source` check is clean. **This proves
parse, cross-reference, symbol resolution, and construction -- nothing about
`graph_match`/matcher correctness**, which needs a real graph and a real device.
Enumeration proves much less than it looks: PR #10839's engine enumerated cleanly on
gfx90a and failed all 27 cases on gfx942, because the packs arch-pruned before the
matcher ever ran on gfx90a.

`tests/test_round_trip.py` checks this in as a permanent (though `-m round_trip`
opt-in, since it depends on a validator binary this repo does not build by default)
regression: point `HIPDNN_VALIDATE_DESCRIPTORS` at your build's binary and run
`.venv/bin/python -m pytest -m round_trip`.

## The pipeline tools

`generate.py` emits a bundle; these audit it. All are host-only and need no GPU. Each
reads the same per-kernel `configs/<slug>.profile.yaml`, so they cannot disagree about
which kernel they are discussing.

| Tool | Answers | Invocation |
|---|---|---|
| `tools/verify_variant_sets.py` | Do the descriptors nest, are their loader tuples unique, and does each one's metadata agree with the spec its binary was built from? | `verify_variant_sets.py [--profile P] LABEL ROOT...` |
| `tools/variant_reachability.py` | Can any shape in the corpus actually select each variant, or is one dead weight? | `variant_reachability.py --kdp K --shapes S [--profile P]` |
| `tools/launch_surface.py` | Is every surface the C++ restates from the kernel's Python declared, guarded and tested? | `launch_surface.py PROFILE --check [--allow-unguarded]` |
| `tools/coverage_gate.py` | Three rungs: descriptors well-formed, engine loads, engine serves. Rung 3 needs a device and reports NOT RUN without one. | `coverage_gate.py --tree T [--profile P] [--validator V]` |
| `tools/knob_sweep.py` | Which knob arms are worth measuring, isolation first then pairwise. | `knob_sweep.py --profile P --shapes S [--plan]` |
| `tools/dispatch_parity.py` | Do the emitted descriptors match what the kernel's real dispatcher resolves? | see `--help` |
| `tools/reconcile_applicability.py` | Does this engine decline anything the reference library serves? | `reconcile_applicability.py --profile P --shapes S [--declines D]` |
| `tools/mine_shapes.py` | Build the shape corpus, refusing categoricals it does not recognise. | see `--help` |

A green tool proves only what it asked. `coverage_gate.py` is explicit about this: it
reports rung 3 as NOT RUN rather than passing, because nothing host-side can prove the
engine served a graph.

`tools/sweep.sh` drives a measurement sweep and requires `EXCLUDE_TENSORS` with no
default -- the tensor names marking graphs that are unservable and dangerous for your op
(for attention, backward graphs, marked by their gradient tensors). An op with no such
class must say `EXCLUDE_TENSORS=none` explicitly. See `tools/README-sweeps.md`.

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

**The tri-state.** A knob absent from an arm is absent from the emitted
`kernel_source.spec`, which tells the builder its own policy decides at build time.
That is NOT the same as pinning it `false`, and both reach metadata as `0`. The shape
states the policy's answer under `resolved`; an arm may also pin `metadata` directly,
for a knob swept in the catalog while the binary is unchanged.

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

kernel_source_kind: embedded_source   # the only implemented kind; anything else is a
                                        # hard ConfigError naming why
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

`rocke` is a later adapter behind the same protocol, added once the packer and kpack
launcher land -- deliberately absent here, not stubbed.

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

### Every path the skill cites must exist (`tests/test_skill_paths.py`)

```bash
.venv/bin/python -m pytest tests/test_skill_paths.py -v
```

`native-pack.md` once told an agent to read `packs/AttentionDenseNative.cpp`, which did
not exist on the branch the skill shipped on -- the file's first instruction was
unresolvable, and one grep would have caught it. Nothing ever ran that grep. This test
does, permanently: it walks every backtick span in the six
`tools/ai/skills/hipdnn-ingestor-engine/*.md` files, extracts the ones that are
unambiguously repo paths (`tests/skill_paths.py` holds the conservative rules -- a known
extension or a known top-level directory prefix, no `$VAR`/`<placeholder>`/glob/shell
metacharacter), and asserts each resolves against `git ls-files`.

Lives here rather than under the skill directory because this is the tool with a venv
and a pytest story already wired; the skill directory has neither. Extraction is
deliberately conservative -- bare words, `$REPO`-rooted paths, `<op>`-style
placeholders, globs and shell one-liners are routed to a separate "skipped, and
counted" bucket rather than either being flagged or silently dropped, and one test
asserts that skip count stays in a sane range so an extractor regression (suddenly
flagging everything, or nothing) is visible. A dedicated negative-test class proves the
check actually fires: it feeds the extractor the literal defect string above and
confirms it flags `packs/AttentionDenseNative.cpp` as dangling on this tree, paired with
a positive case (`packs/ConvNative.cpp`, `packs/PointwiseNative.cpp`, both of which do
exist) proving the negative result comes from the path being genuinely absent, not from
the resolver rejecting everything.
