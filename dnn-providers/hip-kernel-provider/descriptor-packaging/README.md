# hkp_pack -- descriptor packaging

Build-time UKD/KMD/KDP -> kpack packaging. Provider-internal (`tools/hkp_pack.py`);
see `python/hkp_pack/` for the pipeline and `examples/descriptors/` for a real,
minimal authored source root.

## Source roots and what the walk accepts

Each wired root is walked recursively and packs into **its own** `OUT_ROOT`; no two
invocations share a destination and there is no shared stage tree. Each descriptor's
authored subpath is preserved verbatim into the staged and installed trees. Producer
selection is per-UKD on `kernel_source.kind`, never per-folder, so one root feeds every
producer into one kpack per arch. Nothing is registered in CMake: adding a descriptor is
dropping files in a folder.

Two filters decide what a build packs, and both prune exactly as arch pruning does:

- **Family folders.** A descriptor family lives in a top-level child folder of a root
  (`rocKE/` today; `hip/` has no switch). `HKP_DESCRIPTOR_FAMILIES` in
  `HkpPackaging.cmake` maps each switchable folder to the option that enables it, and a
  family whose option is OFF is passed as `--exclude-folder <name>`: its files are
  never read, under every root. A new family is one entry in that table. A folder is
  matched by its exact name as a **top-level** child of the root, so a folder of that
  name deeper in the tree is not a family. A root that is itself placed inside a family
  folder (for example `descriptors/rocKE/...`) is not filtered by that family's switch:
  pointing a root there is the user's responsibility.
- **Producer kinds.** A `kernel_source.kind` this build has no producer for is passed
  as `--disable-kind <kind>`. Its UKDs are dropped from their KDPs, and a KDP left with
  none does not ship. The rocKE producer, and the private rocKE wheels and comgr it
  lowers through, exist only under `HIPKERNELPROVIDER_ENABLE_ROCKE=ON`, so with it OFF
  `rocke` is disabled. A kind outside the vocabulary is still a validation error.

A root, or an arch, left with nothing to pack is skipped, not failed. Only producers
gated by a build option (rocKE today) are disabled by kind; a kind outside that set is
not pruned.

The provider wires six roots: production, plus five over the four authored test sets
(`shared` packs twice, once into each test binary's discovery root).

| Root | Source | Ships |
|---|---|---|
| Production | `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, a `CACHE PATH` defaulting to the in-tree `src/engines/kernel_ingestor_engine/descriptors/` | yes |
| Test | `src/engines/kernel_ingestor_engine/test_descriptors/{shared,unit,integration,archive_fixture}/` | only under `HIPKERNELPROVIDER_ENABLE_TESTS` |

Before wiring any root, configure asks the packer (`pipeline.shipped_engines` and
`pipeline.offered_engines`, under the base interpreter, with the same filters) which
engines each arch's shard would carry and which engines the root carries for any arch. A
root, production or test, named or inherited alike, that is empty or would ship nothing
for any arch this build packs for is **dormant** at configure and skipped at pack, and
any stale output tree is removed; neither is an error, and one STATUS line says why. A
dormant root is recorded so the census and the embedded-source verify step skip it
too. Two answers decide which engine-pinned tests are registered:

- **Per-arch registrations** (the census, and the external and gpu_ref integration
  checks) install into a gfx-specific shard and are gated on
  `hkp_gfx950_attention_dense_available()`, which reads what the shard for that arch
  ships.
- **Host-side sources that pin an engine**, such as the gfx950 knobs suite, are gated on
  `hkp_product_offers_engine()`, which reads what the product root carries for any arch
  under this build's filters. It never reads `GPU_TARGETS`: the host libraries and
  tests must work with another build's arch content.

A probe that cannot answer leaves the root wired, so the packer reports what is wrong
with it, and both predicates then read `HIPKERNELPROVIDER_ENABLE_ROCKE`. A root set but
not a directory is fatal.

Test roots hold no family folder today. The first family folder added under
`test_descriptors/` must also gate the test binary's cases: a fixture root that goes
dormant still leaves cases in that binary that expect its descriptors.

Two rules govern the walk:

- **Hidden paths are skipped, and said so.** A dot-prefixed path segment or filename is
  warned and skipped, as is a `*.json` whose name carries no type token. A root whose
  only KDP sits under a hidden path therefore probes empty and goes dormant. A type-tagged descriptor that is malformed, missing a field, of unknown
  type or carrying a dangling reference still fails.
- **An `embedded_source` `source_file` must act as an identity.** It is never
  normalised, so `..` is rejected (one file would take two identities) and an absolute
  path is rejected (the emitted key must be the same on every machine).

`embedded_source` is a **passthrough** kind: the descriptor is emitted as authored, no
producer runs, and it contributes no code object and no archive entry — the packer only
stamps the shard architecture and records provenance. A root of only passthrough kinds
therefore produces descriptors and **no** archive, and a shard with no compiled variant
holds no `kpack/`. Descriptors but no archive is legal.

`hsaco` names a prebuilt code object: `kernel_source: {kind: "hsaco", file, symbol}`.
`file` resolves relative to the descriptor that names it, must stay inside the root, and
has no root-relative fallback — the same rule as a `hip` `source`. No compile runs: the
bytes are packed as-is into the arch's kpack, the kernel signature is read from the
object's AMDGPU metadata as for a compiled object, and the UKD ships as `kind: kpack`.
The toc key derives from the file's resolved root-relative path, so one file serving
several symbols is one archive entry, and one key claimed by two different files is a
hard error. The packer does not check the object's format or target processor. An
`hsaco` UKD must list the arch(es) its object runs on in `arch` (a generic-target
object lists every arch it runs on); an absent or empty `arch` is rejected. The author's own load test on the target arch
is the only check; no in-tree load test covers `hsaco`. The shipped provenance records
`origin_kind: "hsaco"`, the root-relative `file`, its `sha256` and the `symbol`, and makes no
toolchain claim. As for `hip`, the specialization contract must declare
`metadata_fields: []`: no compiler ran whose specialization a binding could observe.

## Compiler-bound specialization agreement

Packaging consumes UKD `provenance.specialization_contract` as data. It binds no
authoring profile and implements no second policy: there is no packaging `--profile`,
CMake `PROFILES` input or external root manifest. See the
[generator agreement reference](../../../projects/hipdnn/tools/IngestorGenerator/README.md#specialization-agreement)
for authoring and projection semantics.

**Declaration.** `schema_version: 1` plus `consumers`, each with `engine_id`, `kmd_id`,
`metadata_fields`, `matcher_only_fields`, `bindings` and `vocabulary`. The two field
lists exhaustively and disjointly partition the referenced KMD's fields. Binding keys
are exactly `metadata_fields`; each value is exactly `{field: "<attr>"}` or `{method:
"<accessor>"}`. IDs reference existing descriptors without copying KMD type/default
definitions. Shared standalone UKDs carry all consumers; duplicate or conflicting
declarations fail.

**Observations** use the selected compiler interpreter and the actual imported
builder/spec objects; `build_spec` hydrates ordinary defaults and default factories on
the object passed to `builder_fn`. Observe a declared direct field only when the builder
uses it without further resolution; otherwise observe the zero-argument bound effective
accessor the builder consumes, even when the raw field is non-null, because coupling may
override an explicit value (swizzle true with conflict-free V disabled). `None` is
authored intent, not a wildcard. Missing or noncallable accessors, unresolved `None`,
unsupported return types, exceptions and non-repeatable observations block full
agreement. Typing goes through the actual referenced KMD: BOOL stays boolean, a boolean
may intentionally project to 0/1 for INT, FLOAT normalizes numerically, and
descriptor-side type errors are not silently coerced. Observed effective values are
compared against each consumer's completed metadata before publication. Never guess
accessor conventions, copy policy formulas or reconstruct a different spec object for
comparison; matcher-only classification needs a source-use-site audit and independent
review.

**Shared compilation.** All consumers' observation requests are collected before a
shared variant is compiled, and every consumer is compared independently — including on
result reuse, so one successful consumer cannot certify a contradictory second. Reuse is
scoped to the packaging invocation, not a persistent cache.

**Reserved evidence.** Authored `provenance.spec` is preserved untouched; only the
producing compiler writes `provenance.effective_spec`, and an authored rocKE input
supplying a purported record is rejected. Extra/provenance passthrough must not
overwrite fresh observations during UKD rewriting or final publication, and packed-input
validation reads the actual packed record rather than recompiling authored input. The
schema-versioned producing-build record binds effective values and observation requests,
the canonical authored-input digest, observed builder/spec/accessor identities and
origins, consumer UKD/engine/KMD/KDP IDs, KDP/effective architecture and the actual
library/toc-key/symbol/payload SHA256. Each consumer's engine, KMD, KDP header, completed
metadata and declaration are bound by **content digest** rather than carried whole: the
checker rebuilds them from the descriptors in front of it. The binding digest excludes
the evidence itself, and the declaration stays alongside the record so a checker needs no
external profile.

Producer identities are module-qualified names and defining-file content hashes from the
actual imported objects. The defining path is watched for the invocation but never
published: shipping it would make the artifact a function of the building machine's
install layout. An unresolvable required origin fails evidence production, and producer
files must stay stable during the invocation. Wheel stamps are separately labeled build
provenance, not proof of imported origins.

**Full versus structural.** A full check validates the declaration and producing-build
record against current descriptors, schema, metadata, architecture and named payload
bytes; missing, unsupported, stale or mismatched required records fail. A valid packed
artifact can be fully checked **without rocKE installed on the verifying machine**. A
structural-only check passes the properties it checks but does not satisfy compiled
agreement, and mining/analysis profiles cannot supply or override compiler evidence.
Observations establish agreement with the builder's decisions and artifact integrity,
not equivalence of arbitrary machine code or correctness of native dispatch.

## Packed `kernel_source`

The runtime consumes packed per-architecture descriptors with source kind KPACK, not
unlowered rocKE, HIP or hsaco authoring descriptors. A packed `kernel_source` carries **five
mandatory keys**:

```json
{
  "kind": "kpack",
  "library": "../../kpack/hip_kernel_provider_gfx942.kpack",
  "toc_key": "pointwise_add_f32",
  "symbol": "pointwise_add_f32",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "signature": [
    {"kind": "global_buffer", "size": 8, "offset": 0},
    {"kind": "global_buffer", "size": 8, "offset": 8},
    {"kind": "by_value", "size": 4, "offset": 16}
  ]
}
```

`signature` is required: an empty array is legal and means a kernel taking no arguments,
so absence would be indistinguishable from it. Per entry `kind`, `size` and `offset` are
mandatory; `name` is optional, because clang emits argument names for some producers and
omits them for HIP `extern "C" __global__` kernels. `sha256` is shape-checked as 64
lowercase hex.

Neither digest nor signature is hand-authored, and they catch different drift. `sha256`
is byte identity of the **decompressed** code object, so a TOC entry pointing at the
wrong offset cannot silently hand back another entry's object; the loader rehashes before
`hipModuleLoadData` and raises `DIGEST_MISMATCH`. `kernel_signature.py` reads the
argument list back out of the compiled object — sniffing a clang offload bundle or a bare
ELF, and dropping compiler-appended `hidden_` arguments — catching a kernel whose
parameters changed while its bytes stayed internally consistent.

## Emitted-bundle census

Native proof executes typed provider registration/loading plus a finalized census;
source-text symbol matching is not certification. Registration is one call per packed
target, in `src/tests/CMakeLists.txt` beside `hkp_verify_embedded_sources()`:

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

`TARGET` is the census binary, `hip_kernel_provider_census_tests`. The `Test<Name>Packs`
suites are compiled into it and not into `hip_kernel_provider_tests`: every census case
needs a descriptor shard and the census environment, which an ordinary unit run must not
require.

`PACK_NAME` selects the wired pack target whose `OUT_ROOT` and recorded arch list the
entries address. `ARCHES` optionally narrows that list: omitted, the suites register at
every arch the pack target was wired for; given, at the intersection of the named arches
with that list; naming the keyword with no arch is fatal. A suite whose fixtures cover the
whole root, like `TestPointwisePacks` above, omits it. A suite stating the inventory of a
bundle that emits for specific arches names them — the gfx950 dense-attention census
passes `ARCHES gfx950` — so a build packing other arches registers nothing for it rather
than asserting that inventory against a shard that never held it.

Per declared suite and per eligible arch, CMake registers
`hip-kernel-provider-hkp-census-<arch>-<suite>`, invoking
`hip_kernel_provider_census_tests --gtest_filter=<suite>.*` directly, without Python, with
`HIPDNN_TEST_CENSUS_SUITE=<suite>`, `HIPDNN_TEST_EXPECTED_ARCH=<arch>` and
`HIPDNN_DESCRIPTOR_DIR=<OUT_ROOT>/<arch>` — its own shard, not a shared stage tree. Each
entry is an independent process labeled `unit_test;hip-kernel-provider;host`, plus the
tier labels `HKP_PACK_CTEST_CATEGORIES_YAML` assigns it; the installed twin carries the
same labels as the build-tree entry.

```bash
ctest --test-dir <build>/dnn-providers/hip-kernel-provider \
  --no-tests=error -V -R '^hip-kernel-provider-hkp-census-<arch>-<suite>$'
```

A suite is declarable only where it reads **exactly one** pack target's shard. The
authored dialect does not decide this: `TestPointwisePacks` is censused at `unit`
although that set is `embedded_source`, while `TestConvFwdPack` reads the `unit` and
`unit_shared` shards and is censused nowhere. One suite declared at two pack targets is
fatal: the entry name carries arch and suite alone, so the second registration would take
the first one's shard.

Strict mode is active only for a nonempty census-suite variable. Before default-root
setup it rejects missing/empty/nonexistent explicit roots and empty expected arches; the
production loader's normal fallback is unchanged. The named suite must exist and be
nonempty, and every registered case must complete and pass without skipping in each
iteration, with at least one completed iteration. Disabled, filtered-out, sharded-out,
failed or skipped cases, list-only and repeat-zero invocations cannot satisfy the census,
and repeated partial runs cannot accumulate coverage. `EXPECTED_CASES` supplies what
execution cannot: execution obligations are built from the cases the suite registered, so
a deleted case takes its own obligation with it.

The call is made where the target is defined and after it exists. A `PACK_NAME` no pack
target carries, an absent or nonexistent `TARGET`, and an empty recorded arch list are
each fatal rather than a silent drop, because a census that registers nothing is
indistinguishable from one that passed. A `PACK_NAME` the registry records as DORMANT is
the one absence that is not a mistake: it stages no shard, so the call registers nothing
and says so at STATUS, keeping a generated integration's declaration valid on every
configuration. Tests OFF and an empty `SUITES` register nothing. Neither structural nor
host loading proves numerical device behavior. The
[ingestor RUNBOOK](../../../projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine/RUNBOOK.md)
owns the complete create/extend sequence and post-regeneration gates.

## Build speed: put the comgr cache on local storage

Packing a rocKE descriptor set is dominated by lowering each kernel through
`libamd_comgr`, which caches results on disk. That cache defaults to **`~/.cache/comgr`**,
so on a network home directory every lookup is a network round trip and packing slows by
an order of magnitude. Point it somewhere local:

```bash
export AMD_COMGR_CACHE_DIR=/tmp/comgr-cache   # RAM disk or local disk
```

`AMD_COMGR_CACHE=0` disables caching outright, a diagnostic rather than a fix.

| Variable | Effect |
|---|---|
| `HKP_PACK_JOBS` | Prewarm worker count. Defaults to `min(32, ncpu)`; `1` forces the serial path for a clean traceback. |

`HKP_PACK_JOBS` is read by a **direct child run** of `hkp_pack`. Inside the build it comes
from the `PACK_JOBS <n>` argument of each `hkp_wire_pack_target()` call: the test roots
pass `1`, or `2` for `integration`, and the production root passes none, so the packer uses
its default. The reason is in the `hkp_wire_pack_target()` header in
`cmake/HkpPackaging.cmake`.

## Running the tests

```bash
cd dnn-providers/hip-kernel-provider
PYTHONPATH=descriptor-packaging/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests -q
```

`rocm_kpack` (the second `PYTHONPATH` entry, or `--kpack-python-dir` /
`HIPKERNELPROVIDER_ROCM_KPACK_DIR`) is the kpack archive reader/writer most of the suite
round-trips through. Its absence is diagnosed once by the `rocm_kpack_dir` fixture in
`tests/conftest.py`: every test needing it skips with one message naming the dependency.
Set `HIPKERNELPROVIDER_KPACK_REQUIRE_ROCM_KPACK=1` (mirroring `_REQUIRE_HIPCC` /
`_REQUIRE_COMGR`) to turn that skip into a hard failure, e.g. in CI.

`-m quick` selects the load-time/pure-unit subset needing neither `hipcc` nor
`rocm_kpack`/comgr.

Every rocKE test lives in `tests/rocke/`, whose own `conftest.py` owns the rocKE
fixtures (`rocke_available`, `rocke_importable`, `rocke_ukd`) and puts the in-tree rocke
platform and kernels library on `sys.path`, so the command above needs no rocke path. No
test outside that directory imports rocke or needs the rocKE toolchain, so a new rocKE
test goes there. Tests outside it may still author or read rocKE descriptors as JSON:
the disabled-kind and disabled-folder tests in `tests/test_hkp_pack_layout.py` hand
them to a build without rocKE, which must prune them before any producer runs. With
`HIPKERNELPROVIDER_ENABLE_ROCKE=OFF` the registered ctest entries pass
`--ignore=<tests>/rocke` and never collect it; this runs the suite as that build does:

```bash
PYTHONPATH=descriptor-packaging/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests -q \
        --ignore=descriptor-packaging/tests/rocke
```

`tests/test_hkp_pack_wiring.py` drives real sub-configures of `HkpPackaging.cmake` to
hold that build's CMake wiring: a root wired with `ENABLE_ROCKE OFF` packs under an
interpreter without pip with rocke disabled and the excluded folders passed, the
configure-time `shipped_engines` probe answers under that interpreter, and the
registered entries ignore `tests/rocke/`.

The two `conftest.py` files both import as `conftest`, so neither exports helpers, and no
test module imports from another. Helpers both sides use live in plain modules:
`tests/synthesised_objects.py`, `tests/pack_helpers.py` and `tests/cmake_harness.py`.

### Desk-check a variant set (`hkp_pack.desk_check`, `tools/hkp_desk_check.py`)

```
tools/hkp_desk_check.py --mode {full,structural} [--kpack-python-dir D]
                        [--field F] [--drift-field F] <path/to/*.kdp.json>
```

The desk check resolves KDP engine → UED metadata → KMD UUID within the selected
descriptor tree/shards, completing defaults and canonicalizing metadata through that
schema. Tuple identity is engine-wide, including cross-pack overlap in effective
architectures: equal metadata on disjoint architectures is legal, conflicting overlapping
candidates are not. A KDP filename is not a KMD reference. An authored tree without
compiler evidence can receive only a structural result.

`--mode` is required and has no default, because a default would let a structural run
read as a full one. Every run prints a leading `mode=<full|structural>` line and a
`compiled specialization agreement:` line.

- `--mode structural` accepts an authored (pre-pack) or shipped file and reports
  agreement as NOT CHECKED.
- `--mode full` is for a shipped shard. It binds each kernel whose declaration names
  specialized metadata fields to the producing build's record and the archive bytes the
  descriptor names, reporting `OK for N kernel(s)`. A missing, unsupported or mismatched
  record for such a kernel is a failure, never a successful COULD-NOT-CHECK. A kernel
  declaring no specialized metadata field is listed by name under `compiled
  specialization NO COMPILED CLAIM:`, printed alongside `OK for N kernel(s)` in a mixed
  KDP. Only when NO kernel makes a claim does the agreement line itself read `NO
  COMPILED CLAIM`.

The **matcher field list** resolves in a fixed precedence: an explicit `--field` outranks
everything; otherwise, when the bundle declares a `specialization_contract`, **both
halves of its partition are the field list**, since a matcher-only field is one the
matcher still compares; failing that, the fields the descriptors carry.

The metadata/spec **drift** list (`--drift-field`, invariant 1) resolves separately and
inherits none of that: absent an explicit `--drift-field` it is every field carrying both
a spec value and a metadata value, the widest comparison the data admits. The declared
contract is an input to invariant 1, not a bound on it, since confining the audit to the
contract's fields would exit 0 on drift in a third. `--drift-field` narrows it explicitly,
and in the log, for a field whose two sides speak deliberately different vocabularies.

`tests/test_desk_check_invariants.py` exercises the shipped module, with
`tests/rocke/test_desk_check_rocke_corpus.py` holding the cases that need rocKE output.
Keep structural identity, real packed observations and tampered-evidence checks distinct
from native registration and numerical tests.

### Embedded-source verification (`tools/hkp_verify_embedded_sources.py`)

A staged tree holds descriptor JSON only, so an `embedded_source` descriptor resolves its
`source_file` against a key table the build compiles into the binary, and nothing in the
staged tree proves that table holds the named source. This step reads that table and
every `embedded_source` descriptor under the staged roots the binary serves, comparing
**presence** (each named `source_file` is a key) and **location** (the file registered
under that key is the one at the authored location the descriptor's provenance records,
joining the `provenance.source_label` root with `rel_dir` and `source_file`).
`--pack-stamp` adds a rule: a pack root whose stamp is present holds at least one
descriptor. A root whose pack is not wired — the dormant production root — contributes no
stamp and is not checked. It runs over emitted JSON alone and imports no part of the
packer.

**The comparison runs one way, staged descriptor → table, so a pass is not evidence that
a bundle is reachable.** A key no descriptor names is not an error: most embedded kernels
have no descriptor at all. Nor is a descriptor that never reaches a staged root —
authored under a folder no pack is wired to, it is never staged, so this walk never sees
it while the runtime never receives it. Catching that needs the authored tree as a second
input, which provenance cannot supply, because the packer is what writes provenance. The
check to state is *does a shard appear under that pack target's `OUT_ROOT`*, not *did the
verifier pass*. An absent root, an empty root, a root with no `embedded_source`
descriptor and an absent key table each pass — which is why a pass reports the two counts
it compared.

### Real-corpus builder-signature guards (`tests/rocke/test_hkp_pack_rocke.py`)

The real gfx942 `build_*` functions in `rocke/library/kernels/gfx942/` must satisfy
`_require_spec_arch_signature`'s `(spec, *, arch)` contract. The real-builder cases in
`tests/rocke/test_hkp_pack_rocke.py` and the rejection cases in
`tests/rocke/test_hkp_pack_producer_guards.py` cover complementary paths. Signature
acceptance alone is not effective-specialization or numerical proof.

```bash
PYTHONPATH=descriptor-packaging/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests/rocke/test_hkp_pack_rocke.py \
        descriptor-packaging/tests/rocke/test_hkp_pack_producer_guards.py -q
```
