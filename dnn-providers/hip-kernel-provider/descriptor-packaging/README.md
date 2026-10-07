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

The provider wires six roots: production, plus five over the four authored test sets
(`shared` packs twice, once into each test binary's discovery root).

| Root | Source | Ships |
|---|---|---|
| Production | `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, a `CACHE PATH` defaulting to the in-tree `src/engines/kernel_ingestor_engine/descriptors/` | yes |
| Test | `src/engines/kernel_ingestor_engine/test_descriptors/{shared,unit,integration,archive_fixture}/` | only under `HIPKERNELPROVIDER_ENABLE_TESTS` |

Production wiring is gated on the root holding at least one non-hidden `*.kdp.json`,
since a KDP is what arch pruning consumes. With none, packaging is **dormant** and any
stale product tree is removed; neither is an error. A root set but not a directory is
fatal. A KDP that prunes on every arch is a hard failure for a root the build NAMED, and
dormancy for the inherited default root.

Two rules govern the walk:

- **Hidden paths are skipped, and said so.** A dot-prefixed path segment or filename is
  warned and skipped, as is a `*.json` whose name carries no type token. The production
  content gate drops the same segments, so a KDP under a hidden path does not wire
  packaging. A type-tagged descriptor that is malformed, missing a field, of unknown
  type or carrying a dangling reference still fails.
- **An `embedded_source` `source_file` must act as an identity.** It is never
  normalised, so `..` is rejected (one file would take two identities) and an absolute
  path is rejected (the emitted key must be the same on every machine).

`embedded_source` is a **passthrough** kind: the descriptor is emitted as authored, no
producer runs, and it contributes no code object and no archive entry — the packer only
stamps the shard architecture and records provenance. A root of only passthrough kinds
therefore produces descriptors and **no** archive, and a shard with no compiled variant
holds no `kpack/`. Descriptors but no archive is legal; no descriptors never is.

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
pass `1`, or `2` for `integration`, and the production root and the packaging probes pass
none, so the packer uses its default. The reason is in the `hkp_wire_pack_target()` header
in `cmake/HkpPackaging.cmake`.

## Packaging probes

A packaging probe packs one descriptor root for one explicit architecture through the real
packer (`hkp_wire_pack_target()`), independent of the architectures the build targets, and
a ctest entry then asserts the packed output. The packer's architecture list is
`GPU_TARGETS` and nothing else, so without a probe a lane never packs an architecture it
does not build. A probe may use any architecture, built by the lane or not.

Probes are superbuild-only and exist only under `HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES`
(default `OFF`). Architecture probes are automatic (below); `probes/probes.cmake` declares
only named probes. With the option `ON`, the probe pack targets are part of `all`, and the aggregate target `hkp_packaging_probes`
builds them alone. Probe output is written under `<build>/hkp-probes/<NAME>/`, outside both
shipped descriptor trees, and is never installed. With the option `OFF` the only difference
is that cache entry; the probe module is never read.

**What a probe proves**

- The descriptors it packs, taken from the real root (ids, the specialization contract,
  the real KMD/UED/UDD/UHD), load, pass the pack-time agreement check and go through the
  packer for that architecture, including the compile for compiled kinds.
- The architecture and output wiring of the CMake pack target: `--arches`, the stamp, the
  interpreter and wheel directory, comgr forwarding (the last is asserted via
  `provenance-comgr` only; no local mutation measures it).
- The output: exactly one architecture directory, exactly the expected UKDs, an archive
  entry for each kpack-output UKD, and the provenance each producer kind ships.
- rocKE only: the packer records a wheel (`provenance-wheel`). The wheel mutation check
  below shows that the rocKE sources come from that wheel, not from the in-tree source the
  pytest suite imports.

**What it does not prove**

- UKDs it does not pack: the variants of a compile group the default sample leaves out,
  so in particular combinations of spec values (see the sweep below), or the UKDs not
  listed in `UKDS`.
- Anything outside the descriptor root and the packer, for example the gfx950 ASM SDPA
  (prebuilt `.co` files) and C++/runtime paths, and install staging.
- Device behavior: no kernel is launched, no numerics are checked.
- Wildcard descriptors on architectures no lane builds (see below for why none ship).
- That the CI job runs: that is the job of the junit check below.

### Which architectures are probed

Every architecture named explicitly under the production root gets an **automatic probe**
in every lane that does not build it. At configure, `hkp_load_packaging_probes()` runs
`tools/hkp_probe_derive_root.py --list-arches` on
`HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT`: the union of the explicit `arch`
entries of every KDP and UKD (inline and standalone), read with the packer's own loader.
It subtracts the lane's own architectures (`hkp_selected_arches()`, i.e. `GPU_TARGETS`)
and declares `hkp_add_packaging_probe(ARCH <arch>)` for each one left: probe `<arch>`,
ctest entry `hkp-probe-<arch>`, packing the default sample. Configure prints the result:

```
hkp: automatic probe arches: [gfx950] (explicit arches under <root>: [gfx950]; build targets: [gfx942])
hkp: probe gfx950: derived root <build>/hkp-probes/gfx950/root (17 UKD(s) for gfx950)
```

A new architecture, or a new pack under one, is therefore probed with no edit anywhere. A
lane whose `GPU_TARGETS` cover every named architecture declares no probe; that is valid,
and the manifest then lists `hkp-probe-tools` alone.

**Wildcards are excluded.** A descriptor with no `arch` (absent or empty) names no
architecture and gets no automatic probe. That is safe: a wildcard ships for whatever the
build targets, so each lane's own production pack packs it, and its GPU test lanes run it,
on every PR. Only an architecture the lane does not build needs a probe, and only an
explicit `arch` entry can make a descriptor ship there and nowhere else.

**Non-concrete names are skipped.** An explicit `arch` entry that is not a concrete
target, such as a generic target (`gfx9-4-generic`) or a TheRock family name, gets no
automatic probe; configure prints which ones it skipped. No build packs for such a name:
`GPU_TARGETS` drops it, and the packer matches arch names exactly, so it never selects a
family member in its place either. Concrete names with a distinct-target word
(`gfx1250-strict`) are probed like any other arch. The rule is `hkp_is_concrete_arch()`
in `cmake/HkpPackaging.cmake`, the same one `GPU_TARGETS` goes through.

### How to use

`probes/probes.cmake` is for **named `UKDS` probes** only. Add a line when an integration
needs specific UKDs packed on every PR, beyond the default sample: a combination of spec
values the sweep does not guarantee, or a variant with its own known risk.

```cmake
# One integration, a chosen representative set.
hkp_add_packaging_probe(ARCH gfx950 NAME attention_dense
    UKDS attention_dense.fp16_d64_hq8_kv1_c1_bm128_bn64.gfx950
         attention_dense.bf16_d64_hq8_kv1_c1_bm256_bn32.gfx950)
```

- **With `UKDS`**, the probe packs exactly the listed UKDs, selected by UKD `name`. Pick a
  minimal representative set; every listed UKD is packed, and compiled for a compiled
  kind, in every CI run. A KDP that keeps none of the listed UKDs is left out of the
  derived root. Give the line a `NAME`: the default name is the bare architecture, which
  that architecture's automatic probe uses, so the two would be declared twice.
- **Without `UKDS`**, a probe packs the default sample of every KDP in the root that ships
  for `ARCH` (see the derived root below). The automatic probes are of this form; a line
  without `UKDS` is only useful with another `ROOT`.

A listed name that is not kept fails configure: no KDP shipping for `ARCH` holds an inline
UKD with that name that itself ships for `ARCH` (a misspelled name, a UKD for another
architecture, or one reached only through a standalone-UKD reference). Listing a name twice
also fails configure.

| Argument | Meaning |
|---|---|
| `ARCH` | Required. The one architecture to pack, e.g. `gfx950` or `gfx1250-strict`; one concrete target as `hkp_is_concrete_arch()` defines it (not a generic target or TheRock family name). Passed to the packer as `--arches`. |
| `ROOT` | Optional descriptor root to probe. Defaults to the production root (`HIPKERNELPROVIDER_PRODUCTION_DESCRIPTOR_SOURCE_ROOT`). A relative path is resolved against `probes/`; a trailing `/` is ignored. |
| `NAME` | Optional probe name; must match `^[A-Za-z0-9_.+-]+$`, must not be dots only, and must not be `tools`. Defaults to `<ARCH>` for the production root and `<ROOT basename>_<ARCH>` for another `ROOT`. Pack target `hkp_packaging_probe_<NAME>`, ctest entry `hkp-probe-<NAME>`. Declaring a name twice is a configure error. |
| `UKDS` | Optional UKD `name`s to pack instead of the default sample. Each entry is non-empty and listed once. |
| `PACK_JOBS` | Optional prewarm worker count. Omitted, the packer uses its default, `min(32, ncpu)`, as the production root does. |

A probe has no producer-kind argument: the kinds present are discovered from the
descriptors, and the assertion expects each UKD's provenance by its authored
`kernel_source.kind`. Probes cover `hip`, `rocke`, `hsaco` and `embedded_source`, the kinds
registered in `tools/hkp_probe_kinds.py`. Each entry gives the `kernel_source` fields that
define a compile group, the provenance checks a shipped UKD must satisfy, and the packed
**output type**: `kpack` (the packer packs the UKD into the archive and ships a
`kind: kpack` UKD: `hip`, `rocke`, `hsaco`) or `passthrough` (the packer ships the UKD as
authored and nothing enters the archive: `embedded_source`).

The archive checks (`kpack-missing`, `kpack-empty`, `kpack-toc`, `sha256`, `symbol`) run
only when the expected list holds at least one kpack-output UKD. A probe of only
pass-through UKDs ships descriptors and no `kpack/` directory; the archive is then not
required and an archive that happens to exist is ignored.

Configuration fails, never skips, when: `ARCH` is missing or malformed; `NAME` is malformed,
`tools` or declared twice (a named probe whose `NAME` is an automatically probed
architecture); `ROOT` is not a directory; the packer's loader (`load_flat_input`) rejects
the root or the production root; no KDP under the root ships for `ARCH`; a `UKDS` entry is empty, repeated or not kept; a KDP the probe keeps references a
standalone UKD (a `kernelDescriptors` entry that is not an object; derive supports inline
UKDs only); a kept UKD has a kind not registered in `tools/hkp_probe_kinds.py`; two kept
UKDs of one KDP share a name; in the default sample, a UKD's `kernel_source` lacks one of
its kind's compile-group fields; the probe output root would sit under a shipped
descriptor tree; or pytest is not importable by `Python3_EXECUTABLE`.

### The derived root and the default sample

At configure, `tools/hkp_probe_derive_root.py` copies the root into
`<build>/hkp-probes/<NAME>/root`. Every KDP shipping for `ARCH` has its
`kernelDescriptors` reduced to the probed UKDs, among the UKDs that themselves ship for
`ARCH`: the `UKDS` list (one `--ukd` per entry), or else the **default sample**, chosen per
**compile group**. With `UKDS`, a KDP that keeps nothing is left out; other files are copied
unchanged. The compile group is the part of a UKD that selects its compile path:

| Producer kind | Compile group |
|---|---|
| `rocke` | `(kind, source, builder)` |
| `hip` | `(kind, source, build)` |
| `hsaco` | one per KDP (list more with `UKDS`) |
| `embedded_source` | one per KDP (list more with `UKDS`; ships as authored, no archive entry) |

Within a group, a kind with a **sweep field** (`sweep_field` in `tools/hkp_probe_kinds.py`;
`rocke`: `spec`) keeps a **value sweep**: every `(field, value)` pair of `spec` whose field
takes more than one value within the group must appear in at least one kept UKD (values
compared as canonical JSON; a field absent from some UKDs counts absence as a value).
Greedily, the UKD covering the most uncovered pairs is kept, ties broken by sorted `name`,
until every pair is covered; the result is deterministic. A group in which no field varies,
and every group of a kind without a sweep field (`hip`, `hsaco`, `embedded_source`), keeps
one UKD, the first by sorted `name`. Kept UKDs keep their authored order. The gfx950 dense
attention KDP (840 UKDs in one group, 7 varying spec fields) keeps 17.

- **Each value is packed; combinations are not.** The sweep guarantees, say, that some fp16
  UKD and some causal UKD are packed, not that an fp16 causal one is. Breakage specific to
  a combination is caught only if a UKD with it is listed in `UKDS`.
- **The key is the UKD `name`, not its `id`.** Ids are `uuid4` values regenerated on every
  ingestor run; names encode the specialization and are unique.
- **Derive reads the root with the packer's own loader** (`load_flat_input`): it sees
  the descriptors the packer sees (hidden paths skipped) and refuses what the packer
  refuses. Which KDPs ship is the packer's `kdp_survives`, which UKDs ship its
  `arch_matches`: an empty or absent `arch` list is a wildcard. Standalone UKDs are out of
  probe scope by policy.
- Derive also writes `expect.json` (kept KDP, UKD name and kind; pass-through kinds also
  carry the authored `kernel_source`) beside the root; the assertion expects exactly that
  set.
- The derive step re-runs when any file under the root changes, and rewrites only files
  whose content differs, so a reconfigure leaves the pack stamp fresh.

### Running the probes locally

From the repository root. The paths below are the workstation's; substitute your ROCm and
rocm_kpack locations.

```bash
cmake --preset hipdnn-dev-all -GNinja -B build-probe \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider" \
    -DGPU_TARGETS=gfx942 \
    -DHIPKERNELPROVIDER_ENABLE_ROCKE=ON -DHIPDNN_ENABLE_KERNEL_INGESTOR=ON \
    -DHIPDNN_ENABLE_SDPA=ON \
    -DHIPKERNELPROVIDER_KPACK_PYTHON_DIR=/opt/rocm-kpack/python \
    -DCMAKE_PREFIX_PATH=/opt/rocm -DROCM_PATH=/opt/rocm \
    -DPython3_EXECUTABLE=/usr/bin/python3 -DPython_EXECUTABLE=/usr/bin/python3 \
    -DENABLE_CLANG_FORMAT=OFF -DENABLE_CLANG_TIDY=OFF -DROCM_LIBS_ENABLE_ROOT_CTEST=ON \
    -DHIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES=ON
cmake --build build-probe --target hkp_packaging_probes   # or the full build
ctest --test-dir build-probe -R '^hkp-probe-' --no-tests=error --output-on-failure \
    --output-junit probe-junit.xml
python3 .github/scripts/check_probe_junit.py \
    --junit build-probe/probe-junit.xml --manifest build-probe/hkp-probes/manifest.txt
```

The selector is `^hkp-probe-`, anchored: `-R` is an unanchored regex. It matches every
probe (`hkp-probe-<NAME>`) and `hkp-probe-tools`, the pytest suite of the probe tooling
(`probes/tests/`). `--no-tests=error` makes an empty selection a failure instead of a
green run that proved nothing. `manifest.txt` is written at configure and lists the ctest
names that must run; the junit check fails unless every one ran and passed (a skipped,
disabled or missing entry fails), and ignores testcases not in the manifest, so it also
accepts the junit of a full ctest run. Configuring without the option and running the same
build and ctest lines fails (unknown target, then "No tests were found").

In CI both superbuild lanes configure with the option `ON`. The probe packs run in the
Build step as part of `all`; the probe ctest entries run once, inside the lane's normal
"Run tests" ctest, which writes `build/ctest-junit.xml`. The "Check packaging probes ran"
step then runs only `check_probe_junit.py` on that junit against the manifest. It runs
whenever "Run tests" ran, pass or fail.

Locally on Linux the build is quick (the wheel environment plus the probed UKDs) and each
probe ctest entry takes a few seconds; `hkp-probe-tools` takes longer, since it really
compiles hip and rocke fixtures. There is no local Windows reproduction.

### Assertion ids

`tools/hkp_probe_assert.py` prints one line per failure, `hkp_probe_assert: FAIL <id>:
<detail>`, and exits 1; the ids are stable and greppable.

| Id | Fails when |
|---|---|
| `out-root-missing` | The probe output root does not exist: build the probe pack target first. |
| `stamp-missing` | The pack stamp is absent. |
| `arch-dir-missing` / `extra-arch-dir` | `<out>/<arch>/` is absent, or another `gfx*` directory is present. |
| `kpack-missing` / `kpack-empty` | Kpack output is expected and `<out>/<arch>/kpack/hip_kernel_provider_<arch>.kpack` is absent or empty. |
| `no-kdp` | No `*.kdp.json` under `<out>/<arch>`, or one is unreadable or not a JSON object. |
| `ukd-kind` | A shipped UKD's `kernel_source.kind` differs from what its expected kind's output type requires: `kpack` for kpack output, the authored kind for pass-through output. |
| `arch-field` | A shipped UKD's `arch` is not exactly `[<arch>]` (an absent `arch` fails too), or a KDP's `arch`, when present, is not exactly `[<arch>]`. |
| `passthrough-source` | A pass-through UKD's `kernel_source` differs from the authored one recorded in `expect.json`. |
| `passthrough-provenance` | A pass-through UKD's `provenance.source_label` is empty, or `provenance.source_file` differs from `kernel_source.source_file`. |
| `kpack-toc` | The archive is unreadable, or has no entry for a UKD's `toc_key` and architecture. |
| `sha256` | The blob's sha256 differs from `kernel_source.sha256`. |
| `signature` | A kpack UKD's `kernel_source.signature` is empty or not a list. |
| `symbol` | A UKD's `symbol` does not appear in its blob. |
| `provenance-origin` | `origin_kind` differs from the UKD's expected kind (the authored `kernel_source.kind`). |
| `no-rules-for-kind` | A UKD's expected kind has no entry in `tools/hkp_probe_kinds.py`: a new producer kind must add its entry. |
| `provenance-wheel` | `rocke_wheel_sha256` is missing or empty: the packer recorded no rocKE wheel. |
| `provenance-comgr` | The comgr recorded at pack time is not the expected library (or is empty). Catches a stale system comgr shadowing the intended one. |
| `ukd-count` | The number of shipped UKDs found in the expected list (`expect.json`, written by derive) differs from its length, is zero, or a shipped UKD is not in the list. |

### How to extend

Every architecture named explicitly under the production root is probed in each lane that
does not build it, with the default sample; nothing needs declaring.

- **New architecture or new integration:** no edit. Reconfigure and check the
  `hkp: automatic probe arches` and `hkp: probe <arch>` STATUS lines.
- **Specific UKDs packed on every PR** (a combination the sweep does not guarantee): one
  line with `NAME <integration>` and `UKDS` naming them.
- **Another descriptor root:** one line with `ROOT <dir>`.
- **New producer kind:** the packer must support the `kernel_source.kind` first. Then one
  `KINDS` entry in `tools/hkp_probe_kinds.py` (compile-group fields, provenance checks by
  name, output type), one end-to-end test in `probes/tests/`, and the kinds list above.
- **New provenance check:** a function in `_PROVENANCE_CHECKS` (`tools/hkp_probe_assert.py`),
  its name in the kind's `provenance`, a row in the assertion-id table and a test showing
  the check fails on a mutated output.
- **New output type** (for example a runtime-compiled kind): an `_OUTPUT_CHECKS` entry plus
  the derive and assert sites that branch on the output type (`expect.json` content, the
  archive rule). This is a design decision, not an entry.

After adding an architecture or a probe line, reconfigure and run the local block above:
the new `hkp-probe-<NAME>` entry and its manifest line must appear. Then run the
mutation checks below.

### Mutation checks when adding a probe

Each mutation must turn the probe red. After each one: reconfigure, build the probe, run
its ctest entry, then revert the edit.

```bash
cmake build-probe
cmake --build build-probe --target hkp_packaging_probes
ctest --test-dir build-probe -R '^hkp-probe-<NAME>$' --output-on-failure
```

1. **Derived descriptors reach the compiler.**
   - Replace the `ARCH` of a named probe with an architecture no KDP in the root ships for:
     configure must fail (no KDP under the root ships for it).
   - Add a standalone-UKD reference (a non-object `kernelDescriptors` entry) to a KDP the
     probe keeps: configure must fail with
     `standalone UKD references unsupported by probe derive`.
   - For a compiled kind, set a compile-affecting spec field of a probed UKD (one in
     `expect.json`) to a value the compiler rejects: the build must fail in the packer. rocKE dense attention, for
     example: `block_n` in `kernel_source.spec` from `64` to `7` fails with
     `invalid spec ...`.
   - Make a probed UKD's `metadata` disagree with the compiled specialization: the build
     must fail with the pack-time agreement error. rocKE dense attention, for example:
     `block_n` in `metadata` from `64` to `128` fails with
     `compiled specialization disagrees with metadata`.
   - Also run `hip-kernel-provider-hkp-pack` on the mutated tree and record whether it
     stays green. For the gfx950 dense-attention probe it does, which is what makes the
     probe more than the existing test.
2. **rocKE only: the wheel supplies rocKE.** In `rocke/platform/pyproject.toml` change
   `rocke = ["**/*.json"]` to `rocke = ["analysis/**/*.json"]`, so the built wheel lacks
   `core/arch/data/arch_specs.json`. The probe build must fail with
   `FileNotFoundError ... arch_specs.json` from the private wheel directory, while
   `hip-kernel-provider-hkp-pack` (in-tree source) stays green.

   **Procedure hazard: remove the stale build products before and after this mutation.**
   Both are gitignored, and either carries `arch_specs.json` into the rebuilt wheel, so
   the edit is otherwise a silent no-op and the probe stays green:

   ```bash
   rm -rf dnn-providers/hip-kernel-provider/rocke/platform/build \
       dnn-providers/hip-kernel-provider/rocke/platform/python/rocke.egg-info
   ```

   A fresh checkout (CI) has neither directory.
3. **The architecture and outputs are wired.** In `cmake/HkpPackagingProbes.cmake`:
   - Replace the `ARCHES` argument of the pack target with an architecture no KDP in the
     root ships for: the build must fail with
     `produced nothing: no KDP survived arch pruning`.
   - Replace it with an empty string: the build succeeds, and the ctest entry must fail
     `hkp_probe_assert: FAIL arch-dir-missing`.
   - After a good build, delete `build-probe/hkp-probes/<NAME>/out/.hkp-packed.stamp`: the
     ctest entry must fail `hkp_probe_assert: FAIL stamp-missing`.

The wheel, architecture-wiring, spec and metadata mutations above turned the gfx950 probe
red when run locally on Linux. The Windows half of the wheel check, and comgr forwarding
(covered only by `provenance-comgr`), are verified by CI only.

## Running the tests

```bash
cd dnn-providers/hip-kernel-provider
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests -q
```

`rocm_kpack` (the third `PYTHONPATH` entry, or `--kpack-python-dir` /
`HIPKERNELPROVIDER_ROCM_KPACK_DIR`) is the kpack archive reader/writer most of the suite
round-trips through. Its absence is diagnosed once by the `rocm_kpack_dir` fixture in
`tests/conftest.py`: every test needing it skips with one message naming the dependency.
Set `HIPKERNELPROVIDER_KPACK_REQUIRE_ROCM_KPACK=1` (mirroring `_REQUIRE_HIPCC` /
`_REQUIRE_COMGR`) to turn that skip into a hard failure, e.g. in CI.

`-m quick` selects the load-time/pure-unit subset needing neither `hipcc` nor
`rocm_kpack`/comgr.

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

`tests/test_desk_check_invariants.py` exercises the shipped module. Keep structural
identity, real packed observations and tampered-evidence checks distinct from native
registration and numerical tests.

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

### Real-corpus builder-signature guards (`tests/test_hkp_pack_rocke.py`)

The real gfx942 `build_*` functions in `rocke/library/kernels/gfx942/` must satisfy
`_require_spec_arch_signature`'s `(spec, *, arch)` contract. The real-builder cases in
`tests/test_hkp_pack_rocke.py` and the rejection cases in
`tests/test_hkp_pack_producer_guards.py` cover complementary paths. Signature acceptance
alone is not effective-specialization or numerical proof.

```bash
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests/test_hkp_pack_rocke.py \
        descriptor-packaging/tests/test_hkp_pack_producer_guards.py -q
```
