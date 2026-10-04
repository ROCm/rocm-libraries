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
unlowered rocKE/HIP authoring descriptors. A packed `kernel_source` carries **five
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

## Packaging probes

A packaging probe proves that one integration's descriptors **pack for one explicit
architecture**, independent of the architectures the build itself targets. The packer's
architecture list is `GPU_TARGETS` and nothing else, so a lane that builds gfx942 (Linux)
or gfx1151 (Windows) never exercises the gfx950 production bundle. A probe closes that
gap: it runs the real packer through `hkp_wire_pack_target()` with its own architecture,
then a ctest entry asserts the packed output.

**What a probe proves**

- The real production descriptors (UUID ids, the specialization contract, the real
  KMD/UED/UDD/UHD) load, pass the pack-time agreement check and compile through comgr.
- The rocKE sources come from the wheel the packer is given, not from the in-tree source
  tree the pytest suite imports.
- The architecture and output wiring of the CMake pack target: `--arches`, the stamp, the
  interpreter and wheel directory, comgr forwarding (the last is asserted via `provenance-comgr` only; no local mutation measures it).
- The output: a kpack archive for exactly that architecture, with the expected UKD count
  and shipped provenance.

**What it does not prove**

- Instances other than the named representative (see below).
- Device behavior: no kernel is launched, no numerics are checked.
- Anything about integrations without a probe.
- That the CI job runs: that is the workflow's job (`--no-tests=error`, the manifest check
  below).

Probes are build-tree ctest entries only. They exist only under
`HIPKERNELPROVIDER_ENABLE_PACKAGING_PROBES` (default `OFF`, CI-only), are never installed
and write under `<build>/hkp-probes/<name>/`, outside both shipped descriptor trees.
With the option `OFF` the only cache or behavior difference is that cache entry (the new module also becomes a CMake regeneration input, which changes no target).

### Running the probes locally

From the repository root. The paths below are the workstation's; substitute your ROCm and
rocm_kpack locations. The `GPU_TARGETS` value must differ from every probe architecture
(see "Adding a probe").

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
cmake --build build-probe --target hkp_packaging_probes
ctest --test-dir build-probe -R '^hkp-probe-' --no-tests=error --output-on-failure \
    --output-junit probe-junit.xml
python3 .github/scripts/check_probe_junit.py \
    --junit build-probe/probe-junit.xml --manifest build-probe/hkp-probes/manifest.txt
```

The selector is `^hkp-probe-`, anchored: `-R` is an unanchored regex. It matches every
probe (`hkp-probe-<name>`) and `hkp-probe-tools`, the pytest suite of the probe tooling
(`probes/tests/`). `--no-tests=error` makes an empty selection a failure instead of a
green run that proved nothing. `manifest.txt` is written at configure and lists the
ctest names that must run; the junit check fails unless every one ran with no failure,
error or skip. Configuring without the option and running the same build and ctest lines
fails (unknown target, then "No tests were found").

Locally on Linux the build takes about 35 s (the wheel environment plus a one-instance
pack) and the two ctest entries about 3 s. There is no local Windows reproduction.

### `hkp_add_packaging_probe` arguments

Declared in `probes/probes.cmake`, defined in `cmake/HkpPackagingProbes.cmake`.

| Argument | Meaning |
|---|---|
| `NAME` | Probe name. Pack target `hkp_packaging_probe_<NAME>`, ctest entry `hkp-probe-<NAME>`. Declaring a name twice is a configure error. |
| `ARCH` | The one architecture to pack, e.g. `gfx950`. Passed to the packer as `--arches`. |
| `KIND` | Producer kind. Only `rocke` is supported; any other value is a configure error naming the supported kinds. |
| `DERIVE_FROM` | Directory of the real production descriptors to derive the probe root from. |
| `KDP` | File name of the KDP inside `DERIVE_FROM` whose `kernelDescriptors` is trimmed. |
| `INSTANCE` | The UKD `name` kept as the representative instance. |
| `PACK_JOBS` | Optional prewarm worker count; defaults to `1`. |

There is one mode, `DERIVE_FROM`. A probe has no `ROOT` argument for a hand-written or
frozen root: a frozen copy drifts silently from the production schema, which is exactly
what a probe exists to catch. The expected UKD count is fixed at one.

Configuration fails, never skips, when: `INSTANCE` is not found exactly once in `KDP`;
the derived root has no KDP covering `ARCH`; the probe output root would sit under a
shipped descriptor tree; no probe is declared; or pytest is not importable by
`Python3_EXECUTABLE`.

### The derived-root representative-instance rule

At configure, `tools/hkp_probe_derive_root.py` copies the `DERIVE_FROM` directory into
`<build>/hkp-probes/<NAME>/root` unchanged, except that `KDP`'s `kernelDescriptors` is
reduced to the single UKD whose `name` equals `INSTANCE`. The probe therefore packs the
real descriptors but compiles one kernel (about 1 s) instead of the whole family.

- **The key is the UKD `name`, not its `id`.** Ids are `uuid4` values regenerated on every
  ingestor run; names encode the specialization and are unique. Keying on the id would
  turn every legitimate regeneration into a red probe.
- **Choose the instance whose compile path every variant of that family shares.** One
  compile then stands for the family. Breakage specific to the other instances is not
  caught; that is the accepted gap of the design.
- **When the named instance is renamed or removed**, configure fails with
  `hkp_probe_derive: instance '<name>' not found in <kdp> (<N> descriptors)` (or
  `... matched <k> times`). That is the intended signal. Pick another instance of the
  same family from the KDP and update `INSTANCE` in `probes/probes.cmake`.
- The derive step re-runs when any file under `DERIVE_FROM` changes.

### Assertion ids

`tools/hkp_probe_assert.py` prints one line per failure, `hkp_probe_assert: FAIL <id>:
<detail>`, and exits 1; the ids are stable and greppable.

| Id | Fails when |
|---|---|
| `out-root-missing` | The probe output root does not exist: build `hkp_packaging_probes` first. |
| `stamp-missing` | The pack stamp is absent. |
| `arch-dir-missing` / `extra-arch-dir` | `<out>/<arch>/` is absent, or another `gfx*` directory is present. |
| `kpack-missing` / `kpack-empty` | `<out>/<arch>/kpack/hip_kernel_provider_<arch>.kpack` is absent or empty. |
| `no-kdp` | No `*.kdp.json` under `<out>/<arch>`. |
| `ukd-kind` | A shipped UKD's `kernel_source.kind` is not `kpack`. |
| `arch-field` | A KDP or UKD `arch` is not exactly the probe architecture. |
| `kpack-toc` | The archive has no entry for a UKD's `toc_key` and architecture. |
| `sha256` | The blob's sha256 differs from `kernel_source.sha256`. |
| `signature` | A UKD has an empty `signature`. |
| `symbol` | A UKD's `symbol` does not appear in its blob. |
| `provenance-origin` | `origin_kind` is not `rocke` (for `KIND rocke`). |
| `provenance-wheel` | `rocke_wheel_sha256` is missing or empty: the wheel did not supply rocKE. |
| `provenance-comgr` | The comgr recorded at pack time is not the expected library (or is empty). Catches a stale system comgr shadowing the intended one. |
| `ukd-count` | The number of kpack-kind UKDs is not the expected one, or is zero. |

### Adding a probe

Required for every new architecture or integration. **No CMake guard enforces this:** a
new integration without a probe configures, builds and passes CI, and its packaging for
an architecture the lanes do not build is then unchecked.

1. Add one `hkp_add_packaging_probe` declaration to `probes/probes.cmake`, with a comment
   stating why the chosen `INSTANCE` represents its family. No workflow edit: the jobs
   build `hkp_packaging_probes` and select `^hkp-probe-`, and the manifest counts every
   declared probe.
2. **The probe architecture must differ from the lane's `GPU_TARGETS`** (gfx942 on the
   Linux job, gfx1151 on the Windows job). If they are equal, the architecture-wiring
   mutation below becomes unobservable on that job. If a probe must use a lane's
   architecture, change the job's `GPU_TARGETS` first.
3. Reconfigure and run the local block above; the new `hkp-probe-<NAME>` entry and the
   manifest line must appear.
4. Run the mutation checks below. Each must turn the probe red, and each edit must be
   reverted afterwards.

### Mutation checks when adding a probe

After each mutation: reconfigure, then build `hkp_packaging_probes`, then run the probe
ctest entry, then revert the edit.

```bash
cmake build-probe
cmake --build build-probe --target hkp_packaging_probes
ctest --test-dir build-probe -R '^hkp-probe-<NAME>$' --output-on-failure
```

1. **Derived descriptors reach the compiler.**
   - Rename the representative UKD in the production KDP: configure must fail with
     `hkp_probe_derive: instance '<name>' not found`.
   - Set a compile-affecting spec field of that UKD to a value the builder rejects (for
     dense attention: `block_n` in `kernel_source.spec` from `64` to `7`): the build must
     fail in the packer (`invalid spec ...`).
   - Make a UKD `metadata` field disagree with the compiled specialization (`block_n` in
     `metadata` from `64` to `128`): the build must fail with the pack-time agreement
     error (`compiled specialization disagrees with metadata`).
   - Also run `hip-kernel-provider-hkp-pack` on the mutated tree and record whether it
     stays green. For the gfx950 dense-attention probe it does, which is what makes the
     probe more than the existing test.
2. **The wheel supplies rocKE.** In `rocke/platform/pyproject.toml` change
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
   - Replace the `ARCHES` argument of the pack target with the lane's architecture
     (`gfx942`): the build must fail with `produced nothing: no KDP survived arch pruning`.
   - Replace it with an empty string: the build succeeds, and the ctest entry must fail
     `hkp_probe_assert: FAIL arch-dir-missing`.
   - After a good build, delete `build-probe/hkp-probes/<NAME>/out/.hkp-packed.stamp`: the
     ctest entry must fail `hkp_probe_assert: FAIL stamp-missing`.

Every mutation above turned the gfx950 dense-attention probe red when run locally on
Linux. The Windows half of the wheel check, and comgr forwarding (covered only by
`provenance-comgr`), are verified by CI only.

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
