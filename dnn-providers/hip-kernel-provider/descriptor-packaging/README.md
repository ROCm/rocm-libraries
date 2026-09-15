# hkp_pack -- descriptor packaging

Build-time UKD/KMD/KDP -> kpack packaging. Provider-internal (`tools/hkp_pack.py`);
see `python/hkp_pack/` for the pipeline itself and `examples/descriptors/` for a
real, minimal authored source root.

## Source roots and what the walk accepts

Each wired root is walked recursively and packs straight into **its own** `OUT_ROOT`;
no two invocations share a destination and there is no shared stage tree. Each
descriptor's authored subpath is preserved verbatim into the staged and installed
trees. Producer selection is per-UKD on `kernel_source.kind`, never per-folder, so
one root feeds every producer into one kpack per arch. Nothing is registered in
CMake: adding a descriptor is dropping files in a folder.

The provider wires six: the production root, plus five over the four authored test
sets (`shared` packs twice, once into each test binary's discovery root).

| Root | Source | Ships |
|---|---|---|
| Production | `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, a `CACHE PATH` defaulting to the in-tree `src/engines/kernel_ingestor_engine/descriptors/` | yes |
| Test | `src/engines/kernel_ingestor_engine/test_descriptors/{shared,unit,integration,archive_fixture}/` | only under `HIPKERNELPROVIDER_ENABLE_TESTS` |

Production wiring is gated on the root holding at least one non-hidden `*.kdp.json`,
since a KDP is what arch pruning consumes. With none, packaging is **dormant**, any
stale product tree is removed, and neither is an error; a KDP that is present but
pruned on every arch stays a hard failure, which is what separates "nothing to ship"
from "something to ship that did not". A root that is set but is not a directory is
fatal at configure.

Two rules govern the walk itself:

- **Hidden paths are skipped, and said so.** Any dot-prefixed path segment or
  dot-prefixed filename is warned and skipped, as is a `*.json` whose name carries no
  type token — an incidental file or a `.git/` under a user-supplied root is
  tolerated rather than aborting the pack, and nothing passed over is silent. The
  production content gate drops the same segments, so a KDP under a hidden path does
  not wire packaging. A type-tagged descriptor that is malformed, missing a field, of
  an unknown type or carrying a dangling reference still fails.
- **An `embedded_source` `source_file` must be able to act as an identity.** The
  value is never normalised, so a `..` segment is rejected (one file would take two
  identities under two spellings) and an absolute path is rejected (it names a
  location on one machine, while the emitted key must be the same on every machine).

`embedded_source` is a **passthrough** kind: the descriptor is emitted exactly as
authored, no producer runs for it, and it contributes no code object and no archive
entry. The packer stamps the shard architecture and records the authored values in a
provenance block. A root of only passthrough kinds therefore legitimately produces
descriptors and **no** archive, and a shard with no compiled variant holds no
`kpack/` directory. Descriptors but no archive is legal; no descriptors never is.
Compiled-specialization obligations are scoped to the compiling kinds they are
defined for, and stay mandatory for every one of those.

## Compiler-bound specialization agreement

Generic descriptor generation is toolchain-free and supplies declarations, not
compiler evidence. Packaging consumes UKD `provenance.specialization_contract` as
data; it does not bind an authoring profile, redirect compiler import roots or
import a second policy implementation. There is no packaging `--profile` option,
CMake `PROFILES` cache input or external root manifest. Profiles remain
authoring/mining inputs whose relevant declarations travel with generated UKDs.

The declaration has `schema_version: 1` and `consumers`, each containing `engine_id`,
`kmd_id`, `metadata_fields`, `matcher_only_fields`, `bindings` and `vocabulary`.
The two field lists exhaustively and disjointly partition the referenced KMD's
fields. Binding keys are exactly `metadata_fields`; each value is exactly
`{field: "<attr>"}` or `{method: "<accessor>"}`. IDs reference existing descriptors,
without copying KMD type/default definitions. Shared standalone UKDs carry all
consumers; duplicate/conflicting declarations fail. See the
[generator agreement reference](../../../projects/hipdnn/tools/IngestorGenerator/README.md#specialization-agreement)
for authoring and projection semantics.

### Actual compiler observations

Use the selected compiler interpreter and the actual imported builder/spec objects.
`build_spec` hydrates ordinary defaults and default factories on the object passed
to `builder_fn`. Observe declared direct fields only when the builder uses them
without further resolution; otherwise observe the zero-argument bound effective
accessor the builder actually consumes. Read it even when the raw field is non-null:
coupling may override an explicit value, such as swizzle true with conflict-free V
disabled. Never guess accessor conventions, copy policy formulas or reconstruct a
different spec object for comparison.

`None` remains authored intent, not a wildcard or an instruction to substitute
false. Missing/noncallable accessors, unresolved `None`, unsupported return types,
exceptions or non-repeatable observations block full agreement. Matcher-only
classification needs a source-use-site audit and independent review; a consumed
specialization field cannot be exempted merely to make a check pass.

Complete and type metadata through the actual referenced KMD. BOOL remains boolean;
a boolean may intentionally project to 0/1 for INT; FLOAT normalizes numerically.
Descriptor-side type errors are not silently coerced. Compare observed effective
values against each consumer's completed metadata before publication.

### Shared compilation and reserved evidence

Collect every consumer's observation requests before compiling a shared variant.
Serial and prewarm/worker results carry the code object, captured symbol,
architecture and serializable observations keyed by canonical declaration digest.
Every consumer is compared independently, including on result reuse; one successful
consumer cannot certify a contradictory second consumer. Reuse is within the
existing packaging invocation/producer context, not a new persistent observation
cache.

Preserve authored `provenance.spec` separately and untouched. Only the producing
compiler writes the reserved `provenance.effective_spec` record. Authored rocKE
inputs supplying a purported record are rejected. Extra/provenance passthrough
must not overwrite fresh observations during UKD rewriting or final publication.
Packed-input validation reads the actual packed record; it does not pretend to
recompile authored input.

The schema-versioned producing-build record binds effective values and observation
requests, canonical authored-input digest, observed builder/spec/accessor identities
and origins, consumer UKD/engine/KMD IDs, KMD content, completed metadata, KDP/effective
architecture and actual library/toc-key/symbol/payload SHA256. Its binding digest
excludes the evidence itself. The declaration remains alongside the record so a
checker needs no external profile.

Producer identities are qualified names and defining-file paths/content hashes
observed from actual imported objects. Required unresolvable origins fail evidence
production; producer files must remain stable during the invocation. Wheel stamps
are separately labeled build provenance, not proof of imported origins or the
entire transitive toolchain. Existing hermetic wheel selection and wheel-content
build dependencies remain responsible for rebuilds after producer changes.

### Full versus structural checking

A full check validates the self-contained declaration and producing-build record
against current descriptors, schema, metadata, architecture and named payload
bytes. Missing, unsupported, stale or mismatched required records fail. A valid
packed artifact can be fully checked **without rocKE installed on the verifying
machine**: the checker neither imports today's producer nor claims an older
artifact came from it.

An explicit structural-only check may pass the properties it actually checks, but
does not satisfy compiled-specialization agreement. Optional mining/analysis
profiles cannot supply or override compiler evidence. Observations establish
agreement with actual builder decisions and artifact integrity, not formal
equivalence of arbitrary machine code or correctness of native dispatch.

The runtime consumes packed per-architecture descriptors with source kind KPACK,
not unlowered rocKE/HIP authoring descriptors. A packed `kernel_source` carries
**five mandatory keys**; the packager emits them together and an adapter needs every
one of them to name a code object and vouch for it:

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

`signature` is required, not optional: an empty array is legal and means a kernel
taking no arguments, which is exactly why the key itself must be present — absence
would otherwise be indistinguishable from it. Per entry, `kind`, `size` and `offset`
are mandatory and `name` is the one optional key, because clang emits argument names
for some producers and omits them for HIP `extern "C" __global__` kernels; requiring
it would make every HIP-compiled kernel unloadable. `sha256` is shape-checked as 64
lowercase hex.

Neither digest nor signature is hand-authored, and they catch different drift.
`sha256` is byte identity of the **decompressed** code object: a TOC entry pointing
at the wrong offset decompresses cleanly and hands back another entry's object, so
without the digest the wrong kernel launches and nothing reports an error. The
loader rehashes before `hipModuleLoadData` and raises `DIGEST_MISMATCH`.
`kernel_signature.py` reads the argument list back out of the object the packer just
compiled — sniffing a clang offload bundle or a bare ELF rather than assuming, and
dropping compiler-appended `hidden_` arguments — which is what catches a kernel whose
parameters changed while its bytes remain internally consistent. A hand-authored
arity would restate the same assumption that drifted.

Native proof separately executes actual typed provider registration/loading and a
finalized emitted-bundle census; source-text symbol matching is not certification.
Census registration is one call per packed target, made in
`src/tests/CMakeLists.txt` beside `hkp_verify_embedded_sources()`:

```cmake
hkp_register_census_tests(
    TARGET hip_kernel_provider_tests
    PACK_NAME unit
    SUITES TestPointwisePacks)
```

`PACK_NAME` selects the wired pack target whose `OUT_ROOT` and recorded arch list the
entries address. Per declared suite and per arch in that list, CMake registers
`hip-kernel-provider-hkp-census-<arch>-<suite>`, invoking
`hip_kernel_provider_tests --gtest_filter=<suite>.*` directly, without Python, with
`HIPDNN_TEST_CENSUS_SUITE=<suite>`, `HIPDNN_TEST_EXPECTED_ARCH=<arch>` and
`HIPDNN_DESCRIPTOR_DIR=<that pack target's OUT_ROOT>/<arch>` — its own shard, not a
shared stage tree. Each entry is an independent process labeled
`unit_test;hip-kernel-provider;host`.

```bash
ctest --test-dir <build>/dnn-providers/hip-kernel-provider \
  --no-tests=error -V -R '^hip-kernel-provider-hkp-census-<arch>-<suite>$'
```

A suite is declarable only where it reads **exactly one** pack target's shard,
because an entry hands the binary one directory and the guard below requires every
case to pass. The authored dialect does not decide this: `TestPointwisePacks` is
censused at `unit` although that set is `embedded_source`, while `TestConvFwdPack`
reads the `unit` and `unit_shared` shards and is censused nowhere. One suite declared
at two pack targets is fatal — the entry name carries arch and suite alone, so the
second registration would silently take the first one's shard.

Strict mode is active only for a nonempty census-suite variable. Before default-root
setup it rejects missing/empty/nonexistent explicit roots and empty expected arches;
the production loader's normal fallback is unchanged. The exact named suite must
exist and be nonempty. Every registered case must complete and pass without skipping
in each iteration, with at least one completed iteration. Disabled, filtered-out,
sharded-out, failed or skipped cases, list-only and repeat-zero invocations cannot
satisfy the census. Repeated partial runs cannot accumulate coverage.

The call is made where the target is defined and after it exists; there is no
deferral machinery. Each missing prerequisite is fatal rather than a silent drop,
because a census that registers nothing is indistinguishable from one that passed: an
unwired `PACK_NAME` (the message names the wired roots), an absent or nonexistent
`TARGET`, an empty recorded arch list. Tests OFF and an empty `SUITES` register
nothing, which is absence of evidence. Normal non-census invocations retain their
filtering and skip behavior. Neither structural nor host loading proves numerical
device behavior. The
[ingestor RUNBOOK](../../../projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine/RUNBOOK.md)
owns the complete create/extend sequence and post-regeneration gates.

## Build speed: put the comgr cache on local storage

Packing a rocKE descriptor set is dominated by lowering each kernel through
`libamd_comgr`, which caches its results on disk. That cache defaults to
**`~/.cache/comgr`**, so on a machine whose home directory is a network filesystem
every lookup is a network round trip and packing slows by an order of magnitude.
Point it somewhere local:

```bash
export AMD_COMGR_CACHE_DIR=/tmp/comgr-cache   # RAM disk or local disk
```

Measured packing a 2,711-kernel gfx942 set at 32 workers, varying only the cache:
13 s warm on tmpfs, 47 s cold on tmpfs, 597 s warm on a network home. A cold local
cache beat a warm network one by more than 10x, so on a network home the cache costs
more than it saves. `AMD_COMGR_CACHE=0` disables caching outright, which is a
diagnostic rather than a fix.

One knob belongs to the packer itself:

| Variable | Effect |
|---|---|
| `HKP_PACK_JOBS` | Prewarm worker count. Defaults to `min(32, ncpu)`; `1` forces the serial path for a clean traceback. |

That statement is about a **direct child run** of `hkp_pack`, and it stays true there.
Inside the build the cap is not an environment variable anyone exports: it is the
`PACK_JOBS <n>` argument at the `hkp_wire_pack_target()` call site, and the wiring is
what transports it to the tool as `HKP_PACK_JOBS`. **No wrapper needs to export it
any more.**

The argument exists per call site because every root is a separate custom target with
no ordering edge between them, so the generator runs them at once and unbounded pools
multiply. All six wire calls name a value: `1` selects the packer's serial path for
the small roots, and `2` goes to the two roots with enough distinct variants to repay
a pool — the production root and the `integration` test root, which is the one that
exercises the parallel path in a real build. Omitting the argument lets the packer
size itself against the machine, which fits only a root large enough to repay the
startup cost.

## Running the tests

```bash
cd dnn-providers/hip-kernel-provider
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests -q
```

`rocm_kpack` (the third `PYTHONPATH` entry, or `--kpack-python-dir` /
`HIPKERNELPROVIDER_ROCM_KPACK_DIR`) is required by most of the suite -- it is the
kpack archive reader/writer every packing test round-trips through. Its absence is
diagnosed ONCE, clearly, by the `rocm_kpack_dir` fixture in `tests/conftest.py`: every
test needing it skips with one message naming the missing dependency, rather than each
failing separately deep inside `run_pipeline` with the same raw import error repeated
per test. Set `HIPKERNELPROVIDER_KPACK_REQUIRE_ROCM_KPACK=1` (mirrors the existing
`_REQUIRE_HIPCC`/`_REQUIRE_COMGR` pattern) to turn that skip into a hard failure, e.g.
in CI where a missing dependency must not go silent.

`-m quick` selects the load-time/pure-unit subset that needs neither `hipcc` nor
`rocm_kpack`/comgr -- useful on a box provisioned for neither.

### Desk-check a variant set (`hkp_pack.desk_check`, `tools/hkp_desk_check.py`)

The desk check resolves KDP engine → UED metadata → KMD UUID within the selected
descriptor tree/shards. It completes defaults and canonicalizes metadata through
that schema. Tuple identity is engine-wide, including cross-pack overlap in
effective architectures: equal metadata on disjoint architectures is legal,
conflicting overlapping candidates are not. A KDP filename is not a KMD reference.

Full checking uses the embedded declarations and producer-owned effective
observations described above, not merely matching explicitly authored spec keys.
Constructor defaults, omitted policy and coupled effective accessors remain
obligations. An authored tree without compiler evidence can receive only an
explicit structural result; it cannot be labeled compiler-clean.

The matcher field list is resolved in a fixed order of precedence: an explicit
`--field` outranks everything; otherwise, when the bundle declares
`specialization_contract.metadata_fields`, **that declaration is the field list**;
`DEFAULT_MATCHER_FIELDS` is the last-resort fallback left for a bundle that declares
no contract. Desk-checking a bundle against the fields it actually declares is the
point — falling back to the generic list for a bundle that states its own would check
a different set of columns and still print a result.

`tools/hkp_desk_check.py --mode {full,structural} [--kpack-python-dir D]
[--field F] [--drift-field F] <path/to/*.kdp.json>` selects the intended proof
strength explicitly. `--mode` is required and has no default, because a default
would let a structural run read as a full one; every run prints a leading
`mode=<full|structural>` line and a `compiled specialization agreement:` line.
`--mode structural` accepts an authored (pre-pack) or a shipped file and reports
that agreement as NOT CHECKED; `--mode full` is for a shipped shard, and binds each
kernel whose declaration names specialized metadata fields to the producing build's
record and the archive bytes the descriptor names, reporting `OK for N kernel(s)`.
A missing, unsupported or mismatched record for such a kernel is a failure, never a
successful COULD-NOT-CHECK result. A kernel declaring no specialized metadata field
has no record to bind and is listed by name under `compiled specialization NO
COMPILED CLAIM:` — never absorbed into the pass line. That list is printed
alongside `OK for N kernel(s)` in a mixed KDP; only when NO kernel makes a claim
does the agreement line itself read `NO COMPILED CLAIM`. Archive-key/symbol
properties that do not apply to authored input do not establish publication
correctness. Symbol reuse by itself is not evidence that two metadata tuples are
identical.

`tests/test_desk_check_invariants.py` exercises the shipped module. Keep structural
identity, real packed observations and tampered-evidence checks distinct from
native registration and numerical tests; a synthetic predicate or controlled
payload establishes only its specific boundary.

### Embedded-source verification (`tools/hkp_verify_embedded_sources.py`)

A staged tree holds descriptor JSON only, so an `embedded_source` descriptor resolves
its `source_file` against a key table the build compiles into the binary, and nothing
in the staged tree proves that table holds the named source. This build step reads
that table and every `embedded_source` descriptor under the staged roots the binary
serves, and compares the two: **presence** (each named `source_file` is a key) and
**location** (the file registered under that key is the one at the authored location
the descriptor's provenance records, joining the `provenance.source_label` root with
`rel_dir` and `source_file`). `--pack-stamp` adds a separate rule: a pack root whose
stamp is present holds at least one descriptor. A root whose pack is not wired — the
dormant production root — contributes no stamp and is not checked.

It runs over emitted JSON alone and imports no part of the packer, so it restates the
contract instead of recomputing one side of it from the other.

**The comparison runs one way, staged descriptor → table. Neither reverse direction
is checked, so a pass is not evidence that a bundle is reachable.** A key the table
holds that no descriptor names is not an error: most embedded kernels have no
descriptor at all. And per the module's own docstring, *"a descriptor that never
reaches a staged root is not an error either. Authored under a folder no pack is
wired to, it is never staged, so this walk never sees it and passes while the runtime
never receives it."* Catching that needs the authored tree as a second input, which
provenance cannot supply, because the packer is what writes provenance. The check to
state is *does a shard appear under that pack target's `OUT_ROOT`*, not *did the
verifier pass*. An absent root, an empty root, a root with no `embedded_source`
descriptor and an absent key table each pass — which is why a pass reports the two
counts it compared, so a pass over nothing reads differently in the build log from a
step that did not run.

### Real-corpus builder-signature guards (`tests/test_hkp_pack_rocke.py`)

The real gfx942 `build_*` functions in `rocke/library/kernels/gfx942/` must satisfy
`_require_spec_arch_signature`'s `(spec, *, arch)` contract. The real-builder cases
in `tests/test_hkp_pack_rocke.py` and rejection cases in
`tests/test_hkp_pack_producer_guards.py` cover complementary paths. Signature
acceptance alone is not effective-specialization or numerical proof.

```bash
PYTHONPATH=descriptor-packaging/python:rocke/library:rocke/platform/python:/opt/rocm-kpack/python \
    python3 -m pytest descriptor-packaging/tests/test_hkp_pack_rocke.py \
        descriptor-packaging/tests/test_hkp_pack_producer_guards.py -q
```
