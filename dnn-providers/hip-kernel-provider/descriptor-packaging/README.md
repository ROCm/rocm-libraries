# hkp_pack -- descriptor packaging

Build-time UKD/KMD/KDP -> kpack packaging. Provider-internal (`tools/hkp_pack.py`);
see `python/hkp_pack/` for the pipeline itself and `examples/descriptors/` for a
real, minimal authored source root.

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
not unlowered rocKE/HIP authoring descriptors. Native proof separately executes
actual typed provider registration/loading and a finalized emitted-bundle census;
source-text symbol matching is not certification. Host checks use explicit
`HIPDNN_TEST_EXPECTED_ARCH` from configured packaging arches with the corresponding
shard and nonempty exact test selection. Neither structural nor host loading
proves numerical device behavior. The
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
