# Runbook: create or extend an ingestor engine

Read [SKILL.md](SKILL.md)'s entry and completion contracts first. **This is the only
ordered create/extend workflow.** Domain pages explain contracts; they do not
replace stages below. A gate requires its actual observation, not a command copied
into a report, a nonempty artifact or an earlier install's success.

## Paths and interpreters

Resolve absolute paths in the checkout and execution environment. The names below
are shell variables for explicit command arguments, **not implicit tool inputs**:

```bash
REPO=/absolute/path/to/rocm-libraries
PROVIDER="$REPO/dnn-providers/hip-kernel-provider"
GEN="$REPO/projects/hipdnn/tools/IngestorGenerator"
PY="$GEN/.venv/bin/python"
BUILD=/absolute/path/to/build
INSTALL=/absolute/path/to/future-install
SWEEP_ROOT=/existing/device-visible/writable/experiment-root
GENERATED=/absolute/path/to/empty-generation-destination
CONFIG=/absolute/path/to/generator-config.yaml
PROFILE=/absolute/path/to/authoring.profile.yaml
SHAPES=/absolute/path/to/request-shapes.json
CORPUS_DIR=/absolute/path/to/graph-corpora
ARCH=gfx942
ENGINE=hipkernel:Gfx942AttentionDense
```

Use [IngestorGenerator setup](../../../IngestorGenerator/README.md#setup) for its
venv. All generator and Python-tool commands below use that interpreter and an
absolute script path, so they are independent of the current working directory.
Authoring/mining tools that import rocKE need their declared library environment;
this is distinct from production packaging's selected compiler/wheel interpreter.
Do not redirect production compiler imports to an editable profile root or assume
system Python already has the required packages.

Build, install, corpus and output paths must be visible where their commands run.
A workspace symlink into a login machine's local disk is not shared storage. Follow
the active build/test and scheduler skills; stage exact source/artifacts when
needed. Never execute GPU work on a login host. Keep command logs and source,
artifact, machine, device and job identities in the run evidence.

## Sequence and gates

| Stage | Required result |
|---|---|
| 1. Entry and early feasibility | Source/identity/scope recorded; actual target device and writable root; representable graph and capable numerical reference, with no install requirement |
| 2. Contracts, corpus and baseline approval | Graph/edge and kernel/ABI evidence; complete semantic corpus inventory; approved feature/shape/knob boundary and provisional baseline |
| 3. Generate, implement and splice | Final authored baseline, completed applicable native hooks, addition-safe splices, field/placeholder checks and structural results |
| 4. Build, pack, install and host proof | Current install, applicable compiler-bound artifact agreement, actual typed native registration and emitted-bundle census |
| 5. Baseline device proof | Installed preflight, exact-engine quick/standard numerics and required negatives; selected/served/skipped/failed counts |
| 6. Measured tuning and final selection | Built/installed arms measured in order; explicit selection; regenerated final package repeats stages 3–5 |
| 7. Final corpora and runtime reconciliation | Fresh final-artifact measurement and independent correctness; complete outcome join for every input; justified coverage |
| 8. Handoff | Final identities and evidence for every applicable gate; exact limitations and no unexplained in-scope cases |

An unresolved correctness contract, reference, dependency, source or runtime failure
blocks its gate. Do not proceed on an assumed default merely because a time budget
expired. Scope changes require explicit approval. Host-only investigation may
continue while a device allocation is pending, but it cannot make a blocked device
gate green.

## 1. Entry and early feasibility

Record create/extend, source revision and actual source kind, exact engine name and
architecture. For extend, inventory the known-good installed engine and preserve
its identities per [extend.md](extend.md). Choose `direct_load` for embedded HIP or
`packaged` for build-time sources; rocKE is always packaged.

Establish reference capability against the intended graph semantics, not just its
operation name. Read [graph-contract.md](graph-contract.md) and the current
[integration-test reference limits](../../../../../../dnn-providers/integration-tests/README.md#what-the-reference-executors-cannot-verify).
Both CPU and GPU SDPA plans reject `sink_token_tensor_uid`. CPU is not a fallback
for sinks. If no capable independent numerical oracle exists, stop that claimed
feature and obtain an explicit scope/reference decision. Never invent golden
output or treat `auto` falling through to skip as verification.

Run on the **actual execution host with the target GPU**, directly or through the
active allocated-job skill:

```bash
"$PY" "$GEN/tools/device_probe.py" \
  --mode early --arch "$ARCH" --sweep-root "$SWEEP_ROOT"
```

Early mode checks the requested device and existing writable workspace root. It
has **no install prerequisite** and ignores inherited `INSTALL`; do not pass
`--install`. `$INSTALL` may still name a future nonexistent directory. Exit 0 is
early feasibility only, not plugin loading, installed paths or correctness.
Exit 1 is a device/path/write failure; exit 2 is invalid invocation. A scheduler
listing or submission estimate is useful planning evidence, not this gate.

For rocKE, locate every actual builder and its annotated spec; establish the
`(spec, *, arch)` packaging interface and required fields. An unknown architecture
inventory requires source/support investigation, not a guessed arch. Recheck
source-sensitive feasibility after a base/library change.

**Gate:** feasible target/workspace and selected reference, with each missing
prerequisite reported explicitly. Installation is not among these prerequisites.

## 2. Contracts, corpus and baseline approval

Record [graph-contract.md](graph-contract.md)'s node/topology/UID edges, schema-field
dispositions, frontend semantics and actual corpus differences. Then extract
[rocke-mining.md](rocke-mining.md)'s applicability, layout, effective-specialization
bindings, geometry/workspace and fixed/conditional ABI facts. Direct-load paths
extract these from HIP without a fictitious rocKE profile. Extensions explicitly
reuse unchanged contracts and reopen any accepted topology/feature they alter.

Inventory the owners' published results, their benchmark shapes and external
workload graphs with original source identities. Request shapes and graph bundles
are different inputs: `$SHAPES` is a JSON list of semantic requests;
`$CORPUS_DIR/<corpus>` holds the actual graph JSON consumed by benchmarking.

For the supported attention miner, for example:

```bash
"$PY" "$GEN/tools/mine_shapes.py" \
  --published /absolute/path/to/owner-results.csv \
  --graphs "$CORPUS_DIR/servable" --arch "$ARCH" \
  --include-windowed --out "$SHAPES"
```

Use `--rocke-bench <actual-benchmark-tree>` when that is a source, and omit only
sources explicitly absent from the approved scope. Do not drop window/sink or
independent V-dimension semantics to fit a request schema. Reconcile every source
count and exclusion; a path contributing no expected graphs is not evidence of
an empty population. See [workloads.md](workloads.md).

Resolve a provisional rocKE baseline through the actual dispatcher:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" \
  --shapes "$SHAPES" --out "$CONFIG" --report-knobs --report-gaps
"$PY" "$GEN/tools/reconcile_applicability.py" \
  --profile "$PROFILE" --shapes "$SHAPES"
```

The second command is **offline applicability comparison**, not runtime coverage
or numerics. Scope reference candidates to the actual kernel family/algorithm and
supply any required opt-in selector. Missing/broken/noncallable APIs, bad binding,
exceptions and invalid predicate results fail with exit 2; escape flags never
waive them. A validated false predicate is a decline, not an operational error.

Present the graph/feature boundary, covered/servable/total counts by source,
exclusions, architecture, engine identity, knobs and provisional baseline to the
user. Resolve uncertain semantics and obtain confirmation before committing to the
set. A legal cross-product is not a measured shipping choice; tuning follows a
runnable baseline. A genuinely single-candidate engine is valid when its contract
requires no selection—do not add weightless variants to satisfy an arbitrary count.

**Gate:** approved boundary and baseline, complete dispositions and no unresolved
semantic loss or unsupported reference assumption.

## 3. Generate, implement and splice the baseline

Generate into an empty scratch destination, not a live engine directory:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED"
"$PY" "$GEN/tools/verify_variant_sets.py" --mode structural \
  baseline "$GENERATED/descriptors"
```

Inspect the finalized emitted inventory, not counts from pre-dedup YAML. Resolve
KDP → UED → KMD by UUID; metadata identity uses the referenced schema's types and
defaults and overlapping effective architecture. Preserve legal disjoint-arch
variants. At this stage the result is **prebuild structural evidence only** —
nesting, runtime-tuple identity, sentinels and vocabulary. `--mode` is required and
has no default: `--mode structural` runs only the checks that need no compiled
evidence and reports `COMPILED SPECIALIZATION AGREEMENT` as NOT CHECKED by name, so
a structural pass can never be read as compiled agreement. A structural run still
passes with its NOT CHECKED list; a full run does not — there, an unrun check is a
failure. `--profile` is optional here
and carries only the two facts the artifacts do not — which bundle to gate when a
tree hosts more than one engine, and the matcher's vocabulary where no declaration
states it; with no vocabulary declared anywhere the vocabulary check reports NOT
CHECKED rather than passing. A profile can never supply compiler evidence.
Artifact-bound compiler agreement is a separate, stronger check performed in
stage 4 against the packed producing-build record.

For extend, apply [extend.md](extend.md)'s identity remapping and addition-only
splicing. Preserve old IDs/hooks and all unrelated entries, including shared KMDs.
Keep specialization-contract consumer IDs consistent with the final authored tree.
Do not mutate packed evidence to match new descriptors: rebuild from authored input.

Implement [native-pack.md](native-pack.md)'s applicable hooks and tests in their
actual destination files. Graph criteria are for genuine pack narrowing; score and
UHD exist only for a configured heuristic. Do not copy a single-node matcher into
a fusion, or a conditional argument list into a fixed ABI.

Apply the emitted fragments to the actual CMake/registration consumers:

| Splice | Direct-load | Packaged |
|---|---|---|
| `HIPDNN_DESCRIPTOR_FILES` | Install authored runtime descriptors | Do not add unlowered authored descriptors |
| `HIPDNN_INGESTOR_PACK_KERNELS` | Embed actual source files | Not applicable |
| Engine `target_sources` | Native implementation | Native implementation |
| `IngestorPacks.hpp` declaration and `.cpp` table | Both | Both |
| Engine test `target_sources` | Applicable tests/census | Applicable tests/census |

Embedding references use source-file stems; entry-point spelling may differ.
Declaration without the registration table row can vanish from static-archive
consumers. A generated fragment is a proposed edit, not an applied splice.

Stage packaged authored descriptors under the configured
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, preserving the generated relative
subpath. Discover that root in the actual build configuration; do not assume an
example directory or broaden a fixture test merely to hide wrong staging. Direct
load retains its own descriptor tree. Every regeneration updates the authored
staging and compares identities/content, not just equal counts.

Run field auditing once per schema, across all applicable native sources:

```bash
"$PY" "$GEN/tools/field_audit.py" \
  "$REPO/projects/hipdnn/flatbuffers_sdk/schemas/sdpa_attributes.fbs" \
  "$PROVIDER/src/engines/kernel_ingestor_engine/packs/Gfx942AttentionDenseNative.cpp"
```

Use the actual schema/source paths for other operations. Exit 0 covers accessor
references only; review semantics separately. Check placeholders across the exact
selected engine native **and** test paths after splicing. Files are located only at
the engine-specific relative paths this generator's own CMake fragments splice to,
so an unrelated same-basename file elsewhere neither satisfies a missing target nor
manufactures a false ambiguity. For an unspliced generation tree the output
directory is the single searched root:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --check-placeholders
```

After splicing, name one `--emitted-root` per tree the pieces landed in. The check
covers every shippable file the config emits — descriptors as well as the native
pack and the generated test stubs — and the provider splits them: packs land in the
engine directory, the test stubs under `src/tests/engines/.../packs/`, and packaged
authored descriptors under the configured `HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`
staging (direct load keeps its own under the engine's `descriptors/`). Those trees
share no ancestor worth scanning. `--output-dir` is still required by the parser but
is not searched once `--emitted-root` is given:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" \
  --check-placeholders \
  --emitted-root "$PROVIDER/src/engines/kernel_ingestor_engine" \
  --emitted-root "$PROVIDER/src/tests/engines/kernel_ingestor_engine" \
  --emitted-root "$DESCRIPTOR_STAGING_ROOT"
```

Set `DESCRIPTOR_STAGING_ROOT` to the tree holding this bundle's staged descriptors
at their generated relative subpath. Anything the roots do not cover is reported by
relative path, so an incomplete root list fails loudly instead of passing on the
files it happened to see.

A root that does not exist is an error, never an empty search. One relative path
resolving under two roots is an error rather than a pick. A shippable file no root
holds is an unfinished splice and fails, even when every file that was read is clean.

When the engine declares a launch-surface audit, check that existing declaration:

```bash
"$PY" "$GEN/tools/launch_surface.py" "$PROFILE" --check
```

Its structural result is not semantic equivalence of the Python and C++ launchers.

**Gate:** finalized authored inventory, real applicable hooks, completed source/test
splices, no selected-path placeholders, structural checks and reviewed field/ABI
coverage. Full compiler/native/device proof comes later.

## 4. Build, pack, install and prove the host boundary

Use `hipdnn-superbuild` from the repository root with the selected preset, build
path and current dependency setup. `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` enables the
capability; SDPA also needs `HIPDNN_ENABLE_SDPA=ON` consistently in SDK and provider.
Producer switches are different: `HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE` and
`..._HIP` control authored source lowering, with their required source root,
compiler/wheel and kpack dependencies. `HIPKERNELPROVIDER_ENABLE_ROCKE` controls
rocKE engine/dependency readiness and is not a replacement for the production
producer switch. Disabling an incumbent engine is not exact-engine attribution.
Read the current CMake guards and [packaging reference](../../../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).

For packaged output, build the production packaging target as well as provider,
validator and applicable test targets; a plugin build alone is not proof that
current descriptors were packed. Use the existing superbuild mechanism, not a new
runner. Keep full output in logs and let Ninja choose parallelism unless explicitly
instructed otherwise. Install before any installed-tree probe:

```bash
cmake --install "$BUILD" --prefix "$INSTALL"
```

The producing compiler observes declared fields/accessors on the actual hydrated
spec passed to the builder, retaining authored `provenance.spec` separately from
compiler-owned `provenance.effective_spec`. The embedded declaration and evidence
bind every consumer to its current schema, metadata, arch and payload. No packaging
`--profile`, CMake `PROFILES` or external root manifest is required or supported by
this workflow. Generation cannot issue that evidence; a later verifier cannot
substitute today's imported producer for the recorded producing build.

Check the complete final descriptor tree for each selected arch. A representative
full artifact invocation is:

```bash
"$PY" "$GEN/tools/verify_variant_sets.py" --mode full --arch "$ARCH" \
  --profile "$PROFILE" final "$FINAL_DESCRIPTOR_ROOT"
```

Set `FINAL_DESCRIPTOR_ROOT` to the actual installed shard/tree discovered after
installation. `--mode full` is the claim being made here: it binds the descriptors
to the producing compiler's own recorded evidence and to the payload bytes they
name, so a missing, unsupported or mismatched record fails. Under `--mode full` a
check that could not run is also a failure — `GATE FAILED (N check(s) NOT RUN: …)`
— because the full claim covers vocabulary and compiled agreement as well as the
structural properties. Pass a `--profile` whose vocabulary actually covers the
string fields no declaration spells out, or ask for `--mode structural` and take
the narrower claim in writing. `--arch` is required whenever the tree does not pin
exactly one architecture. Add
`--kpack-python-dir <dir>` when the `rocm_kpack` package needed to read those bytes
is not the installed one. For direct-load examples, explicitly report their
structural/native/device proof boundary rather than inventing a rocKE
compiled-specialization claim.

Run standalone structural descriptor validation against the runtime dialect:
authored direct-load descriptors or **packed** per-arch descriptors, never unlowered
rocKE authoring input. Build the validator first if absent; a missing binary can
mean unbuilt targets, disabled capability or the wrong build/install path.

Run the compiled provider host registration/loading and finalized emitted-bundle
census. The existing path executes real `SymbolScope<Handle>` registrations and
`discoverDescriptorSets()` / `loadValidatedDescriptorSets<Handle>()`; source-text
inspection does not certify native hooks. Use fresh processes and explicit
`HIPDNN_TEST_EXPECTED_ARCH` from the configured packaging arch list, not from the
loaded descriptors or GPU detection. Each selected shard needs a nonempty exact
host-test selection and expected finalized identities/counts, source kind, SDK and
arch. Missing registrations or bundles fail; a heuristic-disabled engine also
requires absence of score registration. Host loading **does not prove dispatch**.

**Gate:** build/install succeeded, final artifact checks passed at their declared
strength, real native registrations/load/census passed, and exact install/artifact
identities were retained. Any required unsupported check is a blocker, not a green
structural substitute.

## 5. Baseline device proof from the installation

On the allocated target machine:

```bash
"$PY" "$GEN/tools/device_probe.py" --mode installed --arch "$ARCH" \
  --sweep-root "$SWEEP_ROOT" --install "$INSTALL"
```

This requires the explicit existing installation. Missing or invisible install
fails here even if early feasibility passed. A successful probe still does not
prove engine loading or numerical correctness.

Use `hipdnn-superbuild-test` discovery; the component key is **`hip-kernel`**, not
the provider's project-name spelling:

```bash
"$PY" "$REPO/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts/discover_test_targets.py" \
  --build-dir "$BUILD" --component hip-kernel --scope external-integration
```

For gfx942 dense create, do not accept the helper's first provider-prefixed command
as the dense gate. Inspect the actual installed generated CTest registration:

```bash
ctest --test-dir "$INSTALL" -N -V \
  -R '^hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests$'
```

Require exactly that registration and inspect its command/config for the exact
`hipkernel:Gfx942AttentionDense` engine pin, installed executable/plugin/config
paths and intended quick/standard selection. Then execute it:

```bash
ctest --test-dir "$INSTALL" --no-tests=error -V \
  -R '^hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests$'
```

Run from the configured installed CTest root if the install layout places it in a
subdirectory; resolve that root explicitly rather than substituting a broad build
target. Missing registration, wrong pin/path, zero selected cases, failed numerical
comparison or all-skipped support fails this gate. If running quick and standard
filters separately through the existing runner, preserve the same resolved
article/engine/config and retain each full case denominator.

Quick tier covers feature breadth with small real nontrivial inputs; standard tier
adds affordable numerical depth. Use actual reference capability and runtime cost
when choosing cases. Test required negative applicability paths separately so a
different winning engine cannot hide a rejection. Do not replace meaningful inputs
with zeros: zero output can agree even when nothing was written. NaN/unwritten-output
evidence is a failure, not a tolerance tweak.

Record exact expected/observed engine ID/name, selected/served/skipped/failed cases
and reasons. A UED name maps to the engine's identity; prove that mapping from the
installed engine, not a name-prefix assumption. A registry listing adds no claim
that a graph was matched or dispatched.

For the concrete pointwise extend walkthrough, retain the old installed baseline,
add only HALF/block_size=256 to ADD through stage 3's scratch splice, preserve
MUL/SUB, and update the copied ADD census from three to four. Run an explicit-knob
frontend smoke selecting the new HALF/256 candidate and checking the actual plan's
`hipkernel:Pointwise` engine ID and nontrivial arithmetic result. Use logical dims
`{1,1,1,1}`: the source is single-element. Also exercise existing ADD/MUL/SUB and
required multi-element/two-node declines. A default-FLOAT pass does not exercise
this addition; this is not arbitrary pointwise tensor coverage.

**Gate:** actual intended-engine dispatch and successful reference comparison on
`$ARCH`, with complete case accounting and expected negative behavior.

## 6. Tune only the runnable baseline, then rebuild the final selection

For the real gfx942 create walkthrough, use the stacked dense integration's real
kernel/native implementation and approved corpus. Generated placeholders or a fake
new engine cannot stand in for this path. Keep child engine changes separate from
the tooling change.

Use existing authoring tools to propose bounded candidates:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" --plan
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --isolate --out-dir "$ARM_CONFIG_ROOT"
```

Set `ARM_CONFIG_ROOT` to an experiment-owned output directory. Each supported arm
must be generated, implemented/spliced as applicable, built, packed, installed and
checked before measurement. Use separate install/output trees; do not mutate the
baseline installation in place and call it a comparison.

Populate [the declarative sweep YAML](../../../IngestorGenerator/tools/README-sweeps.md)
from actual corpus counts, total installed KDP entries and proven engine identity.
The committed `configs/sweep-isolation.sweep.yaml.example` is an example, not a
measurement inventory. Replace historical counts and roots. YAML paths resolve
relative to the YAML file, not cwd; there is no shell/environment interpolation.

```bash
"$PY" "$GEN/tools/sweep.py" --config "$SWEEP_CONFIG"
```

Set `SWEEP_CONFIG` to an absolute YAML path. Use one target device/node/session/job,
baseline-first fixed arm order, a gated discarded warmup and at least three rounds
for drift-reporting comparisons. Correctness runs separately after timing, once
per corpus/arm with a capable declared reference. Check exact-engine results;
there is no benchmark `--engine` selection recipe here.

Investigate individually measured survivors, then generate supported pairwise
arms with `knob_sweep.py --pairwise <knob>,<knob> --out-dir <path>`. Build/install
those arms before another sweep. Obtain explicit selection approval from measured
coverage, correctness and per-corpus performance, not just a flattering aggregate.
Generate a shipping set from the actual supported choices:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --knobs "$APPROVED_KNOBS_JSON" --out "$FINAL_CONFIG"
```

The variables name the approved JSON object of knob value lists and final config
path. Repeat **stages 3–5** with that final config: generation, final authored
staging, artifact agreement, real native/census checks, build/install, installed
probe and exact-engine numerics. Reordering work to claim a measured shipping
selection before a baseline runs is invalid. Neither the winning isolation arm
nor an old install certifies the regenerated shipping cross.

**Gate:** justified selection and revalidated final installation. An untuned
extension may explicitly retain its approved baseline; it still takes the final
artifact/corpus branch below.

## 7. Final corpus proof and runtime reconciliation

Run a fresh final-artifact YAML sweep in a new output directory with
`correctness.enabled: true`. Diagnostic resume can reuse current-input-valid phase
evidence, but cross-session results do not establish a single-job comparative
cohort. Timing-only `SWEEP_TIMING_ONLY` is not final success. Require the exact phase
key set, all warmup/timed/correctness gates and validated `SWEEP_DONE`; plausible
JSON or a previous completion sidecar is insufficient.

Then explicitly harvest final phase result files and available engine logs into a
complete outcome ledger for **every input**. This is a required manual verification
step, not an automatic matcher-reason extractor. Retain:

- Canonical semantic request identity and every original corpus/source/graph occurrence.
- Phase/input fingerprint, exact expected engine, observed engine/outcome and evidence path.
- Separate served, explicitly declined, execution-error, missing and ambiguous categories.

Absence of timing is not a decline reason. A reason unavailable in runtime evidence
remains unavailable; do not reconstruct it from offline policy or source. Join
within each corpus/phase and reject missing outcomes, duplicate/ambiguous names or
fingerprint mismatches before making the graph-name-to-reason JSON. Preserve source
occurrences even when mining merged semantic requests. Missing/ambiguous/error
outcomes block runtime coverage acceptance.

For rocKE, reconcile each corpus with **its own complete join**:

```bash
"$PY" "$GEN/tools/reconcile_applicability.py" --profile "$PROFILE" \
  --shapes "$CORPUS_SHAPES" --declines "$RUNTIME_DECLINES"
```

Set both paths to this corpus's request list and runtime-derived decline mapping.
Without the complete join, report this as offline only even if a sparse decline
file could be parsed. Reference-only supported rows require investigation of
missing variants, matcher semantics or reference numerics and the explicit scope
decision gate; never erase them with an unexplained exclusion or escape flag.
Predicate applicability does not establish numerical truth.

Direct-load pointwise uses a small explicit one-element corpus, actual installed
engine results and independent arithmetic/reference correctness, with
`exclude_tensors: none`; it does not invent a rocKE profile or oracle.

Report full corpus denominators, served/reference-validated populations and every
remaining outcome separately from timing. Report geomean-of-ratios and time-weighted
sum-baseline/sum-arm together by source, drift by round, and byte-identical controls
chosen from artifact hashes rather than timings. A minimum served floor is a
measurement gate, not permission to omit other corpus outcomes.

**Gate:** zero wrong answers, complete final-runtime accounting, no missing/ambiguous/
error or unexplained in-scope outcomes, and justified coverage against the approved
scope. Regeneration or changed installed artifacts returns to stages 3–5 and
invalidates any final result bound to the old inputs.

## 8. Handoff and documented walkthrough boundaries

Report per SKILL's completion contract: last completed stage, final source/config/
descriptor/payload/plugin identities, installed tree, all actual commands and
results, engine-attributed case/corpus fractions, exclusions and limitations.
No claim extends beyond the observed device, architecture, reference and graphs.
Keep experiment copies and one-off probes disposable; retain their inputs and
command evidence under the workspace's results policy.

The early documented-command traversal (**C0**) is intentionally smaller than the
full walkthrough (**C1**). After tooling writes settle, on the actual target host,
set an inherited `INSTALL` to a future nonexistent path and run stage 1's early
probe: it must pass when device/root are available. Run stage 5's installed probe
against that same missing path: it must fail. Using the completed generator checkout,
run stage 3's generation and structural verification commands with `configs/scale_add.yaml`
into a clean scratch destination. Resolve tool roots explicitly if the premerge
checkout differs. C0 proves only those early commands, not native/build/device
completion, and is not blocked on compiled-host integration.

C1 follows **the full runbook after native/census integration**: clean gfx942 dense
create using the real stacked implementation, bounded tuning, final regeneration
and corpus join; then the existing-engine HALF/256 one-element pointwise extension.
Require the exact dense CTest target/pin from stage 5, not a generic component PASS.
Exercise failure handling for missing installed tree, unavailable reference, absent
real registration and failed correctness/resume phases; none may emit completion.
A disposable negative join probe must reject omitted and same-named mismatched
outcomes. A prose/source scan is not execution evidence for either walkthrough.
