# Runbook: create or extend an ingestor engine

This is the **only ordered create/extend workflow**. Use [SKILL.md](SKILL.md) for
entry/completion requirements and the linked domain pages for contracts. Each gate
requires current evidence; an old install or copied command is not an observation.

## Paths and interpreters

Resolve absolute paths before executing commands. These are explicit arguments,
not implicit tool inputs:

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

Use [generator setup](../../../IngestorGenerator/README.md#setup). Authoring/mining
imports need the profile's rocKE library environment; production packaging uses
its selected compiler/wheel interpreter instead. Full artifact checking also needs
`rocm_kpack` and its dependencies in `$PY` (including `zstandard` and `msgpack`).
`--kpack-python-dir` supplies an import root, not missing Python dependencies.

Verify source, build, install, corpus and output paths on the execution host. A
workspace symlink backed by login-local **`/var/tmp` is invisible from compute
nodes**, even when its link name is under shared storage. Use shared source or stage
the exact checkout/artifacts to compute-local scratch through the active scheduler
skill. Do not run GPU work on a login host. Retain command logs and source/artifact,
machine, device and allocation identities under the workspace's evidence policy.

## 1. Entry and early feasibility

Record the entry contract; for extend, inventory the installed baseline according
to [extend.md](extend.md). Select `direct_load` or `packaged` before staging files.
Establish that the graph is representable and has a capable independent numerical
reference using [graph-contract.md](graph-contract.md). A reference's skip is not
verification; unavailable semantics require an explicit scope/reference decision.

On the actual allocated execution host:

```bash
"$PY" "$GEN/tools/device_probe.py" \
  --mode early --arch "$ARCH" --sweep-root "$SWEEP_ROOT"
```

Early mode requires the requested device and an existing writable root. It ignores
inherited `INSTALL` and rejects `--install`; `$INSTALL` may name a future directory.
Exit 0 proves feasibility only, exit 1 a device/path/write failure, and exit 2 an
invalid invocation. For rocKE, confirm the actual builder/spec and `(spec, *, arch)`
interface; an unknown architecture inventory needs source investigation.

**Gate:** feasible target/workspace, representable scope and capable reference. A
missing dependency blocks its gate; host-only research may continue while a device
allocation is pending, but cannot discharge device proof.

## 2. Contracts, corpus and baseline approval

Record [graph-contract.md](graph-contract.md)'s topology/UID edges and field
dispositions, then [rocke-mining.md](rocke-mining.md)'s applicability, specialization,
layout, geometry/workspace and ABI evidence. Direct-load engines obtain these facts
from HIP source without inventing a profile. Reuse unchanged extension contracts;
reopen any topology or feature the addition changes.

Inventory owner-published results, owner benchmark shapes and external graphs per
[workloads.md](workloads.md). `$SHAPES` is a JSON list of semantic requests; the
benchmark consumes actual graph JSON under `$CORPUS_DIR/<corpus>`. For the attention
miner, using the sources in scope:

```bash
"$PY" "$GEN/tools/mine_shapes.py" \
  --published /absolute/path/to/owner-results.csv \
  --graphs "$CORPUS_DIR/servable" --arch "$ARCH" \
  --include-windowed --out "$SHAPES"
```

Add `--rocke-bench <actual-benchmark-tree>` when applicable. Reconcile each source's
total, parsed, servable, covered and excluded counts; do not discard window/sink or
independent operand dimensions to fit the request schema.

For rocKE, resolve the provisional baseline through its actual dispatcher:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" \
  --shapes "$SHAPES" --out "$CONFIG" --report-knobs --report-gaps
"$PY" "$GEN/tools/reconcile_applicability.py" \
  --profile "$PROFILE" --shapes "$SHAPES"
```

The second command is **offline applicability**, not runtime coverage or numerics.
Scope reference candidates to the kernel family/algorithm and required opt-in
selector. API failures are operational errors, never unsupported-shape evidence.

Present the feature/shape boundary, per-source coverage and exclusions, architecture,
engine identity, knobs and provisional baseline for approval. A legal cross-product
is not a measured shipping set; a genuinely single-candidate engine needs no extra
variants.

**Gate:** approved baseline and scope, with no unresolved semantic loss or reference
assumption.

## 3. Generate, implement and splice

Generate into an empty scratch directory, never over the live engine:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --dry-run
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED"
"$PY" "$GEN/tools/verify_variant_sets.py" --mode structural \
  baseline "$GENERATED/descriptors"
```

Review the finalized inventory after deduplication. Resolve KDP → UED → KMD by UUID;
tuple identity includes schema types/defaults and effective architecture overlap.
Structural mode reports compiled agreement as `NOT CHECKED` and can exit 0 with
unrun checks. `--profile` optionally selects the bundle and supplies vocabulary;
it cannot supply compiler evidence. Full artifact checking belongs after packing.

Implement [native-pack.md](native-pack.md)'s referenced hooks and behavioral tests.
Extensions remap scratch references and copy only additions per [extend.md](extend.md),
including consumer IDs in shared KDP or per-UKD specialization declarations. Never
edit packed evidence to match a changed descriptor; rebuild from authored inputs.

Apply fragments to their actual consumers, preserving unrelated entries:

| Splice | Direct-load | Packaged |
|---|---|---|
| `HIPDNN_DESCRIPTOR_FILES` | Authored runtime descriptors | Never unlowered authored descriptors |
| `HIPDNN_INGESTOR_PACK_KERNELS` | Actual source-file stems | Not applicable |
| Engine `target_sources` | Native implementation | Native implementation |
| `IngestorPacks.hpp` declaration and `.cpp` table | Both | Both |
| Engine test `target_sources` | Applicable tests | Applicable tests/census |
| `HKP_CENSUS_TEST_SUITES` in `hkp_register_census_tests()` | Not the direct-load inventory gate | Literal generated packaged census suite name |

The census list lives in
`dnn-providers/hip-kernel-provider/descriptor-packaging/cmake/HkpPackaging.cmake`.
Appending the literal `Test<Name>Packs` suite is what creates the census obligation;
an empty list registers none.

Stage packaged authored descriptors under the configured
`HIPKERNELPROVIDER_PRODUCTION_SOURCE_ROOT`, preserving their generated relative
subpath. Update that staging after every regeneration and compare content/identities,
not only counts. A fragment file is not evidence that its splice was applied.

Set `SCHEMA` to the actual operation `.fbs` and `NATIVE_SOURCE` to its implementation;
pass all relevant source files and repeat for each schema in a fusion:

```bash
"$PY" "$GEN/tools/field_audit.py" "$SCHEMA" "$NATIVE_SOURCE"
```

Exit 0 covers lexical accessor references only; review semantic dispositions
separately. For an unspliced tree, check placeholders with:

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" \
  --output-dir "$GENERATED" --check-placeholders
```

For a packaged splice, set `DESCRIPTOR_STAGING_ROOT` to the production tree holding
the generated descriptor subpath, then check all destinations below. For direct-load,
omit the third root: its descriptors already live under the engine root.

```bash
"$PY" "$GEN/generate.py" --config "$CONFIG" --output-dir "$GENERATED" \
  --check-placeholders \
  --emitted-root "$PROVIDER/src/engines/kernel_ingestor_engine" \
  --emitted-root "$PROVIDER/src/tests/engines/kernel_ingestor_engine" \
  --emitted-root "$DESCRIPTOR_STAGING_ROOT"
```

With `--emitted-root`, `--output-dir` is required but not searched. The check covers
all emitted shippable files at engine-specific paths: nonexistent roots, missing
files and ambiguous matches fail. An unrelated same-basename file cannot satisfy it.
If the profile declares a launch-surface audit, also run:

```bash
"$PY" "$GEN/tools/launch_surface.py" "$PROFILE" --check
```

**Gate:** final authored inventory, completed hooks and source/test splices, no
selected-path placeholders, and reviewed structural/field/ABI results. None proves
native loading or numerical dispatch.

## 4. Build, pack, install and prove the host boundary

Configure from `$REPO` using `hipdnn-superbuild`, with
`CMAKE_INSTALL_PREFIX="$INSTALL"`, `HIPDNN_ENABLE_KERNEL_INGESTOR=ON` and
`HIPKERNELPROVIDER_ENABLE_TESTS=ON`. SDPA needs `HIPDNN_ENABLE_SDPA=ON` consistently
in SDK and provider. Packaged producers additionally need
`HIPKERNELPROVIDER_PRODUCTION_ENABLE_ROCKE` and/or `..._HIP`, the authored source
root, and the [packaging dependencies](../../../../../../dnn-providers/hip-kernel-provider/descriptor-packaging/README.md).
`HIPKERNELPROVIDER_ENABLE_ROCKE` does not replace the production producer switch.

Build the provider, validator and required test targets through the configured
superbuild. For packaged engines, run `hkp_packaging_product` after the full build
and after any reconfigure. Require the final staged descriptors, not merely an
up-to-date packaging stamp or an intermediate archive. Then install:

```bash
cmake --install "$BUILD" --prefix "$INSTALL"
```

**Install before provider-wide integration runs.** ASM SDPA loads loose `.co`
kernels from the configured install prefix by default; building its executable or
copying kernels into a build tree does not satisfy that path. Its explicit runtime
override is a separate test setup, not evidence of the final installation. Keep the
configured prefix aligned with `$INSTALL`, rather than relocating only this command.

Set `FINAL_DESCRIPTOR_ROOT` to the actual installed per-arch shard (or the
arch-independent direct-load tree). Resolve `VALIDATOR` to the built
`hipdnn_validate_descriptors` executable and validate the runtime dialect:

```bash
"$VALIDATOR" "$FINAL_DESCRIPTOR_ROOT" --expect-engine "$ENGINE" --json
```

For packaged output, check each selected architecture using the producing-build
record, not today's imported producer:

```bash
"$PY" "$GEN/tools/verify_variant_sets.py" --mode full --arch "$ARCH" \
  --profile "$PROFILE" final "$FINAL_DESCRIPTOR_ROOT"
```

Omit `--profile` when neither bundle selection nor extra vocabulary needs it. Add
`--kpack-python-dir <dir>` if required by the reader environment. Interpret outcomes
separately:

- Use full checking only for **packed `kind: kpack` descriptors**; confirm the input
  dialect instead of inferring it from exit 0. Missing declarations, mismatches
  and required checks `NOT RUN` block acceptance, including missing vocabulary.
- A packed kernel declaring no specialized `metadata_fields` and carrying no
  `effective_spec` reports **`NOT VERIFIED HERE`**. It neither fails the gate nor
  gains compiled-specialization proof. Full-mode exit 0 does not certify those
  binaries; AOT HIP specialization remains outside this check.
- For declared specialization, the per-kernel record binds the current descriptor,
  schema, metadata, architecture and named payload bytes. Generation supplies the
  declaration, not that compiler-owned evidence; see [rocke-mining.md](rocke-mining.md).

### Packaged census: direct native CTest entries

Run real provider registration/loading and inventory checks in fresh processes.
The census **covers packaged engines only**, and it is a direct native obligation
with no Python launcher and no XML guard. For each literal suite in
`HKP_CENSUS_TEST_SUITES` and each configured packaging architecture, CMake registers
an independent test `hip-kernel-provider-hkp-census-<arch>-<suite>` that invokes

```text
hip_kernel_provider_tests --gtest_filter=<suite>.*
```

with `HIPDNN_TEST_CENSUS_SUITE=<suite>`, `HIPDNN_TEST_EXPECTED_ARCH=<arch>` and
`HIPDNN_DESCRIPTOR_DIR=<descriptor-build-dir>/<arch>`. The architecture comes from
the configured packaging list, never from a detected device or from the descriptors
themselves. Set `CENSUS_SUITE` to the generated suite name and `PROVIDER_BUILD` to
the provider's own binary directory (`$BUILD/dnn-providers/hip-kernel-provider` in
the superbuild layout), then run every requested arch's entry:

```bash
ctest --test-dir "$PROVIDER_BUILD" --no-tests=error -V \
  -R "^hip-kernel-provider-hkp-census-${ARCH}-${CENSUS_SUITE}$"
```

Those entries bind the build-tree shard. For **final installed packaged evidence**,
run the same suite against the installed shard by supplying the same explicit
environment to the installed binary:

```bash
HIPDNN_TEST_CENSUS_SUITE="$CENSUS_SUITE" \
HIPDNN_TEST_EXPECTED_ARCH="$ARCH" \
HIPDNN_DESCRIPTOR_DIR="$FINAL_DESCRIPTOR_ROOT" \
"$INSTALL/bin/hip_kernel_provider_tests" --gtest_filter="${CENSUS_SUITE}.*"
```

Adjust the binary path for a nondefault install bindir. A nonempty
`HIPDNN_TEST_CENSUS_SUITE` activates the native strict guard: before default-root
setup it rejects an empty expected arch and a missing, empty or nonexistent explicit
descriptor root, and it rejects an absent or empty named suite. Every registered
case in that suite must execute and pass **without skipping in every iteration**,
with at least one completed iteration. Disabled, filtered-out, sharded-out, failed
or skipped cases, list-only invocations and zero iterations cannot satisfy it, and
repeated partial runs do not accumulate coverage. Normal invocations without the
variable keep ordinary GoogleTest filtering and skip behavior, and the production
runtime's descriptor-root fallback is unchanged.

Declared suites with a missing test target or an empty configured architecture list
are configuration errors. Tests built OFF, or no declared suite, yields **no census
evidence**: that is absence, not a pass. Direct-load engines use their ordinary
unit/inventory suites against their arch-independent descriptor tree and are not
covered by a packaged-census pass; a generated inventory suite invoked directly
still requires an explicit expected arch and descriptor root. Check retained
extension inventory and heuristic-disabled score absence per
[native-pack.md](native-pack.md).

**Gate:** current installation, artifact checks at their stated strength, and real
registration/loading plus applicable inventory/census checks. Report `NOT VERIFIED
HERE` separately. Neither the structural validator's stubs nor host loading proves
dispatch.

## 5. Baseline device proof from the installation

On the allocated target host:

```bash
"$PY" "$GEN/tools/device_probe.py" --mode installed --arch "$ARCH" \
  --sweep-root "$SWEEP_ROOT" --install "$INSTALL"
```

A missing/invisible installation fails even if early feasibility passed. Use
`hipdnn-superbuild-test` discovery with component **`hip-kernel`**:

```bash
"$PY" "$REPO/projects/hipdnn/tools/ai/skills/hipdnn-superbuild-test/scripts/discover_test_targets.py" \
  --build-dir "$BUILD" --component hip-kernel --scope external-integration
```

Do not accept the helper's first provider-prefixed command as exact-engine proof.
The provider's default installed CTest root is **`$INSTALL/bin/hip_kernel_provider`**,
not `$INSTALL`; substitute the configured bindir if customized. For gfx942 dense:

```bash
CTEST_ROOT="$INSTALL/bin/hip_kernel_provider"
DEVICE_TEST=hip_kernel_provider_gfx942_attention_dense_gpu_ref_integration_tests
ctest --test-dir "$CTEST_ROOT" -N -V -R "^${DEVICE_TEST}$"
```

Require exactly that registration. Inspect its command/config for
`hipkernel:Gfx942AttentionDense`, installed executable/plugin/config paths and the
intended quick/standard selection. Then execute:

```bash
ctest --test-dir "$CTEST_ROOT" --no-tests=error -V -R "^${DEVICE_TEST}$"
```

Keep verbose output in the retained log. **`--output-on-failure` hides passing
suites' case counts; an all-skip suite can still report CTest PASS.** Record selected,
served, skipped/declined and failed counts and observed reasons. Missing registration,
zero selected, all-skipped support, wrong engine/path or numerical failure blocks
this gate. Resolve the exact UED name → engine ID from the installation; a prefix
or registry listing is not dispatch attribution.

Use nontrivial inputs for quick feature breadth and bounded standard numerical
depth. Exercise required declines separately; another winning engine must not hide
them. NaN/unwritten output is a failure, not a tolerance adjustment.

Extensions must select the addition explicitly. The disposable pointwise example
adds HALF/block_size=256 to ADD, preserves MUL/SUB and changes ADD's expected census
from three to four. Select HALF/256 on logical dims `{1,1,1,1}`, check the actual
`hipkernel:Pointwise` plan and arithmetic, and retain old ADD/MUL/SUB and required
multi-element/two-node declines. Its source computes one element; a default-FLOAT
pass or this one-element smoke proves no arbitrary-size coverage.

**Gate:** intended-engine dispatch, capable-reference numerics and complete case
accounting on `$ARCH`.

## 6. Tune the runnable baseline and rebuild the final selection

For rocKE, propose bounded candidates with the actual profile:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" --plan
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --isolate --out-dir "$ARM_CONFIG_ROOT"
```

Set `ARM_CONFIG_ROOT` to an experiment-owned directory. Generate, implement/splice,
build, pack, install and check every supported arm before measurement. Keep separate
install/output trees; never mutate the baseline and call it a comparison.

Populate `configs/sweep-isolation.sweep.yaml.example` using actual corpus counts,
installed KDP-entry counts and proven engine identity. YAML paths resolve relative
to the YAML file; there is no shell/environment interpolation. Set `SWEEP_CONFIG`
to its absolute path, then run:

```bash
"$PY" "$GEN/tools/sweep.py" --config "$SWEEP_CONFIG"
```

Follow [workloads.md](workloads.md)'s one-session, baseline-first order, gated
warmup, rounds, isolated caches and separate correctness requirements. The sweep
selects the discovered engine ID through the benchmark's `--engine` option; do not
override its phase-owned arguments.

For rocKE, investigate measured survivors. Set `PAIRWISE_KNOBS` to comma-separated
surviving knob names and `PAIRWISE_CONFIG_ROOT` to their config output directory,
then generate, build and install the supported pairwise arms:

```bash
"$PY" "$GEN/tools/knob_sweep.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --pairwise "$PAIRWISE_KNOBS" --out-dir "$PAIRWISE_CONFIG_ROOT"
```

Obtain selection approval from coverage, correctness and per-corpus measurements.
For the rocKE shipping cross, `APPROVED_KNOBS_JSON` contains the approved JSON object
of knob value lists, **not a filename**:

```bash
"$PY" "$GEN/tools/dispatch_parity.py" --profile "$PROFILE" --shapes "$SHAPES" \
  --knobs "$APPROVED_KNOBS_JSON" --out "$FINAL_CONFIG"
```

Repeat **stages 3–5** with the final config and a new empty generation destination.
Neither an isolation arm nor an old install certifies the regenerated shipping set.
An explicitly untuned extension may retain its approved baseline selection, but
still requires final installed artifact and corpus proof.

**Gate:** justified selection and revalidated final installation.

## 7. Final corpus proof and runtime reconciliation

Run the final installed artifact through a fresh-output YAML sweep with
`correctness.enabled: true`. Require the exact phase key set and validated
`SWEEP_DONE`. Resume may reuse content-valid evidence diagnostically; it does not
make separate sessions a single comparative cohort. `SWEEP_TIMING_ONLY` is not
final success.

Harvest final phase results and available engine logs into [workloads.md](workloads.md)'s
complete per-input outcome ledger. Preserve semantic identity, all original
corpus/source/graph occurrences, current phase/input fingerprints, exact engines
and result/log locations. Distinguish served, explicitly declined, execution-error,
missing and ambiguous outcomes. A missing timing row is not a decline reason.

Join within each corpus/phase before making its graph-name-to-reason JSON. Reject
missing outcomes, duplicate/ambiguous names and mismatched fingerprints. Runtime
reasons unavailable in the evidence stay unavailable; do not reconstruct them from
offline policy. This is an explicit evidence review, not a promised automatic
matcher-reason extractor.

For rocKE, set `CORPUS_SHAPES` and `RUNTIME_DECLINES` to that corpus's requests and
complete runtime-derived mapping, then reconcile:

```bash
"$PY" "$GEN/tools/reconcile_applicability.py" --profile "$PROFILE" \
  --shapes "$CORPUS_SHAPES" --declines "$RUNTIME_DECLINES"
```

Without the complete join, label reconciliation **offline only**. Investigate
reference-only supported rows and obtain explicit scope decisions for exclusions;
escape flags cannot excuse broken reference APIs or unexplained gaps. Direct-load
pointwise instead uses its explicit one-element corpus, installed engine results
and independent arithmetic/reference correctness, with `exclude_tensors: none`.

Report full per-source denominators, served/reference-validated populations and
all remaining outcomes separately from timing. Include geomean-of-ratios,
time-weighted sum-baseline/sum-arm, round drift and byte-identical controls chosen
from artifact hashes.

**Gate:** zero wrong answers, complete final-runtime accounting, and no missing,
ambiguous, erroneous or unexplained in-scope outcomes. Changed installed artifacts
invalidate old evidence and return to stages 3–5.

## 8. Handoff

Report [SKILL.md](SKILL.md)'s completion evidence and exact limitations. Keep
experiment copies/probes disposable and retain their inputs/results under the
workspace evidence policy. For blocked work, name the last completed stage and
missing prerequisite; do not substitute a proposed command or queued job for proof.
