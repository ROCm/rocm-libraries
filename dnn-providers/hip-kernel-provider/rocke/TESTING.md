<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# rocKE Testing Strategy

> **Status: living document / work-in-progress.** This is the source of truth for
> *how rocKE is tested and why*. It describes the **target** strategy and is meant
> to read as an on-ramp for a developer or tester new to rocKE. Where the target
> is not yet reached, the gap is named in [§7](#7-current-state-vs-target-the-gap-registry-wip)
> and marked **⚠ WIP** — this doc does not pretend the aspiration is already true.
> When the code and this document disagree, that disagreement is a bug in one of
> them; fix it, don't route around it.

## 0. Purpose, scope, and the other two docs

**This doc defines the strategy**, validation requirements, and CI replay
procedure. The sibling docs provide the test inventory and general local
testing instructions:

| Doc | Owns | You want it when |
|---|---|---|
| **`TESTING.md`** (this file) | strategy, pre-merge validation, CI replay, what proves what | "how is rocKE tested, and why this way?" |
| [**`platform/tests/README.md`**](platform/tests/README.md) | inventory — what test lives where, exact counts, the CTest target names | "which file covers X? what exactly runs?" |
| [**`platform/dsl_docs/development/testing.md`**](platform/dsl_docs/development/testing.md) | how to run & debug tests locally | "how do I run this / why did it fail?" |

**Scope: the whole rocKE engine** — both the `platform/` tree (core engines, IR,
dispatch) and the `library/` tree (the attention kernel surface). Platform vs.
library is an **emerging modular boundary**; the strategy spans both today and
will evolve to reflect that modularity. Out of scope: `ck4inductor` /
`example/ck_tile/dsl` tests, which drive external packages and live in
`composablekernel/`.

## 1. Two questions: kernel quality and platform quality

rocKE is an **(agentic) kernel-authoring platform**: a DSL and engine that authors
lower into GPU kernels. Every test here serves one of **two distinct quality
questions**, and keeping them separate is the whole point of this document —
conflating them is what produces false confidence and bad coverage decisions.

- **Question A — are the kernels good?** The *artifacts rocKE emits*: are they
  **correct** (right numerics on real hardware) and are they **fast**? This is the
  quality of the product. → [§3](#3-kernel-quality-are-the-kernels-correct-and-fast)
- **Question B — is the authoring platform good?** *rocKE itself* — the DSL,
  engine, IR, dispatch, optimization passes: is it a correct, trustworthy platform
  for authoring kernels (increasingly by agents)? This is the quality of the tool.
  → [§4](#4-platform-quality-is-rocke-a-sound-authoring-platform)

**Why this is hard (the oracle problem).** rocKE's output is LLVM IR / assembly,
not a number you can eyeball. So the first question of any test isn't "did it
pass?" — it's *"how do we even know the right answer?"* In the testing literature
that is the *oracle problem*, and a **test oracle** is whatever mechanism decides
pass/fail. rocKE stacks oracles of differing strength ([§2](#2-how-we-answer-them-the-oracle-types)).

**One structural fact spans both questions:** rocKE ships **two independent
implementations of the same engine** — a **Python engine** and a **C++ engine**
that mirror each other layer-for-layer (see
[`dsl_docs/architecture/engines_and_switching.md`](platform/dsl_docs/architecture/engines_and_switching.md)).
Keeping them equivalent during the migration to C++ is a platform-quality concern
all its own ([§4.3](#43-do-the-two-implementations-agree-the-migration-gate)).

> **Vocabulary bridge.** This doc says *engine* for the product and *the two
> engine implementations* (Python / C++) for its mirrored halves — rocKE is **one
> engine with two implementations**, not two engines. The gate scripts and
> [`dual_backend_unification_rfc.md`](platform/dsl_docs/architecture/dual_backend_unification_rfc.md)
> call the same halves the **dual backend**;
> read "backend" there as "engine implementation." Neither is the per-arch **ISA
> backend** (`Gfx950Backend`, `backend_for(...)`) — a third, unrelated use of the
> word in the code.

## 2. How we answer them — the oracle types

Four kinds of test oracle, in decreasing order of strength. The names are the
standard ones from the testing literature; naming them matters because each proves
something *different*, and conflating them is how false confidence creeps in.

| Oracle type | Question it answers | Mechanism in rocKE | Serves |
|---|---|---|---|
| **Reference oracle** | Is the math correct? | Run on GPU, compare to an independent reference (numpy / torch) | Kernel correctness (A) |
| **Regression / characterization** | Did the output change? | sha256 of canonical IR vs. a pinned baseline | Platform output stability (B) |
| **Differential / pseudo-oracle** | Do the two implementations agree? | **Byte-identity** of emitted IR, Python engine vs C++ engine | Platform equivalence (B) |
| **Implicit oracle** | Did it crash / NaN / hang? | Signal death, non-finite output, timeout | Both A and B |

Two notes that matter:

- **The code's bare word "oracle" is the *reference oracle* row.** When
  [`_emit_common.py`](platform/tests/instances/parity/_emit_common.py) or
  [`fuzz_diff.py`](platform/tests/instances/differential/fuzz_diff.py) say "the
  oracle," they mean the trusted reference implementation — one specific type here,
  not the category. This doc uses the fuller taxonomy; the code's usage is a
  special case of it.
- **Agreement is not correctness.** The differential oracle is a *pseudo-oracle*
  (two implementations checking each other because no cheap true oracle exists).
  Two implementations can agree on a wrong answer. Only the reference oracle
  establishes ground truth — and only on real hardware.

---

## 3. Kernel quality — are the kernels correct and fast?

Question A. The quality of the *artifacts* rocKE emits.

### 3.1 Are the kernels correct?

The **reference oracle**, and the *only* place ground-truth correctness is
established: emit → compile → launch on a **HIP device** → compare to an
independent numpy/torch reference. These lanes are the narrowest and slowest part
of the strategy, require a GPU (skipped, not failed, off-device), and are where
the strategy's biggest holes live (see
[§7](#7-current-state-vs-target-the-gap-registry-wip)):

- The numeric reference lanes (e.g.
  [`instances/differential/numeric.py`](platform/tests/instances/differential/numeric.py))
  drive kernels on device and check the result within tolerance. Model-glue lanes
  exercise end-to-end kernel wiring for real model shapes.
- The pinned SDPA reference lane qualifies old rocKE
  against independent NumPy SDPA offline, then executes old and current rocKE
  on gfx942 using a conservative triangle-inequality error budget. Required
  executions fail when hardware or qualification is missing. The first cohort
  is enrolled in installed tests when a qualified bundle is supplied at build time.
  Provider builds enable `ROCKE_INSTALL_TEST_SDPA_REFERENCE` by default. Run
  `dvc pull dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles/sdpa/gfx942.tar.gz.dvc`
  from the repository root before configuring. TheRock already performs this
  DVC download during source preparation. CMake extracts the archive, validates
  every payload hash against
  [`baseline_lock.json`](library/tests/sdpa_reference/architectures/gfx942/baseline_lock.json), and
  installs it under `engines/test_arch_content/rocke/sdpa/<arch>/` relative to
  the test root (`bin/hip_kernel_provider/` for provider builds). TheRock captures
  this in the test component and splits it into the matching architecture artifact,
  while the Python harness and trusted locks remain in the generic test artifact.
  Missing or corrupt data fails configuration. For builds that intentionally
  omit this GPU lane, use `-DROCKE_INSTALL_TEST_SDPA_REFERENCE=OFF`. This also
  ignores explicit or cached bundle-directory overrides; host checks remain installed.
  In rocm-libraries superbuild mode (`ROCM_LIBS_SUPERBUILD=ON`), this option
  defaults to OFF because the superbuild runs build-tree tests rather than the
  installed reference suite. To package the references in a superbuild, fetch
  the bundle and explicitly pass `-DROCKE_INSTALL_TEST_SDPA_REFERENCE=ON`.
  Existing build directories retain their cached option value.
  To install an unpacked local bundle, pass
  `-DROCKE_TEST_SDPA_REFERENCE_INSTALL_SOURCE_gfx942=<qualified-bundle>`.
  To select a bundle for a pytest run without installing it, set the environment
  variable `ROCKE_TEST_SDPA_REFERENCE_BUNDLE_GFX942`. Normal installed CI leaves
  that override unset and finds the packaged bundle automatically. CTest sets
  `ROCKE_TEST_REQUIRE_SDPA_GPU=1` to enforce required execution. Standalone
  platform builds do not enable archive staging by default.
  For source verification, run
  `python library/tests/run_sdpa_reference.py verify --arch gfx942 --bundle <qualified-bundle> --current-root .`
  from the rocKE root with NumPy, HIP, and COMGR; Torch is not required.
  The installed GPU suite reuses separate baseline and current worker processes
  across cases. Each case still executes both versions twice and validates its
  outputs independently; worker failures and timeouts fail the test.
  To publish a replacement, independently qualify it first, update the source
  lock, and use `python library/tests/sdpa_reference/artifact.py pack --bundle <qualified-bundle> --lock library/tests/sdpa_reference/architectures/gfx942/baseline_lock.json --archive library/tests/reference_bundles/sdpa/gfx942.tar.gz`
  from the rocKE root. Then run `dvc add` and a scoped `dvc push` for the
  archive from the repository root before pushing its Git pointer. Git tracks
  the lock, pointer, and ignore entry; compiled kernels stay in DVC; inputs are regenerated.
  Architecture enrollments are listed in
  [`architectures/registry.json`](library/tests/sdpa_reference/architectures/registry.json),
  shared by CMake and Python. Each architecture owns its case list, source-dispatch
  adapter, and baseline lock under `sdpa_reference/architectures/<arch>/`, with a
  separate DVC archive under `reference_bundles/sdpa/<arch>.tar.gz`. Each operation
  owns its directory under `reference_bundles/`, so future convolution references
  can use `reference_bundles/conv/<arch>.tar.gz` and be updated independently.
  Source pytest looks for extracted bundles under `reference_bundles/sdpa/<arch>/`;
  the runtime override can select another directory. Installed bundles
  live under `engines/test_arch_content/rocke/sdpa/<arch>/` relative to the test
  root. The `engines/test_arch_content` spelling is required by TheRock's artifact
  manifest and the hipkernelprovider kpack splitting handler. `arch_content` is
  reserved for runtime content and must not be used for these test references.
  When reference installation is enabled, every architecture in the registry is
  installed; TheRock splits the payloads by target. Only gfx942 is currently enrolled.
  Adding another architecture requires its adapter, independently qualified lock
  and bundle, and a registry entry; adding an empty registry entry is insufficient.
  Qualification accepts `--arch` and freezes the shared worker and architecture
  adapters into the new bundle. Existing bundles keep their original frozen code.

- **Nothing else in this document proves the math is right.** Byte-identity
  ([§4.3](#43-do-the-two-implementations-agree-the-migration-gate)) and golden IR
  ([§4.2](#42-does-the-platforms-output-stay-stable)) are both blind to a
  wrong-but-stable, wrong-but-agreeing kernel.

### Reference input storage

Schema-2 reference bundles contain compiled kernels, the frozen replay runtime,
and qualification metadata, but no input or output tensor files. Tests regenerate
Q, K, and V using the versioned `numpy-pcg64-normal-f32-v1` recipe: a fresh
PCG64 stream seeded with zero, float32 normal samples in Q/K/V order, followed
by the existing fp16 or round-to-nearest-even bf16 encoding. Each tensor's shape,
dtype, and values must match its qualified digest before either GPU worker runs.
A NumPy distribution implementation change that alters those bytes fails the test;
a seed alone is not considered sufficient evidence of reproducibility.

Generated inputs are passed to both isolated workers through a temporary `.npz`
file, which is removed after the comparison, including on worker failure. This
keeps the frozen baseline worker unchanged. Outputs remain temporary as before.
Neither input nor output tensors are included in the DVC archive or installed
reference bundle. Archive validation rejects `.npz` and `.npy` payloads for schema 2.

The original schema-1 gfx942 bundle was migrated without changing its corpus or
kernels. The migration checked every generated tensor against both the recorded
digest and the stored tensor's dtype, shape, and bytes. All retained payload
hashes, case records, output/reference digests, compiler provenance, and error
budgets were preserved. `storage_migration` records the original manifest hash,
comparison method, and NumPy version; `reference` retains the original
qualification provenance. Only the storage schema and manifest lock changed.

For another schema-1 bundle, use its trusted version-controlled lock to unpack
it, then run this offline maintenance command from the rocKE root:

```bash
python library/tests/run_sdpa_reference.py remove-stored-inputs \
  --arch gfx942 --bundle <original-bundle> --lock <schema-1-lock> \
  --output <new-bundle>
```

The command refuses to migrate inputs that do not reproduce exactly. In that
case, qualify a new corpus against the independent NumPy reference; do not
carry forward old error budgets for different input bytes. After a successful
migration, run verification on the target GPU with the new qualification lock,
pack the bundle, and update the architecture's committed lock and DVC pointer.
Upload the new archive with a scoped `dvc push` before pushing those Git changes.

### 3.2 Are the kernels fast?

Performance is **non-functional** — it answers *"is it fast enough?"*, not *"is it
correct?"* — so it is not a correctness oracle, though its baseline check is a
regression-style comparison.

- **Exists:** a threshold-based perf-baseline gate (per-workload baseline, metric,
  direction, and tolerances such as `max_slowdown` / `min_fraction`), plus the
  `benchmark/` suites. **⚠** This is *not* a sha "golden" — a within-tolerance
  slowdown passes and a correctness-preserving speedup passes.
- **⚠ WIP:** the `benchmark/` suites are orphaned (no gated tier) and there is no
  perf lane in CI. Performance is in scope but only partially wired (gap **G8**).

---

## 4. Platform quality — is rocKE a sound authoring platform?

Question B. The quality of *rocKE itself* as the tool authors lower kernels
through. Most of this is host-only and fast (no GPU), which is exactly why it can
be the widest, cheapest part of the suite.

### 4.1 Does the engine do what it claims?

The platform machinery — the parts an author (human or agent) relies on to turn a
spec into correct IR:

- **DSL / optimization passes** — constant folding, unroll, barrier optimization,
  unrolled lowering.
- **Dispatch / selection** — registry, support gates, arch-family routing,
  split-k. Which kernel gets picked for a problem, decided without emitting.
- **Resolvers & heuristics** — e.g. the LDS-budget resolver.
- **SSOT invariants** — the MMA/arch fragment tables in C++ and Python describe
  the same hardware (a differential check across the two tables).
- **Serialize roundtrip** — `deserialize ∘ serialize == id`, a *property* test
  proven per implementation.
- **Emit → build & reentrancy** — instances build across arches; the engine is
  safe to invoke repeatedly in one process.

*Oracles: mostly self-evident assertions and property checks; SSOT is a
differential check across the two tables.*

### 4.2 Does the platform's output stay stable?

The **regression / characterization** oracle. Golden tests pin the canonical IR of
a representative set so an *unintended* change trips a diff. They are tripwires,
not correctness checks — a golden update is legitimate when the change is intended
and reviewed. rocKE keeps goldens at several granularities (representative-IR sha,
per-config anchors, installed-artifact statics), enumerated in
[`platform/tests/README.md`](platform/tests/README.md).

Note the division of labor: golden catches *any* change to emitted IR — including
a change that lands in **both** engine implementations identically, which
byte-identity ([§4.3](#43-do-the-two-implementations-agree-the-migration-gate)) is
blind to. Golden watches *change*; byte-identity watches *divergence*; neither
watches *correctness* (that's [§3.1](#31-are-the-kernels-correct)).

### 4.3 Do the two implementations agree? (the migration gate)

The **differential byte-identity gate** — the single most load-bearing gate in the
project today, and a platform-quality check, not a kernel one. For every parity
*family*, both engine implementations emit the artifact for a given `(spec, arch)`
and the outputs must be **byte-for-byte identical**. The `*_emit.py` / `*_emit.c`
pairs are the two mutually-checking reference sides (the pseudo-oracle); the corpus
spans **both** the [platform](platform/tests/instances/parity) and
[library](library/tests/parity) parity trees. The gate is driven by
[`check_byte_identity.py`](platform/tools/check_byte_identity.py) /
[`run_diff.py`](platform/tests/instances/differential/run_diff.py); its equivalence
contract is documented in
[`dsl_docs/development/engine_parity.md`](platform/dsl_docs/development/engine_parity.md).

**Its purpose is cross-implementation equivalence** — a migration/refactoring
safety net proving the C++ engine has not diverged from the Python engine (the
dual-engine unification). It answers *"have the two implementations drifted
apart?"* and nothing else:

- It is **not** a kernel-correctness test and **not** a kernel-regression test. A
  change that lands in *both* engines identically (e.g. a shared spec or
  fragment-table edit) leaves byte-identity green while changing the emitted code —
  catching that is [§4.2](#42-does-the-platforms-output-stay-stable)'s job;
  catching wrong-but-agreeing math is [§3.1](#31-are-the-kernels-correct)'s job.
- This gate is **migration-scoped by nature**: its role changes once the C++ engine
  fully subsumes the Python one. Until then it is the contract that lets the
  migration proceed safely.

Key semantics:

- **Both-reject is parity.** For an unsupported `(spec, arch)`, *both* engines must
  reject it — counted parity-faithful (`BOTH_REJECTED`), not failure. Divergence
  is: one accepts and one rejects, or both accept with different bytes. A family
  where that is *all* that happened is a different matter: it compared no bytes,
  so it fails as `ALL_REJECTED` rather than passing as green.
- **CRASH is an implicit-oracle signal**, deliberately *not* laundered into
  "both rejected" — a segfault is a real failure the gate surfaces.
- **RANGE_DRIFT** — the two emitters enumerating different config counts is itself
  a parity failure, caught independently of byte content.
- **The gate fails closed.** Every family status is classified as passing or
  failing in one place (`run_diff.py`'s `GATE_PASS` / `GATE_FAIL`); a status in
  neither fails. A run that compared nothing — no families, no configs, or only
  rejections — is a failure, not a green.
- **Property/fuzz generation**
  ([`fuzz_diff.py`](platform/tests/instances/differential/fuzz_diff.py)) feeds the
  differential oracle with generated `(spec, arch)` inputs rather than a fixed list.

---

## 5. Execution tiers & gating

Four distinct things run here; **do not conflate them**:

| Tier | What | Gated? |
|---|---|---|
| **1. Gate** | relative-path guard → byte-identity gate → pytest (`platform/tests`) → ctest | ✅ blocking |
| **2. Diagnostics** | IR-canonical diff, fuzz diff, per-config golden check | ❌ opt-in |
| **3. GPU / numeric** | reference-oracle kernel-correctness lanes | ❌ skipped off-device |
| **4. Manual demos/tools** | hand-compiled CLIs / demos | ❌ |

**Two entrypoints with different environments and selection.** [`run_all.py`](platform/tests/run_all.py) is
the **developer** runner (guard → gate → pytest → ctest). **CI does not run
`run_all.py`** — it runs
**ctest** against the installed artifact, after the component script installs
the packaged wheels. Its registered pytest entries include the platform suite
and selected library host/GPU suites. Test registration is authoritative in
[`platform/tests/CMakeLists.txt`](platform/tests/CMakeLists.txt) and
[`platform/CMakeLists.txt`](platform/CMakeLists.txt); the provider's external
test-category YAML and TheRock's tier selection determine which registered
tests actually execute. Check the selected list in the job log or with CTest's
verbose listing. Registration alone does not establish CI coverage.

**The tree/gating reality (a second axis, orthogonal to the two questions).**
The developer pytest step collects `platform/tests`. The installed platform
pytest entry excludes the staged library tests, which have separate CTest
entries with explicit file lists. The byte-identity corpus also reaches
`library/`. Tests outside these selections need explicit enrollment; neither
their presence in the source tree nor their installation makes them gated.
Coverage gaps in [§7](#7-current-state-vs-target-the-gap-registry-wip) therefore
need to be checked against both registration and the selected CI tier.

Installed library GPU selection includes attention tests, but their Torch and
architecture gates can still skip all numeric execution. The pinned SDPA lane
uses an independently qualified bundle to run its enrolled cases without Torch.
Inspect collection and GPU comparison counts separately from host-test passes;
directory coverage alone does not establish GPU correctness.

**Harness lane bridge.** The differential harness numbers its lanes `L1…L6` (L1
`verify`, L3 `ll` = the byte-identity gate, L5 the golden anchor, L6 numeric). Read
L3 ↔ [§4.3](#43-do-the-two-implementations-agree-the-migration-gate), L5 ↔
[§4.2](#42-does-the-platforms-output-stay-stable), L6 ↔
[§3.1](#31-are-the-kernels-correct).

### 5.1 Before merging: source backends and installed CI

Run both lanes below when changing backend selection, rejection tests, test
dependencies, or packaging. A source-tree pytest pass does not validate the
installed test artifact. `run_all.py` builds native code and can expose a binding
that the CI test environment does not have.

#### 5.1.1 Source backend matrix

Use a fresh virtualenv outside the platform tree. Install the test requirements
from the CI job's TheRock revision; keep Torch out of the minimal lane. Build a
fresh native extension with `ROCKE_BUILD_PYBIND=ON` for the binding-present rows.
Run each row in a separate process, with `PYTHONPATH` containing the platform
Python root and, only where indicated, the matching `cpp/bindings` build directory.

| Lane | `ROCKE_BACKEND` | `ROCKE_CPP_STRICT` | `rocke_engine` importable? | Rejection observed by the caller |
|---|---|---|---|---|
| Default installed fallback | unset | unset | No | Python exception after the C++ import fails |
| Explicit Python | `python` | unset | Either | Python exception |
| Permissive C++ | `cpp` | `0` | Yes | Python exception after native rejection |
| Strict C++ | `cpp` | `1` | Yes | Native exception; fallback is an error |
| Differential | `both` | `1` | Yes | Python runs first; its rejection prevents a native comparison |

For rejection-contract changes, run these affected modules from `platform/` under
every row, then the complete platform pytest suite in the default missing-binding
lane:

```text
python -m pytest tests/test_rocke.py tests/core/test_gfx1250_scaled_wmma.py tests/core/test_storage.py -q -rs
python -m pytest tests -v -rs --timeout=60
```

Verify imports before each lane: print `rocke.__file__`, and use
`importlib.util.find_spec("rocke_engine")` to check binding absence; in the native
lanes import `rocke_engine` and print its `__file__`. Clear inherited backend,
fixture, and runtime overrides before configuring a lane. In particular, a
developer's `ROCKE_CPP_STRICT=1` must not leak into the default fallback run.
Use a unique temporary directory for each validation run (`TMPDIR` on POSIX,
`TEMP`/`TMP` on Windows). Some tests discover native artifacts under the system
temporary directory; an old `rocke_online` or `rocke_verify` build can otherwise
change which tests run and which library they load.

Native rejection tests must account for strict mode as well as the requested
backend. `resolve_backend() == "cpp"` does not establish which engine produced
the exception. Preserve the exact exception type and diagnostic for each mode;
accepting a tuple of unrelated exceptions would hide a broken dispatch contract.
The missing-binding regression tests deliberately block the import even when a
native extension is available.

Changes to emitted IR still require the byte-identity and golden checks, and
kernel changes require the relevant GPU numerical tests. This matrix adds
fallback coverage; it does not replace those checks.

#### 5.1.2 Installed artifact replay

Start from the failing job's **Print test reproduction command**. It pins the
artifact run, GPU family, test script, requirements files, and tier. TheRock's
`build_tools/github_actions/reproduce_test_failure.py` downloads and tests that
artifact; replaying it reproduces the old result, not an unbuilt local fix.
Validate a fix by rebuilding and packaging the changed revision with the same
settings, then testing its relocated artifact.

Run the component test script's setup as well as its CTest command. The
hipkernelprovider script installs the artifact's `rocke` and `rocke_library`
wheels with `--no-deps --reinstall` before running CTest. Those wheels are part
of the tested configuration: library imports in child processes can depend on
the installed `rocke_library` wheel even when the parent pytest process finds
the staged library through `conftest.py`. Building only the standalone CMake
install does not reproduce this setup. Use wheels built from the same revision
as the artifact, never an editable install or a wheel from an older build.

For example, [job 110907294612](https://github.com/ROCm/rocm-libraries/actions/runs/37019571757/job/110907294612)
used this selection after artifact setup:

```text
ctest -L ^standard$ -LE ex_gpu --output-on-failure --parallel 1 --timeout 7200 --test-dir build/bin/hip_kernel_provider -V --tests-information 1,,1
```

Before execution, use `ctest --test-dir build/bin/hip_kernel_provider -N -V`
with the same selection flags to check the selected names, working directory,
commands, and environment. `ROCKE_ENGINE_test_categories_external.yaml` in the
provider and TheRock's categorization script determine tier membership. A
passing command that selected zero tests is not validation. In particular,
check that `rocke_pytest` is selected when it is part of the intended coverage.

That entry runs from `build/bin/hip_kernel_provider`, with `PYTHONPATH=.`:

```text
python -m pytest ./tests --ignore ./tests/library -v -rs --timeout=60
```

Use a clean runtime environment with no editable source installs, no binding
build directory on `PYTHONPATH`, and no inherited native fixture paths. Verify
that `rocke.__file__` is under the relocated artifact and `rocke_engine` is absent.
Start from an unexecuted install when relocating it; omit `__pycache__` and
pytest caches so cached code objects cannot retain paths to the original tree.
Do not add a source directory to repair an installed import failure: install the
required module or data through CMake and the artifact manifest. Include imported
runner helpers even when their developer entry point is not executed by CI.

Record source SHA, artifact run, Python/dependency versions, LLVM/ROCm versions,
backend variables, selected CTest names, and pytest failures/skips. Compare the
same environment before and after the fix. Use CI-matched LLVM tools for object
tests; an older local compiler failure is separate evidence. Review skipped
tests explicitly: Torch-free execution, missing native fixtures, and unavailable
GPUs each leave different coverage gaps. Host pytest is not a full GPU job replay.

## 6. Invariants & contracts

Cross-cutting properties every change must preserve:

- **Byte-identity (parity)** — the two engine implementations emit identical bytes
  for every `(spec, arch)`, including symmetric rejection ([§4.3](#43-do-the-two-implementations-agree-the-migration-gate)).
- **Roundtrip** — `deserialize ∘ serialize == id` per implementation ([§4.1](#41-does-the-engine-do-what-it-claims)).
- **Copyability / relative-path guard** — no code/build file under a rocKE tree may
  reference an absolute or repo-rooted path; the tree drops into another repo
  verbatim.
- **SSOT** — the MMA/arch fragment tables have one meaning shared by both engines.

## 7. Current state vs target: the gap registry (WIP)

The **single registry** of known gaps between this strategy and today's reality,
grouped by the two questions. Inline ⚠ markers elsewhere point here; close a gap by
deleting its row and its pointer, so the doc self-heals rather than accumulating
stale markers.

**Coverage is emitter-driven.** Byte-identity is a property of one `(spec, arch)`;
arch coverage is exactly the pairs the emitters enumerate — there is no global arch
override. Most common families default to **gfx950**; a minority are arch-prefixed.
**Risk:** it is easy to *believe* an arch is covered when only gfx950 is enumerated.
Read coverage from the emitter configs ([`platform/tests/README.md`](platform/tests/README.md)), never assume.

### 7.1 Kernel-quality gaps (Question A)

| # | Gap | Impact | Target |
|---|---|---|---|
| G1 | **Correctness reference lane is torch-based** | The kernel-correctness reference oracle depends on torch, contradicting the numpy-only target below | De-torch to a numpy reference oracle |
| G2 | **Numeric coverage is narrow** — only fp32/fp16/bf16 across a handful of families | fp8/bf8/int8/mx and conv/moe/grouped-gemm have *differential* agreement but **no reference-oracle check** — two engines could agree on wrong fp8 saturation | Extend correctness lanes to the low-precision & fused families |
| G3 | **No C-engine on-GPU correctness lane** | C++ engine numerics validated only transitively (byte-identity to the Python engine) | Add a C-emitted `.ll` → HSACO → launch → compare lane |
| G4 | **Two overlapping correctness lanes** | Duplication between the platform and legacy numeric lanes | Consolidate to one canonical lane (needs GPU validation) |
| G5 | **Loose correctness verdict** — single worst-case tolerance (fp16 `atol=rtol=1e-2`) | Structural bugs can hide inside dtype-truncation noise; no structural-vs-quantization separation; NaN/Inf/denormal caught only incidentally | Split structural from dtype tolerance; add numeric edge-case tests |
| G8 | **Performance largely ungated** ([§3.2](#32-are-the-kernels-fast)) | Perf regression caught only by a manual smoke gate; benchmark suites orphaned | Wire a perf tier into CI |

> **Aspirational principle (target, not yet true — see G1):** correctness
> reference oracles should be **torch-free** (numpy only); bf16 gets a hand-rolled
> encoding or an explicit `NotImplementedError`, never a silent upcast.

### 7.2 Platform-quality gaps (Question B)

| # | Gap | Impact | Target |
|---|---|---|---|
| G6 | **Orphaned `library/` behavioral pytest** | Attention *platform behavior* (builds, dispatch wiring, golden IR) runs in no gated tier — only its parity emitters ride the gate | Gate a `library` pytest tier |
| G7 | **Orphaned `platform/python` pytest** (heuristics, benchmark) | Platform unit tests outside `platform/tests` never run in gate or CI | Collect or relocate them |

### 7.3 What "green" does and does not mean today

A green gate proves: the two engine implementations agree byte-for-byte across the
enumerated parity corpus (both trees), IR roundtrips, the platform pytest suite
passes, and the copyability contract holds. A green gate does **not** prove: that
the **kernels are correct** on hardware (Question A — needs Tier 3, which itself has
the holes in G1–G5), that the **kernels are fast** (G8), that attention *platform
behavior* is correct (G6), or that any arch beyond the enumerated configs works.

---

*Maintenance: change this document when the testing **strategy** changes. Exact
file paths, counts, and CTest target names live in
[`platform/tests/README.md`](platform/tests/README.md) and the CMake files —
reference them, don't copy them here.*

## Pinned-reference documentation

- [SDPA implementation and usage](docs/sdpa-test-reference.md)
- [GPU attention coverage and gaps](docs/gpu-attention-test-coverage.md)
- [Reference methodology and extension strategy](docs/gpu-ci-pinned-rocke-test-reference-plan.md)
