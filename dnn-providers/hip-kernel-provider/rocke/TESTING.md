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
dispatch) and the `library/` tree (the operation-specific kernel surface). Platform vs.
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
of the strategy, require a GPU (legacy lanes skip off-device; required pinned-reference lanes fail), and are where
the strategy's biggest holes live (see
[§7](#7-current-state-vs-target-the-gap-registry-wip)):

- The numeric reference lanes (e.g.
  [`instances/differential/numeric.py`](platform/tests/instances/differential/numeric.py))
  drive kernels on device and check the result within tolerance. Model-glue lanes
  exercise end-to-end kernel wiring for real model shapes.
- The pinned SDPA reference lane qualifies old rocKE
  against independent NumPy SDPA offline, then executes old and current rocKE
  on each published architecture using a conservative triangle-inequality
  error budget. The gfx1151/gfx1201 cohorts exercise standalone FP16 WMMA kernels. The gfx1201
  replacement compiles for `gfx12-generic`; gfx1200 uses the same compile target
  but remains qualification-only until its own certificate and archive exist. Required
  executions fail when hardware or qualification is missing. Each architecture cohort
  is enrolled in installed tests when a qualified bundle is supplied at build time.
  Provider builds enable `ROCKE_INSTALL_TEST_GPU_REFERENCES` by default. Run
  `dvc pull --recursive dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles`
  from the repository root before configuring. TheRock already performs this
  DVC download during source preparation. CMake extracts the archive, validates
  every payload hash against
  [`baseline_lock.json`](library/tests/sdpa_reference/architectures/gfx942/baseline_lock.json), and
  installs it under `engines/test_arch_content/rocke/sdpa/<arch>/` relative to
  the test root (`bin/hip_kernel_provider/` for provider builds). TheRock captures
  this in the test component and splits it into the matching architecture artifact,
  while the Python harness and trusted locks remain in the generic test artifact.
  Missing or corrupt data fails configuration. For builds that intentionally
  omit all pinned-reference GPU lanes, use `-DROCKE_INSTALL_TEST_GPU_REFERENCES=OFF`.
  Host checks remain installed.
  In rocm-libraries superbuild mode (`ROCM_LIBS_SUPERBUILD=ON`), this option
  defaults to OFF because the superbuild runs build-tree tests rather than the
  installed reference suite. To package the references in a superbuild, fetch
  the bundle and explicitly pass `-DROCKE_INSTALL_TEST_GPU_REFERENCES=ON`.
  Existing build directories retain their cached option value.
  To select a bundle for a pytest run without installing it, pass
  `--rocke-reference-bundle` with `--rocke-reference-operation` and
  `--rocke-reference-arch`. Use `--rocke-reference-lock` for a candidate lock.
  Normal installed CI omits path overrides and finds the packaged bundle
  automatically. CTest's explicit operation/architecture selection makes the
  suite required, including failure on a missing bundle. Standalone
  platform builds do not enable archive staging by default.
  For source verification, run
  `python library/tests/run_sdpa_reference.py verify --arch gfx942 --bundle <qualified-bundle> --current-root .`
  from the rocKE root with NumPy, HIP, and COMGR; Torch is not required.
  Add `--case <id>` to run only selected cases; that subset run is not a full
  verification. The same CLI's `cases` and `diff` commands list an
  architecture's enrolled cases and compare them with a bundle without a GPU;
  see the [workflow](library/tests/reference_common/docs/gpu-reference-workflow.md#decide-what-must-change).
  The installed GPU suite reuses separate baseline and current worker processes
  across cases. Each case still executes both versions twice and validates its
  outputs independently; worker failures and timeouts fail the test.
  For replacement baselines and coverage changes, follow the
  [authoritative reference workflow](library/tests/reference_common/docs/gpu-reference-workflow.md).
  Architecture enrollments are listed in
  [`architectures/registry.json`](library/tests/sdpa_reference/architectures/registry.json),
  used by Python qualification and verification. CMake publication is separate.
  Each architecture has its own source-dispatch adapter and, once published,
  baseline lock under `sdpa_reference/architectures/<arch>/`, with a separate
  DVC archive under `reference_bundles/sdpa/<arch>.tar.gz`. gfx942 and gfx950
  define their own case lists; gfx1151, gfx1200, and gfx1201 share the WMMA
  cohort in `sdpa_reference/implementations/wmma.py`, so editing it requires
  requalifying every published WMMA pair. Each operation owns its directory
  under `reference_bundles/`, and convolution references use
  `reference_bundles/conv/<arch>.tar.gz` and can be updated independently.
  Source pytest looks for extracted bundles under `reference_bundles/sdpa/<arch>/`;
  the runtime override can select another directory. Installed bundles
  live under `engines/test_arch_content/rocke/sdpa/<arch>/` relative to the test
  root. The `engines/test_arch_content` spelling is required by TheRock's artifact
  manifest and the hipkernelprovider kpack splitting handler. `arch_content` is
  reserved for runtime content and must not be used for these test references.
  When reference installation is enabled, CMake installs each pair in
  `platform/cmake/PublishedGpuReferences.cmake`; TheRock splits the payloads by target.
  The operation's architecture registry describes supported qualification cohorts,
  not publication status; that CMake file is the list of published pairs. Adding
  another published pair requires its adapter, independently qualified lock,
  retrievable bundle, and publication entry.
  Qualification accepts `--arch` and freezes the shared worker and architecture
  adapters into the new bundle. Existing bundles keep their original frozen code.

- The [convolution reference lane](library/tests/conv_reference/docs/conv-test-reference.md) implements the
  same offline qualification and GPU replay workflow for bounded forward cohorts.
  It shares artifact, digest, budget, snapshot, and worker transport code with
  SDPA. Its published bundles are installed by default alongside SDPA's. Verify
  replacement candidates with explicit bundle and lock paths through the
  offline CLI. Both operations use NumPy and ROCm without Torch in CI.
  Optional Torch cross-checking belongs only to offline qualification: CPU Torch
  for convolution, and CPU or ROCm GPU Torch for SDPA.
  Separate gradient tests run the dgrad and wgrad cases in
  `library/tests/conv_reference/gradient_cases.py` on gfx942 and gfx950
  against live NumPy oracles. The [coverage guide](docs/conv-coverage-roadmap.md)
  distinguishes production-selected instances from explicit kernel configurations.

- **Nothing else in this document proves the math is right.** Byte-identity
  ([§4.3](#43-do-the-two-implementations-agree-the-migration-gate)) and golden IR
  ([§4.2](#42-does-the-platforms-output-stay-stable)) are both blind to a
  wrong-but-stable, wrong-but-agreeing kernel.

### Reference input storage

Schema-2 reference bundles contain compiled kernels, the frozen replay runtime,
and qualification metadata, but no input or output tensor files. Tests regenerate
SDPA Q, K, and V using the versioned `numpy-pcg64-normal-f32-v1` recipe: a fresh
PCG64 stream seeded with zero, float32 normal samples in Q/K/V order, followed
by the existing fp16 or round-to-nearest-even bf16 encoding. Each tensor's shape,
dtype, and values must match its qualified digest before either GPU worker runs.
Convolution uses `numpy-pcg64-uniform-f32-conv-v1`: seed-zero PCG64 uniform
samples in [-1, 1), activation then weight order, cast to float32 and encoded
to the case's FP16/BF16 storage. Its input digests are checked in the same way.
A NumPy distribution implementation change that alters those bytes fails the test;
a seed alone is not considered sufficient evidence of reproducibility.

Generated inputs are passed to both isolated workers through a temporary `.npz`
file, which is removed after the comparison, including on worker failure. This
keeps the frozen baseline worker unchanged. Outputs remain temporary as before.
Neither input nor output tensors are included in the DVC archive or installed
reference bundle. Archive validation rejects `.npz` and `.npy` payloads for schema 2.

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

### 4.4 Is the emitted IR legal? (the toolchain gate)

§4.2 and §4.3 both compare rocKE's output against *another copy of rocKE's
output* — a golden sha, or the other engine. Neither asks the AMDGPU toolchain
whether the IR is valid at all, so an illegal construct that is emitted stably
and by both engines is green in both. The **emitted-IR validity gate**,
[`check_ir_validity.py`](platform/tools/check_ir_validity.py), asks the external
question: it lowers every case in the representative corpus
([`rocke_ir_parity_harness.cases()`](platform/tests/instances/rocke_ir_parity_harness.py),
the same corpus §4.2 hashes) and pushes each module through `clang` to a linked
hsaco.

Three design points, each of them load-bearing:

- **The oracle is a LINK, not a verify and not a codegen.** A `declare` for an
  intrinsic that does not exist passes `opt -passes=verify`, *and* passes
  `clang -S` — the backend silently treats the unknown `llvm.*` name as an
  ordinary external function and emits a GOT-relative call. The undefined symbol
  appears only when the relocatable is linked. Stopping anywhere earlier makes
  the gate blind to the whole *fictional-intrinsic* bug class.
- **Every compile is a subprocess.** A backend failure is a
  `report_fatal_error`, not an exception: it takes the process down. Compiling
  in-process (via comgr, which is otherwise faster) means one bad module kills
  the run and emits no report. The isolation is a correctness requirement, and
  it makes the per-module diagnostic free.
- **Only the host's own LLVM flavor can be validated.** Modules lower at any
  flavor, but there is no local compiler for the others; those report
  `UNVALIDATED`, never green. On a host with no LLVM tools at all the gate
  self-skips loudly — pass `--strict` to make that a failure instead.

Failures that are known and owned live in a `KNOWN_BAD` allowlist, same
convention and same rule as `KNOWN_VIOLATIONS` in
[`test_library_layering.py`](library/tests/test_library_layering.py): **it only
shrinks.** An entry that starts compiling is itself reported as a failure, so a
fix cannot leave dead weight behind.

### 4.5 Can this target lower this intrinsic? (the arch-domain gate)

§4.4 asks a compiler. The **arch-domain lane** asks *committed measured data*
instead: `tools/gen_arch_domain.py` probes, per (declaration key, gfx target),
whether that declaration actually links, and commits one JSON column per LLVM
flavor under `platform/python/rocke/core/arch/data/`. Both engines consult it at
their demand chokepoint — `_Lowerer._need` in Python, `rocke_ll_need` in C++ —
on every intrinsic a kernel demands. The C++ engine's copy of the table is
generated from the same columns at build time (`tools/gen_arch_domain_cpp.py`),
so there is one table and no hand-maintained mirror; regenerating a column needs
no other step.

What the lane does with a measured negative is `ROCKE_ARCH_DOMAIN`:

| mode | effect |
|---|---|
| `error` | refuse the lowering with `ArchDomainError` (a `NotImplementedError`), unless the cell is waived; both engines refuse with the same message |
| `warn` (default) | report it as `ArchDomainWarning` and keep lowering |
| `off` | skip the lookup |

Any other value raises in both engines rather than falling back to the
default. On the default `cpp` backend a refusal is raised, never "fallen back"
from: the Python lowerer reads the same table and would refuse the same kernel.
The C++ engine has no warning channel, so on that backend the reports come from
a pass over the IR it emitted, which also checks that it refused what it had to.

**Only a definite negative speaks.** `arch_absent` is the sole negative status.
A flavor with no committed column, an unswept target, a key absent from the
table, and the no-data statuses (`target_unsupported`, `toolchain_crash`,
`toolchain_timeout`, `probe_error`) are all silent — 14% of the llvm20 column is
no-data, and warning on those would train people to ignore the lane.
`name_absent` is deliberately *not* reported: §4.4 already catches a reachable
one with a real diagnostic.

**The escape hatch is one cell.** A row can be measured wrong, and in `error`
mode a wrong `arch_absent` would block every kernel that needs that intrinsic
until the column is fixed. `platform/python/rocke/core/arch/data/arch_domain_waivers.json`
lists waived `(key, arch, flavor)` cells, each with the reason the measurement is
wrong; both engines read it through the same build step as the columns, and a
waived cell is reported instead of refused — never silenced. It is for a wrong
measurement, not a kernel bug: a kernel that demands an intrinsic its target
cannot lower is fixed in the kernel. `ROCKE_ARCH_DOMAIN=warn` reports every
negative instead of stopping at the first, which is a way to diagnose, not to
ship.

The gate, [`check_arch_domain.py`](platform/tools/check_arch_domain.py), lowers
the same corpus §4.4 uses at every committed flavor with the lane in `warn` mode
and goes red on:

- a negative with no waiver: a kernel is demanding an intrinsic its target
  cannot lower;
- a stale waiver: its cell no longer measures negative, so the column was fixed
  and the waiver is deleted in the same change;
- when the `rocke_engine` extension is importable, the C++ engine in `error`
  mode refusing a different set of cases than the Python lowerer would.

This gate and §4.4 overlap on purpose and neither subsumes the other:

- **It needs no toolchain.** Lowering is pure Python and the table is committed
  data, so this gate rules on *every* flavor on any host — where §4.4 can only
  ever rule on the host's own and reports `UNVALIDATED` for the rest.
- **It names the intrinsic.** §4.4 reports whatever the backend chose to say,
  often an unnamed fatal error. This reports `(key, arch, flavor)` plus the
  diagnostic the probe recorded.

### 4.6 Does the measured arch-domain artifact reproduce?

The warning gate consumes committed measurements; it does not prove that this
host's toolchain reproduces them. The `rocke_arch_domain_reproducibility` CTest
runs `tools/gen_arch_domain.py --check` over the full intrinsic/architecture
matrix for the host flavor and compares the result with the committed column.
This is an artifact-reproduction gate backed by the host toolchain, not a Python
unit test: it owns external compiler processes, a dedicated timeout and resource
policy, and an explicit CTest skip result.

The gate is registered in both execution environments. A developer build runs
the source tool against the source package data. Provider CI runs a packaged,
test-only copy of the same implementation against the column in the relocated
installed `rocke` package. This follows the library-side test contract: CMake
stages every input, CTest supplies artifact-relative paths, and the test never
configures or reaches back into a source checkout.

The result mapping deliberately preserves the former pytest item's behavior:
no clang, no committed host-flavor column, and a clang/resolved-flavor mismatch
skip; a partial or internally inconsistent LLVM installation fails. Flavor
auto-detection errors, including unavailable COMGR evidence, also remain
failures during this migration.

The reproduction scheduler preserves exhaustive coverage while avoiding
redundant compiler startup. It batches declaration discovery, preflights each
target once, and combines expected-positive calls per architecture. The
committed artifact schedules those batches but cannot make them pass: every call
must survive compile, link, and target assembly, and a failed batch is bisected
to individual verdicts. Expected-negative cells remain individual and use
explicit operand witnesses. Independent work shares a quota-aware bounded
worker pool, with `--jobs` available for an explicit site override.
Expected crashes and timeouts whose normalized evidence still matches are not
retried; changed evidence, unexpected verdicts, and unclassified outcomes still
receive a contention-free retry.

The fast structure and generator units remain source-only in
`platform/tests/tools/test_arch_domain_artifact.py`. Normal pytest and the
`ROCKE_BACKEND=both` pytest pass do not re-run the compiler sweep. The dedicated
CTest is categorized as standard in both build-tree and installed manifests, so
installed-artifact CI can schedule the reproduction result without adding the
external process sweep to the quick or FFM pytest pools. A named skip is not
reproduction coverage: check the CTest result and reason, especially when the
compiler's exact major differs from rocKE's resolved flavor or no committed
column exists. Keeping one generator implementation behind two thin
registrations preserves local coverage while avoiding duplicated test logic.

**Kernel-author routing.** Ordinary kernel work does not need a separate focused
run beyond the full pre-merge `tests/run_all.py`; once that runner has configured
and built native tests, its unfiltered CTest phase includes this sweep. Run the
focused CTest while iterating on intrinsic declarations, architecture-domain
data or generation, LLVM discovery/classification, or this test's CMake/install
wiring. A skip confirms orchestration only, not the committed measurements.

---

## 5. Execution tiers & gating

The execution classes below have different selectors and verdicts; **do not
conflate registration, CI enrollment, hardware selection, and a test actually
performing comparisons**:

| Execution class | Environment and oracle | Verdict |
|---|---|---|
| **Developer pre-merge gate** | `run_all.py`: relative-path guard → byte identity → emitted-IR validity → arch-domain warning gate → platform pytest → conditional differential pytest → all configured build-tree CTests | Blocking. This is a complete gate, not a fast smoke; differential pytest runs only when `rocke_engine` is importable, and the CTest phase includes the standard-tier arch-domain reproduction sweep once native tests are built. |
| **Installed host CI** | Selected CTests from the relocated provider artifact, including platform and explicitly enrolled library pytest cohorts | Blocking when selected. Registration alone is not enrollment. |
| **Installed pinned-reference GPU CI** | One independently qualified reference CTest per published operation/architecture pair | Blocking on its matching required lane; missing hardware, qualification, bundle, or comparisons fails. Other processors' entries skip. |
| **Legacy/manual GPU numeric** | Live NumPy/Torch reference tests and diagnostic harnesses not covered by a required pinned lane | Opt-in or capability-skipped; a skip proves no kernel correctness. |
| **Diagnostics and demos** | IR diff/fuzz tools, per-config golden checks, hand-compiled CLIs, and demos | Opt-in and non-gating. |

**Two entrypoints with different environments and selection.** [`run_all.py`](platform/tests/run_all.py) is
the **developer** runner (guard → byte identity → IR validity → arch-domain
warning gate → platform pytest → differential pytest when `rocke_engine` is
importable → CTest). **CI does not run
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
The published convolution lane uses the same workflow for its forward cohort.
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

On Linux, direct source pytest runs need a matching engine archive for the native
compiler-detection fixture. Build it once outside the source tree and set
`ROCKE_TEST_ENGINE_ARCHIVE` to its absolute path before the commands below:

```bash
cmake -S . -B "$TMPDIR/engine" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DROCKE_BUILD_PYBIND=OFF -DROCKE_INSTALL_TESTS=OFF
cmake --build "$TMPDIR/engine" --target rocke_core --parallel 2
export ROCKE_TEST_ENGINE_ARCHIVE="$TMPDIR/engine/librocke_core.a"
```

Use an absolute, unique `TMPDIR` as described below. With a multi-config generator,
select `--config Release` and use `engine/Release/librocke_core.a`. The developer
runner supplies an archive from its build root to both pytest passes automatically;
an explicit archive override takes precedence. Installed tests use the packaged
archive beside the detector module. Missing or invalid explicit archives fail.
Without either archive, standalone source tests build the engine in their fixture
setup, which counts toward the per-test timeout; prebuilding keeps that work outside
the documented `--timeout=60` run.

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

#### 5.1.3 Writing native recipe replay tests

Native replay APIs can hide a source-build dependency. In particular,
[`online.recipe_cbor_to_llvm()`](platform/python/rocke/portable_ir/src/online.py)
calls `online.load()`, which calls `build_lib()` when it finds no prebuilt shared
library. A test can pass in a source checkout or with a cached library, then fail
in an installed artifact because the source tree and `CMakeLists.txt` are absent.
Do not let replay test setup implicitly configure or build native code.

When adding or changing a native replay test:

- Build the native artifact during build/setup, from the same revision as the
  Python oracle. Run a prebuilt replay CLI, or supply a prebuilt shared-library
  path explicitly before calling the online API. Never fall back to a cached
  library or an automatic source build to make the test pass.
- If the prebuilt CLI is unset or missing, skip the native replay lane with a
  reason identifying the missing artifact, following the existing portable-IR
  tests. Once present, execution errors and IR mismatches are failures; do not
  catch them as capability skips. Keep independent Python coverage enabled.
- Let the launcher supply artifact paths. Installed CTest paths must be relative
  to its working directory so relocation works. Do not infer artifact presence
  or location from test-file paths, nearby source files, or conventional build
  directories. Install required content through CMake and the artifact manifest.
- Keep a suite-specific fixture local to its module. Do not mutate shared
  environment settings during fixture setup or add an autouse fixture that
  changes other suites' native-lane discovery or skips. Shared launcher wiring
  or fixtures need an intentional consumer contract and validation of every
  affected suite.

The existing replay CLI target is `rocke_portable_ir_replay_cli`; its installed
location is `tests/portable_ir/`. `ROCKE_REPLAY_CLI` is the single launcher setting
for native CLI replay, including TF32 and the portable-IR suites. Installed CTest
supplies the relative executable path once for the pytest run, enabling all
consumers of that artifact. It also supplies `ROCKE_PROVENANCE_MANIFEST`, a
build-generated version/build-ID record installed beside the CLI. Source-less
tests validate the CLI against that independent expectation instead of trying
to hash or build from unavailable sources.

For source execution, build the target first and supply its executable through
`ROCKE_REPLAY_CLI`. Reuse this setting for new replay consumers rather than adding
a kernel-specific setting. When changing its launcher wiring, validate every
consumer, including suites whose native lanes previously skipped. Check absent
and present artifacts, a broken executable, and unrelated suites' skips.
Repeat the native lane in a clean
relocated install without source directories or native caches available; a
source-only pass cannot establish installed-artifact support.

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
| G1 | **Many legacy numeric lanes still require Torch** | Pinned SDPA and the bounded forward-convolution cohort run without Torch. Other Torch-gated cohorts remain uncovered in environments without Torch | Extend independently qualified references to the remaining cohorts |
| G2 | **Numeric coverage is narrow** — only fp32/fp16/bf16 across a handful of families | Pinned reference coverage is bounded; broader convolution directions, low-precision and fused families need independent numeric coverage beyond differential agreement | Extend correctness lanes to the low-precision & fused families |
| G3 | **No C-engine on-GPU correctness lane** | C++ engine numerics validated only transitively (byte-identity to the Python engine) | Add a C-emitted `.ll` → HSACO → launch → compare lane |
| G4 | **Two overlapping correctness lanes** | Duplication between the platform and legacy numeric lanes | Consolidate to one canonical lane (needs GPU validation) |
| G5 | **Loose correctness verdict** — single worst-case tolerance (fp16 `atol=rtol=1e-2`) | Structural bugs can hide inside dtype-truncation noise; no structural-vs-quantization separation; NaN/Inf/denormal caught only incidentally | Split structural from dtype tolerance; add numeric edge-case tests |
| G8 | **Performance largely ungated** ([§3.2](#32-are-the-kernels-fast)) | Perf regression caught only by a manual smoke gate; benchmark suites orphaned | Wire a perf tier into CI |

> **Reference CI policy:** pinned-reference verification runs with NumPy and
> ROCm, without installing Torch. Independent oracles, including optional Torch,
> run offline during qualification. BF16 storage has explicit encoding and
> rounding. Extending this coverage to legacy lanes remains G1.

### 7.2 Platform-quality gaps (Question B)

| # | Gap | Impact | Target |
|---|---|---|---|
| G6 | **Partially enrolled `library/` behavioral pytest** | Installed CTest registers explicit attention, convolution, DSA, MLA, GDN, reference-contract, and GPU cohorts, but external tier selection does not currently enroll DSA and source tests outside the explicit CMake selections remain ungated; registration or adding a file under `library/tests/` does not establish CI coverage | Audit registered-but-unselected entries and unregistered files; enroll each in an owned CTest tier or document why it remains manual |
| G7 | **Orphaned `platform/python` pytest** (heuristics, benchmark) | Platform unit tests outside `platform/tests` never run in gate or CI | Collect or relocate them |

### 7.3 What "green" does and does not mean today

A green **developer pre-merge gate** proves: the two engine implementations agree
byte-for-byte across the enumerated parity corpus (both trees), emitted IR passes
the host-toolchain validity gate where available, arch-domain warnings match the
allowlist, platform pytest passes, configured build-tree CTests pass or report an
accepted named skip, and the copyability contract holds. If `rocke_engine` was
not importable, GREEN also does not prove that the differential pytest pass ran;
its explicit skip must be reviewed. A green developer gate does not prove that
kernels are numerically correct on hardware, that they are fast (G8), that every
`library/tests/` file is enrolled (G6), or that an architecture absent from the
enumerated configs works.

A green **matching installed pinned-reference lane** additionally proves the
enrolled operation/architecture cases matched their independent qualified
reference under that lane's numerical contract. A green installed host lane or a
GPU lane that selected no matching test, skipped, or performed zero comparisons
does not make that claim. The remaining numerical holes are G1–G5.

---

*Maintenance: change this document when the testing **strategy** changes. Exact
file paths, counts, and CTest target names live in
[`platform/tests/README.md`](platform/tests/README.md) and the CMake files —
reference them, don't copy them here.*

## Pinned-reference documentation

Detailed guides live in the reference harnesses' `docs/` directories under
`library/tests/`. Kernel authors start with the
[quickstart](library/tests/reference_common/docs/quickstart.md), which covers
running the pinned suites and adding cases, architectures, and operations. The
[documentation hub](library/tests/reference_common/docs/README.md) covers kernel
compatibility, the baseline lifecycle, and a glossary.

- [Adding or updating a reference: authoritative workflow](library/tests/reference_common/docs/gpu-reference-workflow.md)
- [Triage a failing pinned test](library/tests/reference_common/docs/gpu-reference-workflow.md#triage)

- [SDPA implementation and usage](library/tests/sdpa_reference/docs/sdpa-test-reference.md)
- [Convolution implementation and usage](library/tests/conv_reference/docs/conv-test-reference.md)
- [GPU attention coverage and gaps](docs/gpu-attention-test-coverage.md)
- [Reference methodology and extension strategy](library/tests/reference_common/docs/gpu-ci-pinned-rocke-test-reference-plan.md)

## Pinned convolution references

Each published forward suite runs the cases its architecture enrolls and three
failure controls without Torch; from the rocKE root,
`python library/tests/run_conv_reference.py cases --arch <arch>` lists them.
Published bundles are installed by default for provider artifact builds. See the
[convolution reference guide](library/tests/conv_reference/docs/conv-test-reference.md)
for the numerical contract and the [shared workflow](library/tests/reference_common/docs/gpu-reference-workflow.md)
for qualification, publication, and candidate verification.

From an installed provider test directory, run the convolution suites; the
suites of other processors skip as described below:

```bash
ctest -R '^rocke_conv_gpu_gfx[0-9a-z]+_pytest$' --output-on-failure
```

The shared selector consumes TheRock's concrete `AMDGPU_TARGETS` and detected
processor. Another requested processor's suite skips with exit 77; matching
hardware requires its own valid bundle. Build-family names do not enroll a
processor or authorize reuse of another processor's certificate.

Gradient host checks join `rocke_library_conv_pytest`; gradient GPU checks join
`rocke_library_gpu_pytest`. They use live NumPy oracles, including atomic and
two-stage execution, with three repetitions per case. Inspect that every
gradient case in `library/tests/conv_reference/gradient_cases.py` executed on
the matching processor; aggregate passes in the omnibus suite can include host
checks and skip unrelated GPU tests. For a focused installed run:

```bash
python -m pytest tests/library/tests/test_conv_gradient_correctness.py \
  --rocke-reference-operation conv --rocke-reference-arch gfx942 -v -rs
```

Use `gfx950` on that processor. Gradients do not need a pinned bundle. Their
source and installed checks cover Python-emitted kernels, not native provider
or kernel-pack execution. See the [six-lane validation record](docs/conv-six-lane-validation.md).

## Shared GPU reference artifacts

Use [Adding or updating a pinned GPU reference](library/tests/reference_common/docs/gpu-reference-workflow.md)
for archive creation, qualification, publication, and enrollment. Both operations
use `library/tests/reference_common/artifact.py`; archive commands require an
explicit `--operation`, authenticate against a reviewed lock, and use only the
Python standard library. They do not qualify GPU results. Schema-1 tensor bundles
are not supported.

### Installing enrolled GPU reference bundles

`ROCKE_INSTALL_TEST_GPU_REFERENCES` is the single installation switch. It defaults
ON for provider artifact builds and OFF for standalone and rocm-libraries
superbuild configurations. Host reference tests remain installed when it is OFF.
CMake installs all pairs listed in
[`PublishedGpuReferences.cmake`](platform/cmake/PublishedGpuReferences.cmake),
failing on missing archives, missing locks, or failed integrity checks. Fetch all
published archives before configuring. Each operation/architecture keeps its own
archive, committed lock, and installed payload. Qualification registries enable
offline work; only the CMake publication lists enroll installation.

### Running installed reference tests without Torch

Use an environment containing NumPy, pytest, and pytest-timeout, with HIP and
COMGR available. Do not install Torch for these suites. From the installed
provider test directory:

```bash
python -c 'import importlib.util; assert importlib.util.find_spec("torch") is None'
ctest -V -R '^rocke_(reference_common|[a-z0-9_]+_reference_unit|[a-z0-9_]+_gpu_gfx[0-9a-z]+)_pytest$' --output-on-failure --no-tests=error
```

The patterns follow the `rocke_<operation>_reference_unit_pytest` and
`rocke_<operation>_gpu_<arch>_pytest` names, so they need no edit when a pair
is published. Inspect the selected tests: disabling reference installation omits
the GPU entries, and entries for other architectures skip. A successful CTest
command alone does not establish GPU coverage. On the matching GPU, each
operation's entry runs one comparison for every case that
`python library/tests/run_<operation>_reference.py cases --arch <arch>` lists
from the rocKE root, plus three failure controls. Confirm the matching entries
executed with zero selected GPU skips.

For `AMDGPU_TARGETS=gfx1200,gfx1201`, a gfx1200 runner skips the gfx1201 pinned
entry with code 77. There is no installed gfx1200 pinned entry until its own
archive is qualified and published. See
[gfx120X family selection](library/tests/sdpa_reference/docs/sdpa-test-reference.md#gfx120x-family-ci-selection)
for the distinction between a valid skip and GPU reference coverage.

Default publication supplies one bundle for each pair that
`PublishedGpuReferences.cmake` lists. Fetch all of their DVC archives before
configuring; missing published archives fail configuration.

The generic gfx1201 replacement is published. A fresh anonymous download matches
its committed DVC pointer and baseline lock. From the rocm-libraries root:

```bash
dvc pull --recursive dnn-providers/hip-kernel-provider/rocke/library/tests/reference_bundles
```

See the [reference inventory](platform/tests/README.md#installed-pinned-reference-suites)
for the expected entries and cases, and the
[WMMA runtime requirements](library/tests/sdpa_reference/docs/sdpa-test-reference.md#standalone-wmma-coverage)
for the tested gfx1151/gfx1201 environments.

The shared launcher blocks Torch imports in baseline and current worker
interpreters, including frozen replay workers. Neither CPU oracle runs during
verification. This restriction is confined to reference workers and does not
change the runtime's behavior for other callers or the legacy Torch test suites.

Each installed operation/architecture pair has its own CTest entry:
`rocke_<operation>_gpu_<arch>_pytest`, with a five-minute timeout. CTest passes
`--rocke-reference-operation` and `--rocke-reference-arch` to select that cohort;
shared negative checks still run for the selected architecture. A different GPU
lane exits with code 77, recorded by CTest as skipped. On the intended lane,
missing hardware, missing bundles, and validation errors fail the test.

The provider's existing category filters remain unchanged. Reference entries do
not add `ex_gpu_*` labels: the current provider runner would use those labels
as an inclusion filter and omit unrelated tests. Architecture selection occurs
inside the reference test entry, before collection and bundle lookup. TheRock
passes `AMDGPU_FAMILIES` for the family and comma-separated `AMDGPU_TARGETS`
for the fetched target artifacts. The gate uses exact targets when available
and selects by the physical HIP processor when they are absent. It does not
infer reference enrollment from family names. A processor without a published
cohort skips the other processors' suites; this records absent pinned-reference
coverage. Explicit candidate requests still require their bundle and lock.
See the [processor-selection rules](library/tests/reference_common/docs/gpu-reference-workflow.md#processor-selection-in-therock-jobs).
Adding a
published architecture creates a separate CTest automatically, which the
provider category YAML selects by pattern. The new architecture also needs an
adapter, a qualified bundle, and matching CI hardware.
