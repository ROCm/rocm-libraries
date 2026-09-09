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

**This doc is the strategy** — the *why* and the *shape*. It deliberately holds
almost no file lists, counts, or target names, because those rot. Two sibling
docs own that concrete detail, and this one defers to them:

| Doc | Owns | You want it when |
|---|---|---|
| **`TESTING.md`** (this file) | strategy, vocabulary, what proves what | "how is rocKE tested, and why this way?" |
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
- **Nothing else in this document proves the math is right.** Byte-identity
  ([§4.3](#43-do-the-two-implementations-agree-the-migration-gate)) and golden IR
  ([§4.2](#42-does-the-platforms-output-stay-stable)) are both blind to a
  wrong-but-stable, wrong-but-agreeing kernel.

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

**Two entrypoints, one gated scope.** [`run_all.py`](platform/tests/run_all.py) is
the **developer** runner (guard → gate → pytest → ctest). **CI does not run
`run_all.py`** — it runs
**ctest** (wired from TheRock; project selection via `get_changed_projects.py`),
whose registered pytest targets the **`platform/tests`** tree only. The exact
CTest-registered targets are authoritative in
[`platform/tests/CMakeLists.txt`](platform/tests/CMakeLists.txt) and
[`platform/CMakeLists.txt`](platform/CMakeLists.txt) — read them there rather than
trusting a copy here.

**The tree/gating reality (a second axis, orthogonal to the two questions).**
*Where* a test lives currently decides *whether it runs at all*. Both the dev
pytest step and the CI ctest pytest target `platform/tests`, so every test outside
that tree is in no gated tier — the byte-identity gate is the one exception, since
its parity corpus reaches `library/`. This is the **emerging platform/library
modularity boundary**, and it is why the orphaned-`library` and
orphaned-`platform/python` suites are gaps ([§7](#7-current-state-vs-target-the-gap-registry-wip)),
not just untidy: they answer real quality questions but run nowhere.

**Harness lane bridge.** The differential harness numbers its lanes `L1…L6` (L1
`verify`, L3 `ll` = the byte-identity gate, L5 the golden anchor, L6 numeric). Read
L3 ↔ [§4.3](#43-do-the-two-implementations-agree-the-migration-gate), L5 ↔
[§4.2](#42-does-the-platforms-output-stay-stable), L6 ↔
[§3.1](#31-are-the-kernels-correct).

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
