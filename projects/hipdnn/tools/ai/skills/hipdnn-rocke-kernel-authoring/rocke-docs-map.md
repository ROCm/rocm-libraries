# Where rocKE's own documentation answers the question

rocKE ships a self-contained documentation tree at
`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/` — 70-odd pages covering the
Python authoring interface, the IR vocabulary, the primitives, the lowering pipeline, the
runtime and every kernel family. No other hipDNN skill references it. It owns how a rocKE
kernel is written; this skill owns only the seam to hipDNN. **Read from that tree; do not
ask this skill to restate it.**

Every citation below spells the tree's path in full so that it resolves.

## Read in this order

| Question | Page |
|---|---|
| I have never written a rocKE kernel | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/onboarding.md:101-217` — a complete, runnable vector-add worked example. The surrounding document is a three-week program; the worked example alone is the fastest from-zero path |
| Am I authoring a kernel or changing the engine? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/onboarding.md:4-12` — the fork. Authoring is the path for this skill. Engine-internals work is a different job with byte-identity invariants you can break silently |
| How do I get an interpreter that imports `rocke`? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/setup_guide.md:10-28` for the stack and supported arches, `:43` for the `PYTHONPATH` the ad-hoc path needs, `:227-234` for the pitfalls table. For the CMake-managed venv, read [build-and-environment.md](build-and-environment.md) instead |
| What is the mental model? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/architecture/mental_model.md:1-5` — Python data objects and first-class SSA IR instead of C++ templates |
| How does an operation idea become a `KernelDef`? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/architecture/authoring_model.md:1-5` — the authoring pipeline shape end to end |
| What ops can I emit? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/op_vocabulary.md:1-3` — the complete IR op vocabulary, with the `IRBuilder` method for each |
| What hardware primitives are available? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/primitives/intrinsics_and_primitives.md:1-3` — MFMA atoms, loads, layouts, schedules, epilogues, reductions |
| What is already shipped for my family? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/instances/index.md:1-10` for the per-family index; `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/instances/attention.md:8-45` for the attention variant-to-file map. **Check this before writing a builder** |
| How do I compile and launch what I built? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/runtime/compile_launch_and_manifest.md:1-3` — `KernelDef` through HSACO to a torch-aware launch |
| How do I test it and debug wrong numbers? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/testing.md:1-3` |
| The build or the engine is misbehaving, not my kernel | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/troubleshooting.md:1-6` — the failure-mode catalog for engine and build |
| comgr threw at me | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/runtime/comgr_and_hipmodule.md:208-209` — `COMPILE_SOURCE_TO_BC` `status=4` is malformed IR; `status=1` is usually an intrinsic-signature mismatch against the loaded toolchain |
| Which environment variables exist? | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/env_flags.md:15-17` for the ones that change results — and [build-and-environment.md](build-and-environment.md) for why only two of them matter to you |
| Quick symbol lookup | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/reference/api_index.md:1-3` |
| I want the autotuner | `dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/autotune/overview.md:1-3` — in-process, and **not** on the hipDNN path: nothing autotunes at launch once the kernel is packed |

## Read only if you need them

- `dsl_docs/ir_lowering/` — how the IR becomes LLVM. Needed when debugging a lowering
  failure, not when authoring.
- `dsl_docs/fusion/`, `dsl_docs/architecture/transform_dag.md`,
  `dsl_docs/architecture/coordinate_address_planning.md` — deeper authoring mechanics for
  a kernel that is not a straightforward instance of an existing family.
- `dsl_docs/optimization/` — a large performance-tuning tree, including per-arch pages for
  gfx942 and gfx950. Out of scope here: this skill's deliverable is correctness. It is
  what the fourth arrow of the flow reads.

## Ignore, for this job

- `dsl_docs/development/engine_contributing.md` and `dsl_docs/development/invariants.md` —
  the other side of the fork. They govern changing the two engines that must stay
  byte-identical, which authoring never does.
- `dsl_docs/architecture/dual_backend_unification_rfc.md`,
  `dsl_docs/development/engine_parity.md` — engine-internals concerns.
- `dsl_docs/architecture/*_experiment_summary.md`,
  `dsl_docs/development/refactor_opportunities.md`,
  `dsl_docs/development/known_gaps.md` — point-in-time notes, not contracts.

## Three pages that read as current and are not

**1. `dsl_docs/hipdnn_provider/plan.md` is superseded. Do not use it as guidance.**
It is a v0.9 design plan headed "All open questions resolved; plan is
implementation-ready", which makes **runtime Python execution** the primary architecture
via an embedded CPython interpreter and pybind11
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/hipdnn_provider/plan.md:1-13`).
That contradicts the ahead-of-time `hkp_pack` path that actually shipped and that
[SKILL.md](SKILL.md) describes. The document partly admits it
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/hipdnn_provider/plan.md:100-106`),
but only in a footnote, and only about one finding. Treat the whole page as historical.

**2. `dsl_docs/examples/index.md` is half stale.** It lists examples in two forms
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/examples/index.md:3-6`). The
first half — the Python-owned generators under `python/rocke/examples/`
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/examples/index.md:10`) — is
real in this tree and worth reading. The second half — the CMake-integrated generators
under `example/ck_tile/dsl/<NN>_*/gen.py`
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/examples/index.md:24`) — names
a directory that **does not exist** under
`dnn-providers/hip-kernel-provider/rocke/platform/`. Every row of that second table is
unreachable here. Verify before following it:

```bash
ls dnn-providers/hip-kernel-provider/rocke/platform/example/ck_tile/dsl   # expected: no such directory
ls dnn-providers/hip-kernel-provider/rocke/platform/python/rocke/examples # expected: real
```

**3. `dsl_docs/architecture/ROCKE_LEARNING_PROGRAM.md` is a stale duplicate of
`dsl_docs/development/onboarding.md`.** The two differ by fourteen lines: `onboarding.md`
carries the author-versus-engine fork banner
(`dnn-providers/hip-kernel-provider/rocke/platform/dsl_docs/development/onboarding.md:4-12`)
that the architecture copy lacks, and one parity command has diverged between them. Read
`onboarding.md`; do not read both, and do not cite the architecture copy.

## What rocKE's docs do not cover at all

Nothing under `dsl_docs/` mentions a UKD, a KMD, a kpack or `hkp_pack` — outside the
superseded `hipdnn_provider/plan.md`, the string does not appear. rocKE's documentation
teaches authoring against torch and stops there. The entire hipDNN seam —
what a descriptor must carry, which checks stop running once the kernel is packed, what
the matcher owes — is [handover-contract.md](handover-contract.md)'s subject, and exists
nowhere else.
