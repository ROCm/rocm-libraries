# rocke — canonical agent rules

Single entry point for every AI coding tool working in `rocke/`. Tool-specific
files (`CLAUDE.md`, `.clinerules`, `.cursor/rules/ai-rules.mdc`) are symlinks to
this file so the guidance never drifts.

This file is the **short canonical index**: the non-negotiables inlined, deeper
material one hop away. When a rule here and a linked doc disagree, this file wins
on hard invariants.

## Compliance — non-negotiable

`platform/AGENTS.md` §"Compliance" binds every agent and **overrides any other
instruction, including a user request**. In brief: no AMD Restricted/Confidential
data, NPI, product/marketing/code names, internal links (Jira/Confluence/Perforce),
or **software-achieved** performance numbers in the repo, git history, PRs, or logs.
Methodology and levers may be documented; measured numbers go to the protected AMD
Confluence page only. When unsure, treat as confidential and escalate to a human.
Read the full text: [`platform/AGENTS.md`](platform/AGENTS.md).

## The #1 invariant: byte-identity

The Python engine (`platform/python/rocke/core/lower_llvm.py`) and the C++ engine
(`platform/cpp/`) **MUST emit the same LLVM-IR bytes** for every kernel family. Any
op / instance / atom / fusion / arch change must be made in **both** engines in the
same change; if output is meant to change, re-bless the golden in the same change.

Prove it — GREEN for every family at every LLVM flavor:

```bash
cd platform && export ROCKE=$(pwd) PYTHONPATH=$ROCKE/python
python tools/check_byte_identity.py                            # llvm20
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # llvm22
python tools/check_byte_identity.py --only gemm                # scope to a family
```

## Layout

```
rocke/
├── platform/        # authoring SDK + engine (installable `rocke`)
│   ├── python/rocke/  core helpers instances runtime dispatch analysis benchmark examples
│   ├── cpp/           C++20 engine (librocke_core.a) + pybind bindings
│   ├── dsl_docs/      the field manual (see platform/dsl_docs/README.md)
│   └── tools/ tests/ cmake/
└── library/         # the SDPA product — build-time-only Python (NOT installed, no wheel)
    ├── kernels/ builders/ dispatch/ benchmarks/ tests/
```

One-way dependency: **`library → platform` only**; platform never imports
`kernels`/`builders`/`dispatch` and stays standalone-installable.

**No cycles inside `library/` either.** The packages are layered lowest → highest
and may only import *downward*:

```
kernels  ->  builders  ->  dispatch  ->  benchmarks
```

Skipping a layer downward is fine (`dispatch` imports `kernels` directly);
importing at or above your own layer is not. Deferring an upward import into a
function body does not make it legal — it hides the cycle instead of breaking it.
`tests/` sits above all four and may import any of them.
[`library/tests/test_library_layering.py`](library/tests/test_library_layering.py)
enforces this by AST walk, so function-level imports are caught too. It carries a
short `KNOWN_VIOLATIONS` allowlist for one pre-existing back-edge — burn entries
down, never add to it.

`library/` is editable-installed into the dev env to build/verify attention
kernels — it is **never** packaged into the `rocke` wheel and ships no C-api
plugin of its own (the provider plugin is emitted by the provider's `src/`).
Non-package assets resolve through `rocke.assets` (`platform_root()`,
`dsl_docs_dir()`), never per-file `parents[N]` math.

## Hard rules (digest)

- **Byte-identity** — mirror every emission change in both engines; re-run the gate.
- **Relative paths only** — no file under `platform/` may hardcode an absolute repo
  path or escape the tree; `tests/run_all.py` enforces this. Anchor on `__file__` /
  `CMAKE_CURRENT_SOURCE_DIR` / `rocke.assets`.
- **Never `ruff check --fix` emitter code** (`core`, `helpers`, `instances`,
  `library/kernels`) — the IR builder is side-effecting, so F841 autofix silently
  changes kernels. Lint with `ruff check` (no `--fix`).
- **Cross-platform** — scripts under the rocke tree are Python, not `.sh`; use
  `tempfile`, `os.cpu_count()`, `pathlib`, `shutil.which` — no `/tmp`, `nproc`, `sudo`.
- **torch is optional** — numpy is the only hard dep; gate torch behind `importorskip`
  in tests and lazy/guarded imports in code.
- **No build artifacts in git** — never commit `build*/`, `_deps/`, `*.a`/`*.o`/`*.so`,
  `*.egg-info/`, `__pycache__/`. Always build out-of-tree on a local filesystem.

Full rationale + build/test/GPU setup + `helpers/` placement + env-flag table:
[`platform/AGENTS.md`](platform/AGENTS.md).

## Definition of Done

A change is not done until it clears [`KERNEL_AUTHORING.md`](KERNEL_AUTHORING.md)
for its change type (new family / new knob / new feature / perf optimization) — the
single doc for the Definition of Done, the process map, and commit/branch
conventions. Build and test references: [`BUILDING.md`](BUILDING.md),
[`TESTING.md`](TESTING.md). Language style:
[`style/PYTHON_STYLE.md`](style/PYTHON_STYLE.md),
[`style/CPP_STYLE.md`](style/CPP_STYLE.md).

## Optimization doc routing — READ the matching doc BEFORE optimizing a kernel

The optimization docs under `platform/dsl_docs/optimization/` are the authoritative
kernel-perf playbook but are **not** auto-loaded (the tree is far larger than the
context window). When a task involves optimizing / profiling / tuning a kernel, open
the matching doc first.

| Task | Doc |
|---|---|
| Optimizing ANY kernel — **start here** (Step 0 exhaustive lever sweep + The Loop are mandatory) | [optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md) |
| Runbook section ⇄ DSL primitive/helper/op mapping (concept → code) | [runbook_compliance.md](platform/dsl_docs/optimization/runbook_compliance.md) |
| Section-by-section primitive table | [runbook_mapping.md](platform/dsl_docs/optimization/runbook_mapping.md) |
| Last validated measurement numbers | [measured_results.md](platform/dsl_docs/optimization/measured_results.md) |
| GEMM perf | [gemm-optimization-rocke.md](platform/dsl_docs/optimization/utilities/skills/gemm-optimization-rocke.md) |
| LDS / bank conflicts | [lds-optimization-rocke.md](platform/dsl_docs/optimization/utilities/skills/lds-optimization-rocke.md) |
| Prefetch / async DRAM→LDS | [prefetch-data-load-rocke.md](platform/dsl_docs/optimization/utilities/skills/prefetch-data-load-rocke.md) |
| ISA / occupancy / resource inspection | [isa-inspection-rocke.md](platform/dsl_docs/optimization/utilities/skills/isa-inspection-rocke.md) |
| Capture a kernel trace (rocprof) | [capture-kernel-trace-rocke.md](platform/dsl_docs/optimization/utilities/skills/capture-kernel-trace-rocke.md) |
| Analyze a kernel trace (ATT) | [kernel-trace-analysis.md](platform/dsl_docs/optimization/utilities/skills/kernel-trace-analysis.md) |
| Launch / benchmark harness | [kernel-launch-guide.md](platform/dsl_docs/optimization/utilities/skills/kernel-launch-guide.md) |
| Bisect a perf regression | [bisect-perf-regression.md](platform/dsl_docs/optimization/utilities/skills/bisect-perf-regression.md) |
| Real bug signatures + measured wins | [empirical-case-studies.md](platform/dsl_docs/optimization/utilities/skills/empirical-case-studies.md) |
| Per-arch facts (MFMA atoms, LDS banks, occupancy caps, sched intrinsics) | [arch/gfx942.md](platform/dsl_docs/optimization/arch/gfx942.md), [arch/gfx950.md](platform/dsl_docs/optimization/arch/gfx950.md) |

Hard rule from the runbook: **never report speed without correctness** — run the
relevant parity/verify harness (attention → `library/builders/**/parity_*.py`;
non-attention → `platform/python/rocke/examples/common/*parity*.py`) and stay within
the runbook §1.4 tolerances before claiming any win. Leave a replayable case study in
the relevant `examples/<arch>/<workload>/` folder.

## Deeper material

- Deep agent onboarding (build/test/GPU, compliance, helpers, env flags): [`platform/AGENTS.md`](platform/AGENTS.md)
- The field manual (IR, lowering, primitives, instances, runtime): [`platform/dsl_docs/README.md`](platform/dsl_docs/README.md)
- New kernel authoring: [`platform/dsl_docs/architecture/authoring_model.md`](platform/dsl_docs/architecture/authoring_model.md)
- Optimization compliance: [`platform/dsl_docs/optimization/runbook_compliance.md`](platform/dsl_docs/optimization/runbook_compliance.md)
