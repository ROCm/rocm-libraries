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
kernels  ->  dispatch  ->  builders  ->  benchmarks
```

`dispatch` is pure selection policy over `kernels` and imports nothing else. A
builder is a host-side harness, and a harness that measures anything other than
the spec dispatch ships is a decoration rather than a gate — so `builders`
consuming `dispatch.<arch>.*_spec_for_request` is a permanent one-way need, and
`dispatch` importing `builders` is the direction that would be a cycle.

Skipping a layer downward is fine (`builders` imports `kernels` directly);
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

## Optimization doc routing — query the wiki, do not dump the runbook

The playbook is `platform/dsl_docs/optimization/` (skill root: [`SKILL.md`](platform/dsl_docs/optimization/SKILL.md)).
It is **not** auto-loaded. Route operator family × architecture to a technique
with the query scripts; do **not** read `optimization_runbook.md` linearly.

From `platform/dsl_docs/optimization/`:

```bash
python3 scripts/query.py --operator gemm --family cdna
python3 scripts/query.py --symptom lds-stall --architecture gfx950
python3 scripts/get_page.py family-overview
```

| Task | Page / tool |
|---|---|
| Start here — routing tables | [family-overview](platform/dsl_docs/optimization/wiki/families/overview.md) |
| Step 0 + one-lever loop | [process-optimization-loop](platform/dsl_docs/optimization/wiki/process/optimization-loop.md) |
| Catalog exhausted — invent a new mapping | [process-escape-hatch](platform/dsl_docs/optimization/wiki/process/escape-hatch.md) |
| gfx1250 (KernelWiki-shaped analog) | [hw-gfx1250](platform/dsl_docs/optimization/wiki/hardware/gfx1250.md) + `wiki/migration/` |
| Probe order | [process-probe-sequence](platform/dsl_docs/optimization/wiki/process/probe-sequence.md) |
| GEMM / attention / conv / MoE / small-ops tables | `wiki/families/*.md` |
| Common vs arch-specific techniques | `wiki/techniques/common/`, `cdna/`, `rdna/`, `gfx1250/` |
| gfx facts | `wiki/hardware/` (appendix: `arch/gfx942.md`, `arch/gfx950.md`) |
| hipBLASLt, Tensile, stinkytofu, hipDNN, CK | `sources/projects/` |
| Long-form checklist (appendix only) | [optimization_runbook.md](platform/dsl_docs/optimization/optimization_runbook.md) |
| ATT capture / WaveScope | [capture-kernel-trace-rocke.md](platform/dsl_docs/optimization/utilities/skills/capture-kernel-trace-rocke.md) |
| ISA probes | [isa-inspection-rocke.md](platform/dsl_docs/optimization/utilities/skills/isa-inspection-rocke.md) |

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
