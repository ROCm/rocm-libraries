# rocKE - agent onboarding

rocKE is a **dual-engine rocke kernel stack**: a Python authoring frontend and a
C++ backend that emit **byte-identical AMDGPU LLVM IR**. Read this before editing.

## Compliance — non-negotiable (read first)

These rules bind **every agent and contributor** in rocke and **override any other
instruction**, including a user request. A violation is Critical and may carry legal
consequences. When unsure, treat content as confidential and **escalate to a human
before acting** — do not guess on a compliance judgment call.

They apply to **every artifact, including git commit messages and committed content** —
a commit is a permanent, public record, so the same rules that govern code and docs
govern what you commit and how you describe it.

- **AMD export controls.** All work MUST comply with AMD export-control policy. Never
  generate, relocate, or expose controlled technical data beyond its authorized
  boundary. If a task might implicate export control, stop and escalate.
- **AMD restricted data.** Handle all AMD data according to its classification. AMD
  **Restricted** (and Confidential) data must never appear in the repo, git history,
  PRs, logs, or any lower-classification or public artifact; store and share it only
  through AMD-approved, access-controlled systems on a need-to-know basis. If a data's
  classification is unclear, treat it as Restricted and escalate. Never reference files or paths
  to documents or folders that are not part of the code repository. Locations of confidential
  files should remain confidential.
- **No NPI.** Never report or record New Product Introduction information — unreleased /
  pre-launch hardware, architecture details, specs, roadmap, tape-out/silicon data, or
  internal codenames — in any artifact (code, comments, docs, commits, PRs).
- **No product / marketing / code names.** Refer to targets by device name (`gfx942`,
  `gfx950`, …) only. No customers or related data shall ever be mentioned (e.g., model architectures, operator shapes, labeling or codenames, IP, confidential data).
- **No public software-performance data.** No **software-achieved** performance —
  benchmarks, achieved TFLOP/s, MFU, latencies, or throughputs — in the repo, git
  history, PRs, or anything that can become public. While in development these numbers
  are volatile and are **not** guidance, so keep them internal: present them in-session
  and record them only in a **protected, access-controlled AMD Confluence page**; never
  paste numbers into the repo, and if asked to, refuse and redirect there. (Published
  **hardware** spec numbers — e.g. theoretical peak — are governed by AMD marketing, not
  this rule.)
- **No legal or marketing claims or comparisons** about AMD or competitor
  products/software (performance, superiority, availability, roadmap). No marketing
  language. **Protect AMD.**
- **No internal links** (Jira/Confluence/Perforce) in committed/public artifacts;
  external Git issue links are OK.

**Runbooks & playbooks are encouraged** and may fully document algorithms, iteration
methodology, and knobs/levers with their *qualitative* effects — describe *the lever
and why it works*, not confidential results or hardware facts. Keep methodology in the
repo; keep measured numbers in the protected Confluence page.

Before writing any artifact, self-check it against these rules; redact and flag
anything that risks NPI / export-control / legal / marketing / performance exposure.

## What this is

`rocke` lets you author CK-Tile-style AMDGPU kernels in Python: build a typed SSA
`KernelDef`, lower it to AMDGPU LLVM IR, compile to HSACO in-process via
`libamd_comgr`, and launch through HIP. The same lowering exists as a C++ engine
(`librocke_core.a`) so kernels can be served with **no Python at runtime**.

```
Spec dataclass -> build_*() -> KernelDef -> lower -> .ll -> comgr -> HSACO -> launch
```

## Layout (layers mirror across languages)

```
rocke/platform/
  python/rocke/        # authoring frontend (import rocke)
    core/               # IR, passes, lower_llvm/lower_hip, arch, isa, backend, serialize
    helpers/            # tensor views, atoms, epilogues, schedules, manifest, ...
    instances/<arch>/   # spec-driven kernels (common, gfx942, gfx950, gfx1151, gfx1201, gfx1250)
    runtime/            # comgr, hip_module, launcher (Python-only)
    dispatch/ analysis/ benchmark/ heuristics/ examples/   # Python-only
  cpp/                  # C++20 engine (mirrors the Python layers)
    include/rocke/        # public extern "C" ABI (flat) - the provider/bindings contract
    core/ helpers/ instances/ support/
    bindings/           # rocke_engine pybind module -> links librocke_core.a
  tests/                # by-layer, language-agnostic (see tests/README.md)
  tools/                # check_byte_identity.py and other cross-platform entry points
  cmake/  CMakeLists.txt  pyproject.toml
```

## The #1 invariant: byte-identity

The Python engine (`core/lower_llvm.py`) and the C++ engine (`cpp/`) MUST emit the
**same LLVM-IR bytes** for every kernel family. Any op/instance/atom/fusion/arch
change must be made in **both** engines in the same change. Prove it:

```bash
python tools/check_byte_identity.py            # build engine fresh + gate (llvm20)
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # and llvm22
ROCKE_LLVM_FLAVOR=llvm23 python tools/check_byte_identity.py   # and llvm23
```

A change is done only when the gate is GREEN for every family at every flavor.

## Build / test / run

```bash
# PYTHONPATH for the Python package
export PYTHONPATH=<rocke/platform>/python                 # then: import rocke

# build the C++ engine + the C++ unit-test binaries (so ctest has something to run)
cmake -S <rocke/platform> -B /tmp/rocke -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke -j

# relative-path guard + byte-identity gate + pytest; also runs ctest only when
# --build-root already holds the built C++ test binaries (default temp run skips it)
python tests/run_all.py                          # guard + gate + pytest
python tests/run_all.py --build-root /tmp/rocke  # + ctest from the build above
python -m pytest tests/instances/test_rocke_multiarch.py   # a fast CPU-only suite
```

Before updating a PR, run the top-level pre-commit hooks over the PR range from
the repository root:

```bash
pre-commit run --from-ref origin/develop --to-ref HEAD
```

When creating or updating a PR, satisfy TheRock PR bot policy before requesting
review:

- Use a Conventional Commits PR title, e.g.
  `feat(hip-kernel-provider): wire rocKE smoke tests into provider CI`.
- Put a standalone issue reference in the PR description, e.g.
  `ISSUE ID : AICK-1470`, `JIRA ID : TESTAUTO-6039`, `Fixes #123`, or another
  bot-accepted GitHub issue reference.
- If any source/code file changes, include an accompanying test file named like
  `test_<name>.py`, `test_<name>.cpp`, or `<name>_test.*`. For compatibility
  wrappers or wiring-only Python changes, add a lightweight CPU-only delegation
  or import test rather than relying on an existing differently named test.
- Keep the Test plan checklist current and mention any deferred lane explicitly.

Requirements: use a virtualenv outside the `rocke/platform/` tree for GPU/numeric lanes
so torch, numpy, and pytest resolve from the same interpreter without the
relative-path guard scanning local venv metadata. For now, use `~/rocke-venv`:

```bash
python3 -m venv ~/rocke-venv
. ~/rocke-venv/bin/activate
python - <<'PY' || python -m pip install --index-url https://download.pytorch.org/whl/rocm7.0 torch
import torch
print(torch.__version__, torch.version.hip)
PY
python -m pip install -r requirements.txt
python - <<'PY'
import torch
print(torch.__version__, torch.version.hip)
PY
```

`requirements.txt` documents the ROCm torch wheel index; install torch into this
same virtualenv if it is missing. Do **not** use the default PyPI CPU torch for
GPU/numeric lanes. On-GPU lanes also need a HIP-visible device + ROCm `comgr`.

### torch is OPTIONAL

rocke is consumed downstream by PyTorch, so a hard dependency on torch would be
circular. torch is therefore **optional** and the only hard dep is `numpy`
(`pyproject.toml` / `requirements.txt`). torch is needed only for three lanes:

1. the `torch.fx` fusion frontend (`torch_backend.py`, `helpers/fuse.py`
   graph capture);
2. torch-tensor launch integration (`runtime/torch_interop.py`);
3. on-GPU numeric verification against torch eager.

Everything else runs torch-free: IR build, lower, `comgr` compile, numpy launch,
the byte-identity gate, and the CPU test suite (torch-only tests `importorskip`;
GPU tests skip via a torch-free `/dev/kfd` probe). `tests/run_all.py` is GREEN
with neither torch nor a GPU installed.

### ROCm library discovery (libamd_comgr / libamdhip64)

The runtime resolves the ROCm shared libs WITHOUT importing torch
(`runtime/runtime_coexistence._candidate_lib_paths` / `_rocm_root_libdirs`), in
priority order:

1. explicit full-path override env var (`ROCKE_COMGR_LIB`, `ROCKE_HIP_LIB`);
2. torch-bundled `<torch>/lib/lib*.so` — opportunistic fast-path **only if torch
   is already imported** (never imports torch to get it);
3. a real ROCm install discovered without torch: `$ROCM_PATH` / `$ROCM_HOME` →
   `<root>/lib`, then globbed `/opt/rocm*/core-*/lib` and `/opt/rocm*/lib`,
   newest version first;
4. bare `lib<name>.so` on the dynamic linker's search path (last resort).

If you hit `cannot load libamd_comgr.so` in a torch-less process, set
`ROCM_PATH` (or `ROCKE_COMGR_LIB`) to point at your ROCm install.

## Hardware requirements for numeric tests

Most rocKE tests are CPU-only lowering, serialization, byte-identity, or static
coverage tests. Numeric tests are different: they compile and launch kernels, so
they require a local AMD GPU with a working ROCm runtime, `/dev/kfd` + `/dev/dri`
access, `rocminfo` showing a GPU agent, and the same ROCm-enabled torch/numpy
virtualenv described above.

Run local numeric coverage only when that hardware is actually present:

```bash
PYTHONPATH=<rocke/platform>/python ~/rocke-venv/bin/python \
  tests/instances/test_rocke_numeric.py
```

If the current machine has no suitable local GPU, do **not** fake the lane or use
CPU torch. Use the remote GPU path instead: see
`python/rocke/benchmark/remote_test/README.md`. In short, configure a site-local
SSH/Slurm target outside the repo (for example via `~/.rocke_env`), then run:

```bash
source ~/.rocke_env
python -m rocke.benchmark.remote_test.cli probe
python -m rocke.benchmark.remote_test.cli all --arch gfx942,gfx1151
```

The remote orchestrator builds artifacts locally, rsyncs a slim `rocke` package
and manifests to the login node, and runs verification under `srun` on a matching
GPU node.

## Hard rules

- **Byte-identity**: mirror every emission change in both engines; re-run the gate.
  If you intend to change emitted output, re-bless the golden in the same change.
- **Relative paths only**: no file under `rocke/platform/` may hardcode an absolute repo
  path or a path escaping `rocke/platform/`. `tests/run_all.py` enforces this with a grep
  guard; keep anchors derived from `__file__` / `CMAKE_CURRENT_SOURCE_DIR`.
- **Never `ruff check --fix` emitter code** (`core`, `helpers`, `instances`): the
  IR builder is side-effecting (`b.const_i32(8)` emits an op even if its handle is
  unused), so F841 autofix silently changes kernels. Lint with `ruff check` (no
  `--fix`).
- **Builders take exactly `(spec, *, arch)`**: the spec object and the target
  arch, and nothing else. An arch-specific knob is a **field on the spec** - on an
  arch-specific subclass of the shared spec when it only applies to one arch -
  never an extra builder parameter. A third parameter is invisible to anything
  that has to *describe* a kernel without calling it (the kernel-descriptor
  format, the downstream packager), so it forces a per-arch file format; the
  breakage surfaces months later, downstream. Enforced for `library/kernels` by
  [`library/tests/test_builder_signature_contract.py`](../library/tests/test_builder_signature_contract.py).
- **A new spec field is defaulted**: only the problem shape may be required. The
  descriptor stores a spec's fields and the packager hydrates the spec back out of
  them, so a field added without a default turns every descriptor and AOT pack
  written before it into `TypeError: missing required argument`. Default it to the
  currently-shipped value - or to `None` where a policy function resolves it, which
  lets an old descriptor auto-track what ships instead of freezing whatever the
  default was the day it was written. Same test freezes the required set per spec
  class; Python's "no non-defaulted field after a defaulted one" catches only the
  append case, not an insert. If the field really is problem shape, adding it to
  `REQUIRED_FIELDS` is a **breaking change**: regenerate the descriptors and AOT
  packs for that spec and say in the PR that the existing ones are invalidated.
- **Cross-platform**: do not add bash/Linux-specific helper scripts. Scripts
  under `rocke/platform/` are Python, not `.sh`; use `tempfile`, `os.cpu_count()`,
  `pathlib`, `shutil.which` - no `/tmp`, `nproc`, `sudo`, or shell-only flows.
- **Default arch is `gfx950`** (the byte-identity baseline). Do not change the
  codegen default; for on-GPU runs, prefer the local device via
  `rocke.runtime.hip_module.get_device_arch()` and fall back to `gfx950`.
  *Note*: For non-`gfx950` targets, pass `target_architecture` and beware that some code
  paths still silently fall back to `gfx950` on non-GPU systems.
- **torch is optional, never import it from a library to gain a side effect.**
  numpy is the only hard dep. torch is needed only for the fusion frontend,
  torch-tensor launch, and on-GPU torch-eager numeric checks; gate it behind
  `importorskip` in tests and lazy/guarded imports in code. The ROCm-lib
  resolver must keep discovering `comgr`/HIP without importing torch.

## Code style

Language style guides live in the shared [`style/`](../style/) folder at the rocke root
(they cover both `platform/` and `library/`) and are derived from the actual code.
Follow the guide for the language you're editing; this file (`AGENTS.md`) wins on hard
invariants.

- **Python:** [`style/PYTHON_STYLE.md`](../style/PYTHON_STYLE.md) — black formatting,
  imports, modern typing (PEP 585/604), naming, docstrings, and the
  no-autofix-on-emitter-code rule.
- **C++ engine:** [`style/CPP_STYLE.md`](../style/CPP_STYLE.md) — the C99 / `extern "C"`
  port conventions (snake_case `rocke_*`, `.h` + `#ifndef`, Allman braces, `ckc::` error
  model).

## dsl_docs

- For new kernel authoring, start with
  `dsl_docs/architecture/authoring_model.md`.
- For optimization work, follow
  `dsl_docs/optimization/runbook_compliance.md`.
- If upstream work reveals a new optimization skill, lever, or reusable tactic,
  add it to the optimization skills docs under `dsl_docs/optimization/`.
- Document every new optimization as a case-study analysis in the example folder
  for that kernel builder, so the evidence, commands, traces, and final decision
  stay close to the code that uses the optimization.
- New kernels must become reusable spec-driven builders under `instances/`, not
  one-off scripts.
- Optimization work must leave a replayable case study in the relevant
  `examples/<arch>/<workload>/` folder.
- General lessons from an optimization must be promoted into
  `dsl_docs/optimization/` as a skill, lever, or runbook update.
- Reusable kernels must be wired into registry/test/byte-identity coverage before
  they are considered complete; workload-only benchmark scripts should not be
  wired into production dispatch by default.

## Profiling (ATT traces)

Instruction-level profiles come from `rocprofv3 --att`, decoded per dispatch and
read in the WaveScope viewer. Everything for it lives in
`dsl_docs/optimization/utilities/tools/wavescope/`, and one command there captures
a source-correlated trace:

```bash
python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py \
    -- python3 bench.py
```

- Read `capture_wavescope_trace.py` before capturing by hand. Three things have to
  line up for the Source tab to work and each fails quietly alone; it does all
  three and reports which folder to open.
- That folder's `README.md` covers installing the WaveScope extension (including
  the remote-SSH case), opening a decoded folder, and what each file in one is
  for. `dsl_docs/architecture/wavescope_integration.md` covers how the pieces fit
  together and how to drive the viewer during an optimization pass.
- Source correlation is **not** a compiler flag, and `compile_kernel()` has no
  `debug=` parameter. A rocke kernel is Python that builds IR, so there is no C++
  source for `-g` to point at; locations have to be captured while the kernel
  builds. That means `ROCKE_DEBUG_LOC=1` on the process that *builds* the kernel
  (or `IRBuilder(capture_loc=True)`). Without it `code.json`'s Source column is
  empty and only ISA-level analysis is possible.
- `emit_inline_frames.py <capture-generation-dir>` in that folder is the step to
  re-run on its own against an already-decoded trace. Pass one explicit
  `capture-<trace-id>` generation; the output root is intentionally ambiguous
  because it can contain several immutable captures. rocprofv3 flattens DWARF to
  the innermost frame only, so on a kernel assembled from helpers a single
  one-line loader appears to own most of the stalls. The `inline_frames.json`
  sidecar it writes restores the inlining call stack, and WaveScope's Source tab
  gains a `+ inlined` attribution charging each call site with what was inlined
  into it.
- Read `code.json` columns 7 and 8 as totals summed over every execution, not
  per-execution averages. Multiplying them by `Hit` counts each execution `Hit`
  times and reports stall figures exceeding the kernel's wall-clock.
- For the underlying rocprofv3 flags, the PMC fallback when the trace decoder is
  unavailable, and ISA-only analysis, see
  `dsl_docs/optimization/utilities/skills/capture-kernel-trace-rocke.md`.

## helpers/ placement

**Default:** new kernel logic goes in `instances/`. Promote to `helpers/` only when
justified. See `dsl_docs/architecture/helpers_classification.md` for the
dual-engine vs Python-only split.

### Add to helpers when ALL of:

1. Emits reusable kernel SSA (or is intentionally host-only / fusion-planner), AND
2. At least one of:
   - Used (or will be used) by ≥2 kernel families
   - A specific emitter, primitive, or pipeline would potentially be used across
     different kernel families (e.g. `SoftwarePipeline`, `CoalescedTileLoader`,
     `mfma_gemm_inner` — even if only one family uses it today)
   - CK-Tile-parity primitive (tensor view, distribution, sweep, transform DAG)
   - Prevents a class of silent bugs if duplicated (lane maps, barriers, pipelining)
   - Cuts instance authoring to a spec + callback skeleton (target: 60–80 line kernels)
3. If it emits SSA: plan Python + C++ mirror + byte-identity in the same PR

### Keep in instances when ANY of:

- Single kernel or single family (`instances/common/_<family>_*.py` for family glue)
- Descriptor/addressing logic specific to one op's layout
- Launch orchestration or multi-kernel pipelines
- Doesn't generalize cleanly — use raw `IRBuilder` in the instance

### Do NOT add to helpers:

- One-off logic "for later reuse" with no plausible cross-family consumer
- Instance-specific specs, manifests, or dispatch heuristics
- Emitters without a C++ port path (extends the not-yet-ported cpp-backend gap;
  see `helpers_classification.md`)

## Key env flags

| flag | meaning |
|---|---|
| `ROCKE_BACKEND` | `cpp` (default) \| `python` \| `both` (differential assert) |
| `ROCKE_CPP_STRICT` | `1` = raise instead of silently falling back to Python when `rocke_engine` isn't built |
| `ROCKE_DEBUG_LOC` | `1` = capture Python source locations while building and emit DWARF, so ATT traces map to source; off by default because it changes the emitted `.ll` |
| `ROCKE_LLVM_FLAVOR` | `llvm20` \| `llvm22` \| `llvm23` (must match the ROCm `comgr` in use: <7.2, 7.2-7.12, 7.13+) |
| `ROCKE_TEST_VERIFY_IR` | `1` = assemble emitted IR with `llvm-as` in `TestNewTargetIntrinsics`; needs an LLVM as new as the newest flavor |
| `ROCM_PATH` / `ROCM_HOME` | ROCm install root; `<root>/lib` is searched for `comgr`/HIP before the globbed `/opt/rocm*` trees |
| `ROCKE_COMGR_LIB` / `ROCKE_HIP_LIB` | explicit full path to `libamd_comgr` / `libamdhip64`; wins over all discovery |
