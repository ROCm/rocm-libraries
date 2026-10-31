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
- **Public hardware, model, and workload references.** Refer to AMD hardware targets by
  device identifier (`gfx942`, `gfx950`, …), not by product or marketing name. Names of
  publicly released models, public benchmarks/datasets/workloads, public operators, and
  their publicly documented architecture, tensor-shape, or configuration parameters may
  appear in code, documentation, and PRs when they explain functional support, test
  coverage, or benchmark scope. The information must be independently verifiable from a
  public source. Never identify or imply a customer, deployment, or private workload; never
  disclose non-public model data, internal codenames, AMD Restricted/Confidential data, or
  unreleased-product details. When public status is uncertain, treat the information as
  confidential and escalate.
- **Public software-performance evidence.** Measured relative results — ratios or percentage
  changes — may appear only in PR descriptions and qualifying benchmark/result locations.
  A qualifying location is either (a) a Markdown file whose case-insensitive basename
  contains `benchmark`, `result`, or `case_study`, and whose title and opening paragraph
  identify its measured scope; or (b) a `Performance` or `Results` section in a
  workload-specific README whose opening identifies its measured scope. In either case,
  recording benchmark methodology or results must be the document's or section's primary
  purpose.
  Every compared kernel arm must pass an on-silicon numeric gate in the measured run:
  outputs and any mutated state against an independent reference, at a declared input
  dtype and tolerance. Byte-identity, compilation, or successful launch alone is not a
  correctness gate. State the result, workload and relevant axes, baseline implementation
  and version, hardware target, measurement method, reproducible provenance, and material
  caveats. Compare equivalent work or label the mismatch; label diagnostic and
  non-production configurations; and use bounded statements such as "faster at these
  measured configurations", never a general superiority claim.
  **Do not add or update absolute software-achieved benchmark or performance results** in
  public artifacts, including but not limited to latency, throughput, TFLOP/s, MFU, and
  bandwidth. Store them only in an approved, access-controlled internal record. This
  prospective rule supersedes conflicting legacy guidance. Existing public records with
  absolute values are frozen migration debt, not precedent: do not extend, update, or copy
  their measurements; migrate them only in a separately reviewed remediation. Source
  comments and commit messages remain qualitative because measured claims there become
  stale and are difficult to review in context.
- **No legal claims or unsupported marketing claims.** Never make legal claims about AMD or
  other products/software. Do not make unsupported marketing, general superiority,
- **No legal claims or unsupported marketing claims.** Never make legal claims about AMD or
  other products/software. Do not make unsupported marketing, general superiority, performance,
  rule above are permitted; marketing language is not. **Protect AMD.**
- **No internal links** (Jira/Confluence/Perforce) in committed/public artifacts;
  external Git issue links are OK.

**Runbooks & playbooks are encouraged** and may fully document algorithms, iteration
methodology, and knobs/levers with their qualitative effects. Qualifying benchmark/result
documents, as defined above, may also contain relative evidence that meets the policy. Keep
all absolute software measurements and any restricted data in the approved access-controlled
record.

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
    CMakeLists.txt       # stable C++ test-registration orchestrator
    cmake/               # concern-owned core / portable-IR / instance registrations
  tools/                # check_byte_identity.py and other cross-platform entry points
  cmake/  CMakeLists.txt  pyproject.toml
```

## The #1 invariant: byte-identity

The Python engine (`core/lower_llvm.py`) and the C++ engine (`cpp/`) MUST emit the
**same LLVM-IR bytes** for every kernel family. Any op/instance/atom/fusion/arch
change must be made in **both** engines in the same change. Prove it:

```bash
ROCKE_LLVM_FLAVOR=llvm20 python tools/check_byte_identity.py   # build engine fresh + gate
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # and llvm22
ROCKE_LLVM_FLAVOR=llvm23 python tools/check_byte_identity.py   # and llvm23
ROCKE_LLVM_FLAVOR=llvm24 python tools/check_byte_identity.py   # and llvm24
```

A change is done only when the gate is GREEN for every family at every flavor.
The llvm20 and llvm22 runs also re-run the `gfx1250_lowering` family at llvm23,
because its gfx1250-only ops need the LLVM23-generation emission profile.

## Build / test / run

```bash
# PYTHONPATH for the Python package
export PYTHONPATH=<rocke/platform>/python                 # then: import rocke

# build the C++ engine + the C++ unit-test binaries (so ctest has something to run)
cmake -S <rocke/platform> -B /tmp/rocke -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/rocke -j

# Complete developer pre-merge gate. It configures/builds the default build root,
# runs host/corpus gates and pytest, then runs all registered CTests, including
# toolchain integrations. Use --build-root to reuse a chosen out-of-tree build.
python tests/run_all.py
python tests/run_all.py --build-root /tmp/rocke
python -m pytest tests/instances/test_rocke_multiarch.py   # a fast CPU-only suite
```

C++ test registration is deliberately distributed to reduce PR contention.
Keep `tests/CMakeLists.txt` as a stable orchestrator: add or change registrations
in `tests/cmake/CoreTests.cmake`, `PortableIrTests.cmake`, or
`InstanceTests.cmake`. `TestCategoryLabels.cmake` must remain last because it
labels tests after registration and must execute in the same CMake directory
scope. Use `include()`, not `add_subdirectory()`: with the project's pre-3.28
CMake minimum, test properties can only be changed from the directory that
called `add_test()`. Installed Python and library CTest entries still belong in
`platform/CMakeLists.txt`. The ownership map and inventory-comparison recipe are
in [`tests/README.md`](tests/README.md#cmake-registration-ownership).

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

COMGR and HIP use separate discovery policies. Neither loader imports torch
or initializes the SDK to discover a library.

COMGR has exactly two choices:

1. A nonempty `ROCKE_COMGR_LIB` selects that explicit library. A missing or
   unloadable override fails; it does not fall back to another compiler.
2. Otherwise, the installed `rocm-sdk-core` distribution's file metadata
   selects its COMGR payload. This supports both `_rocm_sdk_core` and package
   names carrying TheRock's suffix nonce. The optional `rocke[sdk-core]` extra
   installs it from an AMD wheel index; see
   [compiler installation and release support](dsl_docs/runtime/comgr_and_hipmodule.md#installing-the-compiler).

COMGR does not automatically select torch libraries, mapped foreign compilers,
`ROCM_PATH`, `ROCM_HOME`, `HIP_PATH`, system ROCm trees, or bare library names.
Python and native C++ use the same env-or-SDK policy. Native standalone AUTO
finds SDK distribution metadata under the installed engine or executable's
installation prefix. Build-tree binaries also read an SDK site descriptor generated at the top-level
build root by CMake for its selected Python interpreter. It uses a relative site
path when possible and a marked absolute site path for cross-drive Windows
builds; regenerate absolute descriptors after moving the SDK. The descriptor
stays in the build tree and is never installed. Deployments outside those locations
must use `ROCKE_COMGR_LIB`. Explicit offline IR emission still requires no compiler.
The compatibility hook `comgr.prefer_bundled_lib()` uses this resolver without
importing torch. Successful loads are retained; the loaded compiler's LLVM
version determines AUTO, while package release metadata is informational.

Before loading COMGR on Linux, reject any mapped COMGR that is a different
file from the selected candidate. An alias of the same file is accepted.
The ordinary shared SDK used by TheRock torch and rocKE is therefore one
compiler. A distinct mapped compiler is a conflict, not a fallback candidate;
compile in a separate process when using another compiler.

HIP keeps its existing runtime policy: `ROCKE_HIP_LIB`, then the copy used by
already imported torch (an unambiguous mapped copy, `torch/lib`, or the active
TheRock SDK), then system ROCm roots and the bare library name. Linux does not
add an independent SDK HIP candidate; Windows also discovers SDK runtime DLLs
directly. When several distinct HIP copies are mapped, prefer the sole copy
owned by imported torch or its active SDK rather than address order.

If COMGR is unavailable in a torch-free process, install `rocke[sdk-core]`
from an AMD wheel index or point `ROCKE_COMGR_LIB` at the intended compiler.

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

## Tiling primitives (`helpers/tiling/`)

A human-approachable, ND tiling + MMA primitives layer (`import rocke.helpers.tiling`). Reach for
it on **tiling problems**: MMA/GEMM-shaped kernels, wave-tile subtiling, custom thread/register
distributions, ragged/partial tiles (clipping), interleaved vs canonical operand layouts, and
target-agnostic MMA authoring — where you'd otherwise hand-write raw `TileDistributionEncoding`
integers.

**Before writing tiling code, read the docs** (they are the source of truth for what exists and
how to drive it):
- `helpers/tiling/docs/tiling_api_surface.md` — the **how-to-use catalog**: every surface, its
  default (MMA-driven) mode AND its manual override, a composability matrix, and runnable examples.
  Read this first to decide whether the problem is solvable with what's built.
- `helpers/tiling/docs/tiling_api_contract.md` — the **contract / why**: the audience tiers
  (front door / toolbox / machinery), the glass-box principles, and how a custom override is
  validated identically to a derived one.
- `helpers/tiling/docs/tiling_interleaving_design.md` — the **interleaved-layout SOT**: the recipe,
  which atoms interleave (every dense square atom; the proof sweep is the census), and the
  `InterleavedStyle` API.

**Workflow for a tiling problem:**
1. Read the API surface + contract docs; map the problem onto the built surface (dense MMA GEMM,
   wave-tile subtiling with `order`, clipping/bounds, canonical vs interleaved operand layout,
   arbitrary distributions via `make_tile_desc`, RCR + other data layouts). If a needed feature is
   only `RESERVED`/`PLANNED` in the composability matrix, say so rather than inventing it.
2. Author with the front door: `TileMma` resolves the intrinsic and **processes the whole wave tile
   (owns the M×N×K subtile grid + iteration + `order`)**, taking an optional `style=` (default
   `CanonicalStyle`; `InterleavedStyle` for the wide-coalesced interleaved layout);
   `make_tensor_desc`/`make_window` give ptr-free memory + auto-clipping; `make_fragment` + the
   b-first verbs `load_fragment` / `store_fragment` / `fill_fragment` move data; `make_tile_desc(...)`
   authors any custom distribution as a quantity-major geometric table (no raw encoding integers).
3. **Turn knobs / customize freely:** atom selection (shape tuple or intrinsic name), wave-tile
   size, subtile `order`, operand `style=`, dtypes, target, clip bounds, and fully custom
   distributions — all are data, composable per the matrix. The recorder + the `analysis/` gates
   (`verify_lds_roundtrip` / `verify_mma_soundness`) verify a recorded kernel; reflection
   (`describe`, `render_forward_map`, `render_inverse_map`) shows what any default resolved to.
4. Verify against a numpy golden reference (the demos are the template; torch is not used).

Tests: `PYTHONPATH=platform/python <venv>/python -m pytest platform/python/rocke/tests/helpers/tiling/ -q`.
Demos live under `helpers/tiling/kernels/` (e.g. `tiling_gemm_interleaved_demo.py`,
`tiling_gemm_manual_demo.py`). Built + bit-exact on gfx90a today; interleaved layouts are BUILT for
dense square atoms (`InterleavedStyle`); `c_transpose`, sparse, and MX remain reserved/planned seams.

## helpers/ placement

This section is the one home of the helpers rule; other docs point here.

**Default: reuse, write, fix, all in the kernel's PR.** The `build_*()` body
names what is special about the op (spec, tile shape, atom, descriptor) and
calls helpers for the rest. Raw `IRBuilder` calls in the kernel body are for
short, one-off glue between helper calls.

1. **Use what is there.** Check `helpers/` and the family modules
   (`_<family>_*.py`) before writing new IR. Extend a helper rather than copy it.
2. **Write what is missing, with a test.** IR that repeats, or hides index
   math, lane maps, layout, masking, or a pipeline stage, becomes a named
   helper in the same PR. A brand-new family, a single kernel, or addressing
   specific to one op's layout is not a reason to stay in raw IR: the helper
   goes in that family's module even if nothing else uses it yet. A host-only
   helper gets a unit test (e.g. `tests/test_helpers_spec.py`,
   `library/tests/test_attention_bwd_shared_helpers.py`); an SSA helper is
   covered by the parity pair, golden, and numeric test of the kernel that
   uses it.
3. **Fix what is broken, with a regression test.** If a helper is wrong or
   does not fit, fix the helper, not a workaround in the kernel. Add a test
   that fails without the fix. A fix that changes emission is mirrored in C++
   and re-blesses the affected goldens (KERNEL_AUTHORING.md Process F).

### Where a new helper goes

| Scope | Home |
|---|---|
| One family today | A family helper module next to the kernel: `instances/common/_<family>_*.py`, or `library/kernels/common/_<family>_*.py` for a library kernel |
| ≥2 families, a general emitter/primitive/pipeline (e.g. `SoftwarePipeline`, `CoalescedTileLoader`, `mfma_gemm_inner`), a CK-Tile-parity primitive, or a guard against a class of silent bugs (lane maps, barriers, pipelining) | `helpers/` |

Move a family helper into `helpers/` when a second family needs it
(KERNEL_AUTHORING.md Process G). Launch orchestration and multi-kernel
pipelines stay in the instance.

### The C++ mirror is not a reason to inline

Every emission change is mirrored in the C++ engine anyway (byte-identity), so
inlined raw IR is mirrored too, once per copy. A helper is mirrored once, as one
function (e.g. `library/kernels/common/_fmha_common.py` and
`cpp/instances/common/_fmha_common.cpp`). If it emits SSA: Python + C++ mirror +
byte-identity in the same PR, the same rule the kernel already follows.

### Do NOT

- Write a helper that only renames one builder call. A helper earns its place
  by hiding detail that repeats or that a reader should not have to re-derive.
- Put instance-specific specs, manifests, or dispatch heuristics in `helpers/`.
- Add an SSA emitter to `helpers/` without its C++ port in the same PR.

## Key env flags

| flag | meaning |
|---|---|
| `ROCKE_ARCH_DOMAIN` | `error` \| `warn` (default) \| `off` — what both engines do when a kernel demands an intrinsic the committed arch-domain table says the target cannot lower. `error` refuses the lowering (`ArchDomainError`) unless the cell is waived in `arch_domain_waivers.json`; `warn` reports it; `off` skips the lookup. Any other value raises rather than falling back to the default, because a misspelling is always an attempt to change what the lane does. A wrong measurement is waived per cell, not worked around with this variable; see TESTING.md, the arch-domain gate |
| `ROCKE_BACKEND` | `cpp` (default) \| `python` \| `both` (differential assert) |
| `ROCKE_CPP_STRICT` | `1` = raise instead of silently falling back to Python when `rocke_engine` isn't built |
| `ROCKE_DEBUG_LOC` | `1` = capture Python source locations while building and emit DWARF, so ATT traces map to source; off by default because it changes the emitted `.ll` |
| `ROCKE_LLVM_FLAVOR` | `llvm20` \| `llvm22` \| `llvm23` \| `llvm24` (AUTO queries loaded COMGR: LLVM <=20, 21/22, 23, or 24; newer majors and unavailable evidence raise; explicit lanes support offline emission) |
| `ROCKE_TEST_VERIFY_IR` | `1` = assemble emitted IR with `llvm-as` in `TestNewTargetIntrinsics`; needs an LLVM as new as the newest flavor |
| `ROCM_PATH` / `ROCM_HOME` | ROCm install root for HIP discovery; COMGR uses only the explicit override or the installed SDK |
| `ROCKE_COMGR_LIB` / `ROCKE_HIP_LIB` | explicit full path to `libamd_comgr` / `libamdhip64`; wins over all discovery |
| `ROCKE_ONLINE_LIB` / `ROCKE_REPLAY_CLI` | explicit portable-IR shared engine / standalone replay CLI; explicit paths never fall back |
| `ROCKE_PROVENANCE_MANIFEST` | build-generated expected version/build ID for validating explicit native artifacts in a source-less installed test tree |
