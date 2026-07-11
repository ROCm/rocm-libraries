# rocKE - agent onboarding

rocKE is a **dual-engine rocke kernel stack**: a Python authoring frontend and a
C++ backend that emit **byte-identical AMDGPU LLVM IR**. Read this before editing.

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
```

A change is done only when the gate is GREEN for every family at both flavors.

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
relative-path guard scanning local venv metadata. For now, use `~/rocKE-venv`:

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
2. torch-tensor launch integration (`runtime/torch_module.py`);
3. on-GPU numeric verification against torch eager.

Everything else runs torch-free: IR build, lower, `comgr` compile, numpy launch,
the byte-identity gate, and the CPU test suite (torch-only tests `importorskip`;
GPU tests skip via a torch-free `/dev/kfd` probe). `tests/run_all.py` is GREEN
with neither torch nor a GPU installed.

### ROCm library discovery (libamd_comgr / libamdhip64)

The runtime resolves the ROCm shared libs WITHOUT importing torch
(`runtime/hip_module._candidate_lib_paths` / `_rocm_root_libdirs`), in priority
order:

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
- **Cross-platform**: do not add bash/Linux-specific helper scripts. Scripts
  under `rocke/platform/` are Python, not `.sh`; use `tempfile`, `os.cpu_count()`,
  `pathlib`, `shutil.which` - no `/tmp`, `nproc`, `sudo`, or shell-only flows.
- **Default arch is `gfx950`** (the byte-identity baseline). Do not change the
  codegen default; for on-GPU runs, prefer the local device via
  `rocke.runtime.hip_module.get_device_arch()` and fall back to `gfx950`.
- **torch is optional, never import it from a library to gain a side effect.**
  numpy is the only hard dep. torch is needed only for the fusion frontend,
  torch-tensor launch, and on-GPU torch-eager numeric checks; gate it behind
  `importorskip` in tests and lazy/guarded imports in code. The ROCm-lib
  resolver must keep discovering `comgr`/HIP without importing torch.

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
| `ROCKE_LLVM_FLAVOR` | `llvm20` \| `llvm22` (must match the ROCm `comgr` in use) |
| `ROCM_PATH` / `ROCM_HOME` | ROCm install root; `<root>/lib` is searched for `comgr`/HIP before the globbed `/opt/rocm*` trees |
| `ROCKE_COMGR_LIB` / `ROCKE_HIP_LIB` | explicit full path to `libamd_comgr` / `libamdhip64`; wins over all discovery |
