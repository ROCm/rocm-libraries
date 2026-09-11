# LOCAL_DEV — build every moving part from scratch and run `hipdnn_torch` locally

> [!NOTE]
> [`README.md`](README.md) explains how to *use* the package once the runtime is wired.
> **This file is the from-scratch developer runbook**: how to build the three pieces the
> package sits on top of, wire them together on a dev box, and prove that ops actually
> route through hipDNN (and track the ones that don't). It was written and validated on a
> gfx1151 laptop under WSL2; the caveats below are the ones that actually bit during that
> bring-up, not hypotheticals.

If you just want to run against artifacts someone already built for you, skip to
[§5 Wire the environment](#5-wire-the-environment) and [§6 Verify + prove it's
hipDNN](#6-verify--prove-its-hipdnn).

## 1. What you are assembling

`hipdnn_torch` is pure Python glue. To make a single `F.linear` land on a hipDNN kernel,
**four** independently-built things have to agree at runtime:

| Piece | Where it comes from | Built how |
|-------|--------------------|-----------|
| **PyTorch + ROCm SDK base** | a TheRock ROCm nightly `pip` wheel set (torch + bundled `_rocm_sdk_libraries_gfx1151`) | `pip install` into a venv |
| **Engine plugin** (`libhip_kernel_provider.so` + the AOT `.co`/`family.json` catalog) | `hip-kernel-provider`, on the **catalog branch** | superbuild preset |
| **Frontend bindings** (`hipdnn_frontend_python.abi3.so`) | `projects/hipdnn/python/frontend_bindings`, on **PR #10600's branch** (`users/sareeder/hipdnn-python-graph-bindings`) | standalone CMake, against an SDPA-ON frontend |
| **Injection package** (`hipdnn_torch/`) | this directory, on the **injection branch** | run in place (no build) |

The three checkouts stay **separate on purpose** and are combined only at runtime through
`HIPDNN_TORCH_PROVIDER_SO` + the frontend on `PYTHONPATH`:

- `users/brpepers/aot-catalog-engine` — the provider/engine + catalog. Never merged into
  the injection branch.
- `users/sareeder/hipdnn-python-graph-bindings` (**PR #10600**) — the canonical Python
  graph bindings (rmsnorm / layernorm / sdpa / `NormFwdPhase`). This injection branch used
  to carry its own copies of those bindings; they were **dropped** in favor of #10600, so
  check that branch out for the `frontend_bindings/` you build in §3b.
- `users/brpepers/hipdnn-torch-injection` — **only** this Python package now (no
  frontend-binding commits). It calls the standard binding API #10600 exposes (e.g.
  `SdpaAttributes.set_attn_scale(float)`, not a custom `set_attn_scale_value`).

> [!IMPORTANT]
> The single hardest lesson of this bring-up: **the frontend binds to exactly one
> `libhipdnn_backend.so`, and PyTorch auto-loads its own bundled copy on CUDA init.** If
> your frontend/provider need a *newer* backend ABI than the one torch bundles, you get
> silent `HIPDNN_ATTR_UNKNOWN` errors or missing symbols. See
> [§4 The one-backend rule](#4-the-one-backend-abi-rule) — read it before you build the
> frontend, not after.

## 2. Prerequisites

- A supported GPU visible to ROCm. Validated on **gfx1151** (Radeon 8060S / Strix Halo).
- A ROCm build toolchain: `cmake >= 3.26`, `ninja`, a ROCm/clang compiler that targets
  your arch (the TheRock SDK ships one).
- Python 3.12 (the bindings are built `STABLE_ABI` against 3.12).
- A venv with a **TheRock ROCm nightly** torch. This runbook was validated against:
  - `torch 2.10.0+rocm7.13.0a20260513`
  - the paired `_rocm_sdk_libraries_gfx1151` wheel (installed automatically as a dep)
  ```bash
  python3.12 -m venv ~/aot-ab-venv
  . ~/aot-ab-venv/bin/activate
  # Install the ROCm nightly torch for your arch per TheRock's instructions. The key is
  # that it pulls in a matching _rocm_sdk_libraries_<arch> wheel; that is the SDK the
  # provider and frontend must be built against, and whose libhipdnn_backend.so torch
  # loads at CUDA-init.
  python -c "import torch, glob, site; \
    print(torch.__version__); \
    print(glob.glob(site.getsitepackages()[0] + '/_rocm_sdk_libraries_*'))"
  ```

Pick your two worktrees/checkouts and export their roots — the rest of this doc uses these:

```bash
export CAT=/path/to/worktree/aot-catalog-engine        # provider/catalog branch (also the SDPA-ON frontend)
export INJ=/path/to/worktree/hipdnn-torch-injection    # this Python injection package
export FE=/path/to/worktree/hipdnn-python-graph-bindings # PR #10600 (users/sareeder/...) -- the bindings
export VENV=~/aot-ab-venv
export SDKLIB=$(python -c "import glob,site; print(glob.glob(site.getsitepackages()[0]+'/_rocm_sdk_libraries_gfx1151/lib')[0])")
```

## 3. Build the pieces

### 3a. Engine plugin + catalog (on `$CAT`)

From the **repo root** of the catalog worktree, build the provider with the superbuild
preset (this is the only way to get `hip-kernel-provider`; it is not in the default
build — see the README's "Getting a provider" section):

```bash
cd "$CAT"
cmake --preset hip-kernel-provider -DHIPDNN_ENABLE_SDPA=ON   # SDPA ON here -- required, see note
cmake --build build
```

> [!NOTE]
> **This same configure also builds the hipDNN frontend the bindings later link, so SDPA
> must be turned on here.** `HIPDNN_ENABLE_SDPA` is the one project-wide feature flag
> (`projects/hipdnn/CMakeLists.txt`, default **OFF**) and **no CMake preset forwards it** —
> pass `-DHIPDNN_ENABLE_SDPA=ON` explicitly at *this* configure, before building the
> bindings (§3b), so the exported `hipdnn_frontendConfig.cmake` carries
> `set(HIPDNN_ENABLE_SDPA ON)` and the bindings inherit it. Skip it and the bindings
> silently compile `g.sdpa` / `SdpaAttributes` out. Verify it took:
> ```bash
> grep -r HIPDNN_ENABLE_SDPA "$CAT"/build/**/hipdnn_frontendConfig.cmake   # expect: set(HIPDNN_ENABLE_SDPA ON)
> ```

Products you care about:

```
$CAT/build/lib/hipdnn_plugins/engines/libhip_kernel_provider.so        # the plugin
$CAT/build/lib/hipdnn_plugins/engines/arch_content/aot_catalog/gfx1151 # the 7-family catalog
```

> [!WARNING]
> **Two catalog trees exist under `engines/`, and only one is complete.** The build lays
> down both `engines/aot_catalog/` (the original 4 families: `fmha_wmma_fwd`,
> `gemm_wmma`, `gemm_wmma_universal`, `rmsnorm2d`) **and**
> `engines/arch_content/aot_catalog/` (all **7**: the four above plus `activation`,
> `conv2d_fprop`, `layernorm2d`). If `HIPDNN_AOT_CATALOG_DIR` points at the 4-family tree
> (or you let a default win), then layernorm / silu / gelu / conv2d **silently fall back
> to native** — `provider_ready()` is still `True` and the model still runs, so this is
> easy to miss. Always point at the `arch_content/aot_catalog` tree (see §5).

> [!NOTE]
> **A stale `.so` won't have new adapters.** If you add or change an op adapter in the
> provider, a plain incremental `ninja`/`cmake --build build` relinks it — but if you're
> reusing a `.so` from an earlier checkout, confirm the adapters are present before
> chasing a "didn't route" ghost:
> ```bash
> nm -C "$CAT/build/lib/hipdnn_plugins/engines/libhip_kernel_provider.so" | grep -iE 'layernorm|conv|pointwise|matmul|rmsnorm|sdpa' | head
> ```

### 3b. Frontend bindings (on `$FE` — PR #10600)

The Python graph bindings come from **PR #10600**
(`users/sareeder/hipdnn-python-graph-bindings`), *not* this injection branch — the
injection branch dropped its own copies in favor of #10600. Check that branch out at `$FE`
and build its `frontend_bindings/`. It references recent frontend surface — `BehaviorNote`,
the RFC-0016 `is_override_shape_enabled` graph attribute — that is **newer than what a
several-months-old nightly's header-only frontend provides**, so build against the
**SDPA-ON hipDNN build tree from §3a** (`$CAT/build`), not the pip nightly's stale
artifact:

```bash
cd "$FE/projects/hipdnn/python/frontend_bindings"
cmake -S . -B build -GNinja \
  -DCMAKE_PREFIX_PATH="$CAT/build;$VENV/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151" \
  -DPython_EXECUTABLE="$VENV/bin/python"
cmake --build build
# product: build/hipdnn_frontend_python.abi3.so
```

Two things that cost hours if you get them wrong:

- **SDPA is inherited from the frontend, not a bindings flag.** The `g.sdpa` /
  `SdpaAttributes` surface is `#ifdef HIPDNN_ENABLE_SDPA`-guarded. #10600's bindings expose
  **no** SDPA option of their own — they pick it up automatically because §3a built the
  frontend with `-DHIPDNN_ENABLE_SDPA=ON`, which propagates through the linked
  `hipdnn_frontend` target and its exported `hipdnn_frontendConfig.cmake`. If `g.sdpa` is
  missing after a rebuild, you skipped the flag in §3a — fix it there, not here.
- **`CMAKE_PREFIX_PATH` ordering.** Put the fresh hipDNN build tree (`$CAT/build`, which
  exports `hipdnn_frontendConfig.cmake` / `hipdnn_backendConfig.cmake` for the just-built
  headers *with SDPA ON*) **before** the pip SDK prefix (which supplies `hipConfig.cmake`).
  First match wins; if the stale SDK's frontend config is found first, the build fails on
  the missing new surface **and** you lose the SDPA define.

### 3c. Injection package (on `$INJ`)

Nothing to build — it's run in place. `samples/*` do `sys.path.insert` to import the
package, and `tests/` import it directly. You only need the env from §5.

## 4. The one-backend ABI rule

The frontend `.so` links `libhipdnn_backend.so`. **PyTorch also loads its own bundled
backend** during CUDA init. Under `RTLD_GLOBAL`, first-loaded wins symbol resolution, so
if the two disagree on ABI you get one of:

- `undefined symbol: hipdnnBackendGetSerializedBinaryGraphAndPlan_ext` — torch's bundled
  backend is **older** than the frontend expects.
- `HIPDNN_ATTR_UNKNOWN` on `is_override_shape_enabled` (layernorm/rmsnorm) — the develop
  frontend sets an RFC-0016 attribute unconditionally, and torch's **older** bundled
  backend rejects it.

`bootstrap.py` deliberately `dlopen`s **torch's own** backend first, which is correct
*when torch's bundle is new enough*. When it isn't, you have two options:

1. **Preferred — use a torch nightly whose bundled backend is post-RFC-0016.** Then
   nothing special is needed; the bootstrap's dlopen of torch's backend just works.
2. **Fallback — swap torch's bundled backend for the one you built.** Back it up first,
   then overwrite:
   ```bash
   cp "$SDKLIB/libhipdnn_backend.so" "$SDKLIB/libhipdnn_backend.so.orig"   # once
   cp "$CAT/build/lib/libhipdnn_backend.so" "$SDKLIB/libhipdnn_backend.so"
   ```
   This is what the validated WSL config uses (backup lives at
   `$SDKLIB/libhipdnn_backend.so.orig`). To undo it, copy `.orig` back.

> [!CAUTION]
> `LD_PRELOAD`-ing a newer backend instead of swapping does **not** work here — it drags
> in a second statically-linked LLVM and dies with
> `spirv-expand-step ... registered more than once`. Swap the file (option 2) or match
> the nightly (option 1); don't preload.

## 5. Wire the environment

Everything below is what a validated run exports. `HIPDNN_TORCH_PROVIDER_SO` is the only
*required* variable; the rest pin the pieces you built above and the catalog tree.

```bash
BUILD="$CAT/build"
PROV="$BUILD/lib/hipdnn_plugins/engines/libhip_kernel_provider.so"

export HIPDNN_TORCH_PROVIDER_SO="$PROV"
# Pin the AOT catalog engine. The default policy hands selection to hipDNN across all
# loaded engines; this runbook overrides it to FORCE a single engine on purpose -- we're
# validating the AOT catalog kernels, so we want every routed op served by that one engine
# for deterministic census attribution, not "whatever hipDNN picks." (Forcing the AOT
# engine pins a hipDNN engine, not a kernel source: it serves whatever AOT kernels are in
# its catalog -- from rocKE today, but the engine is source-agnostic.) For real-model perf
# later you'd drop both of these and stay on the default policy (see §8).
export HIPDNN_TORCH_SELECT=force
export HIPDNN_TORCH_ENGINE=AOT_CATALOG_ENGINE
export HIPDNN_TORCH_FRONTEND_DIR="$FE/projects/hipdnn/python/frontend_bindings/build"
# Point at torch's bundled backend (swapped-in develop backend per §4, or a new-enough nightly):
export HIPDNN_TORCH_BACKEND_GLOB="$SDKLIB/libhipdnn_backend.so"

# The 7-family tree (NOT the 4-family engines/aot_catalog) -- see §3a warning:
export HIPDNN_AOT_CATALOG_DIR="$BUILD/lib/hipdnn_plugins/engines/arch_content/aot_catalog"

# WSL2 only: the GPU shim must be discoverable before torch inits CUDA (see §9).
export LD_LIBRARY_PATH="$VENV/wsl-shim:$(dirname "$PROV"):$BUILD/lib:$LD_LIBRARY_PATH"
```

Notes:
- `HIPDNN_TORCH_FRONTEND_DIR` is the raw `frontend_bindings/build` dir; it's used only if
  the `hipdnn-frontend` wheel isn't importable. If you `pip install` the frontend wheel
  instead, you can drop this.
- `HIPDNN_AOT_CATALOG_DIR` is an **engine** variable, not a `hipdnn_torch` one — the
  package doesn't set or clear engine env (by design). The engine reads it; the AOT
  catalog resolver logs which root it chose (see §6).

## 6. Verify + prove it's hipDNN

Three layers of proof, cheapest first. Together they establish "the op ran on hipDNN"
and enumerate every native fallback with a reason.

**(a) It's wired at all — no model:**
```bash
python -c "import hipdnn_torch; print(hipdnn_torch.provider_ready())"   # -> True
```

**(b) Parity + routing, per op — the pytest suite:**
```bash
cd "$INJ/projects/hipdnn/tools/hipdnn_torch"
python -m pytest tests/ -q
```
`tests/test_gates.py` runs on CPU (no GPU/provider needed) and pins the two legitimate
native-routing stops (non-CUDA tensor, unmappable dtype) plus the pure translation
helpers (`_ntuple`/`_resolve_pads` padding math, the `[1,…,1,*ns]` scale-view builder,
the N-D matmul operand builder, the gelu/silu `_mode` map) — it asserts **no** capability
declines, because there are none under the pure-passthrough contract.
`tests/test_parity.py` (auto-skipped unless `provider_ready()`) asserts, for every op,
that the census shows **`aot>0`** (it routed, not fell back) *and* the result matches
native within the dtype tolerance, across the newly translated paths (N-D linear, grouped
conv, `'same'` padding, causal SDPA on permuted views, multi-axis LayerNorm/RMSNorm).

**(c) The census is the proof-of-hipDNN + the fallback ledger.** Every routed call
increments an `aot` counter keyed by shape; every fallback increments `native` **with a
reason**. `native=0` across the board means nothing silently reverted to PyTorch:
```bash
python samples/minimal_block.py     # prints per-op "aot=N native=0" + parity OK
python samples/microbench_ab.py     # per-op A/B: parity, timing, routed-or-fell-back
```
Expect exact-erf `F.gelu` to show up as a fallback when no loaded engine serves it —
it is built and submitted like every other op (no pre-decline), so the reason is a
hipDNN-origin decline from `check_support`, not a gate string. That's the ledger working,
not a bug; the day a builder for it lands, the same call routes with no injection change.

**(d) Engine-level trace — irrefutable proof the catalog served the op.** Set
`HIPDNN_AOT_DEBUG=1` and the AOT catalog engine narrates catalog resolution, family
loads, and per-graph accept/decline:
```bash
HIPDNN_AOT_DEBUG=1 python samples/minimal_block.py 2>&1 | grep aot-catalog
```
A healthy run shows the resolver picking your `HIPDNN_AOT_CATALOG_DIR` and loading all
seven families, e.g.:
```
[hipdnn aot-catalog] resolving catalog for arch gfx1151: root=.../arch_content/aot_catalog (env HIPDNN_AOT_CATALOG_DIR)
[hipdnn aot-catalog] loaded family 'fmha_wmma_fwd_gfx1151'      op_kind='sdpa'        (2 kernels)
[hipdnn aot-catalog] loaded family 'activation_gfx1151'         op_kind='pointwise'   (12 kernels)
[hipdnn aot-catalog] loaded family 'gemm_wmma_universal_gfx1151' op_kind='matmul'     (12 kernels)
[hipdnn aot-catalog] loaded family 'layernorm2d_gfx1151'        op_kind='layernorm'   (18 kernels)
[hipdnn aot-catalog] loaded family 'conv2d_fprop_gfx1151'       op_kind='conv_fprop'  (6 kernels)
[hipdnn aot-catalog] loaded family 'rmsnorm2d_gfx1151'          op_kind='rmsnorm'     (18 kernels)
[hipdnn aot-catalog] loaded family 'gemm_wmma_gfx1151'          op_kind='matmul'      (2 kernels)
[hipdnn aot-catalog] loaded 7 family(ies) for arch gfx1151
```
If you see `loaded 4 family(ies)` you're pointed at the wrong catalog tree (§3a). If you
see the resolver pick a `root=` you didn't set, `HIPDNN_AOT_CATALOG_DIR` isn't taking —
it must be exported before the process starts.

> **Tracking native fallbacks generally.** For any real model, call
> `hipdnn_torch.enable_logging()` to print each fallback as it happens, and
> `print(hipdnn_torch.report())` at the end for the ranked reason list — that list *is*
> the "what hipDNN still needs" backlog. `census()` returns the same counters
> programmatically for CI assertions like "no unexpected fallbacks."

## 7. Running a real model (ComfyUI / LTX-Video)

`samples/ltx_video_ab.py` drives a real LTX-Video diffusion transformer through the
injection with a per-op **device-time** census (needs a ComfyUI checkout via
`COMFYUI_PATH`). One caveat worth stating up front:

> [!NOTE]
> **Empty / random-weight checkpoints only exercise part of the graph.** With no real
> `.safetensors` checkpoint, the transformer stack may not run end-to-end, but the
> **VAE** path (conv2d + activations) does — which is enough to prove conv/activation
> routing on real tensor shapes. Full-model perf numbers need a real checkpoint; op
> *routing* can be verified on the VAE path alone.

## 8. Engine selection: `default` vs `force` (and a future tuning run)

`hipdnn_torch` exposes two selection policies via `HIPDNN_TORCH_SELECT`, and the choice is
independent of any specific engine:

- **`default` (the default).** Hand the graph to hipDNN and let it select across every
  loaded engine; the census reports the *winning* engine per shape. No engine is named. This
  is the engine-agnostic behavior, and the right one for honest full-model performance. This
  selection can be pointed at a **rules file** (via the backend's `HIPDNN_HEUR_CONFIG_PATH`)
  that decides the engine per graph/shape — the injection consumes such a file with no code
  change.
- **`force`.** Pin exactly one engine, named by `HIPDNN_TORCH_ENGINE`; a shape it can't
  serve falls back to native. This gives **deterministic attribution** — every routed op is
  served by that one engine — which is what you want to validate or bench a single engine in
  isolation.

That gives two very different intents, and this runbook is squarely in the first:

- **Validating the AOT catalog kernels (what this runbook does).** We force a single engine
  with `HIPDNN_TORCH_SELECT=force` + `HIPDNN_TORCH_ENGINE=AOT_CATALOG_ENGINE`
  (+ `HIPDNN_AOT_CATALOG_DIR`). Note this pins a *hipDNN engine*, not a kernel source: the
  AOT catalog engine serves whatever ahead-of-time kernels are in its catalog — from rocKE
  today, but the engine is source-agnostic and could serve AOT kernels from anywhere. Every
  routed op is then served by that one engine, which is exactly what we want when the goal
  is to exercise and prove those kernels. Read the census accordingly: **`aot>0` /
  `native=0` means "the pinned engine served it, didn't fall back to native PyTorch" — it is
  NOT a claim that engine is the fastest hipDNN engine for that shape.** The microbench
  speedups in §6 are forced-engine-vs-native, not a cross-engine best.

- **Real-workload model perf (later).** Do **not** force an engine — stay on the `default`
  policy. Let hipDNN select the best engine it has for each problem; the AOT catalog engine
  will increasingly compete with the HIP kernel provider, MIOpen, and the AITER ASM SDPA
  kernels. On this axis the interesting number is best-engine-vs-native, not forced-vs-native.

> **Future: a persistent tuning run (planned injection feature).** A natural third mode would
> have the injection measure the loaded engines on a real model's actual shapes and record the
> winner per graph/shape into a rules file, so a second run replays the tuned selection with no
> re-measurement. hipDNN itself already has C++ auto-tuning that emits such rules, but it is not
> yet exposed in the Python bindings. Either binding that, or recreating the measure-and-record
> loop in the injection, is a separate feature to be developed — it does not exist today.

Practically: keep the pin for the validation/parity work here; drop it when you move to
honest full-model performance comparisons.

## 9. Platform notes

### WSL2 (validated — this is the config all §s above describe)
- **The `librocdxg` GPU shim must be on `LD_LIBRARY_PATH` before torch inits CUDA**, or
  device enumeration fails. Stage it in a dir (this runbook uses `$VENV/wsl-shim/`
  holding `librocdxg.so` + `librocdxg.so.1`) and prepend that dir to `LD_LIBRARY_PATH`
  (§5). Native Linux and Windows do **not** need this.
- The one-backend swap (§4 option 2) was used here because the `2026-05-13` nightly's
  bundled backend predates the RFC-0016 attribute the develop frontend sets.

### Native Linux
- No `librocdxg` shim needed; drop it from `LD_LIBRARY_PATH`.
- Everything else (build steps, the one-backend rule, the catalog-tree gotcha) applies
  unchanged.

### Native Windows (validated — gfx1151 Strix Halo, 2026-08-13)

Validated on a TheRock nightly (`torch 2.9.1+rocm7.15.0a20260715`, driver `32.0.31016.2`),
Python **3.12** venv. The build steps of §3 apply unchanged except for the path/naming
deltas below; the runtime DLL and plugin-shadowing hazards are handled automatically by
`bootstrap.py` — you do not set them by hand.

**Probe the GPU first.** Nightly × driver pairing is not portable: pull your candidate
nightly and run one op before building —
`python -c "import torch; assert torch.cuda.is_available(); (torch.randn(8,8,device='cuda')@torch.randn(8,8,device='cuda')).sum().item()"`.
If it faults (`kpack_load_code_object` / `hipErrorInvalidImage`), try another nightly
before touching the driver. Record the pair that loads.

**Build-product naming/location deltas (vs the `.so`/`lib/` paths in §3):**
- Backend is `hipdnn_backend.dll`; provider plugins are `hip_kernel_provider.dll` (etc.),
  **not** `lib*.so`.
- Plugins land under **`build\bin\hipdnn_plugins\engines\`**, not `build/lib/...`. The
  7-family catalog is at `build\bin\hipdnn_plugins\engines\arch_content\aot_catalog\gfx1151`.
- The frontend binding is **`hipdnn_frontend_python.pyd`**, not `.abi3.so`.
- Build the engine with `-DROCKE_COMGR_LIB=<sdk>/bin/amd_comgr.dll` so rocKE can JIT.

**Split-SDK DLL search (handled by `bootstrap.py`, Windows-only `_add_dll_search_dirs`).**
The nightly splits its runtime SDK into two wheels: `_rocm_sdk_core\bin` (amdhip64,
amd_comgr, hiprtc, hiprtc-builtins — `hip_kernel_provider` links hiprtc for rocKE JIT) and
`_rocm_sdk_libraries\bin` (hipdnn_backend + math libs). Windows `ctypes`/`LoadLibrary` use
the *secure* DLL search, which **ignores `PATH`**, and `RTLD_GLOBAL` is a POSIX no-op here.
`bootstrap.py` `os.add_dll_directory`s the backend dir, the provider dirs, and **every**
sibling `_rocm_sdk_*\bin` before dlopening the backend. Override the set with
`HIPDNN_TORCH_DLL_DIRS` if your layout differs.

**Plugin-loading mode = `absolute` (handled by `bootstrap.py`; cross-platform fix that
Windows exposes).** The nightly ships a **prebuilt** `hip_kernel_provider.dll` beside
`hipdnn_backend.dll` in `_rocm_sdk_libraries\bin\hipdnn_plugins\engines\` **with no
`arch_content/` catalog next to it** (empty/old catalog). In the default `additive` load
mode the backend unions that prebuilt with your local build, and because cross-plugin
engine-id ownership is plain last-writer-wins (no de-dup — verified in the backend loader),
the stale copy can **shadow** `AOT_CATALOG_ENGINE` and silently decline every graph:
`provider_ready()` still returns `True`, but census shows `aot=0` and there is no
`[hipdnn aot-catalog]` trace. `bootstrap.py` now calls
`set_engine_plugin_paths(providers, ABSOLUTE)` by default, so your named provider(s)
**replace** the auto-discovered set and the outcome no longer depends on load order. This
is not Windows-specific code — the loader is platform-identical; Windows merely ships the
colliding prebuilt. Set `HIPDNN_TORCH_PLUGIN_MODE=additive` to opt back into unioning.

**Backend glob.** `bootstrap.py`'s default glob targets the Linux `lib/libhipdnn_backend.so`
layout, which misses on Windows three ways (non-suffixed dir, `bin` not `lib`, `.dll`).
Point `HIPDNN_TORCH_BACKEND_GLOB` at the exact
`...\_rocm_sdk_libraries\bin\hipdnn_backend.dll`.

**No `librocdxg` shim** (that's WSL-only).

**Capturing python stdout/stderr from PowerShell 5.1.** PS 5.1 mangles native-command file
redirection; use `Start-Process -RedirectStandardOutput/-RedirectStandardError` to capture
a run reliably.

## 10. Quick caveat index

| Symptom | Cause | Fix | § |
|---------|-------|-----|---|
| `provider_ready()==True` but layernorm/silu/gelu/conv2d never route | pointed at the 4-family `engines/aot_catalog` tree | set `HIPDNN_AOT_CATALOG_DIR` to `arch_content/aot_catalog` | 3a, 5 |
| SDPA surface (`g.sdpa`/`SdpaAttributes`) missing after rebuild | frontend built SDPA-OFF, so #10600's bindings compile it out | build the frontend with `-DHIPDNN_ENABLE_SDPA=ON` in §3a; it propagates to the bindings via the exported config | 3a, 3b |
| Frontend build fails on `BehaviorNote`/`isKnownBehaviorNote` | frontend prefix too old | put `$CAT/build` first on `CMAKE_PREFIX_PATH` | 3b |
| `undefined symbol: hipdnnBackend...GraphAndPlan_ext` | torch's bundled backend too old | new-enough nightly, or swap backend | 4 |
| `HIPDNN_ATTR_UNKNOWN` (is_override_shape_enabled) | two backends, old torch one wins | swap torch's bundled backend | 4 |
| `spirv-expand-step registered more than once` | tried to `LD_PRELOAD` a 2nd backend | swap the file instead | 4 |
| new adapter added but op still falls back | reusing a stale provider `.so` | relink; verify with `nm -C` | 3a |
| device enumeration fails on WSL | `librocdxg` shim not on `LD_LIBRARY_PATH` | prepend the shim dir | 9 |
| `provider_ready()==True` but census `aot=0`, no `[hipdnn aot-catalog]` trace (esp. Windows) | a prebuilt `hip_kernel_provider` shipped beside the backend shadows your local build (last-writer-wins engine-id) | use `absolute` plugin mode (`bootstrap.py` default; `HIPDNN_TORCH_PLUGIN_MODE`) so your provider replaces the auto-discovered set | 9 |
| backend/provider `.dll` fails to load on Windows despite being on `PATH` | secure DLL search ignores `PATH`; split SDK (`_rocm_sdk_core` vs `_rocm_sdk_libraries`) | `bootstrap.py` `add_dll_directory`s all sibling `_rocm_sdk_*\bin`; override with `HIPDNN_TORCH_DLL_DIRS` | 9 |
| exact-erf `F.gelu` falls back | submitted, but no loaded engine serves it yet (hipDNN-origin decline, not a pre-gate) | use `approximate="tanh"`, or accept the fallback until a builder lands | 6 |

## 11. Reproducing on another machine (share this)

Hand this section to anyone with a working ROCm/hipDNN dev setup who wants to stand
the stack up on their own GPU. It is deliberately **goal-oriented, not a pinned
recipe** — environments vary too much (gfx942/gfx950 CDNA infra boxes and gfx1151;
native Windows, native Linux, or WSL2). Everything OS-specific lives in §9; the arch
target is a build flag. Follow §§1–8 for the build, §9 for your OS, and use the list
below as the invariants-and-acceptance checklist.

**The three PRs** (build separately, combine only at *runtime* — see §1):
- **#10556** `users/brpepers/aot-catalog-engine` — AOT catalog engine in `hip-kernel-provider`; rocKE AOT-creates kernels for your arch.
- **#10600** `users/sareeder/hipdnn-python-graph-bindings` — the hipDNN Python graph bindings the injection depends on.
- **#10562** `users/brpepers/hipdnn-torch-injection` — the `hipdnn_torch` injection package, this runbook, the pytest suite, model samples.

**Invariants (get these right or it silently no-ops):**
1. **Probe the GPU first** on your candidate nightly (one `cuda` matmul) before building — nightly×driver pairing is not portable (§9, and §2/Phase 0).
2. Configure the engine with **`-DHIPDNN_ENABLE_SDPA=ON`** and **`-DGPU_TARGETS=<your arch>`** (§3a). SDPA-ON here also propagates to #10600's bindings.
3. Put the engine **`build/` dir FIRST** on the frontend's `CMAKE_PREFIX_PATH` (§3b) — first match wins.
4. **One-backend rule** (§4): the frontend must bind the same `hipdnn_backend` torch loads.
5. Point **`HIPDNN_AOT_CATALOG_DIR` at `arch_content/aot_catalog/<arch>`**, not the 4-family `engines/aot_catalog` (§3a, §5).
6. Plugin loading is **ABSOLUTE by default** (`bootstrap.py`): your built provider replaces the backend's auto-discovered plugins so an SDK-shipped copy can't shadow it. `provider_ready()==True` + census `aot=0` is the shadowing symptom (§9, §10).

**Two test modes** (§8) — never confuse them:
- **`default` + all providers co-loaded = the pass/fail gate** ("did hipDNN route this graph through *any* engine"). Run `tests/` here.
- **`force` (pin `AOT_CATALOG_ENGINE`) = attribution only.** Its failures are known AOT POC gaps (NCHW/3-D conv, 2-D-only norms/gemm, non-causal SDPA), not defects. Never gate on it.

**Acceptance — structural, because exact counts are arch-specific:**
- `hipdnn_torch.provider_ready()` → `True`.
- `HIPDNN_AOT_DEBUG=1 python samples/minimal_block.py` resolves the catalog to *your* `HIPDNN_AOT_CATALOG_DIR` and prints `loaded N family(ies)` for your arch (7 on gfx1151; may differ on CDNA).
- `pytest tests/ -q` in **default + all-providers** is green, with a handful of `xfailed` for graphs no engine covers on your arch (the gfx1151 set: weightless 3-D `layer_norm`, N-D `linear`+bias, causal SDPA `H=8`, 3-D `rms_norm`).
- A sample shows routed ops with `native=0` and parity within dtype tol.

**What "replicate" means for you:** on another **gfx1151** you should match closely. The
AOT/rocKE kernels are **arch-specific** — the validated families are RDNA/WMMA
(`fmha_wmma`, `gemm_wmma`, …). On **CDNA** (gfx942/gfx950, MFMA/XDL) rocKE may build a
different/smaller set or none; that's expected. There the portable result is that the
**injection stands up and `default` mode routes ops through hipDNN** (MIOpen/hipblaslt
serving), with AOT coverage best-effort. Report back your `loaded N families`, the
default-gate pass/xfail split, and any arch-specific decline reasons.
