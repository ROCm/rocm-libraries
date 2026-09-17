# flyDSL-in-hipDNN — Phase B Handoff (gfx950)

**Goal:** get hipDNN to pick best-of-breed among **flyDSL**, **rocKE**, and **AITER ASM**
for **attention** shapes — first in a synthetic 3-engine head-to-head, then in real model
runs with autotune choosing the winner.

**Status of this doc:** written on the gfx1151 laptop (WSL2) at the end of the POC phase,
to hand off to a fresh session running on/near the **gfx950** box (reached via the Alola
login server → ssh to the gfx950 system). This branch
(`users/brpepers/hipdnn-flydsl-poc`) is based on bharriso's
`rocke-gfx950-dense-autotune-sweep` work, so the rocKE gfx950 autotune/ingestor base is
already underneath us — useful, not accidental.

> Throwaway / go-fast rules still apply: not a PR, doesn't need to be "right," just needs
> to prove the path. No kpack. No committing/pushing beyond this branch without sign-off.

---

## 1. What is proven (done)

- **Phase 0** — a trivial flyDSL kernel compiles + runs on gfx1151 (`vadd.py` →
  `vadd_flydsl.hsaco`, launched with raw `hipModuleLoadData`/`hipModuleLaunchKernel`).
- **Phase A** — that flyDSL HSACO runs *through hipDNN's real dispatch-handler contract*
  via the "native escape hatch" (`packs/TestFlydslRawDispatchHandler.cpp`): a UDD author's
  own `IKernelDispatchHandler` raw-loads the build-time HSACO and runs prepare/launch.
  Selection was still stubbed here.
- **Real selection + execution POC** — `app/flydsl_hipdnn_app.cpp`: a standalone C++ app
  builds a pointwise-ADD graph through the **public hipDNN frontend API** and proves BOTH:
  1. **selection** — `get_ranked_engine_ids()` returns `hipkernel:Flydsl`, and the
     provider logs `ingestor: engine 'hipkernel:Flydsl' selected kernel ...-0007`.
  2. **execution** — `3 + 4 == 7` computed on-device by the flyDSL `vadd_0` kernel.
  App exits 0. Run recipe: `app/run_app.sh` (gfx1151-specific paths — see §6).

**Bottom line:** the plumbing to register a flyDSL engine, have hipDNN *select* it, and
*execute* it through the public API is working end-to-end. What remains is (a) making the
served op **attention** instead of pointwise, and (b) doing it on **gfx950** where the two
competitors live.

---

## 2. Standing decisions (do not relitigate)

1. **kpack is OUT — permanently, even for attention.** kpack is just a packaging format
   for HSACO + metadata. Bare HSACO on disk (what the POC dispatch already does) is fine.
   AITER's asm_sdpa uses kpack only because it's a shipped product; irrelevant to us.
2. **Selection = hipDNN autotune**, not pinning or a hand-rolled bench. The POC's
   `set_preferred_engine_id_ext` pin was POC-only. Real comparison: register all three
   engines against the *same* SDPA graph and let autotune benchmark + pick.
3. **flyDSL itself is untouched** — we use it purely as a *compiler* for our kernels. Our
   kernel **sources** live in this repo (e.g. `vadd.py`), not in flyDSL.
4. **Kernel sourcing = copy from AITER, adapt into flyDSL, hook into hipDNN.** We grab the
   attention kernel(s) we want from the AITER repo, copy them in, and make the changes
   needed to compile them via flyDSL and wire them into hipDNN.
   - **TOP OPEN DESIGN QUESTION:** AITER attention kernels are ASM / CK / Triton / HIP.
     flyDSL consumes its own DSL (see `vadd.py`: `@kernel`, `flydsl.expr.gpu/arith`,
     `Tensor`/`T` types). So "make changes needed" is really a **port into flyDSL's input
     form**, not copy-paste. That port is the crux of the flyDSL-engine work. Decide early
     whether to port a full FMHA or start from a minimal attention tile.
5. **Stage-2 "PyTorch injection" reality:** there is **no** PyTorch attention monkeypatch /
   `SDPBackend` hook in this repo. The only mechanism that forces PyTorch onto hipDNN is
   the **cuDNN shim** (`projects/hipdnn/docs/rfcs/0012_CuDNN_Shim.md`) — intercept the
   cuDNN attention API and redirect into hipDNN. It is an RFC; **not verified to build or
   work**. Stage 2 inherits "is the shim real yet?" as a sub-blocker.

---

## 3. Where the three competitors live

| Engine | Location | Notes |
|---|---|---|
| **AITER ASM** | `dnn-providers/hip-kernel-provider/src/engines/asm_sdpa_engine/` | engine `ASM_SDPA_ENGINE_NAME`; built `gfx942 gfx950`; `HIPDNN_ENGINE_ASM_SDPA=1`; packs `.kpack` (ignore packaging). This is AITER ASM attention. |
| **rocKE** | `dnn-providers/hip-kernel-provider/rocke/` | FMHA MFMA: `rocke/platform/cpp/include/rocke/instance_fmha_mfma.h`. Live attention benches: `rocke/library/benchmarks/gfx9{42,50}/attention/{prefill,decode}/` — confirms rocKE attention is real on gfx950. |
| **flyDSL attention** | **does not exist — must author** | bare HSACO + hand-written descriptors, mirroring the pointwise POC's 6-descriptor pattern (§6). |

**Attention is NOT served by the kernel-ingestor engine** — the ingestor natively serves
only `hipkernel:Pointwise` and `hipkernel:ConvFwd`. Attention is the **SDPA engine's**
domain, and `HIPDNN_ENABLE_SDPA` is currently **OFF** in the frontend build
(`hipdnn_frontendConfig.cmake` ~line 34; `projects/hipdnn/frontend/CMakeLists.txt` ~line 86).
It must be **ON** to construct an SDPA graph.

---

## 4. Two-stage test plan

### Stage 1 — synthetic N-engine head-to-head (mechanism CONFIRMED in the frontend)
Construct an SDPA graph applicable to all engines, enumerate a plan per supporting engine,
run each on the same variant pack, show the same problem served by N engines, and let
autotune rank them. Frontend API that makes this possible (`hipdnn_frontend/Graph.hpp`):
- `create_execution_plans()` — retains **all** supporting engines' configs.
- `build_plans(BuildPlanPolicy::ALL)` — compile every candidate.
- `deselect_engines()` / `get_ranked_engine_ids()` / `get_knobs_for_engine()` — iterate
  engine-by-engine, execute each.
- autotune benchmarking (`barred` / `nonBenchmarkedResults`) — the built-in selector.

**Do the 2-engine version first** (rocKE vs asm_sdpa) to de-risk the enumerate→execute→
autotune loop **before** flyDSL attention exists. Then add flyDSL as the third engine.

### Stage 2 — real models with autotune ON
Force PyTorch attention → hipDNN via the **cuDNN shim** (RFC 0012), run real models with
autotune on, collect best-of-breed results. Blocked on verifying the shim actually works.

---

## 5. Blockers

1. **gfx950 access** — HAVE IT (Alola login → ssh gfx950). This handoff exists so the work
   resumes there natively instead of via a fragile double-hop from the laptop.
2. **SDPA-ON build on gfx950** — `HIPDNN_ENABLE_SDPA` is OFF today; turn it ON.
3. **flyDSL FMHA must be authored** (§2.4) before flyDSL can be the third engine.

---

## 6. File map + run recipe

Checked-in here (SOURCE only — binaries are `.gitignore`d and regenerated):
- `app/flydsl_hipdnn_app.cpp` — the public-API selection+execution proof app.
- `app/CMakeLists.txt`, `app/run_app.sh` — build + run (paths are **gfx1151-specific**).
- `flydsl_descriptors/flydsl/*.json` — the 6-descriptor pack that makes `hipkernel:Flydsl`
  selectable (ued/uhd/kmd/udd/umd + `flydsl_add.kdp.json`). **This is the template** for
  the attention descriptors.
- `vadd.py`, `rmsnorm.py` — flyDSL **kernel sources** (compile → HSACO).
- `run_hsaco.cpp`, `run_rmsnorm.cpp`, `dbg_dump.py` — raw HSACO host harnesses / debug.
- `PHASE0_NOTES.md`, `PHASE_A_NOTES.md`, `PHASE_A2_APP_GUIDE.md`, `KERNEL_SURVEY.md`.
- (provider) `dnn-providers/.../kernel_ingestor_engine/packs/TestFlydslRawDispatchHandler.cpp`
  + its `CMakeLists.txt` wiring — the Phase-A escape-hatch test.

**Run recipe (gfx1151 POC):** `env -u ROCM_PATH bash flydsl_poc_scratch/app/run_app.sh`.
`run_app.sh` prepends the freshly-built `build/lib` to `LD_LIBRARY_PATH` — **critical**:
the ROCm SDK ships a **stale** `libhipdnn_backend.so` that predates attr 609
(`HIPDNN_ATTR_OPERATIONGRAPH_IS_OVERRIDE_SHAPE_ENABLED_EXT`); if it wins via
`LD_LIBRARY_PATH`, `build_operation_graph` fails with `HIPDNN_ATTR_UNKNOWN`. Verify with
`ldd $APP | grep hipdnn` → must resolve to `build/lib`. These absolute paths and the vadd
HSACO are gfx1151 artifacts — **the pointwise POC is not meant to run on gfx950** (different
arch); it's the *method* that carries over.

**Build the repo:** `~/rocm-sdk/build-miopen.sh` style / `env -u ROCM_PATH`; turn SDPA ON
for Stage 1.

---

## 7. Handoff next steps (ordered, gfx950-first)

1. **On the gfx950 box:** fetch this branch, build the provider + backend with
   `HIPDNN_ENABLE_SDPA=ON` (`env -u ROCM_PATH`). Confirm `asm_sdpa` and rocKE attention
   engines load (check provider plugin dir + engine names).
2. **Stage 1 (2-engine):** author a minimal SDPA-graph harness (reuse the app's public-API
   pattern) that enumerates plans for rocKE **and** asm_sdpa on one attention shape, runs
   each, and lets autotune rank them. Proves the loop before flyDSL exists.
3. **Author flyDSL attention:** pull the AITER attention kernel(s), port into a flyDSL
   kernel source in-repo (model after `vadd.py`), compile to bare HSACO, and write the
   attention descriptor pack (copy `flydsl_descriptors/flydsl/*.json`, swap the op/kernel
   metadata) so flyDSL `check_support`s the same SDPA graph.
4. **Stage 1 (3-engine):** add flyDSL to the ranked set; confirm autotune ranks all three.
5. **Stage 2:** verify the cuDNN shim redirects PyTorch attention; run real models with
   autotune ON; collect results.

**Reconstruct context fast:** see the auto-memory files (kept outside the repo, in
`~/.claude/.../memory/`): `project_hipdnn_flydsl_real_selection_done`,
`project_flydsl_hipdnn_phaseB_ground_rules`, `project_flydsl_kernel_landscape`,
`project_hipdnn_flydsl_phaseA_done`, `project_flydsl_phase0_bringup`.
