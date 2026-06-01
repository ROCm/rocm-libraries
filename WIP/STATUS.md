# CK DSL hipDNN Provider — M1 Complete

Branch: `users/dahawkin/ck-dsl-provider`.
Cut from `users/vanantha/ck-dsl-prototype` at `bf7546ed99e`.

## Plan progress (per plan v0.9 §6.2)

| Step | Status |
|---|---|
| Prep P-1 … P-7 | done; synthesis in `WIP/prep_findings/PREP_FINDINGS.md` |
| I-1 skeleton | ✅ `73116d68404` |
| I-2 embedded interpreter | ✅ `e980348fcab` |
| I-3 compile-service bridge | ✅ `2fa22bc9c8e` (+ GIL-dtor fix `0fcec098917`) |
| I-4 KernelArtifact / HipModule | ✅ `a0f8288e395` |
| I-5 JitCache | ✅ `c6c8b480f37` |
| I-6 ConvImplicitGemmAdapter + Spec | ✅ `9f1a92afe58` |
| I-7 PlanBuilder + JIT path | ✅ `a82e0f72c57` |
| I-8 Plan::execute | ✅ `87f45907093` |
| I-9 PerfMeasurement | ✅ `0cbf119bcc7` |
| I-10 Integration test | ✅ `de7bcf4873b` |
| I-11 CI / pre-commit clean | ✅ (this commit) |

## M1 result

End-to-end JIT path works on gfx950 (MI350-series):

- `ninja ck-dsl-provider-unit-check` — green (42 tests, ~10 s wall)
- `ninja ck-dsl-provider-integration-check` — green (1 test, ~9 s wall)
- `IntegrationGpuCkDslConvFp16.BakeOffConv` reports **131 TFLOPS median**
  on the bake-off shape (N=8, 56×56×64→64, 3×3, s=1, p=1, FP16, NHWC).
  Numerical agreement against `CpuFpReferenceConvolution::fprop`
  passes at 5e-2 absolute tolerance over all ~1.6M output elements.
- `pre-commit run` over the full provider tree — clean.

The bake-off example documents 248 TFLOPS on MI300X for the same
configuration (`bake_off_implicit_gemm.py:69-72`). Closing that gap
on MI350 is M2 autotuning work.

## What M1 ships

`dnn-providers/ck-dsl-provider/`:

```
CMakeLists.txt                         finalize_test_targets("ck-dsl-provider")
cmake/                                 version + Python-path helpers
python/ck_dsl_provider/
    compile_service.py                 dispatch on op_kind, conv-igemm builder
src/
    CkDslContainer.{hpp,cpp}           one engine per CK DSL op (M5+ adds siblings)
    CkDslHandle.{hpp,cpp}              stream + container reference + detached buffers
    CkDslContext.hpp                   plan + settings storage
    CkDslPluginPublic.cpp              the only C-ABI source (5 macros + .inl include)
    adapters/conv_implicit_gemm/
        ConvImplicitGemmSpec.hpp       pure-C++ mirror of the dataclass (P-5 defaults)
        ConvImplicitGemmAdapter.cpp    FB ConvolutionFwdAttributes -> Spec
        ConvImplicitGemmPayload.cpp    Spec -> py::dict (Python boundary)
    engines/conv_implicit_gemm/
        CkDslConvImplicitGemmEngine    IEngine -> plan builder
        ConvImplicitGemmPlanBuilder    isApplicable + buildPlan via JitCache
        ConvImplicitGemmPlan           IPlan::execute (uid -> DevPtr -> launch)
    graph/
        GraphSignature                 FNV-1a over op_kind + dtypes + shape + DSL SHA
    perf/
        PerfMeasurement                hipEvent warmup/timed; [CkDslPerf] log line
    python/
        EmbeddedInterpreter            singleton libpython init
        CompileServiceBridge           noopSmoke + compileSmoke + compile(opKind, payload)
        PythonError                    py::error_already_set -> HipdnnPluginException
    runtime/
        KernelArtifact + ArgSchema     P-1's schema-driven HSACO + launch ABI
        LaunchAbi                      contiguous-buffer arg packing
        HipModule                      RAII hipModule_t + hipFunction_t + launch
        JitCache                       mutex-guarded SignatureHash -> shared_ptr<HipModule>
tests/                                 unit tests (host-only + GPU-gated)
integration_tests/                     ninja ck-dsl-provider-integration-check
```

Build artifact: `build/lib/hipdnn_plugins/engines/libck_dsl_provider_plugin.so`.

## Deferred to M1.5 / M2

- **Frontend Graph API + .so plugin loader.** The integration test
  drives `ConvImplicitGemmPlanBuilder` directly rather than going
  through the hipDNN backend's plugin-loader path. The plan-builder
  surface is the same code the backend would call after `dlopen`;
  wiring `hipdnnSetEnginePluginPaths_ext` + `hipdnn_frontend::graph::
  Graph` is additive on top of what M1 already proves.
- **Autotuning.** The constexpr defaults in `ConvImplicitGemmSpec`
  ship the bake-off values verbatim (P-5). Adapter knob surfacing is
  M2.
- **Second op.** M2 adds `CkDslGemmEngine` (or similar) as a sibling
  engine -- the M1 file layout (`engines/<op>/`, `adapters/<op>/`) was
  designed so this is additive.
- **On-disk HSACO cache.** M3 (plan §3.4): `$XDG_CACHE_HOME/
  ck-dsl-provider/<hash>.hsaco`, invalidated on
  `CK_DSL_PROVIDER_VERSION_STRING` change (the same key the in-memory
  cache already uses).

## Hardware constraint to carry forward

`ck_dsl` is gfx950-only (`runtime/comgr.py:210`,
`helpers/compile.py:68,82,129`, `examples/bake_off_implicit_gemm.py:44`
all hardcode `amdgcn-amd-amdhsa--gfx950`; the DSL also emits
MFMA-32×32×16-fp16, `ds_swizzle_b32`, `v_permlane32_swap_b32`,
`ds_read_b64_tr_b{8,16}`, and scaled FP8/BF8 converts unconditionally).
M1 hardware target is **MI350-series**. No fallback to MI300/MI250.
