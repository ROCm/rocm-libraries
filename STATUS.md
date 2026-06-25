# SDPA Forward Integration Test — Status

Branch: `users/dahawkin/sdpa-integration-tests` (off `develop`)
Ticket: ALMIOPEN-2127

## Goal
Add the engine-agnostic SDPA forward integration test on the shared
`IntegrationGraphVerificationHarness` (the harness/pattern ALMIOPEN-2127 names):
a frontend `Graph::sdpa()` graph driven through the harness, DUT engine vs the
configured reference executor (CPU today; GPU once #8438 lands).

## Changes (3 files, + this doc)
- `dnn-providers/integration-tests/src/integration_tests/sdpa/IntegrationGpuSdpaFwdInference.cpp` (new)
  - `SdpaForward<DataType>` harness subclass. Builds Q/K/V (BHSD, packed) and
    `graphObj.sdpa(q, k, v, SdpaAttributes)`; output validated against the reference.
  - Cases: MHA, causal top-left, GQA — head dim 128, bf16 + fp16, scale `1/sqrt(d)`.
  - Whole TU guarded by `#ifdef HIPDNN_ENABLE_SDPA` (frontend `sdpa()` is flag-gated).
- `dnn-providers/integration-tests/src/harness/IntegrationGraphVerificationHarness.hpp`
  - Added `SdpaFwdNode -> sdpa::getToleranceFwd<T>()` tolerance branch + include.
- `dnn-providers/integration-tests/src/integration_tests/sdpa/CMakeLists.txt`
  - Registered the new source.

## CI wiring — decision
No redundant wiring is added here, by design:
- **AITER ASM** forward CI coverage already exists and is wired:
  `dnn-providers/hip-kernel-provider/src/integration_tests/asm_sdpa_engine/IntegrationGpuSdpaForward.cpp`
  (provider-local harness, DUT=AITER ASM vs CPU ref, bf16, cases enumerated from the
  engine's own `cfg_fmha_fwd`, registered via `add_integration_test_target(... LABELS slow)`).
  Wiring this shared-exe suite against AITER ASM as well would be strictly redundant.
- The shared `hipdnn_integration_tests` exe is CI-wired only by **miopen-provider**
  (`add_external_integration_test_target`); MIOpen has no SDPA engine, so these cases
  SKIP there (harmless).
- **This** suite's non-redundant role is the engine-agnostic harness path and the
  **GPU-reference** vehicle. Its CI enablement rides with **#8438** turning on the GPU
  reference executor (`--reference-executor gpu`); that is where the ctest target for
  this suite belongs.

## Feature matrix rationale (AITER ASM fwd applicability)
- dtype: **bf16** dispatches to AITER ASM; **fp16** dispatches to CK. (fp32 unsupported.)
- head dim **128** (registry-supported; matches golden bundles).
- **No stats/LSE** — AITER ASM fwd rejects `generate_stats`.
- GQA requires `nhead_q % nhead_kv == 0`; head counts are multiples of 8 to avoid
  the AITER no-mask kernel downgrade.

## Verification so far (gfx1151 / Strix Halo)
- Builds clean with `-DHIPDNN_ENABLE_SDPA=ON` (Ninja); default (flag OFF) build
  unaffected — TU compiles to nothing, harness branch uses unconditional symbols.
- All 6 cases register; each **builds + validates + lowers** the SDPA graph
  (`validate()` + `build_operation_graph()` succeed), then SKIPs at
  `get_ranked_engine_ids` with "No engine supports this graph".
- Expected on gfx1151: AITER ASM fwd `isApplicable` requires **gfx942/gfx950**, and
  this preset loads no SDPA provider. Graph construction + harness wiring proven;
  numerical DUT-vs-reference must run on gfx942/gfx950.

## How to test on gfx942
Build hipDNN + hip-kernel-provider (AITER ASM) + integration tests, SDPA enabled:

```
cmake --preset hip-kernel-provider -B build -G Ninja -DHIPDNN_ENABLE_SDPA=ON
cmake --build build
```

Run this shared-exe suite (DUT = AITER ASM for bf16, CK for fp16; reference = CPU):

```
./build/bin/hipdnn_integration_tests \
    --reference-executor cpu \
    --gtest_filter='*IntegrationGpuSdpaFwd*'
```

Expected: cases PASS (not SKIP). A SKIP means no engine accepted the graph — check
the provider plugin is loaded (`--test-article <plugin.so>` if discovery misses it)
and the device is gfx942/gfx950. To pin AITER ASM as the DUT, add
`--test-engine <asm sdpa engine name>`.

The existing provider-local AITER test also runs via ctest:
`ctest --test-dir build -R hip_kernel_provider_integration_tests`.

## Not done / next
- GPU-reference CI target for this suite — lands with #8438.
- Update ALMIOPEN-2127 scope text (CPU / AITER ASM / golden; GPU ref in #8438).
- Draft PR.
