# Megakernel engine (ENABLE_MEGAKERNEL_ENGINE): build and test notes

Design: megakernels `docs/hipdnn-route.md` (sections 3-5, decisions D3, D5, D6, D7).
Branch `users/sareeder/megakernel-engine` from `origin/develop` 6a27c440ff9.

## What is here

| Part | Where |
|---|---|
| Option (default OFF), family entry, compile definition: one name `ENABLE_MEGAKERNEL_ENGINE` | `CMakeLists.txt`, `descriptor-packaging/cmake/HkpPackaging.cmake`, `src/engines/kernel_ingestor_engine/CMakeLists.txt` |
| Native UMD (graph + kernel scope) and UDD | `src/engines/kernel_ingestor_engine/packs/MegakernelNative.cpp` |
| Descriptors, lock, DVC pointer | `src/engines/kernel_ingestor_engine/descriptors/megakernel/` |
| Lock check in the pack step | `descriptor-packaging/tools/hkp_megakernel_lock.py`, wired in `hkp_wire_pack_target` |
| Configure DVC check per enabled family (rocKE, megakernel; size or file count) | `CMakeLists.txt` |
| Unit tests: matcher declines | `src/tests/engines/kernel_ingestor_engine/packs/TestMegakernelMatchers.cpp` |
| Tests: lock mismatch fails the tool and the build | `descriptor-packaging/tests/test_hkp_megakernel_lock.py` |
| GPU integration test (stand-in object, Graph.custom_op + execute) | `src/integration_tests/kernel_ingestor_engine/IntegrationGpuMegakernel.cpp`, `test_descriptors/integration/megakernel/` |

Payload (schema 1): `{schema, model, src, arch, cus, kv, block, M, tp, rt, lm_head, nlayers}`;
op id `megakernel.<model>`. Graph scope declines (INFO line
`hipkernel:Megakernel declined custom op '<id>' [<field>]: ...`) on: no/!object JSON
`[payload]`, schema != 1 `[schema]`, missing or mistyped key `[<key>]`, op id model differs
`[model]`, arch != device `[arch]`, cus != device multiProcessorCount `[cus]`, nlayers not a
positive i32 `[nlayers]`. Kernel scope compares model, cus, kv, block, M, tp, rt, lm_head, src
with the KMD record and logs the first differing field (`src` last; kernels of other models
are skipped silently). `prepare()` declines (NOT_APPLICABLE, `[occupancy]`) when
`hipModuleOccupancyMaxActiveBlocksPerMultiprocessor` < 1 for block_threads / lds_bytes. Launch:
grid (cus), block (block_threads), dynamic LDS lds_bytes (0 for FlyDSL objects; agreed with
CoExporter), args (tab addr i64, glob addr i64, nlayers i32) on the handle stream, workspace 0.

## Decisions taken here

- Lock: `manifest.json` is CoExporter's rt0 set (megakernels `users/sareeder/co-export`
  1f9108f, `common/results/co/rt0/final/`, manifest sha256 e689d054dfca...): 16 objects,
  Qwen3.8 TP1 and TP2, Qwen3.6, Gemma4 at M 1, 2, 4, 8; `src` is vLLM's `compile_key`.
  Earlier heads carried the spike's four Qwen3.8 objects as a stand-in.
- The pack step skips the lock check when `artifacts/` is absent (no `dvc pull`): configure
  already warns, as for rocKE, and nothing is packed.
- `lm_head`, `cus`, ... are KMD fields of type bool/int, all `matcher_only_fields`.
- The GPU test uses a stand-in object with the same ABI, because a real megakernel needs
  model weights behind `tab`. Its engine is `hipkernel:megakernel_standin` (integration set
  only, packed only with the option ON).
- `Kernel::function()` accessor added so the pack can query occupancy of the loaded function.

## Results (Alola CPU node ctr-smc-s22-17, hipdnn_latest_gfx942.sqsh, GPU_TARGETS=gfx942, ingestor ON)

Log: `logs/run-off-on.log` (job 68218157, head a90fe00804e).

| Build | Build | Unit tests (`hip_kernel_provider_tests`) | Megakernel cases |
|---|---|---|---|
| OFF | rc 0 | 1109 passed, 68 skipped (no device), 0 failed (`logs/unit-OFF.log`) | 0 (not compiled) |
| ON | rc 0 | 1119 passed, 68 skipped, 0 failed (`logs/unit-ON.log`) | 10 passed (`logs/unit-megakernel-ON.log`) |

- OFF configure: `megakernel` listed as a disabled family folder (`logs/configure-OFF-hkp.log`);
  the OFF build differs from develop only in code under `#ifdef ENABLE_MEGAKERNEL_ENGINE`, the
  family table and the per-family DVC loop. Same pass/skip set as the ON build minus the 10
  megakernel cases.
- ON: product root packs `megakernel/artifacts/megakernel.kdp.json` for gfx942, integration
  root packs the stand-in (`logs/run-off-on.log`, `logs/build-ON-megakernel.log`).
- clang-tidy-20 (repo `.clang-tidy`, warnings as errors) on the 5 touched TUs: rc 0
  (`logs/tidy-ON.log`). A full-tree tidy build found 11 findings in earlier heads, all fixed
  (jobs 68216113, 68216970). clang-format 18 applied to every touched C++ file.
- Lock tests: 11 passed (`logs/pytest-lock.log`): sha mismatch, extra object, missing
  object, stale KDP, malformed lock, unpulled bundle passes, and a sub-build whose pack step
  fails on a corrupt object with the family ON and packs with it OFF.
- DVC: `artifacts.dvc` round trip through a LOCAL remote (`/tmp/mk-dvc-remote`, scratch DVC
  repo): push 6 files, delete artifacts and cache, pull, `hkp_megakernel_lock.py verify` OK;
  the pointer is byte-identical after regenerating the KDP. Nothing was pushed to `ingestor`.

## Not done

- GPU integration test (`IntegrationGpuMegakernel.*`) not run yet: gfx942 nodes are held
  until about 11-01 (Main's sharing rule). Run on a gfx942 node with the ON build:
  `HIP_VISIBLE_DEVICES=0 bin/hip_kernel_provider_integration_tests --gtest_filter='IntegrationGpuMegakernel.*'`.

## DVC upload, once write access to the ingestor bucket exists

From the repository root, with credentials in the environment (never in a config file):

```
dvc push -r ingestor dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/megakernel/artifacts.dvc
```

Then check anonymously that `dvc pull -r ingestor <same pointer>` restores the files and that
`python3 dnn-providers/hip-kernel-provider/descriptor-packaging/tools/hkp_megakernel_lock.py verify --lock <...>/megakernel/manifest.json` passes.
