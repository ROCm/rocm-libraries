# Phase A2 — end-to-end app research findings (from exploration)

Goal: a standalone C++ app builds a hipDNN graph, runs it, and we prove the flyDSL
kernel was SELECTED + used by the kernel-ingestor engine.

## Decisive facts (agent-mapped, to verify as I build)

- **Use a POINTWISE ADD graph.** The kernel-ingestor engine ships only two descriptor
  packs: `hipkernel:Pointwise` and `hipkernel:ConvFwd`
  (`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/descriptors/{pointwise,conv_fwd}`).
  Normalization (RMSNorm/layernorm) is served by the separate `HIP_MLOPS` engine
  (`src/core/Container.cpp:56-72`), NOT the ingestor. So to be dispatched by the
  ingestor, build pointwise ADD.

- **Reference impl to clone:** `dnn-providers/hip-kernel-provider/src/integration_tests/kernel_ingestor_engine/IntegrationGpuKernelIngestor.cpp` — full working graph build+execute through the ingestor.

## Public API sequence (pointwise add)
- Headers: `projects/hipdnn/frontend/include/` (umbrella `hipdnn_frontend.hpp`),
  `hipdnn_frontend/attributes/PointwiseAttributes.hpp`.
- `auto graph = std::make_shared<Graph>(); graph->set_name/set_io_data_type/...(FLOAT)`.
- Tensors via `TensorAttributes` (`set_uid/set_dim/set_stride/set_data_type`).
- `PointwiseAttributes attrs; attrs.set_mode(PointwiseMode::ADD);`
  `auto c = graph->pointwise(a,b,attrs); c->set_uid(3).set_output(true)...`
- Handle: `auto [handle,herr] = hipdnn_frontend::createHipdnnHandle(nullptr);`
- Granular (reveals engine):
  `graph->set_preferred_engine_id_ext(engineNameToId("hipkernel:Pointwise"));`
  then `build_operation_graph(*handle)` → `create_execution_plans()` →
  `check_support()` → `build_plans()`.
- Execute: `get_workspace_size(ws)`; `Workspace ws(bytes)`;
  `variantPack{uid->devptr}`; `graph->execute(*handle, variantPack, ws.get())`.

## Plugin discovery / load
- Backend scans a dir at handle creation. Env `HIPDNN_PLUGIN_DIR` overrides; default
  `hipdnn_plugins/engines/` relative to backend module dir. Every `.so` dlopen'd, probed
  via C ABI `hipdnnEnginePluginGetAllEngineIds`.
- App options: set `HIPDNN_PLUGIN_DIR=/abs/dir/with/libhip_kernel_provider.so`, OR
  `hipdnn_frontend::setEnginePluginPaths({".../libhip_kernel_provider.so"}, MODE_ABSOLUTE)`.
- Audit: `getLoadedEnginePluginPaths(handle, paths)`.

## Ingestor enable + descriptor knobs
- Build-time gate `-DHIPDNN_ENABLE_KERNEL_INGESTOR=ON`. No runtime on/off; the provider
  registers one engine per discovered pack (`Container::getEngineDefinitions()`,
  `src/core/Container.cpp:100-131`).
- `HIPDNN_DESCRIPTOR_DIR` overrides shipped descriptor tree; **`HIPDNN_DESCRIPTOR_RUNTIME_DIR`
  is an ADDITIVE second tree** — drop a flyDSL pack there and it's discovered beside the
  shipped ones (redefining a shipped id is refused; shipped wins).
  (`src/engines/kernel_ingestor_engine/KernelIngestorEngine.hpp:24-36`.)

## Build/link
- Link `hipdnn_frontend` (INTERFACE; pulls `hipdnn_backend` + `hipdnn_data_sdk`).
  Do NOT link the provider — it's dlopen'd at runtime.
- Provider target `hip_kernel_provider` → `libhip_kernel_provider.so`.
- CMake: `find_package(hip REQUIRED); find_package(hipdnn_frontend CONFIG REQUIRED);`
  `target_link_libraries(app PRIVATE hip::host Threads::Threads hipdnn_frontend)`.
  `CMAKE_POSITION_INDEPENDENT_CODE ON` (dlopen/TLS).
- `engineNameToId` / `Workspace`: `<hipdnn_data_sdk/utilities/EngineNames.hpp>`,
  `<hipdnn_data_sdk/utilities/Workspace.hpp>`.

## Proof-of-selection hooks
- `graph->get_ranked_engine_ids(ranked)` — assert our engine id present; pin with
  `set_preferred_engine_id_ext`.
- `HIPDNN_LOG_LEVEL=info` (+ optional `HIPDNN_LOG_FILE`) — plugin emits engine/kernel
  selection lines. Programmatic: `setGlobalLogLevel(HIPDNN_SEV_INFO)` +
  `setUserLogCallback`.
- Benchmarking knob: `create_execution_plan_ext` +
  `KnobSetting(BENCHMARKING_KNOB_NAME, 1)` → emits "benchmarking selected kernel".

## OPEN QUESTION (crux, awaiting agents 1 & 3)
A shipped pack has a runtime DESCRIPTOR (`descriptors/pointwise`) AND a C++ dispatch
handler. How does a descriptor bind to a native C++ `IKernelDispatchHandler`? Two
possibilities for the flyDSL pack:
  (B1) descriptor specifies source-kind HSACO_FILE → hits the `buildIngestorKernelCode`
       throw (unimplemented) → would need to implement that source kind; OR
  (B2) descriptor can name a native/registered dispatch handler → I register a
       Flydsl handler and point a descriptor at it.
Resolving B1 vs B2 (and whether a compiled-in C++ registration path exists) determines
the wiring. See PHASE_A2 open items; do not finalize approach until reconciled.
