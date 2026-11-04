// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Container.hpp"
#include "Handle.hpp"
#include "version.h"

using namespace hip_kernel_provider::core;

#define HIPDNN_PLUGIN_NAME "hip_kernel_provider_plugin"
#define HIPDNN_PLUGIN_VERSION HIP_KERNEL_PROVIDER_VERSION_STRING
#define HIPDNN_PLUGIN_CONTAINER_TYPE Container
#define HIPDNN_PLUGIN_HANDLE_TYPE Handle
#define HIPDNN_PLUGIN_CONTEXT_TYPE Context

// hipDNN's newest engine plugin API (PluginVersionConstants.hpp K_MAX_SUPPORTED_API_VERSION,
// asserted below), so the backend routes the provider every graph the SDK can express. Which of
// them an engine sees is the provider's schema guard's to say (GraphSchema.hpp): an engine that
// has not opted in still declines any graph needing more than providerBaselineGraphSchema().
#define HIPDNN_PLUGIN_API_VERSION "1.4.0"
#include <hipdnn_plugin_sdk/EnginePluginImpl.inl>

#include <string_view>

#include <hipdnn_plugin_sdk/PluginVersionConstants.hpp>

static_assert(std::string_view(HIPDNN_PLUGIN_API_VERSION)
                  == hipdnn_plugin_sdk::K_MAX_SUPPORTED_API_VERSION,
              "the SDK defines another engine plugin API: review the provider's schema guard "
              "(GraphSchema.hpp) and the engines that opt in, then raise this with it");

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"

/// Drops every ingestor pack's cached kpack modules and closes every kpack archive the
/// provider retains, so the next dispatch re-reads its archive from disk. FOR TESTS ONLY.
///
/// Deliberately absent from EnginePluginApi.h: this is not part of the plugin ABI, the
/// loader never calls it, and no product code path does either. It is reachable only by
/// a test that dlsym()s it by name, the same way
/// IntegrationGpuKernelIngestorDirectAbi.SelfRegistersAllEngineIds reaches
/// hipdnnEnginePluginGetAllEngineIds.
///
/// Why it has to exist. Module residency is a process-lifetime guarantee -- one
/// hipModule_t per (archive, toc_key, arch), held for the life of the process by
/// design. A test that deliberately corrupts a staged archive to prove the engine
/// reports the damage cannot observe anything if a resident module already serves the
/// plan: the corrupt bytes are read by nothing and the diagnostic never fires.
/// Resetting is the only way to make the next dispatch genuinely re-read the file,
/// short of forking a process per case.
///
/// Clearing releases each cache's own reference only. A module still held by a live
/// plan stays loaded until that plan drops it (KpackModule's destructor does the
/// hipModuleUnload), so this cannot pull a module out from under a running dispatch.
///
/// HIPDNN_PLUGIN_EXPORT is required, not decorative: hip_kernel_provider_impl is built
/// with CXX_VISIBILITY_PRESET hidden (src/CMakeLists.txt:28), so without it this symbol
/// is compiled but never lands in the dynamic table and the dlsym above fails.
extern "C" HIPDNN_PLUGIN_EXPORT void hipdnnEnginePluginResetKpackModuleCacheForTesting()
{
    hip_kernel_provider::kernel_ingestor_engine::resetIngestorModuleCachesForTesting();
}
#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
