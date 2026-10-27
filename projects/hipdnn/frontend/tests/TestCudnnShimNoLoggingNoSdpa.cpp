// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Compile-only: the shim umbrella with NV_CUDNN_FRONTEND_DISABLE_LOGGING defined and
// HIPDNN_ENABLE_SDPA undefined, two configurations no other target builds.
#ifndef NV_CUDNN_FRONTEND_DISABLE_LOGGING
#error "TestCudnnShimNoLoggingNoSdpa.cpp must be built with NV_CUDNN_FRONTEND_DISABLE_LOGGING"
#endif
#ifdef HIPDNN_ENABLE_SDPA
#error "TestCudnnShimNoLoggingNoSdpa.cpp must be built without HIPDNN_ENABLE_SDPA"
#endif

#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

namespace hipdnn_shim_test
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

// External linkage forces the logging helpers' disabled branches to be emitted, not just parsed.
// NOLINTNEXTLINE(misc-use-internal-linkage)
bool loggingCompiledOut(const fe::BackendDescriptor& descriptor)
{
    CUDNN_FE_LOG(descriptor);
    CUDNN_FE_LOG_LABEL(descriptor);
    CUDNN_FE_LOG_LABEL_ENDL(descriptor);
    CUDNN_FE_LOG_BANNER("disabled");
    fe::getLogger() << descriptor << '\n';
    return !fe::isLoggingEnabled() && !fe::isLoggingTensorDumpEnabled();
}

} // namespace hipdnn_shim_test
