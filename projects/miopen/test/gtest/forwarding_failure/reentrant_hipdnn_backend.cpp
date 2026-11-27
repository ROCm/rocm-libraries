// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// A hipDNN backend that, like a plugin built against MIOpen, owns an MIOpen handle
// and destroys it when hipDNN destroys its own handle. hipDNN accepts it: it reports
// the frontend's major version and can create, destroy and set a stream on a
// handle. It has none of the graph functions, so every graph build fails.
//
// MIOpen destroys its hipDNN handles from miopenDestroy, so hipdnnDestroy here
// re-enters miopenDestroy. If MIOpen still held a lock there, that call would
// never return. script/test_forwarding_failure_modes.py checks that it does.
//
// With REENTRANT_HIPDNN_BACKEND_FAIL_SET_STREAM set, setting a stream fails, so
// the hipDNN handle MIOpen just created for a handle with a stream is destroyed
// again. That destroy re-enters miopenDestroy too.
//
// The backend header is not included: it declares these functions as imports on
// Windows, and this library defines them.

#include <hipdnn_frontend/version.h>
#include <miopen/miopen.h>

#include <cstdio>
#include <cstdlib>
#include <string>

#ifdef _WIN32
#define REENTRANT_BACKEND_EXPORT extern "C" __declspec(dllexport)
#else
#define REENTRANT_BACKEND_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// hipdnnHandle_t is a pointer to this struct, which the backend defines.
struct hipdnnHandle
{
    miopenHandle_t miopen = nullptr;
};

namespace {

// hipdnnStatus_t values; the frontend only tells success from failure here.
constexpr int Success = 0;
constexpr int Failure = 1;

} // namespace

REENTRANT_BACKEND_EXPORT const char* hipdnnVersionString_ext()
{
    static const std::string version = std::to_string(HIPDNN_FRONTEND_VERSION_MAJOR) + ".0.0";
    return version.c_str();
}

REENTRANT_BACKEND_EXPORT int hipdnnCreate(hipdnnHandle** handle)
{
    if(handle == nullptr)
        return Failure;
    auto* created = new hipdnnHandle;
    if(miopenCreate(&created->miopen) != miopenStatusSuccess)
    {
        delete created;
        return Failure;
    }
    *handle = created;
    return Success;
}

REENTRANT_BACKEND_EXPORT int hipdnnDestroy(hipdnnHandle* handle)
{
    if(handle == nullptr)
        return Failure;
    const miopenStatus_t status = miopenDestroy(handle->miopen);
    std::fprintf(
        stderr, "REENTRANT_HIPDNN_BACKEND_DESTROYED_MIOPEN_HANDLE=%d\n", static_cast<int>(status));
    delete handle;
    return Success;
}

REENTRANT_BACKEND_EXPORT int hipdnnSetStream(hipdnnHandle* handle, void* /*stream*/)
{
    static const bool failSetStream =
        std::getenv("REENTRANT_HIPDNN_BACKEND_FAIL_SET_STREAM") != nullptr;
    return handle == nullptr || failSetStream ? Failure : Success;
}

REENTRANT_BACKEND_EXPORT int hipdnnGetStream(hipdnnHandle* handle, void** stream)
{
    if(handle == nullptr || stream == nullptr)
        return Failure;
    *stream = nullptr;
    return Success;
}
