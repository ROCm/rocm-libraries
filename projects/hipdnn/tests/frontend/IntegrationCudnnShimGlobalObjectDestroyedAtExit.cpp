// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A handle from the cuDNN-compatibility shim's create_cudnn_handle(), owned by a
// namespace-scope object and destroyed during process exit. See
// GlobalObjectDestroyedAtExitFixture.hpp for how the child process is checked.
//
// Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in tests/frontend/CMakeLists.txt.

#include <atomic>
#include <cstdlib>
#include <memory>

#include <gtest/gtest.h>

#include <hipdnn_compatibility/cudnn/cudnn.h>
#include <hipdnn_frontend/detail/HipdnnBackendInterface.hpp>
#include <hipdnn_frontend/detail/HipdnnDirectBackendWrapper.hpp>

#include "GlobalObjectDestroyedAtExitFixture.hpp"

namespace
{

using hipdnn_frontend::detail::HipdnnDirectBackendWrapper;
using hipdnn_frontend::detail::IHipdnnBackend;

// True once DestroyObservingBackend::destroy() runs while IHipdnnBackend::isShutdown() is true,
// that is, once a destroy reached the backend after exit began. Trivially destructible, so a
// destructor running during exit can still read it.
std::atomic<bool> gDestroyCalledAfterShutdown{false};

// The direct-link backend this binary's frontend uses, with destroy() also recording calls made
// after exit began. cudnnDestroy's exit guard returns before looking up the backend, so with
// the guard in place this destroy() is never reached during exit.
class DestroyObservingBackend : public HipdnnDirectBackendWrapper
{
public:
    using HipdnnDirectBackendWrapper::HipdnnDirectBackendWrapper;

    hipdnnStatus_t destroy(hipdnnHandle_t handle) override
    {
        if(IHipdnnBackend::isShutdown())
        {
            gDestroyCalledAfterShutdown.store(true, std::memory_order_release);
        }
        return HipdnnDirectBackendWrapper::destroy(handle);
    }
};

// Destroyed after every object below: fails the child if a destroy reached the backend after
// exit began.
class DestroyGuardProbe
{
public:
    DestroyGuardProbe() = default;
    DestroyGuardProbe(const DestroyGuardProbe&) = delete;
    DestroyGuardProbe& operator=(const DestroyGuardProbe&) = delete;
    DestroyGuardProbe(DestroyGuardProbe&&) = delete;
    DestroyGuardProbe& operator=(DestroyGuardProbe&&) = delete;

    ~DestroyGuardProbe()
    {
        if(gDestroyCalledAfterShutdown.load(std::memory_order_acquire))
        {
            std::_Exit(hipdnn_tests::DESTROY_FAILED_AT_EXIT);
        }
    }
};

DestroyGuardProbe gDestroyGuardProbe;
hipdnn_tests::PluginUnloadProbe gPluginUnloadProbe;
// The frontend's own reference to its backend instance is destroyed before gCudnnHandle. This
// one outlives the handle, so a destroy that reaches the backend during exit runs on a live
// object and is recorded rather than touching freed memory.
std::shared_ptr<DestroyObservingBackend> gObservingBackend;
std::unique_ptr<cudnnHandle_t, CudnnHandleDeleter> gCudnnHandle;

class IntegrationCudnnShimGlobalObjectDestroyedAtExit
    : public hipdnn_tests::IntegrationGlobalObjectDestroyedAtExit
{
};

TEST_F(IntegrationCudnnShimGlobalObjectDestroyedAtExit, HandleExitsCleanly)
{
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    gCudnnHandle = create_cudnn_handle();
    ASSERT_NE(gCudnnHandle, nullptr);
    ASSERT_TRUE(isTestGoodPluginLoaded());

    // The backend instance now exists. Replace it with the observing one, which every later
    // lookup returns, including any lookup made during exit.
    const auto real
        = std::dynamic_pointer_cast<HipdnnDirectBackendWrapper>(IHipdnnBackend::getInstance());
    ASSERT_NE(real, nullptr) << "this binary's frontend must link the backend directly";
    gObservingBackend = std::make_shared<DestroyObservingBackend>(real->version());
    IHipdnnBackend::setInstance(gObservingBackend);
}

} // namespace
