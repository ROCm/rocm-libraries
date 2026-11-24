// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/ScopedHipdnnBackendDescriptor.hpp>

#include "GlobalObjectDestroyedAtExitFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_frontend::detail::ScopedHipdnnBackendDescriptor;
using hipdnn_tests::IntegrationGlobalObjectDestroyedAtExit;

namespace
{

hipdnn_tests::BackendReachedProbe gBackendReachedProbe;
hipdnn_tests::PluginUnloadProbe gPluginUnloadProbe;
std::shared_ptr<hipdnn_tests::ExitObservingBackend> gObservingBackend;
HipdnnHandlePtr gHandle;
std::shared_ptr<Graph> gGraph;
hipdnn_tests::LateBackendHandle gBackendHandle;
ScopedHipdnnBackendDescriptor gScopedDescriptor;

// Move-assigns an empty descriptor over gScopedDescriptor from its destructor, which runs during
// exit before gScopedDescriptor's own.
class ScopedDescriptorReplacer
{
public:
    ScopedDescriptorReplacer() = default;
    ScopedDescriptorReplacer(const ScopedDescriptorReplacer&) = delete;
    ScopedDescriptorReplacer& operator=(const ScopedDescriptorReplacer&) = delete;
    ScopedDescriptorReplacer(ScopedDescriptorReplacer&&) = delete;
    ScopedDescriptorReplacer& operator=(ScopedDescriptorReplacer&&) = delete;

    ~ScopedDescriptorReplacer()
    {
        gScopedDescriptor = ScopedHipdnnBackendDescriptor();
    }
};

ScopedDescriptorReplacer gScopedDescriptorReplacer;

std::shared_ptr<Graph> makeConvFpropGraph()
{
    auto graph = std::make_shared<Graph>();
    graph->set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto makeTensor = [](const std::vector<int64_t>& dims, const std::vector<int64_t>& strides) {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_dim(dims).set_stride(strides).set_data_type(DataType::FLOAT);
        return tensor;
    };
    auto x = makeTensor({2, 3, 8, 8}, {192, 64, 8, 1});
    auto w = makeTensor({4, 3, 3, 3}, {27, 9, 3, 1});

    ConvFpropAttributes conv;
    conv.set_padding({1, 1}).set_stride({1, 1}).set_dilation({1, 1});
    graph->conv_fprop(x, w, conv)->set_output(true);
    return graph;
}

TEST_F(IntegrationGlobalObjectDestroyedAtExit, HandleExitsCleanly)
{
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    auto [handle, error] = createHipdnnHandle();
    ASSERT_TRUE(error.is_good()) << error.get_message();
    ASSERT_TRUE(isTestGoodPluginLoaded());
    gHandle = std::move(handle);
    ASSERT_NO_FATAL_FAILURE(observeBackend(gObservingBackend));
}

TEST_F(IntegrationGlobalObjectDestroyedAtExit, BuiltGraphExitsCleanly)
{
    if(!isChild())
    {
        ASSERT_NO_FATAL_FAILURE(useTestGoodEnginePlugin());
        auto [handle, error] = createHipdnnHandle();
        ASSERT_TRUE(error.is_good()) << error.get_message();
        if(makeConvFpropGraph()->build(*handle).is_bad())
        {
            GTEST_SKIP() << "No engine builds the convolution graph on this device";
        }
        handle.reset();
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    auto [handle, error] = createHipdnnHandle();
    ASSERT_TRUE(error.is_good()) << error.get_message();
    gGraph = makeConvFpropGraph();
    const auto buildError = gGraph->build(*handle);
    ASSERT_TRUE(buildError.is_good()) << buildError.get_message();
    handle.reset();
    // The built plan, not the handle, now holds the engine plugin.
    ASSERT_TRUE(isTestGoodPluginLoaded());
}

TEST_F(IntegrationGlobalObjectDestroyedAtExit, BackendHandleExitsCleanly)
{
#if defined(_WIN32)
    GTEST_SKIP() << "The backend's exit handlers run when its DLL detaches, after the "
                    "executable's static destructors, so a late hipdnnDestroy there runs while "
                    "every plugin is still loaded and releases the plugin as before exit";
#else
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    ASSERT_NO_FATAL_FAILURE(prepareChild(gPluginUnloadProbe));
    hipdnnHandle_t handle = nullptr;
    ASSERT_EQ(hipdnnCreate(&handle), HIPDNN_STATUS_SUCCESS);
    gBackendHandle.adopt(handle);
    ASSERT_TRUE(isTestGoodPluginLoaded());
#endif
}

TEST_F(IntegrationGlobalObjectDestroyedAtExit, DescriptorReplacedExitsCleanly)
{
    if(!isChild())
    {
        expectChildExitsCleanly();
        return;
    }

    gScopedDescriptor = ScopedHipdnnBackendDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
    ASSERT_TRUE(gScopedDescriptor.valid());
    ASSERT_NO_FATAL_FAILURE(observeBackend(gObservingBackend));
}

} // namespace
