// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A serialized conv dgrad graph whose stride is far larger than its dims allow is rejected with
// HIPDNN_STATUS_BAD_PARAM when it is deserialized, because the descriptor's finalize checks the
// stride against the dims. Without that check, engine applicability for such a graph has been
// observed not to return for minutes, so the whole sequence up to and including the ENGINEHEUR
// finalize runs on a worker thread with an internal deadline. A worker that misses the deadline
// can be neither joined nor cancelled, so the test reports the failure and ends the process.

#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <string>
#include <type_traits>

namespace
{

namespace fb = hipdnn_flatbuffers_sdk::data_objects;

constexpr auto K_DEADLINE = std::chrono::seconds(60);

struct DescriptorDeleter
{
    void operator()(hipdnnBackendDescriptor_t desc) const
    {
        hipdnnBackendDestroyDescriptor(desc);
    }
};
using ScopedDescriptor
    = std::unique_ptr<std::remove_pointer_t<hipdnnBackendDescriptor_t>, DescriptorDeleter>;

struct HandleDeleter
{
    void operator()(hipdnnHandle_t handle) const
    {
        hipdnnDestroy(handle);
    }
};
using ScopedHandle = std::unique_ptr<std::remove_pointer_t<hipdnnHandle_t>, HandleDeleter>;

struct SequenceOutcome
{
    bool setupFailed = false;
    hipdnnStatus_t deserializeStatus = HIPDNN_STATUS_SUCCESS;
    std::string detail;
};

std::string stepResult(const char* step, hipdnnStatus_t status)
{
    return std::string(step) + " returned " + hipdnnGetErrorString(status);
}

// dy [4,4,4,4], w [4,4,1,1], dx [4,4,4,4] with stride [1, widthStride]; inconsistent for any
// width stride above 3.
flatbuffers::FlatBufferBuilder createHugeStrideDgradGraph(int64_t widthStride)
{
    const auto base = hipdnn_test_sdk::utilities::createValidConvBwdGraph();
    auto graphT = fb::UnPackGraph(base.GetBufferPointer());
    graphT->nodes.at(0)->attributes.AsConvolutionBwdAttributes()->stride[1] = widthStride;
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(fb::Graph::Pack(builder, graphT.get()));
    return builder;
}

// Deserialize, finalize the graph, then finalize an ENGINEHEUR descriptor in FALLBACK mode.
// Returns at the first step that declines, or after the heuristic finalize returns.
SequenceOutcome runDgradSequence(int64_t widthStride)
{
    SequenceOutcome outcome;

    hipdnnHandle_t rawHandle = nullptr;
    const auto createStatus = hipdnnCreate(&rawHandle);
    const ScopedHandle handle(rawHandle);
    if(createStatus != HIPDNN_STATUS_SUCCESS)
    {
        outcome.setupFailed = true;
        outcome.detail = stepResult("hipdnnCreate", createStatus);
        return outcome;
    }

    const auto builder = createHugeStrideDgradGraph(widthStride);

    hipdnnBackendDescriptor_t rawGraph = nullptr;
    auto status = hipdnnBackendCreateAndDeserializeGraph_ext(
        &rawGraph, builder.GetBufferPointer(), builder.GetSize());
    const ScopedDescriptor graph(rawGraph);
    outcome.deserializeStatus = status;
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        outcome.detail = stepResult("hipdnnBackendCreateAndDeserializeGraph_ext", status);
        return outcome;
    }

    status = hipdnnBackendSetAttribute(graph.get(),
                                       HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                       HIPDNN_TYPE_HANDLE,
                                       1,
                                       static_cast<const void*>(&rawHandle));
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        outcome.detail = stepResult("set HIPDNN_ATTR_OPERATIONGRAPH_HANDLE", status);
        return outcome;
    }

    status = hipdnnBackendFinalize(graph.get());
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        outcome.detail = stepResult("graph hipdnnBackendFinalize", status);
        return outcome;
    }

    hipdnnBackendDescriptor_t rawHeuristic = nullptr;
    status = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR, &rawHeuristic);
    const ScopedDescriptor heuristic(rawHeuristic);
    if(status != HIPDNN_STATUS_SUCCESS)
    {
        outcome.setupFailed = true;
        outcome.detail = stepResult("create ENGINEHEUR descriptor", status);
        return outcome;
    }

    hipdnnBackendDescriptor_t graphForHeuristic = graph.get();
    const hipdnnBackendHeurMode_t mode = HIPDNN_HEUR_MODE_FALLBACK;
    hipdnnBackendSetAttribute(heuristic.get(),
                              HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                              HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                              1,
                              static_cast<const void*>(&graphForHeuristic));
    hipdnnBackendSetAttribute(
        heuristic.get(), HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_HEUR_MODE, 1, &mode);

    outcome.detail
        = stepResult("ENGINEHEUR hipdnnBackendFinalize", hipdnnBackendFinalize(heuristic.get()));
    return outcome;
}

void expectRejectedAtDeserializeWithinDeadline(int64_t widthStride)
{
    auto worker = std::async(std::launch::async, runDgradSequence, widthStride);
    if(worker.wait_for(K_DEADLINE) == std::future_status::timeout)
    {
        // The future's destructor would block on the hung worker, so end the process here.
        std::fprintf(stderr,
                     "FAILED: conv dgrad with stride [1, %lld] did not finish deserialize, "
                     "finalize and ENGINEHEUR finalize within %lld s\n",
                     static_cast<long long>(widthStride),
                     static_cast<long long>(K_DEADLINE.count()));
        std::fflush(nullptr);
        std::_Exit(EXIT_FAILURE);
    }

    const auto outcome = worker.get();
    EXPECT_FALSE(outcome.setupFailed) << "stride [1, " << widthStride << "]: " << outcome.detail;
    EXPECT_EQ(outcome.deserializeStatus, HIPDNN_STATUS_BAD_PARAM)
        << "stride [1, " << widthStride << "]: " << outcome.detail;
}

} // namespace

TEST(IntegrationConvDgradHugeStride, DeserializeRejectsWithinDeadline)
{
    SKIP_IF_NO_DEVICES();

    expectRejectedAtDeserializeWithinDeadline(100000);
    expectRejectedAtDeserializeWithinDeadline(1048577);
}
