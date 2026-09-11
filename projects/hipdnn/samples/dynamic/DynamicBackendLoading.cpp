// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Sample for the runtime-load frontend target (hipdnn_frontend_dynamic).
//
// It links ONLY hipdnn_frontend_dynamic, not hipdnn_test_sdk or
// libhipdnn_backend, so creating the handle must resolve the backend at runtime.
// The graph is a deterministic all-ones 1x1 convolution where every output
// element equals the input channel count C.

#include <cmath>
#include <cstdio>
#include <iostream>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

int main()
{
    try
    {
        RETURN_SUCCESS_IF_NO_DEVICE();

        constexpr int64_t N = 2; // Batch size
        constexpr int64_t C = 4; // Input channels
        constexpr int64_t H = 8; // Height
        constexpr int64_t W = 8; // Width
        constexpr int64_t K = 3; // Output channels
        constexpr int64_t R = 1; // Filter height
        constexpr int64_t S = 1; // Filter width

        // Creating the handle instantiates HipdnnDynamicBackendWrapper and
        // opens libhipdnn_backend at runtime.
        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const auto layout = TensorLayout::NCHW;

        graph::Graph graph;
        graph.set_io_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);

        auto xAttr = createTensor({N, C, H, W}, DataType::FLOAT, layout);
        auto wAttr = createTensor({K, C, R, S}, DataType::FLOAT, layout);

        graph::ConvFpropAttributes convAttributes;
        convAttributes.set_name("dynamic_backend_loading");
        convAttributes.set_padding({0, 0});
        convAttributes.set_stride({1, 1});
        convAttributes.set_dilation({1, 1});

        auto yAttr = graph.conv_fprop(xAttr, wAttr, convAttributes);
        yAttr->set_output(true);

        const auto buildStatus = graph.build(*handle);
        if(buildStatus.get_code() == ErrorCode::GRAPH_NOT_SUPPORTED)
        {
            // No applicable engine on this device: graceful skip (matches the
            // skip contract used by the other samples).
            std::cout << "Skipping: no engine has an applicable solution for this graph on the "
                         "current device. ("
                      << buildStatus.get_message() << ")\n";
            return 0;
        }
        HIPDNN_FE_CHECK(buildStatus);
        std::cout << "Dynamic backend loading graph build successful.\n";

        utilities::Tensor<float> xTensor(xAttr->get_dim(), layout);
        utilities::Tensor<float> wTensor(wAttr->get_dim(), layout);
        utilities::Tensor<float> yTensor(yAttr->get_dim(), layout);

        xTensor.fillWithValue(1.0f);
        wTensor.fillWithValue(1.0f);
        yTensor.fillWithValue(0.0f);

        std::unordered_map<int64_t, void*> variantPack;
        variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
        variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
        variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();

        int64_t workspaceSize = 0;
        HIPDNN_FE_CHECK(graph.get_workspace_size(workspaceSize));
        const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

        HIPDNN_FE_CHECK(graph.execute(*handle, variantPack, workspace.get()));

        yTensor.memory().markDeviceModified();
        const auto* yHost = yTensor.memory().hostData();

        // All-ones 1x1 convolution => every output element == C.
        constexpr auto EXPECTED = static_cast<float>(C);
        constexpr float TOLERANCE = 1e-3F;
        int64_t elementCount = 1;
        for(auto dim : yAttr->get_dim())
        {
            elementCount *= dim;
        }

        bool correct = true;
        for(int64_t i = 0; i < elementCount; ++i)
        {
            if(std::fabs(yHost[i] - EXPECTED) > TOLERANCE)
            {
                std::cerr << "Mismatch at " << i << ": got " << yHost[i] << ", expected "
                          << EXPECTED << '\n';
                correct = false;
                break;
            }
        }

        if(!correct)
        {
            std::cout << "Dynamic backend loading sample produced incorrect results.\n";
            return 1;
        }

        std::cout << "Dynamic backend loading sample executed and verified (" << elementCount
                  << " elements == " << EXPECTED << ").\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
