// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Post-install test for the runtime-load frontend target (hipdnn_frontend_dynamic).
//
// This sample is compiled against the installed frontend headers in runtime-load
// mode and links ONLY hipdnn_frontend_dynamic -- it does NOT link
// libhipdnn_backend. It therefore exercises the property the dynamic target
// exists to provide: the backend is resolved at runtime via dlopen/dlsym (by
// HipdnnDynamicBackendWrapper) rather than through a hard link dependency.
//
// Unlike the other samples it deliberately avoids hipdnn_test_sdk (which links
// the direct frontend, and would pull libhipdnn_backend back onto the link line).
// Correctness is checked inline with a deterministic case instead of the CPU
// reference utilities: an all-ones 1x1 convolution with no padding makes every
// output element equal to the input channel count C.

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
        constexpr int64_t N = 2; // Batch size
        constexpr int64_t C = 4; // Input channels
        constexpr int64_t H = 8; // Height
        constexpr int64_t W = 8; // Width
        constexpr int64_t K = 3; // Output channels
        constexpr int64_t R = 1; // Filter height
        constexpr int64_t S = 1; // Filter width

        // Creating the handle goes through detail::hipdnnBackend(), which (in this
        // build) instantiates HipdnnDynamicBackendWrapper and dlopens libhipdnn_backend.
        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const auto layout = TensorLayout::NCHW;

        graph::Graph graph;
        graph.set_io_data_type(DataType::FLOAT).set_compute_data_type(DataType::FLOAT);

        auto xAttr = createTensor({N, C, H, W}, DataType::FLOAT, layout);
        auto wAttr = createTensor({K, C, R, S}, DataType::FLOAT, layout);

        graph::ConvFpropAttributes convAttributes;
        convAttributes.set_name("conv_fprop_dynamic");
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
        std::cout << "Runtime-load graph build successful (backend resolved via dlopen).\n";

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
        constexpr float expected = static_cast<float>(C);
        constexpr float tolerance = 1e-3F;
        int64_t elementCount = 1;
        for(auto dim : yAttr->get_dim())
        {
            elementCount *= dim;
        }

        bool correct = true;
        for(int64_t i = 0; i < elementCount; ++i)
        {
            if(std::fabs(yHost[i] - expected) > tolerance)
            {
                std::cerr << "Mismatch at " << i << ": got " << yHost[i] << ", expected " << expected
                          << '\n';
                correct = false;
                break;
            }
        }

        if(!correct)
        {
            std::cout << "Runtime-load convolution produced incorrect results.\n";
            return 1;
        }

        std::cout << "Runtime-load convolution executed and verified ("
                  << elementCount << " elements == " << expected << ").\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
