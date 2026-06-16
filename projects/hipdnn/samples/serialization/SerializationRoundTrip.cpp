// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <iostream>
#include <string>
#include <unordered_map>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

namespace
{

template <typename InputType>
void executeConvFpropGraph(graph::Graph& graph,
                           hipdnnHandle_t handle,
                           const std::shared_ptr<graph::Tensor_attributes>& xAttr,
                           const std::shared_ptr<graph::Tensor_attributes>& wAttr,
                           const std::shared_ptr<graph::Tensor_attributes>& yAttr,
                           utilities::Tensor<InputType>& xTensor,
                           utilities::Tensor<InputType>& wTensor,
                           utilities::Tensor<InputType>& yTensor)
{
    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor.memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor.memory().deviceData();
    variantPack[yAttr->get_uid()] = yTensor.memory().deviceData();

    int64_t workspaceSize = 0;
    HIPDNN_FE_CHECK(graph.get_workspace_size(workspaceSize));
    const utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    HIPDNN_FE_CHECK(graph.execute(handle, variantPack, workspace.get()));

    yTensor.memory().markDeviceModified();
}

#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
template <typename InputType>
bool testJsonSerialization(const graph::Graph& originalGraph,
                           hipdnnHandle_t handle,
                           utilities::Tensor<InputType>& xTensor,
                           utilities::Tensor<InputType>& wTensor,
                           utilities::Tensor<InputType>& yOriginal,
                           const TensorLayout& layout)
{
    std::cout << "\n--- Testing JSON serialization/deserialization ---\n";

    std::string jsonData;
    HIPDNN_FE_CHECK(originalGraph.serialize(jsonData));
    std::cout << "Serialized to JSON (" << jsonData.size() << " bytes)\n";

    graph::Graph jsonGraph;
    // deserialize(handle, ...) produces a finalized, ready-to-use graph —
    // no separate validate() or build_operation_graph() call is needed.
    HIPDNN_FE_CHECK(jsonGraph.deserialize(handle, jsonData));
    std::cout << "Deserialized from JSON.\n";

    HIPDNN_FE_CHECK(jsonGraph.create_execution_plans());
    HIPDNN_FE_CHECK(jsonGraph.check_support());
    HIPDNN_FE_CHECK(jsonGraph.build_plans());

    auto jsonTensors = jsonGraph.getTensorsByName();
    auto xAttrJson = jsonTensors.at("x");
    auto wAttrJson = jsonTensors.at("w");
    auto yAttrJson = jsonTensors.at("y");

    utilities::Tensor<InputType> yJson(yAttrJson->get_dim(), layout);
    yJson.fillWithValue(static_cast<InputType>(0.0f));

    executeConvFpropGraph(
        jsonGraph, handle, xAttrJson, wAttrJson, yAttrJson, xTensor, wTensor, yJson);
    std::cout << "JSON-deserialized graph execution complete.\n";

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator<InputType>();
    const bool passed = validator->allClose(yOriginal, yJson);
    std::cout << "JSON round-trip result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}
#endif // HIPDNN_FRONTEND_SKIP_JSON_LIB

template <typename InputType>
bool testBinarySerialization(const graph::Graph& originalGraph,
                             hipdnnHandle_t handle,
                             utilities::Tensor<InputType>& xTensor,
                             utilities::Tensor<InputType>& wTensor,
                             utilities::Tensor<InputType>& yOriginal,
                             const TensorLayout& layout)
{
    std::cout << "\n--- Testing binary serialization/deserialization ---\n";

    std::vector<uint8_t> binaryData;
    HIPDNN_FE_CHECK(originalGraph.serialize(binaryData));
    std::cout << "Serialized to binary (" << binaryData.size() << " bytes)\n";

    graph::Graph binaryGraph;
    // deserialize(handle, ...) produces a finalized, ready-to-use graph —
    // no separate validate() or build_operation_graph() call is needed.
    HIPDNN_FE_CHECK(binaryGraph.deserialize(handle, binaryData));
    std::cout << "Deserialized from binary.\n";

    HIPDNN_FE_CHECK(binaryGraph.create_execution_plans());
    HIPDNN_FE_CHECK(binaryGraph.check_support());
    HIPDNN_FE_CHECK(binaryGraph.build_plans());

    auto binaryTensors = binaryGraph.getTensorsByName();
    auto xAttrBin = binaryTensors.at("x");
    auto wAttrBin = binaryTensors.at("w");
    auto yAttrBin = binaryTensors.at("y");

    utilities::Tensor<InputType> yBinary(yAttrBin->get_dim(), layout);
    yBinary.fillWithValue(static_cast<InputType>(0.0f));

    executeConvFpropGraph(
        binaryGraph, handle, xAttrBin, wAttrBin, yAttrBin, xTensor, wTensor, yBinary);
    std::cout << "Binary-deserialized graph execution complete.\n";

    auto validator = hipdnn_test_sdk::utilities::createAllCloseValidator<InputType>();
    const bool passed = validator->allClose(yOriginal, yBinary);
    std::cout << "Binary round-trip result: " << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    const auto inputType = getDataTypeEnumFromType<InputType>();

    std::cout << "\n=== Running serialization round-trip test " << inputType << " [" << layout
              << "] ===\n";

    constexpr int64_t N = 4;
    constexpr int64_t C = 8;
    constexpr int64_t H = 8;
    constexpr int64_t W = 8;
    constexpr int64_t K = 8;
    constexpr int64_t R = 3;
    constexpr int64_t S = 3;
    constexpr int64_t U = 1;
    constexpr int64_t V = 1;
    constexpr int64_t PAD_H = 1;
    constexpr int64_t PAD_W = 1;
    constexpr int64_t DIL_H = 1;
    constexpr int64_t DIL_W = 1;

    // Set names on tensors so we can retrieve them after deserialization
    auto xAttrOrig = createTensor({N, C, H, W}, inputType, layout);
    auto wAttrOrig = createTensor({K, C, R, S}, inputType, layout);
    xAttrOrig->set_name("x");
    wAttrOrig->set_name("w");

    utilities::Tensor<InputType> xTensor(xAttrOrig->get_dim(), layout);
    utilities::Tensor<InputType> wTensor(wAttrOrig->get_dim(), layout);
    xTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    wTensor.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));

    // ==================== ORIGINAL GRAPH ====================
    std::cout << "\n--- Building and executing original graph ---\n";
    graph::Graph originalGraph;
    originalGraph.set_name("original_conv_graph");
    originalGraph.set_io_data_type(inputType)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    graph::ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fprop")
        .set_padding({PAD_H, PAD_W})
        .set_stride({U, V})
        .set_dilation({DIL_H, DIL_W});

    auto yAttrOrig = originalGraph.conv_fprop(xAttrOrig, wAttrOrig, convAttrs);
    yAttrOrig->set_name("y");
    yAttrOrig->set_output(true);

    HIPDNN_FE_CHECK_SKIPPABLE(originalGraph.build(handle));

    utilities::Tensor<InputType> yOriginal(yAttrOrig->get_dim(), layout);
    yOriginal.fillWithValue(static_cast<InputType>(0.0f));

    executeConvFpropGraph(
        originalGraph, handle, xAttrOrig, wAttrOrig, yAttrOrig, xTensor, wTensor, yOriginal);
    std::cout << "Original graph execution complete.\n";

    // ==================== RUN SERIALIZATION TESTS ====================
#ifndef HIPDNN_FRONTEND_SKIP_JSON_LIB
    const bool jsonMatch = testJsonSerialization<InputType>(
        originalGraph, handle, xTensor, wTensor, yOriginal, layout);
#else
    std::cout << "\n--- Skipping JSON serialization test (JSON support disabled) ---\n";
    const bool jsonMatch = true;
#endif

    const bool binaryMatch = testBinarySerialization<InputType>(
        originalGraph, handle, xTensor, wTensor, yOriginal, layout);

    return jsonMatch && binaryMatch;
}

int main(int argc, char* argv[])
{
    try
    {
        auto config = parseCommandLineArgs(argc, argv);

        auto [handle, handleError] = createHipdnnHandle();
        HIPDNN_FE_CHECK(handleError);

        const bool allPassed = run(SampleRunner{*handle, config});

        if(allPassed)
        {
            std::cout << "\nAll serialization round-trip tests PASSED.\n";
            return 0;
        }
        std::cout << "\nSome serialization round-trip tests FAILED.\n";
        return 1;
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "Unhandled exception: %s\n", e.what());
        return 1;
    }
}
