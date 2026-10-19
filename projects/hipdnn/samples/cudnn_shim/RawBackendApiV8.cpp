// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   samples/legacy_samples/test_list.cpp
//     Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: frontend/THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

// cuDNN v8 backend C API sample: the legacy "Tensor creation comparison" case
// (host only), then an fp32 NCHW convolution run purely through cudnnBackend*.

#include "V8SampleUtils.hpp"

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace fe = v8_sample::fe;

void checkStatus(cudnnStatus_t status)
{
    if(status != CUDNN_STATUS_SUCCESS)
    {
        std::array<char, 1024> lastError{};
        cudnnGetLastErrorString(lastError.data(), lastError.size());
        throw std::runtime_error(std::string("cuDNN backend call failed: ")
                                 + cudnnGetErrorString(status) + ": " + lastError.data());
    }
}

// Upstream test_list.cpp "Tensor creation comparison".
void tensorCreationComparison()
{
    // Consider creation of a 2d Tensor
    // n,c,h,w as 4,32,32,32
    std::cout << "Tensor creation comparison\n";
    std::array<int64_t, 4> tensorDim = {4, 32, 32, 32};
    std::array<int64_t, 4> tensorStr = {32768, 1024, 32, 1}; // NCHW format
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    int64_t alignment = sizeof(float);
    int64_t id = 0xD0D0CACA; // Some magic number

    // Creating Frontend code

    try
    {
        auto tensor = fe::TensorBuilder()
                          .setDim(static_cast<int64_t>(tensorDim.size()), tensorDim.data())
                          .setStrides(static_cast<int64_t>(tensorStr.size()), tensorStr.data())
                          .setId(id)
                          .setAlignment(alignment)
                          .setDataType(dataType)
                          .build();
    }
    catch(fe::cudnnException& e)
    {
        std::cout << "Exception in tensor creation " << e.what() << '\n';
    }

    // Equivalent Backend code
    {
        cudnnBackendDescriptor_t tensor;

        // Allocate memory for the descriptor
        // This is a c-style malloc which requires
        // a equivalent 1-time deletion. Raw backend code
        // requires tracking allocation and free unlike raw
        // pointers, else it may lead to memory leak.
        checkStatus(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &tensor));

        // Set the following attributes
        // Dimensions, Strides, Alignment, Id, DataType
        checkStatus(cudnnBackendSetAttribute(
            tensor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType));
        checkStatus(cudnnBackendSetAttribute(tensor,
                                             CUDNN_ATTR_TENSOR_DIMENSIONS,
                                             CUDNN_TYPE_INT64,
                                             static_cast<int64_t>(tensorDim.size()),
                                             tensorDim.data()));
        checkStatus(cudnnBackendSetAttribute(tensor,
                                             CUDNN_ATTR_TENSOR_STRIDES,
                                             CUDNN_TYPE_INT64,
                                             static_cast<int64_t>(tensorStr.size()),
                                             tensorStr.data()));
        checkStatus(cudnnBackendSetAttribute(
            tensor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id));
        checkStatus(cudnnBackendSetAttribute(
            tensor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
        // Finalize the descriptor
        checkStatus(cudnnBackendFinalize(tensor));

        // Free the memory allocated above. Any short-circuit return will
        // cause a memory leak.
        checkStatus(cudnnBackendDestroyDescriptor(tensor));
    }

    std::cout << "\n==============================================================================="
                 "=========\n";
}

class BackendDescriptor
{
public:
    explicit BackendDescriptor(cudnnBackendDescriptorType_t type)
    {
        checkStatus(cudnnBackendCreateDescriptor(type, &_descriptor));
    }

    ~BackendDescriptor()
    {
        cudnnBackendDestroyDescriptor(_descriptor);
    }

    BackendDescriptor(const BackendDescriptor&) = delete;
    BackendDescriptor& operator=(const BackendDescriptor&) = delete;
    BackendDescriptor(BackendDescriptor&&) = delete;
    BackendDescriptor& operator=(BackendDescriptor&&) = delete;

    cudnnBackendDescriptor_t get() const
    {
        return _descriptor;
    }

    // One-element BACKEND_DESCRIPTOR array for cudnnBackendSetAttribute.
    const void* address() const
    {
        return static_cast<const void*>(&_descriptor);
    }

private:
    cudnnBackendDescriptor_t _descriptor = nullptr;
};

void setTensorAttributes(const BackendDescriptor& desc,
                         const hipdnn_data_sdk::utilities::Tensor<float>& tensor,
                         int64_t uid)
{
    const cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    const int64_t alignment = v8_sample::K_ALIGNMENT;
    const auto rank = static_cast<int64_t>(tensor.dims().size());

    checkStatus(cudnnBackendSetAttribute(
        desc.get(), CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dataType));
    checkStatus(cudnnBackendSetAttribute(
        desc.get(), CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, rank, tensor.dims().data()));
    checkStatus(cudnnBackendSetAttribute(
        desc.get(), CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, rank, tensor.strides().data()));
    checkStatus(cudnnBackendSetAttribute(
        desc.get(), CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid));
    checkStatus(cudnnBackendSetAttribute(
        desc.get(), CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment));
    checkStatus(cudnnBackendFinalize(desc.get()));
}

bool runRawConvFprop(cudnnHandle_t handle, const Config& config)
{
    const int64_t n = !config.dims.empty() ? config.dims[0] : 16;
    const int64_t c = config.dims.size() > 1 ? config.dims[1] : 16;
    const int64_t h = config.dims.size() > 2 ? config.dims[2] : 16;
    const int64_t w = config.dims.size() > 3 ? config.dims[3] : 16;
    const int64_t k = !config.filter.empty() ? config.filter[0] : 16;
    const int64_t r = config.filter.size() > 1 ? config.filter[1] : 3;
    const int64_t s = config.filter.size() > 2 ? config.filter[2] : 3;
    const int64_t u = !config.stride.empty() ? config.stride[0] : 1;
    const int64_t v = config.stride.size() > 1 ? config.stride[1] : 1;
    const int64_t padH = !config.padding.empty() ? config.padding[0] : 1;
    const int64_t padW = config.padding.size() > 1 ? config.padding[1] : 1;
    const int64_t dilH = !config.dilation.empty() ? config.dilation[0] : 1;
    const int64_t dilW = config.dilation.size() > 1 ? config.dilation[1] : 1;
    const int64_t outH = (h + 2 * padH - dilH * (r - 1) - 1) / u + 1;
    const int64_t outW = (w + 2 * padW - dilW * (s - 1) - 1) / v + 1;

    std::cout << "Running raw backend API convolution fprop fp32 [NCHW]...\n";

    const std::vector<int64_t> pad = {padH, padW};
    const std::vector<int64_t> stride = {u, v};
    const std::vector<int64_t> dilation = {dilH, dilW};
    const auto spatialDims = static_cast<int64_t>(pad.size());

    hipdnn_data_sdk::utilities::Tensor<float> xTensor({n, c, h, w});
    hipdnn_data_sdk::utilities::Tensor<float> wTensor({k, c, r, s});
    hipdnn_data_sdk::utilities::Tensor<float> yTensor({n, k, outH, outW});

    const std::array<int64_t, 3> uids = {'x', 'w', 'y'};
    const BackendDescriptor xDesc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    const BackendDescriptor wDesc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    const BackendDescriptor yDesc(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    setTensorAttributes(xDesc, xTensor, uids[0]);
    setTensorAttributes(wDesc, wTensor, uids[1]);
    setTensorAttributes(yDesc, yTensor, uids[2]);

    const cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
    const cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    const BackendDescriptor convDesc(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    checkStatus(cudnnBackendSetAttribute(
        convDesc.get(), CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &computeType));
    checkStatus(cudnnBackendSetAttribute(
        convDesc.get(), CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    checkStatus(cudnnBackendSetAttribute(
        convDesc.get(), CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatialDims));
    checkStatus(cudnnBackendSetAttribute(convDesc.get(),
                                         CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                                         CUDNN_TYPE_INT64,
                                         spatialDims,
                                         pad.data()));
    checkStatus(cudnnBackendSetAttribute(convDesc.get(),
                                         CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                                         CUDNN_TYPE_INT64,
                                         spatialDims,
                                         pad.data()));
    checkStatus(cudnnBackendSetAttribute(convDesc.get(),
                                         CUDNN_ATTR_CONVOLUTION_DILATIONS,
                                         CUDNN_TYPE_INT64,
                                         spatialDims,
                                         dilation.data()));
    checkStatus(cudnnBackendSetAttribute(convDesc.get(),
                                         CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                                         CUDNN_TYPE_INT64,
                                         spatialDims,
                                         stride.data()));
    checkStatus(cudnnBackendFinalize(convDesc.get()));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const BackendDescriptor convOp(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    checkStatus(cudnnBackendSetAttribute(convOp.get(),
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         xDesc.address()));
    checkStatus(cudnnBackendSetAttribute(convOp.get(),
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         wDesc.address()));
    checkStatus(cudnnBackendSetAttribute(convOp.get(),
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         yDesc.address()));
    checkStatus(cudnnBackendSetAttribute(convOp.get(),
                                         CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         convDesc.address()));
    checkStatus(cudnnBackendSetAttribute(
        convOp.get(), CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha));
    checkStatus(cudnnBackendSetAttribute(
        convOp.get(), CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta));
    checkStatus(cudnnBackendFinalize(convOp.get()));

    const BackendDescriptor opGraph(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    checkStatus(cudnnBackendSetAttribute(opGraph.get(),
                                         CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                                         CUDNN_TYPE_HANDLE,
                                         1,
                                         static_cast<const void*>(&handle)));
    checkStatus(cudnnBackendSetAttribute(opGraph.get(),
                                         CUDNN_ATTR_OPERATIONGRAPH_OPS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         convOp.address()));
    checkStatus(cudnnBackendFinalize(opGraph.get()));

    const cudnnBackendHeurMode_t heurMode = CUDNN_HEUR_MODE_FALLBACK;
    const BackendDescriptor heur(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    checkStatus(cudnnBackendSetAttribute(heur.get(),
                                         CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         opGraph.address()));
    checkStatus(cudnnBackendSetAttribute(
        heur.get(), CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heurMode));
    checkStatus(cudnnBackendFinalize(heur.get()));

    int64_t configCount = 0;
    checkStatus(cudnnBackendGetAttribute(heur.get(),
                                         CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         0,
                                         &configCount,
                                         nullptr));
    if(configCount == 0)
    {
        return v8_sample::skipVariant("raw backend API conv fprop fp32 [NCHW]");
    }

    // The heuristic writes into a caller-created ENGINECFG in place.
    const BackendDescriptor engineConfig(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    cudnnBackendDescriptor_t engineConfigRaw = engineConfig.get();
    int64_t returnedCount = 0;
    checkStatus(cudnnBackendGetAttribute(heur.get(),
                                         CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         &returnedCount,
                                         static_cast<void*>(&engineConfigRaw)));
    if(returnedCount != 1)
    {
        throw std::runtime_error("CUDNN_ATTR_ENGINEHEUR_RESULTS returned "
                                 + std::to_string(returnedCount) + " configs, expected 1");
    }

    const BackendDescriptor plan(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    checkStatus(cudnnBackendSetAttribute(plan.get(),
                                         CUDNN_ATTR_EXECUTION_PLAN_HANDLE,
                                         CUDNN_TYPE_HANDLE,
                                         1,
                                         static_cast<const void*>(&handle)));
    checkStatus(cudnnBackendSetAttribute(plan.get(),
                                         CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                         CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                         1,
                                         engineConfig.address()));
    checkStatus(cudnnBackendFinalize(plan.get()));

    int64_t workspaceSize = 0;
    int64_t workspaceSizeCount = 0;
    checkStatus(cudnnBackendGetAttribute(plan.get(),
                                         CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                         CUDNN_TYPE_INT64,
                                         1,
                                         &workspaceSizeCount,
                                         &workspaceSize));
    const hipdnn_data_sdk::utilities::Workspace workspace(static_cast<size_t>(workspaceSize));

    xTensor.fillWithRandomValues(0.0f, 1.0f);
    wTensor.fillWithRandomValues(0.0f, 1.0f);
    yTensor.fillWithValue(0.0f);

    const std::array<void*, 3> dataPointers = {xTensor.memory().deviceData(),
                                               wTensor.memory().deviceData(),
                                               yTensor.memory().deviceData()};
    void* workspacePtr = workspace.get();
    const BackendDescriptor variantPack(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    checkStatus(cudnnBackendSetAttribute(variantPack.get(),
                                         CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                         CUDNN_TYPE_VOID_PTR,
                                         static_cast<int64_t>(dataPointers.size()),
                                         static_cast<const void*>(dataPointers.data())));
    checkStatus(cudnnBackendSetAttribute(variantPack.get(),
                                         CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                                         CUDNN_TYPE_INT64,
                                         static_cast<int64_t>(uids.size()),
                                         uids.data()));
    checkStatus(cudnnBackendSetAttribute(variantPack.get(),
                                         CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                         CUDNN_TYPE_VOID_PTR,
                                         1,
                                         static_cast<const void*>(&workspacePtr)));
    checkStatus(cudnnBackendFinalize(variantPack.get()));

    checkStatus(cudnnBackendExecute(handle, plan.get(), variantPack.get()));
    HIP_CHECK(hipDeviceSynchronize());
    yTensor.memory().markDeviceModified();

    hipdnn_data_sdk::utilities::Tensor<float> yRefTensor(yTensor.dims());
    hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
        xTensor, wTensor, yRefTensor, stride, dilation, pad);

    const auto tolerance
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropTolerance<float, float, float>(
            0.0, 1.0, 0.0, 1.0, wTensor.dims());
    const auto yValidator
        = hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float>(tolerance, tolerance);
    const bool valid = hipdnn_test_sdk::utilities::validateAndReport<float>(
        std::cout, "y", yValidator, yRefTensor, yTensor, tolerance, tolerance);

    v8_sample::ranVariant();
    return valid;
}

} // namespace

int main(int argc, char** argv)
{
    return v8_sample::guardedMain([&]() -> int {
        tensorCreationComparison();

        RETURN_SUCCESS_IF_NO_DEVICE();

        const auto config = parseCommandLineArgs(argc, argv);
        const v8_sample::CudnnHandle handle;

        return v8_sample::finish(runRawConvFprop(handle.get(), config));
    });
}
