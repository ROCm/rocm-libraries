// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Builds cuDNN v8 backend descriptors through the shim's C API only, for the
// integration suites that run them against the real hipDNN backend.
#pragma once

#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_shim_v8_test
{

inline std::string lastError()
{
    std::array<char, 1024> message{};
    cudnnGetLastErrorString(message.data(), message.size());
    return message.data();
}

inline std::vector<std::int64_t> packedStrides(const std::vector<std::int64_t>& dims)
{
    std::vector<std::int64_t> strides(dims.size(), 1);
    for(std::size_t i = dims.size() - 1; i > 0; --i)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    return strides;
}

// The decimal engine id a plan's JSON representation carries in "engine":{"id":"..."}.
inline std::string jsonEngineId(const std::string& json)
{
    const std::string key = "\"id\"";
    const auto keyPos = json.find(key);
    if(keyPos == std::string::npos)
    {
        return {};
    }
    const auto open = json.find('"', json.find(':', keyPos + key.size()));
    if(open == std::string::npos)
    {
        return {};
    }
    const auto close = json.find('"', open + 1);
    if(close == std::string::npos)
    {
        return {};
    }
    return json.substr(open + 1, close - open - 1);
}

class DeviceBuffer
{
public:
    explicit DeviceBuffer(std::size_t bytes)
    {
        EXPECT_EQ(hipMalloc(&_ptr, std::max<std::size_t>(bytes, 1)), hipSuccess);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    ~DeviceBuffer()
    {
        if(_ptr != nullptr)
        {
            static_cast<void>(hipFree(_ptr));
        }
    }

    void* get() const
    {
        return _ptr;
    }

private:
    void* _ptr = nullptr;
};

struct TensorOptions
{
    bool isVirtual = false;
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    bool isByValue = false;
};

using DescriptorSlot = std::pair<cudnnBackendAttributeName_t, cudnnBackendDescriptor_t>;

// Owns every descriptor it creates and destroys them in reverse order.
class GraphBuilder
{
public:
    GraphBuilder() = default;
    GraphBuilder(const GraphBuilder&) = delete;
    GraphBuilder& operator=(const GraphBuilder&) = delete;
    GraphBuilder(GraphBuilder&&) = delete;
    GraphBuilder& operator=(GraphBuilder&&) = delete;

    ~GraphBuilder()
    {
        for(auto it = _owned.rbegin(); it != _owned.rend(); ++it)
        {
            cudnnBackendDestroyDescriptor(*it);
        }
    }

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type)
    {
        cudnnBackendDescriptor_t descriptor = nullptr;
        EXPECT_EQ(cudnnBackendCreateDescriptor(type, &descriptor), CUDNN_STATUS_SUCCESS)
            << lastError();
        _owned.push_back(descriptor);
        return descriptor;
    }

    template <typename T>
    static void set(cudnnBackendDescriptor_t descriptor,
                    cudnnBackendAttributeName_t name,
                    cudnnBackendAttributeType_t type,
                    const T& value)
    {
        EXPECT_EQ(
            cudnnBackendSetAttribute(descriptor, name, type, 1, static_cast<const void*>(&value)),
            CUDNN_STATUS_SUCCESS)
            << "attribute " << static_cast<int>(name) << ": " << lastError();
    }

    template <typename T>
    static void setArray(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t name,
                         cudnnBackendAttributeType_t type,
                         const std::vector<T>& values)
    {
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           name,
                                           type,
                                           static_cast<std::int64_t>(values.size()),
                                           static_cast<const void*>(values.data())),
                  CUDNN_STATUS_SUCCESS)
            << "attribute " << static_cast<int>(name) << ": " << lastError();
    }

    static void finalize(cudnnBackendDescriptor_t descriptor)
    {
        EXPECT_EQ(cudnnBackendFinalize(descriptor), CUDNN_STATUS_SUCCESS) << lastError();
    }

    cudnnBackendDescriptor_t tensor(std::int64_t uid,
                                    const std::vector<std::int64_t>& dims,
                                    const TensorOptions& options = {})
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, options.dataType);
        setArray(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, dims);
        setArray(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, packedStrides(dims));
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
        set(descriptor,
            CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            CUDNN_TYPE_INT64,
            static_cast<std::int64_t>(16));
        if(options.isVirtual)
        {
            set(descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, true);
        }
        if(options.isByValue)
        {
            set(descriptor, CUDNN_ATTR_TENSOR_IS_BY_VALUE, CUDNN_TYPE_BOOLEAN, true);
        }
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t virtualTensor(std::int64_t uid, const std::vector<std::int64_t>& dims)
    {
        TensorOptions options;
        options.isVirtual = true;
        return tensor(uid, dims, options);
    }

    // 2-D cross-correlation with symmetric padding.
    cudnnBackendDescriptor_t convolution(std::int64_t padding, std::int64_t stride)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(descriptor,
            CUDNN_ATTR_CONVOLUTION_CONV_MODE,
            CUDNN_TYPE_CONVOLUTION_MODE,
            CUDNN_CROSS_CORRELATION);
        set(descriptor,
            CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
            CUDNN_TYPE_INT64,
            static_cast<std::int64_t>(2));
        const std::vector<std::int64_t> paddings(2, padding);
        const std::vector<std::int64_t> strides(2, stride);
        const std::vector<std::int64_t> dilations(2, 1);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, paddings);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, paddings);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, dilations);
        setArray(descriptor, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, strides);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t pointwise(cudnnPointwiseMode_t mode)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_POINTWISE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, mode);
        set(descriptor, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t matmul()
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_MATMUL_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(descriptor);
        return descriptor;
    }

    cudnnBackendDescriptor_t reduction(cudnnReduceTensorOp_t op)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_REDUCTION_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_REDUCTION_OPERATOR, CUDNN_TYPE_REDUCTION_OPERATOR_TYPE, op);
        set(descriptor, CUDNN_ATTR_REDUCTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        finalize(descriptor);
        return descriptor;
    }

    // 2-D pooling with a 2x2 window and stride and no padding. The shim requires
    // a padding mode; max pooling pads with -inf.
    cudnnBackendDescriptor_t resample(cudnnResampleMode_t mode)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_RESAMPLE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_RESAMPLE_MODE, CUDNN_TYPE_RESAMPLE_MODE, mode);
        set(descriptor, CUDNN_ATTR_RESAMPLE_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(descriptor,
            CUDNN_ATTR_RESAMPLE_PADDING_MODE,
            CUDNN_TYPE_PADDING_MODE,
            mode == CUDNN_RESAMPLE_MAXPOOL ? CUDNN_NEG_INF_PAD : CUDNN_ZERO_PAD);
        set(descriptor,
            CUDNN_ATTR_RESAMPLE_SPATIAL_DIMS,
            CUDNN_TYPE_INT64,
            static_cast<std::int64_t>(2));
        const std::vector<cudnnFraction_t> windows(2, {2, 1});
        const std::vector<cudnnFraction_t> zeros(2, {0, 1});
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_WINDOW_DIMS, CUDNN_TYPE_FRACTION, windows);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_PRE_PADDINGS, CUDNN_TYPE_FRACTION, zeros);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_POST_PADDINGS, CUDNN_TYPE_FRACTION, zeros);
        setArray(descriptor, CUDNN_ATTR_RESAMPLE_STRIDES, CUDNN_TYPE_FRACTION, windows);
        finalize(descriptor);
        return descriptor;
    }

    // An operation with its descriptor-valued attributes set; enum-valued ones
    // (norm mode and phase) are the caller's, as is finalize.
    cudnnBackendDescriptor_t operation(cudnnBackendDescriptorType_t type,
                                       const std::vector<DescriptorSlot>& slots)
    {
        cudnnBackendDescriptor_t descriptor = create(type);
        for(const auto& [name, value] : slots)
        {
            set(descriptor, name, CUDNN_TYPE_BACKEND_DESCRIPTOR, value);
        }
        return descriptor;
    }

    cudnnBackendDescriptor_t finalizedOperation(cudnnBackendDescriptorType_t type,
                                                const std::vector<DescriptorSlot>& slots)
    {
        cudnnBackendDescriptor_t descriptor = operation(type, slots);
        finalize(descriptor);
        return descriptor;
    }

    // Not finalized; a null handle leaves HANDLE unset.
    cudnnBackendDescriptor_t graph(cudnnHandle_t handle,
                                   const std::vector<cudnnBackendDescriptor_t>& ops)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        if(handle != nullptr)
        {
            set(descriptor, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, handle);
        }
        setArray(descriptor, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, ops);
        return descriptor;
    }

    // A finalized single pointwise ReLU graph; every test plugin claims any graph.
    cudnnBackendDescriptor_t reluGraph(cudnnHandle_t handle,
                                       std::int64_t xUid,
                                       std::int64_t yUid,
                                       const std::vector<std::int64_t>& dims)
    {
        cudnnBackendDescriptor_t op = finalizedOperation(
            CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
            {{CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, pointwise(CUDNN_POINTWISE_RELU_FWD)},
             {CUDNN_ATTR_OPERATION_POINTWISE_XDESC, tensor(xUid, dims)},
             {CUDNN_ATTR_OPERATION_POINTWISE_YDESC, tensor(yUid, dims)}});
        cudnnBackendDescriptor_t descriptor = graph(handle, {op});
        finalize(descriptor);
        return descriptor;
    }

    // ENGINEHEUR RESULTS in rank order, filled into caller-created ENGINECFGs.
    std::vector<cudnnBackendDescriptor_t> heuristicConfigs(cudnnBackendDescriptor_t graph)
    {
        cudnnBackendDescriptor_t heuristic = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
        set(heuristic, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, graph);
        set(heuristic, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, CUDNN_HEUR_MODE_INSTANT);
        finalize(heuristic);

        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(heuristic,
                                           CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           0,
                                           &count,
                                           nullptr),
                  CUDNN_STATUS_SUCCESS)
            << lastError();

        std::vector<cudnnBackendDescriptor_t> configs;
        configs.reserve(static_cast<std::size_t>(std::max<std::int64_t>(count, 0)));
        for(std::int64_t i = 0; i < count; ++i)
        {
            configs.push_back(create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
        }
        if(count > 0)
        {
            std::int64_t filled = 0;
            EXPECT_EQ(cudnnBackendGetAttribute(heuristic,
                                               CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                               CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                               count,
                                               &filled,
                                               static_cast<void*>(configs.data())),
                      CUDNN_STATUS_SUCCESS)
                << lastError();
            EXPECT_EQ(filled, count);
        }
        return configs;
    }

    cudnnBackendDescriptor_t engine(cudnnBackendDescriptor_t graph, std::int64_t globalIndex)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, graph);
        set(descriptor, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, globalIndex);
        finalize(descriptor);
        return descriptor;
    }

    // The engine index behind a finalized ENGINECFG, read through a caller-created
    // ENGINE the shim fills in place, as upstream does.
    std::int64_t configEngineIndex(cudnnBackendDescriptor_t config)
    {
        cudnnBackendDescriptor_t engineDescriptor = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(config,
                                           CUDNN_ATTR_ENGINECFG_ENGINE,
                                           CUDNN_TYPE_BACKEND_DESCRIPTOR,
                                           1,
                                           &count,
                                           static_cast<void*>(&engineDescriptor)),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        return getInt64(engineDescriptor, CUDNN_ATTR_ENGINE_GLOBAL_INDEX);
    }

    cudnnBackendDescriptor_t knobChoice(cudnnBackendKnobType_t type, std::int64_t value)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE, CUDNN_TYPE_KNOB_TYPE, type);
        set(descriptor, CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE, CUDNN_TYPE_INT64, value);
        finalize(descriptor);
        return descriptor;
    }

    // Not finalized, so callers can assert on the finalize status.
    cudnnBackendDescriptor_t engineConfig(cudnnBackendDescriptor_t engineDescriptor,
                                          const std::vector<cudnnBackendDescriptor_t>& choices = {})
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
        set(descriptor,
            CUDNN_ATTR_ENGINECFG_ENGINE,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            engineDescriptor);
        if(!choices.empty())
        {
            setArray(descriptor,
                     CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,
                     CUDNN_TYPE_BACKEND_DESCRIPTOR,
                     choices);
        }
        return descriptor;
    }

    // Not finalized.
    cudnnBackendDescriptor_t plan(cudnnHandle_t handle, cudnnBackendDescriptor_t config)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        set(descriptor, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, handle);
        set(descriptor,
            CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
            CUDNN_TYPE_BACKEND_DESCRIPTOR,
            config);
        return descriptor;
    }

    // Not finalized. JSON before HANDLE, the order upstream's loadFromJson uses.
    cudnnBackendDescriptor_t loadedPlan(cudnnHandle_t handle, const std::string& json)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        EXPECT_EQ(cudnnBackendSetAttribute(descriptor,
                                           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                           CUDNN_TYPE_CHAR,
                                           static_cast<std::int64_t>(json.size()),
                                           static_cast<const void*>(json.data())),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        set(descriptor, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, handle);
        return descriptor;
    }

    cudnnBackendDescriptor_t variantPack(const std::vector<std::int64_t>& uids,
                                         const std::vector<void*>& pointers,
                                         void* workspace)
    {
        cudnnBackendDescriptor_t descriptor = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        setArray(descriptor, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, uids);
        setArray(descriptor, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, pointers);
        set(descriptor, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, workspace);
        finalize(descriptor);
        return descriptor;
    }

    static std::int64_t getInt64(cudnnBackendDescriptor_t descriptor,
                                 cudnnBackendAttributeName_t name)
    {
        std::int64_t value = -1;
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(
                      descriptor, name, CUDNN_TYPE_INT64, 1, &count, static_cast<void*>(&value)),
                  CUDNN_STATUS_SUCCESS)
            << "attribute " << static_cast<int>(name) << ": " << lastError();
        return value;
    }

    static std::string planJson(cudnnBackendDescriptor_t planDescriptor)
    {
        std::int64_t count = 0;
        EXPECT_EQ(cudnnBackendGetAttribute(planDescriptor,
                                           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                           CUDNN_TYPE_CHAR,
                                           0,
                                           &count,
                                           nullptr),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        std::string json(static_cast<std::size_t>(std::max<std::int64_t>(count, 0)), '\0');
        EXPECT_EQ(cudnnBackendGetAttribute(planDescriptor,
                                           CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                           CUDNN_TYPE_CHAR,
                                           count,
                                           &count,
                                           static_cast<void*>(json.data())),
                  CUDNN_STATUS_SUCCESS)
            << lastError();
        json.resize(json.find('\0') == std::string::npos ? json.size() : json.find('\0'));
        return json;
    }

private:
    std::vector<cudnnBackendDescriptor_t> _owned;
};

} // namespace hipdnn_shim_v8_test
